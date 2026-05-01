import threading
import time
import json
import subprocess
import sys
from typing import Any
from pathlib import Path

import pandas as pd
import numpy as np

from datetime import datetime, UTC
from decimal import Decimal
from zoneinfo import ZoneInfo

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails

from build_s_universe import build_s_universe

ET: ZoneInfo = ZoneInfo("America/New_York")

FLUSH_SIZE: int = 10_000
RETRY_SCHEDULE: list[int] = [1, 2, 4, 8, 15, 30]
BASE_QUOTES_PATH: Path = Path(__file__).with_name("base_quotes.json")
BASE_QUOTES_REFRESH_LOG_PATH: Path = Path(__file__).with_name("base_quotes_refresh.txt")
BOOTSTRAP_SCRIPT_PATH: Path = Path(__file__).with_name("bootstrap_base_quotes.py")
FALLBACK_PRICE_CSV_PATH: Path = Path(__file__).with_name("current_fallback_prices_v2.csv")
BASE_QUOTES_REFRESH_TIMEOUT_SECS: int = 420
BASE_QUOTES_MAX_AGE_SECS: int = 300
BASE_QUOTES_PERSIST_INTERVAL_SECS: int = 60
INHERITED_TICK_MAX_AGE_SECS: int = 60 * 60

def get_equity_exchange(default_ex: str) -> str:
    now: datetime = datetime.now(ET)
    t: tuple[int, int] = (now.hour, now.minute)
    if t <= (3, 50) or t >= (20, 0):
        return "OVERNIGHT"
    return default_ex


def _parse_trading_hours_endpoint(raw: str, fallback_date: str, tz: ZoneInfo) -> datetime | None:
    raw = raw.strip()
    if not raw:
        return None

    if ":" in raw:
        date_part, time_part = raw.split(":", 1)
    else:
        date_part, time_part = fallback_date, raw

    if len(date_part) != 8 or len(time_part) != 4:
        return None

    try:
        year = int(date_part[0:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        hour = int(time_part[0:2])
        minute = int(time_part[2:4])
    except ValueError:
        return None

    return datetime(year, month, day, hour, minute, tzinfo=tz)


def is_tick_tradable(details: ContractDetails, now: datetime | None = None) -> bool:
    trading_hours: str = getattr(details, "tradingHours", "") or ""
    if not trading_hours:
        return False

    tz_name: str = getattr(details, "timeZoneId", "") or "America/New_York"
    try:
        market_tz = ZoneInfo(tz_name)
    except Exception:
        market_tz = ET

    now_market: datetime = (now or datetime.now(UTC)).astimezone(market_tz)

    for day_block in trading_hours.split(";"):
        block = day_block.strip()
        if not block or ":CLOSED" in block:
            continue

        if ":" not in block:
            continue

        date_part, sessions = block.split(":", 1)
        for session in sessions.split(","):
            session = session.strip()
            if not session or session == "CLOSED" or "-" not in session:
                continue

            start_raw, end_raw = session.split("-", 1)
            start_dt = _parse_trading_hours_endpoint(start_raw, date_part, market_tz)
            end_dt = _parse_trading_hours_endpoint(end_raw, date_part, market_tz)
            if start_dt is None or end_dt is None:
                continue

            if start_dt <= now_market < end_dt:
                return True

    return False


def has_valid_quote_values(quote: dict[str, Any]) -> bool:
    bid = quote.get("bid")
    ask = quote.get("ask")
    bid_size = quote.get("bid_size")
    ask_size = quote.get("ask_size")

    if None in (bid, ask, bid_size, ask_size):
        return False

    try:
        bid_f = float(bid)
        ask_f = float(ask)
        bid_size_f = float(bid_size)
        ask_size_f = float(ask_size)
    except (TypeError, ValueError):
        return False

    if bid_f in (-100.0, -1.0) or ask_f in (-100.0, -1.0):
        return False

    return bid_f > 0.0 and ask_f > 0.0 and bid_size_f > 0.0 and ask_size_f > 0.0


def format_output_symbol(symbol: str, sec_type: str) -> str:
    if sec_type == "CASH":
        return symbol.replace(".", "-")
    return symbol


def json_safe_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_valid_price_tick(price: Any) -> bool:
    value = json_safe_number(price)
    if value is None:
        return False
    return value > 0.0 and value not in (-1.0, -100.0)


def is_valid_size_tick(size: Any) -> bool:
    value = json_safe_number(size)
    if value is None:
        return False
    return value > 0.0


def load_base_quotes(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    payload: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    quotes: dict[int, dict[str, Any]] = {}

    for req_id_str, row in payload.items():
        try:
            req_id = int(req_id_str)
        except ValueError:
            continue

        quotes[req_id] = {
            "symbol": row.get("symbol"),
            "asset_class": row.get("asset_class"),
            "exchange": row.get("exchange"),
            "bid": row.get("bid"),
            "ask": row.get("ask"),
            "bid_size": row.get("bid_size"),
            "ask_size": row.get("ask_size"),
            "tradable": row.get("tradable"),
            "source": row.get("source"),
            "as_of": row.get("as_of"),
            "last_tick_at": row.get("last_tick_at"),
            "local_symbol": row.get("local_symbol"),
            "expiry": row.get("expiry"),
            "notes": row.get("notes"),
        }

    return quotes


def validate_base_quotes(quotes: dict[int, dict[str, Any]], expected_count: int) -> bool:
    if len(quotes) != expected_count:
        return False

    for row in quotes.values():
        bid = row.get("bid")
        ask = row.get("ask")
        if bid is None or ask is None:
            return False
        try:
            if float(bid) <= 0.0 or float(ask) <= 0.0:
                return False
        except (TypeError, ValueError):
            return False

    return True


def parse_quote_as_of(value: str | None) -> datetime | None:
    if not value:
        return None

    for fmt in ("%Y%m%d %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


def quote_tick_is_stale(quote: dict[str, Any], now: datetime | None = None, max_age_secs: int = INHERITED_TICK_MAX_AGE_SECS) -> bool:
    timestamp_value: str | None = quote.get("last_tick_at") or quote.get("as_of")
    last_tick_at: datetime | None = parse_quote_as_of(timestamp_value)
    if last_tick_at is None:
        return True

    current_time: datetime = now or datetime.now(UTC)
    age_secs = (current_time - last_tick_at).total_seconds()
    return age_secs > max_age_secs


def quote_should_be_tradable(quote: dict[str, Any], now: datetime | None = None) -> bool:
    if not has_valid_quote_values(quote):
        return False

    current_time: datetime = now or datetime.now(UTC)
    if quote_tick_is_stale(quote, now=current_time):
        return False

    details: ContractDetails | None = quote.get("contract_details")
    if details is not None and not is_tick_tradable(details, now=current_time):
        return False

    return True


def base_quotes_are_fresh(quotes: dict[int, dict[str, Any]], max_age_secs: int) -> bool:
    if not quotes:
        return False

    newest_as_of: datetime | None = None
    for row in quotes.values():
        as_of = parse_quote_as_of(row.get("as_of"))
        if as_of is None:
            continue
        if newest_as_of is None or as_of > newest_as_of:
            newest_as_of = as_of

    if newest_as_of is None:
        return False

    age_secs = (datetime.now(UTC) - newest_as_of).total_seconds()
    return age_secs <= max_age_secs


def base_quotes_file_is_fresh(path: Path, max_age_secs: int) -> bool:
    if not path.exists():
        return False

    try:
        age_secs = time.time() - path.stat().st_mtime
    except OSError:
        return False

    return age_secs <= max_age_secs


def refresh_base_quotes() -> None:
    if not BOOTSTRAP_SCRIPT_PATH.exists():
        print("Base quote refresh skipped: bootstrap script missing")
        return

    try:
        existing_quotes = load_base_quotes(BASE_QUOTES_PATH)
    except Exception as exc:
        print("Failed to inspect existing base quotes", exc)
        existing_quotes = {}

    quotes_fresh = base_quotes_are_fresh(existing_quotes, BASE_QUOTES_MAX_AGE_SECS)
    file_fresh = base_quotes_file_is_fresh(BASE_QUOTES_PATH, BASE_QUOTES_MAX_AGE_SECS)
    if quotes_fresh and file_fresh:
        print("Base quotes still fresh; skipping refresh")
        return

    if not file_fresh:
        print("Base quotes JSON is older than 5 minutes; refreshing")
    elif not quotes_fresh:
        print("Base quote timestamps are older than 5 minutes; refreshing")

    print("Refreshing base quotes")

    cmd: list[str] = [
        sys.executable,
        str(BOOTSTRAP_SCRIPT_PATH),
        "--json-output",
        str(BASE_QUOTES_PATH),
        "--text-output",
        str(BASE_QUOTES_REFRESH_LOG_PATH),
        "--timeout",
        str(BASE_QUOTES_REFRESH_TIMEOUT_SECS),
    ]

    if FALLBACK_PRICE_CSV_PATH.exists():
        cmd.extend(["--fallback-price-csv", str(FALLBACK_PRICE_CSV_PATH)])

    try:
        subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            check=False,
            timeout=BASE_QUOTES_REFRESH_TIMEOUT_SECS + 30,
        )
    except subprocess.TimeoutExpired:
        print("Base quote refresh timed out; using existing file")
        return
    except Exception as exc:
        print("Base quote refresh failed", exc)
        return

    try:
        refreshed_quotes = load_base_quotes(BASE_QUOTES_PATH)
    except Exception as exc:
        print("Base quote refresh produced invalid JSON", exc)
        return

    expected_count: int = len(build_s_universe())
    if not validate_base_quotes(refreshed_quotes, expected_count):
        print("Base quote refresh rejected; using existing file")
        return
    print("Base quotes refreshed", len(refreshed_quotes))


class IBApp(EWrapper, EClient):

    def __init__(self) -> None:
        EClient.__init__(self, self)

        self.connected_flag: bool = False
        self.shutting_down: bool = False

        self.data: list[dict[str, Any]] = []
        self.tick_count: int = 0

        self.instruments: list[dict[str, Any]] = build_s_universe()

        self.contractDetails_map: dict[int, list[ContractDetails]] = {}
        self.detail_req_to_instr: dict[int, int] = {}
        self.market_req_to_instr: dict[int, int] = {}
        self.active_market_req_by_instr: dict[int, int] = {}
        self.request_id_counter: int = 100_000
        self.quotes: dict[int, dict[str, Any]] = self.get_base_quotes()
        self.last_base_quotes_persist: float = 0.0

        self.current_equity_exchange: str | None = None

    def get_base_quotes(self) -> dict[int, dict[str, Any]]:
        try:
            quotes = load_base_quotes(BASE_QUOTES_PATH)
        except Exception as exc:
            print("Failed to load base quotes", exc)
            return {}

        now_utc: datetime = datetime.now(UTC)
        for quote in quotes.values():
            quote["tradable"] = quote_should_be_tradable(quote, now=now_utc)

        print("Loaded base quotes", len(quotes))
        return quotes

    def next_request_id(self) -> int:
        self.request_id_counter += 1
        return self.request_id_counter

    def refresh_quote_tradability(self, instr_id: int, now: datetime | None = None) -> None:
        quote: dict[str, Any] | None = self.quotes.get(instr_id)
        if quote is None:
            return
        quote["tradable"] = quote_should_be_tradable(quote, now=now)

    def refresh_all_quote_tradability(self, now: datetime | None = None) -> None:
        current_time: datetime = now or datetime.now(UTC)
        for instr_id in list(self.quotes.keys()):
            self.refresh_quote_tradability(instr_id, now=current_time)

    def persist_base_quotes(self, force: bool = False) -> None:
        if self.shutting_down and not force:
            return

        if len(self.quotes) != len(self.instruments):
            return

        now_ts: float = time.time()
        if not force and now_ts - self.last_base_quotes_persist < BASE_QUOTES_PERSIST_INTERVAL_SECS:
            return

        now_utc: datetime = datetime.now(UTC)
        self.refresh_all_quote_tradability(now=now_utc)

        as_of: str = now_utc.strftime("%Y%m%d %H:%M:%S")
        payload: dict[str, dict[str, Any]] = {}

        for req_id, inst in enumerate(self.instruments):
            q: dict[str, Any] = self.quotes.get(req_id, {})
            symbol: str = q.get("symbol") or (
                f"{inst['symbol']}.{inst['currency']}" if inst["secType"] == "CASH" else inst["symbol"]
            )

            payload[str(req_id)] = {
                "symbol": symbol,
                "asset_class": q.get("asset_class", inst["asset_class"]),
                "exchange": q.get("exchange", inst["exchange"]),
                "bid": json_safe_number(q.get("bid")),
                "ask": json_safe_number(q.get("ask")),
                "bid_size": json_safe_number(q.get("bid_size")),
                "ask_size": json_safe_number(q.get("ask_size")),
                "tradable": bool(q.get("tradable", False)),
                "source": q.get("source", "live_snapshot"),
                "as_of": as_of,
                "last_tick_at": q.get("last_tick_at"),
                "local_symbol": q.get("local_symbol"),
                "expiry": q.get("expiry"),
                "notes": q.get("notes"),
            }

        BASE_QUOTES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.last_base_quotes_persist = now_ts
        print("Persisted base quotes", len(payload))

    def request_contract_details(self, instr_id: int, inst: dict[str, Any], exchange: str) -> None:
        contract = Contract()
        contract.symbol = inst["symbol"]
        contract.secType = inst["secType"]
        contract.exchange = exchange
        contract.currency = inst["currency"]

        detail_req_id: int = self.next_request_id()
        self.detail_req_to_instr[detail_req_id] = instr_id
        self.contractDetails_map[instr_id] = []
        self.reqContractDetails(detail_req_id, contract)

    def select_contract_details(self, inst: dict[str, Any], cds: list[ContractDetails]) -> ContractDetails | None:
        if not cds:
            return None

        if inst["secType"] != "FUT":
            return cds[0]

        today: str = datetime.now(UTC).strftime("%Y%m%d")
        valid: list[ContractDetails] = []
        dated: list[ContractDetails] = []

        for details in cds:
            expiry = (details.contract.lastTradeDateOrContractMonth or "").replace("-", "")
            if not expiry:
                continue
            dated.append(details)
            if expiry >= today:
                valid.append(details)

        pool: list[ContractDetails] = valid if valid else dated
        if not pool:
            return cds[0]

        return min(pool, key=lambda d: (d.contract.lastTradeDateOrContractMonth or "").replace("-", ""))

    def nextValidId(self, orderId: int) -> None:

        print("Connected")
        self.connected_flag = True
        self.reqMarketDataType(1)

        for i, inst in enumerate(self.instruments):
            exchange: str = inst["exchange"]
            if inst["asset_class"] == "EQUITY":
                ex: str = get_equity_exchange(inst["exchange"])
                self.current_equity_exchange = ex
                exchange = ex

            self.request_contract_details(i, inst, exchange)

    def contractDetails(self, reqId: int, details: ContractDetails) -> None:
        instr_id: int | None = self.detail_req_to_instr.get(reqId)
        if instr_id is None:
            return

        if instr_id not in self.contractDetails_map:
            self.contractDetails_map[instr_id] = []

        self.contractDetails_map[instr_id].append(details)

    def contractDetailsEnd(self, reqId: int) -> None:
        instr_id: int | None = self.detail_req_to_instr.pop(reqId, None)
        if instr_id is None:
            return

        inst: dict[str, Any] = self.instruments[instr_id]
        cds: list[ContractDetails] = self.contractDetails_map[instr_id]
        chosen_details = self.select_contract_details(inst, cds)
        if chosen_details is None:
            return

        front: Contract = chosen_details.contract

        self.start_market_data(instr_id, front, inst, front.exchange or inst["exchange"], chosen_details)

    def start_market_data(
        self,
        instr_id: int,
        contract: Contract,
        inst: dict[str, Any],
        exchange: str,
        details: ContractDetails | None = None,
    ) -> None:

        if inst["secType"] == "CASH":
            symbol: str = f'{inst["symbol"]}.{inst["currency"]}'
        else:
            symbol = inst["symbol"]
        output_symbol: str = format_output_symbol(symbol, inst["secType"])

        old_market_req_id: int | None = self.active_market_req_by_instr.get(instr_id)
        if old_market_req_id is not None:
            try:
                self.cancelMktData(old_market_req_id)
            except Exception:
                pass
            self.market_req_to_instr.pop(old_market_req_id, None)

        market_req_id: int = self.next_request_id()
        self.active_market_req_by_instr[instr_id] = market_req_id
        self.market_req_to_instr[market_req_id] = instr_id

        existing: dict[str, Any] = self.quotes.get(instr_id, {})
        self.quotes[instr_id] = {
            "symbol": symbol,
            "output_symbol": output_symbol,
            "asset_class": inst["asset_class"],
            "exchange": exchange,
            "bid": existing.get("bid"),
            "ask": existing.get("ask"),
            "bid_size": existing.get("bid_size"),
            "ask_size": existing.get("ask_size"),
            "tradable": bool(existing.get("tradable", False)),
            "trading_hours": getattr(details, "tradingHours", existing.get("trading_hours")) if details is not None else existing.get("trading_hours"),
            "time_zone_id": getattr(details, "timeZoneId", existing.get("time_zone_id")) if details is not None else existing.get("time_zone_id"),
            "contract_details": details if details is not None else existing.get("contract_details"),
            "source": existing.get("source"),
            "as_of": existing.get("as_of"),
            "last_tick_at": existing.get("last_tick_at"),
            "local_symbol": existing.get("local_symbol"),
            "expiry": existing.get("expiry"),
            "notes": existing.get("notes"),
        }

        self.refresh_quote_tradability(instr_id)

        self.reqMktData(market_req_id, contract, "", False, False, [])

    def resubscribe_equities_if_needed(self) -> None:

        if not self.connected_flag:
            return

        sample_inst: dict[str, Any] | None = next((i for i in self.instruments if i["asset_class"] == "EQUITY"), None)
        if sample_inst is None:
            return

        new_ex: str = get_equity_exchange(sample_inst["exchange"])

        if new_ex == self.current_equity_exchange:
            return

        print("Switching equities to", new_ex)
        self.current_equity_exchange = new_ex

        for reqId, inst in enumerate(self.instruments):

            if inst["asset_class"] != "EQUITY":
                continue

            self.request_contract_details(reqId, inst, new_ex)

    def error(
        self,
        reqId: int,
        errorTime: int,
        errorCode: int,
        errorMsg: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        if errorCode == 300:
            return
        print("ERR", reqId, errorCode, errorMsg)
        if errorCode in {1100, 1101, 1102, 2103, 2105}:
            self.connected_flag = False

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:
        instr_id: int | None = self.market_req_to_instr.get(reqId)
        if instr_id is None:
            return

        accepted_tick: bool = False
        if tickType == 1 and is_valid_price_tick(price):
            self.quotes[instr_id]["bid"] = price
            accepted_tick = True

        elif tickType == 2 and is_valid_price_tick(price):
            self.quotes[instr_id]["ask"] = price
            accepted_tick = True

        if accepted_tick:
            self.quotes[instr_id]["last_tick_at"] = datetime.now(UTC).strftime("%Y%m%d %H:%M:%S")

        self.refresh_quote_tradability(instr_id)

        self.record(instr_id)

    def tickSize(self, reqId: int, tickType: int, size: float) -> None:
        instr_id: int | None = self.market_req_to_instr.get(reqId)
        if instr_id is None:
            return

        accepted_tick: bool = False
        if tickType == 0 and is_valid_size_tick(size):
            self.quotes[instr_id]["bid_size"] = size
            accepted_tick = True

        elif tickType == 3 and is_valid_size_tick(size):
            self.quotes[instr_id]["ask_size"] = size
            accepted_tick = True

        if accepted_tick:
            self.quotes[instr_id]["last_tick_at"] = datetime.now(UTC).strftime("%Y%m%d %H:%M:%S")

        self.refresh_quote_tradability(instr_id)

        self.record(instr_id)

    def record(self, reqId: int) -> None:

        if self.shutting_down:
            return

        if len(self.quotes) != len(self.instruments):
            return

        trigger_quote: dict[str, Any] = self.quotes[reqId]
        if trigger_quote.get("tradable") is False:
            return
        if not has_valid_quote_values(trigger_quote):
            return

        ts: datetime = datetime.now(ET)
        row: dict[tuple[str, str], Any] = {
            ("base", "timestamp"): ts,
            ("base", "trigger_symbol"): trigger_quote.get("output_symbol", trigger_quote.get("symbol")),
        }

        #print(trigger_quote.get("output_symbol", trigger_quote.get("symbol")))
        for req_id, inst in enumerate(self.instruments):
            q: dict[str, Any] = self.quotes.get(req_id, {})
            symbol_key: str = q.get("output_symbol") or format_output_symbol(
                f"{inst['symbol']}.{inst['currency']}" if inst["secType"] == "CASH" else inst["symbol"],
                inst["secType"],
            )

            row[(symbol_key, "asset_class")] = q.get("asset_class", inst["asset_class"])
            row[(symbol_key, "exchange")] = q.get("exchange", inst["exchange"])
            row[(symbol_key, "bid")] = np.float64(q["bid"]) if q.get("bid") is not None else np.nan
            row[(symbol_key, "ask")] = np.float64(q["ask"]) if q.get("ask") is not None else np.nan
            row[(symbol_key, "bid_size")] = np.float64(q["bid_size"]) if q.get("bid_size") is not None else np.nan
            row[(symbol_key, "ask_size")] = np.float64(q["ask_size"]) if q.get("ask_size") is not None else np.nan
            row[(symbol_key, "tradable")] = bool(q.get("tradable", False))

        trigger_symbol = row[("base", "trigger_symbol")]
        obupdate = {
            "bid": row[(trigger_symbol, "bid")],
            "ask": row[(trigger_symbol, "ask")],
            "bid_size": row[(trigger_symbol, "bid_size")],
            "ask_size": row[(trigger_symbol, "ask_size")],
            "tradable": row[(trigger_symbol, "tradable")],
        }
        #print(ts, trigger_symbol, obupdate)
        if self.tick_count % 1000 == 0:
            print(self.tick_count)

        self.data.append(row)
        self.tick_count += 1

        if self.tick_count >= FLUSH_SIZE and not self.shutting_down:
            self.flush()

    def flush(self, force: bool = False) -> None:
        if self.shutting_down and not force:
            return

        if not self.data:
            return

        df: pd.DataFrame = pd.DataFrame(self.data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        ts: str = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
        filename: str = f"C:/Users/mnapo/Desktop/HFT supervised/ALL_DATA/ticks/ticks_{ts}.parquet"

        df.to_parquet(filename, index=False)

        print("Flushed", len(df))

        self.data.clear()
        self.tick_count = 0

    def stop(self) -> None:

        self.shutting_down = True

        try:
            self.persist_base_quotes(force=True)
        except Exception:
            pass

        try:
            self.flush(force=True)
        except Exception:
            pass

        try:
            self.disconnect()
        except Exception:
            pass


def start() -> None:
    refresh_base_quotes()

    retry_i: int = 0

    while True:

        app: IBApp = IBApp()

        try:

            print("Connecting")

            app.connect("127.0.0.1", 7497, clientId=1)

            thread: threading.Thread = threading.Thread(target=app.run, daemon=True)
            thread.start()

            last_check: float = 0

            while True:

                time.sleep(1)

                if app.connected_flag:
                    retry_i = 0

                    now: float = time.time()
                    app.persist_base_quotes()
                    if now - last_check >= 60:
                        app.resubscribe_equities_if_needed()
                        last_check = now

                if not app.connected_flag:
                    raise ConnectionError("Lost connection")

        except Exception as e:

            print("Connection issue", e)

            app.stop()

            delay: int = RETRY_SCHEDULE[min(retry_i, len(RETRY_SCHEDULE) - 1)]
            retry_i += 1

            print("Retrying in", delay)

            time.sleep(delay)

start()
