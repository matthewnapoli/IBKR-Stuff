from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ibapi.client import EClient
from ibapi.contract import Contract, ContractDetails
from ibapi.wrapper import EWrapper

from build_s_universe import build_s_universe


ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
DEFAULT_JSON_OUTPUT = ROOT / "base_quotes.json"
DEFAULT_TEXT_OUTPUT = ROOT / "base_quotes_refresh.txt"
INVALID_PRICES = {-1.0, -100.0}
DELAYED_TICK_TYPES = {
    66: 1,  # delayed bid
    67: 2,  # delayed ask
    69: 0,  # delayed bid size
    70: 3,  # delayed ask size
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the base quote snapshot consumed by scrapin2.py."
    )
    parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host.")
    parser.add_argument("--port", type=int, default=7497, help="TWS/Gateway API port.")
    parser.add_argument("--client-id", type=int, default=2, help="IB API client id.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=420.0,
        help="Maximum seconds to wait for contract details and market data.",
    )
    parser.add_argument(
        "--snapshot-grace-secs",
        type=float,
        default=25.0,
        help="Seconds to wait before historical tick backfill is requested for missing quotes.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Path to write the completed base quote JSON.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=DEFAULT_TEXT_OUTPUT,
        help="Path to write a human-readable refresh report.",
    )
    parser.add_argument(
        "--existing-json",
        type=Path,
        default=None,
        help="Optional existing quote JSON to use as fallback. Defaults to --json-output.",
    )
    parser.add_argument(
        "--fallback-price-csv",
        type=Path,
        default=None,
        help="Optional CSV containing fallback prices keyed by symbol.",
    )
    parser.add_argument(
        "--market-data-type",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="IB market data type: 1 live, 2 frozen, 3 delayed, 4 delayed frozen.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write output even if one or more instruments are still missing bid/ask.",
    )
    return parser.parse_args()


def json_safe_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_valid_price(value: Any) -> bool:
    number = json_safe_number(value)
    return number is not None and number > 0.0 and number not in INVALID_PRICES


def is_valid_size(value: Any) -> bool:
    number = json_safe_number(value)
    return number is not None and number > 0.0


def format_symbol(inst: dict[str, Any]) -> str:
    if inst["secType"] == "CASH":
        return f"{inst['symbol']}.{inst['currency']}"
    return inst["symbol"]


def format_output_symbol(symbol: str, sec_type: str) -> str:
    if sec_type == "CASH":
        return symbol.replace(".", "-")
    return symbol


def get_equity_exchange(default_exchange: str) -> str:
    now = datetime.now(ET)
    hhmm = (now.hour, now.minute)
    if hhmm <= (3, 50) or hhmm >= (20, 0):
        return "OVERNIGHT"
    return default_exchange


def parse_trading_hours_endpoint(raw: str, fallback_date: str, tz: ZoneInfo) -> datetime | None:
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
        return datetime(
            int(date_part[0:4]),
            int(date_part[4:6]),
            int(date_part[6:8]),
            int(time_part[0:2]),
            int(time_part[2:4]),
            tzinfo=tz,
        )
    except ValueError:
        return None


def is_tick_tradable(details: ContractDetails | None, now: datetime | None = None) -> bool:
    if details is None:
        return False

    trading_hours = getattr(details, "tradingHours", "") or ""
    if not trading_hours:
        return False

    tz_name = getattr(details, "timeZoneId", "") or "America/New_York"
    try:
        market_tz = ZoneInfo(tz_name)
    except Exception:
        market_tz = ET

    now_market = (now or datetime.now(UTC)).astimezone(market_tz)
    for day_block in trading_hours.split(";"):
        block = day_block.strip()
        if not block or ":CLOSED" in block or ":" not in block:
            continue

        date_part, sessions = block.split(":", 1)
        for session in sessions.split(","):
            session = session.strip()
            if not session or session == "CLOSED" or "-" not in session:
                continue

            start_raw, end_raw = session.split("-", 1)
            start_dt = parse_trading_hours_endpoint(start_raw, date_part, market_tz)
            end_dt = parse_trading_hours_endpoint(end_raw, date_part, market_tz)
            if start_dt is not None and end_dt is not None and start_dt <= now_market < end_dt:
                return True

    return False


def parse_quote_as_of(value: str | None) -> datetime | None:
    if not value:
        return None

    for fmt in ("%Y%m%d %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


def quote_tick_is_stale(
    quote: dict[str, Any],
    now: datetime | None = None,
    max_age_secs: int = 60 * 60,
) -> bool:
    timestamp_value = quote.get("last_tick_at") or quote.get("as_of")
    last_tick_at = parse_quote_as_of(timestamp_value)
    if last_tick_at is None:
        return True

    current_time = now or datetime.now(UTC)
    return (current_time - last_tick_at).total_seconds() > max_age_secs


def load_existing_quotes(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    quotes: dict[str, dict[str, Any]] = {}

    for key, row in payload.items():
        if not isinstance(row, dict):
            continue
        quotes[str(key)] = row
        symbol = row.get("symbol")
        if symbol:
            quotes[str(symbol)] = row
            quotes[str(symbol).replace("-", ".")] = row

    return quotes


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        if name in lowered and lowered[name] not in ("", None):
            return lowered[name]
    return None


def load_fallback_csv(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    quotes: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = first_present(row, ("symbol", "ticker", "local_symbol", "contract"))
            if not symbol:
                continue

            bid = first_present(row, ("bid", "bid_price"))
            ask = first_present(row, ("ask", "ask_price"))
            mid = first_present(row, ("mid", "price", "last", "close", "mark"))

            if not is_valid_price(bid) and is_valid_price(mid):
                bid = mid
            if not is_valid_price(ask) and is_valid_price(mid):
                ask = mid

            quote = {
                "bid": json_safe_number(bid),
                "ask": json_safe_number(ask),
                "bid_size": json_safe_number(first_present(row, ("bid_size", "bidsize", "size"))) or 1.0,
                "ask_size": json_safe_number(first_present(row, ("ask_size", "asksize", "size"))) or 1.0,
                "source": "fallback_csv",
                "local_symbol": first_present(row, ("local_symbol", "localsymbol")),
                "expiry": first_present(row, ("expiry", "last_trade_date", "lasttradedateorcontractmonth")),
                "notes": f"fallback from {path.name}",
            }
            key = str(symbol)
            quotes[key] = quote
            quotes[key.replace("-", ".")] = quote

    return quotes


def seed_quote(
    instr_id: int,
    inst: dict[str, Any],
    existing_quotes: dict[str, dict[str, Any]],
    fallback_quotes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    symbol = format_symbol(inst)
    output_symbol = format_output_symbol(symbol, inst["secType"])
    base = {
        "symbol": symbol,
        "output_symbol": output_symbol,
        "asset_class": inst["asset_class"],
        "exchange": inst["exchange"],
        "bid": None,
        "ask": None,
        "bid_size": None,
        "ask_size": None,
        "tradable": False,
        "source": "missing",
        "as_of": None,
        "last_tick_at": None,
        "local_symbol": None,
        "expiry": None,
        "notes": "no quote received",
    }

    for source in (
        existing_quotes.get(str(instr_id)),
        existing_quotes.get(symbol),
        existing_quotes.get(output_symbol),
        fallback_quotes.get(symbol),
        fallback_quotes.get(output_symbol),
        fallback_quotes.get(inst["symbol"]),
    ):
        if not source:
            continue

        bid = json_safe_number(source.get("bid"))
        ask = json_safe_number(source.get("ask"))
        if not is_valid_price(bid) or not is_valid_price(ask):
            continue

        base.update(
            {
                "bid": bid,
                "ask": ask,
                "bid_size": json_safe_number(source.get("bid_size")) or 1.0,
                "ask_size": json_safe_number(source.get("ask_size")) or 1.0,
                "source": source.get("source") or "fallback",
                "as_of": source.get("as_of"),
                "last_tick_at": source.get("last_tick_at"),
                "local_symbol": source.get("local_symbol"),
                "expiry": source.get("expiry"),
                "notes": source.get("notes") or "carried forward",
            }
        )
        break

    return base


def quote_complete(quote: dict[str, Any]) -> bool:
    return is_valid_price(quote.get("bid")) and is_valid_price(quote.get("ask"))


def choose_front_contract(inst: dict[str, Any], details: list[ContractDetails]) -> ContractDetails | None:
    if not details:
        return None

    if inst["secType"] != "FUT":
        return details[0]

    today = datetime.now(UTC).strftime("%Y%m%d")
    valid: list[ContractDetails] = []
    dated: list[ContractDetails] = []

    for row in details:
        expiry = (row.contract.lastTradeDateOrContractMonth or "").replace("-", "")
        if not expiry:
            continue
        dated.append(row)
        if expiry >= today:
            valid.append(row)

    pool = valid if valid else dated
    if not pool:
        return details[0]

    return min(pool, key=lambda row: (row.contract.lastTradeDateOrContractMonth or "").replace("-", ""))


class BootstrapQuotesApp(EWrapper, EClient):
    def __init__(
        self,
        instruments: list[dict[str, Any]],
        existing_quotes: dict[str, dict[str, Any]],
        fallback_quotes: dict[str, dict[str, Any]],
        market_data_type: int,
    ) -> None:
        EClient.__init__(self, self)
        self.instruments = instruments
        self.market_data_type = market_data_type
        self.quotes = {
            instr_id: seed_quote(instr_id, inst, existing_quotes, fallback_quotes)
            for instr_id, inst in enumerate(instruments)
        }
        self.contract_details: dict[int, list[ContractDetails]] = {}
        self.contract_by_instr: dict[int, Contract] = {}
        self.details_by_instr: dict[int, ContractDetails] = {}
        self.detail_req_to_instr: dict[int, int] = {}
        self.market_req_to_instr: dict[int, int] = {}
        self.hist_req_to_instr: dict[int, int] = {}
        self.snapshot_done: set[int] = set()
        self.history_done: set[int] = set()
        self.errors: list[str] = []
        self.connected_event = threading.Event()
        self.done_event = threading.Event()
        self.lock = threading.RLock()
        self.next_req_id = 10_000
        self.historical_started = False
        self.started_at = time.time()

    def next_request_id(self) -> int:
        with self.lock:
            self.next_req_id += 1
            return self.next_req_id

    def nextValidId(self, orderId: int) -> None:
        with self.lock:
            self.next_req_id = max(self.next_req_id, orderId + 10_000)
        self.connected_event.set()
        self.reqMarketDataType(self.market_data_type)

        for instr_id, inst in enumerate(self.instruments):
            exchange = inst["exchange"]
            if inst["asset_class"] == "EQUITY":
                exchange = get_equity_exchange(exchange)

            contract = Contract()
            contract.symbol = inst["symbol"]
            contract.secType = inst["secType"]
            contract.exchange = exchange
            contract.currency = inst["currency"]

            req_id = self.next_request_id()
            with self.lock:
                self.detail_req_to_instr[req_id] = instr_id
                self.contract_details[instr_id] = []
            self.reqContractDetails(req_id, contract)

    def contractDetails(self, reqId: int, details: ContractDetails) -> None:
        with self.lock:
            instr_id = self.detail_req_to_instr.get(reqId)
            if instr_id is None:
                return
            self.contract_details.setdefault(instr_id, []).append(details)

    def contractDetailsEnd(self, reqId: int) -> None:
        with self.lock:
            instr_id = self.detail_req_to_instr.pop(reqId, None)
            if instr_id is None:
                return
            inst = self.instruments[instr_id]
            details = choose_front_contract(inst, self.contract_details.get(instr_id, []))

        if details is None:
            with self.lock:
                self.errors.append(f"{instr_id} {format_symbol(inst)}: no contract details")
                self.snapshot_done.add(instr_id)
            self.maybe_done()
            return

        contract = details.contract
        symbol = format_symbol(inst)
        output_symbol = format_output_symbol(symbol, inst["secType"])
        exchange = contract.exchange or inst["exchange"]

        with self.lock:
            self.contract_by_instr[instr_id] = contract
            self.details_by_instr[instr_id] = details
            self.quotes[instr_id].update(
                {
                    "symbol": symbol,
                    "output_symbol": output_symbol,
                    "asset_class": inst["asset_class"],
                    "exchange": exchange,
                    "local_symbol": contract.localSymbol or self.quotes[instr_id].get("local_symbol"),
                    "expiry": contract.lastTradeDateOrContractMonth or self.quotes[instr_id].get("expiry"),
                }
            )

            req_id = self.next_request_id()
            self.market_req_to_instr[req_id] = instr_id

        self.reqMktData(req_id, contract, "", True, False, [])

    def tickSnapshotEnd(self, reqId: int) -> None:
        with self.lock:
            instr_id = self.market_req_to_instr.pop(reqId, None)
            if instr_id is not None:
                self.snapshot_done.add(instr_id)
        self.maybe_done()

    def update_quote_from_tick(self, reqId: int, tickType: int, value: Any, is_size: bool) -> None:
        tickType = DELAYED_TICK_TYPES.get(tickType, tickType)
        with self.lock:
            instr_id = self.market_req_to_instr.get(reqId)
            if instr_id is None:
                return

            quote = self.quotes[instr_id]
            accepted = False
            if is_size:
                size = json_safe_number(value)
                if tickType == 0 and is_valid_size(size):
                    quote["bid_size"] = size
                    accepted = True
                elif tickType == 3 and is_valid_size(size):
                    quote["ask_size"] = size
                    accepted = True
            else:
                price = json_safe_number(value)
                if tickType == 1 and is_valid_price(price):
                    quote["bid"] = price
                    accepted = True
                elif tickType == 2 and is_valid_price(price):
                    quote["ask"] = price
                    accepted = True

            if accepted:
                now = datetime.now(UTC).strftime("%Y%m%d %H:%M:%S")
                quote["source"] = "snapshot"
                quote["as_of"] = now
                quote["last_tick_at"] = now
                quote["notes"] = "live snapshot"

        if accepted:
            self.maybe_done()

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:
        self.update_quote_from_tick(reqId, tickType, price, is_size=False)

    def tickSize(self, reqId: int, tickType: int, size: Decimal) -> None:
        self.update_quote_from_tick(reqId, tickType, size, is_size=True)

    def start_historical_backfill(self) -> None:
        with self.lock:
            if self.historical_started:
                return
            self.historical_started = True
            missing_ids = [
                instr_id
                for instr_id, quote in self.quotes.items()
                if not quote_complete(quote) and instr_id in self.contract_by_instr
            ]

        end_time = datetime.now(UTC).strftime("%Y%m%d %H:%M:%S UTC")
        for instr_id in missing_ids:
            with self.lock:
                contract = self.contract_by_instr[instr_id]
                req_id = self.next_request_id()
                self.hist_req_to_instr[req_id] = instr_id
            self.reqHistoricalTicks(req_id, contract, "", end_time, 100, "BID_ASK", 0, True, [])

        with self.lock:
            if not missing_ids:
                self.history_done.update(self.quotes.keys())
        self.maybe_done()

    def historicalTicksBidAsk(self, reqId: int, ticks: list[Any], done: bool) -> None:
        with self.lock:
            instr_id = self.hist_req_to_instr.get(reqId)
            if instr_id is None:
                return

            quote = self.quotes[instr_id]
            for tick in ticks:
                bid = json_safe_number(getattr(tick, "priceBid", None))
                ask = json_safe_number(getattr(tick, "priceAsk", None))
                if not is_valid_price(bid) or not is_valid_price(ask):
                    continue

                quote["bid"] = bid
                quote["ask"] = ask
                quote["bid_size"] = json_safe_number(getattr(tick, "sizeBid", None)) or quote.get("bid_size") or 1.0
                quote["ask_size"] = json_safe_number(getattr(tick, "sizeAsk", None)) or quote.get("ask_size") or 1.0
                quote["source"] = "historical_bid_ask"
                tick_time = getattr(tick, "time", None)
                if tick_time:
                    quote["last_tick_at"] = datetime.fromtimestamp(int(tick_time), UTC).strftime("%Y%m%d %H:%M:%S")
                quote["as_of"] = datetime.now(UTC).strftime("%Y%m%d %H:%M:%S")
                quote["notes"] = "recent bid/ask tick"

            if done:
                self.hist_req_to_instr.pop(reqId, None)
                self.history_done.add(instr_id)
        self.maybe_done()

    def historicalTicks(self, reqId: int, ticks: list[Any], done: bool) -> None:
        with self.lock:
            instr_id = self.hist_req_to_instr.get(reqId)
            if instr_id is None:
                return

            quote = self.quotes[instr_id]
            for tick in ticks:
                price = json_safe_number(getattr(tick, "price", None))
                if not is_valid_price(price):
                    continue
                quote["bid"] = quote.get("bid") if is_valid_price(quote.get("bid")) else price
                quote["ask"] = quote.get("ask") if is_valid_price(quote.get("ask")) else price
                quote["bid_size"] = quote.get("bid_size") or 1.0
                quote["ask_size"] = quote.get("ask_size") or 1.0
                quote["source"] = "historical_price"
                tick_time = getattr(tick, "time", None)
                if tick_time:
                    quote["last_tick_at"] = datetime.fromtimestamp(int(tick_time), UTC).strftime("%Y%m%d %H:%M:%S")
                quote["as_of"] = datetime.now(UTC).strftime("%Y%m%d %H:%M:%S")
                quote["notes"] = "recent price tick"

            if done:
                self.hist_req_to_instr.pop(reqId, None)
                self.history_done.add(instr_id)
        self.maybe_done()

    def error(
        self,
        reqId: int,
        errorTime: int,
        errorCode: int,
        errorMsg: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        if errorCode in {2104, 2106, 2158}:
            return

        with self.lock:
            self.errors.append(f"{reqId}: {errorCode} {errorMsg}")
            instr_id = self.detail_req_to_instr.pop(reqId, None)
            if instr_id is not None:
                self.snapshot_done.add(instr_id)
            instr_id = self.market_req_to_instr.pop(reqId, None)
            if instr_id is not None:
                self.snapshot_done.add(instr_id)
            instr_id = self.hist_req_to_instr.pop(reqId, None)
            if instr_id is not None:
                self.history_done.add(instr_id)
        self.maybe_done()

    def maybe_done(self) -> None:
        with self.lock:
            if all(quote_complete(quote) for quote in self.quotes.values()):
                self.done_event.set()

    def complete_payload(self) -> dict[str, dict[str, Any]]:
        now_utc = datetime.now(UTC)
        as_of = now_utc.strftime("%Y%m%d %H:%M:%S")
        payload: dict[str, dict[str, Any]] = {}

        with self.lock:
            for instr_id, inst in enumerate(self.instruments):
                quote = self.quotes[instr_id].copy()
                details = self.details_by_instr.get(instr_id)
                bid_size = json_safe_number(quote.get("bid_size")) or 1.0
                ask_size = json_safe_number(quote.get("ask_size")) or 1.0
                quote_is_complete = quote_complete(quote)

                payload[str(instr_id)] = {
                    "symbol": quote.get("symbol") or format_symbol(inst),
                    "asset_class": quote.get("asset_class") or inst["asset_class"],
                    "exchange": quote.get("exchange") or inst["exchange"],
                    "bid": json_safe_number(quote.get("bid")),
                    "ask": json_safe_number(quote.get("ask")),
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "tradable": (
                        quote_is_complete
                        and not quote_tick_is_stale(quote, now=now_utc)
                        and is_tick_tradable(details, now=now_utc)
                    ),
                    "source": quote.get("source") or "missing",
                    "as_of": as_of,
                    "last_tick_at": quote.get("last_tick_at"),
                    "local_symbol": quote.get("local_symbol"),
                    "expiry": quote.get("expiry"),
                    "notes": quote.get("notes"),
                }

        return payload

    def missing_symbols(self) -> list[str]:
        with self.lock:
            return [
                f"{instr_id}:{self.quotes[instr_id].get('symbol') or format_symbol(inst)}"
                for instr_id, inst in enumerate(self.instruments)
                if not quote_complete(self.quotes[instr_id])
            ]


def write_json_atomic(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def write_report(
    path: Path,
    payload: dict[str, dict[str, Any]],
    errors: list[str],
    missing: list[str],
    elapsed_secs: float,
    wrote_json: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"bootstrap_base_quotes completed at {datetime.now(UTC).strftime('%Y%m%d %H:%M:%S')} UTC",
        f"elapsed_secs={elapsed_secs:.1f}",
        f"rows={len(payload)}",
        f"wrote_json={wrote_json}",
        f"missing_count={len(missing)}",
        "",
        "sources:",
    ]

    source_counts: dict[str, int] = {}
    for row in payload.values():
        source = str(row.get("source") or "missing")
        source_counts[source] = source_counts.get(source, 0) + 1
    for source, count in sorted(source_counts.items()):
        lines.append(f"  {source}: {count}")

    if missing:
        lines.extend(["", "missing:"])
        lines.extend(f"  {symbol}" for symbol in missing)

    if errors:
        lines.extend(["", "errors:"])
        lines.extend(f"  {error}" for error in errors[-100:])

    lines.extend(["", "quotes:"])
    for key in sorted(payload, key=int):
        row = payload[key]
        lines.append(
            "  "
            f"{key:>3} {row.get('symbol')} "
            f"bid={row.get('bid')} ask={row.get('ask')} "
            f"bid_size={row.get('bid_size')} ask_size={row.get('ask_size')} "
            f"source={row.get('source')} local={row.get('local_symbol')} expiry={row.get('expiry')}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    existing_json = args.existing_json or args.json_output
    instruments = build_s_universe()
    existing_quotes = load_existing_quotes(existing_json)
    fallback_quotes = load_fallback_csv(args.fallback_price_csv)
    app = BootstrapQuotesApp(instruments, existing_quotes, fallback_quotes, args.market_data_type)

    started_at = time.time()
    thread: threading.Thread | None = None

    try:
        app.connect(args.host, args.port, clientId=args.client_id)
        thread = threading.Thread(target=app.run, daemon=True)
        thread.start()
        app.connected_event.wait(timeout=min(args.timeout, 15.0))

        deadline = started_at + args.timeout
        historical_started = False

        while time.time() < deadline:
            if app.done_event.wait(timeout=0.5):
                break

            if not historical_started and time.time() - started_at >= args.snapshot_grace_secs:
                app.start_historical_backfill()
                historical_started = True

        if not historical_started:
            app.start_historical_backfill()
            app.done_event.wait(timeout=max(0.0, deadline - time.time()))
    except Exception as exc:
        with app.lock:
            app.errors.append(f"bootstrap exception: {exc}")
    finally:
        try:
            app.disconnect()
        except Exception:
            pass
        if thread is not None:
            thread.join(timeout=2.0)

    payload = app.complete_payload()
    missing = app.missing_symbols()
    wrote_json = False
    if args.allow_partial or not missing:
        write_json_atomic(args.json_output, payload)
        wrote_json = True

    write_report(
        args.text_output,
        payload,
        app.errors,
        missing,
        elapsed_secs=time.time() - started_at,
        wrote_json=wrote_json,
    )

    if missing and not args.allow_partial:
        print(f"Missing bid/ask for {len(missing)} instrument(s); kept existing JSON output untouched.")
        return 1

    print(f"Wrote {len(payload)} base quote(s) to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
