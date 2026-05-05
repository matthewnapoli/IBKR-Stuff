r"""Run a local L1 snapshot backtest.

Usage:
    python test_runner.py algo.py --data "C:\path\to\ticks.parquet"

The algorithm file can expose either:
    class Trader:
        def run(self, state): ...

or:
    def run(state): ...
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import datetime
import importlib.util
from io import StringIO
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("This runner needs pandas/pyarrow. Run it with .\\twsEnv\\Scripts\\python.exe or install pandas.") from exc

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit("This runner needs pandas/pyarrow. Run it with .\\twsEnv\\Scripts\\python.exe or install pyarrow.") from exc

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from datamodel import OrderDepth, TradingState


TIMESTAMP_COLUMN = ("base", "timestamp")
RAW_TIMESTAMP_COLUMN = repr(TIMESTAMP_COLUMN)
G10_CURRENCIES = {"USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK"}
DEFAULT_POSITION_LIMIT_USD = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local 1-second L1 depth backtest.")
    parser.add_argument("algorithm", type=Path, help="Path to the algorithm .py file.")
    parser.add_argument(
        "--data",
        "--path-to-data",
        required=True,
        dest="data",
        type=Path,
        help="Path to one parquet tick snapshot file.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        help="Optional product list to run within the selected asset-class universe.",
    )
    parser.add_argument("--fx", action="store_true", help="Run FX products.")
    parser.add_argument("--equities", action="store_true", help="Run equity products.")
    parser.add_argument("--index-fut", action="store_true", help="Run index futures.")
    parser.add_argument("--com-fut", action="store_true", help="Run commodity futures.")
    parser.add_argument("--rate-fut", action="store_true", help="Run rate futures.")
    parser.add_argument("--g10", action="store_true", help="Run G10 FX pairs only.")
    parser.add_argument("--all", action="store_true", help="Run all products. This is the default.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional max number of sampled 1-second states to run.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=10_000,
        help="Rows to read from parquet at a time before 1-second sampling.",
    )
    parser.add_argument(
        "--keep-untradable",
        action="store_true",
        help="Include products whose tradable column is false.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("backtests"),
        help="Directory for orderbook and trade parquet logs.",
    )
    parser.add_argument(
        "--name",
        help="Output file prefix. Defaults to algorithm name plus timestamp.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Also print algorithm stdout to the console. By default it is captured into the orderbook parquet log.",
    )
    parser.add_argument(
        "--write-parquets",
        action="store_true",
        help="Also write the legacy orderbooks/trades parquet files.",
    )
    parser.add_argument(
        "--position-limit-usd",
        type=float,
        default=DEFAULT_POSITION_LIMIT_USD,
        help="Per-product absolute position limit in USD equivalent, marked with current mid.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = []
    for column in df.columns:
        if isinstance(column, tuple):
            columns.append(column)
            continue

        if isinstance(column, str) and column.startswith("("):
            try:
                parsed = ast.literal_eval(column)
            except (SyntaxError, ValueError):
                parsed = column
            columns.append(parsed if isinstance(parsed, tuple) else column)
            continue

        columns.append((column, ""))

    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(columns)
    return df


def available_products(df: pd.DataFrame) -> list[str]:
    products = []
    for product in df.columns.get_level_values(0).unique():
        if product == "base":
            continue
        if (product, "bid") in df.columns and (product, "ask") in df.columns:
            products.append(product)
    return products


def selected_asset_classes(args: argparse.Namespace) -> set[str] | None:
    selected = set()
    if args.fx:
        selected.add("FX")
    if args.equities:
        selected.add("EQUITY")
    if args.index_fut:
        selected.add("INDEX_FUTURE")
    if args.com_fut:
        selected.add("FUTURE")
    if args.rate_fut:
        selected.add("RATE")

    if args.all or not selected:
        return None

    return selected


def products_for_asset_classes(df: pd.DataFrame, asset_classes: set[str] | None) -> list[str]:
    products = available_products(df)
    if asset_classes is None:
        return products

    filtered = []
    for product in products:
        if (product, "asset_class") not in df.columns:
            continue

        asset_classes_for_product = df[(product, "asset_class")].dropna()
        if asset_classes_for_product.empty:
            continue

        if asset_classes_for_product.iloc[0] in asset_classes:
            filtered.append(product)

    return filtered


def resolve_products(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    products = products_for_asset_classes(df, selected_asset_classes(args))
    if args.g10:
        products = [product for product in products if is_g10_fx_pair(product)]

    if not args.products:
        return products

    requested = set(args.products)
    filtered = [product for product in products if product in requested]
    missing = sorted(requested - set(filtered))
    if missing:
        print(f"Warning: requested products not found in selected universe: {', '.join(missing)}")
    return filtered


def is_g10_fx_pair(product: str) -> bool:
    base, separator, quote = product.partition("-")
    return bool(separator) and base in G10_CURRENCIES and quote in G10_CURRENCIES


def product_asset_classes(df: pd.DataFrame, products: list[str]) -> dict[str, str | None]:
    asset_classes = {}
    for product in products:
        if (product, "asset_class") not in df.columns:
            asset_classes[product] = None
            continue

        values = df[(product, "asset_class")].dropna()
        asset_classes[product] = None if values.empty else str(values.iloc[0])
    return asset_classes


def load_sampled_snapshots(data_path: Path, read_batch_size: int, show_progress: bool) -> pd.DataFrame:
    if not data_path.is_file():
        raise SystemExit(f"Data path must be one parquet file: {data_path}")
    if read_batch_size <= 0:
        raise SystemExit("--read-batch-size must be positive.")

    parquet_file = pq.ParquetFile(data_path)
    if RAW_TIMESTAMP_COLUMN not in parquet_file.schema.names:
        raise SystemExit(f"Missing timestamp column {RAW_TIMESTAMP_COLUMN!r} in {data_path}")

    sampled_batches: list[pd.DataFrame] = []
    batch_iter = parquet_file.iter_batches(batch_size=read_batch_size)
    if show_progress and tqdm is not None:
        batch_iter = tqdm(
            batch_iter,
            total=math.ceil(parquet_file.metadata.num_rows / read_batch_size),
            desc="Loading 1s snapshots",
            unit="batch",
        )

    for batch in batch_iter:
        df = normalize_columns(batch.to_pandas())
        if TIMESTAMP_COLUMN not in df.columns:
            raise SystemExit(f"Missing timestamp column {TIMESTAMP_COLUMN!r} in {data_path}")

        df = df.dropna(subset=[TIMESTAMP_COLUMN])
        if df.empty:
            continue

        df = df.sort_values(TIMESTAMP_COLUMN)
        df = df.set_index(TIMESTAMP_COLUMN)
        sampled = df.resample("1s").last().ffill()
        if not sampled.empty:
            sampled_batches.append(sampled)

    if not sampled_batches:
        return pd.DataFrame()

    sampled = pd.concat(sampled_batches)
    sampled = sampled.sort_index()
    sampled = sampled.groupby(level=0).last().ffill()
    sampled.index.name = TIMESTAMP_COLUMN
    return sampled


def load_algorithm(path: Path) -> Callable[[TradingState], Any]:
    algorithm_path = path.expanduser().resolve()
    if not algorithm_path.is_file():
        raise SystemExit(f"Algorithm file does not exist: {algorithm_path}")

    sys.path.insert(0, str(algorithm_path.parent))
    spec = importlib.util.spec_from_file_location(algorithm_path.stem, algorithm_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load algorithm file: {algorithm_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[algorithm_path.stem] = module
    spec.loader.exec_module(module)

    if hasattr(module, "Trader"):
        trader = module.Trader()
        if not hasattr(trader, "run") or not callable(trader.run):
            raise SystemExit("Trader exists but does not expose a callable run(state).")
        return trader.run

    if hasattr(module, "run") and callable(module.run):
        return module.run

    raise SystemExit("Algorithm must expose either Trader().run(state) or run(state).")


def build_state(
    timestamp: pd.Timestamp,
    row: pd.Series,
    products: list[str],
    keep_untradable: bool,
    positions: dict[str, float],
) -> TradingState:
    order_depths: dict[str, OrderDepth] = {}

    for product in products:
        if not keep_untradable and (product, "tradable") in row.index and not bool(row[(product, "tradable")]):
            continue

        bid = row.get((product, "bid"))
        ask = row.get((product, "ask"))
        bid_size = row.get((product, "bid_size"), 0)
        ask_size = row.get((product, "ask_size"), 0)

        depth = OrderDepth()
        if is_valid_number(bid) and is_valid_number(bid_size) and bid_size > 0:
            depth.buy_orders[float(bid)] = float(bid_size)
        if is_valid_number(ask) and is_valid_number(ask_size) and ask_size > 0:
            depth.sell_orders[float(ask)] = float(ask_size)

        if depth.buy_orders or depth.sell_orders:
            order_depths[product] = depth

    return TradingState(timestamp=timestamp, order_depths=order_depths, position=positions.copy())


def is_valid_number(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def mid_price(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2


def match_orders(output: Any, state: TradingState, position_limit_usd: float) -> list[dict[str, Any]]:
    if isinstance(output, tuple) and output:
        output = output[0]

    fills = []
    for item in iter_trade_like_items(output):
        order = normalize_trade_item(item)
        if order is None:
            continue

        product = order["product"]
        quantity = float(order["quantity"])
        if quantity == 0:
            continue

        order_price = order.get("price")
        if order_price is None:
            continue

        depth = state.order_depths.get(product)
        if depth is None:
            continue

        fill_price = match_fill_price(quantity, float(order_price), depth)
        if fill_price is None:
            continue

        fill_quantity = clip_quantity_to_position_limit(
            product,
            quantity,
            depth,
            state.position,
            position_limit_usd,
        )
        if fill_quantity == 0:
            continue

        fills.append(
            {
                "timestamp": state.timestamp,
                "product": product,
                "quantity": fill_quantity,
                "price": fill_price,
            }
        )

    return fills


def match_fill_price(quantity: float, order_price: float, depth: OrderDepth) -> float | None:
    if quantity > 0:
        if not depth.sell_orders:
            return None
        best_ask = min(depth.sell_orders)
        return best_ask if order_price >= best_ask else None

    if not depth.buy_orders:
        return None
    best_bid = max(depth.buy_orders)
    return best_bid if order_price <= best_bid else None


def clip_quantity_to_position_limit(
    product: str,
    quantity: float,
    depth: OrderDepth,
    positions: dict[str, float],
    position_limit_usd: float,
) -> float:
    mid = mid_price(depth)
    if mid is None or position_limit_usd <= 0:
        return 0.0

    unit_value_usd = usd_trade_value(product, 1.0, mid)
    if unit_value_usd <= 0:
        return 0.0

    max_abs_position = position_limit_usd / unit_value_usd
    current_position = positions.get(product, 0.0)

    if quantity > 0:
        max_buy_quantity = max_abs_position - current_position
        return max(0.0, min(quantity, max_buy_quantity))

    max_sell_quantity = -max_abs_position - current_position
    return min(0.0, max(quantity, max_sell_quantity))


def iter_trade_like_items(output: Any):
    if output is None:
        return

    if is_dataclass(output):
        yield output
        return

    if isinstance(output, dict):
        if has_trade_keys(output):
            yield output
            return

        for product, value in output.items():
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict):
                        item = {"product": product, **item}
                    elif is_dataclass(item) and not hasattr(item, "product"):
                        item_dict = asdict(item)
                        item = {"product": product, **item_dict}
                    yield item
            elif has_quantity(value):
                yield {"product": product, "quantity": value}
        return

    if isinstance(output, (list, tuple)):
        for item in output:
            yield item


def normalize_trade_item(item: Any) -> dict[str, Any] | None:
    if is_dataclass(item):
        item = asdict(item)

    if isinstance(item, dict):
        product = item.get("product", item.get("symbol"))
        quantity = item.get("quantity", item.get("qty", item.get("size")))
        price = item.get("price")
        if product is None or quantity is None:
            return None
        return {"product": str(product), "quantity": quantity, "price": price}

    if hasattr(item, "product") and hasattr(item, "quantity"):
        return {
            "product": str(item.product),
            "quantity": item.quantity,
            "price": getattr(item, "price", None),
        }

    return None


def has_trade_keys(value: dict[str, Any]) -> bool:
    return ("product" in value or "symbol" in value) and any(key in value for key in ("quantity", "qty", "size"))


def has_quantity(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def apply_trades(
    trades: list[dict[str, Any]],
    cash: float,
    cash_by_product: dict[str, float],
    positions: dict[str, float],
    asset_classes: dict[str, str | None],
) -> float:
    for trade in trades:
        product = trade["product"]
        quantity = float(trade["quantity"])
        price = float(trade["price"])
        trade_value_usd = usd_trade_value(product, quantity, price)
        commission = commission_for_trade(product, quantity, price, asset_classes)
        cash_delta = -(quantity * price) - commission

        cash += cash_delta
        cash_by_product[product] = cash_by_product.get(product, 0.0) + cash_delta
        positions[product] = positions.get(product, 0.0) + quantity
        trade["notional"] = quantity * price
        trade["trade_value_usd"] = trade_value_usd
        trade["commission"] = commission
        trade["cash_after"] = cash
        trade["product_cash_after"] = cash_by_product[product]
        trade["position_after"] = positions[product]

    return cash


def commission_for_trade(
    product: str,
    quantity: float,
    price: float,
    asset_classes: dict[str, str | None],
) -> float:
    if asset_classes.get(product) != "FX":
        return 0.0

    trade_value_usd = usd_trade_value(product, quantity, price)
    tier_1_rate = 0.20 / 10000
    return max(trade_value_usd * tier_1_rate, 2.0)


def usd_trade_value(product: str, quantity: float, price: float) -> float:
    base, separator, quote = product.partition("-")
    if not separator:
        return abs(quantity * price)
    if quote == "USD":
        return abs(quantity * price)
    if base == "USD":
        return abs(quantity)
    return abs(quantity * price)


def run_algorithm_with_log(
    run_algorithm: Callable[[TradingState], Any],
    state: TradingState,
    print_output: bool,
) -> tuple[Any, str]:
    stdout = StringIO()

    with redirect_stdout(stdout):
        output = run_algorithm(state)

    log = stdout.getvalue().rstrip()
    if print_output and log:
        print(log)

    return output, log


def portfolio_mtm(positions: dict[str, float], state: TradingState) -> tuple[float, dict[str, float]]:
    values = {}
    total = 0.0
    for product, quantity in positions.items():
        depth = state.order_depths.get(product)
        mid = mid_price(depth) if depth is not None else None
        if mid is None:
            value = 0.0
        else:
            value = quantity * mid
        values[product] = value
        total += value
    return total, values


def product_pnls(cash_by_product: dict[str, float], mtm_by_product: dict[str, float]) -> dict[str, float]:
    products = set(cash_by_product) | set(mtm_by_product)
    return {
        product: cash_by_product.get(product, 0.0) + mtm_by_product.get(product, 0.0)
        for product in products
    }


def run_backtest(
    run_algorithm: Callable[[TradingState], Any],
    snapshots: pd.DataFrame,
    products: list[str],
    asset_classes: dict[str, str | None],
    limit: int | None,
    keep_untradable: bool,
    show_progress: bool,
    print_output: bool,
    position_limit_usd: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeline = []
    trades_log = []
    cash = 0.0
    cash_by_product = {product: 0.0 for product in products}
    positions = {product: 0.0 for product in products}
    total = min(len(snapshots), limit) if limit is not None else len(snapshots)

    rows = snapshots.iterrows()
    if show_progress and tqdm is not None:
        rows = tqdm(rows, total=total, desc="Backtesting", unit="state")

    for index, (timestamp, row) in enumerate(rows, start=1):
        if limit is not None and index > limit:
            break

        state = build_state(timestamp, row, products, keep_untradable, positions)
        output, lambda_log = run_algorithm_with_log(run_algorithm, state, print_output)
        trades = match_orders(output, state, position_limit_usd)
        cash = apply_trades(trades, cash, cash_by_product, positions, asset_classes)
        mtm, mtm_by_product = portfolio_mtm(positions, state)
        pnl_by_product = product_pnls(cash_by_product, mtm_by_product)
        pnl = sum(pnl_by_product.values())

        for trade in trades:
            trade["pnl_after"] = pnl
            trade["product_pnl_after"] = pnl_by_product.get(trade["product"], 0.0)
        trades_log.extend(trades)

        timeline.append(
            {
                "timestamp": timestamp,
                "cash": cash,
                "mtm": mtm,
                "pnl": pnl,
                "lambda_log": lambda_log,
                "cash_by_product": cash_by_product.copy(),
                "positions": positions.copy(),
                "mtm_by_product": mtm_by_product,
                "pnl_by_product": pnl_by_product,
            }
        )

        if (not show_progress or tqdm is None) and (index == 1 or index == total or index % 1000 == 0):
            print(f"Ran {index}/{total} states at {timestamp} | pnl={pnl:.4f}")

    return timeline, trades_log


def build_orderbook_log(
    snapshots: pd.DataFrame,
    products: list[str],
    timeline: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    timeline_by_timestamp = {entry["timestamp"]: entry for entry in timeline}

    for timestamp, snapshot in snapshots.iloc[: len(timeline)].iterrows():
        pnl_entry = timeline_by_timestamp[timestamp]
        row = {
            "timestamp": timestamp,
            "cash": pnl_entry["cash"],
            "mtm": pnl_entry["mtm"],
            "pnl": pnl_entry["pnl"],
            "lambda_log": pnl_entry["lambda_log"],
        }

        for product in products:
            bid = snapshot.get((product, "bid"))
            ask = snapshot.get((product, "ask"))
            bid_size = snapshot.get((product, "bid_size"))
            ask_size = snapshot.get((product, "ask_size"))
            mid = (float(bid) + float(ask)) / 2 if is_valid_number(bid) and is_valid_number(ask) else None

            row[f"{product}_bid"] = float(bid) if is_valid_number(bid) else None
            row[f"{product}_ask"] = float(ask) if is_valid_number(ask) else None
            row[f"{product}_bid_size"] = float(bid_size) if is_valid_number(bid_size) else None
            row[f"{product}_ask_size"] = float(ask_size) if is_valid_number(ask_size) else None
            row[f"{product}_mid"] = mid
            row[f"{product}_cash"] = pnl_entry["cash_by_product"].get(product, 0.0)
            row[f"{product}_position"] = pnl_entry["positions"].get(product, 0.0)
            row[f"{product}_mtm"] = pnl_entry["mtm_by_product"].get(product, 0.0)
            row[f"{product}_pnl"] = pnl_entry["pnl_by_product"].get(product, 0.0)

        rows.append(row)

    return pd.DataFrame(rows)


def write_logs(
    snapshots: pd.DataFrame,
    products: list[str],
    timeline: list[dict[str, Any]],
    trades_log: list[dict[str, Any]],
    out_dir: Path,
    prefix: str,
    write_parquets: bool,
    position_limit_usd: float,
) -> tuple[Path, Path | None, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    orderbooks_path = out_dir / f"{prefix}_orderbooks.parquet"
    trades_path = out_dir / f"{prefix}_trades.parquet"
    bundle_path = out_dir / f"{prefix}_bundle.json"

    orderbooks_df = build_orderbook_log(snapshots, products, timeline)
    trades_df = pd.DataFrame(
        trades_log,
        columns=[
            "timestamp",
            "product",
            "quantity",
            "price",
            "notional",
            "trade_value_usd",
            "commission",
            "cash_after",
            "product_cash_after",
            "position_after",
            "pnl_after",
            "product_pnl_after",
        ],
    )

    bundle = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "products": products,
        "position_limit_usd": position_limit_usd,
        "commission_model": {
            "FX": {
                "tier": 1,
                "rate_basis_points": 0.20,
                "minimum_usd_per_order": 2.0,
            }
        },
        "orderbooks": dataframe_records(orderbooks_df),
        "trades": dataframe_records(trades_df),
    }
    bundle_path.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False), encoding="utf-8")

    if write_parquets:
        orderbooks_df.to_parquet(orderbooks_path, index=False)
        trades_df.to_parquet(trades_path, index=False)
        return bundle_path, orderbooks_path, trades_path

    return bundle_path, None, None


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for record in df.to_dict(orient="records"):
        records.append({key: json_value(value) for key, value in record.items()})
    return records


def json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> int:
    args = parse_args()
    print("Loading data...")
    snapshots = load_sampled_snapshots(
        args.data.expanduser().resolve(),
        args.read_batch_size,
        not args.no_progress,
    )
    products = resolve_products(snapshots, args)
    if not products:
        raise SystemExit("No products found. Expected per-product bid/ask columns.")

    run_algorithm = load_algorithm(args.algorithm)
    print(f"Loaded {len(snapshots)} 1-second snapshots for {len(products)} products.")
    timeline, trades_log = run_backtest(
        run_algorithm,
        snapshots,
        products,
        product_asset_classes(snapshots, products),
        args.limit,
        args.keep_untradable,
        not args.no_progress,
        args.print_output,
        args.position_limit_usd,
    )
    prefix = args.name or f"{args.algorithm.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle_path, orderbooks_path, trades_path = write_logs(
        snapshots,
        products,
        timeline,
        trades_log,
        args.out_dir.expanduser().resolve(),
        prefix,
        args.write_parquets,
        args.position_limit_usd,
    )
    print(f"Backtest complete. Wrote single-file bundle to {bundle_path}")
    if orderbooks_path is not None and trades_path is not None:
        print(f"Backtest complete. Wrote orderbooks/MTM to {orderbooks_path}")
        print(f"Backtest complete. Wrote trades to {trades_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
