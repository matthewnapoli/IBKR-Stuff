from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "ALL_DATA" / "ticks"
OUT_DIR = ROOT / "ALL_DATA" / "cleaned"
TIMESTAMP_COL = "('base', 'timestamp')"


def stable_value(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ET).isoformat()
        return value.astimezone(ET).isoformat()
    return value


def row_hash(row: dict[str, object]) -> str:
    payload = json.dumps(row, default=stable_value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def day_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ET).strftime("%Y%m%d")
        return value.astimezone(ET).strftime("%Y%m%d")
    return None


def unique_row_indices(table: pa.Table, seen_hashes: set[str]) -> list[int]:
    keep: list[int] = []

    for idx, row in enumerate(table.to_pylist()):
        digest = row_hash(row)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        keep.append(idx)

    return keep


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing raw tick directory: {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now(ET).strftime("%Y%m%d")
    writer: pq.ParquetWriter | None = None
    current_day: str | None = None
    seen_hashes: set[str] = set()

    try:
        for path in sorted(RAW_DIR.glob("ticks_*.parquet")):
            file_day = path.stem.split("_")[1]
            if file_day == today_str:
                continue

            table = pq.read_table(path)
            if TIMESTAMP_COL not in table.column_names:
                raise KeyError(f"{TIMESTAMP_COL} not found in {path}")

            timestamp_values = table[TIMESTAMP_COL].to_pylist()
            timestamp_strings = [day_string(value) for value in timestamp_values]
            days_in_file = sorted({day for day in timestamp_strings if day and day != today_str})

            for day in days_in_file:
                if day != current_day:
                    if writer is not None:
                        writer.close()
                    current_day = day
                    seen_hashes = set()
                    writer = None

                keep_rows = [idx for idx, row_day in enumerate(timestamp_strings) if row_day == day]
                if not keep_rows:
                    continue

                day_table = table.take(pa.array(keep_rows, type=pa.int32()))
                if day_table.num_rows == 0:
                    continue

                keep = unique_row_indices(day_table, seen_hashes)
                if not keep:
                    continue

                unique_table = day_table.take(pa.array(keep, type=pa.int32()))

                if writer is None:
                    output_path = OUT_DIR / f"ticks_{day}.parquet"
                    writer = pq.ParquetWriter(output_path, unique_table.schema)

                writer.write_table(unique_table)
                print(f"{day}: appended {unique_table.num_rows:,} rows from {path.name}")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
