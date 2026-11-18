from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd


FULL_TABLE_SCAN_VERSION = "csv-time-column-chunk-scan-v2"

_CSV_SCAN_READ_OPTIONS = {
    "sep",
    "delimiter",
    "header",
    "names",
    "index_col",
    "usecols",
    "dtype",
    "engine",
    "skiprows",
    "skipfooter",
    "nrows",
    "na_values",
    "keep_default_na",
    "na_filter",
    "verbose",
    "skip_blank_lines",
    "parse_dates",
    "infer_datetime_format",
    "keep_date_col",
    "date_parser",
    "date_format",
    "dayfirst",
    "cache_dates",
    "iterator",
    "chunksize",
    "compression",
    "thousands",
    "decimal",
    "lineterminator",
    "quotechar",
    "quoting",
    "doublequote",
    "escapechar",
    "comment",
    "encoding",
    "encoding_errors",
    "dialect",
    "on_bad_lines",
    "delim_whitespace",
    "low_memory",
    "memory_map",
    "float_precision",
    "storage_options",
}

_CSV_SCAN_CONTROL_OPTIONS = {
    "usecols",
    "dtype",
    "nrows",
    "iterator",
    "chunksize",
    "on_bad_lines",
}


def normalized_csv_scan_read_options(read_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Keep structural CSV reader settings while excluding sampling controls."""
    return {
        key: value
        for key, value in dict(read_kwargs or {}).items()
        if key in _CSV_SCAN_READ_OPTIONS
        and key not in _CSV_SCAN_CONTROL_OPTIONS
        and value is not None
    }


def csv_scan_read_options_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    source = metadata if isinstance(metadata, dict) else {}
    contract = source.get("csv_read_contract")
    if isinstance(contract, dict) and isinstance(contract.get("pandas_kwargs"), dict):
        return dict(contract["pandas_kwargs"])
    dialect = source.get("csv_dialect") if isinstance(source.get("csv_dialect"), dict) else {}
    return {
        "sep": dialect.get("sep", ","),
        "encoding": source.get("csv_encoding", "utf-8-sig"),
        "engine": dialect.get("engine"),
        "decimal": dialect.get("decimal"),
    }


def csv_scan_read_options_fingerprint(read_kwargs: dict[str, Any] | None) -> str:
    options = normalized_csv_scan_read_options(read_kwargs)
    encoded = json.dumps(options, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verified_full_scan_population(
    scan: Any,
    *,
    source_fingerprint: str,
    read_options_fingerprint: str | None = None,
) -> bool:
    """Validate exact row-count evidence independently from timestamp extrema."""
    if not isinstance(scan, dict):
        return False
    if (
        scan.get("status") != "completed"
        or scan.get("scanner_version") not in {
            FULL_TABLE_SCAN_VERSION,
            "csv-time-column-chunk-scan-v1",
        }
        or not source_fingerprint
        or scan.get("source_fingerprint") != source_fingerprint
        or not str(scan.get("column") or "").strip()
        or type(scan.get("rows_scanned")) is not int
        or scan["rows_scanned"] < 0
        or type(scan.get("valid_timestamp_rows")) is not int
        or type(scan.get("invalid_timestamp_rows")) is not int
        or scan["valid_timestamp_rows"] < 0
        or scan["invalid_timestamp_rows"] < 0
        or scan["valid_timestamp_rows"] + scan["invalid_timestamp_rows"] != scan["rows_scanned"]
    ):
        return False
    if (
        read_options_fingerprint is not None
        and scan.get("read_options_fingerprint") != read_options_fingerprint
    ):
        return False

    ranges = scan.get("timestamp_ranges_by_timezone_awareness")
    if isinstance(ranges, dict):
        range_rows = 0
        for item in ranges.values():
            if not isinstance(item, dict) or type(item.get("rows")) is not int or item["rows"] < 0:
                return False
            range_rows += item["rows"]
            if item["rows"] and not (
                item.get("minimum_timestamp") and item.get("maximum_timestamp")
            ):
                return False
        if range_rows != scan["valid_timestamp_rows"]:
            return False
        if scan["valid_timestamp_rows"] == 0 and (
            scan.get("minimum_timestamp") is not None
            or scan.get("maximum_timestamp") is not None
        ):
            return False
        return True

    # Older v1 evidence did not always include per-timezone ranges. Its row
    # population is still exact when timestamp accounting reconciles.
    if scan["valid_timestamp_rows"] == 0:
        return scan.get("minimum_timestamp") is None and scan.get("maximum_timestamp") is None
    return bool(scan.get("minimum_timestamp") and scan.get("maximum_timestamp"))


def scan_csv_time_column(
    path: Path,
    *,
    column: str,
    read_kwargs: dict[str, Any] | None = None,
    chunk_rows: int = 50_000,
) -> dict[str, Any]:
    """Count every CSV record and validate one requested time column in chunks."""
    started = time.perf_counter()
    stat = path.stat()
    options = normalized_csv_scan_read_options(read_kwargs)
    options.update(
        {
            "usecols": [column],
            "chunksize": max(1, int(chunk_rows)),
            "dtype": {column: "string"},
            "on_bad_lines": "error",
        }
    )

    rows = 0
    valid = 0
    invalid = 0
    chunks = 0
    ranges: dict[str, dict[str, pd.Timestamp | int]] = {
        "naive": {"minimum": None, "maximum": None, "rows": 0},
        "aware_utc": {"minimum": None, "maximum": None, "rows": 0},
    }
    for frame in pd.read_csv(path, **options):
        chunks += 1
        rows += len(frame)
        raw_times = frame[column]
        parsed = pd.to_datetime(raw_times, format="mixed", errors="coerce")
        valid += int(parsed.notna().sum())
        invalid += int(parsed.isna().sum())
        for value in parsed.dropna().array:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                group = ranges["naive"]
                normalized = timestamp
            else:
                group = ranges["aware_utc"]
                normalized = timestamp.tz_convert("UTC")
            current_min = group["minimum"]
            current_max = group["maximum"]
            group["minimum"] = normalized if current_min is None or normalized < current_min else current_min
            group["maximum"] = normalized if current_max is None or normalized > current_max else current_max
            group["rows"] = int(group["rows"]) + 1

    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("source_changed_during_scan")
    observed_groups = [key for key, value in ranges.items() if value["rows"]]
    mixed_awareness = len(observed_groups) > 1
    sole_group = ranges[observed_groups[0]] if len(observed_groups) == 1 else None
    timestamp_ranges = {
        key: {
            "rows": int(value["rows"]),
            "minimum_timestamp": value["minimum"].isoformat() if value["minimum"] is not None else None,
            "maximum_timestamp": value["maximum"].isoformat() if value["maximum"] is not None else None,
        }
        for key, value in ranges.items()
    }
    return {
        "schema_version": "autorealize.csv_full_table_scan.v1",
        "scanner_version": FULL_TABLE_SCAN_VERSION,
        "status": "completed",
        "scope": "all_csv_records_one_time_column",
        "column": column,
        "read_options": normalized_csv_scan_read_options(read_kwargs),
        "read_options_fingerprint": csv_scan_read_options_fingerprint(read_kwargs),
        "rows_scanned": rows,
        "valid_timestamp_rows": valid,
        "invalid_timestamp_rows": invalid,
        "minimum_timestamp": sole_group["minimum"].isoformat() if sole_group and sole_group["minimum"] is not None else None,
        "maximum_timestamp": sole_group["maximum"].isoformat() if sole_group and sole_group["maximum"] is not None else None,
        "timestamp_ranges_by_timezone_awareness": timestamp_ranges,
        "chunk_rows": max(1, int(chunk_rows)),
        "chunks_scanned": chunks,
        "timestamp_parse_format": "mixed",
        "file_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "time_zone_policy": (
            "preserve naive wall-clock timestamps; normalize explicit offsets to UTC for comparison; "
            "report separate ranges when naive and aware values are mixed"
        ),
        "mixed_timezone_awareness": mixed_awareness,
    }
