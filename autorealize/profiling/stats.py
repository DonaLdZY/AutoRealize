from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .csv_utils import read_csv_auto
from ..utils.json_table import read_json_as_table


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_ratio: float
    unique_count: int
    row_count: int = 0
    null_count: int = 0
    non_null_count: int = 0
    logical_type: str = "unknown"
    numeric_parse_ratio: float = 0.0
    datetime_parse_ratio: float = 0.0
    format_hints: list[str] = field(default_factory=list)
    sample_values: list[Any] = field(default_factory=list)
    top_values: list[str] = field(default_factory=list)
    value_pattern_hints: list[str] = field(default_factory=list)
    numeric_stats: dict[str, float | int | bool] = field(default_factory=dict)
    abnormal_tokens: list[str] = field(default_factory=list)
    quantiles: dict[str, float] = field(default_factory=dict)
    datetime_stats: dict[str, str] = field(default_factory=dict)


def read_table(
    path: Path,
    *,
    json_flatten_sep: str = "__",
    json_flatten_max_level: int | None = None,
    json_keep_raw_nested_columns: bool = False,
    max_rows: int | None = None,
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    """读取可表格化文件。

    max_rows 用于数据认知预览的安全采样，避免数 GB CSV 在认知阶段被全量加载。
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_auto(path, nrows=max_rows)
    if suffix == ".json":
        df, _ = read_json_as_table(
            path,
            sep=json_flatten_sep,
            max_level=json_flatten_max_level,
            keep_raw_nested_columns=json_keep_raw_nested_columns,
        )
        return df.head(max_rows) if max_rows is not None else df
    return pd.read_excel(path, sheet_name=sheet_name, nrows=max_rows)


def table_probe_sample_rows(path: Path, *, configured_rows: int | None, large_threshold_bytes: int) -> int | None:
    """返回字段统计应读取的行数。None 表示允许全量读取。"""
    if configured_rows is None:
        return None
    try:
        # The current policy always applies the configured profiling cap, but we
        # still inspect the threshold so downstream metadata can distinguish
        # ordinary capped profiling from genuinely large-file protection.
        _ = path.stat().st_size > max(1, int(large_threshold_bytes))
    except Exception:
        pass
    return max(1, int(configured_rows))


def table_sampling_metadata(
    path: Path,
    *,
    configured_rows: int | None,
    large_threshold_bytes: int,
    rows_read: int | None = None,
) -> dict[str, Any]:
    """Describe the deterministic profiling sampling policy for a table."""
    try:
        file_size = int(path.stat().st_size)
    except Exception:
        file_size = 0
    threshold = max(1, int(large_threshold_bytes))
    max_rows = table_probe_sample_rows(
        path,
        configured_rows=configured_rows,
        large_threshold_bytes=large_threshold_bytes,
    )
    is_large_file = bool(file_size and file_size > threshold)
    if max_rows is None:
        reason = "full_scan_allowed"
    elif is_large_file:
        reason = "large_file_row_cap"
    else:
        reason = "configured_row_cap"
    return {
        "configured_max_rows": max_rows,
        "rows_read": rows_read,
        "file_size_bytes": file_size,
        "large_threshold_bytes": threshold,
        "is_large_file": is_large_file,
        "sampling_reason": reason,
        "sampled": bool(max_rows is not None),
    }


def profile_excel_sheets(
    path: Path,
    *,
    max_rows: int | None,
    top_k: int = 10,
    max_profile_columns: int = 80,
    preview_rows: int = 10,
    large_threshold_bytes: int = 256 * 1024 * 1024,
    full_profile_sheet_threshold: int = 10,
    representatives_per_group: int = 1,
) -> list[dict[str, Any]]:
    """Return compact per-sheet profiles for Excel workbooks.

    Policy:
    - Every sheet gets a lightweight inventory: name, shape/header, dtypes and a
      small preview.
    - If a workbook has only a few sheets and is not large, every sheet is
      deeply profiled with a full read.
    - Otherwise sheets with the same/near-same headers are grouped, and only
      representatives are deeply profiled. Non-representatives keep lightweight
      inventory so downstream agents still know the full workbook boundary.
    """
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        return []
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return []
    try:
        sheet_names = [str(x) for x in xls.sheet_names if str(x).strip()]
        sheet_count = len(sheet_names)
        try:
            file_size = int(path.stat().st_size)
        except Exception:
            file_size = 0
        is_large = bool(file_size and file_size > max(1, int(large_threshold_bytes)))
        full_profile_all = sheet_count <= max(1, int(full_profile_sheet_threshold)) and not is_large
        shapes = _excel_sheet_shapes(path)

        inventories: list[dict[str, Any]] = []
        for sheet in sheet_names:
            raw_preview: list[list[Any]] = []
            raw_preview_rows_used = 0
            raw_preview_error = ""
            try:
                raw_df = xls.parse(sheet_name=sheet, header=None, nrows=max(1, int(preview_rows)))
                raw_preview = _raw_rows(raw_df.head(max(1, int(preview_rows))))
                raw_preview_rows_used = int(len(raw_df))
            except Exception as exc:  # noqa: BLE001
                raw_preview_error = str(exc)[:500]
            try:
                preview_df = xls.parse(sheet_name=sheet, nrows=max(1, int(preview_rows)))
            except Exception as exc:  # noqa: BLE001
                item = {
                    "sheet_name": sheet,
                    "error": str(exc)[:500],
                    "raw_preview": raw_preview,
                    "raw_preview_rows_used": raw_preview_rows_used,
                }
                if raw_preview_error:
                    item["raw_preview_error"] = raw_preview_error
                inventories.append(item)
                continue
            shape = shapes.get(sheet)
            shape_estimated = False
            if not shape:
                shape = [int(preview_df.shape[0]), int(preview_df.shape[1])]
                shape_estimated = True
            columns = [str(c) for c in preview_df.columns.tolist()]
            inventories.append(
                {
                    "sheet_name": sheet,
                    "shape": shape,
                    "shape_estimated": shape_estimated,
                    "preview_rows_used": int(len(preview_df)),
                    "columns": columns,
                    "dtypes": {str(k): str(v) for k, v in preview_df.dtypes.to_dict().items()},
                    "preview": _records(preview_df.head(max(1, int(preview_rows)))),
                    "raw_preview": raw_preview,
                    "raw_preview_rows_used": raw_preview_rows_used,
                    **({"raw_preview_error": raw_preview_error} if raw_preview_error else {}),
                    "sheet_name_pattern": _normalize_sheet_name(sheet),
                    "header_signature": _header_signature(columns),
                }
            )

        groups = _group_excel_sheet_inventories(inventories)
        representative_sheets: set[str] = set()
        for group in groups:
            reps = group["sheets"][: max(1, int(representatives_per_group))]
            representative_sheets.update(str(x) for x in reps)
        if sheet_names:
            representative_sheets.add(sheet_names[0])

        out: list[dict[str, Any]] = []
        for inv in inventories:
            sheet = str(inv.get("sheet_name", "") or "")
            if inv.get("error"):
                out.append(inv)
                continue
            group = next((g for g in groups if sheet in g.get("sheets", [])), {})
            is_representative = sheet in representative_sheets
            should_deep_profile = bool(full_profile_all or is_representative)
            profile_rows_limit = None if full_profile_all else max_rows
            entry = {
                **inv,
                "workbook_sheet_count": sheet_count,
                "workbook_file_size_bytes": file_size,
                "workbook_is_large": is_large,
                "profile_policy": "full_all_sheets" if full_profile_all else "representative_by_sheet_group",
                "sheet_group_id": group.get("group_id", ""),
                "sheet_group_size": group.get("sheet_count", 1),
                "sheet_group_sheets": group.get("sheets", [sheet])[:20],
                "sheet_group_representative": group.get("representative", sheet),
                "is_deep_profiled": should_deep_profile,
                "profile_rows_limit": profile_rows_limit,
            }
            if not should_deep_profile:
                out.append(entry)
                continue
            try:
                df = xls.parse(sheet_name=sheet, nrows=profile_rows_limit)
            except Exception as exc:  # noqa: BLE001
                entry["profile_error"] = str(exc)[:500]
                out.append(entry)
                continue
            use_cols = list(df.columns[:max_profile_columns])
            profiles = profile_dataframe(df[use_cols], top_k=top_k) if use_cols else []
            entry.update(
                {
                    "shape_profiled": [int(df.shape[0]), int(df.shape[1])],
                    "shape_sampled": [int(df.shape[0]), int(df.shape[1])],
                    "profiled_column_count": len(use_cols),
                    "column_profiles": [column_profile_to_dict(p) for p in profiles],
                }
            )
            out.append(entry)
        return out
    finally:
        xls.close()


def excel_sheet_groups_from_profiles(sheet_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return workbook-level sheet groups from per-sheet profiles."""
    return _group_excel_sheet_inventories(sheet_profiles)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(k): _json_safe(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _raw_rows(df: pd.DataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for values in df.itertuples(index=False, name=None):
        rows.append([_json_safe(v) for v in values])
    return rows


def _json_safe(value: Any) -> Any:
    try:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Timedelta):
            return str(value)
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def _excel_sheet_shapes(path: Path) -> dict[str, list[int]]:
    """Best-effort sheet shapes without full pandas reads."""
    if path.suffix.lower() != ".xlsx":
        return {}
    try:
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        shapes: dict[str, list[int]] = {}
        for ws in wb.worksheets:
            rows = max(0, int(ws.max_row or 0) - 1)
            cols = max(0, int(ws.max_column or 0))
            shapes[str(ws.title)] = [rows, cols]
        wb.close()
        return shapes
    except Exception:
        return {}


def _normalize_sheet_name(name: str) -> str:
    value = str(name or "").strip().lower()
    value = re.sub(r"\d{4}[-_/年]?\d{1,2}[-_/月]?\d{0,2}日?", "{date}", value)
    value = re.sub(r"\d+", "{num}", value)
    value = re.sub(r"\s+", "", value)
    return value or "{sheet}"


def _header_signature(columns: list[str]) -> str:
    normalized = [re.sub(r"\s+", "", str(c or "").strip().lower()) for c in columns]
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _header_similarity(sig_a: str, sig_b: str) -> float:
    try:
        a = set(json.loads(sig_a))
        b = set(json.loads(sig_b))
    except Exception:
        return 0.0
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _group_excel_sheet_inventories(inventories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for inv in inventories:
        if inv.get("error"):
            continue
        sig = str(inv.get("header_signature", "") or "")
        pattern = str(inv.get("sheet_name_pattern", "") or "{sheet}")
        sheet = str(inv.get("sheet_name", "") or "")
        placed = False
        for group in groups:
            same_pattern = pattern == group.get("sheet_name_pattern")
            similar_header = _header_similarity(sig, str(group.get("header_signature", "") or "")) >= 0.75
            if similar_header and (same_pattern or len(group.get("sheets", [])) >= 1):
                group["sheets"].append(sheet)
                group["sheet_count"] = len(group["sheets"])
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "group_id": f"sheet_group_{len(groups) + 1}",
                    "sheet_name_pattern": pattern,
                    "header_signature": sig,
                    "representative": sheet,
                    "sheets": [sheet],
                    "sheet_count": 1,
                    "columns": [str(c) for c in inv.get("columns", [])],
                }
            )
    return groups


def _numeric_stats(series: pd.Series) -> dict[str, float | int | bool]:
    if pd.api.types.is_bool_dtype(series):
        return {}
    coerced = pd.to_numeric(series, errors="coerce")
    valid = coerced.dropna()
    valid = valid[np.isfinite(valid.to_numpy())]
    if valid.empty:
        return {}
    arr = valid.to_numpy()
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "var": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "has_inf": bool(np.isinf(arr).any()),
    }


def _numeric_quantiles(series: pd.Series) -> dict[str, float]:
    if pd.api.types.is_bool_dtype(series):
        return {}
    coerced = pd.to_numeric(series, errors="coerce")
    valid = coerced.dropna()
    valid = valid[np.isfinite(valid.to_numpy())]
    if valid.empty:
        return {}
    return {
        "q1": float(valid.quantile(0.25)),
        "q3": float(valid.quantile(0.75)),
        "p05": float(valid.quantile(0.05)),
        "p95": float(valid.quantile(0.95)),
    }


def _datetime_stats(series: pd.Series) -> dict[str, str]:
    # 仅对“可能具有日期语义”的列做日期解析，避免把纯数值列误判为时间戳。
    if pd.api.types.is_numeric_dtype(series):
        return {}
    if not (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    ):
        return {}
    sample = series.dropna().astype(str).head(30).tolist()
    if not sample:
        return {}
    date_like_hits = 0
    for v in sample:
        s = v.strip()
        if "-" in s or "/" in s or ":" in s or "年" in s or "月" in s or "日" in s:
            date_like_hits += 1
    if date_like_hits < max(2, int(len(sample) * 0.35)):
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dt = pd.to_datetime(series, errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        return {}
    sorted_dt = dt.sort_values()
    granularity = ""
    if len(sorted_dt) >= 2:
        diffs = sorted_dt.diff().dropna()
        if not diffs.empty:
            median_seconds = float(diffs.dt.total_seconds().median())
            if median_seconds <= 1:
                granularity = "second_or_finer"
            elif median_seconds <= 60:
                granularity = "minute"
            elif median_seconds <= 3600:
                granularity = "hour"
            elif median_seconds <= 86400:
                granularity = "day"
            elif median_seconds <= 86400 * 31:
                granularity = "month_like"
            else:
                granularity = "coarse"
    return {
        "min": dt.min().isoformat(),
        "max": dt.max().isoformat(),
        "range_days": str((dt.max() - dt.min()).days),
        "granularity": granularity,
    }


def _datetime_parse_ratio(series: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(series):
        return 0.0
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.astype(str).head(200)
    date_like = sample.map(lambda v: any(tok in str(v) for tok in ["-", "/", ":", "年", "月", "日"]))
    if float(date_like.mean()) < 0.35:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(non_null, errors="coerce")
    return round(float(parsed.notna().mean()), 6)


def _format_hints(series: pd.Series) -> list[str]:
    hints: list[str] = []
    if pd.api.types.is_integer_dtype(series):
        hints.append("integer_storage")
    elif pd.api.types.is_float_dtype(series):
        hints.append("float_storage")
    elif pd.api.types.is_bool_dtype(series):
        hints.append("boolean_storage")
    elif pd.api.types.is_datetime64_any_dtype(series):
        hints.append("datetime_storage")
    elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        hints.append("text_storage")

    vals = [str(x).strip() for x in series.dropna().astype(str).head(200).tolist() if str(x).strip()]
    if not vals:
        return hints

    def _ratio(pred) -> float:
        return sum(1 for v in vals if pred(v)) / max(1, len(vals))

    if _ratio(lambda v: bool(__import__("re").fullmatch(r"[-+]?\d+", v))) >= 0.8:
        hints.append("integer_string")
    if _ratio(lambda v: bool(__import__("re").fullmatch(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", v))) >= 0.8:
        hints.append("numeric_string")
    if _ratio(lambda v: "%" in v) >= 0.5:
        hints.append("percentage_string")
    if _ratio(lambda v: "," in v and any(ch.isdigit() for ch in v)) >= 0.5:
        hints.append("comma_number_string")
    if _ratio(lambda v: any(tok in v for tok in ["-", "/", ":", "年", "月", "日"])) >= 0.35:
        hints.append("date_time_like_string")
    if _ratio(lambda v: v.lower() in {"true", "false", "yes", "no", "y", "n", "0", "1"}) >= 0.8:
        hints.append("boolean_like_string")
    return list(dict.fromkeys(hints))


def _logical_type(series: pd.Series, numeric_ratio: float, datetime_ratio: float, unique_count: int) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series) or datetime_ratio >= 0.8:
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if numeric_ratio >= 0.95:
        return "numeric_string"
    if numeric_ratio >= 0.6:
        return "mixed_numeric_text"
    if unique_count <= 20:
        return "categorical"
    return "text"


def column_profile_to_dict(profile: ColumnProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "dtype": profile.dtype,
        "logical_type": profile.logical_type,
        "row_count": profile.row_count,
        "null_count": profile.null_count,
        "non_null_count": profile.non_null_count,
        "null_ratio": profile.null_ratio,
        "unique_count": profile.unique_count,
        "numeric_parse_ratio": profile.numeric_parse_ratio,
        "datetime_parse_ratio": profile.datetime_parse_ratio,
        "format_hints": profile.format_hints,
        "sample_values": profile.sample_values,
        "top_values": profile.top_values,
        "value_pattern_hints": profile.value_pattern_hints,
        "numeric_stats": profile.numeric_stats,
        "quantiles": profile.quantiles,
        "datetime_stats": profile.datetime_stats,
        "abnormal_tokens": profile.abnormal_tokens,
    }


def _value_pattern_hints(series: pd.Series) -> list[str]:
    vals = [str(x).strip() for x in series.dropna().astype(str).head(120).tolist() if str(x).strip()]
    if not vals:
        return []
    hints: list[str] = []

    def _ratio(pred) -> float:
        hit = sum(1 for v in vals if pred(v))
        return hit / max(1, len(vals))

    if _ratio(lambda v: bool(pd.notna(v)) and bool(__import__("re").match(r"^\d+\s*型[\u4e00-\u9fffA-Za-z]+$", v))) >= 0.5:
        hints.append("numbered_type_enum")
    # 脱敏编号：如“粤B****8”或“*A23**”
    if _ratio(lambda v: ("*" in v or "×" in v or "X" in v) and any(ch.isdigit() for ch in v)) >= 0.4:
        hints.append("masked_plate_like")
    # 代码标识：大写字母+数字+连接符
    if _ratio(lambda v: bool(__import__("re").match(r"^[A-Za-z0-9_-]{6,}$", v))) >= 0.6:
        hints.append("code_like")
    # 行政区文本
    if _ratio(lambda v: any(k in v for k in ["省", "市", "区", "县", "镇", "街道"])) >= 0.4:
        hints.append("region_name_like")
    return hints


def profile_dataframe(df: pd.DataFrame, top_k: int = 12) -> list[ColumnProfile]:
    def _safe_unique_count(series: pd.Series) -> int:
        try:
            return int(series.nunique(dropna=True))
        except TypeError:
            # list/dict 等不可哈希对象退化为 JSON 字符串后再统计唯一值。
            def _norm(v: Any) -> Any:
                if isinstance(v, (list, dict)):
                    try:
                        return json.dumps(v, ensure_ascii=False, sort_keys=True)
                    except TypeError:
                        return str(v)
                return v

            normalized = series.map(_norm)
            return int(normalized.nunique(dropna=True))

    profiles: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        row_count = int(s.shape[0])
        null_count = int(s.isna().sum())
        non_null_count = int(row_count - null_count)
        null_ratio = float(s.isna().mean())
        unique_count = _safe_unique_count(s)
        sample_values = s.dropna().astype(str).head(top_k).tolist()
        numeric_parse_ratio = 0.0
        if non_null_count > 0 and not pd.api.types.is_bool_dtype(s):
            numeric_parse_ratio = round(float(pd.to_numeric(s.dropna(), errors="coerce").notna().mean()), 6)
        datetime_parse_ratio = _datetime_parse_ratio(s)
        profile = ColumnProfile(
            name=str(col),
            dtype=str(s.dtype),
            null_ratio=null_ratio,
            unique_count=unique_count,
            row_count=row_count,
            null_count=null_count,
            non_null_count=non_null_count,
            numeric_parse_ratio=numeric_parse_ratio,
            datetime_parse_ratio=datetime_parse_ratio,
            format_hints=_format_hints(s),
            sample_values=sample_values,
        )
        profile.numeric_stats = _numeric_stats(s)
        profile.quantiles = _numeric_quantiles(s)
        profile.datetime_stats = _datetime_stats(s)
        profile.logical_type = _logical_type(s, numeric_parse_ratio, datetime_parse_ratio, unique_count)
        try:
            vc = s.dropna().astype(str).value_counts().head(top_k)
            profile.top_values = [f"{idx}({int(cnt)})" for idx, cnt in vc.items()]
        except Exception:
            profile.top_values = []
        profile.value_pattern_hints = _value_pattern_hints(s)
        # 检查“主要是数字但掺杂字符串”的情况。
        if not pd.api.types.is_bool_dtype(s):
            coerced = pd.to_numeric(s, errors="coerce")
            numeric_ratio = float(coerced.notna().mean())
            if numeric_ratio > 0.6:
                mask = coerced.isna() & s.notna()
                bad_tokens = s[mask].astype(str).value_counts().head(top_k).index.tolist()
                profile.abnormal_tokens = [str(x) for x in bad_tokens]
        profiles.append(profile)
    return profiles


def dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in df.columns.tolist()],
    }
