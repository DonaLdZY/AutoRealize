from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ..utils.json_table import read_json_as_table


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_ratio: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)
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
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gb18030")
    if suffix == ".json":
        df, _ = read_json_as_table(
            path,
            sep=json_flatten_sep,
            max_level=json_flatten_max_level,
            keep_raw_nested_columns=json_keep_raw_nested_columns,
        )
        return df
    return pd.read_excel(path)


def _numeric_stats(series: pd.Series) -> dict[str, float | int | bool]:
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
    return {
        "min": dt.min().isoformat(),
        "max": dt.max().isoformat(),
        "range_days": str((dt.max() - dt.min()).days),
    }


def profile_dataframe(df: pd.DataFrame, top_k: int = 12) -> list[ColumnProfile]:
    def _safe_unique_count(series: pd.Series) -> int:
        try:
            return int(series.nunique(dropna=True))
        except TypeError:
            # list/dict 等不可哈希对象退化为 JSON 字符串后再统计唯一值
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
        null_ratio = float(s.isna().mean())
        unique_count = _safe_unique_count(s)
        sample_values = s.dropna().astype(str).head(top_k).tolist()
        profile = ColumnProfile(
            name=str(col),
            dtype=str(s.dtype),
            null_ratio=null_ratio,
            unique_count=unique_count,
            sample_values=sample_values,
        )
        profile.numeric_stats = _numeric_stats(s)
        profile.quantiles = _numeric_quantiles(s)
        profile.datetime_stats = _datetime_stats(s)
        # 检查“主要是数字但掺杂字符串”的情况
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
