from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def _to_dataframe_from_dict_of_lists(
    data: dict[str, Any],
    keep_raw_nested_columns: bool = False,
    sep: str = "__",
) -> pd.DataFrame:
    list_items = {k: v for k, v in data.items() if isinstance(v, list)}
    if not list_items:
        return pd.DataFrame()
    max_len = max((len(v) for v in list_items.values()), default=0)
    if max_len == 0:
        return pd.DataFrame()
    columns: dict[str, list[Any]] = {}
    nested_raw: dict[str, list[Any]] = {}
    for key, values in list_items.items():
        has_nested = any(isinstance(x, (dict, list)) for x in values)
        padded = list(values) + [None] * (max_len - len(values))
        if has_nested:
            if keep_raw_nested_columns:
                nested_raw[f"raw{sep}{key}"] = [
                    json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
                    for x in padded
                ]
            else:
                return pd.DataFrame()
        else:
            columns[str(key)] = padded
    for key, value in data.items():
        if key in list_items:
            continue
        if _is_scalar(value):
            columns[f"root__{key}"] = [value] * max_len
        elif keep_raw_nested_columns:
            columns[f"raw{sep}{key}"] = [json.dumps(value, ensure_ascii=False)] * max_len
    columns.update(nested_raw)
    return pd.DataFrame(columns)


def _append_raw_nested_columns(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    sep: str = "__",
) -> pd.DataFrame:
    if df.empty:
        return df
    raw_keys = set()
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                raw_keys.add(str(k))
    if not raw_keys:
        return df
    out = df.copy()
    for key in sorted(raw_keys):
        col = f"raw{sep}{key}"
        out[col] = [
            json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else None
            for row in rows
        ]
    return out


def json_to_table(
    data: Any,
    sep: str = "__",
    max_level: int | None = None,
    keep_raw_nested_columns: bool = False,
) -> tuple[pd.DataFrame, str]:
    """将 JSON 对象尽量转换为二维表，并返回所采用的策略名。"""
    if isinstance(data, list):
        if len(data) == 0:
            return pd.DataFrame(), "list_empty"
        if all(isinstance(x, dict) for x in data):
            df = pd.json_normalize(data, sep=sep, max_level=max_level)
            if keep_raw_nested_columns:
                df = _append_raw_nested_columns(data, df, sep=sep)
            return df, "list_of_dicts"
        if all(isinstance(x, list) for x in data):
            return pd.DataFrame(data), "list_of_lists"
        if all(_is_scalar(x) for x in data):
            return pd.DataFrame({"value": data}), "list_of_scalars"
        return pd.DataFrame(), "list_mixed_non_tabular"

    if isinstance(data, dict):
        # 常见模式：根字典中某个键是记录数组
        record_keys = [
            k
            for k, v in data.items()
            if isinstance(v, list) and len(v) > 0 and all(isinstance(item, dict) for item in v)
        ]
        if record_keys:
            # 优先最长记录数组
            record_key = max(record_keys, key=lambda k: len(data[k]))
            rows = data[record_key]
            df = pd.json_normalize(rows, sep=sep, max_level=max_level)
            if keep_raw_nested_columns:
                df = _append_raw_nested_columns(rows, df, sep=sep)
            for k, v in data.items():
                if k == record_key:
                    continue
                if _is_scalar(v):
                    df[f"root__{k}"] = v
                elif keep_raw_nested_columns:
                    df[f"raw{sep}{k}"] = json.dumps(v, ensure_ascii=False)
            return df, f"dict_record_list:{record_key}"

        dict_of_lists_df = _to_dataframe_from_dict_of_lists(
            data,
            keep_raw_nested_columns=keep_raw_nested_columns,
            sep=sep,
        )
        if not dict_of_lists_df.empty:
            return dict_of_lists_df, "dict_of_lists"

        if all(_is_scalar(v) for v in data.values()):
            return pd.DataFrame([data]), "dict_single_row"

        return pd.DataFrame(), "dict_non_tabular"

    return pd.DataFrame(), "unsupported_root_type"


def read_json_as_table(
    path: Path,
    sep: str = "__",
    max_level: int | None = None,
    keep_raw_nested_columns: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = load_json_file(path)
    df, strategy = json_to_table(
        data,
        sep=sep,
        max_level=max_level,
        keep_raw_nested_columns=keep_raw_nested_columns,
    )
    meta = {
        "strategy": strategy,
        "root_type": type(data).__name__,
        "tabular_candidate": bool(df.shape[1] > 0 or df.shape[0] > 0),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "sep": sep,
        "max_level": max_level,
        "keep_raw_nested_columns": keep_raw_nested_columns,
    }
    return df, meta
