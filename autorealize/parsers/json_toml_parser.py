from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.json_table import read_json_as_table

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib

from .base import BaseParser, ParsedFile


def _preview_obj(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)[:4000]
    except TypeError:
        return str(obj)[:4000]


def _collect_json_paths(obj: Any, prefix: str = "", out: set[str] | None = None, max_items: int = 200) -> set[str]:
    if out is None:
        out = set()
    if len(out) >= max_items:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            path = f"{prefix}.{key}" if prefix else key
            out.add(path)
            _collect_json_paths(v, path, out, max_items=max_items)
    elif isinstance(obj, list):
        list_path = f"{prefix}[]" if prefix else "[]"
        out.add(list_path)
        for item in obj[:5]:
            _collect_json_paths(item, list_path, out, max_items=max_items)
    return out


def _schema_keys(obj: Any, *, max_depth: int = 2, prefix: str = "") -> set[str]:
    if max_depth < 0:
        return set()
    if isinstance(obj, dict):
        out: set[str] = set()
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.add(key)
            out.update(_schema_keys(v, max_depth=max_depth - 1, prefix=key))
        return out
    if isinstance(obj, list):
        list_path = f"{prefix}[]" if prefix else "[]"
        out = {list_path}
        for item in obj[:3]:
            out.update(_schema_keys(item, max_depth=max_depth - 1, prefix=list_path))
        return out
    return {prefix} if prefix else set()


def _first_level_samples(data: Any, *, limit: int = 8) -> list[Any]:
    if isinstance(data, list):
        return data[:limit]
    if isinstance(data, dict):
        list_values = [v for v in data.values() if isinstance(v, list) and v]
        if list_values:
            return max(list_values, key=len)[:limit]
        return list(data.values())[:limit]
    return [data]


def _schema_similarity(samples: list[Any]) -> dict[str, Any]:
    schemas = [sorted(_schema_keys(item)) for item in samples]
    sets = [set(x) for x in schemas if x]
    if len(sets) <= 1:
        score = 1.0 if sets else 0.0
    else:
        union = set.union(*sets)
        intersection = set.intersection(*sets)
        score = round(len(intersection) / max(1, len(union)), 6)
    return {
        "sample_count": len(samples),
        "schema_similarity": score,
        "schema_samples": schemas[:5],
    }


class JsonParser(BaseParser):
    supported_suffixes = (".json",)
    kind = "structured_document"

    def __init__(
        self,
        flatten_sep: str = "__",
        flatten_max_level: int | None = None,
        keep_raw_nested_columns: bool = False,
        preview_rows: int = 10,
    ) -> None:
        self.flatten_sep = flatten_sep
        self.flatten_max_level = flatten_max_level
        self.keep_raw_nested_columns = keep_raw_nested_columns
        self.preview_rows = max(1, int(preview_rows))

    def parse(self, path: Path) -> ParsedFile:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        schema_meta = _schema_similarity(_first_level_samples(data))
        df, meta = read_json_as_table(
            path,
            sep=self.flatten_sep,
            max_level=self.flatten_max_level,
            keep_raw_nested_columns=self.keep_raw_nested_columns,
        )
        if meta.get("tabular_candidate"):
            preview = df.head(self.preview_rows).to_dict(orient="records")
            return ParsedFile(
                path=path,
                kind="table",
                text_summary=f"JSON 可表格化: {meta.get('strategy')} | {df.shape[0]} x {df.shape[1]}",
                preview=preview,
                columns=[str(c) for c in df.columns.tolist()],
                metadata={
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                    "json_strategy": meta.get("strategy", ""),
                    "json_root_type": meta.get("root_type", ""),
                    "json_first_level_schema": schema_meta,
                    "preview_rows_used": int(min(self.preview_rows, len(df))),
                    "source_format": "json",
                },
            )
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=_preview_obj(data),
            metadata={
                "type": type(data).__name__,
                "json_root_type": type(data).__name__,
                "json_paths_topk": sorted(list(_collect_json_paths(data)))[:60],
                "json_first_level_schema": schema_meta,
                "source_format": "json",
            },
        )


class TomlParser(BaseParser):
    supported_suffixes = (".toml",)
    kind = "structured_document"

    def parse(self, path: Path) -> ParsedFile:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=_preview_obj(data),
            metadata={"keys": list(data.keys())[:30]},
        )
