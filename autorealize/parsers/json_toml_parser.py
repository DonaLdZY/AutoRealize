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


class JsonParser(BaseParser):
    supported_suffixes = (".json",)
    kind = "structured_document"

    def __init__(
        self,
        flatten_sep: str = "__",
        flatten_max_level: int | None = None,
        keep_raw_nested_columns: bool = False,
    ) -> None:
        self.flatten_sep = flatten_sep
        self.flatten_max_level = flatten_max_level
        self.keep_raw_nested_columns = keep_raw_nested_columns

    def parse(self, path: Path) -> ParsedFile:
        data = json.loads(path.read_text(encoding="utf-8"))
        df, meta = read_json_as_table(
            path,
            sep=self.flatten_sep,
            max_level=self.flatten_max_level,
            keep_raw_nested_columns=self.keep_raw_nested_columns,
        )
        if meta.get("tabular_candidate"):
            preview = df.head(20).to_dict(orient="records")
            return ParsedFile(
                path=path,
                kind="table",
                text_summary=f"JSON表格候选: {meta.get('strategy')} | {df.shape[0]} x {df.shape[1]}",
                preview=preview,
                columns=[str(c) for c in df.columns.tolist()],
                metadata={
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                    "json_strategy": meta.get("strategy", ""),
                    "json_root_type": meta.get("root_type", ""),
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
