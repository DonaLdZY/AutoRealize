from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CsvDialectHint:
    sep: str
    engine: str | None = None
    inferred: bool = False
    reason: str = ""


def _decode_sample(raw: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"


def detect_csv_encoding(path: Path, *, max_bytes: int = 65536) -> str:
    """Return the first configured encoding that can decode a small CSV sample."""
    try:
        with path.open("rb") as f:
            raw = f.read(max_bytes)
    except Exception:
        return "utf-8-sig"
    _text, encoding = _decode_sample(raw)
    return encoding


def _read_sample_lines(path: Path, *, max_bytes: int = 65536, max_lines: int = 60) -> list[str]:
    with path.open("rb") as f:
        raw = f.read(max_bytes)
    text, _encoding = _decode_sample(raw)
    return [line.rstrip("\r\n") for line in text.splitlines() if line.strip()][:max_lines]


def _split_whitespace(line: str) -> list[str]:
    return line.strip().split()


def _looks_like_whitespace_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    header_parts = _split_whitespace(lines[0])
    if len(header_parts) < 2:
        return False
    if any("," in part for part in header_parts):
        return False

    checked = 0
    stable = 0
    for line in lines[1:16]:
        parts = _split_whitespace(line)
        if not parts:
            continue
        checked += 1
        if len(parts) == len(header_parts):
            stable += 1
    if checked == 0:
        return False
    comma_counts = [line.count(",") for line in lines[1 : min(len(lines), 16)]]
    comma_unstable = bool(comma_counts) and (max(comma_counts) - min(comma_counts) > 2 or max(comma_counts) >= len(header_parts) * 2)
    return stable / checked >= 0.8 and (comma_unstable or lines[0].count(",") == 0)


def infer_csv_dialect(path: Path) -> CsvDialectHint:
    """Infer a conservative pandas CSV dialect for contest-style tabular files.

    Some datasets use a `.csv` extension while the actual columns are separated
    by spaces and individual fields contain comma-delimited sequences. Pandas'
    default comma parser then treats those sequence values as extra columns.
    """
    try:
        lines = _read_sample_lines(path)
    except Exception:
        return CsvDialectHint(sep=",", inferred=False, reason="sample_unavailable")
    if _looks_like_whitespace_table(lines):
        return CsvDialectHint(sep=r"\s+", engine="python", inferred=True, reason="whitespace_columns_with_comma_lists")
    return CsvDialectHint(sep=",", inferred=False, reason="default_comma")


def read_csv_auto(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV-like file with lightweight delimiter inference.

    Callers can still pass explicit pandas options. If `sep` or `delimiter` is
    provided, this function respects the caller's choice.
    """
    if "sep" not in kwargs and "delimiter" not in kwargs:
        hint = infer_csv_dialect(path)
        kwargs["sep"] = hint.sep
        if hint.engine:
            kwargs.setdefault("engine", hint.engine)
    try:
        return pd.read_csv(path, *args, encoding=kwargs.pop("encoding", "utf-8-sig"), **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, *args, encoding="gb18030", **kwargs)
