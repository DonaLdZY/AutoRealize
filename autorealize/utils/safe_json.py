from __future__ import annotations

import dataclasses
import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert common runtime/scientific Python values into JSON-safe objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
    except Exception:
        pass
    return str(value)


def dumps_json_safe(value: Any, **kwargs: Any) -> str:
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("default", str)
    return json.dumps(json_safe(value), **kwargs)


def write_json_safe(path: Path, value: Any, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json_safe(value, **kwargs), encoding="utf-8")


def append_jsonl_safe(path: Path, value: Any, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(dumps_json_safe(value, **kwargs) + "\n")
