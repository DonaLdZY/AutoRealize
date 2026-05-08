from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

_event_sink_path: Path | None = None
_event_lock = Lock()


def setup_logging(level: int = logging.INFO) -> None:
    """初始化终端日志。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def configure_event_sink(path: Path | None) -> None:
    """配置结构化事件流输出文件。"""
    global _event_sink_path
    _event_sink_path = path
    if _event_sink_path is not None:
        _event_sink_path.parent.mkdir(parents=True, exist_ok=True)


def _event_classification(component: str) -> dict[str, str]:
    c = component.lower()
    if c.startswith("stage."):
        return {"layer": "stage", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("agent."):
        return {"layer": "agent", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("checker."):
        return {"layer": "checker", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("progressive_sampler"):
        return {"layer": "tool", "scope": "progressive_sampler"}
    if c.startswith("pipeline") or c.startswith("finalize"):
        return {"layer": "pipeline", "scope": c.split(".")[0]}
    return {"layer": "misc", "scope": c.split(".")[0] if c else "unknown"}


def log_event(logger: logging.Logger, component: str, event: str, **fields: Any) -> None:
    """统一事件日志格式，便于在终端观察系统状态。"""
    if fields:
        detail = " | ".join(f"{k}={fields[k]}" for k in sorted(fields))
        logger.info("[EVENT] %s | %s | %s", component, event, detail)
    else:
        logger.info("[EVENT] %s | %s", component, event)

    if _event_sink_path is None:
        return
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event": event,
        "fields": fields,
        "classification": _event_classification(component),
    }
    with _event_lock:
        with _event_sink_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
