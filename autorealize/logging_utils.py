from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .utils.safe_json import append_jsonl_safe, json_safe, write_json_safe

_event_sink_path: Path | None = None
_state_sink_path: Path | None = None
_event_lock = Lock()
_event_seq = 0
_run_id = ""
_recent_limit = 200
_telemetry_enabled = True
_recent_events: list[dict[str, Any]] = []
_status_counts: dict[str, int] = {}
_layer_counts: dict[str, int] = {}
_component_counts: dict[str, int] = {}
_active_components: dict[str, dict[str, Any]] = {}
_current_status = "created"
_started_at = ""
_last_event: dict[str, Any] | None = None
_raw_event_log = False

TERMINAL_EVENTS = {"COMPLETED", "FAILED", "SKIPPED", "DELETED", "ROLLED_BACK", "FLUSHED"}
ACTIVE_EVENTS = {"STARTED", "ACTIVATED", "RUNNING", "CREATED", "LEVEL_STARTED", "VERIFY_SCRIPT_RUNNING"}
FAIL_EVENTS = {"FAILED", "ERROR", "READ_FAILED", "PROBE_FAILED"}
SUCCESS_EVENTS = {"COMPLETED", "RUN_COMPLETED", "GENERATED_FILE", "READ_COMPLETED"}

EVENT_TAXONOMY: dict[str, Any] = {
    "schema_version": "autorealize.event_taxonomy.v1",
    "statuses": {
        "running": "组件已创建、激活或正在执行。",
        "completed": "组件或动作已成功完成。",
        "failed": "组件或动作失败，fields.error/reason 通常包含原因。",
        "skipped": "由于配置开关或条件不满足而跳过。",
        "info": "普通信息事件，不表示生命周期状态变化。",
        "created": "组件对象或资源被创建。",
    },
    "layers": {
        "pipeline": "运行级流程、输出整理、收尾动作。",
        "workflow": "两阶段工作流整体生命周期。",
        "module": "数据认知、任务定义等模块。",
        "stage": "模块内部的细分阶段，例如 P1 文件读取。",
        "agent": "Architect 与数据认知探查等智能体或子代理。",
        "checker": "格式校验器。",
        "knowledge": "本地知识库、RAG 清单、知识检索。",
        "llm": "LLM/VLLM 客户端和调用状态。",
        "tool": "表格探查、文件解析、关系发现等工具。",
        "misc": "未归类组件。",
    },
    "common_fields": {
        "file": "相对文件路径或生成文件名。",
        "source": "源文件或源目录。",
        "target": "目标文件或目标目录。",
        "workers": "并行 worker 数。",
        "seconds": "动作耗时秒数。",
        "error": "错误摘要。",
        "reason": "跳过、失败或决策原因。",
        "artifacts": "产物数量或产物路径列表。",
    },
    "frontend_notes": [
        "event_stream.jsonl 可按 seq 增量读取。",
        "current_state.json 可轮询展示当前状态、活跃组件与最近事件。",
        "classification.layer/scope 用于可视化分层泳道。",
        "component + event 可作为前端节点状态更新键。",
    ],
}


def setup_logging(level: int = logging.INFO) -> None:
    """初始化终端日志，统一输出到 stdout。"""
    global _raw_event_log
    _raw_event_log = os.environ.get("AUTOREALIZE_RAW_EVENT_LOG", "0").strip() in {"1", "true", "True"}
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # 降噪：这些模块保留 warning/error，避免终端刷屏。
    noisy_loggers = [
        "autorealize.llm.client",
        "autorealize.cognition",
        "httpx",
        "openai",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_event_sink(
    path: Path | None,
    *,
    state_path: Path | None = None,
    run_id: str = "",
    enabled: bool = True,
    recent_limit: int = 200,
) -> None:
    """配置结构化事件落盘。

    event_stream.jsonl 适合 tail/流式消费；current_state.json 适合前端轮询展示当前状态。
    """
    global _event_sink_path, _state_sink_path, _event_seq, _run_id, _recent_limit, _telemetry_enabled
    global _recent_events, _status_counts, _layer_counts, _component_counts, _active_components
    global _current_status, _started_at, _last_event
    _event_sink_path = path if enabled else None
    _state_sink_path = state_path if enabled else None
    _event_seq = 0
    _run_id = run_id
    _recent_limit = max(10, int(recent_limit))
    _telemetry_enabled = enabled
    _recent_events = []
    _status_counts = {}
    _layer_counts = {}
    _component_counts = {}
    _active_components = {}
    _current_status = "running"
    _started_at = datetime.now(timezone.utc).isoformat()
    _last_event = None
    for p in [_event_sink_path, _state_sink_path]:
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
    _write_state_snapshot()


def get_event_taxonomy() -> dict[str, Any]:
    """返回前端可用的事件协议说明。"""
    return EVENT_TAXONOMY


def write_event_taxonomy(path: Path) -> None:
    """将事件协议说明写入 JSON，便于前端生成图例和筛选器。"""
    write_json_safe(path, EVENT_TAXONOMY, indent=2)


def _event_classification(component: str) -> dict[str, str]:
    c = component.lower()
    if c.startswith("module."):
        return {"layer": "module", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("stage."):
        return {"layer": "stage", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("agent."):
        return {"layer": "agent", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("checker."):
        return {"layer": "checker", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("knowledge."):
        return {"layer": "knowledge", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("workflow"):
        return {"layer": "workflow", "scope": "main"}
    if c.startswith("llm"):
        return {"layer": "llm", "scope": "client"}
    if c.startswith("progressive_sampler"):
        return {"layer": "tool", "scope": "progressive_sampler"}
    if c.startswith("pipeline") or c.startswith("finalize"):
        return {"layer": "pipeline", "scope": c.split(".")[0]}
    return {"layer": "misc", "scope": c.split(".")[0] if c else "unknown"}


def _infer_status(event: str) -> str:
    e = event.upper()
    if e in FAIL_EVENTS or e.endswith("FAILED") or "ERROR" in e:
        return "failed"
    if e in SUCCESS_EVENTS or e.endswith("COMPLETED") or e.endswith("GENERATED"):
        return "completed"
    if e.endswith("SKIPPED") or e == "SKIPPED":
        return "skipped"
    if e in ACTIVE_EVENTS or e.endswith("STARTED") or e.endswith("ACTIVATED") or e.endswith("RUNNING"):
        return "running"
    if e.endswith("CREATED"):
        return "created"
    return "info"


def _infer_severity(status: str, fields: dict[str, Any]) -> str:
    if status == "failed":
        return "error"
    if any(str(k).lower() in {"warning", "warnings", "alerts"} for k in fields):
        return "warning"
    return "info"


def _json_safe(value: Any) -> Any:
    return json_safe(value)


def _sanitize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _json_safe(v) for k, v in fields.items()}


def log_event(logger: logging.Logger, component: str, event: str, **fields: Any) -> None:
    """同时写人类可读终端日志和机器可读事件流。"""
    safe_fields = _sanitize_fields(fields)
    if _raw_event_log:
        if safe_fields:
            detail = " | ".join(f"{k}={safe_fields[k]}" for k in sorted(safe_fields))
            logger.info("[EVENT] %s | %s | %s", component, event, detail)
        else:
            logger.info("[EVENT] %s | %s", component, event)
    else:
        line = _human_terminal_line(component, event, safe_fields)
        if line:
            logger.info(line)

    if not _telemetry_enabled:
        return

    global _event_seq, _last_event, _current_status
    classification = _event_classification(component)
    status = _infer_status(event)
    severity = _infer_severity(status, safe_fields)
    with _event_lock:
        _event_seq += 1
        payload = {
            "schema_version": "autorealize.event.v1",
            "seq": _event_seq,
            "run_id": _run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "event": event,
            "status": status,
            "severity": severity,
            "fields": safe_fields,
            "classification": classification,
        }
        _last_event = payload
        _recent_events.append(payload)
        if len(_recent_events) > _recent_limit:
            del _recent_events[: len(_recent_events) - _recent_limit]
        _status_counts[status] = _status_counts.get(status, 0) + 1
        _layer_counts[classification["layer"]] = _layer_counts.get(classification["layer"], 0) + 1
        _component_counts[component] = _component_counts.get(component, 0) + 1
        _update_active_components(component, event, payload)
        if component == "pipeline" and event == "RUN_COMPLETED":
            _current_status = "completed"
        elif status == "failed" and _current_status != "completed":
            _current_status = "failed"
        elif _current_status == "created":
            _current_status = "running"
        if _event_sink_path is not None:
            append_jsonl_safe(_event_sink_path, payload)
        _write_state_snapshot()


def _update_active_components(component: str, event: str, payload: dict[str, Any]) -> None:
    e = event.upper()
    if e in TERMINAL_EVENTS or e.endswith("COMPLETED") or e.endswith("FAILED") or e.endswith("SKIPPED"):
        _active_components.pop(component, None)
        return
    if e in ACTIVE_EVENTS or e.endswith("STARTED") or e.endswith("ACTIVATED") or e.endswith("RUNNING"):
        _active_components[component] = {
            "component": component,
            "event": event,
            "seq": payload["seq"],
            "ts": payload["ts"],
            "classification": payload["classification"],
            "fields": payload["fields"],
        }


def _write_state_snapshot() -> None:
    if _state_sink_path is None:
        return
    snapshot = {
        "schema_version": "autorealize.state.v1",
        "run_id": _run_id,
        "status": _current_status,
        "started_at": _started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "event_count": _event_seq,
        "status_counts": _status_counts,
        "layer_counts": _layer_counts,
        "component_counts": _component_counts,
        "active_components": list(_active_components.values()),
        "last_event": _last_event,
        "recent_events": _recent_events,
    }
    write_json_safe(_state_sink_path, snapshot, indent=2)


def _human_terminal_line(component: str, event: str, fields: dict[str, Any]) -> str | None:
    e = event.upper()
    f = fields

    def _kv(*keys: str) -> str:
        vals = []
        for k in keys:
            if k in f:
                vals.append(f"{k}={f[k]}")
        return " | ".join(vals)

    # 运行级
    if component == "pipeline" and e == "RUN_STARTED":
        return f"[RUN] 启动 | run_name={f.get('run_name', '')} | input={f.get('input_root', '')}"
    if component == "pipeline" and e == "WORKSPACE_COPIED":
        return f"[RUN] 工作区准备完成 | { _kv('workspace') }"
    if component == "pipeline" and e == "RUN_COMPLETED":
        return f"[RUN] 完成 | { _kv('run_dir') }"

    # 模块级
    if component == "module.data_cognition":
        if e == "ACTIVATED":
            return "[P1] 数据认知启动"
        if e == "FILES_SELECTED":
            return f"[P1] 文件选择完成 | { _kv('selected', 'total_files') }"
        if e == "GENERATING_FILE":
            return f"[P1] 正在生成文件 | { _kv('file') }"
        if e == "GENERATED_FILE":
            return f"[P1] 已生成文件 | { _kv('file') }"
        if e == "COMPLETED":
            return f"[P1] 数据认知完成 | { _kv('files', 'relations') }"
        if e == "SKIPPED":
            return f"[P1] 已跳过 | { _kv('reason') }"

    if component == "module.task_definition":
        if e == "ACTIVATED":
            return "[P2] 任务定义启动"
        if e == "GENERATING_FILE":
            return f"[P2] 正在生成文件 | { _kv('file') }"
        if e == "GENERATED_FILE":
            return f"[P2] 已生成文件 | { _kv('file') }"
        if e == "COMPLETED":
            return "[P2] 任务定义完成"
        if e == "SKIPPED":
            return f"[P2] 已跳过 | { _kv('reason') }"

