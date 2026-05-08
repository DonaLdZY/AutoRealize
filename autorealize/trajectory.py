from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Event:
    ts: str
    stage: str
    kind: str
    payload: dict[str, Any]


class TrajectoryLogger:
    """运行轨迹记录器。"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "trajectory_events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, kind: str, payload: dict[str, Any]) -> None:
        event = Event(
            ts=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            kind=kind,
            payload=payload,
        )
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def write_markdown_index(self) -> None:
        """生成轨迹索引文档。"""
        md_path = self.run_dir / "trajectory.md"
        lines = [
            "# 运行轨迹",
            "",
            f"- 轨迹事件文件: `{self.events_path.name}`",
            "- 结构化监控事件: `event_stream.jsonl`（含分层分类字段，前端可直接消费）",
            "- LLM 调用轨迹: `llm_traces.jsonl`",
            "- 每行一个 JSON 事件，包含阶段、动作、参数、错误与结果。",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")
