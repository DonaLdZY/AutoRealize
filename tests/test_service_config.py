from __future__ import annotations

import json
from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.service_api import _build_snapshot


def test_snapshot_uses_resolved_yaml_filenames_and_limits(tmp_path: Path) -> None:
    report_dir = tmp_path / "realize_report"
    report_dir.mkdir()
    cfg = AutoRealizeConfig()
    cfg.telemetry.current_state_filename = "state.custom.json"
    cfg.telemetry.event_stream_filename = "events.custom.jsonl"
    cfg.service.snapshot_event_limit = 1
    cfg.service.snapshot_file_markdown_chars = 4
    cfg.write_yaml(report_dir / "final_config.yaml")

    (report_dir / "state.custom.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    (report_dir / "events.custom.jsonl").write_text(
        '{"seq": 1}\n{"seq": 2}\n',
        encoding="utf-8",
    )
    cognition_dir = report_dir / "file_cognition"
    cognition_dir.mkdir()
    (cognition_dir / "demo.json").write_text(
        json.dumps({"path": "demo.csv"}),
        encoding="utf-8",
    )
    (cognition_dir / "demo.md").write_text("abcdefgh", encoding="utf-8")

    snapshot = _build_snapshot(str(tmp_path))

    assert snapshot["current_state"]["status"] == "running"
    assert snapshot["events"] == [{"seq": 2}]
    assert snapshot["file_cognition_index"]["demo.csv"]["markdown"] == "abcd"
