from __future__ import annotations

import json
from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.service_api import _build_snapshot, _extract_last_error_from_output


def test_error_summary_preserves_final_deadline_exception():
    text = ("Traceback (most recent call last):\nhttpx.ReadError: socket closed\n"
            "The above exception was the direct cause of the following exception:\n"
            "Traceback (most recent call last):\n  File 'client.py', line 1\n"
            "autorealize.llm.client.StreamDeadlineExceeded: Completion stream exceeded its wall-clock deadline\n")
    assert _extract_last_error_from_output(text) == text.splitlines()[-1]


def test_error_summary_skips_pydantic_documentation():
    text = "ValueError: invalid contract\nFor further information visit https://errors.pydantic.dev/2\n"
    assert _extract_last_error_from_output(text) == "ValueError: invalid contract"


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
