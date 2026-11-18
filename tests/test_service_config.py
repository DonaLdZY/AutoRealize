from __future__ import annotations

import json
import logging
from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.logging_utils import configure_event_sink, log_event
from autorealize.service_api import (
    StartAutoRealizeRequest,
    _build_snapshot,
    _compose_task_hint,
    _extract_last_error_from_output,
    _run_job,
)


def test_error_summary_preserves_final_deadline_exception():
    text = ("Traceback (most recent call last):\nhttpx.ReadError: socket closed\n"
            "The above exception was the direct cause of the following exception:\n"
            "Traceback (most recent call last):\n  File 'client.py', line 1\n"
            "autorealize.llm.client.StreamDeadlineExceeded: Completion stream exceeded its wall-clock deadline\n")
    assert _extract_last_error_from_output(text) == text.splitlines()[-1]


def test_error_summary_skips_pydantic_documentation():
    text = "ValueError: invalid contract\nFor further information visit https://errors.pydantic.dev/2\n"
    assert _extract_last_error_from_output(text) == "ValueError: invalid contract"


def test_autorealize_job_passes_user_evaluation_contract_to_pipeline(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Process:
        returncode = 0

        def communicate(self):
            return "", ""

    class Store:
        def update(self, *_args, **_kwargs):
            return None

        def set_process(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("autorealize.service_api.store", Store())
    def start_process(command, **_kwargs):
        captured["command"] = command
        return Process()

    monkeypatch.setattr("autorealize.service_api.subprocess.Popen", start_process)
    request = StartAutoRealizeRequest(
        task_id="task",
        input_root=str(tmp_path / "input"),
        output_root=str(tmp_path / "runs"),
        run_name="contract-test",
        task_hint="Classify equipment failures.",
        evaluation_contract=(
            "Primary metric: average precision. Use random_state=20260907. "
            "Do not read final labels during search."
        ),
        config_path=str(tmp_path / "missing.yaml"),
        working_dir=str(tmp_path),
    )

    _run_job("job", request)

    command = captured["command"]
    task_argument = command[command.index("--task") + 1]
    assert task_argument == _compose_task_hint(
        request.task_hint,
        request.evaluation_contract,
    )
    assert "random_state=20260907" in task_argument
    assert "Do not read final labels during search." in task_argument


def test_task_hint_keeps_backward_compatibility_without_evaluation_contract():
    assert _compose_task_hint("  Classify equipment failures.  ", "") == (
        "Classify equipment failures."
    )


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


def test_retryable_failure_keeps_run_snapshot_nonterminal(tmp_path: Path) -> None:
    state = tmp_path / "current_state.json"
    events = tmp_path / "event_stream.jsonl"
    configure_event_sink(events, state_path=state, run_id="retry-demo")
    logger = logging.getLogger("test.retry")
    log_event(logger, "stage.P1", "READ_FAILED", file="data.csv", retryable=True)
    snapshot = json.loads(state.read_text(encoding="utf-8"))
    assert snapshot["status"] == "retrying"
    assert snapshot["retryable_failures"] == 1
    assert snapshot["unrecoverable_failures"] == 0
    log_event(logger, "pipeline", "RUN_COMPLETED", run_dir=str(tmp_path))
    snapshot = json.loads(state.read_text(encoding="utf-8"))
    assert snapshot["status"] == "completed"
