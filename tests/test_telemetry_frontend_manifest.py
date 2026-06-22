import json
from pathlib import Path

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.pipeline import AutoRealizePipeline


def test_frontend_manifest_and_event_taxonomy_are_written(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    pd.DataFrame({"id": [1, 2], "sales": [10, 12]}).to_csv(input_root / "sales.csv", index=False)
    (input_root / "readme.md").write_text("forecast next month sales", encoding="utf-8")

    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="forecast next month sales",
        run_name="run_telemetry_manifest",
    )

    report_dir = run_dir / "realize_report"
    manifest = json.loads((report_dir / "frontend_manifest.json").read_text(encoding="utf-8"))
    taxonomy = json.loads((report_dir / "event_taxonomy.json").read_text(encoding="utf-8"))
    cognition_report = json.loads((report_dir / "data_cognition_report.json").read_text(encoding="utf-8"))
    task_report = json.loads((report_dir / "task_definition_report.json").read_text(encoding="utf-8"))
    submission_report = json.loads((report_dir / "submission_report.json").read_text(encoding="utf-8"))
    current_state = json.loads((report_dir / "current_state.json").read_text(encoding="utf-8"))
    first_event = json.loads((report_dir / "event_stream.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert manifest["schema_version"] == "autorealize.frontend_manifest.v1"
    assert manifest["watch"]["event_stream"] == "realize_report/event_stream.jsonl"
    assert any(m["id"] == "data_cognition" for m in manifest["modules"])
    assert taxonomy["schema_version"] == "autorealize.event_taxonomy.v1"
    assert cognition_report["schema_version"] == "autorealize.data_cognition_report.v1"
    assert cognition_report["summary"]["file_count"] >= 2
    assert task_report["schema_version"] == "autorealize.task_definition_report.v1"
    assert submission_report["schema_version"] == "autorealize.submission_report.v1"
    assert "module" in taxonomy["layers"]
    assert current_state["status"] == "completed"
    assert first_event["schema_version"] == "autorealize.event.v1"
    assert "classification" in first_event
