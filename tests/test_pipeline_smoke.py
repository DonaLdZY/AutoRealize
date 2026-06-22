from pathlib import Path
import json

import pandas as pd
import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.llm.client import LLMClient
from autorealize.pipeline import AutoRealizePipeline


def test_pipeline_requires_llm_smoke(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    pd.DataFrame({"id": [1, 2], "amount": ["1.2", "INF"]}).to_csv(input_root / "sales.csv", index=False)
    (input_root / "readme.txt").write_text("forecast next month revenue", encoding="utf-8")

    cfg = AutoRealizeConfig.from_env()
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="forecast next month revenue",
        run_name="run_001_smoke",
    )

    # Kaggle-style required files
    assert (run_dir / "description.md").exists()
    submission_report = json.loads((run_dir / "realize_report" / "submission_report.json").read_text(encoding="utf-8"))
    assert submission_report["source"] in {
        "official_sample_reused",
        "generated_by_llm",
        "skipped_no_authoritative_contract",
        "not_required_by_problem_paradigm",
    }

    # Reports should go under realize_report/
    report_dir = run_dir / "realize_report"
    assert (report_dir / "data_description.md").exists()


def test_real_llm_client_rejects_missing_api_key(tmp_path: Path) -> None:
    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    with pytest.raises(ValueError):
        LLMClient(cfg, tmp_path)

