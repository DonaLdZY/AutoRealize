from pathlib import Path

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.pipeline import AutoRealizePipeline


def test_pipeline_offline_smoke(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    pd.DataFrame({"id": [1, 2], "amount": ["1.2", "INF"]}).to_csv(input_root / "sales.csv", index=False)
    (input_root / "readme.txt").write_text("????????????", encoding="utf-8")

    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="????????",
        run_name="run_001_smoke",
    )

    # Kaggle-style required files
    assert (run_dir / "description.md").exists()
    assert (run_dir / "sample_submission.csv").exists()

    # Reports should go under realize_report/
    report_dir = run_dir / "realize_report"
    assert (report_dir / "data_description.md").exists()
    assert (report_dir / "cleaning_report.md").exists()
