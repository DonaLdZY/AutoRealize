from pathlib import Path

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.pipeline import AutoRealizePipeline


def test_preserve_original_description(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    pd.DataFrame({"id": [1], "v": [1]}).to_csv(input_root / "data.csv", index=False)
    (input_root / "description.md").write_text("origin description", encoding="utf-8")

    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="????",
        run_name="run_002_origin_desc",
    )

    origin = run_dir / "description_origin.md"
    assert origin.exists()
    assert "origin description" in origin.read_text(encoding="utf-8")
