from pathlib import Path

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.pipeline import AutoRealizePipeline


def test_output_data_files_are_flattened_to_run_root(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv(input_root / "sales.csv", index=False)
    (input_root / "note.txt").write_text("input note", encoding="utf-8")

    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="forecast sales",
        run_name="run_flattened_output",
    )

    assert (run_dir / "sales.csv").exists()
    assert (run_dir / "note.txt").exists()
    assert not (run_dir / "data").exists()
    assert (run_dir / "description.md").exists()
    assert (run_dir / "sample_submission.csv").exists()
