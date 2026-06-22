from pathlib import Path

from PIL import Image

from autorealize.config import AutoRealizeConfig
from autorealize.pipeline import AutoRealizePipeline
import autorealize.pipeline as pipeline_mod


def test_file_level_image_uses_vllm_semantic_summary(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    img_path = input_root / "1.jpg"
    Image.new("RGB", (16, 16), color=(220, 120, 80)).save(img_path)

    monkeypatch.setattr(
        pipeline_mod,
        "_infer_single_image_purpose",
        lambda image_file, config: "sample image for leaf classification",
    )

    cfg = AutoRealizeConfig.from_env()
    cfg.vllm.enabled = True
    cfg.switches.run_task_definition = False
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="build an image classification model",
        run_name="run_image_semantic",
    )

    data_desc = (run_dir / "realize_report" / "data_description.md").read_text(encoding="utf-8")
    assert "### 1.jpg" in data_desc
    assert "sample image for leaf classification" in data_desc
