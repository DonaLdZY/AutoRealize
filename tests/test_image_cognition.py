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
        lambda image_file, config: "这是一张用于叶片分类的样本图像",
    )

    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    cfg.vllm.enabled = True
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="构建图像分类模型",
        run_name="run_image_semantic",
    )

    data_desc = (run_dir / "realize_report" / "data_description.md").read_text(encoding="utf-8")
    assert "### 1.jpg" in data_desc
    assert "用于叶片分类的样本图像" in data_desc

