from pathlib import Path

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.parsers.json_toml_parser import JsonParser
from autorealize.pipeline import AutoRealizePipeline, _infer_downstream_context
from autorealize.profiling.stats import read_table
from autorealize.utils.json_table import read_json_as_table


def test_json_parser_detects_tabular_json(tmp_path: Path) -> None:
    parser = JsonParser()
    p = tmp_path / "tmp_json_parser_case.json"
    p.write_text(
        '[{"id": 1, "a": {"x": 10}, "tags": ["u", "v"]}, {"id": 2, "a": {"x": 20}, "tags": ["w"]}]',
        encoding="utf-8",
    )
    try:
        parsed = parser.parse(p)
        assert parsed.kind == "table"
        assert "a__x" in parsed.columns
    finally:
        if p.exists():
            p.unlink()


def test_read_table_supports_json(tmp_path: Path) -> None:
    p = tmp_path / "train.json"
    p.write_text('[{"id": 1, "y": 0}, {"id": 2, "y": 1}]', encoding="utf-8")
    df = read_table(p)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "y"]


def test_pipeline_cleans_json_table_and_keeps_submission(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "train.json").write_text(
        '[{"id": 1, "amount": "1.2"}, {"id": 2, "amount": "INF"}]',
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="预测下个月销售额",
        run_name="run_json_table",
    )
    assert (run_dir / "train.json").exists()
    assert (run_dir / "sample_submission.csv").exists()
    assert (run_dir / "realize_report" / "cleaning_report.md").exists()
    # ?? json ???????????
    assert (run_dir / "train.json").read_text(encoding="utf-8") == '[{"id": 1, "amount": "1.2"}, {"id": 2, "amount": "INF"}]'


def test_pipeline_json_not_cleaned_when_switch_off(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    raw = '[{"id": 1, "amount": "1.2"}, {"id": 2, "amount": "INF"}]'
    json_file = input_root / "train.json"
    json_file.write_text(raw, encoding="utf-8")
    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    cfg.data.enable_json_cleaning = False
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="预测下个月销售额",
        run_name="run_json_skip_cleaning",
    )
    # 认知与文档产物仍会生成
    assert (run_dir / "realize_report" / "data_description.md").exists()
    assert (run_dir / "description.md").exists()
    # 原始 JSON 内容保持不变（未进入清洗写回）
    assert (run_dir / "train.json").read_text(encoding="utf-8") == raw


def test_json_flatten_config_sep_and_max_level(tmp_path: Path) -> None:
    p = tmp_path / "nested.json"
    p.write_text(
        '[{"id": 1, "a": {"b": {"c": 10}}}, {"id": 2, "a": {"b": {"c": 20}}}]',
        encoding="utf-8",
    )
    df, meta = read_json_as_table(p, sep=".", max_level=1, keep_raw_nested_columns=False)
    assert meta["sep"] == "."
    assert meta["max_level"] == 1
    # max_level=1 下不会打平到 a.b.c
    assert "a.b.c" not in df.columns
    assert "a.b" in df.columns


def test_json_flatten_config_keep_raw_nested_columns(tmp_path: Path) -> None:
    p = tmp_path / "nested_raw.json"
    p.write_text(
        '[{"id": 1, "meta": {"x": 1}, "tags": ["a", "b"]}, {"id": 2, "meta": {"x": 2}, "tags": ["c"]}]',
        encoding="utf-8",
    )
    df, meta = read_json_as_table(p, sep="__", max_level=None, keep_raw_nested_columns=True)
    assert meta["keep_raw_nested_columns"] is True
    assert "raw__meta" in df.columns
    assert "raw__tags" in df.columns


def test_infer_context_prefers_train_with_label_and_test_without_label() -> None:
    cfg = AutoRealizeConfig.from_env()
    root = Path("runs/run_017_pizza")
    if not root.exists():
        return
    ctx = _infer_downstream_context(root, [], "预测下发帖人能否收到披萨", cfg)
    assert ctx["train_table"] == "train.json"
    assert ctx["predict_table"] == "test.json"
    assert ctx["target_column"] == "requester_received_pizza"
    assert ctx["id_column"] == "request_id"


def test_infer_context_test_with_empty_label_still_predict_set(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    train.write_text(
        '[{"request_id":"a1","requester_received_pizza":1,"x":10},{"request_id":"a2","requester_received_pizza":0,"x":20}]',
        encoding="utf-8",
    )
    test.write_text(
        '[{"request_id":"b1","requester_received_pizza":null,"x":11},{"request_id":"b2","requester_received_pizza":"","x":12}]',
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    ctx = _infer_downstream_context(tmp_path, [], "预测下发帖人能否收到披萨", cfg)
    assert ctx["train_table"] == "train.json"
    assert ctx["predict_table"] == "test.json"
    assert ctx["target_column"] == "requester_received_pizza"


def test_generate_sample_submission_reuses_camel_case_samplesubmission(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "sampleSubmission.csv").write_text(
        "request_id,requester_received_pizza\nr1,0\nr2,0\n",
        encoding="utf-8",
    )
    (input_root / "train.json").write_text(
        '[{"request_id":"a1","requester_received_pizza":1},{"request_id":"a2","requester_received_pizza":0}]',
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="预测下发帖人能否收到披萨",
        run_name="run_reuse_samplesubmission",
    )
    out = (run_dir / "sample_submission.csv").read_text(encoding="utf-8")
    assert "request_id,requester_received_pizza" in out


def test_generate_sample_submission_reuses_id_target_template(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "sampleSubmission.csv").write_text(
        "id,target\nx1,0\nx2,0\n",
        encoding="utf-8",
    )
    (input_root / "train.json").write_text(
        '[{"id":"x1","label":1},{"id":"x2","label":0}]',
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="二分类预测",
        run_name="run_reuse_id_target_samplesubmission",
    )
    out = (run_dir / "sample_submission.csv").read_text(encoding="utf-8")
    assert "id,target" in out


def test_generate_sample_submission_when_missing_template_uses_inferred_columns_and_predict_rows(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "train.csv").write_text(
        "request_id,requester_received_pizza,feat\n"
        "a1,1,10\n"
        "a2,0,20\n",
        encoding="utf-8",
    )
    (input_root / "test.csv").write_text(
        "request_id,feat\n"
        "b1,11\n"
        "b2,12\n"
        "b3,13\n",
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="二分类预测",
        run_name="run_generate_submission_without_template",
    )
    out_df = pd.read_csv(run_dir / "sample_submission.csv")
    assert list(out_df.columns) == ["request_id", "requester_received_pizza"]
    assert len(out_df) == 3


def test_description_marks_train_only_columns_for_prediction_guardrail(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "train.json").write_text(
        '[{"id":"a1","target":1,"train_only_feature":7},{"id":"a2","target":0,"train_only_feature":9}]',
        encoding="utf-8",
    )
    (input_root / "test.json").write_text(
        '[{"id":"b1"},{"id":"b2"}]',
        encoding="utf-8",
    )
    (input_root / "sampleSubmission.csv").write_text(
        "id,target\nb1,0\nb2,0\n",
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="二分类预测",
        run_name="run_feature_alignment_guardrail",
    )
    desc = (run_dir / "description.md").read_text(encoding="utf-8")
    assert "Train/Test Feature Alignment" in desc
    assert "train_only_feature" in desc
    assert "严禁在验证/预测阶段引用测试集不存在的原始字段" in desc



def test_data_description_refined_by_downstream_context_target_not_misread(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "train.json").write_text(
        '[{"request_id":"a1","requester_received_pizza":1,"f":10},{"request_id":"a2","requester_received_pizza":0,"f":20}]',
        encoding="utf-8",
    )
    (input_root / "test.json").write_text(
        '[{"request_id":"b1","giver_username_if_known":"N/A","f":11},{"request_id":"b2","giver_username_if_known":"N/A","f":12}]',
        encoding="utf-8",
    )
    (input_root / "sampleSubmission.csv").write_text(
        "request_id,requester_received_pizza\nb1,0\nb2,0\n",
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    cfg.llm.api_key = None
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="predict requester_received_pizza",
        run_name="run_refine_target_semantics",
    )
    dd = (run_dir / "realize_report" / "data_description.md").read_text(encoding="utf-8")

    # Ensure downstream-corrected target semantics appear in test/train summaries.
    assert dd.count("requester_received_pizza") >= 3

    lines = dd.splitlines()
    idx_test = lines.index("### test.json")
    test_block = "\n".join(lines[idx_test: idx_test + 8])
    assert "requester_received_pizza" in test_block
    assert "giver_username_if_known???'N/A'" not in test_block
