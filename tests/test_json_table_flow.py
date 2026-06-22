import json
from pathlib import Path

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.models import SubmissionScriptPlan
from autorealize.parsers.json_toml_parser import JsonParser
from autorealize.pipeline import (
    AutoRealizePipeline,
    _infer_downstream_context,
    _try_build_submission_from_plan,
    _validate_generated_submission,
)
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


def test_read_table_supports_utf8_sig_json(tmp_path: Path) -> None:
    p = tmp_path / "predict.json"
    p.write_text('[{"id": 1, "y": null}, {"id": 2, "y": null}]', encoding="utf-8-sig")
    df = read_table(p)
    assert list(df.columns) == ["id", "y"]
    parsed = JsonParser().parse(p)
    assert parsed.kind == "table"


def test_pipeline_reads_json_table_and_keeps_submission(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    raw = '[{"id": 1, "amount": "1.2"}, {"id": 2, "amount": "INF"}]'
    (input_root / "train.json").write_text(raw, encoding="utf-8")
    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="forecast next month sales amount",
        run_name="run_json_table",
    )
    assert (run_dir / "train.json").exists()
    assert (run_dir / "sample_submission.csv").exists()
    assert (run_dir / "train.json").read_text(encoding="utf-8") == raw


def test_pipeline_json_content_is_preserved(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    raw = '[{"id": 1, "amount": "1.2"}, {"id": 2, "amount": "INF"}]'
    (input_root / "train.json").write_text(raw, encoding="utf-8")
    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="forecast next month sales amount",
        run_name="run_json_preserved",
    )
    assert (run_dir / "realize_report" / "data_description.md").exists()
    assert (run_dir / "description.md").exists()
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
    ctx = _infer_downstream_context(root, [], "predict requester_received_pizza", cfg)
    assert ctx["train_table"] == "train.json"
    assert ctx["predict_table"] == "test.json"
    assert ctx["target_column"] == "requester_received_pizza"
    assert ctx["id_column"] == "request_id"


def test_infer_context_test_with_empty_label_still_predict_set(tmp_path: Path) -> None:
    (tmp_path / "train.json").write_text(
        '[{"request_id":"a1","requester_received_pizza":1,"x":10},{"request_id":"a2","requester_received_pizza":0,"x":20}]',
        encoding="utf-8",
    )
    (tmp_path / "test.json").write_text(
        '[{"request_id":"b1","requester_received_pizza":null,"x":11},{"request_id":"b2","requester_received_pizza":"","x":12}]',
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    ctx = _infer_downstream_context(tmp_path, [], "predict requester_received_pizza", cfg)
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
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="predict requester_received_pizza",
        run_name="run_reuse_samplesubmission",
    )
    out = (run_dir / "sample_submission.csv").read_text(encoding="utf-8")
    assert "request_id,requester_received_pizza" in out


def test_generate_sample_submission_reuses_id_target_template(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "sampleSubmission.csv").write_text("id,target\nx1,0\nx2,0\n", encoding="utf-8")
    (input_root / "train.json").write_text('[{"id":"x1","label":1},{"id":"x2","label":0}]', encoding="utf-8")
    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="binary classification prediction",
        run_name="run_reuse_id_target_samplesubmission",
    )
    out = (run_dir / "sample_submission.csv").read_text(encoding="utf-8")
    assert "id,target" in out


def test_generate_sample_submission_when_missing_template_uses_inferred_columns_and_predict_rows(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "train.csv").write_text(
        "request_id,requester_received_pizza,feat\na1,1,10\na2,0,20\n",
        encoding="utf-8",
    )
    (input_root / "test.csv").write_text("request_id,feat\nb1,11\nb2,12\nb3,13\n", encoding="utf-8")
    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="binary classification prediction",
        run_name="run_generate_submission_without_template",
    )
    out_df = pd.read_csv(run_dir / "sample_submission.csv")
    assert list(out_df.columns) == ["request_id", "requester_received_pizza"]
    assert len(out_df) == 3
    desc = (run_dir / "description.md").read_text(encoding="utf-8")
    assert "sample_submission.csv: [request_id, requester_received_pizza]" in desc
    assert "`request_id`" in desc
    assert "`requester_received_pizza`" in desc
    assert "Metric Direction" not in desc
    assert "Leakage Guards" not in desc


def test_evaluation_contract_repair_loop_returns_contract_to_llm(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "train.csv").write_text("id,target,feat\na1,1,10\na2,0,20\n", encoding="utf-8")
    (input_root / "test.csv").write_text("id,feat\nb1,11\n", encoding="utf-8")
    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="force_eval_repair",
        run_name="run_eval_contract_repair",
    )
    report = json.loads((run_dir / "realize_report" / "evaluation_contract_report.json").read_text(encoding="utf-8"))
    assert len(report["revision_log"]) >= 2
    assert report["revision_log"][0]["passed"] is False
    assert report["final"]["passed"] is True
    assert report["reflection_log"]
    assert report["reflection_log"][-1]["is_unambiguous"] is True


def test_generate_sample_submission_plan_can_read_relative_input_file(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "train.csv").write_text("id,target,feat\na1,1,10\na2,0,20\n", encoding="utf-8")
    (input_root / "weather.csv").write_text("id,target\nw1,0.5\nw2,0.7\n", encoding="utf-8")
    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="relative_path_probe",
        run_name="run_submission_relative_path",
    )
    out_df = pd.read_csv(run_dir / "sample_submission.csv")
    assert list(out_df["id"]) == ["w1", "w2"]


def test_submission_plan_can_return_dataframe_from_builder_function(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "original_order_id": ["raw1", "raw2"],
            "vehicle_type": ["truck", "van"],
        }
    )
    plan = SubmissionScriptPlan(
        purpose="builder function style",
        submission_columns=["order_id", "original_order_id", "vehicle_type"],
        python_code=(
            "def generate_submission(df):\n"
            "    out_df = df[['order_id', 'original_order_id', 'vehicle_type']].copy()\n"
            "    return out_df\n"
        ),
        id_column="order_id",
        target_columns=["original_order_id", "vehicle_type"],
    )

    out_df, issues = _try_build_submission_from_plan(plan=plan, df=df, data_root=tmp_path)

    assert issues == []
    assert out_df is not None
    assert list(out_df.columns) == ["order_id", "original_order_id", "vehicle_type"]
    assert list(out_df["order_id"]) == ["o1", "o2"]


def test_submission_plan_projects_extra_operational_columns_to_contract(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "original_order_id": ["raw1", "raw2"],
            "vehicle_type": ["truck", "van"],
        }
    )
    plan = SubmissionScriptPlan(
        purpose="llm produced a solved-looking dispatch table",
        submission_columns=["order_id", "original_order_id", "vehicle_type"],
        python_code=(
            "out_df = pd.DataFrame({\n"
            "    'delivery_day': ['2026-01-01', '2026-01-02'],\n"
            "    'order_id': df['order_id'],\n"
            "    'original_order_id': df['original_order_id'],\n"
            "    'wave_id': ['w1', 'w2'],\n"
            "    'carrier': ['c1', 'c2'],\n"
            "    'vehicle_type': ['truck', 'van'],\n"
            "    'vehicle_count': [1, 1],\n"
            "    'cost': [100.0, 120.0],\n"
            "})\n"
        ),
        id_column="order_id",
        target_columns=["original_order_id", "vehicle_type"],
    )

    out_df, issues = _try_build_submission_from_plan(plan=plan, df=df, data_root=tmp_path)

    assert issues == []
    assert out_df is not None
    assert list(out_df.columns) == ["order_id", "original_order_id", "vehicle_type"]
    assert list(out_df["vehicle_type"]) == ["truck", "van"]


def test_description_marks_train_only_columns_for_prediction_guardrail(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "train.json").write_text(
        '[{"id":"a1","target":1,"train_only_feature":7},{"id":"a2","target":0,"train_only_feature":9}]',
        encoding="utf-8",
    )
    (input_root / "test.json").write_text('[{"id":"b1"},{"id":"b2"}]', encoding="utf-8")
    (input_root / "sampleSubmission.csv").write_text("id,target\nb1,0\nb2,0\n", encoding="utf-8")
    cfg = AutoRealizeConfig.from_env()
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="binary classification prediction",
        run_name="run_feature_alignment_guardrail",
    )
    desc = (run_dir / "description.md").read_text(encoding="utf-8")
    assert "train_only_feature" in desc


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
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="predict requester_received_pizza",
        run_name="run_refine_target_semantics",
    )
    dd = (run_dir / "realize_report" / "data_description.md").read_text(encoding="utf-8")
    assert dd.count("requester_received_pizza") >= 3
    lines = dd.splitlines()
    idx_test = lines.index("### test.json")
    test_block = "\n".join(lines[idx_test : idx_test + 8])
    assert "requester_received_pizza" in test_block
    assert "predict giver_username_if_known" not in test_block


def test_infer_context_generic_semantic_keys_for_matching_like_task(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text(
        "order_id,group_order_id,provider_code,resource_type,target\nO1,G1,C1,T1,1\nO2,G1,C2,T2,0\n",
        encoding="utf-8",
    )
    (tmp_path / "test.csv").write_text(
        "order_id,group_order_id,provider_code,resource_type\nO3,G2,C1,T1\nO4,G3,C2,T3\n",
        encoding="utf-8",
    )
    cfg = AutoRealizeConfig.from_env()
    ctx = _infer_downstream_context(tmp_path, [], "build a matching plan", cfg)
    sem = ctx.get("semantic_keys", {})

    assert sem.get("entity_id_key") in {"order_id", "id"}
    assert sem.get("group_id_key") in {"group_order_id", "group_id", ""}
    rk = sem.get("resource_keys", [])
    assert isinstance(rk, list)
    assert any(x in rk for x in ["provider_code", "resource_type"])
    assert ctx.get("submission_columns", []) == []
    hints = ctx.get("submission_schema_hints", [])
    assert isinstance(hints, list)
    assert len(hints) >= 2
    assert "order_id" in hints


def test_submission_validator_semantic_keys_are_hints_not_hard_fail() -> None:
    df_src = pd.DataFrame({"id": ["a1", "a2"], "x": [1, 2]})
    df_out = pd.DataFrame({"id": ["a1", "a2"], "target": [0.1, 0.2]})
    ctx = {
        "submission_columns": ["id", "target"],
        "semantic_keys": {
            "entity_id_key": "order_id",
            "group_id_key": "group_order_id",
            "resource_keys": ["provider_code", "resource_type"],
        },
    }
    issues = _validate_generated_submission(df_out, ctx=ctx, source_df=df_src)
    assert any(str(x).startswith("hint_") for x in issues)
    assert not any("column_order_mismatch" in str(x) for x in issues)
