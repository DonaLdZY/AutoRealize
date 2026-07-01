from pathlib import Path

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.modules.data_cognition import _table_schema_signature
from autorealize.parsers.table_parser import TableParser
from autorealize.pipeline import AutoRealizePipeline
from autorealize.profiling.stats import excel_sheet_groups_from_profiles, profile_excel_sheets


def test_table_parser_only_reads_preview_for_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "large_like.csv"
    csv_path.write_text("id,value\n1,10\n2,20\n3,30\n", encoding="utf-8")

    parser = TableParser(preview_rows=2, sample_rows=2)
    parsed = parser.parse(csv_path)

    assert parsed.columns == ["id", "value"]
    assert len(parsed.preview) == 2
    assert parsed.metadata["shape"][1] == 2
    assert parsed.metadata["preview_rows_used"] == 2


def test_excel_small_workbook_deep_profiles_every_sheet(tmp_path: Path) -> None:
    workbook = tmp_path / "small.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        for idx in range(3):
            pd.DataFrame({"id": [1, 2], "value": [idx, idx + 1]}).to_excel(
                writer,
                sheet_name=f"Sheet{idx + 1}",
                index=False,
            )

    profiles = profile_excel_sheets(
        workbook,
        max_rows=1,
        preview_rows=2,
        large_threshold_bytes=1024 * 1024,
        full_profile_sheet_threshold=10,
    )

    assert len(profiles) == 3
    assert all(p["is_deep_profiled"] for p in profiles)
    assert all(p["profile_policy"] == "full_all_sheets" for p in profiles)
    assert all(p["profile_rows_limit"] is None for p in profiles)
    assert all(p["preview"] for p in profiles)


def test_excel_parser_preserves_raw_top_rows_for_every_sheet(tmp_path: Path) -> None:
    workbook = tmp_path / "instructions.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(
            [
                ["说明", "订单主表，第二行开始才是业务提示"],
                ["订单号", "数量"],
                ["O1", 3],
            ]
        ).to_excel(writer, sheet_name="订单说明", index=False, header=False)
        pd.DataFrame({"line_id": ["L1"], "cost": [12.5]}).to_excel(writer, sheet_name="成本明细", index=False)

    parsed = TableParser(preview_rows=3).parse(workbook)
    sheets = parsed.metadata["excel_sheets"]

    assert [s["sheet_name"] for s in sheets] == ["订单说明", "成本明细"]
    assert all("raw_preview" in s for s in sheets)
    assert sheets[0]["raw_preview"][0][0] == "说明"
    assert sheets[0]["raw_preview"][1][0] == "订单号"
    assert "订单说明" in parsed.metadata["excel_sheet_names"]
    assert "成本明细" in parsed.metadata["excel_sheet_names"]
    assert "line_id" in parsed.columns


def test_excel_layout_inference_detects_headerless_and_non_default_header(tmp_path: Path) -> None:
    workbook = tmp_path / "layout_cases.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame([["A", 1], ["B", 2]]).to_excel(
            writer,
            sheet_name="headerless_map",
            index=False,
            header=False,
        )
        pd.DataFrame([["note", "open row"], ["id", "amount"], ["A", 1]]).to_excel(
            writer,
            sheet_name="header_row_1",
            index=False,
            header=False,
        )

    profiles = profile_excel_sheets(
        workbook,
        max_rows=10,
        preview_rows=5,
        large_threshold_bytes=1024 * 1024,
        full_profile_sheet_threshold=10,
    )
    by_sheet = {str(p["sheet_name"]): p for p in profiles}

    assert by_sheet["headerless_map"]["layout_kind"] == "headerless_table"
    assert by_sheet["headerless_map"]["read_strategy_kind"] == "header_none_table"
    assert "header=None" in by_sheet["headerless_map"]["recommended_read"]
    assert by_sheet["header_row_1"]["layout_kind"] == "non_default_header"
    assert by_sheet["header_row_1"]["detected_header_row"] == 1
    assert "header=1" in by_sheet["header_row_1"]["recommended_read"]


def test_excel_schema_signature_is_layout_aware_for_sampling(tmp_path: Path) -> None:
    first = tmp_path / "sample_01_map.xlsx"
    second = tmp_path / "sample_02_map.xlsx"
    with pd.ExcelWriter(first) as writer:
        pd.DataFrame([["A", 1], ["B", 2]]).to_excel(writer, sheet_name="map", index=False, header=False)
    with pd.ExcelWriter(second) as writer:
        pd.DataFrame([["X", 9], ["Y", 10]]).to_excel(writer, sheet_name="map", index=False, header=False)

    sig_a = _table_schema_signature(first)
    sig_b = _table_schema_signature(second)

    assert sig_a["signature"] == sig_b["signature"]
    assert sig_a["schema_basis"] == "excel_sheet_layout_and_header_strategy"
    assert sig_a["layout_summary"][0]["layout_kind"] == "headerless_table"
    assert sig_a["layout_summary"][0]["columns_preview"] == ["col1:text", "col2:number"]


def test_excel_many_similar_sheets_group_and_profile_representatives(tmp_path: Path) -> None:
    workbook = tmp_path / "monthly.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        for idx in range(12):
            pd.DataFrame({"order_id": [1, 2], "cost": [idx, idx + 1]}).to_excel(
                writer,
                sheet_name=f"2025-{idx + 1:02d}",
                index=False,
            )

    profiles = profile_excel_sheets(
        workbook,
        max_rows=1,
        preview_rows=2,
        large_threshold_bytes=1024 * 1024,
        full_profile_sheet_threshold=10,
        representatives_per_group=1,
    )
    groups = excel_sheet_groups_from_profiles(profiles)

    assert len(profiles) == 12
    assert len(groups) == 1
    assert groups[0]["sheet_count"] == 12
    assert sum(1 for p in profiles if p["is_deep_profiled"]) == 1
    assert all(p["columns"] == ["order_id", "cost"] for p in profiles)
    assert all(p["preview"] for p in profiles)
    assert all(p["raw_preview"] for p in profiles)


def test_pipeline_records_sampled_column_profiles(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    pd.DataFrame({"store_id": ["s1", "s2", "s3", "s4"], "sales_amount": [10.0, 20.0, 30.0, 40.0]}).to_csv(
        input_root / "sales.csv", index=False
    )

    cfg = AutoRealizeConfig.from_env()
    cfg.data.table_profile_sample_rows = 2
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="forecast next month sales",
        run_name="run_001_large_table_sampling",
    )

    data_description = (run_dir / "realize_report" / "data_description.md").read_text(encoding="utf-8")
    assert "sales_amount" in data_description
    assert (run_dir / "description.md").exists()
    assert (run_dir / "sample_submission.csv").exists()


def test_generated_sample_submission_without_predict_set_is_small_sample(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    rows = 50
    pd.DataFrame(
        {
            "store_id": [f"s{i}" for i in range(rows)],
            "dtdate": ["2025-01-01"] * rows,
            "sales_amount": list(range(rows)),
        }
    ).to_csv(input_root / "sales.csv", index=False)

    cfg = AutoRealizeConfig.from_env()
    cfg.data.generated_sample_submission_max_rows = 7
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint="forecast next month sales",
        run_name="run_002_small_generated_submission",
    )

    out_df = pd.read_csv(run_dir / "sample_submission.csv")
    assert len(out_df) == 7
    report = (run_dir / "realize_report" / "submission_report.json").read_text(encoding="utf-8")
    assert '"sample_rows_only": true' in report
    desc = (run_dir / "description.md").read_text(encoding="utf-8")
    assert "sample_submission.csv" in desc


def test_infer_context_does_not_treat_single_train_table_as_predict_set(tmp_path: Path) -> None:
    from autorealize.pipeline import _infer_downstream_context

    pd.DataFrame(
        {
            "store_id": ["s1", "s2"],
            "dtdate": ["2025-01-01", "2025-01-02"],
            "sales_amount": [10.0, 12.0],
        }
    ).to_csv(tmp_path / "sales.csv", index=False)

    cfg = AutoRealizeConfig.from_env()
    ctx = _infer_downstream_context(tmp_path, [], "forecast next month sales", cfg)

    assert ctx["train_table"] == "sales.csv"
    assert ctx["predict_table"] == ""
    assert ctx["predict_columns"] == []


def test_eval_ambiguity_accepts_fixed_time_window_protocol() -> None:
    from autorealize.report_writer import eval_ambiguity_defects

    text = """
## Overview
- Task Type: time_series_regression
## Evaluation
### Computation Scope
- `y_true` source: validation window labels from `sales_amount`.
### Validation Protocol
- Use a strict rolling-window time split: train window length = 180 days, validation window length = 30 days, step = 30 days.
- Time-series tasks must be split by chronological order and must not leak future information.
### Reporting Rules
- Keep at least 6 decimal places for the primary metric.
"""
    defects = eval_ambiguity_defects(text)
    assert not any("train window" in x or "validation window" in x or "step" in x for x in defects)
