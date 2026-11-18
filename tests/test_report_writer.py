import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.models import DescriptionProtocolBundle, EvaluationContractReview, FileRole, FileSummary
from autorealize.prompts.manager import PromptManager
from autorealize.report_writer import (
    apply_evaluation_contract,
    build_automl_context_pack,
    build_data_access_protocol,
    build_data_schema_contract,
    coverage_defects,
    description_quality_check,
    description_protocol_bundle_defects,
    evaluation_contract_defects,
    finalize_description_markdown,
    render_automl_context_markdown,
    render_description_protocol_markdown,
)


def test_coverage_defects_detect_short_output() -> None:
    original = "这是一个非常详细的需求说明，包含评价指标、submission格式、字段说明和业务约束。" * 30
    generated = "简短描述"
    defects = coverage_defects(generated, original)
    assert any("shorter than original" in d for d in defects)


def test_description_quality_check_detects_missing_sections() -> None:
    text = "# 赛题说明\n\n## 任务概述\nx"
    defects = description_quality_check(text)
    assert any("缺少章节" in d for d in defects)


def test_description_quality_check_detects_internal_pipeline_language() -> None:
    text = (
        "## 任务概述\nx\n"
        "## 数据与读取方式\nx\n"
        "## 字段说明\nx\n"
        "## 任务定义\nP1 数据认知\n"
        "## 评估协议\n### 计算公式\nx\n### 计算范围\nx\n### 验证协议\nx\n### 结果报告要求\nx\n"
        "## 输出或提交格式\nx\n## 关键约束与注意事项\nx\n"
    )
    defects = description_quality_check(text)
    assert any("内部的流程描述" in d for d in defects)


def test_description_quality_check_detects_ambiguous_words() -> None:
    text = (
        "## 任务概述\nx\n"
        "## 数据与读取方式\nx\n"
        "## 字段说明\nx\n"
        "## 任务定义\nx\n"
        "## 评估协议\n### 计算公式\nx\n### 计算范围\nx\n### 验证协议\n推荐使用滚动窗口\n### 结果报告要求\nx\n"
        "## 输出或提交格式\nx\n## 关键约束与注意事项\nx\n"
    )
    defects = description_quality_check(text)
    assert any("评估歧义措辞" in d for d in defects)


def test_optimization_description_avoids_hardcoded_output_contract_templates() -> None:
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="根据订单和承运商数据生成符合评估要求的最低成本方案。",
        task_goal="根据订单和承运商数据生成符合评估要求的提交文件。",
        optimization={
            "input_instance": "订单数据、承运商成本数据和车辆约束共同构成待求解实例。",
            "decision_variables": ["订单分配到承运商", "订单分配到车型"],
            "objective": "最小化总成本与约束违约惩罚之和。",
            "hard_constraints": ["每个订单必须被合法分配一次"],
            "feasibility_checks": ["检查缺行、重复订单和非法车型"],
            "solution_representation": "输出覆盖全部订单的方案表。",
        },
        output={
            "output_kind": "solution_table",
            "output_filename": "submission.csv",
            "sample_submission_required": False,
            "columns": ["订单号", "原始订单号", "限制车型"],
            "row_unit": "一行表示一个订单的决策结果",
            "format_rules": ["列必须来自任务证据，不得新增无关列"],
            "no_sample_submission_reason": "该优化任务没有权威样例时以方案协议为准。",
        },
        evaluation_summary="总成本加约束惩罚，越小越好。",
        constraints=["订单唯一分配"],
    )
    text = render_description_protocol_markdown(
        bundle,
        file_summaries=[],
        downstream_context={
            "task_hint": "根据订单和承运商数据生成符合评估要求的提交文件。",
            "problem_paradigm": "static_optimization",
            "generate_sample_submission": False,
        },
    )

    forbidden = [
        "### Expected Output Contract",
        "决策主键列",
        "动作/分配列",
        "决策键列",
        "固定离线回放集",
        "容量、时窗、唯一分配",
    ]
    for phrase in forbidden:
        assert phrase not in text
    assert "## 输出或提交格式" in text
    assert "输出类型：solution_table" in text
    assert "列顺序：`订单号`，`原始订单号`，`限制车型`" in text
    assert "不得把任务硬套成固定 `id,target` 两列表" not in text


def test_automl_context_keeps_problem_structure_and_required_rl_separate() -> None:
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        problem_structure="decision_optimization",
        decision_temporality="static_instance",
        environment_source="constructed_from_static_data",
        required_method_families=["reinforcement_learning"],
        allowed_method_families=["reinforcement_learning", "hybrid_policy"],
        rl_formulation_candidates=[
            {
                "formulation_type": "constructive_policy",
                "state_candidates": ["current partial solution", "remaining resources"],
                "action_candidates": ["select one legal next decision"],
                "transition_description": "Apply one action to the partial solution.",
                "action_mask_basis": ["task-defined feasibility constraints"],
                "reward_alignment": "Terminal return matches score_solution.",
                "terminal_condition": "No decision units remain.",
                "episode_generation": "Create episodes from static problem instances.",
            }
        ],
        task_goal="Produce a task-valid decision artifact.",
        evaluation_summary="Use one scalar score.",
    )
    pack = build_automl_context_pack(
        bundle,
        [],
        downstream_context={
            "problem_paradigm_review": {
                "rl_required": True,
                "explicit_rl_requested": True,
            }
        },
    )
    text = render_automl_context_markdown(pack)

    assert pack.problem_structure == "decision_optimization"
    assert pack.environment_source == "constructed_from_static_data"
    assert pack.method_strategy["rl_required"] is True
    assert pack.method_strategy["first_draft_method"] == "reinforcement_learning"
    assert "rl_formulation_candidates" in text
    assert "Build a deterministic" not in text
    assert "RL is a later" not in text


def test_automl_handoff_preserves_long_constraint_and_output_contracts() -> None:
    text = render_automl_context_markdown({
        "constraints": [f"Mandatory constraint {index}" for index in range(80)],
        "evaluation_contract": {"submission_checks": [f"Exact check {index}" for index in range(70)]},
        "output_contract": {"columns": [f"output_{index}" for index in range(150)]},
        "data_schema_contract": {"tables": [{"physical_columns_exact": [f"physical_{index}" for index in range(255)]}]},
    })
    assert "Mandatory constraint 79" in text
    assert "Exact check 69" in text
    assert "output_149" in text
    assert "physical_254" in text


def test_data_access_target_prefers_resolved_task_over_label_heuristics() -> None:
    fs = FileSummary(
        path="Steel_industry_data.csv",
        role=FileRole.raw_data_table,
        summary="Energy observations.",
        columns=["date", "Usage_kWh", "Day_of_week", "Load_Type"],
        column_semantics={"Load_Type": "load class label"},
        source_metadata={"target_column_hint": "Usage_kWh"},
    )

    assert build_data_access_protocol([fs]).files[0].target_fields == ["Usage_kWh"]


def test_data_access_target_does_not_match_y_inside_arbitrary_column() -> None:
    fs = FileSummary(
        path="train.csv", role=FileRole.raw_data_table, summary="Training observations.",
        columns=["Day_of_week", "capacity", "y"],
    )

    assert build_data_access_protocol([fs]).files[0].target_fields == ["y"]


def test_data_access_protocol_renders_whitespace_csv_read_example() -> None:
    fs = FileSummary(
        path="trainset.csv",
        role=FileRole.raw_data_table,
        summary="空白分隔 CSV，字段中包含逗号列表。",
        columns=["user_id", "exposed_items", "labels"],
        source_metadata={
            "csv_dialect": {
                "sep": r"\s+",
                "engine": "python",
                "inferred": True,
                "reason": "whitespace_columns_with_comma_lists",
            }
        },
    )
    protocol = build_data_access_protocol([fs])
    bundle = DescriptionProtocolBundle(
        problem_paradigm="ml_dl_prediction",
        overview="预测用户交互结果。",
        data_access=protocol,
        ml_dl={"target": "labels", "prediction_unit": "一行用户曝光记录"},
        output={"output_kind": "submission_table", "output_filename": "submission.csv"},
    )
    text = render_description_protocol_markdown(bundle, [fs])
    assert "pd.read_csv('./input/trainset.csv', sep=r'\\s+', engine='python', encoding='utf-8-sig')" in text


def test_single_profiled_excel_sheet_keeps_explicit_header_and_source_columns() -> None:
    fs = FileSummary(path="relations.xlsx", role="raw_data_table", summary="relations", columns=["source", "target"],
        source_metadata={"excel_sheet_names": ["Relations"], "excel_sheet_profiles": [{
            "sheet_name": "Relations", "layout_kind": "standard_table", "detected_header_row": 0,
            "columns": ["source", "target"], "shape": [2, 2]}]})
    access = build_data_access_protocol([fs]).files[0]
    schema = build_data_schema_contract([fs])["tables"][0]
    assert access.read_contract["pandas_kwargs"] == {"sheet_name": "Relations", "header": 0, "skiprows": 0, "engine": "openpyxl"}
    assert access.read_contract["population_verified"] is False
    assert access.read_contract["columns_exact"] == ["source", "target"]
    assert schema["read_contract"] == access.read_contract
    assert schema["read_example"] == access.read_example


def test_multisheet_workbook_does_not_choose_one_sheet_from_its_profile() -> None:
    fs = FileSummary(path="multi.xlsx", role="raw_data_table", summary="two sources",
        source_metadata={"excel_sheet_names": ["A", "B"], "excel_sheet_profiles": [
            {"sheet_name": name, "layout_kind": "standard_table", "detected_header_row": 0} for name in ["A", "B"]]})
    access = build_data_access_protocol([fs]).files[0]
    assert "sheet_name=None" in access.read_example
    assert access.read_contract == {}


def test_nondefault_and_headerless_sheet_contracts_preserve_reader_semantics() -> None:
    fs = FileSummary(path="layout.xlsx", role="raw_data_table", summary="mixed layouts",
        source_metadata={"excel_sheet_names": ["offset", "raw"], "excel_sheet_profiles": [
            {"sheet_name": "offset", "layout_kind": "non_default_header", "detected_header_row": 2,
             "columns": ["customer", "longitude"], "shape": [10, 2]},
            {"sheet_name": "raw", "layout_kind": "headerless_table", "detected_header_row": None,
             "columns": ["vehicle", "capacity"], "shape": [10, 2]}]})
    pack = build_automl_context_pack(DescriptionProtocolBundle(data_access=build_data_access_protocol([fs])), [fs],
        compiled_context={"table_cards": [{"table_id": f"layout.xlsx::{name}", "source_file": "layout.xlsx",
                                         "sheet_name": name} for name in ["offset", "raw"]]})
    readers = {row["sheet_name"]: row for row in pack.data_access}
    schema = {row["sheet_name"]: row for row in pack.data_schema_contract["tables"]}
    assert readers["offset"]["read_contract"]["pandas_kwargs"]["header"] == 2
    assert readers["offset"]["read_example"] == schema["offset"]["read_example"]
    assert readers["raw"]["read_contract"]["pandas_kwargs"]["header"] is None
    assert readers["raw"]["columns_scope"] == "integer_position_raw_layout"
    assert not readers["raw"].get("columns")
    assert not schema["raw"].get("physical_columns_exact")
    assert schema["raw"]["profiled_signature_fields"] == ["vehicle", "capacity"]


def test_grouped_schema_does_not_erase_individual_sheet_readers() -> None:
    files = [FileSummary(path=f"cost/carrier{number:02}.xlsx", role="raw_data_table", summary="cost",
        source_metadata={"excel_sheet_names": ["rates"], "excel_sheet_profiles": [{
            "sheet_name": "rates", "layout_kind": "standard_table", "detected_header_row": 0,
            "columns": ["carrier", "rate"], "shape": [10, 2]}]}) for number in range(1, 12)]
    pack = build_automl_context_pack(DescriptionProtocolBundle(data_access=build_data_access_protocol(files)), files,
        compiled_context={"table_cards": [{"table_id": f"{fs.path}::rates", "source_file": fs.path,
                                         "sheet_name": "rates"} for fs in files]})
    assert len(pack.data_access) == len(files)
    for row in pack.data_access:
        assert row["read_contract"]["pandas_kwargs"]["header"] == 0
        assert row["path"] in row["read_example"]
        assert "header=None" not in row["read_example"]


def test_actual_excel_read_takes_precedence_over_unverified_layout_guess() -> None:
    fs = FileSummary(path="addresses.xlsx", role="raw_data_table", summary="addresses",
        source_metadata={"excel_sheet_names": ["A"], "excel_sheet_profiles": [{
            "sheet_name": "A", "layout_kind": "non_default_header", "detected_header_row": 2,
            "columns": ["customer", "longitude"], "shape": [10, 2],
            "profiled_read_contract": {"columns_exact": ["customer", "longitude"], "read_parameters_verified": True,
                "pandas_kwargs": {"sheet_name": "A", "header": 0, "skiprows": 0}}}]})
    access = build_data_access_protocol([fs]).files[0]
    schema = build_data_schema_contract([fs])["tables"][0]
    assert access.read_contract["pandas_kwargs"]["header"] == 0
    assert access.read_example == schema["read_example"]
    assert "header=2" not in access.read_example


def test_workbook_selects_only_fully_verified_single_populated_sheet() -> None:
    fs = FileSummary(path="lab.xls", role="raw_data_table", summary="10-row preview",
        source_metadata={"excel_sheet_names": ["Sheet1", "Sheet2"], "excel_sheet_profiles": [
            {"sheet_name": "Sheet1", "layout_kind": "standard_table", "detected_header_row": 0,
             "columns": ["target"], "shape": [10, 1], "shape_profiled": [8036, 1], "profile_policy": "full_all_sheets"},
            {"sheet_name": "Sheet2", "columns": [], "shape": [0, 0], "shape_profiled": [0, 0], "profile_policy": "full_all_sheets"}]})
    access = build_data_access_protocol([fs]).files[0]
    assert "sheet_name='Sheet1'" in access.read_example
    assert access.read_contract == build_data_schema_contract([fs])["tables"][0]["read_contract"]
    assert "verified_rows=8036" in access.row_grain and "verified_rows=0" in access.row_grain
    assert "10-row preview" not in access.row_grain
    pack = build_automl_context_pack(
        DescriptionProtocolBundle(data_access=build_data_access_protocol([fs])), [fs],
        compiled_context={"table_cards": [
            {"table_id": f"lab.xls::{name}", "source_file": "lab.xls", "sheet_name": name}
            for name in ["Sheet1", "Sheet2"]]})
    readers = {entry["sheet_name"]: entry for entry in pack.data_access}
    assert readers["Sheet1"]["read_contract"]["pandas_kwargs"]["sheet_name"] == "Sheet1"
    assert readers["Sheet2"]["read_contract"]["pandas_kwargs"]["sheet_name"] == "Sheet2"
    assert readers["Sheet2"]["read_contract"]["pandas_kwargs"]["header"] is None
    assert "Sheet1" not in readers["Sheet2"]["read_example"]
    assert "target" not in readers["Sheet2"]["read_contract"]["columns_exact"]
    fs.source_metadata["excel_sheet_profiles"][1]["profile_policy"] = "sampled"
    uncertain = build_data_access_protocol([fs]).files[0]
    assert "sheet_name=None" in uncertain.read_example
    assert uncertain.read_contract == {}


def test_data_access_protocol_preserves_executable_csv_read_contract() -> None:
    read_contract = {
        "schema_version": "autorealize.csv_read_contract.v1",
        "verified": True,
        "path": "input_items.csv",
        "reader": "pandas.read_csv",
        "pandas_kwargs": {"sep": ";", "decimal": ",", "encoding": "utf-8-sig"},
        "validated_shape": [203, 19],
        "validated_columns_exact": ["Item ident", "Weight"],
    }
    fs = FileSummary(
        path="input_items.csv",
        role=FileRole.raw_data_table,
        summary="Semicolon-delimited item data.",
        columns=["Item ident", "Weight"],
        source_metadata={
            "csv_dialect": {
                "sep": ";",
                "decimal": ",",
                "inferred": True,
                "reason": "stable_semicolon_delimiter_with_decimal_comma",
            },
            "csv_encoding": "utf-8-sig",
            "csv_read_contract": read_contract,
        },
    )

    protocol = build_data_access_protocol([fs])

    assert protocol.files[0].read_example == (
        "pd.read_csv('./input/input_items.csv', sep=';', decimal=',', encoding='utf-8-sig')"
    )
    assert protocol.files[0].read_contract["path"] == "input_items.csv"
    assert protocol.files[0].read_contract["read_example"] == protocol.files[0].read_example
    assert protocol.files[0].read_contract["population_verified"] is False
    assert "validated_shape" not in protocol.files[0].read_contract
    assert protocol.files[0].read_contract["reader_reported_shape"] == [203, 19]
    schema = build_data_schema_contract([fs])
    assert schema["tables"][0]["read_contract"]["pandas_kwargs"]["decimal"] == ","


def test_legacy_preview_dtype_contract_is_corrected_in_both_handoffs() -> None:
    fs = FileSummary(path="values.csv", role=FileRole.raw_data_table, summary="Numeric observations", columns=["value"], source_metadata={
        "csv_read_contract": {"validated_dtypes": {"value": "int64"}, "preview_rows_validated": 2},
    })
    access = build_data_access_protocol([fs]).files[0].read_contract
    schema = build_data_schema_contract([fs])["tables"][0]["read_contract"]
    for contract in (access, schema):
        assert contract["observed_preview_dtypes"] == {"value": "int64"}
        assert contract["dtype_validation_scope"] == "preview_only"
        assert "validated_dtypes" not in contract
    assert fs.source_metadata["csv_read_contract"]["validated_dtypes"] == {"value": "int64"}


def test_physical_schema_order_comes_from_reader_not_semantic_field_priority():
    from autorealize.context_compiler import build_source_coverage_ledger
    fs = FileSummary(path="data.csv", role=FileRole.raw_data_table, summary="Measurements",
        columns=["ID", "temperature", "Type"], source_metadata={
            "shape": [10000, 3], "profile_sampling": {"sampled": True},
            "csv_read_contract": {"verified": True, "validated_shape": [10000, 3],
                                  "validated_columns_exact": ["ID", "Type", "temperature"]}})
    schema = build_data_schema_contract([fs])["tables"][0]
    assert schema["physical_columns_exact"] == ["ID", "Type", "temperature"]
    ledger = build_source_coverage_ledger(file_summaries=[fs])["entries"][0]
    assert ledger["schema_signature_fields"] == ["ID", "Type", "temperature"]
    assert ledger["schema_signature_scope"].startswith("preview_only")
    assert not ledger.get("verified_shape")
    assert ledger["reader_reported_shape"] == [10000, 3]


def test_repeated_structured_files_are_grouped_in_description() -> None:
    files = [
        FileSummary(
            path=f"cost/carrier{i:02d}_cost.xlsx",
            role=FileRole.raw_data_table,
            summary="carrier cost table",
            columns=["lane", "vehicle_type", "price"],
            column_profiles=[
                {"name": "lane", "logical_type": "categorical", "null_count": 0, "row_count": 10},
                {"name": "vehicle_type", "logical_type": "categorical", "null_count": 0, "row_count": 10},
                {"name": "price", "logical_type": "float", "null_count": 0, "row_count": 10},
            ],
        )
        for i in range(1, 4)
    ]
    protocol = build_data_access_protocol(files)
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="generate a feasible low-cost plan",
        data_access=protocol,
        optimization={
            "input_instance": "orders and carrier costs",
            "objective": "minimize total cost",
            "decision_variables": ["carrier assignment"],
            "hard_constraints": ["all orders assigned once"],
            "feasibility_checks": ["check missing assignments"],
            "solution_representation": "assignment table",
        },
        evaluation_summary="total cost, lower is better",
    )

    text = render_description_protocol_markdown(bundle, files)

    assert "### cost/carrier{id}_cost.xlsx" in text
    assert "for path in sorted(input_dir.glob('cost/carrier*_cost.xlsx')):" in text
    assert "pd.read_excel(path)" in text
    assert "cost/carrier01_cost.xlsx" not in text
    assert "cost/carrier02_cost.xlsx" not in text
    assert "同结构文件组" not in text
    assert "同结构字段组" not in text
    assert "覆盖文件示例" not in text
    assert text.index("## 任务定义") < text.index("## 数据与读取方式")
    assert text.rfind("## 数据与读取方式") > text.rfind("## 关键约束与注意事项")


def test_repeated_file_description_lists_common_and_variant_fields() -> None:
    files = [
        FileSummary(
            path="contracts/承运商01成本.csv",
            role=FileRole.raw_data_table,
            summary="contract table A",
            columns=["lane", "vehicle_type", "cost", "settlement_code"],
        ),
        FileSummary(
            path="contracts/承运商02成本.csv",
            role=FileRole.raw_data_table,
            summary="contract table B",
            columns=["lane", "vehicle_type", "cost", "carrier_code"],
        ),
        FileSummary(
            path="contracts/承运商03成本.csv",
            role=FileRole.raw_data_table,
            summary="contract table C",
            columns=["lane", "vehicle_type", "cost", "region_code"],
        ),
    ]
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="assign orders to carriers",
        data_access=build_data_access_protocol(files),
        optimization={"objective": "minimize cost", "solution_representation": "assignment table"},
    )

    text = render_description_protocol_markdown(bundle, files)

    assert "同组共通字段" in text
    assert "`lane`" in text
    assert "同组差异字段" in text
    assert "settlement_code" in text
    assert "carrier_code" in text


def test_schema_contract_marks_group_union_and_variant_fields() -> None:
    files = [
        FileSummary(
            path="contracts/承运商01成本.csv",
            role=FileRole.raw_data_table,
            summary="contract table A",
            columns=["lane", "vehicle_type", "cost", "settlement_code"],
        ),
        FileSummary(
            path="contracts/承运商02成本.csv",
            role=FileRole.raw_data_table,
            summary="contract table B",
            columns=["lane", "vehicle_type", "cost", "carrier_code"],
        ),
        FileSummary(
            path="contracts/承运商03成本.csv",
            role=FileRole.raw_data_table,
            summary="contract table C",
            columns=["lane", "vehicle_type", "cost", "region_code"],
        ),
    ]

    contract = build_data_schema_contract(files)
    table = contract["tables"][0]
    group = table["schema_group"]

    assert "settlement_code" in table["physical_columns_exact"]
    assert group["shared_physical_columns_exact"] == ["lane", "vehicle_type", "cost"]
    assert group["column_variant_count"] == 3
    assert any(row["fields"] == ["settlement_code"] for row in group["variant_fields_by_file"])


def test_composite_carrier_identifier_stays_in_repeated_schema_group() -> None:
    files = [
        FileSummary(
            path=f"成本/{name} 承运商成本.xlsx",
            role=FileRole.raw_data_table,
            summary="承运商成本表",
            columns=["承运商代码", "计费费率"],
        )
        for name in ["承运商01BZWL01", "承运商02fsd", "承运商14SZ-ZT-01"]
    ]
    contract = build_data_schema_contract(files)
    assert len(contract["tables"]) == 1
    assert contract["tables"][0]["table_id"] == "成本/承运商{id} 承运商成本.xlsx"
    assert contract["tables"][0]["schema_group"]["file_count"] == 3

def test_repeated_structured_file_fields_are_group_level_not_sample_constants() -> None:
    carrier_ids = ["BZWL01", "fsd", "fy"]
    files = [
        FileSummary(
            path=f"成本/承运商{i:02d}{carrier_id} 承运商成本.xlsx",
            role=FileRole.raw_data_table,
            summary="承运商成本表",
            columns=["结算方代码", "车型", "成本"],
            column_semantics={
                "结算方代码": f"结算方代码，此处为{carrier_id}",
                "车型": "承运商可用车型",
                "成本": "运输成本",
            },
            column_semantic_meta={
                "结算方代码": {"source": "llm_field_description"},
                "车型": {"source": "llm_field_description"},
                "成本": {"source": "llm_field_description"},
            },
            column_profiles=[
                {"name": "结算方代码", "logical_type": "categorical", "null_count": 0, "row_count": 10, "top_values": [carrier_id]},
                {"name": "车型", "logical_type": "categorical", "null_count": 0, "row_count": 10},
                {"name": "成本", "logical_type": "float", "null_count": 0, "row_count": 10, "numeric_stats": {"min": i * 10, "max": i * 20}},
            ],
        )
        for i, carrier_id in enumerate(carrier_ids, start=1)
    ]
    protocol = build_data_access_protocol(files)
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="生成低成本运输方案",
        data_access=protocol,
        optimization={
            "input_instance": "订单与承运商成本",
            "objective": "minimize total cost",
            "decision_variables": ["carrier assignment"],
            "hard_constraints": ["all orders assigned once"],
            "feasibility_checks": ["check missing assignments"],
            "solution_representation": "assignment table",
        },
        evaluation_summary="total cost, lower is better",
    )

    text = render_description_protocol_markdown(bundle, files)

    assert "### 成本/承运商{id} 承运商成本.xlsx" in text
    assert "BZWL01" not in text
    assert "fsd" not in text
    assert "fy" not in text
    assert "此处为" not in text
    assert "`结算方代码`：结算方/承运商标识字段" in text
    assert "跨文件随文件名中的 `{id}` 部分变化" in text


def test_repeated_files_with_slightly_different_columns_are_grouped() -> None:
    files = [
        FileSummary(
            path="成本/承运商01BZWL01 承运商成本.xlsx",
            role=FileRole.raw_data_table,
            summary="成本合同表",
            columns=["合同费率Id", "合同编号", "结算方代码", "结算方名称", "起点", "终点", "车型", "计费费率"],
        ),
        FileSummary(
            path="成本/承运商02fsd 承运商成本.xlsx",
            role=FileRole.raw_data_table,
            summary="成本合同表",
            columns=["合同费率Id", "合同编号", "结算方代码", "结算方名称", "起点", "终点", "车型", "计费费率", "保底费"],
        ),
        FileSummary(
            path="成本/承运商06HQ01 承运商成本.xlsx",
            role=FileRole.raw_data_table,
            summary="成本合同表",
            columns=["合同费率Id", "合同编号", "承运商代码", "承运商名称", "起点", "终点", "车型", "计费费率"],
        ),
        FileSummary(
            path="成本/承运商11PDWL 承运商成本.xlsx",
            role=FileRole.raw_data_table,
            summary="成本合同表",
            columns=["合同费率Id", "合同编号", "承运商代码", "承运商名称", "起点", "终点", "车型", "计费费率", "合同有效标志"],
        ),
    ]
    protocol = build_data_access_protocol(files)
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="生成低成本运输方案",
        data_access=protocol,
        optimization={
            "input_instance": "订单与承运商成本",
            "objective": "minimize total cost",
            "decision_variables": ["carrier assignment"],
            "hard_constraints": ["all orders assigned once"],
            "feasibility_checks": ["check missing assignments"],
            "solution_representation": "assignment table",
        },
        evaluation_summary="total cost, lower is better",
    )

    text = render_description_protocol_markdown(bundle, files)
    pack = build_automl_context_pack(bundle, files)
    context = render_automl_context_markdown(pack)

    assert "### 成本/承运商{id} 承运商成本.xlsx" in text
    field_section = text.split("## 字段说明", 1)[1].split("## 数据与读取方式", 1)[0]
    data_access_section = text.split("## 数据与读取方式", 1)[1]
    assert field_section.count("### 成本/承运商{id} 承运商成本.xlsx") == 1
    assert data_access_section.count("### 成本/承运商{id} 承运商成本.xlsx") == 1
    assert "成本/承运商02fsd 承运商成本.xlsx" not in text
    assert "成本/承运商06HQ01 承运商成本.xlsx" not in text
    assert "`结算方代码`" in text
    assert "`承运商代码`" in text
    assert "成本/承运商{id} 承运商成本.xlsx" in context
    assert context.count("### 成本/承运商{id} 承运商成本.xlsx") == 1
    assert "成本/承运商02fsd 承运商成本.xlsx" not in context


def test_repeated_multisheet_workbook_renders_sheet_level_context() -> None:
    files = [
        FileSummary(
            path=f"成本/承运商{i:02d}{carrier} 承运商成本.xlsx",
            role=FileRole.raw_data_table,
            summary="承运商成本 workbook，包含费率表和计费规则 sheet",
            columns=["合同编号", "结算方代码", "起点", "终点", "计费费率", "计费单位"],
            column_semantics={
                "合同编号": "合同记录编号",
                "结算方代码": "承运商或结算主体代码",
                "起点": "合同线路起点",
                "终点": "合同线路终点",
                "计费费率": "合同单位计费费率",
                "计费单位": "费率适用单位",
            },
            column_semantic_meta={name: {"source": "llm_field_description"} for name in ["合同编号", "结算方代码", "起点", "终点", "计费费率", "计费单位"]},
            column_profiles=[
                {"name": "合同编号", "logical_type": "categorical", "null_count": 0, "row_count": 10},
                {"name": "结算方代码", "logical_type": "categorical", "null_count": 0, "row_count": 10, "top_values": [carrier]},
                {"name": "起点", "logical_type": "categorical", "null_count": 0, "row_count": 10},
                {"name": "终点", "logical_type": "categorical", "null_count": 0, "row_count": 10},
                {"name": "计费费率", "logical_type": "float", "null_count": 0, "row_count": 10, "numeric_stats": {"min": 1, "max": 100}},
                {"name": "计费单位", "logical_type": "categorical", "null_count": 0, "row_count": 10, "top_values": ["元/吨"]},
            ],
            source_metadata={
                "excel_sheet_names": ["导出信息", "Sheet1"],
                "sheet_field_descriptions": {
                    "导出信息": {
                        "合同编号": "费率合同编号",
                        "计费费率": "线路合同的单位费率",
                    },
                    "Sheet1": {
                        "合同计费方案": "计费方案名称或编号",
                        "算法规则": "文字描述的计费公式、保底比较和体积折算规则",
                    },
                },
                "excel_sheet_profiles": [
                    {
                        "sheet_name": "导出信息",
                        "shape": [10, 6],
                        "columns": ["合同编号", "结算方代码", "起点", "终点", "计费费率", "计费单位"],
                        "preview": [{"合同编号": "C1", "结算方代码": carrier, "起点": "A", "终点": "B", "计费费率": 12, "计费单位": "元/吨"}],
                        "raw_preview": [["合同编号", "结算方代码", "起点", "终点", "计费费率", "计费单位"]],
                        "column_profiles": [
                            {"name": "合同编号", "logical_type": "categorical", "null_count": 0, "row_count": 10},
                            {"name": "计费费率", "logical_type": "float", "null_count": 0, "row_count": 10, "numeric_stats": {"min": 1, "max": 100}},
                        ],
                    },
                    {
                        "sheet_name": "Sheet1",
                        "shape": [3, 2],
                        "columns": ["合同计费方案", "算法规则"],
                        "preview": [{"合同计费方案": "1-合同运价", "算法规则": "费用=max(交付重量*费率, 保底费)"}],
                        "raw_preview": [["合同计费方案", "算法规则"], ["1-合同运价", "费用=max(交付重量*费率, 保底费)"]],
                        "column_profiles": [
                            {"name": "合同计费方案", "logical_type": "categorical", "null_count": 0, "row_count": 3},
                            {"name": "算法规则", "logical_type": "text", "null_count": 0, "row_count": 3, "top_values": ["费用=max(交付重量*费率, 保底费)"]},
                        ],
                    },
                ],
            },
        )
        for i, carrier in enumerate(["BZWL01", "fsd", "fy"], start=1)
    ]
    protocol = build_data_access_protocol(files)
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="生成低成本运输方案",
        data_access=protocol,
        optimization={
            "input_instance": "订单与承运商成本",
            "objective": "minimize total cost",
            "decision_variables": ["carrier assignment"],
            "hard_constraints": ["all orders assigned once"],
            "feasibility_checks": ["check missing assignments"],
            "solution_representation": "assignment table",
        },
        evaluation_summary="total cost, lower is better",
    )

    text = render_description_protocol_markdown(bundle, files)
    pack = build_automl_context_pack(bundle, files)
    context = render_automl_context_markdown(pack)

    field_section = text.split("## 字段说明", 1)[1].split("## 数据与读取方式", 1)[0]
    data_access_section = text.split("## 数据与读取方式", 1)[1]
    assert field_section.count("### 成本/承运商{id} 承运商成本.xlsx") == 1
    assert data_access_section.count("### 成本/承运商{id} 承运商成本.xlsx") == 1
    assert "#### sheet: 导出信息" in text
    assert "#### sheet: Sheet1" in text
    assert "`算法规则`：文字描述的计费公式、保底比较和体积折算规则" in text
    assert "sheet_frames" in context
    assert "sheet=Sheet1" in context
    assert "费用=max(交付重量*费率, 保底费)" in context
    assert "pd.read_excel(path, sheet_name='Sheet1')" in context


def test_data_access_only_renders_critical_read_examples() -> None:
    files = [
        FileSummary(
            path="train.csv",
            role=FileRole.raw_data_table,
            summary="普通训练集",
            columns=["id", "x", "target"],
            source_metadata={"csv_dialect": {"sep": ",", "inferred": False, "reason": "default_comma"}},
        ),
        FileSummary(
            path="orders.xlsx",
            role=FileRole.raw_data_table,
            summary="订单 Excel，含主表和明细表",
            columns=["订单号", "数量"],
            source_metadata={"excel_sheet_names": ["订单表信息", "订单明细信息", "Sheet1"]},
        ),
    ]
    protocol = build_data_access_protocol(files)
    bundle = DescriptionProtocolBundle(
        problem_paradigm="ml_dl_prediction",
        overview="预测目标值",
        data_access=protocol,
        ml_dl={"target": "target", "prediction_unit": "one row", "validation_design": "holdout"},
        evaluation_summary="RMSE, lower is better",
    )

    text = render_description_protocol_markdown(bundle, files)
    data_access_section = text.split("## 数据与读取方式", 1)[1]

    assert "### train.csv" not in data_access_section
    assert "### orders.xlsx" in data_access_section
    assert "sheet_name=None" in data_access_section
    assert "不要依赖 `pd.read_excel(path)` 默认读取第一个工作表" in data_access_section


def test_prompt_manager_injects_language_prefix_only_for_system_prompts() -> None:
    cfg = AutoRealizeConfig()
    cfg.prompt.output_language = "zh"
    manager = PromptManager(cfg)

    system_prompt = manager.load("system/ml_dl_description_protocol.md")
    fewshot = manager.load("fewshot/task_classifier_fewshot.json")

    assert system_prompt.startswith("输出语言要求：")
    assert not fewshot.startswith("输出语言要求：")


def test_description_protocol_bundle_defects_are_paradigm_specific() -> None:
    ml_bundle = DescriptionProtocolBundle(
        problem_paradigm="ml_dl_prediction",
        overview="Predict a target.",
        data_access=build_data_access_protocol(
            [
                FileSummary(
                    path="train.csv",
                    role=FileRole.raw_data_table,
                    summary="training data",
                    columns=["id", "x", "target"],
                )
            ]
        ),
        ml_dl={"train_data": "train.csv", "prediction_unit": "one row"},
        evaluation_summary="RMSE",
    )
    defects = description_protocol_bundle_defects(ml_bundle)
    assert "description_protocol missing ml_dl.target" in defects
    assert "description_protocol missing ml_dl.validation_design" in defects

    opt_bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="Optimize assignments.",
        data_access=ml_bundle.data_access,
        optimization={"input_instance": "orders and resources", "objective": "minimize cost"},
        evaluation_summary="total cost",
    )
    defects = description_protocol_bundle_defects(opt_bundle)
    assert "description_protocol missing optimization.decision_variables" in defects
    assert "description_protocol missing optimization.hard_constraints" in defects

    rl_bundle = DescriptionProtocolBundle(
        problem_paradigm="reinforcement_learning",
        overview="Learn a policy.",
        data_access=ml_bundle.data_access,
        rl={"state": "inventory", "action": "order quantity", "reward": "profit"},
        evaluation_summary="average return",
    )
    defects = description_protocol_bundle_defects(rl_bundle)
    assert "description_protocol missing rl.transition" in defects
    assert "description_protocol missing rl.terminal_condition" in defects


def test_description_protocol_bundle_defects_cover_hybrid() -> None:
    bundle = DescriptionProtocolBundle(
        problem_paradigm="hybrid_ml_optimization",
        overview="Forecast demand then optimize allocation.",
        data_access=build_data_access_protocol(
            [
                FileSummary(
                    path="orders.csv",
                    role=FileRole.raw_data_table,
                    summary="orders",
                    columns=["order_id", "demand"],
                )
            ]
        ),
        hybrid={
            "prediction_subproblem": "forecast demand",
            "decision_subproblem": "allocate vehicles",
            "handoff": "predicted demand feeds the optimizer",
        },
        evaluation_summary="final plan cost",
    )
    defects = description_protocol_bundle_defects(bundle)
    assert "description_protocol missing hybrid.final_objective" in defects
    assert "description_protocol missing hybrid.validation_design" in defects


def test_evaluation_contract_requires_direction_and_guardrails() -> None:
    contract = EvaluationContractReview(
        passed=True,
        primary_metric="RMSE",
        metric_formula="RMSE = sqrt(mean((y_true - y_pred)^2))",
        prediction_unit="one validation row",
        y_true_source="validation labels",
        y_pred_source="submission prediction column",
        computation_scope="all validation rows",
        aggregation_rule="average over rows",
        validation_protocol="fixed validation split from run configuration",
        submission_checks=["columns match sample_submission.csv"],
        leakage_guards=["no validation labels as features"],
        invalid_solution_rules=["NaN predictions are invalid"],
        tie_break_rules=["lower runtime wins after equal metric"],
    )

    defects = evaluation_contract_defects(contract)
    assert "evaluation_contract metric_direction must be minimize or maximize" in defects
    with pytest.raises(RuntimeError):
        apply_evaluation_contract("## Evaluation\nold\n## Submission Format\nx\n", contract)


def test_final_evaluation_contract_is_optional_without_a_separate_holdout() -> None:
    contract = EvaluationContractReview(
        passed=True,
        primary_metric="MAE",
        metric_direction="minimize",
        metric_formula="mean(abs(y_true-y_pred))",
        prediction_unit="one validation row",
        y_true_source="development labels",
        y_pred_source="prediction column",
        computation_scope="all development rows",
        aggregation_rule="equal row weights",
        validation_protocol="Fixed development split; no independent final population is available.",
        submission_checks=["schema is valid"],
        leakage_guards=["no future labels"],
        invalid_solution_rules=["reject NaN"],
    )
    assert evaluation_contract_defects(contract) == []


def test_final_evaluation_contract_requires_all_fields_when_holdout_is_declared() -> None:
    contract = EvaluationContractReview(
        passed=True,
        primary_metric="MAE",
        metric_direction="minimize",
        metric_formula="mean(abs(y_true-y_pred))",
        prediction_unit="one validation row",
        y_true_source="development labels",
        y_pred_source="prediction column",
        computation_scope="all development rows",
        aggregation_rule="equal row weights",
        validation_protocol="Use the final holdout month once after global candidate selection.",
        submission_checks=["schema is valid"],
        leakage_guards=["no future labels"],
        invalid_solution_rules=["reject NaN"],
    )
    defects = evaluation_contract_defects(contract)
    assert all(
        f"evaluation_contract missing final evaluation field: {name}" in defects
        for name in (
            "final_evaluation_metric",
            "final_evaluation_direction",
            "final_evaluation_formula",
            "final_evaluation_protocol",
        )
    )

    contract.final_evaluation_metric = "MAE"
    contract.final_evaluation_direction = "minimize"
    contract.final_evaluation_formula = "mean(abs(y_true-y_pred))"
    contract.final_evaluation_protocol = "Evaluate the frozen winner once on the final holdout month."
    assert evaluation_contract_defects(contract) == []


def test_evaluation_contract_requires_scalar_score_for_multi_objective() -> None:
    contract = EvaluationContractReview(
        passed=True,
        primary_metric="FailedCountThenCost",
        metric_direction="minimize",
        metric_formula="First minimize failed_count, then minimize total_cost",
        prediction_unit="one dispatch plan",
        y_true_source="fixed order set and cost tables",
        y_pred_source="solution table",
        computation_scope="all orders",
        aggregation_rule="lexicographic comparison over failed_count and total_cost",
        validation_protocol="fixed replay",
        submission_checks=["solution covers all orders"],
        leakage_guards=["no future orders"],
        invalid_solution_rules=["infeasible orders count as failed"],
        tie_break_rules=["lower total_cost wins after equal failed_count"],
    )

    defects = evaluation_contract_defects(contract)
    assert any("final metric_formula" in d for d in defects)

    contract.scalar_score_formula = "score = failed_count * LARGE_PENALTY + total_cost"
    contract.metric_formula = "score = failed_count * LARGE_PENALTY + total_cost"
    defects = evaluation_contract_defects(contract)
    assert not any("final metric_formula" in d for d in defects)


def test_audit_metrics_and_display_order_do_not_become_ranking_objectives() -> None:
    contract = EvaluationContractReview(
        passed=True, executable=True, primary_metric="development_MAE", metric_formula="mean(abs(y-yhat))", metric_direction="minimize",
        prediction_unit="gantry and origin", y_true_source="observed counts", y_pred_source="prediction table",
        computation_scope="all development rows", aggregation_rule="average absolute errors",
        validation_protocol="Fit ensemble weights only in training OOF; report cost and constraint failures.",
        submission_checks=["complete keys"], leakage_guards=["no future labels"], invalid_solution_rules=["reject NaN"],
        audit_metrics=["并列报告 30 分钟及 60 分钟诊断，不用于排序", "cost", "failure_count"],
        tie_break_rules=["仅当主 MAE 完全相等时，按 candidate_id 的字典序确定显示顺序。该规则不构成第二个效果指标，也不改变 metric_formula。",
                         "60 分钟诊断、运行时长、低流量回退数量均不得成为第二排序目标。"],
    )
    assert evaluation_contract_defects(contract) == []
    contract.tie_break_rules.append("lower total_cost wins after equal MAE")
    assert any("ignores tie-break/objective term: cost" in item for item in evaluation_contract_defects(contract))


def test_evaluation_contract_rejects_scalar_score_that_ignores_tie_break_objective() -> None:
    contract = EvaluationContractReview(
        passed=True,
        primary_metric="订单分配成功率",
        metric_direction="maximize",
        metric_formula="N_success / 2104",
        scalar_score_formula="score = N_success / 2104",
        prediction_unit="one order unit",
        y_true_source="validated order set",
        y_pred_source="assignment table",
        computation_scope="all order units",
        aggregation_rule="success rate is primary; lower cost wins after equal success rate",
        validation_protocol="fixed replay over all orders",
        submission_checks=["solution schema"],
        leakage_guards=["no future data"],
        invalid_solution_rules=["infeasible assignments are invalid"],
        tie_break_rules=["lower total_transport_cost wins after equal success rate"],
    )

    defects = evaluation_contract_defects(contract)

    assert any("ignores tie-break/objective term: cost" in d for d in defects)

    contract.scalar_score_formula = "score = N_success * LARGE_REWARD - total_transport_cost"
    contract.metric_formula = "score = N_success * LARGE_REWARD - total_transport_cost"
    defects = evaluation_contract_defects(contract)
    assert not any("ignores tie-break/objective term: cost" in d for d in defects)


def test_evaluation_contract_rejects_competing_metric_and_scalar_score() -> None:
    contract = EvaluationContractReview(
        passed=True,
        primary_metric="AverageEpisodeReturn",
        metric_direction="maximize",
        metric_formula="AverageEpisodeReturn = mean(reward_day)",
        scalar_score_formula="score = - unassigned_order_count * 1e9 - total_transport_cost",
        prediction_unit="one daily solution episode",
        y_true_source="fixed replay data",
        y_pred_source="solution table",
        computation_scope="all delivery days",
        aggregation_rule="rank by the final scalar score over all delivery days",
        validation_protocol="fixed replay",
        submission_checks=["solution schema"],
        leakage_guards=["no future data"],
        invalid_solution_rules=["infeasible solutions receive penalty"],
        tie_break_rules=["cost is included in the final score"],
    )

    defects = evaluation_contract_defects(contract)

    assert any("competing scores" in d for d in defects)


def test_apply_evaluation_contract_renders_anti_gaming_sections() -> None:
    contract = EvaluationContractReview(
        passed=True,
        primary_metric="RMSE",
        metric_direction="minimize",
        metric_formula="RMSE = sqrt(mean((y_true - y_pred)^2))",
        prediction_unit="one validation row",
        y_true_source="validation labels from the fixed split",
        y_pred_source="submission prediction column",
        computation_scope="all validation rows",
        aggregation_rule="average squared error over all rows, then square root",
        validation_protocol="fixed validation split from run configuration",
        submission_checks=["columns match sample_submission.csv exactly"],
        leakage_guards=["no future rows or validation labels may be used as features"],
        invalid_solution_rules=["NaN predictions receive the configured worst score"],
        tie_break_rules=["lower runtime wins after equal metric"],
    )

    text = apply_evaluation_contract("## Evaluation\nold\n## Submission Format\nx\n", contract)
    assert "优化方向：越小越好" in text
    assert "最终评分公式：RMSE = sqrt(mean((y_true - y_pred)^2))" in text
    assert "主指标公式" not in text
    assert "单一可比较分数" not in text
    assert "### 提交校验与防作弊规则" in text
    assert "### 防泄漏要求" in text
    assert "### 非法输出处理" in text
    assert "Metric Direction" not in text


def test_evaluation_contract_accepts_missing_input_rejection_but_not_missing_rules():
    contract = EvaluationContractReview(
        passed=True, primary_metric="MAE", metric_direction="minimize",
        metric_formula="mean(abs(y_true-y_pred))", prediction_unit="row",
        y_true_source="fixed validation labels", y_pred_source="prediction column",
        computation_scope="all fixed validation rows", aggregation_rule="equal row weights",
        validation_protocol="seed=20260907; stratified 60/20/20 split",
        submission_checks=["Reject unknown IDs and any 待确认 capacity date"],
        leakage_guards=["Use train-only transforms; optional audit plots cannot change ranking"],
        invalid_solution_rules=["Reject missing dates; never treat unknown supply as unlimited"],
    )
    assert evaluation_contract_defects(contract) == []
    contract.validation_protocol = "TBD"
    assert evaluation_contract_defects(contract)


def test_software_check_scope_does_not_mix_internal_dispatch_cost_into_ranking():
    contract = EvaluationContractReview(
        passed=True, ranking_scope="software_contract_checks", primary_metric="contract_check_pass_fraction",
        metric_direction="maximize", metric_formula="sum(passed_fixed_checks)/12", prediction_unit="software case",
        y_true_source="fixed software assertions", y_pred_source="solver outputs", computation_scope="12 fixed checks",
        aggregation_rule="equal binary checks including lexicographic cost, unassigned orders and capacity",
        validation_protocol="fixed software fixtures; empirical dispatch unavailable",
        submission_checks=["missing dates must be rejected"], leakage_guards=["no historical claim from fixtures"],
        invalid_solution_rules=["exceptions fail the case"],
        tie_break_rules=["cost inside each instance is verified by the checks, not an extra candidate tie-break"],
    )
    assert evaluation_contract_defects(contract) == []
    contract.ranking_scope = "task_effectiveness"
    assert any("cost" in issue for issue in evaluation_contract_defects(contract))
    contract.validation_protocol = "split: unknown"
    assert evaluation_contract_defects(contract)


def test_automl_context_pack_renders_data_access_and_scalar_score() -> None:
    files = [
        FileSummary(
            path=f"cost/carrier{i:02d}_cost.xlsx",
            role=FileRole.raw_data_table,
            summary="carrier cost table",
            columns=["lane", "vehicle_type", "price"],
        )
        for i in range(1, 4)
    ]
    protocol = build_data_access_protocol(files)
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="Optimize orders.",
        task_goal="Minimize dispatch cost.",
        data_access=protocol,
        optimization={
            "input_instance": "orders and carrier costs",
            "objective": "minimize failed orders then cost",
            "decision_variables": ["order grouping", "vehicle assignment"],
            "hard_constraints": ["capacity"],
            "feasibility_checks": ["coverage"],
            "solution_representation": "solution table",
        },
        output={
            "output_kind": "solution_table",
            "output_filename": "solution.csv",
            "sample_submission_required": False,
            "no_sample_submission_reason": "authoritative task uses solution protocol",
        },
    )
    contract = EvaluationContractReview(
        passed=True,
        primary_metric="FailedCountThenCost",
        metric_direction="minimize",
        metric_formula="score = failed_count * LARGE_PENALTY + total_cost",
        scalar_score_formula="score = failed_count * LARGE_PENALTY + total_cost",
        prediction_unit="one solution",
        y_true_source="orders and constraints",
        y_pred_source="solution table",
        computation_scope="all orders",
        aggregation_rule="one scalar score",
        validation_protocol="fixed replay",
        submission_checks=["solution schema"],
        leakage_guards=["no future data"],
        invalid_solution_rules=["capacity violation is failed"],
        tie_break_rules=["same scalar score ties"],
    )

    pack = build_automl_context_pack(bundle, files, evaluation_contract=contract)
    text = render_automl_context_markdown(pack)

    assert pack.output_contract["sample_submission_required"] is False
    assert "score = failed_count * LARGE_PENALTY + total_cost" in text
    assert "cost/carrier{id}_cost.xlsx" in text
    assert "pd.read_excel(path)" in text


def test_automl_access_uses_exact_csv_schema_and_preserves_provisional_population() -> None:
    columns = [f"feature_{i}" for i in range(185)] + ["target", "excluded_id"]
    fs = FileSummary(
        path="measurements.csv", role=FileRole.raw_data_table, columns=columns, summary="Measurements",
        source_metadata={"shape": [10000, len(columns)], "profile_sampling": {"sampled": True}},
        column_profiles=[{"name": columns[0], "row_count": 500, "null_count": 0}],
    )
    pack = build_automl_context_pack(
        DescriptionProtocolBundle(data_access=build_data_access_protocol([fs])), [fs],
        compiled_context={"table_cards": [{
            "source_file": fs.path, "table_id": fs.path, "table_kind": "csv_table",
            "fields": [{"name": columns[0]}],
        }]},
    )
    assert pack.data_access[0]["columns"] == columns
    assert pack.data_access[0]["columns_scope"] == "physical_schema"
    table = build_data_schema_contract([fs])["tables"][0]
    assert table["row_count_basis"] == "reader_shape_unverified"
    assert table["reader_reported_shape"] == [10000, len(columns)]
    assert table.get("verified_row_count") is None
    assert "worksheet_used_range_shape" not in table


def test_automl_access_does_not_replace_sheet_schema_with_workbook_or_field_preview() -> None:
    fs = FileSummary(
        path="inputs.xlsx", role=FileRole.raw_data_table, columns=["workbook_field"], summary="Orders",
        source_metadata={"excel_sheet_profiles": [{
            "sheet_name": "orders", "columns": ["order", "amount", "owner"], "shape": [20, 3],
        }]},
    )
    pack = build_automl_context_pack(
        DescriptionProtocolBundle(data_access=build_data_access_protocol([fs])), [fs],
        compiled_context={"table_cards": [
            {"source_file": fs.path, "sheet_name": "orders", "fields": [{"name": "order"}]},
            {"source_file": "unprofiled.csv", "field_hints": ["unknown"]},
        ]},
    )
    assert pack.data_access[0]["columns"] == ["order", "amount", "owner"]
    assert pack.data_access[1]["columns_scope"] == "preview_only"


def test_automl_context_renders_entity_alias_candidates_as_unconfirmed() -> None:
    files = [
        FileSummary(
            path="contracts.xlsx",
            role=FileRole.raw_data_table,
            summary="成本合同表。",
            columns=[],
            source_metadata={
                "excel_sheet_profiles": [
                    {
                        "sheet_name": "承运商成本合同表",
                        "shape": [6, 5],
                        "columns": ["结算方代码", "结算方名称", "起点", "终点", "车型"],
                    }
                ]
            },
        ),
        FileSummary(
            path="vehicles.xlsx",
            role=FileRole.raw_data_table,
            summary="每日车辆表。",
            columns=["承运商代码", "承运商名称", "车型"],
            source_metadata={"shape": [4, 3]},
        ),
    ]
    bundle = DescriptionProtocolBundle(
        problem_paradigm="static_optimization",
        overview="验证合同和车辆可行性。",
        task_goal="在合同线路覆盖下分配车辆。",
        data_access=build_data_access_protocol(files),
    )

    pack = build_automl_context_pack(
        bundle,
        files,
        downstream_context={
            "constraint_memory": {
                "entity_alias_candidates": [
                    {
                        "concept_id": "carrier_settlement_entity",
                        "label": "承运商/结算方实体候选字段",
                        "reason": "字段语义可能引用同一参与方，但需要值域核验。",
                        "confidence": "medium",
                        "candidate_fields": [
                            {
                                "source_file": "contracts.xlsx",
                                "sheet_name": "承运商成本合同表",
                                "field": "结算方代码",
                                "semantic_role": "party_identifier",
                                "value_kind": "code",
                            },
                            {
                                "source_file": "vehicles.xlsx",
                                "field": "承运商代码",
                                "semantic_role": "party_identifier",
                                "value_kind": "code",
                            },
                        ],
                    }
                ]
            }
        },
    )
    text = render_automl_context_markdown(pack)

    assert "## Entity Alias Candidates" in text
    assert "not confirmed equivalent keys" in text
    assert "candidate_not_equivalent" in text
    assert "结算方代码" in text
    assert "承运商代码" in text
    assert "recommended_qdi_checks" in text


def test_apply_evaluation_contract_allows_evidence_gap_with_explicit_fixes() -> None:
    contract = EvaluationContractReview(
        passed=False,
        primary_metric="TotalCost",
        metric_direction="minimize",
        metric_formula="TotalCost = shipping_cost + violation_penalty",
        prediction_unit="one order assignment row",
        y_true_source="computed feasibility and cost from provided business tables",
        y_pred_source="assignment columns in submission.csv",
        computation_scope="all orders in the evaluation set",
        aggregation_rule="sum all costs and penalties into one total score",
        validation_protocol="one-shot evaluation over the fixed order set using the provided cost tables",
        submission_checks=["columns match sample_submission.csv exactly"],
        leakage_guards=["do not use external route data unless provided by the task package"],
        invalid_solution_rules=["unassigned or infeasible orders receive the configured penalty"],
        tie_break_rules=["fewer constraint violations wins after equal total cost"],
        issues=["cost table mapping is not explicitly defined"],
        fixes=["provide a deterministic cost-table lookup rule before final scoring"],
    )

    text = apply_evaluation_contract("## Evaluation\nold\n## Submission Format\nx\n", contract)
    assert "### 评估前置条件" in text
    assert "cost table mapping is not explicitly defined" in text
    assert "正式评分前需明确" in text
    assert "blocked_by_evidence_gap" not in text
    assert "Unresolved Evaluation Gaps" not in text


def test_finalize_description_removes_internal_process_sections() -> None:
    text = (
        "# 赛题说明\n\n"
        "## Evaluation\n### Metric Direction\nminimize\n"
        "## Output Layout\n### Directory Tree\nx\n"
        "## Submission Format\nx\n"
        "### Contract Status\npassed=false\n"
    )

    cleaned = finalize_description_markdown(text)

    assert "## 评估协议" in cleaned
    assert "## 输出或提交格式" in cleaned
    assert "Output Layout" not in cleaned
    assert "Contract Status" not in cleaned


def test_finalize_description_drops_duplicate_empty_section_and_long_duplicate_fact() -> None:
    fact = "- " + ("这是一条重复的完整评估约束，所有候选必须使用相同固定人口与指标。" * 5)
    text = (
        "# 赛题说明\n\n"
        "## 任务定义\n- 明确预测目标和时间边界。\n"
        f"{fact}\n\n"
        "## 任务定义\n- 当前材料不足，无法生成该章节。\n\n"
        "## 评估协议\n"
        f"{fact}\n"
    )

    cleaned = finalize_description_markdown(text)

    assert cleaned.count("## 任务定义") == 1
    assert "当前材料不足，无法生成该章节" not in cleaned
    assert cleaned.count(fact) == 1
