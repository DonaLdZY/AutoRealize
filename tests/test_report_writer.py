import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.models import DescriptionProtocolBundle, EvaluationContractReview, FileRole, FileSummary
from autorealize.prompts.manager import PromptManager
from autorealize.report_writer import (
    apply_evaluation_contract,
    build_automl_context_pack,
    build_data_access_protocol,
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
