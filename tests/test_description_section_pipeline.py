from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from autorealize.config import AutoRealizeConfig
from autorealize.models import (
    AutoMLContextPack,
    DescriptionProtocolBundle,
    DescriptionTaskProtocolDraft,
    EvaluationContractReview,
    FileRole,
    FileSummary,
    ProblemParadigmReview,
    SampleSubmissionSpec,
)
from autorealize.modules.task_definition import TaskDefinitionModule
from autorealize.report_writer import render_automl_context_markdown


def _module(tmp_path: Path) -> TaskDefinitionModule:
    cfg = AutoRealizeConfig()
    services = SimpleNamespace(llm_client=None, prompt_mgr=None, registry=None, trajectory=None, knowledge_store=None)
    report_dir = tmp_path / "realize_report"
    report_dir.mkdir()
    return TaskDefinitionModule(cfg, services, tmp_path, report_dir)


class _PromptMgr:
    def load(self, _name: str) -> str:
        return "system prompt"


class _CaptureLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ask_structured(self, *, model_cls, **kwargs):
        self.calls.append(kwargs)
        return model_cls()


def _module_with_capture_llm(tmp_path: Path) -> tuple[TaskDefinitionModule, _CaptureLLM]:
    cfg = AutoRealizeConfig()
    llm = _CaptureLLM()
    services = SimpleNamespace(llm_client=llm, prompt_mgr=_PromptMgr(), registry=None, trajectory=None, knowledge_store=None)
    report_dir = tmp_path / "realize_report"
    report_dir.mkdir()
    return TaskDefinitionModule(cfg, services, tmp_path, report_dir), llm


def test_low_cost_classifier_does_not_treat_split_rule_as_metric(tmp_path: Path) -> None:
    mod, llm = _module_with_capture_llm(tmp_path)
    result = mod._task_classification_from_problem_review(
        ProblemParadigmReview(problem_paradigm="ml_dl_prediction", confidence=0.99),
        {"authoritative_memory": {"evaluation_requirements": [
            "Use chronological 60/20/20 splits.",
            "Primary metric is MAE in kWh.",
        ]}},
    )

    assert result.task_type == "prediction"
    assert result.primary_metric == ""
    assert result.metric_formula == ""
    assert llm.calls == []


def test_task_overview_authority_pack_omits_detailed_evidence_ledgers(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    auth = {
        "summary": "Forecast the next hour.",
        "task_goal": "Predict the process quality result one hour ahead.",
        "input_requirements": ["Use only observations available at the cutoff."],
        "output_requirements": ["Return the next-hour prediction and limits."],
        "evaluation_requirements": ["Use a fixed chronological split."],
        "constraints": ["Do not leak post-cutoff measurements."],
        "leakage_guards": ["Features must be available by the prediction cutoff."],
        "submission_contract": {"format": "json"},
        "authority_conflicts": [{"topic": "quality threshold", "resolution": "do not assume one"}],
        "unresolved_questions": ["Whether the threshold is operationally approved."],
        "evidence_items": [{"source": f"doc-{idx}", "text": "large duplicate evidence"} for idx in range(50)],
        "requirement_fact_matrix": [{"requirement": f"duplicate-{idx}"} for idx in range(100)],
        "context_routing_notes": ["Internal prompt-routing detail."],
    }
    pack = mod._build_task_authority_pack(
        task_hint="Predict the next-hour result.",
        problem_review=ProblemParadigmReview(
            problem_paradigm="ml_dl_prediction",
            problem_structure="time_series_forecast",
            confidence=0.95,
            reasoning="Long internal explanation that is not needed for the reader-facing overview.",
            evidence=["large internal evidence detail"],
            recommended_solver_families=["tree models"],
        ),
        downstream_context={
            "authoritative_memory": auth,
            "train_table": "inferred-train-table",
            "predict_table": "inferred-predict-table",
            "target_column": "inferred-target-column",
            "id_column": "inferred-id-column",
        },
    )

    rendered = str(pack)
    assert pack["authoritative_memory"]["constraints"] == auth["constraints"]
    assert pack["authoritative_memory"]["submission_contract"] == auth["submission_contract"]
    assert "evidence_items" not in pack["authoritative_memory"]
    assert "requirement_fact_matrix" not in pack["authoritative_memory"]
    assert "context_routing_notes" not in pack["authoritative_memory"]
    assert "reasoning" not in pack["problem_paradigm_review"]
    assert "recommended_solver_families" not in pack["problem_paradigm_review"]
    assert "downstream_context" not in pack
    assert "inferred-target-column" not in rendered


def test_evaluation_evidence_pack_keeps_contract_facts_and_filters_duplicate_context(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    mod._question_memory_pack = lambda _context: {
        "summary": "large QDI digest that should not be copied wholesale",
        "answers": [
            {"question": "Target timestamp", "answer": "Use lab sample time.", "script": "x" * 5000},
            {"question": "UI color", "answer": "Blue."},
        ],
        "question_records": [
            {
                "question_id": "q-time",
                "question": "How is the target sampling time interpreted?",
                "status": "answered",
                "short_answer": "Use the lab timestamp as target time; forecast cutoff is one hour earlier.",
            },
            {
                "question_id": "q-ui",
                "question": "Should the settings panel use blue?",
                "status": "answered",
                "short_answer": "Use the neutral palette.",
            },
        ],
        "unresolved_questions": [
            "Lab result publication delay is unknown.",
            "The future navigation color is undecided.",
        ],
        "context_routing_notes": ["internal route data"],
    }
    source = FileSummary(
        path="浓度.xls",
        role=FileRole.raw_data_table,
        summary="Laboratory results.",
        columns=["采样时间", "参数", "部位", "结果"],
        source_metadata={"verified_row_count": 8036, "row_count_basis": "full_sheet_parse"},
    )
    problem_review = ProblemParadigmReview(
        problem_paradigm="ml_dl_prediction",
        problem_structure="time_series_forecast",
        decision_temporality="one hour ahead",
        confidence=0.94,
        reasoning="x" * 10000,
        evidence=["duplicate and verbose evidence"] * 20,
        allowed_method_families=["tree", "linear", "many other methods"],
    )
    pack = mod._build_evaluation_evidence_pack(
        downstream_context={
            "constraint_memory": {"items": [{"description": "irrelevant " + "x" * 10000}]},
            "authoritative_submission_contract": {
                "is_defined": True,
                "columns": ["prediction_cutoff", "predicted_al_n"],
                "row_unit": "one prediction per sample timestamp",
            },
        },
        problem_review=problem_review,
        relations=[{"left_file": "unrelated-a", "right_file": "unrelated-b"}] * 100,
        file_summaries=[source],
        protocol_bundle=DescriptionProtocolBundle(evaluation_summary="Development MAE; final holdout once."),
    )

    assert set(pack) == {
        "problem_summary",
        "authoritative_output_contract",
        "source_coverage",
        "relevant_prior_qdi",
        "protocol_evaluation_summary",
    }
    assert pack["problem_summary"]["problem_paradigm"] == "ml_dl_prediction"
    assert "reasoning" not in pack["problem_summary"]
    assert pack["authoritative_output_contract"]["columns"] == [
        "prediction_cutoff",
        "predicted_al_n",
    ]
    assert pack["source_coverage"]["entries"][0]["source_file"] == "浓度.xls"
    assert "采样时间" in pack["source_coverage"]["entries"][0]["schema_signature_fields"]
    assert [item["question_id"] for item in pack["relevant_prior_qdi"]["relevant_prior_conclusions"]] == [
        "q-time"
    ]
    assert pack["relevant_prior_qdi"]["unresolved_questions"] == [
        "Lab result publication delay is unknown."
    ]
    assert "x" * 1000 not in str(pack)


def test_description_file_pack_excludes_raw_preview_and_keeps_sheet_inventory(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    fs = FileSummary(
        path="cost/carrier_cost.xlsx",
        role=FileRole.raw_data_table,
        summary="Carrier cost rules and contract notes.",
        columns=["order_id", "cost"],
        column_semantics={"order_id": "Order join key", "cost": "Transport cost field"},
        column_profiles=[
            {"name": "order_id", "logical_type": "categorical", "unique_count": 10, "null_ratio": 0.0},
            {"name": "cost", "logical_type": "numeric", "unique_count": 8, "null_ratio": 0.0},
        ],
        source_metadata={
            "shape": [100, 2],
            "preview": [{"order_id": "A001", "cost": 12.3}],
            "probe_results": {"very": "large"},
            "excel_sheet_profiles": [
                {
                    "sheet_name": "contract_notes",
                    "shape": [20, 3],
                    "columns": ["contract", "billing_rule", "note"],
                    "raw_preview": [["contract_a", "tiered"]],
                    "preview": [{"contract": "contract_a"}],
                    "column_profiles": [{"name": "contract", "logical_type": "text", "unique_count": 2}],
                    "profile_policy": "deep_profile_all_small_workbook",
                }
            ],
            "sheet_field_descriptions": {"contract_notes": {"billing_rule": "Contract billing rule text"}},
        },
    )

    packed = mod._compact_file_for_sections(fs, include_profiles=True)

    assert packed["shape"] == [100, 2]
    assert packed["sheets"][0]["sheet_name"] == "contract_notes"
    assert packed["sheets"][0]["field_semantics"]["billing_rule"] == "Contract billing rule text"
    assert "preview" not in str(packed)
    assert "probe_results" not in str(packed)


def test_compose_description_sections_uses_required_order(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    desc = mod._compose_description_sections(
        [
            "## 任务概述\n- A",
            "## 任务定义\n- B",
            "## 评估协议\n- C",
            "## 输出或提交格式\n- D",
            "## 数据说明\n- E",
            "## 关键字段说明\n- F",
            "## 约束与防泄漏\n- G",
            "## 关键坑点与待确认事项\n- H",
        ]
    )

    headers = [line for line in desc.splitlines() if line.startswith("## ")]
    assert headers == [
        "## 任务概述",
        "## 任务定义",
        "## 评估协议",
        "## 输出或提交格式",
        "## 数据说明",
        "## 关键字段说明",
        "## 约束与防泄漏",
        "## 关键坑点与待确认事项",
    ]


def test_static_optimization_can_require_rl_without_changing_problem_structure(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    review = ProblemParadigmReview(
        problem_paradigm="static_optimization",
        problem_structure="decision_optimization",
        explicit_rl_requested=True,
        rl_required=True,
        reasoning="用户要求 PPO/DQN，状态、动作、奖励；任务输出 assignment 明细表，评分 total_penalized_cost。",
        evidence=["必须使用强化学习建模", "输出订单车辆分配方案", "最小化未分配订单和运输成本"],
        key_signals=["强化学习", "车辆调度", "assignment 明细"],
    )

    mod._normalize_method_only_rl_review(
        review,
        original_text="城市配送订单车辆调度分配，输出完整方案，未分配惩罚，总成本最小。PPO或DQN都实现。",
        downstream_context={},
    )

    assert review.problem_paradigm == "static_optimization"
    assert review.problem_structure == "decision_optimization"
    assert review.explicit_rl_requested is True
    assert review.rl_required is True
    assert review.rl_as_required_paradigm is False
    assert "reinforcement_learning" in review.required_method_families
    assert review.environment_source == "constructed_from_static_data"
    assert not any("baseline" in note.lower() for note in review.method_routing_notes)


def test_prediction_with_noncausal_reference_is_not_routed_as_optimization(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    review = ProblemParadigmReview(
        problem_paradigm="hybrid_ml_optimization",
        problem_structure="hybrid_prediction_optimization",
        reasoning=(
            "主链路是监督回归预测；附带蒸汽流量参考，因此范式映射为 hybrid_ml_optimization。"
            "无明确决策变量-约束-目标函数，不输出阀开度指令，不宣称最优。"
        ),
        evidence=["一小时提前预测", "蒸汽建议为历史近邻参考"],
        key_signals=[
            "无明确决策变量-约束-目标函数",
            "不构成静态优化求解问题",
        ],
        required_method_families=[
            "supervised_tabular_regression",
            "time_series_feature_engineering",
        ],
        allowed_method_families=[
            "time_series_supervised_regression",
            "reinforcement_learning",
            "constructed_policy_optimization_from_static_data",
        ],
        recommended_solver_families=[
            "time_series_supervised_regression",
            "reinforcement_learning",
            "constructed_policy_optimization_from_static_data",
        ],
        method_routing_notes=[
            "RL 路由：用户未要求 RL，但可以构造式策略，不改变 hybrid_prediction_optimization。",
        ],
        rl_formulation_candidates=[
            {"formulation_type": "constructed_policy_optimization_from_static_data"},
        ],
    )

    mod._normalize_prediction_with_reference_review(
        review,
        task_hint="预测下一小时 Al(N)，蒸汽建议仅为非因果参考。",
    )

    assert review.problem_paradigm == "ml_dl_prediction"
    assert review.problem_structure == "prediction"
    assert review.rl_required is False
    assert "hybrid_ml_optimization" not in review.reasoning
    assert not any("hybrid_prediction_optimization" in item for item in review.method_routing_notes)
    assert review.rl_formulation_candidates == []
    assert review.environment_source == "not_applicable"
    assert "reinforcement_learning" not in review.allowed_method_families
    assert not any("policy_optimization" in item for item in review.recommended_solver_families)
    assert any("下游按监督预测路由" in note for note in review.method_routing_notes)


def test_prediction_with_reference_is_normalized_without_negative_classifier_claims(tmp_path: Path) -> None:
    review = ProblemParadigmReview(
        problem_paradigm="hybrid_ml_optimization",
        problem_structure="hybrid_prediction_optimization",
        reasoning="预测下一小时压滤液 Al(N)，并根据历史近邻生成蒸汽流量参考建议。",
        evidence=["监督回归预测下一小时 Al(N)", "蒸汽流量建议是历史工况参考"],
        required_method_families=["tabular_supervised_regression"],
    )

    TaskDefinitionModule._normalize_prediction_with_reference_review(
        review,
        task_hint="预测下一小时 Al(N)，附带有数据支持的蒸汽参考建议。",
    )

    assert review.problem_paradigm == "ml_dl_prediction"
    assert review.problem_structure == "prediction"


def test_explicit_prediction_optimization_contract_remains_hybrid(tmp_path: Path) -> None:
    review = ProblemParadigmReview(
        problem_paradigm="hybrid_ml_optimization",
        problem_structure="hybrid_prediction_optimization",
        reasoning="先预测蒸汽负荷，再优化决策变量 steam_flow，以最小化目标函数 energy_cost。",
        evidence=["决策变量 steam_flow", "最小化目标函数 energy_cost"],
        required_method_families=["tabular_supervised_regression", "constrained_optimization"],
    )

    TaskDefinitionModule._normalize_prediction_with_reference_review(
        review,
        task_hint="预测蒸汽负荷；决策变量 steam_flow，最小化 energy_cost。",
    )

    assert review.problem_paradigm == "hybrid_ml_optimization"
    assert review.problem_structure == "hybrid_prediction_optimization"


def test_prediction_reference_protocol_forbids_handoff_leakage_and_unit_guessing(tmp_path: Path) -> None:
    bundle = DescriptionProtocolBundle(
        problem_paradigm="ml_dl_prediction",
        hybrid={"handoff": "predicted_al_n enters the steam advice as quality evidence"},
        ml_dl={
            "leakage_guards": ["Only use observations visible by cutoff."],
        },
        output={"format_rules": ["Return steam advice when supported."]},
    )

    changed = TaskDefinitionModule._normalize_prediction_reference_protocol_bundle(
        bundle,
        task_hint="预测下一小时 Al(N)，附带蒸汽流量参考建议。",
    )

    assert changed
    assert not bundle.hybrid.handoff
    assert any("不得读取预测目标" in item for item in bundle.ml_dl.leakage_guards)
    assert any("不得与 kg/h" in item for item in bundle.output.format_rules)


def test_description_protocol_models_accept_null_inactive_branches() -> None:
    payload = {
        "problem_paradigm": "static_optimization",
        "ml_dl": None,
        "optimization": {"objective": "minimize total cost"},
        "rl": None,
        "hybrid": None,
    }

    draft = DescriptionTaskProtocolDraft.model_validate(payload)
    bundle = DescriptionProtocolBundle.model_validate(payload)

    for protocol in (draft, bundle):
        assert protocol.ml_dl is not None
        assert protocol.optimization.objective == "minimize total cost"
        assert protocol.rl is not None
        assert protocol.hybrid is not None


def test_description_protocol_prompts_forbid_null_protocol_branches() -> None:
    prompt_dir = Path(__file__).parents[1] / "autorealize" / "prompts" / "system"
    prompt_names = [
        "ml_dl_description_protocol.md",
        "optimization_description_protocol.md",
        "rl_description_protocol.md",
        "hybrid_description_protocol.md",
    ]

    for prompt_name in prompt_names:
        text = (prompt_dir / prompt_name).read_text(encoding="utf-8")
        assert "绝不能输出 `null`" in text

def test_artifact_sanity_checks_sample_columns(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    (tmp_path / "sample_submission.csv").write_text("id,pred\n1,0\n", encoding="utf-8")
    data_root = tmp_path / "input"
    data_root.mkdir()
    spec = SampleSubmissionSpec(columns=["id", "target"])
    contract = EvaluationContractReview(primary_metric="score", metric_direction="minimize")
    legacy = SimpleNamespace(_find_missing_file_references=lambda desc, root: [])

    defects = mod._artifact_sanity_check(
        desc="# 璧涢璇存槑\n\n## 浠诲姟姒傝堪\n- test",
        data_root=data_root,
        sample_spec=spec,
        evaluation_contract=contract,
        legacy=legacy,
    )

    assert any("sample_columns_mismatch" in item for item in defects)


def test_sample_spec_source_fields_are_corrected_to_exact_schema(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    spec = SampleSubmissionSpec(
        columns=["总箱数"],
        source_fields={"总箱数": "派生自 `订单明细信息` 中的 `标准箱数` 字段，按 `订单号` 聚合"},
    )
    schema_contract = {
        "tables": [
            {
                "table_id": "orders.xlsx::订单明细信息",
                "physical_columns_exact": ["订单号", "标箱数量", "折算箱数"],
            }
        ]
    }

    corrections = mod._correct_sample_spec_source_fields(spec, schema_contract=schema_contract)

    assert corrections == [
        {
            "output_column": "总箱数",
            "from": "标准箱数",
            "to": "标箱数量",
            "reason": "source_fields alias was not an exact physical column; corrected using exact schema contract",
        }
    ]
    assert "`标箱数量`" in spec.source_fields["总箱数"]
    assert "`标准箱数`" not in spec.source_fields["总箱数"]


def test_unresolved_source_field_alias_is_recorded_not_forced(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    spec = SampleSubmissionSpec(
        columns=["交付日"],
        source_fields={"交付日": "从订单数据中的“交付日”字段获取"},
    )
    schema_contract = {
        "tables": [
            {
                "table_id": "orders.xlsx::订单表信息",
                "sheet_name": "订单表信息",
                "physical_columns_exact": ["订单号", "要求交付时间", "最早交货时间", "最晚交货时间"],
            }
        ]
    }

    corrections = mod._correct_sample_spec_source_fields(spec, schema_contract=schema_contract)

    assert corrections == []
    assert "“交付日”" in spec.source_fields["交付日"]
    assert any("could not resolve source field alias `交付日`" in rule for rule in spec.validation_rules)


def test_automl_context_renders_source_field_corrections() -> None:
    pack = AutoMLContextPack(
        output_contract={
            "output_kind": "solution_table",
            "columns": ["总箱数"],
            "sample_submission_spec": {
                "source_fields": {"交付日": "从订单数据中的“交付日”字段获取"},
                "validation_rules": [
                    "AutoRealize could not resolve source field alias `交付日` for output column `交付日` to an exact physical column; do not access it as a raw dataframe column."
                ],
            },
            "sample_submission_source_field_corrections": [
                {
                    "output_column": "总箱数",
                    "from": "标准箱数",
                    "to": "标箱数量",
                    "reason": "source_fields alias was not an exact physical column; corrected using exact schema contract",
                }
            ],
        }
    )

    rendered = render_automl_context_markdown(pack)

    assert "source_field_corrections" in rendered
    assert "from_alias=标准箱数" in rendered
    assert "to_physical_column=标箱数量" in rendered
    assert "sample_submission_source_fields" in rendered
    assert "could not resolve source field alias `交付日`" in rendered


def test_automl_context_renders_output_schema_rules() -> None:
    pack = AutoMLContextPack(
        output_contract={
            "output_kind": "solution_table",
            "columns": ["方案名", "交付日", "订单集编号"],
        }
    )

    rendered = render_automl_context_markdown(pack)

    assert "output_schema_rules" in rendered
    assert "pd.DataFrame(rows, columns=OUTPUT_COLUMNS)" in rendered
    assert "not raw input dataframe columns" in rendered


def test_output_markdown_receives_source_field_correction_notes(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    spec = SampleSubmissionSpec(
        columns=["总箱数"],
        source_fields={"总箱数": "由订单明细中 `标准箱数` 聚合得到"},
    )
    spec.validation_rules = [
        "AutoRealize corrected source field alias `标准箱数` -> `标箱数量` for output column `总箱数`."
    ]

    rendered = mod._apply_source_field_corrections_to_output_markdown(
        "## 输出或提交格式\n\n| `总箱数` | 来自 `标准箱数` |",
        spec,
        [{"output_column": "总箱数", "from": "标准箱数", "to": "标箱数量"}],
    )

    assert "`标箱数量`" in rendered
    assert "`标准箱数`" not in rendered.split("### 源字段精确性修正", 1)[0]
    assert "输出列是生成结果 schema" in rendered


def test_source_field_alias_correction_handles_path_style_source_fields(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    spec = SampleSubmissionSpec(
        columns=["交付日", "总箱数"],
        source_fields={
            "交付日": "15天订单数据1027-1110.xlsx::订单表信息::交付时间（取日期部分）",
            "总箱数": "15天订单数据1027-1110.xlsx::订单明细信息::标准箱数",
        },
    )
    schema_contract = {
        "tables": [
            {
                "table_id": "15天订单数据1027-1110.xlsx::订单表信息",
                "source_file": "15天订单数据1027-1110.xlsx",
                "sheet_name": "订单表信息",
                "physical_columns_exact": ["订单号", "要求交付时间", "最早交货时间", "最晚交货时间"],
            },
            {
                "table_id": "15天订单数据1027-1110.xlsx::订单明细信息",
                "source_file": "15天订单数据1027-1110.xlsx",
                "sheet_name": "订单明细信息",
                "physical_columns_exact": ["订单号", "标箱数量", "折算箱数"],
                "field_summaries": [{"name": "标箱数量", "meaning": "标准箱数量。"}],
            },
        ]
    }

    corrections = mod._correct_sample_spec_source_fields(spec, schema_contract=schema_contract)

    assert {item["from"]: item["to"] for item in corrections} == {
        "交付时间": "要求交付时间",
        "标准箱数": "标箱数量",
    }
    assert "::要求交付时间" in spec.source_fields["交付日"]
    assert "::标箱数量" in spec.source_fields["总箱数"]
    assert any("`标准箱数` -> `标箱数量`" in rule for rule in spec.validation_rules)


def test_source_alias_guard_marks_corrected_and_unresolved_aliases(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    spec = SampleSubmissionSpec(
        columns=["交付日", "总箱数"],
        source_fields={
            "交付日": "从订单表信息.交付日 获取",
            "总箱数": "订单明细信息.标准箱数 汇总",
        },
    )
    schema_contract = {
        "tables": [
            {
                "table_id": "15天订单数据1027-1110.xlsx::订单表信息",
                "source_file": "15天订单数据1027-1110.xlsx",
                "sheet_name": "订单表信息",
                "physical_columns_exact": ["订单号", "要求交付时间", "最早交货时间", "最晚交货时间"],
            },
            {
                "table_id": "15天订单数据1027-1110.xlsx::订单明细信息",
                "source_file": "15天订单数据1027-1110.xlsx",
                "sheet_name": "订单明细信息",
                "physical_columns_exact": ["订单号", "标箱数量", "折算箱数"],
                "field_summaries": [{"name": "标箱数量", "meaning": "标准箱数量。"}],
            },
        ]
    }
    downstream_context = {
        "constraint_memory": {
            "items": [
                {
                    "summary": "不可合并订单需要单独处理。",
                    "related_fields": ["是否可合并"],
                }
            ]
        }
    }

    guard = mod._collect_source_alias_guard(
        sample_spec=spec,
        schema_contract=schema_contract,
        downstream_context=downstream_context,
    )
    by_alias = {item["alias"]: item for item in guard}

    assert by_alias["标准箱数"]["exact_physical_column"] == "标箱数量"
    assert by_alias["交付日"]["status"] == "unresolved_business_concept"
    assert "要求交付时间" in by_alias["交付日"]["candidate_exact_columns"]
    assert by_alias["是否可合并"]["status"] == "unresolved_business_concept"
    assert "Do not access this alias as a raw column" in by_alias["是否可合并"]["rule"]
    assert "从订单表信息" not in by_alias

    rendered = render_automl_context_markdown(AutoMLContextPack(source_alias_guard=guard))

    assert "## Source Alias Guard" in rendered
    assert "alias=标准箱数" in rendered
    assert "exact_physical_column=标箱数量" in rendered
    assert "alias=交付日" in rendered
    assert "alias=是否可合并" in rendered
    assert "Never use an alias as `df[alias]`" in rendered


def test_problem_paradigm_prompt_keeps_full_original_but_artifacts_data_digest(tmp_path: Path) -> None:
    mod, llm = _module_with_capture_llm(tmp_path)
    tail_marker = "ORIGINAL_REQUIREMENTS_TAIL_MARKER"
    data_marker = "FULL_DATA_DESCRIPTION_MARKER_SHOULD_NOT_BE_IN_PROMPT"
    original = "A" * 13000 + tail_marker
    data_description = "data description before marker\n" + data_marker
    fs = FileSummary(
        path="orders.csv",
        role=FileRole.raw_data_table,
        summary="Order table for delivery optimization.",
        columns=["order_id", "cost"],
        column_semantics={"order_id": "order key", "cost": "transport cost"},
        column_profiles=[
            {"name": "order_id", "logical_type": "categorical", "unique_count": 10, "null_ratio": 0.0},
            {"name": "cost", "logical_type": "numeric", "unique_count": 8, "null_ratio": 0.0},
        ],
        source_metadata={"shape": [100, 2], "preview": [{"order_id": "A001", "cost": 12.3}]},
    )

    mod._classify_problem_paradigm(
        task_hint="minimize transport cost",
        original_text=original,
        data_digest=data_description[:12000],
        data_description_text=data_description,
        downstream_context={
            "task_hint": "minimize transport cost",
            "authoritative_memory": {"task_goal": "minimize transport cost"},
            "constraint_memory": {"summary": "capacity constraints"},
        },
        file_summaries=[fs],
        relations=[],
    )

    assert llm.calls
    stable = llm.calls[0]["static_context_prompt"]
    assert "original_requirements_full" in stable
    assert tail_marker in stable
    assert "headroom_telemetry" not in stable
    assert "artifact_refs" not in stable
    assert "context_shape" not in stable
    assert "data_cognition_digest" not in stable
    assert data_marker not in stable
    assert "source_metadata" not in stable
    assert '"preview":' not in stable
    assert "visible_excerpt" not in stable

    artifact_files = list((tmp_path / "realize_report" / "context_artifacts").glob("*.json"))
    assert artifact_files
    artifact_text = "\n".join(p.read_text(encoding="utf-8") for p in artifact_files)
    assert data_marker in artifact_text
    telemetry_path = tmp_path / "realize_report" / "context_pack_telemetry.jsonl"
    assert telemetry_path.exists()
    telemetry_text = telemetry_path.read_text(encoding="utf-8")
    assert "problem_paradigm_classifier" in telemetry_text
    assert "problem_paradigm_pack" in telemetry_text


def test_generic_section_prompt_keeps_artifact_refs_local(tmp_path: Path) -> None:
    mod, llm = _module_with_capture_llm(tmp_path)
    mod._generate_generic_section(
        section_id="data",
        section_title="鏁版嵁璇存槑",
        evidence_pack={
            "table_index": [{"table_id": "orders.csv", "shape": [10, 2]}],
            "filename_sample_groups": [{"template_path": "orders_*.csv", "count": 12}],
        },
        frozen_sections={"浠诲姟姒傝堪": "## 浠诲姟姒傝堪\n- frozen"},
        original_text="authoritative requirements",
        artifact_refs={"data_description": {"artifact_id": "local-only"}},
    )

    stable = llm.calls[-1]["static_context_prompt"]
    assert "original_requirements_full" in stable
    assert "authoritative requirements" in stable
    assert "artifact_refs" not in stable
    assert "context_shape" not in stable
    assert "data_cognition_digest" not in stable
    assert "visible_excerpt" not in stable


def test_field_section_pack_includes_bounded_field_meanings(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    fs = FileSummary(
        path="orders.xlsx",
        role=FileRole.raw_data_table,
        summary="Order workbook used for delivery optimization.",
        columns=["order_id", "delivery_date", "weight_kg"],
        column_semantics={
            "order_id": "Unique order identifier.",
            "delivery_date": "Required delivery date for the order.",
            "weight_kg": "Order weight used by vehicle capacity constraints.",
        },
        column_profiles=[
            {
                "name": "order_id",
                "logical_type": "categorical",
                "row_count": 100,
                "non_null_count": 100,
                "unique_count": 100,
                "null_ratio": 0.0,
                "top_values": ["A001:1"],
            },
            {
                "name": "delivery_date",
                "logical_type": "datetime",
                "row_count": 100,
                "non_null_count": 100,
                "unique_count": 15,
                "null_ratio": 0.0,
                "datetime_stats": {"min": "2026-01-01", "max": "2026-01-15", "range_days": 14},
            },
            {
                "name": "weight_kg",
                "logical_type": "numeric",
                "row_count": 100,
                "non_null_count": 98,
                "unique_count": 70,
                "null_ratio": 0.02,
                "numeric_stats": {"mean": 12.5, "std": 3.1, "var": 9.61, "min": 1.0, "max": 30.0},
            },
        ],
        source_metadata={
            "shape": [100, 3],
            "preview": [{"order_id": "A001", "delivery_date": "2026-01-01", "weight_kg": 12.5}],
            "probe_results": {"large": "must remain local"},
        },
    )

    pack = mod._build_field_section_pack(
        file_summaries=[fs],
        downstream_context={
            "authoritative_memory": {"task_goal": "assign orders to vehicles while respecting capacity"},
            "constraint_memory": {"summary": "vehicle capacity constraints"},
        },
        relations=[],
    )

    assert pack["table_index"][0]["table_id"] == "orders.xlsx"
    detail = pack["table_field_details"][0]
    fields = {field["name"]: field for field in detail["fields"]}
    assert fields["order_id"]["meaning"] == "Unique order identifier."
    assert fields["delivery_date"]["datetime_stats"]["range_days"] == 14
    assert fields["weight_kg"]["numeric_stats"]["mean"] == 12.5
    text = str(pack)
    assert "probe_results" not in text
    assert "must remain local" not in text
