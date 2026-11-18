from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from autorealize.execution_contract import (
    artifact_consistency_gate_passes,
    build_constraint_ledger,
    build_execution_contract,
)
from autorealize.config import AutoRealizeConfig
from autorealize.llm.client import LLMClient
from autorealize.modules.task_definition import TaskDefinitionModule
from autorealize.models import ConstraintMemory, DescriptionProtocolBundle, EvaluationContractReview, FileSummary, SampleSubmissionSpec, ProblemParadigmReview
from autorealize.pipeline import _extract_constraint_memory
from autorealize.prompt_cache import stable_dynamic_prompt
from autorealize.report_writer import build_automl_context_pack, evaluation_contract_defects, render_automl_context_markdown


def _evaluation() -> dict:
    return {
        "passed": True,
        "executable": True,
        "primary_metric": "MAE",
        "metric_direction": "minimize",
        "metric_formula": "mean(abs(y_true - y_pred))",
        "validation_protocol": "Reserve the final complete calendar month; fit only on earlier months.",
        "assumptions": [],
    }


def _contract(*, context=None, evaluation=None):
    return build_execution_contract(
        bundle={"task_goal": "Forecast monthly revenue", "problem_paradigm": "ml_dl_prediction"},
        context=context or {}, evaluation=evaluation or _evaluation(),
        output={"columns": ["month", "revenue"]},
        data_schema={"tables": [{"path": "sales.csv", "columns": ["month", "revenue"]}]},
    )


def test_static_solver_handoff_preserves_task_contract_without_prediction_defaults():
    bundle = {"problem_paradigm": "static_optimization", "constraints": ["Missing business dates must remain missing"]}
    evaluation = {**_evaluation(), "primary_metric": "contract_checks", "metric_formula": "passed / total",
                  "validation_protocol": "Twelve deterministic software contract checks, no label split."}
    original = deepcopy((bundle, evaluation))
    static = build_execution_contract(bundle=bundle, context={}, evaluation=evaluation,
                                      output={"columns": []}, data_schema={})
    assert (bundle, evaluation) == original
    assert static["constraint_ledger"][0]["description"] == bundle["constraints"][0]
    assert "input-validation and solving entrypoints" in static["reproducibility_requirements"][0]
    assert "training and inference entrypoints" not in "\n".join(static["reproducibility_requirements"])
    assert "software-contract validation distinct" in "\n".join(static["candidate_comparison_rules"])
    assert "not_applicable_static_optimization" in static["identity_semantics"]["split_fingerprint"]
    assert "do not generate evaluation unit IDs, folds" in static["identity_semantics"]["split_fingerprint"]
    assert "explicitly defined task evaluation split" in static["identity_semantics"]["split_fingerprint"]
    prediction = _contract()
    assert "training and inference entrypoints" in prediction["reproducibility_requirements"][0]
    assert any("out-of-fold" in rule for rule in prediction["candidate_comparison_rules"])
    assert prediction["identity_semantics"]["split_fingerprint"].startswith("Generated from actual evaluation unit IDs/folds")


def test_evaluator_identity_changes_with_ranking_rules_but_not_review_prose():
    original = _contract()
    revised = {**_evaluation(), "rationale": "Reviewed again", "evidence": ["README.md"]}
    assert _contract(evaluation=revised)["evaluation_id"] == original["evaluation_id"]
    assert _contract(evaluation=revised)["contract_id"] == original["contract_id"]
    for change in (
        {"ranking_scope": "software_contract_checks"},
        {"scalar_score_formula": "MAE + 2 * violations"},
        {"metric_direction": "maximize"},
        {"validation_protocol": "Hold out two complete months"},
        {"invalid_solution_rules": ["Reject negative revenue"]},
    ):
        result = _contract(evaluation={**_evaluation(), **change})
        assert result["evaluation_id"] != original["evaluation_id"]
        assert result["contract_id"] != original["contract_id"]


def test_constraint_provenance_and_strength_survive_handoff_without_obsolete_rules():
    context = {"constraint_memory": {"items": [
        {"description": "Capacity is 40", "strength": "hard", "status": "superseded"},
        {"description": "Capacity is 50", "strength": "hard", "status": "current",
         "evidence": ["spec.md#capacity"], "acceptance_check": "load <= 50 tonnes"},
        {"description": "Prefer simpler models", "strength": "soft", "status": "current"},
        {"description": "Date-like column", "priority": "high"},
    ]}}
    result = _contract(context=context)
    ledger = result["constraint_ledger"]
    assert [item["description"] for item in ledger] == [
        "Capacity is 50", "Prefer simpler models", "Date-like column",
    ]
    assert [item["strength"] for item in ledger] == ["hard", "soft", "unknown"]
    assert ledger[0]["evidence"] == ["spec.md#capacity"]
    assert ledger[0]["acceptance_check"] == "load <= 50 tonnes"
    assert len({item["constraint_id"] for item in ledger}) == 3


def test_duplicate_evidence_never_downgrades_confirmed_constraints():
    context = {"constraint_memory": {"items": [{
        "description": "CPU only", "strength": "hard", "status": "current", "evidence": ["task_hint"],
    }]}, "authoritative_memory": {"constraints": ["CPU  only"], "source_files": ["requirements.md"]}}
    ledger = build_constraint_ledger({"constraints": ["CPU only", "No GPU execution"]}, context)
    assert len(ledger) == 2
    assert ledger[0]["strength"] == "hard"
    assert ledger[0]["status"] == "current"
    assert {item["source"] for item in ledger[0]["additional_sources"]} == {"authoritative_memory", "task_protocol"}
    assert ledger[1]["description"] == "No GPU execution"
    assert "never grants override" in _contract(context=context)["constraint_resolution_policy"][0]


def test_blocked_evaluation_and_unresolved_hard_constraints_are_not_ready():
    result = _contract(evaluation={**_evaluation(), "executable": False})
    assert result["readiness"] == "blocked"
    context = {"constraint_memory": {"items": [
        {"description": "Unknown legal capacity", "strength": "hard", "status": "unresolved"},
    ]}}
    result = _contract(context=context)
    assert result["readiness"] == "blocked"
    assert "Unknown legal capacity" in result["open_questions"]
    assert any("Unresolved hard constraint" in issue for issue in result["blocking_issues"])


def test_required_temporal_scan_is_part_of_executable_handoff_contract():
    blocked = _contract(context={
        "agent_context_pack": {
            "temporal_scan_requirement": {
                "required": True,
                "status": "blocked",
                "blocking_issues": ["measurements.csv: 2 timestamp values could not be parsed."],
            },
        },
    })
    assert blocked["readiness"] == "blocked"
    assert any("timestamp values could not be parsed" in issue for issue in blocked["blocking_issues"])

    passed = _contract(context={
        "temporal_scan_requirement": {
            "required": True,
            "status": "passed",
            "files": [{"file": "measurements.csv", "rows_scanned": 100}],
        },
    })
    assert passed["readiness"] == "ready"
    assert passed["temporal_scan_requirement"]["status"] == "passed"


def test_single_metric_contract_does_not_require_invented_tie_break_rules():
    contract = EvaluationContractReview(
        **_evaluation(), prediction_unit="one validation row", y_true_source="train.csv.sales",
        y_pred_source="candidate prediction", computation_scope="all fixed validation rows",
        aggregation_rule="mean absolute error over validation rows",
        submission_checks=["Exact output schema and complete row coverage"],
        leakage_guards=["Fit preprocessing only on training rows"],
        invalid_solution_rules=["Reject non-finite or negative predictions"],
        tie_break_rules=[],
    )
    assert evaluation_contract_defects(contract) == []


def test_final_artifact_failures_block_otherwise_executable_contract():
    original = _contract()
    for context in (
        {"artifact_consistency_review": {"passed": False, "issues": []}},
        {"artifact_consistency_review": {"passed": True, "issues": [
            {"severity": "blocking", "message": "Invalid physical source field"},
        ]}},
        {"defects_after_gate": ["Split memberships disagree"]},
    ):
        result = _contract(context=context)
        assert result["readiness"] == "blocked"
        assert result["blocking_issues"]
        assert result["evaluation_id"] == original["evaluation_id"]
    repaired = _contract(context={"artifact_consistency_review": {"passed": True, "issues": []}})
    assert repaired["readiness"] == "ready"


def test_question_resolution_keeps_remaining_gaps_and_original_evidence():
    original = "Split IDs and measured model scores are missing"
    context = {"constraint_memory": {"unresolved_questions": [original, "Capacity dates are missing"]},
               "open_question_resolutions": [{"original_text": original,
                   "remaining_questions": ["Measured model scores are missing"],
                   "evidence_refs": ["qdi:q2"], "reason": "Actual QDI established only split IDs"}]}
    before = deepcopy(context)
    result = _contract(context=context)
    assert result["open_questions"] == ["Measured model scores are missing", "Capacity dates are missing"]
    assert result["resolved_question_history"][0]["original_text"] == original
    assert context == before


def test_resolved_conflicts_are_excluded_and_assumptions_remain_visible():
    context = {"authoritative_memory": {"authority_conflicts": [
        {"status": "resolved", "message": "Older price list replaced"},
    ]}}
    result = _contract(context=context, evaluation={**_evaluation(), "assumptions": ["Use calendar months"]})
    assert result["readiness"] == "ready_with_assumptions"
    assert result["authority_conflicts"] == []
    assert result["assumptions"] == ["Use calendar months"]


def test_current_unresolved_authority_conflict_blocks_handoff():
    context = {"authoritative_memory": {"authority_conflicts": [
        {"status": "unresolved", "message": "Two incompatible official target definitions"},
    ]}}
    result = _contract(context=context)
    assert result["readiness"] == "blocked"
    assert any("authority conflict" in issue for issue in result["blocking_issues"])


def test_shared_unspecified_metadata_is_visible_without_inventing_a_conflict():
    conflict = {"conflict_id": "AC-3", "topic": "Submission filename", "execution_disposition": "unknown",
                "positions": ["task_hint: 未显式规定提交文件名。",
                              "description.md: 未给出提交文件名，仅给出头部格式 Id,SalePrice 及示例 sample_submission.csv。"]}
    context = {"authoritative_memory": {"authority_conflicts": [conflict]}}
    before = deepcopy(context)
    result = _contract(context=context)
    assert result["readiness"] == "ready_with_assumptions"
    assert result["authority_conflicts"] == [conflict]
    assert result["authority_conflict_assessments"][0]["disposition"] == "shared_unspecified"
    assert context == before
    # This classification never fills a required execution field.
    missing_metric = _contract(context=context, evaluation={**_evaluation(), "metric_formula": ""})
    assert missing_metric["readiness"] == "blocked"


def test_one_explicit_rule_or_blocking_flag_cannot_be_waived_as_shared_absence():
    for conflict in (
        {"positions": ["user: 未规定输出文件名", "spec: 必须输出 required.csv"]},
        {"positions": ["user: not specified", "spec: does not specify a filename"], "blocks_execution": True},
        {"positions": ["user: 未规定提交文件名，但禁止输出结果", "spec: 未规定文件名"]},
    ):
        assert _contract(context={"authoritative_memory": {"authority_conflicts": [conflict]}})["readiness"] == "blocked"


def test_unrecognized_authority_conflict_status_fails_closed():
    for status in ("pending_review", "uncertain", "unknown"):
        result = _contract(context={"authoritative_memory": {"authority_conflicts": [
            {"status": status, "message": "Two incompatible official target definitions"},
        ]}})

        assert result["readiness"] == "blocked"
        assert result["authority_conflict_assessments"][0]["disposition"] == "blocking"
        assert any("authority conflict" in issue for issue in result["blocking_issues"])


def test_scoped_unresolved_authority_conflicts_are_retained_without_blocking():
    conflicts = [
        {
            "conflict_id": "C1",
            "topic": "质量合格阈值",
            "description": "设定值与合格标记不一致。",
            "resolution": "保留冲突记录，不推导质量合格阈值。",
            "status": "unresolved",
        },
        {
            "conflict_id": "C2",
            "topic": "蒸汽最佳流量主导因素",
            "description": "天气因素与进料量因素的观点不同。",
            "resolution": "保留双方观点，不判定为硬约束；蒸汽建议仅作数据参考，非因果结论。",
            "status": "unresolved",
        },
        {
            "conflict_id": "C3",
            "topic": "液位适用范围",
            "description": "日常工况和开机工况的液位口径不同。",
            "resolution": "按不同情境分别保留，不得混用。",
            "status": "unresolved",
        },
    ]
    result = _contract(context={
        "authoritative_memory": {
            "authority_conflicts": conflicts,
            "unresolved_questions": ["阈值与工况仍需现场确认"],
        },
    })

    assert result["readiness"] == "ready_with_assumptions"
    assert result["blocking_issues"] == []
    assert result["authority_conflicts"] == conflicts
    assert result["open_questions"] == ["阈值与工况仍需现场确认"]
    assert [item["disposition"] for item in result["authority_conflict_assessments"]] == [
        "retained_non_blocking",
        "retained_non_blocking",
        "retained_non_blocking",
    ]


def test_safe_use_boundary_does_not_unblock_core_target_conflict_or_explicit_blocker():
    conflicts = [
        {
            "topic": "预测目标定义",
            "description": "Two incompatible official target definitions.",
            "resolution": "Retain the conflict; do not infer which target is authoritative.",
            "status": "unresolved",
        },
        {
            "topic": "工况差异",
            "description": "Two operating regimes have different settings.",
            "resolution": "按不同情境分别保留，不得混用。",
            "status": "unresolved",
            "severity": "blocking",
        },
    ]
    result = _contract(context={"authoritative_memory": {"authority_conflicts": conflicts}})

    assert result["readiness"] == "blocked"
    assert len(result["blocking_issues"]) == 2
    assert [item["disposition"] for item in result["authority_conflict_assessments"]] == [
        "blocking",
        "blocking",
    ]


def test_structured_conflict_disposition_requires_explicit_limits_and_respects_resolution_status():
    conflicts = [
        {
            "topic": "质量合格阈值",
            "fact_a": "设定值为0.5。",
            "fact_b": "合格标记与设定值不一致。",
            "resolution_status": "unresolved_conflict_documented",
            "execution_disposition": "non_blocking_with_limits",
            "execution_limits": ["不推导质量阈值；缺少质量证据时蒸汽建议返回 review_required。"],
        },
        {
            "topic": "蒸汽流量因素",
            "resolution_status": "unresolved_conflict_documented",
            "execution_disposition": "non_blocking_with_limits",
            "execution_limits": [],
        },
        {
            "topic": "预测目标定义",
            "resolution_status": "unresolved_conflict_documented",
            "execution_disposition": "non_blocking_with_limits",
            "execution_limits": ["保留双方观点，不选择目标。"],
        },
        {
            "topic": "旧工况记录",
            "resolution_status": "resolved_by_authority",
        },
    ]
    result = _contract(context={"authoritative_memory": {"authority_conflicts": conflicts}})

    assert result["readiness"] == "blocked"
    assert [item["disposition"] for item in result["authority_conflict_assessments"]] == [
        "retained_non_blocking",
        "blocking",
        "blocking",
    ]
    assert any("蒸汽流量因素" in issue for issue in result["blocking_issues"])
    assert any("预测目标定义" in issue for issue in result["blocking_issues"])
    assert all("旧工况记录" not in issue for issue in result["blocking_issues"])


def test_explicit_task_hint_boundary_can_scope_a_core_contract_conflict():
    conflict = {
        "topic": "granularity_and_target",
        "conflict": "Historical documents use minute-level results; task_hint fixes hourly t+1 prediction.",
        "highest_priority": "task_hint",
        "sources": ["task_hint", "historical_results.pdf", "requirements.docx"],
        "execution_disposition": "non_blocking_with_limits",
        "execution_limits": ["Historical minute-level results do not enter the hourly target or metrics."],
    }

    result = _contract(context={"authoritative_memory": {"authority_conflicts": [conflict]}})

    assert result["readiness"] == "ready_with_assumptions"
    assert result["authority_conflict_assessments"][0]["disposition"] == "retained_non_blocking"
    assert not any("authority conflict" in issue for issue in result["blocking_issues"])


def test_core_conflict_requires_task_hint_to_be_a_recorded_highest_priority_source():
    for changes in (
        {"highest_priority": "requirements.docx"},
        {"sources": ["historical_results.pdf", "requirements.docx"]},
    ):
        conflict = {
            "topic": "prediction target",
            "description": "Two incompatible official target definitions.",
            "highest_priority": "task_hint",
            "sources": ["task_hint", "requirements.docx"],
            "execution_disposition": "non_blocking_with_limits",
            "execution_limits": ["Use the documented boundary."],
            **changes,
        }
        result = _contract(context={"authoritative_memory": {"authority_conflicts": [conflict]}})

        assert result["readiness"] == "blocked"
        assert result["authority_conflict_assessments"][0]["disposition"] == "blocking"


def test_description_and_investigation_open_issues_survive_handoff():
    result = _contract(context={
        "description_sections": {"overview_task_definition": {"open_issues": ["Forecast cutoff missing"]}},
        "agent_context_pack": {"question_memory": {"unresolved_questions": ["Currency unit missing"]}},
    })
    assert result["open_questions"] == ["Forecast cutoff missing", "Currency unit missing"]
    assert result["readiness"] == "ready_with_assumptions"


def test_required_rl_is_not_downgraded_by_static_problem_description():
    pack = build_automl_context_pack(
        DescriptionProtocolBundle(
            problem_paradigm="static_optimization", required_method_families=["reinforcement_learning"],
        ), [], {"problem_paradigm_review": {"explicit_rl_requested": True, "rl_required": True}},
    )
    assert pack.method_strategy["first_draft_method"] == "reinforcement_learning"
    assert any("required solution method" in item for item in pack.modeling_boundary)
    assert not any("First implement a scorable static optimization baseline" in item for item in pack.modeling_boundary)


def test_keyword_hints_cannot_reintroduce_business_constraints_after_extraction():
    llm = SimpleNamespace(ask_structured=lambda **kwargs: ConstraintMemory())
    summary = FileSummary(path="sales.csv", role="raw_data_table", summary="Historical price and date observations", columns=["price", "date"])
    result = _extract_constraint_memory(
        llm_client=llm, prompt_mgr=SimpleNamespace(), file_summaries=[summary], task_hint="Forecast sales",
    )
    assert result["items"] == []
    assert result["heuristic_observations"]
    assert all(item["strength"] == "unknown" and item["status"] == "unresolved" for item in result["heuristic_observations"])
    assert all(item["priority"] == "medium" and "约束" not in item["name"]
               for item in result["heuristic_observations"])
    assert "价格/费用线索" in {item["name"] for item in result["heuristic_observations"]}
    assert all("运价" not in item["name"] for item in result["heuristic_observations"])


def test_stable_prompt_prefix_is_identical_for_equivalent_mapping_order():
    first = {"tables": {"b": {"columns": ["id", "value"], "path": "data.csv"}, "a": 1}, "task_hint": "Forecast"}
    second = {"task_hint": "Forecast", "tables": {"a": 1, "b": {"path": "data.csv", "columns": ["id", "value"]}}}
    stable_a, dynamic_a = stable_dynamic_prompt(stable=first, dynamic={"round": 1})
    stable_b, dynamic_b = stable_dynamic_prompt(stable=second, dynamic={"round": 2})
    assert stable_a == stable_b
    assert dynamic_a != dynamic_b
    changed_order = deepcopy(second)
    changed_order["tables"]["b"]["columns"].reverse()
    assert stable_dynamic_prompt(stable=changed_order)[0] != stable_a


def test_context_pack_renders_contract_and_clean_constraint_text():
    context = {"constraint_memory": {"items": [
        {"description": "Use monthly totals", "status": "current", "strength": "hard", "evidence": ["description.md"]},
        {"description": "Outdated daily totals", "status": "superseded"},
    ]}}
    pack = build_automl_context_pack(
        DescriptionProtocolBundle(task_goal="Forecast sales"), [], context,
        evaluation_contract=EvaluationContractReview(**_evaluation()),
    )
    rendered = render_automl_context_markdown(pack)
    assert pack.execution_contract["evaluation_id"].startswith("evaluation:")
    assert "## Execution Contract\n\n```json" in rendered
    assert "description.md" in rendered
    assert "constraint_resolution_policy" in rendered
    assert "never grants override" in rendered
    assert "Outdated daily totals" not in rendered
    assert pack.constraints == ["[hard; current] Use monthly totals"]
    assert rendered.index("## Execution Contract") < rendered.index("## Evaluation Contract Reference")


def test_provider_model_identifier_and_prompt_roles_are_preserved(tmp_path, monkeypatch):
    config = AutoRealizeConfig()
    config.llm.api_key = "test-key"
    config.llm.base_url = "https://compatible-provider.example/v1"
    config.llm.model_name = "gpt-5.6-sol"
    config.llm.enable_cache = False
    client = LLMClient(config, tmp_path)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs["create_kwargs"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage={},
        )

    monkeypatch.setattr(client, "_chat_completion_with_network_retry", completion)
    client.ask_text("stage instructions", "dynamic", "provider_contract", static_user_prompt="stable evidence")
    request = calls[0]
    assert request["model"] == "gpt-5.6-sol"
    assert "prompt_cache_key" not in request
    assert "extra_body" not in request
    assert request["messages"][1]["content"] == "stable evidence"
    assert request["messages"][-1]["content"] == "dynamic"


def test_local_cache_identity_accounts_for_control_prompt_language(tmp_path):
    config = AutoRealizeConfig()
    config.llm.api_key = "test-key"
    client = LLMClient(config, tmp_path)
    args = {"prompt_name": "test", "system_prompt": "stage", "user_prompt": "data"}
    first = client._cache_key(**args)
    config.prompt.control_language = "en"
    assert client._cache_key(**args) != first


def test_local_cache_identity_accounts_for_output_budget(tmp_path):
    config = AutoRealizeConfig()
    config.llm.api_key = "test-key"
    client = LLMClient(config, tmp_path)
    args = {"prompt_name": "test", "system_prompt": "stage", "user_prompt": "data"}

    assert client._cache_key(**args, output_max_tokens=1024) != client._cache_key(
        **args, output_max_tokens=2048,
    )


def test_known_source_filenames_do_not_become_business_field_aliases():
    schema = {"tables": [{"source_file": "train.csv", "physical_columns_exact": ["weekday", "sales"]}]}
    spec = SampleSubmissionSpec(source_fields={"sales": "Read `train.csv` physical field `sales`."})
    assert TaskDefinitionModule._correct_sample_spec_source_fields(spec, schema_contract=schema) == []
    assert spec.validation_rules == []


def test_output_self_reference_is_preserved_while_source_typo_is_corrected(tmp_path, monkeypatch):
    schema = {"tables": [{"source_file": "curve.csv", "physical_columns_exact": ["Time [s]", "Temperature"]}]}
    spec = SampleSubmissionSpec(columns=["time_s"], source_fields={
        "time_s": "Derive output `time_s` from `Time [s]` and inspect `Temperatur`."})
    corrections = TaskDefinitionModule._correct_sample_spec_source_fields(spec, schema_contract=schema)
    assert [(row["from"], row["to"]) for row in corrections] == [("Temperatur", "Temperature")]
    assert spec.source_fields["time_s"] == "Derive output `time_s` from `Time [s]` and inspect `Temperature`."
    module = TaskDefinitionModule(AutoRealizeConfig(), SimpleNamespace(llm_client=None, prompt_mgr=None), tmp_path, tmp_path)
    monkeypatch.setattr(module, "_resolve_exact_source_field_alias",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Derived output treated as input")))
    assert module._resolve_unresolved_source_field_aliases(
        spec, schema_contract=schema, downstream_context={}, original_text="") == []
    guard = module._collect_source_alias_guard(sample_spec=spec, schema_contract=schema, downstream_context={})
    assert len(guard) == 1
    assert guard[0]["alias"] == "time_s"
    assert guard[0]["status"] == "declared_output_not_physical_column"


def test_qualified_physical_reference_is_checked_even_when_output_has_same_name():
    schema = {"tables": [{"source_file": "curve.csv", "physical_columns_exact": ["temperature"]}]}
    spec = SampleSubmissionSpec(columns=["temperatur"], source_fields={
        "temperatur": "Generate `temperatur` from `curve.csv::temperatur`."})
    corrections = TaskDefinitionModule._correct_sample_spec_source_fields(spec, schema_contract=schema)
    assert corrections == []
    guard = TaskDefinitionModule._collect_source_alias_guard(
        sample_spec=spec, schema_contract=schema, downstream_context={})
    source = next(row for row in guard if row["alias"] == "curve.csv::temperatur")
    assert source["status"] == "unresolved_business_concept"
    assert source["candidate_exact_columns"] == ["temperature"]
    assert any("curve.csv::temperatur" in rule for rule in spec.validation_rules)


def test_qualified_source_fields_resolve_within_their_actual_table():
    schema = {"tables": [
        {"source_file": "train.csv", "physical_columns_exact": ["weekday", "sales"]},
        {"source_file": "other.csv", "physical_columns_exact": ["cost"]},
    ]}
    guard = TaskDefinitionModule._collect_source_alias_guard(
        sample_spec=SampleSubmissionSpec(), schema_contract=schema,
        downstream_context={"constraint_memory": {"items": [{"related_fields": ["train.csv.weekday", "train.csv.cost"]}]}},
    )
    by_alias = {entry["alias"]: entry for entry in guard}
    assert by_alias["train.csv.weekday"]["exact_physical_column"] == "weekday"
    assert by_alias["train.csv.cost"]["status"] == "unresolved_business_concept"
    assert "cost" not in by_alias["train.csv.cost"]["candidate_exact_columns"]


def test_pipe_qualified_exact_source_field_is_not_blocked_as_an_alias():
    schema = {"tables": [{
        "table_id": "orders.xlsx::detail",
        "source_file": "orders.xlsx",
        "sheet_name": "detail",
        "physical_columns_exact": ["订单号", "下单重量(kg)"],
    }]}
    guard = TaskDefinitionModule._collect_source_alias_guard(
        sample_spec=SampleSubmissionSpec(),
        schema_contract=schema,
        downstream_context={"constraint_memory": {"items": [{
            "related_fields": ["orders.xlsx | detail | 订单号", "orders.xlsx | detail | 下单重量(kg)"],
        }]}},
    )
    assert guard == [
        {
            "alias": "orders.xlsx | detail | 下单重量(kg)",
            "source": ["constraint_memory.items[0].related_fields"],
            "status": "corrected_to_exact_physical_column",
            "exact_physical_column": "下单重量(kg)",
            "rule": "Use the exact physical column for raw pandas access; keep the alias only as business wording or a derived/output name.",
        },
        {
            "alias": "orders.xlsx | detail | 订单号",
            "source": ["constraint_memory.items[0].related_fields"],
            "status": "corrected_to_exact_physical_column",
            "exact_physical_column": "订单号",
            "rule": "Use the exact physical column for raw pandas access; keep the alias only as business wording or a derived/output name.",
        },
    ]


def test_colon_and_slash_sheet_qualified_fields_resolve_to_their_exact_sources():
    schema = {"tables": [
        {"table_id": "设备.csv", "source_file": "设备.csv",
         "physical_columns_exact": ["SampleTime", "蒸汽进气压力", "蒸汽进气流量"]},
        {"table_id": "浓度.xls::Sheet1", "source_file": "浓度.xls", "sheet_name": "Sheet1",
         "physical_columns_exact": ["参数", "部位", "采样时间", "结果\n[单位为当量]"]},
    ]}
    aliases = [
        "设备.csv:蒸汽进气压力",
        "设备.csv:蒸汽进气流量",
        "浓度.xls/Sheet1:采样时间",
        "浓度.xls/Sheet1:结果\n[单位为当量]",
    ]
    guard = TaskDefinitionModule._collect_source_alias_guard(
        sample_spec=SampleSubmissionSpec(),
        schema_contract=schema,
        downstream_context={"constraint_memory": {"items": [{"related_fields": aliases}]}},
    )
    assert {item["alias"]: item["exact_physical_column"] for item in guard} == {
        alias: alias.rsplit(":", 1)[-1] for alias in aliases
    }
    assert all(item["status"] == "corrected_to_exact_physical_column" for item in guard)


def test_unquoted_slash_sheet_reference_extracts_only_the_qualified_field_alias():
    schema = {"tables": [{
        "table_id": "浓度.xls::Sheet1",
        "source_file": "浓度.xls",
        "sheet_name": "Sheet1",
        "physical_columns_exact": ["采样时间"],
    }]}
    names = TaskDefinitionModule._exact_schema_non_column_names(schema)
    columns = set(TaskDefinitionModule._exact_schema_columns(schema))
    tokens = TaskDefinitionModule._source_field_alias_tokens(
        "按 浓度.xls/Sheet1:采样时间戳 读取",
        names,
        columns,
    )
    assert tokens == ["采样时间戳"]


def test_warning_only_consistency_result_does_not_block_execution_contract():
    review = {"passed": False, "issues": [{"severity": "warning", "message": "Residual uncertainty"}]}
    assert artifact_consistency_gate_passes(review)
    ready = _contract(context={"artifact_consistency_review": review})
    assert ready["readiness"] != "blocked"
    assert "Final artifact consistency review has not passed." not in ready["blocking_issues"]

    assert not artifact_consistency_gate_passes({"passed": False, "issues": []})
    assert not artifact_consistency_gate_passes(
        {"passed": False, "issues": [{"severity": "blocking"}]}
    )
    assert not artifact_consistency_gate_passes(
        {"passed": True, "issues": [{"severity": "unknown"}]}
    )


def test_source_diagnostics_do_not_parse_metadata_or_prose_as_columns():
    schema = {"tables": [{"source_file": "train.csv", "physical_columns_exact": ["date", "Usage_kWh"]}]}
    spec = SampleSubmissionSpec(source_fields={
        "Usage_kWh": "Prediction, not copied from `DataFrame`; label reference `source_file=train.csv`, "
                     "`sheet_name=not applicable`, `physical_field=Usage_kWh`.",
    }, validation_rules=["Finite prediction required", "AutoRealize could not resolve source field alias `DataFrame`"])
    assert TaskDefinitionModule._correct_sample_spec_source_fields(spec, schema_contract=schema) == []
    assert spec.validation_rules == ["Finite prediction required"]
    guard = TaskDefinitionModule._collect_source_alias_guard(
        sample_spec=spec, schema_contract=schema,
        downstream_context={"authoritative_memory": {"constraints": ["Keep all weights. Use two Workers."]}},
    )
    assert guard == []


def test_optional_sample_generation_does_not_create_a_source_requirement():
    assert not TaskDefinitionModule._source_requires_sample({"sample_submission_generation_requested": True}, {})


def test_unit_bearing_multiword_columns_remain_exact_through_repeated_diagnostics(tmp_path, monkeypatch):
    columns = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
               "Torque [Nm]", "Tool wear [min]", "UDI", "Product ID"]
    schema = {"tables": [{"source_file": "ai4i.csv", "physical_columns_exact": columns}]}
    original = ", ".join("ai4i.csv::" + column for column in columns[:6])
    spec = SampleSubmissionSpec(source_fields={"failure_probability": original})
    config = AutoRealizeConfig()
    config.llm.api_key = "test-key"
    module = TaskDefinitionModule(config, SimpleNamespace(llm_client=None, prompt_mgr=None), tmp_path, tmp_path)
    monkeypatch.setattr(module, "_resolve_exact_source_field_alias",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Exact column treated as alias")))
    assert module._resolve_unresolved_source_field_aliases(
        spec, schema_contract=schema, downstream_context={}, original_text="Preserve exact fields",
    ) == []
    for _ in range(3):
        assert TaskDefinitionModule._correct_sample_spec_source_fields(spec, schema_contract=schema) == []
        assert spec.source_fields["failure_probability"] == original
        assert TaskDefinitionModule._collect_source_alias_guard(
            sample_spec=spec, schema_contract=schema, downstream_context={}) == []
    replace = TaskDefinitionModule._replace_source_field_alias
    text = 'ai4i.csv::Process, ai4i.csv::Process temperature [K]; `Process`'
    expected = 'ai4i.csv::Process temperature [K], ai4i.csv::Process temperature [K]; `Process temperature [K]`'
    assert replace(text, "Process", "Process temperature [K]") == expected
    assert replace(expected, "Process", "Process temperature [K]") == expected
    assert replace("Process this input", "Process", "Process temperature [K]") == "Process this input"
    assert TaskDefinitionModule._source_requires_sample({}, {"columns": ["id", "target"]})
    assert TaskDefinitionModule._source_requires_sample({"authoritative_submission_contract": {"sample_filename": "sample.csv"}}, {})


def test_sample_structure_check_never_claims_final_prediction_acceptance():
    pack = build_automl_context_pack(DescriptionProtocolBundle(), [])
    assert pack.output_contract["sample_validation_scope"] == "sample_schema_only"
    assert pack.output_contract["final_prediction_validated"] is False


def test_declared_prediction_output_is_never_fuzzy_mapped_to_leakage_field():
    spec = SampleSubmissionSpec(columns=["predicted_failure"],
        source_fields={"predicted_failure": "Apply frozen probability threshold"})
    guard = TaskDefinitionModule._collect_source_alias_guard(
        sample_spec=spec,
        schema_contract={"tables": [{"source_file": "data.csv", "physical_columns_exact": ["Machine failure", "Product ID"]}]},
        downstream_context={"constraint_memory": {"items": [{"related_fields": ["predicted_failure"]}]}},
    )
    assert guard[0]["status"] == "declared_output_not_physical_column"
    assert "candidate_exact_columns" not in guard[0]
    assert "unresolved" not in TaskDefinitionModule._apply_source_alias_guard_to_description("# Task", guard)


def test_wide_table_tail_column_is_not_fuzzy_corrected_from_truncated_display_schema():
    columns = [f"Unnamed: {index}" for index in range(255)]
    schema = {"tables": [{"source_file": "cooling.csv", "physical_columns_exact": columns[:180],
                         "physical_columns_omitted": 75, "read_contract": {"validated_columns_exact": columns}}]}
    spec = SampleSubmissionSpec(source_fields={"y_pred": "Read `Unnamed: 254` using `cooling.csv`"})
    assert "Unnamed: 254" in TaskDefinitionModule._exact_schema_columns(schema)
    assert TaskDefinitionModule._collect_source_alias_guard(sample_spec=spec, schema_contract=schema,
        downstream_context={"constraint_memory": {"items": [{"related_fields": ["Unnamed: 254"]}]}}) == []
    assert TaskDefinitionModule._correct_sample_spec_source_fields(spec, schema_contract=schema) == []
    assert "Unnamed: 254" in spec.source_fields["y_pred"]
    table = schema["tables"][0]
    table["read_contract"] = {}
    resolved, _ = TaskDefinitionModule._resolve_exact_source_field_alias(
        "Unnamed: 254", schema_contract=schema, physical_columns=columns[:180], non_column_names=set())
    assert resolved is None


def test_qualified_duplicate_delimiter_fix_does_not_guess_other_columns():
    schema = {"tables": [{"table_id": "lab.xls::Sheet1", "source_file": "lab.xls", "sheet_name": "Sheet1",
                         "physical_columns_exact": ["result [unit]", "result [unit2]"]}]}
    spec = SampleSubmissionSpec(source_fields={"row_id": "Label audit uses `lab.xls::Sheet1::result [unit]]`; derive row_id from original row position."})
    corrections = TaskDefinitionModule._correct_sample_spec_source_fields(spec, schema_contract=schema)
    assert len(corrections) == 1
    assert "`result [unit]`" in spec.source_fields["row_id"]
    assert "original row position" in spec.source_fields["row_id"]
    resolved, _ = TaskDefinitionModule._resolve_exact_source_field_alias(
        "lab.xls::Sheet1::result [unit3]", schema_contract=schema, physical_columns=[], non_column_names=set())
    assert resolved is None


def test_generated_sample_records_actual_filename_and_distinguishes_seed_table(tmp_path):
    import pandas as pd
    target = tmp_path / "sample_submission.csv"
    pd.DataFrame({"row_id": ["lab.xls::Sheet1:9"], "y_pred": [0.0]}).to_csv(target, index=False)
    sample = SampleSubmissionSpec(source="generated_spec", sample_filename="sample_prediction_frame.csv",
                                  columns=["row_id", "y_pred"], source_fields={"row_id": "Derived from lab.xls::Sheet1 row number"})
    schema = {"tables": [{"table_id": "lab.xls::Sheet1"}, {"table_id": "weather.csv"}]}
    report = {"source": "sample_spec_builder+validator", "passed": True, "candidate_table": "weather.csv"}
    TaskDefinitionModule._synchronize_generated_sample_metadata(sample, report, schema, target)
    assert sample.sample_filename == "sample_submission.csv"
    assert report["candidate_table"] == "lab.xls::Sheet1"
    assert report["seed_table"] == "weather.csv"
    assert "not a runtime IO trace" in report["source_table_basis"]
    assert report["preview"][0]["row_id"] == "lab.xls::Sheet1:9"
    sample.source = "official_sample"
    sample.sample_filename = "official.csv"
    TaskDefinitionModule._synchronize_generated_sample_metadata(sample, report, schema, target)
    assert sample.sample_filename == "official.csv"


def test_undefined_decision_filename_does_not_become_default_submission():
    module = TaskDefinitionModule.__new__(TaskDefinitionModule)
    problem = ProblemParadigmReview(problem_paradigm="static_optimization")
    bundle = DescriptionProtocolBundle()
    context = {"authoritative_submission_contract": {"is_defined": False, "output_filename": ""},
               "sample_submission_spec": {"should_generate": False, "output_filename": ""}}
    module._sync_sample_requirement(problem, bundle, context, {})
    assert bundle.output.output_filename == ""
    assert context["source_diagnostic_history"]
    bundle.output.output_filename = "explicit-delivery.json"
    module._sync_sample_requirement(problem, bundle, context, {})
    assert bundle.output.output_filename == "explicit-delivery.json"
    bundle.output.output_filename = "submission.csv"
    context["authoritative_submission_contract"]["is_defined"] = True
    context["authoritative_submission_contract"]["output_filename"] = "submission.csv"
    module._sync_sample_requirement(problem, bundle, context, {})
    assert bundle.output.output_filename == "submission.csv"
