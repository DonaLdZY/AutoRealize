from types import SimpleNamespace
import json
import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.models import ArtifactConsistencyIssue, ArtifactConsistencyReview, DescriptionProtocolBundle, EvaluationContractReview, FileSummary, ProblemParadigmReview, SampleSubmissionSpec
from autorealize.modules.task_definition import TaskDefinitionModule
from autorealize.modules.types import RuntimeServices


def contract(**changes):
    values = dict(
        passed=True, executable=True, authority_status="official", primary_metric="MAE",
        metric_direction="minimize", metric_formula="mean(abs(y_true-y_pred))",
        prediction_unit="one timestamp", y_true_source="data.csv.target", y_pred_source="prediction",
        computation_scope="all fixed development timestamps", aggregation_rule="arithmetic mean",
        validation_protocol="Use observed history before each origin, including earlier partitions.",
        submission_checks=["Complete timestamp coverage"], leakage_guards=["No future labels"],
        invalid_solution_rules=["Reject non-finite outputs"],
    )
    return EvaluationContractReview(**(values | changes))


def module_at(tmp_path, response):
    client = SimpleNamespace(ask_structured=lambda **kwargs: response.model_copy(deep=True))
    services = RuntimeServices(client, SimpleNamespace(load=lambda name: "rules"), None,
                               SimpleNamespace(log=lambda *args: None))
    return TaskDefinitionModule(AutoRealizeConfig(), services, tmp_path, tmp_path)


def test_first_round_assumptions_are_never_marked_official(tmp_path):
    module = module_at(tmp_path, contract(assumptions=["Default output filename predictions.csv"]))
    result = module._review_evaluation_contract(original_text="Forecast next period", downstream_context={})
    assert result.authority_status == "autorealize_assumption"
    assert len(module._evaluation_contract_revision_log) == 1
    assert result.assumptions == ["Default output filename predictions.csv"]


def test_problem_formula_is_synchronized_to_single_explicit_task_hint() -> None:
    from autorealize.models import ProblemParadigmReview

    review = ProblemParadigmReview(
        reasoning="先按 标准值=(蒸汽进气压力-600)+蒸汽进气流量 校正。",
        method_routing_notes=["蒸汽参考使用标准值=(蒸汽进气压力-600)+蒸汽进气流量。"],
    )
    task_hint = "压力校正关系：标准值=(蒸汽进气压力-600)*2+蒸汽进气流量。"
    repaired = TaskDefinitionModule._synchronize_authoritative_steam_formula(review, task_hint)
    assert "(蒸汽进气压力-600)+蒸汽进气流量" not in repaired.reasoning
    assert all("*2" in note for note in repaired.method_routing_notes)
    assert "*2" in repaired.reasoning


def test_explicit_time_series_split_and_purge_are_normalized_without_llm_repair() -> None:
    from autorealize.artifact_contract_repair import normalize_explicit_time_series_contract

    review = contract(
        primary_metric="dev_mae_al_n",
        metric_formula="N_dev uses ceil(0.6N) or cumulative proportions.",
        final_evaluation_metric="holdout_mae_al_n",
        final_evaluation_formula="holdout MAE after ceil-based split.",
        computation_scope="Use ceil or cumulative 60/20/20 proportions.",
        validation_protocol=(
            "1) Read labels. 2) Scan timestamps. 3) Sort timestamps. "
            "4) 时间切分：ceil or cumulative proportions. "
            "5) 边界 purge：剔除 s + 1h >= s_min，但又保留 s + 1h <= s_min。 "
            "6) Score candidates."
        ),
        assumptions=[
            "切分按最早 ceil(0.6N)、开发 ceil(0.2N) 划分。",
            "Train-only OOF folds use a fixed seed.",
        ],
        leakage_guards=["边界 purge 要求 s+1h >= s_min。", "No future features."],
    )
    requirements = (
        "最早60%可用标签训练，中间20%开发，最后20%一次性独立留出。"
        "按一小时预测跨度 purge 边界重叠样本。"
    )

    normalized, changed = normalize_explicit_time_series_contract(review, requirements)

    assert {"computation_scope", "validation_protocol", "assumptions", "leakage_guards"} <= set(changed)
    assert "ceil" not in normalized.computation_scope.lower()
    assert "ceil" not in normalized.validation_protocol.lower()
    assert "s + 1h > s_min" in normalized.validation_protocol
    assert "s + 1h <= s_min" in normalized.validation_protocol
    assert "validation_protocol" in normalized.computation_scope
    assert "validation_protocol" in normalized.leakage_guards[-1]
    assert any("取整方式" in item and "向下取整" in item for item in normalized.assumptions)
    assert any("等号样本保留" in item for item in normalized.assumptions)
    assert not any("purge" in item.lower() for item in normalized.leakage_guards)
    assert any("fixed seed" in item for item in normalized.assumptions)
    assert not any("ceil" in item.lower() for item in normalized.assumptions)
    assert normalized.primary_metric == "dev_mae_al_n"
    assert normalized.final_evaluation_metric == "holdout_mae_al_n"
    population_rule = next(item for item in normalized.invalid_solution_rules if "开发段冻结人口" in item)
    assert "该候选整体无效" in population_rule
    assert "不得跳过样本、改变分母" in population_rule
    assert "steam_advice" not in population_rule
    assert "蒸汽" not in population_rule
    assert any(population_rule == item for item in normalized.submission_checks)
    assert {"invalid_solution_rules", "submission_checks"} <= set(changed)


def test_time_series_normalizer_preserves_decimals_and_collapses_generated_contract_rules() -> None:
    from autorealize.artifact_contract_repair import normalize_explicit_time_series_contract

    review = contract(
        primary_metric="development_mae",
        final_evaluation_metric="holdout_mae",
        computation_scope=(
            "Build labels from the verified source. Split by floor(0.6N), floor(0.2N), then remainder."
        ),
        validation_protocol=(
            "1) Read labels. "
            "2) Split available labels using floor(0.6*M), floor(0.2*M), and the remainder. "
            "3) Purge training rows where s + 1h >= s_min_dev. "
            "4) Check threshold 0.5 and parser version 1.2. "
            "5) Purge development rows where s + 1h >= s_min_holdout. "
            "6) Evaluate the selected candidate once."
        ),
        assumptions=[
            "The temporal split is 60/20/20 using ceil rounding.",
            "Purge the earlier partition at both boundaries.",
            "Train-only OOF folds use a fixed seed.",
        ],
        leakage_guards=[
            "The 60/20/20 split and one-hour boundary purge must be applied.",
            "No future features.",
        ],
    )
    requirements = (
        "最早60%可用标签训练，中间20%开发，最后20%一次性留出；"
        "按一小时预测跨度 purge 边界重叠样本。"
    )

    normalized, _ = normalize_explicit_time_series_contract(review, requirements)
    protocol = normalized.validation_protocol

    assert protocol.count("floor(0.6N)") == 1
    assert protocol.count("s + 1h > s_min") == 1
    assert "floor(0.6*M)" not in protocol
    assert "s + 1h >= s_min_dev" not in protocol
    assert "threshold 0.5 and parser version 1.2" in protocol
    assert "0.5" in protocol and "1.2" in protocol
    assert "floor(0.6N)" not in normalized.computation_scope
    assert "validation_protocol" in normalized.computation_scope
    assert any("60/20/20" in item and "向下取整" in item for item in normalized.assumptions)
    assert any("边界 purge" in item and "等号样本保留" in item for item in normalized.assumptions)
    assert any("fixed seed" in item for item in normalized.assumptions)
    assert all("purge" not in item.lower() and "60/20/20" not in item for item in normalized.leakage_guards)

    normalized_again, changed_again = normalize_explicit_time_series_contract(normalized, requirements)
    assert normalized_again.validation_protocol == protocol
    assert changed_again == []


def test_time_series_normalizer_leaves_unspecified_split_untouched() -> None:
    from autorealize.artifact_contract_repair import normalize_explicit_time_series_contract

    original = contract(computation_scope="Use all fixed timestamps.")
    normalized, changed = normalize_explicit_time_series_contract(
        original,
        "Forecast next period using chronological validation.",
    )
    assert normalized is original
    assert changed == []


def test_time_series_normalizer_does_not_override_exact_user_rounding_or_purge_boundary() -> None:
    from autorealize.artifact_contract_repair import normalize_explicit_time_series_contract

    original = contract(
        validation_protocol=(
            "Use ceil(0.6N) for training and ceil(0.2N) for development. "
            "Purge earlier samples where label_time + 1h >= first_scored_time."
        )
    )
    requirements = (
        "最早60%训练、中间20%开发、最后20%留出；明确采用 ceil(0.6N) 和 ceil(0.2N)。"
        "一小时边界 purge：label_time + 1h >= first_scored_time 的样本剔除。"
    )

    normalized, changed = normalize_explicit_time_series_contract(original, requirements)

    assert normalized is original
    assert changed == []
    assert "ceil(0.6N)" in normalized.validation_protocol
    assert "label_time + 1h >=" in normalized.validation_protocol


def test_implicit_time_series_defaults_are_marked_as_assumptions_in_the_final_contract(tmp_path) -> None:
    requirements = (
        "最早60%可用标签训练，中间20%开发，最后20%一次性独立留出。"
        "按一小时预测跨度 purge 边界重叠样本。"
    )
    module = module_at(
        tmp_path,
        contract(authority_status="official", assumptions=[]),
    )
    module.config.prompt.evaluation_contract_max_rounds = 1

    result = module._review_evaluation_contract(
        original_text=requirements,
        downstream_context={},
    )

    assert result.authority_status == "autorealize_assumption"
    assert any("训练和开发样本数分别向下取整" in item for item in result.assumptions)
    assert any("等号样本保留" in item for item in result.assumptions)


def test_hybrid_business_objective_is_preserved_and_evaluation_boundary_is_explicit() -> None:
    bundle = DescriptionProtocolBundle(
        problem_paradigm="hybrid_ml_optimization",
        hybrid={
            "final_objective": (
                "主评估目标为留出集预测 MAE。蒸汽建议以支持样本充分性作为方案质量判据。"
            ),
            "validation_design": (
                "预测侧将留出集作为主目标。决策侧只在训练段拟合参考池。"
            ),
        },
    )
    evaluation = contract(
        primary_metric="dev_mae_al_n",
        metric_direction="minimize",
        final_evaluation_metric="holdout_mae_al_n",
    )

    changed = TaskDefinitionModule._synchronize_hybrid_objective(bundle, evaluation)

    assert changed
    assert bundle.hybrid.final_objective == (
        "主评估目标为留出集预测 MAE。蒸汽建议以支持样本充分性作为方案质量判据。"
    )
    assert "决策侧只在训练段拟合参考池" in bundle.hybrid.validation_design
    assert "dev_mae_al_n" in bundle.hybrid.validation_design
    assert "holdout_mae_al_n" in bundle.hybrid.validation_design
    assert "不参与搜索、排序或调参" in bundle.hybrid.validation_design
    assert "dev_mae_al_n" in bundle.evaluation_summary
    assert "holdout_mae_al_n" in bundle.evaluation_summary


def test_prediction_summary_lists_audit_metrics_without_changing_ranking_objective() -> None:
    bundle = DescriptionProtocolBundle(
        problem_paradigm="ml_dl_prediction",
        evaluation_summary="主评估：development_set_MAE，越低越好。",
    )
    evaluation = contract(
        audit_metrics=[
            "development_rmse: report only",
            "train_median_baseline_dev_mae: report only",
            "train_median_baseline_holdout_mae: report only",
        ],
    )

    changed = TaskDefinitionModule._synchronize_hybrid_objective(bundle, evaluation)

    assert changed
    assert "development_set_MAE" in bundle.evaluation_summary
    assert "development_rmse: report only" in bundle.evaluation_summary
    assert "train_median_baseline_dev_mae: report only" in bundle.evaluation_summary
    assert "train_median_baseline_holdout_mae: report only" in bundle.evaluation_summary
    assert "仅作报告，不参与候选排序" in bundle.evaluation_summary
    assert TaskDefinitionModule._synchronize_hybrid_objective(bundle, evaluation) is False


def test_no_assumptions_do_not_promote_unresolved_provenance(tmp_path):
    module = module_at(tmp_path, contract(authority_status="unresolved"))
    module.config.prompt.evaluation_contract_max_rounds = 1
    result = module._review_evaluation_contract(original_text="Forecast next period", downstream_context={})
    assert result.authority_status == "unresolved"


def test_repair_refreshes_existing_contract_and_both_handoff_objects(tmp_path, monkeypatch):
    repaired = contract(assumptions=["Default output filename predictions.csv"])
    module = module_at(tmp_path, repaired)
    original = contract(validation_protocol="History must not cross partitions.")
    pack = {"evaluation": original.model_dump()}
    protocol = {"evaluation_contract": original.model_dump()}
    context = {"task_hint": "rolling prediction", "description_sections": {}}
    captured = {}

    def compile_contract(**kwargs):
        captured.update(kwargs)
        return repaired

    monkeypatch.setattr(module, "_build_evaluation_evidence_pack", lambda **kwargs: {"full_evidence": True})
    monkeypatch.setattr(module, "_review_evaluation_contract", compile_contract)
    monkeypatch.setattr(module, "_write_automl_context_pack",
                        lambda **kwargs: {"evaluation": kwargs["evaluation_contract"].model_dump()})
    monkeypatch.setattr(module, "_write_main_task_protocol",
                        lambda **kwargs: {"evaluation_contract": kwargs["evaluation_contract"].model_dump(),
                                          "automl_context_pack": kwargs["automl_context_pack"]})
    desc = module._repair_final_evaluation_contract(
        desc="# Task\n\n## 评估协议\nHistory must not cross partitions.\n",
        feedback=["Allow observed prior-partition history"], evaluation_contract=original,
        problem_review=None, protocol_bundle=None, downstream_context=context,
        automl_context_pack=pack, main_task_protocol=protocol,
        repair_context={"original_text": "Full requirements", "file_summaries": [], "relations": []},
    )
    assert "including earlier partitions" in desc
    assert "History must not cross partitions" not in desc
    assert captured["original_text"] == "Full requirements"
    assert captured["evaluation_evidence_pack"] == {"full_evidence": True}
    assert original.model_dump() == repaired.model_dump()
    assert context["evaluation_contract"] == pack["evaluation"] == protocol["evaluation_contract"]
    assert protocol["automl_context_pack"] is pack


def test_rejected_repair_keeps_accepted_objects(tmp_path, monkeypatch):
    import pytest

    module = module_at(tmp_path, contract(passed=False, executable=False))
    monkeypatch.setattr(module, "_build_evaluation_evidence_pack", lambda **kwargs: {})
    original = contract()
    before = original.model_dump()
    pack = {"preserved": True}
    with pytest.raises(RuntimeError, match="repair rejected"):
        module._repair_final_evaluation_contract(
            desc="## 评估协议\nOriginal", feedback=["repair"], evaluation_contract=original,
            problem_review=None, protocol_bundle=None, downstream_context={},
            automl_context_pack=pack, main_task_protocol={},
            repair_context={"original_text": "Full requirements", "file_summaries": [], "relations": []},
        )
    assert original.model_dump() == before
    assert pack == {"preserved": True}


@pytest.mark.parametrize("section, artifact", [
    ("evaluation_contract", ""),
    ("evaluation_contract.validation_protocol", "evaluation_contract"),
    ("evaluation_contract.computation_scope / validation_protocol", "evaluation_contract; description.md; sample_submission_spec"),
    ("validation_protocol", "evaluation_contract; description.md"),
])
def test_auditor_routes_evaluator_subfields_back_to_compiler(tmp_path, monkeypatch, section, artifact):
    module = module_at(tmp_path, contract())
    reviews = iter([
        ArtifactConsistencyReview(passed=False, issues=[ArtifactConsistencyIssue(
            severity="blocking", section=section, artifact=artifact, repair_target="machine_contract",
            message="Threshold fitting scope contradicts the scoring population",
        )]),
        ArtifactConsistencyReview(passed=True),
    ])
    repaired = []
    monkeypatch.setattr(module, "_review_final_artifact_consistency", lambda **kw: next(reviews))
    monkeypatch.setattr(module, "_remember_stage", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_artifact_sanity_check", lambda **kw: [])

    def compile_repair(**kwargs):
        repaired.append(kwargs["feedback"])
        return "## Evaluation\nCorrect contract"

    monkeypatch.setattr(module, "_repair_final_evaluation_contract", compile_repair)
    monkeypatch.setattr(module, "_build_artifact_consistency_patch",
                        lambda **kw: pytest.fail("Machine defect incorrectly routed to prose patcher"))
    desc, audit, defects = module._audit_and_repair_final_artifacts(
        desc="## Evaluation\nConflicting contract", data_root=tmp_path, legacy=None,
        problem_review=None, protocol_bundle=None, evaluation_contract=contract(), sample_spec=None,
        submission_report={}, automl_context_pack={}, main_task_protocol={}, downstream_context={},
        deterministic_defects=[], source_coverage_ledger={}, evaluation_repair_context={},
    )
    assert repaired and "Threshold fitting scope" in repaired[0][0]
    assert audit.passed and not defects and "Correct contract" in desc


def test_auditor_runs_final_review_after_last_configured_repair_pass(tmp_path, monkeypatch):
    module = module_at(tmp_path, contract())
    module.config.prompt.artifact_consistency_max_rounds = 1
    issue = ArtifactConsistencyReview(passed=False, issues=[ArtifactConsistencyIssue(
        issue_id="time", severity="blocking", section="evaluation_contract.validation_protocol",
        artifact="evaluation_contract", repair_target="machine_contract",
        message="Preserve exact cutoff timestamp")])
    reviews = iter([issue, ArtifactConsistencyReview(passed=True)])
    seen_rounds = []
    monkeypatch.setattr(module, "_review_final_artifact_consistency",
                        lambda **kwargs: seen_rounds.append(kwargs["round_idx"]) or next(reviews))
    monkeypatch.setattr(module, "_remember_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_artifact_sanity_check", lambda **kwargs: [])
    monkeypatch.setattr(module, "_repair_final_evaluation_contract", lambda **kwargs: kwargs["desc"])
    monkeypatch.setattr(module, "_build_artifact_consistency_patch",
                        lambda **kwargs: pytest.fail("Machine contract issue reached prose repair"))
    _, audit, defects = module._audit_and_repair_final_artifacts(
        desc="## 评估协议\nExact cutoff", data_root=tmp_path, legacy=None,
        problem_review=None, protocol_bundle=None, evaluation_contract=contract(), sample_spec=None,
        submission_report={}, automl_context_pack={}, main_task_protocol={}, downstream_context={},
        deterministic_defects=[], source_coverage_ledger={}, evaluation_repair_context={},
    )
    assert seen_rounds == [1, 2]
    assert audit.passed and not defects


def test_warning_only_final_review_is_preserved_without_retries(tmp_path, monkeypatch):
    module = module_at(tmp_path, contract())
    module.config.prompt.artifact_consistency_max_rounds = 2
    warning = ArtifactConsistencyReview(
        passed=False,
        issues=[ArtifactConsistencyIssue(
            issue_id="residual-note",
            severity="warning",
            section="data_schema_contract",
            artifact="automl_context_pack",
            message="One low-risk uncertainty remains.",
        )],
    )
    calls = []
    monkeypatch.setattr(
        module,
        "_review_final_artifact_consistency",
        lambda **kwargs: calls.append(kwargs["round_idx"]) or warning,
    )
    monkeypatch.setattr(module, "_remember_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_artifact_sanity_check", lambda **kwargs: [])
    _, audit, defects = module._audit_and_repair_final_artifacts(
        desc="## Task\nCurrent task",
        data_root=tmp_path,
        legacy=None,
        problem_review=None,
        protocol_bundle=None,
        evaluation_contract=contract(),
        sample_spec=None,
        submission_report={},
        automl_context_pack={},
        main_task_protocol={},
        downstream_context={},
        deterministic_defects=[],
        source_coverage_ledger={},
    )
    assert calls == [1]
    assert audit.passed is True
    assert audit.issues[0].severity == "warning"
    assert "warning" in audit.summary
    assert defects == []


def test_consistency_review_payload_excludes_redundant_unrelated_context(tmp_path):
    pack = {
        "data_schema_contract": {"tables": [{"physical_columns_exact": ["SampleTime"]}]},
        "source_alias_guard": [{"alias": "SampleTime"}],
        "evaluation_contract": {"primary_metric": "MAE"},
        "execution_contract": {"contract_id": "task:abc"},
        "data_access": {"large_repeated_notes": "x" * 100_000},
        "relation_cards": [{"large_duplicate": "y" * 100_000}],
    }
    projected = TaskDefinitionModule._artifact_consistency_review_pack(pack)
    assert projected["data_schema_contract"] == pack["data_schema_contract"]
    assert projected["source_alias_guard"] == pack["source_alias_guard"]
    assert "evaluation_contract" not in projected
    assert "execution_contract" not in projected
    assert "data_access" not in projected
    assert "relation_cards" not in projected


def test_sample_requirement_refresh_preserves_other_routing_rules_and_old_note_evidence(tmp_path):
    module = module_at(tmp_path, contract())
    problem = ProblemParadigmReview(requires_sample_submission=True,
        method_routing_notes=["Keep every physical safety constraint", "requires_sample_submission=true because output is a table"])
    bundle = DescriptionProtocolBundle()
    context = {}
    module._sync_sample_requirement(problem, bundle, context, {})
    assert problem.requires_sample_submission is False
    assert bundle.output.sample_submission_required is False
    assert problem.method_routing_notes[0] == "Keep every physical safety constraint"
    assert "requires_sample_submission=false" in problem.method_routing_notes[1]
    assert context["source_diagnostic_history"][0]["previous_notes"] == ["requires_sample_submission=true because output is a table"]


def test_sample_requirement_forces_llm_default_off_without_authoritative_contract(tmp_path):
    module = module_at(tmp_path, contract())
    problem = ProblemParadigmReview(problem_paradigm="ml_dl_prediction", requires_sample_submission=False)
    bundle = DescriptionProtocolBundle()
    spec = SampleSubmissionSpec(should_generate=True, columns=["prediction"])
    context = {"sample_submission_spec": spec.model_dump(), "authoritative_submission_contract": {}}

    module._sync_sample_requirement(problem, bundle, context, {}, sample_spec=spec)

    assert spec.should_generate is False
    assert context["sample_submission_spec"]["should_generate"] is False
    assert context["sample_submission_spec"]["no_sample_submission_reason"]


def test_auditor_repairs_derived_machine_rules_before_prose(tmp_path, monkeypatch):
    module = module_at(tmp_path, contract())
    reviews = iter([
        ArtifactConsistencyReview(passed=False, issues=[ArtifactConsistencyIssue(
            issue_id="rows", severity="blocking", section="row_count_rule; validation_rules",
            artifact="sample_submission_spec", repair_target="machine_contract", message="Separate origins disagree")]),
        ArtifactConsistencyReview(passed=True),
    ])
    called = []
    monkeypatch.setattr(module, "_review_final_artifact_consistency", lambda **kw: next(reviews))
    monkeypatch.setattr(module, "_remember_stage", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_artifact_sanity_check", lambda **kw: [])
    monkeypatch.setattr(module, "_repair_final_derived_contracts", lambda **kw: called.append(kw["review"]))
    monkeypatch.setattr(module, "_build_artifact_consistency_patch", lambda **kw: pytest.fail("Machine rule sent to prose repair"))
    _, audit, _ = module._audit_and_repair_final_artifacts(
        desc="Original", data_root=tmp_path, legacy=None, problem_review=None, protocol_bundle=None,
        evaluation_contract=contract(), sample_spec=None, submission_report={}, automl_context_pack={},
        main_task_protocol={}, downstream_context={}, deterministic_defects=[], source_coverage_ledger={},
        evaluation_repair_context={},
    )
    assert len(called) == 1 and audit.passed


def test_frozen_markdown_mirror_is_routed_to_its_original_section(tmp_path):
    from autorealize.models import ArtifactConsistencyPatch
    module = module_at(tmp_path, ArtifactConsistencyPatch())
    calls = []
    module.services.llm_client.ask_structured = lambda **kw: calls.append(kw) or ArtifactConsistencyPatch()
    issue = ArtifactConsistencyIssue(issue_id="old_prose", repair_target="machine_contract",
        section="main_task_protocol.frozen_description_sections.output.markdown / 输出或提交格式 > 2. 行数",
        artifact="main_task_protocol; description.md", severity="blocking", message="Old union of origins")
    module._build_artifact_consistency_patch(desc="## 输出或提交格式\nOld", review=ArtifactConsistencyReview(issues=[issue]), round_idx=1)
    assert len(calls) == 1
    assert '"section":"输出或提交格式"' in calls[0]["dynamic_user_prompt"]
    assert issue.repair_target == "machine_contract"
    calls.clear()
    issue.section = "main_task_protocol.frozen_description_sections.output.sample_submission_spec"
    module._build_artifact_consistency_patch(desc="Old", review=ArtifactConsistencyReview(issues=[issue]), round_idx=1)
    assert not calls


def test_provider_patch_with_markdown_heading_key_replaces_only_that_section(tmp_path):
    module = module_at(tmp_path, contract())
    desc = "# Task\n\n## 任务定义\nIncorrect boundary\n\n## 评估协议\nKeep evaluator\n"
    patched = module._replace_h2_section(desc, "## 任务定义", "## 任务定义\nAllow observed history\n")
    assert "Incorrect boundary" not in patched
    assert "Allow observed history" in patched
    assert "## 评估协议\nKeep evaluator" in patched
    assert module._replace_h2_section(desc, "### 时间切分", "### 时间切分\nWrong scope") == desc


def test_output_markdown_evidence_routes_to_original_reader_section(tmp_path):
    module = module_at(tmp_path, contract())
    issue = ArtifactConsistencyIssue(issue_id="field-name", repair_target="machine_contract",
        artifact="description_protocol_bundle.output", section="output", severity="blocking",
        message="Output markdown uses a translated key instead of the fixed machine key",
        evidence_refs=["description_protocol_bundle.output.markdown 二、输出 JSON 协议"])
    routed = module._artifact_description_issues(ArtifactConsistencyReview(issues=[issue]))
    assert len(routed) == 1 and routed[0]["section"] == "输出或提交格式"
    assert routed[0]["repair_target"] == "description_section"


def test_invalid_derived_proposal_stops_after_one_paid_attempt(tmp_path, monkeypatch):
    from autorealize.artifact_contract_repair import DerivedContractRepair
    from autorealize.models import SampleSubmissionSpec
    module = module_at(tmp_path, contract())
    sample = SampleSubmissionSpec(validation_rules=["wrong field", "finite output"])
    bundle = DescriptionProtocolBundle()
    patch = dict(path="/sample_submission_spec/validation_rules",
        expected=["wrong field", "finite output"],
        replacement=["exact physical field", "finite output"], issue_ids=["unrelated"],
        evidence_refs=["verified_data_access"],
        reason="Use the verified column name")
    requests = []
    def ask(**kwargs):
        assert sample.validation_rules == ["wrong field", "finite output"]
        requests.append(kwargs)
        return DerivedContractRepair(repairs=[patch])
    module.services.llm_client.ask_structured = ask
    monkeypatch.setattr(module, "_stage_prompt", lambda **kw: ("immutable evidence", "exact fields"))
    monkeypatch.setattr(module, "_write_protocol_artifacts", lambda **kw: None)
    monkeypatch.setattr(module, "_remember_stage", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_write_automl_context_pack", lambda **kw: {})
    monkeypatch.setattr(module, "_write_main_task_protocol", lambda **kw: {})
    with pytest.raises(ValueError, match="Unknown or missing audited issue IDs"):
        module._repair_final_derived_contracts(review=ArtifactConsistencyReview(issues=[ArtifactConsistencyIssue(
            issue_id="source", artifact="sample_submission_spec", section="validation_rules",
            repair_target="machine_contract", severity="blocking", message="Wrong physical field")]),
            evaluation_contract=contract(), problem_review=None, protocol_bundle=bundle, sample_spec=sample,
            downstream_context={}, automl_context_pack={}, main_task_protocol={},
            repair_context={"original_text": "Keep finite output", "file_summaries": [], "relations": []})
    assert len(requests) == 1
    assert sample.validation_rules == ["wrong field", "finite output"]


def test_multi_artifact_boundary_issue_repairs_contract_and_mirrors_in_one_round(tmp_path, monkeypatch):
    from autorealize.models import ArtifactConsistencyPatch

    module = module_at(tmp_path, contract())
    issue = ArtifactConsistencyIssue(issue_id="boundary", severity="blocking", repair_target="machine_contract",
        section="evaluation_contract", artifact="evaluation_contract、description_protocol_bundle、description.md",
        message="Origin event belongs to target, but a mirror also uses it as a feature",
        evidence_refs=["description.md::任务定义/3. 聚合与边界", "description.md::约束与防泄漏/2. 边界",
                       "description.md::unrecognized/section", "evaluation_contract.prediction_unit",
                       "description_protocol_bundle.ml_dl.feature_boundary[0]"])
    reviews = iter([ArtifactConsistencyReview(passed=False, issues=[issue]), ArtifactConsistencyReview(passed=True)])
    calls = []
    monkeypatch.setattr(module, "_review_final_artifact_consistency", lambda **kw: next(reviews))
    monkeypatch.setattr(module, "_remember_stage", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_artifact_sanity_check", lambda **kw: [])
    monkeypatch.setattr(module, "_repair_final_evaluation_contract",
        lambda **kw: calls.append("evaluation") or kw["desc"])
    monkeypatch.setattr(module, "_repair_final_derived_contracts", lambda **kw: calls.append("derived"))
    def patch(**kw):
        calls.append("prose")
        assert {item["section"] for item in module._artifact_description_issues(kw["review"])} == {"任务定义", "约束与防泄漏"}
        return ArtifactConsistencyPatch(revised_sections={"任务定义": "## 任务定义\nEvents strictly before o"})
    monkeypatch.setattr(module, "_build_artifact_consistency_patch", patch)
    legacy = SimpleNamespace(_find_missing_file_references=lambda *a: [], _enforce_existing_file_references=lambda desc, root: desc)
    desc, audit, defects = module._audit_and_repair_final_artifacts(
        desc="## 任务定义\nIncorrect\n\n## 评估协议\nKeep metric", data_root=tmp_path, legacy=legacy,
        problem_review=None, protocol_bundle=None, evaluation_contract=contract(), sample_spec=None,
        submission_report={}, automl_context_pack={}, main_task_protocol={}, downstream_context={},
        deterministic_defects=[], source_coverage_ledger={}, evaluation_repair_context={},
    )
    assert calls == ["evaluation", "derived", "prose"]
    assert audit.passed and not defects and "Events strictly before o" in desc and "Keep metric" in desc


def test_source_refresh_archives_failed_sample_attempt_without_claiming_regeneration(tmp_path, monkeypatch):
    module = module_at(tmp_path, contract())
    monkeypatch.setattr(module, "_write_protocol_artifacts", lambda **kwargs: None)
    spec = SampleSubmissionSpec(columns=["probability"], source_fields={"probability": "data.csv::temperature"})
    fact = "data.csv: verified_row_count=10000, row_count_basis=worksheet_used_range; exclude IDs."
    context = {"sample_submission_spec": spec.model_dump(), "description_sections": {
        "output": {"sample_submission_spec": {"source_fields": {"probability": "stale"}}},
        "data": {"facts_used": [fact]},
    }, "sample_submission_generation_issues": ["Old mapping failed"]}
    failed = {"source": "sample_spec_generation_failed", "passed": False,
              "issues": ["Earlier source mapping was wrong"]}
    (tmp_path / "submission_report.json").write_text(json.dumps(failed), encoding="utf-8")
    module.refresh_output_source_diagnostics(
        desc="## 输出或提交格式\nDerived probability from data.csv::temperature\n",
        sample_spec=spec, protocol_bundle=DescriptionProtocolBundle(), problem_review=ProblemParadigmReview(),
        downstream_context=context, file_summaries=[FileSummary(
            path="data.csv", file_type="csv", role="raw_data_table", summary="Sensor table", columns=["temperature"],
            source_metadata={"shape": [10000, 1], "profile_sampling": {"sampled": True}},
        )],
    )
    current = json.loads((tmp_path / "submission_report.json").read_text(encoding="utf-8"))
    history = json.loads((tmp_path / "submission_diagnostic_history.json").read_text(encoding="utf-8"))
    assert history[0]["report"] == failed
    assert current["source"] == "not_regenerated_after_schema_repair"
    assert current["current_source_mapping_checked"] is True
    assert current["passed"] is False
    assert current["sample_submission_available"] is False
    assert current["final_prediction_validated"] is False
    assert current["issues"] == []
    assert context["sample_submission_status"] == current
    assert context["sample_submission_generation_issues"] == []
    assert context["sample_submission_generation_status"] == current["source"]
    assert context["description_sections"]["data"]["facts_used"] == [
        "data.csv: verified_row_count=unverified, row_count_basis=reader_shape_unverified; exclude IDs."]
    module._sync_final_description_sections("## 输出或提交格式\nCurrent output", context)
    assert context["description_sections"]["output"]["sample_submission_spec"] == spec.model_dump()


def test_audit_keeps_unchanged_contracts_before_description_and_round(tmp_path, monkeypatch):
    from autorealize.models import ArtifactConsistencyReview, SampleSubmissionSpec
    from autorealize.modules import task_definition

    module = module_at(tmp_path, ArtifactConsistencyReview(passed=True))
    requests = []
    original_review_prompt_parts = task_definition.review_prompt_parts

    def capture(**kwargs):
        requests.append(kwargs)
        return original_review_prompt_parts(**kwargs)

    monkeypatch.setattr(task_definition, "review_prompt_parts", capture)
    for round_idx in (1, 2):
        module._review_final_artifact_consistency(
            desc=f"Current complete description {round_idx}", problem_review=ProblemParadigmReview(),
            protocol_bundle=DescriptionProtocolBundle(), evaluation_contract=contract(),
            sample_spec=SampleSubmissionSpec(), submission_report={}, automl_context_pack={"full_schema": True},
            main_task_protocol={}, downstream_context={}, deterministic_defects=[], round_idx=round_idx,
        )
    assert requests[0]["dynamic_title"] == requests[1]["dynamic_title"]
    payload = requests[0]["latest_request"]
    keys = list(payload)
    assert keys.index("evaluation_contract") < keys.index("automl_context_pack") < keys.index("description_markdown") < keys.index("round")
    assert payload["evaluation_contract"] == {
        key: value for key, value in contract().model_dump().items()
        if key in {
            "passed", "executable", "authority_status", "ranking_scope",
            "score_representation", "primary_metric", "metric_direction",
            "metric_formula", "scalar_score_formula", "final_evaluation_metric",
            "final_evaluation_direction", "final_evaluation_formula",
            "final_evaluation_protocol", "prediction_unit", "y_true_source",
            "y_pred_source", "computation_scope", "aggregation_rule",
            "validation_protocol", "submission_checks", "leakage_guards",
            "invalid_solution_rules", "tie_break_rules", "assumptions",
            "residual_issues",
        }
    }
    assert requests[1]["latest_request"]["description_markdown"] == "Current complete description 2"


def test_question_repair_requires_exact_current_text_and_known_evidence(tmp_path, monkeypatch):
    import pytest
    from autorealize.models import ArtifactConsistencyReview, OpenQuestionRepair, OpenQuestionResolution

    original = "Split and measured scores are missing"
    response = OpenQuestionRepair(resolutions=[OpenQuestionResolution(
        original_text=original, remaining_questions=["Measured scores are missing"],
        evidence_refs=["qdi:q2"], reason="The script established the split only",
    )])
    module = module_at(tmp_path, response)
    context = {"agent_context_pack": {"question_memory": {"answers": [
        {"question_id": "q2", "answer": "Actual split IDs", "evidence": ["real-script-result"]},
    ]}}}
    pack = {"execution_contract": {"open_questions": [original]}}
    monkeypatch.setattr(module, "_write_automl_context_pack", lambda **kwargs: {"preserved": True})
    monkeypatch.setattr(module, "_write_main_task_protocol", lambda **kwargs: {"preserved": True})
    kwargs = dict(review=ArtifactConsistencyReview(), evaluation_contract=contract(),
                  problem_review=ProblemParadigmReview(), protocol_bundle=DescriptionProtocolBundle(),
                  downstream_context=context, automl_context_pack=pack, main_task_protocol={},
                  repair_context={"file_summaries": [], "relations": []})
    response.resolutions[0].evidence_refs = ["invented-result"]
    with pytest.raises(RuntimeError, match="visible supporting evidence"):
        module._repair_final_open_questions(**kwargs)
    assert "open_question_resolutions" not in context
    response.resolutions[0].evidence_refs = ["qdi:q2"]
    module._repair_final_open_questions(**kwargs)
    assert context["open_question_resolutions"][0]["remaining_questions"] == ["Measured scores are missing"]
    assert context["agent_context_pack"]["question_memory"]["answers"][0]["answer"] == "Actual split IDs"
