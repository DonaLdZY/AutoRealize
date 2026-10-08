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

    module = module_at(tmp_path, ArtifactConsistencyReview(passed=True))
    requests = []
    def capture(**kwargs):
        requests.append(kwargs)
        return "stable", "dynamic"

    monkeypatch.setattr(module, "_stage_prompt", capture)
    for round_idx in (1, 2):
        module._review_final_artifact_consistency(
            desc=f"Current complete description {round_idx}", problem_review=ProblemParadigmReview(),
            protocol_bundle=DescriptionProtocolBundle(), evaluation_contract=contract(),
            sample_spec=SampleSubmissionSpec(), submission_report={}, automl_context_pack={"full_schema": True},
            main_task_protocol={}, downstream_context={}, deterministic_defects=[], round_idx=round_idx,
        )
    assert requests[0]["stage"] == requests[1]["stage"]
    assert requests[0]["complete_current_snapshot"] is True
    payload = requests[0]["dynamic"]
    keys = list(payload)
    assert keys.index("evaluation_contract") < keys.index("automl_context_pack") < keys.index("description_markdown") < keys.index("round")
    assert payload["evaluation_contract"] == contract().model_dump()
    assert requests[1]["dynamic"]["description_markdown"] == "Current complete description 2"


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
