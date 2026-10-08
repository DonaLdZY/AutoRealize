import json
from copy import deepcopy

import pytest

from autorealize.prompt_cache import lossless_json
from autorealize.review_prompt import decode_review_snapshot, review_prompt_parts


def test_document_terms_are_not_exposed_as_physical_input_columns():
    from autorealize.models import DescriptionProtocolBundle, FileSummary
    from autorealize.report_writer import build_automl_context_pack
    files = [FileSummary(path="requirements.pdf", role="task_requirement", summary="Predict target", columns=["target"]),
             FileSummary(path="raw.csv", role="raw_data_table", summary="Raw data", columns=["feature"])]
    cards = [{"source_file": "requirements.pdf", "table_kind": "document", "role": "task_requirement",
              "field_hints": ["target"]},
             {"source_file": "raw.csv", "table_kind": "csv_table", "role": "raw_data_table",
              "field_hints": ["feature"]}]
    pack = build_automl_context_pack(DescriptionProtocolBundle(), file_summaries=files,
                                    compiled_context={"table_cards": cards})
    assert pack.data_access[0]["columns_scope"] == "document_terms"
    assert pack.data_access[0]["columns"] == ["target"]
    assert pack.data_access[1]["columns_scope"] == "physical_schema"


def decode(stable):
    return decode_review_snapshot(stable.split("\n", 1)[1])


def test_review_deduplicates_across_authority_and_artifacts_with_exact_roundtrip():
    rule = {"constraint": "Do not use holdout for selection. " * 30, "date": None, "count": 0}
    authority = {"a/~": rule}
    evidence = {"same_with_different_key_order": dict(reversed(list(rule.items())))}
    current = {"evaluation": rule, "description": "ALL source rules\n" + "tail" * 2000,
               "round": 1, "deterministic_defects": []}
    before = deepcopy((authority, evidence, current))
    stable, dynamic = review_prompt_parts(authority=authority, stage_evidence=evidence,
                                         latest_request=current, dynamic_title="Current feedback")
    decoded = decode(stable)
    assert decoded == {"authority": authority, "stage_evidence": evidence,
                       "latest_request": {k: v for k, v in current.items() if k not in {"round", "deterministic_defects"}}}
    assert json.loads(dynamic.split("\n", 1)[1]) == {"round": 1, "deterministic_defects": []}
    assert (authority, evidence, current) == before
    assert "$prompt_ref" in stable and "_prompt_truncation" not in stable


def test_changing_only_feedback_preserves_provider_prefix():
    kwargs = dict(authority={"constraints": ["source rule"]}, stage_evidence={"rule": "review all artifacts"},
                  dynamic_title="feedback")
    a = review_prompt_parts(**kwargs, latest_request={"contract": {"score": "MAE"}, "round": 1})
    b = review_prompt_parts(**kwargs, latest_request={"contract": {"score": "MAE"}, "round": 2,
                                                      "deterministic_defects": ["changed output column"]})
    assert a[0] == b[0]
    assert a[1] != b[1]


def test_similar_but_different_contracts_and_numeric_types_are_preserved():
    rule = "required " * 150
    values = [{"rule": rule, "direction": "minimize", "v": False},
              {"rule": rule, "direction": "maximize", "v": 0},
              {"rule": rule, "direction": "minimize", "v": 0.0}]
    stable, _ = review_prompt_parts(authority={}, stage_evidence={}, latest_request={"contracts": values}, dynamic_title="feedback")
    restored = decode(stable)["latest_request"]["contracts"]
    assert json.dumps(restored) == json.dumps(values)


@pytest.mark.parametrize("marker", ["$prompt_ref", "$same_as", "$review_text"])
def test_source_reference_markers_are_not_interpreted_as_generated_aliases(marker):
    payload = {marker: "/source/reference", "rule": ["keep every word " * 1000] * 2}
    stable, _ = review_prompt_parts(authority={}, stage_evidence={}, latest_request=payload, dynamic_title="feedback")
    assert decode(stable)["latest_request"] == payload


def test_repeated_markdown_sections_and_code_roundtrip_without_lost_whitespace():
    body = "Preserve all restrictions. " * 80
    code = "\r\n```python\r\nfor i in range(3):\r\n    print(i)\r\n```\r\n"
    source = {"section": body + code, "description": "# Full task\n\n## Output\n" + body + code}
    stable, _ = review_prompt_parts(authority={}, stage_evidence={}, latest_request=source, dynamic_title="feedback")
    assert decode(stable)["latest_request"] == source
    assert "$review_text" in stable
    assert "_prompt_truncation" not in stable


def test_canonical_matching_saves_more_than_existing_factoring():
    a = {"a": "source constraints " * 400, "b": "source formulas " * 400}
    b = dict(reversed(list(a.items())))
    source = {"authority": a, "artifacts": [b, a]}
    old = lossless_json(source, sort_keys=False)
    new = lossless_json(source, sort_keys=False, canonical_matches=True, min_shared_chars=160)
    assert len(new) < len(old)
    assert decode_review_snapshot(new) == source


def test_gate_uses_complete_evidence_without_memory_llm_calls(tmp_path):
    from types import SimpleNamespace
    from autorealize.config import AutoRealizeConfig
    from autorealize.context_memory import CrossStageContextLedger
    from autorealize.modules.task_definition import TaskDefinitionModule
    from autorealize.modules.types import RuntimeServices

    def fail(**kwargs):
        raise AssertionError("Paid history retrieval must not run for a complete review snapshot")

    cfg = AutoRealizeConfig()
    cfg.prompt.lossless_review_prompts = True
    services = RuntimeServices(SimpleNamespace(ask_structured=fail), SimpleNamespace(load=lambda *a: "rules"), None, None)
    module = TaskDefinitionModule(cfg, services, tmp_path, tmp_path)
    ledger = CrossStageContextLedger(config=cfg, llm_client=services.llm_client, prompt_mgr=services.prompt_mgr,
                                    report_dir=tmp_path, stable_context={"original_requirements_full": "All original requirements"})
    ledger.add("obsolete_draft", {"old": "old draft " * 1000})
    ledger.compaction_count = 1
    module._cross_stage_context = ledger
    for stage in ("artifact_consistency_reviewer", "evaluation_contract_finalizer_3"):
        stable, _ = module._stage_prompt(stage=stage, stable={"evidence": "complete current evidence"},
                                        dynamic={"previous_contract": "full rules " * 5000},
                                        stable_title="evidence", dynamic_title="feedback", dynamic_limit=200)
        assert decode(stable)["latest_request"]["previous_contract"] == "full rules " * 5000
        assert "old draft" not in stable


def test_review_requires_all_coverage_dimensions_and_blocks_inconsistent_verdict():
    from autorealize.review_prompt import ArtifactConsistencyAudit, ReviewCoverage
    coverage = {key: {"status": "consistent", "evidence_refs": ["evaluation_contract"],
                      "reasoning": "The source and artifact agree."}
                for key in ReviewCoverage.model_fields}
    result = ArtifactConsistencyAudit(passed=True, issues=[], coverage=coverage)
    assert len(result.coverage.model_dump()) == 8
    missing = deepcopy(coverage)
    missing.pop("population_arithmetic")
    with pytest.raises(ValueError):
        ArtifactConsistencyAudit(passed=True, coverage=missing)
    broken = deepcopy(coverage)
    broken["population_arithmetic"]["status"] = "conflict"
    with pytest.raises(ValueError, match="locatable issue"):
        ArtifactConsistencyAudit(passed=True, issues=[], coverage=broken)
    with pytest.raises(ValueError, match="coverage conflict"):
        ArtifactConsistencyAudit(passed=True, issues=[{"severity": "warning"}], coverage=broken)
    with pytest.raises(ValueError, match="blocking issue"):
        ArtifactConsistencyAudit(passed=True, issues=[{"severity": "blocking"}], coverage=coverage)


def test_optimized_production_reviewer_uses_full_output_budget(tmp_path):
    from types import SimpleNamespace
    from autorealize.config import AutoRealizeConfig
    from autorealize.models import DescriptionProtocolBundle, EvaluationContractReview, ProblemParadigmReview, SampleSubmissionSpec
    from autorealize.modules.task_definition import TaskDefinitionModule
    from autorealize.modules.types import RuntimeServices
    from autorealize.review_prompt import ArtifactConsistencyAudit
    requests = []
    cfg = AutoRealizeConfig()
    cfg.prompt.lossless_review_prompts = True
    services = RuntimeServices(SimpleNamespace(ask_structured=lambda **kw: requests.append(kw)),
                               SimpleNamespace(load=lambda *a: "Review instructions"), None, None)
    module = TaskDefinitionModule(cfg, services, tmp_path, tmp_path)
    module._review_final_artifact_consistency(
        desc="Every section remains visible", problem_review=ProblemParadigmReview(),
        protocol_bundle=DescriptionProtocolBundle(), evaluation_contract=EvaluationContractReview(),
        sample_spec=SampleSubmissionSpec(), submission_report={}, automl_context_pack={},
        main_task_protocol={}, downstream_context={}, deterministic_defects=[], round_idx=1)
    assert requests[0]["model_cls"] is ArtifactConsistencyAudit
    assert requests[0]["max_tokens"] == 32768
    assert decode(requests[0]["static_context_prompt"])["latest_request"]["description_markdown"] == "Every section remains visible"


@pytest.mark.parametrize("optimized", [False, True])
def test_evaluation_rollout_changes_only_validated_finalizer(tmp_path, optimized):
    from types import SimpleNamespace
    from autorealize.config import AutoRealizeConfig
    from autorealize.models import EvaluationContractReview
    from autorealize.modules.task_definition import TaskDefinitionModule
    from autorealize.modules.types import RuntimeServices
    from autorealize.review_prompt import EVALUATION_REVIEW_INSTRUCTIONS

    requests = []

    class Captured(Exception):
        pass

    def ask(**kwargs):
        requests.append(kwargs)
        raise Captured()

    cfg = AutoRealizeConfig()
    cfg.prompt.lossless_review_prompts = optimized
    services = RuntimeServices(SimpleNamespace(ask_structured=ask),
                               SimpleNamespace(load=lambda *a: "Review instructions"), None, None)
    module = TaskDefinitionModule(cfg, services, tmp_path, tmp_path)
    previous = EvaluationContractReview(primary_metric="development_mae", metric_direction="minimize")
    for rounds, stage in ((3, "evaluation_contract_reviewer_1"), (1, "evaluation_contract_finalizer_1")):
        cfg.prompt.evaluation_contract_max_rounds = rounds
        with pytest.raises(Captured):
            module._review_evaluation_contract(
                original_text="Preserve final holdout isolation", downstream_context={},
                evaluation_evidence_pack={"exact_columns": ["target", "prediction"]}, previous_contract=previous)
        request = requests[-1]
        assert request["prompt_name"] == stage
        if optimized and rounds == 1:
            assert EVALUATION_REVIEW_INSTRUCTIONS in request["system_prompt"]
            restored = decode(request["static_context_prompt"])
            assert restored["latest_request"]["previous_contract"] == previous.model_dump()
            assert restored["stage_evidence"]["evaluation_evidence_pack"] == {"exact_columns": ["target", "prediction"]}
        else:
            assert request["system_prompt"] == "Review instructions"
            assert "previous_contract" in request["dynamic_user_prompt"]
            assert "previous_contract" not in request["static_context_prompt"]
        if rounds == 1:
            assert "finalizer_instruction" in json.loads(request["dynamic_user_prompt"].split("\n", 1)[1])
