from __future__ import annotations

from pathlib import Path
import json

import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.context_compiler import ArtifactStore
from autorealize.investigation import (
    _qdi_answerer_stable_prefix,
    _qdi_compact_constraint_memory,
    _qdi_compact_coverage,
    _qdi_project_cache_context,
    _qdi_table_index_card,
    _qdi_question_cache_context,
    _run_question_investigator_inner,
    _artifact_ids_in,
    _bounded_json_view,
    _empty_working_memory_card,
    _merge_working_memory_card,
    _new_live_action_entry,
    _recent_action_window,
    _working_memory_prompt_view,
)
from autorealize.models import (
    ContextRetrievalRequest,
    InvestigationQuestion,
    QDIActionDigest,
    QDIWorkingMemoryUpdate,
    QuestionInvestigationAction,
    QuestionInvestigationPlan,
    ReadonlyPythonRequest,
)


class _PromptManager:
    def load(self, _name: str) -> str:
        return "fixed system prompt"


class _TwoRoundLLM:
    def __init__(self, recommended_actions: int = 0) -> None:
        self.calls: list[dict] = []
        self.action_calls = 0
        self.recommended_actions = recommended_actions

    def ask_structured(self, *, model_cls, **kwargs):
        self.calls.append({"model_cls": model_cls, **kwargs})
        if model_cls is QuestionInvestigationPlan:
            return QuestionInvestigationPlan(
                recommended_max_actions_per_question=self.recommended_actions,
                questions=[
                    InvestigationQuestion(
                        question_id="q1",
                        question="Which table defines the output key?",
                        category="output",
                    )
                ]
            )
        self.action_calls += 1
        if self.action_calls == 1:
            return QuestionInvestigationAction(
                action="request_context",
                request_context=ContextRetrievalRequest(question_id="q1", query="output key"),
                working_memory_update=QDIWorkingMemoryUpdate(open_gaps=["output key is not confirmed"]),
            )
        return QuestionInvestigationAction(
            action="answer",
            answer="No table card establishes an output key.",
            confidence="low",
            evidence=["action:1/context_retrieved"],
            action_digest_updates=[
                QDIActionDigest(
                    sequence=1,
                    action="request_context",
                    what_was_done="Requested table-card details related to the output key.",
                    key_outputs=["No matching table card established the output key."],
                    temporary_conclusion="The output key remains unspecified in table-card evidence.",
                    remaining_gap="A requirement document may still define it.",
                    evidence_refs=["action:1/context_retrieved"],
                )
            ],
            working_memory_update=QDIWorkingMemoryUpdate(
                confirmed_facts=["The retrieved table-card set did not establish an output key."],
                evidence_refs=["action:1/context_retrieved"],
            ),
        )


class _CaptureReadyLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ask_structured(self, *, model_cls, **kwargs):
        self.calls.append({"model_cls": model_cls, **kwargs})
        return QuestionInvestigationPlan(ready_to_answer=True)


def test_recent_action_window_keeps_only_latest_three() -> None:
    actions = [{"sequence": index, "request": f"request-{index}"} for index in range(1, 5)]

    window = _recent_action_window(actions, count=3)

    assert [item["sequence"] for item in window] == [2, 3, 4]
    assert actions[0]["sequence"] == 1


def test_working_memory_merges_incrementally_without_an_extra_llm_call() -> None:
    card = _empty_working_memory_card("q1")
    first = QDIWorkingMemoryUpdate(
        confirmed_facts=["orders.csv contains 10 rows"],
        evidence_refs=["action:1/result.rows"],
        open_gaps=["join coverage is unknown"],
    )
    second = QDIWorkingMemoryUpdate(
        confirmed_facts=["orders.csv contains 10 rows"],
        temporary_conclusions=["carrier_id may be a join key"],
        invalidated_hypotheses=["order_id is not a carrier key"],
        recommended_next_focus="measure carrier_id coverage",
    )

    card = _merge_working_memory_card(card, first, sequence=1, max_chars=4000)
    card = _merge_working_memory_card(card, second, sequence=2, max_chars=4000)

    assert card["confirmed_facts"] == ["orders.csv contains 10 rows"]
    assert card["temporary_conclusions"] == ["carrier_id may be a join key"]
    assert card["invalidated_hypotheses"] == ["order_id is not a carrier key"]
    assert card["last_updated_sequence"] == 2
    assert "not an authority layer" in card["trust_policy"]


def test_evidence_linked_resolution_retires_only_the_exact_old_interpretation():
    card = _empty_working_memory_card("q1")
    card["open_gaps"] = ["timestamps not checked", "label delay not checked"]
    update = QDIWorkingMemoryUpdate(
        confirmed_facts=["full scan verified all timestamps"], evidence_refs=["action:2/full_scan"],
        resolutions=[{"field": "open_gaps", "text": "timestamps not checked",
                      "evidence_refs": ["action:2/full_scan"], "reason": "Full scan completed"}],
    )
    result = _merge_working_memory_card(card, update, sequence=2, max_chars=4000)
    assert result["open_gaps"] == ["label delay not checked"]
    assert card["open_gaps"] == ["timestamps not checked", "label delay not checked"]
    assert result["confirmed_facts"] == ["full scan verified all timestamps"]


def test_resolution_without_known_evidence_or_exact_match_cannot_erase_gaps():
    card = _empty_working_memory_card("q1")
    card["open_gaps"] = ["timestamps not checked"]
    update = QDIWorkingMemoryUpdate(
        evidence_refs=["action:2/full_scan"],
        resolutions=[
            {"field": "open_gaps", "text": "timestamps not checked",
             "evidence_refs": ["invented"], "reason": "claim"},
            {"field": "open_gaps", "text": "timestamps",
             "evidence_refs": ["action:2/full_scan"], "reason": "partial match"},
        ],
    )
    result = _merge_working_memory_card(card, update, sequence=2, max_chars=4000)
    assert result["open_gaps"] == ["timestamps not checked"]


def test_long_recent_script_is_artifact_backed_and_explicitly_truncated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    action = QuestionInvestigationAction(
        action="request_script",
        request_script=ReadonlyPythonRequest(
            question_id="q1",
            python_code="x" * 500,
        ),
    )

    live = _new_live_action_entry(
        sequence=1,
        question_id="q1",
        action_name="request_script",
        action=action,
        artifact_store=store,
        script_chars=100,
    )

    request = live["request"]
    assert len(request["python_code"]) == 100
    assert request["python_code_truncated"] is True
    assert request["python_code_original_chars"] == 500
    artifact_ids = _artifact_ids_in(live)
    assert len(artifact_ids) == 1


def test_qdi_artifact_read_is_bounded_and_rejects_other_artifact_types(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    allowed = store.put("qdi_script_output_full", "q1:r1", {"rows": list(range(30))})
    rejected = store.put("table_card_full", "orders.csv", {"secret": "not qdi"})

    first = store.read_excerpt(allowed["artifact_id"], max_chars=40, json_path="rows")
    second = store.read_excerpt(
        allowed["artifact_id"],
        offset=first["next_offset"],
        max_chars=40,
        json_path="rows",
    )
    denied = store.read_excerpt(rejected["artifact_id"], max_chars=40)
    traversal = store.read_excerpt("../outside", max_chars=40)

    assert first["status"] == "completed"
    assert first["has_more"] is True
    assert second["offset"] == first["next_offset"]
    assert denied["error"] == "artifact_type_not_allowed"
    assert traversal["error"] == "invalid_artifact_id"


def test_bounded_view_reports_exact_truncation_lengths() -> None:
    view = _bounded_json_view({"payload": "z" * 500}, 80)

    assert view["truncated"] is True
    assert view["visible_chars"] == 80
    assert view["original_chars"] > view["visible_chars"]


def test_stable_prefix_shares_planner_routing_key_and_preserves_changed_rules() -> None:
    from autorealize.llm.client import _provider_prompt_cache_key
    from autorealize.prompt_cache import stable_dynamic_prompt

    context = {"table_cards": [{"table_id": "orders"}], "authoritative_memory": {"capacity": 51}}
    planner, _ = stable_dynamic_prompt(stable=context, dynamic={"max_questions": 6})
    answerer = _qdi_answerer_stable_prefix(context)
    assert planner != answerer
    assert '"table_cards"' in answerer
    assert '"table_index"' in answerer
    cfg = AutoRealizeConfig()
    key_args = {"config": cfg, "base_url": "https://api.openai.com/v1"}
    assert _provider_prompt_cache_key(
        stable_prefix=planner, cache_scope="qdi:project-prefix:v6", **key_args
    ) == _provider_prompt_cache_key(
        stable_prefix=answerer, cache_scope="qdi:project-prefix:v6", **key_args
    )
    changed = _qdi_answerer_stable_prefix({**context, "authoritative_memory": {"capacity": 50}})
    assert _provider_prompt_cache_key(
        stable_prefix=changed, cache_scope="qdi:project-prefix:v6", **key_args
    ) == _provider_prompt_cache_key(
        stable_prefix=answerer, cache_scope="qdi:project-prefix:v6", **key_args
    )
    assert _provider_prompt_cache_key(
        stable_prefix=answerer, cache_scope="qdi:answerer:v6", **key_args
    ) != _provider_prompt_cache_key(
        stable_prefix=answerer, cache_scope="qdi:planner:v6", **key_args
    )


def test_qdi_question_segment_is_route_only_and_does_not_duplicate_table_cards() -> None:
    context = {
        "table_cards": [
            {"table_id": "orders", "source_file": "orders.csv"},
            {"table_id": "customers", "source_file": "customers.csv"},
        ],
        "authoritative_memory": {"capacity": 51},
    }
    question = {
        "question_id": "q1",
        "question": "Which table defines the output key?",
        "candidate_files": ["orders.csv"],
    }
    segment = _qdi_question_cache_context(context, question, context)
    assert "table_cards" not in segment
    assert "table_index" not in segment
    assert segment["question_route"]["candidate_table_ids"] == ["orders"]
    assert segment["question_route"]["all_table_details_are_in_project_prefix"] is True


def test_project_prefix_excludes_task_and_prior_run_state() -> None:
    base = {
        "task_hint": "predict Al(N)",
        "data_root_name": "chemical",
        "authoritative_memory": {"task_goal": "predict target"},
        "table_cards": [{"table_id": "process", "source_file": "process.csv"}],
        "prior_qdi_memory": {
            "answers": [{"question_id": "old", "answer": "old answer"}],
            "source_delta": {"changed_files": ["process.csv"]},
        },
    }
    changed = {
        **base,
        "task_hint": "recommend steam flow",
        "prior_qdi_memory": {
            "answers": [{"question_id": "new", "answer": "new answer"}],
            "source_delta": {"new_files": ["new.csv"]},
        },
    }
    assert _qdi_project_cache_context(base) == _qdi_project_cache_context(changed)


def test_compact_project_routes_every_table_but_coverage_is_exception_first() -> None:
    card = {
        "table_id": "wide",
        "source_file": "wide.csv",
        "shape": [10, 40],
        "fields": [{"name": f"field_{i}", "role": "other"} for i in range(40)],
    }
    index = _qdi_table_index_card(card)
    assert index["table_id"] == "wide"
    assert len(index["field_names"]) == 16
    assert index["field_names_omitted_count"] == 24

    coverage = _qdi_compact_coverage(
        {
            "schema_version": "v1",
            "entries": [
                {"table_id": "ok", "coverage_status": "required", "verified_row_count": 10},
                {"table_id": "bad", "coverage_status": "required", "row_count_conflict": True},
            ],
            "counts": {"required": 2},
            "original_table_count": 2,
            "visible_entry_count": 2,
        }
    )
    assert [item["table_id"] for item in coverage["entries"]] == ["bad", "ok"]
    assert coverage["entry_policy"]


def test_constraint_projection_prioritizes_hard_items_and_counts_omissions() -> None:
    memory = {
        "summary": "constraints",
        "items": [
            {"name": f"soft-{i}", "priority": "low", "strength": "soft", "description": "x"}
            for i in range(20)
        ] + [
            {"name": "hard-rule", "priority": "critical", "strength": "hard", "description": "never leak target"}
        ],
    }
    compact = _qdi_compact_constraint_memory(memory)
    assert compact["items"][0]["name"] == "hard-rule"
    assert compact["items_omitted_count"] == 5


def test_prior_qdi_memory_exposes_source_status_and_incremental_delta_to_planner(tmp_path: Path):
    data_root = tmp_path / "input"
    data_root.mkdir()
    llm = _CaptureReadyLLM()
    cfg = AutoRealizeConfig.from_env()
    prior = {
        "answers": [{
            "question_id": "old-q1",
            "question": "Which file defines the target?",
            "answer": "target.csv",
            "confidence": "high",
            "source_status": "changed",
            "changed_source_files": ["target.csv"],
            "task_scope_status": "different",
            "source_task_hint": "predict a different target",
        }],
        "source_delta": {
            "unchanged_files": ["history.csv"],
            "new_files": ["new_batch.csv"],
            "changed_files": ["target.csv"],
            "removed_files": ["old.csv"],
        },
    }
    _run_question_investigator_inner(
        cfg=cfg,
        llm_client=llm,
        prompt_mgr=_PromptManager(),
        data_root=data_root,
        task_hint="predict target",
        file_summaries=[],
        relation_hints=[],
        constraint_memory={},
        authoritative_memory={},
        knowledge_base={},
        prior_question_memory=prior,
        report_dir=tmp_path / "report",
    )
    planner = next(call for call in llm.calls if call["model_cls"] is QuestionInvestigationPlan)
    dynamic = planner["dynamic_user_prompt"]
    combined = planner["static_context_prompt"] + dynamic
    assert "source_status" in dynamic
    assert "source_inventory_status" in dynamic
    assert "qdi_contract_status" in dynamic
    assert "task_scope_status" in dynamic
    assert "predict a different target" in dynamic
    assert "changed_source_files" in dynamic
    assert "new_batch.csv" in dynamic
    assert "old.csv" in dynamic
    assert "只能作为假设" in combined or "hypotheses" in combined


@pytest.mark.parametrize("recommended_actions", [0, 1, 2])
def test_full_qdi_loop_reuses_prefix_and_updates_memory_without_summary_call(
    tmp_path: Path, recommended_actions: int
) -> None:
    data_root = tmp_path / "input"
    report_dir = tmp_path / "report"
    data_root.mkdir()
    cfg = AutoRealizeConfig.from_env()
    cfg.investigation.max_questions = 1
    cfg.investigation.max_rounds_per_run = 2
    cfg.investigation.max_scripts_per_question = 0
    cfg.investigation.adaptive_budgeting = True
    llm = _TwoRoundLLM(recommended_actions=recommended_actions)

    report = _run_question_investigator_inner(
        cfg=cfg,
        llm_client=llm,
        prompt_mgr=_PromptManager(),
        data_root=data_root,
        task_hint="test",
        file_summaries=[],
        relation_hints=[],
        constraint_memory={},
        authoritative_memory={},
        knowledge_base={},
        report_dir=report_dir,
    )

    answerer_calls = [call for call in llm.calls if call["model_cls"] is QuestionInvestigationAction]
    assert len(llm.calls) == 3
    assert len(answerer_calls) == 2
    assert answerer_calls[0]["prompt_name"] == "question_investigator_action"
    assert answerer_calls[0]["static_context_prompt"] == answerer_calls[1]["static_context_prompt"]
    assert answerer_calls[0]["static_context_prompt"] == llm.calls[0]["static_context_prompt"]
    assert '"initial_question"' in answerer_calls[0]["dynamic_user_prompt"]
    assert "Which table defines the output key?" in answerer_calls[0]["dynamic_user_prompt"]
    assert "route-only manifests" not in answerer_calls[0]["dynamic_user_prompt"]
    assert "context_retrieved" in answerer_calls[1]["dynamic_user_prompt"]
    assert "pending_action_digest_requests" in answerer_calls[1]["dynamic_user_prompt"]
    assert "output key is not confirmed" in answerer_calls[1]["dynamic_user_prompt"]
    assert report["answers"][0]["answer"] == "No table card establishes an output key."
    assert report["action_digest_cards"][0]["digest_source"] == "llm_action_digest"
    assert report["action_digest_cards"][0]["digest_for_sequence"] == 1
    assert "No matching table card" in report["action_digest_cards"][0]["key_outputs"][0]
    assert report["working_memory_cards"][0]["last_updated_sequence"] == 2


def test_qdi_action_prompt_uses_compact_host_owned_decision_state() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "input"
        report_dir = root / "report"
        data_root.mkdir()
        cfg = AutoRealizeConfig.from_env()
        cfg.investigation.max_questions = 1
        cfg.investigation.max_rounds_per_run = 2
        cfg.investigation.max_scripts_per_question = 0
        cfg.investigation.context_retrievals_per_question = 1
        cfg.investigation.document_retrievals_per_question = 0
        cfg.investigation.artifact_retrievals_per_question = 0
        llm = _TwoRoundLLM()

        _run_question_investigator_inner(
            cfg=cfg,
            llm_client=llm,
            prompt_mgr=_PromptManager(),
            data_root=data_root,
            task_hint="test",
            file_summaries=[],
            relation_hints=[],
            constraint_memory={},
            authoritative_memory={},
            knowledge_base={},
            report_dir=report_dir,
        )

        action_prompts = [
            call["dynamic_user_prompt"]
            for call in llm.calls
            if call["model_cls"] is QuestionInvestigationAction
        ]
        assert action_prompts
        assert all('"protocol":"qdi-decision-state-v7"' in prompt for prompt in action_prompts)
        assert all('"decision_state"' in prompt for prompt in action_prompts)
        assert all("Append-only action and digest timeline" not in prompt for prompt in action_prompts)
        assert all("Recent live action window" not in prompt for prompt in action_prompts)
        assert '"next_step_hint"' in action_prompts[-1]
        assert '"available_actions"' in action_prompts[-1]


def test_one_investigation_slot_still_allows_context_retrieval_before_answer(tmp_path: Path) -> None:
    data_root = tmp_path / "input"
    data_root.mkdir()
    cfg = AutoRealizeConfig.from_env()
    cfg.investigation.max_questions = 1
    cfg.investigation.max_rounds_per_run = 1
    cfg.investigation.adaptive_budgeting = True
    llm = _TwoRoundLLM(recommended_actions=2)

    report = _run_question_investigator_inner(
        cfg=cfg,
        llm_client=llm,
        prompt_mgr=_PromptManager(),
        data_root=data_root,
        task_hint="test",
        file_summaries=[],
        relation_hints=[],
        constraint_memory={},
        authoritative_memory={},
        knowledge_base={},
        report_dir=tmp_path / "report",
    )

    assert llm.action_calls == 2
    assert report["action_history"][0]["status"] == "context_retrieved"
    assert report["question_records"][0]["status"] == "resolved"


def test_complete_memory_survives_prompt_projection_and_long_gap_resolution(tmp_path: Path):
    card = _empty_working_memory_card("q1")
    gaps = [f"Gap {i}: " + "scope qualifier " * 70 + "DO NOT use holdout rows" for i in range(30)]
    card = _merge_working_memory_card(card, QDIWorkingMemoryUpdate(
        open_gaps=gaps, confirmed_facts=["Full population: " + "detail " * 180 + "except shutdown rows"],
    ), sequence=1, max_chars=1200)
    assert card["open_gaps"] == gaps
    assert card["confirmed_facts"][0].endswith("except shutdown rows")
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put("qdi_working_memory_full", "q1:1", card)
    card["_full_memory_artifact"] = ref
    view = _working_memory_prompt_view(card, max_chars=1200)
    assert view["truncated"] is True
    assert view["omitted_entries"]["open_gaps"] + len(view["open_gaps"]) == len(gaps)
    assert all(item["text"] == gaps[item["index"]] for item in view["open_gaps"])
    assert card["open_gaps"] == gaps
    result = store.read_excerpt(ref["artifact_id"], json_path="open_gaps[29]", max_chars=3000)
    assert json.loads(result["excerpt"]) == gaps[29]
    resolved = _merge_working_memory_card(card, QDIWorkingMemoryUpdate(
        evidence_refs=["action:2/full_scan"],
        resolutions=[{"field": "open_gaps", "text": gaps[29], "reason": "verified complete scan",
                      "evidence_refs": ["action:2/full_scan"]}],
    ), sequence=2, max_chars=1200)
    assert resolved["open_gaps"] == gaps[:29]
    assert card["open_gaps"] == gaps
