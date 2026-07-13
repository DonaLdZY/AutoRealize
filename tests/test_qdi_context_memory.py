from __future__ import annotations

from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.context_compiler import ArtifactStore
from autorealize.investigation import (
    _qdi_answerer_stable_prefix,
    _run_question_investigator_inner,
    _artifact_ids_in,
    _bounded_json_view,
    _empty_working_memory_card,
    _merge_working_memory_card,
    _new_live_action_entry,
    _recent_action_window,
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
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.action_calls = 0

    def ask_structured(self, *, model_cls, **kwargs):
        self.calls.append({"model_cls": model_cls, **kwargs})
        if model_cls is QuestionInvestigationPlan:
            return QuestionInvestigationPlan(
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


def test_stable_prefix_keeps_global_context_before_question_and_ignores_round_state() -> None:
    context = {"table_cards": [{"table_id": "orders"}], "authoritative_memory": {}}
    first_record = {"question_id": "q1", "question": "First?", "status": "pending", "depth": 0}
    changed_status = {**first_record, "status": "investigating", "short_answer": "dynamic"}
    second_record = {"question_id": "q2", "question": "Second?", "status": "pending", "depth": 0}

    first = _qdi_answerer_stable_prefix(context, first_record)
    same_question_next_round = _qdi_answerer_stable_prefix(context, changed_status)
    second = _qdi_answerer_stable_prefix(context, second_record)

    assert first == same_question_next_round
    assert first.index("Frozen global QDI context") < first.index("Immutable initial current-question card")
    global_block = first.split("2. Immutable initial current-question card", 1)[0]
    assert second.startswith(global_block)


def test_full_qdi_loop_reuses_prefix_and_updates_memory_without_summary_call(tmp_path: Path) -> None:
    data_root = tmp_path / "input"
    report_dir = tmp_path / "report"
    data_root.mkdir()
    cfg = AutoRealizeConfig.from_env()
    cfg.investigation.max_questions = 1
    cfg.investigation.max_rounds_per_run = 2
    cfg.investigation.max_scripts_per_question = 0
    llm = _TwoRoundLLM()

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
    assert "context_retrieved" in answerer_calls[1]["dynamic_user_prompt"]
    assert "pending_action_digest_requests" in answerer_calls[1]["dynamic_user_prompt"]
    assert "output key is not confirmed" in answerer_calls[1]["dynamic_user_prompt"]
    assert report["answers"][0]["answer"] == "No table card establishes an output key."
    assert report["action_digest_cards"][0]["digest_source"] == "llm_action_digest"
    assert report["action_digest_cards"][0]["digest_for_sequence"] == 1
    assert "No matching table card" in report["action_digest_cards"][0]["key_outputs"][0]
    assert report["working_memory_cards"][0]["last_updated_sequence"] == 2
