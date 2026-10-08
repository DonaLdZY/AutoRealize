from copy import deepcopy

from autorealize.investigation import reconcile_question_memory


def test_remaining_uncertainty_is_not_lost_when_answer_is_marked_resolved():
    memory = {
        "summary": "2 resolved, 0 unresolved", "unresolved_questions": [],
        "answers": [
            {"question_id": "q1", "answer": "Known schema", "remaining_uncertainty": ""},
            {"question_id": "q2", "answer": "No official new-data contract found",
             "remaining_uncertainty": "New file format and identity policy remain unknown"},
        ],
        "question_records": [{"question_id": qid, "status": "resolved"} for qid in ("q1", "q2")],
        "progress": {"resolved_questions": 2, "unresolved_questions": 0},
    }
    before = deepcopy(memory)
    result = reconcile_question_memory(memory)
    assert memory == before
    assert result["answers"] == before["answers"]
    assert result["question_records"][1]["status"] == "answered_with_uncertainty"
    assert result["progress"]["resolved_questions"] == 1
    assert result["progress"]["unresolved_questions"] == 1
    assert "identity policy" in result["unresolved_questions"][0]
    assert reconcile_question_memory(result) == result


def test_latest_answer_can_resolve_earlier_uncertainty_without_rewriting_history():
    memory = {"answers": [
        {"question_id": "q1", "remaining_uncertainty": "Unknown row count"},
        {"question_id": "q1", "remaining_uncertainty": "", "answer": "Full parse verified"},
    ], "unresolved_questions": [], "question_records": [{"question_id": "q1", "status": "resolved"}]}
    assert reconcile_question_memory(memory) == memory
