from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.investigation import (
    CrossFileInvestigationTools, InvestigationStepResult, _add_question_record, _deterministic_script_repair, _find_duplicate_question,
    _request_signature, _run_question_investigator_inner,
    run_custom_readonly_python,
    validate_custom_readonly_python,
)
from autorealize.models import (
    InvestigationQuestion, InvestigationToolRequest, QDIDeterministicCheck,
    QDIExtractiveAnswer, QuestionInvestigationAction, QuestionInvestigationPlan,
    ReadonlyPythonRequest,
)
from autorealize.qdi_efficiency import (
    QDIEvidenceCache, compile_checks, extractive_answer, question_context, scoped_timeline,
)


class Prompts:
    def load(self, name):
        return name


class ScriptedLLM:
    def __init__(self, plan, actions):
        self.plan, self.actions, self.calls = plan, iter(actions), []

    def ask_structured(self, model_cls, **kwargs):
        self.calls.append(kwargs)
        return self.plan if model_cls is QuestionInvestigationPlan else next(self.actions)


def run_loop(tmp_path, monkeypatch, llm, context=None, **overrides):
    from autorealize import investigation
    cfg = AutoRealizeConfig()
    cfg.investigation.automatic_verification_question_limit = 0
    cfg.investigation.max_rounds_per_run = 6
    shared_cache_dir = overrides.pop("shared_cache_dir", None)
    report_dir = overrides.pop("report_dir", tmp_path / "report")
    for key, value in overrides.items():
        setattr(cfg.investigation, key, value)
    root = tmp_path / "input"
    root.mkdir(exist_ok=True)
    if context is not None:
        monkeypatch.setattr(investigation, "_build_investigation_context", lambda **_: (deepcopy(context), {}))
    return _run_question_investigator_inner(
        cfg=cfg, llm_client=llm, prompt_mgr=Prompts(), data_root=root, task_hint="test",
        file_summaries=[], relation_hints=[], constraint_memory={}, authoritative_memory={},
        knowledge_base={}, report_dir=report_dir,
        shared_cache_dir=shared_cache_dir,
    )


def test_request_signature_uses_full_code_and_ignores_consumer_question():
    def req(qid, ending):
        return InvestigationToolRequest(question_id=qid, tool_name="custom_readonly_python",
            custom_python=ReadonlyPythonRequest(question_id=qid, input_files=["a.csv"],
                python_code="# shared prefix\n" * 30 + ending))
    assert _request_signature(req("q1", "return 1")) != _request_signature(req("q1", "return 2"))
    assert _request_signature(req("q1", "return 1")) == _request_signature(req("q2", "return 1"))


def test_deterministic_os_path_join_repair_preserves_csv_path_and_avoids_model_repair(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "records.csv").write_text("value\n7\n", encoding="utf-8")
    request = InvestigationToolRequest(
        request_id="script-1",
        question_id="q1",
        tool_name="custom_readonly_python",
        reason="Read a bounded CSV",
        params={},
        custom_python=ReadonlyPythonRequest(
            question_id="q1",
            python_code=(
                "import csv, os\n"
                "def analyze(input_dir: str, scratch_dir: str) -> dict:\n"
                "    name = 'records.csv'\n"
                "    path = os.path.join(input_dir, name)\n"
                "    with open(path, encoding='utf-8') as stream:\n"
                "        return {'values': [int(row['value']) for row in csv.DictReader(stream)]}\n"
            ),
        ),
    )
    failed = InvestigationStepResult(
        request_id="script-1",
        question_id="q1",
        tool_name="custom_readonly_python",
        status="failed",
        reason="Read a bounded CSV",
        result={"error": "custom_python_static_validation_failed", "issues": ["banned_import:os"]},
        error="custom_python_static_validation_failed",
    )

    repaired = _deterministic_script_repair(request, failed)

    assert repaired is not None
    assert "import os" not in repaired.custom_python.python_code
    assert "str(Path(input_dir) / (name))" in repaired.custom_python.python_code
    assert "os.path.join" in request.custom_python.python_code
    execution = run_custom_readonly_python(
        repaired.custom_python.python_code,
        input_dir=root,
        cfg=AutoRealizeConfig(),
    )
    assert execution["values"] == [7]


def test_deterministic_os_repair_declines_semantically_different_operations():
    request = InvestigationToolRequest(
        request_id="script-2",
        question_id="q1",
        tool_name="custom_readonly_python",
        reason="Read cwd",
        params={},
        custom_python=ReadonlyPythonRequest(
            question_id="q1",
            python_code=(
                "import os\n"
                "def analyze(input_dir: str, scratch_dir: str) -> dict:\n"
                "    return {'cwd': os.getcwd()}\n"
            ),
        ),
    )
    failed = InvestigationStepResult(
        request_id="script-2",
        question_id="q1",
        tool_name="custom_readonly_python",
        status="failed",
        reason="Read cwd",
        result={"error": "custom_python_static_validation_failed", "issues": ["banned_import:os"]},
        error="custom_python_static_validation_failed",
    )

    assert _deterministic_script_repair(request, failed) is None
    assert "os.getcwd()" in request.custom_python.python_code


def test_deterministic_os_repair_handles_exists_join_and_pandas_replace(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "data.csv").write_text(" value \n1\n", encoding="utf-8")
    request = InvestigationToolRequest(
        request_id="script-os-path",
        question_id="q1",
        tool_name="custom_readonly_python",
        reason="Read and normalize one table",
        params={},
        custom_python=ReadonlyPythonRequest(
            question_id="q1",
            python_code=(
                "import os, pandas as pd\n"
                "def analyze(input_dir: str, scratch_dir: str) -> dict:\n"
                "    path = os.path.join(input_dir, 'data.csv')\n"
                "    if not os.path.exists(path):\n"
                "        return {'exists': False}\n"
                "    frame = pd.read_csv(path)\n"
                "    frame.columns = frame.columns.str.replace(' ', '')\n"
                "    return {'exists': True, 'columns': list(frame.columns)}\n"
            ),
        ),
    )
    failed = InvestigationStepResult(
        request_id="script-os-path",
        question_id="q1",
        tool_name="custom_readonly_python",
        status="failed",
        reason="Read and normalize one table",
        result={"error": "custom_python_static_validation_failed", "issues": ["banned_import:os"]},
        error="custom_python_static_validation_failed",
    )
    repaired = _deterministic_script_repair(request, failed)
    assert repaired is not None
    assert "os.path" not in repaired.custom_python.python_code
    assert ".exists()" in repaired.custom_python.python_code
    assert ".str.replace(" in repaired.custom_python.python_code
    result = run_custom_readonly_python(repaired.custom_python.python_code, input_dir=root, cfg=AutoRealizeConfig())
    assert result["exists"] is True
    assert result["columns"] == ["value"]


def test_question_id_collision_terminates_and_scope_extensions_are_not_duplicates():
    records = {"q2": {"question": "Are identifiers unique across all records?"}}
    assert _find_duplicate_question(records, "Are identifiers unique across all records? And in each time period?") == ""
    queue, questions = [], {}
    qid = _add_question_record(question=InvestigationQuestion(question_id="q2", question="Which time period?"),
        all_questions=questions, question_records=records, queue=queue,
        parent_question_id="", depth=0, max_total_questions=5)
    assert qid == "q3"


def test_long_question_keeps_critical_tail_in_action_input(tmp_path, monkeypatch):
    tail = "Verify full-file parsed timestamps and both directional coverage denominators."
    question = "Sources and previous evidence: " + "source details; " * 100 + tail
    plan = QuestionInvestigationPlan(questions=[InvestigationQuestion(question_id="q1", question=question)])
    llm = ScriptedLLM(plan, [QuestionInvestigationAction(action="give_up", unresolved_reason="No evidence")])
    report = run_loop(tmp_path, monkeypatch, llm, {"context_policy": {}})
    action_input = llm.calls[1]["dynamic_user_prompt"]
    assert tail in action_input
    assert report["questions"][0]["question"] == question
    assert report["question_records"][0]["question"] == question


def test_extractive_answers_require_complete_exact_authority_quote():
    context = {"authoritative_memory": {"metric": "MAE, lower is better; evaluate held-out rows only.",
                                        "unresolved_questions": ["metric is unknown"]}}
    proposal = QDIExtractiveAnswer(question=InvestigationQuestion(question="Which metric?"),
        source_pointer="/authoritative_memory/metric", quote=context["authoritative_memory"]["metric"])
    assert extractive_answer(context, proposal) == proposal.quote
    proposal.quote = "MAE"
    assert extractive_answer(context, proposal) is None
    proposal.source_pointer = "/authoritative_memory/unresolved_questions/0"
    proposal.quote = "metric is unknown"
    assert extractive_answer(context, proposal) is None
    proposal.source_pointer = "/table_cards/0/metric"
    assert extractive_answer(context, proposal) is None


@pytest.mark.parametrize("valid", [True, False])
def test_planner_direct_quote_saves_call_but_bad_citation_keeps_investigation(tmp_path, monkeypatch, valid):
    context = {"context_policy": {}, "authoritative_memory": {"metric": "MAE on development split"}}
    plan = QuestionInvestigationPlan(ready_to_answer=True, direct_answers=[QDIExtractiveAnswer(
        question=InvestigationQuestion(question_id="q1", question="Which metric?", category="evaluation"),
        source_pointer="/authoritative_memory/metric", quote="MAE on development split" if valid else "MAE",
    )])
    llm = ScriptedLLM(plan, [QuestionInvestigationAction(action="give_up", unresolved_reason="Invalid citation")])
    report = run_loop(tmp_path, monkeypatch, llm, context)
    assert len(llm.calls) == (1 if valid else 2)
    if valid:
        assert report["answers"][0]["evidence"] == ["stable_context:/authoritative_memory/metric"]
        assert report["answers"][0]["answer"] == "MAE on development split"
    else:
        assert report["question_records"][0]["status"] == "unresolved"


def test_question_projection_preserves_connected_sources_rules_and_fallback():
    context = {"table_cards": [{"table_id": n, "source_file": n, "fields": [{"name": "id"}]} for n in "abcd"],
               "relations": [{"left_file": "a", "right_file": "b"}, {"left_file": "b", "right_file": "c"}],
               "authoritative_memory": {"hard_rule": "never access holdout"},
               "constraint_memory": {"unit": "kg"}, "source_coverage_ledger": [{"population": "partial"}]}
    before = deepcopy(context)
    result = question_context(context, {"candidate_files": ["a"]})
    assert [c["table_id"] for c in result["table_cards"]] == list("abc")
    assert len(result["table_index"]) == 4
    for key in ("authoritative_memory", "constraint_memory", "source_coverage_ledger", "relations"):
        assert result[key] == context[key]
    assert context == before
    assert question_context(context, {"candidate_files": ["missing"]}) is context
    assert question_context(context, {}) is context


def check(kind="key_integrity", **kwargs):
    return QDIDeterministicCheck(kind=kind, left={"file": "a.csv"}, columns=["id"], **kwargs)


def test_time_range_keeps_naive_wall_clock_and_aware_utc_ranges_separate(tmp_path):
    (tmp_path / "a.csv").write_text(
        "time\n"
        "2026-01-01 00:00:00\n"
        "2025-12-31T19:30:00-05:00\n"
        "bad timestamp\n"
        '""\n',
        encoding="utf-8",
    )
    request = compile_checks(
        [
            QDIDeterministicCheck(
                kind="time_range",
                left={"file": "a.csv"},
                columns=["time"],
            )
        ],
        "q-time",
    )
    tools = CrossFileInvestigationTools(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        authoritative_memory={},
        knowledge_base={},
    )

    result = tools.execute(request)

    check_result = result.result["checks"][0]
    assert result.status == "completed", check_result
    assert check_result["status"] == "completed", check_result
    facts = check_result["facts"]["time_columns"]["time"]
    ranges = facts["timestamp_ranges_by_timezone_awareness"]
    assert facts["mixed_timezone_awareness"] is True
    assert facts["source_nulls"] == 1
    assert facts["valid_timestamp_rows"] == 2
    assert facts["parse_failures"] == 1
    assert facts["min"] is None
    assert facts["max"] is None
    assert ranges["naive"]["min"] == "2026-01-01T00:00:00"
    assert ranges["naive"]["max"] == "2026-01-01T00:00:00"
    assert ranges["aware_utc"]["min"] == "2026-01-01T00:30:00+00:00"
    assert ranges["aware_utc"]["max"] == "2026-01-01T00:30:00+00:00"


def test_compiled_check_failure_is_reported_as_tool_failure(tmp_path):
    (tmp_path / "a.csv").write_text("id\n1\n", encoding="utf-8")
    request = compile_checks(
        [QDIDeterministicCheck(kind="table_profile", left={"file": "a.csv"}, columns=["missing"])],
        "q-missing-column",
    )
    tools = CrossFileInvestigationTools(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        authoritative_memory={},
        knowledge_base={},
    )

    result = tools.execute(request)

    assert result.status == "failed"
    assert "indexes [0]" in result.error
    assert result.result["checks"][0]["status"] == "failed"
    assert "missing columns" in result.result["checks"][0]["error"]


def test_compiled_checks_execute_in_sandbox_preserve_sampling_and_null_keys(tmp_path):
    (tmp_path / "a.csv").write_text("id,time\n1,2020-01-01\n1,bad\n,2020-02-02\n9,2020-01-01\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("id\n1\n2\n", encoding="utf-8")
    checks = [check(row_limit=3), check("join_coverage", right={"file": "b.csv"}, right_columns=["id"]),
              QDIDeterministicCheck(kind="time_range", left={"file": "a.csv"}, columns=["time"])]
    request = compile_checks(checks, "q1")
    assert validate_custom_readonly_python(request.custom_python.python_code) == []
    tools = CrossFileInvestigationTools(cfg=AutoRealizeConfig(), data_root=tmp_path,
                                       authoritative_memory={}, knowledge_base={})
    result = tools.execute(request)
    assert result.status == "completed", result.model_dump()
    rows = result.result["checks"]
    assert all(r["status"] == "completed" for r in rows), rows
    assert rows[0]["scope"]["complete"] is False
    assert rows[0]["facts"]["null_key_rows"] == 1
    assert rows[0]["facts"]["duplicate_key_rows"] == 1
    assert rows[1]["facts"]["left_nonnull_row_coverage"] == pytest.approx(2 / 3)
    assert rows[1]["facts"]["null_keys_never_match"] is True
    assert rows[2]["facts"]["time_columns"]["time"]["parse_failures"] == 1


def test_headerless_profile_preserves_first_data_row_in_real_sandbox(tmp_path):
    (tmp_path / "raw.csv").write_text("alpha,1\nbeta,2\n", encoding="utf-8")
    checks = [QDIDeterministicCheck(kind="table_profile", left={"file": "raw.csv", "header": None}),
              QDIDeterministicCheck(kind="table_profile", left={"file": "raw.csv", "header": 0})]
    tools = CrossFileInvestigationTools(cfg=AutoRealizeConfig(), data_root=tmp_path,
                                       authoritative_memory={}, knowledge_base={})
    result = tools.execute(compile_checks(checks, "q1"))
    assert result.status == "completed", result.model_dump()
    raw, headed = result.result["checks"]
    assert raw["status"] == headed["status"] == "completed"
    assert raw["scope"]["rows_observed"] == 2
    assert raw["facts"]["columns"] == [0, 1]
    assert headed["scope"]["rows_observed"] == 1
    assert headed["facts"]["columns"] == ["alpha", "1"]


def test_evidence_cache_revalidates_content_and_rejects_arbitrary_code(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("id\n1\n", encoding="utf-8")
    cache = QDIEvidenceCache(tmp_path)
    req = compile_checks([check()], "q1")
    tools = CrossFileInvestigationTools(cfg=AutoRealizeConfig(), data_root=tmp_path,
                                       authoritative_memory={}, knowledge_base={})
    result = tools.execute(req)
    key = cache.key(req)
    cache.put(key, result)
    other = compile_checks([check()], "q2")
    reused = cache.get(cache.key(other), other)
    assert reused is not None
    assert reused.question_id == "q2"
    assert "_evidence_reuse" not in result.result
    original_stat = path.stat()
    path.write_text("id\n2\n", encoding="utf-8")
    import os
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert cache.get(cache.key(req), req) is None  # same size/mtime, changed bytes
    req.custom_python.python_code += "\n# modified"
    assert cache.key(req) is None
    assert QDIEvidenceCache(tmp_path, max_bytes=1).key(other) is None


def test_evidence_cache_reuses_completed_checks_across_runs(tmp_path):
    source_root = tmp_path / "input"
    source_root.mkdir()
    (source_root / "a.csv").write_text("id\n1\n2\n", encoding="utf-8")
    shared_cache = tmp_path / "project-cache"
    request = compile_checks([check()], "first-question")
    tools = CrossFileInvestigationTools(cfg=AutoRealizeConfig(), data_root=source_root,
                                       authoritative_memory={}, knowledge_base={})
    original = tools.execute(request)
    first_run = QDIEvidenceCache(source_root, shared_cache_dir=shared_cache)
    first_run.put(first_run.key(request), original)

    second_request = compile_checks([check()], "second-question")
    second_run = QDIEvidenceCache(source_root, shared_cache_dir=shared_cache)
    reused = second_run.get(second_run.key(second_request), second_request)

    assert reused is not None
    assert reused.question_id == "second-question"
    assert reused.result["_evidence_reuse"]["scope"] == "shared_project"
    assert second_run.shared_hits == 1
    assert second_run.local_hits == 0


@pytest.mark.parametrize("usage_available", [True, False])
def test_qdi_node_cache_reuses_planner_and_action_across_report_runs(
    tmp_path, monkeypatch, usage_available
):
    from autorealize import investigation

    context = {
        "context_policy": {},
        "table_cards": [
            {
                "table_id": "process.csv",
                "source_file": "process.csv",
                "fields": [{"name": "al_n", "logical_type": "numeric"}],
            }
        ],
        "authoritative_memory": {"task_goal": "predict al_n"},
        "constraint_memory": {},
    }
    plan = QuestionInvestigationPlan(
        questions=[InvestigationQuestion(question_id="q1", question="What is the target?")]
    )
    action = QuestionInvestigationAction(
        action="answer",
        answer="The target is al_n.",
        evidence=["process.csv"],
        confidence="high",
    )

    class CacheableLLM:
        def __init__(self):
            self.config = AutoRealizeConfig()
            self.base_url = "https://cache-test.invalid/v1"
            self.calls = []
            self.last_call_metadata = {}

        def ask_structured(self, model_cls, **kwargs):
            self.calls.append({"model_cls": model_cls, **kwargs})
            self.last_call_metadata = {
                "source": "provider",
                "parsed_ok": True,
                "finish_reason": "stop",
            }
            if usage_available:
                self.last_call_metadata["provider_usage_available"] = True
            return plan if model_cls is QuestionInvestigationPlan else action

    shared_cache = tmp_path / "shared-cache"
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first_client = CacheableLLM()
    first = run_loop(
        tmp_path / "first",
        monkeypatch,
        first_client,
        context,
        max_rounds_per_run=1,
        max_scripts_per_question=0,
        context_retrievals_per_question=0,
        document_retrievals_per_question=0,
        artifact_retrievals_per_question=0,
        shared_cache_dir=shared_cache,
        report_dir=tmp_path / "first" / "report",
    )
    second_client = CacheableLLM()
    second = run_loop(
        tmp_path / "second",
        monkeypatch,
        second_client,
        context,
        max_rounds_per_run=1,
        max_scripts_per_question=0,
        context_retrievals_per_question=0,
        document_retrievals_per_question=0,
        artifact_retrievals_per_question=0,
        shared_cache_dir=shared_cache,
        report_dir=tmp_path / "second" / "report",
    )

    assert len(first_client.calls) == 2
    assert second_client.calls == []
    metrics = second["progress"]["cost_controls"]["cache_metrics"]["qdi_node_result_cache"]
    assert metrics["qdi_node_cache_project_hits"] >= 2
    assert second["answers"][0]["answer"] == first["answers"][0]["answer"]


def test_join_preserves_leading_zero_ids_and_literal_na(tmp_path):
    (tmp_path / "a.csv").write_text("id\n001\n1\nNA\n\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("id\n001\nNA\n", encoding="utf-8")
    req = compile_checks([check("join_coverage", right={"file": "b.csv"}, right_columns=["id"])], "q1")
    tools = CrossFileInvestigationTools(cfg=AutoRealizeConfig(), data_root=tmp_path,
                                       authoritative_memory={}, knowledge_base={})
    result = tools.execute(req)
    facts = result.result["checks"][0]["facts"]
    assert facts["unique_keys"] == 3
    assert facts["null_key_rows"] == 1
    assert facts["intersection_unique_keys"] == 2
    assert facts["left_unique_key_coverage"] == pytest.approx(2/3)
    assert facts["right_unique_key_coverage"] == 1


def test_two_questions_reuse_check_and_keep_other_history_retrievable(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    (root / "a.csv").write_text("id\n1\n2\n", encoding="utf-8")
    plan = QuestionInvestigationPlan(questions=[
        InvestigationQuestion(question_id="q1", question="Are keys unique?"),
        InvestigationQuestion(question_id="q2", question="How many keys?"),
    ])
    actions = []
    for _ in range(2):
        actions += [QuestionInvestigationAction(action="request_checks", request_checks=[check()]),
                    QuestionInvestigationAction(action="answer", answer="Two distinct keys", evidence=["check:0"])]
    llm = ScriptedLLM(plan, actions)
    report = run_loop(tmp_path, monkeypatch, llm)
    assert len(llm.calls) == 5
    assert len(report["answers"]) == 2
    assert "_evidence_reuse" in report["step_results"][1]["result"]
    assert report["progress"]["cost_controls"]["evidence_cache_hits"] == 1
    assert "qdi_question_history_full" in llm.calls[3]["dynamic_user_prompt"]
    trace = [{"question_id": "q1", "detail": "first"}, {"question_id": "q2", "detail": "second"}]
    assert scoped_timeline(trace, "q2") == [trace[1]]


def test_repeated_request_forces_terminal_round_without_claiming_answer(tmp_path, monkeypatch):
    plan = QuestionInvestigationPlan(questions=[InvestigationQuestion(question_id="q1", question="Where is ID?")])
    first = QuestionInvestigationAction(action="request_context", request_context={"query": "ID"})
    llm = ScriptedLLM(plan, [first,
                           QuestionInvestigationAction(action="answer", answer="ID is not in the visible cards."),
                           QuestionInvestigationAction(action="give_up", unresolved_reason="No new evidence")])
    report = run_loop(tmp_path, monkeypatch, llm)
    assert '"request_context"' not in llm.calls[2]["dynamic_user_prompt"].split('"available_actions":', 1)[1].split(']', 1)[0]
    assert report["question_records"][0]["status"] == "resolved"


def test_context_retrievals_use_their_quota_without_using_final_investigation_slot(tmp_path, monkeypatch):
    plan = QuestionInvestigationPlan(questions=[InvestigationQuestion(question_id="q1", question="Which scope?")])
    llm = ScriptedLLM(plan, [
        QuestionInvestigationAction(action="request_context", request_context={"query": "first scope"}),
        QuestionInvestigationAction(action="request_context", request_context={"query": "second scope"}),
        QuestionInvestigationAction(action="answer", answer="Scope remains unknown", confidence="low"),
    ])
    report = run_loop(tmp_path, monkeypatch, llm, max_rounds_per_run=1,
                      context_retrievals_per_question=2, document_retrievals_per_question=0,
                      artifact_retrievals_per_question=0,
                      max_retrieval_actions_before_terminal=2)
    assert [item["action"] for item in report["action_history"]] == [
        "request_context", "request_context", "answer",
    ]
    assert len(llm.calls) == 4  # one planner call and three chargeable decisions
    first, second, final = [call["dynamic_user_prompt"] for call in llm.calls[1:]]
    assert all('"remaining_action_rounds":1' in prompt for prompt in (first, second, final))
    assert '"context":{"used":2,"remaining":0,"limit":2}' in final
    assert '"request_context"' not in final.split('"available_actions":', 1)[1].split(']', 1)[0]


def test_rejected_check_batch_still_consumes_script_action_quota(tmp_path, monkeypatch):
    plan = QuestionInvestigationPlan(questions=[InvestigationQuestion(question_id="q1", question="Can the key be checked?")])
    llm = ScriptedLLM(plan, [
        QuestionInvestigationAction(action="request_checks", request_checks=[]),
        QuestionInvestigationAction(action="answer", answer="The invalid check was not run."),
    ])
    report = run_loop(tmp_path, monkeypatch, llm, max_rounds_per_run=2,
                      max_scripts_per_question=1, context_retrievals_per_question=0,
                      document_retrievals_per_question=0, artifact_retrievals_per_question=0)
    next_prompt = llm.calls[2]["dynamic_user_prompt"]
    available = next_prompt.split('"available_actions":', 1)[1].split(']', 1)[0]
    assert '"request_checks"' not in available
    assert '"request_script"' not in available
    assert '"remaining_for_question":0' in next_prompt
    assert report["question_records"][0]["budget_usage"]["script_requests"] == 1
    assert report["tool_requests"] == []
    assert report["action_history"][0]["status"] == "invalid_check_request"


def test_script_then_artifact_retrieval_on_last_slot_preserves_result_tail(tmp_path, monkeypatch):
    plan = QuestionInvestigationPlan(questions=[InvestigationQuestion(question_id="q1", question="What is final value?")])

    class ReadTailLLM:
        def __init__(self):
            self.calls = []
            self.steps = 0

        def ask_structured(self, model_cls, **kwargs):
            self.calls.append(kwargs)
            if model_cls is QuestionInvestigationPlan:
                return plan
            self.steps += 1
            if self.steps == 1:
                return QuestionInvestigationAction(action="request_script", request_script=ReadonlyPythonRequest(
                    question_id="q1", python_code="""
def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"values": list(range(250))}
"""))
            if self.steps == 2:
                match = re.search(r"qdi_script_output_full_[0-9a-f]+", kwargs["dynamic_user_prompt"])
                assert match is not None
                return QuestionInvestigationAction(action="read_qdi_artifact_excerpt",
                    read_qdi_artifact_excerpt={"artifact_id": match.group(0), "json_path": "values", "max_chars": 5000})
            return QuestionInvestigationAction(action="answer", answer="The last value is 249.",
                                               evidence=["artifact:values[249]"])

    llm = ReadTailLLM()
    report = run_loop(tmp_path, monkeypatch, llm, max_rounds_per_run=2,
                      max_scripts_per_question=1, max_result_chars=100,
                      context_retrievals_per_question=0, document_retrievals_per_question=0,
                      artifact_retrievals_per_question=1)
    assert llm.steps == 3
    assert [item["status"] for item in report["action_history"]] == [
        "script_completed", "artifact_excerpt_read", "answered",
    ]
    assert '"remaining_action_rounds":1' in llm.calls[2]["dynamic_user_prompt"]
    assert '"remaining_action_rounds":1' in llm.calls[3]["dynamic_user_prompt"]
    assert '"remaining_for_question":0' in llm.calls[2]["dynamic_user_prompt"]
    assert "249" in llm.calls[3]["dynamic_user_prompt"]
    assert report["answers"][0]["answer"] == "The last value is 249."


def test_failed_artifact_read_uses_retrieval_quota_but_preserves_answer_slot(tmp_path, monkeypatch):
    plan = QuestionInvestigationPlan(questions=[InvestigationQuestion(question_id="q1", question="Where is output key?")])
    llm = ScriptedLLM(plan, [
        QuestionInvestigationAction(action="read_qdi_artifact_excerpt",
                                    read_qdi_artifact_excerpt={"artifact_id": "qdi_missing_123", "max_chars": 100}),
        QuestionInvestigationAction(action="give_up", unresolved_reason="No evidence for output key."),
    ])
    report = run_loop(tmp_path, monkeypatch, llm, max_rounds_per_run=1,
                      context_retrievals_per_question=0, document_retrievals_per_question=0,
                      artifact_retrievals_per_question=1)
    assert [item["status"] for item in report["action_history"]] == [
        "artifact_excerpt_failed", "given_up",
    ]
    final = llm.calls[-1]["dynamic_user_prompt"]
    assert '"qdi_artifact_excerpts":{"used":1,"remaining":0,"limit":1}' in final
    assert '"remaining_action_rounds":1' in final
    assert '"read_qdi_artifact_excerpt"' not in final.split('"available_actions":', 1)[1].split(']', 1)[0]


def test_final_report_retains_complete_answer_and_evidence_lists(tmp_path, monkeypatch):
    text = "scope=" + "x" * 1600 + " full-population-only"
    evidence = [f"source:{i}" for i in range(25)]
    notes = [f"note:{i}" for i in range(25)]
    plan = QuestionInvestigationPlan(questions=[InvestigationQuestion(question_id="q1", question="What is the scope?")])
    llm = ScriptedLLM(plan, [QuestionInvestigationAction(
        action="answer", answer=text, evidence=evidence, downstream_notes=notes,
        used_files=[f"file{i}.csv" for i in range(25)],
    )])
    report = run_loop(tmp_path, monkeypatch, llm, max_rounds_per_run=1,
                      context_retrievals_per_question=0, document_retrievals_per_question=0,
                      artifact_retrievals_per_question=0)
    assert report["answers"][0]["answer"] == text
    assert report["answers"][0]["evidence"] == evidence
    assert report["answers"][0]["downstream_notes"] == notes
    assert report["question_records"][0]["short_answer"] == text
    assert len(report["question_records"][0]["used_files"]) == 25


def test_document_search_delivers_source_neighbors_in_one_action(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    (root / "rules.md").write_text("Score is MAE. Only development rows may be scored. Holdout is forbidden.", encoding="utf-8")
    llm = ScriptedLLM(QuestionInvestigationPlan(questions=[InvestigationQuestion(question_id="q1", question="Score?")]), [
        QuestionInvestigationAction(action="search_document", search_document={"query": "MAE"}),
        QuestionInvestigationAction(action="answer", answer="MAE", evidence=["rules.md"]),
    ])
    report = run_loop(tmp_path, monkeypatch, llm)
    assert len(llm.calls) == 3
    assert "source_context" in llm.calls[-1]["dynamic_user_prompt"]
    assert "Holdout is forbidden" in llm.calls[-1]["dynamic_user_prompt"]
    assert report["action_history"][0]["action"] == "search_document"


def test_check_path_escape_and_missing_column_are_explicit_failed_evidence(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (tmp_path / "outside.csv").write_text("id\nsecret\n", encoding="utf-8")
    req = compile_checks([QDIDeterministicCheck(kind="table_profile", left={"file": "../outside.csv"})], "q1")
    tools = CrossFileInvestigationTools(cfg=AutoRealizeConfig(), data_root=root,
                                       authoritative_memory={}, knowledge_base={})
    result = tools.execute(req)
    assert "secret" not in json.dumps(result.model_dump())
    assert result.status == "failed" or result.result["checks"][0]["status"] == "failed"
