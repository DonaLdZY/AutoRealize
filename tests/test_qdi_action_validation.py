import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from autorealize.config import AutoRealizeConfig
from autorealize.llm.client import LLMClient, _loads_json_object
from autorealize.models import QuestionInvestigationAction, QuestionInvestigationReport


PROVIDER_ERROR_TRACE = (
    '<think>**Verifying data sample consistency**\n\n> \u274c {"detail":"Rate limit exceeded"}'
)


@pytest.mark.parametrize("response", [
    "{}", '{"detail":"Rate limit exceeded"}', '{"error":{"message":"Rate limit exceeded"}}',
    PROVIDER_ERROR_TRACE, '{"action":""}', '{"action":"   "}',
    '{"action":"give_up"}', '{"action":"give_up","unresolved_reason":"   "}',
])
def test_default_action_cannot_turn_invalid_response_into_give_up(response):
    with pytest.raises(ValidationError):
        QuestionInvestigationAction.model_validate(_loads_json_object(response))


@pytest.mark.parametrize("payload", [
    {"action": "answer", "answer": 'The document quotes {"detail":"Rate limit exceeded"}.'},
    {"action": "give_up", "unresolved_reason": "The required source file is unavailable."},
    {"action": "request_script", "request_script": {"python_code": "def analyze(input_dir, scratch_dir):\n    return {}"}},
    {"action": "request_context", "request_context": {"query": "Verify the order identifier"}},
])
def test_normal_actions_and_quoted_error_evidence_remain_valid(payload):
    action = QuestionInvestigationAction.model_validate(payload)
    assert action.action == payload["action"]
    assert QuestionInvestigationAction.model_validate(action.model_dump()) == action
    report = QuestionInvestigationReport(action_history=[action.model_dump()])
    assert QuestionInvestigationReport.model_validate_json(report.model_dump_json()) == report


def _client(tmp_path):
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    cfg.llm.model_name = "gpt-5.6-luna"
    cfg.llm.max_retries = 1
    return LLMClient(cfg, tmp_path)


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage={"prompt_tokens": 0, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}},
    )


def test_extra_root_close_keeps_code_and_memory_without_paid_retry(tmp_path, monkeypatch):
    code = 'def analyze(input_dir, scratch_dir):\n    return {"quoted": "}, not syntax"}'
    prefix = {"action": "request_script", "request_script": {"python_code": code}}
    tail = {"working_memory_update": {"open_gaps": ["Still need full-population coverage"]}, "notes": "Exact scope"}
    malformed = json.dumps(prefix) + ',' + json.dumps(tail)[1:]
    client = _client(tmp_path)
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        return _response(malformed)
    monkeypatch.setattr(client, "_chat_completion_with_network_retry", completion)
    try:
        action = client.ask_structured(QuestionInvestigationAction, "Inspect", "Inspect", "question_investigator_action")
    finally:
        client.client.close()
    assert len(calls) == 1
    assert action.request_script.python_code == code
    assert action.working_memory_update.open_gaps == tail["working_memory_update"]["open_gaps"]
    assert action.notes == "Exact scope"
    assert json.loads(client.trace_path.read_text(encoding="utf-8").splitlines()[-1])["response"] == malformed


@pytest.mark.parametrize("malformed", [
    '{"action":"answer"},{"action":"give_up"}',
    '{"action":"answer"},"action":"give_up"}',
    '{"action":"answer"},"answer":"first","answer":"second"}',
    '{"action":"answer"},"evidence":{"x":1,"x":2}}',
    '{"action":"answer"},"notes":broken}',
    '{"action":"answer"},"answer":"done"} {"ignore":"me"}',
])
def test_extra_close_repair_rejects_ambiguous_or_incomplete_objects(malformed):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _loads_json_object(malformed)


def test_extra_close_repair_does_not_bypass_action_schema():
    with pytest.raises(ValidationError):
        QuestionInvestigationAction.model_validate(_loads_json_object('{"action":"give_up"},"notes":"no reason"}'))


def test_provider_error_is_logged_as_failed_and_never_cached_as_success(tmp_path, monkeypatch):
    client = _client(tmp_path)
    monkeypatch.setattr(client, "_chat_completion_with_network_retry", lambda **_: _response(PROVIDER_ERROR_TRACE))
    try:
        with pytest.raises(RuntimeError, match="LLM structured output failed"):
            client.ask_structured(QuestionInvestigationAction, "Process the current question", "Inspect data", "question_investigator_action")
    finally:
        client.client.close()
    assert not client.cache_path.exists()
    usage_rows = [json.loads(line) for line in client.usage_path.read_text(encoding="utf-8").splitlines()]
    assert len(usage_rows) == 1
    assert usage_rows[0]["parsed_ok"] is False
    assert usage_rows[0]["estimated_prompt_tokens"] > 0
    traces = [json.loads(line) for line in client.trace_path.read_text(encoding="utf-8").splitlines()]
    assert traces[-1]["response"] == PROVIDER_ERROR_TRACE
    assert traces[-1]["parsed_ok"] is False


def test_poisoned_cached_action_is_revalidated_and_replaced_by_valid_response(tmp_path, monkeypatch):
    client = _client(tmp_path)
    client._cache_put("existing-poisoned-key", PROVIDER_ERROR_TRACE)
    monkeypatch.setattr(client, "_cache_key", lambda **_: "existing-poisoned-key")
    valid = '{"action":"give_up","unresolved_reason":"The required source file is unavailable."}'
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _response(valid)

    monkeypatch.setattr(client, "_chat_completion_with_network_retry", completion)
    try:
        action = client.ask_structured(QuestionInvestigationAction, "Process the current question", "Inspect data", "question_investigator_action")
    finally:
        client.client.close()
    assert len(calls) == 1
    assert action.action == "give_up"
    assert action.unresolved_reason == "The required source file is unavailable."
    assert client._cache_get("existing-poisoned-key") == valid
    assert client._usage_summary["cache_hits_local"] == 0
    assert client._usage_summary["calls"] == 1
