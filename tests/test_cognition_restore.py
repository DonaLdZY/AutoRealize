import json
from types import SimpleNamespace

import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.modules.data_cognition import DataCognitionModule


def test_restore_keeps_cognition_constraints_and_does_not_call_provider(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "orders.csv").write_text("order_id\n1\n")
    report = tmp_path / "report"
    report.mkdir()
    raw = {"task_hint": "Match orders", "summary": {"file_count": 1},
           "files": [{"path": "orders.csv", "role": "raw_data_table", "summary": "Order identifiers"}],
           "original_requirement_texts": ["No assumed capacity dates"],
           "table_columns": {"orders.csv": ["order_id"]},
           "constraint_memory": {"hard": ["Keep distinct order IDs"]},
           "question_investigation": {"answer": "Dates missing"},
           "authoritative_memory": {"pricing": "Contract tiers"},
           "knowledge_base": {"facts": ["Original content"]},
           "agent_context_pack": {"exceptions": ["Missing rate remains unassigned"]}}
    checkpoint = report / "data_cognition_report.json"
    checkpoint.write_text(json.dumps(raw))
    original = checkpoint.read_bytes()
    module = DataCognitionModule(AutoRealizeConfig(), SimpleNamespace(), report)
    restored = module.restore(data, "Match orders")
    assert restored.original_requirement_texts == raw["original_requirement_texts"]
    assert restored.constraint_memory == raw["constraint_memory"]
    assert restored.question_memory == raw["question_investigation"]
    assert restored.authoritative_memory == raw["authoritative_memory"]
    assert restored.agent_context_pack == raw["agent_context_pack"]
    assert restored.table_columns == raw["table_columns"]
    assert checkpoint.read_bytes() == original
    with pytest.raises(ValueError, match="unchanged task hint"):
        module.restore(data, "Different task")


def test_large_output_is_bounded_by_context_without_shortening_prompt(tmp_path, monkeypatch):
    import autorealize.llm.client as transport
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    cfg.llm.context_window_tokens = cfg.llm.max_tokens = 278528
    cfg.llm.stream_responses = False
    client = transport.LLMClient(cfg, tmp_path)
    captured = []
    monkeypatch.setattr(client.client.chat.completions, "create",
                        lambda **kwargs: captured.append(kwargs) or SimpleNamespace(choices=[]))
    monkeypatch.setattr(transport, "estimate_text_tokens", lambda _: 60000)
    kwargs = {"model": "test", "max_tokens": 278528,
              "messages": [{"role": "user", "content": "Complete original constraints"}]}
    client._chat_completion_with_network_retry(prompt_name="context-test", mode="text", create_kwargs=kwargs)
    assert captured[0]["messages"] == kwargs["messages"]
    assert captured[0]["max_tokens"] == 278528 - 60000 - int(278528 * .01)
    monkeypatch.setattr(transport, "estimate_text_tokens", lambda _: 278000)
    with pytest.raises(ValueError, match="context budget"):
        client._chat_completion_with_network_retry(prompt_name="context-test", mode="text", create_kwargs=kwargs)
    assert len(captured) == 1
