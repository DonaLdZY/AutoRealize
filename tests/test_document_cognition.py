from __future__ import annotations

import json
from autorealize.cognition import (
    _bound_document_memory,
    _document_cognition_chunks,
    _summarize_document_full_text,
)
from autorealize.config import AutoRealizeConfig
from autorealize.models import CognitionSummary, DocumentCognitionMemory
from autorealize.prompts.manager import PromptManager


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ask_structured(self, *, model_cls, dynamic_user_prompt, prompt_name, static_context_prompt="", **kwargs):
        payload_text = f"{static_context_prompt}\n{dynamic_user_prompt}"
        self.calls.append({"model_cls": model_cls, "prompt_name": prompt_name, "payload": payload_text})
        if model_cls is DocumentCognitionMemory:
            marker = f"fact-{len(self.calls)}"
            return DocumentCognitionMemory(
                concise_purpose="完整文档认知",
                constraints_and_rules=[marker],
                source_anchors=[marker],
            )
        return CognitionSummary(
            file_role_guess="task_requirement",
            concise_summary="已阅读全文",
            detailed_report="基于全部切片重点记忆生成。",
        )


def _prompt_manager() -> PromptManager:
    return PromptManager(AutoRealizeConfig.from_env())


def test_long_document_is_read_chunk_by_chunk_without_replaying_previous_raw_text() -> None:
    cfg = AutoRealizeConfig.from_env()
    cfg.data.document_cognition_chunk_chars = 1000
    cfg.data.document_cognition_chunk_overlap_chars = 0
    llm = RecordingLLM()
    full_text = "A" * 1000 + "B" * 1000 + "C" * 800

    summary, trace = _summarize_document_full_text(
        cfg=cfg,
        llm=llm,
        prompt_mgr=_prompt_manager(),
        base_context={"file": "rules.pdf", "kind": "document", "task": "提取全部规则"},
        full_text=full_text,
        relative_path="rules.pdf",
    )

    chunk_calls = [call for call in llm.calls if call["model_cls"] is DocumentCognitionMemory]
    assert len(chunk_calls) == 3
    assert all(call["prompt_name"] == "cognition_document_chunk" for call in chunk_calls)
    assert "A" * 500 in chunk_calls[0]["payload"]
    assert "A" * 500 not in chunk_calls[1]["payload"]
    assert "B" * 500 in chunk_calls[1]["payload"]
    assert "B" * 500 not in chunk_calls[2]["payload"]
    assert "C" * 500 in chunk_calls[2]["payload"]
    assert llm.calls[-1]["model_cls"] is CognitionSummary
    assert trace["coverage_ratio"] == 1.0
    assert trace["chunks_processed"] == 3
    assert summary.concise_summary == "已阅读全文"


def test_short_document_uses_one_full_summary_call() -> None:
    cfg = AutoRealizeConfig.from_env()
    cfg.data.document_cognition_chunk_chars = 12000
    llm = RecordingLLM()
    full_text = "短文档中的完整规则"

    _, trace = _summarize_document_full_text(
        cfg=cfg,
        llm=llm,
        prompt_mgr=_prompt_manager(),
        base_context={
            "file": "rules.txt",
            "kind": "document",
            "task": "提取规则",
            "document_excerpt": full_text,
        },
        full_text=full_text,
        relative_path="rules.txt",
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["model_cls"] is CognitionSummary
    assert full_text in llm.calls[0]["payload"]
    assert trace["mode"] == "single_call_full_text"


def test_rolling_memory_has_a_total_character_budget() -> None:
    memory = DocumentCognitionMemory(
        concise_purpose="purpose",
        task_goals=["a" * 400 for _ in range(20)],
        constraints_and_rules=["b" * 400 for _ in range(20)],
        evaluation_and_submission=["c" * 400 for _ in range(20)],
    )

    bounded = _bound_document_memory(memory, item_limit=20, max_chars=5000)
    serialized = json.dumps(bounded, ensure_ascii=False)

    assert len(serialized) < 7000
    assert bounded["task_goals"]
    assert bounded["constraints_and_rules"]
    assert bounded["evaluation_and_submission"]


def test_chunking_covers_document_end_to_end() -> None:
    text = "0123456789" * 500
    chunks = _document_cognition_chunks(text, chunk_chars=1000, overlap_chars=100)

    assert chunks[0]["start_char"] == 0
    assert chunks[-1]["end_char"] == len(text)
    assert all(left["end_char"] >= right["start_char"] for left, right in zip(chunks, chunks[1:]))
