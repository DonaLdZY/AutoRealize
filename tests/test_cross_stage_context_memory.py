from __future__ import annotations

from autorealize.config import AutoRealizeConfig
from autorealize.context_memory import CrossStageContextLedger
from autorealize.models import CrossStageMemorySummary, CrossStageRetrievalPlan, CrossStageRetrievalRequest
from autorealize.prompt_cache import estimate_text_tokens


class _PromptManager:
    def load(self, _name: str) -> str:
        return "只使用可见证据并输出严格 JSON。"


class _MemoryLLM:
    def __init__(self) -> None:
        self.compactions = 0
        self.retrievals = 0
        self.retrieval_id = ""

    def ask_structured(self, *, model_cls, **_kwargs):
        if model_cls is CrossStageMemorySummary:
            self.compactions += 1
            # Deliberately omit artifact IDs; the ledger must restore them.
            return CrossStageMemorySummary(task_state="compressed")
        if model_cls is CrossStageRetrievalPlan:
            self.retrievals += 1
            return CrossStageRetrievalPlan(
                needs_retrieval=True,
                requests=[
                    CrossStageRetrievalRequest(
                        artifact_id=self.retrieval_id,
                        json_path="payload",
                        reason="需要精确旧证据",
                        max_chars=2000,
                    ),
                    CrossStageRetrievalRequest(
                        artifact_id="../../outside",
                        reason="非法 ID 必须被忽略",
                    ),
                ],
            )
        raise AssertionError(model_cls)


def test_cross_stage_memory_compacts_and_retrieves_exact_artifact(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    cfg.prompt.prompt_token_budget = 2000
    cfg.context.cross_stage_stable_context_chars = 4000
    cfg.context.cross_stage_memory_trigger_chars = 4000
    cfg.context.cross_stage_memory_entry_chars = 3000
    cfg.context.cross_stage_memory_recent_entries = 1
    cfg.context.cross_stage_retrieval_stage_prefixes = ("critical_",)
    llm = _MemoryLLM()
    ledger = CrossStageContextLedger(
        config=cfg,
        llm_client=llm,
        prompt_mgr=_PromptManager(),
        report_dir=tmp_path,
        stable_context={"original_requirements_full": "规则" * 6000, "task_hint": "预测"},
    )
    stable_prefix = ledger.static_context_prompt
    assert "full_stable_context_artifact" in stable_prefix
    assert estimate_text_tokens(stable_prefix) < 1600

    first = ledger.add("stage_1", {"fact": "A" * 5000})
    ledger.add("stage_2", {"fact": "B" * 5000})
    ledger.add("stage_3", {"fact": "C" * 5000})
    ledger.add("stage_4", {"fact": "D" * 5000})
    ledger.add("stage_5", {"fact": "E" * 5000})
    llm.retrieval_id = first["artifact_id"]

    stable_a, _ = ledger.prompt_parts(
        stage="noncritical_stage",
        stage_evidence={"value": 1},
        latest_request={"instruction": "noop"},
    )
    assert llm.compactions == 1
    assert llm.retrievals == 0
    assert first["artifact_id"] in ledger.summary.evidence_artifact_ids
    assert ledger.compaction_history[0]["trigger"] == "auto"
    assert ledger.compaction_history[0]["before_estimated_tokens"] > 0

    stable_b, dynamic_b = ledger.prompt_parts(
        stage="critical_review",
        stage_evidence={"value": 2},
        latest_request={"instruction": "fresh_tail_marker"},
    )
    assert stable_a == stable_b == stable_prefix
    assert llm.retrievals == 1
    assert first["artifact_id"] in dynamic_b
    assert ledger.retrieval_history[-1]["status"] == "completed"
    assert "fresh_tail_marker" in dynamic_b
    assert dynamic_b.rfind("fresh_tail_marker") > dynamic_b.rfind("recent_stage_entries")

    ledger.prompt_parts(
        stage="critical_review",
        stage_evidence={"value": 2},
        latest_request={"instruction": "fresh_tail_marker"},
    )
    assert llm.retrievals == 1
    assert (tmp_path / "cross_stage_context.json").is_file()


def test_body_after_prefix_scope_preserves_more_body_budget(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    cfg.prompt.prompt_token_budget = 10000
    cfg.context.cross_stage_memory_trigger_tokens = 3000
    cfg.context.cross_stage_stable_context_tokens = 2000
    cfg.context.cross_stage_memory_limit_scope = "body_after_prefix"
    ledger = CrossStageContextLedger(
        config=cfg,
        llm_client=_MemoryLLM(),
        prompt_mgr=_PromptManager(),
        report_dir=tmp_path / "body",
        stable_context={"task_hint": "任务", "background": "背景" * 500},
    )
    body_budget = ledger._live_memory_token_budget()

    cfg_total = AutoRealizeConfig()
    cfg_total.prompt.prompt_token_budget = 10000
    cfg_total.context.cross_stage_memory_trigger_tokens = 3000
    cfg_total.context.cross_stage_stable_context_tokens = 2000
    cfg_total.context.cross_stage_memory_limit_scope = "total"
    total_ledger = CrossStageContextLedger(
        config=cfg_total,
        llm_client=_MemoryLLM(),
        prompt_mgr=_PromptManager(),
        report_dir=tmp_path / "total",
        stable_context={"task_hint": "任务", "background": "背景" * 500},
    )

    assert body_budget == 3000
    assert total_ledger._live_memory_token_budget() < body_budget


def test_cjk_recent_entry_is_bounded_by_token_budget(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    cfg.prompt.prompt_token_budget = 4000
    cfg.context.cross_stage_memory_trigger_tokens = 1200
    cfg.context.cross_stage_memory_recent_entries = 1
    ledger = CrossStageContextLedger(
        config=cfg,
        llm_client=_MemoryLLM(),
        prompt_mgr=_PromptManager(),
        report_dir=tmp_path,
        stable_context={"task_hint": "任务"},
    )
    entry = ledger.add("cjk", {"content": "证据" * 5000})

    assert entry["truncated"] is True
    assert estimate_text_tokens(entry["content_excerpt"]) <= 700


def test_cross_stage_artifact_store_rejects_illegal_id(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    ledger = CrossStageContextLedger(
        config=cfg,
        llm_client=_MemoryLLM(),
        prompt_mgr=_PromptManager(),
        report_dir=tmp_path,
        stable_context={"task_hint": "x"},
    )
    result = ledger.artifact_store.read_excerpt(
        "../../outside",
        allowed_type_prefixes=("cross_stage_",),
    )
    assert result["status"] == "rejected"
    assert result["error"] == "invalid_artifact_id"


def test_oversized_current_dynamic_payload_is_artifact_backed(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    ledger = CrossStageContextLedger(
        config=cfg,
        llm_client=_MemoryLLM(),
        prompt_mgr=_PromptManager(),
        report_dir=tmp_path,
        stable_context={"task_hint": "任务"},
    )
    _, dynamic = ledger.prompt_parts(
        stage="large_stage",
        stage_evidence={"evidence": "早期" * 2000},
        latest_request={"instruction": "LATEST_MARKER", "payload": "最新" * 2000},
        dynamic_limit=1200,
    )

    prompt_artifacts = [
        item
        for item in ledger.artifact_catalog
        if item.get("authority") == "full_dynamic_prompt_payload"
    ]
    assert prompt_artifacts
    assert "LATEST_MARKER" in dynamic
    artifact_id = str(prompt_artifacts[-1]["artifact_id"])
    recovered = ledger.artifact_store.read_excerpt(
        artifact_id,
        json_path="payload.latest_request",
        max_chars=800,
        allowed_type_prefixes=("cross_stage_",),
    )
    assert recovered["status"] == "completed"
    assert "LATEST_MARKER" in recovered["excerpt"]
