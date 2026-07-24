from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .config import AutoRealizeConfig
from .context_compiler import ArtifactStore
from .logging_utils import log_event
from .models import CrossStageMemorySummary, CrossStageRetrievalPlan
from .prompt_cache import estimate_text_tokens, json_block, stable_dynamic_prompt
from .utils.safe_json import json_safe, write_json_safe

logger = logging.getLogger(__name__)


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    value = str(text or "")
    limit = max(1, int(max_tokens))
    if estimate_text_tokens(value) <= limit:
        return value
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(value[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return value[:low]


class CrossStageContextLedger:
    """Provider-friendly task context with artifact-backed lossy memory.

    The stable task/data prefix never changes after construction. Stage outputs
    accumulate in the final dynamic message. When that dynamic memory approaches
    its budget, an LLM compacts older entries while the full payloads remain in
    local artifacts that a later stage can explicitly retrieve.
    """

    def __init__(
        self,
        *,
        config: AutoRealizeConfig,
        llm_client: Any,
        prompt_mgr: Any,
        report_dir: Path,
        stable_context: dict[str, Any],
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.prompt_mgr = prompt_mgr
        self.report_dir = report_dir
        self.artifact_store = ArtifactStore(
            report_dir / "context_artifacts",
            default_visible_limit=int(getattr(config.context, "artifact_visible_excerpt_chars", 1200)),
        )
        full_stable_context = json_safe(stable_context)
        stable_ref = self.artifact_store.put(
            "cross_stage_stable_context",
            "immutable_task_context",
            full_stable_context,
            visible_excerpt="完整稳定任务上下文；仅在常驻前缀不足以支持当前判断时按需取回。",
        )
        self.stable_context = self._bounded_stable_context(full_stable_context, stable_ref)
        self.static_context_prompt = stable_dynamic_prompt(stable=self.stable_context, dynamic={})[0]
        self.entries: list[dict[str, Any]] = []
        self.summary = CrossStageMemorySummary()
        self.artifact_catalog: list[dict[str, Any]] = [
            {
                "artifact_id": str(stable_ref.get("artifact_id") or ""),
                "stage": "immutable_task_context",
                "authority": "authoritative_stable_context",
                "visible_excerpt": str(stable_ref.get("visible_excerpt") or "")[:1200],
                "original_chars": int(stable_ref.get("original_chars") or 0),
            }
        ]
        self.compaction_count = 0
        self.retrieval_count = 0
        self.compaction_history: list[dict[str, Any]] = []
        self.retrieval_history: list[dict[str, Any]] = []
        self._next_sequence = 1
        self._retrieval_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._persist()

    @staticmethod
    def _serialized_chars(value: Any) -> int:
        return len(json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, default=str))

    def _bounded_stable_context(self, stable_context: dict[str, Any], stable_ref: dict[str, Any]) -> dict[str, Any]:
        """Keep the immutable prefix bounded without destroying its evidence."""

        configured = max(
            4000,
            int(getattr(self.config.context, "cross_stage_stable_context_chars", 32000)),
        )
        prompt_budget_chars = max(8000, int(getattr(self.config.prompt, "prompt_token_budget", 12000)) * 4)
        ratio = min(0.9, max(0.4, float(getattr(self.config.context, "cross_stage_headroom_ratio", 0.72))))
        budget = min(configured, max(4000, int(prompt_budget_chars * ratio * 0.72)))
        configured_tokens = getattr(self.config.context, "cross_stage_stable_context_tokens", None)
        token_budget = (
            max(1000, int(configured_tokens))
            if configured_tokens
            else max(1000, int(getattr(self.config.prompt, "prompt_token_budget", 12000) * ratio * 0.65))
        )
        bounded = dict(stable_context)
        bounded["full_stable_context_artifact"] = {
            "artifact_id": str(stable_ref.get("artifact_id") or ""),
            "original_chars": int(stable_ref.get("original_chars") or 0),
            "retrieval_policy": "按需使用 artifact_id 和 json_path 取回；不得从省略内容中猜测。",
        }
        task_hint = bounded.get("task_hint")
        if isinstance(task_hint, str) and estimate_text_tokens(task_hint) > 800:
            bounded["task_hint"] = {
                "artifact_id": str(stable_ref.get("artifact_id") or ""),
                "json_path": "payload.task_hint",
                "visible_excerpt": _truncate_to_token_budget(task_hint, 800),
                "truncated": True,
            }
        if self._serialized_chars(bounded) <= budget and estimate_text_tokens(
            json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str)
        ) <= token_budget:
            return bounded

        protected = {
            "schema_version",
            "context_policy",
            "output_language_policy",
            "task_hint",
            "full_stable_context_artifact",
        }
        replacement_order = sorted(
            (key for key in bounded if key not in protected),
            key=lambda key: self._serialized_chars(bounded.get(key)),
            reverse=True,
        )
        omitted: list[dict[str, Any]] = []
        for key in replacement_order:
            bounded_text = json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str)
            if self._serialized_chars(bounded) <= budget and estimate_text_tokens(bounded_text) <= token_budget:
                break
            value = bounded.get(key)
            text = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, default=str)
            visible_chars = min(1200, max(240, budget // 16))
            bounded[key] = {
                "artifact_id": str(stable_ref.get("artifact_id") or ""),
                "json_path": f"payload.{key}",
                "original_chars": len(text),
                "visible_excerpt": text[:visible_chars],
                "truncated": len(text) > visible_chars,
            }
            omitted.append({"key": key, "original_chars": len(text)})
        bounded["stable_context_compaction"] = {
            "lossy": True,
            "omitted_sections": omitted,
            "full_artifact_id": str(stable_ref.get("artifact_id") or ""),
        }
        for item in omitted:
            bounded_text = json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str)
            if self._serialized_chars(bounded) <= budget and estimate_text_tokens(bounded_text) <= token_budget:
                break
            section = bounded.get(str(item.get("key") or ""))
            if isinstance(section, dict) and isinstance(section.get("visible_excerpt"), str):
                section["visible_excerpt"] = _truncate_to_token_budget(section["visible_excerpt"], 120)
        return bounded

    def add(
        self,
        stage: str,
        payload: Any,
        *,
        authority: str = "derived",
        evidence_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        safe = json_safe(payload)
        text = json.dumps(safe, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        configured_entry_limit = max(
            1000,
            int(getattr(self.config.context, "cross_stage_memory_entry_chars", 12000)),
        )
        recent_count = max(1, int(getattr(self.config.context, "cross_stage_memory_recent_entries", 4)))
        live_budget = self._live_memory_budget()
        live_token_budget = self._live_memory_token_budget()
        adaptive_entry_limit = max(1000, int(live_budget * 0.55 / recent_count))
        entry_limit = min(configured_entry_limit, adaptive_entry_limit)
        adaptive_entry_tokens = max(256, int(live_token_budget * 0.55 / recent_count))
        content_excerpt = _truncate_to_token_budget(text[:entry_limit], adaptive_entry_tokens)
        ref = self.artifact_store.put(
            "cross_stage_memory",
            stage,
            safe,
            visible_excerpt=text[: min(entry_limit, 2000)],
        )
        entry = {
            "sequence": self._next_sequence,
            "stage": str(stage),
            "authority": str(authority),
            "evidence_refs": [str(x) for x in (evidence_refs or []) if str(x).strip()][:20],
            "artifact_id": str(ref.get("artifact_id") or ""),
            "content_excerpt": content_excerpt,
            "truncated": len(content_excerpt) < len(text),
            "original_chars": len(text),
        }
        self._next_sequence += 1
        self.entries.append(entry)
        self.artifact_catalog.append(
            {
                "artifact_id": entry["artifact_id"],
                "stage": str(stage),
                "authority": str(authority),
                "visible_excerpt": str(ref.get("visible_excerpt") or "")[:1200],
                "original_chars": len(text),
            }
        )
        if len(self.artifact_catalog) > 120:
            self.artifact_catalog = [self.artifact_catalog[0], *self.artifact_catalog[-119:]]
        self._persist()
        return entry

    def prompt_parts(
        self,
        *,
        stage: str,
        stage_evidence: Any,
        latest_request: Any,
        dynamic_title: str = "Latest stage request",
        dynamic_limit: int | None = None,
    ) -> tuple[str, str]:
        self._compact_if_needed(stage)
        recovered = self._retrieve_if_needed(stage, latest_request)
        payload = {
            "cross_stage_memory_policy": {
                "summary_is_lossy": True,
                "artifact_ids_are_retrievable": True,
                "hidden_content_must_not_be_inferred": True,
                "authority_order": "original requirements and verified evidence outrank accumulated interpretations",
            },
            "compressed_memory": self.summary.model_dump(),
            "recent_stage_entries": self.entries,
            "recovered_artifact_excerpts": recovered,
            "current_stage": str(stage),
            "current_stage_evidence": json_safe(stage_evidence),
        }
        safe_latest_request = json_safe(latest_request)
        full_payload = {**payload, "latest_request": safe_latest_request}
        serialized_payload = json.dumps(
            full_payload,
            ensure_ascii=False,
            sort_keys=False,
            indent=2,
            default=str,
        )
        if dynamic_limit is not None and dynamic_limit > 0 and len(serialized_payload) > dynamic_limit:
            ref = self.artifact_store.put(
                "cross_stage_prompt_payload",
                str(stage),
                full_payload,
                visible_excerpt=(
                    f"阶段 {stage} 的完整动态 prompt payload；模型只直接看到有界首尾预览。"
                ),
            )
            artifact_id = str(ref.get("artifact_id") or "")
            if artifact_id and not any(
                str(item.get("artifact_id") or "") == artifact_id for item in self.artifact_catalog
            ):
                self.artifact_catalog.append(
                    {
                        "artifact_id": artifact_id,
                        "stage": str(stage),
                        "authority": "full_dynamic_prompt_payload",
                        "visible_excerpt": str(ref.get("visible_excerpt") or "")[:1200],
                        "original_chars": int(ref.get("original_chars") or 0),
                    }
                )
                if len(self.artifact_catalog) > 120:
                    self.artifact_catalog = [self.artifact_catalog[0], *self.artifact_catalog[-119:]]
            payload["full_dynamic_payload_artifact"] = {
                "artifact_id": artifact_id,
                "artifact_path": str(ref.get("artifact_path") or ""),
                "original_chars": int(ref.get("original_chars") or 0),
                "policy": "完整载荷可审计并可供后续关键阶段按需取回；当前不可见部分不得猜测。",
            }
            preview_chars = max(120, min(1200, int(dynamic_limit) // 4))

            def bounded_preview(value: Any, json_path: str) -> Any:
                value_text = json.dumps(
                    json_safe(value),
                    ensure_ascii=False,
                    sort_keys=False,
                    indent=2,
                    default=str,
                )
                if len(value_text) <= preview_chars:
                    return value
                head_chars = max(40, int(preview_chars * 0.4))
                tail_chars = max(40, preview_chars - head_chars)
                return {
                    "truncated": True,
                    "artifact_id": artifact_id,
                    "json_path": json_path,
                    "original_chars": len(value_text),
                    "visible_json_head": value_text[:head_chars],
                    "visible_json_tail": value_text[-tail_chars:],
                }

            payload["current_stage_evidence"] = bounded_preview(
                payload["current_stage_evidence"],
                "payload.current_stage_evidence",
            )
            safe_latest_request = bounded_preview(
                safe_latest_request,
                "payload.latest_request",
            )
            self._persist()
        payload["latest_request"] = safe_latest_request
        # Preserve insertion order so recent evidence and the latest request stay
        # at the tail, where recency-sensitive providers see them last.
        return self.static_context_prompt, json_block(
            dynamic_title,
            payload,
            limit=dynamic_limit,
            sort_keys=False,
        )

    def _memory_chars(self) -> int:
        return len(
            json.dumps(
                {"summary": self.summary.model_dump(), "entries": self.entries},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )

    def _memory_tokens(self) -> int:
        return estimate_text_tokens(
            json.dumps(
                {"summary": self.summary.model_dump(), "entries": self.entries},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )

    def _live_memory_budget(self) -> int:
        configured_trigger = max(
            4000,
            int(getattr(self.config.context, "cross_stage_memory_trigger_chars", 48000)),
        )
        prompt_budget_chars = max(8000, int(getattr(self.config.prompt, "prompt_token_budget", 12000)) * 4)
        ratio = min(0.9, max(0.4, float(getattr(self.config.context, "cross_stage_headroom_ratio", 0.72))))
        available_dynamic = max(4000, int(prompt_budget_chars * ratio) - len(self.static_context_prompt))
        return min(configured_trigger, available_dynamic)

    def _live_memory_token_budget(self) -> int:
        prompt_budget = max(2000, int(getattr(self.config.prompt, "prompt_token_budget", 12000)))
        ratio = min(0.9, max(0.4, float(getattr(self.config.context, "cross_stage_headroom_ratio", 0.72))))
        stable_tokens = estimate_text_tokens(self.static_context_prompt)
        hard_body_limit = max(1000, int(prompt_budget * ratio) - stable_tokens)
        configured = getattr(self.config.context, "cross_stage_memory_trigger_tokens", None)
        compact_limit = max(1000, int(configured)) if configured else max(1000, int(prompt_budget * ratio))
        scope = str(
            getattr(self.config.context, "cross_stage_memory_limit_scope", "body_after_prefix")
            or "body_after_prefix"
        ).strip().lower()
        if scope == "total":
            compact_limit = max(1000, compact_limit - stable_tokens)
        return min(compact_limit, hard_body_limit)

    def _compact_if_needed(self, next_stage: str) -> None:
        if not bool(getattr(self.config.context, "cross_stage_memory_enabled", True)):
            return
        trigger = self._live_memory_budget()
        trigger_tokens = self._live_memory_token_budget()
        max_compactions = max(0, int(getattr(self.config.context, "cross_stage_memory_max_compactions", 6)))
        within_budget = self._memory_chars() <= trigger and self._memory_tokens() <= trigger_tokens
        if within_budget or self.compaction_count >= max_compactions or len(self.entries) <= 1:
            return

        configured_recent_count = max(
            1,
            int(getattr(self.config.context, "cross_stage_memory_recent_entries", 4)),
        )
        recent_count = 0
        recent_chars = 0
        recent_budget = max(1000, int(trigger * 0.55))
        recent_token_budget = max(256, int(trigger_tokens * 0.55))
        for entry in reversed(self.entries):
            entry_chars = self._serialized_chars(entry)
            entry_tokens = estimate_text_tokens(
                json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
            )
            if recent_count >= configured_recent_count or (
                recent_count > 0
                and (
                    recent_chars + entry_chars > recent_budget
                    or sum(
                        estimate_text_tokens(
                            json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                        )
                        for item in self.entries[-recent_count:]
                    )
                    + entry_tokens
                    > recent_token_budget
                )
            ):
                break
            recent_count += 1
            recent_chars += entry_chars
        recent_count = max(1, recent_count)
        split_at = max(1, len(self.entries) - recent_count)
        older = self.entries[:split_at]
        recent = self.entries[split_at:]
        if not older:
            return
        configured_target_chars = max(
            2000,
            int(getattr(self.config.context, "cross_stage_memory_target_chars", 16000)),
        )
        target_chars = min(configured_target_chars, max(2000, int(trigger * 0.4)))
        configured_target_tokens = getattr(self.config.context, "cross_stage_memory_target_tokens", None)
        target_tokens = (
            max(512, int(configured_target_tokens))
            if configured_target_tokens
            else max(512, int(trigger_tokens * 0.4))
        )
        before_chars = self._memory_chars()
        before_tokens = self._memory_tokens()
        source_artifact_ids = [
            str(item.get("artifact_id") or "")
            for item in older
            if str(item.get("artifact_id") or "")
        ]
        log_event(
            logger,
            "context.cross_stage",
            "COMPACTING",
            trigger="auto",
            before_chars=before_chars,
            before_tokens=before_tokens,
            budget_chars=trigger,
            budget_tokens=trigger_tokens,
            compacted_entries=len(older),
        )
        dynamic = json_block(
            "需要压缩的跨阶段动态记忆",
            {
                "instruction": (
                    "把旧阶段记忆压缩成可继续工作的结构化状态。保留已确认事实、决定、矛盾、未决问题和 artifact_id。"
                    "不得把推测提升为事实，不得丢失会影响任务、评估、输出、数据读取或硬约束的内容。"
                ),
                "next_stage": next_stage,
                "target_max_chars": target_chars,
                "previous_summary": self.summary.model_dump(),
                "older_entries": older,
            },
        )
        try:
            compacted = self.llm_client.ask_structured(
                model_cls=CrossStageMemorySummary,
                system_prompt=self.prompt_mgr.load("system/cross_stage_context_compactor.md"),
                user_prompt=dynamic,
                prompt_name=f"cross_stage_context_compactor_{self.compaction_count + 1}",
                max_tokens=max(1200, min(6000, int(target_tokens * 1.35))),
                static_context_prompt=self.static_context_prompt,
                dynamic_user_prompt=dynamic,
            )
            compacted.evidence_artifact_ids = list(
                dict.fromkeys(compacted.evidence_artifact_ids + source_artifact_ids)
            )[:120]
            self.summary = self._bounded_summary(compacted, target_chars, target_tokens)
            source = "llm"
        except Exception as exc:  # noqa: BLE001
            artifact_ids = [str(x.get("artifact_id") or "") for x in older if str(x.get("artifact_id") or "")]
            self.summary = CrossStageMemorySummary(
                task_state=self.summary.task_state,
                confirmed_facts=list(self.summary.confirmed_facts),
                decisions=list(self.summary.decisions),
                unresolved_questions=list(self.summary.unresolved_questions),
                contradictions=list(self.summary.contradictions),
                evidence_artifact_ids=list(dict.fromkeys(self.summary.evidence_artifact_ids + artifact_ids))[:80],
                stage_summaries=list(self.summary.stage_summaries)
                + [f"{item.get('stage')}: full result retained in {item.get('artifact_id')}" for item in older],
            )
            source = "deterministic_fallback"
            log_event(logger, "context.cross_stage", "COMPACTION_FAILED", error=str(exc)[:240])
        self.summary = self._bounded_summary(self.summary, target_chars, target_tokens)
        self.entries = recent
        self.compaction_count += 1
        self.compaction_history.append(
            {
                "compaction": self.compaction_count,
                "trigger": "auto",
                "next_stage": str(next_stage),
                "source": source,
                "before_chars": before_chars,
                "before_estimated_tokens": before_tokens,
                "budget_chars": trigger,
                "budget_estimated_tokens": trigger_tokens,
                "target_chars": target_chars,
                "target_estimated_tokens": target_tokens,
                "compacted_entries": len(older),
                "retained_recent_entries": len(recent),
                "compacted_artifact_ids": source_artifact_ids,
                "after_chars": self._memory_chars(),
                "after_estimated_tokens": self._memory_tokens(),
            }
        )
        self.compaction_history = self.compaction_history[-80:]
        log_event(
            logger,
            "context.cross_stage",
            "COMPACTED",
            source=source,
            older_entries=len(older),
            retained_recent=len(recent),
            compaction=self.compaction_count,
        )
        self._persist()

    def _retrieve_if_needed(self, stage: str, latest_request: Any) -> list[dict[str, Any]]:
        if not bool(getattr(self.config.context, "cross_stage_retrieval_enabled", True)):
            return []
        if self.compaction_count <= 0 or not self.artifact_catalog:
            return []
        prefixes = tuple(
            str(item)
            for item in getattr(
                self.config.context,
                "cross_stage_retrieval_stage_prefixes",
                (),
            )
            if str(item).strip()
        )
        if prefixes and not any(str(stage).startswith(prefix) for prefix in prefixes):
            return []
        cache_key = (str(stage), self.compaction_count)
        if cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]
        max_artifacts = max(1, int(getattr(self.config.context, "cross_stage_retrieval_max_artifacts", 3)))
        excerpt_chars = max(500, int(getattr(self.config.context, "cross_stage_retrieval_excerpt_chars", 6000)))
        dynamic = json_block(
            "压缩记忆取回决策",
            {
                "instruction": (
                    "判断当前阶段是否必须查看被压缩的旧证据。只有摘要不足以安全完成任务时才请求取回。"
                    "只能使用目录中的 artifact_id，最多请求规定数量；不要请求仅为复述背景的内容。"
                ),
                "current_stage": stage,
                "latest_request": json_safe(latest_request),
                "compressed_memory": self.summary.model_dump(),
                "artifact_catalog": self.artifact_catalog[-60:],
                "max_artifacts": max_artifacts,
            },
        )
        try:
            plan = self.llm_client.ask_structured(
                model_cls=CrossStageRetrievalPlan,
                system_prompt=self.prompt_mgr.load("system/cross_stage_context_retriever.md"),
                user_prompt=dynamic,
                prompt_name=f"cross_stage_context_retriever_{stage}",
                max_tokens=1600,
                static_context_prompt=self.static_context_prompt,
                dynamic_user_prompt=dynamic,
            )
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "context.cross_stage", "RETRIEVAL_PLAN_FAILED", stage=stage, error=str(exc)[:240])
            self.retrieval_history.append(
                {
                    "stage": str(stage),
                    "compaction": self.compaction_count,
                    "status": "planner_failed",
                    "error": str(exc)[:1000],
                }
            )
            self.retrieval_history = self.retrieval_history[-120:]
            self._retrieval_cache[cache_key] = []
            self._persist()
            return []
        if not plan.needs_retrieval:
            self.retrieval_history.append(
                {
                    "stage": str(stage),
                    "compaction": self.compaction_count,
                    "status": "not_needed",
                    "reason": str(plan.reason or "")[:1000],
                }
            )
            self.retrieval_history = self.retrieval_history[-120:]
            self._retrieval_cache[cache_key] = []
            self._persist()
            return []
        allowed = {str(item.get("artifact_id") or "") for item in self.artifact_catalog}
        recovered: list[dict[str, Any]] = []
        for request in plan.requests[:max_artifacts]:
            artifact_id = str(request.artifact_id or "").strip()
            if artifact_id not in allowed:
                continue
            result = self.artifact_store.read_excerpt(
                artifact_id,
                max_chars=min(excerpt_chars, max(500, int(request.max_chars or excerpt_chars))),
                json_path=str(request.json_path or ""),
                allowed_type_prefixes=("cross_stage_",),
            )
            recovered.append(
                {
                    "request_reason": str(request.reason or "")[:500],
                    **result,
                }
            )
        self.retrieval_count += len(recovered)
        self.retrieval_history.append(
            {
                "stage": str(stage),
                "compaction": self.compaction_count,
                "status": "completed" if recovered else "no_valid_requests",
                "reason": str(plan.reason or "")[:1000],
                "requested_artifact_ids": [
                    str(request.artifact_id or "") for request in plan.requests[:max_artifacts]
                ],
                "recovered": [
                    {
                        "artifact_id": str(item.get("artifact_id") or ""),
                        "json_path": str(item.get("json_path") or ""),
                        "status": str(item.get("status") or ""),
                        "visible_chars": int(item.get("visible_chars") or 0),
                    }
                    for item in recovered
                ],
            }
        )
        self.retrieval_history = self.retrieval_history[-120:]
        if recovered:
            log_event(
                logger,
                "context.cross_stage",
                "ARTIFACTS_RETRIEVED",
                stage=stage,
                artifacts=[str(x.get("artifact_id") or "") for x in recovered],
            )
        self._retrieval_cache[cache_key] = recovered
        self._persist()
        return recovered

    @staticmethod
    def _bounded_summary(
        summary: CrossStageMemorySummary,
        target_chars: int,
        target_tokens: int,
    ) -> CrossStageMemorySummary:
        data = summary.model_dump()
        data["task_state"] = _truncate_to_token_budget(
            str(data.get("task_state") or "")[:2000],
            max(128, int(target_tokens * 0.3)),
        )
        list_keys = [
            "confirmed_facts",
            "decisions",
            "unresolved_questions",
            "contradictions",
            "evidence_artifact_ids",
            "stage_summaries",
        ]
        for key in list_keys:
            data[key] = [str(x)[:1000] for x in data.get(key, [])[:80] if str(x).strip()]
        reducible_keys = [key for key in list_keys if key != "evidence_artifact_ids"]
        while True:
            serialized = json.dumps(data, ensure_ascii=False, default=str)
            if len(serialized) <= target_chars and estimate_text_tokens(serialized) <= target_tokens:
                break
            key = max(reducible_keys, key=lambda name: len(data.get(name, [])))
            if not data.get(key):
                break
            data[key].pop()
        return CrossStageMemorySummary.model_validate(data)

    def _persist(self) -> None:
        stable_digest = hashlib.sha256(self.static_context_prompt.encode("utf-8", errors="replace")).hexdigest()[:16]
        write_json_safe(
            self.report_dir / "cross_stage_context.json",
            {
                "schema_version": "autorealize.cross_stage_context.v1",
                "stable_context_digest": stable_digest,
                "stable_context_chars": len(self.static_context_prompt),
                "stable_context_estimated_tokens": estimate_text_tokens(self.static_context_prompt),
                "stable_context_budget_chars": int(
                    getattr(self.config.context, "cross_stage_stable_context_chars", 32000)
                ),
                "dynamic_memory_chars": self._memory_chars(),
                "dynamic_memory_budget_chars": self._live_memory_budget(),
                "dynamic_memory_estimated_tokens": self._memory_tokens(),
                "dynamic_memory_budget_tokens": self._live_memory_token_budget(),
                "memory_limit_scope": str(
                    getattr(self.config.context, "cross_stage_memory_limit_scope", "body_after_prefix")
                ),
                "compaction_count": self.compaction_count,
                "retrieval_count": self.retrieval_count,
                "next_sequence": self._next_sequence,
                "summary": self.summary.model_dump(),
                "recent_entries": self.entries,
                "artifact_catalog": self.artifact_catalog,
                "compaction_history": self.compaction_history,
                "retrieval_history": self.retrieval_history,
            },
            indent=2,
        )
