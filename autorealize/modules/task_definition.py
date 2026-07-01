from __future__ import annotations

import csv
import ast
import difflib
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path

from ..agents.architect import Architect
from ..config import AutoRealizeConfig
from ..context_compiler import (
    ArtifactStore,
    build_filename_group_cards,
    build_relation_cards,
    build_table_cards,
    compact_authoritative_memory,
    compact_constraint_memory,
    compact_detail_table_card_for_prompt,
    compact_table_cards_for_prompt,
    context_telemetry,
)
from ..logging_utils import log_event
from ..prompt_cache import stable_dynamic_prompt
from ..report_writer import (
    SECTION_ALIASES,
    append_constraint_memory_section,
    apply_evaluation_contract,
    build_automl_context_pack,
    build_data_access_protocol,
    build_data_schema_contract,
    build_description_markdown,
    coverage_defects,
    description_quality_check,
    description_protocol_bundle_defects,
    eval_ambiguity_defects,
    evaluation_contract_defects,
    finalize_description_markdown,
    render_automl_context_markdown,
    render_description_protocol_markdown,
    sync_submission_format_with_context,
    write_data_description,
)
from ..utils.safe_json import dumps_json_safe, write_json_safe
from ..models import (
    AmbiguityReview,
    DescriptionProtocolBundle,
    DescriptionSectionDraft,
    DescriptionTaskProtocolDraft,
    EvaluationContractReview,
    OutputSectionDraft,
    OverviewTaskDefinitionDraft,
    PipelinePlan,
    ProblemParadigmReview,
    SampleSubmissionSpec,
    SampleSubmissionValidationResult,
    SubmissionScriptPlan,
    TaskClassification,
)
from .types import DataCognitionResult, RuntimeServices, TaskDefinitionResult

logger = logging.getLogger(__name__)


class TaskDefinitionModule:
    """规范书 demo 的第二阶段：意图解析、任务范式识别与 Kaggle 风格任务书生成。"""

    def __init__(self, config: AutoRealizeConfig, services: RuntimeServices, run_dir: Path, report_dir: Path) -> None:
        self.config = config
        self.services = services
        self.run_dir = run_dir
        self.report_dir = report_dir
        self.architect = Architect(config, services.llm_client, services.prompt_mgr)
        self._evaluation_contract_revision_log: list[dict] = []
        self._evaluation_reflection_log: list[dict] = []

    def _default_pipeline_plan(self, *, task_hint: str) -> PipelinePlan:
        """Cheap compatibility plan for the legacy report shape.

        The current source of truth is the paradigm/protocol/evaluation contract
        pipeline. This object keeps downstream report fields populated without
        spending a separate LLM call on a plan that will be superseded later.
        """
        goal = task_hint.strip() or "根据输入数据和权威任务说明生成可执行任务定义。"
        return PipelinePlan(
            task_type="unknown_but_executable",
            objectives=[goal],
            phases=[],
            evaluation_metric="",
            evaluation_formula="",
            submission_spec="",
        )

    def _task_classification_from_problem_review(
        self,
        problem_review: ProblemParadigmReview,
        downstream_context: dict,
    ) -> TaskClassification:
        mapping = {
            "ml_dl_prediction": "prediction",
            "static_optimization": "optimization",
            "reinforcement_learning": "reinforcement_learning",
            "hybrid_ml_optimization": "hybrid_ml_optimization",
            "unknown_but_executable": "unknown_but_executable",
        }
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context, dict) else {}
        eval_reqs = auth.get("evaluation_requirements", []) if isinstance(auth, dict) else []
        primary_metric = str(eval_reqs[0])[:240] if eval_reqs else ""
        return TaskClassification(
            task_type=mapping.get(problem_review.problem_paradigm, "unknown_but_executable"),
            confidence=float(problem_review.confidence or 0.0),
            reasoning=(
                "Derived from ProblemParadigmReview in low-token mode; "
                "legacy task_classifier LLM call was skipped."
            ),
            primary_metric=primary_metric,
            metric_formula="",
        )

    def _read_sample_submission_columns(self, sample_path: Path) -> list[str]:
        if not sample_path.exists():
            return []
        try:
            with sample_path.open("r", encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f), [])
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "module.task_definition", "READ_SAMPLE_HEADER_FAILED", error=str(exc)[:180])
            return []
        return [str(c).strip() for c in header if str(c).strip()]

    def _authoritative_requirement_text(
        self,
        authoritative_memory: dict,
        original_requirement_texts: list[str],
        task_hint: str,
        agent_context_pack: dict | None = None,
    ) -> str:
        memory = authoritative_memory if isinstance(authoritative_memory, dict) else {}
        pack = agent_context_pack if isinstance(agent_context_pack, dict) else {}
        pack_authority = pack.get("authoritative_memory") if isinstance(pack.get("authoritative_memory"), dict) else {}
        if pack_authority.get("has_authoritative_sources"):
            memory = {**memory, **pack_authority}
        parts: list[str] = []

        def _add_block(title: str, value: object) -> None:
            if isinstance(value, str):
                text = value.strip()
                if text:
                    parts.append(f"## {title}\n{text}")
            elif isinstance(value, list):
                items = [str(x).strip() for x in value if str(x).strip()]
                if items:
                    parts.append(f"## {title}\n" + "\n".join(f"- {x}" for x in items))

        priority_order = [str(x).strip() for x in pack.get("priority_order", []) if str(x).strip()]
        if priority_order:
            parts.append("## 上下文优先级\n" + "\n".join(f"- {x}" for x in priority_order[:8]))
        else:
            parts.append(
                "## 上下文优先级\n"
                "- user task hint\n"
                "- existing input description.md\n"
                "- README / official requirement / spec / other task documents\n"
                "- data statistics and LLM inference"
            )
        if task_hint.strip():
            parts.append("## 用户补充需求（最高优先级）\n" + task_hint.strip())

        if memory.get("has_authoritative_sources"):
            _add_block("权威任务摘要", memory.get("summary", ""))
            _add_block("任务目标", memory.get("task_goal", ""))
            _add_block("输入要求", memory.get("input_requirements", []))
            _add_block("输出要求", memory.get("output_requirements", []))
            _add_block("评估要求", memory.get("evaluation_requirements", []))
            _add_block("约束", memory.get("constraints", []))
            _add_block("防泄漏要求", memory.get("leakage_guards", []))
            contract = memory.get("submission_contract") or {}
            if isinstance(contract, dict) and contract.get("is_defined"):
                contract_lines = [
                    f"output_filename: {contract.get('output_filename') or 'submission.csv'}",
                    f"sample_filename: {contract.get('sample_filename') or 'sample_submission.csv'}",
                ]
                cols = [str(x) for x in contract.get("columns", []) if str(x).strip()]
                if cols:
                    contract_lines.append("columns: " + ", ".join(cols))
                for key in ["row_unit", "row_count_rule", "format_description"]:
                    if str(contract.get(key, "")).strip():
                        contract_lines.append(f"{key}: {contract.get(key)}")
                rules = [str(x).strip() for x in contract.get("validation_rules", []) if str(x).strip()]
                if rules:
                    contract_lines.append("validation_rules: " + "; ".join(rules))
                evidence = [str(x).strip() for x in contract.get("evidence", []) if str(x).strip()]
                if evidence:
                    contract_lines.append("evidence: " + "; ".join(evidence[:6]))
                parts.append("## 权威提交/输出合同\n" + "\n".join(f"- {x}" for x in contract_lines))
            evidence_items = memory.get("evidence_items") or []
            evidence_lines = []
            if isinstance(evidence_items, list):
                for item in evidence_items[:12]:
                    if not isinstance(item, dict):
                        continue
                    src = str(item.get("source_path", "")).strip()
                    ev = str(item.get("evidence", "")).strip()
                    if src or ev:
                        evidence_lines.append(f"- {src}: {ev}")
            if evidence_lines:
                parts.append("## 权威证据来源\n" + "\n".join(evidence_lines))

        if original_requirement_texts:
            parts.append("## 原始需求文档摘要\n" + "\n\n".join(original_requirement_texts))
        return "\n\n".join(parts).strip()

    def _apply_authoritative_context(
        self,
        downstream_context: dict,
        authoritative_memory: dict,
        agent_context_pack: dict | None = None,
    ) -> None:
        memory = authoritative_memory if isinstance(authoritative_memory, dict) else {}
        pack = agent_context_pack if isinstance(agent_context_pack, dict) else {}
        downstream_context["agent_context_pack"] = pack
        if pack.get("context_routes"):
            downstream_context["context_routes"] = pack.get("context_routes")
        if pack.get("do_not_invent"):
            downstream_context["do_not_invent"] = pack.get("do_not_invent")
        downstream_context["authoritative_memory"] = memory
        contract = memory.get("submission_contract") if isinstance(memory.get("submission_contract"), dict) else {}
        pack_contract = pack.get("submission_contract") if isinstance(pack.get("submission_contract"), dict) else {}
        if pack_contract.get("is_defined"):
            contract = {**contract, **pack_contract}
        if not contract:
            return
        downstream_context["authoritative_submission_contract"] = contract
        columns = [str(x).strip() for x in contract.get("columns", []) if str(x).strip()]
        is_defined = bool(contract.get("is_defined"))
        if is_defined and columns:
            downstream_context["submission_columns"] = columns
            downstream_context["submission_contract_source"] = contract.get("source", "")
            downstream_context["submission_output_filename"] = contract.get("output_filename", "submission.csv")
            downstream_context["submission_sample_filename"] = contract.get("sample_filename", "sample_submission.csv")
        elif not is_defined:
            downstream_context["submission_contract_source"] = "not_defined_by_authoritative_sources"

    def _compact_agent_context(self, downstream_context: dict, *, route: str = "") -> dict:
        pack = downstream_context.get("agent_context_pack")
        if not isinstance(pack, dict):
            return {}
        data_memory = pack.get("data_memory") if isinstance(pack.get("data_memory"), dict) else {}
        routes = pack.get("context_routes") if isinstance(pack.get("context_routes"), dict) else {}
        route_payload = routes.get(route, {}) if route and isinstance(routes.get(route, {}), dict) else {}
        return {
            "priority_order": pack.get("priority_order", []),
            "do_not_invent": pack.get("do_not_invent", []),
            "route": route_payload,
            "authoritative_memory": pack.get("authoritative_memory", {}),
            "submission_contract": pack.get("submission_contract", {}),
            "constraint_memory": pack.get("constraint_memory", {}),
            "question_memory": pack.get("question_memory", {}),
            "data_memory": {
                "tables": data_memory.get("tables", [])[:20],
                "documents": data_memory.get("documents", [])[:12],
                "relations": data_memory.get("relations", [])[:40],
                "sampled_filename_patterns": data_memory.get("sampled_filename_patterns", [])[:30],
                "filename_sample_groups": data_memory.get("filename_sample_groups", [])[:40],
                "metric_candidates": data_memory.get("metric_candidates", [])[:30],
                "time_clues": data_memory.get("time_clues", [])[:30],
            },
        }

    def _classify_problem_paradigm(
        self,
        *,
        task_hint: str,
        original_text: str,
        data_digest: str,
        downstream_context: dict,
        file_summaries: list | None = None,
        relations: list | None = None,
        data_description_text: str = "",
    ) -> ProblemParadigmReview:
        system = self.services.prompt_mgr.load("system/problem_paradigm_classifier.md")
        problem_paradigm_pack = self._build_problem_paradigm_pack(
            downstream_context=downstream_context,
            file_summaries=file_summaries or [],
            relations=relations or [],
        )
        artifact_refs = self._task_definition_artifact_refs(
            original_text=original_text,
            data_description_text=data_description_text or data_digest,
            downstream_context=downstream_context,
        )
        payload = {
            "context_policy": self._headroom_context_policy(),
            "output_language_policy": self._output_language_policy(),
            "task_hint": task_hint,
            "original_requirements_full": original_text,
            "problem_paradigm_pack": problem_paradigm_pack,
        }
        self._record_headroom_pack_telemetry(
            "problem_paradigm_classifier",
            {
                "original_requirements_full": original_text,
                "problem_paradigm_pack": problem_paradigm_pack,
                "artifact_refs": artifact_refs,
            },
        )
        stable, dynamic = stable_dynamic_prompt(
            stable=payload,
            dynamic={"instruction": "Classify the executable problem paradigm from the evidence above."},
            stable_title="Stable problem paradigm evidence",
            dynamic_title="Dynamic classification request",
        )
        review = self.services.llm_client.ask_structured(
            model_cls=ProblemParadigmReview,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name="problem_paradigm_classifier",
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )
        allowed = {
            "ml_dl_prediction",
            "static_optimization",
            "reinforcement_learning",
            "hybrid_ml_optimization",
            "unknown_but_executable",
        }
        if review.problem_paradigm not in allowed:
            review.problem_paradigm = "unknown_but_executable"
        self._normalize_method_only_rl_review(
            review,
            original_text="\n".join([task_hint or "", original_text or ""]),
            downstream_context=downstream_context,
        )
        return review

    @staticmethod
    def _contains_any(text: str, keywords: list[str] | tuple[str, ...]) -> bool:
        lowered = (text or "").lower()
        return any(str(k).lower() in lowered for k in keywords if str(k).strip())

    @staticmethod
    def _dedupe_strings(values: list[str], *, limit: int = 20) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    def _normalize_method_only_rl_review(
        self,
        review: ProblemParadigmReview,
        *,
        original_text: str,
        downstream_context: dict,
    ) -> None:
        """Keep static decision tasks from being routed into a mandatory-RL template.

        A user may request RL as a modeling method for a dispatch/assignment
        problem, but downstream code still needs one deterministic solution
        table, evaluator, and scalar objective. In that case the task paradigm is
        static optimization and RL is a candidate solver family.
        """
        text_parts = [
            original_text or "",
            review.reasoning or "",
            "\n".join(str(x) for x in review.evidence or []),
            "\n".join(str(x) for x in review.key_signals or []),
            str(downstream_context.get("task_hint", "") or ""),
        ]
        text = "\n".join(text_parts)
        rl_signal = self._contains_any(
            text,
            (
                "reinforcement learning",
                "offline rl",
                "online rl",
                "dqn",
                "ppo",
                "policy",
                "reward",
                "强化学习",
                "状态",
                "动作",
                "奖励",
                "策略",
                "轨迹",
                "课程学习",
            ),
        )
        static_decision_signal = self._contains_any(
            text,
            (
                "static_optimization",
                "assignment",
                "dispatch",
                "routing",
                "scheduling",
                "vehicle",
                "order",
                "cost",
                "constraint",
                "feasible solution",
                "solution table",
                "分配",
                "调度",
                "派单",
                "订单",
                "车辆",
                "车型",
                "承运商",
                "运输成本",
                "总成本",
                "约束",
                "可行",
                "方案",
                "发车",
                "未分配",
            ),
        )
        official_env_signal = self._contains_any(
            text,
            (
                "gym",
                "gymnasium",
                "env.step",
                "reset(",
                "step(action",
                "official simulator",
                "provided simulator",
                "benchmark environment",
                "官方环境",
                "官方仿真器",
                "交互式评估接口",
                "回放评估接口",
                "评估服务器执行策略",
                "提交 policy",
                "提交策略",
            ),
        )
        solution_output_signal = self._contains_any(
            text,
            (
                "assignment 明细",
                "assignment",
                "solution",
                "dispatch plan",
                "完整方案",
                "方案输出",
                "发车记录",
                "订单号列表",
                "score_solution",
                "total_penalized_cost",
                "transport_cost",
                "unassigned",
                "未分配",
                "运输成本",
            ),
        )

        if rl_signal:
            review.explicit_rl_requested = True

        recommended = list(review.recommended_solver_families or [])
        if static_decision_signal:
            recommended.extend(["greedy_baseline", "repair_heuristic", "local_search"])
        if rl_signal:
            recommended.append("rl_candidate")
        review.recommended_solver_families = self._dedupe_strings(recommended, limit=8)

        if (
            review.problem_paradigm == "reinforcement_learning"
            and static_decision_signal
            and solution_output_signal
            and not official_env_signal
        ):
            review.problem_paradigm = "static_optimization"
            review.rl_as_required_paradigm = False
            note = (
                "AutoRealize normalized this as static_optimization: the authoritative deliverable/evaluator "
                "is a complete dispatch/assignment solution with deterministic constraints and scalar cost. "
                "RL is recorded as a requested candidate method, not as the problem paradigm."
            )
            if note not in review.method_routing_notes:
                review.method_routing_notes.append(note)
            review.key_signals = self._dedupe_strings(
                list(review.key_signals or []) + ["method_only_rl_request", "static_solution_output"],
                limit=20,
            )
            review.evidence = self._dedupe_strings(
                list(review.evidence or []) + ["Static assignment/dispatch output and deterministic cost/constraint evaluation dominate the executable contract."],
                limit=20,
            )
        elif review.problem_paradigm == "reinforcement_learning":
            review.rl_as_required_paradigm = True

        if static_decision_signal and rl_signal:
            baseline_note = (
                "Downstream first solution should implement the deterministic evaluator plus a greedy/repair/local-search baseline; "
                "if RL is attempted, it must reuse the same score_solution/validate_solution contract and be compared against the baseline."
            )
            if baseline_note not in review.method_routing_notes:
                review.method_routing_notes.append(baseline_note)

    def _protocol_prompt_for_paradigm(self, paradigm: str) -> str:
        mapping = {
            "ml_dl_prediction": "system/ml_dl_description_protocol.md",
            "static_optimization": "system/optimization_description_protocol.md",
            "reinforcement_learning": "system/rl_description_protocol.md",
            "hybrid_ml_optimization": "system/hybrid_description_protocol.md",
        }
        return mapping.get(paradigm, "system/ml_dl_description_protocol.md")

    def _compact_data_access_inventory(self, deterministic_data_access: object) -> dict:
        """Small data-access digest for LLM task reasoning; full protocol is merged by code."""
        data = deterministic_data_access.model_dump() if hasattr(deterministic_data_access, "model_dump") else deterministic_data_access
        if not isinstance(data, dict):
            return {"files": [], "global_notes": []}
        file_limit = max(1, int(getattr(self.config.prompt, "description_protocol_file_limit", 16)))
        field_limit = max(1, int(getattr(self.config.prompt, "description_protocol_fields_per_file", 12)))
        files = []
        for item in list(data.get("files", []) or [])[:file_limit]:
            if not isinstance(item, dict):
                continue
            files.append(
                {
                    "path": str(item.get("path", "")),
                    "file_role": str(item.get("file_role", "")),
                    "read_method": str(item.get("read_method", "")),
                    "read_example": str(item.get("read_example", "")),
                    "row_grain": str(item.get("row_grain", ""))[:300],
                    "key_fields": [str(x) for x in (item.get("key_fields", []) or [])[:6]],
                    "target_fields": [str(x) for x in (item.get("target_fields", []) or [])[:4]],
                    "relation_keys": [str(x) for x in (item.get("relation_keys", []) or [])[:6]],
                    "important_fields": [str(x) for x in (item.get("important_fields", []) or [])[:field_limit]],
                    "parsing_notes": [str(x)[:240] for x in (item.get("parsing_notes", []) or [])[:3]],
                }
            )
        omitted = max(0, len(list(data.get("files", []) or [])) - len(files))
        return {
            "global_notes": [str(x)[:300] for x in (data.get("global_notes", []) or [])[:4]],
            "files": files,
            "omitted_file_count": omitted,
            "policy": (
                "This is only a compact inventory for task reasoning. "
                "Do not output data_access JSON; deterministic data_access will be merged by code."
            ),
        }

    def _compact_protocol_agent_context(self, downstream_context: dict) -> dict:
        base = self._compact_agent_context(downstream_context, route="description_writer")
        data_memory = base.get("data_memory", {}) if isinstance(base.get("data_memory"), dict) else {}
        field_limit = max(1, int(getattr(self.config.prompt, "description_protocol_fields_per_file", 12)))

        def _compact_table(item: dict) -> dict:
            return {
                "path": str(item.get("path", "")),
                "role": str(item.get("role", "")),
                "summary": str(item.get("summary", ""))[:700],
                "columns": [str(x) for x in (item.get("columns", []) or [])[:field_limit]],
                "warnings": [str(x)[:220] for x in (item.get("warnings", []) or [])[:3]],
            }

        def _compact_doc(item: dict) -> dict:
            return {
                "path": str(item.get("path", "")),
                "role": str(item.get("role", "")),
                "summary": str(item.get("summary", ""))[:900],
                "detailed_report": str(item.get("detailed_report", ""))[:1800],
            }

        return {
            "priority_order": base.get("priority_order", []),
            "do_not_invent": base.get("do_not_invent", []),
            "route": base.get("route", {}),
            "authoritative_memory": base.get("authoritative_memory", {}),
            "submission_contract": base.get("submission_contract", {}),
            "constraint_memory": {
                "summary": str((base.get("constraint_memory") or {}).get("summary", ""))[:1000],
                "items": (base.get("constraint_memory") or {}).get("items", [])[:20],
            },
            "question_memory": {
                "summary": str((base.get("question_memory") or {}).get("summary", ""))[:1200],
                "answers": (base.get("question_memory") or {}).get("answers", [])[:20],
                "context_routing_notes": (base.get("question_memory") or {}).get("context_routing_notes", [])[:20],
            },
            "data_memory": {
                "tables": [_compact_table(x) for x in (data_memory.get("tables", []) or [])[:12] if isinstance(x, dict)],
                "documents": [_compact_doc(x) for x in (data_memory.get("documents", []) or [])[:8] if isinstance(x, dict)],
                "relations": (data_memory.get("relations", []) or [])[:24],
                "sampled_filename_patterns": (data_memory.get("sampled_filename_patterns", []) or [])[:20],
                "filename_sample_groups": (data_memory.get("filename_sample_groups", []) or [])[:20],
            },
        }

    def _bundle_from_protocol_draft(
        self,
        draft: DescriptionTaskProtocolDraft,
        *,
        paradigm: str,
        deterministic_data_access: object,
    ) -> DescriptionProtocolBundle:
        if hasattr(deterministic_data_access, "files"):
            data_access = deterministic_data_access
        else:
            from ..models import DataAccessProtocol

            data = deterministic_data_access.model_dump() if hasattr(deterministic_data_access, "model_dump") else deterministic_data_access
            data_access = DataAccessProtocol.model_validate(data if isinstance(data, dict) else {})
        bundle = DescriptionProtocolBundle(
            problem_paradigm=paradigm,
            overview=draft.overview,
            task_goal=draft.task_goal,
            data_access=data_access,  # type: ignore[arg-type]
            ml_dl=draft.ml_dl,
            optimization=draft.optimization,
            rl=draft.rl,
            hybrid=draft.hybrid,
            output=draft.output,
            evaluation_summary=draft.evaluation_summary,
            constraints=draft.constraints,
            warnings=draft.warnings,
        )
        return bundle

    def _build_description_protocol_bundle(
        self,
        *,
        problem_review: ProblemParadigmReview,
        original_text: str,
        data_digest: str,
        downstream_context: dict,
        deterministic_data_access: object,
        file_summaries: list | None = None,
        relations: list | None = None,
        data_description_text: str = "",
    ) -> DescriptionProtocolBundle:
        paradigm = problem_review.problem_paradigm or "unknown_but_executable"
        system = self.services.prompt_mgr.load(self._protocol_prompt_for_paradigm(paradigm))
        description_protocol_pack = self._build_description_protocol_pack(
            problem_review=problem_review,
            downstream_context=downstream_context,
            deterministic_data_access=deterministic_data_access,
            file_summaries=file_summaries or [],
            relations=relations or [],
        )
        artifact_refs = self._task_definition_artifact_refs(
            original_text=original_text,
            data_description_text=data_description_text or data_digest,
            downstream_context=downstream_context,
        )
        payload = {
            "context_policy": self._headroom_context_policy(),
            "output_language_policy": self._output_language_policy(),
            "instruction": (
                "Generate only the task/evaluation/output protocol draft. "
                "Do not enumerate all files or fields. Do not output data_access; it is generated by code."
            ),
            "original_requirements_full": original_text,
            "description_protocol_pack": description_protocol_pack,
        }
        self._record_headroom_pack_telemetry(
            f"description_protocol_{paradigm}",
            {
                "original_requirements_full": original_text,
                "description_protocol_pack": description_protocol_pack,
                "artifact_refs": artifact_refs,
            },
        )
        max_retries = max(3, int(getattr(self.config.prompt, "description_quality_max_retries", 3)))
        bundle: DescriptionProtocolBundle | None = None
        defects: list[str] = []
        for idx in range(max_retries):
            if idx == 0:
                dynamic_payload = {"instruction": "Generate the first DescriptionTaskProtocolDraft JSON."}
            else:
                dynamic_payload = {
                    "previous_protocol_defects": defects,
                    "repair_instruction": (
                        "Regenerate the full DescriptionTaskProtocolDraft JSON. Fix every listed defect. "
                        "Do not output data_access; deterministic file access will be merged by code. "
                        "Do not add sample_submission requirements unless an authoritative contract exists."
                    ),
                }
            stable, dynamic = stable_dynamic_prompt(
                stable=payload,
                dynamic=dynamic_payload,
                stable_title="Stable description protocol evidence",
                dynamic_title="Dynamic protocol generation request",
            )
            protocol_max_tokens_raw = getattr(self.config.prompt, "description_protocol_max_tokens", None)
            try:
                protocol_max_tokens = int(protocol_max_tokens_raw) if protocol_max_tokens_raw else None
            except (TypeError, ValueError):
                protocol_max_tokens = None
            draft = self.services.llm_client.ask_structured(
                model_cls=DescriptionTaskProtocolDraft,
                system_prompt=system,
                user_prompt=dynamic,
                prompt_name=f"description_protocol_{paradigm}_{idx+1}",
                max_tokens=protocol_max_tokens,
                static_context_prompt=stable,
                dynamic_user_prompt=dynamic,
            )
            bundle = self._bundle_from_protocol_draft(
                draft,
                paradigm=paradigm,
                deterministic_data_access=deterministic_data_access,
            )
            bundle.problem_paradigm = paradigm
            cols = [
                str(x)
                for x in (
                    downstream_context.get("submission_columns", [])
                    or downstream_context.get("generated_submission_columns", [])
                    or []
                )
                if str(x).strip()
            ]
            if cols and not bundle.output.columns:
                bundle.output.columns = cols
            if not bundle.output.output_filename:
                bundle.output.output_filename = str(downstream_context.get("submission_output_filename", "submission.csv"))
            if paradigm in {"static_optimization", "reinforcement_learning"} and not bundle.output.no_sample_submission_reason:
                bundle.output.no_sample_submission_reason = (
                    "优化/强化学习任务以方案或策略评估为中心；只有权威输出合同明确要求时才需要 sample_submission。"
                )
            defects = description_protocol_bundle_defects(bundle, downstream_context)
            log_event(
                logger,
                "module.task_definition.description_protocol",
                "REVIEWED",
                round=idx + 1,
                defects=len(defects),
                paradigm=paradigm,
            )
            if not defects:
                break
        if bundle is None:
            raise RuntimeError("description protocol generation returned no bundle")
        downstream_context["description_protocol_defects"] = defects
        if defects:
            log_event(
                logger,
                "module.task_definition.description_protocol",
                "WARNING",
                defects=defects[:8],
                paradigm=paradigm,
            )
        return bundle

    def _sample_submission_allowed_for_paradigm(self, problem_review: ProblemParadigmReview, downstream_context: dict) -> bool:
        configured = bool(getattr(self.config.switches, "generate_sample_submission", True))
        if not configured:
            return False
        contract = downstream_context.get("authoritative_submission_contract") or {}
        authoritative = bool(isinstance(contract, dict) and contract.get("is_defined"))
        if authoritative:
            return True
        if problem_review.problem_paradigm in {"static_optimization", "reinforcement_learning", "hybrid_ml_optimization"}:
            if problem_review.problem_paradigm != "hybrid_ml_optimization":
                return False
            source = str(problem_review.output_contract_source or "").lower()
            has_columns = bool(downstream_context.get("submission_columns") or downstream_context.get("generated_submission_columns"))
            return bool(problem_review.requires_sample_submission and (authoritative or has_columns or "official" in source or "sample" in source))
        if problem_review.requires_sample_submission:
            return True
        return True

    def _write_protocol_artifacts(
        self,
        *,
        problem_review: ProblemParadigmReview,
        deterministic_data_access: object,
        protocol_bundle: DescriptionProtocolBundle | None = None,
    ) -> None:
        write_json_safe(self.report_dir / "problem_paradigm_report.json", problem_review.model_dump(), indent=2)
        if hasattr(deterministic_data_access, "model_dump"):
            write_json_safe(self.report_dir / "data_access_protocol.json", deterministic_data_access.model_dump(), indent=2)
        if protocol_bundle is not None:
            write_json_safe(self.report_dir / "description_protocol_bundle.json", protocol_bundle.model_dump(), indent=2)

    def _known_data_file_names(self, data_root: Path) -> list[str]:
        names: list[str] = []
        try:
            names = [p.name for p in data_root.rglob("*") if p.is_file()]
        except Exception:
            names = []
        names.extend(["description.md", "description_origin.md", "sample_submission.csv", "submission.csv"])
        return sorted(dict.fromkeys([x for x in names if x]))[:300]

    def _json_dump_prompt(self, payload: object, *, limit: int | None = None) -> str:
        text = dumps_json_safe(payload, indent=2, sort_keys=True)
        return text[:limit] if limit and limit > 0 else text

    def _sample_submission_name(self, path: Path) -> bool:
        normalized = "".join(ch for ch in path.stem.lower() if ch.isalnum())
        return "samplesubmission" in normalized

    def _find_official_sample_submission(self, data_root: Path) -> Path | None:
        try:
            for p in data_root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"} and self._sample_submission_name(p):
                    return p
        except Exception:
            return None
        return None

    def _probe_sample_submission(self, sample_path: Path, data_root: Path | None = None) -> dict:
        if not sample_path or not sample_path.exists():
            return {}
        rel_path = sample_path.name
        if data_root is not None:
            try:
                rel_path = str(sample_path.relative_to(data_root)).replace("\\", "/")
            except Exception:
                rel_path = sample_path.name
        try:
            import pandas as pd

            if sample_path.suffix.lower() == ".csv":
                import autorealize.pipeline as legacy

                df = legacy.read_csv_auto(sample_path, nrows=10)
                header_df = legacy.read_csv_auto(sample_path, nrows=0)
                columns = [str(c) for c in header_df.columns.tolist()]
            elif sample_path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(sample_path, nrows=10)
                columns = [str(c) for c in df.columns.tolist()]
            elif sample_path.suffix.lower() == ".json":
                import autorealize.pipeline as legacy

                df = legacy.read_table(
                    sample_path,
                    json_flatten_sep=self.config.data.json_flatten_sep,
                    json_flatten_max_level=self.config.data.json_flatten_max_level,
                    json_keep_raw_nested_columns=self.config.data.json_keep_raw_nested_columns,
                    max_rows=10,
                )
                columns = [str(c) for c in df.columns.tolist()]
            else:
                return {}
            return {
                "path": rel_path,
                "suffix": sample_path.suffix.lower(),
                "columns": columns,
                "shape_preview": [int(df.shape[0]), int(df.shape[1])],
                "head": df.head(min(5, len(df))).astype(object).where(df.notna(), None).to_dict(orient="records"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"path": rel_path, "error": str(exc)[:500]}

    def _compact_profile_for_section(self, profile: dict) -> dict:
        if not isinstance(profile, dict):
            return {}
        item = {
            "name": str(profile.get("name", "")),
            "logical_type": profile.get("logical_type") or profile.get("dtype"),
            "row_count": profile.get("row_count"),
            "null_ratio": profile.get("null_ratio"),
            "unique_count": profile.get("unique_count"),
        }
        numeric = profile.get("numeric_stats") if isinstance(profile.get("numeric_stats"), dict) else {}
        if numeric:
            item["numeric_stats"] = {
                k: numeric.get(k)
                for k in ["min", "max", "mean", "std"]
                if numeric.get(k) not in (None, "", [], {})
            }
        datetime_stats = profile.get("datetime_stats") if isinstance(profile.get("datetime_stats"), dict) else {}
        if datetime_stats:
            item["datetime_stats"] = {
                k: datetime_stats.get(k)
                for k in ["min", "max", "granularity", "unique_timestamps"]
                if datetime_stats.get(k) not in (None, "", [], {})
            }
        top_values = profile.get("top_values")
        if isinstance(top_values, list) and top_values:
            item["top_values"] = [str(x)[:120] for x in top_values[:8]]
        return {k: v for k, v in item.items() if v not in (None, "", [], {})}

    def _compact_file_for_sections(self, fs, *, include_profiles: bool = False) -> dict:
        meta = getattr(fs, "source_metadata", {}) or {}
        semantics = getattr(fs, "column_semantics", {}) or {}
        field_limit = max(1, int(getattr(self.config.prompt, "description_protocol_fields_per_file", 12)))
        columns = [str(x) for x in (getattr(fs, "columns", []) or [])]
        profiles = [p for p in (getattr(fs, "column_profiles", []) or []) if isinstance(p, dict)]
        profile_by_name = {str(p.get("name", "")): p for p in profiles if str(p.get("name", "")).strip()}

        def _field_entry(col: str) -> dict:
            meaning = str(semantics.get(col, "") or "").strip()
            profile = profile_by_name.get(col, {})
            entry = {
                "name": col,
                "meaning": meaning[:260],
                "logical_type": profile.get("logical_type") or profile.get("dtype"),
                "unique_count": profile.get("unique_count"),
                "null_ratio": profile.get("null_ratio"),
            }
            return {k: v for k, v in entry.items() if v not in (None, "", [], {})}

        priority_words = [
            "id",
            "key",
            "订单",
            "编号",
            "时间",
            "日期",
            "目标",
            "标签",
            "target",
            "成本",
            "费用",
            "价格",
            "score",
            "约束",
            "状态",
            "仓",
            "车辆",
            "路线",
            "输出",
        ]
        ordered_cols = []
        for col in columns:
            text = f"{col} {semantics.get(col, '')}".lower()
            if any(w.lower() in text for w in priority_words):
                ordered_cols.append(col)
        for col in columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
            if len(ordered_cols) >= field_limit:
                break
        field_semantics = [_field_entry(c) for c in ordered_cols[:field_limit]]

        sheets = []
        sheet_profiles = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
        sheet_field_descriptions = (
            meta.get("sheet_field_descriptions") if isinstance(meta.get("sheet_field_descriptions"), dict) else {}
        )
        for sheet in sheet_profiles[:30]:
            if not isinstance(sheet, dict):
                continue
            sheet_name = str(sheet.get("sheet_name", ""))
            sheet_semantics = sheet_field_descriptions.get(sheet_name, {}) if isinstance(sheet_field_descriptions, dict) else {}
            sheet_cols = [str(x) for x in (sheet.get("columns") or [])[:field_limit]]
            sheet_entry = {
                "sheet_name": sheet_name,
                "shape": sheet.get("shape") or sheet.get("shape_profiled") or sheet.get("shape_sampled"),
                "columns": sheet_cols,
                "field_semantics": {
                    str(k): str(v)[:220]
                    for k, v in list((sheet_semantics or {}).items())[:field_limit]
                    if str(k).strip() and str(v).strip()
                },
                "profile_policy": sheet.get("profile_policy"),
                "is_deep_profiled": sheet.get("is_deep_profiled"),
            }
            if include_profiles:
                sheet_entry["column_profiles"] = [
                    self._compact_profile_for_section(p)
                    for p in (sheet.get("column_profiles") or [])[:field_limit]
                    if isinstance(p, dict)
                ]
            sheets.append({k: v for k, v in sheet_entry.items() if v not in (None, "", [], {})})

        reading_notes = []
        for item in (meta.get("read_examples") or []):
            if str(item).strip():
                reading_notes.append(str(item)[:260])
        if str(getattr(fs, "path", "")).lower().endswith((".xlsx", ".xls")) and sheets:
            reading_notes.append("多 sheet Excel：需要按 sheet_name 显式读取相关 sheet。")
        if str(getattr(fs, "path", "")).lower().endswith(".csv"):
            dialect = meta.get("csv_dialect") if isinstance(meta.get("csv_dialect"), dict) else {}
            encoding = str(meta.get("csv_encoding", "") or "")
            sep = dialect.get("sep") or dialect.get("delimiter")
            if encoding and encoding.lower() not in {"utf-8", "utf-8-sig"}:
                reading_notes.append(f"CSV 编码提示：{encoding}")
            if sep and str(sep) not in {",", ""}:
                reading_notes.append(f"CSV 非默认分隔符提示：sep={sep!r}")
        if str(getattr(fs, "path", "")).lower().endswith(".json"):
            strategy = str(meta.get("json_strategy", "") or "")
            if strategy:
                reading_notes.append(f"JSON 读取策略：{strategy}")

        out = {
            "path": str(getattr(fs, "path", "")),
            "role": str(getattr(getattr(fs, "role", ""), "value", getattr(fs, "role", ""))),
            "summary": str(getattr(fs, "summary", "") or "")[:700],
            "shape": meta.get("shape"),
            "shape_estimated": meta.get("shape_estimated"),
            "columns_count": len(columns),
            "field_semantics": field_semantics,
            "sheets": sheets,
            "reading_notes": reading_notes[:6],
            "warnings": [str(x)[:260] for x in (getattr(fs, "warnings", []) or [])[:5]],
        }
        if include_profiles:
            out["column_profiles"] = [self._compact_profile_for_section(p) for p in profiles[:field_limit]]
            sampling = meta.get("profile_sampling") if isinstance(meta.get("profile_sampling"), dict) else {}
            if sampling:
                out["profile_sampling"] = {
                    "rows_read": sampling.get("rows_read"),
                    "configured_max_rows": sampling.get("configured_max_rows"),
                    "sampling_reason": sampling.get("sampling_reason"),
                    "sampled": sampling.get("sampled"),
                }
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}

    def _compact_files_for_sections(self, file_summaries: list, *, include_profiles: bool = False, limit: int | None = None) -> list[dict]:
        max_files = limit or max(1, int(getattr(self.config.prompt, "description_protocol_file_limit", 16)))
        scored = []
        for fs in file_summaries:
            role = str(getattr(getattr(fs, "role", ""), "value", getattr(fs, "role", "")))
            path = str(getattr(fs, "path", ""))
            meta = getattr(fs, "source_metadata", {}) or {}
            score = 0
            if role in {"task_requirement", "data_description"}:
                score += 5
            if meta.get("downstream_role_hint"):
                score += 5
            if path.lower().endswith((".csv", ".xlsx", ".xls", ".json")):
                score += 3
            if meta.get("excel_sheet_profiles"):
                score += 3
            if getattr(fs, "column_semantics", None):
                score += 2
            scored.append((score, path, fs))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self._compact_file_for_sections(fs, include_profiles=include_profiles) for _, _, fs in scored[:max_files]]

    def _compact_relations_for_sections(self, relations: object, *, limit: int = 40) -> list[dict]:
        out: list[dict] = []
        for rel in list(relations or [])[:limit]:
            if hasattr(rel, "__dict__"):
                data = rel.__dict__
            elif isinstance(rel, dict):
                data = rel
            else:
                continue
            item = {
                "left_file": str(data.get("left_file", "")),
                "left_field": str(data.get("left_field", "")),
                "right_file": str(data.get("right_file", "")),
                "right_field": str(data.get("right_field", "")),
                "relation_type": str(data.get("relation_type", "")),
                "confidence": data.get("confidence"),
                "short_evidence": str(data.get("short_evidence", "") or data.get("reason", ""))[:260],
            }
            if not item["left_field"] and data.get("shared_columns"):
                shared = data.get("shared_columns") or []
                if shared:
                    item["left_field"] = str(shared[0])
                    item["right_field"] = str(shared[0])
            out.append({k: v for k, v in item.items() if v not in (None, "", [], {})})
        return out

    def _compact_filename_groups_for_sections(self, downstream_context: dict, *, limit: int = 20) -> list[dict]:
        pack = downstream_context.get("agent_context_pack") if isinstance(downstream_context.get("agent_context_pack"), dict) else {}
        data_memory = pack.get("data_memory") if isinstance(pack.get("data_memory"), dict) else {}
        groups = data_memory.get("filename_sample_groups") if isinstance(data_memory.get("filename_sample_groups"), list) else []
        out = []
        for item in groups[:limit]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "template_path": item.get("template_path") or item.get("pattern") or item.get("path_pattern"),
                    "count": item.get("count") or item.get("file_count"),
                    "role": item.get("role") or item.get("file_group_role"),
                    "structure_consistent": item.get("structure_consistent") or item.get("schema_consistent"),
                    "representative_files": [str(x) for x in (item.get("representative_files") or item.get("sample_files") or [])[:3]],
                    "shared_fields": [str(x) for x in (item.get("shared_fields") or item.get("columns") or [])[:20]],
                    "short_evidence": str(item.get("short_evidence") or item.get("sampling_reason") or item.get("reason") or "")[:260],
                }
            )
        return [{k: v for k, v in item.items() if v not in (None, "", [], {})} for item in out]

    def _context_artifact_store(self) -> ArtifactStore:
        return ArtifactStore(self.report_dir / "context_artifacts")

    def _ranked_file_summaries_for_sections(self, file_summaries: list, *, limit: int | None = None) -> list:
        max_files = limit or max(1, int(getattr(self.config.prompt, "description_protocol_file_limit", 16)))
        scored = []
        for fs in file_summaries or []:
            role = str(getattr(getattr(fs, "role", ""), "value", getattr(fs, "role", "")))
            path = str(getattr(fs, "path", ""))
            meta = getattr(fs, "source_metadata", {}) or {}
            score = 0
            if role in {"task_requirement", "data_description"}:
                score += 8
            if meta.get("downstream_role_hint"):
                score += 6
            if path.lower().endswith((".csv", ".xlsx", ".xls", ".json")):
                score += 4
            if meta.get("excel_sheet_profiles"):
                score += 4
            if getattr(fs, "column_semantics", None):
                score += 3
            if getattr(fs, "column_profiles", None):
                score += 2
            scored.append((score, path, fs))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [fs for _, _, fs in scored[:max_files]]

    def _filename_group_source(self, downstream_context: dict) -> list[dict]:
        groups: list[dict] = []
        pack = downstream_context.get("agent_context_pack") if isinstance(downstream_context.get("agent_context_pack"), dict) else {}
        data_memory = pack.get("data_memory") if isinstance(pack.get("data_memory"), dict) else {}
        if isinstance(data_memory.get("filename_sample_groups"), list):
            groups.extend([x for x in data_memory.get("filename_sample_groups", []) if isinstance(x, dict)])
        kb = downstream_context.get("knowledge_base") if isinstance(downstream_context.get("knowledge_base"), dict) else {}
        if isinstance(kb.get("filename_sample_groups"), list):
            groups.extend([x for x in kb.get("filename_sample_groups", []) if isinstance(x, dict)])
        return groups

    def _filename_group_cards_for_sections(
        self,
        downstream_context: dict,
        *,
        file_summaries: list,
        limit: int = 20,
    ) -> list[dict]:
        return build_filename_group_cards(
            self._filename_group_source(downstream_context),
            file_summaries=file_summaries,
            limit=limit,
        )

    def _table_cards_for_sections(
        self,
        file_summaries: list,
        *,
        limit: int | None = None,
        field_limit: int | None = None,
    ) -> list[dict]:
        ranked = self._ranked_file_summaries_for_sections(file_summaries, limit=limit)
        return build_table_cards(
            ranked,
            artifact_store=self._context_artifact_store(),
            file_limit=len(ranked) or 1,
            sheet_limit=80,
            field_limit=field_limit or max(1, int(getattr(self.config.prompt, "description_protocol_fields_per_file", 12))),
        )

    def _select_relevant_table_cards(
        self,
        table_cards: list[dict],
        *,
        terms: list[str],
        limit: int,
    ) -> list[dict]:
        wanted = [str(x).lower() for x in terms if str(x).strip()]
        if not wanted:
            return table_cards[:limit]
        scored: list[tuple[int, str, dict]] = []
        for card in table_cards:
            if not isinstance(card, dict):
                continue
            field_text = " ".join(
                f"{f.get('name', '')} {f.get('meaning', '')} {f.get('role', '')} {f.get('logical_type', '')}"
                for f in self._table_card_fields(card)
            )
            text = " ".join(
                [
                    str(card.get("table_id", "")),
                    str(card.get("source_file", "")),
                    str(card.get("sheet_name", "")),
                    str(card.get("role", "")),
                    str(card.get("file_cognition", "")),
                    field_text,
                ]
            ).lower()
            score = sum(1 for term in wanted if term in text)
            if score:
                scored.append((score, str(card.get("table_id", "")), card))
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected = [card for _, _, card in scored[:limit]]
        return selected or table_cards[:limit]

    def _table_cards_as_file_index(self, table_cards: list[dict], *, limit: int | None = None) -> list[dict]:
        """Tiny compatibility index for prompt fields still named `files`.

        Headroom-style prompts should not receive a second copy of table cards.
        This keeps only lookup metadata; field details must come from table_cards'
        tiny manifest or a dynamic retrieval.
        """

        out: list[dict] = []
        for card in list(table_cards or [])[: limit or len(table_cards or [])]:
            if not isinstance(card, dict):
                continue
            out.append(
                {
                    k: v
                    for k, v in {
                        "table_id": card.get("table_id"),
                        "source_file": card.get("source_file"),
                        "sheet_name": card.get("sheet_name"),
                        "table_kind": card.get("table_kind"),
                        "role": card.get("role"),
                        "shape": card.get("shape"),
                        "field_count": card.get("field_count"),
                    }.items()
                    if v not in (None, "", [], {})
                }
            )
        return out

    @staticmethod
    def _table_card_fields(card: dict) -> list[dict]:
        """Return prompt-visible field handles from either detail cards or manifests.

        Headroom-style stable cards use `field_index`; retrieved/detail cards may
        still use `fields`. Consumers should not assume full statistics exist.
        """

        raw_fields = card.get("fields")
        if not isinstance(raw_fields, list) or not raw_fields:
            raw_fields = card.get("field_index")
        return [field for field in (raw_fields or []) if isinstance(field, dict)]

    def _compiled_context_for_sections(
        self,
        *,
        file_summaries: list,
        downstream_context: dict,
        relations: list,
        table_limit: int | None = None,
        relation_limit: int = 60,
    ) -> dict:
        detailed_table_cards = self._table_cards_for_sections(file_summaries, limit=table_limit)
        table_cards = compact_table_cards_for_prompt(
            detailed_table_cards,
            field_limit=max(2, min(5, int(getattr(self.config.prompt, "description_protocol_fields_per_file", 12)))),
        )
        relation_cards = build_relation_cards(relations, limit=relation_limit)
        filename_group_cards = self._filename_group_cards_for_sections(
            downstream_context,
            file_summaries=file_summaries,
            limit=30,
        )
        payload = {
            "schema_version": "autorealize.compiled_context.headroom.v1",
            "context_policy": {
                "files_are_table_cards": True,
                "large_objects_local_only": True,
                "authoritative_facts_source": "authoritative_memory",
                "constraint_facts_source": "constraint_memory",
            },
            "table_cards": table_cards,
            "relations": relation_cards,
            "filename_sample_groups": filename_group_cards,
        }
        return payload

    def _field_detail_cards_for_sections(
        self,
        *,
        file_summaries: list,
        downstream_context: dict,
        relations: list,
        table_limit: int,
        field_limit: int,
    ) -> list[dict]:
        """Return bounded field-detail evidence for the field section only.

        Most description stages should see route-only table manifests. The
        `关键字段说明` section is the exception: it needs actual field meanings and
        lightweight statistics, but still not full previews or source metadata.
        """

        detailed_cards = self._table_cards_for_sections(
            file_summaries,
            limit=max(1, int(table_limit)),
            field_limit=max(1, int(field_limit)),
        )
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context.get("authoritative_memory"), dict) else {}
        constraint_memory = downstream_context.get("constraint_memory") if isinstance(downstream_context.get("constraint_memory"), dict) else {}
        authority_terms: list[str] = []
        for key in ["task_goal", "input_requirements", "output_requirements", "evaluation_requirements", "constraints"]:
            value = auth.get(key)
            if isinstance(value, list):
                authority_terms.extend(str(x) for x in value[:12])
            elif value:
                authority_terms.append(str(value))
        if isinstance(constraint_memory, dict):
            authority_terms.append(str(constraint_memory.get("summary", "")))
            authority_terms.extend(str(x) for x in (constraint_memory.get("items") or [])[:12])

        relation_terms: list[str] = []
        for rel in build_relation_cards(relations, limit=40):
            relation_terms.extend(
                str(rel.get(k, ""))
                for k in ["left_file", "left_field", "right_file", "right_field", "short_evidence"]
            )

        default_terms = [
            "id",
            "key",
            "target",
            "label",
            "score",
            "metric",
            "cost",
            "price",
            "amount",
            "quantity",
            "capacity",
            "date",
            "time",
            "output",
            "submission",
            "订单",
            "单号",
            "客户",
            "车辆",
            "车牌",
            "承运商",
            "车型",
            "成本",
            "费用",
            "合同",
            "线路",
            "路线",
            "地址",
            "省份",
            "城市",
            "区县",
            "日期",
            "时间",
            "交付",
            "提货",
            "体积",
            "重量",
            "数量",
            "容量",
            "装载",
            "限制",
            "约束",
            "目标",
            "提交",
            "输出",
        ]
        terms = default_terms + authority_terms + relation_terms
        selected = self._select_relevant_table_cards(detailed_cards, terms=terms, limit=max(1, int(table_limit)))
        return [
            compact_detail_table_card_for_prompt(card, field_limit=max(1, int(field_limit)))
            for card in selected
            if isinstance(card, dict)
        ]

    def _question_memory_pack(self, downstream_context: dict) -> dict:
        pack = downstream_context.get("agent_context_pack") if isinstance(downstream_context.get("agent_context_pack"), dict) else {}
        qmem = pack.get("question_memory") if isinstance(pack.get("question_memory"), dict) else {}
        if not qmem and isinstance(downstream_context.get("knowledge_base"), dict):
            qmem = downstream_context["knowledge_base"].get("question_investigation", {}) or {}
        answers = qmem.get("answers") if isinstance(qmem.get("answers"), list) else []
        unresolved = qmem.get("unresolved_questions") if isinstance(qmem.get("unresolved_questions"), list) else []
        records = qmem.get("question_records") if isinstance(qmem.get("question_records"), list) else []
        return {
            "summary": str(qmem.get("summary", ""))[:1200],
            "answers": [
                {
                    "question": str(a.get("question", ""))[:260] if isinstance(a, dict) else "",
                    "answer": str(a.get("answer", ""))[:500] if isinstance(a, dict) else str(a)[:500],
                    "confidence": a.get("confidence") if isinstance(a, dict) else None,
                }
                for a in answers[:12]
            ],
            "question_records": [
                {
                    "question_id": r.get("question_id"),
                    "question": str(r.get("question", ""))[:260],
                    "status": r.get("status"),
                    "short_answer": str(r.get("short_answer") or r.get("answer") or "")[:360],
                    "unresolved_reason": str(r.get("unresolved_reason", ""))[:360],
                }
                for r in records[:20]
                if isinstance(r, dict)
            ],
            "unresolved_questions": [str(x)[:360] for x in unresolved[:12]],
            "context_routing_notes": [str(x)[:260] for x in (qmem.get("context_routing_notes") or [])[:10]],
        }

    def _build_task_authority_pack(
        self,
        *,
        task_hint: str,
        problem_review: ProblemParadigmReview,
        downstream_context: dict,
    ) -> dict:
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context.get("authoritative_memory"), dict) else {}
        return {
            "task_hint": task_hint,
            "authority_priority": [
                "用户输入最高优先级",
                "已有 description.md 次之",
                "官方说明/其他说明文档再次之",
                "数据统计和 LLM 推断最低",
            ],
            "authoritative_memory": {
                "summary": str(auth.get("summary", ""))[:1200],
                "task_goal": str(auth.get("task_goal", ""))[:1000],
                "input_requirements": [str(x)[:360] for x in (auth.get("input_requirements") or [])[:10]],
                "output_requirements": [str(x)[:360] for x in (auth.get("output_requirements") or [])[:10]],
                "evaluation_requirements": [str(x)[:360] for x in (auth.get("evaluation_requirements") or [])[:10]],
                "constraints": [str(x)[:360] for x in (auth.get("constraints") or [])[:12]],
                "leakage_guards": [str(x)[:360] for x in (auth.get("leakage_guards") or [])[:8]],
                "authority_conflicts": (auth.get("authority_conflicts") or [])[:8],
                "unresolved_questions": [str(x)[:360] for x in (auth.get("unresolved_questions") or [])[:8]],
            },
            "problem_paradigm_review": problem_review.model_dump(),
            "downstream_context": {
                "train_table": downstream_context.get("train_table", ""),
                "predict_table": downstream_context.get("predict_table", ""),
                "target_column": downstream_context.get("target_column", ""),
                "id_column": downstream_context.get("id_column", ""),
                "submission_columns": downstream_context.get("submission_columns", []),
            },
            "question_memory": self._question_memory_pack(downstream_context),
        }

    def _build_evaluation_evidence_pack(
        self,
        *,
        downstream_context: dict,
        problem_review: ProblemParadigmReview,
        relations: list,
        file_summaries: list,
        protocol_bundle: DescriptionProtocolBundle | None = None,
    ) -> dict:
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context.get("authoritative_memory"), dict) else {}
        constraint_memory = downstream_context.get("constraint_memory") if isinstance(downstream_context.get("constraint_memory"), dict) else {}
        compiled = self._compiled_context_for_sections(
            file_summaries=file_summaries,
            downstream_context=downstream_context,
            relations=relations,
            table_limit=20,
            relation_limit=30,
        )
        table_cards = self._select_relevant_table_cards(
            compiled["table_cards"],
            terms=["eval", "metric", "score", "评分", "评估", "目标", "成本", "费用", "label", "target", "提交", "output", "submission"],
            limit=10,
        )
        return {
            "problem_paradigm": problem_review.model_dump(),
            "evaluation_requirements": [str(x)[:500] for x in (auth.get("evaluation_requirements") or [])[:12]],
            "objective_priority": [
                str(x)[:500]
                for x in (auth.get("constraints") or [])
                if any(k in str(x).lower() for k in ["目标", "优先", "成本", "最小", "最大", "score", "metric", "penalty", "reward"])
            ][:10],
            "submission_or_output_requirements": [str(x)[:500] for x in (auth.get("output_requirements") or [])[:12]],
            "authoritative_submission_contract": downstream_context.get("authoritative_submission_contract", {}),
            "constraint_memory": {
                "summary": str(constraint_memory.get("summary", ""))[:1000],
                "items": (constraint_memory.get("items") or [])[:16],
            },
            "leakage_facts": [str(x)[:360] for x in (auth.get("leakage_guards") or [])[:12]],
            # Compatibility key: tiny file/table handles only, not a full metadata bundle.
            "evaluation_related_files": self._table_cards_as_file_index(table_cards),
            "table_index": table_cards,
            "relations": compiled["relations"][:20],
            "filename_sample_groups": compiled["filename_sample_groups"][:12],
            "qdi_evaluation_output_constraint_conclusions": self._question_memory_pack(downstream_context),
            "protocol_evaluation_summary": (
                protocol_bundle.evaluation_summary if protocol_bundle is not None else ""
            ),
        }

    def _build_output_evidence_pack(
        self,
        *,
        data_root: Path,
        downstream_context: dict,
        problem_review: ProblemParadigmReview,
        official_sample_probe: dict,
        file_summaries: list | None = None,
        relations: list | None = None,
        evaluation_contract: EvaluationContractReview | None = None,
    ) -> dict:
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context.get("authoritative_memory"), dict) else {}
        table_cards: list[dict] = []
        relation_cards: list[dict] = []
        filename_group_cards: list[dict] = []
        if file_summaries is not None:
            compiled = self._compiled_context_for_sections(
                file_summaries=file_summaries,
                downstream_context=downstream_context,
                relations=relations or [],
                table_limit=20,
                relation_limit=30,
            )
            table_cards = self._select_relevant_table_cards(
                compiled["table_cards"],
                terms=["submission", "submit", "output", "result", "预测", "提交", "输出", "订单", "id", "target", "score"],
                limit=10,
            )
            relation_cards = compiled["relations"][:20]
            filename_group_cards = compiled["filename_sample_groups"][:12]
        return {
            "configured_generate_sample_submission": bool(getattr(self.config.switches, "generate_sample_submission", True)),
            "problem_paradigm": problem_review.model_dump(),
            "official_sample_probe": official_sample_probe,
            "authoritative_submission_contract": downstream_context.get("authoritative_submission_contract", {}),
            "submission_columns": downstream_context.get("submission_columns", []),
            "generated_submission_columns": downstream_context.get("generated_submission_columns", []),
            "output_requirements": [str(x)[:500] for x in (auth.get("output_requirements") or [])[:12]],
            "evaluation_contract": evaluation_contract.model_dump() if evaluation_contract else {},
            "qdi_output_conclusions": self._question_memory_pack(downstream_context),
            "table_index": table_cards,
            "relations": relation_cards,
            "filename_sample_groups": filename_group_cards,
            "known_output_like_files": [
                str(p.relative_to(data_root)).replace("\\", "/")
                for p in data_root.rglob("*")
                if p.is_file() and any(k in p.name.lower() for k in ["submission", "submit", "output", "result"])
            ][:12],
        }

    def _build_data_section_pack(self, *, file_summaries: list, downstream_context: dict) -> dict:
        compiled = self._compiled_context_for_sections(
            file_summaries=file_summaries,
            downstream_context=downstream_context,
            relations=[],
            table_limit=max(1, int(getattr(self.config.prompt, "description_protocol_file_limit", 16))),
            relation_limit=1,
        )
        return {
            # Compatibility key: tiny file/table handles only.
            "files": self._table_cards_as_file_index(compiled["table_cards"]),
            "filename_sample_groups": compiled["filename_sample_groups"],
            "global_reading_policy": [
                "description.md 只写必要读取方式；详细读取代码进入 automl_context。",
                "普通默认 CSV 不冗余展开读取代码；非默认 CSV、多 sheet Excel、JSON 等需要提示读取方式。",
            ],
        }

    def _build_field_section_pack(self, *, file_summaries: list, downstream_context: dict, relations: list) -> dict:
        table_limit = max(1, int(getattr(self.config.prompt, "description_protocol_file_limit", 16)))
        field_limit = max(1, int(getattr(self.config.prompt, "description_protocol_fields_per_file", 12)))
        compiled = self._compiled_context_for_sections(
            file_summaries=file_summaries,
            downstream_context=downstream_context,
            relations=relations,
            table_limit=table_limit,
            relation_limit=50,
        )
        table_field_details = self._field_detail_cards_for_sections(
            file_summaries=file_summaries,
            downstream_context=downstream_context,
            relations=relations,
            table_limit=table_limit,
            field_limit=field_limit,
        )
        return {
            "context_policy": {
                "table_index_is_route_only": True,
                "table_field_details_are_bounded_prompt_visible_evidence": True,
                "full_profiles_previews_and_source_metadata_are_local_only": True,
            },
            "table_index": compiled["table_cards"],
            "table_field_details": table_field_details,
            "relations": compiled["relations"][:40],
            "filename_sample_groups": compiled["filename_sample_groups"],
            "field_selection_policy": (
                "只写任务相关关键字段；不要求罗列所有列。字段含义优先来自 "
                "table_field_details[].fields[].meaning；若 meaning 为空，只能基于字段名、角色、"
                "类型和统计做保守说明，并标注含义未确认。"
            ),
        }

    def _build_constraint_section_pack(self, *, downstream_context: dict, relations: list) -> dict:
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context.get("authoritative_memory"), dict) else {}
        constraint_memory = downstream_context.get("constraint_memory") if isinstance(downstream_context.get("constraint_memory"), dict) else {}
        relation_cards = build_relation_cards(relations, limit=30)
        return {
            "authoritative_constraints": [str(x)[:500] for x in (auth.get("constraints") or [])[:20]],
            "leakage_guards": [str(x)[:500] for x in (auth.get("leakage_guards") or [])[:12]],
            "constraint_memory": {
                "summary": str(constraint_memory.get("summary", ""))[:1200],
                "items": (constraint_memory.get("items") or [])[:24],
            },
            "relations": relation_cards[:20],
            "qdi_constraint_conclusions": self._question_memory_pack(downstream_context),
        }

    def _build_tips_pack(self, *, downstream_context: dict) -> dict:
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context.get("authoritative_memory"), dict) else {}
        return {
            "authority_conflicts": (auth.get("authority_conflicts") or [])[:12],
            "unresolved_authority_questions": [str(x)[:360] for x in (auth.get("unresolved_questions") or [])[:12]],
            "qdi_memory": self._question_memory_pack(downstream_context),
            "mandatory_unresolved_guidance": (
                "建模、数据处理、特征工程、约束设计和评分实现应避免依赖未验证疑惑；"
                "无法避免时写成可配置假设或保守兜底并记录实验日志。"
            ),
        }

    def _ensure_section_header(self, markdown: str, title: str) -> str:
        text = self._strip_markdown_fence(markdown).strip()
        if not text:
            return f"## {title}\n- 当前材料不足，无法生成该章节。"
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        if first == f"## {title}":
            return text
        if first.startswith("## "):
            lines = text.splitlines()
            lines[0] = f"## {title}"
            return "\n".join(lines).strip()
        return f"## {title}\n{text}"

    def _generate_overview_task_definition_sections(
        self,
        *,
        task_authority_pack: dict,
        original_text: str,
        artifact_refs: dict | None = None,
    ) -> OverviewTaskDefinitionDraft:
        system = self.services.prompt_mgr.load("system/description_overview_task_definition.md")
        stable_payload = {
            "context_policy": self._headroom_context_policy(),
            "output_language_policy": self._output_language_policy(),
            "fixed_rules": [
                "Generate exactly two reader-facing sections.",
                "Do not include data field details or modeling advice in the overview.",
                "Do not invent metrics, target columns, or RL requirements.",
            ],
            "original_requirements_full": original_text,
            "task_authority_pack": task_authority_pack,
        }
        self._record_headroom_pack_telemetry(
            "description_sections_overview_task_definition",
            {
                "original_requirements_full": original_text,
                "task_authority_pack": task_authority_pack,
                "artifact_refs": artifact_refs or {},
            },
        )
        stable, dynamic = stable_dynamic_prompt(
            stable=stable_payload,
            dynamic={"instruction": "Generate frozen overview and task-definition sections."},
            stable_title="Stable overview/task-definition evidence",
            dynamic_title="Dynamic section request",
        )
        draft = self.services.llm_client.ask_structured(
            model_cls=OverviewTaskDefinitionDraft,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name="description_sections_overview_task_definition",
            max_tokens=5000,
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )
        draft.overview_markdown = self._ensure_section_header(draft.overview_markdown, "任务概述")
        draft.task_definition_markdown = self._ensure_section_header(draft.task_definition_markdown, "任务定义")
        return draft

    def _render_evaluation_section(self, contract: EvaluationContractReview, downstream_context: dict) -> str:
        direction = str(contract.metric_direction or "").strip()
        direction_text = "最小化" if direction.lower() == "minimize" else "最大化" if direction.lower() == "maximize" else direction or "未明确"
        paradigm = str(downstream_context.get("problem_paradigm", "") or "")
        true_label = "评估依据" if paradigm in {"static_optimization", "reinforcement_learning", "hybrid_ml_optimization"} else "`y_true` 来源"
        pred_label = "方案/预测来源" if paradigm in {"static_optimization", "reinforcement_learning", "hybrid_ml_optimization"} else "`y_pred` 来源"
        final_formula = str(contract.metric_formula or contract.scalar_score_formula or "").strip() or "未明确"
        lines = [
            "## 评估协议",
            "### 主指标",
            f"- 指标名称：{contract.primary_metric or '未明确'}",
            f"- 优化方向：{direction_text}",
            f"- 预测/决策单元：{contract.prediction_unit or '未明确'}",
            "### 计算公式",
            f"- 最终评分公式：{final_formula}",
        ]
        lines.extend(
            [
                "### 计算范围",
                f"- {true_label}：{contract.y_true_source or '未明确'}",
                f"- {pred_label}：{contract.y_pred_source or '未明确'}",
                f"- 覆盖范围：{contract.computation_scope or '未明确'}",
                f"- 聚合方式：{contract.aggregation_rule or '未明确'}",
                "### 验证协议",
                f"- {contract.validation_protocol or '未明确'}",
                "### 提交校验与防作弊",
            ]
        )
        for item in contract.submission_checks or ["未明确"]:
            lines.append(f"- {item}")
        lines.append("### 防泄漏要求")
        for item in contract.leakage_guards or ["未明确"]:
            lines.append(f"- {item}")
        lines.append("### 非法输出处理")
        for item in contract.invalid_solution_rules or ["未明确"]:
            lines.append(f"- {item}")
        if contract.tie_break_rules:
            lines.append("### 并列与次级目标")
            for item in contract.tie_break_rules:
                lines.append(f"- {item}")
        if contract.audit_metrics:
            lines.append("### 审计指标")
            for item in contract.audit_metrics:
                lines.append(f"- {item}")
        if not contract.passed or contract.issues or contract.fixes:
            lines.append("### 评估前置条件")
            for item in contract.issues or []:
                lines.append(f"- 当前材料限制：{item}")
            for item in contract.fixes or []:
                lines.append(f"- 正式评分前需明确：{item}")
        return "\n".join(lines).strip()

    def _generate_output_section(
        self,
        *,
        output_evidence_pack: dict,
        frozen_sections: dict,
        evaluation_contract: EvaluationContractReview,
        original_text: str,
        artifact_refs: dict | None = None,
    ) -> OutputSectionDraft:
        system = self.services.prompt_mgr.load("system/description_output_section.md")
        stable_payload = {
            "context_policy": self._headroom_context_policy(),
            "output_language_policy": self._output_language_policy(),
            "fixed_rules": [
                "Official sample/output contract has priority.",
                "Do not invent id,target if the task needs a custom solution format.",
                "Do not invent submission columns for input entities that do not exist in data or authoritative documents.",
                "Do not state unverified operational capabilities as hard rules; put them in open_issues or assumptions.",
                "Every submission column must have a source: official sample, input field, deterministic derivation, or explicit decision placeholder.",
                "Submission/output columns are generated result schema, not raw input dataframe columns; state this when the task output is a solution table.",
                "For generated tabular outputs, include a rule that downstream code should build result tables as pd.DataFrame(rows, columns=OUTPUT_COLUMNS) so empty/no-feasible solutions keep schema.",
                "Empty/all-unassigned/no-feasible solutions should be validation/scoring outcomes with counts and reasons, not pandas construction errors.",
                "Only output the output/submission section and sample_submission_spec.",
            ],
            "original_requirements_full": original_text,
            "output_evidence_pack": output_evidence_pack,
            "frozen_task_context": frozen_sections,
            "evaluation_contract": evaluation_contract.model_dump(),
        }
        self._record_headroom_pack_telemetry(
            "description_section_output_spec",
            {
                "original_requirements_full": original_text,
                "output_evidence_pack": output_evidence_pack,
                "frozen_task_context": frozen_sections,
                "evaluation_contract": evaluation_contract.model_dump(),
                "artifact_refs": artifact_refs or {},
            },
        )
        stable, dynamic = stable_dynamic_prompt(
            stable=stable_payload,
            dynamic={"instruction": "Generate output/submission section and sample_submission_spec."},
            stable_title="Stable output section evidence",
            dynamic_title="Dynamic output section request",
        )
        draft = self.services.llm_client.ask_structured(
            model_cls=OutputSectionDraft,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name="description_section_output_spec",
            max_tokens=5000,
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )
        draft.markdown = self._ensure_section_header(draft.markdown, "输出或提交格式")
        official_probe = output_evidence_pack.get("official_sample_probe") if isinstance(output_evidence_pack, dict) else {}
        if isinstance(official_probe, dict) and official_probe.get("columns"):
            cols = [str(x) for x in official_probe.get("columns", []) if str(x).strip()]
            if cols:
                draft.sample_submission_spec.source = "official_sample"
                draft.sample_submission_spec.should_generate = False
                draft.sample_submission_spec.columns = cols
                if not draft.sample_submission_spec.sample_filename:
                    draft.sample_submission_spec.sample_filename = "sample_submission.csv"
        return draft

    def _generate_generic_section(
        self,
        *,
        section_id: str,
        section_title: str,
        evidence_pack: dict,
        frozen_sections: dict,
        original_text: str,
        artifact_refs: dict | None = None,
        extra_rules: list[str] | None = None,
    ) -> DescriptionSectionDraft:
        system = self.services.prompt_mgr.load("system/description_generic_section.md").replace("{section_title}", section_title)
        stable_payload = {
            "context_policy": self._headroom_context_policy(),
            "output_language_policy": self._output_language_policy(),
            "section_id": section_id,
            "section_title": section_title,
            "fixed_rules": extra_rules or [],
            "original_requirements_full": original_text,
            "evidence_pack": evidence_pack,
            "frozen_previous_sections": frozen_sections,
        }
        self._record_headroom_pack_telemetry(
            f"description_section_{section_id}",
            {
                "original_requirements_full": original_text,
                "evidence_pack": evidence_pack,
                "frozen_previous_sections": frozen_sections,
                "artifact_refs": artifact_refs or {},
            },
        )
        stable, dynamic = stable_dynamic_prompt(
            stable=stable_payload,
            dynamic={"instruction": f"Generate only the `{section_title}` section."},
            stable_title=f"Stable {section_id} section evidence",
            dynamic_title=f"Dynamic {section_id} section request",
        )
        draft = self.services.llm_client.ask_structured(
            model_cls=DescriptionSectionDraft,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name=f"description_section_{section_id}",
            max_tokens=4500,
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )
        draft.section_id = section_id
        draft.markdown = self._ensure_section_header(draft.markdown, section_title)
        return draft

    def _compose_description_sections(self, sections: list[str]) -> str:
        clean = [self._strip_markdown_fence(s).strip() for s in sections if str(s or "").strip()]
        return finalize_description_markdown("# 赛题说明\n\n" + "\n\n".join(clean).strip() + "\n")

    def _write_submission_report(self, payload: dict) -> None:
        report = {"schema_version": "autorealize.submission_report.v1", **payload}
        write_json_safe(self.report_dir / "submission_report.json", report, indent=2)

    def _reuse_official_sample_submission(self, sample_src: Path, data_root: Path) -> dict:
        target_file = self.run_dir / "sample_submission.csv"
        probe = self._probe_sample_submission(sample_src, data_root)
        try:
            if sample_src.suffix.lower() == ".csv":
                shutil.copy2(sample_src, target_file)
            else:
                import pandas as pd
                import autorealize.pipeline as legacy

                if sample_src.suffix.lower() in {".xlsx", ".xls"}:
                    df = pd.read_excel(sample_src)
                else:
                    df = legacy.read_table(
                        sample_src,
                        json_flatten_sep=self.config.data.json_flatten_sep,
                        json_flatten_max_level=self.config.data.json_flatten_max_level,
                        json_keep_raw_nested_columns=self.config.data.json_keep_raw_nested_columns,
                    )
                df.to_csv(target_file, index=False, encoding="utf-8-sig")
            columns = self._read_sample_submission_columns(target_file)
            report = {
                "passed": True,
                "source": "official_sample_reused",
                "sample_source": probe.get("path", sample_src.name),
                "target_file": "sample_submission.csv",
                "columns": columns or probe.get("columns", []),
                "preview": probe.get("head", [])[:5],
            }
            self._write_submission_report(report)
            return report
        except Exception as exc:  # noqa: BLE001
            report = {
                "passed": False,
                "source": "official_sample_reuse_failed",
                "sample_source": probe.get("path", sample_src.name),
                "target_file": None,
                "columns": probe.get("columns", []),
                "issues": [str(exc)[:500]],
                "reason": "Official sample was found but could not be copied/converted.",
            }
            self._write_submission_report(report)
            return report

    def _select_sample_builder_table(self, data_root: Path, downstream_context: dict) -> tuple[Path | None, object | None, list[str]]:
        import autorealize.pipeline as legacy

        table_files = [p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}]
        table_files = [p for p in table_files if not self._sample_submission_name(p)]
        if not table_files:
            return None, None, ["no_table_files"]
        preferred_names = [
            str(downstream_context.get("predict_table", "") or ""),
            str(downstream_context.get("train_table", "") or ""),
        ]
        candidate = None
        for name in preferred_names:
            if not name:
                continue
            for p in table_files:
                if p.name == name or str(p.relative_to(data_root)).replace("\\", "/") == name:
                    candidate = p
                    break
            if candidate is not None:
                break
        candidate = candidate or table_files[0]
        try:
            df = legacy.read_table(
                candidate,
                json_flatten_sep=self.config.data.json_flatten_sep,
                json_flatten_max_level=self.config.data.json_flatten_max_level,
                json_keep_raw_nested_columns=self.config.data.json_keep_raw_nested_columns,
                max_rows=legacy.table_probe_sample_rows(
                    candidate,
                    configured_rows=self.config.data.table_profile_sample_rows,
                    large_threshold_bytes=self.config.data.large_table_threshold_bytes,
                ),
            )
            return candidate, df, []
        except Exception as exc:  # noqa: BLE001
            return candidate, None, [f"candidate_read_failed: {candidate.name}: {exc}"]

    def _build_sample_data_access_minipack(
        self,
        *,
        sample_spec: SampleSubmissionSpec,
        file_summaries: list,
        downstream_context: dict,
        relations: list,
    ) -> dict:
        wanted = set()
        for value in (sample_spec.source_fields or {}).values():
            text = str(value)
            for part in text.replace("\\", "/").split():
                if "." in part or "/" in part:
                    wanted.add(part.strip("`'\",:;()[]{}"))
        compiled = self._compiled_context_for_sections(
            file_summaries=file_summaries,
            downstream_context=downstream_context,
            relations=relations,
            table_limit=24,
            relation_limit=30,
        )
        terms = list(wanted) + [str(x) for x in (sample_spec.columns or []) if str(x).strip()]
        table_cards = self._select_relevant_table_cards(compiled["table_cards"], terms=terms, limit=12)
        selected_files = {
            str(card.get("source_file", "") or "")
            for card in table_cards
            if isinstance(card, dict) and str(card.get("source_file", "") or "").strip()
        }
        selected_fields = {
            str(field.get("name", "") or "")
            for card in table_cards
            if isinstance(card, dict)
            for field in self._table_card_fields(card)
            if str(field.get("name", "") or "").strip()
        }
        relation_cards = []
        for rel in compiled["relations"]:
            if not isinstance(rel, dict):
                continue
            if (
                str(rel.get("left_file", "")) in selected_files
                or str(rel.get("right_file", "")) in selected_files
                or str(rel.get("left_field", "")) in selected_fields
                or str(rel.get("right_field", "")) in selected_fields
            ):
                relation_cards.append(rel)
            if len(relation_cards) >= 20:
                break
        return {
            # Compatibility key: tiny file/table handles only.
            "files": self._table_cards_as_file_index(table_cards),
            "table_index": table_cards,
            "exact_source_schema_contract": build_data_schema_contract(file_summaries, max_tables=24),
            "relations": relation_cards,
            "filename_sample_groups": compiled["filename_sample_groups"][:12],
            "downstream_context": {
                "train_table": downstream_context.get("train_table", ""),
                "predict_table": downstream_context.get("predict_table", ""),
                "id_column": downstream_context.get("id_column", ""),
                "target_column": downstream_context.get("target_column", ""),
            },
        }

    @staticmethod
    def _exact_schema_columns(schema_contract: dict) -> set[str]:
        columns: set[str] = set()
        tables = schema_contract.get("tables") if isinstance(schema_contract.get("tables"), list) else []
        for table in tables:
            if not isinstance(table, dict):
                continue
            for col in table.get("physical_columns_exact") or []:
                text = str(col or "").strip()
                if text:
                    columns.add(text)
        return columns

    @staticmethod
    def _exact_schema_non_column_names(schema_contract: dict) -> set[str]:
        names: set[str] = set()
        tables = schema_contract.get("tables") if isinstance(schema_contract.get("tables"), list) else []
        for table in tables:
            if not isinstance(table, dict):
                continue
            for key in ["table_id", "source_file", "sheet_name", "table_kind"]:
                text = str(table.get(key, "") or "").strip()
                if text:
                    names.add(text)
        workbooks = schema_contract.get("workbooks") if isinstance(schema_contract.get("workbooks"), list) else []
        for workbook in workbooks:
            if not isinstance(workbook, dict):
                continue
            text = str(workbook.get("source_file", "") or "").strip()
            if text:
                names.add(text)
            for sheet in workbook.get("valid_sheet_names_exact") or []:
                text = str(sheet or "").strip()
                if text:
                    names.add(text)
        return names

    @staticmethod
    def _source_field_alias_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        for pattern in [r"`([^`]+)`", r"“([^”]+)”", r'"([^"]+)"']:
            for match in re.findall(pattern, text or ""):
                token = str(match or "").strip()
                if token and token not in tokens:
                    tokens.append(token)
        # Machine-facing source fields are often rendered as
        # file.xlsx::sheet::column without quotes.  The final segment is the
        # dangerous one for pandas access, so inspect every :: segment and let
        # the schema-name filter below discard file/sheet names.
        for match in re.findall(r"::([^:\s`'\"，,。；;、\)\]）】]+)", text or ""):
            token = str(match or "").strip()
            token = re.split(r"[（(]", token, maxsplit=1)[0].strip()
            if token and token not in tokens:
                tokens.append(token)
        # Also catch common source references such as "订单明细信息.标准箱数（汇总）".
        # File/sheet/table names are later discarded by the schema-name filter.
        for raw_part in re.split(r"[\s,，;；、]+", text or ""):
            part = str(raw_part or "").strip("`'\"“”‘’()（）[]【】{}<>")
            if "." not in part and "：" not in part and ":" not in part:
                continue
            for segment in re.split(r"[.。:：/\\]+", part):
                token = re.split(r"[（(]", segment, maxsplit=1)[0].strip("`'\"“”‘’()（）[]【】{}<>")
                token = re.sub(r"^(从|来自|源自|派生自|由|按|以|和|与|或)+", "", token).strip()
                if token and token not in tokens:
                    tokens.append(token)
        return tokens

    @staticmethod
    def _source_field_candidate_columns(alias: str, schema_contract: dict, *, limit: int = 6) -> list[str]:
        alias_text = str(alias or "").strip()
        if not alias_text:
            return []
        alias_norm = re.sub(r"[\s_`'\"“”‘’（）()\[\]【】{}<>:：,，.。;；、/\\\-]+", "", alias_text).lower()
        if not alias_norm:
            return []
        candidates: list[tuple[float, int, str]] = []
        tables = schema_contract.get("tables") if isinstance(schema_contract.get("tables"), list) else []
        seen: set[str] = set()
        for table in tables:
            if not isinstance(table, dict):
                continue
            meanings: dict[str, str] = {}
            for field in table.get("field_summaries") or []:
                if isinstance(field, dict):
                    name = str(field.get("name", "") or "")
                    if name:
                        meanings[name] = str(field.get("meaning", "") or "")
            for idx, raw_col in enumerate(table.get("physical_columns_exact") or []):
                col = str(raw_col or "").strip()
                if not col or col in seen:
                    continue
                col_norm = re.sub(r"[\s_`'\"“”‘’（）()\[\]【】{}<>:：,，.。;；、/\\\-]+", "", col).lower()
                meaning_norm = re.sub(
                    r"[\s_`'\"“”‘’（）()\[\]【】{}<>:：,，.。;；、/\\\-]+",
                    "",
                    meanings.get(col, ""),
                ).lower()
                score = difflib.SequenceMatcher(None, alias_norm, col_norm).ratio()
                direct_signal = False
                if "标准箱" in alias_text and "标箱" in col:
                    score = max(score, 0.88)
                    direct_signal = True
                if alias_norm in col_norm or col_norm in alias_norm:
                    score = max(score, 0.82)
                    direct_signal = True
                if alias_norm and alias_norm in meaning_norm:
                    score = max(score, 0.78)
                    direct_signal = True
                if alias_norm.endswith("日") and alias_norm[:-1] and alias_norm[:-1] in col_norm and (
                    "时间" in col or "日期" in col or "日" in col
                ):
                    score = max(score, 0.74)
                    direct_signal = True
                if len(alias_norm) <= 2:
                    short_signal = direct_signal and alias_norm in col_norm
                    if alias_norm == "重量" and any(term in col for term in ["重量", "载重", "毛重"]):
                        short_signal = True
                    if alias_norm == "体积" and "体积" in col:
                        short_signal = True
                    if not short_signal:
                        continue
                if score >= 0.45 and (direct_signal or len(alias_norm) > 2):
                    candidates.append((-score, idx, col))
                    seen.add(col)
        candidates.sort()
        return [col for _score, _idx, col in candidates[: max(1, int(limit))]]

    @staticmethod
    def _resolve_exact_source_field_alias(
        alias: str,
        *,
        schema_contract: dict,
        physical_columns: list[str],
        non_column_names: set[str],
    ) -> tuple[str | None, list[str]]:
        field = str(alias or "").strip()
        if not field or field in physical_columns or field in non_column_names:
            return None, []
        # Very short business words such as "重量" or "体积" are commonly
        # ambiguous across order, line-item, vehicle, and contract tables. Keep
        # them as unresolved with candidates instead of silently choosing one.
        if len(field) <= 2:
            return None, TaskDefinitionModule._source_field_candidate_columns(field, schema_contract)
        matches = difflib.get_close_matches(field, physical_columns, n=1, cutoff=0.72)
        if matches:
            return matches[0], []
        candidates = TaskDefinitionModule._source_field_candidate_columns(field, schema_contract)
        if len(candidates) == 1:
            candidate = candidates[0]
            ratio = difflib.SequenceMatcher(None, field, candidate).ratio()
            if ratio >= 0.72:
                return candidate, []
        return None, candidates

    @staticmethod
    def _replace_source_field_alias(text: str, old: str, new: str) -> str:
        out = str(text or "")
        old = str(old or "").strip()
        new = str(new or "").strip()
        if not old or not new or old == new:
            return out
        for left, right in [("`", "`"), ("“", "”"), ('"', '"')]:
            out = out.replace(f"{left}{old}{right}", f"{left}{new}{right}")
        out = re.sub(rf"(?<=::){re.escape(old)}(?=($|[（(\s`'\"，,。；;、\)\]）】]))", new, out)
        # Safe bare-word replacement for Chinese aliases in source-field prose.
        # The negative CJK/word boundaries avoid changing generated output
        # columns such as "总标准箱数".
        out = re.sub(
            rf"(?<![\w\u4e00-\u9fff]){re.escape(old)}(?![\w\u4e00-\u9fff])",
            new,
            out,
        )
        return out

    @staticmethod
    def _apply_source_field_alias_corrections_to_text(text: str, corrections: list[dict]) -> str:
        out = str(text or "")
        for item in corrections or []:
            out = TaskDefinitionModule._replace_source_field_alias(
                out,
                str(item.get("from", "") or ""),
                str(item.get("to", "") or ""),
            )
        return out

    @staticmethod
    def _apply_source_field_alias_corrections_to_contract(
        contract: EvaluationContractReview,
        corrections: list[dict],
    ) -> EvaluationContractReview:
        if not corrections:
            return contract
        for attr in [
            "metric_formula",
            "scalar_score_formula",
            "prediction_unit",
            "y_true_source",
            "y_pred_source",
            "computation_scope",
            "aggregation_rule",
            "validation_protocol",
            "rationale",
        ]:
            value = getattr(contract, attr, "")
            if isinstance(value, str) and value:
                setattr(contract, attr, TaskDefinitionModule._apply_source_field_alias_corrections_to_text(value, corrections))
        for attr in [
            "submission_checks",
            "leakage_guards",
            "invalid_solution_rules",
            "tie_break_rules",
            "audit_metrics",
            "issues",
            "fixes",
            "evidence",
        ]:
            values = getattr(contract, attr, None)
            if isinstance(values, list):
                setattr(
                    contract,
                    attr,
                    [
                        TaskDefinitionModule._apply_source_field_alias_corrections_to_text(str(value), corrections)
                        for value in values
                    ],
                )
        existing = list(contract.evidence or [])
        for item in corrections[:8]:
            note = (
                "AutoRealize corrected source field alias "
                f"`{item.get('from')}` -> `{item.get('to')}` using the exact source schema contract."
            )
            if note not in existing:
                existing.append(note)
        contract.evidence = existing
        return contract

    @staticmethod
    def _correct_sample_spec_source_fields(
        sample_spec: SampleSubmissionSpec,
        *,
        schema_contract: dict,
    ) -> list[dict]:
        """Repair source_fields that use a prose alias instead of an exact column.

        The LLM may correctly understand a business concept but spell it as a
        semantic alias. Downstream code needs exact physical column names, so we
        conservatively rewrite backticked aliases only when there is a strong
        fuzzy match in the deterministic schema contract.
        """

        physical_columns = sorted(TaskDefinitionModule._exact_schema_columns(schema_contract))
        physical_column_set = set(physical_columns)
        non_column_names = TaskDefinitionModule._exact_schema_non_column_names(schema_contract)
        if not physical_columns or not sample_spec.source_fields:
            return []

        corrections: list[dict] = []
        unresolved: list[dict] = []
        updated: dict[str, str] = {}
        for output_col, raw in sample_spec.source_fields.items():
            text = str(raw or "")
            new_text = text
            for token in TaskDefinitionModule._source_field_alias_tokens(text):
                field = str(token or "").strip()
                if not field or field in physical_column_set or field in non_column_names:
                    continue
                replacement, candidates = TaskDefinitionModule._resolve_exact_source_field_alias(
                    field,
                    schema_contract=schema_contract,
                    physical_columns=physical_columns,
                    non_column_names=non_column_names,
                )
                if not replacement:
                    unresolved.append(
                        {
                            "output_column": str(output_col),
                            "alias": field,
                            "reason": "source_fields alias is not an exact physical column and no safe schema correction was found",
                            "candidate_exact_columns": candidates[:6],
                        }
                    )
                    continue
                if replacement == field:
                    continue
                new_text = TaskDefinitionModule._replace_source_field_alias(new_text, field, replacement)
                corrections.append(
                    {
                        "output_column": str(output_col),
                        "from": field,
                        "to": replacement,
                        "reason": "source_fields alias was not an exact physical column; corrected using exact schema contract",
                    }
                )
            updated[str(output_col)] = new_text
        if corrections or unresolved:
            sample_spec.source_fields = updated
            existing_rules = list(sample_spec.validation_rules or [])
            for item in corrections[:8]:
                existing_rules.append(
                    "AutoRealize corrected source field alias "
                    f"`{item['from']}` -> `{item['to']}` for output column `{item['output_column']}`."
                )
            for item in unresolved[:8]:
                existing_rules.append(
                    "AutoRealize could not resolve source field alias "
                    f"`{item['alias']}` for output column `{item['output_column']}` to an exact physical column; "
                    + (
                        "candidate exact columns: "
                        + ", ".join(f"`{x}`" for x in item.get("candidate_exact_columns", [])[:6])
                        + "; "
                        if item.get("candidate_exact_columns")
                        else ""
                    )
                    + "do not access it as a raw dataframe column. Derive it from exact source fields, provide an evaluation/config rule, or define an explicit exclusion rule."
                )
            sample_spec.validation_rules = list(dict.fromkeys(existing_rules))
        return corrections

    @staticmethod
    def _apply_source_field_corrections_to_output_markdown(
        markdown: str,
        sample_spec: SampleSubmissionSpec,
        corrections: list[dict],
    ) -> str:
        """Keep the reader-facing output section aligned with corrected source fields."""

        text = str(markdown or "")
        for item in corrections or []:
            old = str(item.get("from", "") or "").strip()
            new = str(item.get("to", "") or "").strip()
            if not old or not new or old == new:
                continue
            text = TaskDefinitionModule._replace_source_field_alias(text, old, new)

        correction_rules = [
            str(rule)
            for rule in (sample_spec.validation_rules or [])
            if str(rule).startswith("AutoRealize corrected source field alias")
            or str(rule).startswith("AutoRealize could not resolve source field alias")
        ]
        if correction_rules and "### 源字段精确性修正" not in text:
            lines = [
                text.rstrip(),
                "",
                "### 源字段精确性修正",
                "- 本节中的输出列是生成结果 schema，不是输入 DataFrame 原始列；下游读取源数据时必须使用精确物理字段名。",
            ]
            lines.extend(f"- {rule}" for rule in correction_rules[:12])
            return "\n".join(lines).strip()
        return text.strip()

    @staticmethod
    def _collect_source_alias_guard(
        *,
        sample_spec: SampleSubmissionSpec,
        schema_contract: dict,
        downstream_context: dict,
    ) -> list[dict]:
        physical_columns = sorted(TaskDefinitionModule._exact_schema_columns(schema_contract))
        physical_column_set = set(physical_columns)
        non_column_names = TaskDefinitionModule._exact_schema_non_column_names(schema_contract)
        if not physical_columns:
            return []

        alias_sources: dict[str, set[str]] = {}

        def add(alias: str, source: str) -> None:
            field = str(alias or "").strip().strip("`'\"“”‘’")
            if not field or field in physical_column_set or field in non_column_names:
                return
            if len(field) <= 1:
                return
            alias_sources.setdefault(field, set()).add(source)

        for output_col, raw in (sample_spec.source_fields or {}).items():
            for token in TaskDefinitionModule._source_field_alias_tokens(str(raw or "")):
                add(token, f"sample_submission_spec.source_fields.{output_col}")

        constraint_memory = downstream_context.get("constraint_memory")
        items = constraint_memory.get("items", []) if isinstance(constraint_memory, dict) else []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for field in item.get("related_fields") or []:
                add(str(field), f"constraint_memory.items[{idx}].related_fields")

        authoritative = downstream_context.get("authoritative_memory")
        if isinstance(authoritative, dict):
            for key in ["output_requirements", "evaluation_requirements", "constraints"]:
                for text in authoritative.get(key, []) if isinstance(authoritative.get(key), list) else []:
                    for token in TaskDefinitionModule._source_field_alias_tokens(str(text or "")):
                        add(token, f"authoritative_memory.{key}")

        guard: list[dict] = []
        for alias in sorted(alias_sources):
            replacement, candidates = TaskDefinitionModule._resolve_exact_source_field_alias(
                alias,
                schema_contract=schema_contract,
                physical_columns=physical_columns,
                non_column_names=non_column_names,
            )
            entry: dict[str, object] = {
                "alias": alias,
                "source": sorted(alias_sources[alias])[:6],
            }
            if replacement:
                entry.update(
                    {
                        "status": "corrected_to_exact_physical_column",
                        "exact_physical_column": replacement,
                        "rule": "Use the exact physical column for raw pandas access; keep the alias only as business wording or a derived/output name.",
                    }
                )
            else:
                entry.update(
                    {
                        "status": "unresolved_business_concept",
                        "candidate_exact_columns": candidates[:8],
                        "rule": "Do not access this alias as a raw column. Derive it from exact fields if the derivation is specified; otherwise treat it as unresolved/optional and report the assumption in validation details.",
                    }
                )
            guard.append(entry)
        return guard

    @staticmethod
    def _apply_source_alias_guard_to_description(desc: str, guard: list[dict]) -> str:
        text = str(desc or "")
        aliases = {str(item.get("alias", "")): item for item in guard if isinstance(item, dict)}
        if "标准箱数" in aliases and aliases["标准箱数"].get("exact_physical_column"):
            exact = str(aliases["标准箱数"].get("exact_physical_column"))
            text = text.replace("`标准箱数` 列", f"`{exact}` 列（业务含义为标准箱数量）")
            text = text.replace("订单明细信息.标准箱数", f"订单明细信息.{exact}")
        if "交付日" in aliases:
            text = text.replace(
                "由求解器根据订单的交付日字段（`订单表信息` 或 `订单明细信息` 中的 `交付日` 列）确定",
                "由求解器从订单时间字段派生；当前 exact schema 未发现名为 `交付日` 的物理列，优先从 `要求交付时间` 取日期部分，并在缺失时按评估规则记录不可确认/未分配原因",
            )
            text = text.replace(
                "`交付日` 列",
                "`要求交付时间` 等精确时间字段派生出的交付日（源表不存在同名 `交付日` 物理列）",
            )
        if "是否可合并" in aliases:
            text = text.replace(
                "被标记为不能合并的订单不能与其他订单合并（对应字段 `是否可合并`）。",
                "被标记为不能合并的订单不能与其他订单合并；当前 exact schema 未发现同名物理列 `是否可合并`，不得直接读取该列，无法从权威字段确认时应保守单独成车或在验证摘要中记录待确认假设。",
            )
        if guard and "## Source Alias Guard" not in text and "## 源字段别名警戒" not in text:
            lines = [
                text.rstrip(),
                "",
                "## 源字段别名警戒",
                "- 下列名称在业务说明中出现，但不一定是输入表的物理列；下游代码必须先查 Exact Source Schema Contract，不得直接 `df[业务名]`。",
            ]
            for item in guard[:12]:
                alias = item.get("alias")
                status = item.get("status")
                exact = item.get("exact_physical_column")
                candidates = item.get("candidate_exact_columns")
                if exact:
                    lines.append(f"- `{alias}`: 使用精确物理列 `{exact}` 读取；业务词只作解释或派生名。")
                elif candidates:
                    lines.append(f"- `{alias}`: 未确认同名物理列；候选精确字段 {candidates}，需显式派生或保守记录假设。")
                else:
                    lines.append(f"- `{alias}`: 未找到可安全映射的物理列；不得直接读取，需作为业务规则/待确认事项处理。")
            text = "\n".join(lines).strip()
        return text

    def _artifact_ref_index(self, ref: dict) -> dict:
        if not isinstance(ref, dict):
            return {}
        return {
            k: ref.get(k)
            for k in [
                "artifact_id",
                "artifact_type",
                "source",
                "truncated",
                "original_chars",
                "visible_chars",
                "artifact_path",
            ]
            if ref.get(k) not in (None, "", [], {})
        }

    def _put_context_artifact(self, artifact_type: str, source: str, payload: object) -> dict:
        if payload in (None, "", [], {}):
            return {}
        ref = self._context_artifact_store().put(
            artifact_type,
            source,
            payload,
            visible_excerpt="",
            visible_limit=1,
        )
        return self._artifact_ref_index(ref)

    def _task_definition_artifact_refs(
        self,
        *,
        original_text: str = "",
        data_description_text: str = "",
        current_description: str = "",
        protocol_bundle: DescriptionProtocolBundle | dict | None = None,
        downstream_context: dict | None = None,
    ) -> dict:
        refs: dict[str, dict] = {}
        original_ref = self._put_context_artifact("original_requirements_full", "realize_report/original_requirements.txt", original_text)
        if original_ref:
            refs["original_requirements"] = original_ref
        data_ref = self._put_context_artifact("data_description_full", "realize_report/data_description.md", data_description_text)
        if data_ref:
            refs["data_description"] = data_ref
        if current_description:
            desc_ref = self._put_context_artifact("description_draft_full", "description.md/current_draft", current_description)
            if desc_ref:
                refs["current_description"] = desc_ref
        if protocol_bundle:
            payload = protocol_bundle.model_dump() if hasattr(protocol_bundle, "model_dump") else protocol_bundle
            proto_ref = self._put_context_artifact(
                "description_protocol_bundle_full",
                "realize_report/description_protocol_bundle.json",
                payload,
            )
            if proto_ref:
                refs["description_protocol_bundle"] = proto_ref
        if isinstance(downstream_context, dict):
            for key, artifact_type, source in [
                ("agent_context_pack", "agent_context_pack_full", "data_cognition.agent_context_pack"),
                ("automl_context_pack", "automl_context_pack_full", "realize_report/automl_context_pack.json"),
                ("main_task_protocol", "main_task_protocol_ref", "realize_report/main_task_protocol.json"),
            ]:
                value = downstream_context.get(key)
                if value not in (None, "", [], {}):
                    ref = self._put_context_artifact(artifact_type, source, value)
                    if ref:
                        refs[key] = ref
        return refs

    def _headroom_context_policy(self) -> dict:
        return {
            "original_requirements_full_is_authoritative_and_present": True,
            "legacy_data_digest_not_in_prompt": True,
            "large_local_artifacts_not_visible_in_prompt": True,
            "use_only_visible_evidence_packs": True,
            "stable_prompt_order": "fixed rules/schema -> original requirements -> compact evidence pack; dynamic defects/errors last",
        }

    def _output_language_policy(self) -> dict:
        lang = str(getattr(self.config.prompt, "output_language", "zh") or "zh").strip().lower()
        if lang in {"zh", "cn", "chinese", "中文"}:
            return {
                "target_language": "zh",
                "prose_language": "Chinese",
                "rules": [
                    "无论原始输入是中文、英文还是中英混合，最终 description 的正文解释必须统一使用中文。",
                    "允许保留原始代码、函数名、文件名、Sheet 名、字段名、列名、指标名、JSON key、正则表达式和必要 API 参数。",
                    "不要把英文业务短语直接混入普通说明文字；如需解释英文材料，应翻译或用中文转述。",
                ],
            }
        if lang in {"en", "english"}:
            return {
                "target_language": "en",
                "prose_language": "English",
                "rules": [
                    "Regardless of whether the source material is Chinese, English, or mixed, reader-facing description prose must be written in English.",
                    "Keep original code, function names, file names, sheet names, field names, column names, metric names, JSON keys, regular expressions, and necessary API parameters unchanged.",
                    "Do not mix Chinese prose into ordinary explanations; translate or explain Chinese source material in English.",
                ],
            }
        return {
            "target_language": "auto",
            "prose_language": "follow source material",
            "rules": [
                "No global prose language is enforced because prompt.output_language=auto.",
                "Still avoid unnecessary mixed-language prose within one sentence unless it is a field name, file name, code identifier, or required technical term.",
            ],
        }

    def _headroom_pack_telemetry(self, packs: dict) -> dict:
        telemetry: dict[str, dict] = {}
        for name, payload in packs.items():
            if payload in (None, "", [], {}):
                continue
            if isinstance(payload, dict):
                telemetry[name] = context_telemetry(payload)
            else:
                text = dumps_json_safe(payload, sort_keys=True)
                telemetry[name] = {
                    "chars": len(text),
                    "estimated_tokens": max(1, len(text) // 4) if text else 0,
                    "sha256_16": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16],
                    "artifact_refs": 0,
                    "top_level_chars": {},
                    "contains_forbidden_large_keys": [],
                    "headroom_warnings": ["large_text_pack_over_120k_chars"] if len(text) > 120_000 else [],
                }
        return telemetry

    def _record_headroom_pack_telemetry(self, stage: str, packs: dict) -> None:
        """Write pack observability locally instead of sending it to the LLM.

        Headroom keeps compression telemetry out of the model context. We follow
        the same rule here: prompts carry only task evidence; artifact indexes,
        size/hash/warning details go to a local JSONL ledger for cost analysis.
        """

        if not stage:
            stage = "unknown"
        payload = {
            "stage": stage,
            "packs": self._headroom_pack_telemetry(packs),
        }
        path = self.report_dir / "context_pack_telemetry.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(dumps_json_safe(payload, sort_keys=True) + "\n")

    def _downstream_signal_pack(self, downstream_context: dict) -> dict:
        return {
            "task_hint": downstream_context.get("task_hint", ""),
            "task_type_hint": downstream_context.get("task_type_hint", ""),
            "problem_paradigm": downstream_context.get("problem_paradigm", ""),
            "target_column": downstream_context.get("target_column", ""),
            "y_true_field": downstream_context.get("y_true_field", ""),
            "id_column": downstream_context.get("id_column", ""),
            "train_table": downstream_context.get("train_table", ""),
            "predict_table": downstream_context.get("predict_table", ""),
            "submission_columns": downstream_context.get("submission_columns", []),
            "generated_submission_columns": downstream_context.get("generated_submission_columns", []),
            "generate_sample_submission": downstream_context.get("generate_sample_submission", True),
            "sample_submission_available": downstream_context.get("sample_submission_available", False),
            "submission_contract_source": downstream_context.get("submission_contract_source", ""),
        }

    def _table_card_index_for_task_prompts(
        self,
        table_cards: list[dict],
        *,
        limit: int = 16,
        field_limit: int = 8,
    ) -> list[dict]:
        out: list[dict] = []
        for card in table_cards[: max(1, int(limit))]:
            if not isinstance(card, dict):
                continue
            fields = []
            for field in self._table_card_fields(card)[: max(1, int(field_limit))]:
                if not isinstance(field, dict):
                    continue
                fields.append(
                    {
                        k: field.get(k)
                        for k in ["name", "role", "logical_type"]
                        if field.get(k) not in (None, "", [], {})
                    }
                )
            out.append(
                {
                    k: v
                    for k, v in {
                        "table_id": card.get("table_id"),
                        "source_file": card.get("source_file"),
                        "sheet_name": card.get("sheet_name"),
                        "table_kind": card.get("table_kind"),
                        "role": card.get("role"),
                        "file_cognition": str(card.get("file_cognition", "") or "")[:220],
                        "shape": card.get("shape"),
                        "field_index": fields,
                        "reading_notes": card.get("reading_notes"),
                        "warnings": [str(x)[:180] for x in (card.get("warnings") or [])[:3]],
                    }.items()
                    if v not in (None, "", [], {})
                }
            )
        return out

    def _build_problem_paradigm_pack(
        self,
        *,
        downstream_context: dict,
        file_summaries: list | None = None,
        relations: list | None = None,
    ) -> dict:
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context.get("authoritative_memory"), dict) else {}
        constraint_memory = downstream_context.get("constraint_memory") if isinstance(downstream_context.get("constraint_memory"), dict) else {}
        compiled = self._compiled_context_for_sections(
            file_summaries=file_summaries or [],
            downstream_context=downstream_context,
            relations=relations or [],
            table_limit=14,
            relation_limit=20,
        )
        table_index = self._table_card_index_for_task_prompts(compiled["table_cards"], limit=14, field_limit=6)
        return {
            "purpose": "Classify whether this task is prediction, static optimization, RL, hybrid, or executable-unknown.",
            "downstream_signals": self._downstream_signal_pack(downstream_context),
            "authoritative_memory": compact_authoritative_memory(auth),
            "constraint_memory": compact_constraint_memory(constraint_memory),
            "authoritative_submission_contract": downstream_context.get("authoritative_submission_contract", {}),
            "table_index": table_index,
            "relations": compiled["relations"][:16],
            "filename_sample_groups": compiled["filename_sample_groups"][:10],
            "question_memory": self._question_memory_pack(downstream_context),
            "classification_clues": {
                "prediction_clues": ["train/test split", "target/label/y_true", "sample submission with id+target"],
                "optimization_clues": ["decision variables", "constraints", "cost/objective", "feasible solution"],
                "rl_clues": ["state/action/reward/episode/policy/environment"],
                "hybrid_clues": ["ML prediction feeds an optimization policy or solver"],
            },
        }

    def _build_description_protocol_pack(
        self,
        *,
        problem_review: ProblemParadigmReview,
        downstream_context: dict,
        deterministic_data_access: object,
        file_summaries: list | None = None,
        relations: list | None = None,
    ) -> dict:
        auth = downstream_context.get("authoritative_memory") if isinstance(downstream_context.get("authoritative_memory"), dict) else {}
        constraint_memory = downstream_context.get("constraint_memory") if isinstance(downstream_context.get("constraint_memory"), dict) else {}
        compiled = self._compiled_context_for_sections(
            file_summaries=file_summaries or [],
            downstream_context=downstream_context,
            relations=relations or [],
            table_limit=max(1, int(getattr(self.config.prompt, "description_protocol_file_limit", 16))),
            relation_limit=30,
        )
        table_index = self._table_card_index_for_task_prompts(compiled["table_cards"], limit=18, field_limit=8)
        pack = {
            "problem_paradigm_review": problem_review.model_dump(),
            "downstream_signals": self._downstream_signal_pack(downstream_context),
            "authoritative_memory": compact_authoritative_memory(auth),
            "constraint_memory": compact_constraint_memory(constraint_memory),
            "authoritative_submission_contract": downstream_context.get("authoritative_submission_contract", {}),
            "question_memory": self._question_memory_pack(downstream_context),
            "data_access_inventory": self._compact_data_access_inventory(deterministic_data_access),
            "table_index": table_index,
            "relations": compiled["relations"][:24],
            "filename_sample_groups": compiled["filename_sample_groups"][:12],
        }
        return pack

    def _validate_sample_against_spec(self, out_df, sample_spec: SampleSubmissionSpec) -> list[str]:
        issues: list[str] = []
        cols = [str(c) for c in out_df.columns.tolist()]
        expected = [str(x) for x in sample_spec.columns if str(x).strip()]
        if expected and cols != expected:
            issues.append(f"column_order_mismatch: got={cols}, expected={expected}")
        if len(out_df) == 0:
            issues.append("empty_submission")
        return issues

    def _readonly_sample_script_issues(self, code: str) -> list[str]:
        """Reject obvious write/delete/process/network operations before exec."""
        issues: list[str] = []
        try:
            tree = ast.parse(code or "")
        except SyntaxError as exc:
            return [f"python_syntax_error: {exc}"]
        blocked_import_roots = {"os", "subprocess", "shutil", "socket", "requests", "urllib", "http", "ftplib"}
        blocked_call_names = {
            "open",
            "exec",
            "eval",
            "compile",
            "__import__",
            "remove",
            "unlink",
            "rmdir",
            "rmtree",
            "rename",
            "replace",
            "move",
            "copy",
            "copy2",
            "to_csv",
            "to_excel",
            "to_json",
            "to_parquet",
            "to_pickle",
            "dump",
            "dumps",
        }
        allowed_write_calls = {"print"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                else:
                    names = [(node.module or "").split(".")[0]]
                for name in names:
                    if name in blocked_import_roots:
                        issues.append(f"blocked_import: {name}")
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in blocked_call_names and name not in allowed_write_calls:
                    issues.append(f"blocked_call: {name}")
        return list(dict.fromkeys(issues))

    def _validate_sample_submission_with_llm(
        self,
        *,
        sample_spec: SampleSubmissionSpec,
        generated_columns: list[str],
        generated_preview: list[dict],
        frozen_context: dict,
        rule_issues: list[str],
        original_text: str,
        artifact_refs: dict | None = None,
    ) -> SampleSubmissionValidationResult:
        system = self.services.prompt_mgr.load("system/sample_submission_validator.md")
        stable_payload = {
            "context_policy": self._headroom_context_policy(),
            "original_requirements_full": original_text,
            "sample_submission_spec": sample_spec.model_dump(),
            "frozen_task_eval_output_context": frozen_context,
            "rules": [
                "Check schema/meaning only; placeholder values are allowed.",
                "Official columns, when present, are strict.",
            ],
        }
        self._record_headroom_pack_telemetry(
            "sample_submission_spec_validator",
            {
                "original_requirements_full": original_text,
                "sample_submission_spec": sample_spec.model_dump(),
                "frozen_task_eval_output_context": frozen_context,
                "artifact_refs": artifact_refs or {},
            },
        )
        stable, dynamic = stable_dynamic_prompt(
            stable=stable_payload,
            dynamic={
                "generated_columns": generated_columns,
                "generated_preview": generated_preview[:10],
                "rule_issues": rule_issues,
            },
            stable_title="Stable sample validator evidence",
            dynamic_title="Dynamic generated sample candidate",
            dynamic_limit=10000,
        )
        return self.services.llm_client.ask_structured(
            model_cls=SampleSubmissionValidationResult,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name="sample_submission_spec_validator",
            max_tokens=3000,
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )

    def _generate_sample_submission_from_spec(
        self,
        *,
        data_root: Path,
        sample_spec: SampleSubmissionSpec,
        file_summaries: list,
        downstream_context: dict,
        relations: list,
        frozen_context: dict,
        original_text: str,
        artifact_refs: dict | None = None,
    ) -> dict:
        import autorealize.pipeline as legacy

        target_file = self.run_dir / "sample_submission.csv"
        candidate, df, read_issues = self._select_sample_builder_table(data_root, downstream_context)
        if df is None:
            report = {
                "passed": False,
                "source": "sample_spec_generation_failed",
                "target_file": None,
                "columns": sample_spec.columns,
                "issues": read_issues,
                "reason": "Could not read a candidate input table for spec-driven sample generation.",
            }
            self._write_submission_report(report)
            return report

        minipack = self._build_sample_data_access_minipack(
            sample_spec=sample_spec,
            file_summaries=file_summaries,
            downstream_context=downstream_context,
            relations=relations,
        )
        stable_payload = {
            "context_policy": self._headroom_context_policy(),
            "original_requirements_full": original_text,
            "sample_submission_spec": sample_spec.model_dump(),
            "data_access_minipack": minipack,
            "candidate_table": str(candidate.relative_to(data_root)).replace("\\", "/") if candidate else "",
            "candidate_columns": [str(c) for c in df.columns.tolist()],
            "candidate_shape_sampled": [int(df.shape[0]), int(df.shape[1])],
            "frozen_task_eval_output_context": frozen_context,
        }
        self._record_headroom_pack_telemetry(
            "sample_submission_spec_builder",
            {
                "original_requirements_full": original_text,
                "data_access_minipack": minipack,
                "sample_submission_spec": sample_spec.model_dump(),
                "frozen_task_eval_output_context": frozen_context,
                "artifact_refs": artifact_refs or {},
            },
        )

        def _ask_plan(dynamic_payload: dict, round_idx: int) -> SubmissionScriptPlan:
            stable, dynamic = stable_dynamic_prompt(
                stable=stable_payload,
                dynamic=dynamic_payload,
                stable_title="Stable sample builder evidence",
                dynamic_title="Dynamic sample builder request",
                dynamic_limit=12000,
            )
            plan = self.services.llm_client.ask_structured(
                model_cls=SubmissionScriptPlan,
                system_prompt=self.services.prompt_mgr.load("system/sample_submission_script_builder.md"),
                user_prompt=dynamic,
                prompt_name=f"sample_submission_spec_builder_{round_idx}",
                max_tokens=5000,
                static_context_prompt=stable,
                dynamic_user_prompt=dynamic,
            )
            if sample_spec.columns and plan.submission_columns != sample_spec.columns:
                plan.submission_columns = list(sample_spec.columns)
            return plan

        current_plan = _ask_plan({"instruction": "Generate initial sample_submission builder script."}, 0)
        max_repairs = max(3, int(getattr(self.config.prompt, "description_quality_max_retries", 3)))
        max_validator_rounds = 3
        validator_rounds = 0
        last_issues: list[str] = []
        for round_idx in range(max_repairs + 1):
            static_issues = self._readonly_sample_script_issues(current_plan.python_code)
            if static_issues:
                last_issues = static_issues
                if round_idx >= max_repairs:
                    break
                current_plan = _ask_plan(
                    {
                        "instruction": "Repair previous script; it violated the read-only/static safety policy.",
                        "previous_plan": current_plan.model_dump(),
                        "static_issues": static_issues,
                        "repair_round": round_idx + 1,
                    },
                    round_idx + 1,
                )
                continue
            out_df, plan_issues = legacy._try_build_submission_from_plan(plan=current_plan, df=df, data_root=data_root)
            if out_df is None:
                last_issues = plan_issues
                if round_idx >= max_repairs:
                    break
                current_plan = _ask_plan(
                    {
                        "instruction": "Repair previous script; it failed during execution.",
                        "previous_plan": current_plan.model_dump(),
                        "execution_issues": plan_issues,
                        "repair_round": round_idx + 1,
                    },
                    round_idx + 1,
                )
                continue
            expected = [str(x) for x in sample_spec.columns if str(x).strip()]
            if expected:
                for col in expected:
                    if col not in out_df.columns:
                        out_df[col] = sample_spec.default_values.get(col, "")
                out_df = out_df[expected]
            if not bool(downstream_context.get("predict_table", "")):
                out_df = out_df.head(max(1, int(self.config.data.generated_sample_submission_max_rows))).copy()
            rule_issues = self._validate_sample_against_spec(out_df, sample_spec) + list(plan_issues or [])
            generated_columns = [str(c) for c in out_df.columns.tolist()]
            generated_preview = out_df.head(min(10, len(out_df))).astype(object).where(out_df.notna(), None).to_dict(orient="records")
            validator = self._validate_sample_submission_with_llm(
                sample_spec=sample_spec,
                generated_columns=generated_columns,
                generated_preview=generated_preview,
                frozen_context=frozen_context,
                rule_issues=rule_issues,
                original_text=original_text,
                artifact_refs=artifact_refs,
            )
            validator_rounds += 1
            blocking = rule_issues + [str(x) for x in (validator.issues or []) if str(x).strip()]
            if validator.passed and not rule_issues and not validator.needs_regenerate:
                out_df.to_csv(target_file, index=False, encoding="utf-8-sig")
                report = {
                    "passed": True,
                    "source": "sample_spec_builder+validator",
                    "target_file": "sample_submission.csv",
                    "candidate_table": str(candidate.relative_to(data_root)).replace("\\", "/") if candidate else "",
                    "columns": generated_columns,
                    "preview": generated_preview[:5],
                    "round": round_idx,
                    "validator_rounds": validator_rounds,
                    "issues": [],
                    "purpose": current_plan.purpose,
                }
                self._write_submission_report(report)
                return report
            last_issues = blocking or validator.fixes or ["validator_rejected_sample_submission"]
            if round_idx >= max_repairs or validator_rounds >= max_validator_rounds:
                break
            if validator.revised_python_code.strip():
                current_plan = SubmissionScriptPlan(
                    purpose=(current_plan.purpose or "") + " | revised_by_spec_validator",
                    submission_columns=expected or generated_columns,
                    python_code=validator.revised_python_code,
                    id_column=current_plan.id_column,
                    target_columns=current_plan.target_columns,
                )
            else:
                current_plan = _ask_plan(
                    {
                        "instruction": "Repair script according to validator/rule issues.",
                        "previous_plan": current_plan.model_dump(),
                        "generated_columns": generated_columns,
                        "generated_preview": generated_preview[:5],
                        "issues": last_issues,
                        "repair_round": round_idx + 1,
                    },
                    round_idx + 1,
                )
        report = {
            "passed": False,
            "source": "sample_spec_generation_failed",
            "target_file": None,
            "columns": sample_spec.columns,
            "issues": last_issues,
            "reason": "Spec-driven sample_submission generation did not pass within repair limits; AutoRealize continued.",
        }
        self._write_submission_report(report)
        return report

    def _artifact_sanity_check(
        self,
        *,
        desc: str,
        data_root: Path,
        sample_spec: SampleSubmissionSpec,
        evaluation_contract: EvaluationContractReview,
        legacy,
    ) -> list[str]:
        defects: list[str] = []
        if not str(desc or "").strip():
            defects.append("description_empty")
        if not evaluation_contract.primary_metric:
            defects.append("evaluation_contract_missing_primary_metric")
        sample_path = self.run_dir / "sample_submission.csv"
        if sample_path.exists() and sample_spec.columns:
            cols = self._read_sample_submission_columns(sample_path)
            if cols != sample_spec.columns:
                defects.append(f"sample_columns_mismatch: got={cols}, expected={sample_spec.columns}")
        missing_refs = legacy._find_missing_file_references(desc, data_root)
        defects.extend([f"引用了不存在文件: {x}" for x in missing_refs])
        return list(dict.fromkeys([x for x in defects if str(x).strip()]))

    def _collect_description_defects(
        self,
        *,
        desc: str,
        original_text: str,
        data_root: Path,
        legacy,
        include_eval: bool = True,
    ) -> tuple[list[str], list[str]]:
        defects = description_quality_check(desc) + coverage_defects(desc, original_text)
        if include_eval:
            defects += eval_ambiguity_defects(desc)
        missing_refs = legacy._find_missing_file_references(desc, data_root)
        defects.extend([f"引用了不存在文件: {x}" for x in missing_refs])
        cleaned = [str(x).strip() for x in defects if str(x).strip()]
        return list(dict.fromkeys(cleaned)), missing_refs

    def _strip_markdown_fence(self, text: str) -> str:
        value = (text or "").strip()
        if value.startswith("```"):
            value = value.strip("` \n")
            if value.lower().startswith("markdown"):
                value = value[8:].strip()
        return value

    def _repair_description_with_feedback(
        self,
        *,
        desc: str,
        original_text: str,
        data_digest: str,
        downstream_context: dict,
        data_root: Path,
        legacy,
        stage: str,
        include_eval: bool = True,
        use_full_document: bool = False,
        evaluation_contract: EvaluationContractReview | None = None,
    ) -> tuple[str, list[str]]:
        current = desc
        max_retries = max(3, int(getattr(self.config.prompt, "description_quality_max_retries", 3)))
        repair_log = downstream_context.setdefault("description_repair_log", [])

        for idx in range(max_retries):
            defects, missing_refs = self._collect_description_defects(
                desc=current,
                original_text=original_text,
                data_root=data_root,
                legacy=legacy,
                include_eval=include_eval,
            )
            if not defects:
                return current, defects

            payload = {
                "stage": stage,
                "round": idx + 1,
                "defects": defects,
                "missing_file_references": missing_refs,
            }
            repair_log.append(payload)
            log_event(
                logger,
                "module.task_definition.description_repair",
                "RETRYING",
                stage=stage,
                round=idx + 1,
                defects=len(defects),
                missing_refs=missing_refs[:8],
            )

            if use_full_document:
                system = self.services.prompt_mgr.load("system/description_writer.md")
                artifact_refs = self._task_definition_artifact_refs(
                    original_text=original_text,
                    data_description_text=data_digest,
                    current_description=current,
                    downstream_context=downstream_context,
                )
                stable_payload = {
                    "context_policy": self._headroom_context_policy(),
                    "instruction": (
                        "上一版 description.md 没有通过检查。请根据 errors 修复并重新输出完整、最终、给人看的中文 Markdown。"
                        "不要输出修复说明或审查日志。"
                    ),
                    "hard_rules": [
                        "不得引用 missing_file_references 中的任何文件名。",
                        "只能引用 existing_file_names 中真实存在的输入/输出文件，或系统固定输出 description.md、sample_submission.csv、submission.csv。",
                        "如果某个处理步骤需要临时中间产物，只描述处理逻辑，不要把它写成输入数据文件。",
                        "保留原始需求、权威说明、提交格式和评估协议，不要因为修复文件名而改写任务目标。",
                    ],
                    "existing_file_names": self._known_data_file_names(data_root),
                    "authoritative_context": {
                        "downstream_signals": self._downstream_signal_pack(downstream_context),
                        "authoritative_memory": compact_authoritative_memory(
                            downstream_context.get("authoritative_memory", {})
                        ),
                        "constraint_memory": compact_constraint_memory(downstream_context.get("constraint_memory", {})),
                        "question_memory": self._question_memory_pack(downstream_context),
                    },
                    "evaluation_contract": evaluation_contract.model_dump() if evaluation_contract else {},
                    "original_requirements_full": original_text,
                }
                self._record_headroom_pack_telemetry(
                    f"description_repair_{stage}_{idx+1}",
                    {
                        "original_requirements_full": original_text,
                        "authoritative_context": stable_payload["authoritative_context"],
                        "evaluation_contract": stable_payload["evaluation_contract"],
                        "artifact_refs": artifact_refs,
                    },
                )
                dynamic_payload = {
                    "stage": stage,
                    "round": idx + 1,
                    "errors": defects,
                    "missing_file_references": missing_refs,
                    "current_description_visible_excerpt": current[:8000],
                    "current_description_policy": "Only this excerpt is visible to the model in this repair call. The full current draft is stored locally for audit, not expanded in the prompt.",
                }
                stable, dynamic = stable_dynamic_prompt(
                    stable=stable_payload,
                    dynamic=dynamic_payload,
                    stable_title="Stable description repair context",
                    dynamic_title="Dynamic description repair feedback",
                )
                repaired = self.services.llm_client.ask_text(
                    system_prompt=system,
                    user_prompt=dynamic,
                    prompt_name=f"description_repair_{stage}_{idx+1}",
                    static_user_prompt=stable,
                    dynamic_user_prompt=dynamic,
                )
                repaired = self._strip_markdown_fence(repaired)
            else:
                repaired = legacy._rewrite_mutable_sections_with_llm(
                    llm_client=self.services.llm_client,
                    prompt_mgr=self.services.prompt_mgr,
                    base_desc=current,
                    defects=defects,
                    downstream_context={
                        **downstream_context,
                        "description_repair_errors": defects,
                        "missing_file_references": missing_refs,
                        "existing_file_names": self._known_data_file_names(data_root),
                    },
                    prompt_name=f"description_repair_{stage}_{idx+1}",
                )

            if str(repaired or "").strip():
                current = repaired

        defects, missing_refs = self._collect_description_defects(
            desc=current,
            original_text=original_text,
            data_root=data_root,
            legacy=legacy,
            include_eval=include_eval,
        )
        if missing_refs:
            cleaned = legacy._enforce_existing_file_references(current, data_root)
            if cleaned != current:
                log_event(
                    logger,
                    "module.task_definition.description_repair",
                    "SOFT_CLEANED",
                    stage=stage,
                    removed_refs=missing_refs[:8],
                )
                repair_log.append(
                    {
                        "stage": stage,
                        "round": "soft_clean",
                        "removed_missing_file_references": missing_refs,
                    }
                )
                current = cleaned
                defects, _ = self._collect_description_defects(
                    desc=current,
                    original_text=original_text,
                    data_root=data_root,
                    legacy=legacy,
                    include_eval=include_eval,
                )
        if defects:
            log_event(
                logger,
                "module.task_definition.description_repair",
                "WARNING",
                stage=stage,
                remaining_defects=len(defects),
            )
        return current, defects

    def _safe_apply_evaluation_contract(
        self,
        *,
        desc: str,
        evaluation_contract: EvaluationContractReview,
        original_text: str,
        data_digest: str,
        downstream_context: dict,
        stage: str,
    ) -> tuple[str, EvaluationContractReview]:
        current_contract = evaluation_contract
        last_error = ""
        max_retries = max(3, int(getattr(self.config.prompt, "description_quality_max_retries", 3)))
        for idx in range(max_retries):
            try:
                return apply_evaluation_contract(desc, current_contract, downstream_context), current_contract
            except RuntimeError as exc:
                last_error = str(exc)
                log_event(
                    logger,
                    "module.task_definition.evaluation_contract",
                    "REPAIRING",
                    stage=stage,
                    round=idx + 1,
                    error=last_error[:300],
                )
                current_contract = self._review_evaluation_contract(
                    desc=desc,
                    original_text=original_text,
                    data_digest=data_digest,
                    downstream_context=downstream_context,
                    previous_contract=current_contract,
                    reflection_feedback=[last_error],
                )
        log_event(
            logger,
            "module.task_definition.evaluation_contract",
            "WARNING",
            stage=stage,
            error=last_error[:300],
        )
        return desc, current_contract

    def _compose_final_description(
        self,
        *,
        desc: str,
        original_text: str,
        data_digest: str,
        downstream_context: dict,
        evaluation_contract: EvaluationContractReview,
    ) -> str:
        system = self.services.prompt_mgr.load("system/description_writer.md")
        artifact_refs = self._task_definition_artifact_refs(
            original_text=original_text,
            data_description_text=data_digest,
            current_description=desc,
            downstream_context=downstream_context,
        )
        compact_context = {
            "downstream_signals": self._downstream_signal_pack(downstream_context),
            "authoritative_memory": compact_authoritative_memory(downstream_context.get("authoritative_memory", {})),
            "constraint_memory": compact_constraint_memory(downstream_context.get("constraint_memory", {})),
            "question_memory": self._question_memory_pack(downstream_context),
            "authoritative_submission_contract": downstream_context.get("authoritative_submission_contract", {}),
        }
        composer_payload = {
            "context_policy": self._headroom_context_policy(),
            "output_language_policy": self._output_language_policy(),
            "instruction": (
                "Rewrite the current draft into the final reader-facing description.md. "
                "Official/original task documents and official sample/output contracts have highest priority. "
                "Data profiles may explain files and fields but must not override official task, output, or evaluation rules."
            ),
            "hard_rules": [
                "必须输出一份完整、自然、给人看的 Markdown 赛题说明，并严格遵守 output_language_policy。",
                "不得输出反思过程、审查日志、智能体中间结果、issues/fixes、ambiguity_points、Contract Status。",
                "不得发明提交列、输出文件名、评估指标、指标方向、行数规则或固定随机种子。",
                "若没有权威提交合同，不要编造 sample_submission.csv；按原始说明描述输出协议或说明由下游实验协议决定。",
                "原始 description.md、README、官方规则、官方样例提交优先级高于数据统计和字段名猜测。",
                "除 output_language_policy 允许保留的代码、文件名、字段名、列名、变量名、公式和必要技术名词外，不要混用其它自然语言。",
            ],
            "required_sections": [
                "任务概述",
                "数据与读取方式",
                "字段说明",
                "任务定义",
                "评估协议",
                "输出或提交格式",
                "关键约束与注意事项",
            ],
            "authoritative_context": compact_context,
            "evaluation_contract": evaluation_contract.model_dump(),
            "downstream_context": self._downstream_signal_pack(downstream_context),
            "original_requirements_full": original_text,
        }
        self._record_headroom_pack_telemetry(
            "description_final_composer",
            {
                "original_requirements_full": original_text,
                "authoritative_context": compact_context,
                "evaluation_contract": evaluation_contract.model_dump(),
                "artifact_refs": artifact_refs,
            },
        )
        current = desc
        defects: list[str] = []
        max_rounds = max(3, int(getattr(self.config.prompt, "description_quality_max_retries", 3)))
        for idx in range(max_rounds):
            stable, dynamic = stable_dynamic_prompt(
                stable=composer_payload,
                dynamic={
                    "current_reviewed_draft_visible_excerpt": current[:10000],
                    "current_draft_policy": "Only this excerpt is visible to the model. The full draft and data cognition report are stored locally for audit, not expanded in the prompt.",
                    "previous_defects": defects,
                },
                stable_title="Stable final description composition context",
                dynamic_title="Dynamic current draft and defects",
            )
            composed = self.services.llm_client.ask_text(
                system_prompt=system,
                user_prompt=dynamic,
                prompt_name=f"description_final_composer_{idx+1}",
                static_user_prompt=stable,
                dynamic_user_prompt=dynamic,
            ).strip()
            if composed.startswith("```"):
                composed = composed.strip("` \n")
                if composed.lower().startswith("markdown"):
                    composed = composed[8:].strip()
            if composed:
                current = composed
            defects = description_quality_check(current) + coverage_defects(current, original_text) + eval_ambiguity_defects(current)
            if not defects:
                break
        if defects:
            log_event(
                logger,
                "module.task_definition.description_composer",
                "WARNING",
                defects=len(defects),
            )
        return current


    def _retrieve_relevant_knowledge(self, task_hint: str, downstream_context: dict) -> list[dict]:
        store = getattr(self.services, "knowledge_store", None)
        if store is None:
            return []
        query_parts = [
            task_hint,
            str((downstream_context.get("authoritative_memory") or {}).get("task_goal", "")),
            " ".join([str(x) for x in (downstream_context.get("authoritative_memory") or {}).get("evaluation_requirements", [])[:20]]),
            " ".join([str(x) for x in (downstream_context.get("authoritative_memory") or {}).get("constraints", [])[:20]]),
            str(downstream_context.get("target_column", "")),
            str(downstream_context.get("task_type_hint", "")),
            " ".join([str(x) for x in downstream_context.get("submission_columns", [])]),
            " ".join([str(x) for x in downstream_context.get("train_columns", [])[:40]]),
            " ".join([str(x) for x in downstream_context.get("predict_columns", [])[:40]]),
        ]
        query = "\n".join([x for x in query_parts if x.strip()])
        results = store.search(query, top_k=self.config.knowledge.retrieval_top_k)
        packed: list[dict] = []
        for r in results:
            packed.append(
                {
                    "score": r.score,
                    "reasons": r.reasons,
                    "kind": r.entry.kind,
                    "source": r.entry.source,
                    "text": r.entry.text,
                    "fields": r.entry.fields,
                    "constraints": r.entry.constraints,
                    "tags": r.entry.tags,
                }
            )
        write_json_safe(self.report_dir / "retrieved_knowledge.json", packed, indent=2)
        log_event(logger, "knowledge.local_store", "RETRIEVED", entries=len(packed))
        return packed

    def _review_evaluation_contract(
        self,
        *,
        desc: str = "",
        original_text: str = "",
        data_digest: str = "",
        downstream_context: dict,
        evaluation_evidence_pack: dict | None = None,
        frozen_task_sections: dict | None = None,
        previous_contract: EvaluationContractReview | None = None,
        reflection_feedback: list[str] | None = None,
    ) -> EvaluationContractReview:
        system = self.services.prompt_mgr.load("system/evaluation_contract_reviewer.md")
        artifact_refs = self._task_definition_artifact_refs(
            original_text=original_text,
            data_description_text=data_digest,
            current_description=desc,
            protocol_bundle=downstream_context.get("description_protocol_bundle", {}),
            downstream_context=downstream_context,
        )
        if evaluation_evidence_pack is None:
            compact_context = {
                "downstream_signals": self._downstream_signal_pack(downstream_context),
                "problem_paradigm_review": downstream_context.get("problem_paradigm_review", {}),
                "authoritative_memory": compact_authoritative_memory(downstream_context.get("authoritative_memory", {})),
                "authoritative_submission_contract": downstream_context.get("authoritative_submission_contract", {}),
                "constraint_memory": compact_constraint_memory(downstream_context.get("constraint_memory", {})),
                "question_memory": self._question_memory_pack(downstream_context),
                "retrieved_knowledge": downstream_context.get("retrieved_knowledge", [])[:8],
            }
            evaluation_evidence_pack = {
                "downstream_context": compact_context,
                "legacy_current_evaluation_section": self._h2_section_text(desc, "评估协议")[:5000] if desc else "",
                "legacy_current_output_section": self._h2_section_text(desc, "输出或提交格式")[:4000] if desc else "",
            }
        stable_payload = {
            "context_policy": self._headroom_context_policy(),
            "output_language_policy": self._output_language_policy(),
            "instruction": (
                "Compile or repair a strict EvaluationContractReview from the compact evidence pack. "
                "Do not inspect or rewrite the full description. Normal review rounds may set passed=false "
                "with concise issues/fixes when evidence is insufficient. The finalizer round must instead "
                "write a complete executable contract using explicit AutoRealize-defined assumptions when needed."
            ),
            "original_requirements_full": original_text,
            "evaluation_evidence_pack": evaluation_evidence_pack,
            "frozen_task_sections": frozen_task_sections or {},
        }
        self._record_headroom_pack_telemetry(
            "evaluation_contract_reviewer",
            {
                "original_requirements_full": original_text,
                "evaluation_evidence_pack": evaluation_evidence_pack,
                "frozen_task_sections": frozen_task_sections or {},
                "artifact_refs": artifact_refs,
            },
        )
        reflection_payload = {}
        if previous_contract is not None or reflection_feedback:
            reflection_payload = {
                "reflection_instruction": "动态返修意见如下。请只修复评估协议合同，不要改写其它章节。",
                "reflection_feedback": reflection_feedback or [],
                "previous_contract": previous_contract.model_dump() if previous_contract else {},
            }
        review: EvaluationContractReview | None = None
        revision_log: list[dict] = []
        max_rounds = max(3, int(getattr(self.config.prompt, "evaluation_contract_max_rounds", 3)))
        for idx in range(max_rounds):
            is_finalizer_round = idx == max_rounds - 1
            if review is None:
                dynamic_payload = reflection_payload or {"instruction": "Create the evaluation contract review."}
            else:
                defects = evaluation_contract_defects(review)
                dynamic_payload = {
                    **reflection_payload,
                    "repair_instruction": (
                        "上一轮评估协议合同没有通过审查，必须返修后重新输出完整 JSON。"
                        "如果 passed=false 是因为证据缺口，请先尝试转化为明确的评估假设、惩罚规则、数据需求或外部配置要求。"
                        "只有当缺失信息会让主指标无法计算时才保留 passed=false，并必须给出 issues/fixes。"
                    ),
                    "previous_defects": defects,
                    "previous_issues": review.issues,
                    "previous_fixes": review.fixes,
                    "previous_contract": review.model_dump(),
                }
            if is_finalizer_round:
                finalizer_payload = {
                    **dynamic_payload,
                    "finalizer_instruction": (
                        "这是最后一轮，不再作为 reviewer 返回 passed=false。你现在是最终评估合同作者。"
                        "必须输出完整 EvaluationContractReview JSON，并设置 passed=true。"
                        "不得输出未明确、待补充、待确认、unknown、tbd、推荐、可选、通常、视情况、可以考虑。"
                        "如果官方材料缺少唯一评分公式、权重、惩罚系数或非法解处理规则，"
                        "请将缺口转化为清晰标注的 AutoRealize-defined evaluation assumption、"
                        "由输入数据上界推导的规则，或外部评估配置参数；仍要给出一个当前可执行的默认公式。"
                        "不要编造不存在的输入字段、官方样例列、车牌号、车辆唯一 ID 或官方规则。"
                        "如果需要资源实例但输入没有唯一实体 ID，只能使用方案输出中的确定性派生 ID 或占位资源 ID，"
                        "并在 y_pred_source/submission_checks 中说明。"
                        "issues 和 fixes 必须为空；所有先前问题都要被吸收到 metric_formula、validation_protocol、"
                        "submission_checks、leakage_guards、invalid_solution_rules、tie_break_rules、audit_metrics、evidence 或 rationale 中。"
                    ),
                }
                if review is not None:
                    finalizer_payload.setdefault("previous_defects", evaluation_contract_defects(review))
                    finalizer_payload.setdefault("previous_contract", review.model_dump())
                    finalizer_payload.setdefault("previous_issues", review.issues)
                    finalizer_payload.setdefault("previous_fixes", review.fixes)
                dynamic_payload = finalizer_payload
            stable, dynamic = stable_dynamic_prompt(
                stable=stable_payload,
                dynamic=dynamic_payload,
                stable_title="Stable evaluation contract evidence",
                dynamic_title="Dynamic evaluation review state",
                dynamic_limit=14000,
            )
            review = self.services.llm_client.ask_structured(
                model_cls=EvaluationContractReview,
                system_prompt=system,
                user_prompt=dynamic,
                prompt_name=(
                    f"evaluation_contract_finalizer_{idx+1}"
                    if is_finalizer_round
                    else f"evaluation_contract_reviewer_{idx+1}"
                ),
                static_context_prompt=stable,
                dynamic_user_prompt=dynamic,
            )
            if is_finalizer_round:
                review.passed = True
                review.issues = []
                review.fixes = []
                defects = []
            else:
                defects = evaluation_contract_defects(review)
            revision_log.append(
                {
                    "round": idx + 1,
                    "source": (
                        "finalizer"
                        if is_finalizer_round
                        else "reflection_repair" if reflection_feedback else "contract_review"
                    ),
                    "passed": review.passed,
                    "defects": defects,
                    "issues": review.issues,
                    "fixes": review.fixes,
                }
            )
            log_event(
                logger,
                "module.task_definition.evaluation_contract",
                "REVIEWED",
                round=idx + 1,
                passed=review.passed,
                defects=len(defects),
                finalizer=is_finalizer_round,
            )
            if is_finalizer_round:
                break
            if review.passed and not defects:
                break
            if defects and not any(str(x).startswith("evaluation_contract not passed") for x in defects):
                continue
            if idx >= 1 and not defects:
                break
        assert review is not None
        self._evaluation_contract_revision_log.extend(revision_log)
        self._write_evaluation_contract_report(review)
        self.services.trajectory.log("task_definition_module", "evaluation_contract", review.model_dump())
        return review

    def _write_evaluation_contract_report(self, review: EvaluationContractReview) -> None:
        write_json_safe(
            self.report_dir / "evaluation_contract_report.json",
            {
                "final": review.model_dump(),
                "revision_log": self._evaluation_contract_revision_log,
                "reflection_log": self._evaluation_reflection_log,
            },
            indent=2,
        )

    def _write_automl_context_pack(
        self,
        *,
        protocol_bundle: DescriptionProtocolBundle,
        file_summaries: list,
        downstream_context: dict,
        evaluation_contract: EvaluationContractReview,
        relations: list,
    ) -> dict:
        compiled_context = self._compiled_context_for_sections(
            file_summaries=file_summaries,
            downstream_context=downstream_context,
            relations=relations,
            table_limit=24,
            relation_limit=40,
        )
        pack = build_automl_context_pack(
            protocol_bundle,
            file_summaries=file_summaries,
            downstream_context=downstream_context,
            evaluation_contract=evaluation_contract,
            compiled_context=compiled_context,
        )
        payload = pack.model_dump()
        write_json_safe(self.report_dir / "automl_context_pack.json", payload, indent=2)
        (self.report_dir / "automl_context.md").write_text(
            render_automl_context_markdown(pack),
            encoding="utf-8",
        )
        downstream_context["automl_context_pack"] = payload
        downstream_context["automl_context_artifacts"] = {
            "json": "realize_report/automl_context_pack.json",
            "markdown": "realize_report/automl_context.md",
        }
        log_event(logger, "module.task_definition", "GENERATED_FILE", file="realize_report/automl_context_pack.json")
        log_event(logger, "module.task_definition", "GENERATED_FILE", file="realize_report/automl_context.md")
        return payload

    def _write_main_task_protocol(
        self,
        *,
        task_hint: str,
        problem_review: ProblemParadigmReview,
        protocol_bundle: DescriptionProtocolBundle,
        deterministic_data_access: object,
        evaluation_contract: EvaluationContractReview,
        automl_context_pack: dict,
        downstream_context: dict,
    ) -> dict:
        authoritative_memory = downstream_context.get("authoritative_memory", {})
        agent_context_pack = downstream_context.get("agent_context_pack", {})
        question_memory = {}
        if isinstance(agent_context_pack, dict):
            question_memory = agent_context_pack.get("question_memory", {})
        if not question_memory and isinstance(downstream_context.get("knowledge_base"), dict):
            question_memory = downstream_context["knowledge_base"].get("question_investigation", {})
        payload = {
            "schema_version": "autorealize.main_task_protocol.v1",
            "purpose": "Single backend entry point for task facts; derived human and machine artifacts must not override this protocol.",
            "authority_rules": [
                "user task hint > existing input description.md > README/official/spec/other task documents > data statistics and LLM inference",
                "Official sample_submission or explicitly documented output contracts define tabular submission columns when present.",
                "Heuristic downstream_context fields are non-authoritative unless supported by authoritative evidence.",
            ],
            "task_hint": task_hint,
            "problem_paradigm": problem_review.model_dump(),
            "task_protocol": protocol_bundle.model_dump(),
            "frozen_description_sections": downstream_context.get("description_sections", {}),
            "data_access_protocol": (
                deterministic_data_access.model_dump()
                if hasattr(deterministic_data_access, "model_dump")
                else deterministic_data_access
            ),
            "evaluation_contract": evaluation_contract.model_dump(),
            "output_contract": protocol_bundle.output.model_dump(),
            "sample_submission_spec": downstream_context.get("sample_submission_spec", {}),
            "sample_submission_status": {
                "available": bool(downstream_context.get("sample_submission_available", False)),
                "status": downstream_context.get("sample_submission_generation_status", ""),
                "path": downstream_context.get("generated_submission_path", ""),
                "columns": downstream_context.get("generated_submission_columns", []),
                "issues": downstream_context.get("sample_submission_generation_issues", []),
            },
            "authoritative_memory": authoritative_memory,
            "authority_conflicts": (
                authoritative_memory.get("authority_conflicts", [])
                if isinstance(authoritative_memory, dict)
                else []
            ),
            "question_investigation": question_memory,
            "downstream_context_evidence": downstream_context.get("evidence_levels", {}),
            "automl_context_pack": automl_context_pack,
            "source_artifacts": {
                "description": "description.md",
                "automl_context": "realize_report/automl_context.md",
                "automl_context_pack": "realize_report/automl_context_pack.json",
                "problem_paradigm": "realize_report/problem_paradigm_report.json",
                "data_access_protocol": "realize_report/data_access_protocol.json",
                "description_protocol_bundle": "realize_report/description_protocol_bundle.json",
                "evaluation_contract": "realize_report/evaluation_contract_report.json",
                "question_investigation": "realize_report/question_investigation_report.json",
            },
        }
        write_json_safe(self.report_dir / "main_task_protocol.json", payload, indent=2)
        downstream_context["main_task_protocol"] = {
            "json": "realize_report/main_task_protocol.json",
            "schema_version": payload["schema_version"],
        }
        log_event(logger, "module.task_definition", "GENERATED_FILE", file="realize_report/main_task_protocol.json")
        return payload

    def _h2_section_text(self, desc: str, header: str) -> str:
        aliases = set(SECTION_ALIASES.get(header, (header,)))
        for canonical, names in SECTION_ALIASES.items():
            if header == canonical or header in names:
                aliases = set(names)
                break
        lines = desc.splitlines()
        start = None
        for idx, line in enumerate(lines):
            if line.startswith("## ") and line[3:].strip() in aliases:
                start = idx
                break
        if start is None:
            return ""
        end = len(lines)
        for idx in range(start + 1, len(lines)):
            if lines[idx].startswith("## "):
                end = idx
                break
        return "\n".join(lines[start:end]).strip()

    def _reflect_evaluation_sections(self, desc: str, downstream_context: dict, round_idx: int) -> AmbiguityReview:
        system = self.services.prompt_mgr.load("system/eval_reflector.md")
        evaluation = self._h2_section_text(desc, "评估协议")
        submission = self._h2_section_text(desc, "提交格式")
        stable, dynamic = stable_dynamic_prompt(
            stable={
                "instruction": (
                    "请只检查两个 description 分段是否已经让评估协议无歧义、唯一、可执行。"
                    "如果还存在歧义，请指出具体 ambiguity_points，并给出可直接修改评估协议合同的 fixes。"
                    "只输出严格 JSON；最多列 6 条 ambiguity_points 和 6 条 fixes，每条必须短句，不要展开长解释。"
                ),
                "task_context": {
                    "task_hint": downstream_context.get("task_hint", ""),
                    "task_type_hint": downstream_context.get("task_type_hint", ""),
                    "problem_paradigm": downstream_context.get("problem_paradigm", ""),
                },
            },
            dynamic={"evaluation_section": evaluation, "submission_section": submission},
            stable_title="Stable evaluation reflection rules",
            dynamic_title="Dynamic evaluation sections",
        )
        return self.services.llm_client.ask_structured(
            model_cls=AmbiguityReview,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name=f"evaluation_section_reflector_{round_idx}",
            fewshot="",
            max_tokens=2000,
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )

    def _apply_evaluation_contract_until_unambiguous(
        self,
        *,
        desc: str,
        evaluation_contract: EvaluationContractReview,
        original_text: str,
        data_digest: str,
        downstream_context: dict,
    ) -> tuple[str, EvaluationContractReview]:
        current_contract = evaluation_contract
        current_desc = desc
        max_rounds = max(3, int(getattr(self.config.prompt, "evaluation_reflection_max_rounds", 3)))
        for idx in range(1, max_rounds + 1):
            current_desc, current_contract = self._safe_apply_evaluation_contract(
                desc=current_desc,
                evaluation_contract=current_contract,
                original_text=original_text,
                data_digest=data_digest,
                downstream_context=downstream_context,
                stage=f"evaluation_reflector_{idx}",
            )
            current_desc = sync_submission_format_with_context(current_desc, downstream_context)
            try:
                reflection = self._reflect_evaluation_sections(current_desc, downstream_context, idx)
            except RuntimeError as exc:
                message = str(exc)
                log_event(
                    logger,
                    "module.task_definition.eval_reflector",
                    "WARNING",
                    round=idx,
                    error=message[:300],
                    fallback="use_current_evaluation_contract",
                )
                self._evaluation_reflection_log.append(
                    {
                        "round": idx,
                        "is_unambiguous": False,
                        "ambiguity_points": ["评估反思器输出被截断或解析失败，已保留当前结构化评估合同。"],
                        "fixes": [],
                        "contract_passed": current_contract.passed,
                        "fallback_error": message[:1000],
                    }
                )
                self._write_evaluation_contract_report(current_contract)
                return current_desc, current_contract
            self._evaluation_reflection_log.append(
                {
                    "round": idx,
                    "is_unambiguous": reflection.is_unambiguous,
                    "ambiguity_points": reflection.ambiguity_points,
                    "fixes": reflection.fixes,
                    "contract_passed": current_contract.passed,
                }
            )
            log_event(
                logger,
                "module.task_definition.evaluation_reflector",
                "REVIEWED",
                round=idx,
                is_unambiguous=reflection.is_unambiguous,
                issues=len(reflection.ambiguity_points),
            )
            if reflection.is_unambiguous:
                self._write_evaluation_contract_report(current_contract)
                return current_desc, current_contract
            feedback = reflection.ambiguity_points + reflection.fixes
            if not feedback:
                self._write_evaluation_contract_report(current_contract)
                return current_desc, current_contract
            current_contract = self._review_evaluation_contract(
                desc=current_desc,
                original_text=original_text,
                data_digest=data_digest,
                downstream_context=downstream_context,
                previous_contract=current_contract,
                reflection_feedback=feedback,
            )
        self._write_evaluation_contract_report(current_contract)
        return current_desc, current_contract

    def run(self, data_root: Path, task_hint: str, cognition: DataCognitionResult) -> TaskDefinitionResult:
        import autorealize.pipeline as legacy

        log_event(logger, "module.task_definition", "CREATED")
        log_event(logger, "module.task_definition", "ACTIVATED")
        data_description_path = cognition.data_description_path or (self.report_dir / "data_description.md")
        data_description_text = data_description_path.read_text(encoding="utf-8") if data_description_path.exists() else ""
        data_digest = data_description_text[:16000]
        original_text = self._authoritative_requirement_text(
            cognition.authoritative_memory,
            cognition.original_requirement_texts,
            task_hint,
            cognition.agent_context_pack,
        ) or task_hint
        (self.report_dir / "original_requirements.txt").write_text(original_text or "", encoding="utf-8")

        log_event(logger, "module.task_definition.intent", "ACTIVATED")
        if bool(getattr(self.config.switches, "run_architect_plan", False)) or not bool(
            getattr(self.config.switches, "optimize_llm_cost", True)
        ):
            plan = self.architect.build_plan(task_hint=task_hint, cognition_digest=data_digest)
            c1 = self.architect.critique_plan(plan)
            c2 = self.architect.critique_expansion(plan)
            log_event(
                logger,
                "module.task_definition.intent",
                "COMPLETED",
                plan_severity=c1.severity.value,
                expansion_severity=c2.severity.value,
            )
            self.services.trajectory.log(
                "task_definition_module",
                "critique",
                {"plan_severity": c1.severity.value, "expansion_severity": c2.severity.value, "issues": c1.issues + c2.issues},
            )
        else:
            plan = self._default_pipeline_plan(task_hint=task_hint)
            log_event(logger, "module.task_definition.intent", "SKIPPED", reason="low_token_mode")
            self.services.trajectory.log("task_definition_module", "critique", {"skipped": True, "reason": "low_token_mode"})

        downstream_context = legacy._infer_downstream_context(data_root, cognition.file_summaries, task_hint, self.config)
        downstream_context["task_hint"] = task_hint
        downstream_context["constraint_memory"] = cognition.constraint_memory
        downstream_context["knowledge_base"] = cognition.knowledge_base
        self._apply_authoritative_context(downstream_context, cognition.authoritative_memory, cognition.agent_context_pack)
        retrieved_knowledge = self._retrieve_relevant_knowledge(task_hint, downstream_context)
        downstream_context["retrieved_knowledge"] = retrieved_knowledge
        if self.config.data.auto_generate_predict_split:
            legacy._maybe_generate_predict_split(data_root, downstream_context, self.config)
            downstream_context = legacy._infer_downstream_context(data_root, cognition.file_summaries, task_hint, self.config)
            downstream_context["task_hint"] = task_hint
            downstream_context["constraint_memory"] = cognition.constraint_memory
            downstream_context["knowledge_base"] = cognition.knowledge_base
            self._apply_authoritative_context(downstream_context, cognition.authoritative_memory, cognition.agent_context_pack)
            retrieved_knowledge = self._retrieve_relevant_knowledge(task_hint, downstream_context)
            downstream_context["retrieved_knowledge"] = retrieved_knowledge

        deterministic_data_access = build_data_access_protocol(cognition.file_summaries)
        downstream_context["data_access_protocol"] = deterministic_data_access.model_dump()
        log_event(logger, "module.task_definition.problem_paradigm", "ACTIVATED")
        problem_review = self._classify_problem_paradigm(
            task_hint=task_hint,
            original_text=original_text,
            data_digest=data_digest,
            downstream_context=downstream_context,
            file_summaries=cognition.file_summaries,
            relations=cognition.relation_hints,
            data_description_text=data_description_text,
        )
        downstream_context["problem_paradigm"] = problem_review.problem_paradigm
        downstream_context["problem_paradigm_review"] = problem_review.model_dump()
        self._write_protocol_artifacts(
            problem_review=problem_review,
            deterministic_data_access=deterministic_data_access,
        )
        log_event(
            logger,
            "module.task_definition.problem_paradigm",
            "COMPLETED",
            paradigm=problem_review.problem_paradigm,
            confidence=f"{problem_review.confidence:.3f}",
            requires_sample_submission=problem_review.requires_sample_submission,
        )

        if bool(getattr(self.config.switches, "run_legacy_task_classifier", False)) or not bool(
            getattr(self.config.switches, "optimize_llm_cost", True)
        ):
            task_cls = legacy._classify_task_type(
                llm_client=self.services.llm_client,
                prompt_mgr=self.services.prompt_mgr,
                task_hint=task_hint,
                data_digest=data_digest,
                downstream_context=downstream_context,
                enable_fewshot=self.config.switches.enable_fewshot,
            )
        else:
            task_cls = self._task_classification_from_problem_review(problem_review, downstream_context)
            log_event(logger, "module.task_definition.classifier", "SKIPPED", reason="low_token_mode")
        log_event(
            logger,
            "module.task_definition.classifier",
            "COMPLETED",
            task_type=task_cls.task_type,
            confidence=f"{task_cls.confidence:.3f}",
            primary_metric=task_cls.primary_metric,
        )
        downstream_context["task_type_hint"] = task_cls.task_type
        if task_cls.primary_metric:
            plan.evaluation_metric = task_cls.primary_metric
        if task_cls.metric_formula:
            plan.evaluation_formula = task_cls.metric_formula
        self.services.trajectory.log("task_definition_module", "task_classifier", task_cls.model_dump())

        legacy._refine_file_summaries_by_downstream_context(cognition.file_summaries, downstream_context)
        rel_hints_refined = legacy.detect_relations(
            cognition.table_columns,
            file_summaries=cognition.file_summaries,
            parallel=self.config.parallel.enable_parallel_relations,
            max_workers=self.config.parallel.relations_max_workers,
        )
        dir_summaries_refined = legacy._summarize_dirs(data_root, cognition.file_summaries)
        write_data_description(data_description_path, cognition.file_summaries, dir_summaries_refined, rel_hints_refined)
        append_constraint_memory_section(data_description_path, cognition.constraint_memory)
        data_description_text = data_description_path.read_text(encoding="utf-8")
        data_digest = data_description_text[:16000]

        protocol_bundle = self._build_description_protocol_bundle(
            problem_review=problem_review,
            original_text=original_text,
            data_digest=data_digest,
            downstream_context=downstream_context,
            deterministic_data_access=deterministic_data_access,
            file_summaries=cognition.file_summaries,
            relations=rel_hints_refined,
            data_description_text=data_description_text,
        )
        downstream_context["description_protocol_bundle"] = protocol_bundle.model_dump()
        self._write_protocol_artifacts(
            problem_review=problem_review,
            deterministic_data_access=deterministic_data_access,
            protocol_bundle=protocol_bundle,
        )

        log_event(logger, "module.task_definition.description_sections", "ACTIVATED")
        task_authority_pack = self._build_task_authority_pack(
            task_hint=task_hint,
            problem_review=problem_review,
            downstream_context=downstream_context,
        )
        task_definition_artifact_refs = self._task_definition_artifact_refs(
            original_text=original_text,
            data_description_text=data_description_text,
            protocol_bundle=protocol_bundle,
            downstream_context=downstream_context,
        )
        overview_task = self._generate_overview_task_definition_sections(
            task_authority_pack=task_authority_pack,
            original_text=original_text,
            artifact_refs=task_definition_artifact_refs,
        )
        frozen_sections: dict[str, str] = {
            "任务概述": overview_task.overview_markdown,
            "任务定义": overview_task.task_definition_markdown,
        }

        log_event(logger, "module.task_definition.evaluation_contract", "ACTIVATED")
        evaluation_evidence_pack = self._build_evaluation_evidence_pack(
            downstream_context=downstream_context,
            problem_review=problem_review,
            relations=rel_hints_refined,
            file_summaries=cognition.file_summaries,
            protocol_bundle=protocol_bundle,
        )
        evaluation_contract = self._review_evaluation_contract(
            downstream_context=downstream_context,
            evaluation_evidence_pack=evaluation_evidence_pack,
            frozen_task_sections=frozen_sections,
            original_text=original_text,
            data_digest=data_description_text,
        )
        downstream_context["evaluation_contract"] = evaluation_contract.model_dump()
        evaluation_section = self._render_evaluation_section(evaluation_contract, downstream_context)
        frozen_sections["评估协议"] = evaluation_section
        log_event(
            logger,
            "module.task_definition.evaluation_contract",
            "COMPLETED",
            primary_metric=evaluation_contract.primary_metric,
            metric_direction=evaluation_contract.metric_direction,
            passed=evaluation_contract.passed,
        )

        sample_path = self.run_dir / "sample_submission.csv"
        if sample_path.exists():
            try:
                sample_path.unlink()
            except OSError as exc:
                log_event(
                    logger,
                    "module.task_definition",
                    "REMOVE_STALE_SAMPLE_FAILED",
                    file="sample_submission.csv",
                    error=str(exc)[:180],
                )
        official_sample = self._find_official_sample_submission(data_root)
        official_sample_probe = self._probe_sample_submission(official_sample, data_root) if official_sample else {}
        output_evidence_pack = self._build_output_evidence_pack(
            data_root=data_root,
            downstream_context=downstream_context,
            problem_review=problem_review,
            official_sample_probe=official_sample_probe,
            file_summaries=cognition.file_summaries,
            relations=rel_hints_refined,
            evaluation_contract=evaluation_contract,
        )
        output_draft = self._generate_output_section(
            output_evidence_pack=output_evidence_pack,
            frozen_sections=frozen_sections,
            evaluation_contract=evaluation_contract,
            original_text=original_text,
            artifact_refs=task_definition_artifact_refs,
        )
        sample_spec = output_draft.sample_submission_spec
        if official_sample_probe.get("columns"):
            sample_spec.source = "official_sample"
            sample_spec.should_generate = False
            sample_spec.columns = [str(x) for x in official_sample_probe.get("columns", []) if str(x).strip()]
        protocol_bundle.output.columns = sample_spec.columns or protocol_bundle.output.columns
        protocol_bundle.output.output_filename = sample_spec.output_filename or protocol_bundle.output.output_filename
        protocol_bundle.output.sample_submission_required = bool(
            sample_spec.should_generate or (isinstance(official_sample_probe, dict) and official_sample_probe.get("columns"))
        )
        if sample_spec.format_rules:
            protocol_bundle.output.format_rules = list(dict.fromkeys(protocol_bundle.output.format_rules + sample_spec.format_rules))
        if sample_spec.no_sample_submission_reason:
            protocol_bundle.output.no_sample_submission_reason = sample_spec.no_sample_submission_reason
        source_schema_contract = build_data_schema_contract(cognition.file_summaries, max_tables=80)
        source_field_corrections = self._correct_sample_spec_source_fields(
            sample_spec,
            schema_contract=source_schema_contract,
        )
        source_field_audit_rules = [
            str(rule)
            for rule in (sample_spec.validation_rules or [])
            if str(rule).startswith("AutoRealize corrected source field alias")
            or str(rule).startswith("AutoRealize could not resolve source field alias")
        ]
        if source_field_corrections:
            evaluation_contract = self._apply_source_field_alias_corrections_to_contract(
                evaluation_contract,
                source_field_corrections,
            )
            downstream_context["evaluation_contract"] = evaluation_contract.model_dump()
            evaluation_section = self._apply_source_field_alias_corrections_to_text(
                self._render_evaluation_section(evaluation_contract, downstream_context),
                source_field_corrections,
            )
            frozen_sections["评估协议"] = evaluation_section
            self._write_evaluation_contract_report(evaluation_contract)
        if source_field_corrections or source_field_audit_rules:
            output_draft.markdown = self._apply_source_field_corrections_to_output_markdown(
                output_draft.markdown,
                sample_spec,
                source_field_corrections,
            )
        if source_field_corrections:
            downstream_context["sample_submission_source_field_corrections"] = source_field_corrections
            log_event(
                logger,
                "module.task_definition.sample_submission",
                "SOURCE_FIELDS_CORRECTED",
                corrections=source_field_corrections[:8],
            )
        elif source_field_audit_rules:
            log_event(
                logger,
                "module.task_definition.sample_submission",
                "SOURCE_FIELDS_AUDITED",
                unresolved_rules=source_field_audit_rules[:8],
            )
        downstream_context["sample_submission_spec"] = sample_spec.model_dump()
        source_alias_guard = self._collect_source_alias_guard(
            sample_spec=sample_spec,
            schema_contract=source_schema_contract,
            downstream_context=downstream_context,
        )
        if source_alias_guard:
            downstream_context["source_alias_guard"] = source_alias_guard
            log_event(
                logger,
                "module.task_definition.source_alias_guard",
                "COMPLETED",
                aliases=len(source_alias_guard),
            )

        generate_sample_submission = bool(getattr(self.config.switches, "generate_sample_submission", True))
        downstream_context["sample_submission_generation_requested"] = generate_sample_submission
        submission_report: dict
        if official_sample is not None and generate_sample_submission:
            log_event(logger, "module.task_definition", "REUSING_FILE", file="sample_submission.csv")
            submission_report = self._reuse_official_sample_submission(official_sample, data_root)
        elif generate_sample_submission and sample_spec.should_generate:
            log_event(logger, "module.task_definition", "GENERATING_FILE", file="sample_submission.csv")
            submission_report = self._generate_sample_submission_from_spec(
                data_root=data_root,
                sample_spec=sample_spec,
                file_summaries=cognition.file_summaries,
                downstream_context=downstream_context,
                relations=rel_hints_refined,
                original_text=original_text,
                artifact_refs=task_definition_artifact_refs,
                frozen_context={
                    "任务概述": overview_task.overview_markdown,
                    "任务定义": overview_task.task_definition_markdown,
                    "评估协议": evaluation_section,
                    "输出或提交格式": output_draft.markdown,
                },
            )
        else:
            reason = "disabled_by_config" if not generate_sample_submission else "not_required_by_output_spec"
            submission_report = {
                "passed": True,
                "source": reason,
                "target_file": None,
                "columns": sample_spec.columns,
                "issues": [],
                "problem_paradigm": problem_review.problem_paradigm,
                "reason": sample_spec.no_sample_submission_reason or reason,
            }
            self._write_submission_report(submission_report)
            log_event(logger, "module.task_definition", "SKIPPED", file="sample_submission.csv", reason=reason)

        generated_submission_columns = self._read_sample_submission_columns(sample_path)
        if generated_submission_columns:
            downstream_context["generated_submission_columns"] = generated_submission_columns
            downstream_context["generated_submission_path"] = "sample_submission.csv"
            downstream_context["sample_submission_available"] = True
            downstream_context["sample_submission_generation_status"] = str(submission_report.get("source", "generated"))
            downstream_context["generate_sample_submission"] = True
            if sample_spec.columns and generated_submission_columns != sample_spec.columns:
                downstream_context["sample_submission_generation_issues"] = [
                    f"generated columns differ from sample_submission_spec: {generated_submission_columns} != {sample_spec.columns}"
                ]
            output_draft.markdown = (
                output_draft.markdown.rstrip()
                + "\n\n### 样例文件状态\n"
                + f"- 已生成或复用 `sample_submission.csv`，列顺序为：{', '.join(f'`{c}`' for c in generated_submission_columns)}。"
            )
        else:
            issues = submission_report.get("issues", [])
            downstream_context["sample_submission_available"] = False
            downstream_context["sample_submission_generation_status"] = str(submission_report.get("source", "not_created"))
            downstream_context["sample_submission_generation_issues"] = issues if isinstance(issues, list) else [str(issues)]
            downstream_context["generate_sample_submission"] = False
            if generate_sample_submission and sample_spec.should_generate:
                output_draft.markdown = (
                    output_draft.markdown.rstrip()
                    + "\n\n### 样例文件状态\n"
                    + "- 本轮未成功生成 `sample_submission.csv`；正式输出仍以本节格式要求和评估协议为准。"
                )
        frozen_sections["输出或提交格式"] = output_draft.markdown

        data_section = self._generate_generic_section(
            section_id="data",
            section_title="数据说明",
            evidence_pack=self._build_data_section_pack(
                file_summaries=cognition.file_summaries,
                downstream_context=downstream_context,
            ),
            frozen_sections=frozen_sections,
            original_text=original_text,
            artifact_refs=task_definition_artifact_refs,
            extra_rules=[
                "只写输入文件/文件组作用、shape 和必要读取提示。",
                "不要展开完整 preview，不要把字段说明放到本章主体。",
            ],
        )
        frozen_sections["数据说明"] = data_section.markdown
        field_section = self._generate_generic_section(
            section_id="fields",
            section_title="关键字段说明",
            evidence_pack=self._build_field_section_pack(
                file_summaries=cognition.file_summaries,
                downstream_context=downstream_context,
                relations=rel_hints_refined,
            ),
            frozen_sections=frozen_sections,
            original_text=original_text,
            artifact_refs=task_definition_artifact_refs,
            extra_rules=[
                "只说明任务相关关键字段；多 sheet Excel 需要标注 sheet 级字段边界。",
                "字段含义优先使用前置文件认知结果，不要重新发明。",
            ],
        )
        frozen_sections["关键字段说明"] = field_section.markdown
        constraint_section = self._generate_generic_section(
            section_id="constraints_leakage",
            section_title="约束与防泄漏",
            evidence_pack=self._build_constraint_section_pack(
                downstream_context=downstream_context,
                relations=rel_hints_refined,
            ),
            frozen_sections=frozen_sections,
            original_text=original_text,
            artifact_refs=task_definition_artifact_refs,
            extra_rules=[
                "写硬约束、非法输出、防泄漏和验证边界。",
                "不要发明权威材料中没有的约束。",
            ],
        )
        frozen_sections["约束与防泄漏"] = constraint_section.markdown
        tips_section = self._generate_generic_section(
            section_id="tips",
            section_title="关键坑点与待确认事项",
            evidence_pack=self._build_tips_pack(downstream_context=downstream_context),
            frozen_sections=frozen_sections,
            original_text=original_text,
            artifact_refs=task_definition_artifact_refs,
            extra_rules=[
                "必须包含 QDI 未解问题处理原则。",
                "提醒下游避免依赖未验证疑惑，无法避免时写成可配置假设或保守兜底。",
            ],
        )
        frozen_sections["关键坑点与待确认事项"] = tips_section.markdown

        desc = self._compose_description_sections(
            [
                overview_task.overview_markdown,
                overview_task.task_definition_markdown,
                evaluation_section,
                output_draft.markdown,
                data_section.markdown,
                field_section.markdown,
                constraint_section.markdown,
                tips_section.markdown,
            ]
        )
        desc = self._apply_source_alias_guard_to_description(desc, source_alias_guard)
        desc = legacy._enforce_existing_file_references(desc, data_root)
        defects = self._artifact_sanity_check(
            desc=desc,
            data_root=data_root,
            legacy=legacy,
            sample_spec=sample_spec,
            evaluation_contract=evaluation_contract,
        )
        desc = finalize_description_markdown(desc)
        downstream_context["description_protocol_bundle"] = protocol_bundle.model_dump()
        downstream_context["description_sections"] = {
            "overview_task_definition": overview_task.model_dump(),
            "output": output_draft.model_dump(),
            "data": data_section.model_dump(),
            "fields": field_section.model_dump(),
            "constraints_leakage": constraint_section.model_dump(),
            "tips": tips_section.model_dump(),
        }
        self._write_protocol_artifacts(
            problem_review=problem_review,
            deterministic_data_access=deterministic_data_access,
            protocol_bundle=protocol_bundle,
        )
        log_event(logger, "module.task_definition.description_sections", "COMPLETED", defects=len(defects))
        automl_context_pack = self._write_automl_context_pack(
            protocol_bundle=protocol_bundle,
            file_summaries=cognition.file_summaries,
            downstream_context=downstream_context,
            evaluation_contract=evaluation_contract,
            relations=rel_hints_refined,
        )
        main_task_protocol = self._write_main_task_protocol(
            task_hint=task_hint,
            problem_review=problem_review,
            protocol_bundle=protocol_bundle,
            deterministic_data_access=deterministic_data_access,
            evaluation_contract=evaluation_contract,
            automl_context_pack=automl_context_pack,
            downstream_context=downstream_context,
        )
        desc_path = self.run_dir / "description.md"
        log_event(logger, "module.task_definition", "GENERATING_FILE", file="description.md")
        desc_path.write_text(desc, encoding="utf-8")
        log_event(logger, "module.task_definition", "GENERATED_FILE", file="description.md", defects=len(defects))

        report_payload = {
            "schema_version": "autorealize.task_definition_report.v1",
            "task_hint": task_hint,
            "plan": plan.model_dump() if hasattr(plan, "model_dump") else str(plan),
            "task_classification": task_cls.model_dump(),
            "problem_paradigm": problem_review.model_dump(),
            "description_protocol_bundle": protocol_bundle.model_dump(),
            "evaluation_contract": evaluation_contract.model_dump(),
            "automl_context_pack": automl_context_pack,
            "main_task_protocol": main_task_protocol,
            "downstream_context": downstream_context,
            "defects_after_gate": defects,
            "artifacts": {
                "description": "description.md",
                "sample_submission": "sample_submission.csv" if sample_path.exists() else None,
                "retrieved_knowledge": "retrieved_knowledge.json",
                "evaluation_contract": "evaluation_contract_report.json",
                "problem_paradigm": "problem_paradigm_report.json",
                "data_access_protocol": "data_access_protocol.json",
                "description_protocol_bundle": "description_protocol_bundle.json",
                "automl_context_pack": "automl_context_pack.json",
                "automl_context": "automl_context.md",
                "main_task_protocol": "main_task_protocol.json",
            },
        }
        write_json_safe(self.report_dir / "task_definition_report.json", report_payload, indent=2)
        log_event(logger, "module.task_definition", "GENERATED_FILE", file="realize_report/task_definition_report.json")
        self.services.trajectory.log("task_definition_module", "done", {"defects_after_gate": len(defects)})
        log_event(logger, "module.task_definition", "COMPLETED")
        return TaskDefinitionResult(
            description_path=desc_path,
            sample_submission_path=sample_path if sample_path.exists() else None,
            downstream_context=downstream_context,
            plan=plan,
            defects=defects,
        )
