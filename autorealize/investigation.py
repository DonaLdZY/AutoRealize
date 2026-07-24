from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd

from .config import AutoRealizeConfig
from .context_compiler import ArtifactStore, build_qdi_context_and_details, compact_detail_table_card_for_prompt
from .document_retrieval import LocalDocumentIndex
from .logging_utils import log_event
from .models import (
    InvestigationStepResult,
    InvestigationToolRequest,
    InvestigationAnswer,
    InvestigationQuestion,
    QuestionInvestigationAction,
    QuestionInvestigationAnswerSet,
    QuestionInvestigationPlan,
    QuestionInvestigationReport,
    ReadonlyPythonRequest,
)
from .prompt_cache import json_block, join_blocks, stable_dynamic_prompt
from .profiling.csv_utils import infer_csv_dialect, read_csv_auto
from .profiling.stats import read_table
from .utils.filesystem import rel
from .utils.safe_json import write_json_safe

logger = logging.getLogger(__name__)


BUILTIN_TOOL_NAMES = {"custom_readonly_python"}
QDI_EVIDENCE_ACTIONS = {
    "request_context",
    "search_document",
    "read_document_chunks",
    "read_qdi_artifact_excerpt",
    "request_script",
}


def run_question_investigator(
    *,
    cfg: AutoRealizeConfig,
    llm_client: Any,
    prompt_mgr: Any,
    data_root: Path,
    report_dir: Path,
    task_hint: str,
    file_summaries: list[Any],
    relation_hints: list[Any],
    constraint_memory: dict,
    authoritative_memory: dict,
    knowledge_base: dict,
) -> dict[str, Any]:
    """Run LLM-led cross-file investigation and persist a structured report."""
    report_path = report_dir / "question_investigation_report.json"
    if not bool(getattr(cfg.investigation, "enabled", True)):
        report = QuestionInvestigationReport(enabled=False, summary="Question-driven investigation is disabled.")
        _write_report(report_path, report.model_dump())
        return report.model_dump()

    try:
        report = _run_question_investigator_inner(
            cfg=cfg,
            llm_client=llm_client,
            prompt_mgr=prompt_mgr,
            data_root=data_root,
            task_hint=task_hint,
            file_summaries=file_summaries,
            relation_hints=relation_hints,
            constraint_memory=constraint_memory,
            authoritative_memory=authoritative_memory,
            knowledge_base=knowledge_base,
            report_dir=report_dir,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "module.data_cognition.investigator", "FAILED", error=str(exc)[:240])
        report = {
            "schema_version": "autorealize.question_investigation.v1",
            "enabled": True,
            "summary": "Question-driven investigation failed; downstream agents should rely on other memory layers.",
            "questions": [],
            "script_requests": [],
            "tool_requests": [],
            "step_results": [],
            "answers": [],
            "unresolved_questions": [str(exc)[:1000]],
            "context_routing_notes": ["Investigation failed before producing reliable evidence."],
            "error": str(exc),
        }
    _write_report(report_path, report)
    return report


def _run_question_investigator_inner(
    *,
    cfg: AutoRealizeConfig,
    llm_client: Any,
    prompt_mgr: Any,
    data_root: Path,
    task_hint: str,
    file_summaries: list[Any],
    relation_hints: list[Any],
    constraint_memory: dict,
    authoritative_memory: dict,
    knowledge_base: dict,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    log_event(logger, "module.data_cognition.investigator", "ACTIVATED")
    planner_prompt = prompt_mgr.load("system/question_investigator_planner.md")
    answerer_prompt = prompt_mgr.load("system/question_investigator_answerer.md")
    artifact_store = ArtifactStore(
        (report_dir or data_root) / "context_artifacts",
        default_visible_limit=int(getattr(cfg.context, "artifact_visible_excerpt_chars", 1200)),
    )
    document_index = LocalDocumentIndex.build(
        data_root=data_root,
        store_root=(report_dir or data_root) / "document_store",
        chunk_chars=int(getattr(cfg.investigation, "document_chunk_chars", 2200)),
        chunk_overlap_chars=int(getattr(cfg.investigation, "document_chunk_overlap_chars", 200)),
    )
    context, table_card_details = _build_investigation_context(
        cfg=cfg,
        data_root=data_root,
        task_hint=task_hint,
        file_summaries=file_summaries,
        relation_hints=relation_hints,
        constraint_memory=constraint_memory,
        authoritative_memory=authoritative_memory,
        knowledge_base=knowledge_base,
        artifact_store=artifact_store,
    )
    context["document_manifest"] = document_index.manifest_for_prompt()
    context["context_policy"]["full_document_text_local"] = True
    context["context_policy"]["document_retrieval_actions"] = ["search_document", "read_document_chunks"]
    tools = CrossFileInvestigationTools(
        cfg=cfg,
        data_root=data_root,
        authoritative_memory=authoritative_memory,
        knowledge_base=knowledge_base,
        artifact_store=artifact_store,
    )

    all_questions: dict[str, InvestigationQuestion] = {}
    question_records: dict[str, dict[str, Any]] = {}
    queue: list[str] = []
    all_requests: list[InvestigationToolRequest] = []
    all_results: list[InvestigationStepResult] = []
    answers: list[InvestigationAnswer] = []
    unresolved_questions: list[str] = []
    routing_notes: list[str] = []
    action_history: list[dict[str, Any]] = []
    action_digest_cards: list[dict[str, Any]] = []
    action_timeline: list[dict[str, Any]] = []
    working_memory_cards: dict[str, dict[str, Any]] = {}
    qdi_artifact_ids: set[str] = set()

    max_total_questions = max(1, int(getattr(cfg.investigation, "max_questions", 5)))
    max_actions_per_question = max(1, int(getattr(cfg.investigation, "max_rounds_per_run", 3)))
    max_depth = max(0, int(getattr(cfg.investigation, "question_bfs_max_depth", 3)))
    max_followups_per_question = max(
        0,
        int(getattr(cfg.investigation, "max_followup_questions_per_question", 3)),
    )
    max_scripts_per_question = max(0, int(getattr(cfg.investigation, "max_scripts_per_question", 3)))
    max_scripts_total = max_total_questions * max_scripts_per_question
    max_output_chars = int(getattr(cfg.investigation, "max_result_chars", 20000))

    live_report_path = (report_dir / "question_investigation_report.json") if report_dir is not None else None

    def persist_live_report(
        *,
        phase: str,
        summary: str,
        current_question_id: str = "",
        current_action: str = "",
        action_round: int = 0,
    ) -> None:
        if live_report_path is None:
            return
        live_report = QuestionInvestigationReport(
            enabled=True,
            summary=summary,
            questions=list(all_questions.values()),
            script_requests=[req.custom_python for req in all_requests if req.custom_python.python_code],
            tool_requests=all_requests,
            step_results=all_results,
            answers=answers,
            unresolved_questions=list(dict.fromkeys(unresolved_questions))[:80],
            context_routing_notes=list(dict.fromkeys(routing_notes))[:80],
            question_records=_question_records_for_prompt(question_records),
            action_history=action_history[:200],
            action_digest_cards=action_digest_cards[:200],
            working_memory_cards=list(working_memory_cards.values()),
            progress={
                "phase": phase,
                "current_question_id": current_question_id,
                "current_action": current_action,
                "action_round": action_round,
                "queued_questions": len(queue),
                "total_questions": len(question_records),
                "resolved_questions": len(answers),
                "unresolved_questions": len(
                    [record for record in question_records.values() if record.get("status") == "unresolved"]
                ),
                "actions": len(action_history),
                "script_attempts": len(all_results),
            },
        )
        _write_report(live_report_path, live_report.model_dump())

    persist_live_report(
        phase="planning",
        summary="QDI 正在根据数据认知结果规划需要进一步核实的问题。",
    )

    stable, dynamic = stable_dynamic_prompt(
        stable=context,
        dynamic={
            "instruction": (
                "生成本次问题驱动研究的初始问题队列。只提出真正阻塞任务定义、数据读取、文件关联、输出、评估或硬约束落地的问题；"
                "不要在 planner 阶段请求脚本。把 files 中的 CSV、表格型 JSON、Excel sheet 都按 table/file card 理解；"
                "file_cognition 只是短导航，权威事实看 authoritative_memory，硬约束和业务规则看 constraint_memory。"
            ),
            "max_questions": max_total_questions,
            "previous_question_records": [],
        },
        stable_title="Stable QDI compact context",
        dynamic_title="Dynamic initial QDI planning request",
    )
    plan = llm_client.ask_structured(
        model_cls=QuestionInvestigationPlan,
        system_prompt=planner_prompt,
        user_prompt=dynamic,
        prompt_name="question_investigator_initial_questions",
        static_context_prompt=stable,
        dynamic_user_prompt=dynamic,
    )
    if bool(getattr(cfg.investigation, "adaptive_budgeting", True)):
        if int(plan.recommended_max_questions or 0) > 0:
            max_total_questions = min(max_total_questions, max(1, int(plan.recommended_max_questions)))
        if int(plan.recommended_max_actions_per_question or 0) > 0:
            max_actions_per_question = min(
                max_actions_per_question,
                max(1, int(plan.recommended_max_actions_per_question)),
            )
        max_scripts_total = max_total_questions * max_scripts_per_question
    automatic_questions = _select_auto_verification_questions(
        entity_alias_questions=_entity_alias_verification_questions(context),
        population_questions=_population_verification_questions(context),
        relation_questions=_relation_verification_questions(context),
        reading_strategy_questions=_reading_strategy_verification_questions(context),
        limit=min(
            max(0, int(getattr(cfg.investigation, "automatic_verification_question_limit", 4))),
            max(0, max_total_questions - (1 if plan.questions and not plan.ready_to_answer else 0)),
        ),
    )
    for q in automatic_questions:
        qid = _add_question_record(
            question=q,
            all_questions=all_questions,
            question_records=question_records,
            queue=queue,
            parent_question_id="",
            depth=0,
            max_total_questions=max_total_questions,
        )
        if qid:
            log_event(
                logger,
                "module.data_cognition.investigator",
                "AUTO_VERIFICATION_QUESTION_QUEUED",
                question_id=qid,
                question=str(question_records[qid].get("question") or "")[:500],
                category=str(question_records[qid].get("category") or ""),
                priority=str(question_records[qid].get("priority") or "medium"),
                depth=0,
            )

    planner_questions = [] if plan.ready_to_answer else (plan.questions or [])
    for q in planner_questions[: max(0, max_total_questions - len(question_records))]:
        qid = _add_question_record(
            question=q,
            all_questions=all_questions,
            question_records=question_records,
            queue=queue,
            parent_question_id=str(getattr(q, "parent_question_id", "") or ""),
            depth=int(getattr(q, "depth", 0) or 0),
            max_total_questions=max_total_questions,
        )
        if qid:
            log_event(
                logger,
                "module.data_cognition.investigator",
                "QUESTION_QUEUED",
                question_id=qid,
                question=str(question_records[qid].get("question") or "")[:500],
                category=str(question_records[qid].get("category") or ""),
                priority=str(question_records[qid].get("priority") or "medium"),
                depth=0,
            )

    if not queue and (plan.script_requests or plan.tool_requests):
        for script in (plan.script_requests or [])[:max_total_questions]:
            q = InvestigationQuestion(
                question_id=script.question_id,
                question=script.goal or script.expected_output or script.reason_builtins_insufficient,
                category="other",
                why_blocking="Planner produced an investigation script; converted to a focused QDI question.",
                candidate_files=script.input_files,
            )
            _add_question_record(
                question=q,
                all_questions=all_questions,
                question_records=question_records,
                queue=queue,
                parent_question_id="",
                depth=0,
                max_total_questions=max_total_questions,
            )

    persist_live_report(
        phase="questions_queued",
        summary=f"QDI 已规划 {len(question_records)} 个调查问题，准备逐项核实。",
    )

    seen_request_keys: set[str] = set()
    while queue:
        qid = queue.pop(0)
        record = question_records.get(qid)
        question = all_questions.get(qid)
        if not record or question is None or record.get("status") not in {"pending", "refined"}:
            continue
        record["status"] = "investigating"
        log_event(
            logger,
            "module.data_cognition.investigator",
            "QUESTION_STARTED",
            question_id=qid,
            question=str(record.get("question") or "")[:500],
            category=str(record.get("category") or ""),
            priority=str(record.get("priority") or "medium"),
            depth=int(record.get("depth", 0) or 0),
        )
        persist_live_report(
            phase="investigating",
            summary=f"QDI 正在调查问题：{str(record.get('question') or qid)[:300]}",
            current_question_id=qid,
        )
        current_result: InvestigationStepResult | None = None
        current_request: InvestigationToolRequest | None = None
        scripts_for_question = 0
        context_retrievals_for_question = 0
        document_retrievals_for_question = 0
        artifact_retrievals_for_question = 0
        question_action_ledger: list[dict[str, Any]] = []
        question_live_actions: list[dict[str, Any]] = []
        working_memory = _empty_working_memory_card(qid)
        working_memory_cards[qid] = working_memory
        answerer_stable_prefix = _qdi_answerer_stable_prefix(context, record)

        for action_round in range(1, max_actions_per_question + 1):
            available_actions = _available_qdi_actions(
                question_records=question_records,
                current_record=record,
                scripts_for_question=scripts_for_question,
                context_retrievals_for_question=context_retrievals_for_question,
                document_retrievals_for_question=document_retrievals_for_question,
                has_documents=bool(document_index.documents),
                max_document_retrievals=max(
                    0, int(getattr(cfg.investigation, "document_retrievals_per_question", 4))
                ),
                has_qdi_artifacts=bool(qdi_artifact_ids),
                artifact_retrievals_for_question=artifact_retrievals_for_question,
                max_artifact_retrievals=max(
                    0, int(getattr(cfg.investigation, "artifact_retrievals_per_question", 2))
                ),
                total_scripts=len(all_requests),
                max_scripts_per_question=max_scripts_per_question,
                max_scripts_total=max_scripts_total,
                max_total_questions=max_total_questions,
                max_depth=max_depth,
                max_followups_per_question=max_followups_per_question,
                allow_custom_readonly_python=bool(getattr(cfg.investigation, "allow_custom_readonly_python", True)),
            )
            if action_round >= max_actions_per_question:
                available_actions = _terminal_qdi_actions(available_actions)
            persist_live_report(
                phase="deciding_action",
                summary=f"QDI 正在为问题 {qid} 决定第 {action_round} 个调查动作。",
                current_question_id=qid,
                current_action="decide_next_action",
                action_round=action_round,
            )
            pending_digest_requests = _pending_action_digest_requests(
                action_history,
                action_digest_cards,
                qid,
            )
            dynamic = join_blocks(
                json_block("1. Append-only action and digest timeline", action_timeline),
                json_block("2. Current evidence-linked working memory", working_memory),
                json_block(
                    "3. Recent live action window",
                    _recent_action_window(
                        question_live_actions,
                        count=int(getattr(cfg.investigation, "recent_action_count", 3)),
                    ),
                ),
                json_block("4. Compact question ledger", _question_records_for_prompt(question_records)),
                json_block("5. Current dynamic action state", {
                    "instruction": (
                        "基于当前证据回答、继续探查或标记未解决。action_timeline 是动作与摘要组成的追加式轨迹，"
                        "recent_action_window 是最近动作的精确可见区，working_memory 是证据关联的累计认知。"
                        "任何 truncated=true 的内容都不得根据未显示部分推断；需要旧 QDI 证据时可分段读取其 artifact。"
                        "不要为了保留历史而复写旧脚本；需要文档原文时优先 search_document/read_document_chunks。"
                    ),
                    "available_actions": available_actions,
                    "pending_action_digest_requests": pending_digest_requests,
                    "remaining_script_requests": max(0, max_scripts_total - len(all_requests)),
                    "remaining_followup_questions": max(0, max_total_questions - len(question_records)),
                    "current_depth": int(record.get("depth", 0) or 0),
                    "max_depth": max_depth,
                    "max_output_len": max_output_chars,
                    "current_question": record,
                    "context_retrieval_policy": (
                        "Stable table_cards are route-only manifests. If field meanings, field statistics, "
                        "reading notes, warnings, or sheet details are needed, choose "
                        "request_context; the retrieved excerpt will appear in recent_action_window on the next turn."
                    ),
                    "action_round": action_round,
                    "action_policy": {
                        "do_not_add_duplicate_questions": True,
                        "use_mark_duplicate_when_needed": True,
                        "use_refine_current_question_only_for_narrow_rewording": True,
                        "do_not_print_full_tables_in_scripts": True,
                        "script_results_are_artifact_backed": True,
                        "document_retrieval_does_not_consume_script_budget": True,
                        "artifact_retrieval_does_not_consume_script_or_repair_budget": True,
                        "working_memory_is_interpretive_not_authoritative": True,
                        "every_pending_action_digest_must_be_returned": True,
                        "final_round_allows_terminal_actions_only": action_round >= max_actions_per_question,
                    },
                }),
            )
            action = llm_client.ask_structured(
                model_cls=QuestionInvestigationAction,
                system_prompt=answerer_prompt,
                user_prompt=dynamic,
                prompt_name="question_investigator_action",
                static_context_prompt=answerer_stable_prefix,
                dynamic_user_prompt=dynamic,
            )
            action_name = str(action.action or "").strip().lower()
            log_event(
                logger,
                "module.data_cognition.investigator",
                "ACTION_SELECTED",
                question_id=qid,
                action=action_name,
                action_round=action_round,
                notes=str(action.notes or "")[:500],
            )
            applied_digest_sequences = _apply_action_digest_updates(
                action_history,
                action_digest_cards,
                action_timeline,
                action.action_digest_updates,
                question_id=qid,
                pending_sequences={int(item["sequence"]) for item in pending_digest_requests},
            )
            _backfill_action_digests_from_working_memory(
                action_history,
                action_digest_cards,
                action_timeline,
                action.working_memory_update,
                question_id=qid,
                pending_sequences={int(item["sequence"]) for item in pending_digest_requests}
                - applied_digest_sequences,
            )
            entry = _compact_action_history(qid, action_name, "selected", action)
            entry["sequence"] = len(action_history) + 1
            action_history.append(entry)
            question_action_ledger.append(entry)
            action_timeline.append(entry)
            live_entry = _new_live_action_entry(
                sequence=entry["sequence"],
                question_id=qid,
                action_name=action_name,
                action=action,
                artifact_store=artifact_store,
                script_chars=int(getattr(cfg.investigation, "recent_script_chars", 12000)),
            )
            question_live_actions.append(live_entry)
            qdi_artifact_ids.update(_artifact_ids_in(live_entry))
            working_memory = _merge_working_memory_card(
                working_memory,
                action.working_memory_update,
                sequence=entry["sequence"],
                max_chars=int(getattr(cfg.investigation, "working_memory_max_chars", 12000)),
            )
            working_memory_cards[qid] = working_memory
            persist_live_report(
                phase="executing_action",
                summary=f"QDI 正在执行问题 {qid} 的动作：{action_name or 'unknown'}。",
                current_question_id=qid,
                current_action=action_name,
                action_round=action_round,
            )
            if action_name not in available_actions:
                entry["status"] = "illegal_action"
                live_entry["observation"] = {"status": "illegal_action", "available_actions": available_actions}
                record["status"] = "unresolved"
                record["unresolved_reason"] = f"LLM selected unavailable action `{action_name}`."
                unresolved_questions.append(f"[{qid}] {record['question']} unresolved: {record['unresolved_reason']}")
                break

            if action_name == "answer":
                entry["status"] = "answered"
                live_entry["observation"] = _bounded_json_view(
                    {
                        "answer": action.answer,
                        "evidence": action.evidence,
                        "confidence": action.confidence,
                        "remaining_uncertainty": action.remaining_uncertainty,
                    },
                    int(getattr(cfg.investigation, "recent_result_chars", 8000)),
                )
                answer = InvestigationAnswer(
                    question_id=qid,
                    question=str(record.get("question", "")),
                    answer=str(action.answer or ""),
                    evidence=[str(x) for x in (action.evidence or [])[:12]],
                    confidence=str(action.confidence or "medium"),
                    remaining_uncertainty=str(action.remaining_uncertainty or ""),
                    downstream_notes=[str(x) for x in (action.downstream_notes or [])[:12]],
                )
                answers.append(answer)
                record.update(
                    {
                        "status": "resolved",
                        "short_answer": answer.answer[:1200],
                        "confidence": answer.confidence,
                        "used_files": [str(x) for x in (action.used_files or [])[:20]],
                    }
                )
                _add_followups_from_action(
                    action=action,
                    parent_record=record,
                    all_questions=all_questions,
                    question_records=question_records,
                    queue=queue,
                    max_total_questions=max_total_questions,
                    max_depth=max_depth,
                    max_followups_per_question=max_followups_per_question,
                )
                break

            if action_name == "give_up":
                entry["status"] = "given_up"
                live_entry["observation"] = {
                    "unresolved_reason": str(action.unresolved_reason or action.remaining_uncertainty or "")[:2000],
                    "what_was_tried": [str(x)[:500] for x in (action.what_was_tried or [])[:12]],
                }
                reason = str(action.unresolved_reason or action.remaining_uncertainty or "Evidence is insufficient.").strip()
                record.update({"status": "unresolved", "unresolved_reason": reason[:1200]})
                unresolved_questions.append(f"[{qid}] {record['question']} unresolved: {reason}")
                routing_notes.append(
                    f"[{qid}] Downstream agents must not assert facts that depend on this unresolved QDI question."
                )
                _add_followups_from_action(
                    action=action,
                    parent_record=record,
                    all_questions=all_questions,
                    question_records=question_records,
                    queue=queue,
                    max_total_questions=max_total_questions,
                    max_depth=max_depth,
                    max_followups_per_question=max_followups_per_question,
                )
                break

            if action_name == "mark_duplicate":
                entry["status"] = "marked_duplicate"
                duplicate_of = str(action.duplicate_of_question_id or "").strip()
                record.update({"status": "duplicate", "duplicate_of_question_id": duplicate_of})
                break

            if action_name == "refine_current_question":
                refined = str(action.refined_question or "").strip()
                if refined:
                    record["question"] = refined[:1000]
                    question.question = refined[:1000]
                    record["status"] = "refined"
                entry["status"] = "question_refined"
                live_entry["observation"] = {"refined_question": refined[:1000]}
                continue

            if action_name == "add_followup_questions":
                added = _add_followups_from_action(
                    action=action,
                    parent_record=record,
                    all_questions=all_questions,
                    question_records=question_records,
                    queue=queue,
                    max_total_questions=max_total_questions,
                    max_depth=max_depth,
                    max_followups_per_question=max_followups_per_question,
                )
                record.update(
                    {
                        "status": "expanded" if added else "unresolved",
                        "unresolved_reason": "" if added else "No legal non-duplicate follow-up question could be added.",
                    }
                )
                if not added:
                    unresolved_questions.append(f"[{qid}] {record['question']} unresolved: {record['unresolved_reason']}")
                entry["status"] = "followups_added" if added else "followup_rejected"
                live_entry["observation"] = {"added_followup_count": added}
                break

            if action_name == "request_context":
                entry["digest_policy"] = "summary_follows_as_separate_timeline_event"
                req_ctx = action.request_context
                req_ctx.question_id = req_ctx.question_id or qid
                retrieved = _retrieve_qdi_context_excerpt(
                    context=context,
                    table_card_details=table_card_details,
                    question_record=record,
                    request=req_ctx,
                    max_cards=2,
                )
                retrieval_result = {
                    "cards": retrieved,
                    "retrieval_policy": "Local deterministic context retrieval; this does not execute data scripts.",
                }
                context_retrievals_for_question += 1
                question_action_ledger[-1].update(
                    status="context_retrieved",
                    result="context_retrieved",
                    retrieved_tables=[str(card.get("table_id") or card.get("source_file") or "") for card in retrieved],
                )
                live_entry["observation"] = _artifact_backed_observation(
                    retrieval_result,
                    artifact_store=artifact_store,
                    artifact_type="qdi_context_retrieval_full",
                    source=f"{qid}:action:{entry['sequence']}",
                    max_chars=int(getattr(cfg.investigation, "recent_retrieval_chars", 8000)),
                )
                qdi_artifact_ids.update(_artifact_ids_in(live_entry))
                continue

            if action_name == "search_document":
                entry["digest_policy"] = "summary_follows_as_separate_timeline_event"
                request = action.search_document
                request.question_id = request.question_id or qid
                query = str(request.query or record.get("question", "")).strip()
                result = document_index.search(
                    query,
                    document_ids=request.document_ids,
                    source_files=request.source_files,
                    top_k=min(
                        max(1, int(request.top_k or 0)),
                        max(1, int(getattr(cfg.investigation, "document_search_top_k", 5))),
                    ),
                )
                document_retrievals_for_question += 1
                question_action_ledger[-1].update(
                    status="document_searched",
                    query=query[:500],
                    result="document_search",
                    hits=[
                        {
                            "chunk_id": match.get("chunk_id"),
                            "source_file": match.get("source_file"),
                            "locator": match.get("locator"),
                            "score": match.get("score"),
                            "evidence_excerpt": str(match.get("excerpt") or "")[:300],
                        }
                        for match in result.get("matches", [])
                    ],
                )
                live_entry["observation"] = _bounded_json_view(
                    result,
                    int(getattr(cfg.investigation, "recent_retrieval_chars", 8000)),
                )
                continue

            if action_name == "read_document_chunks":
                entry["digest_policy"] = "summary_follows_as_separate_timeline_event"
                request = action.read_document_chunks
                request.question_id = request.question_id or qid
                result = document_index.read_chunks(
                    request.chunk_ids,
                    neighbor_count=min(max(0, int(request.neighbor_count)), 2),
                    max_chunks=8,
                    max_chars=int(getattr(cfg.investigation, "document_retrieval_max_chars", 12000)),
                )
                document_retrievals_for_question += 1
                question_action_ledger[-1].update(
                    status="document_chunks_read",
                    result="document_chunks_read",
                    chunks=[
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "source_file": chunk.get("source_file"),
                            "locator": chunk.get("locator"),
                            "chars": chunk.get("chars"),
                            "evidence_excerpt": str(chunk.get("text") or "")[:300],
                        }
                        for chunk in result.get("chunks", [])
                    ],
                    truncated=bool(result.get("truncated")),
                )
                live_entry["observation"] = _bounded_json_view(
                    result,
                    int(getattr(cfg.investigation, "recent_retrieval_chars", 8000)),
                )
                continue

            if action_name == "read_qdi_artifact_excerpt":
                entry["digest_policy"] = "summary_follows_as_separate_timeline_event"
                request = action.read_qdi_artifact_excerpt
                request.question_id = request.question_id or qid
                artifact_id = str(request.artifact_id or "").strip()
                if artifact_id not in qdi_artifact_ids:
                    result = {
                        "status": "rejected",
                        "artifact_id": artifact_id,
                        "error": "artifact_not_created_or_exposed_in_current_qdi_run",
                    }
                else:
                    result = artifact_store.read_excerpt(
                        artifact_id,
                        offset=max(0, int(request.offset or 0)),
                        max_chars=min(
                            max(1, int(request.max_chars or 0)),
                            max(1, int(getattr(cfg.investigation, "artifact_retrieval_max_chars", 8000))),
                        ),
                        json_path=str(request.json_path or ""),
                    )
                artifact_retrievals_for_question += 1
                entry.update(
                    status="artifact_excerpt_read" if result.get("status") == "completed" else "artifact_excerpt_failed",
                    result="artifact_excerpt",
                    artifact_id=artifact_id,
                    offset=int(result.get("offset") or 0),
                    next_offset=int(result.get("next_offset") or 0),
                    has_more=bool(result.get("has_more")),
                    error=str(result.get("error") or "")[:500],
                )
                live_entry["observation"] = _bounded_json_view(
                    result,
                    int(getattr(cfg.investigation, "recent_retrieval_chars", 8000)),
                )
                continue

            if action_name == "request_script":
                entry["digest_policy"] = "summary_follows_as_separate_timeline_event"
                script_req = action.request_script
                script_req.question_id = script_req.question_id or qid
                script_req.goal = script_req.goal or str(action.notes or record.get("question", ""))
                req = InvestigationToolRequest(
                    request_id="",
                    question_id=qid,
                    tool_name="custom_readonly_python",
                    reason=script_req.goal or script_req.expected_output or script_req.reason_builtins_insufficient,
                    params={},
                    custom_python=script_req,
                )
                req = _normalize_request(req, len(all_requests) + 1)
                key = _request_signature(req)
                if key in seen_request_keys:
                    current_result = InvestigationStepResult(
                        request_id=f"{req.request_id}_duplicate",
                        question_id=qid,
                        tool_name="custom_readonly_python",
                        status="failed",
                        reason="duplicate_script_request",
                        error="duplicate_script_request",
                    )
                    entry["status"] = "duplicate_script_request"
                    live_entry["observation"] = {"status": "failed", "error": "duplicate_script_request"}
                    continue
                seen_request_keys.add(key)
                all_requests.append(req)
                scripts_for_question += 1
                attempt_results, final_req = _execute_request_with_repair(
                    req=req,
                    tools=tools,
                    llm_client=llm_client,
                    prompt_mgr=prompt_mgr,
                    cfg=cfg,
                    context=context,
                    table_card_details=table_card_details,
                    previous_questions=_question_records_for_prompt(question_records),
                    artifact_store=artifact_store,
                )
                current_request = final_req
                all_results.extend(attempt_results)
                current_result = attempt_results[-1]
                visible = _visible_output_payload(current_result.result, max_output_chars)
                stored_result_ref = (
                    current_result.result.get("_full_result_artifact", {})
                    if isinstance(current_result.result, dict)
                    else {}
                )
                question_action_ledger[-1].update(
                    status="script_completed" if current_result.status == "completed" else "script_failed",
                    result="script_completed" if current_result.status == "completed" else "script_failed",
                    request_id=current_result.request_id,
                    error=str(current_result.error or "")[:500],
                    output_truncated=bool(visible["output_truncated"]),
                    original_output_chars=visible["original_output_chars"],
                    visible_output_chars=visible["visible_output_chars"],
                    result_artifact_id=str(stored_result_ref.get("artifact_id") or ""),
                )
                script_evidence = _current_script_evidence(
                    current_request,
                    current_result,
                    int(getattr(cfg.investigation, "recent_result_chars", 8000)),
                    artifact_store=artifact_store,
                )
                script_evidence.pop("current_script", None)
                live_entry["observation"] = script_evidence
                qdi_artifact_ids.update(_artifact_ids_in(live_entry))
                log_event(
                    logger,
                    "module.data_cognition.investigator",
                    "SCRIPT_COMPLETED" if current_result.status == "completed" else "SCRIPT_FAILED",
                    tool=req.tool_name,
                    request_id=current_result.request_id,
                    question_id=current_result.question_id,
                    attempts=len(attempt_results),
                    error=current_result.error[:240],
                )
                persist_live_report(
                    phase="script_completed" if current_result.status == "completed" else "script_failed",
                    summary=(
                        f"问题 {qid} 的只读脚本已完成，正在分析结果。"
                        if current_result.status == "completed"
                        else f"问题 {qid} 的只读脚本失败，正在决定修复或改用其他证据。"
                    ),
                    current_question_id=qid,
                    current_action=action_name,
                    action_round=action_round,
                )
                continue

        if record.get("status") in {"investigating", "refined"}:
            record["status"] = "unresolved"
            record["unresolved_reason"] = "QDI action rounds exhausted before a final answer."
            unresolved_questions.append(f"[{qid}] {record['question']} unresolved: {record['unresolved_reason']}")

        log_event(
            logger,
            "module.data_cognition.investigator",
            "QUESTION_COMPLETED",
            question_id=qid,
            question=str(record.get("question") or "")[:500],
            question_status=str(record.get("status") or ""),
            short_answer=str(record.get("short_answer") or "")[:500],
            unresolved_reason=str(record.get("unresolved_reason") or "")[:500],
        )
        persist_live_report(
            phase="question_completed",
            summary=f"QDI 已完成问题 {qid}，状态：{str(record.get('status') or 'unknown')}。",
            current_question_id=qid,
        )

    summary = (
        f"问题驱动研究完成：初始问题 {len(plan.questions or [])} 个，"
        f"累计问题 {len(question_records)} 个，已解决 {len(answers)} 个，"
        f"未解决 {len([r for r in question_records.values() if r.get('status') == 'unresolved'])} 个，"
        f"执行脚本 {len(all_results)} 次。"
    )
    report = QuestionInvestigationReport(
        enabled=True,
        summary=summary,
        questions=list(all_questions.values()),
        script_requests=[req.custom_python for req in all_requests if req.custom_python.python_code],
        tool_requests=all_requests,
        step_results=all_results,
        answers=answers,
        unresolved_questions=list(dict.fromkeys(unresolved_questions))[:80],
        context_routing_notes=list(dict.fromkeys(routing_notes))[:80],
        question_records=_question_records_for_prompt(question_records),
        action_history=action_history[:200],
        action_digest_cards=action_digest_cards[:200],
        working_memory_cards=list(working_memory_cards.values()),
        progress={
            "phase": "completed",
            "current_question_id": "",
            "current_action": "",
            "action_round": 0,
            "queued_questions": 0,
            "total_questions": len(question_records),
            "resolved_questions": len(answers),
            "unresolved_questions": len(
                [record for record in question_records.values() if record.get("status") == "unresolved"]
            ),
            "actions": len(action_history),
            "script_attempts": len(all_results),
        },
    )
    log_event(
        logger,
        "module.data_cognition.investigator",
        "COMPLETED",
        questions=len(report.questions),
        scripts=len(report.step_results),
        answers=len(report.answers),
    )
    return report.model_dump()


def _execute_request_with_repair(
    *,
    req: InvestigationToolRequest,
    tools: "CrossFileInvestigationTools",
    llm_client: Any,
    prompt_mgr: Any,
    cfg: AutoRealizeConfig,
    context: dict[str, Any],
    table_card_details: dict[str, dict[str, Any]] | None = None,
    previous_questions: list[dict[str, Any]] | None = None,
    artifact_store: ArtifactStore | None = None,
) -> tuple[list[InvestigationStepResult], InvestigationToolRequest]:
    max_retries = max(0, int(getattr(cfg.investigation, "custom_python_max_retries", 3)))
    results: list[InvestigationStepResult] = []
    current = req
    for attempt in range(max_retries + 1):
        if attempt > 0 and not current.request_id.endswith(f"_retry{attempt + 1}"):
            current.request_id = f"{req.request_id}_retry{attempt + 1}"
        result = tools.execute(current)
        result.result.setdefault("attempt", attempt + 1)
        results.append(result)
        if result.status == "completed":
            return results, current
        if attempt >= max_retries:
            return results, current
        try:
            stable, dynamic = stable_dynamic_prompt(
                stable={
                    "context": _build_script_repair_context(
                        context=context,
                        request=current,
                        table_card_details=table_card_details or {},
                    ),
                    "script_contract": {
                        "function": "def analyze(input_dir: str, scratch_dir: str) -> dict",
                        "input_dir": "read-only",
                        "scratch_dir": "temporary writable, destroyed after execution",
                        "allowed_libraries": [
                            "pandas",
                            "numpy",
                            "json",
                            "math",
                            "statistics",
                            "re",
                            "csv",
                            "collections",
                            "itertools",
                            "pathlib",
                            "datetime",
                            "typing",
                        ],
                        "forbidden": [
                            "network access",
                            "writing outside scratch_dir",
                            "modifying input_dir",
                            "reading outside input_dir/scratch_dir",
                            "large stdout or full table dumps",
                        ],
                    },
                },
                dynamic={
                    "question_id": current.question_id,
                    "question": _question_text(previous_questions or [], current.question_id),
                    "previous_request": current.custom_python.model_dump(),
                    "failed_result": _failed_result_for_prompt(result, artifact_store=artifact_store),
                    "repair_instruction": "Return a corrected ReadonlyPythonRequest. Keep the same question_id and only change python_code/input_files/expected_output if needed.",
                },
                stable_title="Stable read-only script repair context",
                dynamic_title="Dynamic failed script result",
            )
            repaired = llm_client.ask_structured(
                model_cls=ReadonlyPythonRequest,
                system_prompt=prompt_mgr.load("system/question_investigator_script_repair.md"),
                user_prompt=dynamic,
                prompt_name="question_investigator_script_repair",
                static_context_prompt=stable,
                dynamic_user_prompt=dynamic,
            )
            current = InvestigationToolRequest(
                request_id=f"{req.request_id}_retry{attempt + 2}",
                question_id=repaired.question_id or current.question_id,
                tool_name="custom_readonly_python",
                reason=current.reason,
                params={},
                custom_python=repaired,
            )
        except Exception as exc:  # noqa: BLE001
            repair_failed = InvestigationStepResult(
                request_id=f"{req.request_id}_repair_failed_{attempt + 1}",
                question_id=current.question_id,
                tool_name="custom_readonly_python",
                status="failed",
                reason="script_repair_llm_failed",
                result={},
                error=str(exc)[:1000],
            )
            results.append(repair_failed)
            return results, current
    return results, current


def _add_question_record(
    *,
    question: InvestigationQuestion,
    all_questions: dict[str, InvestigationQuestion],
    question_records: dict[str, dict[str, Any]],
    queue: list[str],
    parent_question_id: str,
    depth: int,
    max_total_questions: int,
) -> str:
    if len(question_records) >= max_total_questions:
        return ""
    qtext = str(question.question or "").strip()
    if not qtext:
        return ""
    duplicate = _find_duplicate_question(question_records, qtext)
    if duplicate:
        return ""
    qid = str(question.question_id or "").strip() or f"q{len(question_records) + 1}"
    while qid in question_records:
        qid = f"q{len(question_records) + 1}"
    question.question_id = qid
    question.parent_question_id = parent_question_id
    question.depth = int(depth)
    all_questions[qid] = question
    question_records[qid] = {
        "question_id": qid,
        "question": qtext[:1000],
        "status": "pending",
        "short_answer": "",
        "unresolved_reason": "",
        "confidence": "",
        "used_files": [],
        "parent_id": parent_question_id,
        "depth": int(depth),
        "category": str(question.category or ""),
        "priority": str(question.priority or "medium"),
        "candidate_files": [str(x) for x in (question.candidate_files or [])[:20]],
        "why_blocking": str(question.why_blocking or "")[:800],
    }
    queue.append(qid)
    return qid


def _find_duplicate_question(question_records: dict[str, dict[str, Any]], question: str) -> str:
    norm = _norm_question(question)
    for qid, record in question_records.items():
        existing = _norm_question(str(record.get("question", "")))
        if not existing:
            continue
        if norm == existing:
            return qid
        if len(norm) >= 20 and (norm in existing or existing in norm):
            return qid
    return ""


def _norm_question(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _entity_alias_verification_questions(context: dict[str, Any]) -> list[InvestigationQuestion]:
    """Queue deterministic QDI checks for candidate entity aliases.

    These questions intentionally ask for evidence instead of asserting that
    similarly named business fields are interchangeable keys.
    """
    groups = context.get("entity_alias_candidates") if isinstance(context, dict) else []
    if not isinstance(groups, list):
        return []

    out: list[InvestigationQuestion] = []
    for group in groups[:6]:
        if not isinstance(group, dict):
            continue
        routing = group.get("qdi_routing") if isinstance(group.get("qdi_routing"), dict) else {}
        if routing and not bool(routing.get("recommended")):
            continue
        raw_fields = group.get("candidate_fields")
        fields = [item for item in raw_fields if isinstance(item, dict)] if isinstance(raw_fields, list) else []
        value_kinds = {
            str(item.get("value_kind", "")).strip()
            for item in fields
            if str(item.get("value_kind", "")).strip()
        }
        distinct_field_refs = {
            (
                str(item.get("source_collection", "") or item.get("source_file", "")),
                str(item.get("field", "")),
            )
            for item in fields
        }
        if len(distinct_field_refs) < 2:
            continue

        candidate_files = _dedupe_qdi_strings(
            [str(item.get("source_file", "")) for item in fields],
            limit=12,
        )
        source_collections = _dedupe_qdi_strings(
            [str(item.get("source_collection", "")) for item in fields],
            limit=12,
        )
        field_refs = []
        for item in fields[:16]:
            source = str(item.get("source_file", "") or "").strip()
            field = str(item.get("field", "") or "").strip()
            if not source or not field:
                continue
            sheet = str(item.get("sheet_name", "") or "").strip()
            table_ref = f"{source}::{sheet}" if sheet else source
            field_refs.append(f"{table_ref}.{field}")
        label = str(group.get("label") or "实体别名候选字段").strip()
        semantic_reason = str(group.get("reason", "") or "").strip()
        evidence_status = str(group.get("evidence_status", "semantic_candidate") or "semantic_candidate")
        relevance = str(group.get("task_relevance", "medium") or "medium")
        deterministic_coverage = group.get("deterministic_group_coverage") if isinstance(group.get("deterministic_group_coverage"), dict) else {}
        directional_coverage = deterministic_coverage.get("directional_coverage") if isinstance(deterministic_coverage.get("directional_coverage"), list) else []
        concept_id = re.sub(r"[^A-Za-z0-9_]+", "_", str(group.get("concept_id") or len(out) + 1)).strip("_")
        question_id = f"auto_entity_alias_{concept_id or len(out) + 1}"
        out.append(
            InvestigationQuestion(
                question_id=question_id,
                question=(
                    f"验证 `{label}` 是否可以作为同一实体键使用：对候选字段 "
                    f"{'、'.join(field_refs) if field_refs else '见 entity_alias_candidates'} "
                    f"分别统计非空数量、唯一值数量、两两唯一值交集、left/right coverage 和 join coverage。"
                    f"候选来源集合为 {source_collections or candidate_files}；先在每个 source_collection 内按 alias_family "
                    "读取全部成员文件并对唯一值取并集，再计算集合之间的方向性覆盖率，不能用单个文件代表整个集合。"
                    f"程序已计算的集合并集方向性覆盖证据为 {directional_coverage[:8]}；先复核并解释该证据，"
                    "只有 read_errors 或缺少目标集合时才重新编写脚本计算实体键覆盖。"
                    f"LLM 提出候选的语义理由为：{semantic_reason or '字段语义和来源角色可能相关'}。"
                    f"当前 evidence_status={evidence_status}，task_relevance={relevance}。"
                    "不要预设这些字段等价；覆盖率不足或方向明显不对称时，记录未确认映射及其对下游 join/筛选的影响。"
                    "脚本应使用 pathlib/pandas，先对实际列名做 strip 后保留原名映射，"
                    "并分别输出 `left_covered_by_right_ratio` 与 `right_covered_by_left_ratio` 的明确分母。"
                ),
                category="join_key",
                why_blocking=(
                    "名称不同但语义相近的字段可能表示同一实体，也可能只是相关但不等价的属性；"
                    "错误合并会使跨表 join、覆盖口径、特征构造或约束判断失真。"
                ),
                candidate_files=candidate_files,
                priority="high" if relevance == "high" or "code" in value_kinds else "medium",
            )
        )
    return out


def _population_verification_questions(context: dict[str, Any]) -> list[InvestigationQuestion]:
    candidates = context.get("population_verification_queue") if isinstance(context, dict) else []
    if not isinstance(candidates, list):
        return []
    out: list[InvestigationQuestion] = []
    for idx, item in enumerate(candidates[:4], start=1):
        if not isinstance(item, dict):
            continue
        table_id = str(item.get("table_id", "") or "").strip()
        sensitive = item.get("population_sensitive_fields") if isinstance(item.get("population_sensitive_fields"), list) else []
        if not table_id or not sensitive:
            continue
        out.append(
            InvestigationQuestion(
                question_id=f"auto_population_{idx}",
                question=(
                    f"核验 `{table_id}` 的评估/决策人口口径。程序证据：verified_row_count={item.get('verified_row_count')}，"
                    f"最佳统计主键候选={item.get('best_statistical_key_candidate')}，人口敏感字段={sensitive}。"
                    "结合权威任务文本判断：评估应覆盖全部主键，还是只覆盖必需字段有效的实体；给出 eligibility rule、"
                    "eligible count、excluded count、逐类排除原因与无法派生时的处理。异常日期只能作为 sentinel 候选，"
                    "不得未经原文或分布核验直接排除。不要混用 worksheet used range、解析行数、唯一主键数和非空行数。"
                ),
                category="evaluation_population",
                why_blocking=(
                    "人口口径会同时决定输出完整性、未分配/缺失惩罚、评估分母和最终分数；"
                    "若把物理行数、唯一实体数或必需字段有效数混为一谈，后续评价不可复现。"
                ),
                candidate_files=[table_id.split("::", 1)[0]],
                priority="high",
            )
        )
    return out


def _relation_verification_questions(context: dict[str, Any]) -> list[InvestigationQuestion]:
    candidates = context.get("relation_verification_queue") if isinstance(context, dict) else []
    if not isinstance(candidates, list):
        return []
    out: list[InvestigationQuestion] = []
    for idx, item in enumerate(candidates[:6], start=1):
        if not isinstance(item, dict):
            continue
        left_file = str(item.get("left_file", "") or "")
        right_file = str(item.get("right_file", "") or "")
        left_field = str(item.get("left_field", "") or "")
        right_field = str(item.get("right_field", "") or "")
        if not left_file or not right_file or not left_field or not right_field:
            continue
        out.append(
            InvestigationQuestion(
                question_id=f"auto_relation_{idx}",
                question=(
                    f"核验 `{left_file}`.`{left_field}` 与 `{right_file}`.`{right_field}` 是否是当前任务可用的关联键。"
                    "必须计算双向非空唯一值覆盖率、交集、重复/基数和匹配行比例，并区分业务同义、部分映射与偶然值重叠。"
                ),
                category="join_key",
                why_blocking="错误关联会污染约束、评估人口、输出映射和下游 join。",
                candidate_files=[left_file.split("::", 1)[0], right_file.split("::", 1)[0]],
                priority="high",
            )
        )
    return out


def _reading_strategy_verification_questions(context: dict[str, Any]) -> list[InvestigationQuestion]:
    cards = context.get("table_cards") if isinstance(context, dict) else []
    if not isinstance(cards, list):
        return []
    markers = [
        "header",
        "skiprows",
        "inspect",
        "sheet",
        "dialect",
        "encoding",
        "表头",
        "工作表",
        "读取",
        "解析",
        "文档式",
    ]
    out: list[InvestigationQuestion] = []
    for idx, card in enumerate(cards, start=1):
        if not isinstance(card, dict):
            continue
        table_id = str(card.get("table_id", "") or "")
        notes = [str(x) for x in (card.get("reading_notes") or [])]
        warnings = [str(x) for x in (card.get("warnings") or [])]
        evidence = notes + warnings
        text = " ".join(evidence).lower()
        if not table_id or not evidence or not any(marker in text for marker in markers):
            continue
        out.append(
            InvestigationQuestion(
                question_id=f"auto_reading_{idx}",
                question=(
                    f"核验 `{table_id}` 的实际读取合同。当前候选提示为：{evidence[:6]}。"
                    "请确认精确 sheet、header/skiprows、编码或分隔符；必须通过试读后的真实列稳定性和数据行证据验证。"
                ),
                category="data_access",
                why_blocking="读取参数错误会使后续所有字段语义、关系和评估结论建立在错误表格上。",
                candidate_files=[table_id.split("::", 1)[0]],
                priority="high",
            )
        )
        if len(out) >= 4:
            break
    return out


def _select_auto_verification_questions(
    *,
    entity_alias_questions: list[InvestigationQuestion],
    population_questions: list[InvestigationQuestion],
    limit: int,
    relation_questions: list[InvestigationQuestion] | None = None,
    reading_strategy_questions: list[InvestigationQuestion] | None = None,
) -> list[InvestigationQuestion]:
    """Reserve a small, diverse part of the QDI budget for evidence checks."""

    if limit <= 0:
        return []
    pools = [
        list(entity_alias_questions or []),
        list(population_questions or []),
        list(relation_questions or []),
        list(reading_strategy_questions or []),
    ]
    selected: list[InvestigationQuestion] = []
    seen: set[str] = set()

    # Take one from each evidence family first, then fill any remaining slots.
    for pool in pools:
        if not pool:
            continue
        question = pool.pop(0)
        signature = re.sub(r"\s+", "", str(question.question or "")).casefold()
        if signature and signature not in seen:
            seen.add(signature)
            selected.append(question)
        if len(selected) >= limit:
            return selected

    remaining = [question for pool in pools for question in pool]
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    remaining.sort(
        key=lambda question: (
            priority_rank.get(str(question.priority or "medium").lower(), 1),
            str(question.question_id or ""),
        )
    )
    for question in remaining:
        signature = re.sub(r"\s+", "", str(question.question or "")).casefold()
        if not signature or signature in seen:
            continue
        seen.add(signature)
        selected.append(question)
        if len(selected) >= limit:
            break
    return selected


def _dedupe_qdi_strings(values: list[str], *, limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _available_qdi_actions(
    *,
    question_records: dict[str, dict[str, Any]],
    current_record: dict[str, Any],
    scripts_for_question: int,
    context_retrievals_for_question: int,
    document_retrievals_for_question: int,
    has_documents: bool,
    max_document_retrievals: int,
    has_qdi_artifacts: bool = False,
    artifact_retrievals_for_question: int = 0,
    max_artifact_retrievals: int = 0,
    total_scripts: int,
    max_scripts_per_question: int,
    max_scripts_total: int,
    max_total_questions: int,
    max_depth: int,
    max_followups_per_question: int,
    allow_custom_readonly_python: bool,
) -> list[str]:
    actions = ["answer", "give_up", "refine_current_question", "mark_duplicate"]
    if context_retrievals_for_question < 2:
        actions.insert(1, "request_context")
    if has_documents and document_retrievals_for_question < max_document_retrievals:
        actions.insert(1, "read_document_chunks")
        actions.insert(1, "search_document")
    if has_qdi_artifacts and artifact_retrievals_for_question < max_artifact_retrievals:
        actions.insert(1, "read_qdi_artifact_excerpt")
    if allow_custom_readonly_python and scripts_for_question < max_scripts_per_question and total_scripts < max_scripts_total:
        actions.insert(1, "request_script")
    depth = int(current_record.get("depth", 0) or 0)
    existing_children = [
        r for r in question_records.values()
        if str(r.get("parent_id", "")) == str(current_record.get("question_id", ""))
    ]
    if (
        depth < max_depth
        and len(existing_children) < max_followups_per_question
        and len(question_records) < max_total_questions
    ):
        actions.insert(-1, "add_followup_questions")
    return actions


def _terminal_qdi_actions(actions: list[str]) -> list[str]:
    terminal = {"answer", "give_up", "mark_duplicate", "add_followup_questions"}
    selected = [action for action in actions if action in terminal]
    return selected or ["answer", "give_up"]


def _pending_action_digest_requests(
    action_history: list[dict[str, Any]],
    action_digest_cards: list[dict[str, Any]],
    question_id: str,
) -> list[dict[str, Any]]:
    summarized_sequences = {
        int(card.get("digest_for_sequence", 0) or 0)
        for card in action_digest_cards
        if str(card.get("question_id", "")) == str(question_id)
    }
    pending = []
    for entry in action_history:
        if str(entry.get("question_id", "")) != str(question_id):
            continue
        if str(entry.get("action", "")) not in QDI_EVIDENCE_ACTIONS:
            continue
        if int(entry.get("sequence", 0) or 0) in summarized_sequences:
            continue
        pending.append(
            {
                "sequence": int(entry.get("sequence", 0) or 0),
                "action": str(entry.get("action", "")),
                "status": str(entry.get("status", "")),
                "query": str(entry.get("query") or entry.get("context_query") or entry.get("document_query") or "")[:500],
                "request_id": str(entry.get("request_id", "")),
                "error": str(entry.get("error", ""))[:500],
                "retrieved_tables": [str(x) for x in (entry.get("retrieved_tables", []) or [])[:8]],
                "hits": [
                    {
                        "chunk_id": item.get("chunk_id"),
                        "source_file": item.get("source_file"),
                        "locator": item.get("locator"),
                        "evidence_excerpt": str(item.get("evidence_excerpt") or "")[:300],
                    }
                    for item in (entry.get("hits", []) or [])[:6]
                    if isinstance(item, dict)
                ],
                "chunks": [
                    {
                        "chunk_id": item.get("chunk_id"),
                        "source_file": item.get("source_file"),
                        "locator": item.get("locator"),
                        "evidence_excerpt": str(item.get("evidence_excerpt") or "")[:300],
                    }
                    for item in (entry.get("chunks", []) or [])[:6]
                    if isinstance(item, dict)
                ],
                "result_artifact_id": str(entry.get("result_artifact_id") or entry.get("artifact_id") or ""),
                "output_truncated": bool(entry.get("output_truncated") or entry.get("truncated")),
                "original_output_chars": int(entry.get("original_output_chars", 0) or 0),
                "visible_output_chars": int(entry.get("visible_output_chars", 0) or 0),
                "instruction": "Summarize this completed exploration from the matching recent_action_window evidence.",
            }
        )
    return pending[-3:]


def _apply_action_digest_updates(
    action_history: list[dict[str, Any]],
    action_digest_cards: list[dict[str, Any]],
    action_timeline: list[dict[str, Any]],
    updates: list[Any],
    *,
    question_id: str,
    pending_sequences: set[int],
) -> set[int]:
    applied: set[int] = set()
    by_sequence = {
        int(entry.get("sequence", 0) or 0): entry
        for entry in action_history
        if str(entry.get("question_id", "")) == str(question_id)
    }
    for update in updates or []:
        sequence = int(getattr(update, "sequence", 0) or 0)
        entry = by_sequence.get(sequence)
        if sequence not in pending_sequences or entry is None:
            continue
        card = {
            "event_type": "action_digest",
            "digest_for_sequence": sequence,
            "question_id": str(question_id),
            "action": str(entry.get("action", "")),
            "what_was_done": str(getattr(update, "what_was_done", "") or "")[:1000],
            "key_outputs": [str(x)[:700] for x in (getattr(update, "key_outputs", []) or [])[:6]],
            "temporary_conclusion": str(getattr(update, "temporary_conclusion", "") or "")[:1000],
            "remaining_gap": str(getattr(update, "remaining_gap", "") or "")[:1000],
            "evidence_refs": [str(x)[:500] for x in (getattr(update, "evidence_refs", []) or [])[:8]],
            "digest_source": "llm_action_digest",
        }
        action_digest_cards.append(card)
        action_timeline.append(card)
        applied.add(sequence)
    return applied


def _backfill_action_digests_from_working_memory(
    action_history: list[dict[str, Any]],
    action_digest_cards: list[dict[str, Any]],
    action_timeline: list[dict[str, Any]],
    update: Any,
    *,
    question_id: str,
    pending_sequences: set[int],
) -> None:
    """Preserve an action-linked card even when the model omits the explicit digest field."""
    if not pending_sequences:
        return
    facts = [str(x)[:700] for x in (getattr(update, "confirmed_facts", []) or [])[:4]]
    conclusions = [str(x)[:700] for x in (getattr(update, "temporary_conclusions", []) or [])[:3]]
    gaps = [str(x)[:700] for x in (getattr(update, "open_gaps", []) or [])[:3]]
    refs = [str(x)[:500] for x in (getattr(update, "evidence_refs", []) or [])[:8]]
    for entry in action_history:
        sequence = int(entry.get("sequence", 0) or 0)
        if sequence not in pending_sequences or str(entry.get("question_id", "")) != str(question_id):
            continue
        action = str(entry.get("action", ""))
        status = str(entry.get("status", ""))
        card = {
            "event_type": "action_digest",
            "digest_for_sequence": sequence,
            "question_id": str(question_id),
            "action": action,
            "what_was_done": f"Executed `{action}`; resulting status was `{status}`.",
            "key_outputs": facts + conclusions,
            "temporary_conclusion": conclusions[0] if conclusions else "",
            "remaining_gap": gaps[0] if gaps else "",
            "evidence_refs": refs,
            "digest_source": "llm_working_memory_fallback",
        }
        action_digest_cards.append(card)
        action_timeline.append(card)


def _question_records_for_prompt(question_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in question_records.values():
        out.append(
            {
                "question_id": str(record.get("question_id", "")),
                "question": str(record.get("question", ""))[:1000],
                "status": str(record.get("status", "")),
                "short_answer": str(record.get("short_answer", ""))[:1200],
                "unresolved_reason": str(record.get("unresolved_reason", ""))[:1000],
                "confidence": str(record.get("confidence", "")),
                "used_files": [str(x) for x in (record.get("used_files", []) or [])[:12]],
                "parent_id": str(record.get("parent_id", "")),
                "depth": int(record.get("depth", 0) or 0),
                "category": str(record.get("category", "")),
                "duplicate_of_question_id": str(record.get("duplicate_of_question_id", "")),
            }
        )
    return out


def _current_script_evidence(
    request: InvestigationToolRequest | None,
    result: InvestigationStepResult | None,
    max_output_chars: int,
    *,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any]:
    if request is None and result is None:
        return {}
    payload = result.model_dump() if result is not None else {}
    visible = _visible_output_payload(payload.get("result", {}), max_output_chars)
    was_truncated = bool(payload.get("output_truncated")) or bool(visible["output_truncated"])
    original_chars = max(
        int(payload.get("original_output_chars") or 0),
        int(visible["original_output_chars"]),
    )
    artifact_ref = {}
    stored_ref = payload.get("result", {}).get("_full_result_artifact") if isinstance(payload.get("result"), dict) else None
    if isinstance(stored_ref, dict):
        artifact_ref = stored_ref
    elif result is not None and artifact_store is not None:
        artifact_ref = artifact_store.put(
            "qdi_script_output_full",
            f"{result.question_id}:{result.request_id}",
            payload.get("result", {}),
            visible_excerpt=visible["current_visible_output"],
            visible_limit=max_output_chars,
        )
    return {
        "current_script": request.custom_python.model_dump() if request is not None else {},
        "status": payload.get("status", ""),
        "error": str(payload.get("error", ""))[:2000],
        "current_visible_output": visible["current_visible_output"],
        "truncated": was_truncated,
        "output_truncated": was_truncated,
        "max_output_len": max_output_chars,
        "original_output_chars": original_chars,
        "visible_output_chars": visible["visible_output_chars"],
        "current_output_artifact": artifact_ref,
        "instruction": (
            "完整结构化结果已保存到 current_output_artifact。truncated=true 表示当前 prompt 仅显示前缀；"
            "可以继续使用可见结论，但不能根据未显示部分推断。需要额外细节时生成更聚焦的新脚本。"
        ),
    }


def _visible_output_payload(value: Any, max_output_chars: int) -> dict[str, Any]:
    safe = _json_safe(value)
    text = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    visible = text[:max(1, int(max_output_chars))]
    return {
        "current_visible_output": visible,
        "output_truncated": len(text) > len(visible),
        "original_output_chars": len(text),
        "visible_output_chars": len(visible),
    }


def _build_script_repair_context(
    context: dict[str, Any],
    request: InvestigationToolRequest,
    table_card_details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    files = context.get("table_cards", []) if isinstance(context.get("table_cards"), list) else []
    relations = context.get("relations", []) if isinstance(context.get("relations"), list) else []
    wanted_files = {str(x).strip() for x in (request.custom_python.input_files or []) if str(x).strip()}
    wanted_cols = {str(x).strip() for x in (request.custom_python.focus_columns or []) if str(x).strip()}
    wanted_sheets = {str(x).strip() for x in (request.custom_python.focus_sheets or []) if str(x).strip()}
    related_cards = _select_related_table_cards(
        files,
        table_card_details or {},
        wanted_files=wanted_files,
        wanted_cols=wanted_cols,
        wanted_sheets=wanted_sheets,
        question_text="",
        max_cards=12,
    )
    related_relations = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        if (
            str(rel.get("left_file", "")) in wanted_files
            or str(rel.get("right_file", "")) in wanted_files
            or str(rel.get("left_field", "")) in wanted_cols
            or str(rel.get("right_field", "")) in wanted_cols
        ):
            related_relations.append(rel)
        if len(related_relations) >= 16:
            break
    return {
        "context_policy": context.get("context_policy", {}),
        "table_card_details": related_cards[:12],
        "relation_cards": related_relations[:16],
        "authoritative_memory": context.get("authoritative_memory", {}),
        "constraint_memory": context.get("constraint_memory", {}),
        "script_investigation_policy": context.get("script_investigation_policy", {}),
    }


def _related_table_card_details_for_prompt(
    *,
    context: dict[str, Any],
    table_card_details: dict[str, dict[str, Any]],
    question_record: dict[str, Any],
    request: InvestigationToolRequest | None,
    max_cards: int = 8,
) -> list[dict[str, Any]]:
    if max_cards <= 0:
        return []
    table_index = context.get("table_cards", []) if isinstance(context.get("table_cards"), list) else []
    candidate_files = {
        str(x).strip()
        for x in (question_record.get("candidate_files", []) or [])
        if str(x).strip()
    }
    wanted_cols: set[str] = set()
    wanted_sheets: set[str] = set()
    if request is not None and request.custom_python is not None:
        candidate_files.update(
            str(x).strip()
            for x in (request.custom_python.input_files or [])
            if str(x).strip()
        )
        wanted_cols.update(
            str(x).strip()
            for x in (request.custom_python.focus_columns or [])
            if str(x).strip()
        )
        wanted_sheets.update(
            str(x).strip()
            for x in (request.custom_python.focus_sheets or [])
            if str(x).strip()
        )
    return _select_related_table_cards(
        table_index,
        table_card_details,
        wanted_files=candidate_files,
        wanted_cols=wanted_cols,
        wanted_sheets=wanted_sheets,
        question_text=str(question_record.get("question", "") or ""),
        max_cards=max_cards,
    )


def _retrieve_qdi_context_excerpt(
    *,
    context: dict[str, Any],
    table_card_details: dict[str, dict[str, Any]],
    question_record: dict[str, Any],
    request: Any,
    max_cards: int,
) -> list[dict[str, Any]]:
    candidate_files = {
        str(x).strip()
        for x in (getattr(request, "input_files", []) or [])
        if str(x).strip()
    }
    wanted_cols = {
        str(x).strip()
        for x in (getattr(request, "focus_columns", []) or [])
        if str(x).strip()
    }
    wanted_sheets = {
        str(x).strip()
        for x in (getattr(request, "focus_sheets", []) or [])
        if str(x).strip()
    }
    explicit_table_ids = [
        str(x).strip()
        for x in (getattr(request, "table_ids", []) or [])
        if str(x).strip()
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for table_id in explicit_table_ids:
        card = table_card_details.get(table_id)
        if isinstance(card, dict):
            selected.append(_detail_card_for_prompt(card))
            seen.add(table_id)
        if len(selected) >= max_cards:
            return selected
    if selected:
        return selected
    table_index = context.get("table_cards", []) if isinstance(context.get("table_cards"), list) else []
    return _select_related_table_cards(
        table_index,
        table_card_details,
        wanted_files=candidate_files,
        wanted_cols=wanted_cols,
        wanted_sheets=wanted_sheets,
        question_text=" ".join(
            [
                str(question_record.get("question", "") or ""),
                str(getattr(request, "query", "") or ""),
                str(getattr(request, "reason", "") or ""),
            ]
        ),
        max_cards=max_cards,
    )


def _select_related_table_cards(
    table_index: list[Any],
    table_card_details: dict[str, dict[str, Any]],
    *,
    wanted_files: set[str],
    wanted_cols: set[str],
    wanted_sheets: set[str],
    question_text: str,
    max_cards: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[int, str]] = []
    qtext = _norm_match_text(question_text)
    wanted_files_norm = {_norm_match_text(x) for x in wanted_files if x}
    wanted_cols_norm = {_norm_match_text(x) for x in wanted_cols if x}
    wanted_sheets_norm = {_norm_match_text(x) for x in wanted_sheets if x}

    def score_item(item: dict[str, Any]) -> tuple[int, str] | None:
        if not isinstance(item, dict):
            return None
        table_id = str(item.get("table_id", "") or item.get("source_file", "") or "")
        if not table_id or table_id.startswith("__omitted_"):
            return None
        source = str(item.get("source_file", "") or "")
        sheet_name = str(item.get("sheet_name", "") or "")
        fields = item.get("fields", []) if isinstance(item.get("fields"), list) else []
        if not fields and isinstance(item.get("field_index"), list):
            fields = item.get("field_index", [])
        field_names = [str(x.get("name", "")) for x in fields if isinstance(x, dict)]
        if isinstance(item.get("field_hints"), list):
            field_names.extend(str(x) for x in item.get("field_hints", []) if str(x).strip())
        haystack_parts = [table_id, source, sheet_name, str(item.get("file_cognition", ""))]
        haystack_parts.extend(field_names)
        haystack = _norm_match_text(" ".join(haystack_parts))
        field_norms = {_norm_match_text(x) for x in field_names if x}
        score = 0
        for target in wanted_files_norm:
            if target and (target in haystack or haystack in target):
                score += 100
        for target in wanted_sheets_norm:
            if target and target in _norm_match_text(sheet_name):
                score += 80
        for target in wanted_cols_norm:
            if target and target in field_norms:
                score += 70
        if qtext:
            for token in _match_tokens(qtext):
                if token in haystack:
                    score += 6
        if not wanted_files and not wanted_cols and not wanted_sheets and score == 0:
            score = 1
        if score > 0:
            return score, table_id
        return None

    indexed_ids: set[str] = set()
    for item in table_index:
        if not isinstance(item, dict):
            continue
        table_id = str(item.get("table_id", "") or item.get("source_file", "") or "")
        if table_id:
            indexed_ids.add(table_id)
        scored_item = score_item(item)
        if scored_item is not None:
            scored.append(scored_item)

    # Stable prompts may intentionally omit many table manifests. Keep retrieval
    # reversible by searching the local detail map as a second-tier index.
    for table_id, item in table_card_details.items():
        if table_id in indexed_ids or not isinstance(item, dict):
            continue
        scored_item = score_item(item)
        if scored_item is not None:
            scored.append(scored_item)

    related = []
    seen: set[str] = set()
    for _score, table_id in sorted(scored, key=lambda x: (-x[0], x[1]))[: max(1, int(max_cards))]:
        if table_id in seen:
            continue
        card = table_card_details.get(table_id)
        if isinstance(card, dict):
            related.append(_detail_card_for_prompt(card))
            seen.add(table_id)
    return related


def _detail_card_for_prompt(card: dict[str, Any]) -> dict[str, Any]:
    return _json_safe(compact_detail_table_card_for_prompt(card, field_limit=12))


def _artifact_ref_index_for_prompt(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        k: ref.get(k)
        for k in ["artifact_id", "artifact_type", "source", "truncated", "original_chars", "artifact_path"]
        if ref.get(k) not in (None, "", [], {})
    }


def _norm_match_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _match_tokens(text: str) -> list[str]:
    return [x for x in re.split(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", text.lower()) if len(x) >= 2][:80]


def _failed_result_for_prompt(
    result: InvestigationStepResult,
    *,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any]:
    payload = result.model_dump()
    visible = _visible_output_payload(payload.get("result", {}), max(1000, int(result.max_output_chars or 12000)))
    payload["result"] = {
        "current_visible_output": visible["current_visible_output"],
        "output_truncated": visible["output_truncated"],
        "original_output_chars": visible["original_output_chars"],
        "visible_output_chars": visible["visible_output_chars"],
    }
    stored_ref = result.result.get("_full_result_artifact") if isinstance(result.result, dict) else None
    if isinstance(stored_ref, dict):
        payload["result_artifact"] = stored_ref
    elif artifact_store is not None:
        payload["result_artifact"] = artifact_store.put(
            "qdi_failed_script_result_full",
            f"{result.question_id}:{result.request_id}",
            result.result,
            visible_excerpt=visible["current_visible_output"],
            visible_limit=max(1000, int(result.max_output_chars or 12000)),
        )
    return payload


def _compact_action_history(question_id: str, action_name: str, status: str, action: QuestionInvestigationAction) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "action": action_name,
        "status": status,
        "answer": str(action.answer or "")[:500],
        "unresolved_reason": str(action.unresolved_reason or "")[:500],
        "duplicate_of_question_id": str(action.duplicate_of_question_id or ""),
        "followup_count": len(action.followup_questions or []),
        "requested_script": bool(action.request_script and action.request_script.python_code),
        "context_query": str(action.request_context.query or "")[:500],
        "document_query": str(action.search_document.query or "")[:500],
        "document_ids": [str(x) for x in (action.search_document.document_ids or [])[:8]],
        "requested_chunk_ids": [str(x) for x in (action.read_document_chunks.chunk_ids or [])[:12]],
        "requested_artifact_id": str(action.read_qdi_artifact_excerpt.artifact_id or ""),
        "artifact_offset": max(0, int(action.read_qdi_artifact_excerpt.offset or 0)),
        "working_memory_items": sum(
            len(getattr(action.working_memory_update, field, []) or [])
            for field in (
                "confirmed_facts",
                "temporary_conclusions",
                "evidence_refs",
                "open_gaps",
                "invalidated_hypotheses",
            )
        ),
        "notes": str(action.notes or "")[:500],
    }


def _add_followups_from_action(
    *,
    action: QuestionInvestigationAction,
    parent_record: dict[str, Any],
    all_questions: dict[str, InvestigationQuestion],
    question_records: dict[str, dict[str, Any]],
    queue: list[str],
    max_total_questions: int,
    max_depth: int,
    max_followups_per_question: int,
) -> int:
    parent_id = str(parent_record.get("question_id", ""))
    parent_depth = int(parent_record.get("depth", 0) or 0)
    if parent_depth >= max_depth:
        return 0
    existing_children = [
        r for r in question_records.values()
        if str(r.get("parent_id", "")) == parent_id
    ]
    remaining_for_parent = max(0, max_followups_per_question - len(existing_children))
    added = 0
    for item in (action.followup_questions or [])[:remaining_for_parent]:
        if len(question_records) >= max_total_questions:
            break
        q = InvestigationQuestion(
            question=str(item.question or ""),
            category=str(parent_record.get("category", "")),
            why_blocking=str(item.reason or ""),
            candidate_files=[str(x) for x in (item.candidate_files or [])[:20]],
            priority="medium",
        )
        qid = _add_question_record(
            question=q,
            all_questions=all_questions,
            question_records=question_records,
            queue=queue,
            parent_question_id=parent_id,
            depth=parent_depth + 1,
            max_total_questions=max_total_questions,
        )
        if qid:
            added += 1
    return added


def _question_text(questions: list[dict[str, Any]], question_id: str) -> str:
    for item in questions or []:
        if str(item.get("question_id", "")) == str(question_id):
            return str(item.get("question", ""))
    return str(question_id or "")


def _merge_question(existing: Any, incoming: Any) -> Any:
    """Allow the planner to refine the question list across QDI rounds."""
    for field in ["question", "category", "why_blocking", "priority"]:
        value = str(getattr(incoming, field, "") or "").strip()
        if value:
            setattr(existing, field, value)
    merged_files = []
    for value in list(getattr(existing, "candidate_files", []) or []) + list(getattr(incoming, "candidate_files", []) or []):
        text = str(value).strip()
        if text and text not in merged_files:
            merged_files.append(text)
    existing.candidate_files = merged_files[:40]
    return existing


def _merge_unresolved_failure_notes(
    *,
    questions: list[Any],
    step_results: list[InvestigationStepResult],
    answer_set: QuestionInvestigationAnswerSet,
) -> tuple[list[str], list[str]]:
    """Add deterministic unresolved notes for questions whose scripts never succeeded."""
    unresolved = [str(x).strip() for x in (answer_set.unresolved_questions or []) if str(x).strip()]
    routing_notes = [str(x).strip() for x in (answer_set.context_routing_notes or []) if str(x).strip()]
    results_by_question: dict[str, list[InvestigationStepResult]] = {}
    for result in step_results:
        qid = str(result.question_id or "").strip()
        if qid:
            results_by_question.setdefault(qid, []).append(result)

    question_by_id = {str(getattr(q, "question_id", "") or "").strip(): q for q in questions}
    for qid, results in results_by_question.items():
        if any(str(r.status).lower() == "completed" for r in results):
            continue
        question = str(getattr(question_by_id.get(qid), "question", "") or qid)
        last_error = str(results[-1].error or results[-1].result.get("error", "") if results else "").strip()
        note = (
            f"[{qid}] `{question}` was not resolved because all read-only investigation scripts failed"
            f" after {len(results)} attempt(s)."
        )
        if last_error:
            note += f" Last error: {last_error[:300]}"
        note += " Treat this point as undefined unless another authoritative source resolves it."
        if not _contains_question_note(unresolved, qid):
            unresolved.append(note)
        route_note = (
            f"[{qid}] Downstream task-definition agents must not assert, hard-code, or optimize against facts "
            "that depend on this unresolved investigation point; keep the affected requirement unknown or "
            "state the needed verification explicitly."
        )
        if not _contains_question_note(routing_notes, qid):
            routing_notes.append(route_note)

    return list(dict.fromkeys(unresolved))[:80], list(dict.fromkeys(routing_notes))[:80]


def _contains_question_note(items: list[str], question_id: str) -> bool:
    marker = f"[{question_id}]"
    return any(marker in str(item) for item in items)


class CrossFileInvestigationTools:
    """Script-only read-only executor used by the LLM investigator.

    Older deterministic helpers remain as internal Python methods for debugging
    and compatibility, but the public `execute()` surface intentionally accepts
    only LLM-authored `custom_readonly_python` requests. This keeps the
    Question-Driven Investigator aligned with the current backend design:
    no tool menu, just sandboxed evidence scripts.
    """

    def __init__(
        self,
        *,
        cfg: AutoRealizeConfig,
        data_root: Path,
        authoritative_memory: dict,
        knowledge_base: dict,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.data_root = data_root.resolve()
        self.authoritative_memory = authoritative_memory if isinstance(authoritative_memory, dict) else {}
        self.knowledge_base = knowledge_base if isinstance(knowledge_base, dict) else {}
        self.artifact_store = artifact_store

    def execute(self, request: InvestigationToolRequest) -> InvestigationStepResult:
        tool = (request.tool_name or "").strip()
        if tool not in BUILTIN_TOOL_NAMES:
            return _failed_result(request, f"unsupported_tool:{tool}")
        try:
            if not bool(getattr(self.cfg.investigation, "allow_custom_readonly_python", True)):
                return _failed_result(request, "custom_readonly_python_disabled")
            result = run_custom_readonly_python(
                request.custom_python.python_code,
                input_dir=self.data_root,
                cfg=self.cfg,
            )
            max_chars = int(getattr(self.cfg.investigation, "max_result_chars", 20000))
            visible = _visible_output_payload(result, max_chars)
            full_result_artifact = {}
            if self.artifact_store is not None:
                full_result_artifact = self.artifact_store.put(
                    "qdi_script_output_full",
                    f"{request.question_id}:{request.request_id}",
                    result,
                    visible_excerpt=visible["current_visible_output"],
                    visible_limit=max_chars,
                )
            visible_result = _truncate_result(result, max_chars)
            if isinstance(visible_result, dict) and full_result_artifact:
                visible_result["_full_result_artifact"] = full_result_artifact
            if isinstance(result, dict) and result.get("error"):
                return _failed_result(request, str(result.get("error")), result=visible_result)
            return InvestigationStepResult(
                request_id=request.request_id,
                question_id=request.question_id,
                tool_name=tool,
                status="completed",
                reason=request.reason,
                result=visible_result if isinstance(visible_result, dict) else {"value": visible_result},
                output_truncated=bool(visible["output_truncated"]),
                max_output_chars=max_chars,
                original_output_chars=int(visible["original_output_chars"]),
                visible_output_chars=int(visible["visible_output_chars"]),
            )
        except Exception as exc:  # noqa: BLE001
            return _failed_result(request, str(exc))

    def schema_compare(self, params: dict[str, Any]) -> dict[str, Any]:
        files = _as_list(params.get("files"))[:12]
        schemas: list[dict[str, Any]] = []
        column_sets: list[set[str]] = []
        for name in files:
            path = self._resolve_input_file(name)
            df = self._read_table(path, max_rows=200)
            cols = [str(c) for c in df.columns]
            column_sets.append(set(cols))
            schemas.append(
                {
                    "file": rel(path, self.data_root),
                    "shape_sampled": [int(df.shape[0]), int(df.shape[1])],
                    "columns": cols,
                    "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                }
            )
        common = sorted(set.intersection(*column_sets)) if column_sets else []
        union = sorted(set.union(*column_sets)) if column_sets else []
        return {
            "files": schemas,
            "common_columns": common,
            "all_columns": union,
            "only_in_file": [
                {
                    "file": item["file"],
                    "columns": sorted(set(item["columns"]) - set(common)),
                }
                for item in schemas
            ],
        }

    def join_coverage(self, params: dict[str, Any]) -> dict[str, Any]:
        left_path, right_path, left_keys, right_keys = self._join_params(params)
        left = self._read_table(left_path)
        right = self._read_table(right_path)
        left_key_series, left_missing = _composite_key(left, left_keys)
        right_key_series, right_missing = _composite_key(right, right_keys)
        right_unique = set(right_key_series[right_key_series.notna()].astype(str))
        left_non_null = left_key_series.dropna().astype(str)
        matched = left_non_null.isin(right_unique)
        left_duplicate_keys = int(left_non_null.duplicated().sum())
        right_non_null = right_key_series.dropna().astype(str)
        right_duplicate_keys = int(right_non_null.duplicated().sum())
        return {
            "left_file": rel(left_path, self.data_root),
            "right_file": rel(right_path, self.data_root),
            "left_keys": left_keys,
            "right_keys": right_keys,
            "left_rows_sampled": int(left.shape[0]),
            "right_rows_sampled": int(right.shape[0]),
            "left_key_null_rows": int(left_missing.sum()),
            "right_key_null_rows": int(right_missing.sum()),
            "left_duplicate_key_rows": left_duplicate_keys,
            "right_duplicate_key_rows": right_duplicate_keys,
            "matched_left_key_rows": int(matched.sum()),
            "unmatched_left_key_rows": int((~matched).sum()),
            "left_to_right_coverage": round(float(matched.mean()), 6) if len(matched) else 0.0,
            "unmatched_left_keys_sample": left_non_null[~matched].drop_duplicates().head(self._max_sample_rows()).tolist(),
        }

    def anti_join(self, params: dict[str, Any]) -> dict[str, Any]:
        left_path, right_path, left_keys, right_keys = self._join_params(params)
        side = str(params.get("side") or "left").lower()
        left = self._read_table(left_path)
        right = self._read_table(right_path)
        left_key_series, _ = _composite_key(left, left_keys)
        right_key_series, _ = _composite_key(right, right_keys)
        if side == "right":
            other_keys = set(left_key_series.dropna().astype(str))
            keys = right_key_series.astype(str)
            mask = right_key_series.notna() & (~keys.isin(other_keys))
            sample = right.loc[mask].head(self._max_sample_rows())
            sample_keys = keys[mask].drop_duplicates().head(self._max_sample_rows()).tolist()
            source_file = right_path
        else:
            other_keys = set(right_key_series.dropna().astype(str))
            keys = left_key_series.astype(str)
            mask = left_key_series.notna() & (~keys.isin(other_keys))
            sample = left.loc[mask].head(self._max_sample_rows())
            sample_keys = keys[mask].drop_duplicates().head(self._max_sample_rows()).tolist()
            source_file = left_path
        return {
            "side": side if side == "right" else "left",
            "source_file": rel(source_file, self.data_root),
            "unmatched_rows": int(mask.sum()),
            "unmatched_keys_sample": sample_keys,
            "unmatched_rows_sample": _records(sample),
        }

    def foreign_key_check(self, params: dict[str, Any]) -> dict[str, Any]:
        child_file = params.get("child_file") or params.get("left_file")
        parent_file = params.get("parent_file") or params.get("right_file")
        child_keys = _keys(params, "child_keys", "child_key") or _keys(params, "left_keys", "left_key")
        parent_keys = _keys(params, "parent_keys", "parent_key") or _keys(params, "right_keys", "right_key")
        return self.join_coverage(
            {
                "left_file": child_file,
                "right_file": parent_file,
                "left_keys": child_keys,
                "right_keys": parent_keys,
            }
        )

    def group_cardinality(self, params: dict[str, Any]) -> dict[str, Any]:
        file_path = self._resolve_input_file(params.get("file"))
        group_by = _keys(params, "group_by", "group_by")
        count_fields = _as_list(params.get("count_fields"))[:8]
        df = self._read_table(file_path)
        _ensure_columns(df, group_by)
        grouped = df.groupby(group_by, dropna=False)
        sizes = grouped.size().sort_values(ascending=False)
        result: dict[str, Any] = {
            "file": rel(file_path, self.data_root),
            "group_by": group_by,
            "rows_sampled": int(df.shape[0]),
            "group_count": int(sizes.shape[0]),
            "group_size_stats": _series_stats(sizes),
            "top_groups": [
                {"key": _json_safe(idx), "rows": int(value)}
                for idx, value in sizes.head(self._max_sample_rows()).items()
            ],
        }
        field_stats: dict[str, Any] = {}
        for col in count_fields:
            if col in df.columns:
                field_stats[col] = {
                    "nunique_per_group_stats": _series_stats(grouped[col].nunique(dropna=True)),
                    "non_null_per_group_stats": _series_stats(grouped[col].count()),
                }
        if field_stats:
            result["field_group_stats"] = field_stats
        return result

    def sample_id_group_check(self, params: dict[str, Any]) -> dict[str, Any]:
        groups = [x for x in self.knowledge_base.get("filename_sample_groups", []) if isinstance(x, dict)]
        required_kinds = {str(x).strip() for x in _as_list(params.get("required_kinds")) if str(x).strip()}
        kind_counts: dict[str, int] = {}
        missing_examples: list[dict[str, Any]] = []
        for item in groups:
            kinds = {str(x) for x in (item.get("data_kinds") or {}).values()}
            for kind in kinds:
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            missing = sorted(required_kinds - kinds)
            if missing and len(missing_examples) < self._max_sample_rows():
                missing_examples.append(
                    {
                        "sample_id": item.get("sample_id"),
                        "missing_kinds": missing,
                        "files": item.get("files", [])[:12],
                    }
                )
        return {
            "group_count": len(groups),
            "required_kinds": sorted(required_kinds),
            "kind_counts": kind_counts,
            "groups_sample": groups[: self._max_sample_rows()],
            "missing_required_kind_examples": missing_examples,
        }

    def submission_shape_check(self, params: dict[str, Any]) -> dict[str, Any]:
        sample_file = params.get("sample_file") or params.get("file") or "sample_submission.csv"
        path = self._resolve_input_file(sample_file)
        df = self._read_table(path, max_rows=200)
        contract = self._submission_contract()
        sample_cols = [str(c) for c in df.columns]
        expected_cols = [str(c) for c in contract.get("columns", []) if str(c).strip()]
        return {
            "sample_file": rel(path, self.data_root),
            "columns": sample_cols,
            "row_count_sampled": int(df.shape[0]),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "preview": _records(df.head(self._max_sample_rows())),
            "authoritative_contract": contract,
            "matches_authoritative_columns": bool(expected_cols and sample_cols == expected_cols),
            "column_mismatch": {
                "expected": expected_cols,
                "got": sample_cols,
            }
            if expected_cols and sample_cols != expected_cols
            else {},
        }

    def csv_dialect_probe(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_input_file(params.get("file"))
        if path.suffix.lower() != ".csv":
            return {"error": f"not_csv:{rel(path, self.data_root)}"}
        hint = infer_csv_dialect(path)
        encoding = _detect_encoding(path)
        try:
            df = read_csv_auto(path, nrows=min(20, self._max_sample_rows()))
            read_ok = True
            error = ""
        except Exception as exc:  # noqa: BLE001
            df = pd.DataFrame()
            read_ok = False
            error = str(exc)
        sep_repr = repr(hint.sep)
        kwargs = [f"sep={sep_repr}", f"encoding={encoding!r}"]
        if hint.engine:
            kwargs.append(f"engine={hint.engine!r}")
        read_example = f"pd.read_csv({str(path.name)!r}, {', '.join(kwargs)})"
        return {
            "file": rel(path, self.data_root),
            "read_ok": read_ok,
            "error": error,
            "encoding": encoding,
            "sep": hint.sep,
            "engine": hint.engine,
            "inferred": hint.inferred,
            "reason": hint.reason,
            "read_example": read_example,
            "columns": [str(c) for c in df.columns],
            "preview": _records(df.head(self._max_sample_rows())) if read_ok else [],
        }

    def _join_params(self, params: dict[str, Any]) -> tuple[Path, Path, list[str], list[str]]:
        left_file = params.get("left_file")
        right_file = params.get("right_file")
        left_keys = _keys(params, "left_keys", "left_key")
        right_keys = _keys(params, "right_keys", "right_key")
        if not right_keys and left_keys:
            right_keys = list(left_keys)
        if not left_keys and right_keys:
            left_keys = list(right_keys)
        if not left_file or not right_file or not left_keys or not right_keys:
            raise ValueError("join tool requires left_file/right_file and left_keys/right_keys")
        return self._resolve_input_file(left_file), self._resolve_input_file(right_file), left_keys, right_keys

    def _read_table(self, path: Path, max_rows: int | None = None) -> pd.DataFrame:
        if max_rows is None:
            max_rows = getattr(self.cfg.investigation, "tool_sample_rows", 50000)
        return read_table(
            path,
            json_flatten_sep=self.cfg.data.json_flatten_sep,
            json_flatten_max_level=self.cfg.data.json_flatten_max_level,
            json_keep_raw_nested_columns=self.cfg.data.json_keep_raw_nested_columns,
            max_rows=max_rows,
        )

    def _resolve_input_file(self, value: Any) -> Path:
        if value is None or not str(value).strip():
            raise ValueError("missing file path")
        raw = str(value).strip().replace("\\", "/").lstrip("./")
        path = (self.data_root / raw).resolve()
        if not _is_relative_to(path, self.data_root):
            raise PermissionError(f"path escapes input_dir: {value}")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"input file not found: {value}")
        return path

    def _max_sample_rows(self) -> int:
        return max(1, int(getattr(self.cfg.investigation, "max_sample_rows", 20)))

    def _submission_contract(self) -> dict[str, Any]:
        contract = self.authoritative_memory.get("submission_contract")
        return contract if isinstance(contract, dict) else {}


def run_custom_readonly_python(code: str, *, input_dir: Path, cfg: AutoRealizeConfig) -> dict[str, Any]:
    """Execute LLM-provided analysis code with input_dir read-only and scratch_dir temporary writable."""
    code = str(code or "").strip()
    if not code:
        return {"error": "empty_custom_python_code"}
    issues = validate_custom_readonly_python(code)
    if issues:
        return {"error": "custom_python_static_validation_failed", "issues": issues}

    timeout = float(getattr(cfg.investigation, "custom_python_timeout_seconds", 30.0))
    max_stdout = int(getattr(cfg.investigation, "custom_python_max_stdout_chars", 12000))
    with tempfile.TemporaryDirectory(prefix="autorealize_scratch_") as scratch:
        scratch_dir = Path(scratch).resolve()
        wrapper_path = scratch_dir / "run_custom_readonly.py"
        result_path = scratch_dir / "__autorealize_result.json"
        stdout_path = scratch_dir / "__autorealize_stdout.log"
        stderr_path = scratch_dir / "__autorealize_stderr.log"
        wrapper_path.write_text(_custom_python_wrapper(code), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                proc = subprocess.run(
                    [sys.executable, str(wrapper_path), str(input_dir.resolve()), str(scratch_dir), str(result_path)],
                    cwd=str(scratch_dir),
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            return {
                "error": f"custom_python_timeout:{timeout}",
                "stdout_tail": _read_text_tail(stdout_path, max_stdout),
                "stderr_tail": _read_text_tail(stderr_path, max_stdout),
            }
        stdout_info = _read_text_tail_info(stdout_path, max_stdout)
        stderr_info = _read_text_tail_info(stderr_path, max_stdout)
        stdout = stdout_info["visible_tail"]
        stderr = stderr_info["visible_tail"]
        if proc.returncode != 0:
            return {
                "error": f"custom_python_exit_code:{proc.returncode}",
                "stdout_tail": stdout,
                "stderr_tail": stderr,
            }
        if not result_path.exists():
            return {
                "error": "custom_python_missing_result_file",
                "stdout_tail": stdout,
                "stderr_tail": stderr,
            }
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "error": f"custom_python_result_not_json:{exc}",
                "stdout_tail": stdout,
                "stderr_tail": stderr,
            }
        if not isinstance(result, dict):
            return {"error": "custom_python_result_not_object", "result": _json_safe(result)}
        result["_scratch_destroyed_after_execution"] = True
        if stdout_info["original_bytes"]:
            result["_stdout_capture"] = stdout_info
        if stderr.strip():
            result["_stderr_capture"] = stderr_info
        return result


def _read_text_tail(path: Path, limit: int) -> str:
    return str(_read_text_tail_info(path, limit)["visible_tail"])


def _read_text_tail_info(path: Path, limit: int) -> dict[str, Any]:
    if limit <= 0 or not path.exists():
        return {"visible_tail": "", "truncated": False, "original_bytes": 0, "visible_chars": 0}
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max(limit * 4, limit)))
            visible = stream.read().decode("utf-8", errors="replace")[-limit:]
            return {
                "visible_tail": visible,
                "truncated": size > len(visible.encode("utf-8", errors="replace")),
                "original_bytes": size,
                "visible_chars": len(visible),
            }
    except OSError:
        return {"visible_tail": "", "truncated": False, "original_bytes": 0, "visible_chars": 0}


def validate_custom_readonly_python(code: str) -> list[str]:
    """Static safety checks for custom read-only analysis code."""
    issues: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax_error:{exc}"]

    banned_import_roots = {
        "os",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "shutil",
        "pickle",
        "joblib",
        "importlib",
        "tempfile",
        "http",
        "ftplib",
        "pathlib2",
    }
    allowed_import_roots = {
        "__future__",
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "statsmodels",
        "polars",
        "pyarrow",
        "fastparquet",
        "networkx",
        "rapidfuzz",
        "xarray",
        "h5py",
        "tables",
        "zarr",
        "openpyxl",
        "xlrd",
        "pyxlsb",
        "odf",
        "json",
        "math",
        "statistics",
        "re",
        "csv",
        "collections",
        "itertools",
        "pathlib",
        "datetime",
        "typing",
    }
    banned_calls = {"eval", "exec", "compile", "__import__", "input", "breakpoint", "help"}
    banned_attrs = {
        "unlink",
        "rmdir",
        "rename",
        "replace",
        "remove",
        "removedirs",
        "rmtree",
        "copy",
        "copyfile",
        "move",
        "chmod",
        "chown",
        "symlink_to",
        "hardlink_to",
    }
    has_analyze = False

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            pass
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            pass
        else:
            issues.append(f"top_level_side_effect_not_allowed:{type(node).__name__}")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "analyze":
            has_analyze = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in banned_import_roots:
                    issues.append(f"banned_import:{alias.name}")
                elif root not in allowed_import_roots:
                    issues.append(f"import_not_allowlisted:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in banned_import_roots:
                issues.append(f"banned_import:{node.module}")
            elif root and root not in allowed_import_roots:
                issues.append(f"import_not_allowlisted:{node.module}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in banned_calls:
                issues.append(f"banned_call:{name}")
            attr = _call_attr(node.func)
            if attr in banned_attrs:
                issues.append(f"banned_method:{attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            issue = _literal_path_issue(node.value)
            if issue:
                issues.append(issue)

    if not has_analyze:
        issues.append("missing_analyze_function")
    return list(dict.fromkeys(issues))[:40]


def _custom_python_wrapper(user_code: str) -> str:
    return (
        _CUSTOM_PYTHON_WRAPPER_PREFIX
        + "\nUSER_CODE = "
        + repr(user_code)
        + "\n"
        + _CUSTOM_PYTHON_WRAPPER_SUFFIX
    )


_CUSTOM_PYTHON_WRAPPER_PREFIX = r'''
from __future__ import annotations

import builtins
import json
from pathlib import Path
import socket
import sys
import traceback

import numpy as np
import pandas as pd

INPUT_DIR = Path(sys.argv[1]).resolve()
SCRATCH_DIR = Path(sys.argv[2]).resolve()
RESULT_PATH = Path(sys.argv[3]).resolve()

_orig_open = builtins.open
_orig_path_open = Path.open
_orig_path_read_text = Path.read_text
_orig_path_read_bytes = Path.read_bytes
_orig_path_write_text = Path.write_text
_orig_path_write_bytes = Path.write_bytes
_orig_path_iterdir = Path.iterdir
_orig_path_glob = Path.glob
_orig_path_rglob = Path.rglob


def _under(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents


def _resolve_user_path(file):
    if isinstance(file, int):
        return None
    return Path(file).resolve()


def _mode_is_write(mode):
    mode = str(mode or "r")
    return any(ch in mode for ch in ["w", "a", "x", "+"])


def _guard_read_path(file):
    path = _resolve_user_path(file)
    if path is None:
        return
    if not (_under(path, INPUT_DIR) or _under(path, SCRATCH_DIR)):
        raise PermissionError(f"readonly sandbox refused read outside input_dir/scratch_dir: {path}")


def _guard_write_path(file):
    path = _resolve_user_path(file)
    if path is None:
        return
    if not _under(path, SCRATCH_DIR):
        raise PermissionError(f"readonly sandbox refused write outside scratch_dir: {path}")


def _safe_open(file, *args, **kwargs):
    mode = args[0] if args else kwargs.get("mode", "r")
    if _mode_is_write(mode):
        _guard_write_path(file)
    else:
        _guard_read_path(file)
    return _orig_open(file, *args, **kwargs)


def _safe_path_open(self, *args, **kwargs):
    mode = args[0] if args else kwargs.get("mode", "r")
    if _mode_is_write(mode):
        _guard_write_path(self)
    else:
        _guard_read_path(self)
    return _orig_path_open(self, *args, **kwargs)


def _safe_read_text(self, *args, **kwargs):
    _guard_read_path(self)
    return _orig_path_read_text(self, *args, **kwargs)


def _safe_read_bytes(self, *args, **kwargs):
    _guard_read_path(self)
    return _orig_path_read_bytes(self, *args, **kwargs)


def _safe_write_text(self, *args, **kwargs):
    _guard_write_path(self)
    return _orig_path_write_text(self, *args, **kwargs)


def _safe_write_bytes(self, *args, **kwargs):
    _guard_write_path(self)
    return _orig_path_write_bytes(self, *args, **kwargs)


def _safe_iterdir(self):
    _guard_read_path(self)
    return _orig_path_iterdir(self)


def _safe_glob(self, pattern):
    _guard_read_path(self)
    return _orig_path_glob(self, pattern)


def _safe_rglob(self, pattern):
    _guard_read_path(self)
    return _orig_path_rglob(self, pattern)


builtins.open = _safe_open
Path.open = _safe_path_open
Path.read_text = _safe_read_text
Path.read_bytes = _safe_read_bytes
Path.write_text = _safe_write_text
Path.write_bytes = _safe_write_bytes
Path.iterdir = _safe_iterdir
Path.glob = _safe_glob
Path.rglob = _safe_rglob


def _blocked_socket(*args, **kwargs):
    raise PermissionError("network access is disabled in custom_readonly_python")


socket.socket = _blocked_socket


def _patch_pandas_writer(cls, method_name):
    if not hasattr(cls, method_name):
        return
    original = getattr(cls, method_name)

    def _safe_writer(self, path_or_buf=None, *args, **kwargs):
        if path_or_buf is not None and not hasattr(path_or_buf, "write"):
            _guard_write_path(path_or_buf)
        return original(self, path_or_buf, *args, **kwargs)

    setattr(cls, method_name, _safe_writer)


for _method in ["to_csv", "to_excel", "to_json", "to_parquet", "to_pickle", "to_feather", "to_hdf"]:
    _patch_pandas_writer(pd.DataFrame, _method)
    _patch_pandas_writer(pd.Series, _method)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        visible = [_json_safe(v) for v in values[:200]]
        if len(values) > len(visible):
            visible.append({
                "_truncated_sequence": True,
                "original_items": len(values),
                "visible_items": 200,
            })
        return visible
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value
'''


_CUSTOM_PYTHON_WRAPPER_SUFFIX = r'''
try:
    namespace = {}
    exec(compile(USER_CODE, "<custom_readonly_python>", "exec"), namespace)
    analyze = namespace.get("analyze")
    if not callable(analyze):
        raise RuntimeError("custom code must define analyze(input_dir: str, scratch_dir: str) -> dict")
    result = analyze(str(INPUT_DIR), str(SCRATCH_DIR))
    _orig_path_write_text(
        RESULT_PATH,
        json.dumps(_json_safe(result), ensure_ascii=False, default=str),
        encoding="utf-8",
    )
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


def _build_investigation_context(
    *,
    cfg: AutoRealizeConfig,
    data_root: Path,
    task_hint: str,
    file_summaries: list[Any],
    relation_hints: list[Any],
    constraint_memory: dict,
    authoritative_memory: dict,
    knowledge_base: dict,
    artifact_store: ArtifactStore | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    return build_qdi_context_and_details(
        cfg=cfg,
        data_root=data_root,
        task_hint=task_hint,
        file_summaries=file_summaries,
        relation_hints=relation_hints,
        constraint_memory=constraint_memory,
        authoritative_memory=authoritative_memory,
        knowledge_base=knowledge_base,
        artifact_store=artifact_store,
    )


def _legacy_build_investigation_context_unused(
    *,
    cfg: AutoRealizeConfig,
    data_root: Path,
    task_hint: str,
    file_summaries: list[Any],
    relation_hints: list[Any],
    constraint_memory: dict,
    authoritative_memory: dict,
    knowledge_base: dict,
) -> dict[str, Any]:
    files = []
    for fs in file_summaries[:120]:
        files.append(
            {
                "path": str(getattr(fs, "path", "")),
                "role": getattr(getattr(fs, "role", ""), "value", str(getattr(fs, "role", ""))),
                "file_cognition": str(getattr(fs, "summary", ""))[:1000],
                "tables_or_sheets": _compact_table_sheet_inventory(fs),
                "field_semantics": _compact_field_semantics(fs),
                "warnings": [str(x)[:300] for x in (getattr(fs, "warnings", []) or [])[:8]],
                "reading_notes": _compact_reading_notes(fs),
            }
        )
    relations = []
    for hint in relation_hints[:80]:
        payload = hint.model_dump() if hasattr(hint, "model_dump") else dict(getattr(hint, "__dict__", {}))
        relations.append(
            {
                "left_file": str(payload.get("left_file", "")),
                "left_field": str(payload.get("left_field", "")),
                "right_file": str(payload.get("right_file", "")),
                "right_field": str(payload.get("right_field", "")),
                "relation_type": str(payload.get("relation_type", "shared_attribute")),
                "confidence": payload.get("confidence", 0.0),
                "short_evidence": str(payload.get("short_evidence", "") or payload.get("reason", ""))[:800],
            }
        )
    compact_filename_groups = _compact_filename_sample_groups(
        (knowledge_base or {}).get("filename_sample_groups", []),
        file_summaries=file_summaries,
    )
    return {
        "task_hint": task_hint,
        "data_root_name": data_root.name,
        "files": files,
        "relations": relations,
        "constraint_memory": _compact_constraint_memory(constraint_memory),
        "authoritative_memory": _compact_authoritative_memory(authoritative_memory),
        "sampled_filename_patterns": _compact_sampled_patterns((knowledge_base or {}).get("sampled_filename_patterns", [])),
        "filename_sample_groups": compact_filename_groups,
        "field_glossary": _compact_field_glossary((knowledge_base or {}).get("field_glossary") or {}),
        "script_investigation_policy": {
            "input_dir": "read-only",
            "scratch_dir": "temporary writable and destroyed after custom Python execution",
            "custom_python_contract": "def analyze(input_dir: str, scratch_dir: str) -> dict",
            "allowed_libraries": [
                "pandas",
                "numpy",
                "json",
                "math",
                "statistics",
                "re",
                "csv",
                "collections",
                "itertools",
                "pathlib",
                "datetime",
                "typing",
            ],
            "output_policy": "Return a compact JSON-compatible dict. Do not print full tables; aggregate, sample, and truncate.",
        },
        "limits": {
            "max_questions": getattr(cfg.investigation, "max_questions", 5),
            "max_rounds_per_run": getattr(cfg.investigation, "max_rounds_per_run", 3),
            "allow_custom_readonly_python": getattr(cfg.investigation, "allow_custom_readonly_python", True),
            "question_bfs_max_depth": getattr(cfg.investigation, "question_bfs_max_depth", 3),
            "max_followup_questions_per_question": getattr(
                cfg.investigation,
                "max_followup_questions_per_question",
                3,
            ),
        },
    }


def _compact_source_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in [
        "parsed_kind",
        "shape",
        "columns",
        "probe_result_keys",
        "compact_image_dir",
        "sample_files",
        "csv_dialect",
        "csv_encoding",
        "excel_sheet_names",
        "excel_default_sheet",
        "excel_sheets",
        "excel_sheet_profiles",
        "json_first_level_schema",
        "json_strategy",
        "json_root_type",
        "preview_rows_used",
        "profile_sampling",
    ]:
        if key in meta:
            out[key] = _json_safe(meta[key])
    if "probe_results" in meta:
        out["probe_results"] = _truncate_result(meta["probe_results"], 4000)
    sheet_profiles = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
    if sheet_profiles:
        out["sheet_inventory"] = [
            {
                "sheet_name": sheet.get("sheet_name", ""),
                "shape": sheet.get("shape"),
                "read_examples": [
                    f"pd.read_excel(path, sheet_name={str(sheet.get('sheet_name', ''))!r})",
                    f"pd.read_excel(path, sheet_name={str(sheet.get('sheet_name', ''))!r}, header=None)",
                ],
                "columns": [str(x) for x in (sheet.get("columns") or [])[:40]],
                "raw_preview": _truncate_result(sheet.get("raw_preview", []), 1200),
                "preview": _truncate_result(sheet.get("preview", []), 1200),
            }
            for sheet in sheet_profiles[:30]
            if isinstance(sheet, dict)
        ]
    return out


def _compact_table_sheet_inventory(fs: Any) -> list[dict[str, Any]]:
    meta = getattr(fs, "source_metadata", {}) or {}
    sheet_profiles = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
    if sheet_profiles:
        out = []
        for sheet in sheet_profiles[:80]:
            if not isinstance(sheet, dict):
                continue
            out.append(
                {
                    "sheet_name": str(sheet.get("sheet_name", "")),
                    "shape": sheet.get("shape") or sheet.get("shape_profiled") or sheet.get("shape_sampled"),
                    "columns": [str(x) for x in (sheet.get("columns") or [])[:80]],
                    "profile_policy": str(sheet.get("profile_policy", "")),
                    "sheet_group_id": str(sheet.get("sheet_group_id", "")),
                    "sheet_group_size": sheet.get("sheet_group_size", 1),
                    "is_deep_profiled": bool(sheet.get("is_deep_profiled")),
                    "column_profiles": _compact_column_profiles(sheet.get("column_profiles", []), limit=40),
                }
            )
        return out
    return [
        {
            "table": str(getattr(fs, "path", "")),
            "shape": meta.get("shape"),
            "columns": [str(x) for x in list(getattr(fs, "columns", []) or [])[:120]],
            "column_profiles": _compact_column_profiles(getattr(fs, "column_profiles", []) or [], limit=80),
        }
    ]


def _compact_column_profiles(profiles: Any, *, limit: int) -> list[dict[str, Any]]:
    out = []
    for p in list(profiles or [])[:limit]:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "logical_type": p.get("logical_type") or p.get("dtype"),
                "row_count": p.get("row_count"),
                "non_null_count": p.get("non_null_count"),
                "null_ratio": p.get("null_ratio"),
                "unique_count": p.get("unique_count"),
                "top_values": p.get("top_values", [])[:6] if isinstance(p.get("top_values"), list) else p.get("top_values"),
                "numeric_stats": _small_dict(p.get("numeric_stats", {}), keys=["mean", "std", "var", "min", "max"]),
                "datetime_stats": _small_dict(p.get("datetime_stats", {}), keys=["min", "max", "range_days", "granularity"]),
            }
        )
    return out


def _compact_field_semantics(fs: Any) -> dict[str, str]:
    semantics = getattr(fs, "column_semantics", {}) or {}
    if not semantics:
        return {}
    selected: dict[str, str] = {}
    priority_cols = []
    for col, desc in semantics.items():
        text = f"{col} {desc}".lower()
        if _is_core_field_text(text):
            priority_cols.append((col, desc))
    for col, desc in priority_cols[:80]:
        selected[str(col)] = str(desc)[:300]
    if len(selected) < 40:
        for col, desc in semantics.items():
            if str(col) in selected:
                continue
            selected[str(col)] = str(desc)[:300]
            if len(selected) >= 40:
                break
    return selected


def _compact_reading_notes(fs: Any) -> list[str]:
    path = str(getattr(fs, "path", "") or "")
    meta = getattr(fs, "source_metadata", {}) or {}
    notes: list[str] = []
    if path.lower().endswith((".xlsx", ".xls")):
        sheet_profiles = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
        if sheet_profiles:
            sheet_names = [str(s.get("sheet_name", "")) for s in sheet_profiles[:20] if isinstance(s, dict)]
            notes.append(f"Excel workbook; read needed sheets explicitly with pandas.read_excel(..., sheet_name=...). Sheets: {', '.join(sheet_names)}")
    if path.lower().endswith(".csv"):
        dialect = meta.get("csv_dialect") if isinstance(meta.get("csv_dialect"), dict) else {}
        encoding = str(meta.get("csv_encoding", "") or "")
        sep = dialect.get("sep") or dialect.get("delimiter")
        engine = dialect.get("engine")
        if encoding and encoding.lower() not in {"utf-8", "utf-8-sig"}:
            notes.append(f"CSV encoding hint: {encoding}")
        if sep and str(sep) not in {",", ""}:
            notes.append(f"CSV non-default separator hint: sep={sep!r}, engine={engine or 'default'}")
    if path.lower().endswith(".json"):
        strategy = str(meta.get("json_strategy", "") or "")
        if strategy:
            notes.append(f"JSON parse strategy: {strategy}")
    return notes[:8]


def _compact_constraint_memory(memory: Any) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    return {
        "summary": str(memory.get("summary", ""))[:1200],
        "items": memory.get("items", [])[:40] if isinstance(memory.get("items", []), list) else [],
        "unresolved_questions": [str(x)[:500] for x in (memory.get("unresolved_questions", []) or [])[:20]],
    }


def _compact_authoritative_memory(memory: Any) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    return {
        "summary": str(memory.get("summary", ""))[:1500],
        "task_goal": str(memory.get("task_goal", ""))[:1500],
        "source_files": [str(x) for x in (memory.get("source_files", []) or [])[:30]],
        "input_requirements": [str(x)[:500] for x in (memory.get("input_requirements", []) or [])[:20]],
        "output_requirements": [str(x)[:500] for x in (memory.get("output_requirements", []) or [])[:20]],
        "evaluation_requirements": [str(x)[:500] for x in (memory.get("evaluation_requirements", []) or [])[:20]],
        "constraints": [str(x)[:500] for x in (memory.get("constraints", []) or [])[:30]],
        "leakage_guards": [str(x)[:500] for x in (memory.get("leakage_guards", []) or [])[:20]],
        "unresolved_questions": [str(x)[:500] for x in (memory.get("unresolved_questions", []) or [])[:20]],
        "authority_conflicts": (memory.get("authority_conflicts", []) or [])[:20],
        "submission_contract": memory.get("submission_contract", {}) if isinstance(memory.get("submission_contract", {}), dict) else {},
    }


def _compact_sampled_patterns(patterns: Any) -> list[dict[str, Any]]:
    out = []
    for item in list(patterns or [])[:40]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "directory": str(item.get("directory", "")),
                "pattern": str(item.get("pattern", "")),
                "total": item.get("total"),
                "sampled_count": len(item.get("sampled", []) or []),
                "skipped_count": len(item.get("skipped", []) or []),
                "sampled": [str(x) for x in (item.get("sampled", []) or [])[:3]],
                "sampling_reason": str(item.get("sampling_reason", ""))[:300],
            }
        )
    return out


def _compact_filename_sample_groups(groups: Any, *, file_summaries: list[Any]) -> list[dict[str, Any]]:
    summary_by_path = {str(getattr(fs, "path", "")): fs for fs in file_summaries}
    out = []
    for group in list(groups or [])[:80]:
        if not isinstance(group, dict):
            continue
        files = [str(x) for x in (group.get("files", []) or [])]
        reps = files[:3]
        column_profile = _filename_group_column_profile(files, summary_by_path)
        shared_columns = column_profile.get("shared_fields", [])
        variant_fields_by_file = column_profile.get("variant_fields_by_file", [])
        field_presence = column_profile.get("field_presence", [])
        out.append(
            {
                "directory": str(group.get("directory", "")),
                "template_or_sample_id": str(group.get("sample_id", "")),
                "file_count": len(files),
                "representative_files": reps,
                "data_kinds": {str(k): str(v) for k, v in list((group.get("data_kinds", {}) or {}).items())[:8]},
                "shared_fields": shared_columns[:40],
                "variant_fields_by_file": variant_fields_by_file[:12],
                "field_presence": field_presence[:24],
                "structure_consistent": bool(shared_columns) and not bool(variant_fields_by_file),
                "short_evidence": (
                    f"文件组 `{group.get('sample_id', '')}` 共 {len(files)} 个文件；"
                    f"代表文件 {', '.join(reps)}；共享字段 {', '.join(shared_columns[:12]) or '未知'}；"
                    f"差异字段 {len(field_presence)} 个。"
                ),
            }
        )
    return out


def _immutable_question_card(record: dict[str, Any]) -> dict[str, Any]:
    """Freeze question identity separately from changing status/budget fields."""
    return {
        "question_id": str(record.get("question_id", "")),
        "question": str(record.get("question", ""))[:1000],
        "category": str(record.get("category", "")),
        "why_blocking": str(record.get("why_blocking", ""))[:1000],
        "candidate_files": [str(x) for x in (record.get("candidate_files", []) or [])[:20]],
        "parent_id": str(record.get("parent_id", "")),
        "depth": int(record.get("depth", 0) or 0),
    }


def _qdi_answerer_stable_prefix(context: dict[str, Any], record: dict[str, Any]) -> str:
    """Keep global context first so providers can reuse it across QDI questions."""
    return join_blocks(
        json_block("1. Frozen global QDI context", context),
        json_block("2. Immutable initial current-question card", _immutable_question_card(record)),
    )


def _empty_working_memory_card(question_id: str) -> dict[str, Any]:
    return {
        "question_id": str(question_id or ""),
        "confirmed_facts": [],
        "temporary_conclusions": [],
        "evidence_refs": [],
        "open_gaps": [],
        "invalidated_hypotheses": [],
        "recommended_next_focus": "",
        "last_updated_sequence": 0,
        "trust_policy": (
            "This is an LLM-authored interpretation of visible evidence, not an authority layer. "
            "Parser/script facts and exact source excerpts take precedence."
        ),
    }


def _merge_working_memory_card(
    card: dict[str, Any],
    update: Any,
    *,
    sequence: int,
    max_chars: int,
) -> dict[str, Any]:
    merged = dict(card or {})
    list_fields = (
        "confirmed_facts",
        "temporary_conclusions",
        "evidence_refs",
        "open_gaps",
        "invalidated_hypotheses",
    )
    for field in list_fields:
        existing = [str(x)[:800] for x in (merged.get(field) or []) if str(x).strip()]
        incoming = [str(x)[:800] for x in (getattr(update, field, []) or []) if str(x).strip()]
        merged[field] = _dedupe_qdi_strings(existing + incoming, limit=24)
    focus = str(getattr(update, "recommended_next_focus", "") or "").strip()
    if focus:
        merged["recommended_next_focus"] = focus[:1200]
    merged["last_updated_sequence"] = int(sequence)
    limit = max(1000, int(max_chars))
    trim_order = (
        "temporary_conclusions",
        "open_gaps",
        "invalidated_hypotheses",
        "confirmed_facts",
        "evidence_refs",
    )
    while len(json.dumps(merged, ensure_ascii=False, sort_keys=True, default=str)) > limit:
        changed = False
        for field in trim_order:
            values = merged.get(field)
            if isinstance(values, list) and len(values) > 1:
                values.pop(0)
                changed = True
                break
        if not changed:
            merged["recommended_next_focus"] = str(merged.get("recommended_next_focus") or "")[:400]
            break
    return merged


def _recent_action_window(actions: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    return actions[-max(1, int(count)) :]


def _bounded_json_view(value: Any, max_chars: int) -> dict[str, Any]:
    safe = _json_safe(value)
    text = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return {
            "content": safe,
            "truncated": False,
            "original_chars": len(text),
            "visible_chars": len(text),
        }
    return {
        "visible_excerpt": text[:limit],
        "truncated": True,
        "original_chars": len(text),
        "visible_chars": limit,
    }


def _artifact_backed_observation(
    value: Any,
    *,
    artifact_store: ArtifactStore,
    artifact_type: str,
    source: str,
    max_chars: int,
) -> dict[str, Any]:
    view = _bounded_json_view(value, max_chars)
    text = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, default=str)
    ref = artifact_store.put(
        artifact_type,
        source,
        value,
        visible_excerpt=text[: max(1, int(max_chars))],
        visible_limit=max_chars,
    )
    view["artifact_ref"] = ref
    return view


def _new_live_action_entry(
    *,
    sequence: int,
    question_id: str,
    action_name: str,
    action: QuestionInvestigationAction,
    artifact_store: ArtifactStore,
    script_chars: int,
) -> dict[str, Any]:
    request: dict[str, Any]
    if action_name == "request_script":
        request = action.request_script.model_dump()
        code = str(request.get("python_code") or "")
        if len(code) > max(1, int(script_chars)):
            ref = artifact_store.put(
                "qdi_script_source_full",
                f"{question_id}:action:{sequence}",
                request,
                visible_excerpt=code[: max(1, int(script_chars))],
                visible_limit=script_chars,
            )
            request["python_code"] = code[: max(1, int(script_chars))]
            request["python_code_truncated"] = True
            request["python_code_original_chars"] = len(code)
            request["python_code_visible_chars"] = len(request["python_code"])
            request["source_artifact"] = ref
    elif action_name == "request_context":
        request = action.request_context.model_dump()
    elif action_name == "search_document":
        request = action.search_document.model_dump()
    elif action_name == "read_document_chunks":
        request = action.read_document_chunks.model_dump()
    elif action_name == "read_qdi_artifact_excerpt":
        request = action.read_qdi_artifact_excerpt.model_dump()
    elif action_name == "add_followup_questions":
        request = {"followup_questions": [item.model_dump() for item in (action.followup_questions or [])]}
    else:
        request = {
            "answer": str(action.answer or "")[:2000],
            "unresolved_reason": str(action.unresolved_reason or "")[:2000],
            "refined_question": str(action.refined_question or "")[:1000],
            "duplicate_of_question_id": str(action.duplicate_of_question_id or ""),
            "notes": str(action.notes or "")[:1000],
        }
    return {
        "sequence": int(sequence),
        "question_id": str(question_id),
        "action": str(action_name),
        "request": request,
        "observation": {"status": "pending_local_execution"},
    }


def _artifact_ids_in(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        artifact_id = str(value.get("artifact_id") or "").strip()
        if artifact_id:
            found.add(artifact_id)
        for nested in value.values():
            found.update(_artifact_ids_in(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_artifact_ids_in(nested))
    return found


def _shared_columns_for_files(files: list[str], summary_by_path: dict[str, Any]) -> list[str]:
    sets = []
    for path in files:
        fs = summary_by_path.get(path)
        cols = [str(x) for x in (getattr(fs, "columns", []) or [])] if fs is not None else []
        if cols:
            sets.append(set(cols))
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def _filename_group_column_profile(files: list[str], summary_by_path: dict[str, Any]) -> dict[str, Any]:
    observed: list[tuple[str, list[str]]] = []
    for path in files:
        fs = summary_by_path.get(str(path))
        cols = [str(x) for x in (getattr(fs, "columns", []) or [])] if fs is not None else []
        cols = [x for x in cols if x.strip()]
        if cols:
            observed.append((str(path), cols))
    if not observed:
        return {}

    sets = [set(cols) for _, cols in observed]
    common_set = set.intersection(*sets) if sets else set()
    union: list[str] = []
    for _, cols in observed:
        for col in cols:
            if col not in union:
                union.append(col)

    shared_fields = [col for col in union if col in common_set]
    variant_fields_by_file: list[dict[str, Any]] = []
    for path, cols in observed[:16]:
        only_fields = [col for col in cols if col not in common_set]
        if only_fields:
            variant_fields_by_file.append(
                {
                    "file": path,
                    "fields": only_fields[:24],
                    "omitted": max(0, len(only_fields) - 24),
                }
            )

    field_presence: list[dict[str, Any]] = []
    for col in union:
        if col in common_set:
            continue
        present = [path for path, cols in observed if col in set(cols)]
        field_presence.append(
            {
                "field": col,
                "present_in_count": len(present),
                "example_files": present[:3],
            }
        )

    return {
        "observed_file_count": len(observed),
        "shared_fields": shared_fields,
        "variant_fields_by_file": variant_fields_by_file,
        "field_presence": field_presence,
    }


def _compact_field_glossary(glossary: Any) -> list[dict[str, Any]]:
    if not isinstance(glossary, dict):
        return []
    priority = []
    fallback = []
    for field, info in glossary.items():
        meaning = str((info or {}).get("meaning", "") if isinstance(info, dict) else "")
        item = {
            "field": str(field),
            "meaning": meaning[:300],
            "role": _field_role(str(field), meaning),
            "files": [str(x) for x in ((info or {}).get("files", []) if isinstance(info, dict) else [])[:8]],
        }
        if item["role"] != "other":
            priority.append(item)
        else:
            fallback.append(item)
    return (priority + fallback)[:120]


def _field_role(field: str, meaning: str) -> str:
    text = f"{field} {meaning}".lower()
    if any(k in text for k in ["id", "key", "code", "编号", "编码", "订单号", "主键", "外键"]):
        return "id_or_key"
    if any(k in text for k in ["date", "time", "日期", "时间", "月份", "交付"]):
        return "time"
    if any(k in text for k in ["target", "label", "y", "目标", "标签"]):
        return "target"
    if any(k in text for k in ["cost", "price", "amount", "fee", "rate", "金额", "成本", "价格", "费用", "单价"]):
        return "cost_or_value"
    if any(k in text for k in ["constraint", "limit", "capacity", "约束", "限制", "容量", "车型"]):
        return "constraint"
    if any(k in text for k in ["submission", "output", "预测", "提交", "输出", "是否有效"]):
        return "output"
    return "other"


def _is_core_field_text(text: str) -> bool:
    return _field_role(text, "") != "other"


def _small_dict(value: Any, *, keys: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {k: value.get(k) for k in keys if k in value}


def _normalize_request(req: InvestigationToolRequest, idx: int) -> InvestigationToolRequest:
    if not req.request_id:
        req.request_id = f"req_{idx}"
    req.tool_name = str(req.tool_name or "").strip()
    if req.tool_name not in BUILTIN_TOOL_NAMES:
        req.tool_name = "custom_readonly_python" if req.custom_python.python_code else req.tool_name
    if not req.question_id and req.custom_python.question_id:
        req.question_id = req.custom_python.question_id
    return req


def _request_signature(req: InvestigationToolRequest) -> str:
    payload = {
        "tool_name": req.tool_name,
        "question_id": req.question_id,
        "params": req.params,
        "code": req.custom_python.python_code[:200] if req.tool_name == "custom_readonly_python" else "",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _failed_result(request: InvestigationToolRequest, error: str, *, result: dict[str, Any] | None = None) -> InvestigationStepResult:
    return InvestigationStepResult(
        request_id=request.request_id,
        question_id=request.question_id,
        tool_name=request.tool_name,
        status="failed",
        reason=request.reason,
        result=result or {},
        error=str(error),
    )


def _write_report(path: Path, data: dict[str, Any]) -> None:
    write_json_safe(path, data, indent=2)


def _compact_for_prompt(value: dict[str, Any], cfg: AutoRealizeConfig) -> dict[str, Any]:
    return _truncate_result(value, int(getattr(cfg.investigation, "max_result_chars", 20000)))


def _truncate_result(value: Any, max_chars: int) -> Any:
    safe = _json_safe(value)
    text = json.dumps(safe, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return safe
    return {
        "_truncated": True,
        "chars": len(text),
        "preview_json": text[:max_chars],
    }


def _json_safe(value: Any) -> Any:
    try:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(r) for r in df.to_dict(orient="records")]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _keys(params: dict[str, Any], list_key: str, single_key: str) -> list[str]:
    raw = params.get(list_key)
    if raw is None:
        raw = params.get(single_key)
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return [str(x) for x in _as_list(raw) if str(x).strip()]


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")


def _composite_key(df: pd.DataFrame, keys: list[str]) -> tuple[pd.Series, pd.Series]:
    _ensure_columns(df, keys)
    key_df = df[keys]
    missing = key_df.isna().any(axis=1)
    text = key_df.astype(str).agg("\u241f".join, axis=1)
    text[missing] = pd.NA
    return text, missing


def _series_stats(series: pd.Series) -> dict[str, Any]:
    if series.empty:
        return {}
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "min": _json_safe(numeric.min()),
        "p25": _json_safe(numeric.quantile(0.25)),
        "median": _json_safe(numeric.median()),
        "mean": _json_safe(numeric.mean()),
        "p75": _json_safe(numeric.quantile(0.75)),
        "max": _json_safe(numeric.max()),
    }


def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:65536]
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _call_attr(func: ast.AST) -> str:
    return func.attr if isinstance(func, ast.Attribute) else ""


def _literal_path_issue(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\") or text.startswith("/"):
        return f"absolute_path_literal_not_allowed:{text[:80]}"
    parts = re.split(r"[\\/]+", text)
    if ".." in parts:
        return f"path_traversal_literal_not_allowed:{text[:80]}"
    return ""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
