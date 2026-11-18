from __future__ import annotations

import ast
import hashlib
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
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
from .prompt_cache import dynamic_json_block, json_block, join_blocks, stable_dynamic_prompt
from .qdi_efficiency import (
    QDIEvidenceCache, QDINodeResultCache, compile_checks, compact_live_window, extractive_answer,
    is_compiled_check, question_context, repeated_action_count, request_signature, scoped_timeline,
)
from .profiling.csv_utils import infer_csv_dialect, read_csv_auto
from .profiling.full_scan import (
    FULL_TABLE_SCAN_VERSION,
    csv_scan_read_options_from_metadata,
    csv_scan_read_options_fingerprint,
    verified_full_scan_population,
)
from .profiling.stats import read_table
from .utils.filesystem import rel
from .utils.safe_json import write_json_safe

logger = logging.getLogger(__name__)

QDI_CACHE_LAYOUT = "project-evidence-question-route-dynamic-v10"
QDI_PROJECT_CONTEXT_LAYOUT = "autorealize.qdi_project_context.v10"
QDI_RETRIEVAL_ACTIONS = {
    "request_context",
    "search_document",
    "read_document_chunks",
    "read_qdi_artifact_excerpt",
}

# The local Pydantic model remains the authority for validation.  This compact
# field guide replaces the much larger generated JSON Schema in the Action
# prompt; it is intentionally versioned with the QDI cache layout.
QDI_ACTION_SCHEMA_PROMPT = r"""
Compact QDI Action JSON protocol. Output one object with required string
`action`; choose only:
answer | give_up | request_checks | request_script | request_context |
search_document | read_document_chunks | read_qdi_artifact_excerpt |
refine_current_question | mark_duplicate | add_followup_questions.

Common optional fields: question_id, answer, confidence, used_files[],
evidence[], downstream_notes[], remaining_uncertainty, unresolved_reason,
what_was_tried[], duplicate_of_question_id, refined_question, notes.

Nested request shapes:
- request_checks: {"request_checks":[{"kind":"table_profile|key_integrity|join_coverage|time_range","left":{"file":"...","sheet":0,"header":0,"encoding":"utf-8-sig","delimiter":","},"right":{...},"columns":["..."],"right_columns":["..."],"row_limit":50000}]}
- request_script: {"request_script":{"question_id":"...","goal":"...","reason_builtins_insufficient":"...","input_files":[],"focus_sheets":[],"focus_columns":[],"expected_output":"...","python_code":"def analyze(input_dir: str, scratch_dir: str) -> dict: ..."}}
- request_context: {"request_context":{"table_ids":[],"input_files":[],"focus_sheets":[],"focus_columns":[],"query":"...","reason":"..."}}
- search_document: {"search_document":{"query":"...","document_ids":[],"source_files":[],"top_k":5,"reason":"..."}}
- read_document_chunks: {"read_document_chunks":{"chunk_ids":[],"neighbor_count":1,"reason":"..."}}
- read_qdi_artifact_excerpt: {"read_qdi_artifact_excerpt":{"artifact_id":"...","offset":0,"max_chars":8000,"json_path":"","reason":"..."}}
- working_memory_update: {"confirmed_facts":[],"temporary_conclusions":[],"evidence_refs":[],"open_gaps":[],"invalidated_hypotheses":[],"recommended_next_focus":"","resolutions":[{"field":"open_gaps|confirmed_facts|temporary_conclusions","text":"...","evidence_refs":[],"reason":"..."}]}
- action_digest_updates: [{"sequence":0,"action":"...","what_was_done":"...","key_outputs":[],"temporary_conclusion":"","remaining_gap":"","evidence_refs":[]}]
- followup_questions: [{"question":"...","reason":"...","why_not_duplicate":"...","candidate_files":[]}]

For `give_up`, `unresolved_reason` must be non-empty. Use only fields needed
for the selected action; omitted fields are empty defaults.
""".strip()


def reconcile_question_memory(
    memory: dict,
    *,
    verified_full_scans: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Keep answered questions with explicit residual uncertainty open in handoffs."""
    result = deepcopy(memory)
    scan_reconciliation_changed = _reconcile_completed_full_scan_gaps(
        result,
        verified_full_scans or {},
    )
    latest = {answer.get("question_id"): answer for answer in result.get("answers", [])
              if isinstance(answer, dict) and answer.get("question_id")}
    residual = {qid: str(answer.get("remaining_uncertainty") or "").strip()
                for qid, answer in latest.items() if str(answer.get("remaining_uncertainty") or "").strip()}
    if not residual:
        if scan_reconciliation_changed:
            if result.get("summary"):
                result["summary"] = (
                    f"已记录 {len(latest)} 个问题的回答，其中 0 个仍有明确不确定性；"
                    f"保留 {len(result.get('unresolved_questions') or [])} 项未解决事项。"
                    "回答原文与取证历史保持不变。"
                )
            progress = result.get("progress")
            if isinstance(progress, dict):
                unresolved_records = {
                    record.get("question_id")
                    for record in result.get("question_records", [])
                    if isinstance(record, dict)
                    and record.get("status") in {"unresolved", "answered_with_uncertainty"}
                }
                progress["answered_questions"] = len(latest)
                progress["resolved_questions"] = len(latest)
                progress["unresolved_questions"] = len(unresolved_records)
        return result
    unresolved = list(result.get("unresolved_questions") or [])
    for qid, uncertainty in residual.items():
        item = f"[{qid}] Remaining uncertainty: {uncertainty}"
        if item not in unresolved:
            unresolved.append(item)
    for record in result.get("question_records", []) or []:
        if isinstance(record, dict) and record.get("question_id") in residual:
            record["status"] = "answered_with_uncertainty"
            record["unresolved_reason"] = residual[record["question_id"]]
    if result.get("summary") and "summary_before_uncertainty_reconciliation" not in result:
        result["summary_before_uncertainty_reconciliation"] = result["summary"]
    result["summary"] = (
        f"已记录 {len(latest)} 个问题的回答，其中 {len(residual)} 个仍有明确不确定性；"
        f"保留 {len(unresolved)} 项未解决事项。回答原文与取证历史保持不变。"
    )
    result["unresolved_questions"] = unresolved
    progress = result.get("progress")
    if isinstance(progress, dict):
        progress["answered_questions"] = len(latest)
        progress["resolved_questions"] = len(latest) - len(residual)
        progress["unresolved_questions"] = len({
            record.get("question_id") for record in result.get("question_records", [])
            if isinstance(record, dict) and record.get("status") in {"unresolved", "answered_with_uncertainty"}
        } | set(residual))
    return result


def _host_verified_full_scan_files(file_summaries: list[Any]) -> dict[str, dict[str, Any]]:
    """Return only current, host-generated full-scan evidence bound to its source."""
    verified: dict[str, dict[str, Any]] = {}
    for summary in file_summaries:
        path = str(getattr(summary, "path", "") or "").replace("\\", "/").removeprefix("./")
        metadata = getattr(summary, "source_metadata", None)
        if not path or not isinstance(metadata, dict):
            continue
        scan = metadata.get("full_table_scan")
        fingerprint = str(metadata.get("source_fingerprint") or "")
        if (
            not isinstance(scan, dict)
            or str(scan.get("file") or "").replace("\\", "/").removeprefix("./") != path
            or scan.get("scanner_version") != FULL_TABLE_SCAN_VERSION
            or not fingerprint
            or scan.get("source_fingerprint") != fingerprint
        ):
            continue
        read_options_fingerprint = csv_scan_read_options_fingerprint(
            csv_scan_read_options_from_metadata(metadata)
        )
        if scan.get("read_options_fingerprint") != read_options_fingerprint:
            continue
        if not verified_full_scan_population(
            scan,
            source_fingerprint=fingerprint,
            read_options_fingerprint=read_options_fingerprint,
        ):
            continue
        verified[path] = {
            "source_fingerprint": fingerprint,
            "read_options_fingerprint": read_options_fingerprint,
            "scanner_version": FULL_TABLE_SCAN_VERSION,
            "column": str(scan.get("column") or ""),
            "rows_scanned": scan["rows_scanned"],
            "valid_timestamp_rows": scan["valid_timestamp_rows"],
            "invalid_timestamp_rows": scan["invalid_timestamp_rows"],
            "minimum_timestamp": scan.get("minimum_timestamp"),
            "maximum_timestamp": scan.get("maximum_timestamp"),
        }
    return verified


def _remove_verified_full_scan_claim(
    text: Any,
    scan_files: dict[str, dict[str, Any]],
    referenced_sources: Any = None,
) -> tuple[str, list[dict[str, Any]]]:
    value = str(text or "").strip()
    if not value or not scan_files:
        return value, []
    normalized_references = {
        str(item).replace("\\", "/").strip().removeprefix("./")
        for item in referenced_sources
        if str(item).strip()
    } if isinstance(referenced_sources, (list, tuple, set)) else set()
    referenced_matches = normalized_references & scan_files.keys()
    file_reference_pattern = re.compile(
        r"`([^`\r\n]+\.(?:csv|xlsx?|xls|jsonl?|parquet|tsv|txt|md|docx?|pdf))`|"
        r"([^\s`'\"，,；;。()（）]+\.(?:csv|xlsx?|xls|jsonl?|parquet|tsv|txt|md|docx?|pdf))(?=$|[^\w.-])",
        flags=re.I,
    )
    prefix_match = re.match(r"^(\[[^\]]+\]\s*Remaining uncertainty:\s*)", value)
    prefix = prefix_match.group(1) if prefix_match else ""
    body = value[len(prefix):] if prefix else value
    clauses = re.split(r"(?<=[；;。])\s*", body)
    kept: list[str] = []
    resolved: list[dict[str, Any]] = []
    for clause in clauses:
        clause_text = clause.strip()
        explicit_matches = [
            source for source in scan_files
            if source in clause_text or Path(source).name in clause_text
        ]
        file_mentions = [
            (match.group(1) or match.group(2) or "").replace("\\", "/").strip()
            for match in file_reference_pattern.finditer(clause_text)
        ]
        unknown_file_mention = any(
            not any(
                source.casefold() in mention.casefold()
                or Path(source).name.casefold() in mention.casefold()
                for source in scan_files
            )
            for mention in file_mentions
        )
        if unknown_file_mention:
            matched_sources = set()
        elif explicit_matches:
            matched_sources = set(explicit_matches)
        elif file_mentions:
            matched_sources = set()
        else:
            matched_sources = referenced_matches if len(referenced_matches) == 1 else set()
        stale_scan_claim = (
            len(matched_sources) == 1
            and any(marker in clause_text for marker in ("全表", "完整扫描", "全量扫描"))
            and any(marker in clause_text for marker in ("未实测", "尚未实测", "尚未完成", "未完成", "仅前", "非全表"))
            and any(marker in clause_text for marker in ("时间范围", "总行数", "记录数", "时间列"))
        )
        if stale_scan_claim:
            matched_file = next(iter(matched_sources))
            resolved.append({
                "claim": clause_text,
                "source_file": matched_file,
                "evidence": "completed_full_scan_with_row_count_and_time_range",
            })
        elif clause_text:
            kept.append(clause_text)

    remaining = "；".join(kept).strip()
    remaining = re.sub(r"^[；;。]\s*|[；;。]\s*$", "", remaining).strip()
    return f"{prefix}{remaining}" if remaining and prefix else remaining, resolved


def _reconcile_completed_full_scan_gaps(
    memory: dict[str, Any],
    scan_files: dict[str, dict[str, Any]],
) -> bool:
    scans = scan_files
    if not scans:
        return False
    changed = False
    audit = list(memory.get("evidence_reconciliations") or [])
    known = {
        (str(item.get("claim") or ""), str(item.get("source_file") or ""))
        for item in audit if isinstance(item, dict)
    }

    def reconcile_text(text: Any, referenced_sources: Any = None) -> str:
        nonlocal changed
        remaining, resolved = _remove_verified_full_scan_claim(text, scans, referenced_sources)
        changed = changed or remaining != str(text or "").strip()
        for item in resolved:
            key = (item["claim"], item["source_file"])
            if key not in known:
                audit.append({**item, **scans[item["source_file"]]})
                known.add(key)
        return remaining

    for answer in memory.get("answers", []) or []:
        if not isinstance(answer, dict):
            continue
        current = answer.get("remaining_uncertainty")
        remaining = reconcile_text(current, answer.get("used_files"))
        if remaining != str(current or "").strip():
            answer.setdefault("historical_remaining_uncertainty", str(current or ""))
            answer["remaining_uncertainty"] = remaining
    for record in memory.get("question_records", []) or []:
        if not isinstance(record, dict):
            continue
        current = record.get("unresolved_reason")
        remaining = reconcile_text(current, record.get("used_files"))
        if remaining != str(current or "").strip():
            record.setdefault("historical_unresolved_reason", str(current or ""))
            record["unresolved_reason"] = remaining
            if not remaining and record.get("status") == "answered_with_uncertainty":
                record["status"] = "resolved"
    answer_sources = {
        str(answer.get("question_id") or ""): answer.get("used_files")
        for answer in memory.get("answers", []) or []
        if isinstance(answer, dict)
    }
    record_sources = {
        str(record.get("question_id") or ""): record.get("used_files")
        for record in memory.get("question_records", []) or []
        if isinstance(record, dict)
    }
    reconciled_questions = []
    for question in memory.get("unresolved_questions", []) or []:
        match = re.match(r"^\[([^\]]+)\]", str(question))
        question_id = match.group(1) if match else ""
        remaining = reconcile_text(
            question,
            record_sources.get(question_id) or answer_sources.get(question_id),
        )
        if remaining:
            reconciled_questions.append(remaining)
    new_questions = list(dict.fromkeys(reconciled_questions))
    changed = changed or new_questions != list(memory.get("unresolved_questions") or [])
    memory["unresolved_questions"] = new_questions
    if audit:
        memory["evidence_reconciliations"] = audit
    return changed


def _prior_qdi_answers_for_prompt(
    memory: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    raw_answers = memory.get("answers") if isinstance(memory.get("answers"), list) else []
    bounded_limit = max(0, int(limit))
    selected = raw_answers[-bounded_limit:] if bounded_limit else []
    result = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        changed_files = item.get("changed_source_files")
        if not isinstance(changed_files, list):
            changed_files = []
        result.append({
            "question_id": str(item.get("question_id") or ""),
            "question": str(item.get("question") or "")[:900],
            "answer": str(item.get("answer") or "")[:1400],
            "evidence": [str(value)[:300] for value in evidence[:6]],
            "confidence": str(item.get("confidence") or "medium"),
            "remaining_uncertainty": str(item.get("remaining_uncertainty") or "")[:500],
            "source_status": str(item.get("source_status") or "unknown"),
            "source_inventory_status": str(item.get("source_inventory_status") or "unknown"),
            "qdi_contract_status": str(item.get("qdi_contract_status") or "unknown"),
            "changed_source_files": [str(value)[:180] for value in changed_files[:12]],
            "task_scope_status": str(item.get("task_scope_status") or "unknown"),
            "source_task_hint": (
                str(item.get("source_task_hint") or "")[:240]
                if item.get("task_scope_status") != "same"
                else ""
            ),
        })
    return result


BUILTIN_TOOL_NAMES = {"custom_readonly_python"}
QDI_EVIDENCE_ACTIONS = {
    "request_context",
    "search_document",
    "read_document_chunks",
    "read_qdi_artifact_excerpt",
    "request_script",
    "request_checks",
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
    prior_question_memory: dict | None = None,
    shared_cache_dir: Path | None = None,
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
            prior_question_memory=prior_question_memory or {},
            shared_cache_dir=shared_cache_dir,
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
    prior_question_memory: dict | None = None,
    shared_cache_dir: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    log_event(logger, "module.data_cognition.investigator", "ACTIVATED")
    planner_prompt = prompt_mgr.load("system/question_investigator_planner.md")
    answerer_prompt = prompt_mgr.load("system/question_investigator_answerer_v7.md")
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
    previous = prior_question_memory if isinstance(prior_question_memory, dict) else {}
    try:
        prior_answer_limit = max(0, int(getattr(cfg.context, "project_prior_answer_limit", 8)))
    except (TypeError, ValueError):
        prior_answer_limit = 8
    previous_answers = _prior_qdi_answers_for_prompt(previous, prior_answer_limit)
    raw_source_delta = previous.get("source_delta")
    if not isinstance(raw_source_delta, dict):
        raw_source_delta = {}
    context["prior_qdi_memory"] = {
        "available": bool(previous_answers),
        "answers": previous_answers,
        "source_delta": {
            "unchanged_files": [str(x)[:180] for x in (raw_source_delta.get("unchanged_files") or [])[:80]],
            "new_files": [str(x)[:180] for x in (raw_source_delta.get("new_files") or [])[:80]],
            "changed_files": [str(x)[:180] for x in (raw_source_delta.get("changed_files") or [])[:80]],
            "removed_files": [str(x)[:180] for x in (raw_source_delta.get("removed_files") or [])[:80]],
        },
        "reuse_policy": (
            "Prior answers are hypotheses, not current task facts. Unchanged source files establish only that "
            "the cited inputs are unchanged. When task_scope_status is different or unknown, do not inherit "
            "task-specific conclusions; recheck any conclusion needed for this task against its current contract "
            "and evidence. Answers with source_status=changed or unknown must also be rechecked. If "
            "source_inventory_status=changed, compare the question with the new/changed/removed file delta; a "
            "whole-project conclusion may be stale even when its cited files are unchanged. Reuse only when the "
            "delta is clearly irrelevant to that question; if relevance is uncertain, recheck it. Answers with "
            "qdi_contract_status=changed or unknown were produced under a different or unrecorded QDI protocol "
            "and cannot suppress a current question until their evidence is rechecked. The current question "
            "still runs through QDI."
        ),
    }
    context["context_policy"]["full_document_text_local"] = True
    context["context_policy"]["document_retrieval_actions"] = ["search_document", "read_document_chunks"]
    verified_full_scans = _host_verified_full_scan_files(file_summaries)
    # Keep the stable project facts in a deterministic artifact.  Run-specific
    # task wording and prior-question deltas stay in the dynamic request; if
    # they were embedded here, a new question would invalidate the provider
    # prefix and destroy cross-run cache reuse.
    full_project_context_ref = artifact_store.put(
        "qdi_project_context_full",
        "project",
        context,
        visible_excerpt=(
            "完整 QDI 项目上下文；稳定前缀只展示可复用项目卡。需要未展示的精确字段时按 json_path 分段取回。"
        ),
        visible_limit=320,
    )
    context["_qdi_full_project_context_artifact"] = _artifact_ref_index_for_prompt(
        full_project_context_ref
    )
    tools = CrossFileInvestigationTools(
        cfg=cfg,
        data_root=data_root,
        authoritative_memory=authoritative_memory,
        knowledge_base=knowledge_base,
        artifact_store=artifact_store,
    )
    evidence_cache = QDIEvidenceCache(data_root, shared_cache_dir=shared_cache_dir)
    node_cache = QDINodeResultCache(shared_cache_dir=shared_cache_dir)
    cost_controls = {"planner_calls": 0, "action_calls": 0, "direct_answers": 0,
                     "repetition_terminations": 0, "evidence_cache_hits": 0,
                     "evidence_cache_local_hits": 0, "evidence_cache_shared_hits": 0,
                     "questions_skipped_by_authoritative_hint": 0,
                     "action_static_chars_before_projection": 0, "action_static_chars_sent": 0,
                     "action_project_context_chars_sent": 0,
                     "action_question_context_chars_sent": 0,
                     "action_dynamic_chars_sent": 0,
                     "planner_project_context_chars_sent": 0,
                     "planner_question_context_chars_sent": 0,
                     "qdi_cache_layout": QDI_CACHE_LAYOUT}

    all_questions: dict[str, InvestigationQuestion] = {}
    question_records: dict[str, dict[str, Any]] = {}
    queue: list[str] = []
    all_requests: list[InvestigationToolRequest] = []
    # Counts requested script/check actions, including a rejected check batch.
    # all_requests remains the list of actually compiled/executed requests.
    script_actions_total = 0
    all_results: list[InvestigationStepResult] = []
    answers: list[InvestigationAnswer] = []
    unresolved_questions: list[str] = []
    routing_notes: list[str] = []
    action_history: list[dict[str, Any]] = []
    action_digest_cards: list[dict[str, Any]] = []
    action_timeline: list[dict[str, Any]] = []
    working_memory_cards: dict[str, dict[str, Any]] = {}
    qdi_artifact_ids: set[str] = {
        str(full_project_context_ref.get("artifact_id") or "")
    } - {""}

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
        cost_controls["evidence_cache_hits"] = evidence_cache.hits
        cost_controls["evidence_cache_local_hits"] = evidence_cache.local_hits
        cost_controls["evidence_cache_shared_hits"] = evidence_cache.shared_hits
        cost_controls.update(node_cache.stats())
        cost_controls["cache_metrics"] = _qdi_cache_metrics(
            llm_client=llm_client,
            node_cache=node_cache,
            evidence_cache=evidence_cache,
        )
        live_report = QuestionInvestigationReport(
            enabled=True,
            summary=summary,
            questions=list(all_questions.values()),
            script_requests=[req.custom_python for req in all_requests if req.custom_python.python_code],
            tool_requests=all_requests,
            step_results=all_results,
            answers=answers,
            unresolved_questions=list(dict.fromkeys(unresolved_questions)),
            context_routing_notes=list(dict.fromkeys(routing_notes)),
            question_records=list(question_records.values()),
            action_history=action_history,
            action_digest_cards=action_digest_cards,
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
                "cost_controls": dict(cost_controls),
            },
        )
        _write_report(
            live_report_path,
            reconcile_question_memory(
                live_report.model_dump(),
                verified_full_scans=verified_full_scans,
            ),
        )

    persist_live_report(
        phase="planning",
        summary="QDI 正在根据数据认知结果规划需要进一步核实的问题。",
    )

    planner_project_context = _qdi_project_cache_context(context, cfg)
    planner_question_context = _qdi_question_cache_context(context, {})
    planner_segments, planner_stable_context = _qdi_stable_context_segments(
        planner_project_context,
        planner_question_context,
    )
    stable = "\n\n".join(segment["content"] for segment in planner_segments)
    planner_dynamic_payload = {
            "task_hint": _qdi_bounded_text(context.get("task_hint"), 2400),
            "instruction": (
                "生成本次问题驱动研究的初始问题队列。只提出真正阻塞任务定义、数据读取、文件关联、输出、评估或硬约束落地的问题；"
                "不要在 planner 阶段请求脚本。把 files 中的 CSV、表格型 JSON、Excel sheet 都按 table/file card 理解；"
                "file_cognition 只是短导航，权威事实看 authoritative_memory，硬约束和业务规则看 constraint_memory。"
            ),
            "max_questions": max_total_questions,
            "previous_question_records": [
                {
                    "question_id": str(item.get("question_id") or ""),
                    "question": str(item.get("question") or "")[:800],
                    "status": "historical_answer",
                    "short_answer": str(item.get("answer") or "")[:1000],
                    "confidence": str(item.get("confidence") or "medium"),
                    "source_status": str(item.get("source_status") or "unknown"),
                    "source_inventory_status": str(item.get("source_inventory_status") or "unknown"),
                    "qdi_contract_status": str(item.get("qdi_contract_status") or "unknown"),
                    "task_scope_status": str(item.get("task_scope_status") or "unknown"),
                    "source_task_hint": str(item.get("source_task_hint") or "")[:240],
                    "changed_source_files": [str(x)[:180] for x in (item.get("changed_source_files") or [])[:8]],
                }
                for item in previous_answers[:30]
            ],
            "prior_source_delta": context["prior_qdi_memory"]["source_delta"],
            "prior_qdi_reuse_policy": context["prior_qdi_memory"]["reuse_policy"],
            "full_project_context_artifact": context.get("_qdi_full_project_context_artifact", {}),
    }
    dynamic = dynamic_json_block(
        "Dynamic initial QDI planning request",
        planner_dynamic_payload,
        planner_stable_context,
    )
    cost_controls["planner_project_context_chars_sent"] = len(planner_segments[0]["content"])
    cost_controls["planner_question_context_chars_sent"] = len(planner_segments[1]["content"])
    cost_controls["project_context_projection"] = {
        "full_context_chars": len(json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)),
        "stable_project_prefix_chars": len(planner_segments[0]["content"]),
        "stable_route_segment_chars": len(planner_segments[1]["content"]),
        "project_prefix_reduction_ratio": round(
            1
            - len(planner_segments[0]["content"])
            / max(1, len(json.dumps(context, ensure_ascii=False, sort_keys=True, default=str))),
            6,
        ),
        "table_index_count": len(planner_project_context.get("table_index") or []),
        "table_card_count": len(planner_project_context.get("table_cards") or []),
        "coverage_attention_count": len(
            (planner_project_context.get("source_coverage_ledger") or {}).get("entries") or []
        ),
        "full_project_context_artifact": planner_project_context.get(
            "full_project_context_artifact", {}
        ),
    }
    cost_controls["planner_calls"] += 1
    plan = _qdi_structured_call(
        node_cache=node_cache,
        cost_controls=cost_controls,
        llm_client=llm_client,
        model_cls=QuestionInvestigationPlan,
        system_prompt=planner_prompt,
        user_prompt=dynamic,
        prompt_name="question_investigator_initial_questions",
        # Keep the compatibility field limited to the reusable project prefix.
        # The complete planner routing context is carried by the explicit
        # stable_context_segments argument below.
        static_context_prompt=planner_segments[0]["content"],
        dynamic_user_prompt=dynamic,
        stable_context_segments=planner_segments,
        max_tokens=max(
            1024,
            int(getattr(cfg.investigation, "planner_max_output_tokens", 8192)),
        ),
    )
    if bool(getattr(cfg.investigation, "adaptive_budgeting", True)):
        if int(plan.recommended_max_questions or 0) > 0:
            max_total_questions = min(max_total_questions, max(1, int(plan.recommended_max_questions)))
        if int(plan.recommended_max_actions_per_question or 0) > 0:
            # A tool action needs a subsequent terminal round to interpret its evidence.
            max_actions_per_question = min(
                max_actions_per_question,
                max(2, int(plan.recommended_max_actions_per_question)),
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
        fixed_reason = _question_fixed_by_task_hint(q, task_hint)
        if fixed_reason:
            cost_controls["questions_skipped_by_authoritative_hint"] += 1
            routing_notes.append(
                f"[{q.question_id or 'auto'}] skipped: task_hint already fixes this rule ({fixed_reason})."
            )
            continue
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
        fixed_reason = _question_fixed_by_task_hint(q, task_hint)
        if fixed_reason:
            cost_controls["questions_skipped_by_authoritative_hint"] += 1
            routing_notes.append(
                f"[{q.question_id or 'planner'}] skipped: task_hint already fixes this rule ({fixed_reason})."
            )
            continue
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

    # Extractive answers never override deterministic verification questions.
    for proposal in plan.direct_answers:
        qid = _add_question_record(
            question=proposal.question, all_questions=all_questions,
            question_records=question_records, queue=queue, parent_question_id="",
            depth=0, max_total_questions=max_total_questions,
        )
        if not qid:
            qid = _find_duplicate_question(question_records, proposal.question.question)
        if not qid or qid.startswith("auto_"):
            continue
        quote = extractive_answer(context, proposal)
        if quote is None:
            routing_notes.append(f"[{qid}] Planner citation failed exact-source validation; normal investigation retained.")
            continue
        if question_records[qid]["status"] == "resolved":
            continue
        evidence = ["stable_context:" + proposal.source_pointer]
        answers.append(InvestigationAnswer(
            question_id=qid, question=question_records[qid]["question"], answer=quote,
            evidence=evidence, confidence="medium",
            downstream_notes=["Exact authority quote selected by planner; no new empirical verification was performed."],
        ))
        question_records[qid].update(status="resolved", short_answer=quote, confidence="medium")
        entry = {"sequence": len(action_history) + 1, "question_id": qid,
                 "action": "planner_direct_answer", "status": "answered",
                 "answer": quote, "evidence": evidence}
        action_history.append(entry)
        action_timeline.append(entry)
        cost_controls["direct_answers"] += 1

    completed_question_artifacts: dict[str, dict] = {}
    full_context_chars = len(stable_dynamic_prompt(stable=context)[0])
    answerer_project_context = _qdi_project_cache_context(context, cfg)
    answerer_project_segments, _ = _qdi_stable_context_segments(answerer_project_context, None)
    answerer_stable_prefix = answerer_project_segments[0]["content"]
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
        retrieval_actions_since_terminal = 0
        investigation_actions_used = 0
        question_action_ledger: list[dict[str, Any]] = []
        question_live_actions: list[dict[str, Any]] = []
        working_memory = _empty_working_memory_card(qid)
        working_memory_cards[qid] = working_memory
        initial_question = _immutable_question_card(record)
        answerer_context = question_context(context, initial_question)
        question_stable_context = _qdi_question_cache_context(context, initial_question, answerer_context)
        qdi_stable_segments, qdi_stable_context = _qdi_stable_context_segments(
            answerer_project_context,
            question_stable_context,
        )
        answerer_stable_prefix = qdi_stable_segments[0]["content"]
        question_stable_prefix = qdi_stable_segments[1]["content"]

        max_context_retrievals = max(
            0, int(getattr(cfg.investigation, "context_retrievals_per_question", 2))
        )
        max_document_retrievals = max(
            0, int(getattr(cfg.investigation, "document_retrievals_per_question", 4))
        )
        max_artifact_retrievals = max(
            0, int(getattr(cfg.investigation, "artifact_retrievals_per_question", 2))
        )
        max_retrieval_actions_before_terminal = max(
            1, int(getattr(cfg.investigation, "max_retrieval_actions_before_terminal", 1))
        )
        max_decision_calls = max_actions_per_question + max_context_retrievals + max_document_retrievals + max_artifact_retrievals
        for decision_call_index in range(1, max_decision_calls + 1):
            action_round = investigation_actions_used + 1
            remaining_investigation_actions = max(0, max_actions_per_question - investigation_actions_used)
            available_actions = _available_qdi_actions(
                question_records=question_records,
                current_record=record,
                scripts_for_question=scripts_for_question,
                context_retrievals_for_question=context_retrievals_for_question,
                document_retrievals_for_question=document_retrievals_for_question,
                has_documents=bool(document_index.documents),
                max_context_retrievals=max_context_retrievals,
                max_document_retrievals=max_document_retrievals,
                has_qdi_artifacts=bool(qdi_artifact_ids),
                artifact_retrievals_for_question=artifact_retrievals_for_question,
                max_artifact_retrievals=max_artifact_retrievals,
                total_scripts=script_actions_total,
                max_scripts_per_question=max_scripts_per_question,
                max_scripts_total=max_scripts_total,
                max_total_questions=max_total_questions,
                max_depth=max_depth,
                max_followups_per_question=max_followups_per_question,
                allow_custom_readonly_python=bool(getattr(cfg.investigation, "allow_custom_readonly_python", True)),
            )
            retrieval_continuation_actions = _qdi_retrieval_continuation_actions(
                question_live_actions,
                document_retrievals_for_question=document_retrievals_for_question,
                max_document_retrievals=max_document_retrievals,
                artifact_retrievals_for_question=artifact_retrievals_for_question,
                max_artifact_retrievals=max_artifact_retrievals,
            )
            if (
                context_retrievals_for_question
                and max_retrieval_actions_before_terminal <= 1
            ):
                available_actions = [
                    action for action in available_actions if action != "request_context"
                ]
            available_actions = _apply_qdi_retrieval_action_policy(
                available_actions,
                retrieval_actions_since_terminal=retrieval_actions_since_terminal,
                max_retrieval_actions_before_terminal=max_retrieval_actions_before_terminal,
                continuation_actions=retrieval_continuation_actions,
            )
            if remaining_investigation_actions <= 1:
                available_actions = _terminal_qdi_actions(available_actions)
                available_actions = _apply_qdi_retrieval_action_policy(
                    available_actions,
                    retrieval_actions_since_terminal=retrieval_actions_since_terminal,
                    max_retrieval_actions_before_terminal=max_retrieval_actions_before_terminal,
                    continuation_actions=retrieval_continuation_actions,
                )
            repetition_limit = max(0, int(getattr(cfg.investigation, "max_repeated_actions", 2)))
            repetition_exhausted = bool(repetition_limit and repeated_action_count(question_live_actions) >= repetition_limit)
            if repetition_exhausted:
                available_actions = ["answer", "give_up"]
                cost_controls["repetition_terminations"] += 1
            persist_live_report(
                phase="deciding_action",
                summary=f"QDI 正在为问题 {qid} 执行第 {decision_call_index} 次动作决策（调查动作槽 {action_round}）。",
                current_question_id=qid,
                current_action="decide_next_action",
                action_round=action_round,
            )
            pending_digest_requests = _pending_action_digest_requests(
                action_history,
                action_digest_cards,
                qid,
            )
            dynamic_payload = {
                "decision_state": _qdi_decision_state(
                    initial_question=initial_question,
                    record=record,
                    question_records=question_records,
                    question_live_actions=question_live_actions,
                    action_digest_cards=action_digest_cards,
                    working_memory=working_memory,
                    current_question_id=qid,
                    available_actions=available_actions,
                    pending_action_digest_requests=pending_digest_requests,
                    investigation_actions_used=investigation_actions_used,
                    remaining_investigation_actions=remaining_investigation_actions,
                    max_actions_per_question=max_actions_per_question,
                    scripts_for_question=scripts_for_question,
                    script_actions_total=script_actions_total,
                    max_scripts_per_question=max_scripts_per_question,
                    max_scripts_total=max_scripts_total,
                    context_retrievals_for_question=context_retrievals_for_question,
                    max_context_retrievals=max_context_retrievals,
                    document_retrievals_for_question=document_retrievals_for_question,
                    max_document_retrievals=max_document_retrievals,
                    artifact_retrievals_for_question=artifact_retrievals_for_question,
                    max_artifact_retrievals=max_artifact_retrievals,
                    retrieval_actions_since_terminal=retrieval_actions_since_terminal,
                    max_retrieval_actions_before_terminal=max_retrieval_actions_before_terminal,
                    retrieval_continuation_actions=retrieval_continuation_actions,
                    decision_call_index=decision_call_index,
                    max_decision_calls=max_decision_calls,
                    max_total_questions=max_total_questions,
                    max_depth=max_depth,
                    max_output_chars=max_output_chars,
                    task_hint=_qdi_bounded_text(context.get("task_hint"), 1600),
                    full_project_context_artifact=context.get(
                        "_qdi_full_project_context_artifact", {}
                    ),
                    repetition_exhausted=repetition_exhausted,
                    cfg=cfg,
                ),
            }
            dynamic = dynamic_json_block(
                "Dynamic QDI action",
                dynamic_payload,
                qdi_stable_context,
                preserve_string_keys={"question"},
            )
            cost_controls["action_calls"] += 1
            cost_controls["action_static_chars_before_projection"] += full_context_chars
            cost_controls["action_static_chars_sent"] += len(answerer_stable_prefix) + len(question_stable_prefix)
            cost_controls["action_project_context_chars_sent"] += len(answerer_stable_prefix)
            cost_controls["action_question_context_chars_sent"] += len(question_stable_prefix)
            cost_controls["action_dynamic_chars_sent"] += len(dynamic)
            action = _qdi_structured_call(
                node_cache=node_cache,
                cost_controls=cost_controls,
                llm_client=llm_client,
                model_cls=QuestionInvestigationAction,
                    system_prompt=answerer_prompt,
                user_prompt=dynamic,
                    prompt_name="question_investigator_action",
                static_context_prompt=answerer_stable_prefix,
                dynamic_user_prompt=dynamic,
                stable_context_segments=qdi_stable_segments,
                schema_prompt=QDI_ACTION_SCHEMA_PROMPT,
                max_tokens=max(
                    1024,
                    int(getattr(cfg.investigation, "action_max_output_tokens", 6144)),
                ),
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
            action_ref = artifact_store.put(
                "qdi_action_full", f"{qid}:action:{entry['sequence']}", action.model_dump(),
            )
            entry["action_artifact_ref"] = _artifact_ref_index_for_prompt(action_ref)
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
            live_entry["action_artifact_ref"] = entry["action_artifact_ref"]
            question_live_actions.append(live_entry)
            qdi_artifact_ids.update(_artifact_ids_in(live_entry))
            working_memory = _merge_working_memory_card(
                working_memory,
                action.working_memory_update,
                sequence=entry["sequence"],
                max_chars=int(getattr(cfg.investigation, "working_memory_max_chars", 12000)),
            )
            memory_ref = artifact_store.put(
                "qdi_working_memory_full", f"{qid}:action:{entry['sequence']}", working_memory,
            )
            working_memory["_full_memory_artifact"] = _artifact_ref_index_for_prompt(memory_ref)
            qdi_artifact_ids.add(memory_ref["artifact_id"])
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

            if action_name in QDI_RETRIEVAL_ACTIONS:
                retrieval_actions_since_terminal += 1

            if action_name not in {"request_context", "search_document", "read_document_chunks", "read_qdi_artifact_excerpt"}:
                investigation_actions_used += 1

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
                    evidence=[str(x) for x in (action.evidence or [])],
                    confidence=str(action.confidence or "medium"),
                    remaining_uncertainty=str(action.remaining_uncertainty or ""),
                    downstream_notes=[str(x) for x in (action.downstream_notes or [])],
                )
                answers.append(answer)
                record.update(
                    {
                        "status": "resolved",
                        "short_answer": answer.answer,
                        "confidence": answer.confidence,
                        "used_files": [str(x) for x in (action.used_files or [])],
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
                    "unresolved_reason": str(action.unresolved_reason or action.remaining_uncertainty or ""),
                    "what_was_tried": [str(x) for x in (action.what_was_tried or [])],
                }
                reason = str(action.unresolved_reason or action.remaining_uncertainty or "Evidence is insufficient.").strip()
                record.update({"status": "unresolved", "unresolved_reason": reason})
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
                    record["question"] = refined
                    question.question = refined
                    record["status"] = "refined"
                entry["status"] = "question_refined"
                live_entry["observation"] = {"refined_question": refined}
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
                live_entry["status"] = "context_retrieved"
                live_entry["observation"] = _artifact_backed_observation(
                    retrieval_result,
                    artifact_store=artifact_store,
                    artifact_type="qdi_context_retrieval_full",
                    source=f"{qid}:action:{entry['sequence']}",
                    max_chars=int(getattr(cfg.investigation, "recent_retrieval_chars", 8000)),
                )
                # A context request exposes the original profile references as well
                # as the bounded card view, including fields absent from that view.
                live_entry["source_profile_artifacts"] = [
                    _artifact_ref_index_for_prompt(ref)
                    for card in retrieved for ref in (card.get("artifact_refs") or [])
                    if isinstance(ref, dict)
                    and ref.get("artifact_type") in {"file_profile_full", "excel_sheet_profile_full"}
                ]
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
                    excerpt_chars=500,
                )
                matches = result.get("matches", [])
                live_entry["continuation_chunk_ids"] = [
                    str(match.get("chunk_id") or "")
                    for match in matches
                    if isinstance(match, dict) and str(match.get("chunk_id") or "").strip()
                ]
                if matches:
                    # One deterministic search can also deliver exact nearby text.
                    result["source_context"] = document_index.read_chunks(
                        [matches[0]["chunk_id"]], neighbor_count=1, max_chunks=3,
                        max_chars=min(5000, max(1, int(getattr(
                            cfg.investigation, "document_retrieval_max_chars", 12000)))),
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
                live_entry["status"] = "document_searched"
                live_entry["observation"] = _artifact_backed_observation(
                    result, artifact_store=artifact_store,
                    artifact_type="qdi_document_search_full", source=f"{qid}:search:{entry['sequence']}",
                    max_chars=int(getattr(cfg.investigation, "recent_retrieval_chars", 8000)),
                )
                qdi_artifact_ids.update(_artifact_ids_in(live_entry))
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
                live_entry["status"] = "document_chunks_read"
                live_entry["observation"] = _artifact_backed_observation(
                    result,
                    artifact_store=artifact_store,
                    artifact_type="qdi_document_chunks_full",
                    source=f"{qid}:document_chunks:{entry['sequence']}",
                    max_chars=int(getattr(cfg.investigation, "recent_retrieval_chars", 8000)),
                )
                qdi_artifact_ids.update(_artifact_ids_in(live_entry))
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
                        allowed_type_prefixes=("qdi_", "file_profile_full", "excel_sheet_profile_full"),
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
                live_entry["status"] = entry["status"]
                # read_excerpt already enforces its own bound and next_offset. A second
                # preview clip would hide bytes that the cursor claims were delivered.
                live_entry["observation"] = result
                continue

            if action_name in {"request_script", "request_checks"}:
                entry["digest_policy"] = "summary_follows_as_separate_timeline_event"
                scripts_for_question += 1
                script_actions_total += 1
                if action_name == "request_checks":
                    try:
                        req = compile_checks(action.request_checks, qid)
                    except ValueError as exc:
                        entry["status"] = "invalid_check_request"
                        live_entry["observation"] = {"status": "failed", "error": str(exc)}
                        continue
                else:
                    script_req = action.request_script
                    script_req.question_id = qid
                    script_req.goal = script_req.goal or str(action.notes or record.get("question", ""))
                    req = InvestigationToolRequest(
                        question_id=qid, tool_name="custom_readonly_python",
                        reason=script_req.goal or script_req.expected_output or script_req.reason_builtins_insufficient,
                        custom_python=script_req,
                    )
                req = _normalize_request(req, len(all_requests) + 1)
                all_requests.append(req)
                attempt_results, final_req = _execute_request_with_repair(
                    req=req,
                    tools=tools,
                    llm_client=llm_client,
                    prompt_mgr=prompt_mgr,
                    cfg=cfg,
                    context=context,
                    table_card_details=table_card_details,
                    previous_questions=list(question_records.values()),
                    artifact_store=artifact_store,
                    evidence_cache=evidence_cache,
                )
                current_request = final_req
                all_results.extend(attempt_results)
                current_result = attempt_results[-1]
                if current_result.result.get("_evidence_reuse"):
                    entry["evidence_reuse"] = current_result.result["_evidence_reuse"]
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
                    error=str(current_result.error or "")[:240],
                    retryable=current_result.status != "completed",
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
            record["unresolved_reason"] = "QDI investigation-action budget exhausted before a final answer."
            unresolved_questions.append(f"[{qid}] {record['question']} unresolved: {record['unresolved_reason']}")

        record["budget_usage"] = {
            "investigation_actions_used": investigation_actions_used,
            "investigation_action_limit": max_actions_per_question,
            "model_action_decisions": decision_call_index,
            "model_action_decision_limit": max_decision_calls,
            "script_requests": scripts_for_question,
            "context_retrievals": context_retrievals_for_question,
            "document_retrievals": document_retrievals_for_question,
            "artifact_retrievals": artifact_retrievals_for_question,
            "retrieval_actions_since_terminal": retrieval_actions_since_terminal,
            "max_retrieval_actions_before_terminal": max_retrieval_actions_before_terminal,
        }

        history_ref = artifact_store.put(
            "qdi_question_history_full", qid,
            {"timeline": scoped_timeline(action_timeline, qid), "live_actions": question_live_actions,
             "working_memory": working_memory, "question": record},
        )
        history_ref_index = _artifact_ref_index_for_prompt(history_ref)
        completed_question_artifacts[qid] = history_ref_index
        record["history_artifact_ref"] = history_ref_index
        qdi_artifact_ids.add(history_ref["artifact_id"])

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
    cost_controls["evidence_cache_hits"] = evidence_cache.hits
    cost_controls["evidence_cache_local_hits"] = evidence_cache.local_hits
    cost_controls["evidence_cache_shared_hits"] = evidence_cache.shared_hits
    report = QuestionInvestigationReport(
        enabled=True,
        summary=summary,
        questions=list(all_questions.values()),
        script_requests=[req.custom_python for req in all_requests if req.custom_python.python_code],
        tool_requests=all_requests,
        step_results=all_results,
        answers=answers,
        unresolved_questions=list(dict.fromkeys(unresolved_questions)),
        context_routing_notes=list(dict.fromkeys(routing_notes)),
        question_records=list(question_records.values()),
        action_history=action_history,
        action_digest_cards=action_digest_cards,
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
            "cost_controls": dict(cost_controls),
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
    return reconcile_question_memory(
        report.model_dump(),
        verified_full_scans=verified_full_scans,
    )


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
    evidence_cache: QDIEvidenceCache | None = None,
) -> tuple[list[InvestigationStepResult], InvestigationToolRequest]:
    max_retries = max(0, int(getattr(cfg.investigation, "custom_python_max_retries", 3)))
    results: list[InvestigationStepResult] = []
    current = req
    if is_compiled_check(req):
        max_retries = 0  # Fixed code is not rewritten by an LLM; fix parameters next round.
    for attempt in range(max_retries + 1):
        if attempt > 0 and not current.request_id.endswith(f"_retry{attempt + 1}"):
            current.request_id = f"{req.request_id}_retry{attempt + 1}"
        cache_key = evidence_cache.key(current) if evidence_cache is not None else None
        result = evidence_cache.get(cache_key, current) if evidence_cache is not None else None
        deterministic_pending = False
        if result is None:
            result = tools.execute(current)
            if result.status != "completed":
                deterministic = _deterministic_script_repair(current, result)
                if deterministic is not None:
                    deterministic_pending = True
            if deterministic_pending:
                result.result["deterministic_repair"] = {
                    "rule": "os_path_join_to_pathlib",
                    "status": "applied",
                    "llm_repair_call_avoided": True,
                }
                result.result.setdefault("attempt", attempt + 1)
                results.append(result)
                current = deterministic
                continue
            # A source mutation during execution invalidates the result for reuse.
            if evidence_cache is not None and cache_key == evidence_cache.key(current):
                evidence_cache.put(cache_key, result)
        result.result.setdefault("attempt", attempt + 1)
        results.append(result)
        if result.status == "completed":
            return results, current
        if (result.error == "custom_python_result_size_limit_exceeded"
                or (isinstance(result.result, dict)
                    and result.result.get("error") == "custom_python_result_size_limit_exceeded")):
            # The host deliberately refused to persist a partial JSON value. A
            # repair model cannot recover the missing tail; the next QDI action
            # must choose a smaller aggregate or a scoped script.
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
                    "repair_budget": {
                        "used": attempt,
                        "remaining_including_this_attempt": max_retries - attempt,
                        "limit": max_retries,
                        "does_not_consume_investigation_action_or_script_request_quota": True,
                        "model_call_still_has_token_cost": True,
                    },
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


def _deterministic_script_repair(
    request: InvestigationToolRequest,
    result: InvestigationStepResult,
) -> InvestigationToolRequest | None:
    """Convert a bounded set of read-only ``os.path`` calls to pathlib."""
    if not request.custom_python or not result.error:
        return None
    if result.error != "custom_python_static_validation_failed":
        return None
    issues = result.result.get("issues") if isinstance(result.result, dict) else None
    if not isinstance(issues, list) or "banned_import:os" not in issues:
        return None
    code = str(request.custom_python.python_code or "")
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    import_nodes: list[ast.Import] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        if any(alias.name == "os" for alias in node.names):
            if any(alias.name == "os" and alias.asname for alias in node.names):
                return None
            import_nodes.append(node)
    if not import_nodes:
        return None

    replacements: list[tuple[ast.Call, str]] = []
    os_loads = [node for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id == "os" and isinstance(node.ctx, ast.Load)]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.keywords:
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
        ):
            continue
        method = func.attr
        if method == "join" and len(node.args) >= 2:
            if not isinstance(node.args[0], ast.Name) or node.args[0].id not in {"input_dir", "scratch_dir"}:
                return None
            parts = [ast.get_source_segment(code, arg) for arg in node.args]
            if any(part is None for part in parts):
                return None
            expression = f"Path({parts[0]})"
            for part in parts[1:]:
                expression += f" / ({part})"
            replacements.append((node, f"str({expression})"))
        elif method in {"exists", "isfile", "isdir", "getsize"} and len(node.args) == 1:
            argument = ast.get_source_segment(code, node.args[0])
            if argument is None:
                return None
            path_expr = f"Path({argument})"
            if method == "getsize":
                replacement = f"{path_expr}.stat().st_size"
            else:
                path_method = {"exists": "exists", "isfile": "is_file", "isdir": "is_dir"}[method]
                replacement = f"{path_expr}.{path_method}()"
            replacements.append((node, replacement))
        else:
            # Leave ambiguous path operations to model repair.
            continue

    if not replacements:
        return None

    matched_os_loads = {
        id(node.func.value.value)
        for node, _ in replacements
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
    }
    if len(os_loads) != len(matched_os_loads) or {id(node) for node in os_loads} != matched_os_loads:
        return None

    lines = code.splitlines(keepends=True)
    line_offsets: list[int] = []
    byte_cursor = 0
    for line in lines:
        line_offsets.append(byte_cursor)
        byte_cursor += len(line.encode("utf-8"))
    source_bytes = code.encode("utf-8")

    def span(node: ast.AST) -> tuple[int, int]:
        start = line_offsets[node.lineno - 1] + node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        return start, end

    edits: list[tuple[int, int, bytes]] = []
    for node, replacement in replacements:
        start, end = span(node)
        edits.append((start, end, replacement.encode("utf-8")))

    for node in import_nodes:
        start, end = span(node)
        remaining = [alias for alias in node.names if alias.name != "os"]
        if remaining:
            rendered = "import " + ", ".join(
                alias.name + (f" as {alias.asname}" if alias.asname else "")
                for alias in remaining
            )
            edits.append((start, end, rendered.encode("utf-8")))
        else:
            edits.append((start, end, b""))

    has_path_import = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pathlib"
        and any(alias.name == "Path" and (alias.asname or "Path") == "Path" for alias in node.names)
        for node in ast.walk(tree)
    )
    if not has_path_import:
        leading = []
        for node in tree.body:
            is_docstring = (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and not leading
            )
            is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
            if is_docstring or is_future:
                leading.append(node)
                continue
            break
        insert_at = span(leading[-1])[1] if leading else 0
        insertion = b"\nfrom pathlib import Path\n" if insert_at else b"from pathlib import Path\n"
        edits.append((insert_at, insert_at, insertion))

    for start, end, replacement in sorted(edits, key=lambda item: (item[0], item[1]), reverse=True):
        source_bytes = source_bytes[:start] + replacement + source_bytes[end:]
    repaired = source_bytes.decode("utf-8")
    remaining = validate_custom_readonly_python(repaired)
    if not remaining:
        copied = deepcopy(request)
        copied.custom_python.python_code = repaired
        return copied
    return None


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
    next_id = len(question_records) + 1
    while qid in question_records:
        qid = f"q{next_id}"
        next_id += 1
    question.question_id = qid
    question.parent_question_id = parent_question_id
    question.depth = int(depth)
    all_questions[qid] = question
    question_records[qid] = {
        "question_id": qid,
        "question": qtext,
        "status": "pending",
        "short_answer": "",
        "unresolved_reason": "",
        "confidence": "",
        "used_files": [],
        "parent_id": parent_question_id,
        "depth": int(depth),
        "category": str(question.category or ""),
        "priority": str(question.priority or "medium"),
        "candidate_files": [str(x) for x in (question.candidate_files or [])],
        "why_blocking": str(question.why_blocking or ""),
    }
    queue.append(qid)
    return qid


def _question_fixed_by_task_hint(
    question: InvestigationQuestion,
    task_hint: str,
) -> str:
    """Return a reason when the user has already fixed the requested rule.

    QDI is for unresolved evidence gaps. Re-asking for a decision that is
    explicitly stated in the user's task hint creates expensive, contradictory
    history and often turns a known constraint into an artificial open issue.
    The checks below are intentionally narrow and only suppress a question
    when the hint contains the corresponding positive rule.
    """

    qtext = str(getattr(question, "question", "") or "").strip().lower()
    hint = str(task_hint or "").strip().lower()
    if not qtext or not hint:
        return ""

    if (
        ("聚合" in qtext or "平均" in qtext or "重复时间" in qtext or "同一采样时间" in qtext)
        and "算术平均" in hint
        and ("同一采样时间" in hint or "重复" in hint)
    ):
        return "label aggregation is explicitly arithmetic mean"
    if (
        ("划分" in qtext or "切分" in qtext or "purge" in qtext)
        and "60%" in hint
        and "20%" in hint
        and "purge" in hint
    ):
        return "chronological split and purge are explicit"
    if (
        ("化验" in qtext or "标签" in qtext or "结果" in qtext)
        and ("历史特征" in qtext or "在线" in qtext or "可见" in qtext)
        and ("不使用无法证明发布时间" in hint or "不得使用无法证明发布时间" in hint)
    ):
        return "unverifiable assay visibility is explicitly excluded"
    return ""


def _find_duplicate_question(question_records: dict[str, dict[str, Any]], question: str) -> str:
    norm = _norm_question(question)
    for qid, record in question_records.items():
        existing = _norm_question(str(record.get("question", "")))
        if not existing:
            continue
        if norm == existing:
            return qid
        # Substring overlap may add a different population, period or constraint;
        # it cannot justify dropping that investigation.
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
        covered_pairs = {
            frozenset((str(item.get("left_collection", "")), str(item.get("right_collection", ""))))
            for item in directional_coverage if isinstance(item, dict)
        }
        missing_pairs = [
            [left, right] for i, left in enumerate(source_collections)
            for right in source_collections[i + 1:]
            if frozenset((left, right)) not in covered_pairs
        ]
        concept_id = re.sub(r"[^A-Za-z0-9_]+", "_", str(group.get("concept_id") or len(out) + 1)).strip("_")
        question_id = f"auto_entity_alias_{concept_id or len(out) + 1}"
        measurement_candidate = _entity_alias_is_measurement(group, field_refs)
        if measurement_candidate:
            question_text = (
                f"核验 `{label}` 中的候选字段是否表达同一测量定义，或只是名称相近/相关的不同测点："
                f"{'、'.join(field_refs) if field_refs else '见 entity_alias_candidates'}。"
                "先从精确物理 schema 确认每个字段是否存在，再分别确认参数/部位、工艺位置、单位、测量对象、采样频率与可见时间。"
                "对连续测量值不得用唯一值精确交集、覆盖率或 join coverage 判定实体等价；这些统计只适用于有业务依据的实体键。"
                "如字段存在且确有同一测量定义，再按已验证的时间可见性对齐后比较有效样本数、范围与误差/相关性；"
                "相关性不能证明语义等价或因果关系。不存在的字段应记录 schema 不存在，不得反复要求全量数值交集。"
                f"候选语义理由：{semantic_reason or '字段语义可能相关'}；evidence_status={evidence_status}，task_relevance={relevance}。"
                "字段存在性、测量语义、实体关联键和预测时点可见性分别给出结论；未知项保留为未确认。"
            )
            category = "field_semantics"
            why_blocking = (
                "测量值字段即使名称相似或数值相关，也可能对应不同工艺位置、单位或采样时点；"
                "错误复用会污染标签、特征或建议。"
            )
        else:
            question_text = (
                f"验证 `{label}` 是否可以作为同一实体键使用：对候选字段 "
                f"{'、'.join(field_refs) if field_refs else '见 entity_alias_candidates'} "
                f"分别统计非空数量、唯一值数量、两两唯一值交集、left/right coverage 和 join coverage。"
                f"候选来源集合为 {source_collections or candidate_files}；先在每个 source_collection 内按 alias_family "
                "读取全部成员文件并对唯一值取并集，再计算集合之间的方向性覆盖率，不能用单个文件代表整个集合。"
                f"程序已计算的集合并集方向性覆盖证据为 {directional_coverage[:8]}；先复核并解释该证据，"
                f"当前缺少覆盖证据的来源对为 {missing_pairs}。缺少来源对时必须补齐取证，不能只解释已有的一对就结束。"
                "已有统计只有在来源、行覆盖范围、读取成功状态与比较规则都满足当前问题时才可直接复用；"
                "read_errors、采样/截断、范围不明或格式不一致时，只补算缺失范围，不重复已充分验证的相同计算。"
                "有界字段画像中的时间极值不代表全量文件起止时间；字面值零交集不证明解析后的日期也无交集。"
                "时间字段需分别核验解析后精确时间及适当粒度的方向覆盖，报告分母与解析失败数；"
                "粒度聚合不能自动确立可用于预测的时间匹配规则。"
                f"LLM 提出候选的语义理由为：{semantic_reason or '字段语义和来源角色可能相关'}。"
                f"当前 evidence_status={evidence_status}，task_relevance={relevance}。"
                "不要预设这些字段等价；覆盖率不足或方向明显不对称时，记录未确认映射及其对下游 join/筛选的影响。"
                "脚本应使用 pathlib/pandas，先对实际列名做 strip 后保留原名映射，"
                "并分别输出 `left_covered_by_right_ratio` 与 `right_covered_by_left_ratio` 的明确分母。"
            )
            category = "join_key"
            why_blocking = (
                "名称不同但语义相近的字段可能表示同一实体，也可能只是相关但不等价的属性；"
                "错误合并会使跨表 join、覆盖口径、特征构造或约束判断失真。"
            )
        out.append(
            InvestigationQuestion(
                question_id=question_id,
                question=question_text,
                category=category,
                why_blocking=why_blocking,
                candidate_files=candidate_files,
                priority="high" if relevance == "high" or "code" in value_kinds else "medium",
            )
        )
    return out


def _entity_alias_is_measurement(group: dict[str, Any], field_refs: list[str]) -> bool:
    identity = " ".join(
        str(group.get(key) or "")
        for key in ("concept_id", "label", "reason", "alias_family")
    )
    identity += " " + " ".join(field_refs)
    return bool(re.search(
        r"(?i)(measurement|concentration|temperature|pressure|flow(?:_rate)?|result|"
        r"浓度|测量|测点|温度|压力|流量|结果|产量|含量)",
        identity,
    ))


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
        kind = str(card.get("table_kind", "") or "")
        source_suffix = Path(table_id.split("::", 1)[0]).suffix.lower()
        if kind == "document" or source_suffix in {".pdf", ".doc", ".docx", ".md", ".txt"}:
            continue
        # An actually empty worksheet has no reader ambiguity to resolve.
        # Asking QDI to inspect header/skiprows again only burns a model call
        # and can create a false "missing schema" issue downstream.
        verified_rows = card.get("verified_row_count")
        shape = card.get("shape")
        if (
            (isinstance(verified_rows, (int, float)) and int(verified_rows) == 0)
            or (isinstance(shape, (list, tuple)) and len(shape) >= 1 and int(shape[0] or 0) == 0)
            or str(card.get("layout_kind") or "").strip().lower() in {"empty", "empty_sheet", "empty_or_unreadable"}
        ):
            continue
        notes = [str(x) for x in (card.get("reading_notes") or [])]
        warnings = [str(x) for x in (card.get("warnings") or [])]
        risks = [str(x) for x in (card.get("reading_risks") or [])]
        evidence = notes + warnings + risks
        text = " ".join(evidence).lower()
        if not table_id or not evidence or not any(marker in text for marker in markers):
            continue
        try:
            header_confidence = float(card.get("header_confidence") or 0)
        except (TypeError, ValueError):
            header_confidence = 0.0
        reading_warning = any(marker in " ".join(warnings + risks).lower() for marker in markers)
        if (
            card.get("layout_kind") == "standard_table"
            and card.get("read_strategy_kind") == "default_header"
            and header_confidence >= 0.75
            and not risks
            and not reading_warning
        ):
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
    max_context_retrievals: int,
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
    if context_retrievals_for_question < max_context_retrievals:
        actions.insert(1, "request_context")
    if has_documents and document_retrievals_for_question < max_document_retrievals:
        actions.insert(1, "search_document")
    if has_qdi_artifacts and artifact_retrievals_for_question < max_artifact_retrievals:
        actions.insert(1, "read_qdi_artifact_excerpt")
    if allow_custom_readonly_python and scripts_for_question < max_scripts_per_question and total_scripts < max_scripts_total:
        actions.insert(1, "request_script")
        actions.insert(1, "request_checks")
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
    terminal = {
        "answer", "give_up", "mark_duplicate", "add_followup_questions",
        "request_context", "search_document", "read_document_chunks", "read_qdi_artifact_excerpt",
    }
    selected = [action for action in actions if action in terminal]
    return selected or ["answer", "give_up"]


def _qdi_retrieval_continuation_actions(
    question_live_actions: list[dict[str, Any]],
    *,
    document_retrievals_for_question: int,
    max_document_retrievals: int,
    artifact_retrievals_for_question: int,
    max_artifact_retrievals: int,
) -> list[str]:
    """Return only continuations justified by the last retrieval result."""
    if not question_live_actions:
        return []
    latest = question_live_actions[-1]
    action = str(latest.get("action") or "")
    observation = latest.get("observation") if isinstance(latest.get("observation"), dict) else {}
    payload = observation.get("content") if isinstance(observation.get("content"), dict) else observation

    if (
        action == "read_qdi_artifact_excerpt"
        and artifact_retrievals_for_question < max_artifact_retrievals
        and payload.get("status") == "completed"
        and payload.get("has_more") is True
    ):
        return ["read_qdi_artifact_excerpt"]

    if action == "search_document" and document_retrievals_for_question < max_document_retrievals:
        chunk_ids = [
            str(value).strip()
            for value in (latest.get("continuation_chunk_ids") or [])
            if str(value).strip()
        ]
        if not chunk_ids:
            matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
            chunk_ids = [
                str(item.get("chunk_id") or "").strip()
                for item in matches
                if isinstance(item, dict) and str(item.get("chunk_id") or "").strip()
            ]
        if chunk_ids:
            return ["read_document_chunks"]
    return []


def _apply_qdi_retrieval_action_policy(
    actions: list[str],
    *,
    retrieval_actions_since_terminal: int,
    max_retrieval_actions_before_terminal: int,
    continuation_actions: list[str],
) -> list[str]:
    """Cap repeated retrieval while keeping useful local evidence actions available."""
    allowed = list(actions)
    allowed.extend(action for action in continuation_actions if action not in allowed)
    if retrieval_actions_since_terminal < max(1, int(max_retrieval_actions_before_terminal)):
        return allowed
    terminal = {"answer", "give_up", "mark_duplicate", "add_followup_questions"}
    evidence_actions = {"request_checks", "request_script"}
    retained = [
        action for action in allowed
        if action in terminal or action in continuation_actions or action in evidence_actions
    ]
    return retained or ["answer", "give_up"]


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
                "question_truncated": len(str(record.get("question", ""))) > 1000,
                "status": str(record.get("status", "")),
                "short_answer": str(record.get("short_answer", ""))[:1200],
                "short_answer_truncated": len(str(record.get("short_answer", ""))) > 1200,
                "unresolved_reason": str(record.get("unresolved_reason", ""))[:1000],
                "unresolved_reason_truncated": len(str(record.get("unresolved_reason", ""))) > 1000,
                "confidence": str(record.get("confidence", "")),
                "used_files": [str(x) for x in (record.get("used_files", []) or [])[:12]],
                "used_files_omitted_count": max(0, len(record.get("used_files", []) or []) - 12),
                "history_artifact_ref": record.get("history_artifact_ref", {}),
                "parent_id": str(record.get("parent_id", "")),
                "depth": int(record.get("depth", 0) or 0),
                "category": str(record.get("category", "")),
                "duplicate_of_question_id": str(record.get("duplicate_of_question_id", "")),
            }
        )
    return out


def _qdi_question_ledger_for_decision(
    question_records: dict[str, dict[str, Any]],
    *,
    current_question_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return only cross-question routing state needed by the action model.

    The complete question records are persisted in the report and history
    artifacts. Re-sending every question's full prose and answer on every
    action turn made the old protocol grow quadratically with the number of
    questions. The current question is already sent as an immutable card, so
    other questions only need a short status/answer pointer to avoid duplicate
    follow-ups.
    """
    rows: list[dict[str, Any]] = []
    for record in question_records.values():
        qid = str(record.get("question_id") or "")
        if not qid or qid == str(current_question_id):
            continue
        rows.append(
            {
                "question_id": qid,
                "question": str(record.get("question") or "")[:320],
                "status": str(record.get("status") or ""),
                "short_answer": str(record.get("short_answer") or "")[:500],
                "unresolved_reason": str(record.get("unresolved_reason") or "")[:320],
                "confidence": str(record.get("confidence") or ""),
                "history_artifact_ref": record.get("history_artifact_ref", {}),
                "parent_id": str(record.get("parent_id") or ""),
                "depth": int(record.get("depth", 0) or 0),
            }
        )
        if len(rows) >= max(0, int(limit)):
            break
    return rows


def _qdi_digest_cards_for_decision(
    action_digest_cards: list[dict[str, Any]],
    *,
    question_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep a small, evidence-linked memory of completed prior actions."""
    rows: list[dict[str, Any]] = []
    for card in reversed(action_digest_cards):
        if str(card.get("question_id") or "") != str(question_id):
            continue
        rows.append(
            {
                "sequence": int(card.get("digest_for_sequence", 0) or 0),
                "action": str(card.get("action") or ""),
                "what_was_done": str(card.get("what_was_done") or "")[:500],
                "key_outputs": [str(x)[:320] for x in (card.get("key_outputs") or [])[:4]],
                "temporary_conclusion": str(card.get("temporary_conclusion") or "")[:500],
                "remaining_gap": str(card.get("remaining_gap") or "")[:500],
                "evidence_refs": [str(x)[:240] for x in (card.get("evidence_refs") or [])[:6]],
            }
        )
        if len(rows) >= max(0, int(limit)):
            break
    rows.reverse()
    return rows


def _qdi_latest_action_for_decision(
    question_live_actions: list[dict[str, Any]],
    *,
    max_chars: int,
) -> dict[str, Any]:
    """Project the latest observation without replaying the full action trace."""
    if not question_live_actions:
        return {}
    entry = question_live_actions[-1]
    action = str(entry.get("action") or "")
    raw_request = entry.get("request") if isinstance(entry.get("request"), dict) else {}
    request: dict[str, Any] = {}
    if action == "request_script":
        for key in (
            "question_id",
            "goal",
            "reason_builtins_insufficient",
            "input_files",
            "focus_sheets",
            "focus_columns",
            "expected_output",
            "python_code_sha256",
            "python_code_truncated",
            "python_code_original_chars",
            "source_artifact",
        ):
            if key in raw_request:
                request[key] = raw_request[key]
    elif action == "request_checks":
        request = {
            "checks": raw_request.get("checks", [])[:4],
        }
    else:
        for key in (
            "question_id",
            "table_ids",
            "input_files",
            "focus_sheets",
            "focus_columns",
            "query",
            "reason",
            "document_ids",
            "source_files",
            "top_k",
            "chunk_ids",
            "neighbor_count",
            "artifact_id",
            "json_path",
            "offset",
            "max_chars",
        ):
            if key in raw_request:
                request[key] = raw_request[key]
    observation = entry.get("observation", {})
    observation_view = _bounded_json_view(observation, max(1000, int(max_chars)))
    if observation_view.get("truncated"):
        observation_payload: Any = {
            "visible_excerpt": observation_view.get("visible_excerpt", ""),
            "truncated": True,
            "original_chars": observation_view.get("original_chars", 0),
            "visible_chars": observation_view.get("visible_chars", 0),
        }
        if isinstance(observation, dict):
            for key in ("artifact_ref", "artifact_id", "status", "error", "has_more", "next_offset"):
                if key in observation:
                    observation_payload[key] = observation[key]
    else:
        observation_payload = observation_view.get("content", observation)
    continuation_chunk_ids = [
        str(value)
        for value in (entry.get("continuation_chunk_ids") or [])
        if str(value).strip()
    ]
    if continuation_chunk_ids and isinstance(observation_payload, dict):
        observation_payload["available_chunk_ids"] = continuation_chunk_ids
    return {
        "sequence": int(entry.get("sequence", 0) or 0),
        "action": action,
        "status": str(entry.get("status") or ""),
        "request": request,
        "observation": observation_payload,
    }


def _qdi_next_step_hint(
    *,
    question_live_actions: list[dict[str, Any]],
    working_memory: dict[str, Any],
    available_actions: list[str],
    repetition_exhausted: bool,
    retrieval_actions_since_terminal: int,
    max_retrieval_actions_before_terminal: int,
    retrieval_continuation_actions: list[str],
) -> str:
    """Give the model a deterministic routing hint derived from host state."""
    if repetition_exhausted:
        return "The same evidence request has repeated. Interpret the existing evidence and answer or give_up."
    if not question_live_actions:
        return "Start with the smallest action that can resolve the blocking question; prefer a typed check over custom code."
    if retrieval_actions_since_terminal >= max_retrieval_actions_before_terminal:
        evidence_actions = [
            action for action in available_actions if action in {"request_checks", "request_script"}
        ]
        if retrieval_continuation_actions or evidence_actions:
            return (
                "No more ordinary retrievals are allowed. Continue only with the explicitly allowed artifact/document "
                "continuation, or use one bounded check/script for a concrete measurable gap; otherwise answer or give_up."
            )
        return "The retrieval budget is exhausted. Use the latest evidence to answer or give_up; do not request another retrieval."
    latest = question_live_actions[-1]
    observation = latest.get("observation") if isinstance(latest.get("observation"), dict) else {}
    status = str(
        observation.get("status")
        or latest.get("status")
        or ""
    )
    action = str(latest.get("action") or "")
    if status in {"script_failed", "artifact_excerpt_failed"}:
        return "The latest evidence action failed. Choose a different bounded source or give_up; do not repeat the same failed request."
    if status in {"context_retrieved", "document_searched", "document_chunks_read", "artifact_excerpt_read",
                  "script_completed", "checks_completed"}:
        if working_memory.get("open_gaps"):
            return "Use the latest evidence to close one listed gap. Request a new computation only if the gap is still measurable and no existing artifact answers it."
        return "The latest evidence is available. Prefer answer now; investigate further only for a concrete unresolved gap."
    if action in {"answer", "give_up", "mark_duplicate"}:
        return "This question is terminal; emit the corresponding terminal action."
    return f"Continue from action `{action}` using only a concrete, non-duplicate next step."


def _qdi_decision_state(
    *,
    initial_question: dict[str, Any],
    record: dict[str, Any],
    question_records: dict[str, dict[str, Any]],
    question_live_actions: list[dict[str, Any]],
    action_digest_cards: list[dict[str, Any]],
    working_memory: dict[str, Any],
    current_question_id: str,
    available_actions: list[str],
    pending_action_digest_requests: list[dict[str, Any]],
    investigation_actions_used: int,
    remaining_investigation_actions: int,
    max_actions_per_question: int,
    scripts_for_question: int,
    script_actions_total: int,
    max_scripts_per_question: int,
    max_scripts_total: int,
    context_retrievals_for_question: int,
    max_context_retrievals: int,
    document_retrievals_for_question: int,
    max_document_retrievals: int,
    artifact_retrievals_for_question: int,
    max_artifact_retrievals: int,
    retrieval_actions_since_terminal: int,
    max_retrieval_actions_before_terminal: int,
    retrieval_continuation_actions: list[str],
    decision_call_index: int,
    max_decision_calls: int,
    max_total_questions: int,
    max_depth: int,
    max_output_chars: int,
    task_hint: str,
    full_project_context_artifact: dict[str, Any],
    repetition_exhausted: bool,
    cfg: AutoRealizeConfig,
) -> dict[str, Any]:
    """Build the host-owned v6 action state frame.

    This is intentionally a projection, not a lossy rewrite of the report:
    full actions, scripts, results and memories remain addressable through
    artifacts. The model receives only state that can change its next action.
    """
    return {
        "protocol": str(getattr(cfg.investigation, "decision_state_protocol", "qdi-decision-state-v7")),
        "scope": {
            "task_hint": str(task_hint or ""),
            "task_hint_policy": "Dynamic task scope; project facts remain in the stable prefix.",
            "full_project_context_artifact": _artifact_ref_index_for_prompt(
                full_project_context_artifact
            )
            if isinstance(full_project_context_artifact, dict)
            else {},
        },
        # Compatibility marker for older local traces/consumers. The full
        # question lives only in decision_state.current_question; this alias
        # deliberately carries no duplicate prose.
        "initial_question": {
            "alias_of": "decision_state.current_question",
            "question_id": str(initial_question.get("question_id") or ""),
        },
        "current_question": initial_question,
        "question_status": {
            "status": str(record.get("status") or ""),
            "confidence": str(record.get("confidence") or ""),
            "short_answer": str(record.get("short_answer") or "")[:800],
            "unresolved_reason": str(record.get("unresolved_reason") or "")[:600],
            "used_files": [str(x) for x in (record.get("used_files") or [])[:12]],
        },
        "evidence_state": {
            "working_memory": _working_memory_prompt_view(
                working_memory,
                max_chars=int(getattr(cfg.investigation, "working_memory_max_chars", 12000)),
            ),
            "latest_action": _qdi_latest_action_for_decision(
                question_live_actions,
                max_chars=int(getattr(cfg.investigation, "decision_latest_evidence_chars", 6000)),
            ),
            "prior_evidence_digests": _qdi_digest_cards_for_decision(
                action_digest_cards,
                question_id=current_question_id,
                limit=int(getattr(cfg.investigation, "decision_digest_count", 4)),
            ),
            "pending_action_digest_requests": pending_action_digest_requests,
            "full_history_policy": (
                "Complete action history and large evidence are local artifacts. "
                "Use read_qdi_artifact_excerpt only when an omitted exact detail changes the decision."
            ),
        },
        "other_question_ledger": _qdi_question_ledger_for_decision(
            question_records,
            current_question_id=current_question_id,
            limit=int(getattr(cfg.investigation, "decision_question_ledger_limit", 8)),
        ),
        "decision": {
            "available_actions": available_actions,
            "next_step_hint": _qdi_next_step_hint(
                question_live_actions=question_live_actions,
                working_memory=working_memory,
                available_actions=available_actions,
                repetition_exhausted=repetition_exhausted,
                retrieval_actions_since_terminal=retrieval_actions_since_terminal,
                max_retrieval_actions_before_terminal=max_retrieval_actions_before_terminal,
                retrieval_continuation_actions=retrieval_continuation_actions,
            ),
            "investigation_action_budget": {
                "used": investigation_actions_used,
                "remaining_slots": remaining_investigation_actions,
                "limit": max_actions_per_question,
            },
            "script_budget": {
                "remaining_for_question": max(0, max_scripts_per_question - scripts_for_question),
                "remaining_for_run": max(0, max_scripts_total - script_actions_total),
                "repair_attempts_per_script_max": max(
                    0, int(getattr(cfg.investigation, "custom_python_max_retries", 3))
                ),
                "custom_python_max_result_bytes": max(
                    1024, int(getattr(cfg.investigation, "custom_python_max_result_bytes", 67108864))
                ),
            },
            "retrieval_budget": {
                "context": {
                    "used": context_retrievals_for_question,
                    "remaining": max(0, max_context_retrievals - context_retrievals_for_question),
                    "limit": max_context_retrievals,
                },
                "documents_shared_search_and_read": {
                    "used": document_retrievals_for_question,
                    "remaining": max(0, max_document_retrievals - document_retrievals_for_question),
                    "limit": max_document_retrievals,
                },
                "qdi_artifact_excerpts": {
                    "used": artifact_retrievals_for_question,
                    "remaining": max(0, max_artifact_retrievals - artifact_retrievals_for_question),
                    "limit": max_artifact_retrievals,
                },
            },
            "retrieval_round": {
                "actions_since_terminal": retrieval_actions_since_terminal,
                "max_actions_before_terminal": max_retrieval_actions_before_terminal,
                "continuation_actions_allowed": retrieval_continuation_actions,
                "terminal_decision_required": (
                    retrieval_actions_since_terminal >= max_retrieval_actions_before_terminal
                    and not retrieval_continuation_actions
                    and not any(action in available_actions for action in ("request_checks", "request_script"))
                ),
                "additional_evidence_actions_allowed": [
                    action for action in available_actions if action in {"request_checks", "request_script"}
                ],
                "continuation_rule": (
                    "After the retrieval cap, do not make ordinary repeated retrievals. A concrete measurable gap may use "
                    "request_checks/request_script if listed; artifact has_more=true and exposed document chunk_ids permit "
                    "their listed continuation. Successful request_context is not repeated by default."
                ),
            },
            "decision_call_index": decision_call_index,
            "max_decision_calls_including_retrievals": max_decision_calls,
            "decision_budget": {
                "used": decision_call_index,
                "remaining": max(0, max_decision_calls - decision_call_index),
                "limit": max_decision_calls,
                "counts_model_action_decisions_only": True,
            },
            "remaining_followup_questions": max(0, max_total_questions - len(question_records)),
            "current_depth": int(record.get("depth", 0) or 0),
            "max_depth": max_depth,
            "max_output_len": max_output_chars,
            "remaining_action_rounds": remaining_investigation_actions,
            "repeated_request_budget_exhausted": repetition_exhausted,
            "action_policy": {
                "answer_from_visible_evidence_before_new_work": True,
                "do_not_add_duplicate_questions": True,
                "use_mark_duplicate_when_needed": True,
                "use_refine_current_question_only_for_narrow_rewording": True,
                "prefer_request_checks_for_supported_typed_checks": True,
                "batch_independent_checks_in_one_request": True,
                "do_not_print_full_tables_in_scripts": True,
                "script_results_are_artifact_backed": True,
                "retrieval_actions_do_not_consume_investigation_action_slots": True,
                "retrieval_actions_still_require_a_model_decision_and_provider_call": True,
                "retrieval_round_cap_is_host_enforced": True,
                "context_retrieval_is_not_repeated_after_success": True,
                "artifact_continuation_requires_has_more": True,
                "document_read_continuation_requires_search_chunk_ids": True,
                "checks_or_scripts_remain_available_for_measurable_gaps": True,
                "script_actions_consume_one_investigation_action_slot_and_one_script_quota": True,
                "last_investigation_action_slot_allows_retrieval_or_terminal_actions_only": (
                    remaining_investigation_actions <= 1
                ),
            },
        },
    }


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
    output = payload.get("result", {})
    content = {key: value for key, value in output.items() if key != "_full_result_artifact"} if isinstance(output, dict) else output
    upstream_omissions = {}
    if isinstance(content, dict) and content.get("_truncated") and isinstance(content.get("visible_fields"), dict):
        upstream_omissions = {
            "omitted_field_count": content.get("omitted_field_count", 0),
            "omitted_fields": content.get("omitted_fields", []),
        }
        content = content["visible_fields"]
    visible = _visible_output_payload(content, max_output_chars)
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
        "current_output_artifact": _artifact_ref_index_for_prompt(artifact_ref),
        "upstream_omitted_fields": upstream_omissions,
        "instruction": (
            "完整结构化结果已保存到 current_output_artifact。truncated=true 表示当前 prompt 仅显示选中的完整字段；"
            "可以继续使用可见结论，但不能根据未显示部分推断。旧结果细节优先用 read_qdi_artifact_excerpt 取回；只有新的计算才需要脚本。"
        ),
    }


def _visible_output_payload(value: Any, max_output_chars: int) -> dict[str, Any]:
    safe = _json_safe(value)
    text = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    limit = max(1, int(max_output_chars))
    if len(text) <= limit:
        visible = text
    else:
        view: dict[str, Any] = {
            "_truncated": True,
            "original_chars": len(text),
            "visible_fields": {},
            "omitted_field_count": 0,
        }
        omitted_names: list[str] = []
        if isinstance(safe, dict):
            for key, item in safe.items():
                candidate = {**view, "visible_fields": {**view["visible_fields"], str(key): item}}
                if len(json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)) <= limit:
                    view = candidate
                else:
                    view["omitted_field_count"] += 1
                    omitted_names.append(str(key))
        else:
            view["root_value_omitted"] = True
            view["omitted_field_count"] = 1
        # Names are helpful for direct json_path reads, but only include the
        # complete list if it fits. Otherwise the exact structure remains in
        # the artifact and the generating script reveals its field paths.
        named_view = {**view, "omitted_fields": omitted_names}
        if omitted_names and len(json.dumps(named_view, ensure_ascii=False, sort_keys=True, default=str)) <= limit:
            view = named_view
        visible = json.dumps(view, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "current_visible_output": visible,
        "output_truncated": len(text) > limit,
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
    table_index = context.get("table_cards", []) if isinstance(context.get("table_cards"), list) else []
    cards_by_id = {
        str(item.get("table_id") or item.get("source_file") or ""): item
        for item in table_index
        if isinstance(item, dict) and str(item.get("table_id") or item.get("source_file") or "")
    }
    cards_by_id.update(
        {
            str(table_id): card
            for table_id, card in table_card_details.items()
            if isinstance(card, dict)
        }
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for table_id in explicit_table_ids:
        card = cards_by_id.get(table_id)
        if isinstance(card, dict):
            selected.append(_context_card_for_retrieval(card, wanted_cols))
            seen.add(table_id)
        if len(selected) >= max_cards:
            return selected
    if selected:
        return selected
    matched = _select_related_table_cards(
        table_index,
        cards_by_id,
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
    return [
        _context_card_for_retrieval(cards_by_id.get(str(card.get("table_id")), card), wanted_cols)
        for card in matched
    ]


def _context_card_for_retrieval(card: dict[str, Any], wanted_columns: set[str]) -> dict[str, Any]:
    """Return whole-field prompt cards with exact schema and a path back to every full profile."""
    full_card = deepcopy(card)
    fields = full_card.get("fields")
    fields = fields if isinstance(fields, list) else []
    if wanted_columns:
        fields = sorted(fields, key=lambda field: (
            0 if isinstance(field, dict) and str(field.get("name")) in wanted_columns else 1,
            str(field.get("name", "")) if isinstance(field, dict) else "",
        ))
        full_card["fields"] = fields
    view = _detail_card_for_prompt(full_card)
    view["fields"] = deepcopy(fields[:12])
    for key in ("reading_notes", "warnings", "recommended_read", "row_count_note"):
        if key in full_card:
            view[key] = deepcopy(full_card[key])
    full_field_names = [str(x) for x in (card.get("physical_columns_exact") or [])]
    if not full_field_names:
        full_field_names = [str(field.get("name", "")) for field in fields if isinstance(field, dict)]
    view["field_count"] = len(full_field_names)
    view["fields_omitted_count"] = max(0, len(full_field_names) - len(view.get("fields", [])))
    view["field_catalog"] = full_field_names
    view["artifact_refs"] = [
        _artifact_ref_index_for_prompt(ref)
        for ref in (card.get("artifact_refs") or [])
        if isinstance(ref, dict)
    ]
    view["detail_policy"] = (
        "Visible fields retain whole stored-card entries, which are still summaries of the source profiles. "
        "field_catalog lists the physical schema; complete original profiles and meanings are recoverable from artifact_refs."
    )
    view["artifact_json_path_guide"] = {
        str(ref.get("artifact_type")): (
            {"all_field_names": "columns", "full_field_profiles": "column_profiles"}
            if str(ref.get("artifact_type")) == "excel_sheet_profile_full"
            else {"all_field_profiles": "column_profiles", "field_meanings": "column_semantics",
                  "excel_sheets": "source_metadata.excel_sheet_profiles"}
        )
        for ref in (card.get("artifact_refs") or [])
        if isinstance(ref, dict) and ref.get("artifact_type") in {"file_profile_full", "excel_sheet_profile_full"}
    }
    return view


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
        if not isinstance(card, dict):
            card = next(
                (
                    item for item in table_index
                    if isinstance(item, dict)
                    and str(item.get("table_id") or item.get("source_file") or "") == table_id
                ),
                None,
            )
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
        payload["result_artifact"] = _artifact_ref_index_for_prompt(stored_ref)
    elif artifact_store is not None:
        ref = artifact_store.put(
            "qdi_failed_script_result_full",
            f"{result.question_id}:{result.request_id}",
            result.result,
            visible_excerpt=visible["current_visible_output"],
            visible_limit=max(1000, int(result.max_output_chars or 12000)),
        )
        payload["result_artifact"] = _artifact_ref_index_for_prompt(ref)
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
    """Run custom scripts and engine-compiled checks through one read-only sandbox.

    Legacy internal helper names remain unavailable to the public executor.
    Typed checks are compiled from validated arguments rather than model code.
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
            visible_result = _truncate_result(result, max_chars) if full_result_artifact else result
            if isinstance(visible_result, dict) and full_result_artifact:
                visible_result["_full_result_artifact"] = _artifact_ref_index_for_prompt(full_result_artifact)
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
    """Run bounded analysis with best-effort path guards, not OS-level isolation."""
    code = str(code or "").strip()
    if not code:
        return {"error": "empty_custom_python_code"}
    issues = validate_custom_readonly_python(code)
    if issues:
        return {"error": "custom_python_static_validation_failed", "issues": issues}

    timeout = float(getattr(cfg.investigation, "custom_python_timeout_seconds", 30.0))
    max_stdout = max(1, int(getattr(cfg.investigation, "custom_python_max_stdout_chars", 12000)))
    with tempfile.TemporaryDirectory(prefix="autorealize_scratch_") as scratch:
        scratch_dir = Path(scratch).resolve()
        wrapper_path = scratch_dir / "run_custom_readonly.py"
        result_path = scratch_dir / "__autorealize_result.json"
        max_result_bytes = max(
            1024, int(getattr(cfg.investigation, "custom_python_max_result_bytes", 67108864))
        )
        wrapper_path.write_text(
            _custom_python_wrapper(code, max_result_bytes=max_result_bytes), encoding="utf-8"
        )
        env = _custom_readonly_python_environment()
        proc = subprocess.Popen(
            [
                sys.executable, "-u", str(wrapper_path), str(input_dir.resolve()),
                str(scratch_dir), str(result_path),
            ],
            cwd=str(scratch_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_capture, stdout_thread = _start_bounded_pipe_capture(proc.stdout, max_stdout)
        stderr_capture, stderr_thread = _start_bounded_pipe_capture(proc.stderr, max_stdout)
        timed_out = False
        try:
            return_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            return_code = proc.wait()
        finally:
            stdout_thread.join()
            stderr_thread.join()
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()

        stdout_info = _finish_bounded_pipe_capture(stdout_capture, max_stdout)
        stderr_info = _finish_bounded_pipe_capture(stderr_capture, max_stdout)
        stdout = stdout_info["visible_tail"]
        stderr = stderr_info["visible_tail"]
        if timed_out:
            return {
                "error": f"custom_python_timeout:{timeout}",
                "stdout_tail": stdout,
                "stderr_tail": stderr,
                "_stdout_capture": stdout_info,
                "_stderr_capture": stderr_info,
            }
        if return_code != 0:
            return {
                "error": f"custom_python_exit_code:{return_code}",
                "stdout_tail": stdout,
                "stderr_tail": stderr,
                "_stdout_capture": stdout_info,
                "_stderr_capture": stderr_info,
            }
        if not result_path.exists():
            return {
                "error": "custom_python_missing_result_file",
                "stdout_tail": stdout,
                "stderr_tail": stderr,
                "_stdout_capture": stdout_info,
                "_stderr_capture": stderr_info,
            }
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "error": f"custom_python_result_not_json:{exc}",
                "stdout_tail": stdout,
                "stderr_tail": stderr,
                "_stdout_capture": stdout_info,
                "_stderr_capture": stderr_info,
            }
        if not isinstance(result, dict):
            return {"error": "custom_python_result_not_object", "result": _json_safe(result)}
        result["_scratch_destroyed_after_execution"] = True
        if stdout_info["original_bytes"]:
            result["_stdout_capture"] = stdout_info
        if stderr.strip():
            result["_stderr_capture"] = stderr_info
        return result


def _custom_readonly_python_environment() -> dict[str, str]:
    """Pass only platform/runtime variables needed by the local analysis child."""
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "APPDATA",
        "USERPROFILE",
        "HOME",
        "PYTHONUSERBASE",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed
    }
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
    )
    return env


def _start_bounded_pipe_capture(stream, max_chars: int):
    max_bytes = max(4, max_chars * 4)
    capture = {"tail": bytearray(), "original_bytes": 0}

    def drain() -> None:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            capture["original_bytes"] += len(chunk)
            capture["tail"].extend(chunk)
            if len(capture["tail"]) > max_bytes:
                del capture["tail"][:-max_bytes]

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return capture, thread


def _finish_bounded_pipe_capture(capture: dict[str, Any], max_chars: int) -> dict[str, Any]:
    visible = bytes(capture["tail"]).decode("utf-8", errors="replace")[-max_chars:]
    return {
        "visible_tail": visible,
        "truncated": int(capture["original_bytes"]) > len(visible.encode("utf-8", errors="replace")),
        "original_bytes": int(capture["original_bytes"]),
        "visible_chars": len(visible),
    }


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
        "pyarrow",
        "fastparquet",
        "polars",
        "xarray",
        "h5py",
        "tables",
        "zarr",
    }
    allowed_import_roots = {
        "__future__",
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "statsmodels",
        "networkx",
        "rapidfuzz",
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
    banned_calls = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "help",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }
    banned_reflection_names = {
        "__builtins__",
        "__globals__",
        "__dict__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "__import__",
    }
    banned_attrs = {
        "unlink",
        "rmdir",
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

    pandas_aliases = {"pd", "pandas"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pandas":
                    pandas_aliases.add(alias.asname or "pandas")
        elif isinstance(node, ast.ImportFrom) and node.module == "pandas":
            pandas_aliases.add(node.module)
    dataframe_names, series_names = _readonly_pandas_names(tree, pandas_aliases)

    def attribute_chain(node: ast.AST) -> list[str]:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return list(reversed(parts))

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
                if alias.name == "pandas.io" or alias.name.startswith("pandas.io."):
                    issues.append(f"banned_pandas_internal_io_import:{alias.name}")
                elif root in banned_import_roots:
                    issues.append(f"banned_import:{alias.name}")
                elif root not in allowed_import_roots:
                    issues.append(f"import_not_allowlisted:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            module = node.module or ""
            if module == "pandas.io" or module.startswith("pandas.io."):
                issues.append(f"banned_pandas_internal_io_import:{module}")
            elif module == "pandas" and any(alias.name == "io" for alias in node.names):
                issues.append("banned_pandas_internal_io_import:pandas.io")
            elif root in banned_import_roots:
                issues.append(f"banned_import:{node.module}")
            elif root and root not in allowed_import_roots:
                issues.append(f"import_not_allowlisted:{node.module}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in banned_calls:
                issues.append(f"banned_call:{name}")
            attr = _call_attr(node.func)
            if attr in {"rename", "replace"} and not _is_readonly_data_transform(
                node, attr, dataframe_names, series_names
            ):
                issues.append(f"banned_method:{attr}")
            if attr in banned_attrs and not (attr == "copy" and _is_memory_copy_call(node)):
                issues.append(f"banned_method:{attr}")
        elif isinstance(node, ast.Name) and node.id in banned_reflection_names:
            issues.append(f"banned_python_reflection_name:{node.id}")
        elif isinstance(node, ast.Attribute):
            chain = attribute_chain(node)
            if len(chain) >= 2 and chain[0] in pandas_aliases and chain[1] == "io":
                issues.append(f"banned_pandas_internal_io_access:{'.'.join(chain[:2])}")
            if node.attr.startswith("__") and node.attr.endswith("__"):
                issues.append(f"banned_python_reflection_attribute:{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            issue = _literal_path_issue(node.value)
            if issue:
                issues.append(issue)

    if not has_analyze:
        issues.append("missing_analyze_function")
    return list(dict.fromkeys(issues))[:40]


def _readonly_pandas_names(
    tree: ast.AST,
    pandas_aliases: set[str],
) -> tuple[set[str], set[str]]:
    dataframes: set[str] = set()
    series: set[str] = set()
    dataframe_transforms = {
        "copy", "rename", "replace", "drop", "dropna", "fillna", "astype",
        "sort_values", "sort_index", "reset_index", "set_index", "assign",
        "query", "head", "tail", "groupby", "merge", "join", "select_dtypes",
    }

    def dataframe_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in dataframes
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id in pandas_aliases:
                return node.func.attr in {"DataFrame", "read_csv", "read_excel", "read_parquet", "read_json"}
            return (
                isinstance(owner, ast.Name)
                and owner.id in dataframes
                and node.func.attr in dataframe_transforms
            )
        return False

    def series_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in series
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id in dataframes
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id in dataframes and node.attr not in {"columns", "index"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in series
                and node.func.attr in {"copy", "rename", "replace", "astype", "fillna", "dropna"}
            )
        return False

    def assignment_targets(node: ast.AST) -> list[ast.Name]:
        if isinstance(node, ast.Name):
            return [node]
        if isinstance(node, (ast.Tuple, ast.List)):
            return [name for child in node.elts for name in assignment_targets(child)]
        return []

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            value = None
            targets: list[ast.Name] = []
            if isinstance(node, ast.Assign):
                value = node.value
                targets = [name for target in node.targets for name in assignment_targets(target)]
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                targets = assignment_targets(node.target)
            elif isinstance(node, ast.NamedExpr):
                value = node.value
                targets = assignment_targets(node.target)
            if value is None or not targets:
                continue
            is_frame = dataframe_expr(value)
            is_series = series_expr(value)
            for target in targets:
                if is_frame and target.id not in dataframes:
                    dataframes.add(target.id)
                    changed = True
                elif is_series and target.id not in series:
                    series.add(target.id)
                    changed = True
    return dataframes, series


def _is_readonly_data_transform(
    node: ast.Call,
    attr: str,
    dataframe_names: set[str],
    series_names: set[str],
) -> bool:
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    if receiver is None:
        return False
    if attr == "rename":
        return isinstance(receiver, ast.Name) and (
            receiver.id in dataframe_names or receiver.id in series_names
        )
    if isinstance(receiver, ast.Constant) and isinstance(receiver.value, str):
        return True
    if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name) and receiver.func.id == "str":
        return True
    if isinstance(receiver, ast.Attribute) and receiver.attr == "str":
        return True
    if isinstance(receiver, ast.Name):
        return receiver.id in dataframe_names or receiver.id in series_names
    if isinstance(receiver, ast.Subscript) and isinstance(receiver.value, ast.Name):
        return receiver.value.id in dataframe_names
    return False


def _is_memory_copy_call(node: ast.Call) -> bool:
    # File copies require a destination; only allow explicit memory-copy forms.
    if node.args:
        return False
    if not node.keywords:
        return True
    if len(node.keywords) != 1:
        return False
    keyword = node.keywords[0]
    if not isinstance(keyword.value, ast.Constant):
        return False
    value = keyword.value.value
    if keyword.arg == "deep":
        return isinstance(value, bool)
    if keyword.arg == "order":
        return isinstance(value, str) and value in {"C", "F", "A", "K"}
    return False


def _custom_python_wrapper(user_code: str, *, max_result_bytes: int = 67108864) -> str:
    return (
        _CUSTOM_PYTHON_WRAPPER_PREFIX
        + "\nUSER_CODE = "
        + repr(user_code)
        + "\nMAX_RESULT_BYTES = "
        + str(max(1024, int(max_result_bytes)))
        + "\n"
        + _CUSTOM_PYTHON_WRAPPER_SUFFIX
    )


_CUSTOM_PYTHON_WRAPPER_PREFIX = r'''
from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import socket
import stat as stat_module
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
_orig_path_exists = Path.exists
_orig_path_is_file = Path.is_file
_orig_path_is_dir = Path.is_dir
_orig_path_stat = Path.stat


def _under(path, root):
    path = Path(os.path.realpath(os.fspath(path)))
    root = Path(os.path.realpath(os.fspath(root)))
    return path == root or root in path.parents


def _resolve_user_path(file):
    if isinstance(file, int):
        return None
    return Path(os.path.realpath(os.fspath(file)))


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


def _safe_glob(self, pattern, *args, **kwargs):
    _guard_read_path(self)
    for path in _orig_path_glob(self, pattern, *args, **kwargs):
        _guard_read_path(path)
        yield path


def _safe_rglob(self, pattern, *args, **kwargs):
    _guard_read_path(self)
    for path in _orig_path_rglob(self, pattern, *args, **kwargs):
        _guard_read_path(path)
        yield path


def _safe_exists(self):
    _guard_read_path(self)
    try:
        _orig_path_stat(self)
        return True
    except (OSError, ValueError):
        return False


def _safe_is_file(self):
    _guard_read_path(self)
    try:
        return stat_module.S_ISREG(_orig_path_stat(self).st_mode)
    except (OSError, ValueError):
        return False


def _safe_is_dir(self):
    _guard_read_path(self)
    try:
        return stat_module.S_ISDIR(_orig_path_stat(self).st_mode)
    except (OSError, ValueError):
        return False


def _safe_stat(self, *args, **kwargs):
    _guard_read_path(self)
    return _orig_path_stat(self, *args, **kwargs)


builtins.open = _safe_open
Path.open = _safe_path_open
Path.read_text = _safe_read_text
Path.read_bytes = _safe_read_bytes
Path.write_text = _safe_write_text
Path.write_bytes = _safe_write_bytes
Path.iterdir = _safe_iterdir
Path.glob = _safe_glob
Path.rglob = _safe_rglob
Path.exists = _safe_exists
Path.is_file = _safe_is_file
Path.is_dir = _safe_is_dir
Path.stat = _safe_stat


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


def _guard_reader_source(source):
    if isinstance(source, (str, bytes, os.PathLike)):
        path = os.fsdecode(os.fspath(source))
        if "://" in path:
            raise PermissionError("remote paths are disabled in custom_readonly_python")
        _guard_read_path(path)
    elif isinstance(source, (list, tuple)):
        for item in source:
            _guard_reader_source(item)


def _patch_pandas_reader(function_name, source_argument):
    if not hasattr(pd, function_name):
        return
    original = getattr(pd, function_name)

    def _safe_reader(*args, **kwargs):
        source = args[0] if args else kwargs.get(source_argument)
        _guard_reader_source(source)
        if kwargs.get("storage_options") or kwargs.get("filesystem") is not None:
            raise PermissionError("external filesystems are disabled in custom_readonly_python")
        return original(*args, **kwargs)

    setattr(pd, function_name, _safe_reader)


def _patch_pandas_file_constructor(class_name):
    cls = getattr(pd, class_name, None)
    if cls is None or not hasattr(cls, "__init__"):
        return
    original = cls.__init__

    def _safe_init(self, source, *args, **kwargs):
        _guard_reader_source(source)
        return original(self, source, *args, **kwargs)

    cls.__init__ = _safe_init


for _reader, _source_argument in {
    "read_csv": "filepath_or_buffer",
    "read_fwf": "filepath_or_buffer",
    "read_table": "filepath_or_buffer",
    "read_excel": "io",
    "read_json": "path_or_buf",
    "read_html": "io",
    "read_xml": "path_or_buffer",
    "read_parquet": "path",
    "read_feather": "path",
    "read_orc": "path",
    "read_pickle": "filepath_or_buffer",
    "read_hdf": "path_or_buf",
    "read_sas": "filepath_or_buffer",
    "read_stata": "filepath_or_buffer",
}.items():
    _patch_pandas_reader(_reader, _source_argument)

_patch_pandas_file_constructor("ExcelFile")
_patch_pandas_file_constructor("HDFStore")


for _method in ["to_csv", "to_excel", "to_json", "to_parquet", "to_pickle", "to_feather", "to_hdf"]:
    _patch_pandas_writer(pd.DataFrame, _method)
    _patch_pandas_writer(pd.Series, _method)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
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
    encoder = json.JSONEncoder(ensure_ascii=False, default=str, separators=(",", ":"))
    serialized_bytes = 0
    exceeded_limit = False
    with _orig_path_open(RESULT_PATH, "w", encoding="utf-8") as result_stream:
        for chunk in encoder.iterencode(_json_safe(result)):
            chunk_bytes = len(chunk.encode("utf-8"))
            if serialized_bytes + chunk_bytes > MAX_RESULT_BYTES:
                exceeded_limit = True
                break
            result_stream.write(chunk)
            serialized_bytes += chunk_bytes
    if exceeded_limit:
        _orig_path_write_text(
            RESULT_PATH,
            json.dumps({
                "error": "custom_python_result_size_limit_exceeded",
                "max_result_bytes": MAX_RESULT_BYTES,
                "serialized_result_bytes_at_least": serialized_bytes + chunk_bytes,
            }, ensure_ascii=False),
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
        "question": str(record.get("question", "")),
        "category": str(record.get("category", "")),
        "why_blocking": str(record.get("why_blocking", "")),
        "candidate_files": [str(x) for x in (record.get("candidate_files", []) or [])],
        "parent_id": str(record.get("parent_id", "")),
        "depth": int(record.get("depth", 0) or 0),
    }


def _qdi_cache_metrics(
    *,
    llm_client: Any,
    node_cache: QDINodeResultCache,
    evidence_cache: QDIEvidenceCache,
) -> dict[str, Any]:
    """Return QDI cache metrics without merging different cache layers."""
    summary = getattr(llm_client, "_usage_summary", {})
    by_prompt = summary.get("by_prompt", {}) if isinstance(summary, dict) else {}
    qdi_rows = [
        row for name, row in by_prompt.items()
        if str(name).startswith("question_investigator")
        and isinstance(row, dict)
    ] if isinstance(by_prompt, dict) else []

    def total(field: str) -> int:
        return sum(int(row.get(field, 0) or 0) for row in qdi_rows)

    provider_input = total("prompt_tokens")
    provider_hit = total("prompt_cache_hit_tokens")
    provider_miss = total("prompt_cache_miss_tokens")
    provider_known = total("provider_cache_known_prompt_tokens")
    local_hits = total("cache_hits_local")
    return {
        "metric_semantics": {
            "provider_prompt_cache": "Provider-reported prompt prefix cache tokens for QDI network calls.",
            "qdi_node_result_cache": "Validated local/shared QDI decision results; a hit avoids a provider call.",
            "local_response_cache": "Exact LLM response cache hits; separate from provider prompt caching.",
            "qdi_evidence_cache": "Validated deterministic evidence/tool result reuse; separate from all LLM caches.",
        },
        "provider_prompt_cache": {
            "input_tokens": provider_input,
            "hit_tokens": provider_hit,
            "miss_tokens": provider_miss,
            "known_input_tokens": provider_known,
            "telemetry_coverage": round(provider_known / provider_input, 6) if provider_input else None,
            "hit_ratio_known": round(provider_hit / provider_known, 6) if provider_known else None,
            "hit_ratio_lower_bound": round(provider_hit / provider_input, 6) if provider_input else None,
        },
        "qdi_node_result_cache": node_cache.stats(),
        "local_response_cache": {
            "hits": local_hits,
            "estimated_prompt_tokens_avoided": total(
                "estimated_prompt_tokens_local_cache_hits"
            ),
        },
        "qdi_evidence_cache": {
            "hits": int(getattr(evidence_cache, "hits", 0) or 0),
            "local_hits": int(getattr(evidence_cache, "local_hits", 0) or 0),
            "shared_hits": int(getattr(evidence_cache, "shared_hits", 0) or 0),
        },
    }


def _qdi_answerer_stable_prefix(context: dict[str, Any]) -> str:
    """Return the project prefix shared by Planner and every QDI question."""
    segments, _ = _qdi_stable_context_segments(_qdi_project_cache_context(context), None)
    return segments[0]["content"]


def _qdi_answerer_cache_context(context: dict[str, Any]) -> dict[str, Any]:
    """Compatibility alias for callers that need only the shared project facts."""
    return _qdi_project_cache_context(context)


def _qdi_cfg_int(cfg: Any | None, name: str, default: int) -> int:
    investigation = getattr(cfg, "investigation", None) if cfg is not None else None
    try:
        return max(1, int(getattr(investigation, name, default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _qdi_bounded_text(value: Any, limit: int) -> str:
    return str(value or "")[: max(0, int(limit))]


def _qdi_unique_strings(values: Any, *, limit: int, item_chars: int = 700) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        text = _qdi_bounded_text(value, item_chars).strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _qdi_compact_authoritative_memory(memory: Any) -> dict[str, Any]:
    """Keep authority-bearing facts while removing repeated prose and telemetry."""
    if not isinstance(memory, dict):
        return {}
    out: dict[str, Any] = {}
    for key, limit, chars in (
        ("summary", 1, 1800),
        ("task_goal", 1, 1800),
        ("source_files", 40, 220),
        ("input_requirements", 18, 700),
        ("output_requirements", 18, 700),
        ("evaluation_requirements", 18, 700),
        ("constraints", 28, 700),
        ("leakage_guards", 18, 700),
        ("unresolved_questions", 16, 600),
    ):
        value = memory.get(key)
        if isinstance(value, list):
            bounded = _qdi_unique_strings(value, limit=limit, item_chars=chars)
        elif value not in (None, ""):
            bounded = _qdi_bounded_text(value, chars)
        else:
            bounded = ""
        if bounded not in ("", [], {}, None):
            out[key] = bounded
    for key in ("has_authoritative_sources", "authority_conflicts"):
        value = memory.get(key)
        if value not in (None, "", [], {}):
            if isinstance(value, list):
                out[key] = value[:16]
            else:
                out[key] = value
    contract = memory.get("submission_contract")
    if isinstance(contract, dict):
        out["submission_contract"] = {
            key: contract[key]
            for key in (
                "is_defined",
                "is_authoritative",
                "output_filename",
                "sample_filename",
                "columns",
                "row_unit",
                "row_count_rule",
                "format_description",
                "validation_rules",
                "source",
            )
            if contract.get(key) not in (None, "", [], {})
        }
    return out


def _qdi_compact_constraint_memory(memory: Any) -> dict[str, Any]:
    """Retain high-impact constraint semantics without replaying the ledger.

    The complete constraint memory is stored in the project artifact.  The
    provider prefix only needs enough information to route a question.  Keep
    hard/current items first, expose a small exact excerpt, and make omission
    explicit so an omitted item can never be interpreted as a negative fact.
    """
    if not isinstance(memory, dict):
        return {}
    out: dict[str, Any] = {}
    summary = _qdi_bounded_text(memory.get("summary"), 1400).strip()
    if summary:
        out["summary"] = summary
    source_items = [item for item in list(memory.get("items") or []) if isinstance(item, dict)]
    priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    strength_rank = {"hard": 0, "soft": 1, "unknown": 2}
    status_rank = {"current": 0, "unresolved": 1, "current_candidate": 2}
    source_items.sort(
        key=lambda item: (
            priority_rank.get(str(item.get("priority") or "medium").lower(), 4),
            strength_rank.get(str(item.get("strength") or "unknown").lower(), 3),
            status_rank.get(str(item.get("status") or "unresolved").lower(), 3),
            str(item.get("name") or ""),
        )
    )
    visible_limit = 16
    items: list[dict[str, Any]] = []
    for item in source_items[:visible_limit]:
        if not isinstance(item, dict):
            continue
        compact = {
            key: item[key]
            for key in (
                "name",
                "description",
                "priority",
                "strength",
                "status",
                "related_fields",
                "acceptance_check",
                "status_reason",
            )
            if item.get(key) not in (None, "", [], {})
        }
        if isinstance(compact.get("description"), str):
            compact["description"] = compact["description"][:420]
        if isinstance(compact.get("acceptance_check"), str):
            compact["acceptance_check"] = compact["acceptance_check"][:360]
        if isinstance(compact.get("status_reason"), str):
            compact["status_reason"] = compact["status_reason"][:280]
        if isinstance(compact.get("related_fields"), list):
            compact["related_fields"] = [str(x)[:100] for x in compact["related_fields"][:8]]
        evidence = _qdi_unique_strings(item.get("evidence"), limit=1, item_chars=180)
        if evidence:
            compact["evidence"] = evidence
        if compact:
            items.append(compact)
    if items:
        out["items"] = items
    if len(source_items) > len(items):
        out["items_omitted_count"] = len(source_items) - len(items)
    unresolved = _qdi_unique_strings(memory.get("unresolved_questions"), limit=12, item_chars=500)
    if unresolved:
        out["unresolved_questions"] = unresolved
    aliases = memory.get("entity_alias_candidates")
    if isinstance(aliases, list):
        out["entity_alias_candidate_count"] = len(aliases)
    return out


def _qdi_table_index_card(card: dict[str, Any]) -> dict[str, Any]:
    fields = card.get("fields") if isinstance(card.get("fields"), list) else []
    field_names: list[str] = []
    field_roles: dict[str, str] = {}
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        if not name or name in field_names:
            continue
        field_names.append(name[:120])
        role = str(field.get("role") or "").strip()
        if role and role != "other":
            field_roles[name[:120]] = role[:80]
    if not field_names:
        field_names = [
            str(value)[:120]
            for value in (card.get("physical_columns_exact") or card.get("field_hints") or [])
            if str(value).strip()
        ]
    field_name_count = len(field_names)
    result = {
        key: card.get(key)
        for key in (
            "table_id",
            "source_file",
            "sheet_name",
            "table_kind",
            "role",
            "shape",
            "verified_row_count",
            "row_count_basis",
            "layout_kind",
            "sheet_group_id",
            "sheet_group_size",
        )
        if card.get(key) not in (None, "", [], {})
    }
    if field_names:
        result["field_names"] = field_names[:16]
        if field_name_count > 16:
            result["field_names_omitted_count"] = field_name_count - 16
    if field_roles:
        result["field_roles"] = {
            name: role
            for name, role in list(field_roles.items())[:16]
        }
        if len(field_roles) > 16:
            result["field_roles_omitted_count"] = len(field_roles) - 16
    artifact_refs = card.get("artifact_refs")
    if isinstance(artifact_refs, list):
        refs = [
            _artifact_ref_index_for_prompt(ref)
            for ref in artifact_refs[:3]
            if isinstance(ref, dict) and ref.get("artifact_id")
        ]
        if refs:
            result["artifact_refs"] = refs
    return result


def _qdi_compact_project_table_cards(
    cards: Any,
    *,
    cfg: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a route-complete index and a bounded evidence-card subset."""
    source_cards = [card for card in list(cards or []) if isinstance(card, dict)]
    source_cards.sort(
        key=lambda card: (
            str(card.get("source_file") or ""),
            str(card.get("sheet_name") or ""),
            str(card.get("table_id") or ""),
        )
    )
    table_index = [_qdi_table_index_card(card) for card in source_cards]
    card_limit = _qdi_cfg_int(cfg, "project_table_card_limit", 24)
    field_limit = _qdi_cfg_int(cfg, "project_table_card_field_limit", 12)
    max_chars = _qdi_cfg_int(cfg, "project_table_cards_max_chars", 60000)
    per_source_limit = _qdi_cfg_int(cfg, "project_table_cards_per_source", 8)

    def score(card: dict[str, Any]) -> tuple[int, str]:
        role = str(card.get("role") or "").lower()
        kind = str(card.get("table_kind") or "").lower()
        priority = 0
        if role in {"task_requirement", "data_description"} or kind == "document":
            priority += 4
        if role in {"raw_data_table", "label_table", "output_table"}:
            priority += 3
        if card.get("row_count_conflict") or card.get("reading_risks"):
            priority += 1
        return -priority, str(card.get("table_id") or card.get("source_file") or "")

    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    used_chars = 0
    omitted: list[dict[str, Any]] = []
    for card in sorted(source_cards, key=score):
        source = str(card.get("source_file") or card.get("table_id") or "")
        if len(selected) >= card_limit or source_counts.get(source, 0) >= per_source_limit:
            omitted.append(_qdi_table_index_card(card))
            continue
        projected = _json_safe(compact_detail_table_card_for_prompt(card, field_limit=field_limit))
        size = len(json.dumps(projected, ensure_ascii=False, sort_keys=True, default=str))
        if selected and used_chars + size > max_chars:
            omitted.append(_qdi_table_index_card(card))
            continue
        selected.append(projected)
        source_counts[source] = source_counts.get(source, 0) + 1
        used_chars += size
    if omitted:
        selected.append(
            {
                "table_id": "__omitted_project_table_cards__",
                "table_kind": "omitted_table_manifest",
                "omitted_count": len(omitted),
                "omitted_table_ids": [
                    str(item.get("table_id") or item.get("source_file") or "")
                    for item in omitted[:24]
                ],
                "omitted_table_ids_are_sample": len(omitted) > 24,
                "detail_policy": (
                    "Only the bounded evidence-card subset is visible here. "
                    "Use request_context for a table card or read_qdi_artifact_excerpt "
                    "on full_project_context_artifact for exact omitted details."
                ),
            }
        )
    return table_index, selected


def _qdi_compact_relations(value: Any, *, limit: int) -> list[dict[str, Any]]:
    relations = [item for item in list(value or []) if isinstance(item, dict)]
    relations.sort(
        key=lambda item: (
            str(item.get("left_file") or ""),
            str(item.get("right_file") or ""),
            str(item.get("left_field") or ""),
            str(item.get("right_field") or ""),
        )
    )
    out: list[dict[str, Any]] = []
    for item in relations[: max(1, int(limit))]:
        compact = {
            key: item.get(key)
            for key in (
                "left_file",
                "left_field",
                "right_file",
                "right_field",
                "relation_type",
                "confidence",
                "short_evidence",
            )
            if item.get(key) not in (None, "", [], {})
        }
        if isinstance(compact.get("short_evidence"), str):
            compact["short_evidence"] = compact["short_evidence"][:320]
        if compact:
            out.append(compact)
    return out


def _qdi_compact_coverage(value: Any, *, limit: int = 80) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    source_entries = [item for item in list(value.get("entries") or []) if isinstance(item, dict)]

    def attention_score(item: dict[str, Any]) -> tuple[int, str]:
        score = 0
        if item.get("row_count_conflict"):
            score -= 6
        if item.get("verified_row_count") in (None, ""):
            score -= 4
        if item.get("recommended_read") or item.get("layout_kind"):
            score -= 2
        if str(item.get("coverage_status") or "") in {"required", "supporting"}:
            score -= 1
        return score, str(item.get("table_id") or item.get("source_file") or "")

    # table_index already carries the complete route inventory.  Coverage is
    # therefore an exception ledger in the stable prefix, not a second copy of
    # every table schema.
    attention_entries = sorted(source_entries, key=attention_score)
    visible_limit = max(1, min(int(limit), 24))
    entries: list[dict[str, Any]] = []
    for item in attention_entries[:visible_limit]:
        if not isinstance(item, dict):
            continue
        is_attention = bool(
            item.get("row_count_conflict")
            or item.get("verified_row_count") in (None, "")
            or item.get("recommended_read")
            or item.get("layout_kind")
            or str(item.get("coverage_status") or "") not in {"documentation", "required", "supporting"}
        )
        if not is_attention and len(entries) >= 8:
            continue
        compact = {
            key: item.get(key)
            for key in (
                "table_id",
                "source_file",
                "sheet_name",
                "table_kind",
                "source_role",
                "coverage_status",
                "verified_shape",
                "verified_row_count",
                "row_count_basis",
                "row_count_conflict",
                "schema_signature_fields",
                "recommended_read",
                "layout_kind",
                "primary_key_candidates",
            )
            if item.get(key) not in (None, "", [], {})
        }
        if isinstance(compact.get("schema_signature_fields"), list):
            compact["schema_signature_fields"] = [
                str(x)[:100] for x in compact["schema_signature_fields"][:12]
            ]
        entries.append(compact)
    out = {
        key: value[key]
        for key in ("schema_version", "counts", "original_table_count", "visible_entry_count", "omitted_table_count")
        if value.get(key) not in (None, "", [], {})
    }
    if entries:
        out["entries"] = entries
    out["entry_policy"] = (
        "table_index is the complete source route; entries here are only coverage exceptions "
        "and a small required/supporting sample. Read the full ledger artifact for exact inventory."
    )
    out["visible_entry_count"] = len(entries)
    out["omitted_entry_count"] = max(0, len(source_entries) - len(entries))
    return out


def _qdi_compact_verification_queue(value: Any, *, limit: int = 6) -> list[dict[str, Any]]:
    """Project only the routing fields of deterministic verification work.

    Keep the historical list shape so prompt/report consumers do not need a
    schema migration; the final marker carries the omission count.
    """
    source = [item for item in list(value or []) if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    for item in source[: max(1, int(limit))]:
        compact = {
            key: item.get(key)
            for key in (
                "question_id",
                "category",
                "priority",
                "reason",
                "left_file",
                "left_field",
                "right_file",
                "right_field",
                "relation_type",
                "table_id",
                "source_file",
                "check_kind",
                "recommended_action",
            )
            if item.get(key) not in (None, "", [], {})
        }
        if isinstance(compact.get("reason"), str):
            compact["reason"] = compact["reason"][:320]
        if compact:
            out.append(compact)
    if len(source) > len(out):
        out.append(
            {
                "queue_policy": (
                    "Only compact routing candidates are visible; use the full "
                    "project-context artifact for omitted verification details."
                ),
                "omitted_count": len(source) - len(out),
            }
        )
    return out


def _qdi_compact_aliases(value: Any, *, limit: int) -> list[dict[str, Any]]:
    groups = [item for item in list(value or []) if isinstance(item, dict)]
    rank = {"high": 0, "medium": 1, "low": 2}
    groups.sort(
        key=lambda item: (
            rank.get(str(item.get("task_relevance") or "medium").lower(), 3),
            str(item.get("concept_id") or ""),
        )
    )
    out: list[dict[str, Any]] = []
    for group in groups[: max(1, int(limit))]:
        compact = {
            key: group.get(key)
            for key in (
                "concept_id",
                "label",
                "reason",
                "confidence",
                "task_relevance",
                "evidence_status",
                "status",
                "caution",
                "recommended_qdi_checks",
            )
            if group.get(key) not in (None, "", [], {})
        }
        candidates = []
        for item in list(group.get("candidate_fields") or [])[:12]:
            if not isinstance(item, dict):
                continue
            candidates.append(
                {
                    key: item.get(key)
                    for key in ("source_file", "source_collection", "sheet_name", "field", "alias_family", "status")
                    if item.get(key) not in (None, "", [], {})
                }
            )
        if candidates:
            compact["candidate_fields"] = candidates
        if isinstance(compact.get("reason"), str):
            compact["reason"] = compact["reason"][:520]
        if isinstance(compact.get("caution"), str):
            compact["caution"] = compact["caution"][:520]
        if compact:
            out.append(compact)
    return out


def _qdi_compact_document_manifest(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in list(value or [])[:120]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                key: item.get(key)
                for key in (
                    "document_id",
                    "source_file",
                    "suffix",
                    "pages",
                    "paragraphs",
                    "chunk_count",
                    "text_chars",
                    "requires_ocr",
                    "parse_error",
                    "full_text_local",
                )
                if item.get(key) not in (None, "", [], {})
            }
        )
    return out


def _qdi_compact_prior_memory(value: Any, *, answer_limit: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    answers: list[dict[str, Any]] = []
    for item in list(value.get("answers") or [])[: max(1, int(answer_limit))]:
        if not isinstance(item, dict):
            continue
        answers.append(
            {
                key: item.get(key)
                for key in (
                    "question_id",
                    "question",
                    "answer",
                    "evidence",
                    "confidence",
                    "remaining_uncertainty",
                    "source_status",
                    "source_inventory_status",
                    "qdi_contract_status",
                    "changed_source_files",
                )
                if item.get(key) not in (None, "", [], {})
            }
        )
    delta = value.get("source_delta") if isinstance(value.get("source_delta"), dict) else {}
    result = {
        "available": bool(value.get("available") or answers),
        "reuse_policy": _qdi_bounded_text(value.get("reuse_policy"), 700),
    }
    if answers:
        result["answers"] = answers
    if delta:
        result["source_delta"] = {
            key: [str(x)[:180] for x in list(delta.get(key) or [])[:40]]
            for key in ("unchanged_files", "new_files", "changed_files", "removed_files")
            if delta.get(key)
        }
    return {key: val for key, val in result.items() if val not in (None, "", [], {})}


def _qdi_project_cache_context(
    context: dict[str, Any],
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Project facts that remain stable across questions and QDI stages.

    The full context is retained in ``qdi_project_context_full``.  This
    projection is intentionally a small, deterministic cache prefix: all table
    identifiers remain routable, while detailed cards and long ledgers are
    available through bounded retrieval actions.
    """
    source = dict(context or {})
    table_index, table_cards = _qdi_compact_project_table_cards(
        source.get("table_cards"),
        cfg=cfg,
    )
    relation_limit = _qdi_cfg_int(cfg, "project_relation_limit", 24)
    relation_cards = _qdi_compact_relations(source.get("relations"), limit=relation_limit)
    verified_relations = _qdi_compact_relations(
        source.get("verified_relation_cards"),
        limit=relation_limit,
    )
    field_glossary = []
    for item in list(source.get("field_glossary") or [])[
        : _qdi_cfg_int(cfg, "project_field_glossary_limit", 40)
    ]:
        if not isinstance(item, dict):
            continue
        field_glossary.append(
            {
                key: item.get(key)
                for key in ("field", "meaning", "role", "files")
                if item.get(key) not in (None, "", [], {})
            }
        )
    stable: dict[str, Any] = {
        "schema_version": QDI_PROJECT_CONTEXT_LAYOUT,
        "context_policy": {
            "stable_prefix_is_immutable": True,
            "task_hint_is_dynamic": True,
            "prior_qdi_memory_is_dynamic": True,
            "prior_qdi_answers_are_hypotheses_until_current_source_check": True,
            "full_project_context_is_local_artifact": True,
            "table_index_is_route_complete": True,
            "table_cards_are_bounded_evidence": True,
            "omitted_details_are_not_negative_evidence": True,
            "request_context_is_preferred_for_one_or_two_table_details": True,
            "read_qdi_artifact_excerpt_is_preferred_for_exact_long_details": True,
            "authoritative_memory_and_constraint_memory_are_compact_views": True,
        },
        "data_root_name": _qdi_bounded_text(source.get("data_root_name"), 240),
        "authoritative_memory": _qdi_compact_authoritative_memory(source.get("authoritative_memory")),
        "constraint_memory": _qdi_compact_constraint_memory(source.get("constraint_memory")),
        "table_index": table_index,
        "table_cards": table_cards,
        "relations": relation_cards,
        "verified_relation_cards": verified_relations,
        "relation_verification_queue": _qdi_compact_verification_queue(
            source.get("relation_verification_queue"),
            limit=6,
        ),
        "population_verification_queue": _qdi_compact_verification_queue(
            source.get("population_verification_queue"),
            limit=6,
        ),
        "source_coverage_ledger": _qdi_compact_coverage(source.get("source_coverage_ledger")),
        "document_manifest": _qdi_compact_document_manifest(source.get("document_manifest")),
        "entity_alias_candidates": _qdi_compact_aliases(
            source.get("entity_alias_candidates"),
            limit=_qdi_cfg_int(cfg, "project_entity_alias_limit", 16),
        ),
        "field_glossary": field_glossary,
        "sampled_filename_patterns": list(source.get("sampled_filename_patterns") or [])[:40],
        "filename_sample_groups": list(source.get("filename_sample_groups") or [])[
            : _qdi_cfg_int(cfg, "project_filename_group_limit", 16)
        ],
        "script_investigation_policy": {
            "input_dir": "read-only",
            "scratch_dir": "temporary writable and destroyed after execution",
            "custom_python_contract": "def analyze(input_dir: str, scratch_dir: str) -> dict",
            "output_policy": "Return compact aggregate JSON; never print full tables.",
        },
        "limits": {
            key: source.get("limits", {}).get(key)
            for key in (
                "max_questions",
                "max_rounds_per_run",
                "question_bfs_max_depth",
                "max_followup_questions_per_question",
                "allow_custom_readonly_python",
            )
            if isinstance(source.get("limits"), dict)
            and source.get("limits", {}).get(key) not in (None, "")
        },
    }
    return {
        key: value
        for key, value in stable.items()
        if value not in (None, "", [], {})
    } | {"zz_qdi_cache_layout": QDI_CACHE_LAYOUT}


def _qdi_question_cache_context(
    context: dict[str, Any],
    question: dict[str, Any],
    routed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Question route only; all bounded table evidence stays in project prefix."""
    route = routed_context if isinstance(routed_context, dict) else context
    cards = route.get("table_cards") if isinstance(route.get("table_cards"), list) else []
    candidate_files = {
        str(item).replace("\\", "/")
        for item in (question.get("candidate_files") or [])
        if str(item).strip()
    }

    def card_names(card: dict[str, Any]) -> set[str]:
        return {
            str(card.get(key, "")).replace("\\", "/")
            for key in ("table_id", "source_file")
            if str(card.get(key, "")).strip()
        }

    routed_cards = [
        card for card in cards
        if isinstance(card, dict) and (not candidate_files or card_names(card) & candidate_files)
    ]
    result: dict[str, Any] = {
        "question_route": {
            "candidate_files": sorted(candidate_files),
            "candidate_table_ids": [
                str(card.get("table_id") or "")
                for card in routed_cards
                if str(card.get("table_id") or "").strip()
            ][:80],
            "candidate_source_files": sorted({
                str(card.get("source_file") or "")
                for card in routed_cards
                if str(card.get("source_file") or "").strip()
            })[:80],
            "all_table_details_are_in_project_prefix": True,
        },
        "question_context_policy": {
            "table_details": "All bounded table cards are in the stable project prefix; this segment carries route identifiers only.",
            "missing_details_are_not_negative_evidence": True,
        },
        "zz_qdi_segment": "question-route-v6",
    }
    if question:
        result["initial_question"] = {
            "question_id": str(question.get("question_id") or ""),
            "question": str(question.get("question") or ""),
            "category": str(question.get("category") or ""),
            "why_blocking": str(question.get("why_blocking") or ""),
            "candidate_files": [str(x) for x in (question.get("candidate_files") or [])],
            "parent_id": str(question.get("parent_id") or ""),
            "depth": int(question.get("depth", 0) or 0),
        }
    return result


def _qdi_stable_context_segments(
    project_context: dict[str, Any],
    question_context: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    segments = [{
        "name": "project",
        "content": json_block("QDI project context", project_context, sort_keys=True),
    }]
    lookup = dict(project_context)
    if question_context is not None:
        segments.append({
            "name": "question",
            "content": json_block("QDI question context", question_context, sort_keys=True),
        })
        lookup.update(question_context)
    return segments, lookup


def _qdi_structured_call(
    *,
    node_cache: QDINodeResultCache,
    cost_controls: dict[str, Any],
    llm_client: Any,
    model_cls: Any,
    system_prompt: str,
    user_prompt: str,
    prompt_name: str,
    fewshot: str = "",
    max_tokens: int | None = None,
    static_context_prompt: str = "",
    dynamic_user_prompt: str | None = None,
    stable_context_segments: list[dict[str, str]] | None = None,
    schema_prompt: str = "",
) -> Any:
    """Call structured LLM with a validated, versioned QDI decision cache."""
    request = {
        "cache_layout": QDI_CACHE_LAYOUT,
        "prompt_name": prompt_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "fewshot": fewshot,
        "max_tokens": max_tokens,
        "static_context_prompt": static_context_prompt,
        "dynamic_user_prompt": dynamic_user_prompt if dynamic_user_prompt is not None else user_prompt,
        "stable_context_segments": stable_context_segments or [],
        "schema_prompt": schema_prompt,
    }
    key = node_cache.key(model_cls=model_cls, llm_client=llm_client, request=request)
    cached = node_cache.get(key, model_cls)
    if cached is not None:
        cost_controls.update(node_cache.stats())
        return cached
    result = llm_client.ask_structured(
        model_cls=model_cls,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_name=prompt_name,
        fewshot=fewshot,
        max_tokens=max_tokens,
        static_context_prompt=static_context_prompt,
        dynamic_user_prompt=dynamic_user_prompt,
        stable_context_segments=stable_context_segments,
        schema_prompt=schema_prompt,
    )
    metadata = getattr(llm_client, "last_call_metadata", None)
    cacheable = bool(
        isinstance(metadata, dict)
        and metadata.get("parsed_ok")
        and (
            metadata.get("source") == "local_response_cache"
            or (
                metadata.get("source") == "provider"
                and metadata.get("finish_reason") == "stop"
            )
        )
    )
    node_cache.put(key, result, cacheable=cacheable)
    cost_controls.update(node_cache.stats())
    return result


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
    merged.pop("_full_memory_artifact", None)
    # Exact references prevent a new conclusion from erasing unrelated gaps.
    # Full resolution records remain in the existing action trajectory.
    evidence_refs = set(merged.get("evidence_refs") or []) | set(getattr(update, "evidence_refs", []) or [])
    list_fields = (
        "confirmed_facts",
        "temporary_conclusions",
        "evidence_refs",
        "open_gaps",
        "invalidated_hypotheses",
    )
    for field in list_fields:
        existing = [str(x) for x in (merged.get(field) or []) if str(x).strip()]
        incoming = [str(x) for x in (getattr(update, field, []) or []) if str(x).strip()]
        merged[field] = list(dict.fromkeys(existing + incoming))
    for resolution in getattr(update, "resolutions", []) or []:
        if (resolution.field in {"open_gaps", "confirmed_facts", "temporary_conclusions"}
                and resolution.reason.strip() and resolution.evidence_refs
                and set(resolution.evidence_refs).issubset(evidence_refs)):
            merged[resolution.field] = [item for item in merged.get(resolution.field, [])
                                        if item != resolution.text]
    focus = str(getattr(update, "recommended_next_focus", "") or "").strip()
    if focus:
        merged["recommended_next_focus"] = focus
    merged["last_updated_sequence"] = int(sequence)
    return merged


def _working_memory_prompt_view(card: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    """Project complete memory into whole, attributable entries with a readable original."""
    limit = max(1000, int(max_chars))
    serialized = json.dumps(card, ensure_ascii=False, sort_keys=True, default=str)
    if len(serialized) <= limit:
        return card
    ref = card.get("_full_memory_artifact")
    view: dict[str, Any] = {
        "question_id": card.get("question_id", ""),
        "last_updated_sequence": card.get("last_updated_sequence", 0),
        "trust_policy": card.get("trust_policy", ""),
        "truncated": True,
        "original_chars": len(serialized),
        "full_memory_artifact": _artifact_ref_index_for_prompt(ref) if isinstance(ref, dict) else {},
        "omitted_entries": {},
        "instruction": "Only included whole entries are visible; use full_memory_artifact to read omitted exact text.",
    }
    # Pending gaps and verified facts win prompt space over tentative interpretations.
    fields = (
        "open_gaps", "confirmed_facts", "invalidated_hypotheses",
        "evidence_refs", "temporary_conclusions",
    )
    for field in fields:
        items = list(card.get(field) or [])
        view[field] = []
        view["omitted_entries"][field] = len(items)
    focus = str(card.get("recommended_next_focus") or "")
    if focus:
        view["recommended_next_focus_omitted"] = True
    for field in fields:
        items = list(card.get(field) or [])
        for index in range(len(items) - 1, -1, -1):
            item = items[index]
            entry = {"index": index, "text": item}
            view[field].append(entry)
            view["omitted_entries"][field] -= 1
            if len(json.dumps(view, ensure_ascii=False, sort_keys=True, default=str)) > limit:
                view[field].pop()
                view["omitted_entries"][field] += 1
    if focus:
        view["recommended_next_focus"] = focus
        view.pop("recommended_next_focus_omitted", None)
        if len(json.dumps(view, ensure_ascii=False, sort_keys=True, default=str)) > limit:
            view.pop("recommended_next_focus")
            view["recommended_next_focus_omitted"] = True
    return view


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
    view["artifact_ref"] = _artifact_ref_index_for_prompt(ref)
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
    if action_name == "request_checks":
        request = {"checks": [check.model_dump() for check in action.request_checks]}
    elif action_name == "request_script":
        request = action.request_script.model_dump()
        code = str(request.get("python_code") or "")
        request["python_code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
        if code:
            ref = artifact_store.put(
                "qdi_script_source_full",
                f"{question_id}:action:{sequence}",
                request,
                visible_excerpt=code[: max(1, int(script_chars))],
                visible_limit=script_chars,
            )
            request["python_code"] = code[: max(1, int(script_chars))]
            request["python_code_truncated"] = len(code) > max(1, int(script_chars))
            request["python_code_original_chars"] = len(code)
            request["python_code_visible_chars"] = len(request["python_code"])
            request["source_artifact"] = _artifact_ref_index_for_prompt(ref)
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
    return request_signature(req)


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
    view = _visible_output_payload(value, max_chars)
    if not view["output_truncated"]:
        return _json_safe(value)
    return json.loads(view["current_visible_output"])


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
    # A slash alone is also a common missing-value marker. Actual I/O remains
    # guarded by resolved input/scratch paths in the execution wrapper.
    if not text or text == "/":
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
