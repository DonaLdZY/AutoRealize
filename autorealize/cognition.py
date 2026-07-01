from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import AutoRealizeConfig
from .llm.client import LLMClient
from .logging_utils import log_event
from .models import CognitionProbePlan, CognitionSummary, FileRole, FileSummary
from .prompt_cache import stable_dynamic_prompt
from .profiling.stats import column_profile_to_dict, profile_dataframe, read_table, table_probe_sample_rows
from .prompts.manager import PromptManager

logger = logging.getLogger(__name__)


def llm_cognition_for_file(
    cfg: AutoRealizeConfig,
    llm: LLMClient,
    prompt_mgr: PromptManager,
    file_path: Path,
    relative_path: str,
    parsed_kind: str,
    parsed_text_summary: str,
    parsed_columns: list[str],
    parsed_preview: list[dict[str, Any]],
    task_hint: str,
    source_metadata: dict[str, Any] | None = None,
    column_profiles: list[dict[str, Any]] | None = None,
    heuristic_field_semantics: dict[str, str] | None = None,
) -> FileSummary | None:
    document_like = parsed_kind in {"document", "structured_document", "archive"}
    document_excerpt = parsed_text_summary[:18000] if document_like else parsed_text_summary[:4000]
    deterministic_context = _compact_deterministic_file_context(
        source_metadata=source_metadata or {},
        column_profiles=column_profiles or [],
        heuristic_field_semantics=heuristic_field_semantics or {},
        preview_rows=getattr(cfg.data, "preview_rows", 10),
    )
    base_context = {
        "file": relative_path,
        "kind": parsed_kind,
        "columns": parsed_columns[:80],
        "preview": parsed_preview[: max(1, int(getattr(cfg.data, "preview_rows", 10)))],
        "summary": parsed_text_summary[:4000],
        "document_excerpt": document_excerpt,
        "document_excerpt_chars": len(document_excerpt),
        "source_text_chars": len(parsed_text_summary or ""),
        "deterministic_profile": deterministic_context,
        "task": task_hint,
    }

    probe_plan = CognitionProbePlan(
        need_more_probe=False,
        probe_actions=[],
        action_specs=[],
        focus_columns=[str(x) for x in parsed_columns[:8]],
        hypotheses=[],
        reason="Single-file LLM cognition consumes deterministic preview/statistics only; no probe tools are called in this stage.",
    )
    log_event(
        logger,
        "agent.file_cognition_probe",
        "SKIPPED",
        file=relative_path,
        kind=parsed_kind,
        need_more_probe=bool(probe_plan.need_more_probe),
        action_count=_probe_plan_action_count(probe_plan),
        focus_columns=[str(x) for x in (probe_plan.focus_columns or [])[:8]],
        reason=str(probe_plan.reason or "")[:240],
    )
    probe_results: dict[str, Any] = {}

    log_event(logger, "agent.file_cognition_summary", "ACTIVATED", file=relative_path)
    try:
        summary = _summarize_file(cfg, llm, prompt_mgr, base_context, probe_plan, probe_results)
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "agent.file_cognition_summary", "FAILED", file=relative_path, error=str(exc)[:240])
        raise
    log_event(
        logger,
        "agent.file_cognition_summary",
        "COMPLETED",
        file=relative_path,
        role=summary.file_role_guess,
        key_columns=len(summary.key_columns or []),
        field_descriptions=len(summary.field_descriptions or {}),
        sheet_field_descriptions=sum(len(v or {}) for v in (summary.sheet_field_descriptions or {}).values()),
    )
    role = _map_role(summary.file_role_guess)
    sheet_semantics = {
        str(sheet): {str(k): str(v) for k, v in (fields or {}).items() if str(k).strip() and str(v).strip()}
        for sheet, fields in (summary.sheet_field_descriptions or {}).items()
        if str(sheet).strip() and isinstance(fields, dict)
    }
    return FileSummary(
        path=relative_path,
        role=role,
        summary=summary.concise_summary,
        detailed_report=summary.detailed_report,
        columns=summary.key_columns or parsed_columns[:30],
        extracted_knowledge=(summary.key_facts or [])[:40],
        warnings=summary.risks,
        related_files=summary.related_hints,
        column_semantics={str(k): str(v) for k, v in (summary.field_descriptions or {}).items() if str(k).strip() and str(v).strip()},
        column_semantic_meta={
            str(k): {"confidence": "medium", "confidence_score": 0.7, "source": "llm_field_description"}
            for k, v in (summary.field_descriptions or {}).items()
            if str(k).strip() and str(v).strip()
        },
        source_metadata={
            "parsed_kind": parsed_kind,
            "sheet_field_descriptions": sheet_semantics,
            "probe_plan": probe_plan.model_dump(),
            "probe_result_keys": list(probe_results.keys()) if isinstance(probe_results, dict) else [],
            "probe_results": _compact_probe_results(probe_results),
            "cognition_trace": _build_cognition_trace(
                file=relative_path,
                parsed_kind=parsed_kind,
                probe_plan=probe_plan,
                probe_results=probe_results,
                summary=summary,
            ),
        },
    )


def _compact_deterministic_file_context(
    *,
    source_metadata: dict[str, Any],
    column_profiles: list[dict[str, Any]],
    heuristic_field_semantics: dict[str, str],
    preview_rows: int,
) -> dict[str, Any]:
    """Compact parser/statistics output for one-shot LLM field understanding."""
    meta = source_metadata if isinstance(source_metadata, dict) else {}
    out: dict[str, Any] = {
        "profile_source": "deterministic_parser_and_statistics",
        "llm_task": (
            "Use the preview rows, column profiles, sheet metadata, and heuristic field semantics "
            "to write concise file cognition and human-readable field_descriptions. "
            "Do not request tools or invent facts not supported by these statistics."
        ),
        "shape": meta.get("shape"),
        "shape_estimated": meta.get("shape_estimated"),
        "preview_rows_used": meta.get("preview_rows_used"),
        "profile_sampling": meta.get("profile_sampling"),
        "csv_dialect": meta.get("csv_dialect"),
        "csv_encoding": meta.get("csv_encoding"),
        "json_strategy": meta.get("json_strategy"),
        "json_root_type": meta.get("json_root_type"),
        "json_first_level_schema": meta.get("json_first_level_schema"),
        "heuristic_field_semantics": {
            str(k): str(v)[:240]
            for k, v in (heuristic_field_semantics or {}).items()
            if str(k).strip() and str(v).strip()
        },
        "column_profiles": _compact_column_profiles_for_llm(column_profiles, limit=80),
    }
    dtypes = meta.get("dtypes") if isinstance(meta.get("dtypes"), dict) else {}
    if dtypes:
        out["dtypes"] = {str(k): str(v) for k, v in list(dtypes.items())[:80]}
    preview = meta.get("preview") if isinstance(meta.get("preview"), list) else []
    if preview:
        out["parser_preview"] = _compact_preview_records(preview, max_rows=max(1, int(preview_rows)), max_cols=20)
    sheet_names = meta.get("excel_sheet_names") if isinstance(meta.get("excel_sheet_names"), list) else []
    if sheet_names:
        out["excel_sheet_names"] = [str(x) for x in sheet_names[:30]]
        out["excel_default_sheet"] = meta.get("excel_default_sheet")
    sheet_groups = meta.get("excel_sheet_groups") if isinstance(meta.get("excel_sheet_groups"), list) else []
    if sheet_groups:
        out["excel_sheet_groups"] = [
            {
                "group_id": item.get("group_id", ""),
                "sheet_name_pattern": item.get("sheet_name_pattern", ""),
                "header_signature": item.get("header_signature", ""),
                "representative": item.get("representative", ""),
                "sheet_count": item.get("sheet_count", 0),
                "sheets": [str(x) for x in (item.get("sheets") or [])[:12]],
            }
            for item in sheet_groups[:10]
            if isinstance(item, dict)
        ]
    sheet_profiles = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
    if sheet_profiles:
        out["excel_sheet_profiles"] = _compact_excel_sheet_profiles_for_llm(
            sheet_profiles,
            preview_rows=max(1, int(preview_rows)),
        )
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def _compact_column_profiles_for_llm(profiles: list[dict[str, Any]], *, limit: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles[:limit]:
        if not isinstance(profile, dict):
            continue
        item = {
            "name": str(profile.get("name", "")),
            "dtype": profile.get("dtype"),
            "logical_type": profile.get("logical_type"),
            "row_count": profile.get("row_count"),
            "null_ratio": profile.get("null_ratio"),
            "unique_count": profile.get("unique_count"),
            "numeric_parse_ratio": profile.get("numeric_parse_ratio"),
            "datetime_parse_ratio": profile.get("datetime_parse_ratio"),
            "format_hints": [str(x) for x in (profile.get("format_hints") or [])[:6]],
            "value_pattern_hints": [str(x) for x in (profile.get("value_pattern_hints") or [])[:6]],
            "sample_values": [str(x)[:120] for x in (profile.get("sample_values") or [])[:5]],
            "top_values": [str(x)[:120] for x in (profile.get("top_values") or [])[:10]],
            "abnormal_tokens": [str(x)[:120] for x in (profile.get("abnormal_tokens") or [])[:6]],
        }
        numeric_stats = profile.get("numeric_stats") if isinstance(profile.get("numeric_stats"), dict) else {}
        if numeric_stats:
            item["numeric_stats"] = {
                k: _json_safe_value(numeric_stats.get(k))
                for k in ["min", "max", "mean", "std", "negative_ratio", "zero_ratio"]
                if k in numeric_stats
            }
        quantiles = profile.get("quantiles") if isinstance(profile.get("quantiles"), dict) else {}
        if quantiles:
            item["quantiles"] = {str(k): _json_safe_value(v) for k, v in quantiles.items()}
        datetime_stats = profile.get("datetime_stats") if isinstance(profile.get("datetime_stats"), dict) else {}
        if datetime_stats:
            item["datetime_stats"] = {
                k: _json_safe_value(datetime_stats.get(k))
                for k in ["min", "max", "granularity", "unique_timestamps"]
                if k in datetime_stats
            }
        rows.append({k: v for k, v in item.items() if v not in (None, "", [], {})})
    return rows


def _compact_excel_sheet_profiles_for_llm(
    sheet_profiles: list[dict[str, Any]],
    *,
    max_sheets: int = 80,
    preview_rows: int = 10,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sheet in sheet_profiles[:max_sheets]:
        if not isinstance(sheet, dict):
            continue
        entry = {
            "sheet_name": sheet.get("sheet_name", ""),
            "shape": sheet.get("shape"),
            "shape_profiled": sheet.get("shape_profiled", sheet.get("shape_sampled")),
            "preview_rows_used": sheet.get("preview_rows_used"),
            "columns": [str(x) for x in (sheet.get("columns") or [])[:30]],
            "layout_kind": sheet.get("layout_kind"),
            "header_confidence": sheet.get("header_confidence"),
            "detected_header_row": sheet.get("detected_header_row"),
            "read_strategy_kind": sheet.get("read_strategy_kind"),
            "recommended_read": sheet.get("recommended_read"),
            "reading_risks": [str(x) for x in (sheet.get("reading_risks") or [])[:5]],
            "is_deep_profiled": sheet.get("is_deep_profiled"),
            "profile_policy": sheet.get("profile_policy"),
            "sheet_group_id": sheet.get("sheet_group_id"),
            "sheet_group_size": sheet.get("sheet_group_size"),
            "sheet_group_representative": sheet.get("sheet_group_representative"),
            "raw_preview_note": "header=None top-left cells; preserves opening notes that pandas may treat as headers",
            "raw_preview": _compact_raw_preview_rows(sheet.get("raw_preview", []), max_rows=preview_rows, max_cols=16),
            "preview": _compact_preview_records(sheet.get("preview", []), max_rows=preview_rows, max_cols=16),
            "column_profiles": _compact_column_profiles_for_llm(sheet.get("column_profiles", []), limit=24),
        }
        out.append({k: v for k, v in entry.items() if v not in (None, "", [], {})})
    return out


def _compact_preview_records(records: Any, *, max_rows: int, max_cols: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(records, list):
        return rows
    for rec in records[:max_rows]:
        if not isinstance(rec, dict):
            continue
        row: dict[str, Any] = {}
        for idx, (key, value) in enumerate(rec.items()):
            if idx >= max_cols:
                row["_truncated_columns"] = max(0, len(rec) - idx)
                break
            row[str(key)] = str(_json_safe_value(value))[:160] if value is not None else None
        rows.append(row)
    return rows


def _compact_raw_preview_rows(records: Any, *, max_rows: int, max_cols: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    if not isinstance(records, list):
        return rows
    for raw in records[:max_rows]:
        if not isinstance(raw, list):
            continue
        row = [
            str(_json_safe_value(value))[:160] if value is not None else None
            for value in raw[:max_cols]
        ]
        if len(raw) > max_cols:
            row.append(f"... {len(raw) - max_cols} more cells")
        rows.append(row)
    return rows


def _plan_probe(
    cfg: AutoRealizeConfig,
    llm: LLMClient,
    prompt_mgr: PromptManager,
    base_context: dict[str, Any],
) -> CognitionProbePlan:
    system = prompt_mgr.load("system/cognition_probe_planner.md")
    try:
        stable, dynamic = stable_dynamic_prompt(
            stable=base_context,
            dynamic={"instruction": "Plan the minimum useful probe actions for this file."},
            stable_title="Stable file context",
            dynamic_title="Dynamic probe request",
        )
        return llm.ask_structured(
            model_cls=CognitionProbePlan,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name="cognition_probe_plan",
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM probe planning failed: {exc}") from exc


def _plan_probe_with_error_feedback(
    cfg: AutoRealizeConfig,
    llm: LLMClient,
    prompt_mgr: PromptManager,
    base_context: dict[str, Any],
    previous_plan: CognitionProbePlan,
    probe_error: str,
    attempt: int,
) -> CognitionProbePlan:
    system = prompt_mgr.load("system/cognition_probe_planner.md")
    stable, dynamic = stable_dynamic_prompt(
        stable=base_context,
        dynamic={
            "previous_probe_plan": previous_plan.model_dump(),
            "probe_execution_error": probe_error,
            "retry_attempt": attempt,
            "instruction": "Revise the probe plan to avoid missing columns and unsupported actions.",
        },
        stable_title="Stable file context",
        dynamic_title="Dynamic retry feedback",
    )
    return llm.ask_structured(
        model_cls=CognitionProbePlan,
        system_prompt=system,
        user_prompt=dynamic,
        prompt_name="cognition_probe_plan_retry",
        static_context_prompt=stable,
        dynamic_user_prompt=dynamic,
    )


def _json_safe_value(v: Any) -> Any:
    try:
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        if isinstance(v, pd.Timedelta):
            return str(v)
        if pd.isna(v):
            return None
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def _json_safe_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _json_safe_value(v) for k, v in rec.items()}


def _compact_probe_results(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return _json_safe_value(str(value)[:240])
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= 40:
                out["_truncated_keys"] = len(value) - idx
                break
            out[str(k)] = _compact_probe_results(v, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_compact_probe_results(x, depth=depth + 1) for x in value[:40]]
    return _json_safe_value(value)


def _probe_plan_action_count(plan: CognitionProbePlan) -> int:
    base_actions = list(dict.fromkeys([str(x) for x in (plan.probe_actions or []) if str(x).strip()]))
    specs = [x for x in getattr(plan, "action_specs", []) if isinstance(x, dict)]
    return len(base_actions) + len(specs[:8])


def _probe_result_status(value: Any) -> str:
    if isinstance(value, dict) and value.get("error"):
        return "failed"
    return "completed"


def _summarize_probe_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        if isinstance(value, list):
            profile_summary = _summarize_profile_rows(value)
            if profile_summary:
                return {"items": len(value), "profiles": profile_summary}
            return {"items": len(value), "sample": _compact_probe_results(value[:3])}
        return {"value": _compact_probe_results(value)}

    out: dict[str, Any] = {}
    for key in ["rows_total", "rows_matched", "ratio", "column", "valid_datetime_count", "groups_checked", "violation_groups"]:
        if key in value:
            out[key] = _json_safe_value(value[key])
    if value.get("error"):
        out["error"] = str(value.get("error"))[:240]
    if value.get("warnings"):
        out["warnings"] = [str(x)[:180] for x in list(value.get("warnings") or [])[:5]]
    if value.get("columns"):
        out["columns"] = [str(x) for x in list(value.get("columns") or [])[:12]]
    if value.get("group_by"):
        out["group_by"] = [str(x) for x in list(value.get("group_by") or [])[:6]]

    profile_summary = _summarize_profile_rows(value.get("profiles"))
    if profile_summary:
        out["profiles"] = profile_summary

    if isinstance(value.get("value_counts"), dict):
        vc_summary: dict[str, Any] = {}
        for col, rows in list(value["value_counts"].items())[:6]:
            if isinstance(rows, list):
                vc_summary[str(col)] = [
                    {
                        "value": str(row.get("value", ""))[:80],
                        "count": row.get("count"),
                        "ratio": row.get("ratio"),
                    }
                    for row in rows[:5]
                    if isinstance(row, dict)
                ]
        out["value_counts"] = vc_summary

    if isinstance(value.get("numeric_summary"), list):
        out["numeric_summary"] = [
            {
                "column": row.get("column"),
                "valid_count": row.get("valid_count"),
                "mean": row.get("mean"),
                "std": row.get("std"),
                "min": row.get("min"),
                "max": row.get("max"),
                "zero_ratio": row.get("zero_ratio"),
                "negative_ratio": row.get("negative_ratio"),
            }
            for row in value["numeric_summary"][:8]
            if isinstance(row, dict)
        ]

    if isinstance(value.get("preview"), list):
        out["preview_rows"] = len(value["preview"])
        out["preview_sample"] = _compact_probe_results(value["preview"][:3])
    if isinstance(value.get("rows"), list):
        out["result_rows"] = len(value["rows"])
        out["result_sample"] = _compact_probe_results(value["rows"][:3])
    return out or _compact_probe_results(value)


def _summarize_profile_rows(value: Any) -> list[dict[str, Any]]:
    """Summarize column profile rows without passing large raw profiler JSON."""
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in value[:16]:
        if not isinstance(row, dict):
            continue
        item: dict[str, Any] = {}
        for key in [
            "name",
            "logical_type",
            "dtype",
            "row_count",
            "non_null_count",
            "null_ratio",
            "unique_count",
            "numeric_parse_ratio",
            "datetime_parse_ratio",
        ]:
            if key in row:
                item[key] = _json_safe_value(row.get(key))
        if isinstance(row.get("top_values"), list):
            item["top_values"] = [str(x)[:80] for x in row.get("top_values", [])[:5]]
        if isinstance(row.get("sample_values"), list):
            item["sample_values"] = [str(x)[:80] for x in row.get("sample_values", [])[:5]]
        if isinstance(row.get("value_pattern_hints"), list):
            item["value_pattern_hints"] = [str(x)[:80] for x in row.get("value_pattern_hints", [])[:4]]
        if item:
            rows.append(item)
    return rows


def _compact_probe_observations(probe_results: dict[str, Any]) -> dict[str, Any]:
    """Build a complete, compact observation object for the LLM summarizer."""
    if not isinstance(probe_results, dict):
        return {"result": _compact_probe_results(probe_results)}
    out: dict[str, Any] = {}
    for idx, (key, value) in enumerate(probe_results.items()):
        if idx >= 18:
            out["_truncated_result_keys"] = len(probe_results) - idx
            break
        out[str(key)] = _summarize_probe_result(value)
    return out


def _result_for_action_spec(probe_results: dict[str, Any], idx: int, action: str) -> tuple[str, Any]:
    prefix = f"action_spec_{idx}_"
    for key, value in probe_results.items():
        if str(key).startswith(prefix):
            return str(key), value
    key = f"action_spec_{idx}_{action or 'unknown'}"
    return key, None


def _append_trace_action(
    actions: list[dict[str, Any]],
    *,
    result_key: str,
    action: str,
    reason: str = "",
    columns: list[Any] | None = None,
    conditions: list[Any] | None = None,
    result: Any = None,
) -> None:
    item: dict[str, Any] = {
        "result_key": result_key,
        "action": action,
        "reason": reason[:300],
        "columns": [str(x) for x in (columns or [])[:16]],
        "status": "pending" if result is None else _probe_result_status(result),
    }
    if conditions:
        item["conditions"] = _compact_probe_results(conditions[:6])
    if result is not None:
        item["result_summary"] = _summarize_probe_result(result)
    actions.append(item)


def _build_cognition_trace(
    *,
    file: str,
    parsed_kind: str,
    probe_plan: CognitionProbePlan,
    probe_results: dict[str, Any],
    summary: CognitionSummary,
) -> dict[str, Any]:
    result_map = probe_results if isinstance(probe_results, dict) else {}
    actions: list[dict[str, Any]] = []
    probe_actions = {str(x) for x in (probe_plan.probe_actions or []) if str(x).strip()}
    focus = [str(x) for x in (probe_plan.focus_columns or [])[:8]]

    if "preview_head" in probe_actions:
        _append_trace_action(
            actions,
            result_key="preview_head",
            action="preview_head",
            reason="查看文件开头样例，确认字段与真实取值是否支持当前理解。",
            columns=focus,
            result=result_map.get("preview_head"),
        )
    if any(x in probe_actions for x in ["profile_numeric", "profile_categorical", "check_nulls", "check_inf"]):
        _append_trace_action(
            actions,
            result_key="profiles",
            action="profile_columns",
            reason="检查关键字段的数据类型、缺失、枚举样例、异常 token 和取值模式。",
            columns=focus,
            result=result_map.get("profiles"),
        )
    if "value_counts_topk" in probe_actions:
        _append_trace_action(
            actions,
            result_key="value_counts_topk",
            action="value_counts_topk",
            reason="查看关键类别字段的高频取值，避免只凭列名猜测。",
            columns=focus,
            result=result_map.get("value_counts_topk"),
        )
    if "numeric_summary" in probe_actions:
        _append_trace_action(
            actions,
            result_key="numeric_summary",
            action="numeric_summary",
            reason="查看关键数值字段的统计特征、分位数和异常比例。",
            columns=focus,
            result=result_map.get("numeric_summary"),
        )

    specs = [x for x in getattr(probe_plan, "action_specs", []) if isinstance(x, dict)]
    for idx, spec in enumerate(specs[:8]):
        action = str(spec.get("action", "unknown"))
        key, value = _result_for_action_spec(result_map, idx, action)
        _append_trace_action(
            actions,
            result_key=key,
            action=action,
            reason=str(spec.get("reason", "")),
            columns=list(spec.get("columns") or []),
            conditions=[x for x in (spec.get("conditions") or []) if isinstance(x, dict)],
            result=value,
        )

    return {
        "schema_version": "autorealize.cognition_trace.v1",
        "file": file,
        "kind": parsed_kind,
        "probe_needed": bool(probe_plan.need_more_probe),
        "planning_reason": str(probe_plan.reason or ""),
        "focus_columns": focus,
        "questions_to_check": [str(x) for x in (probe_plan.hypotheses or [])[:6]],
        "actions": actions,
        "result_keys": list(result_map.keys())[:40],
        "summary_status": "generated",
        "summary_role": summary.file_role_guess,
        "summary_key_columns": [str(x) for x in (summary.key_columns or [])[:30]],
        "summary_field_description_count": len(summary.field_descriptions or {}),
        "summary_detailed_report_chars": len(summary.detailed_report or ""),
    }


def _existing_columns(df: pd.DataFrame, cols: list[Any], limit: int = 12) -> list[str]:
    out: list[str] = []
    for c in cols[:limit]:
        name = str(c)
        if name in df.columns and name not in out:
            out.append(name)
    return out


def _apply_probe_conditions(df: pd.DataFrame, conditions: list[dict[str, Any]]) -> tuple[pd.Series, list[str]]:
    mask = pd.Series(True, index=df.index)
    warnings: list[str] = []
    for cond in conditions[:8]:
        col = str(cond.get("column", ""))
        op = str(cond.get("op", "eq")).lower()
        value = cond.get("value")
        if col not in df.columns:
            warnings.append(f"condition_column_not_found:{col}")
            continue
        s = df[col]
        try:
            if op == "eq":
                cur = s.astype(str).eq(str(value)) if s.dtype == object else s.eq(value)
            elif op == "ne":
                cur = s.astype(str).ne(str(value)) if s.dtype == object else s.ne(value)
            elif op in {"gt", "ge", "lt", "le"}:
                numeric = pd.to_numeric(s, errors="coerce")
                try:
                    cmp_value = float(value)
                except Exception:
                    warnings.append(f"condition_value_not_numeric:{col}")
                    continue
                if op == "gt":
                    cur = numeric.gt(cmp_value)
                elif op == "ge":
                    cur = numeric.ge(cmp_value)
                elif op == "lt":
                    cur = numeric.lt(cmp_value)
                else:
                    cur = numeric.le(cmp_value)
            elif op == "contains":
                cur = s.astype(str).str.contains(str(value), na=False, regex=False)
            elif op == "in":
                values = value if isinstance(value, list) else [value]
                values = {str(x) for x in values}
                cur = s.astype(str).isin(values)
            elif op == "is_null":
                cur = s.isna()
            elif op == "not_null":
                cur = s.notna()
            else:
                warnings.append(f"unsupported_condition_op:{op}")
                continue
            mask &= cur.fillna(False)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"condition_failed:{col}:{op}:{str(exc)[:80]}")
    return mask, warnings


def _safe_numeric_summary(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in columns[:12]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            rows.append({"column": col, "valid_count": 0, "warning": "not_numeric_or_all_null"})
            continue
        rows.append(
            {
                "column": col,
                "valid_count": int(valid.shape[0]),
                "mean": _json_safe_value(valid.mean()),
                "std": _json_safe_value(valid.std()),
                "var": _json_safe_value(valid.var()),
                "min": _json_safe_value(valid.min()),
                "p05": _json_safe_value(valid.quantile(0.05)),
                "q1": _json_safe_value(valid.quantile(0.25)),
                "median": _json_safe_value(valid.median()),
                "q3": _json_safe_value(valid.quantile(0.75)),
                "p95": _json_safe_value(valid.quantile(0.95)),
                "max": _json_safe_value(valid.max()),
                "zero_ratio": round(float((valid == 0).mean()), 6),
                "negative_ratio": round(float((valid < 0).mean()), 6),
            }
        )
    return rows


def _execute_action_spec(df: pd.DataFrame, spec: dict[str, Any], default_focus: list[str], cfg: AutoRealizeConfig) -> dict[str, Any]:
    action = str(spec.get("action", "")).strip()
    reason = str(spec.get("reason", "")).strip()
    limit = max(1, min(50, int(spec.get("limit") or 10)))
    columns = _existing_columns(df, [*list(spec.get("columns") or []), *default_focus], limit=16)
    conditions = [x for x in (spec.get("conditions") or []) if isinstance(x, dict)]
    mask, condition_warnings = _apply_probe_conditions(df, conditions)
    scoped = df.loc[mask]
    out: dict[str, Any] = {
        "action": action,
        "reason": reason,
        "rows_total": int(len(df)),
        "rows_matched": int(len(scoped)),
    }
    if condition_warnings:
        out["warnings"] = condition_warnings

    if action == "condition_ratio":
        out["ratio"] = round(float(len(scoped) / max(1, len(df))), 6)
        return out

    if action in {"profile_numeric", "profile_categorical"}:
        use_cols = columns or [str(c) for c in scoped.columns[:8]]
        prof = profile_dataframe(scoped[use_cols], top_k=cfg.data.category_top_k)
        out["columns"] = use_cols
        out["profiles"] = [
            {
                **column_profile_to_dict(p),
                "sample_values": p.sample_values[:10],
                "top_values": p.top_values[:8],
                "value_pattern_hints": p.value_pattern_hints[:6],
                "abnormal_tokens": p.abnormal_tokens[:8],
            }
            for p in prof
        ]
        return out

    if action == "check_nulls":
        use_cols = columns or [str(c) for c in scoped.columns[:8]]
        out["columns"] = use_cols
        out["nulls"] = [
            {
                "column": col,
                "null_count": int(scoped[col].isna().sum()),
                "null_ratio": round(float(scoped[col].isna().mean()), 6) if len(scoped) else 0.0,
            }
            for col in use_cols
            if col in scoped.columns
        ]
        return out

    if action == "check_inf":
        use_cols = columns or [str(c) for c in scoped.columns[:8]]
        rows = []
        for col in use_cols:
            if col not in scoped.columns:
                continue
            numeric = pd.to_numeric(scoped[col], errors="coerce")
            inf_mask = numeric.isin([float("inf"), float("-inf")])
            rows.append(
                {
                    "column": col,
                    "inf_count": int(inf_mask.sum()),
                    "inf_ratio": round(float(inf_mask.mean()), 6) if len(numeric) else 0.0,
                }
            )
        out["columns"] = use_cols
        out["inf_values"] = rows
        return out

    if action == "filter_preview":
        use_cols = columns or [str(c) for c in df.columns[:8]]
        out["columns"] = use_cols[:12]
        out["preview"] = [_json_safe_record(r) for r in scoped[use_cols[:12]].head(limit).to_dict(orient="records")]
        return out

    if action == "value_counts_topk":
        rows = {}
        for col in columns[:8]:
            vc = scoped[col].astype(str).value_counts(dropna=False).head(limit)
            rows[col] = [{"value": str(k), "count": int(v), "ratio": round(float(v / max(1, len(scoped))), 6)} for k, v in vc.items()]
        out["value_counts"] = rows
        return out

    if action == "numeric_summary":
        use_cols = columns or [str(c) for c in scoped.select_dtypes(include="number").columns[:8]]
        out["numeric_summary"] = _safe_numeric_summary(scoped, use_cols)
        return out

    if action == "groupby_agg":
        group_by = _existing_columns(df, list(spec.get("group_by") or []), limit=4)
        aggs = [x for x in (spec.get("aggregations") or []) if isinstance(x, dict)]
        if not group_by:
            out["error"] = "missing_valid_group_by"
            return out
        if not aggs:
            out["error"] = "missing_aggregations"
            return out
        named_aggs = {}
        for idx, agg in enumerate(aggs[:8]):
            col = str(agg.get("column", ""))
            fn = str(agg.get("agg", "count")).lower()
            if col not in scoped.columns:
                continue
            if fn not in {"count", "nunique", "mean", "sum", "std", "min", "max"}:
                continue
            named_aggs[f"{col}__{fn}__{idx}"] = (col, fn)
        if not named_aggs:
            out["error"] = "no_valid_aggregations"
            return out
        grouped = scoped.groupby(group_by, dropna=False).agg(**named_aggs).reset_index()
        out["group_by"] = group_by
        out["aggregations"] = list(named_aggs.keys())
        out["rows"] = [_json_safe_record(r) for r in grouped.head(limit).to_dict(orient="records")]
        return out

    if action == "time_granularity":
        col = columns[0] if columns else ""
        if not col:
            out["error"] = "missing_time_column"
            return out
        dt = pd.to_datetime(scoped[col], errors="coerce").dropna().sort_values()
        out["column"] = col
        out["valid_datetime_count"] = int(dt.shape[0])
        if dt.empty:
            return out
        deltas = dt.diff().dropna().dt.total_seconds()
        out["min"] = _json_safe_value(dt.iloc[0])
        out["max"] = _json_safe_value(dt.iloc[-1])
        out["range_days"] = round(float((dt.iloc[-1] - dt.iloc[0]).total_seconds() / 86400), 6)
        out["top_delta_seconds"] = [
            {"seconds": float(k), "count": int(v)}
            for k, v in deltas.value_counts().head(limit).items()
        ]
        return out

    if action == "uniqueness":
        use_cols = columns or [str(c) for c in df.columns[:8]]
        subset = scoped[use_cols]
        out["columns"] = use_cols
        out["unique_count"] = int(subset.drop_duplicates().shape[0])
        out["duplicate_count"] = int(max(0, len(subset) - subset.drop_duplicates().shape[0]))
        out["duplicate_ratio"] = round(float(out["duplicate_count"] / max(1, len(subset))), 6)
        return out

    if action == "functional_dependency":
        determinant = columns[:4]
        dependent = str(spec.get("dependent_column", ""))
        if dependent not in df.columns or not determinant:
            out["error"] = "missing_determinant_or_dependent_column"
            return out
        nunique = scoped.groupby(determinant, dropna=False)[dependent].nunique(dropna=False)
        violations = nunique[nunique > 1]
        out["determinant_columns"] = determinant
        out["dependent_column"] = dependent
        out["groups_checked"] = int(nunique.shape[0])
        out["violation_groups"] = int(violations.shape[0])
        out["violation_ratio"] = round(float(violations.shape[0] / max(1, nunique.shape[0])), 6)
        if not violations.empty:
            bad_keys = violations.head(limit).index.tolist()
            out["violation_examples"] = [str(x) for x in bad_keys]
        return out

    return {"action": action, "reason": reason, "error": "unsupported_action"}


def _execute_probe_actions(
    file_path: Path,
    plan: CognitionProbePlan,
    cfg: AutoRealizeConfig,
    *,
    relative_path: str = "",
    attempt: int = 1,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    suffix = file_path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".json"}:
        return result
    try:
        df = read_table(
            file_path,
            json_flatten_sep=cfg.data.json_flatten_sep,
            json_flatten_max_level=cfg.data.json_flatten_max_level,
            json_keep_raw_nested_columns=cfg.data.json_keep_raw_nested_columns,
            max_rows=table_probe_sample_rows(
                file_path,
                configured_rows=cfg.data.table_profile_sample_rows,
                large_threshold_bytes=cfg.data.large_table_threshold_bytes,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "agent.file_cognition_probe", "FAILED", file=relative_path or str(file_path), attempt=attempt, error=str(exc)[:240])
        return {"error": str(exc)}

    requested_focus = [str(x) for x in (plan.focus_columns or [])][:8]
    if requested_focus:
        existing = [c for c in requested_focus if c in df.columns]
        missing = [c for c in requested_focus if c not in df.columns]
        focus = existing if existing else [str(c) for c in df.columns[:8]]
        if missing:
            result["probe_warnings"] = [f"focus_columns_not_found: {missing}"]
            logger.warning("[P1-LLM] 探查列不存在，自动降级: file=%s missing=%s", file_path, missing)
    else:
        focus = [str(c) for c in df.columns[:8]]

    def _do_preview() -> tuple[str, Any]:
        rows = []
        for rec in df.head(cfg.data.preview_rows).to_dict(orient="records"):
            rows.append(_json_safe_record(rec))
        return "preview_head", rows

    def _do_profiles() -> tuple[str, Any]:
        use_cols = [c for c in focus if c in df.columns]
        if not use_cols:
            return "profiles", []
        prof = profile_dataframe(df[use_cols], top_k=cfg.data.category_top_k)
        return (
            "profiles",
            [
                {
                    **column_profile_to_dict(p),
                    "sample_values": p.sample_values[:10],
                    "top_values": p.top_values[:8],
                    "value_pattern_hints": p.value_pattern_hints[:6],
                    "abnormal_tokens": p.abnormal_tokens,
                }
                for p in prof
            ],
        )

    ProbeJob = tuple[str, str, str, Callable[[], tuple[str, Any]]]
    jobs: list[ProbeJob] = []
    if "preview_head" in plan.probe_actions:
        jobs.append(("preview_head", "preview_head", "查看文件开头样例", _do_preview))
    if any(x in plan.probe_actions for x in ["profile_numeric", "check_nulls", "check_inf", "profile_categorical", "value_counts_topk"]):
        jobs.append(("profiles", "profile_columns", "检查字段类型、缺失和异常值", _do_profiles))
    if "value_counts_topk" in plan.probe_actions:
        jobs.append(
            (
                "value_counts_topk",
                "value_counts_topk",
                "查看关键类别字段的高频取值",
                lambda: ("value_counts_topk", _execute_action_spec(df, {"action": "value_counts_topk", "columns": focus, "limit": cfg.data.category_top_k}, focus, cfg)),
            )
        )
    if "numeric_summary" in plan.probe_actions:
        jobs.append(
            (
                "numeric_summary",
                "numeric_summary",
                "查看关键数值字段的统计特征",
                lambda: ("numeric_summary", _execute_action_spec(df, {"action": "numeric_summary", "columns": focus}, focus, cfg)),
            )
        )

    specs = [x for x in getattr(plan, "action_specs", []) if isinstance(x, dict)]
    for idx, spec in enumerate(specs[:8]):
        action = str(spec.get("action", "unknown"))
        jobs.append(
            (
                f"action_spec_{idx}_{action}",
                action,
                str(spec.get("reason", ""))[:240],
                lambda spec=spec, idx=idx: (f"action_spec_{idx}_{str(spec.get('action', 'unknown'))}", _execute_action_spec(df, spec, focus, cfg)),
            )
        )

    def _run_job(job: ProbeJob) -> tuple[str, Any]:
        default_key, action, reason, fn = job
        log_event(
            logger,
            "agent.file_cognition_probe",
            "ACTION_STARTED",
            file=relative_path or str(file_path),
            attempt=attempt,
            action=action,
            result_key=default_key,
            reason=reason[:240],
        )
        try:
            key, value = fn()
        except Exception as exc:  # noqa: BLE001
            key = default_key
            value = {"action": action, "reason": reason, "error": str(exc)}
        summary = _summarize_probe_result(value)
        event = "ACTION_FAILED" if isinstance(value, dict) and value.get("error") else "ACTION_COMPLETED"
        log_event(
            logger,
            "agent.file_cognition_probe",
            event,
            file=relative_path or str(file_path),
            attempt=attempt,
            action=action,
            result_key=key,
            rows_total=summary.get("rows_total"),
            rows_matched=summary.get("rows_matched"),
            error=summary.get("error", ""),
            warnings=summary.get("warnings", []),
        )
        return key, value

    if cfg.parallel.enable_parallel_probe_actions and len(jobs) > 1:
        workers = max(1, int(cfg.parallel.probe_max_workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_job, job) for job in jobs]
            for fut in as_completed(futures):
                k, v = fut.result()
                result[k] = v
    else:
        for job in jobs:
            k, v = _run_job(job)
            result[k] = v
    return result


def _summarize_file(
    cfg: AutoRealizeConfig,
    llm: LLMClient,
    prompt_mgr: PromptManager,
    base_context: dict[str, Any],
    probe_plan: CognitionProbePlan,
    probe_results: dict[str, Any],
) -> CognitionSummary:
    system = prompt_mgr.load("system/cognition_summarizer.md")
    safe_probe = probe_results
    if isinstance(safe_probe, dict):
        safe_probe = dict(safe_probe)
        if "error" in safe_probe and "not in index" in str(safe_probe.get("error")):
            safe_probe["probe_warnings"] = list(dict.fromkeys((safe_probe.get("probe_warnings") or []) + ["detected_missing_focus_columns"]))
    compact_observations = _compact_probe_observations(safe_probe if isinstance(safe_probe, dict) else {"result": safe_probe})
    observations = {
        "tool_observation_summary": compact_observations,
        "output_limits": {
            "concise_summary": "2-6 sentences",
            "detailed_report": "documents: 800-2500 Chinese characters unless the task rules require more; tables: 300-1200 Chinese characters",
            "field_descriptions": "one concise sentence per important real column; do not copy profiler JSON",
            "sheet_field_descriptions": "for multi-sheet Excel, map each sheet name to concise descriptions for that sheet's important columns",
            "key_facts": "up to 20 concrete facts",
            "risks": "up to 10 concrete risks",
            "related_hints": "up to 10 cross-file hints",
        },
        "internal_note": "Only output final conclusions. Never copy raw tool/profile JSON or internal action keys.",
    }
    stable, dynamic = stable_dynamic_prompt(
        stable=base_context,
        dynamic=observations,
        stable_title="Stable file context",
        dynamic_title="Dynamic tool observations",
    )
    structured_max_tokens = getattr(cfg.llm, "structured_max_tokens", None)
    try:
        structured_max_tokens = int(structured_max_tokens) if structured_max_tokens else None
    except (TypeError, ValueError):
        structured_max_tokens = None
    return llm.ask_structured(
        model_cls=CognitionSummary,
        system_prompt=system,
        user_prompt=dynamic,
        prompt_name="cognition_summary",
        static_context_prompt=stable,
        dynamic_user_prompt=dynamic,
        max_tokens=structured_max_tokens,
    )


def _map_role(text: str) -> FileRole:
    m = text.strip().lower()
    mp = {
        "task_requirement": FileRole.task_requirement,
        "data_description": FileRole.data_description,
        "raw_data_table": FileRole.raw_data_table,
        "code_or_config": FileRole.code_or_config,
        "image_or_media": FileRole.image_or_media,
        "unknown": FileRole.unknown,
    }
    return mp.get(m, FileRole.unknown)

