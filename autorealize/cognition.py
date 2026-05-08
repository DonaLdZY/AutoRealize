from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AutoRealizeConfig
from .llm.client import LLMClient
from .models import CognitionProbePlan, CognitionSummary, FileRole, FileSummary
from .profiling.stats import profile_dataframe
from .prompts.manager import PromptManager

logger = logging.getLogger(__name__)


def llm_cognition_for_file(
    cfg: AutoRealizeConfig,
    llm: LLMClient | None,
    prompt_mgr: PromptManager,
    file_path: Path,
    relative_path: str,
    parsed_kind: str,
    parsed_text_summary: str,
    parsed_columns: list[str],
    parsed_preview: list[dict[str, Any]],
    task_hint: str,
) -> FileSummary | None:
    if llm is None:
        return None

    base_context = {
        "file": relative_path,
        "kind": parsed_kind,
        "columns": parsed_columns[:80],
        "preview": parsed_preview[:20],
        "summary": parsed_text_summary[:2000],
        "task": task_hint,
    }

    probe_plan = _plan_probe(cfg, llm, prompt_mgr, base_context)
    probe_results: dict[str, Any] = {}
    if probe_plan.need_more_probe:
        logger.info("[P1-LLM] 正在生成探查计划: %s", relative_path)
        probe_results = _execute_probe_actions(file_path, probe_plan, cfg)
        logger.info("[P1-LLM] 探查完成: %s | actions=%s", relative_path, len(probe_results))

    summary = _summarize_file(cfg, llm, prompt_mgr, base_context, probe_plan, probe_results)
    role = _map_role(summary.file_role_guess)
    return FileSummary(
        path=relative_path,
        role=role,
        summary=summary.concise_summary,
        columns=summary.key_columns or parsed_columns[:30],
        warnings=summary.risks,
        related_files=summary.related_hints,
        source_metadata={"parsed_kind": parsed_kind},
    )


def _plan_probe(
    cfg: AutoRealizeConfig,
    llm: LLMClient,
    prompt_mgr: PromptManager,
    base_context: dict[str, Any],
) -> CognitionProbePlan:
    system = prompt_mgr.load("system/cognition_probe_planner.md")
    user = f"文件上下文:\n{json.dumps(base_context, ensure_ascii=False)}"
    try:
        return llm.ask_structured(
            model_cls=CognitionProbePlan,
            system_prompt=system,
            user_prompt=user,
            prompt_name="cognition_probe_plan",
        )
    except Exception:
        return CognitionProbePlan(need_more_probe=False, probe_actions=[], focus_columns=[], reason="fallback")


def _execute_probe_actions(file_path: Path, plan: CognitionProbePlan, cfg: AutoRealizeConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    suffix = file_path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".json"}:
        return result
    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".json":
            from .utils.json_table import read_json_as_table

            df, _ = read_json_as_table(
                file_path,
                sep=cfg.data.json_flatten_sep,
                max_level=cfg.data.json_flatten_max_level,
                keep_raw_nested_columns=cfg.data.json_keep_raw_nested_columns,
            )
        else:
            df = pd.read_excel(file_path)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    focus = plan.focus_columns[:8] if plan.focus_columns else [str(c) for c in df.columns[:8]]

    def _do_preview() -> tuple[str, Any]:
        return "preview_head", df.head(cfg.data.preview_rows).to_dict(orient="records")

    def _do_profiles() -> tuple[str, Any]:
        prof = profile_dataframe(df[focus], top_k=cfg.data.category_top_k)
        return (
            "profiles",
            [
                {
                    "name": p.name,
                    "dtype": p.dtype,
                    "null_ratio": p.null_ratio,
                    "unique_count": p.unique_count,
                    "sample_values": p.sample_values[:10],
                    "numeric_stats": p.numeric_stats,
                    "abnormal_tokens": p.abnormal_tokens,
                }
                for p in prof
            ],
        )

    jobs = []
    if "preview_head" in plan.probe_actions:
        jobs.append(_do_preview)
    if any(x in plan.probe_actions for x in ["profile_numeric", "check_nulls", "check_inf", "profile_categorical", "value_counts_topk"]):
        jobs.append(_do_profiles)

    if cfg.parallel.enable_parallel_probe_actions and len(jobs) > 1:
        workers = max(1, int(cfg.parallel.probe_max_workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(job) for job in jobs]
            for fut in as_completed(futures):
                k, v = fut.result()
                result[k] = v
    else:
        for job in jobs:
            k, v = job()
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
    user = (
        f"基础上下文:\n{json.dumps(base_context, ensure_ascii=False)}\n\n"
        f"探查计划:\n{json.dumps(probe_plan.model_dump(), ensure_ascii=False)}\n\n"
        f"探查结果:\n{json.dumps(probe_results, ensure_ascii=False, default=str)[:12000]}"
    )
    return llm.ask_structured(
        model_cls=CognitionSummary,
        system_prompt=system,
        user_prompt=user,
        prompt_name="cognition_summary",
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
