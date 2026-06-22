from __future__ import annotations

import base64
import builtins
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from .agents.architect import Architect
from .agents.orchestrator import Orchestrator
from .config import AutoRealizeConfig
from .cognition import llm_cognition_for_file
from .llm.client import LLMClient
from .logging_utils import configure_event_sink, log_event, write_event_taxonomy
from .knowledge.local_store import LocalKnowledgeStore
from .models import ConstraintMemory, FileRole, FileSummary, SubmissionCheckVerdict, SubmissionScriptPlan, TaskClassification
from .models import AmbiguityReview
from .parsers import build_registry
from .prompt_cache import stable_dynamic_prompt
from .profiling.csv_utils import read_csv_auto
from .profiling.relations import detect_relations
from .profiling.stats import (
    column_profile_to_dict,
    excel_sheet_groups_from_profiles,
    profile_dataframe,
    profile_excel_sheets,
    read_table,
    table_probe_sample_rows,
    table_sampling_metadata,
)
from .prompts.manager import PromptManager
from .report_writer import (
    SECTION_ALIASES,
    append_constraint_memory_section,
    apply_eval_fixes,
    build_description_markdown,
    coverage_defects,
    description_quality_check,
    eval_ambiguity_defects,
    write_data_description,
)
from .trajectory import TrajectoryLogger
from .modules.data_cognition import DataCognitionModule
from .modules.task_definition import TaskDefinitionModule
from .modules.types import DataCognitionResult, RuntimeServices, TaskDefinitionResult
from .utils.archives import archive_stem, extract_archive, is_archive_file
from .utils.filesystem import rel, safe_copytree, walk_dirs, walk_files

logger = logging.getLogger(__name__)
NETWORK_RETRY_MAX_ATTEMPTS = 5
NETWORK_RETRY_MAX_SLEEP_SECONDS = 30.0


def _is_retryable_network_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    try:
        if int(status_code) in {429, 500, 502, 503, 504}:
            return True
    except Exception:
        pass
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if any(
        key in name
        for key in [
            "timeout",
            "connection",
            "ratelimit",
            "internalserver",
            "apierror",
            "apiconnection",
            "badgateway",
            "serviceunavailable",
            "gateway",
            "httpstatus",
        ]
    ):
        return True
    return any(
        key in msg
        for key in [
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection refused",
            "10061",
            "actively refused",
            "积极拒绝",
            "temporary failure",
            "temporarily unavailable",
            "bad gateway",
            "502",
            "503",
            "504",
            "rate limit",
            "too many requests",
            "getaddrinfo",
            "11001",
            "name resolution",
            "name or service not known",
            "server disconnected",
            "remote protocol error",
        ]
    )


def _openai_create_with_network_retry(client: OpenAI, *, label: str, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(1, NETWORK_RETRY_MAX_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            retryable = _is_retryable_network_error(exc)
            logger.warning(
                "%s request failed, retryable=%s, attempt=%s/%s: %s",
                label,
                retryable,
                attempt,
                NETWORK_RETRY_MAX_ATTEMPTS,
                exc,
            )
            if (not retryable) or attempt >= NETWORK_RETRY_MAX_ATTEMPTS:
                raise
            time.sleep(min(NETWORK_RETRY_MAX_SLEEP_SECONDS, 5.0 * attempt))
    if last_exc is not None:
        raise last_exc


def _json_safe(value):
    """Convert pandas/numpy-ish values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, pd.Timedelta):
        return None if pd.isna(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_dumps_safe(value, **kwargs) -> str:
    return json.dumps(_json_safe(value), **kwargs)


def _extract_constraint_memory(
    *,
    llm_client: LLMClient,
    prompt_mgr: PromptManager,
    file_summaries: list[FileSummary],
    task_hint: str,
) -> dict:
    """抽取跨阶段复用的关键约束记忆。"""
    # 规则先验：先从字段和摘要里抓明显约束线索。
    hints: list[dict] = []
    keyword_groups = {
        "运价/费率约束": ["运价", "费率", "单价", "成本", "合同", "price", "rate", "cost"],
        "容量约束": ["体积", "重量", "装载", "容量", "载重", "volume", "weight", "capacity"],
        "时序/日分配约束": ["每日", "日期", "时间", "day", "date", "time"],
        "地址/区间约束": ["发货", "收货", "地址", "起点", "终点", "区间", "origin", "destination"],
    }
    for fs in file_summaries:
        text = f"{fs.path} {fs.summary} {' '.join(fs.columns[:80])}".lower()
        for cname, kws in keyword_groups.items():
            if any(k.lower() in text for k in kws):
                hints.append(
                    {
                        "name": cname,
                        "description": f"从 `{fs.path}` 检测到与“{cname}”相关信息。",
                        "evidence": [fs.path, fs.summary[:160]],
                        "related_fields": [c for c in fs.columns if any(k.lower() in str(c).lower() for k in kws)][:12],
                        "priority": "high" if "约束" in cname or "容量" in cname else "medium",
                    }
                )

    dedup: dict[str, dict] = {}
    for it in hints:
        n = it.get("name", "")
        if n not in dedup:
            dedup[n] = it
        else:
            dedup[n]["evidence"] = list(dict.fromkeys((dedup[n].get("evidence", []) + it.get("evidence", []))))[:6]
            dedup[n]["related_fields"] = list(
                dict.fromkeys((dedup[n].get("related_fields", []) + it.get("related_fields", [])))
            )[:20]
    rule_items = list(dedup.values())[:20]

    system = (
        "你是约束抽取器。请仅输出 JSON，结构必须满足给定 schema。"
        "你的任务是从文件摘要与字段中提取对任务有决定作用的业务约束、数据约束、评估约束。"
    )
    stable, dynamic = stable_dynamic_prompt(
        stable={
            "task_hint": task_hint,
            "file_summaries": [fs.model_dump() for fs in file_summaries][:80],
        },
        dynamic={"instruction": "提炼可执行约束，不要泛泛而谈；每条约束给证据与相关字段。"},
        stable_title="Stable constraint extraction context",
        dynamic_title="Dynamic constraint extraction request",
        stable_limit=12000,
    )
    mem = llm_client.ask_structured(
        model_cls=ConstraintMemory,
        system_prompt=system,
        user_prompt=dynamic,
        prompt_name="constraint_memory_extractor",
        static_context_prompt=stable,
        dynamic_user_prompt=dynamic,
    )
    out = mem.model_dump()
    # Rule hints may enrich the LLM result, but they are never a replacement for LLM extraction.
    if rule_items:
        exist = {str(x.get("name", "")) for x in out.get("items", [])}
        for it in rule_items:
            if it.get("name") not in exist:
                out.setdefault("items", []).append(it)
    return out


class AutoRealizePipeline:
    def __init__(self, config: AutoRealizeConfig | None = None) -> None:
        self.config = config or AutoRealizeConfig.from_env()

    def run(
        self,
        input_root: Path,
        output_root: Path,
        task_hint: str,
        run_name: str,
    ) -> Path:
        run_started_at = time.perf_counter()
        # RUN_STARTED 只保留一条（包含 output_root），避免终端重复刷屏。
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        data_out = run_dir / "data"
        safe_copytree(input_root, data_out)
        source_file_set = {rel(p, data_out) for p in walk_files(data_out)}
        report_dir = run_dir / "realize_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        configure_event_sink(
            report_dir / self.config.telemetry.event_stream_filename,
            state_path=report_dir / self.config.telemetry.current_state_filename,
            run_id=run_name,
            enabled=self.config.telemetry.enabled,
            recent_limit=self.config.telemetry.recent_events_limit,
        )
        log_event(logger, "pipeline", "RUN_STARTED", run_name=run_name, input_root=str(input_root), output_root=str(run_dir))
        if self.config.telemetry.write_config_snapshot:
            (report_dir / "final_config.json").write_text(
                _json_dumps_safe(self.config.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if self.config.telemetry.write_config_schema:
            (report_dir / "config_schema.json").write_text(
                _json_dumps_safe(self.config.schema_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        write_event_taxonomy(report_dir / "event_taxonomy.json")
        log_event(logger, "pipeline", "WORKSPACE_COPIED", workspace=str(data_out))
        self._expand_archives(data_out, report_dir)
        _preserve_original_description(data_out, run_dir)
        source_file_set = {rel(p, data_out) for p in walk_files(data_out)}

        traj = TrajectoryLogger(report_dir)
        traj.log("bootstrap", "start", {"input_root": str(input_root), "run_name": run_name})
        registry = build_registry(self.config)

        llm_client = LLMClient(self.config, report_dir)
        llm_client.health_check()
        traj.log("bootstrap", "llm", {"enabled": True, "model": self.config.llm.model_name})
        log_event(logger, "llm", "CLIENT_READY", enabled=True, model=self.config.llm.model_name)

        prompt_mgr = PromptManager(self.config)
        knowledge_store = None
        if self.config.knowledge.enabled:
            knowledge_store = LocalKnowledgeStore(
                report_dir / self.config.knowledge.store_filename,
                max_entry_chars=self.config.knowledge.max_entry_chars,
                boost_structured=self.config.knowledge.boost_structured_knowledge,
            )
            log_event(logger, "knowledge.local_store", "CREATED", file=self.config.knowledge.store_filename)
        services = RuntimeServices(
            llm_client=llm_client,
            prompt_mgr=prompt_mgr,
            registry=registry,
            trajectory=traj,
            knowledge_store=knowledge_store,
        )
        log_event(logger, "workflow", "CREATED", design="spec_demo_two_modules")
        log_event(logger, "workflow", "ACTIVATED")

        cognition_result = DataCognitionResult()
        task_result = TaskDefinitionResult()

        if self.config.switches.run_data_cognition:
            cognition_module = DataCognitionModule(self.config, services, report_dir)
            cognition_result = cognition_module.run(data_out, task_hint)
        else:
            log_event(logger, "module.data_cognition", "SKIPPED", reason="switch_disabled")

        if self.config.switches.run_task_definition:
            task_module = TaskDefinitionModule(self.config, services, run_dir, report_dir)
            task_result = task_module.run(data_out, task_hint, cognition_result)
        else:
            log_event(logger, "module.task_definition", "SKIPPED", reason="switch_disabled")

        if knowledge_store is not None:
            knowledge_store.flush()
            if self.config.knowledge.write_rag_manifest:
                knowledge_store.write_manifest(report_dir / "rag_manifest.json")
            log_event(logger, "knowledge.local_store", "FLUSHED")
        log_event(logger, "workflow", "COMPLETED")

        traj.write_markdown_index()
        summary = {
            "schema_version": "autorealize.run_summary.v1",
            "run_name": run_name,
            "input_root": str(input_root),
            "data_output_root": str(data_out),
            "run_dir": str(run_dir),
            "task_hint": task_hint,
            "duration_seconds_before_flatten": round(time.perf_counter() - run_started_at, 4),
            "modules": {
                "data_cognition": {
                    "enabled": bool(self.config.switches.run_data_cognition),
                    "files": len(cognition_result.file_summaries),
                    "relations": len(cognition_result.relation_hints),
                    "artifact": "data_description.md",
                },
                "task_definition": {
                    "enabled": bool(self.config.switches.run_task_definition),
                    "description": "description.md",
                    "sample_submission": "sample_submission.csv" if task_result.sample_submission_path else None,
                    "defects_after_gate": len(task_result.defects),
                },
            },
            "telemetry": {
                "event_stream": self.config.telemetry.event_stream_filename,
                "current_state": self.config.telemetry.current_state_filename,
                "event_taxonomy": "event_taxonomy.json",
                "frontend_manifest": "frontend_manifest.json",
                "final_config": "final_config.json",
                "config_schema": "config_schema.json",
            },
            "knowledge": {
                "enabled": bool(self.config.knowledge.enabled),
                "store": self.config.knowledge.store_filename,
                "rag_manifest": "rag_manifest.json" if self.config.knowledge.write_rag_manifest else None,
            },
        }
        (report_dir / "run_summary.json").write_text(_json_dumps_safe(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        traj.log("finalize", "done", summary)
        self._flatten_data_to_root(run_dir, data_out, report_dir)
        _write_frontend_manifest(run_dir, report_dir, run_name, input_root, task_hint, self.config)
        _append_output_layout_to_description(run_dir, report_dir)
        log_event(logger, "pipeline", "RUN_COMPLETED", run_dir=str(run_dir))
        return run_dir

    def _expand_archives(self, data_out: Path, run_dir: Path) -> None:
        if not self.config.data.enable_archive_extraction:
            return
        archive_files = [p for p in walk_files(data_out) if is_archive_file(p)]
        if not archive_files:
            return
        log_lines: list[str] = []
        for arc in archive_files:
            target_dir = arc.parent / f"{archive_stem(arc)}__extracted"
            try:
                result = extract_archive(
                    arc,
                    target_dir,
                    max_files=self.config.data.archive_extract_file_limit,
                )
                log_lines.append(
                    f"- {rel(arc, data_out)} -> {rel(target_dir, data_out)} | type={result.archive_type} | files={result.extracted_files} | warning={result.warning or 'none'}"
                )
                if (
                    self.config.data.archive_extract_file_limit > 0
                    and result.extracted_files > self.config.data.archive_extract_file_limit
                ):
                    log_lines.append(
                        f"  limit_warning: extracted_files={result.extracted_files} > archive_extract_file_limit={self.config.data.archive_extract_file_limit}"
                    )
                if not self.config.data.keep_archive_after_extract and arc.exists():
                    arc.unlink()
            except Exception as exc:  # noqa: BLE001
                log_lines.append(f"- {rel(arc, data_out)} | extract_error={exc}")
        if log_lines:
            (run_dir / "archive_extraction.log").write_text("\n".join(log_lines), encoding="utf-8")

    def _flatten_data_to_root(self, run_dir: Path, data_out: Path, report_dir: Path) -> None:
        log_event(logger, "finalize.flatten", "ACTIVATED", source=str(data_out), target=str(run_dir))
        if not data_out.exists() or not data_out.is_dir():
            log_event(logger, "finalize.flatten", "SKIPPED", reason="data_out_missing")
            return
        reserved = {"description.md", "description_origin.md", "sample_submission.csv", "realize_report"}
        conflicts: list[str] = []
        for src in sorted(walk_files(data_out)):
            rel_path = src.relative_to(data_out)
            dst = run_dir / rel_path
            if rel_path.parts and rel_path.parts[0] == "realize_report":
                continue
            if rel_path.name in reserved:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                conflicts.append(str(rel_path).replace("\\", "/"))
                continue
            shutil.move(str(src), str(dst))
        try:
            shutil.rmtree(data_out)
        except Exception:
            pass
        if conflicts:
            (report_dir / "flatten_conflicts.log").write_text("\n".join(conflicts), encoding="utf-8")
            log_event(logger, "finalize.flatten", "COMPLETED_WITH_CONFLICTS", conflicts=len(conflicts))
        else:
            log_event(logger, "finalize.flatten", "COMPLETED", moved="all")




def _preserve_original_description(data_root: Path, run_dir: Path) -> None:
    """如果输入数据中已存在 description.md，则复制一份到输出根目录的 description_origin.md。"""
    candidates = [p for p in walk_files(data_root) if p.name.lower() == "description.md"]
    if not candidates:
        return
    # 选择层级最浅、路径最短的原始 description.md 作为备份。
    candidates = sorted(candidates, key=lambda x: (len(x.parts), str(x)))
    src = candidates[0]
    target = run_dir / "description_origin.md"
    if not target.exists():
        shutil.copy2(src, target)


def _append_output_layout_to_description(run_dir: Path, report_dir: Path) -> None:
    """将输出目录结构写入过程报告，不污染最终 description.md。"""
    lines: list[str] = [
        "# 输出目录结构",
        "",
        "## 目录树",
        "```text",
    ]

    def walk_tree(root: Path, prefix: str = "") -> None:
        children = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for idx, child in enumerate(children):
            branch = "└── " if idx == len(children) - 1 else "├── "
            lines.append(prefix + branch + child.name)
            if child.is_dir():
                ext = "    " if idx == len(children) - 1 else "│   "
                walk_tree(child, prefix + ext)

    walk_tree(run_dir)
    lines.append("```")
    lines.append("")
    lines.append("## 文件与目录职责")

    role_map = {
        "description.md": "面向 ML-Master/AutoML 的任务说明文档（Kaggle 风格）",
        "sample_submission.csv": "提交样例文件（优先复用原始样例）",
        "description_origin.md": "原始数据中自带的 description.md 备份",
        "realize_report": "AutoRealize 过程报告目录（认知/任务定义/轨迹/日志）",
        "data_description.md": "原始数据认知文档",
        "trajectory_events.jsonl": "结构化运行事件轨迹",
        "trajectory.md": "运行轨迹索引",
        "llm_traces.jsonl": "LLM 请求与响应轨迹",
        "llm_usage.jsonl": "每次 LLM/VLLM 调用的 token usage 账本",
        "llm_usage_summary.json": "LLM/VLLM token usage 汇总与 provider cache 统计",
        "event_stream.jsonl": "全量结构化事件流（前端监控首选数据源）",
        "run_summary.json": "本次运行摘要",
    }

    # 目录条目
    for d in sorted([p for p in run_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        lines.append(f"- `{d.name}/`: {role_map.get(d.name, '自动生成目录')}")

    # 文件条目（按文件名去重）
    seen = set()
    for f in sorted([p for p in run_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower()):
        if f.name in seen:
            continue
        seen.add(f.name)
        lines.append(f"- `{f.name}`: {role_map.get(f.name, '自动生成文件')}")

    (report_dir / "output_layout.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_frontend_manifest(
    run_dir: Path,
    report_dir: Path,
    run_name: str,
    input_root: Path,
    task_hint: str,
    config: AutoRealizeConfig,
) -> None:
    """写出前端入口清单，集中暴露运行产物、事件源与可配置能力。"""
    def exists_rel(path: str) -> dict:
        p = run_dir / path
        return {"path": path, "exists": p.exists(), "kind": "dir" if p.is_dir() else "file"}

    report_files = [
        "realize_report/event_stream.jsonl",
        "realize_report/current_state.json",
        "realize_report/event_taxonomy.json",
        "realize_report/final_config.json",
        "realize_report/config_schema.json",
        "realize_report/run_summary.json",
        "realize_report/trajectory.md",
        "realize_report/trajectory_events.jsonl",
        "realize_report/llm_traces.jsonl",
        "realize_report/llm_usage.jsonl",
        "realize_report/llm_usage_summary.json",
        "realize_report/data_description.md",
        "realize_report/data_cognition_report.json",
        "realize_report/task_definition_report.json",
        "realize_report/submission_report.json",
        "realize_report/knowledge_base.json",
        "realize_report/knowledge_store.jsonl",
        "realize_report/retrieved_knowledge.json",
        "realize_report/rag_manifest.json",
        "realize_report/file_cognition",
    ]
    output_files = ["description.md", "sample_submission.csv", "description_origin.md"]
    data_files = []
    for p in sorted(walk_files(run_dir), key=lambda x: str(x).lower()):
        if report_dir in p.parents:
            continue
        if p.name in {"description.md", "sample_submission.csv", "description_origin.md"}:
            continue
        data_files.append(str(p.relative_to(run_dir)).replace("\\", "/"))

    manifest = {
        "schema_version": "autorealize.frontend_manifest.v1",
        "run_name": run_name,
        "input_root": str(input_root),
        "run_dir": str(run_dir),
        "task_hint": task_hint,
        "watch": {
            "event_stream": "realize_report/event_stream.jsonl",
            "current_state": "realize_report/current_state.json",
            "event_taxonomy": "realize_report/event_taxonomy.json",
        },
        "modules": [
            {
                "id": "data_cognition",
                "title": "数据认知",
                "main_artifacts": ["realize_report/data_description.md", "realize_report/file_cognition", "realize_report/knowledge_base.json"],
                "event_scopes": ["module.data_cognition", "stage.P1", "knowledge.local_store"],
                "enabled": bool(config.switches.run_data_cognition),
            },
            {
                "id": "task_definition",
                "title": "任务定义",
                "main_artifacts": ["description.md", "sample_submission.csv", "realize_report/retrieved_knowledge.json"],
                "event_scopes": ["module.task_definition", "checker.sample_submission"],
                "enabled": bool(config.switches.run_task_definition),
            },
        ],
        "artifacts": {
            "outputs": [exists_rel(x) for x in output_files],
            "reports": [exists_rel(x) for x in report_files],
            "data_files": data_files,
        },
        "config_entrypoints": {
            "snapshot": "realize_report/final_config.json",
            "schema": "realize_report/config_schema.json",
            "cli_print_default": "python -m autorealize.cli --print-default-config",
            "cli_write_default": "python -m autorealize.cli --write-default-config config.json",
        },
        "frontend_notes": [
            "优先轮询 current_state.json 展示状态，按 seq 增量读取 event_stream.jsonl 追加细节。",
            "模块卡片可用 modules[*].event_scopes 过滤事件。",
            "点击文件产物时优先打开 artifacts 中 exists=true 的条目。",
        ],
    }
    out = report_dir / "frontend_manifest.json"
    out.write_text(_json_dumps_safe(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event(logger, "pipeline.frontend_manifest", "GENERATED_FILE", file="realize_report/frontend_manifest.json")

def _collect_inventory(data_root: Path) -> dict:
    files = walk_files(data_root)
    table_ext = {".csv", ".xlsx", ".xls", ".json"}
    doc_ext = {".txt", ".md", ".doc", ".docx", ".pdf", ".rst"}
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
    archive_ext = {".zip", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z"}
    table_count = 0
    document_count = 0
    image_count = 0
    archive_count = 0
    has_task_doc = False
    for f in files:
        suffix = f.suffix.lower()
        name = f.name.lower()
        if suffix in table_ext:
            table_count += 1
        if suffix in doc_ext:
            document_count += 1
            if any(k in name for k in ["task", "readme", "description", "requirement", "需求", "任务"]):
                has_task_doc = True
        if suffix in image_ext:
            image_count += 1
        if suffix in archive_ext:
            archive_count += 1
    return {
        "file_count": len(files),
        "table_count": table_count,
        "document_count": document_count,
        "image_count": image_count,
        "archive_count": archive_count,
        "has_task_doc": has_task_doc,
    }

def _infer_role(path: str, kind: str, text_summary: str) -> FileRole:
    lpath = path.lower()
    ltext = text_summary.lower()
    if kind == "table":
        return FileRole.raw_data_table
    if kind == "archive":
        return FileRole.data_description
    if kind == "image":
        return FileRole.image_or_media
    if any(k in lpath for k in ["description", "readme", "task", "requirement", "spec"]):
        return FileRole.task_requirement
    if any(k in ltext for k in ["任务", "目标", "submission", "evaluate", "评估指标", "需求"]):
        return FileRole.task_requirement
    if kind in {"document", "structured_document"}:
        return FileRole.data_description
    if any(k in lpath for k in [".py", ".ipynb", ".sql", ".toml", ".yaml", ".yml"]):
        return FileRole.code_or_config
    return FileRole.unknown


def _summarize_dirs(root: Path, files: list[FileSummary]) -> list[str]:
    by_dir: dict[str, list[FileSummary]] = {}
    for f in files:
        d = str(Path(f.path).parent).replace("\\", "/")
        by_dir.setdefault(d, []).append(f)
    lines: list[str] = []
    for d in walk_dirs(root):
        rd = rel(d, root)
        group = by_dir.get(rd, [])
        if not group:
            continue
        roles = {}
        for g in group:
            roles[g.role.value] = roles.get(g.role.value, 0) + 1
        role_desc = ", ".join([f"{k}:{v}" for k, v in sorted(roles.items(), key=lambda x: -x[1])])
        lines.append(f"`{rd}`: 文件数 {len(group)}，角色分布 {role_desc}")
    return lines


def _digest_data_inventory(file_summaries: list[FileSummary]) -> str:
    parts = []
    for fs in file_summaries[:80]:
        parts.append(f"- {fs.path} ({fs.role.value}): {fs.summary[:140]}")
    return "\n".join(parts)


def _guess_column_semantics(columns: list[str], profiles: list[dict], task_hint: str) -> dict[str, str]:
    profile_by_name = {str(p.get("name", "")): p for p in profiles if str(p.get("name", "")).strip()}
    semantics: dict[str, str] = {}
    for col in columns:
        name = str(col).strip()
        if not name:
            continue
        lower = name.lower()
        profile = profile_by_name.get(name, {})
        logical = str(profile.get("logical_type", "") or profile.get("dtype", "")).lower()
        parts: list[str] = []
        if any(k in lower for k in ["id", "key", "code"]) or any(k in name for k in ["编号", "代码", "编码", "订单号", "单号"]):
            parts.append("业务实体或关联键字段")
        elif any(k in lower for k in ["target", "label", "class", "category", "y"]) or any(k in name for k in ["标签", "目标", "类别", "结果"]):
            parts.append("可能的标签、目标或结果字段")
        elif any(k in lower for k in ["date", "time", "dt"]) or any(k in name for k in ["日期", "时间", "时刻"]):
            parts.append("时间字段")
        elif any(k in lower for k in ["cost", "price", "fee", "amount", "rate", "score"]) or any(k in name for k in ["成本", "价格", "费用", "金额", "费率", "得分"]):
            parts.append("金额、成本、价格或评分字段")
        elif any(k in lower for k in ["qty", "count", "num", "volume", "weight"]) or any(k in name for k in ["数量", "件数", "体积", "重量", "容量"]):
            parts.append("数量、容量或规模字段")
        elif any(k in name for k in ["类型", "类别", "车型", "状态"]) or logical == "categorical":
            parts.append("分类或枚举字段")
        elif logical in {"integer", "float", "numeric_string", "mixed_numeric_text"}:
            parts.append("数值型字段")
        elif logical == "datetime":
            parts.append("时间字段")
        else:
            parts.append("业务属性字段")

        null_ratio = profile.get("null_ratio")
        unique_count = profile.get("unique_count")
        stats = profile.get("numeric_stats") if isinstance(profile.get("numeric_stats"), dict) else {}
        datetime_stats = profile.get("datetime_stats") if isinstance(profile.get("datetime_stats"), dict) else {}
        extras: list[str] = []
        if unique_count is not None:
            extras.append(f"去重值约 {unique_count} 个")
        if null_ratio is not None:
            try:
                extras.append(f"缺失率 {float(null_ratio):.2%}")
            except Exception:
                pass
        if stats:
            extras.append(
                "数值范围 "
                + f"{stats.get('min')} 到 {stats.get('max')}"
                + (f"，均值 {stats.get('mean'):.6g}" if isinstance(stats.get("mean"), (int, float)) else "")
            )
        if datetime_stats:
            granularity = datetime_stats.get("granularity") or ""
            extras.append(
                f"时间范围 {datetime_stats.get('min')} 到 {datetime_stats.get('max')}"
                + (f"，粒度 {granularity}" if granularity else "")
            )
        semantics[name] = "；".join([parts[0], *extras[:3]])
    return semantics


def _guess_semantic_meta(columns: list[str], profiles: list[dict], semantics: dict[str, str], *, source: str) -> dict[str, dict]:
    return {
        str(col): {
            "source": source,
            "confidence": "medium",
            "confidence_score": 0.6,
            "evidence": "deterministic_column_profile",
        }
        for col in columns
        if str(col) in semantics
    }


def _fill_deterministic_file_memory(fs: FileSummary, parsed_kind: str) -> None:
    """Populate human-readable cognition fields without calling an LLM."""
    meta = fs.source_metadata or {}
    facts: list[str] = []
    risks: list[str] = []
    related: list[str] = list(fs.related_files or [])
    role = fs.role.value if hasattr(fs.role, "value") else str(fs.role)

    shape = meta.get("shape")
    if isinstance(shape, list) and len(shape) >= 2:
        facts.append(f"结构化表规模约为 {shape[0]} 行、{shape[1]} 列。")
    if meta.get("shape_estimated"):
        risks.append("行数或 shape 由轻量方式估算，必要时应在建模代码中重新核对。")
    dialect = meta.get("csv_dialect") if isinstance(meta.get("csv_dialect"), dict) else {}
    if dialect:
        if dialect.get("inferred"):
            facts.append(
                "CSV 需要非默认读取方式："
                f"sep={dialect.get('sep')!r}"
                + (f", engine={dialect.get('engine')!r}" if dialect.get("engine") else "")
                + f"，原因={dialect.get('reason') or '自动探测'}。"
            )
        else:
            facts.append("CSV 可按默认逗号分隔读取。")
    sheets = meta.get("excel_sheets") if isinstance(meta.get("excel_sheets"), list) else []
    sheet_names = meta.get("excel_sheet_names") if isinstance(meta.get("excel_sheet_names"), list) else []
    if sheet_names:
        facts.append(f"Excel 工作簿包含 {len(sheet_names)} 个 sheet：" + "、".join(f"`{x}`" for x in sheet_names[:12]) + (" 等。" if len(sheet_names) > 12 else "。"))
        if len(sheet_names) > 1:
            risks.append("多 sheet Excel 不能依赖 pandas 默认读取第一个 sheet，后续代码应显式指定 sheet_name。")
    json_schema = meta.get("json_first_level_schema") if isinstance(meta.get("json_first_level_schema"), dict) else {}
    if json_schema:
        facts.append(
            f"JSON 第一层样本数 {json_schema.get('sample_count')}，schema 相似度 {json_schema.get('schema_similarity')}。"
        )
    if fs.column_profiles:
        high_null = []
        keyish = []
        for p in fs.column_profiles:
            name = str(p.get("name", "")).strip()
            if not name:
                continue
            try:
                if float(p.get("null_ratio") or 0.0) >= 0.98:
                    high_null.append(name)
            except Exception:
                pass
            lower = name.lower()
            if any(k in lower for k in ["id", "key", "code"]) or any(k in name for k in ["编号", "代码", "订单号", "单号"]):
                keyish.append(name)
        if keyish:
            fs.key_entities = list(dict.fromkeys([*fs.key_entities, *keyish]))[:20]
            facts.append("疑似主键/关联键字段：" + "、".join(f"`{x}`" for x in keyish[:12]) + "。")
        if high_null:
            risks.append("以下字段几乎全空：" + "、".join(f"`{x}`" for x in high_null[:12]) + "。")

    if parsed_kind in {"document", "structured_document"}:
        facts.append("该文件按说明文档或结构化文档读取，内容会作为任务定义和约束抽取的候选证据。")
        if fs.path.lower().endswith("description.md"):
            facts.append("该文件名为 description.md，在权威优先级中高于其他说明文档，仅低于用户输入。")

    lines = [
        f"文件角色判定为 `{role}`。",
        str(fs.summary or "").strip(),
    ]
    if facts:
        lines.append("关键事实：" + " ".join(facts[:8]))
    if risks:
        lines.append("风险与注意事项：" + " ".join(risks[:8]))
    if sheets:
        sheet_bits = []
        for item in sheets[:8]:
            if not isinstance(item, dict):
                continue
            cols = [str(x) for x in (item.get("columns") or [])[:8]]
            shape_item = item.get("shape") or []
            sheet_bits.append(
                f"`{item.get('sheet_name')}` shape={shape_item} columns={cols}"
            )
        if sheet_bits:
            lines.append("Sheet 概览：" + "；".join(sheet_bits))
    fs.detailed_report = "\n".join([x for x in lines if x]).strip()
    fs.extracted_knowledge = list(dict.fromkeys([*facts, *fs.extracted_knowledge]))[:40]
    fs.warnings = list(dict.fromkeys([*fs.warnings, *risks]))[:40]
    fs.related_files = related[:12]


def _refine_semantics_by_relations(file_summaries: list[FileSummary], relation_hints: list) -> None:
    by_path = {fs.path: fs for fs in file_summaries}
    for rh in relation_hints:
        lf = by_path.get(getattr(rh, "left_file", ""))
        rf = by_path.get(getattr(rh, "right_file", ""))
        if lf is None or rf is None:
            continue
        reason = str(getattr(rh, "short_evidence", "") or getattr(rh, "reason", ""))
        pairs = []
        if str(getattr(rh, "left_field", "") or "").strip() or str(getattr(rh, "right_field", "") or "").strip():
            pairs = [(lf, str(getattr(rh, "left_field", "") or "")), (rf, str(getattr(rh, "right_field", "") or ""))]
        else:
            shared = [str(x) for x in (getattr(rh, "shared_columns", []) or [])]
            pairs = [(fs, col_low) for col_low in shared for fs in [lf, rf]]
        for fs, target_col in pairs:
            for c in fs.columns:
                if str(c).lower().replace(" ", "") != str(target_col).lower().replace(" ", ""):
                    continue
                cur = fs.column_semantics.get(c, "")
                if fs.column_semantic_meta.get(c, {}).get("source") != "llm_field_description":
                    continue
                meta = fs.column_semantic_meta.get(c, {})
                if "join" not in cur and "关联" not in cur:
                    fs.column_semantics[c] = f"{cur}（跨表可关联字段）".strip()
                prev = float(meta.get("confidence_score", 0.6))
                boosted = min(0.95, prev + 0.08)
                fs.column_semantic_meta[c] = {
                    "confidence_score": round(boosted, 3),
                    "confidence": "high" if boosted >= 0.8 else ("medium" if boosted >= 0.65 else "low"),
                    "source": "cross_file_refine",
                    "evidence": reason or "shared_column_relation",
                }


def _refine_file_summaries_by_downstream_context(file_summaries: list[FileSummary], downstream_context: dict) -> None:
    """Use P2 train/test hints without overwriting LLM-authored data descriptions."""
    train_name = str(downstream_context.get("train_table", "") or "").strip()
    predict_name = str(downstream_context.get("predict_table", "") or "").strip()
    target_col = str(downstream_context.get("target_column", "") or "").strip()
    submission_cols = [str(x) for x in downstream_context.get("submission_columns", []) if str(x).strip()]

    for fs in file_summaries:
        name = Path(fs.path).name
        lower_name = name.lower()

        if "samplesubmission" in "".join(ch for ch in lower_name if ch.isalnum()) or "sample_submission" in lower_name:
            fs.role = FileRole.task_requirement
            cols_text = ", ".join(submission_cols) if submission_cols else "（列名未识别）"
            if not fs.summary:
                fs.summary = f"提交样例/格式文件，列顺序为：{cols_text}。"
            continue

        if train_name and name == train_name:
            fs.role = FileRole.raw_data_table
            fs.source_metadata = {**(fs.source_metadata or {}), "downstream_role_hint": "train_table", "target_column_hint": target_col}
            if target_col and target_col not in fs.summary:
                fs.summary = (fs.summary.rstrip() + f" 下游任务将 `{target_col}` 识别为训练目标列。").strip()
            continue

        if predict_name and name == predict_name:
            fs.role = FileRole.raw_data_table
            fs.source_metadata = {**(fs.source_metadata or {}), "downstream_role_hint": "predict_table", "target_column_hint": target_col}
            if target_col and target_col not in fs.summary:
                fs.summary = (
                    fs.summary.rstrip()
                    + f" 预测阶段使用该表作为待预测输入；目标列 `{target_col}` 来自训练数据或提交样例，不应作为预测输入特征。"
                ).strip()
            continue

        if train_name and train_name in fs.path:
            fs.role = FileRole.raw_data_table
            fs.source_metadata = {**(fs.source_metadata or {}), "downstream_role_hint": "train_table_derived", "target_column_hint": target_col}
            if target_col and target_col not in fs.summary:
                fs.summary = (fs.summary.rstrip() + f" 下游任务将 `{target_col}` 识别为训练目标列。").strip()
        elif predict_name and predict_name in fs.path:
            fs.role = FileRole.raw_data_table
            fs.source_metadata = {**(fs.source_metadata or {}), "downstream_role_hint": "predict_table_derived", "target_column_hint": target_col}
            if target_col and target_col not in fs.summary:
                fs.summary = (
                    fs.summary.rstrip()
                    + f" 预测阶段使用该表作为待预测输入；目标列 `{target_col}` 来自训练数据或提交样例，不应作为预测输入特征。"
                ).strip()


def _validate_generated_submission(
    out_df: pd.DataFrame,
    *,
    ctx: dict,
    source_df: pd.DataFrame,
) -> list[str]:
    """Validate generated sample_submission against inferred semantic contract."""
    issues: list[str] = []
    cols = [str(c) for c in out_df.columns.tolist()]
    submission_cols = [str(x) for x in ctx.get("submission_columns", []) if str(x).strip()]

    semantic = ctx.get("semantic_keys", {}) or {}
    entity_id_key = str(semantic.get("entity_id_key", "") or "")
    group_id_key = str(semantic.get("group_id_key", "") or "")
    resource_keys = [str(x) for x in semantic.get("resource_keys", []) if str(x).strip()]

    if submission_cols and cols != submission_cols:
        issues.append(f"column_order_mismatch: got={cols}, expected={submission_cols}")

    # semantic keys are hints only, not hard failures.
    if entity_id_key and entity_id_key not in cols:
        issues.append(f"hint_missing_entity_id_key: {entity_id_key}")

    if group_id_key and group_id_key not in cols:
        issues.append(f"hint_missing_group_id_key: {group_id_key}")

    if resource_keys and not any(k in cols for k in resource_keys):
        issues.append(f"hint_missing_resource_keys: expected_any_of={resource_keys}")

    if len(out_df) == 0:
        issues.append("empty_submission")

    if len(source_df) > 0 and len(out_df) > len(source_df) * 3:
        issues.append(f"abnormal_row_count: out={len(out_df)}, source={len(source_df)}")

    return issues



def _evaluate_submission_with_llm(
    *,
    llm_client: LLMClient,
    task_hint: str,
    task_type_hint: str,
    candidate_name: str,
    source_columns: list[str],
    source_preview: list[dict],
    generated_columns: list[str],
    generated_preview: list[dict],
    semantic_keys: dict,
    constraint_memory: dict | None = None,
) -> SubmissionCheckVerdict:
    stable, dynamic = stable_dynamic_prompt(
        stable={
            "rules": [
                "只检查 schema/列语义，不检查样例值是否已经是优化后的最终答案。",
                "候选 sample_submission 必须服务下游 ML-Master/AutoML，而不是服务 AutoRealize 自身。",
                "优先判断提交文件是否表达题目真正需要预测/决策的对象，而不是套用固定模板。",
                "不要把所有问题硬套成 id+target；优化、推荐、编排、分配类问题可以有多列决策输出。",
                "sample_submission.csv 是格式/列契约样例，不是已经求解完成的最优方案。",
                "只有当缺少必要列、列含义无法表达任务输出、列顺序违反官方样例时，才 needs_regenerate=true。",
                "若原始数据/文档中有官方 sample submission 语义，必须优先尊重官方样例。",
            ],
            "task_hint": task_hint,
            "task_type_hint": task_type_hint,
            "semantic_keys": semantic_keys,
            "constraint_memory": constraint_memory or {},
        },
        dynamic={
            "candidate_table": candidate_name,
            "source_columns": source_columns,
            "source_preview": source_preview,
            "generated_columns": generated_columns,
            "generated_preview": generated_preview,
        },
        stable_title="Stable sample submission checker rules",
        dynamic_title="Dynamic sample submission candidate",
        stable_limit=9000,
        dynamic_limit=9000,
    )
    return llm_client.ask_structured(
        model_cls=SubmissionCheckVerdict,
        system_prompt="你是严格的提交格式检查器。只能输出 JSON。",
        user_prompt=dynamic,
        prompt_name="sample_submission_checker",
        static_context_prompt=stable,
        dynamic_user_prompt=dynamic,
    )


def _schema_blocking_checker_issues(issues: list[str]) -> list[str]:
    """Keep only checker issues that should block a sample_submission schema.

    The LLM checker sometimes drifts into judging whether the example rows are
    already optimized. That is wrong for sample_submission.csv: placeholders are
    acceptable as long as the columns can express the downstream output.
    """
    blocking: list[str] = []
    value_only_markers = [
        "placeholder",
        "default",
        "not optimized",
        "not solved",
        "no actual optimization",
        "占位",
        "默认",
        "全为",
        "均为",
        "相同",
        "待优化",
        "未优化",
        "优化结果",
        "实际调度",
        "实际分配",
        "决策依据",
        "未按",
        "没有按",
        "未体现",
        "未考虑",
        "未提供任何优化",
        "运力结算",
        "装载约束",
        "业务约束",
        "求解",
        "最优",
    ]
    hard_schema_markers = [
        "column_order_mismatch",
        "empty_submission",
        "列顺序",
        "字段缺失",
        "缺少字段",
        "缺少列",
        "不包含",
        "未包含",
        "无法表达",
        "不能表达",
        "schema",
        "column",
    ]
    for issue in issues:
        text = str(issue).strip()
        lower = text.lower()
        if not text:
            continue
        is_value_only = any(marker in lower or marker in text for marker in value_only_markers)
        is_schema = any(marker in lower or marker in text for marker in hard_schema_markers)
        if is_value_only:
            continue
        if is_schema:
            blocking.append(issue)
            continue
        # Conservative default: unclear checker complaints remain blocking.
        blocking.append(issue)
    return blocking



def _try_build_submission_from_plan(
    *,
    plan: SubmissionScriptPlan,
    df: pd.DataFrame,
    data_root: Path,
) -> tuple[pd.DataFrame | None, list[str]]:
    issues: list[str] = []

    def _resolve_input_path(path_like):
        if not isinstance(path_like, (str, Path)):
            return path_like
        path = Path(path_like)
        if path.is_absolute() or path.exists():
            return path_like
        candidate = data_root / path
        if candidate.exists():
            return candidate
        for p in walk_files(data_root):
            if p.name == path.name:
                return p
        return path_like

    class _PathAwarePandas:
        def __getattr__(self, name: str):
            return getattr(pd, name)

        def read_csv(self, filepath_or_buffer, *args, **kwargs):
            resolved = _resolve_input_path(filepath_or_buffer)
            if isinstance(resolved, (str, Path)):
                return read_csv_auto(Path(resolved), *args, **kwargs)
            return pd.read_csv(resolved, *args, **kwargs)

        def read_excel(self, io, *args, **kwargs):
            return pd.read_excel(_resolve_input_path(io), *args, **kwargs)

        def read_json(self, path_or_buf, *args, **kwargs):
            return pd.read_json(_resolve_input_path(path_or_buf), *args, **kwargs)

        def read_parquet(self, path, *args, **kwargs):
            return pd.read_parquet(_resolve_input_path(path), *args, **kwargs)

    path_aware_pd = _PathAwarePandas()

    def _path_aware_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "pandas":
            return path_aware_pd
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins = vars(builtins).copy()
    safe_builtins["__import__"] = _path_aware_import
    generated_path = data_root / "sample_submission.csv"
    generated_existed_before = generated_path.exists()
    old_cwd = Path.cwd()

    def _as_dataframe(candidate) -> pd.DataFrame | None:
        if isinstance(candidate, pd.DataFrame):
            return candidate
        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                if isinstance(item, pd.DataFrame):
                    return item
        return None

    def _call_submission_builder(fn) -> tuple[pd.DataFrame | None, str | None]:
        try:
            signature = inspect.signature(fn)
            params = list(signature.parameters.values())
            positional = [
                p
                for p in params
                if p.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            required_positional = [p for p in positional if p.default is inspect.Parameter.empty]
            result = fn() if not required_positional and not has_varargs else fn(df.copy())
        except Exception as first_exc:  # noqa: BLE001
            try:
                result = fn(df.copy())
            except Exception as second_exc:  # noqa: BLE001
                return None, f"{first_exc}; retry_with_df_failed: {second_exc}"
        return _as_dataframe(result), None

    try:
        exec_vars: dict = {
            "__builtins__": safe_builtins,
            "pd": path_aware_pd,
            "df": df.copy(),
            "out_df": None,
            "data_root": str(data_root),
            "input_root": str(data_root),
        }
        os.chdir(data_root)
        exec(plan.python_code, exec_vars, exec_vars)  # noqa: S102
        out_df = _as_dataframe(exec_vars.get("out_df"))
        if not isinstance(out_df, pd.DataFrame):
            for name in ("submission", "sample_submission", "result", "output"):
                candidate_df = _as_dataframe(exec_vars.get(name))
                if candidate_df is not None:
                    out_df = candidate_df
                    break
        builder_errors: list[str] = []
        if not isinstance(out_df, pd.DataFrame):
            for name in ("generate_submission", "build_submission", "create_submission", "make_submission", "main"):
                fn = exec_vars.get(name)
                if not callable(fn):
                    continue
                candidate_df, err = _call_submission_builder(fn)
                if err:
                    builder_errors.append(f"{name}: {err}")
                    continue
                if candidate_df is not None:
                    out_df = candidate_df
                    break
        if not isinstance(out_df, pd.DataFrame) and generated_path.exists():
            out_df = read_csv_auto(generated_path)
        if not isinstance(out_df, pd.DataFrame) or out_df.empty:
            if builder_errors:
                return None, ["plan_python_output_empty_or_invalid", *builder_errors[:3]]
            return None, ["plan_python_output_empty_or_invalid"]
        expected = [str(x) for x in plan.submission_columns if str(x).strip()]
        if expected and list(out_df.columns) != expected:
            for col in expected:
                if col not in out_df.columns:
                    if col in df.columns:
                        source_values = df[col].reset_index(drop=True)
                        out_df = out_df.reset_index(drop=True)
                        out_df[col] = source_values.reindex(range(len(out_df))).fillna("").to_list()
                    else:
                        out_df[col] = 0.0 if col == expected[-1] else ""
            out_df = out_df[expected]
        return out_df, issues
    except Exception as exc:  # noqa: BLE001
        return None, [f"plan_python_exec_error: {exc}"]
    finally:
        os.chdir(old_cwd)
        if generated_path.exists() and not generated_existed_before:
            try:
                generated_path.unlink()
            except Exception:
                pass



def _generate_sample_submission(
    data_root: Path,
    run_dir: Path,
    cfg: AutoRealizeConfig,
    *,
    downstream_context: dict | None = None,
    llm_client: LLMClient,
) -> None:
    """Generate sample_submission for downstream AutoML. Requires LLM when no official sample exists."""
    target_file = run_dir / "sample_submission.csv"
    ctx = downstream_context or _infer_downstream_context(data_root, [], "", cfg)
    if target_file.exists():
        _write_submission_report(
            run_dir,
            {
                "passed": True,
                "source": "preexisting_output",
                "target_file": "sample_submission.csv",
                "reason": "target file already exists",
            },
        )
        return

    def _is_sample_submission_name(path: Path) -> bool:
        normalized = "".join(ch for ch in path.stem.lower() if ch.isalnum())
        return "samplesubmission" in normalized

    existing_samples = [
        p
        for p in walk_files(data_root)
        if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}
        and _is_sample_submission_name(p)
    ]
    if existing_samples:
        sample_src = existing_samples[0]
        if sample_src.suffix.lower() == ".csv":
            shutil.copy2(sample_src, target_file)
            _write_submission_report(
                run_dir,
                {
                    "passed": True,
                    "source": "official_sample_reused",
                    "sample_source": str(sample_src.relative_to(data_root)).replace("\\", "/"),
                    "target_file": "sample_submission.csv",
                    "columns": list(read_csv_auto(target_file, nrows=0).columns),
                },
            )
            return
        try:
            df_sample = pd.read_excel(sample_src)
            df_sample.to_csv(target_file, index=False, encoding="utf-8-sig")
            _write_submission_report(
                run_dir,
                {
                    "passed": True,
                    "source": "official_sample_reused",
                    "sample_source": str(sample_src.relative_to(data_root)).replace("\\", "/"),
                    "target_file": "sample_submission.csv",
                    "columns": [str(c) for c in df_sample.columns.tolist()],
                },
            )
            return
        except Exception:  # noqa: BLE001
            pass

    task_hint = str(ctx.get("task_hint", "")).strip()
    task_type_hint = str(ctx.get("task_type_hint", "")).strip()
    predict_name = str(ctx.get("predict_table", "")).strip()
    has_real_predict_table = bool(predict_name)
    confirmed_submission_cols = [str(x) for x in ctx.get("submission_columns", []) if str(x).strip()]
    authoritative_contract = ctx.get("authoritative_submission_contract") or {}
    if not isinstance(authoritative_contract, dict):
        authoritative_contract = {}
    authoritative_contract_defined = bool(authoritative_contract.get("is_defined"))
    task_type_lower = task_type_hint.lower()
    looks_like_rl_or_optimization = any(
        key in task_type_lower
        for key in [
            "reinforcement",
            "rl",
            "optimization",
            "optimisation",
            "planning",
            "scheduling",
            "assignment",
        ]
    )
    if not confirmed_submission_cols and (
        looks_like_rl_or_optimization or not has_real_predict_table or not authoritative_contract_defined
    ):
        _write_submission_report(
            run_dir,
            {
                "passed": True,
                "source": "skipped_no_authoritative_contract",
                "target_file": None,
                "columns": [],
                "task_type_hint": task_type_hint,
                "real_predict_table": has_real_predict_table,
                "authoritative_contract_defined": authoritative_contract_defined,
                "issues": [],
                "reason": (
                    "No official sample_submission or authoritative output contract was found. "
                    "AutoRealize will not invent a submission schema; downstream AutoML must follow description.md/evaluation protocol."
                ),
            },
        )
        log_event(
            logger,
            "checker.sample_submission",
            "SKIPPED",
            reason="no_authoritative_contract",
            task_type_hint=task_type_hint,
            real_predict_table=has_real_predict_table,
        )
        return

    table_files = [p for p in walk_files(data_root) if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}]
    if not table_files:
        _write_submission_report(
            run_dir,
            {
                "passed": True,
                "source": "skipped_no_authoritative_contract",
                "target_file": None,
                "columns": [],
                "task_type_hint": task_type_hint,
                "issues": [],
                "reason": "No table files or official sample_submission were found; no sample schema was generated.",
            },
        )
        log_event(logger, "checker.sample_submission", "SKIPPED", reason="no_table_files")
        return

    id_col = str(ctx.get("id_column", "id")).strip() or "id"
    target_col = str(ctx.get("target_column", "target")).strip() or "target"
    schema_hints = [str(x) for x in ctx.get("submission_schema_hints", []) if str(x).strip()]

    semantic = ctx.get("semantic_keys", {}) or {}

    candidate = None
    if predict_name:
        for p in table_files:
            if p.name == predict_name:
                candidate = p
                break
    if candidate is None:
        non_submission_tables = [p for p in table_files if not _is_sample_submission_name(p)]
        candidate = non_submission_tables[0] if non_submission_tables else table_files[0]

    try:
        df = read_table(
            candidate,
            json_flatten_sep=cfg.data.json_flatten_sep,
            json_flatten_max_level=cfg.data.json_flatten_max_level,
            json_keep_raw_nested_columns=cfg.data.json_keep_raw_nested_columns,
            max_rows=table_probe_sample_rows(
                candidate,
                configured_rows=cfg.data.table_profile_sample_rows,
                large_threshold_bytes=cfg.data.large_table_threshold_bytes,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        issues = [f"candidate_read_failed: {candidate.name}: {exc}"]
        _write_submission_report(
            run_dir,
            {
                "passed": False,
                "source": "skipped_generation_failed",
                "candidate_table": candidate.name,
                "target_file": None,
                "columns": [],
                "issues": issues,
                "reason": "Candidate table could not be read; sample_submission generation was skipped without interrupting AutoRealize.",
            },
        )
        log_event(logger, "checker.sample_submission", "WARNING", reason="candidate_read_failed", error=str(exc)[:240])
        return
    if df.empty:
        issues = [f"candidate_empty: {candidate.name}"]
        _write_submission_report(
            run_dir,
            {
                "passed": False,
                "source": "skipped_generation_failed",
                "candidate_table": candidate.name,
                "target_file": None,
                "columns": [],
                "issues": issues,
                "reason": "Candidate table is empty; sample_submission generation was skipped without interrupting AutoRealize.",
            },
        )
        log_event(logger, "checker.sample_submission", "WARNING", reason="candidate_empty", candidate_table=candidate.name)
        return

    if id_col not in df.columns:
        inferred_id = None
        for c in df.columns:
            lc = str(c).lower()
            if "id" in lc or "number" in lc or "order" in lc or "code" in lc:
                inferred_id = str(c)
                break
        id_col = inferred_id or str(df.columns[0])

    preview = _json_safe(df.head(min(30, len(df))).to_dict(orient="records"))
    plan_prompt = (
        "Return strict JSON with fields: purpose, submission_columns, python_code, id_column, target_columns.\n"
        "Goal: design sample_submission schema from task semantics and dataset structure.\n"
        f"candidate_table={candidate.name}\n"
        f"task_hint={task_hint}\n"
        f"task_type_hint={task_type_hint}\n"
        f"confirmed_submission_columns={confirmed_submission_cols}\n"
        f"submission_schema_hints={schema_hints}\n"
        f"semantic_keys={_json_dumps_safe(semantic, ensure_ascii=False)}\n"
        f"constraint_memory={_json_dumps_safe(ctx.get('constraint_memory', {}), ensure_ascii=False)[:4000]}\n"
        f"columns={list(df.columns)}\n"
        f"preview={_json_dumps_safe(preview, ensure_ascii=False)[:6000]}\n"
        "Constraints:\n"
        "1) If confirmed_submission_columns is non-empty, output out_df with exactly those columns in that order.\n"
        "2) If confirmed_submission_columns is empty, design the schema from the task semantics; submission_schema_hints are non-binding hints, not a required template.\n"
        "3) Do not force id+target or any fixed template when optimization/recommendation/custom tasks need richer output fields.\n"
        "4) Use original field names from data whenever they accurately express the output meaning.\n"
        "5) Reflect critical constraints from constraint_memory in submission field design.\n"
        "6) python_code should use the provided DataFrame variable df and assign the final DataFrame to out_df.\n"
        "7) do not read input files or write output files unless absolutely necessary; the execution sandbox will save out_df.\n"
    )

    def _align_confirmed_columns(plan: SubmissionScriptPlan) -> SubmissionScriptPlan:
        if confirmed_submission_cols and plan.submission_columns != confirmed_submission_cols:
            return SubmissionScriptPlan(
                purpose=(plan.purpose or "") + " | aligned_to_confirmed_submission_columns",
                submission_columns=confirmed_submission_cols,
                python_code=plan.python_code,
                id_column=plan.id_column,
                target_columns=[c for c in confirmed_submission_cols[1:]],
            )
        return plan

    def _write_generation_skipped(
        *,
        issues: list[str],
        reason: str,
        round_idx: int | None = None,
        generated_columns: list[str] | None = None,
        generated_preview: list[dict] | None = None,
        non_blocking_checker_warnings: list[str] | None = None,
    ) -> None:
        _write_submission_report(
            run_dir,
            {
                "passed": False,
                "source": "skipped_generation_failed",
                "candidate_table": candidate.name,
                "task_type_hint": task_type_hint,
                "target_file": None,
                "columns": generated_columns or [],
                "preview": (generated_preview or [])[:5],
                "issues": issues,
                "non_blocking_checker_warnings": non_blocking_checker_warnings or [],
                "round": round_idx,
                "reason": reason,
            },
        )
        log_event(
            logger,
            "checker.sample_submission",
            "WARNING",
            reason="generation_skipped",
            issues=issues[:5],
            round=round_idx,
        )

    def _plan_with_feedback(
        *,
        base_prompt: str,
        previous_plan: SubmissionScriptPlan | None,
        feedback: list[str],
        round_idx: int,
        generated_columns: list[str] | None = None,
        generated_preview: list[dict] | None = None,
    ) -> SubmissionScriptPlan:
        repair_payload = {
            "instruction": (
                "The previous sample_submission plan failed validation or execution. "
                "Regenerate the strict JSON plan. Do not repeat the same mistake."
            ),
            "repair_round": round_idx,
            "feedback_errors": feedback,
            "previous_plan": previous_plan.model_dump() if previous_plan else {},
            "previous_generated_columns": generated_columns or [],
            "previous_generated_preview": generated_preview or [],
            "hard_rules": [
                "python_code must assign a non-empty pandas DataFrame to out_df, or define a callable that returns one.",
                "Do not read files by bare relative names unless the file exists under data_root; prefer using the provided df.",
                "If confirmed_submission_columns is non-empty, output exactly those columns in that order.",
                "If no authoritative contract exists, do not invent a fixed id+target template.",
                "Use original column names when they express the required output semantics.",
            ],
        }
        plan = llm_client.ask_structured(
            model_cls=SubmissionScriptPlan,
            system_prompt="You are a schema planner repairing sample_submission generation. Output JSON only.",
            user_prompt=_json_dumps_safe(repair_payload, ensure_ascii=False, indent=2)[:10000],
            prompt_name=f"sample_submission_script_plan_repair_{round_idx}",
            static_context_prompt=base_prompt,
            dynamic_user_prompt="Repair feedback JSON:\n"
            + _json_dumps_safe(repair_payload, ensure_ascii=False, indent=2)[:10000],
        )
        return _align_confirmed_columns(plan)

    try:
        current_plan = _align_confirmed_columns(llm_client.ask_structured(
            model_cls=SubmissionScriptPlan,
            system_prompt="You are a schema planner for sample_submission. Output JSON only.",
            user_prompt="Generate the initial sample_submission schema plan.",
            prompt_name="sample_submission_script_plan",
            static_context_prompt=plan_prompt,
            dynamic_user_prompt="Generate the initial sample_submission schema plan.",
        ))

        max_repair_rounds = max(2, int(getattr(cfg.prompt, "description_quality_max_retries", 3)))
        final_rejection_issues: list[str] = []
        for round_idx in range(max_repair_rounds + 1):
            out_df, plan_issues = _try_build_submission_from_plan(plan=current_plan, df=df, data_root=data_root)
            if out_df is None:
                final_rejection_issues = plan_issues
                log_event(
                    logger,
                    "checker.sample_submission",
                    "REPAIRING",
                    reason="plan_python_failed",
                    round=round_idx,
                    issues=plan_issues[:5],
                )
                if round_idx >= max_repair_rounds:
                    _write_generation_skipped(
                        issues=final_rejection_issues,
                        reason="LLM sample_submission code did not produce a valid DataFrame after repair rounds.",
                        round_idx=round_idx,
                    )
                    return
                current_plan = _plan_with_feedback(
                    base_prompt=plan_prompt,
                    previous_plan=current_plan,
                    feedback=["Generated python did not produce a valid non-empty DataFrame.", *plan_issues],
                    round_idx=round_idx + 1,
                )
                continue

            rule_issues = _validate_generated_submission(out_df, ctx=ctx, source_df=df)
            generated_cols = [str(c) for c in out_df.columns.tolist()]
            generated_preview = _json_safe(out_df.head(min(20, len(out_df))).to_dict(orient="records"))

            try:
                checker = _evaluate_submission_with_llm(
                    llm_client=llm_client,
                    task_hint=task_hint,
                    task_type_hint=task_type_hint,
                    candidate_name=candidate.name,
                    source_columns=[str(c) for c in df.columns.tolist()],
                    source_preview=preview,
                    generated_columns=generated_cols,
                    generated_preview=generated_preview,
                    semantic_keys=semantic,
                    constraint_memory=ctx.get("constraint_memory", {}),
                )
            except Exception as exc:  # noqa: BLE001
                final_rejection_issues = [f"sample_submission_checker_failed: {exc}"]
                _write_generation_skipped(
                    issues=final_rejection_issues,
                    reason="LLM checker failed; sample_submission generation was skipped without interrupting AutoRealize.",
                    round_idx=round_idx,
                    generated_columns=generated_cols,
                    generated_preview=generated_preview,
                )
                return

            checker_issues = list(checker.issues or [])
            blocking_checker_issues = _schema_blocking_checker_issues(checker_issues)
            hard_rule_issues = [x for x in rule_issues if not x.startswith("hint_")]
            if (checker.passed and not checker.needs_regenerate or not blocking_checker_issues) and not hard_rule_issues:
                if not has_real_predict_table:
                    out_df = out_df.head(max(1, int(cfg.data.generated_sample_submission_max_rows))).copy()
                out_df.to_csv(target_file, index=False, encoding="utf-8-sig")
                validation = {
                    "passed": True,
                    "issues": blocking_checker_issues + hard_rule_issues + plan_issues,
                    "non_blocking_checker_warnings": [
                        x for x in checker_issues if x not in blocking_checker_issues
                    ],
                    "source": "llm_plan+llm_checker",
                    "round": round_idx,
                    "real_predict_table": has_real_predict_table,
                    "sample_rows_only": not has_real_predict_table,
                }
                (run_dir / "realize_report" / "sample_submission_validation.json").write_text(
                    _json_dumps_safe(validation, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _write_submission_report(
                    run_dir,
                    {
                        "passed": True,
                        "source": "llm_plan+llm_checker",
                        "candidate_table": candidate.name,
                        "task_type_hint": task_type_hint,
                        "purpose": current_plan.purpose,
                        "columns": generated_cols,
                        "preview": generated_preview[:5],
                        "issues": blocking_checker_issues + hard_rule_issues + plan_issues,
                        "non_blocking_checker_warnings": [
                            x for x in checker_issues if x not in blocking_checker_issues
                        ],
                        "round": round_idx,
                        "target_file": "sample_submission.csv",
                        "real_predict_table": has_real_predict_table,
                        "sample_rows_only": not has_real_predict_table,
                    },
                )
                log_event(logger, "checker.sample_submission", "COMPLETED", passed=True)
                return

            repair_feedback = blocking_checker_issues + hard_rule_issues + plan_issues
            if round_idx >= max_repair_rounds:
                final_rejection_issues = repair_feedback
                _write_generation_skipped(
                    issues=final_rejection_issues,
                    reason="LLM checker did not approve a valid sample_submission schema after repair rounds.",
                    round_idx=round_idx,
                    generated_columns=generated_cols,
                    generated_preview=generated_preview,
                    non_blocking_checker_warnings=[
                        x for x in checker_issues if x not in blocking_checker_issues
                    ],
                )
                return

            if blocking_checker_issues and checker.needs_regenerate and checker.revised_python_code.strip():
                revised_cols = [str(x) for x in checker.revised_submission_columns if str(x).strip()]
                current_plan = SubmissionScriptPlan(
                    purpose=(current_plan.purpose or "") + " | revised_by_checker",
                    submission_columns=revised_cols or current_plan.submission_columns,
                    python_code=checker.revised_python_code,
                    id_column=current_plan.id_column,
                    target_columns=current_plan.target_columns,
                )
            else:
                log_event(
                    logger,
                    "checker.sample_submission",
                    "REPAIRING",
                    reason="checker_rejected_without_repair_code",
                    round=round_idx,
                    issues=repair_feedback[:5],
                )
                current_plan = _plan_with_feedback(
                    base_prompt=plan_prompt,
                    previous_plan=current_plan,
                    feedback=[
                        "Checker rejected the generated sample_submission but did not provide repair code.",
                        *repair_feedback,
                    ],
                    round_idx=round_idx + 1,
                    generated_columns=generated_cols,
                    generated_preview=generated_preview,
                )
    except Exception as exc:
        _write_generation_skipped(
            issues=[str(exc)],
            reason="Unexpected sample_submission generation failure; stage was skipped without interrupting AutoRealize.",
        )
        return


def _write_submission_report(run_dir: Path, payload: dict) -> None:
    report_dir = run_dir / "realize_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "autorealize.submission_report.v1",
        **payload,
    }
    (report_dir / "submission_report.json").write_text(
        _json_dumps_safe(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _infer_downstream_context(
    data_root: Path,
    file_summaries: list[FileSummary],
    task_hint: str,
    cfg: AutoRealizeConfig,
) -> dict:
    """Infer train/predict/label semantics and non-binding submission hints."""
    table_paths = [p for p in walk_files(data_root) if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}]
    hint_lower = task_hint.lower()
    placeholders = {"", "nan", "none", "null", "na", "n/a", "unknown", "?"}
    label_priority = ["target", "label", "y"]

    def _read_small_table(path: Path, nrows: int = 200) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return read_csv_auto(path, nrows=nrows)
        if path.suffix.lower() == ".json":
            from .utils.json_table import read_json_as_table

            df, _ = read_json_as_table(
                path,
                sep=cfg.data.json_flatten_sep,
                max_level=cfg.data.json_flatten_max_level,
                keep_raw_nested_columns=cfg.data.json_keep_raw_nested_columns,
            )
            return df.head(nrows)
        return pd.read_excel(path, nrows=nrows)

    def _label_score(series: pd.Series) -> float:
        if series is None or len(series) == 0:
            return 0.0
        s = series.copy()
        st = s.astype(str).str.strip().str.lower()
        is_missing = s.isna() | st.isin(placeholders)
        non_missing = int((~is_missing).sum())
        if non_missing == 0:
            return 0.0
        uniq = int(st[~is_missing].nunique())
        if uniq == 0:
            return 0.0
        return non_missing / max(len(s), 1)

    def _unique_ratio(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return 0.0
        s = df[col]
        if len(s) == 0:
            return 0.0
        nn = s.dropna()
        if len(nn) == 0:
            return 0.0
        return float(nn.nunique()) / float(len(nn))

    def _best_col_by_keywords(columns: list[str], keyword_groups: list[list[str]]) -> str:
        lower = [str(c).lower() for c in columns]
        for group in keyword_groups:
            if group == ["id"]:
                for idx, lc in enumerate(lower):
                    raw = str(columns[idx])
                    if lc == "id" or lc.endswith("_id") or raw.endswith("编号") or raw.endswith("代码") or raw.endswith("编码"):
                        return raw
                continue
            for idx, lc in enumerate(lower):
                if all(k in lc for k in group):
                    return str(columns[idx])
            for idx, lc in enumerate(lower):
                if any(k in lc for k in group):
                    return str(columns[idx])
        for c in columns:
            cs = str(c)
            for group in keyword_groups:
                if any(k in cs for k in group):
                    return cs
        return ""

    def _best_group_key(columns: list[str]) -> str:
        lower = [str(c).lower() for c in columns]
        strong_patterns = [
            ["group", "id"],
            ["origin", "order"],
            ["parent", "id"],
            ["batch", "id"],
        ]
        for pat in strong_patterns:
            for i, lc in enumerate(lower):
                if all(k in lc for k in pat):
                    return str(columns[i])

        for c in columns:
            cs = str(c)
            if ("原始" in cs and ("订单" in cs or "单号" in cs)) or ("父级" in cs) or ("批次" in cs):
                return cs
        return ""

    def _best_id_column(columns: list[str], df: pd.DataFrame) -> str:
        candidates: list[tuple[float, str]] = []
        for c in columns:
            name = str(c)
            lc = name.lower()
            score = 0.0
            if any(x in lc for x in ["origin", "group", "parent", "batch"]):
                score -= 1.5
            if "id" == lc or lc.endswith("_id"):
                score += 4.0
            if "order" in lc or "request" in lc or "code" in lc or "number" in lc:
                score += 2.5
            if any(k in name for k in ["编号", "代码", "编码", "单号", "订单", "请求", "主键"]):
                score += 2.5
            if score <= 0:
                continue
            score += _unique_ratio(df, name) * 3.0
            candidates.append((score, name))
        if not candidates:
            return ""
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][1]

    def _best_time_column(columns: list[str]) -> str:
        for c in columns:
            lc = str(c).lower()
            if lc.startswith("dt") or any(k in lc for k in ["date", "time", "month", "day"]):
                return str(c)
            if any(k in str(c) for k in ["日期", "时间", "月份", "每日"]):
                return str(c)
        return ""

    def _is_train_filename(name_lower: str) -> bool:
        return re.search(r"(^|[_\-.])(train|training)([_\-.]|$)", name_lower) is not None

    def _is_test_filename(name_lower: str) -> bool:
        return re.search(r"(^|[_\-.])(test|testing)([_\-.]|$)", name_lower) is not None

    table_infos: list[dict] = []
    # Only official sample/submission files may populate this hard contract.
    submission_columns: list[str] = []
    for table in table_paths:
        try:
            df = _read_small_table(table, nrows=500)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue

        columns = [str(c) for c in df.columns.tolist()]
        lower_cols = {c.lower(): c for c in columns}
        name_lower = table.name.lower()
        is_train_name = _is_train_filename(name_lower)
        is_test_name = _is_test_filename(name_lower)
        is_submission = (
            "sample_submission" in name_lower
            or "samplesubmission" in name_lower
            or name_lower.startswith("submission")
            or "_submission" in name_lower
        )
        if is_submission and not submission_columns:
            submission_columns = columns

        label_candidates: list[tuple[str, float]] = []
        for lp in label_priority:
            if lp in lower_cols:
                col = lower_cols[lp]
                label_candidates.append((col, _label_score(df[col])))
        for c in columns:
            lc = c.lower()
            if lc in {"target", "label", "y"} and lc not in label_priority:
                label_candidates.append((c, _label_score(df[c])))
        label_candidates = [(c, s) for c, s in label_candidates if s > 0.0]
        label_col = ""
        label_score = 0.0
        if label_candidates:
            label_col, label_score = sorted(label_candidates, key=lambda x: (-x[1], x[0]))[0]

        entity_id_key = _best_col_by_keywords(
            columns,
            [["order", "id"], ["request", "id"], ["entity", "id"], ["单", "号"], ["订单"], ["id"]],
        )
        group_id_key = _best_group_key(columns)
        resource_key_1 = _best_col_by_keywords(
            columns,
            [["carrier", "code"], ["provider", "code"], ["vendor", "code"], ["承运", "商"], ["供应", "商"]],
        )
        resource_key_2 = _best_col_by_keywords(
            columns,
            [["vehicle", "type"], ["truck", "type"], ["resource", "type"], ["车型"], ["车辆"]],
        )

        id_col = _best_id_column(columns, df)
        if entity_id_key:
            id_col = entity_id_key
        if group_id_key and group_id_key == entity_id_key:
            group_id_key = ""

        table_infos.append(
            {
                "path": str(table),
                "name": table.name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": columns,
                "id_col": id_col,
                "entity_id_key": entity_id_key,
                "group_id_key": group_id_key,
                "resource_key_1": resource_key_1,
                "resource_key_2": resource_key_2,
                "time_col": _best_time_column(columns),
                "is_train_name": is_train_name,
                "is_test_name": is_test_name,
                "is_submission": is_submission,
                "label_col": label_col,
                "label_score": float(label_score),
                "has_usable_label": bool(label_col),
            }
        )

    has_named_train = any(t["is_train_name"] for t in table_infos)
    has_named_test = any(t["is_test_name"] for t in table_infos)

    named_train_infos = [t for t in table_infos if t["is_train_name"] and not t["is_submission"]]
    named_test_infos = [t for t in table_infos if t["is_test_name"] and not t["is_submission"]]
    train_schema_hint = sorted(named_train_infos, key=lambda t: -t["rows"])[0] if named_train_infos else None
    test_schema_hint = sorted(named_test_infos, key=lambda t: -t["rows"])[0] if named_test_infos else None

    if train_schema_hint and test_schema_hint and not train_schema_hint["has_usable_label"]:
        test_cols = set(test_schema_hint["columns"])
        train_only = [c for c in train_schema_hint["columns"] if c not in test_cols]
        if len(train_only) == 1:
            label_col = train_only[0]
            train_schema_hint["label_col"] = label_col
            train_schema_hint["label_score"] = _label_score(_read_small_table(Path(train_schema_hint["path"]), nrows=500)[label_col])
            train_schema_hint["has_usable_label"] = True
        else:
            scored_same_name: list[tuple[float, str]] = []
            test_df = _read_small_table(Path(test_schema_hint["path"]), nrows=500)
            train_df = _read_small_table(Path(train_schema_hint["path"]), nrows=500)
            for c in train_schema_hint["columns"]:
                if c not in test_cols or c not in train_df.columns or c not in test_df.columns:
                    continue
                train_score = _label_score(train_df[c])
                test_score = _label_score(test_df[c])
                if train_score > 0 and test_score == 0:
                    scored_same_name.append((train_score, c))
            if scored_same_name:
                _, label_col = sorted(scored_same_name, key=lambda x: (-x[0], x[1]))[0]
                train_schema_hint["label_col"] = label_col
                train_schema_hint["label_score"] = float(_label_score(_read_small_table(Path(train_schema_hint["path"]), nrows=500)[label_col]))
                train_schema_hint["has_usable_label"] = True

    train_table = None
    train_named = [t for t in table_infos if t["is_train_name"] and t["has_usable_label"]]
    if train_named:
        train_table = sorted(train_named, key=lambda t: (-t["label_score"], -t["rows"]))[0]
    else:
        any_labeled = [t for t in table_infos if t["has_usable_label"] and not t["is_submission"]]
        if has_named_test:
            any_labeled = [t for t in any_labeled if not t["is_test_name"]]
        if any_labeled:
            train_table = sorted(any_labeled, key=lambda t: (-t["label_score"], -t["rows"]))[0]

    pred_table = None
    test_no_label = [t for t in table_infos if t["is_test_name"] and not t["has_usable_label"] and not t["is_submission"]]
    if test_no_label:
        pred_table = sorted(test_no_label, key=lambda t: -t["rows"])[0]
    else:
        any_no_label = [t for t in table_infos if not t["has_usable_label"] and not t["is_submission"]]
        if has_named_train:
            any_no_label = [t for t in any_no_label if not t["is_train_name"]]
        # 只有一个普通业务表时，它更可能是历史训练/分析表，而不是独立预测集。
        # 预测集必须由 test/predict 命名、缺失标签字段、或官方说明明确指向；这里不凭空制造 predict_table。
        if len([t for t in table_infos if not t["is_submission"]]) <= 1:
            any_no_label = []
        if any_no_label:
            pred_table = sorted(any_no_label, key=lambda t: -t["rows"])[0]

    if train_table is None and pred_table is None and table_infos:
        non_submission = [t for t in table_infos if not t["is_submission"]]
        if non_submission:
            train_table = sorted(non_submission, key=lambda t: -t["rows"])[0]

    id_column = pred_table["id_col"] if pred_table else (train_table["id_col"] if train_table else "")
    target_column = train_table["label_col"] if train_table and train_table.get("has_usable_label") else ""
    y_true_field = target_column

    entity_id_key = ""
    group_id_key = ""
    resource_keys: list[str] = []
    for t in [pred_table, train_table]:
        if not t:
            continue
        if not entity_id_key and t.get("entity_id_key"):
            entity_id_key = str(t.get("entity_id_key"))
        if not group_id_key and t.get("group_id_key"):
            group_id_key = str(t.get("group_id_key"))
        for rk in [t.get("resource_key_1"), t.get("resource_key_2")]:
            if rk and str(rk) not in resource_keys:
                resource_keys.append(str(rk))

    optimization_hint = any(k in hint_lower for k in [
        "optimization", "dispatch", "matching", "routing", "assignment", "schedule", "plan",
        "优化", "调度", "匹配", "分配", "排程", "规划",
    ])

    if train_table and train_table.get("time_col") and train_table["has_usable_label"]:
        task_type_hint = "time_series_regression"
    elif optimization_hint:
        task_type_hint = "optimization"
    elif any(k in hint_lower for k in ["classification", "class", "分类", "类别", "true", "false"]):
        task_type_hint = "binary_classification"
    elif train_table and train_table["has_usable_label"]:
        task_type_hint = "regression"
    else:
        task_type_hint = "optimization_or_rl"

    submission_schema_hints: list[str] = []
    if optimization_hint or entity_id_key or group_id_key or resource_keys:
        for c in [entity_id_key or id_column, group_id_key, *resource_keys]:
            if c and c not in submission_schema_hints:
                submission_schema_hints.append(c)
    elif id_column:
        submission_schema_hints.append(id_column)
    if target_column and target_column not in submission_schema_hints:
        submission_schema_hints.append(target_column)

    train_columns = train_table["columns"] if train_table else []
    predict_columns = pred_table["columns"] if pred_table else []
    train_only_columns = [c for c in train_columns if c not in predict_columns and c != target_column]
    predict_only_columns = [c for c in predict_columns if c not in train_columns]

    def _train_table_evidence(table: dict | None) -> str:
        if not table:
            return "unknown"
        if table.get("is_train_name") and table.get("has_usable_label"):
            return "named_train_table_with_label"
        if table.get("has_usable_label"):
            return "label_column_heuristic"
        return "fallback_largest_non_submission_table_heuristic"

    def _predict_table_evidence(table: dict | None) -> str:
        if not table:
            return "unknown"
        if table.get("is_test_name") and not table.get("has_usable_label"):
            return "named_test_or_predict_table_without_label"
        return "multi_table_without_label_heuristic"

    evidence_levels = {
        "train_table": _train_table_evidence(train_table),
        "predict_table": _predict_table_evidence(pred_table),
        "id_column": "column_name_and_uniqueness_heuristic" if id_column else "unknown",
        "target_column": "label_column_heuristic" if target_column else "unknown",
        "submission_columns": "official_submission_file" if submission_columns else "unknown",
        "task_type_hint": "task_hint_keyword_or_schema_heuristic" if task_type_hint else "unknown",
    }
    heuristic_fields = [
        name
        for name, evidence in evidence_levels.items()
        if evidence.endswith("_heuristic") or "heuristic" in evidence
    ]

    return {
        "task_hint": task_hint,
        "id_column": id_column,
        "target_column": target_column,
        "y_true_field": y_true_field,
        "submission_columns": submission_columns,
        "submission_schema_hints": submission_schema_hints,
        "task_type_hint": task_type_hint,
        "semantic_keys": {
            "entity_id_key": entity_id_key or id_column,
            "group_id_key": group_id_key,
            "resource_keys": resource_keys,
        },
        "has_official_test_labels": False,
        "detected_tables": [t["name"] for t in table_infos][:20]
        or [fs.path for fs in file_summaries if fs.role == FileRole.raw_data_table][:20],
        "train_table": train_table["name"] if train_table else "",
        "predict_table": pred_table["name"] if pred_table else "",
        "evidence_levels": evidence_levels,
        "heuristic_fields": heuristic_fields,
        "train_columns": train_columns[:200],
        "predict_columns": predict_columns[:200],
        "train_only_columns": train_only_columns[:200],
        "predict_only_columns": predict_only_columns[:200],
    }


def _classify_task_type(
    llm_client: LLMClient,
    prompt_mgr: PromptManager,
    task_hint: str,
    data_digest: str,
    downstream_context: dict,
    enable_fewshot: bool = False,
) -> TaskClassification:
    system = prompt_mgr.load("system/task_classifier.md")
    fewshot = prompt_mgr.load("fewshot/task_classifier_fewshot.json") if enable_fewshot else ""
    light_ctx = {
        "task_hint": task_hint,
        "train_table": downstream_context.get("train_table", ""),
        "predict_table": downstream_context.get("predict_table", ""),
        "id_column": downstream_context.get("id_column", ""),
        "target_column": downstream_context.get("target_column", ""),
        "task_type_hint_pre": downstream_context.get("task_type_hint", ""),
        "confirmed_submission_columns": downstream_context.get("submission_columns", []),
        "submission_schema_hints": downstream_context.get("submission_schema_hints", []),
        "authoritative_memory": downstream_context.get("authoritative_memory", {}),
        "authoritative_submission_contract": downstream_context.get("authoritative_submission_contract", {}),
        "context_priority_order": (downstream_context.get("agent_context_pack") or {}).get("priority_order", []),
        "do_not_invent": downstream_context.get("do_not_invent", []),
        "task_classifier_route": (downstream_context.get("context_routes") or {}).get("task_classifier", {}),
        "train_columns": downstream_context.get("train_columns", [])[:80],
        "predict_columns": downstream_context.get("predict_columns", [])[:80],
    }
    stable, dynamic = stable_dynamic_prompt(
        stable={
            "task_hint": task_hint,
            "data_digest": data_digest[:5000],
            "structured_clues": light_ctx,
        },
        dynamic={"instruction": "Classify the task type and evaluation hints from the stable evidence."},
        stable_title="Stable task classification evidence",
        dynamic_title="Dynamic task classification request",
    )
    return llm_client.ask_structured(
        model_cls=TaskClassification,
        system_prompt=system,
        user_prompt=dynamic,
        prompt_name="task_classifier",
        fewshot=fewshot,
        static_context_prompt=stable,
        dynamic_user_prompt=dynamic,
    )


def _select_cognition_files(data_root: Path, config: AutoRealizeConfig) -> tuple[list[Path], dict[str, list[Path]]]:
    files = walk_files(data_root)
    by_dir: dict[str, list[Path]] = {}
    for f in files:
        by_dir.setdefault(rel(f.parent, data_root), []).append(f)
    selected: list[Path] = []
    compact_image_dirs: dict[str, list[Path]] = {}
    for drel, group in by_dir.items():
        image_files = [p for p in group if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}]
        non_images = [p for p in group if p not in image_files]
        selected.extend(non_images)
        if len(image_files) > config.data.image_dir_compact_threshold:
            sample_count = max(1, config.data.image_dir_sample_file_count)
            samples = sorted(image_files)[:sample_count]
            selected.extend(samples)
            compact_image_dirs[drel] = samples
        else:
            selected.extend(image_files)
    return sorted(set(selected)), compact_image_dirs


def _should_run_llm_file_cognition(
    *,
    config: AutoRealizeConfig,
    parsed_kind: str,
    role: FileRole,
    relative_path: str,
    metadata: dict,
    allow_hint: bool | None = None,
) -> bool:
    mode = str(getattr(config.data, "llm_file_cognition_mode", "all") or "all").lower()
    if mode in {"all", "selective"}:
        return True
    if mode in {"none", "off", "disabled"}:
        return False

    document_like = parsed_kind in {"document", "structured_document", "archive"}
    if mode == "documents_only":
        return document_like or role == FileRole.task_requirement

    if document_like or role == FileRole.task_requirement:
        return True
    if bool(allow_hint):
        return True

    # In selective mode, keep LLM for files where deterministic reading is most
    # likely to be insufficient or dangerous for downstream AutoML.
    if parsed_kind == "table":
        dialect = metadata.get("csv_dialect")
        if isinstance(dialect, dict) and dialect.get("inferred"):
            return True
        sheets = metadata.get("excel_sheet_names") or []
        if isinstance(sheets, list) and len(sheets) > 1:
            return True
        lower = str(relative_path).lower()
        important_name_markers = [
            "sample_submission",
            "submission",
            "description",
            "readme",
            "requirement",
            "requirements",
            "spec",
            "task",
            "规则",
            "需求",
            "任务",
            "说明",
            "方案",
        ]
        return any(marker in lower for marker in important_name_markers)
    return False


def _cognize_one_file(
    *,
    file: Path,
    data_root: Path,
    registry,
    config: AutoRealizeConfig,
    llm_client: LLMClient,
    prompt_mgr: PromptManager,
    task_hint: str,
    allow_llm_cognition: bool | None = None,
) -> dict:
    rpath = rel(file, data_root)
    log_event(logger, "stage.P1", "READING_FILE", file=rpath)
    t0 = time.perf_counter()
    try:
        parsed = registry.parse(file)
        dt = time.perf_counter() - t0
        log_event(logger, "stage.P1", "READ_COMPLETED", file=rpath, file_type=parsed.kind, seconds=f"{dt:.2f}")
        role = _infer_role(rpath, parsed.kind, parsed.text_summary)
        warnings = []
        if parsed.kind == "table" and parsed.metadata.get("shape", [0, 0])[0] == 0:
            warnings.append("空表")
        fs = FileSummary(
            path=rpath,
            role=role,
            summary=parsed.text_summary[:600],
            columns=parsed.columns,
            warnings=warnings,
            source_metadata={
                **(parsed.metadata or {}),
                "kind": parsed.kind,
                "preview": _json_safe((parsed.preview or [])[: max(1, int(config.data.preview_rows))]),
            },
        )
        if parsed.kind == "table":
            try:
                df_stats = read_table(
                    file,
                    json_flatten_sep=config.data.json_flatten_sep,
                    json_flatten_max_level=config.data.json_flatten_max_level,
                    json_keep_raw_nested_columns=config.data.json_keep_raw_nested_columns,
                    max_rows=table_probe_sample_rows(
                        file,
                        configured_rows=config.data.table_profile_sample_rows,
                        large_threshold_bytes=config.data.large_table_threshold_bytes,
                    ),
                )
                prof = profile_dataframe(df_stats, top_k=config.data.category_top_k)
                sample_rows = table_probe_sample_rows(
                    file,
                    configured_rows=config.data.table_profile_sample_rows,
                    large_threshold_bytes=config.data.large_table_threshold_bytes,
                )
                fs.source_metadata["profile_sampling"] = table_sampling_metadata(
                    file,
                    configured_rows=config.data.table_profile_sample_rows,
                    large_threshold_bytes=config.data.large_table_threshold_bytes,
                    rows_read=len(df_stats),
                )
                if file.suffix.lower() in {".xlsx", ".xls"}:
                    excel_profiles = profile_excel_sheets(
                        file,
                        max_rows=sample_rows,
                        top_k=config.data.category_top_k,
                        preview_rows=config.data.preview_rows,
                        large_threshold_bytes=config.data.large_table_threshold_bytes,
                        full_profile_sheet_threshold=10,
                        representatives_per_group=config.data.pattern_sample_file_count,
                    )
                    fs.source_metadata["excel_sheet_profiles"] = excel_profiles
                    fs.source_metadata["excel_sheet_groups"] = excel_sheet_groups_from_profiles(excel_profiles)
                if sample_rows is not None:
                    fs.warnings.append(f"字段统计基于前 {min(sample_rows, len(df_stats))} 行采样，未全量扫描。")
                fs.column_profiles = [
                    {
                        **column_profile_to_dict(p),
                        "top_values": p.top_values[:12],
                        "value_pattern_hints": p.value_pattern_hints[:8],
                        "abnormal_tokens": p.abnormal_tokens[:8],
                    }
                    for p in prof
                ]
                fs.column_semantics = _guess_column_semantics(
                    columns=parsed.columns,
                    profiles=fs.column_profiles,
                    task_hint=task_hint,
                )
                fs.column_semantic_meta = _guess_semantic_meta(
                    columns=parsed.columns,
                    profiles=fs.column_profiles,
                    semantics=fs.column_semantics,
                    source="heuristic",
                )
            except Exception as stats_exc:  # noqa: BLE001
                fs.warnings.append(f"字段统计失败: {stats_exc}")
        if parsed.kind == "image":
            log_event(logger, "stage.P1.image", "ACTIVATED", file=rpath)
            image_semantic = _infer_single_image_purpose(file, config, llm_client=llm_client)
            if image_semantic:
                fs.summary = f"{image_semantic} | {parsed.text_summary[:200]}"
                log_event(logger, "stage.P1.image", "COMPLETED", file=rpath, semantic_summary=True)
            else:
                log_event(logger, "stage.P1.image", "COMPLETED", file=rpath, semantic_summary=False)
        _fill_deterministic_file_memory(fs, parsed.kind)
        should_run_llm = _should_run_llm_file_cognition(
            config=config,
            parsed_kind=parsed.kind,
            role=role,
            relative_path=rpath,
            metadata=parsed.metadata or {},
            allow_hint=allow_llm_cognition,
        )
        if parsed.kind in {"table", "document", "structured_document", "archive"} and should_run_llm:
            log_event(logger, "agent.file_cognition", "CREATED", file=rpath, kind=parsed.kind)
            log_event(logger, "agent.file_cognition", "ACTIVATED", file=rpath)
            try:
                fs_llm = llm_cognition_for_file(
                    cfg=config,
                    llm=llm_client,
                    prompt_mgr=prompt_mgr,
                    file_path=file,
                    relative_path=rpath,
                    parsed_kind=parsed.kind,
                    parsed_text_summary=parsed.text_summary,
                    parsed_columns=parsed.columns,
                    parsed_preview=parsed.preview,
                    task_hint=task_hint,
                    source_metadata=fs.source_metadata,
                    column_profiles=fs.column_profiles,
                    heuristic_field_semantics=fs.column_semantics,
                )
                if fs_llm is not None:
                    base_columns = fs.columns[:]
                    base_warnings = fs.warnings[:]
                    base_semantics = dict(fs.column_semantics)
                    base_profiles = list(fs.column_profiles)
                    base_source_metadata = dict(fs.source_metadata or {})
                    fs = fs_llm
                    fs.source_metadata = {
                        **(parsed.metadata or {}),
                        **base_source_metadata,
                        **(fs.source_metadata or {}),
                    }
                    if parsed.kind == "table" and base_columns:
                        # Keep the full physical schema for data files. LLM key_columns are useful
                        # for focus, but must not replace the complete field list consumed by AutoML.
                        fs.columns = base_columns
                    elif not fs.columns:
                        fs.columns = base_columns
                    if base_warnings:
                        fs.warnings = list(dict.fromkeys((fs.warnings or []) + base_warnings))
                    if base_semantics:
                        merged = dict(base_semantics)
                        merged.update(fs.column_semantics or {})
                        fs.column_semantics = merged
                    if not getattr(fs, "column_semantic_meta", None):
                        fs.column_semantic_meta = {}
                    if base_profiles:
                        base_meta = _guess_semantic_meta(
                            columns=base_columns,
                            profiles=base_profiles,
                            semantics=base_semantics,
                            source="heuristic",
                        )
                        for k, v in base_meta.items():
                            if k not in fs.column_semantic_meta:
                                fs.column_semantic_meta[k] = v
                    if base_profiles and not fs.column_profiles:
                        fs.column_profiles = base_profiles
                    if not fs.summary:
                        fs.summary = parsed.text_summary[:600]
            except Exception as llm_exc:  # noqa: BLE001
                log_event(logger, "agent.file_cognition", "FAILED", file=rpath, error=str(llm_exc)[:180])
                raise RuntimeError(f"LLM file cognition failed for {rpath}: {llm_exc}") from llm_exc
            log_event(logger, "agent.file_cognition", "COMPLETED", file=rpath)
        elif parsed.kind in {"table", "document", "structured_document", "archive"}:
            log_event(
                logger,
                "agent.file_cognition",
                "SKIPPED",
                file=rpath,
                kind=parsed.kind,
                reason="llm_file_cognition_mode",
            )
        return {
            "rpath": rpath,
            "fs": fs,
            "columns": parsed.columns,
            "is_requirement": (role == FileRole.task_requirement),
            "summary_text": parsed.text_summary if role == FileRole.task_requirement else "",
        }
    except Exception as exc:  # noqa: BLE001
        dt = time.perf_counter() - t0
        log_event(logger, "stage.P1", "READ_FAILED", file=rpath, seconds=f"{dt:.2f}", error=str(exc)[:180])
        return {
            "rpath": rpath,
            "fs": FileSummary(
                path=rpath,
                role=FileRole.unknown,
                summary=f"解析失败: {exc}",
                warnings=["解析失败，已跳过"],
            ),
            "columns": [],
            "is_requirement": False,
            "summary_text": "",
        }


def _infer_image_dir_purpose(
    data_root: Path,
    dir_rel: str,
    sample_files: list[Path],
    config: AutoRealizeConfig,
    llm_client: LLMClient | None = None,
) -> str:
    base_summary = (
        f"目录 `{dir_rel}` 含大量图片文件，推断为图像样本目录；"
        f"已抽样 {len(sample_files)} 张图片用于用途识别。"
    )
    if not config.vllm.enabled or not sample_files:
        return base_summary
    try:
        client = OpenAI(api_key=config.vllm.api_key, base_url=config.vllm.base_url)
        user_content: list[dict] = [
            {
                "type": "text",
                "text": "请用一句中文判断这些样本图像目录的用途（如训练集/测试集），不要输出冗余解释。",
            }
        ]
        image_payload_chars = 0
        for p in sample_files[: config.vllm.max_images_per_dir]:
            mime = "image/jpeg"
            if p.suffix.lower() == ".png":
                mime = "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            image_payload_chars += len(b64)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        t0 = time.perf_counter()
        resp = _openai_create_with_network_retry(
            client,
            label="vllm_image_dir",
            model=config.vllm.model_name,
            messages=[
                {"role": "system", "content": "你是数据集目录识别助手。"},
                {"role": "user", "content": user_content},
            ],
            stream=False,
        )
        if llm_client is not None:
            choice = resp.choices[0] if getattr(resp, "choices", None) else None
            llm_client._log_provider_usage(
                prompt_name="vllm_image_dir",
                mode="vision",
                response=resp,
                seconds=time.perf_counter() - t0,
                finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                parsed_ok=True,
                source="vllm_provider",
                model_name=config.vllm.model_name,
                prompt_parts=[
                    {"name": "system_prompt", "role": "system", "content": "你是数据集目录识别助手。"},
                    {
                        "name": "user_text",
                        "role": "user",
                        "content": "请用一句中文判断这些样本图像目录的用途（如训练集/测试集），不要输出冗余解释。",
                    },
                    {
                        "name": "image_payload_base64",
                        "role": "user",
                        "chars": image_payload_chars,
                        "utf8_bytes": image_payload_chars,
                        "estimated_tokens": max(1, image_payload_chars // 4),
                    },
                ],
            )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return f"{base_summary} 视觉抽样结论: {text[:300]}"
        return base_summary
    except Exception as exc:  # noqa: BLE001
        if config.vllm.fail_silently:
            return f"{base_summary} 视觉抽样失败，降级为元数据模式。"
        return f"{base_summary} 视觉抽样失败: {exc}"


def _infer_single_image_purpose(
    image_file: Path,
    config: AutoRealizeConfig,
    llm_client: LLMClient | None = None,
) -> str:
    if not config.vllm.enabled:
        return ""
    try:
        client = OpenAI(api_key=config.vllm.api_key, base_url=config.vllm.base_url)
        suffix = image_file.suffix.lower()
        mime = "image/jpeg"
        if suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        elif suffix in {".tif", ".tiff"}:
            mime = "image/tiff"
        elif suffix == ".gif":
            mime = "image/gif"
        b64 = base64.b64encode(image_file.read_bytes()).decode("ascii")
        t0 = time.perf_counter()
        resp = _openai_create_with_network_retry(
            client,
            label="vllm_single_image",
            model=config.vllm.model_name,
            messages=[
                {"role": "system", "content": "你是数据集图像语义识别助手。"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请用一句中文描述这张图片在数据集中的语义用途，不要复述元数据。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                },
            ],
            stream=False,
        )
        if llm_client is not None:
            choice = resp.choices[0] if getattr(resp, "choices", None) else None
            llm_client._log_provider_usage(
                prompt_name="vllm_single_image",
                mode="vision",
                response=resp,
                seconds=time.perf_counter() - t0,
                finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                parsed_ok=True,
                source="vllm_provider",
                model_name=config.vllm.model_name,
                prompt_parts=[
                    {"name": "system_prompt", "role": "system", "content": "你是数据集图像语义识别助手。"},
                    {
                        "name": "user_text",
                        "role": "user",
                        "content": "请用一句中文描述这张图片在数据集中的语义用途，不要复述元数据。",
                    },
                    {
                        "name": "image_payload_base64",
                        "role": "user",
                        "chars": len(b64),
                        "utf8_bytes": len(b64),
                        "estimated_tokens": max(1, len(b64) // 4),
                    },
                ],
            )
        text = (resp.choices[0].message.content or "").strip()
        return text[:300]
    except Exception:  # noqa: BLE001
        if config.vllm.fail_silently:
            return ""
        return "图片语义识别失败"


def _run_eval_reflector(
    llm_client: LLMClient,
    prompt_mgr: PromptManager,
    desc: str,
    original_text: str,
    task_hint: str,
    data_digest: str,
    enable_fewshot: bool = False,
) -> tuple[str, list[str]]:
    """Low-context evaluation ambiguity reflection loop."""
    system = prompt_mgr.load("system/eval_reflector.md")
    fewshot = prompt_mgr.load("fewshot/eval_ambiguity_fewshot.json") if enable_fewshot else ""
    defects: list[str] = []
    current = desc
    for idx in range(2):
        stable, dynamic = stable_dynamic_prompt(
            stable={
                "instruction": (
                    "请只基于 description 文本判断评估协议是否无歧义。输出严格 JSON；"
                    "最多列 6 条 ambiguity_points 和 6 条 fixes，每条必须短句，不要展开长解释。"
                )
            },
            dynamic={"description": current[:12000]},
            stable_title="Stable evaluation ambiguity review rules",
            dynamic_title="Dynamic description text",
        )
        try:
            review = llm_client.ask_structured(
                model_cls=AmbiguityReview,
                system_prompt=system,
                user_prompt=dynamic,
                prompt_name=f"eval_reflector_{idx+1}",
                fewshot=fewshot,
                max_tokens=2000,
                static_context_prompt=stable,
                dynamic_user_prompt=dynamic,
            )
        except RuntimeError as exc:
            log_event(
                logger,
                "module.task_definition.eval_reflector",
                "WARNING",
                round=idx + 1,
                error=str(exc)[:300],
                fallback="keep_current_description",
            )
            return current, defects
        if review.is_unambiguous:
            break
        defects.extend(review.ambiguity_points)
        rewritten = _rewrite_mutable_sections_with_llm(
            llm_client=llm_client,
            prompt_mgr=prompt_mgr,
            base_desc=current,
            defects=review.ambiguity_points + review.fixes,
            downstream_context={
                "task_hint": task_hint,
                "target_column": "",
                "y_true_field": "",
                "submission_columns": [],
                "task_type_hint": "",
            },
            prompt_name=f"description_refine_by_reflector_{idx+1}",
        )
        current = rewritten
    return current, defects


def _resolve_eval_ambiguity(
    desc: str,
    downstream_context: dict,
    llm_client: LLMClient,
    prompt_mgr: PromptManager,
    data_root: Path,
    enable_fewshot: bool = False,
) -> str:
    current = desc
    y_true_field = str(downstream_context.get("y_true_field", downstream_context.get("target_column", "target")))
    max_rounds = 3
    for _ in range(max_rounds):
        defects = eval_ambiguity_defects(current)
        if not defects:
            logger.info("[P2-Reflect] 歧义检查通过")
            return current
        logger.info("[P2-Reflect] 发现评估歧义，尝试修复: %s", defects[:3])
        # 先做规则化修复，保持可控可复现。
        patched = apply_eval_fixes(current, y_true_field=y_true_field)
        if patched != current:
            current = patched
            current = _enforce_existing_file_references(current, data_root)
            continue
        # 规则修不动时必须启用低上下文反思智能体。
        reviewed = _run_eval_reflector_once(
            llm_client,
            prompt_mgr,
            current,
            downstream_context=downstream_context,
            y_true_field=y_true_field,
            enable_fewshot=enable_fewshot,
        )
        if reviewed == current:
            return current
        current = _enforce_existing_file_references(reviewed, data_root)
    return current


def _run_eval_reflector_once(
    llm_client: LLMClient,
    prompt_mgr: PromptManager,
    desc: str,
    downstream_context: dict,
    y_true_field: str,
    enable_fewshot: bool = False,
) -> str:
    system = prompt_mgr.load("system/eval_reflector.md")
    fewshot = prompt_mgr.load("fewshot/eval_ambiguity_fewshot.json") if enable_fewshot else ""
    stable, dynamic = stable_dynamic_prompt(
        stable={
            "instruction": (
                "只基于 description 文本做检查，不允许引用任何外部上下文。"
                "若存在歧义，请输出结构化修复建议；若无歧义，请输出 is_unambiguous=true。"
                "最多列 6 条 ambiguity_points 和 6 条 fixes，每条必须短句。"
            )
        },
        dynamic={"description": desc[:12000]},
        stable_title="Stable single-pass evaluation reflection rules",
        dynamic_title="Dynamic description text",
    )
    try:
        review = llm_client.ask_structured(
            model_cls=AmbiguityReview,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name="eval_reflector_once",
            fewshot=fewshot,
            max_tokens=2000,
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )
    except RuntimeError as exc:
        log_event(
            logger,
            "module.task_definition.eval_reflector",
            "WARNING",
            error=str(exc)[:300],
            fallback="keep_current_description",
        )
        return desc
    if review.is_unambiguous or not review.fixes:
        return desc
    patched = apply_eval_fixes(desc, y_true_field=y_true_field)
    return _rewrite_mutable_sections_with_llm(
        llm_client=llm_client,
        prompt_mgr=prompt_mgr,
        base_desc=patched,
        defects=review.ambiguity_points + review.fixes,
        downstream_context=downstream_context,
        prompt_name="description_eval_section_rewrite",
    )


def _split_h2_sections(text: str) -> tuple[list[str], dict[str, str]]:
    def canonical_header(raw: str) -> str:
        header = re.sub(r"^\d+\.\s+", "", str(raw or "").strip())
        for canonical, aliases in SECTION_ALIASES.items():
            if header == canonical or header in aliases:
                return canonical
        return header

    lines = text.splitlines()
    order: list[str] = []
    sections: dict[str, str] = {}
    current = "__preamble__"
    bucket: list[str] = []
    order.append(current)
    for line in lines:
        if line.startswith("## "):
            sections[current] = "\n".join(bucket).rstrip() + "\n"
            current = canonical_header(line[3:].strip())
            order.append(current)
            bucket = [f"## {current}"]
        else:
            bucket.append(line)
    sections[current] = "\n".join(bucket).rstrip() + "\n"
    return order, sections


def _merge_mutable_sections(base_desc: str, rewritten_part: str) -> str:
    mutable_headers = {"任务定义", "评估协议", "输出或提交格式", "提交格式"}
    order_base, sections_base = _split_h2_sections(base_desc)
    _, sections_new = _split_h2_sections(rewritten_part)
    for h in mutable_headers:
        if h in sections_new:
            sections_base[h] = sections_new[h]
    merged: list[str] = []
    for h in order_base:
        merged.append(sections_base.get(h, ""))
    return "\n".join([x.rstrip("\n") for x in merged]).strip() + "\n"


def _rewrite_mutable_sections_with_llm(
    llm_client: LLMClient,
    prompt_mgr: PromptManager,
    base_desc: str,
    defects: list[str],
    downstream_context: dict,
    prompt_name: str,
) -> str:
    order, sections = _split_h2_sections(base_desc)
    _ = order
    mutable_now = "\n\n".join(
        [
            sections.get("任务定义", ""),
            sections.get("评估协议", ""),
            sections.get("输出或提交格式", "") or sections.get("提交格式", ""),
        ]
    ).strip()
    system = prompt_mgr.load("system/description_section_rewriter.md")
    stable, dynamic = stable_dynamic_prompt(
        stable={
            "rules": [
                "你不得引用不存在的文件名；若未识别预测文件，必须明确写“未提供独立预测文件，由训练数据切分验证”。",
                "只输出三个二级章节：任务定义 / 评估协议 / 输出或提交格式。",
                "不要输出反思过程、审查日志、Contract Status、issues/fixes、ambiguity_points 等中间过程。",
            ],
            "constraint_context": downstream_context,
        },
        dynamic={"mutable_sections": mutable_now[:12000], "defects": defects},
        stable_title="Stable mutable section rewrite rules",
        dynamic_title="Dynamic mutable sections and defects",
        stable_limit=6000,
    )
    rewritten_part = llm_client.ask_text(
        system_prompt=system,
        user_prompt=dynamic,
        prompt_name=prompt_name,
        static_user_prompt=stable,
        dynamic_user_prompt=dynamic,
    )
    merged = _merge_mutable_sections(base_desc, rewritten_part)
    return merged


def _enforce_existing_file_references(desc: str, data_root: Path) -> str:
    existing = {p.name for p in walk_files(data_root)}
    # 保留系统输出文件名
    existing |= {"description.md", "description_origin.md", "sample_submission.csv", "submission.csv"}

    pattern = re.compile(r"`([^`]+\.(?:csv|xlsx|xls|json|parquet|txt|md))`", flags=re.I)
    lines = desc.splitlines()
    new_lines: list[str] = []
    for line in lines:
        bad = False
        for m in pattern.finditer(line):
            fname = Path(m.group(1)).name
            if fname not in existing:
                bad = True
                break
        if bad:
            # 删除引用不存在文件的整行，避免 description 幻觉文件名。
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


def _find_missing_file_references(desc: str, data_root: Path) -> list[str]:
    existing = {p.name for p in walk_files(data_root)}
    existing |= {"description.md", "description_origin.md", "sample_submission.csv", "submission.csv"}
    pattern = re.compile(r"`([^`]+\.(?:csv|xlsx|xls|json|parquet|txt|md))`", flags=re.I)
    missing: list[str] = []
    for m in pattern.finditer(desc):
        fname = Path(m.group(1)).name
        if fname not in existing:
            missing.append(fname)
    uniq: list[str] = []
    seen = set()
    for x in missing:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _maybe_generate_predict_split(data_root: Path, downstream_context: dict, cfg: AutoRealizeConfig) -> None:
    train_name = str(downstream_context.get("train_table", "") or "").strip()
    predict_name = str(downstream_context.get("predict_table", "") or "").strip()
    target_col = str(downstream_context.get("target_column", "") or "").strip()
    task_type = str(downstream_context.get("task_type_hint", "") or "").lower()
    if predict_name:
        return
    if not train_name:
        return
    train_path = None
    for p in walk_files(data_root):
        if p.name == train_name and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}:
            train_path = p
            break
    if train_path is None:
        return
    try:
        df = read_table(
            train_path,
            json_flatten_sep=cfg.data.json_flatten_sep,
            json_flatten_max_level=cfg.data.json_flatten_max_level,
            json_keep_raw_nested_columns=cfg.data.json_keep_raw_nested_columns,
            max_rows=table_probe_sample_rows(
                train_path,
                configured_rows=cfg.data.table_profile_sample_rows,
                large_threshold_bytes=cfg.data.large_table_threshold_bytes,
            ),
        )
    except Exception:
        return
    if df.empty:
        return
    out = df.copy()
    time_col = ""
    for c in out.columns:
        lc = str(c).lower()
        if lc.startswith("dt") or "date" in lc or "time" in lc:
            time_col = str(c)
            break
    log_event(logger, "agent.predict_split_generator", "CREATED", train_table=train_name)
    log_event(logger, "agent.predict_split_generator", "ACTIVATED", train_table=train_name)
    out = _generate_predict_split_dataframe(
        train_df=out,
        task_type=task_type,
        target_col=target_col,
        time_col=time_col,
        cfg=cfg,
    )
    out_name = f"{Path(train_name).stem}__predict_split.csv"
    out_path = data_root / out_name
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log_event(
        logger,
        "agent.predict_split_generator",
        "COMPLETED",
        file=out_name,
        rows=int(len(out)),
    )


def _generate_predict_split_dataframe(
    *,
    train_df: pd.DataFrame,
    task_type: str,
    target_col: str,
    time_col: str,
    cfg: AutoRealizeConfig,
) -> pd.DataFrame:
    """Generate a lightweight prediction split from the inferred training table."""
    out = train_df.copy()
    tt = (task_type or "").lower()
    if "time_series" in tt and time_col and time_col in out.columns:
        dt = pd.to_datetime(out[time_col], errors="coerce")
        valid = dt.notna()
        if valid.any():
            max_day = dt[valid].max()
            min_keep = max_day - pd.Timedelta(days=max(1, int(cfg.data.generated_predict_horizon_days)))
            out = out[dt >= min_keep].copy()
    else:
        ratio = float(cfg.data.generated_predict_split_ratio)
        n = max(1, int(len(out) * ratio))
        out = out.tail(n).copy()
    if target_col and target_col in out.columns:
        out[target_col] = pd.NA
    return out

