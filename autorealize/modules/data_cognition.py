from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AutoRealizeConfig
from ..investigation import run_question_investigator
from ..logging_utils import log_event
from ..knowledge.base import KnowledgeEntry
from ..knowledge.local_store import make_entry_id
from ..models import AuthoritativeTaskMemory, FileGroupingRegexPlan, FileRole, FileSamplingReview, FileSummary
from ..prompt_cache import stable_dynamic_prompt
from ..profiling.csv_utils import read_csv_auto
from ..profiling.relations import detect_relations
from ..report_writer import append_constraint_memory_section, format_column_profile_inline, write_data_description
from ..utils.filesystem import rel, walk_dirs, walk_files
from ..utils.safe_json import dumps_json_safe, write_json_safe
from .types import DataCognitionResult, RuntimeServices

logger = logging.getLogger(__name__)


class DataCognitionModule:
    """规范书 demo 的第一阶段：多源异构数据认知与元知识库构建。"""

    def __init__(self, config: AutoRealizeConfig, services: RuntimeServices, report_dir: Path) -> None:
        self.config = config
        self.services = services
        self.report_dir = report_dir

    def run(self, data_root: Path, task_hint: str) -> DataCognitionResult:
        import autorealize.pipeline as legacy

        log_event(logger, "module.data_cognition", "CREATED")
        log_event(logger, "module.data_cognition", "ACTIVATED", data_root=str(data_root))
        tree_text = self._write_directory_tree(data_root)
        selected_files, compact_image_dirs, sampled_patterns, filename_sample_groups = self._select_files_with_pattern_sampling(data_root)
        log_event(
            logger,
            "module.data_cognition",
            "FILES_SELECTED",
            total_files=len(list(walk_files(data_root))),
            selected=len(selected_files),
            compact_image_dirs=len(compact_image_dirs),
            sampled_patterns=len(sampled_patterns),
        )
        for item in sampled_patterns[:80]:
            skipped = [str(x) for x in item.get("skipped", [])]
            log_event(
                logger,
                "module.data_cognition.sampling",
                "PATTERN_SAMPLED",
                directory=str(item.get("directory", "")),
                pattern=str(item.get("pattern", "")),
                total=item.get("total"),
                sampled=[str(x) for x in item.get("sampled", [])][:40],
                skipped=skipped[:120],
                skipped_count=len(skipped),
                reason=str(item.get("sampling_reason", "")),
            )

        file_summaries: list[FileSummary] = []
        table_columns: dict[str, list[str]] = {}
        original_requirement_texts: list[str] = []
        per_file_dir = self.report_dir / "file_cognition"
        per_file_dir.mkdir(parents=True, exist_ok=True)
        llm_cognition_paths = self._select_llm_cognition_paths(selected_files, data_root)

        def _one(file: Path) -> dict:
            result = legacy._cognize_one_file(
                file=file,
                data_root=data_root,
                registry=self.services.registry,
                config=self.config,
                llm_client=self.services.llm_client,
                prompt_mgr=self.services.prompt_mgr,
                task_hint=task_hint,
                allow_llm_cognition=rel(file, data_root) in llm_cognition_paths,
            )
            self._write_per_file_cognition(per_file_dir, result["fs"])
            return result

        if self.config.parallel.enable_parallel_cognition and len(selected_files) > 1:
            workers = max(1, int(self.config.parallel.cognition_max_workers))
            log_event(logger, "module.data_cognition.parallel", "ACTIVATED", workers=workers, files=len(selected_files))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_one, file) for file in selected_files]
                for fut in as_completed(futures):
                    self._accept_file_result(fut.result(), file_summaries, table_columns, original_requirement_texts)
            log_event(logger, "module.data_cognition.parallel", "COMPLETED", files=len(selected_files))
        else:
            for file in selected_files:
                self._accept_file_result(_one(file), file_summaries, table_columns, original_requirement_texts)

        for dir_rel, sample_files in compact_image_dirs.items():
            log_event(logger, "module.data_cognition.image_dir", "ACTIVATED", dir=dir_rel, samples=len(sample_files))
            vision_summary = legacy._infer_image_dir_purpose(
                data_root,
                dir_rel,
                sample_files,
                self.config,
                llm_client=self.services.llm_client,
            )
            file_summaries.append(
                FileSummary(
                    path=f"{dir_rel}/",
                    role=FileRole.data_description,
                    summary=vision_summary,
                    columns=[],
                    warnings=[],
                    source_metadata={"compact_image_dir": True, "sample_files": [rel(p, data_root) for p in sample_files]},
                )
            )
            log_event(logger, "module.data_cognition.image_dir", "COMPLETED", dir=dir_rel)

        filename_relations = self._apply_filename_sample_relations(file_summaries, filename_sample_groups)
        for fs in file_summaries:
            self._write_per_file_cognition(per_file_dir, fs)

        log_event(logger, "module.data_cognition.relations", "ACTIVATED", tables=len(table_columns))
        relation_hints = detect_relations(
            table_columns,
            file_summaries=file_summaries,
            parallel=self.config.parallel.enable_parallel_relations,
            max_workers=self.config.parallel.relations_max_workers,
        )
        legacy._refine_semantics_by_relations(file_summaries, relation_hints)
        rel_map: dict[str, list[str]] = {}
        for hint in relation_hints:
            rel_map.setdefault(hint.left_file, []).append(hint.right_file)
            rel_map.setdefault(hint.right_file, []).append(hint.left_file)
        for fs in file_summaries:
            fs.related_files = list(dict.fromkeys((fs.related_files or []) + rel_map.get(fs.path, [])))[:12]
            self._write_per_file_cognition(per_file_dir, fs)
        log_event(logger, "module.data_cognition.relations", "COMPLETED", relations=len(relation_hints))

        dir_summaries = self._summarize_dirs(data_root, file_summaries, sampled_patterns)
        data_description_path = self.report_dir / "data_description.md"
        log_event(logger, "module.data_cognition", "GENERATING_FILE", file="realize_report/data_description.md")
        write_data_description(data_description_path, file_summaries, dir_summaries, relation_hints)

        log_event(logger, "module.data_cognition.constraints", "ACTIVATED")
        constraint_memory = legacy._extract_constraint_memory(
            llm_client=self.services.llm_client,
            prompt_mgr=self.services.prompt_mgr,
            file_summaries=file_summaries,
            task_hint=task_hint,
        )
        log_event(
            logger,
            "module.data_cognition.constraints",
            "COMPLETED",
            constraints=len(constraint_memory.get("items", [])) if isinstance(constraint_memory, dict) else 0,
        )
        log_event(logger, "module.data_cognition.authority", "ACTIVATED")
        authoritative_memory = self._extract_authoritative_memory(
            data_root=data_root,
            task_hint=task_hint,
            file_summaries=file_summaries,
        )
        write_json_safe(self.report_dir / "authoritative_task_memory.json", authoritative_memory, indent=2)
        log_event(
            logger,
            "module.data_cognition.authority",
            "COMPLETED",
            sources=len(authoritative_memory.get("source_files", [])) if isinstance(authoritative_memory, dict) else 0,
            has_contract=bool((authoritative_memory.get("submission_contract") or {}).get("is_defined"))
            if isinstance(authoritative_memory, dict)
            else False,
        )
        knowledge_base = self._build_meta_knowledge_base(
            file_summaries=file_summaries,
            relation_count=len(relation_hints),
            constraint_memory=constraint_memory,
            authoritative_memory=authoritative_memory,
            directory_tree=tree_text,
            sampled_patterns=sampled_patterns,
            filename_sample_groups=filename_sample_groups,
        )
        if self._should_run_question_investigator(file_summaries, constraint_memory, authoritative_memory):
            question_memory = run_question_investigator(
                cfg=self.config,
                llm_client=self.services.llm_client,
                prompt_mgr=self.services.prompt_mgr,
                data_root=data_root,
                report_dir=self.report_dir,
                task_hint=task_hint,
                file_summaries=file_summaries,
                relation_hints=relation_hints,
                constraint_memory=constraint_memory,
                authoritative_memory=authoritative_memory,
                knowledge_base=knowledge_base,
            )
        else:
            question_memory = {
                "schema_version": "autorealize.question_investigation.v1",
                "enabled": False,
                "summary": "Skipped by on-demand trigger: no blocking data-access, output, evaluation, or constraint gap detected.",
                "questions": [],
                "script_requests": [],
                "tool_requests": [],
                "step_results": [],
                "answers": [],
                "unresolved_questions": [],
                "context_routing_notes": [],
            }
            write_json_safe(self.report_dir / "question_investigation_report.json", question_memory, indent=2)
            log_event(logger, "module.data_cognition.investigator", "SKIPPED", reason="on_demand_no_blocking_signal")
        knowledge_base["question_investigation"] = question_memory
        agent_context_pack = self._build_agent_context_pack(
            task_hint=task_hint,
            file_summaries=file_summaries,
            relation_hints=relation_hints,
            constraint_memory=constraint_memory,
            authoritative_memory=authoritative_memory,
            knowledge_base=knowledge_base,
            sampled_patterns=sampled_patterns,
            filename_sample_groups=filename_sample_groups,
            question_memory=question_memory,
        )
        write_json_safe(self.report_dir / "constraint_memory.json", constraint_memory, indent=2)
        write_json_safe(self.report_dir / "knowledge_base.json", knowledge_base, indent=2)
        write_json_safe(self.report_dir / "agent_context_pack.json", agent_context_pack, indent=2)
        cognition_report = {
            "schema_version": "autorealize.data_cognition_report.v1",
            "task_hint": task_hint,
            "directory_tree": "directory_tree.txt",
            "data_description": "data_description.md",
            "file_cognition_dir": "file_cognition",
            "files": [fs.model_dump() for fs in file_summaries],
            "directories": dir_summaries,
            "relations": [r.model_dump() if hasattr(r, "model_dump") else r.__dict__ for r in relation_hints],
            "filename_relations": filename_relations,
            "sampled_filename_patterns": sampled_patterns,
            "filename_sample_groups": filename_sample_groups,
            "compact_image_dirs": {
                k: [rel(p, data_root) for p in v]
                for k, v in compact_image_dirs.items()
            },
            "constraint_memory": constraint_memory,
            "authoritative_memory": authoritative_memory,
            "question_investigation": question_memory,
            "knowledge_base": knowledge_base,
            "agent_context_pack": agent_context_pack,
            "summary": {
                "file_count": len(file_summaries),
                "table_count": sum(1 for fs in file_summaries if fs.role == FileRole.raw_data_table),
                "requirement_doc_count": sum(1 for fs in file_summaries if fs.role == FileRole.task_requirement),
                "relation_count": len(relation_hints),
                "constraint_count": len(constraint_memory.get("items", [])) if isinstance(constraint_memory, dict) else 0,
            },
        }
        write_json_safe(self.report_dir / "data_cognition_report.json", cognition_report, indent=2)
        log_event(logger, "module.data_cognition", "GENERATED_FILE", file="realize_report/data_cognition_report.json")
        append_constraint_memory_section(data_description_path, constraint_memory)
        self._append_knowledge_sections(data_description_path, knowledge_base)
        self._publish_knowledge(file_summaries, constraint_memory, knowledge_base)
        log_event(logger, "module.data_cognition", "GENERATED_FILE", file="realize_report/data_description.md")
        log_event(logger, "module.data_cognition", "COMPLETED", files=len(file_summaries), relations=len(relation_hints))
        self.services.trajectory.log("data_cognition_module", "done", {"files": len(file_summaries), "relations": len(relation_hints)})
        return DataCognitionResult(
            file_summaries=file_summaries,
            original_requirement_texts=original_requirement_texts,
            table_columns=table_columns,
            relation_hints=relation_hints,
            constraint_memory=constraint_memory,
            authoritative_memory=authoritative_memory,
            question_memory=question_memory,
            knowledge_base=knowledge_base,
            agent_context_pack=agent_context_pack,
            data_description_path=data_description_path,
        )

    def _select_llm_cognition_paths(self, files: list[Path], data_root: Path) -> set[str]:
        mode = str(getattr(self.config.data, "llm_file_cognition_mode", "all") or "all").lower()
        if mode in {"all", "selective"}:
            return {rel(p, data_root) for p in files}
        if mode in {"none", "off", "disabled"}:
            return set()
        doc_ext = {".txt", ".md", ".rst", ".log", ".docx", ".pdf", ".toml", ".yaml", ".yml"}
        selected: set[str] = set()
        for p in files:
            if p.suffix.lower() in doc_ext or _is_task_like_file(p):
                selected.add(rel(p, data_root))
        return selected

    def _should_run_question_investigator(
        self,
        file_summaries: list[FileSummary],
        constraint_memory: dict,
        authoritative_memory: dict,
    ) -> bool:
        if not bool(getattr(self.config.investigation, "enabled", True)):
            return False
        mode = str(getattr(self.config.investigation, "trigger_mode", "on_demand") or "on_demand").lower()
        if mode in {"disabled", "off", "false"}:
            return False
        if mode == "always":
            return True

        contract = (authoritative_memory or {}).get("submission_contract") if isinstance(authoritative_memory, dict) else {}
        if isinstance(contract, dict) and contract.get("unresolved_questions"):
            return True
        if isinstance(authoritative_memory, dict) and authoritative_memory.get("unresolved_questions"):
            return True
        if isinstance(authoritative_memory, dict) and any(fs.role == FileRole.task_requirement for fs in file_summaries):
            has_eval = bool(authoritative_memory.get("evaluation_requirements"))
            has_output = bool(authoritative_memory.get("output_requirements")) or bool(
                isinstance(contract, dict) and contract.get("is_defined")
            )
            if not has_eval and not has_output:
                return True

        for fs in file_summaries:
            meta = fs.source_metadata or {}
            blocking_warnings = [
                str(w)
                for w in (fs.warnings or [])
                if "字段统计基于前" not in str(w) and "未全量扫描" not in str(w)
            ]
            if fs.role == FileRole.unknown or blocking_warnings:
                return True
            if meta.get("csv_dialect") and isinstance(meta.get("csv_dialect"), dict):
                if meta["csv_dialect"].get("inferred"):
                    return True
            sheets = meta.get("excel_sheet_names") or []
            if isinstance(sheets, list) and len(sheets) > 1:
                return True
            text = " ".join([fs.summary or "", fs.detailed_report or "", *list(fs.risks if hasattr(fs, "risks") else [])]).lower()
            if any(k in text for k in ["unclear", "不确定", "无法确定", "需确认", "读取失败", "解析失败"]):
                return True
        items = (constraint_memory or {}).get("items", []) if isinstance(constraint_memory, dict) else []
        return any(str(item).lower().find("不确定") >= 0 for item in items[:20])

    def _write_directory_tree(self, data_root: Path) -> str:
        lines = [data_root.name or "."]
        for p in sorted(data_root.rglob("*"), key=lambda x: str(x).lower()):
            try:
                rp = p.relative_to(data_root)
            except ValueError:
                continue
            depth = len(rp.parts) - 1
            lines.append(f"{'  ' * depth}- {rp.name}{'/' if p.is_dir() else ''}")
        text = "\n".join(lines)
        (self.report_dir / "directory_tree.txt").write_text(text, encoding="utf-8")
        log_event(logger, "module.data_cognition", "GENERATED_FILE", file="realize_report/directory_tree.txt")
        return text

    def _select_files_with_pattern_sampling(self, data_root: Path) -> tuple[list[Path], dict[str, list[Path]], list[dict], list[dict]]:
        selected: list[Path] = []
        compact_image_dirs: dict[str, list[Path]] = {}
        sampled_patterns: list[dict] = []
        filename_sample_groups: list[dict] = []
        image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
        # 需求/说明类文档必须全读；表格、JSON、日志若命名高度统一则允许抽样。
        always_full_ext = {".txt", ".md", ".doc", ".docx", ".pdf", ".toml", ".yaml", ".yml"}
        tabular_ext = {".csv", ".xlsx", ".xls"}
        generic_max_samples = max(1, int(getattr(self.config.data, "pattern_sample_file_count", 3)))
        tabular_max_samples = max(
            1,
            int(
                getattr(
                    self.config.data,
                    "similar_table_sample_file_count",
                    getattr(self.config.data, "tabular_pattern_sample_file_count", 2),
                )
            ),
        )
        generic_min_group = max(2, int(getattr(self.config.data, "filename_pattern_min_group", 20)))
        tabular_min_group = max(
            2,
            int(
                getattr(
                    self.config.data,
                    "similar_table_min_files_to_sample",
                    getattr(self.config.data, "tabular_pattern_min_group", 3),
                )
            ),
        )
        use_schema_signature = bool(
            getattr(
                self.config.data,
                "similar_table_use_header_signature",
                getattr(self.config.data, "tabular_pattern_use_schema_signature", True),
            )
        )
        llm_grouping_cache: dict[str, list[dict[str, Any]]] = {}
        llm_grouping_enabled = bool(getattr(self.config.data, "enable_llm_filename_grouping", False))

        for d in walk_dirs(data_root):
            group = [p for p in walk_files(d) if p.parent == d]
            if not group:
                continue
            drel = rel(d, data_root)
            image_files = [p for p in group if p.suffix.lower() in image_ext]
            non_images = [p for p in group if p not in image_files]

            if len(image_files) > self.config.data.image_dir_compact_threshold:
                sample_count = max(1, self.config.data.image_dir_sample_file_count)
                samples = sorted(image_files)[:sample_count]
                selected.extend(samples)
                compact_image_dirs[drel] = samples
            else:
                selected.extend(image_files)

            llm_patterns = self._llm_filename_grouping_patterns(drel, non_images, llm_grouping_cache)
            filename_sample_groups.extend(
                _extract_filename_sample_groups(
                    drel,
                    group,
                    llm_patterns,
                    data_root,
                    allow_legacy_fallback=False,
                )
            )

            pattern_groups: dict[tuple[str, str, str], list[Path]] = {}
            pattern_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
            eligible_files: list[Path] = []
            for p in non_images:
                suffix = p.suffix.lower()
                if suffix in always_full_ext or _is_task_like_file(p):
                    selected.append(p)
                    continue
                eligible_files.append(p)

            initially_ungrouped: list[Path] = []
            for p in eligible_files:
                suffix = p.suffix.lower()
                match_info: dict[str, Any] | None = None
                if not llm_grouping_enabled:
                    initially_ungrouped.append(p)
                    continue
                match_info = _llm_grouping_match_for_file(p, llm_patterns)
                if not match_info:
                    initially_ungrouped.append(p)
                    continue
                pattern = str(match_info.get("pattern") or "")
                schema_info: dict[str, Any] = {}
                schema_sig = ""
                if suffix in tabular_ext and use_schema_signature:
                    schema_info = _table_schema_signature(p)
                    schema_sig = str(schema_info.get("signature") or "schema_unreadable")
                key = (pattern, suffix, schema_sig)
                pattern_groups.setdefault(key, []).append(p)
                meta = pattern_meta.setdefault(
                    key,
                    {
                        "schema_signature": schema_sig,
                        "columns_preview": schema_info.get("columns_preview", []),
                        "column_count": schema_info.get("column_count"),
                        "schema_error": schema_info.get("error"),
                    },
                )
                if match_info:
                    meta.update(
                        {
                            "regex_name": match_info.get("regex_name", ""),
                            "regex": match_info.get("regex", ""),
                            "regex_reason": match_info.get("regex_reason", ""),
                            "regex_confidence": match_info.get("regex_confidence", 0.0),
                        }
                    )

            sampling_candidates = _build_sampling_candidates_from_pattern_groups(
                drel,
                pattern_groups,
                pattern_meta,
                data_root,
                tabular_ext=tabular_ext,
                generic_min_group=generic_min_group,
                tabular_min_group=tabular_min_group,
                generic_max_samples=generic_max_samples,
                tabular_max_samples=tabular_max_samples,
                candidate_pool=eligible_files,
                initially_ungrouped=initially_ungrouped,
                reason_prefix="llm_regex" if llm_grouping_enabled else "same_directory",
            )
            reviewed = self._review_sampling_candidates(
                drel,
                sampling_candidates,
                data_root,
                require_llm_review=llm_grouping_enabled,
            )
            reviewed_file_set: set[Path] = set()
            for item in reviewed:
                reviewed_file_set.update({x.resolve() for x in item.get("_files", []) or []})
                selected.extend(item.pop("_sample_paths", []))
                item.pop("_files", None)
                item.pop("_skipped_paths", None)
                item.pop("_candidate_pool", None)
                item.pop("_initially_ungrouped_paths", None)
                item.pop("_data_root", None)
                sampled_patterns.append(item)
            selected.extend([p for p in eligible_files if p.resolve() not in reviewed_file_set])
        return sorted(set(selected)), compact_image_dirs, sampled_patterns, filename_sample_groups

    def _review_sampling_candidates(
        self,
        drel: str,
        candidates: list[dict[str, Any]],
        data_root: Path,
        *,
        require_llm_review: bool = False,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if bool(getattr(self.config.switches, "optimize_llm_cost", True)) and not require_llm_review:
            reviewed: list[dict[str, Any]] = []
            for item in candidates:
                item["review"] = {
                    "decision": "deterministic_accept",
                    "reason": "low-token mode accepts rule-verified same-pattern sampling without LLM review",
                }
                item.pop("_skipped_paths", None)
                reviewed.append(item)
            return reviewed
        llm_client = self.services.llm_client
        prompt_mgr = self.services.prompt_mgr
        reviewed: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = list(candidates)
        chunk_size = 12
        max_rounds = 2 if require_llm_review else 1
        for round_index in range(max_rounds):
            if not pending:
                break
            next_round: list[dict[str, Any]] = []
            for start in range(0, len(pending), chunk_size):
                chunk = pending[start : start + chunk_size]
                stable, dynamic = stable_dynamic_prompt(
                    stable={
                        "instruction": (
                            "Review the concrete file sampling plan after regex matching. "
                            "For every pattern_id, decide whether the system may skip the listed will_skip files, "
                            "or whether it should read all files / add extra representative files. "
                            "If the regex is too broad or too narrow, return rewrite_regex so the system can rebuild "
                            "the will-read / will-skip plan and ask you to confirm it once more."
                        )
                    },
                    dynamic={
                        "directory": drel,
                        "review_round": round_index + 1,
                        "max_review_rounds": max_rounds,
                        "plans": [_sampling_review_payload(item) for item in chunk],
                    },
                    stable_title="Stable sampling review rules",
                    dynamic_title="Dynamic sampling candidates",
                )
                try:
                    review = llm_client.ask_structured(
                        model_cls=FileSamplingReview,
                        system_prompt=prompt_mgr.load("system/file_sampling_reviewer.md"),
                        user_prompt=dynamic,
                        prompt_name="file_sampling_reviewer",
                        static_context_prompt=stable,
                        dynamic_user_prompt=dynamic,
                    )
                    review_by_id = {str(item.pattern_id): item for item in review.items}
                except Exception as exc:  # noqa: BLE001
                    log_event(
                        logger,
                        "module.data_cognition.sampling_review",
                        "FAILED",
                        dir=drel,
                        round=round_index + 1,
                        error=str(exc)[:240],
                    )
                    review_by_id = {}
                for item in chunk:
                    review_item = review_by_id.get(str(item.get("pattern_id")))
                    rewrite_regex = str(getattr(review_item, "rewrite_regex", "") or "").strip() if review_item else ""
                    if rewrite_regex and round_index + 1 < max_rounds:
                        rewritten = self._build_rewritten_sampling_candidates(drel, item, review_item, data_root)
                        if rewritten:
                            next_round.extend(rewritten)
                            log_event(
                                logger,
                                "module.data_cognition.sampling_review",
                                "REWRITE_REQUESTED",
                                directory=str(item.get("directory", "")),
                                pattern_id=str(item.get("pattern_id", "")),
                                rewritten_candidates=len(rewritten),
                            )
                            continue
                        applied = _force_full_sampling_plan(
                            item,
                            data_root,
                            decision="force_full_read_invalid_rewrite_regex",
                            reason="Reviewer returned rewrite_regex, but it failed validation or did not produce a sampleable plan.",
                        )
                    elif rewrite_regex:
                        applied = _force_full_sampling_plan(
                            item,
                            data_root,
                            decision="force_full_read_rewrite_not_confirmed",
                            reason="Reviewer requested another regex rewrite after the final confirmation round.",
                        )
                    else:
                        applied = _apply_sampling_review(item, review_item, data_root)
                    reviewed.append(applied)
                    log_event(
                        logger,
                        "module.data_cognition.sampling_review",
                        "REVIEWED",
                        directory=str(applied.get("directory", "")),
                        pattern_id=str(applied.get("pattern_id", "")),
                        pattern=str(applied.get("pattern", "")),
                        decision=str((applied.get("review") or {}).get("decision", "")),
                        sampled_count=len(applied.get("sampled", [])),
                        skipped_count=len(applied.get("skipped", [])),
                    )
            pending = next_round
        return reviewed

    def _build_rewritten_sampling_candidates(
        self,
        drel: str,
        item: dict[str, Any],
        review: Any,
        data_root: Path,
    ) -> list[dict[str, Any]]:
        candidate_pool = sorted(set(item.get("_candidate_pool", []) or item.get("_files", []) or []))
        if not candidate_pool:
            return []
        names = sorted({p.name for p in candidate_pool})
        candidate = {
            "name": f"rewrite_for_{item.get('pattern_id', '')}",
            "regex": str(getattr(review, "rewrite_regex", "") or ""),
            "sample_id_group": str(getattr(review, "rewrite_sample_id_group", "") or "sample_id"),
            "data_kind_group": str(getattr(review, "rewrite_data_kind_group", "") or "data_kind"),
            "applies_to_suffixes": list(getattr(review, "rewrite_applies_to_suffixes", []) or []),
            "reason": str(getattr(review, "reason", "") or "")[:300],
            "confidence": 0.0,
        }
        validated = _validate_llm_grouping_regex(candidate, names)
        if not validated:
            return []

        tabular_ext = {".csv", ".xlsx", ".xls"}
        generic_max_samples = max(1, int(getattr(self.config.data, "pattern_sample_file_count", 3)))
        tabular_max_samples = max(
            1,
            int(
                getattr(
                    self.config.data,
                    "similar_table_sample_file_count",
                    getattr(self.config.data, "tabular_pattern_sample_file_count", 2),
                )
            ),
        )
        generic_min_group = max(2, int(getattr(self.config.data, "filename_pattern_min_group", 20)))
        tabular_min_group = max(
            2,
            int(
                getattr(
                    self.config.data,
                    "similar_table_min_files_to_sample",
                    getattr(self.config.data, "tabular_pattern_min_group", 3),
                )
            ),
        )
        use_schema_signature = bool(
            getattr(
                self.config.data,
                "similar_table_use_header_signature",
                getattr(self.config.data, "tabular_pattern_use_schema_signature", True),
            )
        )

        pattern_groups: dict[tuple[str, str, str], list[Path]] = {}
        pattern_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
        ungrouped: list[Path] = []
        for path in candidate_pool:
            match_info = _llm_grouping_match_for_file(path, [validated])
            if not match_info:
                ungrouped.append(path)
                continue
            suffix = path.suffix.lower()
            schema_info: dict[str, Any] = {}
            schema_sig = ""
            if suffix in tabular_ext and use_schema_signature:
                schema_info = _table_schema_signature(path)
                schema_sig = str(schema_info.get("signature") or "schema_unreadable")
            key = (str(match_info.get("pattern") or ""), suffix, schema_sig)
            pattern_groups.setdefault(key, []).append(path)
            pattern_meta.setdefault(
                key,
                {
                    "schema_signature": schema_sig,
                    "columns_preview": schema_info.get("columns_preview", []),
                    "column_count": schema_info.get("column_count"),
                    "schema_error": schema_info.get("error"),
                    "regex_name": match_info.get("regex_name", ""),
                    "regex": match_info.get("regex", ""),
                    "regex_reason": match_info.get("regex_reason", ""),
                    "regex_confidence": match_info.get("regex_confidence", 0.0),
                    "rewrite_from_pattern_id": item.get("pattern_id"),
                },
            )

        return _build_sampling_candidates_from_pattern_groups(
            drel,
            pattern_groups,
            pattern_meta,
            data_root,
            tabular_ext=tabular_ext,
            generic_min_group=generic_min_group,
            tabular_min_group=tabular_min_group,
            generic_max_samples=generic_max_samples,
            tabular_max_samples=tabular_max_samples,
            candidate_pool=candidate_pool,
            initially_ungrouped=ungrouped,
            reason_prefix="llm_rewritten_regex",
        )

    def _llm_filename_grouping_patterns(
        self,
        drel: str,
        files: list[Path],
        cache: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        if drel in cache:
            return cache[drel]
        cache[drel] = []
        if not bool(getattr(self.config.data, "enable_llm_filename_grouping", False)):
            return []
        llm_client = self.services.llm_client
        prompt_mgr = self.services.prompt_mgr
        names = sorted({p.name for p in files})
        if len(names) < 3:
            return []
        max_names = max(10, int(getattr(self.config.data, "llm_filename_grouping_max_names", 80)))
        suffix_counts: dict[str, int] = {}
        for p in files:
            suffix_counts[p.suffix.lower()] = suffix_counts.get(p.suffix.lower(), 0) + 1
        stable, dynamic = stable_dynamic_prompt(
            stable={
                "instruction": (
                    "Propose regexes only when repeated sample-id/data-kind filename structures exist. "
                    "Regexes must be Python-compatible and include named groups sample_id and data_kind."
                )
            },
            dynamic={
                "directory": drel,
                "suffix_counts": suffix_counts,
                "file_names": names[:max_names],
            },
            stable_title="Stable filename grouping rules",
            dynamic_title="Dynamic directory file names",
        )
        plan = llm_client.ask_structured(
            model_cls=FileGroupingRegexPlan,
            system_prompt=prompt_mgr.load("system/file_grouping_regex_planner.md"),
            user_prompt=dynamic,
            prompt_name="file_grouping_regex_planner",
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )

        max_patterns = max(1, int(getattr(self.config.data, "llm_filename_grouping_max_patterns", 6)))
        accepted: list[dict[str, Any]] = []
        for candidate in plan.candidates[:max_patterns]:
            validated = _validate_llm_grouping_regex(candidate.model_dump(), names)
            if validated:
                accepted.append(validated)
        if accepted:
            log_event(
                logger,
                "module.data_cognition.filename_grouping",
                "LLM_PATTERNS_ACCEPTED",
                dir=drel,
                patterns=len(accepted),
            )
        cache[drel] = accepted
        return accepted

    def _accept_file_result(
        self,
        result: dict,
        file_summaries: list[FileSummary],
        table_columns: dict[str, list[str]],
        original_requirement_texts: list[str],
    ) -> None:
        file_summaries.append(result["fs"])
        if result.get("columns"):
            table_columns[result["rpath"]] = result["columns"]
        if result.get("is_requirement") and result.get("summary_text"):
            original_requirement_texts.append(str(result["summary_text"]))

    def _apply_filename_sample_relations(self, file_summaries: list[FileSummary], filename_sample_groups: list[dict]) -> list[dict]:
        by_path = {fs.path: fs for fs in file_summaries}
        relations: list[dict] = []
        for group in filename_sample_groups:
            files = [str(x) for x in group.get("files", []) if str(x) in by_path]
            if len(files) < 2:
                continue
            sample_id = str(group.get("sample_id", ""))
            data_kinds = group.get("data_kinds", {}) if isinstance(group.get("data_kinds"), dict) else {}
            for path in files:
                peers = [x for x in files if x != path]
                fs = by_path[path]
                kind = str(data_kinds.get(path, "sample_part"))
                fact = f"文件名样本ID `{sample_id}` 将该文件与 {', '.join(peers[:8])} 关联为同一个样本/井的一组数据。"
                fs.related_files = list(dict.fromkeys([*peers, *fs.related_files]))[:12]
                fs.related_files = _filter_generic_related_hints(fs.related_files, strong_peers=set(peers))
                fs.extracted_knowledge = list(dict.fromkeys([fact, *fs.extracted_knowledge]))[:40]
                fs.source_metadata = fs.source_metadata or {}
                fs.source_metadata["filename_sample_id"] = sample_id
                fs.source_metadata["filename_data_kind"] = kind
                fs.source_metadata["filename_group_files"] = files
                relations.append(
                    {
                        "relation_type": "same_filename_sample_id",
                        "sample_id": sample_id,
                        "file": path,
                        "data_kind": kind,
                        "related_files": peers,
                    }
                )
        return relations

    def _safe_cognition_filename(self, rel_path: str) -> str:
        normalized = rel_path.replace("\\", "/").strip("/")
        if not normalized:
            normalized = "root"
        # use full relative path + slash marker to avoid same-name collisions across dirs
        encoded = normalized.replace("/", "__SLASH__")
        encoded = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", encoded).strip("_")
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
        return f"{encoded}__{digest}" if encoded else f"root__{digest}"

    def _write_per_file_cognition(self, out_dir: Path, fs: FileSummary) -> None:
        safe_name = self._safe_cognition_filename(fs.path)
        payload = fs.model_dump()
        json_path = out_dir / f"{safe_name}.json"
        md_path = out_dir / f"{safe_name}.md"
        write_json_safe(json_path, payload, indent=2)
        lines = [f"# {fs.path}", "", f"- 角色: `{fs.role.value}`", f"- 摘要: {fs.summary}"]
        if str(fs.detailed_report or "").strip():
            lines.append("")
            lines.append("## 详细认知报告")
            lines.append(str(fs.detailed_report).strip())
        if fs.extracted_knowledge:
            lines.append("")
            lines.append("- 关键知识明细:")
            for x in fs.extracted_knowledge[:30]:
                lines.append(f"  - {x}")
        if fs.key_entities:
            lines.append(f"- 关键实体: {', '.join(fs.key_entities[:20])}")
        if fs.related_files:
            lines.append(f"- 可能关联文件: {', '.join(fs.related_files[:12])}")
        data_profiles = {str(p.get("name", "")): p for p in (fs.column_profiles or []) if str(p.get("name", "")).strip()}
        if fs.column_semantics and data_profiles:
            lines.append("")
            lines.append("## 字段语义")
            for col, meaning in fs.column_semantics.items():
                lines.append(f"- `{col}`: {meaning}")
        if fs.column_profiles:
            lines.append("")
            lines.append("## 字段结构与质量（全部数据字段）")
            for p in fs.column_profiles:
                name = str(p.get("name", "")).strip()
                if not name:
                    continue
                lines.append(f"- `{name}`: {format_column_profile_inline(p)}")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        log_event(logger, "module.data_cognition.file_artifact", "GENERATED_FILE", file=str(md_path.name), source=fs.path)

    def _summarize_dirs(self, root: Path, files: list[FileSummary], sampled_patterns: list[dict]) -> list[str]:
        by_dir: dict[str, list[FileSummary]] = {}
        for f in files:
            d = str(Path(f.path).parent).replace("\\", "/")
            by_dir.setdefault(d, []).append(f)
        pattern_by_dir: dict[str, list[dict]] = {}
        for item in sampled_patterns:
            pattern_by_dir.setdefault(str(item["directory"]), []).append(item)
        lines: list[str] = []
        for d in walk_dirs(root):
            rd = rel(d, root)
            group = by_dir.get(rd, [])
            patterns = pattern_by_dir.get(rd, [])
            if not group and not patterns:
                continue
            roles: dict[str, int] = {}
            for g in group:
                roles[g.role.value] = roles.get(g.role.value, 0) + 1
            role_desc = ", ".join([f"{k}:{v}" for k, v in sorted(roles.items(), key=lambda x: -x[1])]) or "无逐文件认知"
            line = f"`{rd}`: 文件认知数 {len(group)}，角色分布 {role_desc}"
            if patterns:
                pats = "; ".join(
                    [
                        f"{p['pattern']} 共 {p['total']} 个，抽样 {len(p['sampled'])} 个，跳过 {len(p.get('skipped', []))} 个"
                        for p in patterns[:5]
                    ]
                )
                line += f"，统一命名模式: {pats}"
            lines.append(line)
        return lines

    def _read_authoritative_text(self, path: Path, max_chars: int = 24000) -> str:
        if path.suffix.lower() not in {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}:
            return ""
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return path.read_text(encoding=enc, errors="ignore")[:max_chars]
            except Exception:
                continue
        return ""

    def _is_authoritative_doc_name(self, path: Path) -> tuple[bool, str, str]:
        name = path.name.lower()
        stem = path.stem.lower()
        rel_name = str(path).replace("\\", "/").lower()
        if name == "description.md":
            return True, "original_description", "high"
        if stem.startswith("readme"):
            return True, "readme", "high"
        if any(k in rel_name for k in ["requirement", "requirements", "task", "spec", "规则", "需求", "任务", "说明"]):
            return True, "requirement_doc", "high"
        return False, "", "low"

    def _is_sample_submission_name(self, path: Path) -> bool:
        compact = "".join(ch for ch in path.stem.lower() if ch.isalnum())
        return "samplesubmission" in compact or path.name.lower() == "sample_submission.csv"

    def _sample_submission_columns(self, path: Path) -> list[str]:
        try:
            if path.suffix.lower() == ".csv":
                return [str(c).strip() for c in read_csv_auto(path, nrows=0).columns if str(c).strip()]
            if path.suffix.lower() in {".xlsx", ".xls"}:
                return [str(c).strip() for c in pd.read_excel(path, nrows=0).columns if str(c).strip()]
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger,
                "module.data_cognition.authority",
                "SAMPLE_HEADER_READ_FAILED",
                file=str(path),
                error=str(exc)[:180],
            )
        return []

    def _extract_authoritative_memory(
        self,
        *,
        data_root: Path,
        task_hint: str,
        file_summaries: list[FileSummary],
    ) -> dict:
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _add_source(
            *,
            source_path: str,
            source_type: str,
            priority: str,
            text: str = "",
            columns: list[str] | None = None,
            evidence: str = "",
        ) -> None:
            key = f"{source_type}:{source_path}"
            if key in seen:
                return
            seen.add(key)
            payload = {
                "source_path": source_path,
                "source_type": source_type,
                "priority": priority,
                "text": text[:24000],
                "columns": columns or [],
                "evidence": evidence[:2000],
            }
            if payload["text"] or payload["columns"] or payload["evidence"]:
                sources.append(payload)

        if task_hint.strip():
            _add_source(
                source_path="task_hint",
                source_type="user_hint",
                priority="highest",
                text=task_hint.strip(),
                evidence=task_hint.strip(),
            )

        for p in walk_files(data_root):
            if not p.is_file():
                continue
            is_doc, source_type, priority = self._is_authoritative_doc_name(p)
            if is_doc:
                text = self._read_authoritative_text(p)
                if text.strip():
                    _add_source(
                        source_path=rel(p, data_root),
                        source_type=source_type,
                        priority=priority,
                        text=text,
                        evidence=text[:1000],
                    )
            if p.suffix.lower() in {".csv", ".xlsx", ".xls"} and self._is_sample_submission_name(p):
                columns = self._sample_submission_columns(p)
                _add_source(
                    source_path=rel(p, data_root),
                    source_type="official_sample",
                    priority="high",
                    columns=columns,
                    evidence=f"Official sample submission file with columns: {columns}",
                )

        for fs in file_summaries:
            if fs.role != FileRole.task_requirement:
                continue
            summary_text = "\n".join(
                [
                    fs.summary or "",
                    fs.detailed_report or "",
                    *[str(x) for x in (fs.extracted_knowledge or [])[:20]],
                ]
            ).strip()
            if summary_text:
                _add_source(
                    source_path=fs.path,
                    source_type="requirement_doc",
                    priority="high",
                    text=summary_text,
                    evidence=summary_text[:1000],
                )

        if not sources:
            return AuthoritativeTaskMemory().model_dump()

        priority_rank = {
            "highest": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }
        source_type_rank = {
            "user_hint": 0,
            "original_description": 1,
            "official_sample": 1,
            "readme": 2,
            "requirement_doc": 2,
        }
        sources.sort(
            key=lambda s: (
                priority_rank.get(str(s.get("priority", "")).lower(), 9),
                source_type_rank.get(str(s.get("source_type", "")).lower(), 9),
                str(s.get("source_path", "")),
            )
        )

        stable, dynamic = stable_dynamic_prompt(
            stable={
                "instruction": (
                "Extract only authoritative task memory from the provided sources. "
                "Do not invent metrics, submission columns, output filenames, row counts, or constraints. "
                "If the sources do not explicitly define the submission/output contract, set "
                "submission_contract.is_defined=false and submission_contract.columns=[]. "
                "Authority priority is strict: user_hint outranks existing description.md, existing description.md "
                "outranks README/official/spec/other requirement documents, and all documents outrank data profiles "
                "or LLM inference. If sources conflict, keep the highest-priority source, preserve evidence, and record "
                "the conflict in authority_conflicts or context_routing_notes."
                )
            },
            dynamic={"sources": sources[:24]},
            stable_title="Stable authoritative extraction rules",
            dynamic_title="Dynamic authoritative sources",
        )
        memory = self.services.llm_client.ask_structured(
            model_cls=AuthoritativeTaskMemory,
            system_prompt=(
                "You extract high-priority task contracts from original task documents. "
                "You must preserve evidence and leave unknown fields empty instead of guessing."
            ),
            user_prompt=dynamic,
            prompt_name="authoritative_task_memory",
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )
        data = memory.model_dump()
        data["source_files"] = list(
            dict.fromkeys(
                [
                    *[str(x) for x in data.get("source_files", []) if str(x).strip()],
                    *[str(s.get("source_path", "")) for s in sources if str(s.get("source_path", "")).strip()],
                ]
            )
        )[:80]

        official_samples = [s for s in sources if s.get("source_type") == "official_sample" and s.get("columns")]
        if official_samples:
            first = official_samples[0]
            contract = data.get("submission_contract") or {}
            contract["is_defined"] = True
            contract["is_authoritative"] = True
            contract["sample_filename"] = str(Path(str(first.get("source_path", "sample_submission.csv"))).name)
            contract["columns"] = [str(c) for c in first.get("columns", []) if str(c).strip()]
            contract["source"] = str(first.get("source_path", ""))
            evidence = list(contract.get("evidence", []) or [])
            evidence.append(str(first.get("evidence", "")))
            contract["evidence"] = [x for x in dict.fromkeys(evidence) if x][:12]
            contract["confidence"] = max(float(contract.get("confidence") or 0.0), 0.98)
            data["submission_contract"] = contract

        data["has_authoritative_sources"] = True
        return AuthoritativeTaskMemory.model_validate(data).model_dump()

    def _build_meta_knowledge_base(
        self,
        *,
        file_summaries: list[FileSummary],
        relation_count: int,
        constraint_memory: dict,
        authoritative_memory: dict,
        directory_tree: str,
        sampled_patterns: list[dict],
        filename_sample_groups: list[dict],
    ) -> dict:
        entities: dict[str, list[str]] = {}
        metrics: list[dict] = []
        time_clues: list[dict] = []
        field_glossary: dict[str, dict] = {}
        for fs in file_summaries:
            for ent in fs.key_entities:
                entities.setdefault(ent, []).append(fs.path)
            for col, meaning in fs.column_semantics.items():
                field_glossary.setdefault(col, {"meaning": meaning, "files": []})
                field_glossary[col]["files"].append(fs.path)
                lower = f"{col} {meaning}".lower()
                if any(k in lower for k in ["value", "metric", "score", "amount", "price", "cost", "rate", "count", "数量", "数值", "指标", "分数", "金额", "单价", "成本"]):
                    metrics.append({"field": col, "file": fs.path, "meaning": meaning})
                if any(k in lower for k in ["date", "time", "month", "day", "日期", "时间", "每日", "月份"]):
                    time_clues.append({"field": col, "file": fs.path, "meaning": meaning})
        return {
            "directory_tree_head": directory_tree[:8000],
            "sampled_filename_patterns": sampled_patterns,
            "filename_sample_groups": filename_sample_groups,
            "authoritative_task_memory": authoritative_memory,
            "authoritative_submission_contract": (authoritative_memory or {}).get("submission_contract", {}),
            "authoritative_sources": (authoritative_memory or {}).get("source_files", []),
            "entity_alignment": entities,
            "field_glossary": field_glossary,
            "metric_candidates": metrics[:80],
            "time_clues": time_clues[:80],
            "implicit_relation_count": relation_count,
            "business_constraints": constraint_memory.get("items", []) if isinstance(constraint_memory, dict) else [],
        }

    def _build_agent_context_pack(
        self,
        *,
        task_hint: str,
        file_summaries: list[FileSummary],
        relation_hints: list[Any],
        constraint_memory: dict,
        authoritative_memory: dict,
        knowledge_base: dict,
        sampled_patterns: list[dict],
        filename_sample_groups: list[dict],
        question_memory: dict | None = None,
    ) -> dict:
        """Build the compact memory endpoint consumed by downstream agents.

        The pack is intentionally layered: official/original task evidence can
        constrain the task contract, while data profiles only support or
        challenge that contract. This prevents heuristic data cognition from
        silently inventing submission schemas, metrics, or random seeds.
        """
        memory = authoritative_memory if isinstance(authoritative_memory, dict) else {}
        contract = memory.get("submission_contract") if isinstance(memory.get("submission_contract"), dict) else {}
        constraints = constraint_memory if isinstance(constraint_memory, dict) else {}

        source_files = [str(x) for x in memory.get("source_files", []) if str(x).strip()]
        evidence_items = []
        for item in memory.get("evidence_items", []) if isinstance(memory.get("evidence_items"), list) else []:
            if not isinstance(item, dict):
                continue
            evidence_items.append(
                {
                    "source_path": str(item.get("source_path", "")),
                    "source_type": str(item.get("source_type", "")),
                    "priority": str(item.get("priority", "")),
                    "evidence": str(item.get("evidence", ""))[:1000],
                }
            )

        table_memories: list[dict[str, Any]] = []
        document_memories: list[dict[str, Any]] = []
        for fs in file_summaries:
            payload = {
                "path": fs.path,
                "role": fs.role.value if hasattr(fs.role, "value") else str(fs.role),
                "summary": str(fs.summary or "")[:1200],
                "detailed_report": str(fs.detailed_report or "")[:6000],
                "key_entities": [str(x) for x in (fs.key_entities or [])[:20]],
                "related_files": [str(x) for x in (fs.related_files or [])[:20]],
                "columns": [str(x) for x in (fs.columns or [])[:120]],
                "field_descriptions": {
                    str(col): str(desc)[:500]
                    for col, desc in (fs.column_semantics or {}).items()
                    if fs.column_semantic_meta.get(col, {}).get("source") == "llm_field_description"
                },
                "field_profiles": [
                    {
                        "name": str(p.get("name", "")),
                        "logical_type": p.get("logical_type") or p.get("dtype"),
                        "null_ratio": p.get("null_ratio"),
                        "unique_count": p.get("unique_count"),
                        "numeric_stats": p.get("numeric_stats"),
                        "datetime_stats": p.get("datetime_stats"),
                        "top_values": p.get("top_values", [])[:8] if isinstance(p.get("top_values"), list) else p.get("top_values"),
                    }
                    for p in (fs.column_profiles or [])[:120]
                    if str(p.get("name", "")).strip()
                ],
                "warnings": [str(x) for x in (fs.warnings or [])[:20]],
            }
            if fs.role == FileRole.raw_data_table:
                table_memories.append(payload)
            elif fs.role in {FileRole.task_requirement, FileRole.data_description}:
                document_memories.append(payload)

        relation_memory: list[dict[str, Any]] = []
        for hint in relation_hints[:120]:
            if hasattr(hint, "model_dump"):
                relation_memory.append(hint.model_dump())
            else:
                relation_memory.append(dict(getattr(hint, "__dict__", {})))

        submission_contract = {
            "is_defined": bool(contract.get("is_defined")),
            "is_authoritative": bool(contract.get("is_authoritative")),
            "output_filename": str(contract.get("output_filename") or "submission.csv"),
            "sample_filename": str(contract.get("sample_filename") or "sample_submission.csv"),
            "columns": [str(x) for x in contract.get("columns", []) if str(x).strip()],
            "column_descriptions": contract.get("column_descriptions", {}),
            "row_unit": str(contract.get("row_unit", "")),
            "row_count_rule": str(contract.get("row_count_rule", "")),
            "format_description": str(contract.get("format_description", "")),
            "validation_rules": [str(x) for x in contract.get("validation_rules", []) if str(x).strip()],
            "source": str(contract.get("source", "")),
            "evidence": [str(x) for x in contract.get("evidence", []) if str(x).strip()][:12],
            "confidence": contract.get("confidence", 0.0),
            "unresolved_questions": [str(x) for x in contract.get("unresolved_questions", []) if str(x).strip()],
        }

        authoritative_summary = {
            "has_authoritative_sources": bool(memory.get("has_authoritative_sources")),
            "source_files": source_files[:80],
            "summary": str(memory.get("summary", ""))[:2000],
            "task_goal": str(memory.get("task_goal", ""))[:2000],
            "input_requirements": [str(x) for x in memory.get("input_requirements", []) if str(x).strip()][:30],
            "output_requirements": [str(x) for x in memory.get("output_requirements", []) if str(x).strip()][:30],
            "evaluation_requirements": [str(x) for x in memory.get("evaluation_requirements", []) if str(x).strip()][:30],
            "constraints": [str(x) for x in memory.get("constraints", []) if str(x).strip()][:40],
            "leakage_guards": [str(x) for x in memory.get("leakage_guards", []) if str(x).strip()][:30],
            "unresolved_questions": [str(x) for x in memory.get("unresolved_questions", []) if str(x).strip()][:30],
            "context_routing_notes": [str(x) for x in memory.get("context_routing_notes", []) if str(x).strip()][:30],
            "authority_conflicts": [
                item for item in (memory.get("authority_conflicts", []) or [])[:30] if isinstance(item, dict)
            ],
            "evidence_items": evidence_items[:40],
        }

        route_base = {
            "priority_order": [
                "user task hint",
                "existing input description.md",
                "README / official requirement / spec / other task documents",
                "official sample_submission or explicitly documented output contract",
                "LLM-extracted constraint memory with evidence",
                "data field profiles and relation probes",
                "filename sampling/grouping clues",
            ],
            "do_not_invent": [
                "Do not invent submission columns or output filenames when no authoritative source defines them.",
                "Do not invent a primary metric, metric direction, row count rule, or fixed random seed.",
                "Do not override user task hints, existing description.md, README/spec constraints, or official samples with data-profile heuristics.",
                "Do not treat field names such as id/target as a submission contract unless official evidence says so.",
                "For RL or optimization tasks without an official tabular submission contract, keep the output protocol from the original description instead of fabricating sample_submission.csv.",
            ],
        }

        context_routes = {
            "task_classifier": {
                **route_base,
                "must_read": [
                    "authoritative_memory.task_goal",
                    "authoritative_memory.evaluation_requirements",
                    "submission_contract",
                    "data_memory.tables[].columns",
                ],
                "allowed_inference": "Infer task type only when official text is silent; never decide submission schema here.",
            },
            "description_writer": {
                **route_base,
                "must_read": [
                    "authoritative_memory",
                    "question_memory.answers and context_routing_notes",
                    "submission_contract",
                    "constraint_memory",
                    "data_memory.tables",
                    "filename/sample grouping memory",
                ],
                "writing_policy": [
                    "Final description.md is a polished reader-facing Chinese task document.",
                    "No reflection logs, issues/fixes, ambiguity_points, or internal agent process notes.",
                    "Use data profiles to explain files and fields; use official docs to define task/output/evaluation.",
                ],
            },
            "evaluation_contract_agent": {
                **route_base,
                "must_read": [
                    "authoritative_memory.evaluation_requirements",
                    "question_memory.answers related to evaluation/output/constraints",
                    "submission_contract.validation_rules",
                    "constraint_memory.items",
                    "data_memory.tables field profiles needed for y_true/y_pred",
                ],
                "repair_policy": "If strictness fails, return feedback to the LLM and convert gaps into explicit assumptions/rules unless the metric is truly uncomputable.",
            },
            "sample_submission_builder": {
                **route_base,
                "must_read": ["submission_contract"],
                "activation_policy": "Only build sample_submission.csv when config enables it and authoritative submission_contract.is_defined=true.",
            },
            "automl": {
                **route_base,
                "must_read": [
                    "final description.md",
                    "question_memory answers for data reading, join keys, output and evaluation",
                    "authoritative_memory",
                    "evaluation_contract",
                    "data_memory.tables field profiles",
                ],
                "priority": "When final description conflicts with original official docs, prefer original official docs and report the conflict.",
            },
        }

        return {
            "schema_version": "autorealize.agent_context_pack.v1",
            "purpose": "Shared compact memory and routing policy for AutoRealize downstream agents.",
            "task_hint": task_hint,
            "priority_order": route_base["priority_order"],
            "do_not_invent": route_base["do_not_invent"],
            "authoritative_memory": authoritative_summary,
            "submission_contract": submission_contract,
            "constraint_memory": {
                "summary": constraints.get("summary", ""),
                "items": constraints.get("items", [])[:80] if isinstance(constraints.get("items", []), list) else [],
            },
            "question_memory": _compact_question_memory(question_memory or (knowledge_base or {}).get("question_investigation", {})),
            "data_memory": {
                "tables": table_memories[:80],
                "documents": document_memories[:40],
                "relations": relation_memory,
                "sampled_filename_patterns": sampled_patterns[:80],
                "filename_sample_groups": filename_sample_groups[:120],
                "field_glossary": (knowledge_base or {}).get("field_glossary", {}),
                "metric_candidates": (knowledge_base or {}).get("metric_candidates", [])[:80],
                "time_clues": (knowledge_base or {}).get("time_clues", [])[:80],
            },
            "context_routes": context_routes,
        }

    def _append_knowledge_sections(self, path: Path, knowledge_base: dict) -> None:
        text = path.read_text(encoding="utf-8") if path.exists() else "# 数据认知文档\n"
        if "## 元知识库摘要" in text:
            return
        lines = ["", "## 元知识库摘要"]
        question_memory = _compact_question_memory(knowledge_base.get("question_investigation", {}))
        if question_memory.get("answers"):
            lines.append("### 关键调查结论")
            if question_memory.get("summary"):
                lines.append(str(question_memory["summary"]))
            for item in question_memory["answers"][:20]:
                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                confidence = str(item.get("confidence", "")).strip()
                if question or answer:
                    lines.append(f"- {question}: {answer} | confidence={confidence}")
        if question_memory.get("unresolved_questions"):
            lines.append("### 仍需注意的问题")
            for item in question_memory["unresolved_questions"][:20]:
                lines.append(f"- {item}")
        if knowledge_base.get("sampled_filename_patterns"):
            lines.append("### 高度统一命名模式")
            for item in knowledge_base["sampled_filename_patterns"][:20]:
                detail = f"- `{item['directory']}`: `{item['pattern']}` 共 {item['total']} 个，抽样: {', '.join(item['sampled'])}"
                if item.get("skipped"):
                    detail += f"，跳过 {len(item.get('skipped', []))} 个同组文件"
                if item.get("columns_preview"):
                    detail += f"，表头示例: {', '.join([str(x) for x in item.get('columns_preview', [])[:12]])}"
                lines.append(detail)
        if knowledge_base.get("filename_sample_groups"):
            lines.append("### 文件名样本ID关联")
            for item in knowledge_base["filename_sample_groups"][:30]:
                files = ", ".join([str(x) for x in item.get("files", [])[:8]])
                kinds = ", ".join(sorted({str(x) for x in (item.get("data_kinds") or {}).values()}))
                lines.append(f"- sample_id `{item.get('sample_id')}`: 数据组成={kinds}; files={files}")
        if knowledge_base.get("metric_candidates"):
            lines.append("### 指标口径候选")
            for item in knowledge_base["metric_candidates"][:30]:
                lines.append(f"- `{item['field']}` in `{item['file']}`: {item['meaning']}")
        if knowledge_base.get("time_clues"):
            lines.append("### 时序线索")
            for item in knowledge_base["time_clues"][:30]:
                lines.append(f"- `{item['field']}` in `{item['file']}`: {item['meaning']}")
        if knowledge_base.get("field_glossary"):
            lines.append("### 统一字段语义索引")
            for field, info in list(knowledge_base["field_glossary"].items())[:80]:
                files = ", ".join(list(dict.fromkeys(info.get("files", [])))[:8])
                lines.append(f"- `{field}`: {info.get('meaning', '')} | files={files}")
        path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

    def _publish_knowledge(self, file_summaries: list[FileSummary], constraint_memory: dict, knowledge_base: dict) -> None:
        store = getattr(self.services, "knowledge_store", None)
        if store is None:
            return
        entries: list[KnowledgeEntry] = []
        for fs in file_summaries:
            tags = [fs.role.value]
            if fs.role == FileRole.task_requirement:
                tags.append("task_requirement")
            text = (
                f"file: {fs.path}\n"
                f"role: {fs.role.value}\n"
                f"summary: {fs.summary}\n"
                f"detailed_report: {str(fs.detailed_report or '')[:6000]}"
            )
            entries.append(
                KnowledgeEntry(
                    entry_id=make_entry_id("file_summary", fs.path, text),
                    kind="file_summary",
                    source=fs.path,
                    text=text,
                    entities=fs.key_entities,
                    fields=fs.columns[:80],
                    tags=tags,
                    metadata={"source_metadata": fs.source_metadata, "warnings": fs.warnings},
                )
            )
            for col, meaning in fs.column_semantics.items():
                col_text = f"field `{col}` in `{fs.path}` semantic_guess: {meaning}"
                entries.append(
                    KnowledgeEntry(
                        entry_id=make_entry_id("field_glossary", fs.path, col_text),
                        kind="field_glossary",
                        source=fs.path,
                        text=col_text,
                        fields=[col],
                        tags=["field", fs.role.value],
                    )
                )
            for profile in fs.column_profiles:
                name = str(profile.get("name", ""))
                if not name:
                    continue
                stat_text = (
                    f"field `{name}` profile in `{fs.path}`: dtype={profile.get('dtype')}, "
                    f"null_ratio={profile.get('null_ratio')}, unique={profile.get('unique_count')}, "
                    f"numeric_stats={profile.get('numeric_stats')}, quantiles={profile.get('quantiles')}, "
                    f"datetime_stats={profile.get('datetime_stats')}, abnormal_tokens={profile.get('abnormal_tokens')}"
                )
                entries.append(
                    KnowledgeEntry(
                        entry_id=make_entry_id("field_profile", fs.path, stat_text),
                        kind="field_profile",
                        source=fs.path,
                        text=stat_text,
                        fields=[name],
                        tags=["profile", "statistics"],
                    )
                )
        for item in constraint_memory.get("items", []) if isinstance(constraint_memory, dict) else []:
            name = str(item.get("name", ""))
            desc = str(item.get("description", ""))
            fields = [str(x) for x in item.get("related_fields", []) if str(x).strip()]
            evidence = [str(x) for x in item.get("evidence", []) if str(x).strip()]
            priority = str(item.get("priority", "medium"))
            c_text = f"constraint: {name}\ndescription: {desc}\nevidence: {'; '.join(evidence[:6])}\nrelated_fields: {', '.join(fields[:20])}"
            tags = ["constraint", f"priority:{priority}"]
            if priority == "high":
                tags.append("hard_constraint")
            entries.append(
                KnowledgeEntry(
                    entry_id=make_entry_id("constraint", name, c_text),
                    kind="constraint",
                    source=evidence[0] if evidence else "constraint_memory",
                    text=c_text,
                    fields=fields,
                    constraints=[name],
                    tags=tags,
                )
            )
        for item in knowledge_base.get("metric_candidates", [])[:80]:
            text = f"metric_candidate: `{item.get('field')}` in `{item.get('file')}`: {item.get('meaning')}"
            entries.append(
                KnowledgeEntry(
                    entry_id=make_entry_id("metric", str(item.get("file", "")), text),
                    kind="metric",
                    source=str(item.get("file", "")),
                    text=text,
                    fields=[str(item.get("field", ""))],
                    tags=["metric", "evaluation"],
                )
            )
        for item in knowledge_base.get("time_clues", [])[:80]:
            text = f"time_clue: `{item.get('field')}` in `{item.get('file')}`: {item.get('meaning')}"
            entries.append(
                KnowledgeEntry(
                    entry_id=make_entry_id("time_clue", str(item.get("file", "")), text),
                    kind="time_clue",
                    source=str(item.get("file", "")),
                    text=text,
                    fields=[str(item.get("field", ""))],
                    tags=["time", "split"],
                )
            )
        question_memory = _compact_question_memory(knowledge_base.get("question_investigation", {}))
        for item in question_memory.get("answers", [])[:80]:
            question = str(item.get("question", ""))
            answer = str(item.get("answer", ""))
            evidence = [str(x) for x in item.get("evidence", []) if str(x).strip()]
            notes = [str(x) for x in item.get("downstream_notes", []) if str(x).strip()]
            text = (
                f"investigation_question: {question}\n"
                f"answer: {answer}\n"
                f"evidence: {'; '.join(evidence[:8])}\n"
                f"downstream_notes: {'; '.join(notes[:8])}"
            )
            entries.append(
                KnowledgeEntry(
                    entry_id=make_entry_id("question_investigation", question, text),
                    kind="question_investigation",
                    source="question_investigation_report",
                    text=text,
                    tags=["question_investigation", "verified_memory"],
                )
            )
        store.add_many(entries)
        log_event(logger, "knowledge.local_store", "ADDED", entries=len(entries))


def _compact_question_memory(question_memory: dict | None) -> dict:
    if not isinstance(question_memory, dict):
        return {"summary": "", "answers": [], "unresolved_questions": [], "context_routing_notes": []}
    answers = []
    for item in question_memory.get("answers", []) if isinstance(question_memory.get("answers", []), list) else []:
        if not isinstance(item, dict):
            continue
        answers.append(
            {
                "question_id": str(item.get("question_id", "")),
                "question": str(item.get("question", ""))[:600],
                "answer": str(item.get("answer", ""))[:1200],
                "evidence": [str(x)[:500] for x in (item.get("evidence", []) or [])[:10]],
                "confidence": str(item.get("confidence", "")),
                "remaining_uncertainty": str(item.get("remaining_uncertainty", ""))[:800],
                "downstream_notes": [str(x)[:500] for x in (item.get("downstream_notes", []) or [])[:10]],
            }
        )
    return {
        "summary": str(question_memory.get("summary", ""))[:2000],
        "answers": answers[:80],
        "unresolved_questions": [str(x)[:800] for x in (question_memory.get("unresolved_questions", []) or [])[:40]],
        "context_routing_notes": [str(x)[:800] for x in (question_memory.get("context_routing_notes", []) or [])[:40]],
    }


def _sampling_pattern_id(drel: str, pattern: str, suffix: str, schema_sig: str) -> str:
    payload = json.dumps(
        {
            "directory": drel,
            "pattern": pattern,
            "suffix": suffix,
            "schema_signature": schema_sig,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"pat_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _sampling_review_payload(item: dict[str, Any]) -> dict[str, Any]:
    files = [Path(p).name for p in item.get("_files", [])]
    return {
        "pattern_id": item.get("pattern_id"),
        "directory": item.get("directory"),
        "pattern": item.get("pattern"),
        "regex_name": item.get("regex_name"),
        "regex": item.get("regex"),
        "regex_reason": item.get("regex_reason"),
        "regex_confidence": item.get("regex_confidence"),
        "suffix": item.get("suffix"),
        "total": item.get("total"),
        "sampling_reason": item.get("sampling_reason"),
        "schema_signature": item.get("schema_signature"),
        "columns_preview": item.get("columns_preview") or [],
        "column_count": item.get("column_count"),
        "schema_error": item.get("schema_error"),
        "will_read": item.get("sampled", [])[:30],
        "will_skip": item.get("skipped", [])[:120],
        "will_skip_count": len(item.get("skipped", []) or []),
        "unmatched_candidate_files": [rel(p, item.get("_data_root")) for p in (item.get("_initially_ungrouped_paths") or [])[:80]]
        if item.get("_data_root")
        else [],
        "unmatched_candidate_count": len(item.get("_initially_ungrouped_paths") or []),
        "matched_file_names_head": files[:80],
        "matched_file_names_tail": files[-20:] if len(files) > 80 else [],
    }


def _apply_sampling_review(item: dict[str, Any], review: Any, data_root: Path) -> dict[str, Any]:
    out = dict(item)
    files: list[Path] = list(item.get("_files", []) or [])
    original_samples: list[Path] = list(item.get("_sample_paths", []) or [])

    if review is None:
        sample_paths = files
        sample_set = {p.resolve() for p in sample_paths}
        skipped_paths = [p for p in files if p.resolve() not in sample_set]
        out["_sample_paths"] = sample_paths
        out["_skipped_paths"] = skipped_paths
        out["sampled"] = [rel(p, data_root) for p in sample_paths]
        out["skipped"] = [rel(p, data_root) for p in skipped_paths]
        out["review"] = {
            "decision": "force_full_read_missing_review",
            "reason": "LLM did not return a review item for this pattern_id, so no files were skipped.",
        }
        return out

    force_full = bool(getattr(review, "force_full_read", False)) or not bool(getattr(review, "accept_sampling", True))
    extra_names = {normalize_rel_path_for_compare(x) for x in list(getattr(review, "extra_sample_files", []) or [])}
    rel_to_path = {normalize_rel_path_for_compare(rel(p, data_root)): p for p in files}
    basename_to_paths: dict[str, list[Path]] = {}
    for p in files:
        basename_to_paths.setdefault(normalize_rel_path_for_compare(p.name), []).append(p)

    if force_full:
        sample_paths = files
        decision = "force_full_read"
    else:
        sample_paths = list(original_samples)
        for key in extra_names:
            path = rel_to_path.get(key)
            if path is None:
                basename_matches = basename_to_paths.get(key, [])
                if len(basename_matches) == 1:
                    path = basename_matches[0]
            if path is not None and path not in sample_paths:
                sample_paths.append(path)
        decision = "accept_sampling_with_extra_files" if len(sample_paths) > len(original_samples) else "accept_sampling"

    sample_set = {p.resolve() for p in sample_paths}
    skipped_paths = [p for p in files if p.resolve() not in sample_set]
    out["_sample_paths"] = sample_paths
    out["_skipped_paths"] = skipped_paths
    out["sampled"] = [rel(p, data_root) for p in sample_paths]
    out["skipped"] = [rel(p, data_root) for p in skipped_paths]
    out["review"] = {
        "decision": decision,
        "accept_sampling": bool(getattr(review, "accept_sampling", True)),
        "force_full_read": bool(getattr(review, "force_full_read", False)),
        "extra_sample_files": list(getattr(review, "extra_sample_files", []) or []),
        "rewrite_regex": str(getattr(review, "rewrite_regex", "") or "")[:500],
        "reason": str(getattr(review, "reason", "") or "")[:800],
        "risk": str(getattr(review, "risk", "") or "")[:800],
    }
    return out


def _force_full_sampling_plan(item: dict[str, Any], data_root: Path, *, decision: str, reason: str) -> dict[str, Any]:
    out = dict(item)
    files: list[Path] = list(item.get("_files", []) or [])
    out["_sample_paths"] = files
    out["_skipped_paths"] = []
    out["sampled"] = [rel(p, data_root) for p in files]
    out["skipped"] = []
    out["review"] = {
        "decision": decision,
        "accept_sampling": False,
        "force_full_read": True,
        "extra_sample_files": [],
        "rewrite_regex": "",
        "reason": str(reason)[:800],
        "risk": "sampling_disabled",
    }
    return out


def normalize_rel_path_for_compare(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("./").strip()


def _llm_grouping_match_for_file(path: Path, patterns: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in patterns:
        suffixes = item.get("applies_to_suffixes") or []
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        match = item["compiled"].match(path.name)
        if not match:
            continue
        groups = match.groupdict()
        data_kind = str(groups.get(item.get("data_kind_group", "data_kind"), "")).strip()
        if not data_kind:
            continue
        ext = path.suffix.lower()
        if data_kind.lower().endswith(ext):
            data_kind = data_kind[: -len(ext)]
        safe_kind = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", data_kind).strip("_")
        return {
            "pattern": f"{{id}}__{safe_kind}",
            "sample_id": str(groups.get(item.get("sample_id_group", "sample_id"), "")).strip(),
            "data_kind": safe_kind,
            "regex_name": str(item.get("name", ""))[:80],
            "regex": str(item.get("regex", ""))[:500],
            "regex_reason": str(item.get("reason", ""))[:300],
            "regex_confidence": float(item.get("confidence", 0.0) or 0.0),
            "compiled": item.get("compiled"),
            "sample_id_group": item.get("sample_id_group", "sample_id"),
            "data_kind_group": item.get("data_kind_group", "data_kind"),
        }
    return None


def _filename_group_parts(
    path: Path,
    patterns: list[dict[str, Any]],
    *,
    allow_legacy_fallback: bool,
) -> tuple[str, str] | None:
    for item in patterns:
        suffixes = item.get("applies_to_suffixes") or []
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        match = item["compiled"].match(path.name)
        if not match:
            continue
        groups = match.groupdict()
        sample_id = str(groups.get(item.get("sample_id_group", "sample_id"), "")).strip()
        data_kind = str(groups.get(item.get("data_kind_group", "data_kind"), "")).strip()
        if sample_id and data_kind:
            ext = path.suffix.lower()
            if data_kind.lower().endswith(ext):
                data_kind = data_kind[: -len(ext)]
            return sample_id, _safe_data_kind(data_kind)

    if not allow_legacy_fallback:
        return None

    stem = path.stem
    match = re.match(r"(?i)^(?P<sample_id>[0-9a-f]{6,32})(?:__(?P<data_kind>.+))?$", stem)
    if match:
        return match.group("sample_id"), _safe_data_kind(match.group("data_kind") or path.suffix.lower().lstrip(".") or "file")
    match = re.match(r"(?i)^(?P<sample_id>[0-9a-f]{6,32})[_\-.](?P<data_kind>.+)$", stem)
    if match:
        return match.group("sample_id"), _safe_data_kind(match.group("data_kind"))
    return None


def _safe_data_kind(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(value)).strip("_")
    return cleaned or "file"


def _extract_filename_sample_groups(
    drel: str,
    files: list[Path],
    patterns: list[dict[str, Any]],
    data_root: Path,
    *,
    allow_legacy_fallback: bool,
) -> list[dict]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in files:
        if path.is_dir():
            continue
        parts = _filename_group_parts(path, patterns, allow_legacy_fallback=allow_legacy_fallback)
        if not parts:
            continue
        sample_id, data_kind = parts
        item = grouped.setdefault(sample_id, {"directory": drel, "sample_id": sample_id, "files": [], "data_kinds": {}})
        rpath = rel(path, data_root)
        item["files"].append(rpath)
        item["data_kinds"][rpath] = data_kind
    out = []
    for item in grouped.values():
        if len(item["files"]) < 2:
            continue
        item["files"] = sorted(item["files"])
        out.append(item)
    return sorted(out, key=lambda x: (str(x.get("directory", "")), str(x.get("sample_id", ""))))


def _build_sampling_candidates_from_pattern_groups(
    drel: str,
    pattern_groups: dict[tuple[str, str, str], list[Path]],
    pattern_meta: dict[tuple[str, str, str], dict[str, Any]],
    data_root: Path,
    *,
    tabular_ext: set[str],
    generic_min_group: int,
    tabular_min_group: int,
    generic_max_samples: int,
    tabular_max_samples: int,
    candidate_pool: list[Path],
    initially_ungrouped: list[Path],
    reason_prefix: str,
) -> list[dict[str, Any]]:
    sampling_candidates: list[dict[str, Any]] = []
    for (pattern, suffix, schema_sig), files in pattern_groups.items():
        files_sorted = sorted(files)
        min_group = tabular_min_group if suffix in tabular_ext else generic_min_group
        if len(files_sorted) < min_group:
            continue
        max_samples = tabular_max_samples if suffix in tabular_ext else generic_max_samples
        samples = files_sorted[:max_samples]
        skipped = [x for x in files_sorted if x not in samples]
        meta = pattern_meta.get((pattern, suffix, schema_sig), {})
        pattern_id = _sampling_pattern_id(drel, pattern, suffix, schema_sig)
        sampling_candidates.append(
            {
                "pattern_id": pattern_id,
                "directory": drel,
                "pattern": f"{pattern}{suffix}",
                "regex_name": meta.get("regex_name") or "",
                "regex": meta.get("regex") or "",
                "regex_reason": meta.get("regex_reason") or "",
                "regex_confidence": meta.get("regex_confidence"),
                "suffix": suffix,
                "total": len(files_sorted),
                "sampled": [rel(x, data_root) for x in samples],
                "skipped": [rel(x, data_root) for x in skipped],
                "planned_sampled": [rel(x, data_root) for x in samples],
                "planned_skipped": [rel(x, data_root) for x in skipped],
                "sampling_reason": (
                    f"{reason_prefix}_filename_pattern_and_schema"
                    if suffix in tabular_ext
                    else f"{reason_prefix}_filename_pattern"
                ),
                "schema_signature": meta.get("schema_signature") or None,
                "columns_preview": meta.get("columns_preview") or [],
                "column_count": meta.get("column_count"),
                "schema_error": meta.get("schema_error"),
                "_files": files_sorted,
                "_sample_paths": samples,
                "_skipped_paths": skipped,
                "_candidate_pool": sorted(candidate_pool),
                "_initially_ungrouped_paths": sorted(initially_ungrouped),
                "_data_root": data_root,
            }
        )
    return sampling_candidates


def _filter_generic_related_hints(related: list[str], *, strong_peers: set[str]) -> list[str]:
    if strong_peers:
        return list(dict.fromkeys([str(x) for x in related if str(x).strip() in strong_peers]))[:12]
    out: list[str] = []
    path_like = re.compile(r"^[^\s,，;；。]+(?:\.(?:csv|xlsx|xls|json|png|jpg|jpeg|webp|tif|tiff|txt|md|pdf|docx?))$", flags=re.I)
    for item in related:
        text = str(item).strip()
        if not text:
            continue
        if text in strong_peers:
            out.append(text)
            continue
        # Once exact sample-id relations exist, related_files should be machine-usable paths,
        # not free-form speculative notes generated from a single-file view.
        if strong_peers and not path_like.match(text):
            continue
        if path_like.match(text):
            out.append(text)
    return list(dict.fromkeys(out))[:12]


def _validate_llm_grouping_regex(candidate: dict[str, Any], names: list[str]) -> dict[str, Any] | None:
    regex = str(candidate.get("regex", "")).strip()
    if not regex or len(regex) > 500:
        return None
    # Keep regexes bounded: no lookarounds/backrefs/nested catch-all patterns.
    forbidden = ["(?=", "(?!", "(?<=", "(?<!", "\\1", "\\2", ".*.*"]
    if any(token in regex for token in forbidden):
        return None
    try:
        compiled = re.compile(regex)
    except re.error:
        return None
    sample_group = str(candidate.get("sample_id_group") or "sample_id")
    kind_group = str(candidate.get("data_kind_group") or "data_kind")
    group_names = set(compiled.groupindex)
    if sample_group not in group_names or kind_group not in group_names:
        return None

    matches: list[re.Match[str]] = []
    kinds: dict[str, set[str]] = {}
    suffixes: set[str] = set()
    for name in names:
        m = compiled.match(name)
        if not m:
            continue
        # Require full filename match even if LLM forgot $.
        if m.end() != len(name):
            continue
        groups = m.groupdict()
        sample_id = str(groups.get(sample_group, "")).strip()
        data_kind = str(groups.get(kind_group, "")).strip()
        if not sample_id or not data_kind:
            continue
        matches.append(m)
        suffixes.add(Path(name).suffix.lower())
        kinds.setdefault(data_kind, set()).add(sample_id)
    if len(matches) < 3:
        return None
    if not any(len(ids) >= 3 for ids in kinds.values()):
        return None
    applies_to_suffixes = []
    for x in candidate.get("applies_to_suffixes", []):
        suffix = str(x).strip().lower()
        if not suffix:
            continue
        applies_to_suffixes.append(suffix if suffix.startswith(".") else f".{suffix}")
    if not applies_to_suffixes:
        applies_to_suffixes = sorted(suffixes)
    return {
        "name": str(candidate.get("name", ""))[:80],
        "regex": regex,
        "compiled": compiled,
        "sample_id_group": sample_group,
        "data_kind_group": kind_group,
        "applies_to_suffixes": applies_to_suffixes,
        "reason": str(candidate.get("reason", ""))[:300],
        "confidence": float(candidate.get("confidence", 0.0) or 0.0),
    }


def _is_task_like_file(path: Path) -> bool:
    lower_name = path.name.lower()
    markers = ["readme", "description", "requirement", "task", "spec", "需求", "任务", "说明"]
    return any(k in lower_name for k in markers)


def _table_schema_signature(path: Path) -> dict[str, Any]:
    """Read only table headers so similar sample files can be grouped safely."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = read_csv_auto(path, nrows=0)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=0)
        else:
            return {"signature": "", "columns_preview": [], "column_count": None}
        columns = [str(c) for c in df.columns]
        normalized = [re.sub(r"\s+", " ", c).strip().lower() for c in columns]
        payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return {
            "signature": f"cols:{len(columns)}:{digest}",
            "columns_preview": columns[:30],
            "column_count": len(columns),
        }
    except Exception as exc:
        return {
            "signature": f"schema_error:{type(exc).__name__}",
            "columns_preview": [],
            "column_count": None,
            "error": str(exc)[:300],
        }

