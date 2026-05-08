from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
import shutil
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from .agents.architect import Architect
from .agents.ground import GroundExecutor
from .agents.ground_agents import PredictSplitGeneratorGroundAgent
from .agents.orchestrator import Orchestrator
from .config import AutoRealizeConfig
from .cognition import llm_cognition_for_file
from .llm.client import LLMClient
from .logging_utils import configure_event_sink, log_event
from .models import FileRole, FileSummary, SubmissionScriptPlan, TaskClassification
from .models import AmbiguityReview
from .parsers import build_registry
from .profiling.relations import detect_relations
from .profiling.stats import profile_dataframe, read_table
from .prompts.manager import PromptManager
from .report_writer import (
    apply_eval_fixes,
    build_description_markdown,
    coverage_defects,
    description_quality_check,
    eval_ambiguity_defects,
    write_cleaning_report,
    write_data_description,
)
from .trajectory import TrajectoryLogger
from .utils.archives import archive_stem, extract_archive, is_archive_file
from .utils.filesystem import rel, safe_copytree, walk_dirs, walk_files

logger = logging.getLogger(__name__)


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
        log_event(logger, "pipeline", "RUN_STARTED", run_name=run_name, input_root=str(input_root))
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        data_out = run_dir / "data"
        safe_copytree(input_root, data_out)
        source_file_set = {rel(p, data_out) for p in walk_files(data_out)}
        report_dir = run_dir / "realize_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        configure_event_sink(report_dir / "event_stream.jsonl")
        log_event(logger, "pipeline", "WORKSPACE_COPIED", workspace=str(data_out))
        self._expand_archives(data_out, report_dir)
        _preserve_original_description(data_out, run_dir)
        source_file_set = {rel(p, data_out) for p in walk_files(data_out)}

        traj = TrajectoryLogger(report_dir)
        traj.log("bootstrap", "start", {"input_root": str(input_root), "run_name": run_name})
        registry = build_registry(self.config)

        llm_client = None
        try:
            llm_client = LLMClient(self.config, report_dir)
            traj.log("bootstrap", "llm", {"enabled": True, "model": self.config.llm.model_name})
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 鍒濆鍖栧け璐ワ紝杩涘叆绂荤嚎鍏滃簳妯″紡: %s", exc)
            traj.log("bootstrap", "llm", {"enabled": False, "error": str(exc)})
        log_event(logger, "llm", "CLIENT_READY", enabled=bool(llm_client), model=self.config.llm.model_name)

        prompt_mgr = PromptManager(self.config)
        orchestrator = Orchestrator(self.config)
        architect = Architect(self.config, llm_client, prompt_mgr)
        log_event(logger, "agent.orchestrator", "CREATED", mode=("auto" if self.config.switches.auto_mode else "interactive"))
        log_event(logger, "agent.orchestrator", "ACTIVATED")
        inventory = _collect_inventory(data_out)
        decision = orchestrator.decide(task_hint=task_hint, data_root=data_out, inventory=inventory)
        log_event(
            logger,
            "agent.orchestrator",
            "COMPLETED",
            run_data_cognition=decision.run_data_cognition,
            run_task_definition=decision.run_task_definition,
            run_data_cleaning=decision.run_data_cleaning,
        )
        logger.info(
            "[编排] 决策: data_cognition=%s | task_definition=%s | data_cleaning=%s | mode=%s | rationale=%s",
            decision.run_data_cognition,
            decision.run_task_definition,
            decision.run_data_cleaning,
            decision.mode,
            decision.rationale,
        )
        for phase in decision.phase_plans:
            logger.info(
                "[编排] 阶段=%s(%s) | enabled=%s | score=%.4f | weight=%.2f | depends_on=%s | reason=%s",
                phase.phase_id,
                phase.title,
                phase.enabled,
                phase.score,
                phase.weight,
                ",".join(phase.depends_on) if phase.depends_on else "-",
                phase.reason,
            )
        traj.log("orchestrator", "inventory", inventory)
        traj.log("orchestrator", "decision", decision.to_dict())

        # Stage 1: 鏁版嵁璁ょ煡
        file_summaries: list[FileSummary] = []
        table_columns: dict[str, list[str]] = {}
        original_requirement_texts: list[str] = []

        if decision.run_data_cognition:
            log_event(logger, "stage.P1", "ACTIVATED")
            selected_files, compact_image_dirs = _select_cognition_files(data_out, self.config)
            log_event(logger, "stage.P1", "FILES_SELECTED", count=len(selected_files), compact_image_dirs=len(compact_image_dirs))
            if self.config.parallel.enable_parallel_cognition and len(selected_files) > 1:
                workers = max(1, int(self.config.parallel.cognition_max_workers))
                log_event(logger, "stage.P1.parallel", "ACTIVATED", workers=workers, files=len(selected_files))
                futures = []
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for file in selected_files:
                        futures.append(
                            ex.submit(
                                _cognize_one_file,
                                file=file,
                                data_root=data_out,
                                registry=registry,
                                config=self.config,
                                llm_client=llm_client,
                                prompt_mgr=prompt_mgr,
                                task_hint=task_hint,
                            )
                        )
                    for fut in as_completed(futures):
                        res = fut.result()
                        file_summaries.append(res["fs"])
                        if res["columns"]:
                            table_columns[res["rpath"]] = res["columns"]
                        if res["is_requirement"] and res["summary_text"]:
                            original_requirement_texts.append(res["summary_text"])
                log_event(logger, "stage.P1.parallel", "COMPLETED", files=len(selected_files))
            else:
                for file in selected_files:
                    res = _cognize_one_file(
                        file=file,
                        data_root=data_out,
                        registry=registry,
                        config=self.config,
                        llm_client=llm_client,
                        prompt_mgr=prompt_mgr,
                        task_hint=task_hint,
                    )
                    file_summaries.append(res["fs"])
                    if res["columns"]:
                        table_columns[res["rpath"]] = res["columns"]
                    if res["is_requirement"] and res["summary_text"]:
                        original_requirement_texts.append(res["summary_text"])
            # 澶у浘鐗囩洰褰曞彧淇濈暀鏍锋湰鍥炬枃浠剁骇璁ょ煡锛屽苟琛ュ厖鐩綍绾х敤閫旀憳瑕?
            for dir_rel, sample_files in compact_image_dirs.items():
                log_event(logger, "agent.image_dir_summary", "ACTIVATED", dir=dir_rel, samples=len(sample_files))
                vision_summary = _infer_image_dir_purpose(data_out, dir_rel, sample_files, self.config)
                file_summaries.append(
                    FileSummary(
                        path=f"{dir_rel}/",
                        role=FileRole.data_description,
                        summary=vision_summary,
                        columns=[],
                        warnings=[],
                    )
                )
                log_event(logger, "agent.image_dir_summary", "COMPLETED", dir=dir_rel)
            rel_hints = detect_relations(
                table_columns,
                parallel=self.config.parallel.enable_parallel_relations,
                max_workers=self.config.parallel.relations_max_workers,
            )
            rel_map: dict[str, list[str]] = {}
            for h in rel_hints:
                rel_map.setdefault(h.left_file, []).append(h.right_file)
                rel_map.setdefault(h.right_file, []).append(h.left_file)
            for fs in file_summaries:
                fs.related_files = rel_map.get(fs.path, [])[:8]

            dir_summaries = _summarize_dirs(data_out, file_summaries)
            log_event(logger, "stage.P1", "GENERATING_FILE", file="realize_report/data_description.md")
            write_data_description(report_dir / "data_description.md", file_summaries, dir_summaries, rel_hints)
            log_event(logger, "stage.P1", "GENERATED_FILE", file="realize_report/data_description.md")
            log_event(logger, "stage.P1", "COMPLETED", files=len(file_summaries), relations=len(rel_hints))
            traj.log("data_cognition", "done", {"files": len(file_summaries), "relations": len(rel_hints)})

        # Stage 2: 浠诲姟瀹氫箟
        plan = None
        if decision.run_task_definition:
            log_event(logger, "agent.architect", "CREATED")
            log_event(logger, "stage.P2", "ACTIVATED")
            data_digest = (report_dir / "data_description.md").read_text(encoding="utf-8")[:12000]
            original_text = "\n\n".join(original_requirement_texts) if original_requirement_texts else task_hint
            log_event(logger, "agent.architect", "ACTIVATED", task="build_plan")
            plan = architect.build_plan(task_hint=task_hint, cognition_digest=data_digest)
            log_event(logger, "agent.architect", "COMPLETED", task="build_plan")
            c1 = architect.critique_plan(plan)
            c2 = architect.critique_expansion(plan)
            log_event(logger, "stage.P2", "PLAN_CRITIQUE_COMPLETED", plan_severity=c1.severity.value, expansion_severity=c2.severity.value)
            traj.log(
                "task_definition",
                "critique",
                {"plan_severity": c1.severity.value, "expansion_severity": c2.severity.value, "issues": c1.issues + c2.issues},
            )
            downstream_context = _infer_downstream_context(data_out, file_summaries, task_hint, self.config)
            if self.config.data.auto_generate_predict_split:
                _maybe_generate_predict_split(data_out, downstream_context, self.config)
                # 生成后重新推断，确保上下文与目录一致。
                downstream_context = _infer_downstream_context(data_out, file_summaries, task_hint, self.config)
            task_cls = _classify_task_type(
                llm_client=llm_client,
                prompt_mgr=prompt_mgr,
                task_hint=task_hint,
                data_digest=data_digest,
                downstream_context=downstream_context,
            )
            if task_cls is not None:
                log_event(
                    logger,
                    "agent.task_classifier",
                    "COMPLETED",
                    task_type=task_cls.task_type,
                    confidence=f"{task_cls.confidence:.3f}",
                    primary_metric=task_cls.primary_metric,
                )
                traj.log("task_definition", "task_classifier", task_cls.model_dump())
                downstream_context["task_type_hint"] = task_cls.task_type
                if task_cls.primary_metric:
                    plan.evaluation_metric = task_cls.primary_metric
                if task_cls.metric_formula:
                    plan.evaluation_formula = task_cls.metric_formula
                if task_cls.submission_schema_hint:
                    downstream_context["submission_columns"] = [str(x) for x in task_cls.submission_schema_hint if str(x).strip()]
            # P2 获取到更强的 train/test/label 证据后，回写修正 P1 文件摘要，避免前后矛盾。
            _refine_file_summaries_by_downstream_context(file_summaries, downstream_context)
            # 重新生成 data_description，确保交叉阅读（含 sample_submission）后的结论落盘。
            rel_hints_refined = detect_relations(
                table_columns,
                parallel=self.config.parallel.enable_parallel_relations,
                max_workers=self.config.parallel.relations_max_workers,
            )
            dir_summaries_refined = _summarize_dirs(data_out, file_summaries)
            write_data_description(report_dir / "data_description.md", file_summaries, dir_summaries_refined, rel_hints_refined)
            desc = build_description_markdown(
                plan,
                original_text,
                _digest_data_inventory(file_summaries),
                file_summaries=file_summaries,
                downstream_context=downstream_context,
            )
            defects = description_quality_check(desc) + coverage_defects(desc, original_text)
            missing_refs = _find_missing_file_references(desc, data_out)
            defects.extend([f"引用了不存在文件: {x}" for x in missing_refs])
            if defects and llm_client is not None:
                for i in range(self.config.prompt.description_quality_max_retries):
                    regenerated = _rewrite_mutable_sections_with_llm(
                        llm_client=llm_client,
                        prompt_mgr=prompt_mgr,
                        base_desc=desc,
                        defects=defects,
                        downstream_context=downstream_context,
                        prompt_name=f"description_retry_{i+1}",
                    )
                    if self.config.prompt.enforce_description_real_file_refs:
                        missing_refs = _find_missing_file_references(regenerated, data_out)
                        if missing_refs:
                            new_defects = (
                                description_quality_check(regenerated)
                                + coverage_defects(regenerated, original_text)
                                + [f"引用了不存在文件: {x}" for x in missing_refs]
                            )
                        else:
                            new_defects = description_quality_check(regenerated) + coverage_defects(regenerated, original_text)
                    else:
                        regenerated = _enforce_existing_file_references(regenerated, data_out)
                        new_defects = description_quality_check(regenerated) + coverage_defects(regenerated, original_text)
                    if not new_defects:
                        desc = regenerated
                        defects = []
                        break
                    defects = new_defects
            # 浣庝笂涓嬫枃鍙嶆€濇鏌ワ細浠呭熀浜庡綋鍓嶆枃妗ｏ紝寰幆鍒版棤姝т箟鎴栬揪鍒伴噸璇曚笂闄?
            desc = _resolve_eval_ambiguity(
                desc=desc,
                downstream_context=downstream_context,
                llm_client=llm_client,
                prompt_mgr=prompt_mgr,
                data_root=data_out,
            )
            log_event(logger, "stage.P2", "EVAL_CHECK_COMPLETED")

            # 浠呭鏈€缁堢増鏈仛涓€娆￠獙鏀讹紝閬垮厤娌跨敤鏃╂湡缂洪櫡
            defects = description_quality_check(desc) + coverage_defects(desc, original_text) + eval_ambiguity_defects(desc)
            missing_refs = _find_missing_file_references(desc, data_out)
            defects.extend([f"引用了不存在文件: {x}" for x in missing_refs])
            defects = list(dict.fromkeys(defects))
            if self.config.prompt.enforce_description_real_file_refs and missing_refs:
                raise RuntimeError(f"description 引用了不存在文件，且重试后仍失败: {missing_refs[:8]}")
            log_event(logger, "stage.P2", "GENERATING_FILE", file="description.md")
            (run_dir / "description.md").write_text(desc, encoding="utf-8")
            log_event(logger, "stage.P2", "GENERATED_FILE", file="description.md", defects=len(defects))
            traj.log("task_definition", "done", {"defects_after_gate": len(defects)})
            log_event(logger, "stage.P2", "GENERATING_FILE", file="sample_submission.csv")
            _generate_sample_submission(
                data_out,
                run_dir,
                self.config,
                downstream_context=downstream_context,
                llm_client=llm_client,
            )
            log_event(logger, "stage.P2", "GENERATED_FILE", file="sample_submission.csv")
            log_event(logger, "stage.P2", "COMPLETED")

        # Stage 3: 鏁版嵁娓呮礂
        cleaning_lines: list[str] = []
        if decision.run_data_cleaning:
            log_event(logger, "stage.P3", "ACTIVATED")
            candidate_files: list[Path] = []
            for file in walk_files(data_out):
                rfile = rel(file, data_out)
                if rfile not in source_file_set:
                    continue
                suffix = file.suffix.lower()
                if suffix not in {".csv", ".xlsx", ".xls", ".json"}:
                    continue
                if suffix == ".json" and not self.config.data.enable_json_cleaning:
                    log_event(logger, "stage.P3", "SKIP_FILE", file=rfile, reason="json_cleaning_disabled")
                    continue
                candidate_files.append(file)

            def _clean_one(file: Path) -> list[str]:
                rfile = rel(file, data_out)
                local_lines: list[str] = []
                try:
                    ground = GroundExecutor(self.config, architect, data_out, report_dir)
                    log_event(logger, "stage.P3", "CLEANING_FILE", file=rfile)
                    result = ground.execute_for_table(file, task_hint)
                    log_event(logger, "stage.P3", "CLEANING_RESULT", file=result.file, action=result.action, success=result.success)
                    local_lines.append(
                        f"- `{result.file}` | action={result.action} | success={result.success} | reason={result.reason}"
                    )
                    if result.monitor_alerts:
                        local_lines.append(f"  monitor: {'; '.join(result.monitor_alerts)}")
                    if result.contract_issues:
                        local_lines.append(f"  contract: {'; '.join(result.contract_issues)}")
                    if result.constraint_issues:
                        local_lines.append(f"  constraints: {'; '.join(result.constraint_issues)}")
                    if result.checker_reason:
                        local_lines.append(f"  checker: {result.checker_reason}")
                    if result.artifacts:
                        local_lines.append(f"  artifacts: {'; '.join(result.artifacts)}")
                    scripts_dir = report_dir / "cleaning_scripts"
                    scripts_dir.mkdir(parents=True, exist_ok=True)
                    script_file = scripts_dir / (result.file.replace("/", "__") + ".py")
                    script_file.write_text(result.script, encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    log_event(logger, "stage.P3", "CLEANING_FAILED", file=rfile, error=str(exc)[:180])
                    local_lines.append(f"- `{rfile}` | success=False | error={exc}")
                return local_lines

            if self.config.parallel.enable_parallel_cleaning and len(candidate_files) > 1:
                workers = max(1, int(self.config.parallel.cleaning_max_workers))
                log_event(logger, "stage.P3.parallel", "ACTIVATED", workers=workers, files=len(candidate_files))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_clean_one, f) for f in candidate_files]
                    for fut in as_completed(futures):
                        cleaning_lines.extend(fut.result())
                log_event(logger, "stage.P3.parallel", "COMPLETED", files=len(candidate_files))
            else:
                for f in candidate_files:
                    cleaning_lines.extend(_clean_one(f))
            log_event(logger, "stage.P3", "GENERATING_FILE", file="realize_report/cleaning_report.md")
            write_cleaning_report(report_dir / "cleaning_report.md", cleaning_lines)
            log_event(logger, "stage.P3", "GENERATED_FILE", file="realize_report/cleaning_report.md")
            log_event(logger, "stage.P3", "COMPLETED", tables=len(cleaning_lines))
            traj.log("data_cleaning", "done", {"table_files": len(cleaning_lines)})

        # 姹囨€荤储寮?
        traj.write_markdown_index()
        summary = {
            "run_name": run_name,
            "input_root": str(input_root),
            "data_output_root": str(data_out),
            "task_hint": task_hint,
        }
        (report_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        traj.log("finalize", "done", summary)
        self._flatten_data_to_root(run_dir, data_out, report_dir)
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
    # ????????????
    candidates = sorted(candidates, key=lambda x: (len(x.parts), str(x)))
    src = candidates[0]
    target = run_dir / "description_origin.md"
    if not target.exists():
        shutil.copy2(src, target)


def _append_output_layout_to_description(run_dir: Path, report_dir: Path) -> None:
    """在 description.md 末尾追加输出目录结构与文件职责说明。"""
    desc_path = run_dir / "description.md"
    if not desc_path.exists():
        return

    lines: list[str] = [
        "",
        "## Output Layout",
        "### Directory Tree",
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
    lines.append("### File & Directory Roles")

    role_map = {
        "description.md": "面向 ML-Master/AutoML 的任务说明文档（Kaggle 风格）",
        "sample_submission.csv": "提交样例文件（优先复用原始样例）",
        "description_origin.md": "原始数据中自带的 description.md 备份",
        "realize_report": "AutoRealize 过程报告目录（认知/清洗/轨迹/日志）",
        "data_description.md": "原始数据认知文档",
        "cleaning_report.md": "数据清洗报告（目标、动作、结果）",
        "trajectory_events.jsonl": "结构化运行事件轨迹",
        "trajectory.md": "运行轨迹索引",
        "llm_traces.jsonl": "LLM 请求与响应轨迹",
        "event_stream.jsonl": "全量结构化事件流（前端监控首选数据源）",
        "cleaning_scripts": "清洗脚本留档",
        "ground_artifacts": "Ground Agent 执行产物目录",
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

    text = desc_path.read_text(encoding="utf-8")
    if "## Output Layout" not in text:
        text = text.rstrip() + "\n" + "\n".join(lines) + "\n"
        desc_path.write_text(text, encoding="utf-8")

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
    hint = (task_hint or "").lower()
    prof_map = {str(p.get("name", "")): p for p in profiles}
    out: dict[str, str] = {}
    for c in columns:
        name = str(c)
        lc = name.lower()
        p = prof_map.get(name, {})
        unique_count = int(p.get("unique_count") or 0)
        null_ratio = float(p.get("null_ratio") or 0.0)

        if "id" in lc or "编号" in name or "单号" in name:
            out[name] = "主键/单据标识字段（用于去重、关联或提交索引）"
            continue
        if any(k in lc for k in ["date", "time", "timestamp"]) or lc.startswith("dt") or any(k in name for k in ["日期", "时间"]):
            out[name] = "时间字段（用于时间切分、滚动验证与时序特征）"
            continue
        if any(k in lc for k in ["store", "shop", "branch"]) or any(k in name for k in ["门店", "店号", "店铺", "仓"]):
            out[name] = "业务实体字段（门店/仓/站点等），可作为分组键或主特征"
            continue
        if any(k in lc for k in ["goods", "sku", "item", "product"]) or any(k in name for k in ["商品", "货品", "品类"]):
            out[name] = "商品维度字段（SKU/货品/品类等），可用于细粒度预测"
            continue
        if any(k in lc for k in ["money", "amount", "price", "sales", "revenue"]) or any(k in name for k in ["金额", "销售额", "单价", "毛利", "折扣"]):
            if "predict" in hint or "预测" in task_hint:
                out[name] = "业务数值字段（金额/销售相关），可能是目标值或关键解释变量"
            else:
                out[name] = "业务数值字段（金额/价格/收入）"
            continue
        if any(k in lc for k in ["qty", "quantity", "count", "num"]) or any(k in name for k in ["数量", "件数", "库存"]):
            out[name] = "数量/计数字段（可用于需求强度与供给能力建模）"
            continue
        if any(k in lc for k in ["target", "label"]) or lc in {"y", "fmoney", "requester_received_pizza"}:
            out[name] = "目标标签字段（训练可用，测试集通常缺失）"
            continue
        if p.get("datetime_stats"):
            out[name] = "可解析为日期时间的字段（建议保留时间语义）"
            continue
        if p.get("numeric_stats"):
            if unique_count <= 2 and null_ratio < 0.2:
                out[name] = "二值/状态型数值字段（可能表示是否、开关或事件状态）"
            else:
                out[name] = "数值型特征字段（可用于建模）"
            continue
        if unique_count == 1:
            out[name] = "常量字段（当前文件内取值单一，跨文件拼接时可能仍有作用）"
            continue
        if ("predict" in hint or "预测" in task_hint) and unique_count < 80:
            out[name] = "离散业务属性字段（可做类别编码）"
        else:
            out[name] = "文本/类别字段（需结合业务文档确认语义）"
    return out


def _refine_file_summaries_by_downstream_context(file_summaries: list[FileSummary], downstream_context: dict) -> None:
    """使用 P2 识别到的 train/test/label 语义回写 P1 摘要，修正早期误判。"""
    train_name = str(downstream_context.get("train_table", "") or "").strip()
    predict_name = str(downstream_context.get("predict_table", "") or "").strip()
    target_col = str(downstream_context.get("target_column", "") or "").strip()
    id_col = str(downstream_context.get("id_column", "") or "").strip()
    submission_cols = [str(x) for x in downstream_context.get("submission_columns", []) if str(x).strip()]
    if not target_col:
        return

    for fs in file_summaries:
        name = Path(fs.path).name
        lower_name = name.lower()

        # 强化 sample_submission 语义：它是格式规范，不是训练/预测数据。
        if "samplesubmission" in "".join(ch for ch in lower_name if ch.isalnum()) or "sample_submission" in lower_name:
            fs.role = FileRole.task_requirement
            cols_text = ", ".join(submission_cols) if submission_cols else "（列名未识别）"
            fs.summary = f"官方提交样例文件，定义提交列格式：{cols_text}。"
            if not fs.column_semantics and submission_cols:
                fs.column_semantics = {c: "提交文件约束列" for c in submission_cols}
            continue

        # 明确 train 任务语义
        if train_name and name == train_name:
            fs.role = FileRole.raw_data_table
            fs.summary = f"训练数据表，包含目标标签 `{target_col}`，用于模型训练与验证。"
            if fs.column_semantics and target_col in fs.column_semantics:
                fs.column_semantics[target_col] = "目标标签字段（训练可用，预测阶段不可见）"
            continue

        # 明确 test 任务语义，覆盖“错误目标列猜测”
        if predict_name and name == predict_name:
            fs.role = FileRole.raw_data_table
            fs.summary = (
                f"预测数据表，不包含可用于训练的真实标签 `{target_col}`；"
                f"需基于可用特征生成 `{target_col}` 预测结果并按样例提交。"
            )
            if fs.column_semantics and target_col in fs.column_semantics:
                fs.column_semantics[target_col] = "提交目标字段（预测输出列，不是测试真值标签）"
            continue

        # 其余同名衍生文件（如 extracted）也做一致性修正
        if train_name and train_name in fs.path:
            fs.role = FileRole.raw_data_table
            fs.summary = f"训练数据衍生表，语义同 `{train_name}`，目标标签为 `{target_col}`。"
        elif predict_name and predict_name in fs.path:
            fs.role = FileRole.raw_data_table
            fs.summary = (
                f"预测数据衍生表，语义同 `{predict_name}`，用于生成 `{target_col}` 预测值。"
            )


def _generate_sample_submission(
    data_root: Path,
    run_dir: Path,
    cfg: AutoRealizeConfig,
    downstream_context: dict | None = None,
    llm_client: LLMClient | None = None,
) -> None:
    """Generate a fallback sample submission file for downstream AutoML."""
    target_file = run_dir / "sample_submission.csv"
    if target_file.exists():
        return
    # 浼樺厛澶嶇敤鏁版嵁涓凡鏈?sample_submission锛岄伩鍏嶉敊璇帹鏂负涓ゅ垪琛ㄣ€?
    def _is_sample_submission_name(path: Path) -> bool:
        # 兼容 sample_submission / sampleSubmission / sample-submission 等命名
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
            return
        try:
            df_sample = pd.read_excel(sample_src)
            df_sample.to_csv(target_file, index=False, encoding="utf-8-sig")
            return
        except Exception:  # noqa: BLE001
            pass
    # 未提供官方 sample_submission 时，优先由 LLM 生成“构建样例提交”的脚本与列契约。
    table_files = [p for p in walk_files(data_root) if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}]
    if not table_files:
        return
    ctx = downstream_context or _infer_downstream_context(data_root, [], "", cfg)
    task_hint = str(ctx.get("task_hint", "")).strip()
    task_type_hint = str(ctx.get("task_type_hint", "")).strip()
    predict_name = str(ctx.get("predict_table", "")).strip()
    id_col = str(ctx.get("id_column", "id")).strip() or "id"
    target_col = str(ctx.get("target_column", "target")).strip() or "target"
    submission_cols = [str(x) for x in ctx.get("submission_columns", []) if str(x).strip()]

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
        if candidate.suffix.lower() == ".csv":
            df = pd.read_csv(candidate)
        elif candidate.suffix.lower() == ".json":
            from .utils.json_table import read_json_as_table

            df, _ = read_json_as_table(
                candidate,
                sep=cfg.data.json_flatten_sep,
                max_level=cfg.data.json_flatten_max_level,
                keep_raw_nested_columns=cfg.data.json_keep_raw_nested_columns,
            )
        else:
            df = pd.read_excel(candidate)
    except Exception:  # noqa: BLE001
        return
    if df.empty:
        return

    if id_col not in df.columns:
        fallback_id = None
        for c in df.columns:
            lc = str(c).lower()
            if "id" in lc or "number" in lc or "order" in lc:
                fallback_id = str(c)
                break
        id_col = fallback_id or str(df.columns[0])

    if not submission_cols:
        submission_cols = [id_col, target_col]

    # LLM 先决：无官方样例时，优先让 LLM 从任务语义生成提交格式与脚本。
    if llm_client is not None:
        try:
            preview = df.head(min(30, len(df))).to_dict(orient="records")
            lower_cols = [str(c).lower() for c in df.columns]
            sales_hint = (
                any(k in (task_hint or "") for k in ["销量", "销售额", "下个月", "次月"])
                or any(k in (task_hint or "").lower() for k in ["sales", "revenue", "forecast"])
                or (
                    any("store" in c or "shop" in c for c in lower_cols)
                    and any(c.startswith("dt") or "date" in c or "time" in c for c in lower_cols)
                    and any("money" in c or "sales" in c or "revenue" in c for c in lower_cols)
                )
            )
            sales_rule = ""
            if sales_hint:
                sales_rule = (
                    "本任务属于“下个月销量/销售额预测”场景："
                    "submission_columns 必须优先采用 [门店键, 日期键, 预测销量/销售额] 的三列结构。"
                    "若存在 `sstoreno`、`dtdate`、`fmoney`，优先输出 [sstoreno, dtdate, fmoney]。"
                    "禁止仅输出 [id, target] 两列。"
                )
            prompt = (
                "请输出严格 JSON，字段包含: purpose, submission_columns, python_code, id_column, target_columns。\n"
                "目标：从任务语义与数据字段出发，设计 sample_submission 列结构，并生成 Python 脚本创建 DataFrame。\n"
                f"候选预测表: {candidate.name}\n"
                f"任务描述: {task_hint}\n"
                f"任务类型提示: {task_type_hint}\n"
                f"建议 id 列: {id_col}\n"
                f"建议目标列: {submission_cols[1:] if len(submission_cols) >= 2 else [target_col]}\n"
                f"下游推断建议 submission 列: {submission_cols}\n"
                f"表头: {list(df.columns)}\n"
                f"预览: {json.dumps(preview, ensure_ascii=False)[:6000]}\n"
                f"{sales_rule}\n"
                "要求：\n"
                "1) 你给出的 `submission_columns` 必须与脚本生成的 out_df 列顺序完全一致；\n"
                "2) 脚本最后把结果写到变量 `out_df`（DataFrame）；\n"
                "3) 若任务是分类概率提交，可输出 id+多概率列；若是回归/时序预测，输出业务键+目标值。\n"
            )
            plan = llm_client.ask_structured(
                model_cls=SubmissionScriptPlan,
                system_prompt="你是提交样例脚本生成器。必须仅输出 JSON。",
                user_prompt=prompt,
                prompt_name="sample_submission_script_plan",
            )
            local_vars: dict = {"pd": pd, "df": df.copy(), "out_df": None}
            exec(plan.python_code, {}, local_vars)  # noqa: S102
            out_df = local_vars.get("out_df")
            if isinstance(out_df, pd.DataFrame) and not out_df.empty:
                if plan.submission_columns:
                    expected = [str(x) for x in plan.submission_columns if str(x).strip()]
                    if expected and list(out_df.columns) != expected:
                        # 强制按契约重排/补列，避免脚本与结构化输出不一致。
                        for col in expected:
                            if col not in out_df.columns:
                                out_df[col] = 0.0 if col == expected[-1] else ""
                        out_df = out_df[expected]
                    submission_cols = expected or submission_cols
                out_df.to_csv(target_file, index=False, encoding="utf-8-sig")
                return
        except Exception:
            pass

    out = pd.DataFrame()
    # 以 submission_cols 为唯一输出契约来源，避免与 id_col 推断冲突。
    if submission_cols:
        for i, col in enumerate(submission_cols):
            if i == len(submission_cols) - 1:
                # 最后一列默认为目标列，样例文件统一填充占位值。
                out[col] = 0.0
                continue
            if col in df.columns:
                out[col] = df[col]
            else:
                # 键列缺失时保底使用 id 列或空串。
                if i == 0 and id_col in df.columns:
                    out[col] = df[id_col].astype(str).tolist()
                else:
                    out[col] = ""
    else:
        out[id_col] = df[id_col].astype(str).tolist()
        out[target_col] = 0.0
    out.to_csv(target_file, index=False, encoding="utf-8-sig")


def _infer_downstream_context(
    data_root: Path,
    file_summaries: list[FileSummary],
    task_hint: str,
    cfg: AutoRealizeConfig,
) -> dict:
    """推断下游 AutoML 所需的 train/test/label 语义，优先使用强规则避免歧义。"""
    table_paths = [p for p in walk_files(data_root) if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}]
    hint_lower = task_hint.lower()
    placeholders = {"", "nan", "none", "null", "na", "n/a", "unknown", "?"}
    label_priority = [
        "requester_received_pizza",
        "transported",
        "species",
        "fmoney",
        "target",
        "label",
        "y",
    ]

    def _read_small_table(path: Path, nrows: int = 200) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=nrows)
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

    def _best_id_column(columns: list[str]) -> str:
        id_keywords = ["id", "order_id", "order", "request_id", "passengerid"]
        for c in columns:
            lc = c.lower()
            if any(k == lc or k in lc for k in id_keywords):
                return c
        return columns[0] if columns else "id"

    def _best_time_column(columns: list[str]) -> str:
        for c in columns:
            lc = str(c).lower()
            if lc.startswith("dt") or any(k in lc for k in ["date", "time", "month", "day"]):
                return str(c)
            if any(k in str(c) for k in ["日期", "时间", "月份", "天"]):
                return str(c)
        return ""

    def _best_store_like_column(columns: list[str]) -> str:
        for c in columns:
            lc = str(c).lower()
            if any(k in lc for k in ["store", "shop", "branch", "station"]):
                return str(c)
            if any(k in str(c) for k in ["门店", "店号", "店铺", "仓"]):
                return str(c)
        return ""

    def _best_sales_target(columns: list[str]) -> str:
        priority = [
            "fmoney",
            "sales",
            "sale_amount",
            "revenue",
            "amount",
            "qty",
            "quantity",
            "fquantity",
        ]
        lower_map = {str(c).lower(): str(c) for c in columns}
        for p in priority:
            if p in lower_map:
                return lower_map[p]
        for c in columns:
            name = str(c)
            lc = name.lower()
            if any(k in lc for k in ["money", "sales", "revenue", "amount"]):
                return name
            if any(k in name for k in ["销售额", "销量", "金额"]):
                return name
        return ""

    def _is_train_filename(name_lower: str) -> bool:
        return re.search(r"(^|[_\-.])(train|training)([_\-.]|$)", name_lower) is not None

    def _is_test_filename(name_lower: str) -> bool:
        return re.search(r"(^|[_\-.])(test|testing)([_\-.]|$)", name_lower) is not None

    table_infos: list[dict] = []
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

        # 识别 label 可用性：列存在 + 非空非占位值
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

        table_infos.append(
            {
                "path": str(table),
                "name": table.name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": columns,
                "id_col": _best_id_column(columns),
                "time_col": _best_time_column(columns),
                "store_col": _best_store_like_column(columns),
                "sales_target_col": _best_sales_target(columns),
                "is_train_name": is_train_name,
                "is_test_name": is_test_name,
                "is_submission": is_submission,
                "label_col": label_col,
                "label_score": float(label_score),
                "has_usable_label": bool(label_col),
            }
        )

    # 强规则：train+有label 优先作为训练；test+无label 优先作为预测
    has_named_train = any(t["is_train_name"] for t in table_infos)
    has_named_test = any(t["is_test_name"] for t in table_infos)

    train_table = None
    train_named = [t for t in table_infos if t["is_train_name"] and t["has_usable_label"]]
    if train_named:
        train_table = sorted(train_named, key=lambda t: (-t["label_score"], -t["rows"]))[0]
    else:
        any_labeled = [t for t in table_infos if t["has_usable_label"] and not t["is_submission"]]
        # 当存在显式 test 命名文件时，禁止把 test 文件当作训练集
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
        # 当存在显式 train 命名文件时，禁止把 train 文件当作预测集
        if has_named_train:
            any_no_label = [t for t in any_no_label if not t["is_train_name"]]
        if any_no_label:
            pred_table = sorted(any_no_label, key=lambda t: -t["rows"])[0]

    # 目标列/ID 推断
    id_column = pred_table["id_col"] if pred_table else (train_table["id_col"] if train_table else "id")
    target_column = train_table["label_col"] if train_table else "target"
    y_true_field = target_column

    has_sales_shape = False
    if train_table:
        has_sales_shape = bool(
            str(train_table.get("store_col") or "").strip()
            and str(train_table.get("time_col") or "").strip()
            and str(train_table.get("sales_target_col") or "").strip()
        )
    is_sales_forecast = (
        any(k in hint_lower for k in ["sales", "revenue", "forecast"])
        or any(k in task_hint for k in ["销量", "销售额", "次月", "下个月"])
        or has_sales_shape
    )

    if not train_table:
        if "transport" in hint_lower or "shipping" in hint_lower or "logistics" in hint_lower:
            target_column = "Transported"
            y_true_field = "Transported"
        elif is_sales_forecast:
            target_column = "predicted_sales"
            y_true_field = "fmoney"

    # 销量/销售额预测任务下，优先用业务字段作为提交锚点，避免退化为仅 id,target。
    if is_sales_forecast and train_table:
        sales_target = train_table.get("sales_target_col") or _best_sales_target(train_table.get("columns", []))
        if sales_target:
            target_column = str(sales_target)
            y_true_field = str(sales_target)

    # 类型推断：先看 label 特征，再看任务关键词
    if "下个月" in task_hint or "次月" in task_hint or "时间序列" in task_hint or has_sales_shape:
        task_type_hint = "time_series_regression"
    elif target_column.lower() in {"requester_received_pizza", "transported"}:
        task_type_hint = "binary_classification"
    elif any(k in hint_lower for k in ["classification", "class", "分类", "是否", "true", "false"]):
        task_type_hint = "binary_classification"
    elif train_table and train_table["has_usable_label"]:
        task_type_hint = "regression"
    else:
        task_type_hint = "optimization_or_rl"

    if not submission_columns:
        if is_sales_forecast:
            submission_columns = []
            store_col = ""
            time_col = ""
            if pred_table:
                store_col = str(pred_table.get("store_col") or "")
                time_col = str(pred_table.get("time_col") or "")
            if not store_col and train_table:
                store_col = str(train_table.get("store_col") or "")
            if not time_col and train_table:
                time_col = str(train_table.get("time_col") or "")
            if store_col:
                submission_columns.append(store_col)
            if time_col and time_col not in submission_columns:
                submission_columns.append(time_col)
            if target_column not in submission_columns:
                submission_columns.append(target_column)
            if len(submission_columns) < 2:
                submission_columns = [id_column, target_column]
        else:
            submission_columns = [id_column, target_column]

    train_columns = train_table["columns"] if train_table else []
    predict_columns = pred_table["columns"] if pred_table else []
    train_only_columns = [c for c in train_columns if c not in predict_columns and c != target_column]
    predict_only_columns = [c for c in predict_columns if c not in train_columns]

    return {
        "task_hint": task_hint,
        "id_column": id_column,
        "target_column": target_column,
        "y_true_field": y_true_field,
        "submission_columns": submission_columns,
        "task_type_hint": task_type_hint,
        "has_official_test_labels": False,
        "detected_tables": [t["name"] for t in table_infos][:20]
        or [fs.path for fs in file_summaries if fs.role == FileRole.raw_data_table][:20],
        "train_table": train_table["name"] if train_table else "",
        "predict_table": pred_table["name"] if pred_table else "",
        "train_columns": train_columns[:200],
        "predict_columns": predict_columns[:200],
        "train_only_columns": train_only_columns[:200],
        "predict_only_columns": predict_only_columns[:200],
    }


def _classify_task_type(
    llm_client: LLMClient | None,
    prompt_mgr: PromptManager,
    task_hint: str,
    data_digest: str,
    downstream_context: dict,
) -> TaskClassification | None:
    if llm_client is None:
        return None
    system = prompt_mgr.load("system/task_classifier.md")
    fewshot = prompt_mgr.load("fewshot/task_classifier_fewshot.json")
    light_ctx = {
        "task_hint": task_hint,
        "train_table": downstream_context.get("train_table", ""),
        "predict_table": downstream_context.get("predict_table", ""),
        "id_column": downstream_context.get("id_column", ""),
        "target_column": downstream_context.get("target_column", ""),
        "task_type_hint_pre": downstream_context.get("task_type_hint", ""),
        "submission_columns_pre": downstream_context.get("submission_columns", []),
        "train_columns": downstream_context.get("train_columns", [])[:80],
        "predict_columns": downstream_context.get("predict_columns", [])[:80],
    }
    user = (
        f"任务提示:\n{task_hint}\n\n"
        f"数据摘要:\n{data_digest[:5000]}\n\n"
        f"结构化线索:\n{json.dumps(light_ctx, ensure_ascii=False)}"
    )
    try:
        return llm_client.ask_structured(
            model_cls=TaskClassification,
            system_prompt=system,
            user_prompt=user,
            prompt_name="task_classifier",
            fewshot=fewshot,
        )
    except Exception:
        return None


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


def _cognize_one_file(
    *,
    file: Path,
    data_root: Path,
    registry,
    config: AutoRealizeConfig,
    llm_client: LLMClient | None,
    prompt_mgr: PromptManager,
    task_hint: str,
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
            source_metadata=parsed.metadata or {},
        )
        if parsed.kind == "table":
            try:
                df_stats = read_table(
                    file,
                    json_flatten_sep=config.data.json_flatten_sep,
                    json_flatten_max_level=config.data.json_flatten_max_level,
                    json_keep_raw_nested_columns=config.data.json_keep_raw_nested_columns,
                )
                prof = profile_dataframe(df_stats, top_k=config.data.category_top_k)
                fs.column_profiles = [
                    {
                        "name": p.name,
                        "dtype": p.dtype,
                        "null_ratio": p.null_ratio,
                        "unique_count": p.unique_count,
                        "numeric_stats": p.numeric_stats,
                        "quantiles": p.quantiles,
                        "datetime_stats": p.datetime_stats,
                        "abnormal_tokens": p.abnormal_tokens[:8],
                    }
                    for p in prof
                ][:120]
                fs.column_semantics = _guess_column_semantics(
                    columns=parsed.columns,
                    profiles=fs.column_profiles,
                    task_hint=task_hint,
                )
            except Exception as stats_exc:  # noqa: BLE001
                fs.warnings.append(f"字段统计失败: {stats_exc}")
        if parsed.kind == "image":
            log_event(logger, "stage.P1.image", "ACTIVATED", file=rpath)
            image_semantic = _infer_single_image_purpose(file, config)
            if image_semantic:
                fs.summary = f"{image_semantic} | {parsed.text_summary[:200]}"
                log_event(logger, "stage.P1.image", "COMPLETED", file=rpath, semantic_summary=True)
            else:
                log_event(logger, "stage.P1.image", "COMPLETED", file=rpath, semantic_summary=False)
        if llm_client is not None and parsed.kind in {"table", "document", "structured_document", "archive"}:
            log_event(logger, "agent.file_cognition", "CREATED", file=rpath, kind=parsed.kind)
            log_event(logger, "agent.file_cognition", "ACTIVATED", file=rpath)
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
            )
            if fs_llm is not None:
                base_columns = fs.columns[:]
                base_warnings = fs.warnings[:]
                base_semantics = dict(fs.column_semantics)
                base_profiles = list(fs.column_profiles)
                fs = fs_llm
                if not fs.columns:
                    fs.columns = base_columns
                if not fs.source_metadata:
                    fs.source_metadata = parsed.metadata or {}
                if base_warnings:
                    fs.warnings = list(dict.fromkeys((fs.warnings or []) + base_warnings))
                if base_semantics:
                    merged = dict(base_semantics)
                    merged.update(fs.column_semantics or {})
                    fs.column_semantics = merged
                if base_profiles and not fs.column_profiles:
                    fs.column_profiles = base_profiles
                if not fs.summary:
                    fs.summary = parsed.text_summary[:600]
            log_event(logger, "agent.file_cognition", "COMPLETED", file=rpath)
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
        for p in sample_files[: config.vllm.max_images_per_dir]:
            mime = "image/jpeg"
            if p.suffix.lower() == ".png":
                mime = "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        resp = client.chat.completions.create(
            model=config.vllm.model_name,
            messages=[
                {"role": "system", "content": "你是数据集目录识别助手。"},
                {"role": "user", "content": user_content},
            ],
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return f"{base_summary} 视觉抽样结论: {text[:300]}"
        return base_summary
    except Exception as exc:  # noqa: BLE001
        if config.vllm.fail_silently:
            return f"{base_summary} 视觉抽样失败，降级为元数据模式。"
        return f"{base_summary} 视觉抽样失败: {exc}"


def _infer_single_image_purpose(image_file: Path, config: AutoRealizeConfig) -> str:
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
        resp = client.chat.completions.create(
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
) -> tuple[str, list[str]]:
    """Low-context evaluation ambiguity reflection loop."""
    system = prompt_mgr.load("system/eval_reflector.md")
    fewshot = prompt_mgr.load("fewshot/eval_ambiguity_fewshot.json")
    defects: list[str] = []
    current = desc
    for idx in range(2):
        user = (
            "璇峰彧鍩轰簬浠ヤ笅description鏂囨湰鍒ゆ柇璇勪及鍗忚鏄惁鏃犳涔夛細\n\n"
            f"{current[:12000]}\n\n"
            "输出严格JSON。"
        )
        try:
            review = llm_client.ask_structured(
                model_cls=AmbiguityReview,
                system_prompt=system,
                user_prompt=user,
                prompt_name=f"eval_reflector_{idx+1}",
                fewshot=fewshot,
            )
        except Exception as exc:  # noqa: BLE001
            defects.append(f"eval_reflector璋冪敤澶辫触: {exc}")
            break
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
    llm_client: LLMClient | None,
    prompt_mgr: PromptManager,
    data_root: Path,
) -> str:
    current = desc
    y_true_field = str(downstream_context.get("y_true_field", downstream_context.get("target_column", "target")))
    max_rounds = 3
    for _ in range(max_rounds):
        defects = eval_ambiguity_defects(current)
        if not defects:
            logger.info("[P2-Reflect] 姝т箟妫€鏌ラ€氳繃")
            return current
        logger.info("[P2-Reflect] 鍙戠幇姝т箟锛屽皾璇曚慨澶? %s", defects[:3])
        # 鍏堝仛瑙勫垯鍖栦慨澶嶏紝淇濇寔鍙帶鍙鐜?
        patched = apply_eval_fixes(current, y_true_field=y_true_field)
        if patched != current:
            current = patched
            current = _enforce_existing_file_references(current, data_root)
            continue
        # 瑙勫垯淇笉鍔ㄦ椂鍐嶅惎鐢ㄩ浂涓婁笅鏂囧弽鎬濇櫤鑳戒綋
        if llm_client is None:
            return current
        reviewed = _run_eval_reflector_once(
            llm_client,
            prompt_mgr,
            current,
            downstream_context=downstream_context,
            y_true_field=y_true_field,
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
) -> str:
    system = prompt_mgr.load("system/eval_reflector.md")
    fewshot = prompt_mgr.load("fewshot/eval_ambiguity_fewshot.json")
    user = (
        "鍙熀浜庝笅闈㈣繖浠?description 鏂囨湰鍋氭鏌ワ紝涓嶅厑璁稿紩鐢ㄤ换浣曞閮ㄤ笂涓嬫枃銆俓n"
        "鑻ュ瓨鍦ㄦ涔夛紝璇疯緭鍑虹粨鏋勫寲淇寤鸿锛涜嫢鏃犳涔夛紝璇疯緭鍑?is_unambiguous=true銆俓n\n"
        f"{desc[:12000]}"
    )
    review = llm_client.ask_structured(
        model_cls=AmbiguityReview,
        system_prompt=system,
        user_prompt=user,
        prompt_name="eval_reflector_once",
        fewshot=fewshot,
    )
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
    lines = text.splitlines()
    order: list[str] = []
    sections: dict[str, str] = {}
    current = "__preamble__"
    bucket: list[str] = []
    order.append(current)
    for line in lines:
        if line.startswith("## "):
            sections[current] = "\n".join(bucket).rstrip() + "\n"
            current = line[3:].strip()
            order.append(current)
            bucket = [line]
        else:
            bucket.append(line)
    sections[current] = "\n".join(bucket).rstrip() + "\n"
    return order, sections


def _merge_mutable_sections(base_desc: str, rewritten_part: str) -> str:
    mutable_headers = {"Task Definition", "Evaluation", "Submission Format"}
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
            sections.get("Task Definition", ""),
            sections.get("Evaluation", ""),
            sections.get("Submission Format", ""),
        ]
    ).strip()
    system = prompt_mgr.load("system/description_section_rewriter.md")
    user = (
        f"当前可变区块:\n{mutable_now[:12000]}\n\n"
        f"缺陷清单:\n{json.dumps(defects, ensure_ascii=False)}\n\n"
        f"约束上下文:\n{json.dumps(downstream_context, ensure_ascii=False)[:3000]}\n\n"
        "你不得引用不存在的文件名；若未识别预测文件，必须明确写“未提供独立预测文件，由训练数据切分验证”。\n"
        "只输出三个二级章节：Task Definition / Evaluation / Submission Format。"
    )
    rewritten_part = llm_client.ask_text(
        system_prompt=system,
        user_prompt=user,
        prompt_name=prompt_name,
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
    generator = PredictSplitGeneratorGroundAgent(cfg)
    out = generator.generate(
        train_df=out,
        task_type=task_type,
        target_col=target_col,
        time_col=time_col,
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

