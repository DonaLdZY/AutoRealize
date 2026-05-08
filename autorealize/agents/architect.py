from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import AutoRealizeConfig
from ..llm.client import LLMClient
from ..models import CheckerVerdict, CritiqueResult, CritiqueSeverity, GroundAction, PipelinePlan, PlanPhase
from ..prompts.manager import PromptBlock, PromptManager

logger = logging.getLogger(__name__)


class Architect:
    """架构师：负责计划、动作设计与审查。"""

    def __init__(self, config: AutoRealizeConfig, llm: LLMClient | None, prompt_mgr: PromptManager) -> None:
        self.config = config
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    def _load_fewshot_text(self, rel: str) -> str:
        try:
            content = self.prompt_mgr.load(rel)
            obj = json.loads(content)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            return ""

    def build_plan(self, task_hint: str, cognition_digest: str) -> PipelinePlan:
        if self.llm is None:
            return self._fallback_plan(task_hint)
        system = self.prompt_mgr.load("system/architect_plan.md")
        user = f"任务描述:\n{task_hint}\n\n数据认知摘要:\n{cognition_digest[:12000]}"
        fewshot = self._load_fewshot_text("fewshot/plan_fewshot.json") if self.config.switches.enable_fewshot else ""
        try:
            return self.llm.ask_structured(
                model_cls=PipelinePlan,
                system_prompt=system,
                user_prompt=user,
                prompt_name="architect_plan",
                fewshot=fewshot,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("build_plan 失败，使用兜底: %s", exc)
            return self._fallback_plan(task_hint)

    def critique_plan(self, plan: PipelinePlan) -> CritiqueResult:
        issues: list[str] = []
        if not plan.evaluation_metric or not plan.evaluation_formula:
            issues.append("缺少统一评估指标/公式")
        if len(plan.phases) == 0:
            issues.append("计划缺少阶段")
        sev = CritiqueSeverity.none if not issues else CritiqueSeverity.major
        return CritiqueResult(level="plan", severity=sev, issues=issues, suggestions=["补全评估与阶段"])

    def critique_expansion(self, plan: PipelinePlan) -> CritiqueResult:
        issues: list[str] = []
        for phase in plan.phases:
            if len(phase.substeps) == 0 or len(phase.substeps) > 3:
                issues.append(f"{phase.phase_id} 子步骤数量不在 1-3 范围")
        sev = CritiqueSeverity.none if not issues else CritiqueSeverity.major
        return CritiqueResult(level="expansion", severity=sev, issues=issues, suggestions=["每阶段控制 1-3 子步骤"])

    def propose_action(
        self,
        relative_file: str,
        table_summary: dict[str, Any],
        task_hint: str,
        error_context: str = "",
    ) -> GroundAction:
        if self.llm is None:
            return self._normalize_action(self._fallback_action(relative_file, table_summary))
        system = self.prompt_mgr.load("system/architect_action.md")
        blocks = [
            PromptBlock("task", 1, f"任务: {task_hint}"),
            PromptBlock("file", 1, f"文件: {relative_file}"),
            PromptBlock("summary", 2, f"数据摘要: {json.dumps(table_summary, ensure_ascii=False)[:10000]}"),
            PromptBlock("error", 1, f"错误上下文: {error_context[:2000]}" if error_context else ""),
        ]
        user = self.prompt_mgr.build([b for b in blocks if b.content])
        fewshot = (
            self._load_fewshot_text("fewshot/ground_action_fewshot.json")
            if self.config.switches.enable_fewshot
            else ""
        )
        try:
            action = self.llm.ask_structured(
                model_cls=GroundAction,
                system_prompt=system,
                user_prompt=user,
                prompt_name="architect_action",
                fewshot=fewshot,
            )
            return self._normalize_action(action)
        except Exception as exc:  # noqa: BLE001
            logger.warning("propose_action 失败，使用兜底: %s", exc)
            return self._normalize_action(self._fallback_action(relative_file, table_summary))

    def _normalize_action(self, action: GroundAction) -> GroundAction:
        """?? agent_type/toolset??? Ground ?????"""
        at = (action.agent_type or "").strip().lower()
        if not at:
            at = "transformer"
            action.agent_type = at
        if not action.toolset:
            if at in {"reader"}:
                action.toolset = ["table_io"]
            elif at in {"profiler"}:
                action.toolset = ["stats_profile"]
            elif at in {"join_planner", "schema_mapper"}:
                action.toolset = ["table_io", "stats_profile"]
            elif at in {"constraint_author"}:
                action.toolset = ["contract_check", "constraint_engine"]
            elif at in {"submission_formatter"}:
                action.toolset = ["table_io"]
            elif at in {"validator"}:
                action.toolset = ["python_sandbox", "contract_check", "constraint_engine"]
            elif at in {"noop_keeper"}:
                action.toolset = []
            else:
                action.toolset = [
                    "table_io",
                    "python_sandbox",
                    "contract_check",
                    "constraint_engine",
                    "monitor",
                    "checker",
                ]
        return action

    def checker_verdict(self, purpose: str, before_preview: list[dict], after_preview: list[dict]) -> CheckerVerdict:
        if self.llm is None:
            verify_script = (
                "def stage_transform(df):\n"
                "    # 离线兜底核验：仅检查输出仍是可读DataFrame\n"
                "    if df is None:\n"
                "        raise ValueError('df 为空')\n"
                "    return df\n"
            )
            return CheckerVerdict(
                passed=True,
                reason="离线兜底：提供基础核验脚本。",
                verify_script=verify_script,
            )
        system = self.prompt_mgr.load("system/checker.md")
        user = (
            f"动作目的:\n{purpose}\n\n"
            f"清洗前切片:\n{json.dumps(before_preview, ensure_ascii=False)[:4000]}\n\n"
            f"清洗后切片:\n{json.dumps(after_preview, ensure_ascii=False)[:4000]}"
        )
        try:
            return self.llm.ask_structured(
                model_cls=CheckerVerdict,
                system_prompt=system,
                user_prompt=user,
                prompt_name="checker_verdict",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("checker_verdict 失败，默认不通过: %s", exc)
            return CheckerVerdict(passed=False, reason=f"checker 调用失败: {exc}")

    def _fallback_plan(self, task_hint: str) -> PipelinePlan:
        lower_hint = task_hint.lower()
        is_classification = any(k in lower_hint for k in ["是否", "分类", "真假", "true", "false", "transported"])
        is_time_series = any(k in task_hint for k in ["下个月", "次月", "时间序列", "环比", "同比"])
        if is_classification:
            task_type = "binary_classification"
            metric = "Accuracy"
            formula = "Accuracy = (TP + TN) / (TP + TN + FP + FN)"
        elif is_time_series:
            task_type = "time_series_regression"
            metric = "RMSE"
            formula = "RMSE = sqrt(mean((y_pred - y_true)^2))"
        else:
            task_type = "optimization_or_rl"
            metric = "MatchSuccessRate"
            formula = "MatchSuccessRate = matched_orders / total_orders"
        return PipelinePlan(
            task_type=task_type,
            objectives=[task_hint],
            phases=[
                PlanPhase(
                    phase_id="P1",
                    title="数据认知",
                    objective="识别文件作用、字段含义与跨文件关系",
                    inputs=["raw_data"],
                    outputs=["data_description"],
                    substeps=["文档摘要", "表格统计", "关系发现"],
                ),
                PlanPhase(
                    phase_id="P2",
                    title="任务定义",
                    objective="生成可执行 description.md",
                    inputs=["data_description"],
                    outputs=["description.md"],
                    substeps=["指标定义", "submission 规范定义"],
                ),
                PlanPhase(
                    phase_id="P3",
                    title="数据清洗",
                    objective="最小改动修复异常并输出清洗报告",
                    inputs=["workspace_copy"],
                    outputs=["cleaning_report.md"],
                    substeps=["异常检测", "脚本执行与回滚", "检查验收"],
                ),
            ],
            evaluation_metric=metric,
            evaluation_formula=formula,
            submission_spec="sample_submission.csv: [id, target]",
        )

    def _fallback_action(self, relative_file: str, table_summary: dict[str, Any]) -> GroundAction:
        columns = [str(c) for c in table_summary.get("columns", [])]
        nullable = [
            c
            for c, r in table_summary.get("null_ratios", {}).items()
            if isinstance(r, (int, float)) and r >= self.config.data.mostly_null_threshold
        ]
        if nullable:
            drop_col = nullable[0]
            code = (
                "def stage_transform(df):\n"
                "    out = df.copy()\n"
                f"    if '{drop_col}' in out.columns:\n"
                f"        out = out.drop(columns=['{drop_col}'])\n"
                "    return out\n"
            )
            action = "transform"
            agent_type = "repairer"
            reason = f"{drop_col} 几乎全空，按最小改动删除。"
            remove_columns = [drop_col]
        else:
            code = "def stage_transform(df):\n    return df.copy()\n"
            action = "noop"
            agent_type = "noop_keeper"
            reason = "未发现必须修改的异常，保持原样。"
            remove_columns = []
        return GroundAction(
            target_file=relative_file,
            purpose="执行最小必要清洗并避免破坏关键字段。",
            agent_type=agent_type,
            toolset=["table_io","python_sandbox","contract_check","constraint_engine","monitor","checker"],
            action=action,
            reason=reason,
            python_code=code,
            contract={
                "required_input_columns": [],
                "preserve_columns": [c for c in columns[:8]],
                "remove_columns": remove_columns,
                "add_columns": [],
                "row_count_rule": "same",
                "value_constraints": {},
                "post_conditions": ["输出必须为可读取 DataFrame"],
            },
        )
