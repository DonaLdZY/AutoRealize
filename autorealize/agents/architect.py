from __future__ import annotations

import json

from ..config import AutoRealizeConfig
from ..llm.client import LLMClient
from ..models import CritiqueResult, CritiqueSeverity, PipelinePlan
from ..prompt_cache import stable_dynamic_prompt
from ..prompts.manager import PromptManager


class Architect:
    """Architect: build and critique task plans. Requires an available LLM."""

    def __init__(self, config: AutoRealizeConfig, llm: LLMClient, prompt_mgr: PromptManager) -> None:
        self.config = config
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    def _load_fewshot_text(self, rel: str) -> str:
        content = self.prompt_mgr.load(rel)
        obj = json.loads(content)
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def build_plan(self, task_hint: str, cognition_digest: str) -> PipelinePlan:
        system = self.prompt_mgr.load("system/architect_plan.md")
        stable, dynamic = stable_dynamic_prompt(
            stable={
                "task_hint": task_hint,
                "data_cognition_digest": cognition_digest[:12000],
            },
            dynamic={"instruction": "Build the task definition plan."},
            stable_title="Stable task and data cognition context",
            dynamic_title="Dynamic planning request",
        )
        fewshot = self._load_fewshot_text("fewshot/plan_fewshot.json") if self.config.switches.enable_fewshot else ""
        return self.llm.ask_structured(
            model_cls=PipelinePlan,
            system_prompt=system,
            user_prompt=dynamic,
            prompt_name="architect_plan",
            fewshot=fewshot,
            static_context_prompt=stable,
            dynamic_user_prompt=dynamic,
        )

    def critique_plan(self, plan: PipelinePlan) -> CritiqueResult:
        issues: list[str] = []
        if not plan.evaluation_metric or not plan.evaluation_formula:
            issues.append("缺少统一评估指标/公式")
        if len(plan.phases) == 0:
            issues.append("计划缺少阶段")
        sev = CritiqueSeverity.none if not issues else CritiqueSeverity.major
        return CritiqueResult(level="plan", severity=sev, issues=issues, suggestions=["补全评估与阶段定义"])

    def critique_expansion(self, plan: PipelinePlan) -> CritiqueResult:
        issues: list[str] = []
        for phase in plan.phases:
            if len(phase.substeps) == 0 or len(phase.substeps) > 3:
                issues.append(f"{phase.phase_id} 子步骤数量不在 1-3 范围")
        sev = CritiqueSeverity.none if not issues else CritiqueSeverity.major
        return CritiqueResult(level="expansion", severity=sev, issues=issues, suggestions=["每阶段控制 1-3 个子步骤"])
