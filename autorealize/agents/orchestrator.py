from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import AutoRealizeConfig


@dataclass
class OrchestratorPhasePlan:
    """编排师输出的阶段计划。"""

    phase_id: str
    title: str
    objective: str
    enabled: bool
    weight: float
    score: float
    reason: str
    depends_on: list[str]


@dataclass
class OrchestratorDecision:
    run_data_cognition: bool
    run_task_definition: bool
    run_data_cleaning: bool
    mode: str
    phase_plans: list[OrchestratorPhasePlan]
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class Orchestrator:
    """任务编排师（论文式职责）：任务分析、阶段分解、调度决策，不直接执行代码。"""

    def __init__(self, config: AutoRealizeConfig) -> None:
        self.config = config

    def decide(
        self,
        task_hint: str = "",
        data_root: Path | None = None,
        inventory: dict | None = None,
    ) -> OrchestratorDecision:
        s = self.config.switches
        mode = "auto" if s.auto_mode else "interactive"
        inv = inventory or {}
        file_count = int(inv.get("file_count", 0))
        table_count = int(inv.get("table_count", 0))
        doc_count = int(inv.get("document_count", 0))
        image_count = int(inv.get("image_count", 0))
        archive_count = int(inv.get("archive_count", 0))
        has_task_doc = bool(inv.get("has_task_doc", False))

        if mode != "auto" or not self.config.orchestrator.auto_enable_weighted_routing:
            phase_plans = [
                OrchestratorPhasePlan(
                    phase_id="P1",
                    title="数据认知",
                    objective="建立数据全局认知与文件关系图",
                    enabled=s.run_data_cognition,
                    weight=1.0,
                    score=1.0 if s.run_data_cognition else 0.0,
                    reason="手动/非配重模式：遵循显式开关",
                    depends_on=[],
                ),
                OrchestratorPhasePlan(
                    phase_id="P2",
                    title="任务定义",
                    objective="输出无歧义、可执行的ML任务书",
                    enabled=s.run_task_definition,
                    weight=1.0,
                    score=1.0 if s.run_task_definition else 0.0,
                    reason="手动/非配重模式：遵循显式开关",
                    depends_on=["P1"],
                ),
                OrchestratorPhasePlan(
                    phase_id="P3",
                    title="数据清洗",
                    objective="在副本上修复异常并出具清洗报告",
                    enabled=s.run_data_cleaning,
                    weight=1.0,
                    score=1.0 if s.run_data_cleaning else 0.0,
                    reason="手动/非配重模式：遵循显式开关",
                    depends_on=["P1", "P2"],
                ),
            ]
            return OrchestratorDecision(
                run_data_cognition=s.run_data_cognition,
                run_task_definition=s.run_task_definition,
                run_data_cleaning=s.run_data_cleaning,
                mode=mode,
                phase_plans=phase_plans,
                rationale="direct_switch_mode",
            )

        hint = task_hint.lower()
        sig_cognition = 0.45
        if file_count > 0:
            sig_cognition += 0.15
        if doc_count > 0:
            sig_cognition += 0.1
        if archive_count > 0:
            sig_cognition += 0.1
        if image_count > 100:
            sig_cognition += 0.05
        sig_cognition = min(sig_cognition, 1.0)

        sig_task_def = 0.7
        if has_task_doc:
            sig_task_def += 0.15
        if len(task_hint.strip()) <= 20:
            sig_task_def += 0.1
        sig_task_def = min(sig_task_def, 1.0)

        sig_cleaning = 0.2
        if table_count > 0:
            sig_cleaning += 0.45
        if any(k in hint for k in ["预测", "分类", "回归", "优化", "匹配", "kaggle", "销量", "销售额"]):
            sig_cleaning += 0.15
        if any(k in hint for k in ["清洗", "异常", "缺失", "nan", "inf"]):
            sig_cleaning += 0.15
        sig_cleaning = min(sig_cleaning, 1.0)

        ocfg = self.config.orchestrator
        p1_score = ocfg.weight_data_cognition * sig_cognition
        p2_score = ocfg.weight_task_definition * sig_task_def
        p3_score = ocfg.weight_data_cleaning * sig_cleaning
        threshold = ocfg.base_min_activation_score

        run_p1 = s.run_data_cognition and (p1_score >= threshold)
        run_p2 = s.run_task_definition and (ocfg.always_run_task_definition or (p2_score >= threshold))
        run_p3 = s.run_data_cleaning and (p3_score >= threshold) and table_count > 0

        phase_plans = [
            OrchestratorPhasePlan(
                phase_id="P1",
                title="数据认知",
                objective="建立数据全局认知与文件关系图",
                enabled=run_p1,
                weight=ocfg.weight_data_cognition,
                score=round(p1_score, 4),
                reason=(f"signals(file={file_count}, doc={doc_count}, archive={archive_count}, image={image_count})"),
                depends_on=[],
            ),
            OrchestratorPhasePlan(
                phase_id="P2",
                title="任务定义",
                objective="输出无歧义、可执行的ML任务书",
                enabled=run_p2,
                weight=ocfg.weight_task_definition,
                score=round(p2_score, 4),
                reason=(
                    f"signals(task_doc={has_task_doc}, task_len={len(task_hint.strip())}, always={ocfg.always_run_task_definition})"
                ),
                depends_on=["P1"],
            ),
            OrchestratorPhasePlan(
                phase_id="P3",
                title="数据清洗",
                objective="在副本上修复异常并出具清洗报告",
                enabled=run_p3,
                weight=ocfg.weight_data_cleaning,
                score=round(p3_score, 4),
                reason=(f"signals(table={table_count}, task_keywords={self._cleaning_keyword_flag(hint)})"),
                depends_on=["P1", "P2"],
            ),
        ]
        rationale = f"weighted_auto_mode(threshold={threshold}, data_root={str(data_root) if data_root else 'n/a'})"
        return OrchestratorDecision(
            run_data_cognition=run_p1,
            run_task_definition=run_p2,
            run_data_cleaning=run_p3,
            mode=mode,
            phase_plans=phase_plans,
            rationale=rationale,
        )

    @staticmethod
    def _cleaning_keyword_flag(task_hint_lower: str) -> str:
        words = ["清洗", "异常", "缺失", "nan", "inf", "预测", "分类", "回归", "优化", "匹配"]
        hits = [w for w in words if w in task_hint_lower]
        return "|".join(hits) if hits else "none"
