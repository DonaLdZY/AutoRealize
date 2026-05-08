from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import pandas as pd

from ..config import AutoRealizeConfig
from ..execution.progressive import ProgressiveSampler
from ..logging_utils import log_event
from ..models import GroundAction

logger = logging.getLogger(__name__)

@dataclass
class AgentRunResult:
    success: bool
    after_df: pd.DataFrame | None
    error: str = ""
    note: str = ""
    artifact_files: list[str] = field(default_factory=list)


class BaseGroundAgent(ABC):
    """Ground 子代理基类。"""

    required_tools: set[str] = set()
    writes_data: bool = False

    def __init__(
        self,
        toolset: list[str] | None = None,
        artifact_root: Path | None = None,
        source_file: str = "",
    ) -> None:
        self.toolset = {str(x).strip() for x in (toolset or []) if str(x).strip()}
        self.artifact_root = artifact_root
        self.source_file = source_file

    def _check_toolset(self) -> tuple[bool, str]:
        if not self.required_tools:
            return True, ""
        missing = sorted(t for t in self.required_tools if t not in self.toolset)
        if missing:
            return False, f"toolset缺失: {missing}"
        return True, ""

    def _write_artifact(self, name: str, payload: dict) -> str | None:
        if self.artifact_root is None:
            return None
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        out = self.artifact_root / f"{name}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)

    def invoke(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        """统一包装 Ground 子智能体调用，输出生命周期日志。"""
        agent_label = f"ground.{self.__class__.__name__}"
        log_event(
            logger,
            agent_label,
            "ACTIVATED",
            source_file=self.source_file or "-",
            action=action.action,
            purpose=(action.purpose[:80] if action.purpose else "-"),
        )
        try:
            result = self.run(action, before_df, sampler)
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger,
                agent_label,
                "FAILED",
                source_file=self.source_file or "-",
                error=str(exc)[:180],
            )
            return AgentRunResult(success=False, after_df=None, error=str(exc))

        log_event(
            logger,
            agent_label,
            "COMPLETED" if result.success else "FAILED",
            source_file=self.source_file or "-",
            note=(result.note[:120] if result.note else "-"),
            artifacts=len(result.artifact_files or []),
        )
        return result

    @abstractmethod
    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        raise NotImplementedError


class ReaderGroundAgent(BaseGroundAgent):
    required_tools = {"table_io"}
    writes_data = False

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        ok, msg = self._check_toolset()
        if not ok:
            return AgentRunResult(success=False, after_df=None, error=msg)
        return AgentRunResult(success=True, after_df=before_df.copy(), note="reader: 仅读取")


class ProfilerGroundAgent(BaseGroundAgent):
    required_tools = {"stats_profile"}
    writes_data = False

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        ok, msg = self._check_toolset()
        if not ok:
            return AgentRunResult(success=False, after_df=None, error=msg)
        return AgentRunResult(success=True, after_df=before_df.copy(), note="profiler: 仅统计")


class NoopKeeperGroundAgent(BaseGroundAgent):
    writes_data = False

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        return AgentRunResult(success=True, after_df=before_df.copy(), note="noop_keeper: 保持原样")


class ValidatorGroundAgent(BaseGroundAgent):
    required_tools = {"python_sandbox"}
    writes_data = False

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        ok, msg = self._check_toolset()
        if not ok:
            return AgentRunResult(success=False, after_df=None, error=msg)
        code = action.python_code.strip()
        if not code:
            return AgentRunResult(success=True, after_df=before_df.copy(), note="validator: 无脚本，跳过")
        progressive = sampler.run(code, before_df)
        if not progressive.success or progressive.final_df is None:
            error = progressive.records[-1].error if progressive.records else "validator执行失败"
            return AgentRunResult(success=False, after_df=None, error=error)
        return AgentRunResult(success=True, after_df=before_df.copy(), note="validator: 校验脚本通过")


class JoinPlannerGroundAgent(BaseGroundAgent):
    required_tools = {"table_io", "stats_profile"}
    writes_data = False

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        ok, msg = self._check_toolset()
        if not ok:
            return AgentRunResult(success=False, after_df=None, error=msg)
        columns = [str(c) for c in before_df.columns.tolist()]
        key_candidates = [c for c in columns if "id" in c.lower() or "order" in c.lower()][:5]
        payload = {
            "agent_type": "join_planner",
            "source_file": self.source_file,
            "row_count": int(len(before_df)),
            "columns": columns,
            "candidate_join_keys": key_candidates,
            "join_plan": [
                {
                    "step": 1,
                    "description": "单表场景无需执行真实join；保留候选键供跨表阶段复用",
                    "join_type": "n/a",
                }
            ],
            "risks": ["当前为单表执行入口，未进行跨表基数验证"],
        }
        art = self._write_artifact("join_plan", payload)
        return AgentRunResult(
            success=True,
            after_df=before_df.copy(),
            note="join_planner: 已生成连接规划（只读）",
            artifact_files=[art] if art else [],
        )


class SchemaMapperGroundAgent(BaseGroundAgent):
    required_tools = {"table_io", "stats_profile"}
    writes_data = False

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        ok, msg = self._check_toolset()
        if not ok:
            return AgentRunResult(success=False, after_df=None, error=msg)
        mappings = []
        for col in [str(c) for c in before_df.columns.tolist()]:
            lc = col.lower()
            semantic = "feature"
            if "id" in lc:
                semantic = "identifier"
            elif "date" in lc or "time" in lc:
                semantic = "timestamp"
            elif "amount" in lc or "price" in lc:
                semantic = "numeric_metric"
            mappings.append(
                {
                    "source_column": col,
                    "semantic_role": semantic,
                    "confidence": 0.6,
                    "normalized_name": col.strip().lower().replace(" ", "_"),
                }
            )
        payload = {
            "agent_type": "schema_mapper",
            "source_file": self.source_file,
            "mappings": mappings,
            "conflicts": [],
        }
        art = self._write_artifact("schema_mapping", payload)
        return AgentRunResult(
            success=True,
            after_df=before_df.copy(),
            note="schema_mapper: 已生成字段语义映射（只读）",
            artifact_files=[art] if art else [],
        )


class ConstraintAuthorGroundAgent(BaseGroundAgent):
    required_tools = {"contract_check", "constraint_engine"}
    writes_data = False

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        ok, msg = self._check_toolset()
        if not ok:
            return AgentRunResult(success=False, after_df=None, error=msg)
        generated = {
            "agent_type": "constraint_author",
            "source_file": self.source_file,
            "base_contract": {
                "required_input_columns": action.contract.required_input_columns,
                "preserve_columns": action.contract.preserve_columns,
                "remove_columns": action.contract.remove_columns,
                "add_columns": action.contract.add_columns,
                "row_count_rule": action.contract.row_count_rule,
                "value_constraints": action.contract.value_constraints,
                "post_conditions": action.contract.post_conditions,
            },
            "suggested_additions": [],
        }
        if len(before_df) > 0:
            for c in [str(x) for x in before_df.columns[:5]]:
                if c not in action.contract.value_constraints:
                    null_ratio = float(before_df[c].isna().mean())
                    if null_ratio < 0.02:
                        generated["suggested_additions"].append(
                            {"column": c, "constraint": {"max_null_ratio": 0.02}}
                        )
        art = self._write_artifact("constraints_patch", generated)
        return AgentRunResult(
            success=True,
            after_df=before_df.copy(),
            note="constraint_author: 已产出可执行约束（只读）",
            artifact_files=[art] if art else [],
        )


class SubmissionFormatterGroundAgent(BaseGroundAgent):
    required_tools = {"table_io"}
    writes_data = False

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        ok, msg = self._check_toolset()
        if not ok:
            return AgentRunResult(success=False, after_df=None, error=msg)
        cols = [str(c) for c in before_df.columns.tolist()]
        id_col = next((c for c in cols if "id" in c.lower()), cols[0] if cols else "id")
        payload = {
            "agent_type": "submission_formatter",
            "source_file": self.source_file,
            "submission_contract": {
                "file_name": "submission.csv",
                "columns": [id_col, "target"],
                "row_count_rule": "equal_to_test_rows",
                "column_order_fixed": True,
            },
        }
        art = self._write_artifact("submission_contract", payload)
        return AgentRunResult(
            success=True,
            after_df=before_df.copy(),
            note="submission_formatter: 已生成提交契约（只读）",
            artifact_files=[art] if art else [],
        )


class TransformerGroundAgent(BaseGroundAgent):
    required_tools = {"python_sandbox"}
    writes_data = True

    def run(self, action: GroundAction, before_df: pd.DataFrame, sampler: ProgressiveSampler) -> AgentRunResult:
        ok, msg = self._check_toolset()
        if not ok:
            return AgentRunResult(success=False, after_df=None, error=msg)
        progressive = sampler.run(action.python_code, before_df)
        if not progressive.success or progressive.final_df is None:
            error = progressive.records[-1].error if progressive.records else "transformer执行失败"
            return AgentRunResult(success=False, after_df=None, error=error)
        return AgentRunResult(success=True, after_df=progressive.final_df, note="transformer: 变换完成")


class RepairerGroundAgent(TransformerGroundAgent):
    writes_data = True


class GroundAgentFactory:
    """根据 agent_type + toolset 实例化 Ground 子代理。"""

    _map = {
        "reader": ReaderGroundAgent,
        "profiler": ProfilerGroundAgent,
        "join_planner": JoinPlannerGroundAgent,
        "schema_mapper": SchemaMapperGroundAgent,
        "constraint_author": ConstraintAuthorGroundAgent,
        "submission_formatter": SubmissionFormatterGroundAgent,
        "transformer": TransformerGroundAgent,
        "repairer": RepairerGroundAgent,
        "validator": ValidatorGroundAgent,
        "noop_keeper": NoopKeeperGroundAgent,
    }

    @classmethod
    def create(
        cls,
        agent_type: str,
        toolset: list[str] | None = None,
        artifact_root: Path | None = None,
        source_file: str = "",
    ) -> BaseGroundAgent:
        key = (agent_type or "").strip().lower()
        agent_cls = cls._map.get(key, TransformerGroundAgent)
        agent = agent_cls(toolset=toolset, artifact_root=artifact_root, source_file=source_file)
        log_event(
            logger,
            "agent.factory",
            "CREATED",
            requested_type=key or "transformer",
            actual_class=agent_cls.__name__,
            source_file=source_file or "-",
        )
        return agent

    @classmethod
    def available_types(cls) -> list[str]:
        return sorted(cls._map.keys())


class PredictSplitGeneratorGroundAgent:
    """按任务类型从训练数据生成预测切分集（不改动原 train）。"""

    def __init__(self, config: AutoRealizeConfig) -> None:
        self.config = config

    def generate(
        self,
        train_df: pd.DataFrame,
        task_type: str,
        target_col: str,
        time_col: str = "",
    ) -> pd.DataFrame:
        out = train_df.copy()
        tt = (task_type or "").lower()
        if "time_series" in tt and time_col and time_col in out.columns:
            dt = pd.to_datetime(out[time_col], errors="coerce")
            valid = dt.notna()
            if valid.any():
                max_day = dt[valid].max()
                min_keep = max_day - pd.Timedelta(days=max(1, int(self.config.data.generated_predict_horizon_days)))
                out = out[dt >= min_keep].copy()
        else:
            ratio = float(self.config.data.generated_predict_split_ratio)
            n = max(1, int(len(out) * ratio))
            out = out.tail(n).copy()
        if target_col and target_col in out.columns:
            out[target_col] = pd.NA
        return out
