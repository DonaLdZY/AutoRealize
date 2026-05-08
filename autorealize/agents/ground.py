from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import AutoRealizeConfig
from ..execution.contracts import check_contract
from ..execution.constraint_engine import ConstraintEngine
from ..execution.monitor import RuleMonitor
from ..execution.progressive import ProgressiveSampler
from ..execution.rollback import SnapshotManager
from ..logging_utils import log_event
from ..profiling.stats import profile_dataframe, read_table
from ..utils.filesystem import rel
from .architect import Architect
from .ground_agents import GroundAgentFactory

logger = logging.getLogger(__name__)


@dataclass
class CleaningActionLog:
    file: str
    action: str
    reason: str
    success: bool
    monitor_alerts: list[str] = field(default_factory=list)
    contract_issues: list[str] = field(default_factory=list)
    constraint_issues: list[str] = field(default_factory=list)
    checker_reason: str = ""
    script: str = ""
    artifacts: list[str] = field(default_factory=list)


class GroundExecutor:
    """Ground 层：真正执行脚本、监控、回滚。"""

    def __init__(
        self,
        config: AutoRealizeConfig,
        architect: Architect,
        workspace_root: Path,
        run_dir: Path,
    ) -> None:
        self.config = config
        self.architect = architect
        self.workspace_root = workspace_root
        self.run_dir = run_dir
        self.monitor = RuleMonitor()
        self.constraint_engine = ConstraintEngine(
            fail_on_unknown_rule=self.config.switches.constraint_fail_on_unknown_rule
        )
        self.sampler = ProgressiveSampler(config)
        self.snapshots = SnapshotManager(workspace_root, run_dir / "snapshots")

    def _table_summary(self, df: pd.DataFrame) -> dict:
        prof = profile_dataframe(df, top_k=self.config.data.category_top_k)
        return {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": [p.name for p in prof],
            "null_ratios": {p.name: p.null_ratio for p in prof},
            "numeric_stats": {p.name: p.numeric_stats for p in prof if p.numeric_stats},
            "abnormal_tokens": {p.name: p.abnormal_tokens for p in prof if p.abnormal_tokens},
        }

    def execute_for_table(self, file_path: Path, task_hint: str) -> CleaningActionLog:
        relative = rel(file_path, self.workspace_root)
        logger.info("[P3] [Ground] 激活清洗智能体: file=%s", relative)
        log_event(logger, "agent.ground", "ACTIVATED", file=relative)
        before_df = read_table(
            file_path,
            json_flatten_sep=self.config.data.json_flatten_sep,
            json_flatten_max_level=self.config.data.json_flatten_max_level,
            json_keep_raw_nested_columns=self.config.data.json_keep_raw_nested_columns,
        )
        summary = self._table_summary(before_df)

        action = self.architect.propose_action(relative, summary, task_hint)
        logger.info("[P3] [Ground] 动作提案: action=%s reason=%s", action.action, action.reason[:120])
        log_event(
            logger,
            "agent.architect",
            "COMPLETED",
            task="propose_action",
            file=relative,
            action=action.action,
            agent_type=action.agent_type,
        )
        snap = self.snapshots.create(
            name=relative.replace("/", "__") + ".before",
            files=[file_path],
        )
        logger.info("[P3] [Ground] 已创建快照: %s", snap.name)
        log_event(logger, "agent.snapshot", "CREATED", file=relative, snapshot=snap.name)
        action_log = CleaningActionLog(
            file=relative,
            action=f"{action.agent_type}/{action.action}",
            reason=action.reason,
            success=False,
            script=action.python_code,
        )
        artifact_root = self.run_dir / "ground_artifacts" / relative.replace("/", "__")
        agent = GroundAgentFactory.create(
            action.agent_type,
            toolset=action.toolset,
            artifact_root=artifact_root,
            source_file=relative,
        )
        logger.info(
            "[P3] [Ground] ??????: type=%s class=%s toolset=%s writes_data=%s",
            action.agent_type,
            agent.__class__.__name__,
            action.toolset,
            getattr(agent, "writes_data", False),
        )
        log_event(
            logger,
            "agent.ground_worker",
            "CREATED",
            file=relative,
            agent_type=action.agent_type,
            worker_class=agent.__class__.__name__,
            writes_data=getattr(agent, "writes_data", False),
        )

        if action.action == "noop":
            action_log.success = True
            return action_log

        last_error = ""
        for retry in range(self.config.sampling.max_refine_per_level + 1):
            logger.info("[P3] [Ground] ?????: file=%s retry=%s type=%s", relative, retry, action.agent_type)
            log_event(
                logger,
                "agent.ground_worker",
                "ACTIVATED",
                file=relative,
                retry=retry,
                agent_type=action.agent_type,
            )
            agent_result = agent.invoke(action, before_df, self.sampler)
            if not agent_result.success or agent_result.after_df is None:
                last_error = agent_result.error or "???????"
                logger.warning("[P3] [Ground] ???????: %s", last_error[:200])
                log_event(logger, "agent.ground_worker", "FAILED", file=relative, retry=retry, error=last_error[:180])
                if retry < self.config.sampling.max_refine_per_level:
                    logger.info("[P3] [Ground] ??????????")
                    log_event(logger, "agent.architect", "ACTIVATED", task="repair_action", file=relative, next_retry=retry + 1)
                    action = self.architect.propose_action(relative, summary, task_hint, error_context=last_error)
                    action_log.action = f"{action.agent_type}/{action.action}"
                    action_log.script = action.python_code
                    agent = GroundAgentFactory.create(
                        action.agent_type,
                        toolset=action.toolset,
                        artifact_root=artifact_root,
                        source_file=relative,
                    )
                    log_event(logger, "agent.architect", "COMPLETED", task="repair_action", file=relative, next_retry=retry + 1)
                    continue
                self.snapshots.rollback(snap)
                logger.warning("[P3] [Ground] ??????: %s", snap.name)
                log_event(logger, "agent.snapshot", "ROLLED_BACK", file=relative, snapshot=snap.name)
                action_log.reason += f" | ????: {last_error[:200]}"
                log_event(logger, "agent.ground", "COMPLETED", file=relative, success=False)
                return action_log

            after_df = agent_result.after_df
            if agent_result.artifact_files:
                action_log.artifacts.extend(agent_result.artifact_files)
            log_event(
                logger,
                "agent.ground_worker",
                "COMPLETED",
                file=relative,
                retry=retry,
                artifacts=len(agent_result.artifact_files or []),
            )
            if not getattr(agent, "writes_data", False):
                action_log.success = True
                action_log.reason += f" | {agent_result.note or '????????/???'}"
                log_event(logger, "agent.ground_worker", "DELETED", file=relative, reason="read_only_completed")
                log_event(logger, "agent.ground", "COMPLETED", file=relative, success=True, mode="read_only")
                return action_log
            contract_result = check_contract(before_df, after_df, action.contract)
            action_log.contract_issues = contract_result.issues
            logger.info("[P3] [Contract] 检查结果: passed=%s issues=%s", contract_result.passed, len(contract_result.issues))
            log_event(logger, "checker.contract", "COMPLETED", file=relative, passed=contract_result.passed, issues=len(contract_result.issues))
            constraint_result = self.constraint_engine.evaluate(before_df, after_df, action.contract)
            action_log.constraint_issues = constraint_result.issues
            logger.info("[P3] [ConstraintEngine] ????: passed=%s rules=%s issues=%s", constraint_result.passed, constraint_result.checked_rules, len(constraint_result.issues))
            log_event(
                logger,
                "checker.constraint_engine",
                "COMPLETED",
                file=relative,
                passed=constraint_result.passed,
                checked_rules=constraint_result.checked_rules,
                issues=len(constraint_result.issues),
            )
            monitor_verdict = self.monitor.evaluate(before_df, after_df, revision_count=retry)
            action_log.monitor_alerts = monitor_verdict.alerts
            logger.info("[P3] [Monitor] 检查结果: ok=%s severity=%s", monitor_verdict.ok, monitor_verdict.severity)
            log_event(logger, "checker.monitor", "COMPLETED", file=relative, ok=monitor_verdict.ok, severity=monitor_verdict.severity)

            checker_reason = ""
            checker_passed = True
            if self.config.switches.enable_checker_agent:
                logger.info("[P3] [Checker] 激活检查智能体（低上下文）")
                log_event(logger, "checker.low_context_agent", "ACTIVATED", file=relative)
                checker = self.architect.checker_verdict(
                    action.purpose,
                    before_df.head(self.config.data.preview_rows).to_dict(orient="records"),
                    after_df.head(self.config.data.preview_rows).to_dict(orient="records"),
                )
                checker_reason = checker.reason
                # 规则：
                # 1) checker 可仅凭切片直接否决（passed=False 即否决）
                # 2) checker 判通过时，必须执行并通过 verify_script 才可最终通过
                if not checker.passed:
                    checker_passed = False
                    checker_reason += " | checker基于切片直接判定不通过。"
                    logger.warning("[P3] [Checker] 判定不通过: %s", checker.reason[:180])
                    log_event(logger, "checker.low_context_agent", "FAILED", file=relative, reason=checker.reason[:180])
                else:
                    verify_script = checker.verify_script.strip()
                    if not verify_script:
                        checker_passed = False
                        checker_reason += " | checker未提供核验脚本，按不通过处理。"
                        logger.warning("[P3] [Checker] 判定通过但未提供核验脚本")
                        log_event(logger, "checker.low_context_agent", "FAILED", file=relative, reason="verify_script_missing")
                    else:
                        logger.info("[P3] [Checker] 正在执行核验脚本")
                        log_event(logger, "checker.low_context_agent", "VERIFY_SCRIPT_RUNNING", file=relative)
                        verify_result = self.sampler.run(verify_script, after_df)
                        if verify_result.success:
                            checker_passed = True
                            checker_reason += " | 核验脚本渐进式执行通过。"
                            logger.info("[P3] [Checker] 核验脚本通过")
                            log_event(logger, "checker.low_context_agent", "COMPLETED", file=relative, verify_script_passed=True)
                        else:
                            checker_passed = False
                            checker_reason += (
                                " | 核验脚本渐进式执行失败。"
                                f" error={verify_result.records[-1].error if verify_result.records else 'unknown'}"
                            )
                            logger.warning("[P3] [Checker] 核验脚本失败")
                            log_event(logger, "checker.low_context_agent", "FAILED", file=relative, verify_script_passed=False)
            action_log.checker_reason = checker_reason

            contract_passed = contract_result.passed if self.config.switches.enable_contract_check else True
            constraint_passed = constraint_result.passed if self.config.switches.enable_constraint_engine else True
            if contract_passed and constraint_passed and monitor_verdict.ok and checker_passed:
                _write_back_table(file_path, after_df)
                action_log.success = True
                logger.info("[P3] [Ground] 三重检查通过，已写回: %s", relative)
                log_event(logger, "agent.ground_worker", "DELETED", file=relative, reason="write_back_completed")
                log_event(logger, "agent.ground", "COMPLETED", file=relative, success=True, wrote_back=True)
                return action_log

            # 检查不过则回滚并尝试重生
            self.snapshots.rollback(snap)
            logger.warning("[P3] [Ground] 检查未通过，已回滚: %s", relative)
            log_event(logger, "agent.snapshot", "ROLLED_BACK", file=relative, snapshot=snap.name)
            if retry < self.config.sampling.max_refine_per_level:
                err = {
                    "contract_issues": contract_result.issues,
                    "constraint_issues": constraint_result.issues,
                    "monitor_alerts": monitor_verdict.alerts,
                    "checker_reason": checker_reason,
                }
                logger.info("[P3] [Ground] 基于失败信息重生成脚本")
                log_event(logger, "agent.architect", "ACTIVATED", task="replan_after_failed_checks", file=relative, next_retry=retry + 1)
                action = self.architect.propose_action(
                    relative,
                    summary,
                    task_hint,
                    error_context=json.dumps(err, ensure_ascii=False, default=str),
                )
                log_event(logger, "agent.architect", "COMPLETED", task="replan_after_failed_checks", file=relative, next_retry=retry + 1)
                current_code = action.python_code
                action_log.script = current_code
                continue
            action_log.reason += " | 合规检查失败，已回滚。"
            logger.warning("[P3] [Ground] 超过重试上限，结束: %s", relative)
            log_event(logger, "agent.ground_worker", "DELETED", file=relative, reason="retry_limit_exceeded")
            log_event(logger, "agent.ground", "COMPLETED", file=relative, success=False)
            return action_log

        self.snapshots.rollback(snap)
        logger.warning("[P3] [Ground] 进入兜底回滚: %s", relative)
        log_event(logger, "agent.snapshot", "ROLLED_BACK", file=relative, snapshot=snap.name)
        log_event(logger, "agent.ground_worker", "DELETED", file=relative, reason="fallback_rollback")
        log_event(logger, "agent.ground", "COMPLETED", file=relative, success=False)
        return action_log


def _write_back_table(path: Path, df: pd.DataFrame) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif suffix == ".json":
        # JSON 表格场景统一回写为 records，便于下游按表读取
        path.write_text(df.to_json(orient="records", force_ascii=False), encoding="utf-8")
    else:
        df.to_excel(path, index=False)
