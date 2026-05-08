from pathlib import Path

import pandas as pd

from autorealize.agents.architect import Architect
from autorealize.agents.ground import GroundExecutor
from autorealize.config import AutoRealizeConfig
from autorealize.models import CheckerVerdict, DataContract, GroundAction
from autorealize.prompts.manager import PromptManager


class _StubArchitect(Architect):
    def __init__(self, cfg: AutoRealizeConfig) -> None:
        super().__init__(cfg, llm=None, prompt_mgr=PromptManager(cfg))

    def propose_action(self, relative_file, table_summary, task_hint, error_context=""):  # type: ignore[override]
        code = "def stage_transform(df):\n    return df.copy()\n"
        return GroundAction(
            target_file=relative_file,
            purpose="仅用于测试 checker gate",
            agent_type="transformer",
            toolset=["python_sandbox", "contract_check", "constraint_engine", "monitor", "checker"],
            action="transform",
            reason="test",
            python_code=code,
            contract=DataContract(
                required_input_columns=[],
                preserve_columns=[],
                remove_columns=[],
                add_columns=[],
                row_count_rule="same",
                value_constraints={},
                post_conditions=[],
            ),
        )

    def checker_verdict(self, purpose, before_preview, after_preview):  # type: ignore[override]
        # 关键：passed=True 但不给 verify_script，必须被 gate 拦截
        return CheckerVerdict(passed=True, reason="切片看起来没问题", verify_script="")


def test_checker_must_have_verify_script(tmp_path: Path) -> None:
    csv = tmp_path / "x.csv"
    pd.DataFrame({"id": [1, 2], "v": [10, 20]}).to_csv(csv, index=False)

    cfg = AutoRealizeConfig.from_env()
    cfg.switches.enable_checker_agent = True
    cfg.switches.enable_contract_check = True
    arch = _StubArchitect(cfg)
    ground = GroundExecutor(cfg, arch, tmp_path, tmp_path / "run")

    result = ground.execute_for_table(csv, "测试任务")
    assert result.success is False
    assert "未提供核验脚本" in result.checker_reason
