from pathlib import Path

import pandas as pd

from autorealize.agents.architect import Architect
from autorealize.agents.ground import GroundExecutor
from autorealize.config import AutoRealizeConfig
from autorealize.models import CheckerVerdict, DataContract, GroundAction
from autorealize.prompts.manager import PromptManager


class _ConstraintFailArchitect(Architect):
    def __init__(self, cfg: AutoRealizeConfig) -> None:
        super().__init__(cfg, llm=None, prompt_mgr=PromptManager(cfg))

    def propose_action(self, relative_file, table_summary, task_hint, error_context=""):  # type: ignore[override]
        code = (
            "def stage_transform(df):\n"
            "    out = df.copy()\n"
            "    if 'amount' in out.columns:\n"
            "        out.loc[0, 'amount'] = -1\n"
            "    return out\n"
        )
        return GroundAction(
            target_file=relative_file,
            purpose="测试约束引擎拦截",
            agent_type="repairer",
            toolset=["python_sandbox", "contract_check", "constraint_engine", "monitor", "checker"],
            action="transform",
            reason="inject negative",
            python_code=code,
            contract=DataContract(
                required_input_columns=["amount"],
                preserve_columns=["amount"],
                remove_columns=[],
                add_columns=[],
                row_count_rule="same",
                value_constraints={"amount": {"min": 0}},
                post_conditions=["row_count_same"],
            ),
        )

    def checker_verdict(self, purpose, before_preview, after_preview):  # type: ignore[override]
        verify_script = "def stage_transform(df):\n    return df\n"
        return CheckerVerdict(passed=True, reason="ok", verify_script=verify_script)


def test_ground_constraint_engine_blocks_and_rolls_back(tmp_path: Path) -> None:
    csv = tmp_path / "x.csv"
    original = pd.DataFrame({"id": [1, 2], "amount": [10, 20]})
    original.to_csv(csv, index=False)

    cfg = AutoRealizeConfig.from_env()
    cfg.switches.enable_checker_agent = True
    cfg.switches.enable_contract_check = True
    cfg.switches.enable_constraint_engine = True
    cfg.sampling.max_refine_per_level = 0

    arch = _ConstraintFailArchitect(cfg)
    ground = GroundExecutor(cfg, arch, tmp_path, tmp_path / "run")

    result = ground.execute_for_table(csv, "测试任务")
    assert result.success is False
    assert any("min" in x for x in result.constraint_issues)

    after = pd.read_csv(csv)
    assert after.equals(original)
