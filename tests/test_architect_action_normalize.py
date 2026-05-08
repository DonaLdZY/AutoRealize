from autorealize.agents.architect import Architect
from autorealize.config import AutoRealizeConfig
from autorealize.models import DataContract, GroundAction
from autorealize.prompts.manager import PromptManager


def test_architect_normalize_action_fills_toolset() -> None:
    cfg = AutoRealizeConfig.from_env()
    arch = Architect(cfg, llm=None, prompt_mgr=PromptManager(cfg))
    action = GroundAction(
        target_file="x.csv",
        purpose="test",
        agent_type="repairer",
        toolset=[],
        action="transform",
        reason="r",
        python_code="def stage_transform(df):\n    return df\n",
        contract=DataContract(),
    )
    normalized = arch._normalize_action(action)
    assert normalized.agent_type == "repairer"
    assert "python_sandbox" in normalized.toolset
    assert "constraint_engine" in normalized.toolset


def test_architect_normalize_action_for_join_planner() -> None:
    cfg = AutoRealizeConfig.from_env()
    arch = Architect(cfg, llm=None, prompt_mgr=PromptManager(cfg))
    action = GroundAction(
        target_file="x.csv",
        purpose="join",
        agent_type="join_planner",
        toolset=[],
        action="analyze",
        reason="r",
        python_code="def stage_transform(df):\n    return df\n",
        contract=DataContract(),
    )
    normalized = arch._normalize_action(action)
    assert normalized.toolset == ["table_io", "stats_profile"]
