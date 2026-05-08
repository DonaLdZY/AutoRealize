import json
from pathlib import Path

import pandas as pd

from autorealize.agents.ground_agents import GroundAgentFactory
from autorealize.execution.progressive import ProgressiveSampler
from autorealize.config import AutoRealizeConfig
from autorealize.models import DataContract, GroundAction


def _sampler() -> ProgressiveSampler:
    cfg = AutoRealizeConfig.from_env()
    cfg.sampling.max_refine_per_level = 0
    return ProgressiveSampler(cfg)


def _base_action(agent_type: str, toolset: list[str]) -> GroundAction:
    return GroundAction(
        target_file="x.csv",
        purpose="test",
        agent_type=agent_type,
        toolset=toolset,
        action="analyze",
        reason="r",
        python_code="def stage_transform(df):\n    return df\n",
        contract=DataContract(),
    )


def test_readonly_agents_write_structured_artifacts(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2], "amount": [10.0, 20.0], "date": ["2026-01-01", "2026-01-02"]})
    sampler = _sampler()

    cases = [
        ("join_planner", ["table_io", "stats_profile"], "join_plan.json"),
        ("schema_mapper", ["table_io", "stats_profile"], "schema_mapping.json"),
        ("constraint_author", ["contract_check", "constraint_engine"], "constraints_patch.json"),
        ("submission_formatter", ["table_io"], "submission_contract.json"),
    ]

    for idx, (agent_type, toolset, fname) in enumerate(cases, 1):
        root = tmp_path / f"art_{idx}"
        agent = GroundAgentFactory.create(
            agent_type=agent_type,
            toolset=toolset,
            artifact_root=root,
            source_file="x.csv",
        )
        result = agent.run(_base_action(agent_type, toolset), df, sampler)
        assert result.success is True
        target = root / fname
        assert target.exists()
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload.get("agent_type") == agent_type
