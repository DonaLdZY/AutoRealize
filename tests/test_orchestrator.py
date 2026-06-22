from pathlib import Path

from autorealize.agents.orchestrator import Orchestrator
from autorealize.config import AutoRealizeConfig


def test_orchestrator_manual_mode_respects_switches() -> None:
    cfg = AutoRealizeConfig.from_env()
    cfg.switches.auto_mode = False
    cfg.switches.run_data_cognition = True
    cfg.switches.run_task_definition = False

    decision = Orchestrator(cfg).decide(task_hint="forecast sales", data_root=Path("."), inventory={"file_count": 1})
    assert decision.mode == "interactive"
    assert decision.run_data_cognition is True
    assert decision.run_task_definition is False
    assert len(decision.phase_plans) == 2


def test_orchestrator_auto_weighted_returns_two_phases() -> None:
    cfg = AutoRealizeConfig.from_env()
    cfg.switches.auto_mode = True
    cfg.orchestrator.auto_enable_weighted_routing = True
    cfg.orchestrator.base_min_activation_score = 0.4

    decision = Orchestrator(cfg).decide(
        task_hint="build transport matching system",
        data_root=Path("."),
        inventory={
            "file_count": 5,
            "table_count": 0,
            "document_count": 3,
            "image_count": 0,
            "archive_count": 0,
            "has_task_doc": True,
        },
    )

    assert decision.mode == "auto"
    assert decision.run_task_definition is True
    assert [p.phase_id for p in decision.phase_plans] == ["P1", "P2"]
