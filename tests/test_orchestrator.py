from pathlib import Path

from autorealize.agents.orchestrator import Orchestrator
from autorealize.config import AutoRealizeConfig


def test_orchestrator_manual_mode_respects_switches() -> None:
    cfg = AutoRealizeConfig.from_env()
    cfg.switches.auto_mode = False
    cfg.switches.run_data_cognition = True
    cfg.switches.run_task_definition = False
    cfg.switches.run_data_cleaning = True

    decision = Orchestrator(cfg).decide(task_hint="???????", data_root=Path("."), inventory={"file_count": 1})
    assert decision.mode == "interactive"
    assert decision.run_data_cognition is True
    assert decision.run_task_definition is False
    assert decision.run_data_cleaning is True
    assert len(decision.phase_plans) == 3


def test_orchestrator_auto_weighted_skips_cleaning_without_tables() -> None:
    cfg = AutoRealizeConfig.from_env()
    cfg.switches.auto_mode = True
    cfg.orchestrator.auto_enable_weighted_routing = True
    cfg.orchestrator.base_min_activation_score = 0.4

    decision = Orchestrator(cfg).decide(
        task_hint="???????????",
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
    assert decision.run_data_cleaning is False
