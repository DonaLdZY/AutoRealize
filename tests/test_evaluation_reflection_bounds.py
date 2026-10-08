from types import SimpleNamespace

import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.models import EvaluationContractReview
from autorealize.modules import task_definition


@pytest.mark.parametrize("rounds", [1, 2])
def test_last_reflection_cannot_return_unapplied_repair(monkeypatch, rounds):
    module = task_definition.TaskDefinitionModule.__new__(task_definition.TaskDefinitionModule)
    module.config = AutoRealizeConfig()
    module.config.prompt.evaluation_reflection_max_rounds = rounds
    module._evaluation_reflection_log = []
    repairs = []
    snapshots = []
    module._safe_apply_evaluation_contract = lambda **kwargs: (kwargs["desc"], kwargs["evaluation_contract"])
    module._reflect_evaluation_sections = lambda *args: SimpleNamespace(
        is_unambiguous=False, ambiguity_points=["Conflicting split"], fixes=["Reconcile split"])
    module._write_evaluation_contract_report = lambda value: snapshots.append(value.model_dump())
    monkeypatch.setattr(task_definition, "sync_submission_format_with_context", lambda desc, context: desc)

    def repair(**kwargs):
        repairs.append(kwargs)
        return EvaluationContractReview(passed=True, executable=True)

    module._review_evaluation_contract = repair
    with pytest.raises(RuntimeError, match="unresolved ambiguity"):
        module._apply_evaluation_contract_until_unambiguous(
            desc="original", evaluation_contract=EvaluationContractReview(passed=True, executable=True),
            original_text="requirements", data_digest="facts", downstream_context={})
    assert len(module._evaluation_reflection_log) == rounds
    assert len(repairs) == rounds - 1
    assert snapshots[-1]["passed"] is False
    assert "Conflicting split" in snapshots[-1]["residual_issues"]
