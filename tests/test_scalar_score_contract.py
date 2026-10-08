from fractions import Fraction
from types import SimpleNamespace

import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.execution_contract import build_execution_contract
from autorealize.models import EvaluationContractReview
from autorealize.modules.task_definition import TaskDefinitionModule
from autorealize.modules.types import RuntimeServices
from autorealize.report_writer import evaluation_contract_defects


def contract(**changes):
    values = dict(
        passed=True, executable=True, primary_metric="dispatch_score", metric_direction="maximize",
        metric_formula="score = completed_orders - total_cost / (1000 + total_cost)",
        prediction_unit="one frozen order population", y_true_source="orders and contract rates",
        y_pred_source="validated assignments", computation_scope="all frozen unique orders",
        aggregation_rule="Aggregate completed orders and total contract cost before scalarization.",
        validation_protocol="One shared evaluator, scale 1000 currency units frozen before search.",
        submission_checks=["Unique order assignment"], leakage_guards=["No future facts"],
        invalid_solution_rules=["Reject missing dates; no invented costs"],
        assumptions=["Fixed cost scale B=1000 currency units; test currency precision on frozen inputs."],
        audit_metrics=["completed_orders", "total_cost"],
    )
    return EvaluationContractReview(**(values | changes))


@pytest.mark.parametrize("formula", [
    "字典序分层，不合成单一标量：先最大化 completed_orders；相同时最小化 total_cost=sum(cost(o))。权重未定义。",
    "Lexicographic ranking without a scalar: maximize completed_orders then minimize total_cost=sum(cost(o))",
    "First maximize completed_orders, then minimize total_cost",
    "First maximize completed_orders, then minimize total_cost=sum(cost(o))",
    "字典序：第一层 completed_orders=count(valid_orders)，第二层 total_cost=sum(cost(o))",
    "(completed_orders, -total_cost)",
    "score = [completed_orders, -total_cost]",
])
def test_multilevel_formula_cannot_pass_even_when_model_claims_passed(formula):
    result = contract(metric_formula=formula, aggregation_rule="lexicographic")
    assert evaluation_contract_defects(result)


@pytest.mark.parametrize("shape", ["lexicographic", "vector", "pareto"])
def test_explicit_shape_is_checked_even_for_software_scope(shape):
    for scope in ("task_effectiveness", "software_contract_checks"):
        result = contract(score_representation=shape, ranking_scope=scope)
        assert any("score_representation" in defect for defect in evaluation_contract_defects(result))


def test_scalar_keeps_business_components_and_assumptions():
    assert evaluation_contract_defects(contract()) == []
    assert evaluation_contract_defects(contract(
        metric_formula="mean(abs(y_true-y_pred))", primary_metric="MAE", metric_direction="minimize",
        aggregation_rule="mean over rows", assumptions=[],
    )) == []


def test_legacy_scalar_alias_cannot_smuggle_a_different_score():
    assert any("competing scores" in defect for defect in evaluation_contract_defects(
        contract(scalar_score_formula="score = completed_orders")
    ))


def test_execution_handoff_rejects_non_scalar_and_preserves_legacy_scalar_identity():
    def build(evaluation):
        return build_execution_contract(bundle={}, context={}, evaluation=evaluation, output={}, data_schema={})
    current = contract().model_dump()
    legacy = {key: value for key, value in current.items() if key != "score_representation"}
    assert build(current)["evaluation_id"] == build(legacy)["evaluation_id"]
    broken = build({**legacy, "metric_formula": "字典序分层，不合成单一标量"})
    assert broken["readiness"] == "blocked"
    assert any("one numeric scalar" in item for item in broken["blocking_issues"])


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_real_compilation_loop_repairs_a_first_round_false_pass(tmp_path, repair_succeeds):
    bad = contract(metric_formula="字典序分层，不合成单一标量；cost=sum(cost(o))")
    responses = iter([bad, contract() if repair_succeeds else bad.model_copy(deep=True)])
    calls = []

    def ask(**kwargs):
        calls.append(kwargs)
        return next(responses)

    services = RuntimeServices(SimpleNamespace(ask_structured=ask), SimpleNamespace(load=lambda name: "rules"),
                               None, SimpleNamespace(log=lambda *args: None))
    config = AutoRealizeConfig()
    config.prompt.evaluation_contract_max_rounds = 2
    module = TaskDefinitionModule(config, services, tmp_path, tmp_path)
    result = module._review_evaluation_contract(original_text="优先完成更多订单，其次节省成本", downstream_context={})
    assert len(calls) == 2
    assert module._evaluation_contract_revision_log[0]["defects"]
    assert result.passed is repair_succeeds
    assert result.executable is repair_succeeds
    if repair_succeeds:
        assert result.authority_status == "autorealize_assumption"
    else:
        assert any("scalar" in item for item in result.residual_issues)


def test_prompt_example_preserves_integer_order_priority_in_exact_arithmetic():
    # Validate the conditional example in the prompt, not a universal float guarantee.
    def score(orders, cost):
        return orders - Fraction(cost, 1000 + cost)

    for orders in (0, 1, 2104):
        assert score(orders + 1, 10**30) > score(orders, 0)
        assert score(orders, 100) > score(orders, 101)
    # Coverage minus normalized cost reverses this same business priority.
    assert Fraction(1001, 2104) - 1 < Fraction(1000, 2104)
