import pandas as pd

from autorealize.execution.constraint_engine import ConstraintEngine
from autorealize.models import DataContract


def test_constraint_engine_value_constraints_pass() -> None:
    before = pd.DataFrame({"id": [1, 2], "amount": [1.0, 2.0], "flag": ["A", "B"]})
    after = before.copy()
    contract = DataContract(
        value_constraints={
            "id": {"unique": True, "not_null": True, "non_negative": True},
            "amount": {"min": 0, "max": 10, "no_inf": True},
            "flag": {"allow_values": ["A", "B", "C"]},
        },
        post_conditions=["row_count_same", "no_inf", "unique:id", "null_ratio<=0.0:id"],
    )
    result = ConstraintEngine().evaluate(before, after, contract)
    assert result.passed is True
    assert result.checked_rules >= 4


def test_constraint_engine_value_constraints_fail() -> None:
    before = pd.DataFrame({"id": [1, 2], "amount": [1.0, 2.0], "flag": ["A", "B"]})
    after = pd.DataFrame({"id": [1, 1], "amount": [1.0, float("inf")], "flag": ["A", "X"]})
    contract = DataContract(
        value_constraints={
            "id": {"unique": True},
            "amount": {"no_inf": True},
            "flag": {"allow_values": ["A", "B"]},
        },
        post_conditions=["unique:id"],
    )
    result = ConstraintEngine().evaluate(before, after, contract)
    assert result.passed is False
    assert any("unique" in x for x in result.issues)
    assert any("allow_values" in x for x in result.issues)
