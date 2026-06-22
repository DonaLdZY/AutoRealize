import pandas as pd

from autorealize.cognition import _execute_action_spec
from autorealize.config import AutoRealizeConfig


def test_action_specs_support_basic_null_and_inf_checks() -> None:
    df = pd.DataFrame(
        {
            "param": ["chloride", None, "ph"],
            "site": ["A", "B", None],
            "result [unit]": [1.0, float("inf"), None],
            "sample_time": ["2025-01-01", None, "2025-01-03"],
        }
    )
    cfg = AutoRealizeConfig.from_env()

    null_out = _execute_action_spec(
        df,
        {
            "action": "check_nulls",
            "reason": "check key columns before modeling",
            "columns": ["param", "site", "result [unit]", "sample_time"],
        },
        [],
        cfg,
    )

    assert "error" not in null_out
    null_counts = {row["column"]: row["null_count"] for row in null_out["nulls"]}
    assert null_counts == {
        "param": 1,
        "site": 1,
        "result [unit]": 1,
        "sample_time": 1,
    }

    inf_out = _execute_action_spec(
        df,
        {
            "action": "check_inf",
            "reason": "check invalid numeric sentinels",
            "columns": ["result [unit]"],
        },
        [],
        cfg,
    )

    assert "error" not in inf_out
    assert inf_out["inf_values"] == [
        {"column": "result [unit]", "inf_count": 1, "inf_ratio": 0.333333}
    ]


def test_action_specs_support_profile_actions() -> None:
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "category": ["A", "B", "A"],
        }
    )
    cfg = AutoRealizeConfig.from_env()

    numeric_out = _execute_action_spec(
        df,
        {"action": "profile_numeric", "columns": ["value"]},
        [],
        cfg,
    )
    categorical_out = _execute_action_spec(
        df,
        {"action": "profile_categorical", "columns": ["category"]},
        [],
        cfg,
    )

    assert "error" not in numeric_out
    assert "error" not in categorical_out
    assert [row["name"] for row in numeric_out["profiles"]] == ["value"]
    assert [row["name"] for row in categorical_out["profiles"]] == ["category"]
