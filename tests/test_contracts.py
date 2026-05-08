import pandas as pd

from autorealize.execution.contracts import check_contract
from autorealize.models import DataContract


def test_contract_check_basic() -> None:
    before = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    after = pd.DataFrame({"a": [1, 2], "c": [5, 6]})
    c = DataContract(
        required_input_columns=["a"],
        preserve_columns=["a"],
        remove_columns=["b"],
        add_columns=["c"],
        row_count_rule="same",
    )
    result = check_contract(before, after, c)
    assert result.passed
