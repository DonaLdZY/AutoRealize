from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..models import DataContract


@dataclass
class ContractCheckResult:
    passed: bool
    issues: list[str]


def check_contract(before_df: pd.DataFrame, after_df: pd.DataFrame, contract: DataContract) -> ContractCheckResult:
    issues: list[str] = []
    after_cols = {str(c) for c in after_df.columns.tolist()}
    before_cols = {str(c) for c in before_df.columns.tolist()}

    for c in contract.required_input_columns:
        if c not in before_cols:
            issues.append(f"缺少输入列: {c}")
    for c in contract.preserve_columns:
        if c not in after_cols:
            issues.append(f"保留列缺失: {c}")
    for c in contract.remove_columns:
        if c in after_cols:
            issues.append(f"应删除列仍存在: {c}")
    for c in contract.add_columns:
        if c not in after_cols:
            issues.append(f"应新增列缺失: {c}")

    row_rule = contract.row_count_rule
    if row_rule == "same" and len(before_df) != len(after_df):
        issues.append("行数规则 same 违反")
    if row_rule == "less" and len(after_df) > len(before_df):
        issues.append("行数规则 less 违反")
    if row_rule == "greater" and len(after_df) < len(before_df):
        issues.append("行数规则 greater 违反")

    return ContractCheckResult(passed=not issues, issues=issues)
