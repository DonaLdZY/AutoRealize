from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..models import DataContract


@dataclass
class ConstraintCheckResult:
    passed: bool
    issues: list[str]
    checked_rules: int


class ConstraintEngine:
    """可执行约束引擎：执行 value_constraints 与 post_conditions 的硬规则校验。"""

    def __init__(self, fail_on_unknown_rule: bool = False) -> None:
        self.fail_on_unknown_rule = fail_on_unknown_rule

    def evaluate(self, before_df: pd.DataFrame, after_df: pd.DataFrame, contract: DataContract) -> ConstraintCheckResult:
        issues: list[str] = []
        checked = 0

        for col, spec in (contract.value_constraints or {}).items():
            if not isinstance(spec, dict):
                issues.append(f"value_constraints[{col}] 不是对象")
                continue
            if col not in after_df.columns:
                issues.append(f"value_constraints 列不存在: {col}")
                continue
            series = after_df[col]
            checked += self._check_column_constraints(before_df, after_df, col, series, spec, issues)

        for cond in (contract.post_conditions or []):
            checked += 1
            ok, msg = self._check_post_condition(before_df, after_df, str(cond))
            if not ok:
                issues.append(msg)

        return ConstraintCheckResult(passed=not issues, issues=issues, checked_rules=checked)

    def _check_column_constraints(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        col: str,
        series: pd.Series,
        spec: dict[str, Any],
        issues: list[str],
    ) -> int:
        checked = 0

        if spec.get("not_null") is True:
            checked += 1
            null_count = int(series.isna().sum())
            if null_count > 0:
                issues.append(f"{col} not_null 失败: null_count={null_count}")

        if "max_null_ratio" in spec:
            checked += 1
            limit = float(spec.get("max_null_ratio"))
            null_ratio = float(series.isna().mean()) if len(series) > 0 else 0.0
            if null_ratio > limit:
                issues.append(f"{col} max_null_ratio 失败: ratio={null_ratio:.4f} > {limit}")

        if spec.get("unique") is True:
            checked += 1
            dup = int(series.duplicated().sum())
            if dup > 0:
                issues.append(f"{col} unique 失败: duplicated={dup}")

        if "dtype_in" in spec:
            checked += 1
            allow = [str(x) for x in spec.get("dtype_in", [])]
            dtype = str(series.dtype)
            if allow and dtype not in allow:
                issues.append(f"{col} dtype_in 失败: dtype={dtype}, allow={allow}")

        if spec.get("non_negative") is True:
            checked += 1
            num = pd.to_numeric(series, errors="coerce")
            bad = int((num < 0).fillna(False).sum())
            if bad > 0:
                issues.append(f"{col} non_negative 失败: negative_count={bad}")

        if "min" in spec:
            checked += 1
            floor = float(spec.get("min"))
            num = pd.to_numeric(series, errors="coerce")
            bad = int((num < floor).fillna(False).sum())
            if bad > 0:
                issues.append(f"{col} min 失败: count={bad}, min={floor}")

        if "max" in spec:
            checked += 1
            ceil = float(spec.get("max"))
            num = pd.to_numeric(series, errors="coerce")
            bad = int((num > ceil).fillna(False).sum())
            if bad > 0:
                issues.append(f"{col} max 失败: count={bad}, max={ceil}")

        if "allow_values" in spec:
            checked += 1
            allow_set = {str(x) for x in spec.get("allow_values", [])}
            if allow_set:
                s = series.dropna().astype(str)
                bad = s[~s.isin(list(allow_set))]
                if not bad.empty:
                    issues.append(f"{col} allow_values 失败: bad_examples={bad.head(3).tolist()}")

        if "forbid_values" in spec:
            checked += 1
            forbid = {str(x) for x in spec.get("forbid_values", [])}
            if forbid:
                s = series.dropna().astype(str)
                bad = s[s.isin(list(forbid))]
                if not bad.empty:
                    issues.append(f"{col} forbid_values 失败: bad_examples={bad.head(3).tolist()}")

        if "regex" in spec:
            checked += 1
            pattern = str(spec.get("regex"))
            try:
                rgx = re.compile(pattern)
                s = series.dropna().astype(str)
                bad = s[~s.map(lambda x: bool(rgx.fullmatch(x)))]
                if not bad.empty:
                    issues.append(f"{col} regex 失败: pattern={pattern}, bad_examples={bad.head(3).tolist()}")
            except re.error as exc:
                issues.append(f"{col} regex 非法: {exc}")

        if spec.get("no_inf") is True:
            checked += 1
            inf_count = _count_inf(series)
            if inf_count > 0:
                issues.append(f"{col} no_inf 失败: inf_like_count={inf_count}")

        if spec.get("unchanged_from_input") is True:
            checked += 1
            if col not in before_df.columns:
                issues.append(f"{col} unchanged_from_input 失败: before_df 无该列")
            else:
                b = before_df[col].reset_index(drop=True)
                a = after_df[col].reset_index(drop=True)
                if len(a) != len(b):
                    issues.append(f"{col} unchanged_from_input 失败: 行数变化")
                else:
                    ne = int((a.astype(str) != b.astype(str)).sum())
                    if ne > 0:
                        issues.append(f"{col} unchanged_from_input 失败: changed_count={ne}")

        return checked

    def _check_post_condition(self, before_df: pd.DataFrame, after_df: pd.DataFrame, cond: str) -> tuple[bool, str]:
        rule = cond.strip()
        if not rule:
            return True, ""

        if rule == "no_inf":
            for c in after_df.columns:
                if _count_inf(after_df[c]) > 0:
                    return False, f"post_condition no_inf 失败: 列 {c} 存在 inf-like 值"
            return True, ""

        if rule == "row_count_same":
            return (len(before_df) == len(after_df), f"post_condition row_count_same 失败: before={len(before_df)} after={len(after_df)}")

        if rule.startswith("row_count_change_ratio<="):
            try:
                threshold = float(rule.split("<=", 1)[1])
            except Exception:
                return False, f"post_condition 解析失败: {rule}"
            base = max(len(before_df), 1)
            ratio = abs(len(after_df) - len(before_df)) / base
            return (ratio <= threshold, f"post_condition row_count_change_ratio<= 失败: ratio={ratio:.4f} > {threshold}")

        if rule.startswith("unique:"):
            col = rule.split(":", 1)[1].strip()
            if col not in after_df.columns:
                return False, f"post_condition unique 失败: 列不存在 {col}"
            dup = int(after_df[col].duplicated().sum())
            return (dup == 0, f"post_condition unique 失败: {col} duplicated={dup}")

        if rule.startswith("null_ratio<=") and ":" in rule:
            left, col = rule.split(":", 1)
            try:
                threshold = float(left.split("<=", 1)[1])
            except Exception:
                return False, f"post_condition 解析失败: {rule}"
            col = col.strip()
            if col not in after_df.columns:
                return False, f"post_condition null_ratio 失败: 列不存在 {col}"
            ratio = float(after_df[col].isna().mean()) if len(after_df) > 0 else 0.0
            return (ratio <= threshold, f"post_condition null_ratio<= 失败: {col} ratio={ratio:.4f} > {threshold}")

        if rule.startswith("range:"):
            parts = rule.split(":")
            if len(parts) != 4:
                return False, f"post_condition range 解析失败: {rule}"
            _, col, min_v, max_v = parts
            col = col.strip()
            if col not in after_df.columns:
                return False, f"post_condition range 失败: 列不存在 {col}"
            try:
                min_f = float(min_v)
                max_f = float(max_v)
            except Exception:
                return False, f"post_condition range 数值解析失败: {rule}"
            num = pd.to_numeric(after_df[col], errors="coerce")
            bad = int(((num < min_f) | (num > max_f)).fillna(False).sum())
            return (bad == 0, f"post_condition range 失败: {col} out_of_range_count={bad}")

        if rule.startswith("regex:"):
            parts = rule.split(":", 2)
            if len(parts) != 3:
                return False, f"post_condition regex 解析失败: {rule}"
            _, col, pattern = parts
            col = col.strip()
            if col not in after_df.columns:
                return False, f"post_condition regex 失败: 列不存在 {col}"
            try:
                rgx = re.compile(pattern)
            except re.error as exc:
                return False, f"post_condition regex 非法: {exc}"
            s = after_df[col].dropna().astype(str)
            bad = s[~s.map(lambda x: bool(rgx.fullmatch(x)))]
            return (bad.empty, f"post_condition regex 失败: {col} bad_examples={bad.head(3).tolist()}")

        if rule.startswith("no_nan_in:"):
            cols = [c.strip() for c in rule.split(":", 1)[1].split(",") if c.strip()]
            for col in cols:
                if col not in after_df.columns:
                    return False, f"post_condition no_nan_in 失败: 列不存在 {col}"
                null_cnt = int(after_df[col].isna().sum())
                if null_cnt > 0:
                    return False, f"post_condition no_nan_in 失败: {col} null_count={null_cnt}"
            return True, ""

        # 自然语言兼容哨兵（来自 fallback contract 常见表达）
        if "输出必须为可读取" in rule or "输出必须为可读取 DataFrame" in rule:
            return True, ""

        if rule in {"任意", "any", "none", "无"}:
            return True, ""

        if self.fail_on_unknown_rule:
            return False, f"post_condition 不支持的规则: {rule}"
        return True, ""


def _count_inf(series: pd.Series) -> int:
    num = pd.to_numeric(series, errors="coerce")
    cnt_num = int(num.map(lambda x: isinstance(x, (int, float)) and math.isinf(float(x)) if pd.notna(x) else False).sum())
    text = series.dropna().astype(str).str.strip().str.lower()
    cnt_text = int(text.isin(["inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"]).sum())
    return cnt_num + cnt_text
