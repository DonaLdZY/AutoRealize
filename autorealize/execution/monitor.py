from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ..models import MonitorVerdict


@dataclass
class MonitorThresholds:
    revision_warn: int = 2
    revision_critical: int = 4
    row_drop_warn: float = 0.30
    row_drop_critical: float = 0.90
    row_growth_warn: float = 5.0
    null_increase_warn_pp: float = 0.20


class RuleMonitor:
    """规则监控器（不调用 LLM）。"""

    def __init__(self, thresholds: MonitorThresholds | None = None) -> None:
        self.t = thresholds or MonitorThresholds()
        self.error_patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"KeyError", re.I), "列名不存在，可能 schema 理解错误"),
            (re.compile(r"MemoryError", re.I), "内存不足，建议先降采样"),
            (re.compile(r"UnicodeDecodeError", re.I), "编码问题，尝试 gb18030/utf-8-sig"),
            (re.compile(r"ParserError", re.I), "解析失败，检查分隔符或损坏文件"),
        ]

    def analyze_error(self, error_text: str) -> list[str]:
        hints: list[str] = []
        for pattern, hint in self.error_patterns:
            if pattern.search(error_text):
                hints.append(hint)
        return hints

    def evaluate(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        revision_count: int = 0,
    ) -> MonitorVerdict:
        alerts: list[str] = []
        severity = "none"
        ok = True

        if revision_count >= self.t.revision_warn:
            alerts.append(f"脚本重试次数较高: {revision_count}")
            severity = "warning"
        if revision_count >= self.t.revision_critical:
            alerts.append(f"脚本重试达到临界: {revision_count}")
            severity = "critical"
            ok = False

        b_rows = max(len(before_df), 1)
        a_rows = len(after_df)
        drop_ratio = max((b_rows - a_rows) / b_rows, 0.0)
        growth_ratio = max((a_rows - b_rows) / b_rows, 0.0)

        if drop_ratio >= self.t.row_drop_warn:
            alerts.append(f"行数下降明显: {drop_ratio:.2%}")
            severity = max(severity, "warning", key=_sev_rank)
        if drop_ratio >= self.t.row_drop_critical:
            alerts.append(f"行数下降过大: {drop_ratio:.2%}")
            severity = "critical"
            ok = False
        if growth_ratio >= self.t.row_growth_warn:
            alerts.append(f"行数增长过大: {growth_ratio:.2%}")
            severity = max(severity, "warning", key=_sev_rank)

        before_null = float(before_df.isna().mean().mean()) if not before_df.empty else 0.0
        after_null = float(after_df.isna().mean().mean()) if not after_df.empty else 0.0
        if after_null - before_null >= self.t.null_increase_warn_pp:
            alerts.append(
                f"空值率提升显著: before={before_null:.2%}, after={after_null:.2%}"
            )
            severity = max(severity, "warning", key=_sev_rank)

        return MonitorVerdict(ok=ok, severity=severity, alerts=alerts)


def _sev_rank(s: str) -> int:
    order = {"none": 0, "warning": 1, "critical": 2}
    return order.get(s, 0)
