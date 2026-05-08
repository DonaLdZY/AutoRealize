import pandas as pd

from autorealize.execution.monitor import RuleMonitor


def test_monitor_detects_row_drop() -> None:
    m = RuleMonitor()
    before = pd.DataFrame({"x": list(range(100))})
    after = before.head(2).copy()
    verdict = m.evaluate(before, after, revision_count=0)
    assert verdict.ok is False
    assert verdict.severity == "critical"
