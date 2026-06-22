from __future__ import annotations

from pathlib import Path

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.investigation import (
    CrossFileInvestigationTools,
    run_custom_readonly_python,
    validate_custom_readonly_python,
)
from autorealize.models import InvestigationToolRequest
from autorealize.models import ReadonlyPythonRequest
from autorealize.profiling.csv_utils import infer_csv_dialect


def _tools(input_dir: Path) -> CrossFileInvestigationTools:
    cfg = AutoRealizeConfig.from_env()
    cfg.investigation.tool_sample_rows = None
    return CrossFileInvestigationTools(
        cfg=cfg,
        data_root=input_dir,
        authoritative_memory={},
        knowledge_base={},
    )


def test_legacy_builtin_tool_names_are_not_executable_by_investigator(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame({"order_id": [1, 2, 3], "carrier": ["A", "B", "C"]}).to_csv(input_dir / "orders.csv", index=False)
    pd.DataFrame({"carrier": ["A", "B"], "rate": [10, 20]}).to_csv(input_dir / "cost.csv", index=False)
    tools = _tools(input_dir)

    req = InvestigationToolRequest(
        request_id="r1",
        question_id="q1",
        tool_name="join_coverage",
        params={
            "left_file": "orders.csv",
            "right_file": "cost.csv",
            "left_keys": ["carrier"],
            "right_keys": ["carrier"],
        },
    )
    result = tools.execute(req)

    assert result.status == "failed"
    assert result.error == "unsupported_tool:join_coverage"


def test_csv_dialect_probe_detects_whitespace_table_with_comma_lists(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "trainset.csv").write_text(
        "id features label\n"
        "1 1,2,3 0\n"
        "2 4,5,6 1\n",
        encoding="utf-8",
    )
    hint = infer_csv_dialect(input_dir / "trainset.csv")

    assert hint.sep == r"\s+"
    assert hint.engine == "python"
    assert hint.inferred is True
    assert hint.reason == "whitespace_columns_with_comma_lists"


def test_custom_readonly_python_can_write_scratch_but_not_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame({"x": [1, 2]}).to_csv(input_dir / "data.csv", index=False)
    cfg = AutoRealizeConfig.from_env()

    ok_code = """
from pathlib import Path
import pandas as pd

def analyze(input_dir: str, scratch_dir: str) -> dict:
    df = pd.read_csv(Path(input_dir) / "data.csv")
    out = Path(scratch_dir) / "preview.csv"
    df.head(1).to_csv(out, index=False)
    return {"rows": int(len(df)), "scratch_file_exists": out.exists()}
"""
    ok = run_custom_readonly_python(ok_code, input_dir=input_dir, cfg=cfg)

    assert ok["rows"] == 2
    assert ok["scratch_file_exists"] is True
    assert ok["_scratch_destroyed_after_execution"] is True

    bad_code = """
from pathlib import Path

def analyze(input_dir: str, scratch_dir: str) -> dict:
    Path(input_dir, "evil.txt").write_text("nope", encoding="utf-8")
    return {"ok": True}
"""
    bad = run_custom_readonly_python(bad_code, input_dir=input_dir, cfg=cfg)

    assert "error" in bad
    assert not (input_dir / "evil.txt").exists()


def test_investigator_execute_accepts_only_custom_readonly_python(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(input_dir / "data.csv", index=False)
    tools = _tools(input_dir)

    req = InvestigationToolRequest(
        request_id="r1",
        question_id="q1",
        tool_name="custom_readonly_python",
        custom_python=ReadonlyPythonRequest(
            question_id="q1",
            python_code="""
from pathlib import Path
import pandas as pd

def analyze(input_dir: str, scratch_dir: str) -> dict:
    df = pd.read_csv(Path(input_dir) / "data.csv")
    return {"rows": int(len(df)), "sum_x": int(df["x"].sum())}
""",
        ),
    )
    result = tools.execute(req)

    assert result.status == "completed"
    assert result.result["rows"] == 3
    assert result.result["sum_x"] == 6


def test_custom_python_static_validation_rejects_network_and_process_imports() -> None:
    issues = validate_custom_readonly_python(
        """
import os
import requests

def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"cwd": os.getcwd()}
"""
    )

    assert "banned_import:os" in issues
    assert "banned_import:requests" in issues
