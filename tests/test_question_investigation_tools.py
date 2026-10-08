from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.investigation import (
    CrossFileInvestigationTools,
    _available_qdi_actions,
    _terminal_qdi_actions,
    _reading_strategy_verification_questions,
    run_custom_readonly_python,
    validate_custom_readonly_python,
)
from autorealize.models import InvestigationToolRequest
from autorealize.models import ReadonlyPythonRequest
from autorealize.parsers.table_parser import TableParser
from autorealize.profiling.csv_utils import infer_csv_dialect, read_csv_auto


def test_reading_checks_skip_documents_and_verified_standard_sheet() -> None:
    from autorealize.context_compiler import file_to_table_cards
    from autorealize.models import FileRole, FileSummary

    sheet = FileSummary(
        path="links.xlsx", role=FileRole.raw_data_table, summary="",
        source_metadata={"excel_sheet_profiles": [{
            "sheet_name": "links", "columns": ["id"], "layout_kind": "standard_table",
            "header_confidence": 0.804, "read_strategy_kind": "default_header",
            "recommended_read": "pd.read_excel(path, sheet_name='links')",
            "raw_preview": [["id"], [1]], "reading_risks": [],
        }]},
    )
    cards = file_to_table_cards(sheet)
    cards.extend([
        {"table_id": "requirements.pdf", "warnings": ["CSV encoding and header need verification"]},
        {"table_id": "requirements.docx", "table_kind": "document", "warnings": ["Inspect sheet mapping"]},
    ])
    assert _reading_strategy_verification_questions({"table_cards": cards}) == []


@pytest.mark.parametrize("changes", [
    {"layout_kind": "headerless_table", "read_strategy_kind": "header_none_table"},
    {"layout_kind": "non_default_header", "read_strategy_kind": "explicit_header_row"},
    {"layout_kind": "sparse_or_irregular_sheet", "read_strategy_kind": "header_none_inspect"},
    {"layout_kind": "document_like_sheet", "read_strategy_kind": "header_none_document"},
    {"header_confidence": 0.4},
    {"header_confidence": "unknown"},
    {"warnings": ["Header mismatch after full read"]},
    {"reading_risks": ["Merged cells may conceal labels"]},
    {"table_id": "cooling.csv", "table_kind": "csv_table", "layout_kind": "", "reading_notes": ["CSV non-default separator hint: sep=';' and encoding='cp1252'"]},
])
def test_reading_checks_preserve_uncertain_or_abnormal_reader_contracts(changes) -> None:
    card = {
        "table_id": "links.xlsx::links", "table_kind": "excel_sheet",
        "layout_kind": "standard_table", "read_strategy_kind": "default_header",
        "header_confidence": 0.9, "reading_notes": ["Recommended read for this sheet: pd.read_excel(path)"],
    }
    card.update(changes)
    questions = _reading_strategy_verification_questions({"table_cards": [card]})
    assert len(questions) == 1
    assert questions[0].category == "data_access"


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


def test_csv_dialect_probe_rereads_semicolon_decimal_comma_and_observes_numeric_flags(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input_trucks.csv"
    path.write_text(
        "Id truck;Weight;Stack with multiple docks\n"
        "T1;386,750;0\n"
        "T2;401,250;1\n",
        encoding="utf-8",
    )

    hint = infer_csv_dialect(path)
    frame = read_csv_auto(path)
    parsed = TableParser(preview_rows=20).parse(path)
    contract = parsed.metadata["csv_read_contract"]

    assert hint.sep == ";"
    assert hint.decimal == ","
    assert "stable_semicolon_delimiter_with_decimal_comma" in hint.reason
    assert frame.shape == (2, 3)
    assert frame["Weight"].tolist() == pytest.approx([386.75, 401.25])
    assert parsed.metadata["shape"] == [2, 3]
    assert contract["pandas_kwargs"] == {
        "sep": ";",
        "encoding": "utf-8-sig",
        "decimal": ",",
    }
    assert contract["validated_columns_exact"] == [
        "Id truck",
        "Weight",
        "Stack with multiple docks",
    ]
    assert contract["boolean_like_columns"]["Stack with multiple docks"] == {
        "representation": "numeric_0_1",
        "observed_values": ["0", "1"],
    }


def test_preview_integers_do_not_become_full_table_dtype_constraints(tmp_path: Path) -> None:
    path = tmp_path / "values.csv"
    path.write_text("value\n0\n0\n0.125\n", encoding="utf-8")
    parsed = TableParser(preview_rows=2).parse(path)
    contract = parsed.metadata["csv_read_contract"]
    assert contract["observed_preview_dtypes"]["value"] == "int64"
    assert contract["dtype_validation_scope"] == "preview_only"
    assert "validated_dtypes" not in contract
    frame = pd.read_csv(path, **contract["pandas_kwargs"])
    assert frame["value"].iloc[-1] == pytest.approx(0.125)


def test_reactive_power_is_not_a_boolean_flag(tmp_path: Path) -> None:
    path = tmp_path / "values.csv"
    path.write_text("Leading_Current_Reactive_Power_kVarh,is_active\n0,0\n0,0\n0.125,1\n", encoding="utf-8")
    contract = TableParser(preview_rows=2).parse(path).metadata["csv_read_contract"]
    assert "Leading_Current_Reactive_Power_kVarh" not in contract["boolean_like_columns"]
    assert "is_active" in contract["boolean_like_columns"]


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


def test_custom_python_static_validation_allows_local_analytics_imports() -> None:
    issues = validate_custom_readonly_python(
        """
import polars as pl
import pyarrow as pa
import scipy.stats
from networkx import Graph
from rapidfuzz import fuzz
from sklearn.metrics import mean_squared_error
from statsmodels import api as sm

def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"ok": True}
"""
    )

    assert issues == []


def test_slash_missing_marker_does_not_request_script_repair(tmp_path: Path) -> None:
    code = '''
def analyze(input_dir: str, scratch_dir: str) -> dict:
    values = ["A", "/", "B", "/"]
    return {"missing": sum(value == "/" for value in values)}
'''
    assert validate_custom_readonly_python(code) == []
    assert run_custom_readonly_python(code, input_dir=tmp_path, cfg=AutoRealizeConfig())["missing"] == 2


def test_slash_literal_does_not_allow_root_directory_access(tmp_path: Path) -> None:
    code = '''
from pathlib import Path
def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"entries": [str(path) for path in Path("/").iterdir()]}
'''
    result = run_custom_readonly_python(code, input_dir=tmp_path, cfg=AutoRealizeConfig())
    assert result["error"] == "custom_python_exit_code:1"
    assert "refused read outside input_dir/scratch_dir" in result["stderr_tail"]


@pytest.mark.parametrize("literal", ["/etc/passwd", "C:/Windows/system.ini", "../secrets.txt"])
def test_full_absolute_and_traversal_literals_remain_rejected(literal) -> None:
    code = f"def analyze(input_dir: str, scratch_dir: str) -> dict:\n    return {{'path': {literal!r}}}"
    assert any("path_" in issue for issue in validate_custom_readonly_python(code))


def test_document_retrieval_actions_do_not_depend_on_script_budget() -> None:
    actions = _available_qdi_actions(
        question_records={"q1": {"question_id": "q1", "depth": 0}},
        current_record={"question_id": "q1", "depth": 0},
        scripts_for_question=3,
        context_retrievals_for_question=2,
        document_retrievals_for_question=0,
        has_documents=True,
        max_document_retrievals=4,
        total_scripts=10,
        max_scripts_per_question=3,
        max_scripts_total=10,
        max_total_questions=5,
        max_depth=3,
        max_followups_per_question=3,
        allow_custom_readonly_python=True,
    )

    assert "request_script" not in actions
    assert "search_document" in actions
    assert "read_document_chunks" in actions


def test_last_qdi_round_keeps_only_terminal_actions() -> None:
    actions = _terminal_qdi_actions(
        [
            "answer",
            "request_script",
            "search_document",
            "read_qdi_artifact_excerpt",
            "add_followup_questions",
            "give_up",
            "refine_current_question",
            "mark_duplicate",
        ]
    )

    assert actions == ["answer", "add_followup_questions", "give_up", "mark_duplicate"]
