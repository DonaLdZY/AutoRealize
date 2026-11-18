from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.context_compiler import ArtifactStore
from autorealize.investigation import (
    CrossFileInvestigationTools,
    _available_qdi_actions,
    _apply_qdi_retrieval_action_policy,
    _qdi_retrieval_continuation_actions,
    _terminal_qdi_actions,
    _question_fixed_by_task_hint,
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


def test_reading_checks_skip_verified_empty_sheet() -> None:
    card = {
        "table_id": "浓度.xls::Sheet2",
        "table_kind": "excel_sheet",
        "layout_kind": "empty_or_unreadable",
        "read_strategy_kind": "header_none_inspect",
        "verified_row_count": 0,
        "shape": [0, 0],
        "reading_risks": ["Sheet could not be previewed"],
    }
    assert _reading_strategy_verification_questions({"table_cards": [card]}) == []


def test_qdi_does_not_reask_rules_explicitly_fixed_by_task_hint() -> None:
    from autorealize.models import InvestigationQuestion

    hint = (
        "重复采样时间按组内有限结果算术平均。"
        "最早60%训练、中间20%开发、最后20%留出，按一小时预测跨度 purge。"
        "不使用无法证明发布时间的化验结果作为在线历史特征。"
    )
    assert _question_fixed_by_task_hint(
        InvestigationQuestion(question="同一采样时间多条标签记录的聚合口径是什么？"),
        hint,
    )
    assert _question_fixed_by_task_hint(
        InvestigationQuestion(question="请确定 60/20/20 切分和 purge 规范"),
        hint,
    )
    assert _question_fixed_by_task_hint(
        InvestigationQuestion(question="化验结果能否作为在线历史特征？"),
        hint,
    )


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


def test_custom_readonly_python_child_does_not_inherit_credentials(tmp_path: Path, monkeypatch) -> None:
    import autorealize.investigation as investigation

    captured = {}
    real_popen = investigation.subprocess.Popen

    def capture_popen(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return real_popen(*args, **kwargs)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setenv("AUTOREALIZE_TEST_TOKEN", "test-token-secret")
    monkeypatch.setattr(investigation.subprocess, "Popen", capture_popen)

    result = run_custom_readonly_python(
        "def analyze(input_dir: str, scratch_dir: str) -> dict:\n    return {'ok': True}",
        input_dir=tmp_path,
        cfg=AutoRealizeConfig(),
    )

    assert result["ok"] is True
    child_env = {key.upper(): value for key, value in captured["env"].items()}
    assert "DEEPSEEK_API_KEY" not in child_env
    assert "OPENAI_API_KEY" not in child_env
    assert "AUTOREALIZE_TEST_TOKEN" not in child_env
    assert child_env["PYTHONDONTWRITEBYTECODE"] == "1"


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


def test_sandbox_preserves_full_sequences_and_artifact_keeps_tail_over_view_budget(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    cfg = AutoRealizeConfig()
    cfg.investigation.max_result_chars = 100
    tools = CrossFileInvestigationTools(
        cfg=cfg, data_root=tmp_path, authoritative_memory={}, knowledge_base={}, artifact_store=store,
    )
    result = tools.execute(InvestigationToolRequest(
        request_id="r-full-list", question_id="q1", tool_name="custom_readonly_python",
        custom_python=ReadonlyPythonRequest(question_id="q1", python_code="""
def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"values": list(range(250))}
"""),
    ))
    assert result.status == "completed", result.model_dump()
    ref = result.result["_full_result_artifact"]
    restored = store.read_excerpt(ref["artifact_id"], json_path="values", max_chars=10000)
    values = __import__("json").loads(restored["excerpt"])
    assert len(values) == 250
    assert values[-1] == 249
    assert result.output_truncated is True


def test_oversized_structured_result_fails_explicitly_instead_of_losing_tail(tmp_path: Path) -> None:
    cfg = AutoRealizeConfig()
    cfg.investigation.custom_python_max_result_bytes = 1024
    result = run_custom_readonly_python("""
def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"values": list(range(1000))}
""", input_dir=tmp_path, cfg=cfg)
    assert result["error"] == "custom_python_result_size_limit_exceeded"
    assert result["max_result_bytes"] == 1024


def test_custom_python_stdout_is_drained_without_unbounded_capture(tmp_path: Path) -> None:
    cfg = AutoRealizeConfig()
    cfg.investigation.custom_python_max_stdout_chars = 128
    result = run_custom_readonly_python("""
def analyze(input_dir: str, scratch_dir: str) -> dict:
    print("x" * 500000)
    return {"ok": True}
""", input_dir=tmp_path, cfg=cfg)

    assert result["ok"] is True
    assert result["_stdout_capture"]["original_bytes"] > 500000
    assert result["_stdout_capture"]["visible_chars"] == 128
    assert result["_stdout_capture"]["truncated"] is True


def test_script_prompt_preview_selects_whole_top_level_fields(tmp_path: Path) -> None:
    cfg = AutoRealizeConfig()
    cfg.investigation.max_result_chars = 180
    store = ArtifactStore(tmp_path / "artifacts")
    tools = CrossFileInvestigationTools(cfg=cfg, data_root=tmp_path,
                                        authoritative_memory={}, knowledge_base={}, artifact_store=store)
    request = InvestigationToolRequest(
        request_id="r-whole-fields", question_id="q1", tool_name="custom_readonly_python",
        custom_python=ReadonlyPythonRequest(question_id="q1", python_code="""
def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"target_count": 17, "long_note": "qualifier " * 200, "scope": "full file except shutdown"}
"""),
    )
    result = tools.execute(request)
    assert result.status == "completed"
    assert result.output_truncated is True
    assert "preview_json" not in json.dumps(result.result)
    assert result.result["_full_result_artifact"]["artifact_id"]
    visible_fields = result.result.get("visible_fields", {})
    assert visible_fields.get("target_count") == 17
    assert visible_fields.get("scope") == "full file except shutdown"
    assert "long_note" in result.result.get("omitted_fields", []) or result.result["omitted_field_count"] >= 1
    ref = result.result["_full_result_artifact"]
    restored = store.read_excerpt(ref["artifact_id"], json_path="long_note", max_chars=3000)
    assert json.loads(restored["excerpt"]) == "qualifier " * 200


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


def test_custom_python_static_validation_rejects_native_file_access_libraries() -> None:
    issues = validate_custom_readonly_python(
        """
import pyarrow as pa
import polars as pl
from h5py import File

def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"ok": True}
"""
    )

    assert "banned_import:pyarrow" in issues
    assert "banned_import:polars" in issues
    assert "banned_import:h5py" in issues


def test_pandas_native_reader_paths_are_checked_before_backend_open(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (tmp_path / "outside.parquet").write_bytes(b"not opened")
    code = """
import pandas as pd
from pathlib import Path

def analyze(input_dir: str, scratch_dir: str) -> dict:
    frame = pd.read_parquet(Path(input_dir).parent / "outside.parquet")
    return {"rows": len(frame)}
"""

    result = run_custom_readonly_python(code, input_dir=input_dir, cfg=AutoRealizeConfig())

    assert result["error"] == "custom_python_exit_code:1"
    assert "refused read outside input_dir/scratch_dir" in result["stderr_tail"]


def test_custom_python_reflection_cannot_recover_dynamic_imports(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    outside = tmp_path / "outside-canary.txt"
    outside.write_text("must not be read", encoding="utf-8")
    code = """
from pathlib import Path
import pandas as pd

def analyze(input_dir: str, scratch_dir: str) -> dict:
    importer = globals()["__builtins__"]["__import__"]
    os_module = importer("os")
    target = Path(input_dir).parent / "outside-canary.txt"
    descriptor = os_module.open(str(target), os_module.O_RDONLY)
    try:
        return {"outside_read": os_module.read(descriptor, 128).decode("utf-8")}
    finally:
        os_module.close(descriptor)
"""

    issues = validate_custom_readonly_python(code)
    result = run_custom_readonly_python(code, input_dir=input_dir, cfg=AutoRealizeConfig())

    assert "banned_call:globals" in issues
    assert result["error"] == "custom_python_static_validation_failed"
    assert "outside_read" not in result
    assert outside.read_text(encoding="utf-8") == "must not be read"

    direct_import_issues = validate_custom_readonly_python(
        """
import pandas as pd
def analyze(input_dir: str, scratch_dir: str) -> dict:
    importer = __builtins__["__import__"]
    return {"os": importer("os").name}
"""
    )
    assert "banned_python_reflection_name:__builtins__" in direct_import_issues


def test_custom_python_cannot_reach_unchecked_pandas_parquet_engine() -> None:
    code = """
import pandas as pd
def analyze(input_dir: str, scratch_dir: str) -> dict:
    engine = pd.io.parquet.get_engine("pyarrow")
    table = engine.api.parquet.read_table("outside.parquet")
    return {"rows": table.num_rows}
"""
    issues = validate_custom_readonly_python(code)

    assert "banned_pandas_internal_io_access:pd.io" in issues


def test_custom_python_cannot_import_pandas_io_helpers() -> None:
    code = """
from pandas.io.parquet import get_engine
def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"engine": get_engine("pyarrow").__class__.__name__}
"""
    issues = validate_custom_readonly_python(code)

    assert "banned_pandas_internal_io_import:pandas.io.parquet" in issues
    assert "banned_python_reflection_attribute:__class__" in issues


def test_custom_python_static_validation_allows_readonly_pandas_and_string_renames() -> None:
    issues = validate_custom_readonly_python(
        """
import pandas as pd
def analyze(input_dir: str, scratch_dir: str) -> dict:
    frame = pd.DataFrame({"value": [" 1 "]})
    frame.columns = frame.columns.str.replace(" ", "_")
    renamed = frame.rename(columns={"value_": "value"})
    values = renamed["value"].replace("/", None)
    return {"values": values.tolist()}
"""
    )
    assert issues == []


def test_custom_python_static_validation_still_rejects_path_mutation() -> None:
    issues = validate_custom_readonly_python(
        """
from pathlib import Path
def analyze(input_dir: str, scratch_dir: str) -> dict:
    source = Path(input_dir) / "data.csv"
    source.rename(source.with_suffix(".bak"))
    return {"ok": True}
"""
    )
    assert "banned_method:rename" in issues


def test_path_metadata_reads_are_restricted_to_input_and_scratch(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-qdi-metadata.txt"
    outside.write_text("private", encoding="utf-8")
    code = """
from pathlib import Path
def analyze(input_dir: str, scratch_dir: str) -> dict:
    outside = Path(input_dir).parent / "outside-qdi-metadata.txt"
    return {"exists": Path(input_dir).exists(), "outside": outside.exists()}
"""
    result = run_custom_readonly_python(code, input_dir=tmp_path, cfg=AutoRealizeConfig())
    assert result["error"] == "custom_python_exit_code:1"
    assert "refused read outside input_dir/scratch_dir" in result["stderr_tail"]


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
        max_context_retrievals=2,
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
    # A chunk read is opened only after a successful search exposes chunk ids.
    assert "read_document_chunks" not in actions


def test_retrieval_policy_requires_terminal_unless_continuation_is_proven() -> None:
    base = ["answer", "give_up", "request_context", "search_document", "read_qdi_artifact_excerpt"]
    assert _apply_qdi_retrieval_action_policy(
        base,
        retrieval_actions_since_terminal=1,
        max_retrieval_actions_before_terminal=1,
        continuation_actions=[],
    ) == ["answer", "give_up"]

    document_continuation = _qdi_retrieval_continuation_actions(
        [{
            "action": "search_document",
            "observation": {
                "content": {"matches": [{"chunk_id": "doc:0"}]},
            },
        }],
        document_retrievals_for_question=1,
        max_document_retrievals=2,
        artifact_retrievals_for_question=0,
        max_artifact_retrievals=2,
    )
    assert document_continuation == ["read_document_chunks"]
    assert _apply_qdi_retrieval_action_policy(
        base,
        retrieval_actions_since_terminal=1,
        max_retrieval_actions_before_terminal=1,
        continuation_actions=document_continuation,
    ) == ["answer", "give_up", "read_document_chunks"]

    artifact_continuation = _qdi_retrieval_continuation_actions(
        [{
            "action": "read_qdi_artifact_excerpt",
            "observation": {"status": "completed", "has_more": True},
        }],
        document_retrievals_for_question=0,
        max_document_retrievals=2,
        artifact_retrievals_for_question=1,
        max_artifact_retrievals=2,
    )
    assert artifact_continuation == ["read_qdi_artifact_excerpt"]

    exhausted_artifact = _qdi_retrieval_continuation_actions(
        [{
            "action": "read_qdi_artifact_excerpt",
            "observation": {"status": "completed", "has_more": False},
        }],
        document_retrievals_for_question=0,
        max_document_retrievals=2,
        artifact_retrievals_for_question=1,
        max_artifact_retrievals=2,
    )
    assert exhausted_artifact == []

    measured_gap_actions = _apply_qdi_retrieval_action_policy(
        ["answer", "give_up", "request_context", "request_checks", "request_script"],
        retrieval_actions_since_terminal=1,
        max_retrieval_actions_before_terminal=1,
        continuation_actions=[],
    )
    assert measured_gap_actions == ["answer", "give_up", "request_checks", "request_script"]


def test_last_qdi_round_allows_retrieval_before_terminal_answer() -> None:
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

    assert actions == ["answer", "search_document", "read_qdi_artifact_excerpt", "add_followup_questions", "give_up", "mark_duplicate"]
