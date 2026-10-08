import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.investigation import (
    run_custom_readonly_python,
    validate_custom_readonly_python,
)


@pytest.mark.parametrize("arguments", [
    "", "deep=True", "deep=False", "order='C'", "order='F'", "order='A'", "order='K'",
])
def test_static_validation_allows_only_memory_copy_arguments(arguments):
    code = f"def analyze(input_dir, scratch_dir):\n    return value.copy({arguments})"
    assert validate_custom_readonly_python(code) == []


@pytest.mark.parametrize("arguments", [
    "destination", "True", "'C'", "target=destination", "dest=destination",
    "deep=destination", "order=destination", "deep='output.csv'", "order='output.csv'",
    "deep=True, target=destination", "*arguments", "**options", "**{'target': destination}",
])
def test_static_validation_rejects_copy_destinations_and_ambiguous_arguments(arguments):
    code = f"def analyze(input_dir, scratch_dir):\n    return value.copy({arguments})"
    assert "banned_method:copy" in validate_custom_readonly_python(code)


@pytest.mark.parametrize("arguments", ["", "deep=True", "deep=False"])
def test_sandbox_can_copy_filtered_dataframe_without_mutating_input(tmp_path, arguments):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    data = input_dir / "data.csv"
    original = "order,value\n1,2\n,3\n1,4\n"
    data.write_text(original, encoding="utf-8")
    code = f'''
from pathlib import Path
import pandas as pd

def analyze(input_dir: str, scratch_dir: str) -> dict:
    df = pd.read_csv(Path(input_dir) / "data.csv")
    valid = df["order"].notna()
    base = df.loc[valid].copy({arguments})
    base["_oid"] = df.loc[valid, "order"].values
    base["value"] = 10
    return {{"rows": len(base), "sum": int(base["value"].sum()),
            "source_sum": int(df["value"].sum()), "source_has_oid": "_oid" in df.columns}}
'''
    assert validate_custom_readonly_python(code) == []
    result = run_custom_readonly_python(code, input_dir=input_dir, cfg=AutoRealizeConfig())
    assert "error" not in result, result
    assert result["rows"] == 2
    assert result["sum"] == 20
    assert result["source_sum"] == 9
    assert result["source_has_oid"] is False
    assert data.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("order", ["C", "F", "A", "K"])
def test_sandbox_can_copy_numpy_array_with_order(tmp_path, order):
    code = f'''
import numpy as np

def analyze(input_dir: str, scratch_dir: str) -> dict:
    original = np.array([[1, 2], [3, 4]])
    copied = original.copy(order={order!r})
    copied[0, 0] = 100
    return {{"source": int(original[0, 0]), "copied": int(copied[0, 0]),
            "shares_memory": bool(np.shares_memory(original, copied))}}
'''
    result = run_custom_readonly_python(code, input_dir=tmp_path, cfg=AutoRealizeConfig())
    assert "error" not in result, result
    assert result["source"] == 1
    assert result["copied"] == 100
    assert result["shares_memory"] is False


@pytest.mark.parametrize("arguments", [
    "destination", "target=destination", "dest=destination", "**{'target': destination}",
])
def test_sandbox_rejects_copy_outside_input_and_scratch(tmp_path, arguments):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "data.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    destination = tmp_path / "escaped.csv"
    code = f'''
from pathlib import Path

def analyze(input_dir: str, scratch_dir: str) -> dict:
    source = Path(input_dir) / "data.csv"
    destination = Path(input_dir).parent / "escaped.csv"
    source.copy({arguments})
    return {{"copied": True}}
'''
    result = run_custom_readonly_python(code, input_dir=input_dir, cfg=AutoRealizeConfig())
    assert result["error"] == "custom_python_static_validation_failed"
    assert "banned_method:copy" in result["issues"]
    assert not destination.exists()
    assert source.read_text(encoding="utf-8") == "value\n1\n"


def test_file_copy_helpers_and_imports_remain_rejected():
    code = '''
import shutil

def analyze(input_dir: str, scratch_dir: str) -> dict:
    shutil.copyfile(input_dir, scratch_dir)
    return {}
'''
    issues = validate_custom_readonly_python(code)
    assert "banned_import:shutil" in issues
    assert "banned_method:copyfile" in issues
