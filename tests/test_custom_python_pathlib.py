import sys

import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.investigation import run_custom_readonly_python


@pytest.mark.parametrize("method,pattern", [("glob", "**/*.csv"), ("rglob", "*.csv")])
@pytest.mark.parametrize("options", [
    "",
    ", case_sensitive=False, recurse_symlinks=False",
    ", case_sensitive=True, recurse_symlinks=True",
])
def test_sandbox_globs_preserve_pathlib_options(tmp_path, method, pattern, options):
    if options and sys.version_info < (3, 13):
        pytest.skip("Explicit recurse_symlinks requires Python 3.13 or later")
    input_dir = tmp_path / "input"
    (input_dir / "nested").mkdir(parents=True)
    for name in ("data.csv", "upper.CSV", "nested/deep.csv"):
        (input_dir / name).write_text("id\n1\n", encoding="utf-8")
    code = f'''
from pathlib import Path

def analyze(input_dir: str, scratch_dir: str) -> dict:
    root = Path(input_dir)
    return {{"files": sorted(path.relative_to(root).as_posix() for path in root.{method}({pattern!r}{options}))}}
'''
    result = run_custom_readonly_python(code, input_dir=input_dir, cfg=AutoRealizeConfig())
    expected = ["data.csv", "nested/deep.csv"]
    if "case_sensitive=False" in options or (not options and sys.platform == "win32"):
        expected.append("upper.CSV")
    assert "error" not in result, result
    assert result["files"] == sorted(expected)


@pytest.mark.parametrize("method", ["glob", "rglob"])
def test_sandbox_globs_reject_roots_outside_input(tmp_path, method):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_text("private\n1\n", encoding="utf-8")
    code = f'''
from pathlib import Path

def analyze(input_dir: str, scratch_dir: str) -> dict:
    root = Path(input_dir).parent / "outside"
    return {{"files": [str(path) for path in root.{method}("*.csv")]}}
'''
    result = run_custom_readonly_python(code, input_dir=input_dir, cfg=AutoRealizeConfig())
    assert result["error"] == "custom_python_exit_code:1"
    assert "refused read outside input_dir/scratch_dir" in result["stderr_tail"]


@pytest.mark.parametrize("method", ["glob", "rglob"])
@pytest.mark.skipif(sys.version_info < (3, 13), reason="recurse_symlinks requires Python 3.13 or later")
def test_sandbox_globs_reject_recursive_symlinks_outside_input(tmp_path, method):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_text("private\n1\n", encoding="utf-8")
    try:
        (input_dir / "linked").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    pattern = "**/*.csv" if method == "glob" else "*.csv"
    code = f'''
from pathlib import Path

def analyze(input_dir: str, scratch_dir: str) -> dict:
    root = Path(input_dir)
    return {{"files": [str(path) for path in root.{method}({pattern!r}, recurse_symlinks=True)]}}
'''
    result = run_custom_readonly_python(code, input_dir=input_dir, cfg=AutoRealizeConfig())
    assert result["error"] == "custom_python_exit_code:1"
    assert "refused read outside input_dir/scratch_dir" in result["stderr_tail"]
