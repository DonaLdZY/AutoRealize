from __future__ import annotations

import io
import time
import traceback
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SandboxResult:
    success: bool
    output_df: pd.DataFrame | None
    stdout: str
    error: str
    exec_seconds: float


def run_code(code: str, df: pd.DataFrame) -> SandboxResult:
    """在受限命名空间执行代码。"""
    local_ns: dict[str, Any] = {}
    global_ns: dict[str, Any] = {"pd": pd}
    in_df = df.copy()
    stdout_buffer = io.StringIO()
    start = time.time()
    old_stdout = None
    try:
        import sys

        old_stdout = sys.stdout
        sys.stdout = stdout_buffer
        exec(code, global_ns, local_ns)
        fn = None
        for name, obj in local_ns.items():
            if callable(obj) and name.startswith("stage_"):
                fn = obj
                break
        if fn is not None:
            out = fn(in_df)
        else:
            out = local_ns.get("output_df", None)
        if out is None:
            return SandboxResult(
                success=False,
                output_df=None,
                stdout=stdout_buffer.getvalue(),
                error="脚本未产生 output_df 且无 stage_* 函数。",
                exec_seconds=time.time() - start,
            )
        if not isinstance(out, pd.DataFrame):
            return SandboxResult(
                success=False,
                output_df=None,
                stdout=stdout_buffer.getvalue(),
                error="输出不是 DataFrame。",
                exec_seconds=time.time() - start,
            )
        return SandboxResult(
            success=True,
            output_df=out,
            stdout=stdout_buffer.getvalue(),
            error="",
            exec_seconds=time.time() - start,
        )
    except Exception:  # noqa: BLE001
        return SandboxResult(
            success=False,
            output_df=None,
            stdout=stdout_buffer.getvalue(),
            error=traceback.format_exc(),
            exec_seconds=time.time() - start,
        )
    finally:
        if old_stdout is not None:
            import sys

            sys.stdout = old_stdout
