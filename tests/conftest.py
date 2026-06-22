from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _test_body_uses_full_pipeline(item: pytest.Item) -> bool:
    """Detect old smoke tests that now run the real LLM-backed pipeline.

    AutoRealize no longer provides a fake/offline LLM path. These tests are
    still valuable, but they are integration tests and must be opted into
    explicitly so a normal unit-test run does not spend minutes calling an API.
    """
    obj = getattr(item, "obj", None)
    if obj is None:
        return False
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return False
    return "AutoRealizePipeline(" in source or "pipeline.run(" in source


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_real_llm = _truthy_env("AUTOREALIZE_REAL_LLM_TESTS") or _truthy_env(
        "AUTOREALIZE_REAL_LLM_FULL_PIPELINE"
    )
    run_full_pipeline = _truthy_env("AUTOREALIZE_REAL_LLM_FULL_PIPELINE")
    skip_real_llm = pytest.mark.skip(
        reason="set AUTOREALIZE_REAL_LLM_TESTS=1 to run real LLM integration tests"
    )
    skip_full_pipeline = pytest.mark.skip(
        reason="set AUTOREALIZE_REAL_LLM_FULL_PIPELINE=1 to run full real LLM pipeline tests"
    )

    for item in items:
        if _test_body_uses_full_pipeline(item):
            item.add_marker(pytest.mark.real_llm)
            item.add_marker(pytest.mark.full_pipeline)

        is_full = item.get_closest_marker("full_pipeline") is not None
        is_real = item.get_closest_marker("real_llm") is not None

        if is_full and not run_full_pipeline:
            item.add_marker(skip_full_pipeline)
        elif is_real and not run_real_llm:
            item.add_marker(skip_real_llm)
