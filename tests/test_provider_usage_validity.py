import json
from types import SimpleNamespace

import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.llm.client import LLMClient


@pytest.mark.parametrize("raw_usage", [
    {"prompt_tokens": 0, "completion_tokens": 10, "total_tokens": 10,
     "prompt_tokens_details": {"cached_tokens": 0}},
    {"prompt_tokens": 0, "completion_tokens": 10, "prompt_cache_hit_tokens": 50,
     "prompt_cache_miss_tokens": 10, "cache_write_tokens": 20},
    {"completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}},
])
def test_nonempty_requests_with_unknown_input_do_not_claim_known_usage(tmp_path, raw_usage):
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    client = LLMClient(cfg, tmp_path)
    try:
        row = client._log_provider_usage(
            prompt_name="question_investigator_action", mode="structured",
            response=SimpleNamespace(usage=raw_usage), seconds=1,
            prompt_parts=[{"name": "stable_context", "content": "All mandatory constraints. " * 200}],
        )
    finally:
        client.client.close()

    assert row["prompt_usage_available"] is False
    assert row["usage_available"] is False
    assert row["provider_cache_tokens_known"] is False
    assert row["provider_cache_write_tokens_known"] is False
    assert row["raw_usage"] == raw_usage
    assert row["estimated_prompt_tokens"] > 0
    assert row["prompt_parts"][0]["estimated_tokens"] > 0
    assert row["prompt_cache_hit_tokens"] == 0
    assert row["prompt_cache_miss_tokens"] == 0
    assert row["cache_write_tokens"] == 0
    assert row["completion_tokens"] == 10
    summary = json.loads(client.usage_summary_path.read_text(encoding="utf-8"))
    assert summary["provider_usage_missing_calls"] == 1
    assert summary["by_prompt"]["question_investigator_action"]["provider_usage_missing_calls"] == 1
    assert summary["provider_cache_known_prompt_tokens"] == 0
    assert summary["provider_cache_write_known_calls"] == 0
    assert summary["provider_cache_write_unknown_calls"] == 1
    assert summary["provider_cache_hit_ratio"] is None
    assert summary["estimated_prompt_tokens"] == row["estimated_prompt_tokens"]
    brief = json.loads(client.usage_brief_path.read_text(encoding="utf-8"))
    assert brief["provider_usage_missing_calls"] == 1
    assert brief["estimated_prompt_tokens"] == row["estimated_prompt_tokens"]


def test_unknown_input_does_not_make_mixed_summary_cache_coverage_complete(tmp_path):
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    client = LLMClient(cfg, tmp_path)
    try:
        for usage in [
            {"prompt_tokens": 0, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}},
            {"prompt_tokens": 120, "completion_tokens": 12, "prompt_cache_hit_tokens": 80,
             "prompt_cache_miss_tokens": 40},
        ]:
            row = client._log_provider_usage(
                prompt_name="mixed", mode="text", response=SimpleNamespace(usage=usage), seconds=1,
                prompt_parts=[{"name": "dynamic_payload", "content": "A nonempty request"}],
            )
    finally:
        client.client.close()
    assert row["prompt_usage_available"] is True
    assert row["provider_cache_tokens_known"] is True
    summary = json.loads(client.usage_summary_path.read_text(encoding="utf-8"))
    assert summary["prompt_tokens"] == 120
    assert summary["completion_tokens"] == 22
    assert summary["provider_cache_known_prompt_tokens"] == 120
    assert summary["known_provider_cache_hit_ratio"] == 0.666667
    assert summary["provider_usage_missing_calls"] == 1
    assert summary["provider_cache_hit_ratio"] is None
    assert summary["provider_cache_miss_ratio"] is None
    assert summary["provider_cache_telemetry_coverage"] is None
    assert summary["provider_cache_observed_hit_ratio_lower_bound"] is None
