from __future__ import annotations

import json
from types import SimpleNamespace

from autorealize.config import AutoRealizeConfig
from autorealize.llm.client import (
    LLMClient,
    _prompt_part_stats,
    _usage_cache_tokens,
    _usage_to_dict,
)


def test_usage_cache_tokens_supports_deepseek_fields() -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 8,
        "total_tokens": 108,
        "prompt_cache_hit_tokens": 70,
        "prompt_cache_miss_tokens": 30,
    }

    assert _usage_cache_tokens(usage) == (70, 30)


def test_usage_cache_tokens_supports_openai_prompt_details() -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 8,
        "total_tokens": 108,
        "prompt_tokens_details": {"cached_tokens": 64},
    }

    assert _usage_cache_tokens(usage) == (64, 36)


def test_usage_to_dict_supports_model_dump_objects() -> None:
    class UsageObject:
        def model_dump(self) -> dict:
            return {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            }

    assert _usage_to_dict(UsageObject()) == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }


def test_prompt_part_stats_supports_synthetic_lengths() -> None:
    rows, total = _prompt_part_stats(
        [
            {"name": "text", "role": "user", "content": "abcd"},
            {"name": "image_payload_base64", "role": "user", "chars": 400, "estimated_tokens": 100},
        ],
        provider_prompt_tokens=200,
    )

    assert total >= 101
    assert rows[1]["name"] == "image_payload_base64"
    assert rows[1]["chars"] == 400
    assert rows[1]["estimated_tokens"] == 100
    assert rows[1]["provider_prompt_tokens_estimate"] > rows[0]["provider_prompt_tokens_estimate"]


def test_log_provider_usage_writes_jsonl_summary_and_prompt_parts(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    client = LLMClient(cfg, tmp_path)
    response = SimpleNamespace(
        usage={
            "prompt_tokens": 120,
            "completion_tokens": 12,
            "total_tokens": 132,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 40,
        }
    )

    row = client._log_provider_usage(
        prompt_name="unit_prompt",
        mode="structured",
        response=response,
        seconds=1.25,
        attempt=2,
        finish_reason="stop",
        max_tokens=8000,
        parsed_ok=True,
        prompt_parts=[
            {"name": "system_prompt", "role": "system", "content": "You are a tester."},
            {"name": "stable_context", "role": "user", "content": "上下文" * 20},
            {"name": "dynamic_payload", "role": "user", "content": "payload"},
        ],
    )

    assert row["prompt_tokens"] == 120
    assert row["completion_tokens"] == 12
    assert row["prompt_cache_hit_tokens"] == 80
    assert row["prompt_cache_miss_tokens"] == 40
    assert row["estimated_prompt_tokens"] > 0
    assert [p["name"] for p in row["prompt_parts"]] == ["system_prompt", "stable_context", "dynamic_payload"]
    usage_rows = [json.loads(line) for line in (tmp_path / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert usage_rows[-1]["prompt_name"] == "unit_prompt"
    assert usage_rows[-1]["provider_cache_tokens_known"] is True
    assert usage_rows[-1]["prompt_parts"][1]["name"] == "stable_context"
    assert usage_rows[-1]["prompt_parts"][1]["provider_prompt_tokens_estimate"] > 0
    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["calls"] == 1
    assert summary["prompt_tokens"] == 120
    assert summary["completion_tokens"] == 12
    assert summary["prompt_cache_hit_tokens"] == 80
    assert summary["prompt_cache_miss_tokens"] == 40
    assert summary["provider_cache_hit_ratio"] == 0.666667
    assert summary["by_prompt"]["unit_prompt"]["calls"] == 1
    assert summary["by_prompt"]["unit_prompt"]["by_part_ranked"][0]["estimated_tokens"] >= 1
    assert summary["by_prompt_part_ranked"][0]["prompt_name"] == "unit_prompt"


def test_log_provider_usage_marks_unknown_cache_fields(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    client = LLMClient(cfg, tmp_path)
    response = SimpleNamespace(
        usage={
            "prompt_tokens": 25,
            "completion_tokens": 5,
            "total_tokens": 30,
        }
    )

    row = client._log_provider_usage(
        prompt_name="unknown_cache_prompt",
        mode="text",
        response=response,
        seconds=0.5,
        parsed_ok=True,
    )

    assert row["provider_cache_tokens_known"] is False
    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["provider_cache_unknown_prompt_tokens"] == 25
    assert summary["provider_cache_known_prompt_tokens"] == 0


def test_log_local_cache_usage_is_separate_from_provider_calls(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    client = LLMClient(cfg, tmp_path)

    client._log_local_cache_usage(
        prompt_name="cached_prompt",
        mode="structured",
        prompt_parts=[{"name": "system_prompt", "role": "system", "content": "cached"}],
    )

    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["calls"] == 0
    assert summary["cache_hits_local"] == 1
    assert summary["by_prompt"]["cached_prompt"]["cache_hits_local"] == 1
    assert summary["by_prompt"]["cached_prompt"]["estimated_prompt_tokens"] > 0
