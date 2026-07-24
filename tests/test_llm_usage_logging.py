from __future__ import annotations

import json
from types import SimpleNamespace

from pydantic import BaseModel, Field

from autorealize.config import AutoRealizeConfig
from autorealize.llm.client import (
    AGENT_INSTRUCTIONS_TITLE,
    CACHE_FRIENDLY_SYSTEM_PROMPT,
    LLMClient,
    _apply_deepseek_request_options,
    _degraded_request_kwargs,
    _deepseek_cost_breakdown,
    _deepseek_pricing_usd_per_1m,
    _normalize_base_url,
    _provider_prompt_cache_key,
    _prompt_stage,
    _prompt_part_stats,
    _schema_text_for_prompt,
    _usage_cache_tokens,
    _usage_cache_write_tokens,
    _usage_to_dict,
)
from autorealize.prompt_cache import STABLE_CONTEXT_TITLE, stable_dynamic_prompt
from autorealize.prompt_cache import json_block


class _StructuredChild(BaseModel):
    code: str = Field(description="Exact source code.")


class _StructuredPayload(BaseModel):
    child: _StructuredChild
    count: int = Field(description="Observed count.")


def test_stable_context_precedes_agent_schema_and_dynamic_payload(tmp_path, monkeypatch) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    cfg.llm.enable_cache = False
    client = LLMClient(cfg, tmp_path)
    calls: list[dict] = []

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"child":{"code":"A"},"count":2}'),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
    )

    def fake_completion(**kwargs):
        calls.append(kwargs["create_kwargs"])
        return response

    monkeypatch.setattr(client, "_chat_completion_with_network_retry", fake_completion)
    parsed = client.ask_structured(
        model_cls=_StructuredPayload,
        system_prompt="Stage-specific reviewer.",
        user_prompt="current",
        prompt_name="cache_layout_test",
        static_context_prompt="authoritative stable evidence",
        dynamic_user_prompt="current changing request",
    )

    assert parsed.count == 2
    messages = calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": CACHE_FRIENDLY_SYSTEM_PROMPT}
    assert "authoritative stable evidence" in messages[1]["content"]
    assert messages[2]["content"].startswith(AGENT_INSTRUCTIONS_TITLE)
    assert "Required JSON Schema" in messages[3]["content"]
    assert "JSON 示例" in messages[3]["content"]
    assert '{"child":{"code":""},"count":0}' in messages[3]["content"]
    assert "current changing request" in messages[-1]["content"]
    assert "prompt_cache_key" not in calls[0]
    assert calls[0]["max_tokens"] == 32768


def test_normal_text_and_structured_requests_enforce_output_floor(tmp_path, monkeypatch) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    cfg.llm.enable_cache = False
    cfg.llm.minimum_output_tokens = 32768
    cfg.llm.max_tokens = 1024
    cfg.llm.structured_max_tokens = 2048
    client = LLMClient(cfg, tmp_path)
    calls: list[dict] = []
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"child":{"code":"A"},"count":2}'),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        ),
    ]

    def fake_completion(**kwargs):
        calls.append(kwargs["create_kwargs"])
        return responses[len(calls) - 1]

    monkeypatch.setattr(client, "_chat_completion_with_network_retry", fake_completion)
    assert client.ask_text("system", "user", "text_floor_test") == "ok"
    parsed = client.ask_structured(
        _StructuredPayload,
        "system",
        "user",
        "structured_floor_test",
        max_tokens=3072,
    )

    assert parsed.count == 2
    assert [call["max_tokens"] for call in calls] == [32768, 32768]


def test_health_check_keeps_small_non_business_output_cap(tmp_path, monkeypatch) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    client = LLMClient(cfg, tmp_path)
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs["create_kwargs"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"), finish_reason="stop")],
            usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        )

    monkeypatch.setattr(client, "_chat_completion_with_network_retry", fake_completion)
    client.health_check()

    assert calls[0]["max_tokens"] == 8


def test_deepseek_official_endpoint_is_normalized_to_beta_for_downstream_capabilities() -> None:
    assert _normalize_base_url(
        "deepseek-v4-pro",
        "https://api.deepseek.com",
    ) == "https://api.deepseek.com/beta"
    assert _normalize_base_url(
        "deepseek-v4-pro",
        "https://api.deepseek.com/beta",
    ) == "https://api.deepseek.com/beta"


def test_deepseek_thinking_options_follow_official_request_shape() -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.model_name = "deepseek-v4-pro"
    cfg.llm.enable_thinking = None
    cfg.llm.reasoning_effort = "xhigh"
    kwargs = {
        "temperature": 0.2,
        "top_p": 0.8,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5,
    }

    _apply_deepseek_request_options(kwargs, cfg, structured=False)

    assert kwargs == {"reasoning_effort": "max"}

    cfg.llm.enable_thinking = False
    disabled_kwargs = {"temperature": 0.2}
    _apply_deepseek_request_options(disabled_kwargs, cfg, structured=False)
    assert disabled_kwargs["temperature"] == 0.2
    assert disabled_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in disabled_kwargs


def test_deepseek_insufficient_system_resource_finish_reason_retries(tmp_path, monkeypatch) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    cfg.llm.model_name = "deepseek-v4-pro"
    client = LLMClient(cfg, tmp_path)
    calls = 0
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="insufficient_system_resource")],
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop")],
        ),
    ]

    def fake_create(**_kwargs):
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
    )
    monkeypatch.setattr("autorealize.llm.client.time.sleep", lambda _seconds: None)

    result = client._chat_completion_with_network_retry(
        prompt_name="deepseek_resource_retry_test",
        mode="text",
        create_kwargs={"model": "deepseek-v4-pro", "messages": []},
    )

    assert calls == 2
    assert result.choices[0].finish_reason == "stop"


def test_provider_prompt_cache_key_auto_is_openai_only() -> None:
    cfg = AutoRealizeConfig()
    stable_prefix = "authoritative stable evidence"

    openai_key = _provider_prompt_cache_key(
        config=cfg,
        base_url="https://api.openai.com/v1",
        stable_prefix=stable_prefix,
    )
    assert openai_key is not None
    assert openai_key == _provider_prompt_cache_key(
        config=cfg,
        base_url="https://api.openai.com/v1/",
        stable_prefix=stable_prefix,
    )
    assert _provider_prompt_cache_key(
        config=cfg,
        base_url="https://api.deepseek.com/beta",
        stable_prefix=stable_prefix,
    ) is None

    cfg.llm.prompt_cache_key_mode = "enabled"
    assert _provider_prompt_cache_key(
        config=cfg,
        base_url="https://compatible-provider.example/v1",
        stable_prefix=stable_prefix,
    ) == openai_key

    cfg.llm.prompt_cache_key_mode = "disabled"
    assert _provider_prompt_cache_key(
        config=cfg,
        base_url="https://api.openai.com/v1",
        stable_prefix=stable_prefix,
    ) is None


def test_openai_calls_reuse_stable_prefix_cache_key_across_dynamic_tails(tmp_path, monkeypatch) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    cfg.llm.base_url = "https://api.openai.com/v1"
    cfg.llm.model_name = "gpt-5.6"
    cfg.llm.enable_cache = False
    client = LLMClient(cfg, tmp_path)
    calls: list[dict] = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage={"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
    )

    def fake_completion(**kwargs):
        calls.append(kwargs["create_kwargs"])
        return response

    monkeypatch.setattr(client, "_chat_completion_with_network_retry", fake_completion)
    for dynamic_tail in ["first request", "second request"]:
        client.ask_text(
            system_prompt="Review the current request.",
            user_prompt=dynamic_tail,
            prompt_name="provider_cache_key_test",
            static_user_prompt="shared immutable evidence",
            dynamic_user_prompt=dynamic_tail,
        )

    assert calls[0]["prompt_cache_key"] == calls[1]["prompt_cache_key"]
    assert calls[0]["messages"][-1]["content"] != calls[1]["messages"][-1]["content"]


def test_prompt_cache_key_degrades_when_sdk_or_provider_rejects_it() -> None:
    kwargs = {"model": "gpt-5.6", "messages": [], "prompt_cache_key": "autorealize:test"}
    degraded, key = _degraded_request_kwargs(
        kwargs,
        TypeError("Completions.create() got an unexpected keyword argument 'prompt_cache_key'"),
    )

    assert key == "prompt_cache_key"
    assert degraded is not None
    assert "prompt_cache_key" not in degraded


def test_stable_prompt_puts_original_requirements_first() -> None:
    stable, dynamic = stable_dynamic_prompt(
        stable={
            "stage_specific": "later",
            "original_requirements_full": "authoritative original text",
        },
        dynamic={"error": "latest"},
        stable_title="varying title that must not lead the prefix",
    )

    assert stable.startswith(f"{STABLE_CONTEXT_TITLE}\n{{\n  \"original_requirements_full\"")
    assert stable.index("authoritative original text") < stable.index("stage_specific")
    assert "latest" in dynamic


def test_json_block_limit_keeps_valid_head_and_latest_tail() -> None:
    block = json_block(
        "dynamic",
        {"early": "A" * 500, "latest_request": "LATEST_TAIL"},
        limit=420,
        sort_keys=False,
    )
    payload = json.loads(block.split("\n", 1)[1])

    assert payload["_prompt_truncation"]["truncated"] is True
    assert "LATEST_TAIL" in payload["visible_json_tail"]


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


def test_usage_cache_write_tokens_supports_chat_and_response_details() -> None:
    assert _usage_cache_write_tokens(
        {"prompt_tokens_details": {"cache_write_tokens": 32}}
    ) == 32
    assert _usage_cache_write_tokens(
        {"input_tokens_details": {"cache_write_tokens": 48}}
    ) == 48


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


def test_deepseek_cost_breakdown_uses_hit_miss_output_prices() -> None:
    breakdown = _deepseek_cost_breakdown(
        prompt_tokens=150,
        cached_tokens=100,
        miss_tokens=40,
        completion_tokens=20,
    )

    assert breakdown["unknown_input_tokens"] == 10
    assert breakdown["cache_hit_input_usd"] == 0.000000363
    assert breakdown["cache_miss_input_usd"] == 0.0000174
    assert breakdown["unknown_input_as_miss_usd"] == 0.00000435
    assert breakdown["output_usd"] == 0.0000174
    assert breakdown["total_unknown_as_miss_usd"] == 0.000039512
    assert _deepseek_pricing_usd_per_1m("deepseek-v4-flash")["cache_miss_input"] == 0.14
    assert _deepseek_pricing_usd_per_1m("deepseek-reasoner") == _deepseek_pricing_usd_per_1m(
        "deepseek-v4-flash"
    )


def test_prompt_stage_groups_known_autorealize_prompts() -> None:
    assert _prompt_stage("question_investigator_action_1") == "qdi"
    assert _prompt_stage("description_protocol_static_optimization_1") == "description_protocol"
    assert _prompt_stage("evaluation_contract_reviewer_2") == "evaluation_contract"


def test_compact_structured_schema_removes_prompt_metadata_but_preserves_validation_shape() -> None:
    full_schema = _StructuredPayload.model_json_schema()
    compact_text = _schema_text_for_prompt(full_schema, compact=True)
    compact_schema = json.loads(compact_text)

    assert len(compact_text) < len(json.dumps(full_schema, ensure_ascii=False, indent=2))
    assert "title" not in compact_text
    assert "description" not in compact_text
    assert "$defs" in compact_schema
    assert compact_schema["required"] == ["child", "count"]
    assert compact_schema["properties"]["child"]["$ref"].startswith("#/$defs/")
    assert _StructuredPayload.model_validate({"child": {"code": "A"}, "count": 2}).count == 2


def test_structured_length_retry_disables_deepseek_thinking(monkeypatch, tmp_path) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    cfg.llm.model_name = "deepseek-v4-pro"
    cfg.llm.enable_thinking = True
    cfg.llm.structured_disable_thinking = False
    cfg.llm.structured_reasoning_fallback_on_length = True
    cfg.llm.structured_reasoning_fallback_ratio = 0.75
    cfg.llm.max_retries = 2
    client = LLMClient(cfg, tmp_path)
    calls: list[dict] = []
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"child":'),
                    finish_reason="length",
                )
            ],
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 4096,
                "total_tokens": 4196,
                "completion_tokens_details": {"reasoning_tokens": 4090},
            },
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"child":{"code":"A"},"count":2}'),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
        ),
    ]

    def fake_completion(**kwargs):
        calls.append(kwargs["create_kwargs"])
        return responses[len(calls) - 1]

    monkeypatch.setattr(client, "_chat_completion_with_network_retry", fake_completion)

    parsed = client.ask_structured(
        model_cls=_StructuredPayload,
        system_prompt="Return structured data.",
        user_prompt="Build the payload.",
        prompt_name="structured_reasoning_fallback_test",
        max_tokens=4096,
    )

    assert parsed == _StructuredPayload(child=_StructuredChild(code="A"), count=2)
    assert len(calls) == 2
    assert calls[0]["extra_body"]["thinking"]["type"] == "enabled"
    assert calls[1]["extra_body"]["thinking"]["type"] == "disabled"
    assert calls[0]["max_tokens"] == 32768
    assert calls[1]["max_tokens"] == 32768
    assert "temperature" not in calls[0]
    assert calls[1]["temperature"] == cfg.llm.temperature
    usage_rows = [json.loads(line) for line in (tmp_path / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["parsed_ok"] for row in usage_rows] == [False, True]
    assert usage_rows[0]["raw_usage"]["completion_tokens_details"]["reasoning_tokens"] == 4090


def test_log_provider_usage_writes_jsonl_summary_and_prompt_parts(tmp_path) -> None:
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    cfg.llm.model_name = "deepseek-v4-pro"
    client = LLMClient(cfg, tmp_path)
    response = SimpleNamespace(
        id="chatcmpl-deepseek-test",
        model="deepseek-v4-pro-202607",
        system_fingerprint="fp_deepseek_v4_test",
        usage={
            "prompt_tokens": 120,
            "completion_tokens": 12,
            "total_tokens": 132,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 40,
            "prompt_tokens_details": {"cache_write_tokens": 24},
            "completion_tokens_details": {"reasoning_tokens": 7},
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
    assert row["reasoning_tokens"] == 7
    assert row["response_id"] == "chatcmpl-deepseek-test"
    assert row["response_model"] == "deepseek-v4-pro-202607"
    assert row["system_fingerprint"] == "fp_deepseek_v4_test"
    assert row["prompt_cache_hit_tokens"] == 80
    assert row["prompt_cache_miss_tokens"] == 40
    assert row["cache_write_tokens"] == 24
    assert row["provider_cache_write_tokens_known"] is True
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
    assert summary["reasoning_tokens"] == 7
    assert summary["provider_response_models"] == {"deepseek-v4-pro-202607": 1}
    assert summary["system_fingerprints"] == {"fp_deepseek_v4_test": 1}
    assert summary["prompt_cache_hit_tokens"] == 80
    assert summary["prompt_cache_miss_tokens"] == 40
    assert summary["cache_write_tokens"] == 24
    assert summary["provider_cache_write_known_calls"] == 1
    assert summary["provider_cache_write_unknown_calls"] == 0
    assert summary["provider_cache_hit_ratio"] == 0.666667
    assert summary["deepseek_cost_breakdown_usd"]["cache_miss_input_tokens"] == 40
    assert summary["by_prompt"]["unit_prompt"]["calls"] == 1
    assert summary["by_prompt"]["unit_prompt"]["by_part_ranked"][0]["estimated_tokens"] >= 1
    assert summary["by_prompt_part_ranked"][0]["prompt_name"] == "unit_prompt"
    brief = json.loads((tmp_path / "llm_usage_brief.json").read_text(encoding="utf-8"))
    assert brief["schema_version"] == "autorealize.llm_usage_brief.v2"
    assert brief["deepseek_pricing_usd_per_1m"]["cache_miss_input"] == 0.435
    assert brief["deepseek_cost_breakdown_usd"]["output_tokens"] == 12
    assert brief["reasoning_tokens"] == 7
    assert brief["cache_write_tokens"] == 24
    assert brief["provider_cache_write_known_calls"] == 1
    assert brief["top_prompts_by_estimated_cost"][0]["cache_write_tokens"] == 24
    assert brief["by_stage"][0]["cache_write_tokens"] == 24
    assert brief["top_prompts_by_estimated_cost"][0]["stage"] == "other"
    assert brief["by_stage"][0]["stage"] == "other"


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
    assert row["provider_cache_write_tokens_known"] is False
    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["provider_cache_unknown_prompt_tokens"] == 25
    assert summary["provider_cache_known_prompt_tokens"] == 0
    assert summary["provider_cache_write_unknown_calls"] == 1


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
