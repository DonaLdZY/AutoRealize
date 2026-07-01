from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
from copy import deepcopy
from pathlib import Path
from threading import Lock, BoundedSemaphore
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config import AutoRealizeConfig
from ..logging_utils import log_event
from ..models import LLMTrace
from ..utils.safe_json import append_jsonl_safe, write_json_safe

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
NETWORK_RETRY_MAX_ATTEMPTS = 5
NETWORK_RETRY_MAX_SLEEP_SECONDS = 30.0
DEEPSEEK_RMB_PER_1M_CACHE_HIT_INPUT = 0.025
DEEPSEEK_RMB_PER_1M_CACHE_MISS_INPUT = 3.0
DEEPSEEK_RMB_PER_1M_OUTPUT = 6.0


def _is_deepseek_model(model_name: str) -> bool:
    return (model_name or "").strip().lower().startswith("deepseek")


def _estimate_deepseek_rmb(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    miss_tokens: int,
    completion_tokens: int,
    unknown_prompt_as_miss: bool,
) -> float:
    unknown_prompt_tokens = max(0, prompt_tokens - cached_tokens - miss_tokens)
    billed_miss_tokens = miss_tokens + (unknown_prompt_tokens if unknown_prompt_as_miss else 0)
    return (
        cached_tokens * DEEPSEEK_RMB_PER_1M_CACHE_HIT_INPUT
        + billed_miss_tokens * DEEPSEEK_RMB_PER_1M_CACHE_MISS_INPUT
        + completion_tokens * DEEPSEEK_RMB_PER_1M_OUTPUT
    ) / 1_000_000.0


def _deepseek_cost_breakdown(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    miss_tokens: int,
    completion_tokens: int,
) -> dict[str, float | int]:
    unknown_prompt_tokens = max(0, int(prompt_tokens) - int(cached_tokens) - int(miss_tokens))
    cache_hit_rmb = int(cached_tokens) * DEEPSEEK_RMB_PER_1M_CACHE_HIT_INPUT / 1_000_000.0
    cache_miss_rmb = int(miss_tokens) * DEEPSEEK_RMB_PER_1M_CACHE_MISS_INPUT / 1_000_000.0
    unknown_as_miss_rmb = unknown_prompt_tokens * DEEPSEEK_RMB_PER_1M_CACHE_MISS_INPUT / 1_000_000.0
    output_rmb = int(completion_tokens) * DEEPSEEK_RMB_PER_1M_OUTPUT / 1_000_000.0
    return {
        "cache_hit_input_tokens": int(cached_tokens),
        "cache_miss_input_tokens": int(miss_tokens),
        "unknown_input_tokens": unknown_prompt_tokens,
        "output_tokens": int(completion_tokens),
        "cache_hit_input_rmb": round(cache_hit_rmb, 6),
        "cache_miss_input_rmb": round(cache_miss_rmb, 6),
        "unknown_input_as_miss_rmb": round(unknown_as_miss_rmb, 6),
        "output_rmb": round(output_rmb, 6),
        "total_cache_known_only_rmb": round(cache_hit_rmb + cache_miss_rmb + output_rmb, 6),
        "total_unknown_as_miss_rmb": round(cache_hit_rmb + cache_miss_rmb + unknown_as_miss_rmb + output_rmb, 6),
    }


def _prompt_stage(prompt_name: str) -> str:
    name = (prompt_name or "").strip().lower()
    if name.startswith("question_investigator") or name.startswith("qdi"):
        return "qdi"
    if name.startswith("cognition") or name.startswith("file_") or name.startswith("llm_file"):
        return "data_cognition"
    if name.startswith("problem_paradigm"):
        return "problem_paradigm"
    if name.startswith("description_protocol"):
        return "description_protocol"
    if name.startswith("description_"):
        return "description_sections"
    if name.startswith("evaluation_contract") or name.startswith("eval_"):
        return "evaluation_contract"
    if name.startswith("sample_submission"):
        return "sample_submission"
    if name.startswith("architect"):
        return "architect_plan"
    if name.startswith("llm_health"):
        return "health_check"
    return "other"


def _normalize_base_url(model_name: str, base_url: str) -> str:
    base = (base_url or "").strip()
    if _is_deepseek_model(model_name) and base.rstrip("/") in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        return "https://api.deepseek.com/beta"
    return base


def _example_from_schema(schema: dict) -> Any:
    if not isinstance(schema, dict):
        return ""
    if "anyOf" in schema and schema["anyOf"]:
        return _example_from_schema(schema["anyOf"][0])
    if "$defs" in schema and "$ref" in schema:
        ref = str(schema["$ref"]).split("/")[-1]
        return _example_from_schema(schema.get("$defs", {}).get(ref, {}))
    schema_type = schema.get("type")
    if schema_type == "object":
        return {str(k): _example_from_schema(v) for k, v in (schema.get("properties", {}) or {}).items()}
    if schema_type == "array":
        return [_example_from_schema(schema.get("items", {}))]
    if schema_type == "boolean":
        return False
    if schema_type in {"integer", "number"}:
        return 0
    if schema.get("enum"):
        return schema["enum"][0]
    return ""


def _normalize_deepseek_reasoning_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    value = str(effort).strip().lower()
    if value in {"default", "none", "null"}:
        return None
    return {"low": "high", "medium": "high", "xhigh": "max"}.get(value, value)


def _normal_max_tokens(value: Any) -> int | None:
    try:
        tokens = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return None
    return tokens if tokens > 0 else None


def _deepseek_extra_body(config: AutoRealizeConfig, *, structured: bool) -> dict[str, Any]:
    if not _is_deepseek_model(config.llm.model_name):
        return {}
    body: dict[str, Any] = {}
    thinking = False if structured and config.llm.structured_disable_thinking else config.llm.enable_thinking
    if thinking is not None:
        body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        effort = _normalize_deepseek_reasoning_effort(config.llm.reasoning_effort)
        if thinking and effort:
            body["reasoning_effort"] = effort
    return body


def _is_provider_parameter_error(exc: Exception) -> bool:
    """Return True when an OpenAI-compatible provider rejects optional params."""
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if any(key in name for key in ["badrequest", "invalidrequest"]):
        return True
    return any(
        key in msg
        for key in [
            "unsupported parameter",
            "unknown parameter",
            "invalid parameter",
            "extra_body",
            "response_format",
            "json_object",
            "reasoning_effort",
            "thinking",
            "extra fields not permitted",
        ]
    )


def _degraded_request_kwargs(create_kwargs: dict[str, Any], exc: Exception) -> tuple[dict[str, Any] | None, str]:
    """Drop optional DeepSeek/provider params when a compatible proxy rejects them."""
    if not _is_provider_parameter_error(exc):
        return None, ""

    msg = str(exc).lower()
    if "extra_body" in create_kwargs and any(
        key in msg for key in ["extra_body", "thinking", "reasoning_effort", "unknown parameter", "invalid parameter"]
    ):
        degraded = dict(create_kwargs)
        degraded.pop("extra_body", None)
        return degraded, "extra_body"

    if "response_format" in create_kwargs and any(
        key in msg for key in ["response_format", "json_object", "unsupported parameter", "unknown parameter", "invalid parameter"]
    ):
        degraded = dict(create_kwargs)
        degraded.pop("response_format", None)
        return degraded, "response_format"

    return None, ""


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return match.group(0)


def _escape_control_chars_in_json_strings(text: str) -> str:
    """Escape raw control characters that some LLMs emit inside JSON strings."""
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string and (ord(ch) < 0x20):
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(f"\\u{ord(ch):04x}")
            continue
        out.append(ch)
    return "".join(out)


def _loads_json_object(text: str) -> dict[str, Any]:
    raw = _extract_json_object(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # LLMs sometimes put literal newlines/tabs inside JSON string values.
        # strict=False accepts those control characters and preserves content.
        try:
            obj = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            obj = json.loads(_escape_control_chars_in_json_strings(raw))
    if not isinstance(obj, dict):
        raise ValueError("Structured LLM response is not a JSON object")
    return obj


def _preview_value(value: Any, *, limit: int = 220) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit].replace("\n", "\\n")


def _format_validation_error(exc: ValidationError) -> str:
    """Render Pydantic errors with field paths instead of only a docs URL."""
    parts: list[str] = []
    for err in exc.errors()[:12]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
        input_value = err.get("input", None)
        parts.append(
            f"{loc}: {err.get('msg', '')} "
            f"(type={err.get('type', '')}, input_type={type(input_value).__name__}, "
            f"input={_preview_value(input_value)})"
        )
    if len(exc.errors()) > 12:
        parts.append(f"... {len(exc.errors()) - 12} more validation errors")
    return "Pydantic validation failed: " + "; ".join(parts)


def _stringify_schema_value(value: Any) -> str:
    """Convert common LLM list/object mistakes into readable string fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if all(not isinstance(x, (dict, list)) for x in value):
            return "；".join(str(x) for x in value)
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _set_path_value(root: Any, loc: tuple[Any, ...], value: Any) -> bool:
    if not loc:
        return False
    cur = root
    for part in loc[:-1]:
        if isinstance(cur, dict):
            if part not in cur:
                return False
            cur = cur[part]
            continue
        if isinstance(cur, list) and isinstance(part, int) and 0 <= part < len(cur):
            cur = cur[part]
            continue
        return False
    last = loc[-1]
    if isinstance(cur, dict):
        cur[last] = value
        return True
    if isinstance(cur, list) and isinstance(last, int) and 0 <= last < len(cur):
        cur[last] = value
        return True
    return False


def _validate_with_string_coercion(model_cls: type[T], obj: dict[str, Any]) -> tuple[T, str]:
    """Validate JSON and repair only the common string_type LLM shape error."""
    try:
        return model_cls.model_validate(obj), ""
    except ValidationError as exc:
        string_errors = [err for err in exc.errors() if err.get("type") == "string_type"]
        if not string_errors:
            raise
        repaired = deepcopy(obj)
        repaired_locs: list[str] = []
        for err in string_errors:
            loc = tuple(err.get("loc", ()))
            if _set_path_value(repaired, loc, _stringify_schema_value(err.get("input"))):
                repaired_locs.append(".".join(str(x) for x in loc) or "<root>")
        if not repaired_locs:
            raise
        parsed = model_cls.model_validate(repaired)
        return parsed, f"normalized string_type fields: {', '.join(repaired_locs[:20])}"


def _format_parse_error(exc: Exception, *, finish_reason: str, max_tokens: int, text: str) -> str:
    if isinstance(exc, ValidationError):
        message = _format_validation_error(exc)
    else:
        message = str(exc)
    if finish_reason == "length":
        message = f"{message}; finish_reason=length; response may be truncated; max_tokens={max_tokens}"
    elif not text.strip():
        message = f"{message}; empty structured response content"
    if text.strip():
        message = f"{message}; response_head={_preview_value(text, limit=360)}"
    return message


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            data = usage.model_dump()
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    if isinstance(usage, dict):
        return dict(usage)
    out: dict[str, Any] = {}
    for key in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "cached_tokens",
    ]:
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    for key in ["prompt_tokens_details", "completion_tokens_details"]:
        details = getattr(usage, key, None)
        if details is None:
            continue
        if hasattr(details, "model_dump"):
            try:
                details = details.model_dump()
            except Exception:
                details = None
        if isinstance(details, dict):
            out[key] = details
    return out


def _usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    pools = [usage, details, completion_details]
    for key in keys:
        for pool in pools:
            value = pool.get(key) if isinstance(pool, dict) else None
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
    return None


def _usage_cache_tokens(usage: dict[str, Any]) -> tuple[int | None, int | None]:
    cached = _usage_int(
        usage,
        "prompt_cache_hit_tokens",
        "cached_tokens",
        "cache_read_input_tokens",
    )
    missed = _usage_int(
        usage,
        "prompt_cache_miss_tokens",
        "cache_miss_input_tokens",
    )
    prompt = _usage_int(usage, "prompt_tokens")
    if missed is None and prompt is not None and cached is not None:
        missed = max(0, prompt - cached)
    return cached, missed


def _estimate_text_tokens(text: str) -> int:
    """Cheap local estimate for locating large prompt parts.

    Provider usage reports exact totals only. This estimate is intentionally
    lightweight and dependency-free; use it for relative attribution, not billing.
    """
    if not text:
        return 0
    cjk = 0
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
    non_cjk = max(0, len(text) - cjk)
    return max(1, int(round(cjk + non_cjk / 4)))


def _prompt_part_stats(
    prompt_parts: list[dict[str, Any]] | None,
    *,
    provider_prompt_tokens: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    if not prompt_parts:
        return rows, 0
    for idx, part in enumerate(prompt_parts):
        raw_content = part.get("content", None)
        has_content = raw_content is not None
        content = str(raw_content or "") if has_content else ""
        synthetic_chars = part.get("chars", None)
        synthetic_bytes = part.get("utf8_bytes", None)
        synthetic_tokens = part.get("estimated_tokens", None)
        if not content and synthetic_chars is None and synthetic_tokens is None:
            continue
        if has_content:
            chars = len(content)
            utf8_bytes = len(content.encode("utf-8"))
            estimated_tokens = _estimate_text_tokens(content)
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        else:
            try:
                chars = int(synthetic_chars or 0)
            except Exception:
                chars = 0
            try:
                utf8_bytes = int(synthetic_bytes if synthetic_bytes is not None else chars)
            except Exception:
                utf8_bytes = chars
            try:
                estimated_tokens = int(synthetic_tokens if synthetic_tokens is not None else max(1, chars // 4))
            except Exception:
                estimated_tokens = max(1, chars // 4)
            digest = str(part.get("sha256_16", "synthetic"))[:16]
        rows.append(
            {
                "index": idx,
                "name": str(part.get("name", f"part_{idx}") or f"part_{idx}"),
                "role": str(part.get("role", "") or ""),
                "chars": chars,
                "utf8_bytes": utf8_bytes,
                "estimated_tokens": estimated_tokens,
                "sha256_16": digest,
            }
        )
    total_estimated = sum(int(x.get("estimated_tokens", 0) or 0) for x in rows)
    for row in rows:
        estimated = int(row.get("estimated_tokens", 0) or 0)
        row["share_of_estimated_prompt"] = round(estimated / total_estimated, 6) if total_estimated else 0.0
        row["provider_prompt_tokens_estimate"] = (
            int(round(provider_prompt_tokens * estimated / total_estimated))
            if provider_prompt_tokens and total_estimated
            else 0
        )
    return rows, total_estimated


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(self, config: AutoRealizeConfig, run_dir: Path) -> None:
        self.config = config
        if not self.config.llm.api_key:
            raise ValueError("Missing LLM API key. Set DEEPSEEK_API_KEY or configure llm.api_key.")
        self.base_url = _normalize_base_url(self.config.llm.model_name, self.config.llm.base_url)
        if self.base_url != self.config.llm.base_url:
            log_event(
                logger,
                "llm.client",
                "BASE_URL_NORMALIZED",
                model=self.config.llm.model_name,
                from_url=self.config.llm.base_url,
                to_url=self.base_url,
            )
        self.client = OpenAI(api_key=self.config.llm.api_key, base_url=self.base_url)
        self.trace_path = run_dir / "llm_traces.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = run_dir / "llm_cache.jsonl"
        self.usage_path = run_dir / "llm_usage.jsonl"
        self.usage_summary_path = run_dir / "llm_usage_summary.json"
        self.usage_brief_path = run_dir / "llm_usage_brief.json"
        self._cache: dict[str, str] = {}
        self._cache_lock = Lock()
        self._trace_lock = Lock()
        self._usage_lock = Lock()
        self._usage_summary: dict[str, Any] = {
            "calls": 0,
            "seconds": 0.0,
            "cache_hits_local": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "provider_cache_known_prompt_tokens": 0,
            "provider_cache_unknown_prompt_tokens": 0,
            "provider_usage_missing_calls": 0,
            "estimated_prompt_tokens": 0,
            "by_prompt_part": {},
            "by_prompt": {},
        }
        self._request_gate = BoundedSemaphore(max(1, int(self.config.llm.max_concurrent_requests)))
        self._load_cache()

    def health_check(self) -> None:
        """Fail fast when the configured LLM endpoint is unreachable."""
        log_event(
            logger,
            "llm.client",
            "HEALTH_CHECK_STARTED",
            model=self.config.llm.model_name,
            base_url=self.base_url,
        )
        create_kwargs = {
            "model": self.config.llm.model_name,
            "temperature": 0,
            "max_tokens": 8,
            "messages": [
                {"role": "system", "content": "You are a health check endpoint."},
                {"role": "user", "content": "Return OK."},
            ],
            "stream": False,
            "timeout": self.config.llm.request_timeout_seconds,
        }
        prompt_parts = [
            {"name": "system_prompt", "role": "system", "content": "You are a health check endpoint."},
            {"name": "user_prompt", "role": "user", "content": "Return OK."},
        ]
        extra_body = _deepseek_extra_body(self.config, structured=False)
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        try:
            t0 = time.perf_counter()
            response = self._chat_completion_with_network_retry(
                prompt_name="llm_health_check",
                mode="health_check",
                create_kwargs=create_kwargs,
            )
            self._log_provider_usage(
                prompt_name="llm_health_check",
                mode="health_check",
                response=response,
                seconds=time.perf_counter() - t0,
                max_tokens=8,
                parsed_ok=True,
                prompt_parts=prompt_parts,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError("LLM health check returned no choices")
        except Exception as exc:
            log_event(logger, "llm.client", "HEALTH_CHECK_FAILED", error=str(exc)[:240])
            raise RuntimeError(f"LLM health check failed: {exc}") from exc
        log_event(logger, "llm.client", "HEALTH_CHECK_COMPLETED")

    def _cache_key(self, *, prompt_name: str, system_prompt: str, user_prompt: str, schema: str = "") -> str:
        payload = json.dumps(
            {
                "model": self.config.llm.model_name,
                "temperature": self.config.llm.temperature,
                "prompt_name": prompt_name,
                "system": system_prompt,
                "user": user_prompt,
                "schema": schema,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self._cache_lock:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    self._cache[str(obj["key"])] = str(obj["response"])
                except Exception:
                    continue

    def _cache_get(self, key: str) -> str | None:
        if not self.config.llm.enable_cache:
            return None
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, response: str) -> None:
        if not self.config.llm.enable_cache:
            return
        with self._cache_lock:
            self._cache[key] = response
            append_jsonl_safe(self.cache_path, {"key": key, "response": response})

    def _log_trace(self, trace: LLMTrace) -> None:
        with self._trace_lock:
            append_jsonl_safe(self.trace_path, trace.model_dump())

    def _log_local_cache_usage(
        self,
        *,
        prompt_name: str,
        mode: str,
        prompt_parts: list[dict[str, Any]] | None = None,
    ) -> None:
        part_rows, estimated_prompt_tokens = _prompt_part_stats(prompt_parts)
        row = {
            "ts": time.time(),
            "prompt_name": prompt_name,
            "mode": mode,
            "source": "local_cache",
            "model": self.config.llm.model_name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "prompt_parts": part_rows,
        }
        with self._usage_lock:
            self._usage_summary["cache_hits_local"] = int(self._usage_summary.get("cache_hits_local", 0)) + 1
            self._usage_summary["estimated_prompt_tokens"] = (
                int(self._usage_summary.get("estimated_prompt_tokens", 0)) + estimated_prompt_tokens
            )
            self._accumulate_prompt_parts_locked(prompt_name=prompt_name, part_rows=part_rows)
            by_prompt = self._usage_summary.setdefault("by_prompt", {})
            item = by_prompt.setdefault(
                prompt_name,
                {
                    "calls": 0,
                    "cache_hits_local": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 0,
                    "provider_cache_known_prompt_tokens": 0,
                    "provider_cache_unknown_prompt_tokens": 0,
                    "provider_usage_missing_calls": 0,
                    "estimated_prompt_tokens": 0,
                    "by_part": {},
                },
            )
            item["cache_hits_local"] = int(item.get("cache_hits_local", 0)) + 1
            item["estimated_prompt_tokens"] = int(item.get("estimated_prompt_tokens", 0)) + estimated_prompt_tokens
            self._accumulate_prompt_parts_for_prompt_locked(item, part_rows)
            self._append_usage_row_locked(row)
            self._write_usage_summary_locked()
        log_event(
            logger,
            "llm.client",
            "USAGE_RECORDED",
            prompt=prompt_name,
            mode=mode,
            source="local_cache",
            prompt_tokens=0,
            cached_tokens=0,
            miss_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def _log_provider_usage(
        self,
        *,
        prompt_name: str,
        mode: str,
        response: Any,
        seconds: float,
        attempt: int | None = None,
        finish_reason: str = "",
        max_tokens: int | None = None,
        parsed_ok: bool | None = None,
        source: str = "provider",
        model_name: str | None = None,
        prompt_parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        usage = _usage_to_dict(getattr(response, "usage", None))
        usage_available = bool(usage)
        prompt_tokens = _usage_int(usage, "prompt_tokens") or 0
        completion_tokens = _usage_int(usage, "completion_tokens") or 0
        total_tokens = _usage_int(usage, "total_tokens") or (prompt_tokens + completion_tokens)
        cached_tokens, miss_tokens = _usage_cache_tokens(usage)
        cache_tokens_known = cached_tokens is not None or miss_tokens is not None
        part_rows, estimated_prompt_tokens = _prompt_part_stats(
            prompt_parts,
            provider_prompt_tokens=prompt_tokens,
        )
        row = {
            "ts": time.time(),
            "prompt_name": prompt_name,
            "mode": mode,
            "source": source,
            "model": model_name or self.config.llm.model_name,
            "attempt": attempt,
            "seconds": round(float(seconds), 4),
            "finish_reason": finish_reason,
            "max_tokens": max_tokens,
            "parsed_ok": parsed_ok,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_cache_hit_tokens": cached_tokens or 0,
            "prompt_cache_miss_tokens": miss_tokens or 0,
            "provider_cache_tokens_known": cache_tokens_known,
            "usage_available": usage_available,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "prompt_parts": part_rows,
            "raw_usage": usage,
        }
        with self._usage_lock:
            self._usage_summary["calls"] = int(self._usage_summary.get("calls", 0)) + 1
            self._usage_summary["seconds"] = round(float(self._usage_summary.get("seconds", 0.0) or 0.0) + float(seconds or 0.0), 4)
            if not usage_available:
                self._usage_summary["provider_usage_missing_calls"] = (
                    int(self._usage_summary.get("provider_usage_missing_calls", 0)) + 1
                )
            for key in [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ]:
                self._usage_summary[key] = int(self._usage_summary.get(key, 0)) + int(row.get(key, 0) or 0)
            self._usage_summary["estimated_prompt_tokens"] = (
                int(self._usage_summary.get("estimated_prompt_tokens", 0)) + estimated_prompt_tokens
            )
            self._accumulate_prompt_parts_locked(prompt_name=prompt_name, part_rows=part_rows)
            cache_bucket = (
                "provider_cache_known_prompt_tokens"
                if cache_tokens_known
                else "provider_cache_unknown_prompt_tokens"
            )
            self._usage_summary[cache_bucket] = int(self._usage_summary.get(cache_bucket, 0)) + prompt_tokens
            by_prompt = self._usage_summary.setdefault("by_prompt", {})
            item = by_prompt.setdefault(
                prompt_name,
                {
                    "calls": 0,
                    "seconds": 0.0,
                    "cache_hits_local": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 0,
                    "provider_cache_known_prompt_tokens": 0,
                    "provider_cache_unknown_prompt_tokens": 0,
                    "provider_usage_missing_calls": 0,
                    "estimated_prompt_tokens": 0,
                    "by_part": {},
                },
            )
            item["calls"] = int(item.get("calls", 0)) + 1
            item["seconds"] = round(float(item.get("seconds", 0.0) or 0.0) + float(seconds or 0.0), 4)
            if not usage_available:
                item["provider_usage_missing_calls"] = int(item.get("provider_usage_missing_calls", 0)) + 1
            for key in [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ]:
                item[key] = int(item.get(key, 0)) + int(row.get(key, 0) or 0)
            item[cache_bucket] = int(item.get(cache_bucket, 0)) + prompt_tokens
            item["estimated_prompt_tokens"] = int(item.get("estimated_prompt_tokens", 0)) + estimated_prompt_tokens
            self._accumulate_prompt_parts_for_prompt_locked(item, part_rows)
            self._append_usage_row_locked(row)
            self._write_usage_summary_locked()
        logger.info(
            "[llm_usage] prompt=%s mode=%s source=%s input=%s cached=%s miss=%s output=%s total=%s est_input=%s usage_available=%s cache_known=%s",
            prompt_name,
            mode,
            source,
            prompt_tokens,
            cached_tokens or 0,
            miss_tokens or 0,
            completion_tokens,
            total_tokens,
            estimated_prompt_tokens,
            usage_available,
            cache_tokens_known,
        )
        log_event(
            logger,
            "llm.client",
            "USAGE_RECORDED",
            prompt=prompt_name,
            mode=mode,
            source=source,
            attempt=attempt,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens or 0,
            miss_tokens=miss_tokens or 0,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            usage_available=usage_available,
            cache_tokens_known=cache_tokens_known,
        )
        return row

    def _accumulate_prompt_parts_locked(self, *, prompt_name: str, part_rows: list[dict[str, Any]]) -> None:
        by_part = self._usage_summary.setdefault("by_prompt_part", {})
        for row in part_rows:
            key = f"{prompt_name}:{row.get('name', '')}"
            item = by_part.setdefault(
                key,
                {
                    "prompt_name": prompt_name,
                    "part_name": row.get("name", ""),
                    "role": row.get("role", ""),
                    "calls": 0,
                    "chars": 0,
                    "utf8_bytes": 0,
                    "estimated_tokens": 0,
                    "provider_prompt_tokens_estimate": 0,
                },
            )
            item["calls"] = int(item.get("calls", 0)) + 1
            for field in ["chars", "utf8_bytes", "estimated_tokens", "provider_prompt_tokens_estimate"]:
                item[field] = int(item.get(field, 0)) + int(row.get(field, 0) or 0)

    @staticmethod
    def _accumulate_prompt_parts_for_prompt_locked(prompt_item: dict[str, Any], part_rows: list[dict[str, Any]]) -> None:
        by_part = prompt_item.setdefault("by_part", {})
        for row in part_rows:
            key = str(row.get("name", ""))
            item = by_part.setdefault(
                key,
                {
                    "role": row.get("role", ""),
                    "calls": 0,
                    "chars": 0,
                    "utf8_bytes": 0,
                    "estimated_tokens": 0,
                    "provider_prompt_tokens_estimate": 0,
                },
            )
            item["calls"] = int(item.get("calls", 0)) + 1
            for field in ["chars", "utf8_bytes", "estimated_tokens", "provider_prompt_tokens_estimate"]:
                item[field] = int(item.get(field, 0)) + int(row.get(field, 0) or 0)

    def _append_usage_row_locked(self, row: dict[str, Any]) -> None:
        append_jsonl_safe(self.usage_path, row)

    def _write_usage_summary_locked(self) -> None:
        summary = dict(self._usage_summary)
        prompt_tokens = int(summary.get("prompt_tokens", 0) or 0)
        cached = int(summary.get("prompt_cache_hit_tokens", 0) or 0)
        missed = int(summary.get("prompt_cache_miss_tokens", 0) or 0)
        known_prompt_tokens = int(summary.get("provider_cache_known_prompt_tokens", 0) or 0)
        estimated_prompt_tokens = int(summary.get("estimated_prompt_tokens", 0) or 0)
        summary["provider_cache_hit_ratio"] = round(cached / prompt_tokens, 6) if prompt_tokens else 0.0
        summary["provider_cache_miss_ratio"] = round(missed / prompt_tokens, 6) if prompt_tokens else 0.0
        summary["known_provider_cache_hit_ratio"] = (
            round(cached / known_prompt_tokens, 6) if known_prompt_tokens else 0.0
        )
        summary["known_provider_cache_miss_ratio"] = (
            round(missed / known_prompt_tokens, 6) if known_prompt_tokens else 0.0
        )
        completion_tokens = int(summary.get("completion_tokens", 0) or 0)
        model_name = str(self.config.llm.model_name or "")
        if _is_deepseek_model(model_name):
            summary["deepseek_pricing_rmb_per_1m"] = {
                "cache_hit_input": DEEPSEEK_RMB_PER_1M_CACHE_HIT_INPUT,
                "cache_miss_input": DEEPSEEK_RMB_PER_1M_CACHE_MISS_INPUT,
                "output": DEEPSEEK_RMB_PER_1M_OUTPUT,
            }
            summary["deepseek_cost_breakdown_rmb"] = _deepseek_cost_breakdown(
                prompt_tokens=prompt_tokens,
                cached_tokens=cached,
                miss_tokens=missed,
                completion_tokens=completion_tokens,
            )
            summary["estimated_deepseek_rmb_cache_known_only"] = round(
                _estimate_deepseek_rmb(
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached,
                    miss_tokens=missed,
                    completion_tokens=completion_tokens,
                    unknown_prompt_as_miss=False,
                ),
                6,
            )
            summary["estimated_deepseek_rmb_unknown_prompt_as_miss"] = round(
                _estimate_deepseek_rmb(
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached,
                    miss_tokens=missed,
                    completion_tokens=completion_tokens,
                    unknown_prompt_as_miss=True,
                ),
                6,
            )
        by_part = summary.get("by_prompt_part", {})
        if isinstance(by_part, dict):
            part_rows = sorted(
                by_part.values(),
                key=lambda x: int(x.get("estimated_tokens", 0) or 0),
                reverse=True,
            )
            for row in part_rows:
                est = int(row.get("estimated_tokens", 0) or 0)
                row["share_of_estimated_prompt"] = (
                    round(est / estimated_prompt_tokens, 6) if estimated_prompt_tokens else 0.0
                )
            summary["by_prompt_part_ranked"] = part_rows
        by_prompt = summary.get("by_prompt", {})
        if isinstance(by_prompt, dict):
            for prompt_item in by_prompt.values():
                if _is_deepseek_model(model_name):
                    prompt_prompt_tokens = int(prompt_item.get("prompt_tokens", 0) or 0)
                    prompt_cached = int(prompt_item.get("prompt_cache_hit_tokens", 0) or 0)
                    prompt_missed = int(prompt_item.get("prompt_cache_miss_tokens", 0) or 0)
                    prompt_completion = int(prompt_item.get("completion_tokens", 0) or 0)
                    prompt_item["deepseek_cost_breakdown_rmb"] = _deepseek_cost_breakdown(
                        prompt_tokens=prompt_prompt_tokens,
                        cached_tokens=prompt_cached,
                        miss_tokens=prompt_missed,
                        completion_tokens=prompt_completion,
                    )
                    prompt_item["estimated_deepseek_rmb_cache_known_only"] = round(
                        _estimate_deepseek_rmb(
                            prompt_tokens=prompt_prompt_tokens,
                            cached_tokens=prompt_cached,
                            miss_tokens=prompt_missed,
                            completion_tokens=prompt_completion,
                            unknown_prompt_as_miss=False,
                        ),
                        6,
                    )
                    prompt_item["estimated_deepseek_rmb_unknown_prompt_as_miss"] = round(
                        _estimate_deepseek_rmb(
                            prompt_tokens=prompt_prompt_tokens,
                            cached_tokens=prompt_cached,
                            miss_tokens=prompt_missed,
                            completion_tokens=prompt_completion,
                            unknown_prompt_as_miss=True,
                        ),
                        6,
                    )
                prompt_est = int(prompt_item.get("estimated_prompt_tokens", 0) or 0)
                prompt_parts = prompt_item.get("by_part", {})
                if not isinstance(prompt_parts, dict):
                    continue
                ranked = sorted(
                    prompt_parts.values(),
                    key=lambda x: int(x.get("estimated_tokens", 0) or 0),
                    reverse=True,
                )
                for row in ranked:
                    est = int(row.get("estimated_tokens", 0) or 0)
                    row["share_of_estimated_prompt"] = round(est / prompt_est, 6) if prompt_est else 0.0
                prompt_item["by_part_ranked"] = ranked
        write_json_safe(self.usage_summary_path, summary, indent=2)
        write_json_safe(self.usage_brief_path, self._build_usage_brief(summary), indent=2)

    def _build_usage_brief(self, summary: dict[str, Any]) -> dict[str, Any]:
        by_prompt = summary.get("by_prompt", {})
        prompt_rows = []
        if isinstance(by_prompt, dict):
            for name, item in by_prompt.items():
                if not isinstance(item, dict):
                    continue
                cost_breakdown = (
                    item.get("deepseek_cost_breakdown_rmb")
                    if isinstance(item.get("deepseek_cost_breakdown_rmb"), dict)
                    else {}
                )
                prompt_rows.append(
                    {
                        "prompt_name": name,
                        "stage": _prompt_stage(str(name)),
                        "calls": int(item.get("calls", 0) or 0),
                        "local_cache_hits": int(item.get("cache_hits_local", 0) or 0),
                        "seconds": round(float(item.get("seconds", 0.0) or 0.0), 4),
                        "input_tokens": int(item.get("prompt_tokens", 0) or 0),
                        "cache_hit_tokens": int(item.get("prompt_cache_hit_tokens", 0) or 0),
                        "cache_miss_tokens": int(item.get("prompt_cache_miss_tokens", 0) or 0),
                        "unknown_input_tokens": int(cost_breakdown.get("unknown_input_tokens", 0) or 0),
                        "output_tokens": int(item.get("completion_tokens", 0) or 0),
                        "deepseek_cost_breakdown_rmb": cost_breakdown,
                        "estimated_deepseek_rmb": item.get("estimated_deepseek_rmb_unknown_prompt_as_miss"),
                    }
                )
        prompt_rows.sort(
            key=lambda row: (
                float(row.get("estimated_deepseek_rmb") or 0.0),
                int(row.get("cache_miss_tokens", 0) or 0),
                int(row.get("output_tokens", 0) or 0),
            ),
            reverse=True,
        )
        stage_rows_by_name: dict[str, dict[str, Any]] = {}
        for row in prompt_rows:
            stage = str(row.get("stage") or "other")
            item = stage_rows_by_name.setdefault(
                stage,
                {
                    "stage": stage,
                    "calls": 0,
                    "local_cache_hits": 0,
                    "seconds": 0.0,
                    "input_tokens": 0,
                    "cache_hit_tokens": 0,
                    "cache_miss_tokens": 0,
                    "unknown_input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_deepseek_rmb": 0.0,
                },
            )
            item["calls"] = int(item.get("calls", 0)) + int(row.get("calls", 0) or 0)
            item["local_cache_hits"] = int(item.get("local_cache_hits", 0)) + int(row.get("local_cache_hits", 0) or 0)
            item["seconds"] = round(float(item.get("seconds", 0.0) or 0.0) + float(row.get("seconds", 0.0) or 0.0), 4)
            for key in [
                "input_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
                "unknown_input_tokens",
                "output_tokens",
            ]:
                item[key] = int(item.get(key, 0)) + int(row.get(key, 0) or 0)
            item["estimated_deepseek_rmb"] = round(
                float(item.get("estimated_deepseek_rmb", 0.0) or 0.0)
                + float(row.get("estimated_deepseek_rmb", 0.0) or 0.0),
                6,
            )
        stage_rows = sorted(
            stage_rows_by_name.values(),
            key=lambda row: (
                float(row.get("estimated_deepseek_rmb") or 0.0),
                int(row.get("cache_miss_tokens", 0) or 0),
                int(row.get("output_tokens", 0) or 0),
            ),
            reverse=True,
        )
        return {
            "schema_version": "autorealize.llm_usage_brief.v1",
            "model": self.config.llm.model_name,
            "calls": int(summary.get("calls", 0) or 0),
            "llm_seconds": round(float(summary.get("seconds", 0.0) or 0.0), 4),
            "input_tokens": int(summary.get("prompt_tokens", 0) or 0),
            "cache_hit_tokens": int(summary.get("prompt_cache_hit_tokens", 0) or 0),
            "cache_miss_tokens": int(summary.get("prompt_cache_miss_tokens", 0) or 0),
            "output_tokens": int(summary.get("completion_tokens", 0) or 0),
            "provider_cache_hit_ratio": summary.get("provider_cache_hit_ratio", 0.0),
            "provider_cache_miss_ratio": summary.get("provider_cache_miss_ratio", 0.0),
            "estimated_deepseek_rmb_cache_known_only": summary.get("estimated_deepseek_rmb_cache_known_only"),
            "estimated_deepseek_rmb_unknown_prompt_as_miss": summary.get("estimated_deepseek_rmb_unknown_prompt_as_miss"),
            "deepseek_cost_breakdown_rmb": summary.get("deepseek_cost_breakdown_rmb", {}),
            "deepseek_pricing_rmb_per_1m": summary.get("deepseek_pricing_rmb_per_1m", {}),
            "by_stage": stage_rows,
            "top_prompts_by_estimated_cost": prompt_rows[:20],
        }

    @staticmethod
    def _is_retryable_llm_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError)):
            return True
        name = exc.__class__.__name__.lower()
        msg = str(exc).lower()
        if any(
            key in name
            for key in [
                "timeout",
                "connection",
                "ratelimit",
                "internalserver",
                "apierror",
                "apiconnection",
                "badgateway",
                "serviceunavailable",
                "gateway",
                "httpstatus",
            ]
        ):
            return True
        if any(
            key in msg
            for key in [
                "timeout",
                "timed out",
                "connection reset",
                "connection aborted",
                "connection refused",
                "10061",
                "actively refused",
                "绉瀬鎷掔粷",
                "temporary failure",
                "temporarily unavailable",
                "bad gateway",
                "502",
                "503",
                "504",
                "rate limit",
                "too many requests",
                "gateway",
                "server disconnected",
                "remote protocol error",
                "getaddrinfo",
                "11001",
                "name resolution",
                "name or service not known",
            ]
        ):
            return True
        return False

    def _chat_completion_with_network_retry(
        self,
        *,
        prompt_name: str,
        mode: str,
        create_kwargs: dict,
        parse_attempt: int | None = None,
    ):
        last_exc: Exception | None = None
        active_kwargs = dict(create_kwargs)
        degraded_keys: set[str] = set()
        for net_attempt in range(1, NETWORK_RETRY_MAX_ATTEMPTS + 1):
            try:
                with self._request_gate:
                    return self.client.chat.completions.create(**active_kwargs)
            except Exception as exc:
                last_exc = exc
                degraded, degraded_key = _degraded_request_kwargs(active_kwargs, exc)
                if degraded is not None and degraded_key not in degraded_keys:
                    degraded_keys.add(degraded_key)
                    active_kwargs = degraded
                    logger.warning(
                        "[LLM] provider rejected optional parameter `%s`; retrying without it: prompt=%s mode=%s err=%s",
                        degraded_key,
                        prompt_name,
                        mode,
                        exc,
                    )
                    log_event(
                        logger,
                        "llm.client",
                        "REQUEST_DEGRADED",
                        prompt=prompt_name,
                        mode=mode,
                        attempt=parse_attempt,
                        dropped_parameter=degraded_key,
                        error=str(exc)[:180],
                    )
                    continue
                retryable = self._is_retryable_llm_error(exc)
                log_event(
                    logger,
                    "llm.client",
                    "REQUEST_FAILED",
                    prompt=prompt_name,
                    mode=mode,
                    attempt=parse_attempt,
                    network_attempt=net_attempt,
                    retryable=retryable,
                    error=str(exc)[:180],
                )
                if (not retryable) or net_attempt >= NETWORK_RETRY_MAX_ATTEMPTS:
                    raise
                sleep_secs = min(NETWORK_RETRY_MAX_SLEEP_SECONDS, 5.0 * net_attempt)
                logger.warning(
                    "[LLM] network error; retrying prompt=%s mode=%s parse_attempt=%s net_attempt=%s/%s sleep=%.1fs err=%s",
                    prompt_name,
                    mode,
                    parse_attempt if parse_attempt is not None else "-",
                    net_attempt,
                    NETWORK_RETRY_MAX_ATTEMPTS,
                    sleep_secs,
                    exc,
                )
                log_event(
                    logger,
                    "llm.client",
                    "REQUEST_RETRYING",
                    prompt=prompt_name,
                    mode=mode,
                    attempt=parse_attempt,
                    network_attempt=net_attempt,
                    sleep_seconds=f"{sleep_secs:.1f}",
                )
                time.sleep(sleep_secs)
        if last_exc is not None:
            raise last_exc

    def ask_text(
        self,
        system_prompt: str,
        user_prompt: str,
        prompt_name: str,
        *,
        static_user_prompt: str = "",
        dynamic_user_prompt: str | None = None,
    ) -> str:
        """Ask for free-form text.

        Callers may provide a stable user prefix and a dynamic user tail. This
        keeps large task/data context in a provider-cache-friendly position while
        preserving backwards compatibility with the old single user prompt.
        """
        if static_user_prompt or dynamic_user_prompt is not None:
            static_user = str(static_user_prompt or "").strip()
            dynamic_user = str(dynamic_user_prompt if dynamic_user_prompt is not None else user_prompt)
            base_user = "\n\n".join(x for x in [static_user, dynamic_user] if x.strip())
            messages = [{"role": "system", "content": system_prompt}]
            prompt_parts = [{"name": "system_prompt", "role": "system", "content": system_prompt}]
            if static_user:
                messages.append({"role": "user", "content": static_user})
                prompt_parts.append({"name": "static_user_prompt", "role": "user", "content": static_user})
            messages.append({"role": "user", "content": dynamic_user})
            prompt_parts.append({"name": "dynamic_user_prompt", "role": "user", "content": dynamic_user})
        else:
            base_user = user_prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_parts = [
                {"name": "system_prompt", "role": "system", "content": system_prompt},
                {"name": "user_prompt", "role": "user", "content": user_prompt},
            ]

        cache_key = self._cache_key(prompt_name=prompt_name, system_prompt=system_prompt, user_prompt=base_user)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info("[LLM] text cache hit: prompt=%s", prompt_name)
            log_event(logger, "llm.client", "CACHE_HIT", prompt=prompt_name, mode="text")
            self._log_local_cache_usage(prompt_name=prompt_name, mode="text", prompt_parts=prompt_parts)
            if self.config.llm.trace_cache_hits:
                self._log_trace(
                    LLMTrace(
                        prompt_name=prompt_name,
                        request=base_user[:10000],
                        response=cached[:12000],
                        parsed_ok=True,
                    )
                )
            return cached

        logger.info("[LLM] generating text: prompt=%s model=%s", prompt_name, self.config.llm.model_name)
        log_event(logger, "llm.client", "REQUEST_STARTED", prompt=prompt_name, mode="text", model=self.config.llm.model_name)
        t0 = time.perf_counter()
        text_max_tokens = _normal_max_tokens(self.config.llm.max_tokens)
        create_kwargs = {
            "model": self.config.llm.model_name,
            "temperature": self.config.llm.temperature,
            "messages": messages,
            "stream": False,
            "timeout": self.config.llm.request_timeout_seconds,
        }
        if text_max_tokens is not None:
            create_kwargs["max_tokens"] = text_max_tokens
        extra_body = _deepseek_extra_body(self.config, structured=False)
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        try:
            response = self._chat_completion_with_network_retry(
                prompt_name=prompt_name,
                mode="text",
                create_kwargs=create_kwargs,
            )
        except Exception as exc:
            raise
        text = response.choices[0].message.content or ""
        dt = time.perf_counter() - t0
        logger.info("[LLM] text completed: prompt=%s | %.2fs", prompt_name, dt)
        choice = response.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        usage_row = self._log_provider_usage(
            prompt_name=prompt_name,
            mode="text",
            response=response,
            seconds=dt,
            finish_reason=finish_reason,
            max_tokens=text_max_tokens,
            parsed_ok=True,
            prompt_parts=prompt_parts,
        )
        log_event(
            logger,
            "llm.client",
            "REQUEST_COMPLETED",
            prompt=prompt_name,
            mode="text",
            seconds=f"{dt:.2f}",
            finish_reason=finish_reason,
            prompt_tokens=usage_row.get("prompt_tokens", 0),
            completion_tokens=usage_row.get("completion_tokens", 0),
            total_tokens=usage_row.get("total_tokens", 0),
            cached_tokens=usage_row.get("prompt_cache_hit_tokens", 0),
            miss_tokens=usage_row.get("prompt_cache_miss_tokens", 0),
        )
        self._cache_put(cache_key, text)
        self._log_trace(
            LLMTrace(
                prompt_name=prompt_name,
                request=base_user[:10000],
                response=text[:12000],
                parsed_ok=True,
            )
        )
        return text

    def ask_structured(
        self,
        model_cls: type[T],
        system_prompt: str,
        user_prompt: str,
        prompt_name: str,
        fewshot: str = "",
        max_tokens: int | None = None,
        static_context_prompt: str = "",
        dynamic_user_prompt: str | None = None,
    ) -> T:
        schema = model_cls.model_json_schema()
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        schema_example = json.dumps(_example_from_schema(schema), ensure_ascii=False, indent=2)
        json_rules_prefix = (
            "Return one valid JSON object only. Do not output markdown fences or explanatory text.\n"
            "The word json is intentionally included for JSON Output mode."
        )
        json_rules_suffix = "Only output the JSON object. No markdown, no prose, no code fences."
        # Keep the schema and JSON rules as a stable prefix, then put run-specific
        # payload in the final user message. This mirrors Claude Code/Codex-style
        # prompt-cache hygiene while still preserving a deterministic local cache key.
        static_user = (
            f"{json_rules_prefix}\n"
            f"JSON example:\n{schema_example}\n\n"
            f"Required JSON Schema:\n{schema_text}\n"
            f"{json_rules_suffix}"
        )
        if fewshot:
            static_user = f"Few-shot examples:\n{fewshot}\n\n{static_user}"
        stable_context_user = ""
        if static_context_prompt:
            stable_context_user = f"Stable task/data context:\n{static_context_prompt}"
        dynamic_user = f"Dynamic input payload:\n{dynamic_user_prompt if dynamic_user_prompt is not None else user_prompt}"
        base_user = "\n\n".join(x for x in [static_user, stable_context_user, dynamic_user] if x.strip())
        output_max_tokens = (
            _normal_max_tokens(max_tokens)
            or _normal_max_tokens(getattr(self.config.llm, "structured_max_tokens", None))
            or _normal_max_tokens(self.config.llm.max_tokens)
        )

        cache_key = self._cache_key(
            prompt_name=prompt_name,
            system_prompt=system_prompt,
            user_prompt=base_user,
            schema=schema_text,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            try:
                parsed, repair_note = _validate_with_string_coercion(model_cls, _loads_json_object(cached))
                logger.info("[LLM] structured cache hit: prompt=%s", prompt_name)
                log_event(logger, "llm.client", "CACHE_HIT", prompt=prompt_name, mode="structured")
                cache_prompt_parts = [
                    {"name": "system_prompt", "role": "system", "content": system_prompt},
                ]
                if fewshot:
                    cache_prompt_parts.append({"name": "fewshot", "role": "user", "content": str(fewshot)})
                cache_prompt_parts.extend(
                    [
                        {"name": "json_rules", "role": "user", "content": f"{json_rules_prefix}\n{json_rules_suffix}"},
                        {"name": "json_example", "role": "user", "content": schema_example},
                        {"name": "json_schema", "role": "user", "content": schema_text},
                    ]
                )
                if stable_context_user:
                    cache_prompt_parts.append(
                        {"name": "stable_context", "role": "user", "content": stable_context_user}
                    )
                cache_prompt_parts.append({"name": "dynamic_payload", "role": "user", "content": dynamic_user})
                self._log_local_cache_usage(
                    prompt_name=prompt_name,
                    mode="structured",
                    prompt_parts=cache_prompt_parts,
                )
                if repair_note:
                    log_event(
                        logger,
                        "llm.client",
                        "PARSE_NORMALIZED",
                        prompt=prompt_name,
                        mode="structured",
                        source="cache",
                        note=repair_note[:240],
                    )
                if self.config.llm.trace_cache_hits:
                    self._log_trace(
                        LLMTrace(
                            prompt_name=prompt_name,
                            request=base_user[:12000],
                            response=cached[:12000],
                            parsed_ok=True,
                        )
                    )
                return parsed
            except Exception:
                pass

        last_error = ""
        for attempt in range(1, self.config.llm.max_retries + 1):
            logger.info(
                "[LLM] generating structured output: prompt=%s attempt=%s/%s model=%s",
                prompt_name,
                attempt,
                self.config.llm.max_retries,
                self.config.llm.model_name,
            )
            log_event(
                logger,
                "llm.client",
                "REQUEST_STARTED",
                prompt=prompt_name,
                mode="structured",
                attempt=attempt,
                model=self.config.llm.model_name,
            )
            t0 = time.perf_counter()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": static_user},
            ]
            prompt_parts = [
                {"name": "system_prompt", "role": "system", "content": system_prompt},
            ]
            if fewshot:
                prompt_parts.append({"name": "fewshot", "role": "user", "content": str(fewshot)})
            prompt_parts.extend(
                [
                    {"name": "json_rules", "role": "user", "content": f"{json_rules_prefix}\n{json_rules_suffix}"},
                    {"name": "json_example", "role": "user", "content": schema_example},
                    {"name": "json_schema", "role": "user", "content": schema_text},
                ]
            )
            if stable_context_user:
                messages.append({"role": "user", "content": stable_context_user})
                prompt_parts.append({"name": "stable_context", "role": "user", "content": stable_context_user})
            messages.append({"role": "user", "content": dynamic_user})
            prompt_parts.append({"name": "dynamic_payload", "role": "user", "content": dynamic_user})
            if last_error and self.config.llm.enforce_json_retry:
                retry_error_prompt = (
                    "The previous response was not a valid JSON object for the schema.\n"
                    f"Error:\n{last_error}\n"
                    "Return exactly one valid JSON object now. No markdown, no prose, no code fences."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": retry_error_prompt,
                    }
                )
                prompt_parts.append({"name": "retry_parse_error", "role": "user", "content": retry_error_prompt})
            create_kwargs = {
                "model": self.config.llm.model_name,
                "temperature": self.config.llm.temperature,
                "messages": messages,
                "stream": False,
                "timeout": self.config.llm.request_timeout_seconds,
                "response_format": {"type": "json_object"},
            }
            if output_max_tokens is not None:
                create_kwargs["max_tokens"] = output_max_tokens
            extra_body = _deepseek_extra_body(self.config, structured=True)
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            try:
                response = self._chat_completion_with_network_retry(
                    prompt_name=prompt_name,
                    mode="structured",
                    parse_attempt=attempt,
                    create_kwargs=create_kwargs,
                )
            except Exception as exc:
                raise
            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            try:
                json_obj = _loads_json_object(text)
                parsed, repair_note = _validate_with_string_coercion(model_cls, json_obj)
                dt = time.perf_counter() - t0
                logger.info("[LLM] structured output completed: prompt=%s attempt=%s | %.2fs", prompt_name, attempt, dt)
                usage_row = self._log_provider_usage(
                    prompt_name=prompt_name,
                    mode="structured",
                    response=response,
                    seconds=dt,
                    attempt=attempt,
                    finish_reason=finish_reason,
                    max_tokens=output_max_tokens,
                    parsed_ok=True,
                    prompt_parts=prompt_parts,
                )
                log_event(
                    logger,
                    "llm.client",
                    "REQUEST_COMPLETED",
                    prompt=prompt_name,
                    mode="structured",
                    attempt=attempt,
                    seconds=f"{dt:.2f}",
                    finish_reason=finish_reason,
                    max_tokens=output_max_tokens,
                    prompt_tokens=usage_row.get("prompt_tokens", 0),
                    completion_tokens=usage_row.get("completion_tokens", 0),
                    total_tokens=usage_row.get("total_tokens", 0),
                    cached_tokens=usage_row.get("prompt_cache_hit_tokens", 0),
                    miss_tokens=usage_row.get("prompt_cache_miss_tokens", 0),
                )
                if repair_note:
                    log_event(
                        logger,
                        "llm.client",
                        "PARSE_NORMALIZED",
                        prompt=prompt_name,
                        mode="structured",
                        attempt=attempt,
                        note=repair_note[:240],
                    )
                self._cache_put(cache_key, text)
                self._log_trace(
                    LLMTrace(
                        prompt_name=prompt_name,
                        request=base_user[:12000],
                        response=text[:12000],
                        parsed_ok=True,
                    )
                )
                return parsed
            except (ValueError, json.JSONDecodeError, ValidationError) as exc:
                dt = time.perf_counter() - t0
                usage_row = self._log_provider_usage(
                    prompt_name=prompt_name,
                    mode="structured",
                    response=response,
                    seconds=dt,
                    attempt=attempt,
                    finish_reason=finish_reason,
                    max_tokens=output_max_tokens,
                    parsed_ok=False,
                    prompt_parts=prompt_parts,
                )
                last_error = _format_parse_error(
                    exc,
                    finish_reason=finish_reason,
                    max_tokens=output_max_tokens,
                    text=text,
                )
                will_retry = attempt < self.config.llm.max_retries
                self._log_trace(
                    LLMTrace(
                        prompt_name=prompt_name,
                        request=base_user[:12000],
                        response=text[:12000],
                        parsed_ok=False,
                        error=last_error[:2000],
                    )
                )
                logger.warning("Structured output parse failed: attempt=%s, error=%s", attempt, last_error)
                log_event(
                    logger,
                    "llm.client",
                    "PARSE_RETRYING" if will_retry else "PARSE_FAILED",
                    prompt=prompt_name,
                    mode="structured",
                    attempt=attempt,
                    finish_reason=finish_reason,
                    max_tokens=output_max_tokens,
                    prompt_tokens=usage_row.get("prompt_tokens", 0),
                    completion_tokens=usage_row.get("completion_tokens", 0),
                    total_tokens=usage_row.get("total_tokens", 0),
                    cached_tokens=usage_row.get("prompt_cache_hit_tokens", 0),
                    miss_tokens=usage_row.get("prompt_cache_miss_tokens", 0),
                    error=last_error[:180],
                )
        raise RuntimeError(f"LLM structured output failed: {last_error}")
