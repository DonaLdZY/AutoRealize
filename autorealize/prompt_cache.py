"""Helpers for provider-side input-cache friendly prompt layouts.

The core rule is to keep long, reusable task/data context before small
round-specific feedback. Providers can only reuse cached prefixes when the
beginning of the request remains stable.
"""

from __future__ import annotations

import json
from typing import Any


STABLE_CONTEXT_TITLE = "Stable authoritative task/data context"
_STABLE_PRIORITY_KEYS = (
    "original_requirements_full",
    "schema_version",
    "task_hint",
    "authoritative_memory",
    "constraint_memory",
    "context_policy",
    "output_language_policy",
)


def estimate_text_tokens(text: str) -> int:
    """Cheap dependency-free estimate for mixed CJK and Latin prompt text."""

    value = str(text or "")
    if not value:
        return 0
    cjk = 0
    for char in value:
        code = ord(char)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
    return max(1, int(round(cjk + (len(value) - cjk) / 4)))


def json_block(
    title: str,
    payload: Any,
    *,
    limit: int | None = None,
    sort_keys: bool = True,
) -> str:
    """Render deterministic JSON under a stable title."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys, indent=2, default=str)
    if limit is not None and limit > 0 and len(text) > limit:
        original_chars = len(text)
        visible_budget = max(80, int(limit) - 240)
        head_chars = max(20, int(visible_budget * 0.35))
        tail_chars = max(20, visible_budget - head_chars)
        while True:
            envelope = {
                "_prompt_truncation": {
                    "truncated": True,
                    "original_chars": original_chars,
                    "policy": "保留结构化头部和更大比例的最新尾部；不得猜测中间省略内容。",
                },
                "visible_json_head": text[:head_chars],
                "visible_json_tail": text[-tail_chars:],
            }
            bounded = json.dumps(envelope, ensure_ascii=False, sort_keys=False, indent=2)
            if len(bounded) <= limit or (head_chars <= 20 and tail_chars <= 20):
                # Valid JSON is more important than a tiny limit overrun caused
                # by the fixed truncation metadata itself.
                text = bounded
                break
            overflow = len(bounded) - limit
            tail_reduction = min(max(1, int(overflow * 0.65)), max(0, tail_chars - 20))
            head_reduction = min(max(1, overflow - tail_reduction), max(0, head_chars - 20))
            tail_chars -= tail_reduction
            head_chars -= head_reduction
    return f"{title}\n{text}"


def text_block(title: str, text: Any, *, limit: int | None = None) -> str:
    """Render text under a stable title."""
    value = str(text or "")
    if limit is not None and limit > 0:
        value = value[:limit]
    return f"{title}\n{value}"


def join_blocks(*blocks: str) -> str:
    """Join non-empty prompt blocks without changing their internal content."""
    return "\n\n".join(block.strip() for block in blocks if str(block or "").strip())


def _stable_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    ordered: dict[str, Any] = {}
    for key in _STABLE_PRIORITY_KEYS:
        if key in payload:
            ordered[key] = payload[key]
    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def stable_dynamic_prompt(
    *,
    stable: Any = "",
    dynamic: Any = "",
    stable_title: str = "Stable context",
    dynamic_title: str = "Dynamic request",
    stable_limit: int | None = None,
    dynamic_limit: int | None = None,
) -> tuple[str, str]:
    """Return two user-message payloads: reusable prefix and changing tail."""
    stable_text = (
        json_block(
            STABLE_CONTEXT_TITLE,
            _stable_payload(stable),
            limit=stable_limit,
            sort_keys=False,
        )
        if isinstance(stable, (dict, list, tuple))
        else text_block(STABLE_CONTEXT_TITLE, stable, limit=stable_limit)
    )
    dynamic_text = (
        json_block(dynamic_title, dynamic, limit=dynamic_limit)
        if isinstance(dynamic, (dict, list, tuple))
        else text_block(dynamic_title, dynamic, limit=dynamic_limit)
    )
    return stable_text, dynamic_text
