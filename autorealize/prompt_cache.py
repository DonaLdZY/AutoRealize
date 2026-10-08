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
    text = lossless_json(payload, sort_keys=sort_keys)
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
            bounded = json.dumps(envelope, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
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


def lossless_json(
    payload: Any, *, sort_keys: bool = True,
    canonical_matches: bool = False, min_shared_chars: int = 512,
) -> str:
    """Factor exact repeated values using backward JSON Pointers, without clipping."""
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"), default=str)

    plain = dumps(payload)
    # Avoid ambiguity with source objects that already use this marker.
    if len(plain) < 2048 or '"$prompt_ref"' in plain:
        return plain
    safe = json.loads(plain)
    seen: dict[str, str] = {}

    def visit(value: Any, pointer: str) -> Any:
        serialized = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if canonical_matches else dumps(value)
        )
        if len(serialized) >= min_shared_chars:
            previous = seen.get(serialized)
            if previous is not None:
                return {"$prompt_ref": previous}
            seen[serialized] = pointer
        if isinstance(value, dict):
            return {key: visit(item, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, pointer + "/" + str(index)) for index, item in enumerate(value)]
        return value

    factored = dumps({
        "_prompt_encoding": "Lossless input encoding: a sole $prompt_ref object aliases the complete earlier value at that JSON Pointer in this envelope. Resolve aliases when reading; all facts and constraints apply in every referenced location. This is not the requested output schema.",
        "data": visit(safe, "/data"),
    })
    return factored if len(factored) + 200 < len(plain) else plain


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
    if isinstance(payload, (list, tuple)):
        return [_canonical_payload(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    ordered: dict[str, Any] = {}
    for key in _STABLE_PRIORITY_KEYS:
        if key in payload:
            ordered[key] = _canonical_payload(payload[key])
    for key in sorted(payload, key=str):
        if key not in ordered:
            ordered[key] = _canonical_payload(payload[key])
    return ordered


def _canonical_payload(payload: Any) -> Any:
    """Normalize mapping order without reordering semantic arrays or identifiers."""
    if isinstance(payload, dict):
        return {key: _canonical_payload(payload[key]) for key in sorted(payload, key=str)}
    if isinstance(payload, (list, tuple)):
        return [_canonical_payload(item) for item in payload]
    return payload


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
