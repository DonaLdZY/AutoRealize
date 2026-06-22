"""Helpers for provider-side input-cache friendly prompt layouts.

The core rule is to keep long, reusable task/data context before small
round-specific feedback. Providers can only reuse cached prefixes when the
beginning of the request remains stable.
"""

from __future__ import annotations

import json
from typing import Any


def json_block(title: str, payload: Any, *, limit: int | None = None) -> str:
    """Render deterministic JSON under a stable title."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
    if limit is not None and limit > 0:
        text = text[:limit]
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
        json_block(stable_title, stable, limit=stable_limit)
        if isinstance(stable, (dict, list, tuple))
        else text_block(stable_title, stable, limit=stable_limit)
    )
    dynamic_text = (
        json_block(dynamic_title, dynamic, limit=dynamic_limit)
        if isinstance(dynamic, (dict, list, tuple))
        else text_block(dynamic_title, dynamic, limit=dynamic_limit)
    )
    return stable_text, dynamic_text
