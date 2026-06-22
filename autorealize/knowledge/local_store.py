from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .base import KnowledgeEntry, RetrievalResult
from ..utils.safe_json import dumps_json_safe, write_json_safe


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")


def make_entry_id(kind: str, source: str, text: str) -> str:
    raw = f"{kind}\n{source}\n{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


class LocalKnowledgeStore:
    """Append-only JSONL knowledge store with lightweight lexical retrieval.

    The class is intentionally simple: it is fast, deterministic, easy to inspect,
    and can be replaced later by a vector DB adapter with the same public methods.
    """

    def __init__(self, path: Path, *, max_entry_chars: int = 2400, boost_structured: bool = True) -> None:
        self.path = path
        self.max_entry_chars = max_entry_chars
        self.boost_structured = boost_structured
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, KnowledgeEntry] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                entry = KnowledgeEntry(**obj)
                self._entries[entry.entry_id] = entry
            except Exception:
                continue

    def add(self, entry: KnowledgeEntry) -> None:
        if len(entry.text) > self.max_entry_chars:
            entry.text = entry.text[: self.max_entry_chars] + "..."
        self._entries[entry.entry_id] = entry

    def add_many(self, entries: list[KnowledgeEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def search(self, query: str, *, top_k: int = 10, tags: list[str] | None = None) -> list[RetrievalResult]:
        q_tokens = tokenize(query)
        tag_filter = set(tags or [])
        results: list[RetrievalResult] = []
        for entry in self._entries.values():
            if tag_filter and not (tag_filter & set(entry.tags)):
                continue
            haystack = " ".join([
                entry.kind,
                entry.source,
                entry.text,
                " ".join(entry.entities),
                " ".join(entry.fields),
                " ".join(entry.constraints),
                " ".join(entry.tags),
            ])
            e_tokens = tokenize(haystack)
            overlap = q_tokens & e_tokens
            score = float(len(overlap))
            reasons = [f"token_overlap={len(overlap)}"]
            if self.boost_structured:
                if entry.kind in {"constraint", "metric", "field_glossary", "task_requirement"}:
                    score += 2.0
                    reasons.append("structured_boost")
                if "hard_constraint" in entry.tags:
                    score += 1.5
                    reasons.append("hard_constraint_boost")
                if "evaluation" in entry.tags:
                    score += 1.0
                    reasons.append("evaluation_boost")
            if score > 0 or not q_tokens:
                results.append(RetrievalResult(entry=entry, score=score, reasons=reasons))
        results.sort(key=lambda r: (-r.score, r.entry.source, r.entry.kind))
        return results[:top_k]

    def flush(self) -> None:
        lines = [dumps_json_safe(e.__dict__) for e in self._entries.values()]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def write_manifest(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "format": "autorealize.local_knowledge_store.v1",
            "entry_count": len(self._entries),
            "store_file": str(self.path.name),
            "kinds": {},
            "tags": {},
        }
        for entry in self._entries.values():
            payload["kinds"][entry.kind] = payload["kinds"].get(entry.kind, 0) + 1
            for tag in entry.tags:
                payload["tags"][tag] = payload["tags"].get(tag, 0) + 1
        write_json_safe(path, payload, indent=2)
