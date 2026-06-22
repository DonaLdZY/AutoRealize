from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class KnowledgeEntry:
    """A normalized knowledge item that can later be indexed by RAG/vector stores."""

    entry_id: str
    kind: str
    source: str
    text: str
    entities: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    entry: KnowledgeEntry
    score: float
    reasons: list[str] = field(default_factory=list)


class KnowledgeStore(Protocol):
    def add(self, entry: KnowledgeEntry) -> None:
        ...

    def add_many(self, entries: list[KnowledgeEntry]) -> None:
        ...

    def search(self, query: str, *, top_k: int = 10, tags: list[str] | None = None) -> list[RetrievalResult]:
        ...

    def flush(self) -> None:
        ...
