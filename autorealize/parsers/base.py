from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedFile:
    path: Path
    kind: str
    text_summary: str
    preview: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseParser:
    """解析器基类。"""

    supported_suffixes: tuple[str, ...] = ()
    kind: str = "unknown"

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_suffixes

    def parse(self, path: Path) -> ParsedFile:
        raise NotImplementedError
