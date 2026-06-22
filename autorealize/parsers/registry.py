from __future__ import annotations

from pathlib import Path

from .base import BaseParser, ParsedFile


class ParserRegistry:
    """注册表模式：按后缀分发解析器。"""

    def __init__(self) -> None:
        self.parsers: list[BaseParser] = []

    def register(self, parser: BaseParser) -> None:
        self.parsers.append(parser)

    def parse(self, path: Path) -> ParsedFile:
        for parser in self.parsers:
            if parser.can_parse(path):
                return parser.parse(path)
        return ParsedFile(path=path, kind="binary", text_summary="未识别格式，跳过深度解析。")
