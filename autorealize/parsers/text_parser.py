from __future__ import annotations

from pathlib import Path

from .base import BaseParser, ParsedFile


class TextParser(BaseParser):
    supported_suffixes = (".txt", ".md", ".rst", ".log")
    kind = "document"

    def __init__(self, encodings: tuple[str, ...]) -> None:
        self.encodings = encodings

    def parse(self, path: Path) -> ParsedFile:
        text = ""
        for enc in self.encodings:
            try:
                text = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if not text:
            text = "文件可读性较差或为空。"
        summary = text[:3000]
        return ParsedFile(path=path, kind=self.kind, text_summary=summary, metadata={"chars": len(text)})
