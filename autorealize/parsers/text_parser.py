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
        encoding_used = ""
        for enc in self.encodings:
            try:
                text = path.read_text(encoding=enc)
                encoding_used = enc
                break
            except UnicodeDecodeError:
                continue
        if not text:
            text = "文件为空，或无法按候选编码可靠读取。"
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=text,
            metadata={"chars": len(text), "encoding": encoding_used, "full_text_extracted": True},
        )
