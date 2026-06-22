from __future__ import annotations

from pathlib import Path

from docx import Document

from .base import BaseParser, ParsedFile


class DocxParser(BaseParser):
    supported_suffixes = (".docx",)
    kind = "document"

    def parse(self, path: Path) -> ParsedFile:
        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=text[:24000] if text else "空文档，或仅包含图片/形状等非文本元素。",
            metadata={"paragraph_count": len(paragraphs)},
        )
