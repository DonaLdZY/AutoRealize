from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .base import BaseParser, ParsedFile


class PdfParser(BaseParser):
    supported_suffixes = (".pdf",)
    kind = "document"

    def parse(self, path: Path) -> ParsedFile:
        reader = PdfReader(str(path))
        snippets = []
        max_pages = min(8, len(reader.pages))
        for i in range(max_pages):
            txt = (reader.pages[i].extract_text() or "").strip()
            if txt:
                snippets.append(txt[:1200])
        merged = "\n\n".join(snippets)
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=merged[:6000] if merged else "PDF 可解析文本为空。",
            metadata={"pages": len(reader.pages)},
        )
