from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .base import BaseParser, ParsedFile


class PdfParser(BaseParser):
    supported_suffixes = (".pdf",)
    kind = "document"

    def parse(self, path: Path) -> ParsedFile:
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        snippets: list[str] = []
        pages_extracted: list[int] = []
        for i in range(total_pages):
            txt = (reader.pages[i].extract_text() or "").strip()
            if txt:
                pages_extracted.append(i + 1)
                snippets.append(f"[page {i + 1}]\n{txt}")
        merged = "\n\n".join(snippets)
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=merged if merged else "PDF 未提取到可用文本。",
            metadata={
                "pages": total_pages,
                "pages_extracted": pages_extracted,
                "full_text_extracted": len(pages_extracted) == total_pages,
                "requires_ocr": total_pages > 0 and not pages_extracted,
                "chars": len(merged),
            },
        )
