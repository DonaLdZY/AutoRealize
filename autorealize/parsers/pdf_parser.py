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
        if total_pages <= 20:
            page_indexes = list(range(total_pages))
        else:
            # Long PDFs are sampled by section to keep prompt context bounded.
            page_indexes = sorted(set([*range(10), total_pages // 2, max(0, total_pages - 2), total_pages - 1]))

        snippets: list[str] = []
        for i in page_indexes:
            txt = (reader.pages[i].extract_text() or "").strip()
            if txt:
                snippets.append(f"[page {i + 1}]\n{txt[:2200]}")
        merged = "\n\n".join(snippets)
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=merged[:24000] if merged else "PDF 未提取到可用文本。",
            metadata={"pages": total_pages, "pages_sampled": [i + 1 for i in page_indexes], "chars": len(merged)},
        )
