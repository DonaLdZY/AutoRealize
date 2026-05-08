from __future__ import annotations

from pathlib import Path

from PIL import Image

from .base import BaseParser, ParsedFile


class ImageParser(BaseParser):
    supported_suffixes = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")
    kind = "image"

    def parse(self, path: Path) -> ParsedFile:
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
            fmt = img.format
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=f"图片元数据: {fmt}, {w}x{h}, mode={mode}",
            metadata={"format": fmt, "width": w, "height": h, "mode": mode},
        )
