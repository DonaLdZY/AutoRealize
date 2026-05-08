from __future__ import annotations

from pathlib import Path

from ..utils.archives import ARCHIVE_EXTENSIONS, list_archive_members
from .base import BaseParser, ParsedFile


class ArchiveParser(BaseParser):
    supported_suffixes = tuple({Path("x" + ext).suffix.lower() for ext in ARCHIVE_EXTENSIONS})
    kind = "archive"

    def can_parse(self, path: Path) -> bool:
        low = path.name.lower()
        return any(low.endswith(ext) for ext in ARCHIVE_EXTENSIONS)

    def parse(self, path: Path) -> ParsedFile:
        listed = list_archive_members(path)
        head = listed.members[:80]
        summary = (
            f"压缩包类型: {listed.archive_type}; 文件数: {listed.member_count}; "
            f"示例: {', '.join(head[:10]) if head else '无'}"
        )
        if listed.warning:
            summary += f"; warning={listed.warning}"
        return ParsedFile(
            path=path,
            kind=self.kind,
            text_summary=summary,
            metadata={
                "archive_type": listed.archive_type,
                "member_count": listed.member_count,
                "members_preview": head,
                "warning": listed.warning,
            },
        )
