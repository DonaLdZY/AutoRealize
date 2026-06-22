from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from autorealize.parsers.archive_parser import ArchiveParser
from autorealize.utils.archives import extract_archive, is_archive_file, list_archive_members


def test_archive_parser_zip_and_list(tmp_path: Path) -> None:
    zpath = tmp_path / "demo.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.txt", "hello")
        zf.writestr("dir/b.csv", "x,y\n1,2\n")

    assert is_archive_file(zpath) is True
    listed = list_archive_members(zpath)
    assert listed.archive_type == "zip"
    assert listed.member_count == 2

    parsed = ArchiveParser().parse(zpath)
    assert parsed.kind == "archive"
    assert parsed.metadata["member_count"] == 2


def test_archive_extract_tar(tmp_path: Path) -> None:
    tpath = tmp_path / "demo.tar.gz"
    src = tmp_path / "src.txt"
    src.write_text("ok", encoding="utf-8")
    with tarfile.open(tpath, "w:gz") as tf:
        tf.add(src, arcname="nested/src.txt")

    out = tmp_path / "out"
    result = extract_archive(tpath, out, max_files=10)
    assert result.extracted_files == 1
    assert (out / "nested" / "src.txt").exists()
