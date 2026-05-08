from __future__ import annotations

import io
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


ARCHIVE_EXTENSIONS = (
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar",
    ".zip",
    ".rar",
)


@dataclass
class ArchiveListResult:
    archive_type: str
    members: list[str]
    member_count: int
    warning: str = ""


@dataclass
class ArchiveExtractResult:
    archive_type: str
    extracted_files: int
    target_dir: str
    warning: str = ""


def is_archive_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def archive_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".rar"):
        return "rar"
    if any(name.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        return "tar"
    return "unknown"


def archive_stem(path: Path) -> str:
    name = path.name
    low = name.lower()
    for ext in ARCHIVE_EXTENSIONS:
        if low.endswith(ext):
            return name[: len(name) - len(ext)]
    return path.stem


def list_archive_members(path: Path, limit: int = 300) -> ArchiveListResult:
    kind = archive_type(path)
    if kind == "zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n and not n.endswith("/")]
        return ArchiveListResult(archive_type=kind, members=names[:limit], member_count=len(names))

    if kind == "tar":
        with tarfile.open(path) as tf:
            names = [m.name for m in tf.getmembers() if m.isfile()]
        return ArchiveListResult(archive_type=kind, members=names[:limit], member_count=len(names))

    if kind == "rar":
        try:
            import rarfile  # type: ignore

            with rarfile.RarFile(path) as rf:
                names = [n for n in rf.namelist() if n and not n.endswith("/")]
            return ArchiveListResult(archive_type=kind, members=names[:limit], member_count=len(names))
        except Exception as exc:  # noqa: BLE001
            return ArchiveListResult(
                archive_type=kind,
                members=[],
                member_count=0,
                warning=f"RAR 列表读取失败: {exc}",
            )

    return ArchiveListResult(archive_type="unknown", members=[], member_count=0, warning="未知压缩格式")


def extract_archive(path: Path, target_dir: Path, max_files: int = 50000) -> ArchiveExtractResult:
    kind = archive_type(path)
    target_dir.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        count, truncated = _safe_extract_zip(path, target_dir, max_files=max_files)
        warning = "已达到解压文件上限，后续文件未展开" if truncated else ""
        return ArchiveExtractResult(archive_type=kind, extracted_files=count, target_dir=str(target_dir), warning=warning)
    if kind == "tar":
        count, truncated = _safe_extract_tar(path, target_dir, max_files=max_files)
        warning = "已达到解压文件上限，后续文件未展开" if truncated else ""
        return ArchiveExtractResult(archive_type=kind, extracted_files=count, target_dir=str(target_dir), warning=warning)
    if kind == "rar":
        try:
            import rarfile  # type: ignore

            count, truncated = _safe_extract_rar(path, target_dir, rarfile, max_files=max_files)
            warning = "已达到解压文件上限，后续文件未展开" if truncated else ""
            return ArchiveExtractResult(archive_type=kind, extracted_files=count, target_dir=str(target_dir), warning=warning)
        except Exception as exc:  # noqa: BLE001
            return ArchiveExtractResult(
                archive_type=kind,
                extracted_files=0,
                target_dir=str(target_dir),
                warning=f"RAR 解压失败: {exc}",
            )
    return ArchiveExtractResult(archive_type="unknown", extracted_files=0, target_dir=str(target_dir), warning="未知压缩格式")


def _safe_extract_zip(path: Path, target_dir: Path, max_files: int = 50000) -> tuple[int, bool]:
    count = 0
    truncated = False
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if count >= max_files:
                truncated = True
                break
            dst = _safe_member_path(target_dir, info.filename)
            if dst is None:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, dst.open("wb") as out:
                out.write(src.read())
            count += 1
    return count, truncated


def _safe_extract_tar(path: Path, target_dir: Path, max_files: int = 50000) -> tuple[int, bool]:
    count = 0
    truncated = False
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if count >= max_files:
                truncated = True
                break
            dst = _safe_member_path(target_dir, member.name)
            if dst is None:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            with dst.open("wb") as out:
                out.write(data)
            count += 1
    return count, truncated


def _safe_extract_rar(path: Path, target_dir: Path, rarfile_mod, max_files: int = 50000) -> tuple[int, bool]:
    count = 0
    truncated = False
    with rarfile_mod.RarFile(path) as rf:
        for name in rf.namelist():
            if not name or name.endswith("/"):
                continue
            if count >= max_files:
                truncated = True
                break
            dst = _safe_member_path(target_dir, name)
            if dst is None:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = rf.read(name)
            with dst.open("wb") as out:
                out.write(data)
            count += 1
    return count, truncated


def _safe_member_path(target_dir: Path, member_name: str) -> Path | None:
    # 防止压缩包路径穿越：忽略绝对路径和上跳路径
    raw = member_name.replace("\\", "/").lstrip("/")
    if not raw or raw.startswith("../") or "/../" in raw:
        return None
    dst = target_dir / raw
    try:
        dst.resolve().relative_to(target_dir.resolve())
    except Exception:  # noqa: BLE001
        return None
    return dst
