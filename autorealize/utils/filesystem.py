from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable


def safe_copytree(src: Path, dst: Path) -> None:
    """复制原始数据到工作副本。"""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def walk_files(root: Path) -> list[Path]:
    """跨平台递归列出文件。"""
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            files.append(Path(dirpath) / name)
    return sorted(files)


def walk_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for dirpath, _, _ in os.walk(root):
        dirs.append(Path(dirpath))
    return sorted(dirs)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def snapshot_files(paths: Iterable[Path], snapshot_dir: Path, root: Path) -> None:
    """保存文件快照（按相对路径存储）。"""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        target = snapshot_dir / rel(p, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def restore_snapshot(snapshot_dir: Path, root: Path) -> None:
    for p in snapshot_dir.rglob("*"):
        if p.is_file():
            rel_path = p.relative_to(snapshot_dir)
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
