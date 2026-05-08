from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils.filesystem import restore_snapshot, snapshot_files


@dataclass
class SnapshotRecord:
    name: str
    path: Path


class SnapshotManager:
    """清洗动作前后快照管理。"""

    def __init__(self, workspace_root: Path, snapshot_root: Path) -> None:
        self.workspace_root = workspace_root
        self.snapshot_root = snapshot_root
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str, files: list[Path]) -> SnapshotRecord:
        path = self.snapshot_root / name
        snapshot_files(files, path, self.workspace_root)
        return SnapshotRecord(name=name, path=path)

    def rollback(self, snap: SnapshotRecord) -> None:
        restore_snapshot(snap.path, self.workspace_root)
