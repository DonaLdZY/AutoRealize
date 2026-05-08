from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class RelationHint:
    left_file: str
    right_file: str
    shared_columns: list[str]
    reason: str


def detect_relations(
    file_columns: dict[str, list[str]],
    *,
    parallel: bool = False,
    max_workers: int = 4,
) -> list[RelationHint]:
    """根据同名字段做轻量关系发现。"""
    items = list(file_columns.items())
    pairs: list[tuple[tuple[str, list[str]], tuple[str, list[str]]]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append((items[i], items[j]))

    def _work(pair: tuple[tuple[str, list[str]], tuple[str, list[str]]]) -> RelationHint | None:
        (lf, lcols), (rf, rcols) = pair
        shared = sorted(set(c.lower() for c in lcols) & set(c.lower() for c in rcols))
        if not shared:
            return None
        return RelationHint(
            left_file=lf,
            right_file=rf,
            shared_columns=shared[:12],
            reason="字段名存在交集，可能可 join 或用于一致性校验。",
        )

    hints: list[RelationHint] = []
    if parallel and len(pairs) > 4:
        workers = max(1, int(max_workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_work, p) for p in pairs]
            for fut in as_completed(futures):
                hint = fut.result()
                if hint is not None:
                    hints.append(hint)
    else:
        for p in pairs:
            hint = _work(p)
            if hint is not None:
                hints.append(hint)
    return hints
