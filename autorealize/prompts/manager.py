from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import AutoRealizeConfig


@dataclass
class PromptBlock:
    name: str
    priority: int
    content: str


class PromptManager:
    """Prompt 预算与块拼装。"""

    def __init__(self, config: AutoRealizeConfig) -> None:
        self.config = config
        self.root = Path(__file__).resolve().parent

    def load(self, rel_path: str) -> str:
        return (self.root / rel_path).read_text(encoding="utf-8")

    def estimate_tokens(self, text: str) -> int:
        # 粗略估算：中文/英文混合场景按 4 字符~1 token。
        return max(1, len(text) // 4)

    def build(self, blocks: list[PromptBlock]) -> str:
        sorted_blocks = sorted(blocks, key=lambda b: b.priority)
        result: list[str] = []
        used = 0
        budget = self.config.prompt.prompt_token_budget
        for block in sorted_blocks:
            token = self.estimate_tokens(block.content)
            if used + token > budget:
                continue
            result.append(block.content)
            used += token
        return "\n\n".join(result)

    def project_columns(self, columns: list[str], objective: str) -> list[str]:
        """宽表上下文优化：只保留更可能相关的列。"""
        if len(columns) <= self.config.prompt.wide_table_column_threshold:
            return columns
        keys = [k for k in objective.lower().replace("，", " ").replace(",", " ").split() if k]
        scored = []
        for col in columns:
            lc = col.lower()
            score = sum(1 for k in keys if k in lc)
            scored.append((score, col))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[: self.config.prompt.wide_table_column_threshold]]
        return top
