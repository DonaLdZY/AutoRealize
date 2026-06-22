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
    """Prompt loading, language prefixing, and lightweight block assembly."""

    def __init__(self, config: AutoRealizeConfig) -> None:
        self.config = config
        self.root = Path(__file__).resolve().parent

    def _language_prefix(self) -> str:
        lang = str(getattr(self.config.prompt, "output_language", "zh") or "zh").strip().lower()
        if lang in {"zh", "cn", "chinese", "中文"}:
            return (
                "输出语言要求：除代码、文件名、字段名、列名、指标名、正则表达式、JSON schema 字段名和必要 API 参数外，"
                "所有面向用户或写入最终文档的自然语言必须使用中文。不要中英文混写；如果原文是英文，也要用中文解释其含义。"
            )
        if lang in {"en", "english"}:
            return (
                "Output language requirement: except for code, file names, field names, column names, metric names, "
                "regular expressions, JSON schema keys, and necessary API parameters, all user-facing natural language "
                "must be written in English. Do not mix Chinese and English prose."
            )
        return ""

    def load(self, rel_path: str) -> str:
        text = (self.root / rel_path).read_text(encoding="utf-8-sig")
        if str(rel_path).replace("\\", "/").startswith("system/"):
            prefix = self._language_prefix()
            if prefix:
                return f"{prefix}\n\n{text}"
        return text

    def estimate_tokens(self, text: str) -> int:
        # Rough estimate for mixed Chinese/English prompts: about 4 chars per token.
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
        """For wide tables, keep the columns most likely related to the current objective."""
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
