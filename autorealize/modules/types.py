from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..models import FileSummary
from ..profiling.relations import RelationHint


@dataclass
class RuntimeServices:
    """跨模块共享的运行服务。"""

    llm_client: object
    prompt_mgr: object
    registry: object
    trajectory: object
    knowledge_store: object | None = None


@dataclass
class DataCognitionResult:
    """数据认知模块产物。"""

    file_summaries: list[FileSummary] = field(default_factory=list)
    original_requirement_texts: list[str] = field(default_factory=list)
    table_columns: dict[str, list[str]] = field(default_factory=dict)
    relation_hints: list[RelationHint] = field(default_factory=list)
    constraint_memory: dict = field(default_factory=dict)
    authoritative_memory: dict = field(default_factory=dict)
    question_memory: dict = field(default_factory=dict)
    knowledge_base: dict = field(default_factory=dict)
    agent_context_pack: dict = field(default_factory=dict)
    data_description_path: Path | None = None


@dataclass
class TaskDefinitionResult:
    """任务定义模块产物。"""

    description_path: Path | None = None
    sample_submission_path: Path | None = None
    downstream_context: dict = field(default_factory=dict)
    plan: object | None = None
    defects: list[str] = field(default_factory=list)
