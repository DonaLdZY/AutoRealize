from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FileRole(str, Enum):
    task_requirement = "task_requirement"
    data_description = "data_description"
    raw_data_table = "raw_data_table"
    code_or_config = "code_or_config"
    image_or_media = "image_or_media"
    unknown = "unknown"


class CritiqueSeverity(str, Enum):
    none = "none"
    minor = "minor"
    major = "major"
    critical = "critical"


class FileSummary(BaseModel):
    path: str
    role: FileRole
    summary: str
    key_entities: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    column_semantics: dict[str, str] = Field(default_factory=dict)
    column_profiles: list[dict[str, Any]] = Field(default_factory=list)


class DirectorySummary(BaseModel):
    path: str
    summary: str
    dominant_types: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)


class DataContract(BaseModel):
    """模式契约。"""

    required_input_columns: list[str] = Field(default_factory=list)
    preserve_columns: list[str] = Field(default_factory=list)
    remove_columns: list[str] = Field(default_factory=list)
    add_columns: list[str] = Field(default_factory=list)
    row_count_rule: str = "any"  # any/same/less/greater
    value_constraints: dict[str, dict[str, Any]] = Field(default_factory=dict)
    post_conditions: list[str] = Field(default_factory=list)


class GroundAction(BaseModel):
    """Ground Agent 执行动作定义。"""

    target_file: str
    purpose: str
    # reader|profiler|transformer|repairer|validator|noop_keeper|...
    agent_type: str = Field(default="transformer")
    toolset: list[str] = Field(default_factory=list)
    action: str = Field(description="drop|keep|transform|analyze|noop")
    reason: str
    python_code: str = ""
    contract: DataContract


class CritiqueResult(BaseModel):
    level: str  # plan|expansion
    severity: CritiqueSeverity
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class PlanPhase(BaseModel):
    phase_id: str
    title: str
    objective: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    substeps: list[str] = Field(default_factory=list)


class PipelinePlan(BaseModel):
    task_type: str
    objectives: list[str] = Field(default_factory=list)
    phases: list[PlanPhase] = Field(default_factory=list)
    evaluation_metric: str
    evaluation_formula: str
    submission_spec: str


class CheckerVerdict(BaseModel):
    passed: bool
    reason: str
    verify_script: str = ""


class MonitorVerdict(BaseModel):
    ok: bool
    severity: str
    alerts: list[str] = Field(default_factory=list)


class LLMTrace(BaseModel):
    prompt_name: str
    request: str
    response: str
    parsed_ok: bool
    error: str = ""


class AmbiguityReview(BaseModel):
    """低上下文反思检查输出。"""

    is_unambiguous: bool
    ambiguity_points: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)


class CognitionProbePlan(BaseModel):
    """数据认知阶段的低成本探查计划（P1）。"""

    need_more_probe: bool
    probe_actions: list[str] = Field(default_factory=list)
    # preview_head, profile_numeric, profile_categorical, check_nulls, check_inf, value_counts_topk
    focus_columns: list[str] = Field(default_factory=list)
    reason: str = ""


class CognitionSummary(BaseModel):
    """文件级认知结构化总结。"""

    file_role_guess: str
    concise_summary: str
    key_columns: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    related_hints: list[str] = Field(default_factory=list)


class SubmissionScriptPlan(BaseModel):
    """无官方样例时，生成 sample_submission 的脚本计划。"""

    purpose: str = ""
    submission_columns: list[str] = Field(default_factory=list)
    python_code: str
    id_column: str = "id"
    target_columns: list[str] = Field(default_factory=list)


class TaskClassification(BaseModel):
    """任务分型结果（用于分模板生成 description/submission 约束）。"""

    task_type: str = "regression"
    confidence: float = 0.0
    reasoning: str = ""
    primary_metric: str = ""
    metric_formula: str = ""
    submission_schema_hint: list[str] = Field(default_factory=list)
