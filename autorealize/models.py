from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints, field_validator

ShortReviewText = Annotated[str, StringConstraints(max_length=240)]


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
    detailed_report: str = ""
    key_entities: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    column_semantics: dict[str, str] = Field(default_factory=dict)
    column_semantic_meta: dict[str, dict[str, Any]] = Field(default_factory=dict)
    column_profiles: list[dict[str, Any]] = Field(default_factory=list)
    extracted_knowledge: list[str] = Field(default_factory=list)


class DirectorySummary(BaseModel):
    path: str
    summary: str
    dominant_types: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)


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


class RLFormulationCandidate(BaseModel):
    """Evidence-grounded way to turn a decision problem into an RL process."""

    formulation_type: str = "constructive_policy"
    state_candidates: list[str] = Field(default_factory=list)
    action_candidates: list[str] = Field(default_factory=list)
    transition_description: str = ""
    action_mask_basis: list[str] = Field(default_factory=list)
    reward_alignment: str = ""
    terminal_condition: str = ""
    episode_generation: str = ""
    generalization_target: str = "instance_specific"
    evidence_refs: list[str] = Field(default_factory=list)
    unresolved_points: list[str] = Field(default_factory=list)


class ProblemParadigmReview(BaseModel):
    """Problem-structure and method-policy routing result."""

    problem_paradigm: str = "unknown_but_executable"
    problem_structure: str = "unknown"
    decision_temporality: str = "not_applicable"
    environment_source: str = "not_applicable"
    confidence: float = 0.0
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    key_signals: list[str] = Field(default_factory=list)
    requires_sample_submission: bool = False
    output_contract_source: str = ""
    explicit_rl_requested: bool = False
    rl_as_required_paradigm: bool = False
    rl_required: bool = False
    required_method_families: list[str] = Field(default_factory=list)
    allowed_method_families: list[str] = Field(default_factory=list)
    recommended_solver_families: list[str] = Field(default_factory=list)
    method_routing_notes: list[str] = Field(default_factory=list)
    rl_formulation_candidates: list[RLFormulationCandidate] = Field(default_factory=list)


class DataAccessFileProtocol(BaseModel):
    """How one important input file should be read and interpreted."""

    path: str
    file_role: str = ""
    read_method: str = ""
    read_example: str = ""
    read_contract: dict[str, Any] = Field(default_factory=dict)
    row_grain: str = ""
    key_fields: list[str] = Field(default_factory=list)
    target_fields: list[str] = Field(default_factory=list)
    relation_keys: list[str] = Field(default_factory=list)
    important_fields: list[str] = Field(default_factory=list)
    parsing_notes: list[str] = Field(default_factory=list)


class DataAccessProtocol(BaseModel):
    """Authoritative data access hints for downstream AutoML/AutoRL agents."""

    files: list[DataAccessFileProtocol] = Field(default_factory=list)
    global_notes: list[str] = Field(default_factory=list)


class MLDLProtocol(BaseModel):
    """Prediction-style ML/DL task contract."""

    train_data: str = ""
    predict_data: str = ""
    prediction_unit: str = ""
    target: str = ""
    feature_boundary: list[str] = Field(default_factory=list)
    validation_design: str = ""
    leakage_guards: list[str] = Field(default_factory=list)


class OptimizationProtocol(BaseModel):
    """Static optimization / combinatorial decision task contract."""

    input_instance: str = ""
    decision_variables: list[str] = Field(default_factory=list)
    objective: str = ""
    hard_constraints: list[str] = Field(default_factory=list)
    soft_constraints: list[str] = Field(default_factory=list)
    feasibility_checks: list[str] = Field(default_factory=list)
    solution_representation: str = ""


class RLProtocol(BaseModel):
    """Sequential decision / reinforcement learning task contract."""

    environment: str = ""
    state: str = ""
    action: str = ""
    transition: str = ""
    reward: str = ""
    terminal_condition: str = ""
    policy_output: str = ""
    evaluation_episodes: str = ""
    illegal_action_handling: list[str] = Field(default_factory=list)


class HybridProtocol(BaseModel):
    """Hybrid prediction + optimization task contract."""

    prediction_subproblem: str = ""
    decision_subproblem: str = ""
    handoff: str = ""
    final_objective: str = ""
    validation_design: str = ""


class OutputProtocol(BaseModel):
    """Final output/submission protocol. It may be a table, a solution, or a policy."""

    output_kind: str = "submission_table"
    output_filename: str = "submission.csv"
    sample_submission_required: bool = False
    columns: list[str] = Field(default_factory=list)
    row_unit: str = ""
    format_rules: list[str] = Field(default_factory=list)
    no_sample_submission_reason: str = ""


class DescriptionSectionDraft(BaseModel):
    """One frozen reader-facing description section."""

    section_id: str = ""
    markdown: str = ""
    facts_used: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)


class OverviewTaskDefinitionDraft(BaseModel):
    """Frozen overview and task-definition sections generated together."""

    overview_markdown: str = ""
    task_definition_markdown: str = ""
    facts_used: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)


class SampleSubmissionSpec(BaseModel):
    """Machine-readable contract used by the sample_submission builder."""

    should_generate: bool = False
    source: str = ""  # official_sample/generated_spec/not_required/unknown
    output_filename: str = "submission.csv"
    sample_filename: str = "sample_submission.csv"
    columns: list[str] = Field(default_factory=list)
    column_meanings: dict[str, str] = Field(default_factory=dict)
    row_unit: str = ""
    row_count_rule: str = ""
    source_fields: dict[str, str] = Field(default_factory=dict)
    default_values: dict[str, Any] = Field(default_factory=dict)
    format_rules: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    no_sample_submission_reason: str = ""


class OutputSectionDraft(BaseModel):
    """Reader-facing output section plus the machine-readable sample spec."""

    markdown: str = ""
    sample_submission_spec: SampleSubmissionSpec = Field(default_factory=SampleSubmissionSpec)
    facts_used: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)


class SampleSubmissionValidationResult(BaseModel):
    """LLM validator result for a generated sample_submission candidate."""

    passed: bool = False
    issues: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)
    needs_regenerate: bool = False
    revised_python_code: str = ""
    rationale: str = ""


class DescriptionProtocolBundle(BaseModel):
    """Structured source of truth used to render the final Kaggle-style description."""

    problem_paradigm: str = "unknown_but_executable"
    problem_structure: str = "unknown"
    decision_temporality: str = "not_applicable"
    environment_source: str = "not_applicable"
    required_method_families: list[str] = Field(default_factory=list)
    allowed_method_families: list[str] = Field(default_factory=list)
    rl_formulation_candidates: list[RLFormulationCandidate] = Field(default_factory=list)
    overview: str = ""
    task_goal: str = ""
    data_access: DataAccessProtocol = Field(default_factory=DataAccessProtocol)
    ml_dl: MLDLProtocol = Field(default_factory=MLDLProtocol)
    optimization: OptimizationProtocol = Field(default_factory=OptimizationProtocol)
    rl: RLProtocol = Field(default_factory=RLProtocol)
    hybrid: HybridProtocol = Field(default_factory=HybridProtocol)
    output: OutputProtocol = Field(default_factory=OutputProtocol)
    evaluation_summary: str = ""
    constraints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("ml_dl", "optimization", "rl", "hybrid", mode="before")
    @classmethod
    def normalize_null_protocol_branches(cls, value: Any) -> Any:
        """Treat an LLM's null inactive branch like an omitted default branch."""

        return {} if value is None else value


class DescriptionTaskProtocolDraft(BaseModel):
    """LLM-authored task/evaluation/output protocol without deterministic file access details.

    Data access is generated by code from parser metadata and merged later. Keeping
    it out of the LLM schema prevents huge JSON outputs for wide/multi-file tasks.
    """

    problem_paradigm: str = "unknown_but_executable"
    problem_structure: str = "unknown"
    decision_temporality: str = "not_applicable"
    environment_source: str = "not_applicable"
    required_method_families: list[str] = Field(default_factory=list)
    allowed_method_families: list[str] = Field(default_factory=list)
    rl_formulation_candidates: list[RLFormulationCandidate] = Field(default_factory=list)
    overview: str = ""
    task_goal: str = ""
    ml_dl: MLDLProtocol = Field(default_factory=MLDLProtocol)
    optimization: OptimizationProtocol = Field(default_factory=OptimizationProtocol)
    rl: RLProtocol = Field(default_factory=RLProtocol)
    hybrid: HybridProtocol = Field(default_factory=HybridProtocol)
    output: OutputProtocol = Field(default_factory=OutputProtocol)
    evaluation_summary: str = ""
    constraints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("ml_dl", "optimization", "rl", "hybrid", mode="before")
    @classmethod
    def normalize_null_protocol_branches(cls, value: Any) -> Any:
        """Treat an LLM's null inactive branch like an omitted default branch."""

        return {} if value is None else value


class InvestigationQuestion(BaseModel):
    """A blocking question that should be answered from data evidence."""

    question_id: str = ""
    question: str = ""
    category: str = ""  # data_access/join_key/output/evaluation/constraint/target/other
    why_blocking: str = ""
    candidate_files: list[str] = Field(default_factory=list)
    priority: str = "medium"  # high/medium/low
    parent_question_id: str = ""
    depth: int = 0


class ReadonlyPythonRequest(BaseModel):
    """Custom read-only Python analysis requested by the investigator."""

    question_id: str = ""
    goal: str = ""
    # Historical field name kept for schema/report compatibility; it now means
    # "why this script is needed".
    reason_builtins_insufficient: str = ""
    input_files: list[str] = Field(default_factory=list)
    focus_sheets: list[str] = Field(default_factory=list)
    focus_columns: list[str] = Field(default_factory=list)
    expected_output: str = ""
    python_code: str = ""


class InvestigationToolRequest(BaseModel):
    """One evidence-gathering request.

    Current production flow only supports `custom_readonly_python`; other tool
    names are legacy compatibility and are rejected by the executor.
    """

    request_id: str = ""
    question_id: str = ""
    tool_name: str = ""
    reason: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    custom_python: ReadonlyPythonRequest = Field(default_factory=ReadonlyPythonRequest)


class QuestionInvestigationPlan(BaseModel):
    """LLM plan for cross-file question-driven investigation."""

    ready_to_answer: bool = False
    planning_notes: str = ""
    routing_reasons: list[str] = Field(default_factory=list)
    recommended_max_questions: int = 0
    recommended_max_actions_per_question: int = 0
    questions: list[InvestigationQuestion] = Field(default_factory=list)
    script_requests: list[ReadonlyPythonRequest] = Field(default_factory=list)
    # Compatibility field. Planners should leave this empty and use
    # `script_requests` for all investigations.
    tool_requests: list[InvestigationToolRequest] = Field(default_factory=list)


class InvestigationStepResult(BaseModel):
    """Result of one sandboxed read-only investigation script."""

    request_id: str = ""
    question_id: str = ""
    tool_name: str = ""
    status: str = "completed"  # completed/failed/skipped
    reason: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    output_truncated: bool = False
    max_output_chars: int = 0
    original_output_chars: int = 0
    visible_output_chars: int = 0


class InvestigationAnswer(BaseModel):
    """Final answer for one blocking investigation question."""

    question_id: str = ""
    question: str = ""
    answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: str = "medium"  # high/medium/low
    remaining_uncertainty: str = ""
    downstream_notes: list[str] = Field(default_factory=list)


class QuestionInvestigationAnswerSet(BaseModel):
    """LLM-produced final investigation answers after seeing tool observations."""

    summary: str = ""
    answers: list[InvestigationAnswer] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    context_routing_notes: list[str] = Field(default_factory=list)


class QuestionInvestigationFollowupQuestion(BaseModel):
    """A bounded follow-up question created while resolving one QDI question."""

    question: str = ""
    reason: str = ""
    why_not_duplicate: str = ""
    candidate_files: list[str] = Field(default_factory=list)


class ContextRetrievalRequest(BaseModel):
    """Request a small local context excerpt by table/card id.

    This is the QDI equivalent of Headroom CCR retrieval: the stable prompt
    carries light card indexes and artifact ids, while detailed field/table
    evidence is only injected into the next turn when the model asks for it.
    """

    question_id: str = ""
    table_ids: list[str] = Field(default_factory=list)
    input_files: list[str] = Field(default_factory=list)
    focus_sheets: list[str] = Field(default_factory=list)
    focus_columns: list[str] = Field(default_factory=list)
    query: str = ""
    reason: str = ""


class DocumentSearchRequest(BaseModel):
    """Search full locally extracted document chunks without an LLM call."""

    question_id: str = ""
    query: str = ""
    document_ids: list[str] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)
    top_k: int = 5
    reason: str = ""


class DocumentChunkReadRequest(BaseModel):
    """Read exact chunks and optional neighbors from the local document store."""

    question_id: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    neighbor_count: int = 1
    reason: str = ""


class QDIArtifactExcerptRequest(BaseModel):
    """Read a bounded excerpt from an artifact created by the current QDI run."""

    question_id: str = ""
    artifact_id: str = ""
    offset: int = 0
    max_chars: int = 8000
    json_path: str = ""
    reason: str = ""


class QDIWorkingMemoryUpdate(BaseModel):
    """Evidence-linked incremental memory emitted by an existing QDI action call."""

    confirmed_facts: list[str] = Field(default_factory=list)
    temporary_conclusions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)
    invalidated_hypotheses: list[str] = Field(default_factory=list)
    recommended_next_focus: str = ""


class QDIActionDigest(BaseModel):
    """LLM-authored digest of a previously executed evidence-gathering action."""

    sequence: int = 0
    action: str = ""
    what_was_done: str = ""
    key_outputs: list[str] = Field(default_factory=list)
    temporary_conclusion: str = ""
    remaining_gap: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class QuestionInvestigationAction(BaseModel):
    """Tool-call style action for the single-question QDI loop."""

    action: str = "give_up"  # answer/request_script/request_context/search_document/read_document_chunks/read_qdi_artifact_excerpt/...
    question_id: str = ""
    answer: str = ""
    confidence: str = "medium"
    used_files: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    downstream_notes: list[str] = Field(default_factory=list)
    remaining_uncertainty: str = ""
    unresolved_reason: str = ""
    what_was_tried: list[str] = Field(default_factory=list)
    duplicate_of_question_id: str = ""
    refined_question: str = ""
    request_script: ReadonlyPythonRequest = Field(default_factory=ReadonlyPythonRequest)
    request_context: ContextRetrievalRequest = Field(default_factory=ContextRetrievalRequest)
    search_document: DocumentSearchRequest = Field(default_factory=DocumentSearchRequest)
    read_document_chunks: DocumentChunkReadRequest = Field(default_factory=DocumentChunkReadRequest)
    read_qdi_artifact_excerpt: QDIArtifactExcerptRequest = Field(default_factory=QDIArtifactExcerptRequest)
    working_memory_update: QDIWorkingMemoryUpdate = Field(default_factory=QDIWorkingMemoryUpdate)
    action_digest_updates: list[QDIActionDigest] = Field(default_factory=list)
    followup_questions: list[QuestionInvestigationFollowupQuestion] = Field(default_factory=list)
    notes: str = ""


class QuestionInvestigationReport(BaseModel):
    """Persisted investigation report for downstream context management."""

    schema_version: str = "autorealize.question_investigation.v1"
    enabled: bool = True
    summary: str = ""
    questions: list[InvestigationQuestion] = Field(default_factory=list)
    script_requests: list[ReadonlyPythonRequest] = Field(default_factory=list)
    # Compatibility mirror of executed script requests.
    tool_requests: list[InvestigationToolRequest] = Field(default_factory=list)
    step_results: list[InvestigationStepResult] = Field(default_factory=list)
    answers: list[InvestigationAnswer] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    context_routing_notes: list[str] = Field(default_factory=list)
    question_records: list[dict[str, Any]] = Field(default_factory=list)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    action_digest_cards: list[dict[str, Any]] = Field(default_factory=list)
    working_memory_cards: list[dict[str, Any]] = Field(default_factory=list)
    # Live progress is persisted throughout QDI so the frontend does not have to
    # wait for the final report before showing useful investigation state.
    progress: dict[str, Any] = Field(default_factory=dict)


class LLMTrace(BaseModel):
    prompt_name: str
    request: str
    response: str
    parsed_ok: bool
    error: str = ""


class AmbiguityReview(BaseModel):
    """低上下文反思检查输出。"""

    is_unambiguous: bool
    ambiguity_points: list[ShortReviewText] = Field(default_factory=list, max_length=6)
    fixes: list[ShortReviewText] = Field(default_factory=list, max_length=6)


class EvaluationContractReview(BaseModel):
    """Structured contract produced by the evaluation rigor agent."""

    passed: bool = False
    executable: bool = False
    authority_status: str = "unresolved"  # official/autorealize_assumption/unresolved
    primary_metric: str = ""
    metric_direction: str = ""  # minimize/maximize
    metric_formula: str = ""
    scalar_score_formula: str = ""
    prediction_unit: str = ""
    y_true_source: str = ""
    y_pred_source: str = ""
    computation_scope: str = ""
    aggregation_rule: str = ""
    validation_protocol: str = ""
    submission_checks: list[str] = Field(default_factory=list)
    leakage_guards: list[str] = Field(default_factory=list)
    invalid_solution_rules: list[str] = Field(default_factory=list)
    tie_break_rules: list[str] = Field(default_factory=list)
    audit_metrics: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    residual_issues: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    rationale: str = ""


class AuthorityEvidenceItem(BaseModel):
    """One source that may authoritatively define the task contract."""

    source_path: str = ""
    source_type: str = ""  # original_description/readme/requirement_doc/official_sample/user_hint
    priority: str = "medium"  # high/medium/low
    evidence: str = ""


class SubmissionContract(BaseModel):
    """Frozen or partially frozen submission/output contract from authoritative sources."""

    is_defined: bool = False
    is_authoritative: bool = False
    output_filename: str = "submission.csv"
    sample_filename: str = "sample_submission.csv"
    columns: list[str] = Field(default_factory=list)
    column_descriptions: dict[str, str] = Field(default_factory=dict)
    row_unit: str = ""
    row_count_rule: str = ""
    format_description: str = ""
    validation_rules: list[str] = Field(default_factory=list)
    source: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    unresolved_questions: list[str] = Field(default_factory=list)


class RequirementFactItem(BaseModel):
    """One routed document statement with conflict/supersession status."""

    statement: str = ""
    fact_type: str = "current_rule"  # current_rule/current_answer/question/future_roadmap/example
    status: str = "current"  # current/resolved/superseded/unresolved/not_current
    topics: list[str] = Field(default_factory=list)
    source: str = ""
    evidence: str = ""
    resolved_or_superseded_by: str = ""


class AuthoritativeTaskMemory(BaseModel):
    """High-priority task memory extracted from original/official/user task docs."""

    has_authoritative_sources: bool = False
    summary: str = ""
    task_goal: str = ""
    input_requirements: list[str] = Field(default_factory=list)
    output_requirements: list[str] = Field(default_factory=list)
    evaluation_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    leakage_guards: list[str] = Field(default_factory=list)
    submission_contract: SubmissionContract = Field(default_factory=SubmissionContract)
    evidence_items: list[AuthorityEvidenceItem] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)
    authority_conflicts: list[dict[str, str]] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    context_routing_notes: list[str] = Field(default_factory=list)
    requirement_fact_matrix: list[RequirementFactItem] = Field(default_factory=list)


class CognitionProbePlan(BaseModel):
    """数据认知阶段的低成本探查计划（P1）。"""

    need_more_probe: bool
    probe_actions: list[str] = Field(default_factory=list)
    # preview_head, profile_numeric, profile_categorical, check_nulls, check_inf, value_counts_topk
    action_specs: list[dict[str, Any]] = Field(default_factory=list)
    focus_columns: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    reason: str = ""


class CognitionSummary(BaseModel):
    """文件级认知结构化总结。"""

    file_role_guess: str
    concise_summary: str
    detailed_report: str = ""
    key_columns: list[str] = Field(default_factory=list)
    field_descriptions: dict[str, str] = Field(default_factory=dict)
    sheet_field_descriptions: dict[str, dict[str, str]] = Field(default_factory=dict)
    key_facts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    related_hints: list[str] = Field(default_factory=list)


class DocumentCognitionMemory(BaseModel):
    """Bounded rolling memory while first-stage cognition reads a long document."""

    processed_chunks: int = 0
    total_chunks: int = 0
    covered_chars: int = 0
    concise_purpose: str = ""
    task_goals: list[str] = Field(default_factory=list)
    business_context: list[str] = Field(default_factory=list)
    data_objects_and_fields: list[str] = Field(default_factory=list)
    constraints_and_rules: list[str] = Field(default_factory=list)
    formulas_thresholds_units: list[str] = Field(default_factory=list)
    evaluation_and_submission: list[str] = Field(default_factory=list)
    methods_and_workflows: list[str] = Field(default_factory=list)
    risks_conflicts_unknowns: list[str] = Field(default_factory=list)
    source_anchors: list[str] = Field(default_factory=list)


class FileGroupingRegexCandidate(BaseModel):
    """LLM-proposed filename grouping pattern for safe sampling."""

    name: str = ""
    regex: str
    sample_id_group: str = "sample_id"
    data_kind_group: str = "data_kind"
    applies_to_suffixes: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0


class FileGroupingRegexPlan(BaseModel):
    """Candidate regexes proposed by LLM; code validates before use."""

    candidates: list[FileGroupingRegexCandidate] = Field(default_factory=list)
    notes: str = ""


class FileSamplingReviewItem(BaseModel):
    """LLM review result for one concrete filename sampling plan."""

    pattern_id: str
    accept_sampling: bool = True
    force_full_read: bool = False
    extra_sample_files: list[str] = Field(default_factory=list)
    rewrite_regex: str = ""
    rewrite_sample_id_group: str = "sample_id"
    rewrite_data_kind_group: str = "data_kind"
    rewrite_applies_to_suffixes: list[str] = Field(default_factory=list)
    reason: str = ""
    risk: str = ""


class FileSamplingReview(BaseModel):
    """LLM review of actual will-read / will-skip lists after regex matching."""

    items: list[FileSamplingReviewItem] = Field(default_factory=list)
    notes: str = ""


class SubmissionScriptPlan(BaseModel):
    """无官方样例时，生成 sample_submission 的脚本计划。"""

    purpose: str = ""
    submission_columns: list[str] = Field(default_factory=list)
    python_code: str
    id_column: str = "id"
    target_columns: list[str] = Field(default_factory=list)


class SubmissionCheckVerdict(BaseModel):
    """sample_submission 生成后检查结果。"""

    passed: bool = False
    reason: str = ""
    issues: list[str] = Field(default_factory=list)
    needs_regenerate: bool = False
    revised_submission_columns: list[str] = Field(default_factory=list)
    revised_python_code: str = ""


class ConstraintItem(BaseModel):
    """结构化约束条目。"""

    name: str
    description: str
    evidence: list[str] = Field(default_factory=list)
    related_fields: list[str] = Field(default_factory=list)
    priority: str = "medium"  # high/medium/low
    status: str = "current_candidate"  # current_candidate/superseded/not_current/unresolved
    status_reason: str = ""


class EntityAliasFieldCandidate(BaseModel):
    """One exact physical field referenced by an LLM alias hypothesis."""

    source_file: str = ""
    sheet_name: str = ""
    field: str = ""
    semantic_role: str = ""
    value_kind: str = "unknown"  # code/name/id/category/unknown
    evidence: str = ""


class EntityAliasCandidateGroup(BaseModel):
    """Cross-source semantic alias hypothesis; never authoritative by itself."""

    concept_id: str = ""
    label: str = ""
    reason: str = ""
    confidence: str = "low"
    task_relevance: str = "medium"  # high/medium/low
    relevance_reason: str = ""
    candidate_fields: list[EntityAliasFieldCandidate] = Field(default_factory=list)
    caution: str = ""


class ConstraintMemory(BaseModel):
    """跨阶段复用的约束记忆。"""

    summary: str = ""
    items: list[ConstraintItem] = Field(default_factory=list)
    entity_alias_candidates: list[EntityAliasCandidateGroup] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class TaskClassification(BaseModel):
    """任务分型结果。不要在此阶段决定提交文件 schema。"""

    task_type: str = "regression"
    confidence: float = 0.0
    reasoning: str = ""
    primary_metric: str = ""
    metric_formula: str = ""


class AutoMLContextPack(BaseModel):
    """Compact supplemental facts for downstream fixed context."""

    schema_version: str = "autorealize.automl_context_pack.v1"
    purpose: str = "Concise facts that supplement description.md without prescribing an AutoML strategy."
    priority_rules: list[str] = Field(default_factory=list)
    problem_paradigm: str = "unknown_but_executable"
    problem_structure: str = "unknown"
    decision_temporality: str = "not_applicable"
    environment_source: str = "not_applicable"
    task_goal: str = ""
    data_orchestration: list[str] = Field(default_factory=list)
    data_access: list[dict[str, Any]] = Field(default_factory=list)
    data_schema_contract: dict[str, Any] = Field(default_factory=dict)
    source_alias_guard: list[dict[str, Any]] = Field(default_factory=list)
    entity_alias_candidates: list[dict[str, Any]] = Field(default_factory=list)
    source_coverage_ledger: dict[str, Any] = Field(default_factory=dict)
    relation_verification_queue: list[dict[str, Any]] = Field(default_factory=list)
    requirement_fact_matrix: list[dict[str, Any]] = Field(default_factory=list)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    evaluation_contract: dict[str, Any] = Field(default_factory=dict)
    method_strategy: dict[str, Any] = Field(default_factory=dict)
    relation_cards: list[dict[str, Any]] = Field(default_factory=list)
    filename_sample_groups: list[dict[str, Any]] = Field(default_factory=list)
    context_shape: dict[str, Any] = Field(default_factory=dict)
    modeling_boundary: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    leakage_guards: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    source_artifacts: dict[str, str] = Field(default_factory=dict)


class CrossStageMemorySummary(BaseModel):
    """Lossy prompt memory whose source evidence remains artifact-backed."""

    schema_version: str = "autorealize.cross_stage_memory.v1"
    task_state: str = ""
    confirmed_facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    evidence_artifact_ids: list[str] = Field(default_factory=list)
    stage_summaries: list[str] = Field(default_factory=list)


class CrossStageRetrievalRequest(BaseModel):
    artifact_id: str = ""
    json_path: str = ""
    reason: str = ""
    max_chars: int = 4000


class CrossStageRetrievalPlan(BaseModel):
    needs_retrieval: bool = False
    reason: str = ""
    requests: list[CrossStageRetrievalRequest] = Field(default_factory=list)


class KnowledgeRerankItem(BaseModel):
    candidate_id: str = ""
    relevance: str = "irrelevant"
    score: float = 0.0
    reason: str = ""


class KnowledgeRerankDecision(BaseModel):
    items: list[KnowledgeRerankItem] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)


class DownstreamContextResolutionItem(BaseModel):
    """One bounded semantic choice from deterministic downstream candidates."""

    dimension: str = ""
    candidate_id: str = ""
    confidence: float = 0.0
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    remain_unresolved: bool = False


class DownstreamContextResolution(BaseModel):
    """LLM disambiguation result; candidate IDs are validated locally."""

    choices: list[DownstreamContextResolutionItem] = Field(default_factory=list)
    unresolved_dimensions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class SourceFieldAliasResolutionItem(BaseModel):
    output_column: str = ""
    alias: str = ""
    candidate_id: str = ""
    confidence: float = 0.0
    reason: str = ""
    remain_unresolved: bool = False


class SourceFieldAliasResolution(BaseModel):
    choices: list[SourceFieldAliasResolutionItem] = Field(default_factory=list)
    unresolved_aliases: list[str] = Field(default_factory=list)


class ArtifactConsistencyIssue(BaseModel):
    issue_id: str = ""
    severity: str = "warning"
    category: str = ""
    artifact: str = ""
    section: str = ""
    message: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    repair_target: str = "none"
    repair_instruction: str = ""


class ArtifactConsistencyReview(BaseModel):
    passed: bool = True
    summary: str = ""
    issues: list[ArtifactConsistencyIssue] = Field(default_factory=list)


class ArtifactConsistencyPatch(BaseModel):
    revised_sections: dict[str, str] = Field(default_factory=dict)
    addressed_issue_ids: list[str] = Field(default_factory=list)
    unresolved_issue_ids: list[str] = Field(default_factory=list)
