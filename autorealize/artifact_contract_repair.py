"""Bounded, evidence-linked repairs of derived output and target descriptions."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import DescriptionProtocolBundle, EvaluationContractReview, SampleSubmissionSpec


class ContractFieldRepair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    expected: Any
    replacement: Any
    issue_ids: list[str]
    evidence_refs: list[str]
    reason: str


class DerivedContractRepair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repairs: list[ContractFieldRepair] = Field(default_factory=list)
    unresolved_issue_ids: list[str] = Field(default_factory=list)


def normalize_explicit_time_series_contract(
    review: EvaluationContractReview,
    requirements: str,
) -> tuple[EvaluationContractReview, list[str]]:
    """Resolve directly stated chronological split and purge rules locally."""
    text = str(requirements or "")
    has_fixed_split = all(
        re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        for pattern in (
            r"(?:最早|前)\s*60\s*%",
            r"(?:中间|接续)\s*20\s*%",
            r"(?:最后|剩余|末尾)\s*20\s*%",
        )
    )
    has_one_hour_purge = bool(
        re.search(r"(?:一小时|1\s*(?:小时|h))[^。\n]{0,40}(?:purge|边界|清洗|剔除)", text, re.IGNORECASE)
        or re.search(r"(?:purge|边界|清洗|剔除)[^。\n]{0,40}(?:一小时|1\s*(?:小时|h))", text, re.IGNORECASE)
    )
    if not has_fixed_split or not has_one_hour_purge:
        return review, []
    exact_rounding = re.search(
        r"\b(?:floor|ceil|ceiling|round)\b|向下取整|向上取整|四舍五入|取整方式",
        text,
        re.IGNORECASE,
    )
    exact_purge_boundary = re.search(
        r"(?:\b[A-Za-z_]\w*\s*\+\s*1\s*(?:h|hour|小时)\s*(?:>=|<=|>|<|≥|≤)\s*"
        r"[A-Za-z_]\w*|\b[A-Za-z_]\w*\s*(?:>=|<=|>|<|≥|≤)\s*"
        r"[A-Za-z_]\w*\s*-\s*1\s*(?:h|hour|小时))"
        r"|(?:等号|等于).{0,20}(?:保留|剔除|包含|排除)|"
        r"(?:保留|剔除|包含|排除).{0,20}(?:等号|等于)",
        text,
        re.IGNORECASE,
    )
    # Exact user-authored rounding or boundary semantics outrank this
    # deterministic fallback. Do not rewrite them with the default convention.
    if exact_rounding or exact_purge_boundary:
        return review, []

    normalized = review.model_copy(deep=True)
    requires_base_coverage_screen = bool(re.search(
        r"(?:基础设备(?:历史)?覆盖(?:前置)?筛查|基础覆盖(?:前置)?筛查|N_eligible)",
        text,
        re.IGNORECASE,
    ))
    if requires_base_coverage_screen:
        split_rule = (
            "评估人口按 prediction_unit 与 y_true_source 定义并完成标签有效性处理后，先对每个标签样本执行"
            "基础设备历史覆盖前置筛查，记录 N_label、N_drop_base，并令 N_eligible=N_label-N_drop_base，"
            "仅以筛查后的 N_eligible 按样本时间升序切分，切分取整唯一固定为：训练段第 1 至 "
            "floor(0.6*N_eligible) 个样本，开发段接续取 floor(0.2*N_eligible) 个样本，"
            "留出段取其余全部样本（余数归入留出段），在两处边界 purge 后冻结各段人口，"
            "并报告筛查、切分与 purge 前后数量。"
        )
    else:
        split_rule = (
            "评估人口按 prediction_unit 与 y_true_source 定义并完成标签有效性处理后，"
            "按样本时间升序排列，记总样本数为 N，切分取整唯一固定为：训练段第 1 至 "
            "floor(0.6N) 个样本，开发段接续取 floor(0.2N) 个样本，留出段取其余全部样本"
            "（余数归入留出段），在两处边界 purge 后冻结各段人口，并报告切分与 purge 前后数量。"
        )
    purge_rule = (
        "一小时边界 purge：设后段最早被评分样本的标签时刻为 s_min，较早分段中"
        "标签时刻 s 满足 s + 1h > s_min 的样本剔除，满足 s + 1h <= s_min 的样本保留"
        "（等号保留），训练/开发与开发/留出边界分别执行并报告清洗前后及剔除数量。"
    )
    selection_rule = (
        f"候选只按冻结开发集上的 {normalized.primary_metric or 'primary_metric'} "
        f"（{normalized.metric_direction or 'direction'}）选择；最终留出指标在候选冻结后由引擎"
        "计算一次，不参与搜索、排序、权重或阈值调整。"
    )
    normalized.computation_scope = _normalize_rule_clauses(
        normalized.computation_scope,
        split_rule="",
        purge_rule="",
        selection_rule="",
    )
    scope_reference = "分段与边界处理以 validation_protocol 为准。"
    if scope_reference not in normalized.computation_scope:
        normalized.computation_scope = "；".join(
            part for part in (normalized.computation_scope.strip("；。 "), scope_reference) if part
        )
    normalized.validation_protocol = _normalize_validation_steps(
        normalized.validation_protocol,
        split_rule=split_rule,
        purge_rule=purge_rule,
    )
    if _is_unweighted_mae(normalized.primary_metric) and _mae_formula_needs_repair(
        normalized.metric_formula,
        forbidden_population=("holdout", "final", "test", "留出", "测试"),
    ):
        normalized.metric_formula = (
            f"{normalized.primary_metric} = mean(|y_true_i - y_pred_i|) over the frozen "
            "development population after purge; N_dev is fixed before candidate execution, "
            "and invalid predictions invalidate the candidate rather than changing its denominator."
        )
    if _is_unweighted_mae(normalized.final_evaluation_metric) and _mae_formula_needs_repair(
        normalized.final_evaluation_formula,
        forbidden_population=("development", "开发"),
    ):
        normalized.final_evaluation_formula = (
            f"{normalized.final_evaluation_metric} = mean(|y_true_i - y_pred_i|) over the frozen "
            "holdout population after purge; evaluate once only after candidate selection is frozen."
        )

    fixed_population_rule = (
        "开发段冻结人口中的每条样本必须恰好产生一个符合输出合同、有限且可对齐的预测值。任一样本推理异常、"
        "缺失预测、非数值、NaN/Inf、错时或不符合任务输出合同，该候选整体无效并排除排序；不得跳过样本、"
        "改变分母或用占位值修补。在线单条推理失败不授权从开发评估人口删行。"
    )
    normalized.invalid_solution_rules = list(
        dict.fromkeys(list(normalized.invalid_solution_rules or []) + [fixed_population_rule])
    )
    normalized.submission_checks = list(
        dict.fromkeys(list(normalized.submission_checks or []) + [fixed_population_rule])
    )

    normalized.assumptions = [
        str(item).strip()
        for item in normalized.assumptions
        if str(item or "").strip()
        and not _contains_split_rule(str(item))
        and not _contains_purge_rule(str(item))
    ]
    normalized.assumptions.extend([
        (
            "实现约定：任务指定按时间顺序 60/20/20 切分但未指定取整方式；"
            "训练和开发样本数分别向下取整，余数全部归入留出段。"
        ),
        (
            "实现约定：任务指定一小时边界 purge 但未指定等号处理；"
            "以标签时刻为准，在训练/开发和开发/留出两处边界剔除 s + 1h > s_min 的较早样本，"
            "等号样本保留。"
        ),
    ])
    normalized.assumptions = list(dict.fromkeys(normalized.assumptions))

    normalized.leakage_guards = [
        str(item).strip()
        for item in normalized.leakage_guards
        if str(item or "").strip()
        and not _contains_split_rule(str(item))
        and not _contains_purge_rule(str(item))
    ]
    leakage_reference = "分段与边界处理以 validation_protocol 为准。"
    if leakage_reference not in normalized.leakage_guards:
        normalized.leakage_guards.append(leakage_reference)
    normalized.leakage_guards = list(dict.fromkeys(normalized.leakage_guards))

    changes = []
    for field in ("computation_scope", "metric_formula", "final_evaluation_formula",
                  "validation_protocol", "assumptions", "leakage_guards",
                  "invalid_solution_rules", "submission_checks"):
        if getattr(normalized, field) != getattr(review, field):
            changes.append(field)
    return normalized, changes


def _replace_numbered_protocol_step(protocol: str, number: int, replacement: str) -> str:
    text = str(protocol or "")
    markers = list(re.finditer(r"(?<!\w)(\d+)\)\s*", text))
    for index, marker in enumerate(markers):
        if int(marker.group(1)) != number:
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        return text[:marker.start()] + replacement.rstrip("。") + "。" + text[end:]
    return text


_SPLIT_RULE_PATTERN = re.compile(
    r"(?:60\s*/\s*20\s*/\s*20|(?:最早|前)\s*60\s*%|(?:中间|接续)\s*20\s*%|"
    r"(?:最后|剩余|末尾)\s*20\s*%|floor\s*\(\s*0\.6|ceil|train_test_split|切分取整|时间切分)",
    re.IGNORECASE,
)
_PURGE_RULE_PATTERN = re.compile(
    r"(?:\bpurge\b|边界.{0,20}(?:清洗|剔除)|(?:清洗|剔除).{0,20}边界)",
    re.IGNORECASE,
)


def _contains_split_rule(text: str) -> bool:
    value = str(text or "")
    matches = list(_SPLIT_RULE_PATTERN.finditer(value))
    if not matches:
        return False
    if any(match.group(0).casefold() != "ceil" for match in matches):
        return True
    return bool(re.search(
        r"(?:split|train|dev|holdout|训练|开发|留出|切分)",
        value,
        re.IGNORECASE,
    ))


def _contains_purge_rule(text: str) -> bool:
    return bool(_PURGE_RULE_PATTERN.search(str(text or "")))


def _split_rule_clauses(text: str) -> list[str]:
    return [
        clause.strip()
        for clause in re.split(r"(?<=[。；;])\s*|\r?\n", str(text or ""))
        if clause.strip()
    ]


def _normalize_rule_clauses(
    text: str,
    *,
    split_rule: str,
    purge_rule: str,
    selection_rule: str,
) -> str:
    preserved = [
        clause
        for clause in _split_rule_clauses(text)
        if not _contains_split_rule(clause) and not _contains_purge_rule(clause)
    ]
    return "；".join(
        item for item in [*preserved, split_rule, purge_rule, selection_rule] if str(item).strip()
    )


def _normalize_validation_steps(protocol: str, *, split_rule: str, purge_rule: str) -> str:
    text = str(protocol or "")
    marker_pattern = re.compile(
        r"(?:^|[ \t\u3000。；])(?P<paren_number>\d+)[)、]\s+"
        r"|^[ \t\u3000]*(?P<dot_number>\d+)\.\s+",
        re.MULTILINE,
    )
    markers = []
    for match in marker_pattern.finditer(text):
        number_group = "paren_number" if match.group("paren_number") is not None else "dot_number"
        markers.append((match.start(number_group), match.end()))
    if not markers:
        clauses = _split_rule_clauses(text)
        rewritten: list[str] = []
        inserted = False
        for clause in clauses:
            has_rule = _contains_split_rule(clause) or _contains_purge_rule(clause)
            if has_rule and not inserted:
                rewritten.extend([split_rule, purge_rule])
                inserted = True
            if not has_rule:
                rewritten.append(clause)
        if not inserted:
            rewritten.extend([split_rule, purge_rule])
        return "；".join(rewritten)

    prefix = text[: markers[0][0]].strip()
    steps: list[str] = []
    insertion_index: int | None = None
    for index, (marker_start, marker_end) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        item = text[marker_end : end].strip()
        clauses = _split_rule_clauses(item)
        has_rule = any(_contains_split_rule(clause) or _contains_purge_rule(clause) for clause in clauses)
        if has_rule and insertion_index is None:
            insertion_index = len(steps)
        preserved = [
            clause
            for clause in clauses
            if not _contains_split_rule(clause) and not _contains_purge_rule(clause)
        ]
        steps.extend(preserved)

    if insertion_index is None:
        insertion_index = len(steps)
    steps[insertion_index:insertion_index] = [split_rule, purge_rule]
    numbered = [f"{index}) {step.rstrip('。；; ')}。" for index, step in enumerate(steps, 1)]
    return " ".join(([prefix] if prefix else []) + numbered)


def _is_unweighted_mae(metric: str) -> bool:
    tokens = re.findall(r"[a-z]+", str(metric or "").lower())
    return "mae" in tokens and not any(token in tokens for token in ("weighted", "wmae"))


def _mae_formula_needs_repair(formula: str, *, forbidden_population: tuple[str, ...]) -> bool:
    text = str(formula or "")
    lowered = text.lower()
    if any(token in lowered for token in forbidden_population):
        return True
    describes_mean_absolute_error = (
        (
            "mean" in lowered
            and (
                "abs(" in lowered
                or "absolute" in lowered
                or re.search(r"mean\s*\(\s*\|[^|]+\|", lowered)
            )
        )
        or "mean_absolute_error" in lowered
    )
    return not describes_mean_absolute_error


def section_fact_evidence_refs(evidence, issues):
    """Accept cited dictionary subfields only when they exist in supplied evidence."""
    allowed = set(evidence)
    for issue in issues:
        for reference in issue.evidence_refs:
            value = evidence
            for key in reference.split("."):
                if not isinstance(value, dict) or key not in value:
                    break
                value = value[key]
            else:
                allowed.add(reference)
    return allowed


def section_fact_fields(sections, issues):
    allowed = {}
    for issue in issues:
        match = re.fullmatch(r"(?:main_task_protocol\.frozen_description_sections|description_sections)\.([a-z_]+)\.facts_used(?:\[(\d+)\])?", issue.section.strip())
        if not match or issue.repair_target != "machine_contract":
            continue
        name, selected = match.groups()
        facts = sections.get(name, {}).get("facts_used", [])
        for index, fact in enumerate(facts):
            if isinstance(fact, str) and (selected is None or index == int(selected)):
                allowed[f"/description_sections/{name}/facts_used/{index}"] = fact
    return allowed


def apply_section_fact_repairs(sections, repair, *, issues, evidence_refs):
    allowed = section_fact_fields(sections, issues)
    accepted = {issue.issue_id for issue in issues}
    candidate, seen = deepcopy(sections), set()
    if not repair.repairs:
        raise ValueError("No fact repair returned")
    for item in repair.repairs:
        if (item.path not in allowed or item.path in seen or item.expected != allowed[item.path]
                or not isinstance(item.replacement, str) or not item.replacement.strip()
                or item.replacement == item.expected or not item.reason.strip()
                or not item.issue_ids or not set(item.issue_ids).issubset(accepted)
                or not item.evidence_refs or not set(item.evidence_refs).issubset(evidence_refs)):
            raise ValueError("Fact repair lacks an allowed leaf, exact old value or declared evidence")
        _, _, name, _, index = item.path.split("/")
        candidate[name]["facts_used"][int(index)] = item.replacement
        seen.add(item.path)
    return candidate


def repairable_fields(bundle, sample, *, allow_output_columns=False):
    fields = {
        "/sample_submission_spec/row_unit": sample.row_unit,
        "/sample_submission_spec/row_count_rule": sample.row_count_rule,
        "/sample_submission_spec/format_rules": sample.format_rules,
        "/sample_submission_spec/validation_rules": sample.validation_rules,
        "/description_protocol_bundle/ml_dl/target": bundle.ml_dl.target,
        "/description_protocol_bundle/ml_dl/prediction_unit": bundle.ml_dl.prediction_unit,
        "/description_protocol_bundle/ml_dl/validation_design": bundle.ml_dl.validation_design,
        "/description_protocol_bundle/output/row_unit": bundle.output.row_unit,
        "/description_protocol_bundle/output/format_rules": bundle.output.format_rules,
    }
    if bundle.problem_paradigm == "hybrid_ml_optimization":
        fields["/description_protocol_bundle/hybrid/handoff"] = bundle.hybrid.handoff
    if allow_output_columns:
        fields["/description_protocol_bundle/output/columns"] = bundle.output.columns
    list_paths = (
        "/sample_submission_spec/format_rules",
        "/sample_submission_spec/validation_rules",
        "/description_protocol_bundle/ml_dl/feature_boundary",
        "/description_protocol_bundle/ml_dl/leakage_guards",
        "/description_protocol_bundle/output/format_rules",
    )
    for path in list_paths:
        value = fields.get(path)
        if isinstance(value, list):
            fields.update({f"{path}/{index}": item for index, item in enumerate(value)})
    for group in ("column_meanings", "source_fields"):
        for key, value in getattr(sample, group).items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            fields[f"/sample_submission_spec/{group}/" + escaped] = value
    for index, value in enumerate(bundle.ml_dl.feature_boundary):
        fields[f"/description_protocol_bundle/ml_dl/feature_boundary/{index}"] = value
    for index, value in enumerate(bundle.ml_dl.leakage_guards):
        fields[f"/description_protocol_bundle/ml_dl/leakage_guards/{index}"] = value
    return fields


def apply_derived_repairs(bundle, sample, repair, *, issue_ids, evidence_refs, allow_output_columns=False):
    allowed = repairable_fields(bundle, sample, allow_output_columns=allow_output_columns)
    candidate = {"description_protocol_bundle": bundle.model_dump(), "sample_submission_spec": sample.model_dump()}
    candidate = deepcopy(candidate)
    seen = set()
    if not repair.repairs:
        raise ValueError("No evidence-backed derived contract repair was returned")
    for item in repair.repairs:
        if item.path not in allowed:
            raise ValueError(f"Unlisted repair path {item.path}; use exactly a current_repairable_fields key, without appending an index")
        if item.path in seen:
            raise ValueError(f"Duplicate repair path {item.path}")
        if item.expected != allowed[item.path]:
            raise ValueError(f"Expected value mismatch at {item.path}; copy the full current value exactly")
        if not item.issue_ids or not set(item.issue_ids).issubset(issue_ids):
            raise ValueError(f"Unknown or missing audited issue IDs at {item.path}")
        if not item.evidence_refs or not set(item.evidence_refs).issubset(evidence_refs):
            raise ValueError(f"Unknown or missing evidence IDs at {item.path}; use allowed_evidence_refs")
        if not item.reason.strip() or item.replacement == item.expected:
            raise ValueError(f"Missing reason or unchanged replacement at {item.path}")
        previous = allowed[item.path]
        if type(item.replacement) is not type(previous) or not item.replacement:
            raise ValueError("Derived contract repair cannot erase a rule or change its type")
        if isinstance(previous, list) and not all(isinstance(value, str) and value.strip() for value in item.replacement):
            raise ValueError("Contract rules must remain nonempty strings")
        seen.add(item.path)
        parts = [part.replace("~1", "/").replace("~0", "~") for part in item.path.split("/")[1:]]
        parent = candidate
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
        parent[key] = item.replacement
    return (DescriptionProtocolBundle.model_validate(candidate["description_protocol_bundle"]),
            SampleSubmissionSpec.model_validate(candidate["sample_submission_spec"]))
