"""Deterministic task handoff shared by search, reusable artifacts, and reporting."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .scalar_score import scalar_score_shape_defects


_INACTIVE_STATUSES = {"superseded", "not_current", "resolved"}
_EVALUATION_FIELDS = (
    "ranking_scope", "primary_metric", "metric_direction", "metric_formula", "scalar_score_formula", "prediction_unit",
    "y_true_source", "y_pred_source", "computation_scope", "aggregation_rule",
    "validation_protocol", "submission_checks", "leakage_guards",
    "invalid_solution_rules", "tie_break_rules", "assumptions",
)


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strings(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        values = [values] if values else []
    return list(dict.fromkeys(str(item).strip() for item in values if str(item or "").strip()))


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def build_constraint_ledger(bundle: dict, context: dict) -> list[dict[str, Any]]:
    """Retain provenance and state; priority never determines constraint strength."""
    memory = _mapping(context.get("constraint_memory"))
    authority = _mapping(context.get("authoritative_memory"))
    entries: list[dict[str, Any]] = []
    for raw in memory.get("items", []) or []:
        item = raw if isinstance(raw, dict) else {"description": str(raw)}
        status = str(item.get("status") or "current_candidate")
        if status in _INACTIVE_STATUSES:
            continue
        description = str(item.get("description") or item.get("name") or "").strip()
        if not description:
            continue
        strength = item.get("strength", "unknown")
        entries.append({
            "description": description,
            "strength": strength if strength in {"hard", "soft"} else "unknown",
            "status": status,
            "evidence": _strings(item.get("evidence")),
            "related_fields": _strings(item.get("related_fields")),
            "acceptance_check": str(item.get("acceptance_check") or ""),
            "source": "constraint_memory",
        })
    sources = (
        (authority.get("constraints", []), "authoritative_memory", "unknown"),
        (bundle.get("constraints", []), "task_protocol", "unknown"),
        (_mapping(bundle.get("optimization")).get("hard_constraints", []), "optimization_protocol", "hard"),
        (_mapping(bundle.get("optimization")).get("soft_constraints", []), "optimization_protocol", "soft"),
    )
    def key(description: str) -> str:
        return re.sub(r"\s+", " ", description).strip()

    seen = {key(item["description"]): item for item in entries}
    for values, source, strength in sources:
        for description in _strings(values):
            normalized = key(description)
            if normalized in seen:
                prior = seen[normalized]
                prior.setdefault("additional_sources", []).append({
                    "source": source, "description": description, "strength": strength,
                    "evidence": _strings(authority.get("source_files")) if source == "authoritative_memory" else [],
                })
                if strength != "unknown" and prior["strength"] == "unknown":
                    prior["strength"] = strength
                    prior["strength_source"] = source
                continue
            entries.append({
                "description": description,
                "strength": strength,
                "status": "current" if source == "authoritative_memory" else "current_candidate",
                "evidence": _strings(authority.get("source_files")) if source == "authoritative_memory" else [],
                "related_fields": [],
                "acceptance_check": "",
                "source": source,
            })
            seen[normalized] = entries[-1]
    for item in entries:
        item["constraint_id"] = "constraint:" + _fingerprint({
            key: item[key] for key in ("description", "source", "evidence")
        })[:16]
    return entries


def build_execution_contract(
    *, bundle: dict, context: dict, evaluation: dict, output: dict, data_schema: dict,
) -> dict[str, Any]:
    authority = _mapping(context.get("authoritative_memory"))
    memory = _mapping(context.get("constraint_memory"))
    ledger = build_constraint_ledger(bundle, context)
    stage_issues = [
        issue
        for section in _mapping(context.get("description_sections")).values()
        for issue in _strings(_mapping(section).get("open_issues"))
    ]
    investigation = _mapping(_mapping(context.get("agent_context_pack")).get("question_memory"))
    if not investigation:
        investigation = _mapping(_mapping(context.get("knowledge_base")).get("question_investigation"))
    questions = _strings(
        _strings(authority.get("unresolved_questions"))
        + _strings(memory.get("unresolved_questions"))
        + stage_issues + _strings(investigation.get("unresolved_questions"))
        + [item["description"] for item in ledger if item["status"] == "unresolved"]
    )
    question_history = []
    for resolution in context.get("open_question_resolutions", []) or []:
        if not isinstance(resolution, dict) or resolution.get("original_text") not in questions:
            continue
        index = questions.index(resolution["original_text"])
        questions[index:index + 1] = _strings(resolution.get("remaining_questions", []))
        question_history.append(resolution)
    questions = _strings(questions)
    conflicts = [
        item for item in (authority.get("authority_conflicts") or [])
        if not isinstance(item, dict) or item.get("status") not in _INACTIVE_STATUSES
    ]
    assumptions = _strings(evaluation.get("assumptions"))
    blockers = []
    if "artifact_consistency_review" in context:
        review = _mapping(context["artifact_consistency_review"])
        if review.get("passed") is not True:
            blockers.append("Final artifact consistency review has not passed.")
        blockers.extend(
            f"Final artifact consistency: {issue.get('message') or issue.get('category') or 'blocking issue'}"
            for issue in review.get("issues", []) or []
            if isinstance(issue, dict) and issue.get("severity") == "blocking"
        )
    blockers.extend(_strings(context.get("defects_after_gate")))
    if evaluation.get("passed") is not True or evaluation.get("executable") is not True:
        blockers.append("Evaluation contract has not passed executable validation.")
    for name in ("primary_metric", "metric_formula", "validation_protocol"):
        if not str(evaluation.get(name) or "").strip():
            blockers.append(f"Missing evaluation field: {name}.")
    if evaluation.get("metric_direction") not in {"minimize", "maximize"}:
        blockers.append("Metric direction must be minimize or maximize.")
    blockers.extend(scalar_score_shape_defects(evaluation))
    blockers.extend(
        f"Unresolved hard constraint: {item['description']}"
        for item in ledger if item["strength"] == "hard" and item["status"] == "unresolved"
    )
    blockers.extend(
        f"Unresolved authority conflict: {json.dumps(item, ensure_ascii=False, sort_keys=True)}"
        for item in conflicts if isinstance(item, dict) and (
            item.get("status") == "unresolved" or item.get("severity") == "blocking"
        )
    )
    evaluation_snapshot = {key: evaluation.get(key) for key in _EVALUATION_FIELDS}
    evaluation_id = "evaluation:" + _fingerprint(evaluation_snapshot)
    snapshot = {
        "task_goal": bundle.get("task_goal") or bundle.get("overview") or context.get("task_hint", ""),
        "problem_paradigm": bundle.get("problem_paradigm"),
        "task_boundaries": {key: bundle.get(key) for key in (
            "problem_structure", "decision_temporality", "environment_source",
            "ml_dl", "optimization", "rl", "hybrid", "allowed_method_families",
        )},
        "required_method_families": _strings(
            _strings(bundle.get("required_method_families"))
            + _strings(_mapping(context.get("problem_paradigm_review")).get("required_method_families"))
        ),
        "evaluation_id": evaluation_id,
        "output": {key: output.get(key) for key in (
            "output_kind", "output_filename", "columns", "row_unit", "format_rules",
            "sample_submission_required", "authoritative_submission_contract",
        )},
        "data_schema": data_schema,
        "constraint_ledger": ledger,
    }
    is_static_optimization = str(
        bundle.get("problem_paradigm")
        or _mapping(context.get("problem_paradigm_review")).get("problem_paradigm")
        or context.get("problem_paradigm") or ""
    ) == "static_optimization"
    reproducibility_requirements = ([
        "Deliver reusable input-validation and solving entrypoints, dependencies, configuration, and any task-required solver artifacts.",
        "Use exact task inputs and return the task-defined decision, infeasibility or missing-input response. Prediction-only training, labels and holdout requirements do not automatically apply to a static solver; any learned subcomponent must follow its explicitly authorized evaluation protocol.",
        "Persist exact physical and logical schemas, input fingerprints, mapping versions, units, solver parameters, seed/configuration and these contract identifiers.",
        "Smoke-test reloading the solver configuration and executing the task-defined checks in a fresh process; report missing verification, operational data or unresolved business mappings honestly.",
    ] if is_static_optimization else [
        "Deliver reusable training and inference entrypoints, dependencies, configuration, and persisted model or policy artifacts.",
        "Inference must load the saved preprocessing and model state in a fresh process; it must not retrain or read validation labels.",
        "Persist exact feature and output schemas, data and split fingerprints, seed/configuration, and these contract identifiers.",
        "Smoke-test reload and inference on a held-out input using the task's output checks; report missing verification honestly.",
    ])
    candidate_comparison_rules = ([
        "Freeze one deterministic feasibility evaluator and objective before comparing solver configurations.",
        "Compare only solutions evaluated on the same physical input, mapping version, contract_id and objective direction; changed rules require reevaluation.",
        "Check all hard constraints before ranking; infeasible, non-finite, or unevaluated solutions cannot become a best feasible result.",
        "Combined candidate solutions must use compatible task-defined decision keys and preserve every hard constraint; do not impose prediction blending or out-of-fold labels on a solver.",
        "Keep software-contract validation distinct from executed decision quality and future operational validation. Missing inputs never establish a measured feasible decision or business benefit.",
    ] if is_static_optimization else [
        "Freeze one evaluator, eligible population, validation split, and metric direction before scoring search nodes.",
        "Compare a node only with scores from the same evaluation_id and realized split fingerprint; a changed evaluator requires reevaluation.",
        "Check hard constraints before ranking; failed, infeasible, non-finite, or unevaluated nodes cannot become a best feasible result.",
        "Ensemble only compatible outputs aligned by exact evaluation unit and out-of-fold or held-out predictions; train blend weights without test labels.",
        "Keep search validation separate from untouched final-test evaluation and report selection bias when no final holdout exists.",
    ])
    report_requirements = ([
        "Trace objectives, feasibility rules, physical readers and missing inputs to source evidence; distinguish confirmed requirements, assumptions, and unresolved issues.",
        "Compare measured feasible solutions on the shared objective and constraint checks, including runtime and missing results.",
        "Describe selected rule configuration, solver entrypoint, reproducibility commands, blocked-data behavior and limitations.",
        "Claim the best observed feasible solution within the explored budget; do not claim global optimality or production savings without evidence.",
    ] if is_static_optimization else [
        "Trace objectives and constraints to source evidence; distinguish confirmed requirements, soft preferences, assumptions, and unresolved issues.",
        "Compare measured candidates on the shared metric and validation protocol, including constraint feasibility, runtime, and missing results.",
        "Describe ensemble members and weights, selected artifacts, reproduction commands, limitations, and verified inference behavior.",
        "Claim best observed feasible candidate within the explored budget; do not claim global optimality or production readiness without evidence.",
    ])
    candidate_comparison_rules.append(
        "Rank by exactly one finite numeric scalar with the frozen formula, direction, weights, scales and aggregation. "
        "No tuple, Pareto ranking, independent business tie-break or candidate-specific scalarization. "
        "Persist raw objective components for audit; they do not create a second ranking."
    )
    return {
        "schema_version": "autorealize.execution_contract.v1",
        "contract_id": "task:" + _fingerprint(snapshot),
        "evaluation_id": evaluation_id,
        "readiness": "blocked" if blockers else "ready_with_assumptions" if (
            assumptions or questions or conflicts
            or any(item["status"] != "current" or item["strength"] == "unknown" for item in ledger)
        ) else "ready",
        "identity_semantics": {
            "contract_id": "Deterministic task-rule fingerprint, not an execution or performance claim.",
            "evaluation_id": "Deterministic evaluation-rule fingerprint shared by comparable candidates and runs; valid before any model executes.",
            "evaluation_run_id": "Generated by the evaluator at execution time to distinguish actual evaluation runs; not available from task compilation.",
            "split_fingerprint": (
                "Follow the frozen static evaluation contract. When no split applies, preserve its exact "
                "not-applicable marker (not_applicable_static_optimization by default); do not generate "
                "evaluation unit IDs, folds or a runtime split. Fingerprint actual inputs separately. "
                "Only an explicitly defined task evaluation split can override this no-split default."
                if is_static_optimization else
                "Generated from actual evaluation unit IDs/folds at execution time, separate from the rule fingerprint."
            ),
        },
        "constraint_ledger": ledger,
        "constraint_resolution_policy": [
            "All current hard constraints apply together. Ledger order never grants override priority.",
            "Unknown strength means unclassified evidence, not permission to relax a matching hard or soft constraint. A current_candidate paraphrase cannot downgrade a current rule.",
            "Exact whitespace-normalized duplicate descriptions share one entry with additional source records. Other paraphrases retain their wording and provenance; do not infer that they supersede existing rules.",
            "Only explicit evidence-backed supersession can retire a rule. Incompatible current hard constraints require an authority resolution before execution; never resolve them by taking the last entry.",
        ],
        "assumptions": assumptions,
        "open_questions": questions,
        "resolved_question_history": question_history,
        "authority_conflicts": conflicts,
        "blocking_issues": blockers,
        "reproducibility_requirements": reproducibility_requirements,
        "candidate_comparison_rules": candidate_comparison_rules,
        "report_requirements": report_requirements,
    }
