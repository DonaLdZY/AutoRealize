"""Bounded, evidence-linked repairs of derived output and target descriptions."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import DescriptionProtocolBundle, SampleSubmissionSpec


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
    if allow_output_columns:
        fields["/description_protocol_bundle/output/columns"] = bundle.output.columns
    for key, value in sample.column_meanings.items():
        escaped = key.replace("~", "~0").replace("/", "~1")
        fields["/sample_submission_spec/column_meanings/" + escaped] = value
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
        if (item.path not in allowed or item.path in seen or item.expected != allowed[item.path]
                or not item.issue_ids or not set(item.issue_ids).issubset(issue_ids)
                or not item.evidence_refs or not set(item.evidence_refs).issubset(evidence_refs)
                or not item.reason.strip() or item.replacement == item.expected):
            raise ValueError("Derived contract repair lacks an allowed path, exact previous value, or visible evidence")
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
