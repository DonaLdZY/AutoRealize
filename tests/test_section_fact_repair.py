from copy import deepcopy
from types import SimpleNamespace

import pytest

from autorealize.artifact_contract_repair import DerivedContractRepair, apply_section_fact_repairs, section_fact_fields, section_fact_evidence_refs


def test_only_audited_fact_leaf_changes_and_other_constraints_survive():
    sections = {"data": {"markdown": "Verified reader uses header=0", "facts_used": ["Reader requires header=2", "Do not assume capacity dates"]},
                "fields": {"facts_used": ["Preserve units"]}}
    issues = [SimpleNamespace(section="main_task_protocol.frozen_description_sections.data.facts_used[0]", repair_target="machine_contract", issue_id="reader")]
    path = "/description_sections/data/facts_used/0"
    patch = {"path": path, "expected": sections["data"]["facts_used"][0], "replacement": "Verified reader uses header=0",
             "issue_ids": ["reader"], "evidence_refs": ["verified_data_access"], "reason": "Original-file reader verification"}
    before = deepcopy(sections)
    updated = apply_section_fact_repairs(sections, DerivedContractRepair(repairs=[patch]), issues=issues, evidence_refs={"verified_data_access"})
    assert sections == before
    assert updated["data"]["facts_used"] == ["Verified reader uses header=0", "Do not assume capacity dates"]
    assert updated["fields"] == sections["fields"]
    assert section_fact_fields(sections, issues) == {path: patch["expected"]}
    for changes in ({"path": "/description_sections/data/markdown"}, {"path": "/description_sections/data/facts_used/1"},
                    {"replacement": ""}, {"expected": "not original"}, {"evidence_refs": ["invented"]}):
        with pytest.raises(ValueError):
            apply_section_fact_repairs(sections, DerivedContractRepair(repairs=[{**patch, **changes}]), issues=issues, evidence_refs={"verified_data_access"})
    assert sections == before


def test_cited_evidence_subfield_must_exist_in_the_supplied_snapshot():
    evidence = {"evaluation_contract": {"submission_checks": ["explicit empty array"]},
                "current_section_markdown": {"output": "Empty assignments required"}}
    issue = SimpleNamespace(evidence_refs=["evaluation_contract.submission_checks",
        "evaluation_contract.invented", "missing.submission_checks", "evaluation_contract.submission_checks.fake"])
    allowed = section_fact_evidence_refs(evidence, [issue])
    assert allowed == {"evaluation_contract", "current_section_markdown", "evaluation_contract.submission_checks"}
