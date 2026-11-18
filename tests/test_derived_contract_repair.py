import pytest

from autorealize.artifact_contract_repair import DerivedContractRepair, apply_derived_repairs, repairable_fields
from autorealize.models import DescriptionProtocolBundle, SampleSubmissionSpec


def example_patch(**changes):
    return {"path": "/sample_submission_spec/row_count_rule", "expected": "union of separate origins",
            "replacement": "G x common_origins x {30,60}", "issue_ids": ["shared_origin"],
            "evidence_refs": ["evaluation_contract"], "reason": "Preserve common forecast origins", **changes}


def test_derived_repair_preserves_unrelated_constraints_and_original_objects():
    bundle = DescriptionProtocolBundle(constraints=["Never use future labels"], required_method_families=["graph"])
    sample = SampleSubmissionSpec(row_count_rule="union of separate origins", columns=["id", "prediction"],
                                  validation_rules=["finite predictions", "complete coverage"])
    new_bundle, new_sample = apply_derived_repairs(bundle, sample, DerivedContractRepair(repairs=[example_patch()]),
        issue_ids={"shared_origin"}, evidence_refs={"evaluation_contract"})
    assert new_bundle == bundle
    assert sample.row_count_rule == "union of separate origins"
    assert new_sample.row_count_rule == "G x common_origins x {30,60}"
    assert new_sample.validation_rules == sample.validation_rules
    assert new_sample.columns == sample.columns


def test_output_columns_remain_immutable_without_explicit_non_authoritative_scope():
    bundle = DescriptionProtocolBundle()
    bundle.output.columns = ["decision"]
    patch = DerivedContractRepair(repairs=[example_patch(path="/description_protocol_bundle/output/columns",
        expected=["decision"], replacement=["decision", "audit_ref"])])
    with pytest.raises(ValueError):
        apply_derived_repairs(bundle, SampleSubmissionSpec(), patch, issue_ids={"shared_origin"}, evidence_refs={"evaluation_contract"})
    updated, _ = apply_derived_repairs(bundle, SampleSubmissionSpec(), patch,
        issue_ids={"shared_origin"}, evidence_refs={"evaluation_contract"}, allow_output_columns=True)
    assert updated.output.columns == ["decision", "audit_ref"]
    assert bundle.output.columns == ["decision"]


@pytest.mark.parametrize("changes", [
    {"path": "/description_protocol_bundle/constraints"}, {"expected": "wrong old value"},
    {"replacement": ""}, {"replacement": []}, {"evidence_refs": ["invented_evidence"]},
    {"issue_ids": ["unrelated_issue"]}, {"reason": ""},
])
def test_derived_repair_rejects_unbounded_or_unjustified_changes(changes):
    sample = SampleSubmissionSpec(row_count_rule="union of separate origins")
    with pytest.raises(ValueError):
        apply_derived_repairs(DescriptionProtocolBundle(), sample, DerivedContractRepair(repairs=[example_patch(**changes)]),
            issue_ids={"shared_origin"}, evidence_refs={"evaluation_contract"})
    assert sample.row_count_rule == "union of separate origins"


def test_partially_invalid_patch_does_not_apply_earlier_operations():
    sample = SampleSubmissionSpec(row_count_rule="union of separate origins")
    repair = DerivedContractRepair(repairs=[example_patch(), example_patch()])
    with pytest.raises(ValueError):
        apply_derived_repairs(DescriptionProtocolBundle(), sample, repair,
                             issue_ids={"shared_origin"}, evidence_refs={"evaluation_contract"})
    assert sample.row_count_rule == "union of separate origins"


def test_boundary_repair_changes_only_the_audited_entry():
    bundle = DescriptionProtocolBundle()
    bundle.ml_dl.feature_boundary = ["Use events at or before o", "Fit preprocessing on fit only"]
    bundle.ml_dl.leakage_guards = ["Holdout is used once"]
    repair = DerivedContractRepair(repairs=[example_patch(
        path="/description_protocol_bundle/ml_dl/feature_boundary/0", expected=bundle.ml_dl.feature_boundary[0],
        replacement="Use raw events strictly before o; completed buckets have right endpoint <= o",
    )])
    updated, _ = apply_derived_repairs(bundle, SampleSubmissionSpec(), repair,
        issue_ids={"shared_origin"}, evidence_refs={"evaluation_contract"})
    assert updated.ml_dl.feature_boundary[1:] == bundle.ml_dl.feature_boundary[1:]
    assert updated.ml_dl.leakage_guards == bundle.ml_dl.leakage_guards
    assert bundle.ml_dl.feature_boundary[0] == "Use events at or before o"
    repair.repairs[0].path = "/description_protocol_bundle/ml_dl/feature_boundary"
    with pytest.raises(ValueError):
        apply_derived_repairs(bundle, SampleSubmissionSpec(), repair,
            issue_ids={"shared_origin"}, evidence_refs={"evaluation_contract"})


def test_derived_contract_repair_accepts_only_explicit_list_leaf_and_handoff_paths():
    bundle = DescriptionProtocolBundle(
        problem_paradigm="hybrid_ml_optimization",
        hybrid={"handoff": "predicted_al_n is quality evidence"},
        output={"format_rules": ["rule one", "rule two"]},
    )
    fields = repairable_fields(bundle, SampleSubmissionSpec())
    assert fields["/description_protocol_bundle/hybrid/handoff"] == "predicted_al_n is quality evidence"
    assert fields["/description_protocol_bundle/output/format_rules/1"] == "rule two"

    repair = DerivedContractRepair(repairs=[
        example_patch(
            path="/description_protocol_bundle/hybrid/handoff",
            expected="predicted_al_n is quality evidence",
            replacement="Only observations visible by cutoff and frozen training references may enter steam advice.",
        ),
        example_patch(
            path="/description_protocol_bundle/output/format_rules/1",
            expected="rule two",
            replacement="Preserve the source unit and do not compare it with kg/h until verified.",
        ),
    ])
    updated, _ = apply_derived_repairs(
        bundle,
        SampleSubmissionSpec(),
        repair,
        issue_ids={"shared_origin"},
        evidence_refs={"evaluation_contract"},
    )

    assert updated.hybrid.handoff.startswith("Only observations")
    assert updated.output.format_rules == [
        "rule one",
        "Preserve the source unit and do not compare it with kg/h until verified.",
    ]
    assert bundle.output.format_rules == ["rule one", "rule two"]
