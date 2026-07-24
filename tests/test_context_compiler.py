from __future__ import annotations

import json
from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.context_compiler import (
    ArtifactStore,
    build_population_verification_queue,
    build_qdi_context_bundle,
    build_qdi_table_card_details,
    compact_table_cards_for_prompt,
    context_telemetry,
    reconcile_table_shape,
)
from autorealize.entity_alias import build_entity_alias_candidates
from autorealize.investigation import (
    _entity_alias_verification_questions,
    _population_verification_questions,
    _related_table_card_details_for_prompt,
    _retrieve_qdi_context_excerpt,
    _select_auto_verification_questions,
)
from autorealize.models import ConstraintMemory, FileRole, FileSummary
from autorealize.modules.task_definition import TaskDefinitionModule
from autorealize.pipeline import _compact_entity_alias_schema, _extract_constraint_memory
from autorealize.profiling.relations import RelationHint


def test_qdi_context_compiler_uses_table_cards_and_artifact_refs(tmp_path: Path) -> None:
    fs = FileSummary(
        path="cost/carrier_cost.xlsx",
        role=FileRole.raw_data_table,
        summary="Carrier contract workbook with cost tables and instructions.",
        columns=["order_id", "cost"],
        column_semantics={"order_id": "order key", "cost": "transport cost"},
        column_profiles=[
            {
                "name": "order_id",
                "logical_type": "categorical",
                "row_count": 10,
                "non_null_count": 10,
                "unique_count": 10,
                "null_ratio": 0.0,
                "top_values": ["A001:1"],
            }
        ],
        source_metadata={
            "shape": [10, 2],
            "preview": [{"order_id": "A001", "cost": 12.3}],
            "probe_results": {"huge": "payload"},
            "excel_sheet_profiles": [
                {
                    "sheet_name": "cost_table",
                    "shape": [10, 2],
                    "columns": ["order_id", "cost"],
                    "raw_preview": [["order_id", "cost"], ["A001", 12.3]],
                    "preview": [{"order_id": "A001", "cost": 12.3}],
                    "column_profiles": [
                        {
                            "name": "cost",
                            "logical_type": "numeric",
                            "row_count": 10,
                            "non_null_count": 10,
                            "unique_count": 8,
                            "null_ratio": 0.0,
                            "numeric_stats": {"mean": 20.0, "std": 3.0, "var": 9.0, "min": 10.0, "max": 30.0},
                        }
                    ],
                }
            ],
            "sheet_field_descriptions": {"cost_table": {"cost": "contract transport cost"}},
        },
    )
    artifact_store = ArtifactStore(tmp_path / "artifacts")

    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="minimize cost",
        file_summaries=[fs],
        relation_hints=[
            RelationHint(
                left_file="orders.csv",
                left_field="order_id",
                right_file="cost/carrier_cost.xlsx::cost_table",
                right_field="order_id",
                relation_type="one_to_many",
                confidence=0.88,
                short_evidence="orders order_id unique; cost table repeats order_id.",
            )
        ],
        constraint_memory={"summary": "capacity must hold", "items": ["no overload"]},
        authoritative_memory={"task_goal": "minimize cost"},
        knowledge_base={
            "filename_sample_groups": [
                {
                    "sample_id": "carrier{id}_cost.xlsx",
                    "files": ["cost/carrier01_cost.xlsx", "cost/carrier02_cost.xlsx"],
                    "role": "carrier cost workbooks",
                }
            ]
        },
        artifact_store=artifact_store,
    )

    details = build_qdi_table_card_details(file_summaries=[fs], artifact_store=artifact_store)
    text = json.dumps(context, ensure_ascii=False, sort_keys=True)
    assert "files" not in context
    assert context["table_cards"][0]["table_kind"] == "excel_sheet"
    assert context["table_cards"][0]["sheet_name"] == "cost_table"
    stable_fields = {
        field["name"]: field for field in context["table_cards"][0]["fields"]
    }
    assert stable_fields["cost"]["meaning"] == "contract transport cost"
    assert stable_fields["cost"]["logical_type"] == "numeric"
    assert "field_index" not in context["table_cards"][0]
    assert "artifact_refs" not in context["table_cards"][0]
    detailed_fields = {
        field["name"]: field
        for field in details["cost/carrier_cost.xlsx::cost_table"]["fields"]
    }
    assert detailed_fields["cost"]["numeric_stats"]["mean"] == 20.0
    assert context["relations"][0]["relation_type"] == "one_to_many"
    assert context["filename_sample_groups"][0]["file_count"] == 2
    assert context_telemetry(context)["contains_forbidden_large_keys"] == []
    assert context_telemetry(context)["artifact_refs"] == 0
    assert "source_metadata" not in text
    assert "probe_results" not in text
    assert "raw_preview" not in text
    assert '"preview":' not in text
    assert "visible_excerpt" not in text
    assert '"numeric_stats"' in text
    assert '"mean": 20.0' in text
    assert "reading_notes" in text
    assert "file_cognition" in text
    assert list((tmp_path / "artifacts").glob("*.json"))


def test_filename_group_cards_include_shared_and_variant_fields(tmp_path: Path) -> None:
    files = [
        FileSummary(
            path="contracts/carrier_a.csv",
            role=FileRole.raw_data_table,
            summary="carrier contract A",
            columns=["lane", "vehicle_type", "cost", "settlement_code"],
            source_metadata={"shape": [10, 4]},
        ),
        FileSummary(
            path="contracts/carrier_b.csv",
            role=FileRole.raw_data_table,
            summary="carrier contract B",
            columns=["lane", "vehicle_type", "cost", "carrier_code"],
            source_metadata={"shape": [10, 4]},
        ),
        FileSummary(
            path="contracts/carrier_c.csv",
            role=FileRole.raw_data_table,
            summary="carrier contract C",
            columns=["lane", "vehicle_type", "cost", "region_code"],
            source_metadata={"shape": [10, 4]},
        ),
    ]

    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="assign orders to carriers",
        file_summaries=files,
        relation_hints=[],
        constraint_memory={},
        authoritative_memory={},
        knowledge_base={
            "filename_sample_groups": [
                {
                    "sample_id": "contracts/carrier_{id}.csv",
                    "files": [fs.path for fs in files],
                    "role": "carrier contract tables",
                }
            ]
        },
    )

    group = context["filename_sample_groups"][0]
    assert group["shared_fields"] == ["lane", "vehicle_type", "cost"]
    assert group["structure_consistent"] is False
    assert any("settlement_code" in row["fields"] for row in group["variant_fields_by_file"])
    assert any(item["field"] == "carrier_code" for item in group["field_presence"])


def test_table_card_details_include_excel_read_strategy(tmp_path: Path) -> None:
    fs = FileSummary(
        path="notes.xlsx",
        role=FileRole.raw_data_table,
        summary="Workbook with a headerless mapping sheet.",
        columns=[],
        source_metadata={
            "excel_sheet_profiles": [
                {
                    "sheet_name": "mapping",
                    "shape": [2, 2],
                    "columns": ["A", "1"],
                    "layout_kind": "headerless_table",
                    "read_strategy_kind": "header_none_table",
                    "detected_header_row": None,
                    "recommended_read": "pd.read_excel(path, sheet_name='mapping', header=None)",
                    "reading_risks": ["First row looks like data rather than field names."],
                }
            ]
        },
    )

    details = build_qdi_table_card_details(file_summaries=[fs], artifact_store=ArtifactStore(tmp_path / "artifacts"))
    card = details["notes.xlsx::mapping"]

    assert card["layout_kind"] == "headerless_table"
    assert card["read_strategy_kind"] == "header_none_table"
    assert "header=None" in card["recommended_read"]
    assert any("headerless" in note.lower() for note in card["reading_notes"])


def test_reconcile_table_shape_prefers_full_profile_over_excel_used_range() -> None:
    facts = reconcile_table_shape(
        {"shape": [20_381, 4]},
        [
            {"name": "record_id", "row_count": 2_104, "non_null_count": 2_104},
            {"name": "event_time", "row_count": 2_104, "non_null_count": 2_033},
        ],
    )

    assert facts["verified_shape"] == [2_104, 4]
    assert facts["verified_row_count"] == 2_104
    assert facts["worksheet_used_range_shape"] == [20_381, 4]
    assert facts["row_count_conflict"] is True


def test_population_verification_is_driven_by_generic_profile_evidence() -> None:
    queue = build_population_verification_queue(
        [
            {
                "table_id": "users.csv",
                "verified_row_count": 100,
                "primary_key_candidates": [{"field": "user_id", "status": "candidate_not_confirmed"}],
                "fields": [
                    {
                        "name": "signup_time",
                        "role": "time",
                        "logical_type": "datetime",
                        "non_null_count": 93,
                    }
                ],
            }
        ]
    )

    assert len(queue) == 1
    assert queue[0]["table_id"] == "users.csv"
    assert queue[0]["population_sensitive_fields"][0]["missing_count"] == 7
    assert "order" not in queue[0]["question"].lower()


def test_auto_verification_questions_are_diverse_and_budgeted() -> None:
    context = {
        "population_verification_queue": [
            {
                "table_id": "users.csv",
                "verified_row_count": 100,
                "best_statistical_key_candidate": {"field": "user_id"},
                "population_sensitive_fields": [{"field": "signup_time", "missing_count": 7}],
            },
            {
                "table_id": "events.csv",
                "verified_row_count": 500,
                "best_statistical_key_candidate": {"field": "event_id"},
                "population_sensitive_fields": [{"field": "event_time", "missing_count": 4}],
            },
        ]
    }
    population = _population_verification_questions(context)
    entity = [
        type(population[0])(
            question_id="alias_1",
            question="Verify candidate entity aliases using directional coverage.",
            category="join_key",
            why_blocking="The join remains unverified.",
            priority="high",
        )
    ]

    selected = _select_auto_verification_questions(
        entity_alias_questions=entity,
        population_questions=population,
        limit=2,
    )

    assert len(selected) == 2
    assert {question.category for question in selected} == {"join_key", "evaluation_population"}


def test_source_coverage_requires_file_and_sheet_identity() -> None:
    ledger = {
        "entries": [
            {
                "table_id": "book.xlsx::Sheet1",
                "source_file": "book.xlsx",
                "sheet_name": "Sheet1",
                "coverage_status": "required",
            }
        ]
    }

    defects = TaskDefinitionModule._source_coverage_defects(object(), "## 数据说明\n使用 Sheet1。", ledger)
    covered = TaskDefinitionModule._source_coverage_defects(
        object(),
        "## 数据说明\n从 book.xlsx 的 Sheet1 读取数据。",
        ledger,
    )

    assert defects == ["description_missing_source_coverage: book.xlsx::Sheet1"]
    assert covered == []


def test_entity_alias_candidates_feed_qdi_validation_questions(tmp_path: Path) -> None:
    files = [
        FileSummary(
            path="accounts.xlsx",
            role=FileRole.raw_data_table,
            summary="Account registry.",
            columns=[],
            source_metadata={
                "excel_sheet_profiles": [
                    {
                        "sheet_name": "accounts",
                        "shape": [8, 2],
                        "columns": ["account_code", "account_name"],
                    }
                ]
            },
        ),
        FileSummary(
            path="events.csv",
            role=FileRole.raw_data_table,
            summary="Events keyed by customer.",
            columns=["customer_code", "event_value"],
            source_metadata={"shape": [4, 2]},
        ),
    ]
    alias_candidates = [
        {
            "concept_id": "customer_account",
            "label": "customer/account entity identifier",
            "reason": "Both fields identify the party attached to a business record.",
            "confidence": "medium",
            "candidate_fields": [
                {
                    "source_file": "accounts.xlsx",
                    "sheet_name": "accounts",
                    "field": "account_code",
                    "semantic_role": "party_identifier",
                    "value_kind": "code",
                    "evidence": "Account registry key.",
                },
                {
                    "source_file": "events.csv",
                    "field": "customer_code",
                    "semantic_role": "party_identifier",
                    "value_kind": "code",
                    "evidence": "Customer reference on each event.",
                },
            ],
        }
    ]

    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="Join events to the account registry.",
        file_summaries=files,
        relation_hints=[],
        constraint_memory={"entity_alias_candidates": alias_candidates},
        authoritative_memory={},
        knowledge_base={},
    )

    groups = context["entity_alias_candidates"]
    assert groups
    group = groups[0]
    assert group["status"] == "candidate_not_equivalent"
    fields = {item["field"] for item in group["candidate_fields"]}
    assert fields == {"account_code", "customer_code"}

    questions = _entity_alias_verification_questions(context)
    assert len(questions) == 1
    assert questions[0].category == "join_key"
    assert questions[0].priority == "high"
    assert "join coverage" in questions[0].question
    assert "成本/合同" not in questions[0].question


def test_entity_alias_candidates_require_llm_proposal_and_exact_schema() -> None:
    files = [
        FileSummary(
            path="accounts.csv",
            role=FileRole.raw_data_table,
            summary="Account registry.",
            columns=["account_code"],
        ),
        FileSummary(
            path="events.csv",
            role=FileRole.raw_data_table,
            summary="Customer events.",
            columns=["customer_code"],
        ),
    ]

    assert build_entity_alias_candidates(files, llm_candidates=[]) == []

    groups = build_entity_alias_candidates(
        files,
        llm_candidates=[
            {
                "concept_id": "customer_account",
                "label": "customer/account identifier",
                "candidate_fields": [
                    {
                        "source_file": "accounts.csv",
                        "field": "account_code",
                        "semantic_role": "party_identifier",
                        "value_kind": "code",
                    },
                    {
                        "source_file": "events.csv",
                        "field": "invented_customer_id",
                        "semantic_role": "party_identifier",
                        "value_kind": "code",
                    },
                ],
            }
        ],
    )

    assert groups == []


def test_constraint_extractor_reuses_one_llm_call_for_generic_alias_candidates() -> None:
    files = [
        FileSummary(
            path="accounts.csv",
            role=FileRole.raw_data_table,
            summary="Account registry.",
            columns=["account_code", "account_name"],
            source_metadata={"preview": [{"account_code": "A"}], "probe_results": {"large": "payload"}},
        ),
        FileSummary(
            path="events.csv",
            role=FileRole.raw_data_table,
            summary="Customer events.",
            columns=["customer_code", "event_value"],
        ),
    ]

    class AliasLLM:
        def __init__(self) -> None:
            self.calls = []

        def ask_structured(self, **kwargs):
            self.calls.append(kwargs)
            return ConstraintMemory(
                entity_alias_candidates=[
                    {
                        "concept_id": "customer_account",
                        "label": "customer/account identifier",
                        "reason": "Both fields identify the party attached to a record.",
                        "candidate_fields": [
                            {
                                "source_file": "accounts.csv",
                                "field": "account_code",
                                "semantic_role": "party_identifier",
                                "value_kind": "code",
                            },
                            {
                                "source_file": "events.csv",
                                "field": "customer_code",
                                "semantic_role": "party_identifier",
                                "value_kind": "code",
                            },
                        ],
                    }
                ]
            )

    llm = AliasLLM()
    memory = _extract_constraint_memory(
        llm_client=llm,
        prompt_mgr=object(),
        file_summaries=files,
        task_hint="Join events to accounts.",
        authoritative_memory={},
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["prompt_name"] == "constraint_memory_extractor"
    assert "entity_alias_schema" in llm.calls[0]["static_context_prompt"]
    assert "source_metadata" not in llm.calls[0]["static_context_prompt"]
    assert memory["entity_alias_candidates"][0]["concept_id"] == "customer_account"


def test_identical_schema_files_share_alias_prompt_budget() -> None:
    files = [
        FileSummary(
            path=f"batch/source_{index}.csv",
            role=FileRole.raw_data_table,
            summary="Repeated export.",
            columns=["entity_code", "value"],
        )
        for index in range(12)
    ]

    schema = _compact_entity_alias_schema(files, max_fields=2)

    assert len(schema["groups"]) == 1
    assert schema["groups"][0]["fields"] == ["entity_code", "value"]
    assert len(schema["groups"][0]["source_files"]) == 12
    assert schema["truncated"] is False
    assert schema["visible_field_count"] == 2
    assert schema["total_field_count"] == 2


def test_alias_schema_truncation_is_counted_and_visible_to_qdi(tmp_path: Path) -> None:
    files = [
        FileSummary(
            path="first.csv",
            role=FileRole.raw_data_table,
            summary="First source.",
            columns=["a", "b", "c"],
        ),
        FileSummary(
            path="second.csv",
            role=FileRole.raw_data_table,
            summary="Second source.",
            columns=["x", "y"],
        ),
    ]

    schema = _compact_entity_alias_schema(files, max_fields=2)
    telemetry = {
        key: schema[key]
        for key in [
            "truncated",
            "visible_field_count",
            "total_field_count",
            "omitted_field_count",
            "visible_schema_count",
            "omitted_schema_count",
            "omitted_sources",
        ]
    }
    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="Relate the relevant entities.",
        file_summaries=files,
        relation_hints=[],
        constraint_memory={"entity_alias_schema_telemetry": telemetry},
        authoritative_memory={},
        knowledge_base={},
    )

    assert schema["truncated"] is True
    assert schema["groups"][0]["fields"] == ["a", "b"]
    assert schema["groups"][0]["fields_truncated"] is True
    assert schema["visible_field_count"] == 2
    assert schema["total_field_count"] == 5
    assert schema["omitted_field_count"] == 3
    assert schema["visible_schema_count"] == 1
    assert schema["omitted_schema_count"] == 1
    assert schema["omitted_sources"] == ["first.csv", "second.csv"]
    assert context["entity_alias_schema_telemetry"] == telemetry


def test_alias_schema_group_cap_does_not_count_hidden_fields_as_visible() -> None:
    files = [
        FileSummary(
            path=f"source_{index}.csv",
            role=FileRole.raw_data_table,
            summary="Distinct source.",
            columns=[f"field_{index}"],
        )
        for index in range(125)
    ]

    schema = _compact_entity_alias_schema(files, max_fields=600)

    assert len(schema["groups"]) == 120
    assert schema["visible_schema_count"] == 120
    assert schema["omitted_schema_count"] == 5
    assert schema["visible_field_count"] == 120
    assert schema["total_field_count"] == 125
    assert schema["omitted_field_count"] == 5
    assert schema["truncated"] is True
    assert schema["omitted_sources"] == [f"source_{index}.csv" for index in range(120, 125)]


def test_llm_alias_candidate_gets_deterministic_directional_coverage(tmp_path: Path) -> None:
    (tmp_path / "accounts.csv").write_text("account_code\nA\nB\n", encoding="utf-8")
    (tmp_path / "events.csv").write_text("customer_code\nA\nB\nC\n", encoding="utf-8")
    files = [
        FileSummary(
            path="accounts.csv",
            role=FileRole.raw_data_table,
            summary="Account registry.",
            columns=["account_code"],
            source_metadata={"shape": [2, 1]},
        ),
        FileSummary(
            path="events.csv",
            role=FileRole.raw_data_table,
            summary="Customer events.",
            columns=["customer_code"],
            source_metadata={"shape": [3, 1]},
        ),
    ]
    candidates = [
        {
            "concept_id": "customer_account",
            "label": "customer/account identifier",
            "candidate_fields": [
                {
                    "source_file": "accounts.csv",
                    "field": "account_code",
                    "semantic_role": "party_identifier",
                    "value_kind": "code",
                },
                {
                    "source_file": "events.csv",
                    "field": "customer_code",
                    "semantic_role": "party_identifier",
                    "value_kind": "code",
                },
            ],
        }
    ]

    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="Join events to accounts.",
        file_summaries=files,
        relation_hints=[],
        constraint_memory={"entity_alias_candidates": candidates},
        authoritative_memory={},
        knowledge_base={},
    )
    coverage = context["entity_alias_candidates"][0]["deterministic_group_coverage"]["directional_coverage"]
    alias_group = context["entity_alias_candidates"][0]

    assert len(coverage) == 1
    assert coverage[0]["intersection_count"] == 2
    assert coverage[0]["left_covered_by_right_ratio"] == 1.0
    assert coverage[0]["right_covered_by_left_ratio"] == 0.666667
    assert alias_group["evidence_status"] == "asymmetric_coverage"
    assert alias_group["qdi_routing"]["recommended"] is True
    assert len(_entity_alias_verification_questions(context)) == 1


def test_complete_alias_value_overlap_does_not_force_qdi(tmp_path: Path) -> None:
    (tmp_path / "left.csv").write_text("left_code\nA\nB\n", encoding="utf-8")
    (tmp_path / "right.csv").write_text("right_code\nA\nB\n", encoding="utf-8")
    files = [
        FileSummary(path="left.csv", role=FileRole.raw_data_table, summary="Left source.", columns=["left_code"]),
        FileSummary(
            path="right.csv", role=FileRole.raw_data_table, summary="Right source.", columns=["right_code"]
        ),
    ]
    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="Join the two relevant sources.",
        file_summaries=files,
        relation_hints=[],
        constraint_memory={
            "entity_alias_candidates": [
                {
                    "concept_id": "shared_entity",
                    "task_relevance": "high",
                    "candidate_fields": [
                        {"source_file": "left.csv", "field": "left_code", "value_kind": "code"},
                        {"source_file": "right.csv", "field": "right_code", "value_kind": "code"},
                    ],
                }
            ]
        },
        authoritative_memory={},
        knowledge_base={},
    )
    alias_group = context["entity_alias_candidates"][0]

    assert alias_group["evidence_status"] == "strong_value_overlap"
    assert alias_group["qdi_routing"] == {
        "recommended": False,
        "reason": "complete_strong_value_evidence_already_visible",
    }
    assert _entity_alias_verification_questions(context) == []


def test_disjoint_low_relevance_alias_candidate_does_not_force_qdi(tmp_path: Path) -> None:
    (tmp_path / "left.csv").write_text("left_code\nA\nB\n", encoding="utf-8")
    (tmp_path / "right.csv").write_text("right_code\nX\nY\n", encoding="utf-8")
    files = [
        FileSummary(path="left.csv", role=FileRole.raw_data_table, summary="Left source.", columns=["left_code"]),
        FileSummary(
            path="right.csv", role=FileRole.raw_data_table, summary="Right source.", columns=["right_code"]
        ),
    ]
    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="Forecast an unrelated target.",
        file_summaries=files,
        relation_hints=[],
        constraint_memory={
            "entity_alias_candidates": [
                {
                    "concept_id": "unrelated_codes",
                    "task_relevance": "low",
                    "candidate_fields": [
                        {"source_file": "left.csv", "field": "left_code", "value_kind": "code"},
                        {"source_file": "right.csv", "field": "right_code", "value_kind": "code"},
                    ],
                }
            ]
        },
        authoritative_memory={},
        knowledge_base={},
    )
    alias_group = context["entity_alias_candidates"][0]

    assert alias_group["evidence_status"] == "value_disjoint"
    assert alias_group["qdi_routing"] == {"recommended": False, "reason": "low_task_relevance"}
    assert _entity_alias_verification_questions(context) == []


def test_qdi_related_table_details_are_dynamic_and_focused(tmp_path: Path) -> None:
    order_fs = FileSummary(
        path="orders.csv",
        role=FileRole.raw_data_table,
        summary="Order table.",
        columns=["order_id", "customer"],
        column_semantics={"order_id": "order key", "customer": "customer name"},
        column_profiles=[
            {
                "name": "order_id",
                "logical_type": "categorical",
                "row_count": 2,
                "non_null_count": 2,
                "unique_count": 2,
                "top_values": ["A:1", "B:1"],
            }
        ],
        source_metadata={"shape": [2, 2]},
    )
    cost_fs = FileSummary(
        path="cost.csv",
        role=FileRole.raw_data_table,
        summary="Cost table.",
        columns=["carrier", "cost"],
        column_semantics={"carrier": "carrier id", "cost": "transport cost"},
        column_profiles=[
            {
                "name": "cost",
                "logical_type": "numeric",
                "row_count": 2,
                "non_null_count": 2,
                "unique_count": 2,
                "numeric_stats": {"mean": 15.0, "std": 5.0, "var": 25.0, "min": 10.0, "max": 20.0},
            }
        ],
        source_metadata={"shape": [2, 2]},
    )
    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="minimize cost",
        file_summaries=[order_fs, cost_fs],
        relation_hints=[],
        constraint_memory={},
        authoritative_memory={},
        knowledge_base={},
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )
    details = build_qdi_table_card_details(
        file_summaries=[order_fs, cost_fs],
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )

    selected = _related_table_card_details_for_prompt(
        context=context,
        table_card_details=details,
        question_record={
            "question_id": "q1",
            "question": "How should transport cost be calculated?",
            "candidate_files": ["cost.csv"],
        },
        request=None,
        max_cards=1,
    )

    assert [card["table_id"] for card in selected] == ["cost.csv"]
    cost_fields = {field["name"]: field for field in selected[0]["fields"]}
    assert cost_fields["cost"]["numeric_stats"]["mean"] == 15.0
    stable_text = json.dumps(context, ensure_ascii=False, sort_keys=True)
    assert '"numeric_stats"' in stable_text
    assert '"mean": 15.0' in stable_text
    assert "raw_preview" not in stable_text


def test_table_manifest_can_omit_cards_but_context_retrieval_still_finds_details(tmp_path: Path) -> None:
    cards = [
        {
            "table_id": f"book.xlsx::sheet_{idx}",
            "source_file": "book.xlsx",
            "sheet_name": f"sheet_{idx}",
            "table_kind": "excel_sheet",
            "shape": [10, 2],
            "fields": [{"name": f"field_{idx}", "logical_type": "numeric"}],
        }
        for idx in range(6)
    ]
    stable_cards = compact_table_cards_for_prompt(cards, max_cards=3, per_source_limit=2)
    context = {"table_cards": stable_cards}
    details = {card["table_id"]: card for card in cards}

    assert any(card.get("table_kind") == "omitted_table_cards" for card in stable_cards)
    assert "book.xlsx::sheet_5" not in json.dumps(stable_cards, ensure_ascii=False)

    request = type(
        "Req",
        (),
        {
            "table_ids": [],
            "input_files": ["book.xlsx"],
            "focus_sheets": ["sheet_5"],
            "focus_columns": ["field_5"],
            "query": "need omitted sheet",
            "reason": "test",
        },
    )()
    retrieved = _retrieve_qdi_context_excerpt(
        context=context,
        table_card_details=details,
        question_record={"question": "What is in sheet_5?"},
        request=request,
        max_cards=1,
    )

    assert retrieved
    assert retrieved[0]["table_id"] == "book.xlsx::sheet_5"
