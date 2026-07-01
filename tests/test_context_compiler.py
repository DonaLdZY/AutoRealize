from __future__ import annotations

import json
from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.context_compiler import (
    ArtifactStore,
    build_qdi_context_bundle,
    build_qdi_table_card_details,
    compact_table_cards_for_prompt,
    context_telemetry,
)
from autorealize.investigation import (
    _entity_alias_verification_questions,
    _related_table_card_details_for_prompt,
    _retrieve_qdi_context_excerpt,
)
from autorealize.models import FileRole, FileSummary
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
    assert context["table_cards"][0]["field_hints"]
    assert "field_index" not in context["table_cards"][0]
    assert "fields" not in context["table_cards"][0]
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
    assert "numeric_stats" not in text
    assert "top_values" not in text
    assert "reading_notes" not in text
    assert "file_cognition" not in text
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


def test_entity_alias_candidates_feed_qdi_validation_questions(tmp_path: Path) -> None:
    files = [
        FileSummary(
            path="contracts.xlsx",
            role=FileRole.raw_data_table,
            summary="成本合同表。",
            columns=[],
            source_metadata={
                "excel_sheet_profiles": [
                    {
                        "sheet_name": "承运商成本合同表",
                        "shape": [8, 5],
                        "columns": ["结算方代码", "结算方名称", "起点", "终点", "车型"],
                    }
                ]
            },
        ),
        FileSummary(
            path="vehicles.xlsx",
            role=FileRole.raw_data_table,
            summary="每日车辆表。",
            columns=["承运商代码", "承运商名称", "车型"],
            source_metadata={"shape": [4, 3]},
        ),
    ]

    context = build_qdi_context_bundle(
        cfg=AutoRealizeConfig(),
        data_root=tmp_path,
        task_hint="验证车辆和合同线路是否可用。",
        file_summaries=files,
        relation_hints=[],
        constraint_memory={},
        authoritative_memory={},
        knowledge_base={},
    )

    groups = context["entity_alias_candidates"]
    assert groups
    group = groups[0]
    assert group["status"] == "candidate_not_equivalent"
    fields = {item["field"] for item in group["candidate_fields"]}
    assert {"结算方代码", "承运商代码"} <= fields

    questions = _entity_alias_verification_questions(context)
    assert len(questions) == 1
    assert questions[0].category == "join_key"
    assert questions[0].priority == "high"
    assert "join coverage" in questions[0].question
    assert "线路/车型/成本可行性覆盖率" in questions[0].question


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
    assert "numeric_stats" not in stable_text


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
