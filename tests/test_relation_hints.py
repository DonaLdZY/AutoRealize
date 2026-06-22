from __future__ import annotations

from autorealize.models import FileRole, FileSummary
from autorealize.profiling.relations import detect_relations


def _profile(name: str, *, unique_count: int, row_count: int, logical_type: str = "text") -> dict:
    return {
        "name": name,
        "logical_type": logical_type,
        "row_count": row_count,
        "non_null_count": row_count,
        "unique_count": unique_count,
        "top_values": ["A:2", "B:1"],
    }


def test_detect_relations_infers_one_to_many_from_unique_ratios() -> None:
    orders = FileSummary(
        path="orders.xlsx::订单表信息",
        role=FileRole.raw_data_table,
        summary="订单主表",
        columns=["订单号", "客户"],
        column_semantics={"订单号": "订单唯一编号"},
        column_profiles=[_profile("订单号", unique_count=100, row_count=100)],
    )
    details = FileSummary(
        path="orders.xlsx::订单明细信息",
        role=FileRole.raw_data_table,
        summary="订单明细表",
        columns=["订单号", "商品"],
        column_semantics={"订单号": "订单编号，关联订单主表"},
        column_profiles=[_profile("订单号", unique_count=100, row_count=320)],
    )

    hints = detect_relations(
        {orders.path: orders.columns, details.path: details.columns},
        file_summaries=[orders, details],
    )

    order_hint = next(h for h in hints if h.left_field == "订单号" and h.right_field == "订单号")
    assert order_hint.relation_type == "one_to_many"
    assert order_hint.confidence > 0.7
    assert "左侧唯一率" in order_hint.short_evidence
    assert "右侧唯一率" in order_hint.short_evidence


def test_detect_relations_keeps_legacy_shared_columns_fields() -> None:
    left = FileSummary(
        path="a.csv",
        role=FileRole.raw_data_table,
        summary="",
        columns=["user_id"],
        column_profiles=[_profile("user_id", unique_count=3, row_count=10)],
    )
    right = FileSummary(
        path="b.csv",
        role=FileRole.raw_data_table,
        summary="",
        columns=["user_id"],
        column_profiles=[_profile("user_id", unique_count=4, row_count=12)],
    )

    hint = detect_relations({left.path: left.columns, right.path: right.columns}, file_summaries=[left, right])[0]

    assert hint.shared_columns == ["user_id"]
    assert hint.reason == hint.short_evidence
    assert hint.relation_type in {"many_to_many", "shared_attribute"}
