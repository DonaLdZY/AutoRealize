from autorealize.pipeline import _enforce_existing_file_references, _find_missing_file_references


def test_file_reference_validation_accepts_globs_that_match_real_inputs(tmp_path):
    costs = tmp_path / "成本"
    costs.mkdir()
    (costs / "承运商01 A 承运商成本.xlsx").write_bytes(b"fixture")
    (tmp_path / "Orders.XLSX").write_bytes(b"fixture")
    description = (
        "- 逐个读取 `成本/承运商* 承运商成本.xlsx`。\n"
        "- 订单来自 `orders.xlsx`。\n"
    )

    assert _find_missing_file_references(description, tmp_path) == []
    assert _enforce_existing_file_references(description, tmp_path) == description.rstrip("\n")


def test_file_reference_validation_still_rejects_unmatched_globs(tmp_path):
    (tmp_path / "orders.xlsx").write_bytes(b"fixture")
    description = (
        "- 保留 `orders.xlsx`。\n"
        "- 不存在 `成本/承运商* 承运商成本.xlsx`。\n"
    )

    assert _find_missing_file_references(description, tmp_path) == ["承运商* 承运商成本.xlsx"]
    assert _enforce_existing_file_references(description, tmp_path) == "- 保留 `orders.xlsx`。"
