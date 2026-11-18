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


def test_generated_artifacts_are_not_checked_as_input_files(tmp_path):
    (tmp_path / "设备.csv").write_text("SampleTime,value\n", encoding="utf-8")
    description = (
        "## 数据说明\n"
        "- 读取 `设备.csv`。\n"
        "## 输出或提交格式\n"
        "- 交付 `selected_config.json` 和 `frozen_model.joblib`。\n"
    )

    assert _find_missing_file_references(description, tmp_path) == []
    assert _enforce_existing_file_references(description, tmp_path) == description.rstrip("\n")


def test_missing_source_reference_is_still_rejected_outside_output_section(tmp_path):
    description = (
        "## 数据说明\n"
        "- 读取 `missing_source.csv`。\n"
        "## 输出或提交格式\n"
        "- 生成 `selected_config.json`。\n"
    )

    assert _find_missing_file_references(description, tmp_path) == ["missing_source.csv"]
    assert _enforce_existing_file_references(description, tmp_path) == (
        "## 数据说明\n## 输出或提交格式\n- 生成 `selected_config.json`。"
    )


def test_package_artifact_dependencies_are_valid_in_any_description_section(tmp_path):
    (tmp_path / "设备.csv").write_text("SampleTime,value\n", encoding="utf-8")
    description = (
        "## 关键字段说明\n"
        "- 训练配方从 `artifact/selected_config.json` 加载。\n"
        "- 推理模型来自 `artifacts/frozen_model.joblib`。\n"
    )

    assert _find_missing_file_references(description, tmp_path) == []
    assert _enforce_existing_file_references(description, tmp_path) == description.rstrip("\n")


def test_package_artifact_filename_is_not_mistaken_for_input_when_root_is_named(tmp_path):
    description = (
        "## 评估协议\n"
        "- `final_evaluate(data, artifact_dir)` 从传入的 artifact_dir 读取 `selected_config.json`。\n"
    )

    assert _find_missing_file_references(description, tmp_path) == []
    assert _enforce_existing_file_references(description, tmp_path) == description.rstrip("\n")


def test_unlisted_package_file_without_artifact_path_is_still_checked_as_input(tmp_path):
    description = (
        "## 数据说明\n"
        "- 读取 `selected_config.json` 作为输入参数。\n"
    )

    assert _find_missing_file_references(description, tmp_path) == ["selected_config.json"]
