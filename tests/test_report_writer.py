from autorealize.report_writer import coverage_defects, description_quality_check


def test_coverage_defects_detect_short_output() -> None:
    original = "这是一个非常详细的需求说明，包含评价指标、submission格式、字段说明和业务约束。" * 30
    generated = "简短描述"
    defects = coverage_defects(generated, original)
    assert any("长度明显短于原始需求" in d for d in defects)


def test_description_quality_check_detects_missing_sections() -> None:
    text = "# description\n\n## Overview\nx"
    defects = description_quality_check(text)
    assert any("缺少章节" in d for d in defects)


def test_description_quality_check_detects_internal_pipeline_language() -> None:
    text = (
        "## Overview\nx\n"
        "## Data Inventory\nx\n"
        "## Task Definition\nP1 数据认知\n"
        "## Evaluation\n### Formal Formula\nx\n### Computation Scope\nx\n### Validation Protocol\nx\n### Reporting Rules\nx\n"
        "## Submission Format\nx\n## Modeling Boundary\nx\n## Constraints & Risks\nx\n"
    )
    defects = description_quality_check(text)
    assert any("内部的流程描述" in d for d in defects)


def test_description_quality_check_detects_ambiguous_words() -> None:
    text = (
        "## Overview\nx\n"
        "## Data Inventory\nx\n"
        "## Task Definition\nx\n"
        "## Evaluation\n### Formal Formula\nx\n### Computation Scope\nx\n### Validation Protocol\n推荐使用滚动窗口\n### Reporting Rules\nx\n"
        "## Submission Format\nx\n## Modeling Boundary\nx\n## Constraints & Risks\nx\n"
    )
    defects = description_quality_check(text)
    assert any("评估歧义措辞" in d for d in defects)
