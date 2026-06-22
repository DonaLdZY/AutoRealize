from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from autorealize.config import AutoRealizeConfig
from autorealize.models import EvaluationContractReview, FileRole, FileSummary, SampleSubmissionSpec
from autorealize.modules.task_definition import TaskDefinitionModule


def _module(tmp_path: Path) -> TaskDefinitionModule:
    cfg = AutoRealizeConfig()
    services = SimpleNamespace(llm_client=None, prompt_mgr=None, registry=None, trajectory=None, knowledge_store=None)
    report_dir = tmp_path / "realize_report"
    report_dir.mkdir()
    return TaskDefinitionModule(cfg, services, tmp_path, report_dir)


def test_description_file_pack_excludes_raw_preview_and_keeps_sheet_inventory(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    fs = FileSummary(
        path="成本/承运商1成本.xlsx",
        role=FileRole.raw_data_table,
        summary="承运商成本规则与合同说明。",
        columns=["订单号", "成本"],
        column_semantics={"订单号": "订单关联键", "成本": "运输成本字段"},
        column_profiles=[
            {"name": "订单号", "logical_type": "categorical", "unique_count": 10, "null_ratio": 0.0},
            {"name": "成本", "logical_type": "numeric", "unique_count": 8, "null_ratio": 0.0},
        ],
        source_metadata={
            "shape": [100, 2],
            "preview": [{"订单号": "A001", "成本": 12.3}],
            "probe_results": {"very": "large"},
            "excel_sheet_profiles": [
                {
                    "sheet_name": "合同说明",
                    "shape": [20, 3],
                    "columns": ["合同", "计费方式", "备注"],
                    "raw_preview": [["合同A", "按阶梯计费"]],
                    "preview": [{"合同": "合同A"}],
                    "column_profiles": [{"name": "合同", "logical_type": "text", "unique_count": 2}],
                    "profile_policy": "deep_profile_all_small_workbook",
                }
            ],
            "sheet_field_descriptions": {"合同说明": {"计费方式": "合同计费规则文字说明"}},
        },
    )

    packed = mod._compact_file_for_sections(fs, include_profiles=True)

    assert packed["shape"] == [100, 2]
    assert packed["sheets"][0]["sheet_name"] == "合同说明"
    assert packed["sheets"][0]["field_semantics"]["计费方式"] == "合同计费规则文字说明"
    assert "preview" not in str(packed)
    assert "probe_results" not in str(packed)


def test_compose_description_sections_uses_required_order(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    desc = mod._compose_description_sections(
        [
            "## 任务概述\n- A",
            "## 任务定义\n- B",
            "## 评估协议\n- C",
            "## 输出或提交格式\n- D",
            "## 数据说明\n- E",
            "## 关键字段说明\n- F",
            "## 约束与防泄漏\n- G",
            "## 关键坑点与待确认事项\n- H",
        ]
    )

    headers = [line for line in desc.splitlines() if line.startswith("## ")]
    assert headers == [
        "## 任务概述",
        "## 任务定义",
        "## 评估协议",
        "## 输出或提交格式",
        "## 数据说明",
        "## 关键字段说明",
        "## 约束与防泄漏",
        "## 关键坑点与待确认事项",
    ]


def test_artifact_sanity_checks_sample_columns(tmp_path: Path) -> None:
    mod = _module(tmp_path)
    (tmp_path / "sample_submission.csv").write_text("id,pred\n1,0\n", encoding="utf-8")
    data_root = tmp_path / "input"
    data_root.mkdir()
    spec = SampleSubmissionSpec(columns=["id", "target"])
    contract = EvaluationContractReview(primary_metric="score", metric_direction="minimize")
    legacy = SimpleNamespace(_find_missing_file_references=lambda desc, root: [])

    defects = mod._artifact_sanity_check(
        desc="# 赛题说明\n\n## 任务概述\n- test",
        data_root=data_root,
        sample_spec=spec,
        evaluation_contract=contract,
        legacy=legacy,
    )

    assert any("sample_columns_mismatch" in item for item in defects)
