from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from autorealize.config import AutoRealizeConfig
from autorealize.models import (
    ArtifactConsistencyIssue,
    ArtifactConsistencyPatch,
    ArtifactConsistencyReview,
    DownstreamContextResolution,
    DownstreamContextResolutionItem,
    FileRole,
    FileSummary,
)
from autorealize.modules.task_definition import TaskDefinitionModule
from autorealize.modules.types import RuntimeServices
from autorealize.pipeline import _infer_downstream_context


class _PromptManager:
    def load(self, _name: str) -> str:
        return "只选择输入候选。"


class _ResolutionLLM:
    def __init__(self, decision: DownstreamContextResolution) -> None:
        self.decision = decision
        self.calls = 0

    def ask_structured(self, **_kwargs):
        self.calls += 1
        return self.decision


def test_infer_downstream_context_handles_chinese_multisheet_and_placeholders(tmp_path) -> None:
    path = tmp_path / "业务数据.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            {"订单编号": ["A", "B"], "特征": [1, 2], "标签": [10.0, 20.0]}
        ).to_excel(writer, sheet_name="训练数据", index=False)
        pd.DataFrame(
            {"订单编号": ["C", "D"], "特征": [3, 4], "标签": ["待预测", "-"]}
        ).to_excel(writer, sheet_name="待预测数据", index=False)

    summary = FileSummary(
        path=path.name,
        role=FileRole.raw_data_table,
        summary="同一工作簿内分别存放训练与待预测数据。",
        columns=["订单编号", "特征", "标签"],
        column_semantics={"订单编号": "实体标识", "标签": "监督标签/真实值"},
    )
    context = _infer_downstream_context(tmp_path, [summary], "预测每个订单的标签", AutoRealizeConfig())

    assert context["train_table"] == path.name
    assert context["train_sheet"] == "训练数据"
    assert context["predict_table"] == path.name
    assert context["predict_sheet"] == "待预测数据"
    assert context["target_column"] == "标签"
    assert context["id_column"] == "订单编号"
    assert context["inference_candidates"]["train_table"]
    assert context["inference_resolution"]["train_table"]["selected_candidate_id"]


def test_downstream_llm_cannot_select_field_from_wrong_table(tmp_path) -> None:
    decision = DownstreamContextResolution(
        choices=[
            DownstreamContextResolutionItem(
                dimension="train_table",
                candidate_id="train_table:1",
                confidence=0.95,
                reason="权威说明指向 B",
            ),
            DownstreamContextResolutionItem(
                dimension="target_column",
                candidate_id="target_column:wrong",
                confidence=0.99,
                reason="字段语义相似",
            ),
        ]
    )
    llm = _ResolutionLLM(decision)
    services = RuntimeServices(
        llm_client=llm,
        prompt_mgr=_PromptManager(),
        registry=SimpleNamespace(),
        trajectory=SimpleNamespace(),
    )
    module = TaskDefinitionModule(AutoRealizeConfig(), services, tmp_path, tmp_path)
    context = {
        "train_table": "A.csv",
        "train_sheet": "",
        "predict_table": "",
        "target_column": "old_target",
        "y_true_field": "old_target",
        "authoritative_memory": {},
        "constraint_memory": {},
        "inference_candidates": {
            "train_table": [
                {
                    "candidate_id": "train_table:0",
                    "value": "A.csv",
                    "source_file": "A.csv",
                    "sheet_name": "",
                    "score": 0.6,
                },
                {
                    "candidate_id": "train_table:1",
                    "value": "B.csv",
                    "source_file": "B.csv",
                    "sheet_name": "",
                    "score": 0.8,
                },
            ],
            "target_column": [
                {
                    "candidate_id": "target_column:wrong",
                    "value": "target_a",
                    "source_file": "A.csv",
                    "sheet_name": "",
                    "score": 0.9,
                }
            ],
        },
        "inference_resolution": {
            "train_table": {
                "selected_candidate_id": "train_table:0",
                "selected_value": "A.csv",
                "confidence": 0.6,
                "unresolved": True,
            },
            "target_column": {
                "selected_candidate_id": "target_column:wrong",
                "selected_value": "old_target",
                "confidence": 0.6,
                "unresolved": True,
            },
        },
        "detected_table_units": [
            {"source_file": "A.csv", "sheet_name": "", "columns": ["target_a"]},
            {"source_file": "B.csv", "sheet_name": "", "columns": ["target_b"]},
        ],
    }

    resolved = module._resolve_downstream_context_semantics(
        data_root=tmp_path,
        task_hint="任务",
        original_text="B.csv 是训练表",
        downstream_context=context,
        file_summaries=[],
    )

    assert llm.calls == 1
    assert resolved["train_table"] == "B.csv"
    assert resolved["target_column"] == "old_target"
    rejected = resolved["downstream_context_resolution_report"]["rejected"]
    assert any(item["reason"] == "field_not_in_selected_train_table" for item in rejected)


def test_final_audit_rejects_patch_that_introduces_missing_file(tmp_path, monkeypatch) -> None:
    services = RuntimeServices(
        llm_client=SimpleNamespace(),
        prompt_mgr=_PromptManager(),
        registry=SimpleNamespace(),
        trajectory=SimpleNamespace(),
    )
    module = TaskDefinitionModule(AutoRealizeConfig(), services, tmp_path, tmp_path)
    review = ArtifactConsistencyReview(
        passed=False,
        issues=[
            ArtifactConsistencyIssue(
                issue_id="I1",
                severity="blocking",
                section="数据说明",
                message="需要修复",
                repair_target="description_section",
            )
        ],
    )
    patch = ArtifactConsistencyPatch(
        revised_sections={"数据说明": "## 数据说明\n- 从 `missing.csv` 读取。"},
        addressed_issue_ids=["I1"],
    )
    monkeypatch.setattr(module, "_review_final_artifact_consistency", lambda **_kwargs: review)
    monkeypatch.setattr(module, "_build_artifact_consistency_patch", lambda **_kwargs: patch)
    legacy = SimpleNamespace(
        _find_missing_file_references=lambda text, _root: ["missing.csv"] if "missing.csv" in text else [],
        _enforce_existing_file_references=lambda text, _root: text,
    )
    original = "# 任务\n\n## 数据说明\n- 从 `real.csv` 读取。\n"
    (tmp_path / "real.csv").write_text("x\n1\n", encoding="utf-8")

    final, _, _ = module._audit_and_repair_final_artifacts(
        desc=original,
        data_root=tmp_path,
        legacy=legacy,
        problem_review=None,  # type: ignore[arg-type]
        protocol_bundle=None,  # type: ignore[arg-type]
        evaluation_contract=None,  # type: ignore[arg-type]
        sample_spec=None,  # type: ignore[arg-type]
        submission_report={},
        automl_context_pack={},
        main_task_protocol={},
        downstream_context={},
        deterministic_defects=[],
        source_coverage_ledger={},
    )

    assert final == original
    report = (tmp_path / "artifact_consistency_report.json").read_text(encoding="utf-8")
    assert "patch_rejected" in report
    assert "missing.csv" in report
