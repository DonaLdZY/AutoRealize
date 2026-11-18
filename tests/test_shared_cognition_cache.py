from __future__ import annotations

import json
import os
from types import SimpleNamespace

import autorealize.pipeline as legacy
from autorealize.config import AutoRealizeConfig
from autorealize.models import FileRole, FileSummary
from autorealize.models import InvestigationQuestion, QuestionInvestigationAction, QuestionInvestigationPlan
from autorealize.modules.data_cognition import (
    DataCognitionModule,
    FILE_COGNITION_SCOPE,
    _annotate_prior_qdi_memory,
    _source_inventory_fingerprint,
)
from autorealize.modules.types import RuntimeServices
from autorealize.prompts.manager import PromptManager


def _module(tmp_path):
    module = object.__new__(DataCognitionModule)
    module.report_dir = tmp_path / "report"
    module.report_dir.mkdir()
    module.config = SimpleNamespace()
    module.services = SimpleNamespace()
    return module


def test_source_inventory_tracks_content_fingerprint_and_mtime(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    source = root / "history.csv"
    source.write_text("value\n1\n", encoding="utf-8")

    module = _module(tmp_path)
    first = module._source_inventory(root)["history.csv"]
    source.touch()
    second = module._source_inventory(root)["history.csv"]
    assert second["mtime_ns"] >= first["mtime_ns"]
    assert second["fingerprint"] == first["fingerprint"]

    source.write_text("value\n2\n", encoding="utf-8")
    third = module._source_inventory(root)["history.csv"]
    assert third["fingerprint"] != first["fingerprint"]
    assert third["sha256"] != first["sha256"]


def test_incremental_delta_bypasses_representative_sampling(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    selected = [root / "batch-001.csv"]
    changed = root / "batch-999.csv"
    changed.write_text("value\n9\n", encoding="utf-8")

    result, forced = DataCognitionModule._include_incremental_sources(
        data_root=root,
        selected_files=selected,
        new_files=["batch-999.csv"],
        changed_files=[],
        cache_available=True,
    )
    assert [path.name for path in result] == ["batch-001.csv", "batch-999.csv"]
    assert forced == ["batch-999.csv"]

    first_run, first_forced = DataCognitionModule._include_incremental_sources(
        data_root=root,
        selected_files=selected,
        new_files=["batch-999.csv"],
        changed_files=[],
        cache_available=False,
    )
    assert first_run == selected
    assert first_forced == []


def test_prior_qdi_memory_marks_missing_and_changed_sources(tmp_path):
    inventory = {
        "stable.csv": {"fingerprint": "stable-v2"},
        "changed.csv": {"fingerprint": "changed-v2"},
    }
    memory = {
        "answers": [
            {"question": "stable", "answer": "yes", "source_fingerprints": {"stable.csv": "stable-v2"}},
            {"question": "changed", "answer": "old", "source_fingerprints": {"changed.csv": "changed-v1"}},
            {"question": "removed", "answer": "old", "source_fingerprints": {"removed.csv": "removed-v1"}},
            {"question": "unknown", "answer": "uncertain"},
        ]
    }
    result = _annotate_prior_qdi_memory(memory, inventory)
    statuses = {item["question"]: item["source_status"] for item in result["answers"]}
    assert statuses == {"stable": "unchanged", "changed": "changed", "removed": "changed", "unknown": "unknown"}
    assert result["answers"][1]["changed_source_files"] == ["changed.csv"]
    assert result["answers"][2]["changed_source_files"] == ["removed.csv"]


def test_prior_qdi_memory_tracks_inventory_changes_separately_from_cited_sources():
    inventory = {"stable.csv": {"fingerprint": "stable-v1"}}
    memory = {
        "answers": [
            {
                "question": "stable source question",
                "answer": "stable answer",
                "source_fingerprints": {"stable.csv": "stable-v1"},
                "source_inventory_sha256": _source_inventory_fingerprint(inventory),
                "qdi_signature": "qdi-v1",
            },
            {
                "question": "legacy cache without inventory fingerprint",
                "answer": "uncertain",
                "source_fingerprints": {"stable.csv": "stable-v1"},
            },
        ]
    }

    unchanged = _annotate_prior_qdi_memory(memory, inventory, qdi_signature="qdi-v1")
    assert unchanged["answers"][0]["source_status"] == "unchanged"
    assert unchanged["answers"][0]["source_inventory_status"] == "unchanged"
    assert unchanged["answers"][0]["qdi_contract_status"] == "unchanged"
    assert unchanged["answers"][1]["source_inventory_status"] == "unknown"
    assert unchanged["answers"][1]["qdi_contract_status"] == "unknown"

    changed = _annotate_prior_qdi_memory(
        memory,
        {**inventory, "new.csv": {"fingerprint": "new-v1"}},
        qdi_signature="qdi-v2",
    )
    assert changed["answers"][0]["source_status"] == "unchanged"
    assert changed["answers"][0]["source_inventory_status"] == "changed"
    assert changed["answers"][0]["qdi_contract_status"] == "changed"


def test_cognition_signature_covers_probe_prompt_and_generation_parameters():
    from autorealize.config import AutoRealizeConfig

    config = AutoRealizeConfig()
    prompt_values = {
        "system/cognition_probe_planner.md": "probe-v1",
        "system/cognition_summarizer.md": "summary-v1",
        "system/cognition_document_compactor.md": "compactor-v1",
        "system/question_investigator_planner.md": "qdi-planner-v1",
        "system/question_investigator_answerer_v7.md": "qdi-answerer-v1",
        "system/question_investigator_script_repair.md": "qdi-script-repair-v1",
    }

    class Prompts:
        def load(self, name):
            return prompt_values[name]

    module = object.__new__(DataCognitionModule)
    module.config = config
    module.services = SimpleNamespace(prompt_mgr=Prompts())
    initial = module._cognition_signature()
    prompt_values["system/cognition_probe_planner.md"] = "probe-v2"
    assert module._cognition_signature() != initial

    prompt_values["system/cognition_probe_planner.md"] = "probe-v1"
    initial = module._cognition_signature()
    config.llm.temperature = 0.25
    assert module._cognition_signature() != initial

    initial = module._qdi_signature()
    config.llm.api_key = "must-not-change-signature"
    assert module._qdi_signature() == initial
    prompt_values["system/question_investigator_planner.md"] = "qdi-planner-v2"
    assert module._qdi_signature() != initial


def test_save_shared_cache_tolerates_legacy_shapes_and_deduplicates_qdi(tmp_path):
    module = _module(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cognition_cache.json").write_text(
        json.dumps({
            "schema_version": "autorealize.shared_cognition_cache.v2",
            "files": [],
            "question_investigation": {"answers": {}},
        }),
        encoding="utf-8",
    )
    summary = FileSummary(path="data.csv", role=FileRole.raw_data_table, summary="table")
    inventory = {
        "data.csv": {
            "path": "data.csv", "size": 5, "mtime_ns": 1, "mtime": 1.0,
            "sha256": "sha", "fingerprint": "fp",
        }
    }
    module._save_shared_cache(
        cache_dir,
        inventory,
        [summary],
        {
            "answers": [{"question_id": "q1", "question": "What?", "answer": "data.csv"}],
            "question_records": [{"question_id": "q1", "used_files": ["data.csv"]}],
        },
        cognition_signature="sig",
        task_hint="predict target",
    )
    payload = json.loads((cache_dir / "cognition_cache.json").read_text(encoding="utf-8"))
    assert payload["files"]["data.csv"]["fingerprint"] == "fp"
    assert len(payload["question_investigation"]["answers"]) == 1
    assert payload["question_investigation"]["answers"][0]["source_fingerprints"] == {"data.csv": "fp"}


def test_save_shared_cache_preserves_deleted_and_all_qdi_source_fingerprints(tmp_path):
    module = _module(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    prior_paths = ["removed.csv", *(f"part-{index}.csv" for index in range(25))]
    previous_inventory = {
        path: {"path": path, "fingerprint": f"fp:{path}"}
        for path in prior_paths
    }
    (cache_dir / "cognition_cache.json").write_text(
        json.dumps({
            "schema_version": "autorealize.shared_cognition_cache.v2",
            "source_inventory": previous_inventory,
            "files": {},
            "question_investigation": {"answers": []},
        }),
        encoding="utf-8",
    )
    current_inventory = {path: value for path, value in previous_inventory.items() if path != "removed.csv"}
    summary = FileSummary(path="part-0.csv", role=FileRole.raw_data_table, summary="table")
    module._save_shared_cache(
        cache_dir,
        current_inventory,
        [summary],
        {
            "answers": [{"question_id": "q1", "question": "Which sources?", "answer": "reviewed"}],
            "question_records": [{"question_id": "q1", "used_files": prior_paths}],
        },
        cognition_signature="sig",
        task_hint="compare batches",
    )

    payload = json.loads((cache_dir / "cognition_cache.json").read_text(encoding="utf-8"))
    answer = payload["question_investigation"]["answers"][0]
    assert answer["source_fingerprints"] == {path: f"fp:{path}" for path in prior_paths}
    status = _annotate_prior_qdi_memory(payload["question_investigation"], current_inventory)["answers"][0]
    assert status["source_status"] == "changed"
    assert status["changed_source_files"] == ["removed.csv"]


def test_multiple_research_runs_reuse_files_and_send_prior_evidence_to_real_qdi(tmp_path, monkeypatch):
    """Exercise actual module orchestration, cache IO and QDI with a local LLM fixture."""
    root = tmp_path / "data"
    root.mkdir()
    for name, value in [("stable.csv", 1), ("change.csv", 2), ("removed.csv", 3)]:
        (root / name).write_text(f"value\n{value}\n", encoding="utf-8")
    cache = tmp_path / "shared-cache"
    monkeypatch.setenv("AUTOREALIZE_SHARED_CACHE_DIR", str(cache))
    config = AutoRealizeConfig()
    config.parallel.enable_parallel_cognition = False
    config.data.enable_llm_filename_grouping = False
    config.investigation.trigger_mode = "always"
    config.investigation.automatic_verification_question_limit = 0
    config.investigation.max_questions = 1
    recognized = []

    def cognize(*, file, task_hint, **kwargs):
        recognized.append((file.name, task_hint))
        summary = FileSummary(
            path=file.name, role=FileRole.raw_data_table,
            summary=file.read_text(encoding="utf-8"), columns=["value"],
        )
        return {"fs": summary, "rpath": file.name, "columns": ["value"], "is_requirement": False}

    monkeypatch.setattr(legacy, "_cognize_one_file", cognize)

    class Client:
        def __init__(self):
            self.calls = []

        def ask_structured(self, *, model_cls, **kwargs):
            self.calls.append({"model_cls": model_cls, **kwargs})
            if model_cls is QuestionInvestigationPlan:
                return QuestionInvestigationPlan(questions=[InvestigationQuestion(
                    question_id="q1", question="Which observations are current?", candidate_files=["stable.csv"],
                )])
            if model_cls is QuestionInvestigationAction:
                return QuestionInvestigationAction(
                    action="answer", answer="See the current file cards, not an inferred model result.",
                    used_files=[path.name for path in sorted(root.glob("*.csv"))],
                    evidence=["file cards"],
                )
            return model_cls()

    def run(index, task_hint):
        report = tmp_path / f"report-{index}"
        report.mkdir()
        client = Client()
        services = RuntimeServices(
            llm_client=client, prompt_mgr=PromptManager(config), registry=None,
            trajectory=SimpleNamespace(log=lambda *args: None),
        )
        module = DataCognitionModule(config, services, report)
        # A deterministic representative selector deliberately omits delta
        # files in the third run, just like a repeated-filename batch can do.
        candidates = sorted(root.glob("*.csv")) if index < 3 else [root / "stable.csv"]
        monkeypatch.setattr(module, "_select_files_with_pattern_sampling", lambda _: (candidates, {}, [], []))
        result = module.run(root, task_hint)
        payload = json.loads((report / "data_cognition_report.json").read_text(encoding="utf-8"))
        return result, payload, client

    _, first, first_client = run(1, "Forecast output A")
    assert first["incremental"]["reused_files"] == []
    assert {name for name, _ in recognized} == {"stable.csv", "change.csv", "removed.csv"}
    assert {hint for _, hint in recognized} == {FILE_COGNITION_SCOPE}
    assert len([call for call in first_client.calls if call["model_cls"] is QuestionInvestigationPlan]) == 1

    recognized.clear()
    _, second, second_client = run(2, "Forecast a different output B")
    assert recognized == []
    assert second["incremental"]["reused_files"] == ["change.csv", "removed.csv", "stable.csv"]
    assert second["question_investigation"]["prior_memory"]["answers"][0]["source_status"] == "unchanged"
    assert second["question_investigation"]["prior_memory"]["answers"][0]["qdi_contract_status"] == "unchanged"
    assert len([call for call in second_client.calls if call["model_cls"] is QuestionInvestigationAction]) == 1

    old_stat = (root / "change.csv").stat()
    (root / "change.csv").write_text("value\n9\n", encoding="utf-8")
    # Detect content changes even when size AND mtime happen to match.
    os.utime(root / "change.csv", ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    (root / "stable.csv").touch()
    (root / "removed.csv").unlink()
    (root / "new.csv").write_text("value\n4\n", encoding="utf-8")
    recognized.clear()
    result, third, third_client = run(3, "Forecast output C")
    assert {name for name, _ in recognized} == {"change.csv", "new.csv"}
    delta = third["incremental"]
    assert delta["reused_files"] == ["stable.csv"]
    assert delta["changed_files"] == ["change.csv"]
    assert delta["new_files"] == ["new.csv"]
    assert delta["removed_files"] == ["removed.csv"]
    assert delta["incremental_forced_files"] == ["change.csv", "new.csv"]
    assert {summary.path for summary in result.file_summaries} == {"stable.csv", "change.csv", "new.csv"}
    assert next(summary for summary in result.file_summaries if summary.path == "change.csv").summary == "value\n9\n"
    old_answer = third["question_investigation"]["prior_memory"]["answers"][0]
    assert old_answer["source_status"] == "changed"
    assert old_answer["source_inventory_status"] == "changed"
    assert old_answer["qdi_contract_status"] == "unchanged"
    assert set(old_answer["changed_source_files"]) == {"change.csv", "removed.csv"}
    planner = next(call for call in third_client.calls if call["model_cls"] is QuestionInvestigationPlan)
    assert '"source_status":"changed"' in planner["dynamic_user_prompt"]
    assert "new.csv" in planner["dynamic_user_prompt"]
    assert "removed.csv" in planner["dynamic_user_prompt"]
    assert "hypotheses" in planner["static_context_prompt"]
    persisted = json.loads((cache / "cognition_cache.json").read_text(encoding="utf-8"))
    assert "removed.csv" not in persisted["files"]
    assert "removed.csv" not in persisted["source_inventory"]

    # Changed model/reading contract invalidates summaries even for stable data.
    recognized.clear()
    config.llm.model_name += "-new-contract"
    _, fourth, _ = run(4, "Forecast output D")
    assert recognized == [("stable.csv", FILE_COGNITION_SCOPE)]
    assert fourth["incremental"]["reused_files"] == []
