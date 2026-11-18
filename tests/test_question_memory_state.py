from copy import deepcopy

from autorealize.investigation import (
    _host_verified_full_scan_files,
    _prior_qdi_answers_for_prompt,
    reconcile_question_memory,
)
from autorealize.models import FileRole, FileSummary
from autorealize.profiling.full_scan import scan_csv_time_column


def test_remaining_uncertainty_is_not_lost_when_answer_is_marked_resolved():
    memory = {
        "summary": "2 resolved, 0 unresolved", "unresolved_questions": [],
        "answers": [
            {"question_id": "q1", "answer": "Known schema", "remaining_uncertainty": ""},
            {"question_id": "q2", "answer": "No official new-data contract found",
             "remaining_uncertainty": "New file format and identity policy remain unknown"},
        ],
        "question_records": [{"question_id": qid, "status": "resolved"} for qid in ("q1", "q2")],
        "progress": {"resolved_questions": 2, "unresolved_questions": 0},
    }
    before = deepcopy(memory)
    result = reconcile_question_memory(memory)
    assert memory == before
    assert result["answers"] == before["answers"]
    assert result["question_records"][1]["status"] == "answered_with_uncertainty"
    assert result["progress"]["resolved_questions"] == 1
    assert result["progress"]["unresolved_questions"] == 1
    assert "identity policy" in result["unresolved_questions"][0]
    assert reconcile_question_memory(result) == result


def test_latest_answer_can_resolve_earlier_uncertainty_without_rewriting_history():
    memory = {"answers": [
        {"question_id": "q1", "remaining_uncertainty": "Unknown row count"},
        {"question_id": "q1", "remaining_uncertainty": "", "answer": "Full parse verified"},
    ], "unresolved_questions": [], "question_records": [{"question_id": "q1", "status": "resolved"}]}
    assert reconcile_question_memory(memory) == memory


def test_prior_qdi_prompt_context_obeys_configured_answer_limit():
    memory = {
        "answers": [
            {"question_id": f"q{index}", "question": f"Question {index}", "answer": f"Answer {index}"}
            for index in range(12)
        ]
    }

    bounded = _prior_qdi_answers_for_prompt(memory, 3)

    assert [item["question_id"] for item in bounded] == ["q9", "q10", "q11"]
    assert _prior_qdi_answers_for_prompt(memory, 0) == []


def test_model_reported_full_scan_cannot_retire_full_population_uncertainty():
    stale = "设备表全表时间范围与总行数未实测（本次仅前 50000 行）；标签重复组冲突数尚未统计。"
    memory = {
        "answers": [{
            "question_id": "old",
            "answer": "Partial scan",
            "remaining_uncertainty": stale,
            "used_files": ["设备.csv"],
        }],
        "unresolved_questions": [f"[old] Remaining uncertainty: {stale}"],
        "question_records": [{
            "question_id": "old",
            "status": "resolved",
            "unresolved_reason": stale,
            "used_files": ["设备.csv"],
        }],
        "tool_requests": [{
            "request_id": "req-full",
            "custom_python": {"input_files": ["设备.csv"]},
        }],
        "step_results": [{
            "request_id": "req-full",
            "status": "completed",
            "result": {
                "row_count_total": 730135,
                "sample_time_min": "2025-01-01 00:00:19",
                "sample_time_max": "2026-05-25 16:51:36",
            },
        }],
    }

    result = reconcile_question_memory(memory)

    assert result["answers"][0]["remaining_uncertainty"] == stale
    assert result["unresolved_questions"] == [f"[old] Remaining uncertainty: {stale}"]
    assert result["question_records"][0]["unresolved_reason"] == stale
    assert "evidence_reconciliations" not in result


def test_current_host_scan_retires_only_its_bound_row_and_time_range_gap(tmp_path):
    source = tmp_path / "设备.csv"
    source.write_text("time,value\n2026-01-01 00:00:00,1\n2026-01-01 00:01:00,2\n", encoding="utf-8")
    read_kwargs = {"sep": ",", "encoding": "utf-8"}
    scan = scan_csv_time_column(source, column="time", read_kwargs=read_kwargs)
    scan.update(file="设备.csv", source_fingerprint="source-v1")
    file_summary = FileSummary(
        path="设备.csv",
        role=FileRole.raw_data_table,
        summary="CSV table",
        columns=["time", "value"],
        source_metadata={
            "source_fingerprint": "source-v1",
            "csv_read_contract": {"pandas_kwargs": read_kwargs},
            "full_table_scan": scan,
        },
    )
    verified_scans = _host_verified_full_scan_files([file_summary])
    stale = "设备表全表时间范围与总行数未实测（本次仅前 50000 行）；设备.csv 全表数据内容尚未完成核验。"
    memory = {
        "answers": [{
            "question_id": "old",
            "answer": "Prior result",
            "remaining_uncertainty": stale,
            "used_files": ["设备.csv"],
        }],
        "unresolved_questions": [f"[old] Remaining uncertainty: {stale}"],
        "question_records": [{
            "question_id": "old",
            "status": "answered_with_uncertainty",
            "unresolved_reason": stale,
            "used_files": ["设备.csv"],
        }],
    }

    result = reconcile_question_memory(memory, verified_full_scans=verified_scans)

    assert result["answers"][0]["remaining_uncertainty"] == "设备.csv 全表数据内容尚未完成核验"
    assert result["question_records"][0]["unresolved_reason"] == "设备.csv 全表数据内容尚未完成核验"
    assert len(result["evidence_reconciliations"]) == 1
    reconciliation = result["evidence_reconciliations"][0]
    assert reconciliation["source_file"] == "设备.csv"
    assert reconciliation["source_fingerprint"] == "source-v1"
    assert reconciliation["rows_scanned"] == 2


def test_full_scan_reconciliation_does_not_borrow_used_file_for_other_file_claim():
    stale = (
        "设备.csv 全表时间范围尚未实测；weather.csv 全表时间范围尚未实测。"
    )
    memory = {
        "summary": "1 resolved, 0 unresolved",
        "answers": [{
            "question_id": "q1",
            "remaining_uncertainty": stale,
            "used_files": ["设备.csv"],
        }],
        "question_records": [{
            "question_id": "q1",
            "status": "answered_with_uncertainty",
            "unresolved_reason": stale,
            "used_files": ["设备.csv"],
        }],
        "unresolved_questions": [f"[q1] Remaining uncertainty: {stale}"],
        "progress": {"answered_questions": 1, "resolved_questions": 0, "unresolved_questions": 1},
    }
    scans = {
        "设备.csv": {
            "source_fingerprint": "device-v1",
            "read_options_fingerprint": "options-v1",
            "rows_scanned": 10,
        },
    }

    result = reconcile_question_memory(memory, verified_full_scans=scans)

    assert result["answers"][0]["remaining_uncertainty"] == "weather.csv 全表时间范围尚未实测"
    assert result["question_records"][0]["unresolved_reason"] == "weather.csv 全表时间范围尚未实测"
    assert result["unresolved_questions"] == [
        "[q1] Remaining uncertainty: weather.csv 全表时间范围尚未实测"
    ]
    assert result["progress"] == {
        "answered_questions": 1,
        "resolved_questions": 0,
        "unresolved_questions": 1,
    }
    assert result["summary_before_uncertainty_reconciliation"] == "1 resolved, 0 unresolved"


def test_full_scan_reconciliation_refreshes_status_summary_and_progress_when_last_gap_closes():
    stale = "设备.csv 全表时间范围尚未实测，仅检查前 50000 行。"
    memory = {
        "summary": "1 resolved, 0 unresolved",
        "answers": [{
            "question_id": "q1",
            "remaining_uncertainty": stale,
            "used_files": ["设备.csv"],
        }],
        "question_records": [{
            "question_id": "q1",
            "status": "answered_with_uncertainty",
            "unresolved_reason": stale,
            "used_files": ["设备.csv"],
        }],
        "unresolved_questions": [f"[q1] Remaining uncertainty: {stale}"],
        "progress": {"answered_questions": 1, "resolved_questions": 0, "unresolved_questions": 1},
    }

    result = reconcile_question_memory(memory, verified_full_scans={
        "设备.csv": {
            "source_fingerprint": "device-v1",
            "read_options_fingerprint": "options-v1",
            "rows_scanned": 10,
        },
    })

    assert result["answers"][0]["remaining_uncertainty"] == ""
    assert result["question_records"][0]["status"] == "resolved"
    assert result["unresolved_questions"] == []
    assert result["progress"] == {
        "answered_questions": 1,
        "resolved_questions": 1,
        "unresolved_questions": 0,
    }
    assert "0 个仍有明确不确定性" in result["summary"]


def test_host_scan_with_stale_source_or_read_options_is_not_trusted(tmp_path):
    source = tmp_path / "设备.csv"
    source.write_text("time,value\n2026-01-01 00:00:00,1\n", encoding="utf-8")
    read_kwargs = {"sep": ",", "encoding": "utf-8"}
    scan = scan_csv_time_column(source, column="time", read_kwargs=read_kwargs)
    scan.update(file="设备.csv", source_fingerprint="source-old")
    summary = FileSummary(
        path="设备.csv",
        role=FileRole.raw_data_table,
        summary="CSV table",
        columns=["time", "value"],
        source_metadata={
            "source_fingerprint": "source-current",
            "csv_read_contract": {"pandas_kwargs": {**read_kwargs, "decimal": ","}},
            "full_table_scan": scan,
        },
    )

    assert _host_verified_full_scan_files([summary]) == {}


def test_sampled_scan_does_not_retire_full_scan_gap():
    memory = {
        "unresolved_questions": ["设备.csv 全表时间范围尚未实测，仅扫描前 50000 行。"],
        "tool_requests": [{
            "request_id": "req-sample",
            "custom_python": {"input_files": ["设备.csv"]},
        }],
        "step_results": [{
            "request_id": "req-sample",
            "status": "completed",
            "result": {
                "rows_observed": 50000,
                "sample_time_min": "2026-04-20",
                "sample_time_max": "2026-05-25",
            },
        }],
    }

    result = reconcile_question_memory(memory)

    assert result["unresolved_questions"] == memory["unresolved_questions"]
    assert "evidence_reconciliations" not in result
