from __future__ import annotations

from pathlib import Path

import pandas as pd

from autorealize.context_compiler import build_table_cards, compact_table_cards_for_prompt, reconcile_table_shape
from autorealize.models import FileRole, FileSummary
from autorealize.modules.data_cognition import (
    _full_scan_targets,
    _scan_requested_csvs,
    _temporal_scan_requirement,
)
from autorealize.profiling.full_scan import (
    FULL_TABLE_SCAN_VERSION,
    csv_scan_read_options_fingerprint,
    scan_csv_time_column,
    verified_full_scan_population,
)
from autorealize.report_writer import build_data_access_protocol, build_data_schema_contract, write_data_description
from autorealize.profiling.relations import RelationHint


def _device_file(path: Path) -> None:
    pd.DataFrame(
        {
            "SampleTime": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
                "bad timestamp",
                "2026-01-01 00:03:00",
                "2026-01-01 00:04:00",
            ],
            "flow": [1, 2, 3, 4, 5],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_chunk_scan_counts_full_file_and_time_range(tmp_path: Path) -> None:
    source = tmp_path / "设备.csv"
    _device_file(source)

    result = scan_csv_time_column(
        source,
        column="SampleTime",
        read_kwargs={"encoding": "utf-8-sig", "sep": ","},
        chunk_rows=2,
    )

    assert result["status"] == "completed"
    assert result["rows_scanned"] == 5
    assert result["valid_timestamp_rows"] == 4
    assert result["invalid_timestamp_rows"] == 1
    assert result["chunks_scanned"] == 3
    assert result["read_options"]["encoding"] == "utf-8-sig"
    assert len(result["read_options_fingerprint"]) == 64
    assert result["minimum_timestamp"].startswith("2026-01-01T00:00:00")
    assert result["maximum_timestamp"].startswith("2026-01-01T00:04:00")


def test_chunk_scan_does_not_merge_naive_and_offset_timestamp_ranges(tmp_path: Path) -> None:
    source = tmp_path / "mixed-timezones.csv"
    source.write_text(
        "time,value\n"
        "2026-01-01 00:00:00,1\n"
        "2025-12-31T19:30:00-05:00,2\n",
        encoding="utf-8",
    )

    result = scan_csv_time_column(source, column="time", chunk_rows=1)

    assert result["status"] == "completed"
    assert result["mixed_timezone_awareness"] is True
    assert result["minimum_timestamp"] is None
    assert result["maximum_timestamp"] is None
    ranges = result["timestamp_ranges_by_timezone_awareness"]
    assert ranges["naive"]["minimum_timestamp"] == "2026-01-01T00:00:00"
    assert ranges["aware_utc"]["minimum_timestamp"] == "2026-01-01T00:30:00+00:00"


def test_full_population_can_be_verified_when_all_timestamps_are_invalid(tmp_path: Path) -> None:
    source = tmp_path / "no-valid-times.csv"
    source.write_text("time,value\nbad,1\nalso bad,2\n", encoding="utf-8")
    read_kwargs = {"sep": ",", "encoding": "utf-8"}
    scan = scan_csv_time_column(source, column="time", read_kwargs=read_kwargs)
    scan["source_fingerprint"] = "source-fingerprint-invalid-times"

    assert scan["rows_scanned"] == 2
    assert scan["valid_timestamp_rows"] == 0
    assert scan["invalid_timestamp_rows"] == 2
    assert scan["minimum_timestamp"] is None
    assert scan["maximum_timestamp"] is None
    assert verified_full_scan_population(
        scan,
        source_fingerprint="source-fingerprint-invalid-times",
        read_options_fingerprint=csv_scan_read_options_fingerprint(read_kwargs),
    )

    metadata = {
        "shape": [2, 2],
        "source_fingerprint": "source-fingerprint-invalid-times",
        "csv_read_contract": {"pandas_kwargs": read_kwargs},
        "full_table_scan": scan,
        "profile_sampling": {"sampled": True, "rows_read": 2},
    }
    shape = reconcile_table_shape(metadata, is_worksheet=False)
    assert shape["verified_row_count"] == 2
    assert shape["row_count_basis"] == "csv_full_table_scan"

    fs = FileSummary(
        path="no-valid-times.csv",
        role=FileRole.raw_data_table,
        summary="preview only",
        columns=["time", "value"],
        source_metadata=metadata,
    )
    protocol = build_data_access_protocol([fs]).files[0]
    assert protocol.read_contract["population_verified"] is True
    assert "全表扫描记录数: 2 x 2" in protocol.row_grain
    assert "无有效时间值" in protocol.row_grain


def test_requested_scan_is_fingerprint_bound_and_reused(tmp_path: Path) -> None:
    source = tmp_path / "设备.csv"
    _device_file(source)
    fs = FileSummary(
        path="设备.csv",
        role=FileRole.raw_data_table,
        summary="表格行列: 100 x 2（CSV 行数快速估计，字段类型基于预览切片）",
        columns=["SampleTime", "flow"],
        source_metadata={
            "shape": [100, 2],
            "shape_estimated": True,
            "csv_encoding": "utf-8-sig",
            "csv_dialect": {"sep": ","},
            "csv_read_contract": {
                "read_parameters_verified": True,
                "pandas_kwargs": {"sep": ",", "encoding": "utf-8-sig"},
            },
            "profile_sampling": {
                "sampled": True,
                "rows_read": 2,
                "configured_max_rows": 2,
            },
        },
        warnings=[
            "字段统计基于前 2 行采样，未全量扫描。",
            "行数或 shape 由轻量方式估算，必要时应在建模代码中重新核对。",
        ],
    )
    inventory = {
        "设备.csv": {
            "fingerprint": "source-fingerprint-1",
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
        }
    }
    task_text = "请对完整 CSV 做只读分块扫描，设备.csv 的时间列是 SampleTime。"

    first = _scan_requested_csvs(
        data_root=tmp_path,
        file_summaries=[fs],
        source_inventory=inventory,
        task_text=task_text,
    )
    second = _scan_requested_csvs(
        data_root=tmp_path,
        file_summaries=[fs],
        source_inventory=inventory,
        task_text=task_text,
    )

    assert first[0]["rows_scanned"] == 5
    assert second[0]["reused_from_shared_cache"] is True
    assert fs.source_metadata["shape"] == [5, 2]
    assert fs.source_metadata["shape_estimated"] is False
    assert fs.source_metadata["csv_read_contract"]["population_verified"] is True
    assert "字段画像读取前 2 行" in fs.summary
    assert reconcile_table_shape(fs.source_metadata, is_worksheet=False)["row_count_basis"] == "csv_full_table_scan"
    assert all("未全量扫描" not in warning for warning in fs.warnings)

    fs.source_metadata["csv_read_contract"]["pandas_kwargs"]["decimal"] = ","
    third = _scan_requested_csvs(
        data_root=tmp_path,
        file_summaries=[fs],
        source_inventory=inventory,
        task_text=task_text,
    )
    assert third[0]["status"] == "completed"
    assert third[0]["reused_from_shared_cache"] is False
    assert third[0]["read_options_fingerprint"] != first[0]["read_options_fingerprint"]


def test_temporal_task_scans_host_table_without_explicit_scan_wording(tmp_path: Path) -> None:
    source = tmp_path / "forecast.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
            "value": [1, 2],
        }
    ).to_csv(source, index=False)
    fs = FileSummary(
        path="forecast.csv",
        role=FileRole.raw_data_table,
        summary="time series observations",
        columns=["timestamp", "value"],
        source_metadata={"csv_read_contract": {"pandas_kwargs": {"sep": ",", "encoding": "utf-8"}}},
    )
    inventory = {"forecast.csv": {"fingerprint": "forecast-fingerprint"}}

    targets = _full_scan_targets([fs], "Forecast the next hour from the historical measurements.")
    scans = _scan_requested_csvs(
        data_root=tmp_path,
        file_summaries=[fs],
        source_inventory=inventory,
        task_text="Forecast the next hour from the historical measurements.",
    )
    requirement = _temporal_scan_requirement(
        "Forecast the next hour from the historical measurements.",
        scans,
    )

    assert targets == [(fs, "timestamp")]
    assert scans[0]["rows_scanned"] == 2
    assert requirement["required"] is True
    assert requirement["status"] == "passed"
    assert requirement["files"][0]["source_fingerprint"] == "forecast-fingerprint"


def test_temporal_scan_requirement_blocks_missing_column_and_invalid_timestamps() -> None:
    fs = FileSummary(
        path="records.csv",
        role=FileRole.raw_data_table,
        summary="records",
        columns=["value"],
    )
    assert _full_scan_targets([fs], "时序预测下一小时的数值") == [(fs, None)]

    missing_column = _temporal_scan_requirement("时序预测", [
        {"file": "records.csv", "status": "failed", "error": "required_time_column_not_found"},
    ])
    invalid_values = _temporal_scan_requirement("Forecast next month", [
        {
            "file": "forecast.csv",
            "column": "timestamp",
            "status": "completed",
            "rows_scanned": 4,
            "valid_timestamp_rows": 3,
            "invalid_timestamp_rows": 1,
            "source_fingerprint": "current-source",
        },
    ])

    assert missing_column["status"] == "blocked"
    assert any("did not complete" in item for item in missing_column["blocking_issues"])
    assert invalid_values["status"] == "blocked"
    assert any("could not be parsed" in item for item in invalid_values["blocking_issues"])


def test_scan_evidence_reaches_qdi_and_automl_read_contracts(tmp_path: Path) -> None:
    source = tmp_path / "设备.csv"
    _device_file(source)
    fingerprint = "source-fingerprint-2"
    read_kwargs = {"sep": ",", "encoding": "utf-8-sig"}
    fs = FileSummary(
        path="设备.csv",
        role=FileRole.raw_data_table,
        summary="表格行列: 5 x 2",
        columns=["SampleTime", "flow"],
        source_metadata={
            "shape": [5, 2],
            "source_fingerprint": fingerprint,
            "csv_encoding": "utf-8-sig",
            "csv_dialect": {"sep": ","},
            "csv_read_contract": {
                "read_parameters_verified": True,
                "reader": "pandas.read_csv",
                "pandas_kwargs": read_kwargs,
            },
            "profile_sampling": {"sampled": True, "rows_read": 2, "configured_max_rows": 2},
            "full_table_scan": {
                "schema_version": "autorealize.csv_full_table_scan.v1",
                "scanner_version": FULL_TABLE_SCAN_VERSION,
                "status": "completed",
                "scope": "all_csv_records_one_time_column",
                "column": "SampleTime",
                "rows_scanned": 5,
                "valid_timestamp_rows": 4,
                "invalid_timestamp_rows": 1,
                "minimum_timestamp": "2026-01-01T00:00:00",
                "maximum_timestamp": "2026-01-01T00:04:00",
                "timestamp_ranges_by_timezone_awareness": {
                    "naive": {
                        "rows": 4,
                        "minimum_timestamp": "2026-01-01T00:00:00",
                        "maximum_timestamp": "2026-01-01T00:04:00",
                    },
                    "aware_utc": {"rows": 0, "minimum_timestamp": None, "maximum_timestamp": None},
                },
                "read_options_fingerprint": csv_scan_read_options_fingerprint(read_kwargs),
                "chunk_rows": 50000,
                "source_fingerprint": fingerprint,
                "preview_scope": {"kind": "field_profile_preview", "rows_read": 2, "sampled": True},
            },
        },
    )

    access = build_data_access_protocol([fs]).files[0]
    schema = build_data_schema_contract([fs])["tables"][0]
    cards = compact_table_cards_for_prompt(build_table_cards([fs]), field_limit=4)
    description = tmp_path / "data_description.md"
    write_data_description(description, [fs], [], [])

    assert access.read_method == "pandas.read_csv(encoding='utf-8-sig')"
    assert "encoding='utf-8-sig'" in access.read_example
    assert access.read_contract["population_verified"] is True
    assert access.read_contract["verified_shape"] == [5, 2]
    assert "全表扫描记录数: 5 x 2" in access.row_grain
    assert schema["row_count_basis"] == "csv_full_table_scan"
    assert cards[0]["full_table_scan"]["minimum_timestamp"] == "2026-01-01T00:00:00"
    assert "全表分块扫描：5 条记录" in description.read_text(encoding="utf-8")
