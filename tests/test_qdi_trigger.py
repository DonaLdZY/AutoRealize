from types import SimpleNamespace

from autorealize.modules.data_cognition import DataCognitionModule


def _module() -> DataCognitionModule:
    config = SimpleNamespace(
        investigation=SimpleNamespace(enabled=True, trigger_mode="on_demand"),
    )
    return DataCognitionModule(config, services=None, report_dir=None)


def test_on_demand_qdi_runs_when_requested_full_scan_failed() -> None:
    module = _module()

    should_run = module._should_run_question_investigator(
        [],
        {},
        {},
        full_scan_evidence=[{"file": "device.csv", "status": "failed", "error": "bad encoding"}],
    )

    assert should_run is True


def test_on_demand_qdi_skips_when_full_scan_completed_and_no_other_gap_exists() -> None:
    module = _module()

    should_run = module._should_run_question_investigator(
        [],
        {},
        {},
        full_scan_evidence=[{"file": "device.csv", "status": "completed", "rows_scanned": 100}],
    )

    assert should_run is False
