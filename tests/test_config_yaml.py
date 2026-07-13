from __future__ import annotations

import json
from pathlib import Path

import yaml

from autorealize.config import AutoRealizeConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_leaf_descriptions(node: dict) -> None:
    for value in node.get("properties", {}).values():
        if value.get("type") == "object":
            _assert_leaf_descriptions(value)
        else:
            assert value.get("description", "").strip()


def test_commented_default_yaml_loads(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-test-key")
    cfg = AutoRealizeConfig.from_file(REPO_ROOT / "config" / "config.yaml")

    assert cfg.llm.api_key == "env-test-key"
    assert cfg.llm.max_concurrent_requests == 100
    assert cfg.parallel.cognition_max_workers == 100
    assert cfg.prompt.output_language == "zh"
    assert cfg.telemetry.config_snapshot_filename == "final_config.yaml"
    assert cfg.service.snapshot_event_limit == 400
    assert cfg.service.stop_wait_seconds == 15.0
    assert cfg.to_dict()["llm"]["api_key"] is None
    assert cfg.schema_dict()["properties"]["llm"]["properties"]["api_key"]["default"] is None
    _assert_leaf_descriptions(cfg.schema_dict())


def test_yaml_round_trip_and_legacy_json_compatibility(tmp_path: Path) -> None:
    cfg = AutoRealizeConfig()
    cfg.prompt.output_language = "en"
    cfg.data.text_encodings = ("utf-8", "gb18030")

    yaml_path = tmp_path / "config.yaml"
    cfg.write_yaml(yaml_path)
    loaded = AutoRealizeConfig.from_file(yaml_path)
    assert loaded.prompt.output_language == "en"
    assert loaded.data.text_encodings == ("utf-8", "gb18030")
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["prompt"]["output_language"] == "en"

    json_path = tmp_path / "legacy.json"
    json_path.write_text(
        json.dumps({"prompt": {"output_language": "zh"}}),
        encoding="utf-8",
    )
    assert AutoRealizeConfig.from_file(json_path).prompt.output_language == "zh"


def test_config_api_key_has_priority_over_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  api_key: config-key\n", encoding="utf-8")

    assert AutoRealizeConfig.from_file(path).llm.api_key == "config-key"
