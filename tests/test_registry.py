from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.parsers import build_registry


def test_registry_can_parse_text(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    reg = build_registry(AutoRealizeConfig())
    parsed = reg.parse(p)
    assert parsed.kind == "document"
    assert "hello" in parsed.text_summary
