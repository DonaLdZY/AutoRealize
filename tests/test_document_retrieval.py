from __future__ import annotations

from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.document_retrieval import LocalDocumentIndex
from autorealize.investigation import _current_script_evidence, run_custom_readonly_python
from autorealize.investigation import CrossFileInvestigationTools
from autorealize.context_compiler import ArtifactStore
from autorealize.models import InvestigationStepResult, InvestigationToolRequest, ReadonlyPythonRequest
from autorealize.parsers.pdf_parser import PdfParser


def test_pdf_parser_visits_every_page(monkeypatch, tmp_path: Path) -> None:
    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, _path: str) -> None:
            self.pages = [Page(f"第 {index} 页全文") for index in range(1, 26)]

    monkeypatch.setattr("autorealize.parsers.pdf_parser.PdfReader", Reader)
    path = tmp_path / "rules.pdf"
    path.write_bytes(b"fake")

    parsed = PdfParser().parse(path)

    assert parsed.metadata["pages"] == 25
    assert parsed.metadata["pages_extracted"] == list(range(1, 26))
    assert "第 25 页全文" in parsed.text_summary


def test_document_index_searches_full_text_and_reads_neighbors(tmp_path: Path) -> None:
    data_root = tmp_path / "input"
    data_root.mkdir()
    text = "开头说明。\n" + ("普通内容。" * 500) + "最终评分采用加权成本公式。\n" + ("尾部内容。" * 100)
    (data_root / "rules.txt").write_text(text, encoding="utf-8")

    index = LocalDocumentIndex.build(
        data_root=data_root,
        store_root=tmp_path / "document_store",
        chunk_chars=600,
        chunk_overlap_chars=80,
    )
    result = index.search("最终评分 加权成本", top_k=3)

    assert result["matches"]
    assert result["matches"][0]["source_file"] == "rules.txt"
    chunk_id = result["matches"][0]["chunk_id"]
    retrieved = index.read_chunks([chunk_id], neighbor_count=1, max_chars=5000)
    assert any("最终评分采用加权成本公式" in chunk["text"] for chunk in retrieved["chunks"])
    assert len(retrieved["chunks"]) >= 2


def test_long_script_result_is_preserved_and_prompt_view_is_marked_truncated(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    cfg = AutoRealizeConfig.from_env()
    code = """
def analyze(input_dir: str, scratch_dir: str) -> dict:
    print("diagnostic" * 2000)
    return {"payload": "x" * 30000, "count": 7}
"""

    result = run_custom_readonly_python(code, input_dir=input_dir, cfg=cfg)

    assert result["count"] == 7
    assert len(result["payload"]) == 30000
    assert result["_stdout_capture"]["truncated"] is True
    request = InvestigationToolRequest(
        request_id="r1",
        question_id="q1",
        tool_name="custom_readonly_python",
        custom_python=ReadonlyPythonRequest(question_id="q1", python_code=code),
    )
    step_result = InvestigationStepResult(
        request_id="r1",
        question_id="q1",
        tool_name="custom_readonly_python",
        result=result,
        max_output_chars=1000,
    )
    evidence = _current_script_evidence(
        request,
        step_result,
        1000,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )

    assert evidence["truncated"] is True
    assert evidence["original_output_chars"] > evidence["visible_output_chars"]
    assert evidence["current_output_artifact"]["artifact_id"]
    assert "truncated=true" in evidence["instruction"]

    artifact_path = tmp_path / "artifacts" / f"{evidence['current_output_artifact']['artifact_id']}.json"
    stored = artifact_path.read_text(encoding="utf-8")
    assert stored.count("x") >= 30000


def test_executor_preserves_original_size_through_artifact_backed_prompt_view(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    cfg = AutoRealizeConfig.from_env()
    cfg.investigation.max_result_chars = 1000
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    tools = CrossFileInvestigationTools(
        cfg=cfg,
        data_root=input_dir,
        authoritative_memory={},
        knowledge_base={},
        artifact_store=artifact_store,
    )
    request = InvestigationToolRequest(
        request_id="r2",
        question_id="q2",
        tool_name="custom_readonly_python",
        custom_python=ReadonlyPythonRequest(
            question_id="q2",
            python_code="""
def analyze(input_dir: str, scratch_dir: str) -> dict:
    return {"payload": "z" * 10000}
""",
        ),
    )

    result = tools.execute(request)
    evidence = _current_script_evidence(request, result, 1000, artifact_store=artifact_store)

    assert result.output_truncated is True
    assert evidence["truncated"] is True
    assert evidence["original_output_chars"] == result.original_output_chars
    assert evidence["original_output_chars"] > 10000
