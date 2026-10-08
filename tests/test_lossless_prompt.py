import json
from pathlib import Path

from autorealize.config import AutoRealizeConfig
from autorealize.context_compiler import compact_authoritative_memory, compact_constraint_memory
from autorealize.llm.client import _apply_deepseek_request_options
from autorealize.prompt_cache import lossless_json


def test_repeated_constraints_round_trip_with_escaped_paths():
    constraint = {"rule": "FYP01 cannot share a vehicle. No stop-count cap. " * 30,
                  "limits": [None, False, 0, 5.5]}
    source = {"a/~": constraint, "nested": [constraint, {"different": "tail constraint"}]}
    encoded = lossless_json(source)
    envelope = json.loads(encoded)
    assert len(encoded) < len(json.dumps(source))

    def resolve(value):
        if isinstance(value, dict) and set(value) == {"$prompt_ref"}:
            target = envelope
            for part in value["$prompt_ref"].split("/")[1:]:
                key = part.replace("~1", "/").replace("~0", "~")
                target = target[int(key)] if isinstance(target, list) else target[key]
            return resolve(target)
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    assert resolve(envelope["data"]) == source


def test_source_reference_marker_remains_literal():
    source = {"$prompt_ref": "/original", "long": ["x" * 3000] * 2}
    assert json.loads(lossless_json(source)) == source


def test_gpt_reasoning_effort_is_sent_verbatim():
    cfg = AutoRealizeConfig()
    cfg.llm.model_name = "gpt-5.6-luna"
    cfg.llm.reasoning_effort = "xhigh"
    params = {"model": cfg.llm.model_name, "temperature": 0.5}
    _apply_deepseek_request_options(params, cfg, structured=True)
    assert params == {"model": "gpt-5.6-luna", "reasoning_effort": "xhigh"}


def test_authoritative_tail_constraints_survive_compaction():
    memory = {"items": [{"id": i, "text": "weight limit " * 100, "status": "resolved"} for i in range(65)],
              "constraints": ["No stop-count cap. " * 100] * 50,
              "unresolved_questions": ["requires verification"] * 31}
    assert compact_constraint_memory(memory) == memory
    assert compact_authoritative_memory(memory) == memory


def test_agent_pack_preserves_source_facts_and_does_not_invent_filenames():
    from autorealize.models import FileRole, FileSummary, SubmissionContract
    from autorealize.modules.data_cognition import DataCognitionModule

    authority = {"constraints": [f"Mandatory {i}" for i in range(130)],
                 "submission_contract": SubmissionContract().model_dump()}
    constraints = {"items": [{"description": f"Rule {i}"} for i in range(130)]}
    columns = [f"field_{i}" for i in range(255)]
    files = [FileSummary(path="data.csv", role=FileRole.raw_data_table, summary="data", columns=columns)]
    pack = DataCognitionModule._build_agent_context_pack(
        object(), task_hint="task", file_summaries=files, relation_hints=[],
        constraint_memory=constraints, authoritative_memory=authority, knowledge_base={},
        sampled_patterns=[], filename_sample_groups=[])
    assert pack["authoritative_memory"] == authority
    assert pack["constraint_memory"] == constraints
    assert pack["submission_contract"]["output_filename"] == ""
    assert pack["submission_contract"]["sample_filename"] == ""
    assert pack["data_memory"]["tables"][0]["columns"] == columns


def test_investigated_constraint_tail_survives_both_stage_handoffs():
    from autorealize.modules.data_cognition import _compact_question_memory
    from autorealize.modules.task_definition import TaskDefinitionModule

    answer = {"question": "Capacity semantics?", "answer": "Documented context. " * 200 + "NO_ASSUMED_DATES",
              "remaining_uncertainty": "Only model delivery is authorized.",
              "evidence": ["source.xlsx:capacity"], "downstream_notes": ["Reject undated inference."]}
    memory = _compact_question_memory({"answers": [answer] * 25})
    packed = TaskDefinitionModule._question_memory_pack(object(), {"agent_context_pack": {"question_memory": memory}})
    assert len(packed["answers"]) == 25
    assert packed["answers"][-1]["answer"] == answer["answer"]
    assert packed["answers"][-1]["evidence"] == answer["evidence"]
    assert packed["answers"][-1]["remaining_uncertainty"] == answer["remaining_uncertainty"]


def test_provider_sdk_cannot_multiply_logged_retries(tmp_path):
    from autorealize.llm.client import LLMClient
    cfg = AutoRealizeConfig()
    cfg.llm.api_key = "test-key"
    client = LLMClient(cfg, tmp_path)
    assert client.client.max_retries == 0
    client.client.close()


def test_authority_does_not_use_absolute_run_directory_names(tmp_path):
    from autorealize.modules.data_cognition import DataCognitionModule

    module = object.__new__(DataCognitionModule)
    data_root = tmp_path / "tasks" / "ai4i"
    data_root.mkdir(parents=True)
    raw = data_root / "ai4i2020.csv"
    rule = data_root / "rules.md"

    assert module._is_authoritative_doc_name(raw, data_root) == (False, "", "low")
    assert module._is_authoritative_doc_name(rule, data_root) == (True, "requirement_doc", "high")
    # The same raw table must stay non-authoritative even when the parent path
    # itself contains the generic word used by the run workspace.
    assert module._is_authoritative_doc_name(Path("D:/runs/tasks/ai4i2020.csv"), Path("D:/runs")) == (False, "", "low")


def test_authority_sources_keep_late_rules_and_all_named_documents(tmp_path):
    from autorealize.models import AuthoritativeTaskMemory
    from autorealize.modules.data_cognition import DataCognitionModule

    class FakeClient:
        def __init__(self):
            self.dynamic = ""

        def ask_structured(self, **kwargs):
            self.dynamic = kwargs["dynamic_user_prompt"]
            return AuthoritativeTaskMemory()

    root = tmp_path / "tasks" / "many"
    root.mkdir(parents=True)
    for index in range(90):
        (root / f"rule-{index:02d}.md").write_text(
            f"rule {index}\n" + ("x" * (25000 if index == 89 else 50)) + f"\nLATE_RULE_{index}", encoding="utf-8"
        )
    client = FakeClient()
    module = object.__new__(DataCognitionModule)
    module.services = type("Services", (), {"llm_client": client})()

    memory = module._extract_authoritative_memory(data_root=root, task_hint="", file_summaries=[])

    assert len(memory["source_files"]) == 90
    assert "rule-89.md" in client.dynamic
    assert "LATE_RULE_89" in client.dynamic


def test_authority_preserves_named_rule_tables_and_non_utf8_text(tmp_path):
    from autorealize.modules.data_cognition import DataCognitionModule
    module = object.__new__(DataCognitionModule)
    assert not module._is_authoritative_doc_name(tmp_path / "requirements" / "measurements.csv")[0]
    assert not module._is_authoritative_doc_name(tmp_path / "docs" / "measurements.csv", tmp_path)[0]
    assert module._is_authoritative_doc_name(tmp_path / "vehicle-rules.csv", tmp_path)[0]
    assert module._is_authoritative_doc_name(tmp_path / "docs" / "late.md", tmp_path)[0]
    path = tmp_path / "业务规则说明.txt"
    text = "末尾规则：不得假定运力日期。"
    path.write_bytes(text.encode("gb18030"))
    assert module._is_authoritative_doc_name(path, tmp_path)[0]
    assert module._read_authoritative_text(path) == text


def test_authority_reads_original_docx_instead_of_promoting_llm_advice(tmp_path):
    from docx import Document
    from autorealize.models import AuthoritativeTaskMemory, FileSummary
    from autorealize.modules.data_cognition import DataCognitionModule

    path = tmp_path / "plant.docx"
    document = Document()
    document.add_paragraph("Steam valve increments must be <= 5%. Startup constraints may be deferred.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Pump cycle"
    table.cell(0, 1).text = "900 seconds"
    document.save(path)
    captured = []
    class Client:
        def ask_structured(self, **kwargs):
            captured.append(kwargs["dynamic_user_prompt"])
            return AuthoritativeTaskMemory()
    module = object.__new__(DataCognitionModule)
    module.services = type("Services", (), {"llm_client": Client()})()
    summary = FileSummary(path="plant.docx", role="task_requirement", summary="Generated summary",
                          detailed_report="A stakeholder must approve all lag windows before any model is fitted.")
    module._extract_authoritative_memory(data_root=tmp_path, task_hint="Predict concentration", file_summaries=[summary])
    assert "900 seconds" in captured[0] and "may be deferred" in captured[0]
    assert "stakeholder must approve" not in captured[0]
    assert "Generated summary" not in captured[0]


def test_unreadable_requirement_summary_keeps_inference_provenance(tmp_path):
    from autorealize.models import AuthoritativeTaskMemory, FileSummary
    from autorealize.modules.data_cognition import DataCognitionModule
    captured = []
    class Client:
        def ask_structured(self, **kwargs):
            captured.append(kwargs["dynamic_user_prompt"])
            return AuthoritativeTaskMemory()
    module = object.__new__(DataCognitionModule)
    module.services = type("Services", (), {"llm_client": Client()})()
    module._extract_authoritative_memory(data_root=tmp_path, task_hint="", file_summaries=[
        FileSummary(path="scanned.png", role="task_requirement", summary="Unverified constraint summary")])
    assert '"source_type":"llm_inference"' in captured[0]
    assert '"priority":"low"' in captured[0]
