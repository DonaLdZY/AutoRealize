from pathlib import Path

from autorealize.knowledge.base import KnowledgeEntry
from autorealize.knowledge.local_store import LocalKnowledgeStore, make_entry_id


def test_local_knowledge_store_search_and_manifest(tmp_path: Path) -> None:
    store = LocalKnowledgeStore(tmp_path / "knowledge_store.jsonl")
    entry = KnowledgeEntry(
        entry_id=make_entry_id("constraint", "doc.md", "销售额不能为负"),
        kind="constraint",
        source="doc.md",
        text="销售额不能为负，预测结果低于0按硬约束违规处理。",
        fields=["sales"],
        constraints=["non_negative_sales"],
        tags=["constraint", "hard_constraint"],
    )
    store.add(entry)
    store.flush()
    results = store.search("预测 销售额 不能为负", top_k=3)
    assert results
    assert results[0].entry.kind == "constraint"
    assert results[0].score > 0

    manifest = tmp_path / "rag_manifest.json"
    store.write_manifest(manifest)
    text = manifest.read_text(encoding="utf-8")
    assert "constraint" in text
    assert "entry_count" in text
