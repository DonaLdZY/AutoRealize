from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .utils.safe_json import json_safe, write_json_safe
from .entity_alias import build_entity_alias_candidates


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    source: str
    visible_excerpt: str = ""
    truncated: bool = False
    original_chars: int = 0
    visible_chars: int = 0
    artifact_path: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "source": self.source,
            "visible_excerpt": self.visible_excerpt,
            "truncated": self.truncated,
            "original_chars": self.original_chars,
            "visible_chars": self.visible_chars,
            "artifact_path": self.artifact_path,
        }


class ArtifactStore:
    """Local CCR-style store for large context objects.

    Prompts should carry the returned ArtifactRef, not the full payload. This is
    intentionally deterministic and non-LLM: the full object stays on disk for
    audit/replay while compact cards enter the model context.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        artifact_type: str,
        source: str,
        payload: Any,
        *,
        visible_excerpt: str = "",
        visible_limit: int = 1200,
    ) -> dict[str, Any]:
        safe = json_safe(payload)
        text = json.dumps(safe, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        digest = hashlib.sha256(
            f"{artifact_type}\n{source}\n{text}".encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        artifact_id = f"{_safe_id(artifact_type)}_{digest}"
        artifact_path = ""
        if self.root is not None:
            path = self.root / f"{artifact_id}.json"
            write_json_safe(
                path,
                {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "source": source,
                    "payload": safe,
                },
                indent=2,
            )
            artifact_path = str(path.name)
        visible = str(visible_excerpt or text[: max(1, int(visible_limit))])
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            source=str(source),
            visible_excerpt=visible,
            truncated=len(text) > len(visible),
            original_chars=len(text),
            visible_chars=len(visible),
            artifact_path=artifact_path,
        ).model_dump()


def build_qdi_context_bundle(
    *,
    cfg: Any,
    data_root: Path,
    task_hint: str,
    file_summaries: list[Any],
    relation_hints: list[Any],
    constraint_memory: dict,
    authoritative_memory: dict,
    knowledge_base: dict,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Compile Headroom-style compact context for QDI."""

    detailed_table_cards = build_table_cards(
        file_summaries,
        artifact_store=artifact_store,
        file_limit=120,
        sheet_limit=80,
        field_limit=80,
    )
    return build_qdi_context_bundle_from_table_cards(
        cfg=cfg,
        data_root=data_root,
        task_hint=task_hint,
        detailed_table_cards=detailed_table_cards,
        relation_hints=relation_hints,
        constraint_memory=constraint_memory,
        authoritative_memory=authoritative_memory,
        knowledge_base=knowledge_base,
        file_summaries=file_summaries,
    )


def build_qdi_context_bundle_from_table_cards(
    *,
    cfg: Any,
    data_root: Path,
    task_hint: str,
    detailed_table_cards: list[dict[str, Any]],
    relation_hints: list[Any],
    constraint_memory: dict,
    authoritative_memory: dict,
    knowledge_base: dict,
    file_summaries: list[Any],
) -> dict[str, Any]:
    """Build the stable QDI prompt context from already materialized details."""

    table_cards = compact_table_cards_for_prompt(
        detailed_table_cards,
        field_limit=3,
        max_cards=48,
        per_source_limit=6,
    )
    relation_cards = build_relation_cards(relation_hints, limit=100)
    filename_group_cards = build_filename_group_cards(
        (knowledge_base or {}).get("filename_sample_groups", []),
        file_summaries=file_summaries,
        limit=80,
    )
    entity_alias_candidates = build_entity_alias_candidates(
        file_summaries,
        filename_sample_groups=(knowledge_base or {}).get("filename_sample_groups", []),
    )
    return {
        "schema_version": "autorealize.qdi_context.headroom.v1",
        "task_hint": task_hint,
        "data_root_name": data_root.name,
        "context_policy": {
            "files_are_table_cards": True,
            "large_objects_local_only": True,
            "retrieved_navigation_notes_are_not_authority": True,
            "authoritative_facts_source": "authoritative_memory",
            "constraint_facts_source": "constraint_memory",
            "historical_script_outputs_not_in_prompt": True,
            "stable_table_cards_are_light_index": True,
            "detailed_field_statistics_are_dynamic_only": True,
            "stable_table_cards_are_route_only": True,
            "retrieve_details_action": "request_context",
            "telemetry_is_local_not_prompt": True,
        },
        "table_cards": table_cards,
        "relations": relation_cards,
        "constraint_memory": compact_constraint_memory(constraint_memory),
        "authoritative_memory": compact_authoritative_memory(authoritative_memory),
        "sampled_filename_patterns": compact_sampled_patterns(
            (knowledge_base or {}).get("sampled_filename_patterns", [])
        ),
        "filename_sample_groups": filename_group_cards,
        "entity_alias_candidates": entity_alias_candidates,
        "field_glossary": compact_field_glossary((knowledge_base or {}).get("field_glossary") or {}),
        "script_investigation_policy": {
            "input_dir": "read-only",
            "scratch_dir": "temporary writable and destroyed after custom Python execution",
            "custom_python_contract": "def analyze(input_dir: str, scratch_dir: str) -> dict",
            "allowed_libraries": [
                "pandas",
                "numpy",
                "json",
                "math",
                "statistics",
                "re",
                "csv",
                "collections",
                "itertools",
                "pathlib",
                "datetime",
                "typing",
            ],
            "output_policy": "Return compact JSON-compatible facts; never print full tables.",
        },
        "limits": {
            "max_questions": getattr(cfg.investigation, "max_questions", 5),
            "max_rounds_per_run": getattr(cfg.investigation, "max_rounds_per_run", 3),
            "allow_custom_readonly_python": getattr(cfg.investigation, "allow_custom_readonly_python", True),
            "question_bfs_max_depth": 3,
            "max_followup_questions_per_question": 3,
        },
    }


def build_qdi_context_and_details(
    *,
    cfg: Any,
    data_root: Path,
    task_hint: str,
    file_summaries: list[Any],
    relation_hints: list[Any],
    constraint_memory: dict,
    authoritative_memory: dict,
    knowledge_base: dict,
    artifact_store: ArtifactStore | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build stable prompt context plus local-only detailed table card index."""

    detailed_table_cards = build_table_cards(
        file_summaries,
        artifact_store=artifact_store,
        file_limit=120,
        sheet_limit=80,
        field_limit=80,
    )
    context = build_qdi_context_bundle_from_table_cards(
        cfg=cfg,
        data_root=data_root,
        task_hint=task_hint,
        detailed_table_cards=detailed_table_cards,
        relation_hints=relation_hints,
        constraint_memory=constraint_memory,
        authoritative_memory=authoritative_memory,
        knowledge_base=knowledge_base,
        file_summaries=file_summaries,
    )
    return context, _table_card_detail_map(detailed_table_cards)


def build_qdi_table_card_details(
    *,
    file_summaries: list[Any],
    artifact_store: ArtifactStore | None = None,
) -> dict[str, dict[str, Any]]:
    """Build detailed table cards for deterministic, per-question retrieval.

    The returned object is intentionally not part of the stable prompt. QDI
    action/repair calls select a small subset and place it in the dynamic tail.
    """

    detailed = build_table_cards(
        file_summaries,
        artifact_store=artifact_store,
        file_limit=120,
        sheet_limit=80,
        field_limit=80,
    )
    return _table_card_detail_map(detailed)


def _table_card_detail_map(detailed: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for card in detailed:
        if not isinstance(card, dict):
            continue
        table_id = str(card.get("table_id", "") or card.get("source_file", "") or "").strip()
        if table_id and table_id not in out:
            out[table_id] = card
    return out


def compact_table_cards_for_prompt(
    cards: list[dict[str, Any]],
    *,
    field_limit: int = 3,
    max_cards: int = 48,
    per_source_limit: int = 6,
) -> list[dict[str, Any]]:
    """Return a Headroom-style route-only table manifest for stable prompts.

    This is deliberately closer to Headroom's marker/index idea than to a data
    pack: stable prompts get just enough identifiers to request details, while
    field semantics, statistics, reading notes, warnings, previews, and artifact
    refs stay in local detail cards until `request_context` asks for them.
    """

    out: list[dict[str, Any]] = []
    omitted_by_source: dict[str, int] = {}
    seen_by_source: dict[str, int] = {}
    max_cards = max(1, int(max_cards))
    per_source_limit = max(1, int(per_source_limit))
    for card in list(cards or []):
        if not isinstance(card, dict):
            continue
        source = str(card.get("source_file") or card.get("table_id") or "unknown")
        if len(out) >= max_cards or seen_by_source.get(source, 0) >= per_source_limit:
            omitted_by_source[source] = omitted_by_source.get(source, 0) + 1
            continue
        out.append(_route_only_table_card(card, field_limit=field_limit))
        seen_by_source[source] = seen_by_source.get(source, 0) + 1
    if omitted_by_source:
        out.append(
            _drop_empty(
                {
                    "table_id": "__omitted_table_manifest__",
                    "table_kind": "omitted_table_cards",
                    "omitted_table_count": sum(omitted_by_source.values()),
                    "omitted_by_source": [
                        {"source_file": source, "omitted_tables": count}
                        for source, count in sorted(omitted_by_source.items())[:20]
                    ],
                    "detail_policy": (
                        "Additional table manifests and all table details are stored locally. "
                        "Use request_context with input_files, focus_sheets, focus_columns, "
                        "or query to retrieve focused details."
                    ),
                }
            )
        )
    return out


def _route_only_table_card(card: dict[str, Any], *, field_limit: int = 3) -> dict[str, Any]:
    fields = card.get("fields", []) if isinstance(card.get("fields"), list) else []
    field_hints = _route_field_hints(fields, limit=field_limit)
    return _drop_empty(
                    {
                        "table_id": card.get("table_id"),
                        "source_file": card.get("source_file"),
                        "sheet_name": card.get("sheet_name"),
                        "table_kind": card.get("table_kind"),
                        "role": card.get("role"),
                        "shape": card.get("shape"),
                        "layout_kind": _abnormal_layout_value(card.get("layout_kind")),
                        "read_strategy_kind": _abnormal_layout_value(card.get("read_strategy_kind")),
                        "sheet_group_id": card.get("sheet_group_id"),
                        "sheet_group_size": card.get("sheet_group_size"),
                        "is_deep_profiled": card.get("is_deep_profiled"),
                        "field_hints": field_hints,
                        "field_count": len(fields),
            "detail_policy": "Route-only table manifest. Use request_context for field meanings, statistics, reading notes, warnings, or sheet details.",
        }
    )


def _route_field_hints(fields: list[Any], *, limit: int) -> list[str]:
    """Tiny field-name hints for routing only, never a field profile."""

    limit = max(0, int(limit))
    if limit <= 0:
        return []
    scored: list[tuple[int, int, str]] = []
    for idx, field in enumerate(fields):
        if not isinstance(field, dict):
            continue
        name = str(field.get("name", "") or "").strip()
        if not name:
            continue
        meaning = str(field.get("meaning", "") or "")
        role = str(field.get("role", "") or field_role(name, meaning))
        score = 0
        if role != "other":
            score += 20
        if str(field.get("logical_type", "")).lower() in {"datetime", "date", "numeric", "integer", "float"}:
            score += 4
        if field.get("unique_count") not in (None, "", [], {}):
            score += 2
        scored.append((-score, idx, name[:80]))
    out: list[str] = []
    seen: set[str] = set()
    for _score, _idx, name in sorted(scored)[:limit]:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def compact_detail_table_card_for_prompt(card: dict[str, Any], *, field_limit: int = 12) -> dict[str, Any]:
    """Return a bounded table detail excerpt for one retrieval turn.

    Full field profiles and previews remain in ArtifactStore; this prompt-side
    object mirrors Headroom's marker-plus-local-store design.
    """

    fields = card.get("fields", []) if isinstance(card.get("fields"), list) else []
    compact_fields: list[dict[str, Any]] = []
    for field in fields[: max(1, int(field_limit))]:
        if not isinstance(field, dict):
            continue
        numeric = field.get("numeric_stats") if isinstance(field.get("numeric_stats"), dict) else {}
        datetime_stats = field.get("datetime_stats") if isinstance(field.get("datetime_stats"), dict) else {}
        compact_fields.append(
            _drop_empty(
                {
                    "name": field.get("name"),
                    "meaning": str(field.get("meaning", "") or "")[:180],
                    "role": field.get("role"),
                    "logical_type": field.get("logical_type"),
                    "row_count": field.get("row_count"),
                    "non_null_count": field.get("non_null_count"),
                    "null_ratio": field.get("null_ratio"),
                    "unique_count": field.get("unique_count"),
                    "top_values": [str(x)[:80] for x in (field.get("top_values") or [])[:3]],
                    "numeric_stats": _small_dict(numeric, ["min", "max", "mean", "std"]),
                    "datetime_stats": _small_dict(datetime_stats, ["min", "max", "range_days", "granularity"]),
                }
            )
        )
    return _drop_empty(
        {
            "table_id": card.get("table_id"),
            "source_file": card.get("source_file"),
            "sheet_name": card.get("sheet_name"),
            "table_kind": card.get("table_kind"),
            "role": card.get("role"),
            "file_cognition": str(card.get("file_cognition", "") or "")[:260],
            "shape": card.get("shape"),
            "layout_kind": card.get("layout_kind"),
            "header_confidence": card.get("header_confidence"),
            "detected_header_row": card.get("detected_header_row"),
            "read_strategy_kind": card.get("read_strategy_kind"),
            "recommended_read": str(card.get("recommended_read", "") or "")[:240],
            "fields": compact_fields,
            "reading_notes": [str(x)[:180] for x in (card.get("reading_notes") or [])[:4]],
            "warnings": [str(x)[:180] for x in (card.get("warnings") or [])[:4]],
            "detail_policy": "This is a bounded retrieved excerpt; full profile remains local and is not part of the prompt.",
        }
    )


def _prefer_specific_artifact_refs(refs: Any) -> list[dict[str, Any]]:
    items = [ref for ref in list(refs or []) if isinstance(ref, dict)]
    if not items:
        return []
    preferred = [
        ref
        for ref in items
        if str(ref.get("artifact_type", "")).endswith("_profile_full")
        and str(ref.get("artifact_type", "")) != "file_profile_full"
    ]
    return preferred or items


def _artifact_refs_as_index(refs: Any, *, max_refs: int = 4) -> list[dict[str, Any]]:
    out = []
    for ref in list(refs or [])[: max(0, int(max_refs))]:
        if not isinstance(ref, dict):
            continue
        out.append(
            _drop_empty(
                {
                    "artifact_id": ref.get("artifact_id"),
                    "artifact_type": ref.get("artifact_type"),
                    "source": ref.get("source"),
                    "truncated": ref.get("truncated"),
                    "original_chars": ref.get("original_chars"),
                    "artifact_path": ref.get("artifact_path"),
                }
            )
        )
    return out


def build_table_cards(
    file_summaries: list[Any],
    *,
    artifact_store: ArtifactStore | None = None,
    file_limit: int = 120,
    sheet_limit: int = 80,
    field_limit: int = 80,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for fs in list(file_summaries or [])[: max(1, int(file_limit))]:
        cards.extend(
            file_to_table_cards(
                fs,
                artifact_store=artifact_store,
                sheet_limit=sheet_limit,
                field_limit=field_limit,
            )
        )
    return cards


def file_to_table_cards(
    fs: Any,
    *,
    artifact_store: ArtifactStore | None = None,
    sheet_limit: int = 80,
    field_limit: int = 80,
) -> list[dict[str, Any]]:
    path = str(getattr(fs, "path", "") or "")
    role = str(getattr(getattr(fs, "role", ""), "value", getattr(fs, "role", "")))
    summary = str(getattr(fs, "summary", "") or "")[:900]
    meta = getattr(fs, "source_metadata", {}) or {}
    semantics = getattr(fs, "column_semantics", {}) or {}
    profiles = [p for p in (getattr(fs, "column_profiles", []) or []) if isinstance(p, dict)]
    artifact_refs = []
    if artifact_store is not None:
        artifact_refs.append(
            artifact_store.put(
                "file_profile_full",
                path,
                _file_payload(fs),
                visible_excerpt=f"{path}: {summary[:300]}",
                visible_limit=600,
            )
        )

    sheet_profiles = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
    if sheet_profiles:
        cards: list[dict[str, Any]] = []
        sheet_semantics_all = meta.get("sheet_field_descriptions") if isinstance(meta.get("sheet_field_descriptions"), dict) else {}
        for sheet in sheet_profiles[: max(1, int(sheet_limit))]:
            if not isinstance(sheet, dict):
                continue
            sheet_name = str(sheet.get("sheet_name", "") or "")
            sheet_semantics = sheet_semantics_all.get(sheet_name, {}) if isinstance(sheet_semantics_all, dict) else {}
            sheet_profiles_compact = sheet.get("column_profiles", []) if isinstance(sheet.get("column_profiles"), list) else []
            refs = list(artifact_refs)
            if artifact_store is not None:
                refs.append(
                    artifact_store.put(
                        "excel_sheet_profile_full",
                        f"{path}::{sheet_name}",
                        sheet,
                        visible_excerpt=f"{path}::{sheet_name}",
                        visible_limit=400,
                    )
                )
            cards.append(
                _drop_empty(
                    {
                        "table_id": f"{path}::{sheet_name}" if sheet_name else path,
                        "source_file": path,
                        "sheet_name": sheet_name,
                        "table_kind": "excel_sheet",
                        "role": role,
                        "file_cognition": summary[:350],
                        "shape": sheet.get("shape") or sheet.get("shape_profiled") or sheet.get("shape_sampled"),
                        "profile_policy": sheet.get("profile_policy"),
                        "sheet_group_id": sheet.get("sheet_group_id"),
                        "sheet_group_size": sheet.get("sheet_group_size"),
                        "is_deep_profiled": sheet.get("is_deep_profiled"),
                        "layout_kind": sheet.get("layout_kind"),
                        "header_confidence": sheet.get("header_confidence"),
                        "detected_header_row": sheet.get("detected_header_row"),
                        "read_strategy_kind": sheet.get("read_strategy_kind"),
                        "recommended_read": sheet.get("recommended_read"),
                        "fields": _field_cards(
                            [str(x) for x in (sheet.get("columns") or [])],
                            sheet_profiles_compact,
                            sheet_semantics,
                            limit=field_limit,
                        ),
                        "reading_notes": _reading_notes(path, meta, sheet_name=sheet_name, sheet=sheet),
                        "warnings": _warnings(fs, sheet),
                        "artifact_refs": refs,
                    }
                )
            )
        return cards

    columns = [str(x) for x in (getattr(fs, "columns", []) or [])]
    return [
        _drop_empty(
            {
                "table_id": path,
                "source_file": path,
                "table_kind": _table_kind(path, meta),
                "role": role,
                "file_cognition": summary[:350],
                "shape": meta.get("shape") or meta.get("shape_estimated"),
                "fields": _field_cards(columns, profiles, semantics, limit=field_limit),
                "reading_notes": _reading_notes(path, meta),
                "warnings": _warnings(fs, {}),
                "artifact_refs": artifact_refs,
            }
        )
    ]


def build_relation_cards(relations: Any, *, limit: int = 100) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in list(relations or [])[: max(1, int(limit))]:
        if hasattr(rel, "model_dump"):
            data = rel.model_dump()
        elif hasattr(rel, "__dict__"):
            data = dict(rel.__dict__)
        elif isinstance(rel, dict):
            data = rel
        else:
            continue
        shared = data.get("shared_columns") or []
        left_field = str(data.get("left_field", "") or (shared[0] if shared else ""))
        right_field = str(data.get("right_field", "") or (shared[0] if shared else ""))
        out.append(
            _drop_empty(
                {
                    "left_file": str(data.get("left_file", "")),
                    "left_field": left_field,
                    "right_file": str(data.get("right_file", "")),
                    "right_field": right_field,
                    "relation_type": str(data.get("relation_type", "shared_attribute") or "shared_attribute"),
                    "confidence": data.get("confidence", 0.0),
                    "short_evidence": str(data.get("short_evidence", "") or data.get("reason", ""))[:500],
                }
            )
        )
    return out


def build_filename_group_cards(groups: Any, *, file_summaries: list[Any], limit: int = 80) -> list[dict[str, Any]]:
    summary_by_path = {str(getattr(fs, "path", "")): fs for fs in file_summaries or []}
    out: list[dict[str, Any]] = []
    for group in list(groups or [])[: max(1, int(limit))]:
        if not isinstance(group, dict):
            continue
        files = [str(x) for x in (group.get("files", []) or [])]
        reps = files[:3]
        column_profile = _filename_group_column_profile(files, summary_by_path)
        shared_fields = column_profile.get("shared_fields", [])
        variant_fields_by_file = column_profile.get("variant_fields_by_file", [])
        field_presence = column_profile.get("field_presence", [])
        layout_variants = column_profile.get("layout_variants", [])
        sample_id = str(group.get("sample_id", "") or group.get("pattern", "") or "")
        out.append(
            _drop_empty(
                {
                    "directory": str(group.get("directory", "")),
                    "template_path_or_sample_id": sample_id,
                    "file_count": len(files),
                    "role": str(group.get("role", "") or group.get("data_kind", "")),
                    "structure_consistent": bool(shared_fields) and not bool(variant_fields_by_file),
                    "representative_files": reps,
                    "shared_fields": shared_fields[:40],
                    "variant_fields_by_file": variant_fields_by_file[:12],
                    "field_presence": field_presence[:24],
                    "layout_variants": layout_variants[:12],
                    "data_kinds": {
                        str(k): str(v)
                        for k, v in list((group.get("data_kinds", {}) or {}).items())[:8]
                    },
                    "short_evidence": (
                        f"Filename group `{sample_id}` has {len(files)} files; "
                        f"representatives: {', '.join(reps)}; "
                        f"shared fields: {', '.join(shared_fields[:12]) or 'unknown'}; "
                        f"variant fields: {len(field_presence)}."
                    ),
                }
            )
        )
    return out


def compact_constraint_memory(memory: Any) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    return {
        "summary": str(memory.get("summary", ""))[:1200],
        "items": memory.get("items", [])[:40] if isinstance(memory.get("items", []), list) else [],
        "unresolved_questions": [str(x)[:500] for x in (memory.get("unresolved_questions", []) or [])[:20]],
    }


def compact_authoritative_memory(memory: Any) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    return {
        "summary": str(memory.get("summary", ""))[:1500],
        "task_goal": str(memory.get("task_goal", ""))[:1500],
        "source_files": [str(x) for x in (memory.get("source_files", []) or [])[:30]],
        "input_requirements": [str(x)[:500] for x in (memory.get("input_requirements", []) or [])[:20]],
        "output_requirements": [str(x)[:500] for x in (memory.get("output_requirements", []) or [])[:20]],
        "evaluation_requirements": [str(x)[:500] for x in (memory.get("evaluation_requirements", []) or [])[:20]],
        "constraints": [str(x)[:500] for x in (memory.get("constraints", []) or [])[:30]],
        "leakage_guards": [str(x)[:500] for x in (memory.get("leakage_guards", []) or [])[:20]],
        "unresolved_questions": [str(x)[:500] for x in (memory.get("unresolved_questions", []) or [])[:20]],
        "authority_conflicts": (memory.get("authority_conflicts", []) or [])[:20],
        "submission_contract": memory.get("submission_contract", {}) if isinstance(memory.get("submission_contract", {}), dict) else {},
    }


def compact_sampled_patterns(patterns: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in list(patterns or [])[:40]:
        if not isinstance(item, dict):
            continue
        out.append(
            _drop_empty(
                {
                    "directory": str(item.get("directory", "")),
                    "pattern": str(item.get("pattern", "")),
                    "total": item.get("total"),
                    "sampled_count": len(item.get("sampled", []) or []),
                    "skipped_count": len(item.get("skipped", []) or []),
                    "sampled": [str(x) for x in (item.get("sampled", []) or [])[:3]],
                    "sampling_reason": str(item.get("sampling_reason", ""))[:300],
                }
            )
        )
    return out


def compact_field_glossary(glossary: Any) -> list[dict[str, Any]]:
    if not isinstance(glossary, dict):
        return []
    priority = []
    fallback = []
    for field, info in glossary.items():
        meaning = str((info or {}).get("meaning", "") if isinstance(info, dict) else "")
        item = {
            "field": str(field),
            "meaning": meaning[:180],
            "role": field_role(str(field), meaning),
            "files": [str(x) for x in ((info or {}).get("files", []) if isinstance(info, dict) else [])[:5]],
        }
        if item["role"] != "other":
            priority.append(item)
        else:
            fallback.append(item)
    return (priority + fallback)[:60]


def context_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    safe = json_safe(payload)
    text = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    top_level_chars: dict[str, int] = {}
    if isinstance(safe, dict):
        for key, value in safe.items():
            top_level_chars[str(key)] = len(
                json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            )
    warnings = []
    if len(text) > 120_000:
        warnings.append("large_stable_pack_over_120k_chars")
    if any(v > 60_000 for v in top_level_chars.values()):
        warnings.append("large_top_level_prompt_part_over_60k_chars")
    forbidden = _contains_forbidden_large_keys(payload)
    return {
        "chars": len(text),
        "estimated_tokens": max(1, len(text) // 4) if text else 0,
        "sha256_16": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16],
        "table_cards": len(payload.get("table_cards", []) or payload.get("table_index", []) or payload.get("files", []) or []),
        "relation_cards": len(payload.get("relations", []) or []),
        "artifact_refs": _count_artifact_refs(payload),
        "top_level_chars": dict(sorted(top_level_chars.items(), key=lambda item: item[1], reverse=True)[:12]),
        "contains_forbidden_large_keys": forbidden,
        "headroom_warnings": warnings + (["forbidden_large_keys_present"] if forbidden else []),
    }


def _field_cards(columns: list[str], profiles: Any, semantics: Any, *, limit: int) -> list[dict[str, Any]]:
    profile_by_name = {
        str(p.get("name", "")): p
        for p in (profiles or [])
        if isinstance(p, dict) and str(p.get("name", "")).strip()
    }
    semantics = semantics if isinstance(semantics, dict) else {}
    ordered = _order_columns(columns, semantics, profile_by_name)[: max(1, int(limit))]
    cards: list[dict[str, Any]] = []
    for col in ordered:
        profile = profile_by_name.get(col, {})
        meaning = str(semantics.get(col, "") or "").strip()
        numeric = _small_dict(profile.get("numeric_stats", {}), ["mean", "std", "var", "min", "max"])
        datetime_stats = _small_dict(profile.get("datetime_stats", {}), ["min", "max", "range_days", "granularity"])
        cards.append(
            _drop_empty(
                {
                    "name": col,
                    "meaning": meaning[:260],
                    "role": field_role(col, meaning),
                    "logical_type": profile.get("logical_type") or profile.get("dtype"),
                    "row_count": profile.get("row_count"),
                    "non_null_count": profile.get("non_null_count"),
                    "null_ratio": profile.get("null_ratio"),
                    "unique_count": profile.get("unique_count"),
                    "top_values": _top_values(profile.get("top_values")),
                    "numeric_stats": numeric,
                    "datetime_stats": datetime_stats,
                }
            )
        )
    return cards


def _order_columns(columns: list[str], semantics: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> list[str]:
    priority: list[str] = []
    for col in columns:
        text = f"{col} {semantics.get(col, '')}".lower()
        if field_role(col, str(semantics.get(col, ""))) != "other" or _has_profile_signal(profiles.get(col, {})):
            priority.append(col)
    out = []
    for col in priority + columns:
        if col not in out:
            out.append(col)
    return out


def field_role(field: str, meaning: str = "") -> str:
    text = f"{field} {meaning}".lower()
    if any(k in text for k in ["id", "key", "code", "??", "??", "???", "??", "??", "??", "??"]):
        return "id_or_key"
    if any(k in text for k in ["date", "time", "??", "??", "??", "??"]):
        return "time"
    if any(k in text for k in ["target", "label", "??", "??"]):
        return "target"
    if any(k in text for k in ["cost", "price", "amount", "fee", "rate", "??", "??", "??", "??", "??", "score", "??"]):
        return "cost_or_value"
    if any(k in text for k in ["constraint", "limit", "capacity", "??", "??", "??", "??", "??"]):
        return "constraint"
    if any(k in text for k in ["submission", "output", "??", "??", "??", "????"]):
        return "output"
    return "other"

def _reading_notes(path: str, meta: dict[str, Any], *, sheet_name: str = "", sheet: dict[str, Any] | None = None) -> list[str]:
    notes: list[str] = []
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        sheet = sheet if isinstance(sheet, dict) else {}
        layout_kind = str(sheet.get("layout_kind", "") or "")
        recommended = str(sheet.get("recommended_read", "") or "")
        detected = sheet.get("detected_header_row")
        if sheet_name:
            if recommended:
                notes.append(f"Recommended read for this sheet: {recommended}.")
            else:
                notes.append(f"Read this Excel sheet explicitly: pandas.read_excel(path, sheet_name={sheet_name!r}).")
        else:
            notes.append("Multi-sheet Excel: read needed sheets explicitly with pandas.read_excel(..., sheet_name=...).")
        if layout_kind == "headerless_table":
            notes.append("Detected likely headerless table; do not trust pandas default columns until header=None is checked.")
        elif layout_kind == "non_default_header":
            notes.append(f"Detected likely non-default header row {detected}; use explicit header row or inspect header=None.")
        elif layout_kind == "document_like_sheet":
            notes.append("Detected document-like sheet; treat it as notes/rules/key-value content rather than an ordinary dataframe.")
        elif layout_kind == "sparse_or_irregular_sheet":
            notes.append("Detected sparse/irregular sheet; inspect with header=None before deciding a table schema.")
        raw_preview = sheet.get("raw_preview")
        if raw_preview:
            notes.append("This sheet has header=None raw preview in artifact; opening rows may contain notes or non-default headers.")
    if lower.endswith(".csv"):
        dialect = meta.get("csv_dialect") if isinstance(meta.get("csv_dialect"), dict) else {}
        encoding = str(meta.get("csv_encoding", "") or "")
        sep = dialect.get("sep") or dialect.get("delimiter")
        engine = dialect.get("engine")
        if encoding and encoding.lower() not in {"utf-8", "utf-8-sig"}:
            notes.append(f"CSV encoding hint: {encoding}")
        if sep and str(sep) not in {",", ""}:
            notes.append(f"CSV non-default separator hint: sep={sep!r}, engine={engine or 'default'}")
    if lower.endswith(".json"):
        strategy = str(meta.get("json_strategy", "") or "")
        if strategy:
            notes.append(f"JSON parse strategy: {strategy}")
    for item in meta.get("read_examples", []) or []:
        text = str(item).strip()
        if text:
            notes.append(text[:260])
    return list(dict.fromkeys(notes))[:8]


def _warnings(fs: Any, sheet: dict[str, Any]) -> list[str]:
    out = [str(x)[:260] for x in (getattr(fs, "warnings", []) or [])[:5]]
    for risk in (sheet.get("reading_risks") or [])[:4] if isinstance(sheet, dict) else []:
        text = str(risk).strip()
        if text:
            out.append(text[:260])
    sampling = sheet.get("profile_sampling") if isinstance(sheet, dict) and isinstance(sheet.get("profile_sampling"), dict) else {}
    if sampling.get("sampled"):
        out.append(f"Profile statistics are sampled: {sampling.get('rows_read')} rows.")
    return list(dict.fromkeys(out))[:8]


def _file_payload(fs: Any) -> dict[str, Any]:
    if hasattr(fs, "model_dump"):
        return fs.model_dump()
    return dict(getattr(fs, "__dict__", {}) or {})


def _table_kind(path: str, meta: dict[str, Any]) -> str:
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv_table"
    if lower.endswith(".json") or meta.get("json_strategy"):
        return "json_table_or_document"
    if lower.endswith((".xlsx", ".xls")):
        return "excel_workbook"
    if lower.endswith((".md", ".txt", ".pdf", ".doc", ".docx")):
        return "document"
    return "file"


def _top_values(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return json_safe(value[:6])


def _small_dict(value: Any, keys: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {k: value.get(k) for k in keys if value.get(k) not in (None, "", [], {})}


def _has_profile_signal(profile: dict[str, Any]) -> bool:
    if not isinstance(profile, dict):
        return False
    text = " ".join(str(profile.get(k, "")) for k in ["logical_type", "dtype"]).lower()
    return any(k in text for k in ["datetime", "date", "numeric", "float", "int"])


def _shared_columns_for_files(files: list[str], summary_by_path: dict[str, Any]) -> list[str]:
    sets = []
    for path in files:
        fs = summary_by_path.get(path)
        cols = [str(x) for x in (getattr(fs, "columns", []) or [])] if fs is not None else []
        if cols:
            sets.append(set(cols))
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def _filename_group_column_profile(files: list[str], summary_by_path: dict[str, Any]) -> dict[str, Any]:
    """Summarize shared and per-file variant columns for a filename group."""
    observed: list[tuple[str, list[str]]] = []
    layout_observed: list[tuple[str, list[dict[str, Any]]]] = []
    for path in files:
        fs = summary_by_path.get(str(path))
        cols = [str(x) for x in (getattr(fs, "columns", []) or [])] if fs is not None else []
        cols = [x for x in cols if x.strip()]
        if cols:
            observed.append((str(path), cols))
        layouts = _excel_layouts_for_summary(fs) if fs is not None else []
        if layouts:
            layout_observed.append((str(path), layouts))
    if not observed and not layout_observed:
        return {}

    sets = [set(cols) for _, cols in observed]
    common_set = set.intersection(*sets) if sets else set()
    union: list[str] = []
    for _, cols in observed:
        for col in cols:
            if col not in union:
                union.append(col)

    shared_fields = [col for col in union if col in common_set]
    variant_fields_by_file: list[dict[str, Any]] = []
    for path, cols in observed[:16]:
        only_fields = [col for col in cols if col not in common_set]
        if only_fields:
            variant_fields_by_file.append(
                {
                    "file": path,
                    "fields": only_fields[:24],
                    "omitted": max(0, len(only_fields) - 24),
                }
            )

    field_presence: list[dict[str, Any]] = []
    for col in union:
        if col in common_set:
            continue
        present = [path for path, cols in observed if col in set(cols)]
        field_presence.append(
            {
                "field": col,
                "present_in_count": len(present),
                "example_files": present[:3],
            }
        )

    layout_variants: list[dict[str, Any]] = []
    layout_buckets: dict[tuple[str, str, str, str], list[str]] = {}
    for path, layouts in layout_observed:
        for layout in layouts:
            key = (
                str(layout.get("sheet_name", "")),
                str(layout.get("layout_kind", "")),
                str(layout.get("read_strategy_kind", "")),
                str(layout.get("detected_header_row", "")),
            )
            layout_buckets.setdefault(key, []).append(path)
    for (sheet_name, layout_kind, read_strategy, detected), paths in sorted(layout_buckets.items()):
        if layout_kind and layout_kind != "standard_table":
            layout_variants.append(
                {
                    "sheet_name": sheet_name,
                    "layout_kind": layout_kind,
                    "read_strategy_kind": read_strategy,
                    "detected_header_row": detected,
                    "present_in_count": len(paths),
                    "example_files": paths[:3],
                }
            )

    return {
        "observed_file_count": len(observed),
        "shared_fields": shared_fields,
        "variant_fields_by_file": variant_fields_by_file,
        "field_presence": field_presence,
        "layout_variants": layout_variants,
    }


def _excel_layouts_for_summary(fs: Any) -> list[dict[str, Any]]:
    meta = getattr(fs, "source_metadata", {}) if fs is not None else {}
    if not isinstance(meta, dict):
        return []
    sheets = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
    if not sheets:
        sheets = meta.get("excel_sheets") if isinstance(meta.get("excel_sheets"), list) else []
    out: list[dict[str, Any]] = []
    for sheet in sheets[:40]:
        if not isinstance(sheet, dict):
            continue
        out.append(
            _drop_empty(
                {
                    "sheet_name": str(sheet.get("sheet_name", "")),
                    "layout_kind": str(sheet.get("layout_kind", "")),
                    "read_strategy_kind": str(sheet.get("read_strategy_kind", "")),
                    "detected_header_row": sheet.get("detected_header_row"),
                }
            )
        )
    return out


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {k: json_safe(v) for k, v in data.items() if v not in (None, "", [], {})}


def _abnormal_layout_value(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "standard_table", "default_header"}:
        return ""
    return text


def _safe_id(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text).strip().lower())
    return safe.strip("_") or "artifact"


def _count_artifact_refs(value: Any) -> int:
    if isinstance(value, dict):
        count = 0
        if {"artifact_id", "artifact_type", "source"}.issubset(value.keys()):
            count += 1
        for item in value.values():
            count += _count_artifact_refs(item)
        return count
    if isinstance(value, list):
        return sum(_count_artifact_refs(x) for x in value)
    return 0


def _contains_forbidden_large_keys(value: Any) -> list[str]:
    forbidden = {"source_metadata", "raw_preview", "probe_results", "excel_sheet_profiles", "detailed_report"}
    found: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if str(key) in forbidden:
                    found.add(str(key))
                walk(val)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(value)
    return sorted(found)
