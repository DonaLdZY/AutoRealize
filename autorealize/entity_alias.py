from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

from .profiling.csv_utils import read_csv_auto


def build_entity_alias_candidates(
    file_summaries: list[Any],
    *,
    llm_candidates: list[Any] | None = None,
    limit: int = 80,
    filename_sample_groups: list[Any] | None = None,
    source_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Validate LLM-proposed aliases against the exact parsed physical schema."""

    if not llm_candidates:
        return []
    source_display = _build_source_display_map(filename_sample_groups, source_aliases)
    source_paths = [str(getattr(fs, "path", "") or "") for fs in file_summaries or []]
    parent_counts = Counter(
        str(Path(path.replace("\\", "/")).parent).replace("\\", "/")
        for path in source_paths
        if path
    )
    schema = _physical_schema_index(file_summaries)
    groups: list[dict[str, Any]] = []
    total_fields = 0
    for group_index, raw_group in enumerate(llm_candidates[:20], start=1):
        group = _as_dict(raw_group)
        raw_fields = group.get("candidate_fields") if isinstance(group.get("candidate_fields"), list) else []
        fields: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for raw_field in raw_fields:
            candidate = _as_dict(raw_field)
            source_requested = str(candidate.get("source_file", "") or "").strip().replace("\\", "/")
            sheet_requested = str(candidate.get("sheet_name", "") or "").strip()
            field_requested = str(candidate.get("field", "") or "").strip()
            resolved = _resolve_physical_field(
                schema,
                source_file=source_requested,
                sheet_name=sheet_requested,
                field=field_requested,
            )
            if resolved is None:
                rejected.append(
                    {
                        "source_file": source_requested,
                        "sheet_name": sheet_requested,
                        "field": field_requested,
                        "reason": "not_found_in_exact_parsed_schema",
                    }
                )
                continue
            source_file, sheet_name, field = resolved
            key = (source_file, sheet_name, field)
            if key in seen:
                continue
            seen.add(key)
            display_source = source_display.get(source_file, source_file)
            parent = str(Path(source_file.replace("\\", "/")).parent).replace("\\", "/")
            if display_source != source_file:
                source_collection = display_source
            elif parent not in {"", "."} and parent_counts[parent] >= 3:
                source_collection = parent
            else:
                source_collection = source_file
            if sheet_name:
                source_collection = f"{source_collection}::{sheet_name}"
            fields.append(
                {
                    "source_file": source_file,
                    "source_collection": source_collection,
                    "sheet_name": sheet_name,
                    "field": field,
                    "alias_family": str(candidate.get("semantic_role", "") or "entity_key").strip(),
                    "value_kind": _normalized_value_kind(candidate.get("value_kind")),
                    "evidence": str(candidate.get("evidence", "") or "")[:300],
                    "status": "llm_candidate_schema_verified",
                }
            )
            total_fields += 1
            if total_fields >= max(1, int(limit)):
                break
        distinct_refs = {(item["source_collection"], item["field"]) for item in fields}
        if len(distinct_refs) < 2:
            continue
        concept_id = _safe_concept_id(group.get("concept_id"), fallback=f"entity_alias_{group_index}")
        groups.append(
            {
                "concept_id": concept_id,
                "label": str(group.get("label", "") or concept_id)[:160],
                "reason": str(group.get("reason", "") or "")[:500],
                "confidence": str(group.get("confidence", "low") or "low"),
                "task_relevance": _normalized_relevance(group.get("task_relevance")),
                "relevance_reason": str(group.get("relevance_reason", "") or "")[:300],
                "status": "candidate_not_equivalent",
                "candidate_fields": fields,
                "alias_families": sorted({item["alias_family"] for item in fields}),
                "value_kinds": sorted({item["value_kind"] for item in fields}),
                "caution": (
                    "This semantic alias was proposed by an LLM and its field references were verified against the "
                    "physical schema. Equivalence is still unconfirmed until value overlap, directional coverage, "
                    "join behavior, or authoritative evidence supports it."
                ),
                "schema_validation": {
                    "accepted_field_count": len(fields),
                    "rejected_fields": rejected[:12],
                },
                "recommended_qdi_checks": [
                    "Union non-null unique values within each source_collection before comparing source collections.",
                    "Report intersection count and both directional coverage ratios with explicit denominators.",
                    "Do not promote semantic similarity to key equivalence when value evidence is weak or asymmetric.",
                ],
            }
        )
        if total_fields >= max(1, int(limit)):
            break
    return groups


def enrich_entity_alias_candidates_with_coverage(
    candidates: list[dict[str, Any]],
    *,
    data_root: Path,
    max_unique_values: int = 100_000,
) -> list[dict[str, Any]]:
    """Measure collection-level directional coverage for candidate aliases.

    Values are unioned within each source collection before comparison. This
    avoids the common error of comparing one repeated contract/resource file
    with a complete master table and treating that asymmetric percentage as a
    global mapping failure.
    """

    enriched: list[dict[str, Any]] = []
    for group in candidates or []:
        if not isinstance(group, dict):
            continue
        item = dict(group)
        fields = [field for field in (item.get("candidate_fields") or []) if isinstance(field, dict)]
        collection_values: dict[tuple[str, str], set[str]] = {}
        family_values: dict[tuple[str, str, str], set[str]] = {}
        read_errors: list[dict[str, str]] = []
        for field in fields:
            source_file = str(field.get("source_file", "") or "").strip()
            field_name = str(field.get("field", "") or "").strip()
            if not source_file or not field_name:
                continue
            collection = str(field.get("source_collection", "") or source_file)
            value_kind = str(field.get("value_kind", "") or "unknown")
            family = str(field.get("alias_family", "") or "unknown")
            values, error = _read_alias_values(
                data_root=data_root,
                source_file=source_file,
                sheet_name=str(field.get("sheet_name", "") or ""),
                field_name=field_name,
                detected_header_row=field.get("detected_header_row"),
                max_unique_values=max_unique_values,
            )
            if error:
                read_errors.append({"source_file": source_file, "field": field_name, "error": error})
                continue
            collection_values.setdefault((collection, value_kind), set()).update(values)
            family_values.setdefault((collection, value_kind, family), set()).update(values)

        profiles = []
        for (collection, value_kind), values in sorted(collection_values.items()):
            family_counts = {
                family: len(family_set)
                for (family_collection, family_kind, family), family_set in sorted(family_values.items())
                if family_collection == collection and family_kind == value_kind
            }
            profiles.append(
                {
                    "source_collection": collection,
                    "value_kind": value_kind,
                    "unique_value_count": len(values),
                    "alias_family_unique_counts": family_counts,
                    "value_sample": sorted(values)[:12],
                }
            )

        coverage = []
        collection_items = sorted(collection_values.items())
        for idx, ((left_collection, left_kind), left_values) in enumerate(collection_items):
            for (right_collection, right_kind), right_values in collection_items[idx + 1 :]:
                if left_collection == right_collection or left_kind != right_kind:
                    continue
                intersection = left_values & right_values
                coverage.append(
                    {
                        "left_collection": left_collection,
                        "right_collection": right_collection,
                        "value_kind": left_kind,
                        "left_unique_count": len(left_values),
                        "right_unique_count": len(right_values),
                        "intersection_count": len(intersection),
                        "left_covered_by_right_ratio": round(len(intersection) / max(1, len(left_values)), 6),
                        "right_covered_by_left_ratio": round(len(intersection) / max(1, len(right_values)), 6),
                        "status": "deterministic_collection_union_evidence",
                    }
                )
        compact_errors = []
        seen_errors: set[tuple[str, str]] = set()
        for error_item in read_errors:
            source = str(error_item.get("source_file", "") or "")
            collection = next(
                (
                    str(field.get("source_collection", "") or source)
                    for field in fields
                    if str(field.get("source_file", "") or "") == source
                ),
                source,
            )
            error = str(error_item.get("error", "") or "")
            key = (collection, error)
            if key in seen_errors:
                continue
            seen_errors.add(key)
            compact_errors.append(
                {
                    "source_collection": collection,
                    "error": error,
                    "affected_member_count": sum(
                        1
                        for candidate in read_errors
                        if str(candidate.get("error", "") or "") == error
                        and any(
                            str(field.get("source_collection", "") or field.get("source_file", "")) == collection
                            and str(field.get("source_file", "") or "") == str(candidate.get("source_file", "") or "")
                            for field in fields
                        )
                    ),
                }
            )
        item["deterministic_group_coverage"] = {
            "policy": (
                "Unique values are unioned across every readable member in a source_collection before directional "
                "coverage is computed. Ratios are asymmetric and their denominators are named explicitly."
            ),
            "collection_profiles": profiles,
            "directional_coverage": coverage,
            "read_errors": compact_errors[:20],
        }
        item["evidence_status"] = _alias_evidence_status(coverage, compact_errors)
        item["qdi_routing"] = _alias_qdi_routing(item)
        enriched.append(item)
    return enriched


def _alias_evidence_status(coverage: list[dict[str, Any]], read_errors: list[dict[str, Any]]) -> str:
    if not coverage:
        return "value_evidence_unavailable" if read_errors else "semantic_candidate"
    intersections = [int(item.get("intersection_count") or 0) for item in coverage]
    if intersections and all(value == 0 for value in intersections):
        return "value_disjoint"
    ratio_pairs = [
        (
            float(item.get("left_covered_by_right_ratio") or 0.0),
            float(item.get("right_covered_by_left_ratio") or 0.0),
        )
        for item in coverage
    ]
    if read_errors:
        return "partial_with_read_errors"
    if ratio_pairs and all(min(left, right) >= 0.95 for left, right in ratio_pairs):
        return "strong_value_overlap"
    if any(max(left, right) >= 0.9 and min(left, right) < 0.9 for left, right in ratio_pairs):
        return "asymmetric_coverage"
    if any(value == 0 for value in intersections) and any(value > 0 for value in intersections):
        return "mixed_value_evidence"
    return "partial_value_overlap"


def _alias_qdi_routing(group: dict[str, Any]) -> dict[str, Any]:
    evidence_status = str(group.get("evidence_status", "semantic_candidate") or "semantic_candidate")
    relevance = _normalized_relevance(group.get("task_relevance"))
    if relevance == "low":
        return {"recommended": False, "reason": "low_task_relevance"}
    if evidence_status == "strong_value_overlap":
        return {"recommended": False, "reason": "complete_strong_value_evidence_already_visible"}
    if evidence_status == "value_disjoint" and relevance != "high":
        return {"recommended": False, "reason": "disjoint_values_without_high_task_relevance"}
    return {
        "recommended": True,
        "reason": f"{relevance}_task_relevance_with_{evidence_status}",
    }


def _read_alias_values(
    *,
    data_root: Path,
    source_file: str,
    sheet_name: str,
    field_name: str,
    detected_header_row: Any,
    max_unique_values: int,
) -> tuple[set[str], str]:
    path = Path(source_file)
    if not path.is_absolute():
        path = data_root / source_file
    if not path.is_file():
        return set(), "source_not_found"
    try:
        import pandas as pd

        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            headers: list[int | None] = []
            try:
                headers.append(int(detected_header_row))
            except (TypeError, ValueError):
                headers.append(0)
            headers.extend(value for value in range(0, 11) if value not in headers)
            frame = None
            actual_field = ""
            for header in headers:
                probe = pd.read_excel(path, sheet_name=sheet_name or 0, header=header)
                columns = {_normalize_column_name(column): str(column) for column in probe.columns}
                actual_field = columns.get(_normalize_column_name(field_name), "")
                if actual_field:
                    frame = probe
                    break
            if frame is None:
                return set(), "field_not_found_in_header_rows_0_10"
        elif suffix in {".csv", ".tsv"}:
            frame = read_csv_auto(path, sep="\t") if suffix == ".tsv" else read_csv_auto(path)
            columns = {_normalize_column_name(column): str(column) for column in frame.columns}
            actual_field = columns.get(_normalize_column_name(field_name), "")
            if not actual_field:
                return set(), "field_not_found"
        else:
            return set(), f"unsupported_source_type:{suffix}"
        values: set[str] = set()
        for raw in frame[actual_field].dropna().tolist():
            normalized = _normalize_entity_value(raw)
            if normalized:
                values.add(normalized)
            if len(values) >= max(1, int(max_unique_values)):
                break
        return values, ""
    except Exception as exc:  # noqa: BLE001
        return set(), f"{type(exc).__name__}: {str(exc)[:240]}"


def _normalize_column_name(value: Any) -> str:
    return "".join(str(value or "").strip().split()).casefold()


def _normalize_entity_value(value: Any) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "nat"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return "".join(text.split()).casefold()


def _build_source_display_map(
    filename_sample_groups: list[Any] | None,
    source_aliases: dict[str, str] | None,
) -> dict[str, str]:
    out = {
        str(k): str(v)
        for k, v in (source_aliases or {}).items()
        if str(k).strip() and str(v).strip()
    }
    for group in filename_sample_groups or []:
        if not isinstance(group, dict):
            continue
        template = str(
            group.get("sample_id")
            or group.get("pattern")
            or group.get("template_path_or_sample_id")
            or group.get("template_path")
            or ""
        ).strip()
        if not template:
            continue
        files = group.get("files")
        if not isinstance(files, list):
            files = group.get("representative_files") if isinstance(group.get("representative_files"), list) else []
        for path in files:
            text = str(path or "").strip()
            if text:
                out[text] = template
    return out


def _iter_tables(fs: Any) -> list[dict[str, Any]]:
    path = str(getattr(fs, "path", "") or "")
    meta = getattr(fs, "source_metadata", {}) or {}
    sheet_profiles = meta.get("excel_sheet_profiles") if isinstance(meta, dict) else []
    if isinstance(sheet_profiles, list) and sheet_profiles:
        out: list[dict[str, Any]] = []
        for sheet in sheet_profiles:
            if not isinstance(sheet, dict):
                continue
            out.append(
                {
                    "source_file": path,
                    "sheet_name": str(sheet.get("sheet_name", "") or ""),
                    "columns": [str(x) for x in (sheet.get("columns") or []) if str(x).strip()],
                }
            )
        return out
    return [
        {
            "source_file": path,
            "sheet_name": "",
            "columns": [str(x) for x in (getattr(fs, "columns", []) or []) if str(x).strip()],
        }
    ]


def _physical_schema_index(file_summaries: list[Any]) -> dict[str, dict[str, dict[str, str]]]:
    index: dict[str, dict[str, dict[str, str]]] = {}
    for fs in file_summaries or []:
        for table in _iter_tables(fs):
            source_file = str(table.get("source_file", "") or "").replace("\\", "/")
            sheet_name = str(table.get("sheet_name", "") or "")
            if not source_file:
                continue
            table_index = index.setdefault(source_file, {})
            table_index[sheet_name] = {
                _normalize_column_name(column): str(column)
                for column in (table.get("columns") or [])
                if str(column).strip()
            }
    return index


def _resolve_physical_field(
    schema: dict[str, dict[str, dict[str, str]]],
    *,
    source_file: str,
    sheet_name: str,
    field: str,
) -> tuple[str, str, str] | None:
    if not source_file or not field:
        return None
    source_norm = source_file.replace("\\", "/").casefold()
    source_matches = [path for path in schema if path.replace("\\", "/").casefold() == source_norm]
    if len(source_matches) != 1:
        return None
    actual_source = source_matches[0]
    sheets = schema[actual_source]
    if sheet_name:
        sheet_norm = _normalize_column_name(sheet_name)
        sheet_matches = [name for name in sheets if _normalize_column_name(name) == sheet_norm]
        if len(sheet_matches) != 1:
            return None
        actual_sheet = sheet_matches[0]
    elif len(sheets) == 1:
        actual_sheet = next(iter(sheets))
    else:
        # A workbook field reference must identify its sheet when multiple sheets exist.
        return None
    actual_field = sheets[actual_sheet].get(_normalize_column_name(field))
    if not actual_field:
        return None
    return actual_source, actual_sheet, actual_field


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        data = value.model_dump()
        return data if isinstance(data, dict) else {}
    return {}


def _safe_concept_id(value: Any, *, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "").strip()).strip("_").lower()
    return text[:80] or fallback


def _normalized_value_kind(value: Any) -> str:
    text = str(value or "unknown").strip().casefold()
    if any(token in text for token in ["code", "identifier", "id", "key", "编码", "代码", "标识"]):
        return "code"
    if any(token in text for token in ["name", "label", "名称", "姓名"]):
        return "name"
    if any(token in text for token in ["category", "type", "enum", "类别", "类型", "枚举"]):
        return "category"
    return "unknown"


def _normalized_relevance(value: Any) -> str:
    text = str(value or "medium").strip().casefold()
    if text in {"high", "medium", "low"}:
        return text
    if text in {"高", "重要", "critical"}:
        return "high"
    if text in {"低", "不相关", "irrelevant"}:
        return "low"
    return "medium"
