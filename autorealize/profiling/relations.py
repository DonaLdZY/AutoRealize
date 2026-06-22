from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class RelationHint:
    """Field-level relation hint with legacy compatibility fields."""

    left_file: str
    right_file: str
    shared_columns: list[str] = field(default_factory=list)
    reason: str = ""
    left_field: str = ""
    right_field: str = ""
    relation_type: str = "shared_attribute"
    confidence: float = 0.5
    short_evidence: str = ""


def detect_relations(
    file_columns: dict[str, list[str]],
    *,
    file_summaries: list[Any] | None = None,
    parallel: bool = False,
    max_workers: int = 4,
) -> list[RelationHint]:
    """Discover compact field-level relation hints.

    The old implementation only emitted file-pair shared column names. This
    version keeps those compatibility fields, but prefers one RelationHint per
    candidate field pair with rule-generated cardinality/evidence.
    """

    summary_index = _build_summary_index(file_summaries or [])
    items = list(file_columns.items())
    pairs: list[tuple[tuple[str, list[str]], tuple[str, list[str]]]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append((items[i], items[j]))

    def _work(pair: tuple[tuple[str, list[str]], tuple[str, list[str]]]) -> list[RelationHint]:
        (lf, lcols), (rf, rcols) = pair
        candidates = _candidate_field_pairs(lcols, rcols, summary_index.get(lf, {}), summary_index.get(rf, {}))
        hints: list[RelationHint] = []
        for left_field, right_field, match_reason in candidates[:24]:
            hint = _build_relation_hint(
                lf,
                left_field,
                rf,
                right_field,
                match_reason=match_reason,
                left_summary=summary_index.get(lf, {}),
                right_summary=summary_index.get(rf, {}),
            )
            if hint.confidence >= 0.35:
                hints.append(hint)
        return hints

    hints: list[RelationHint] = []
    if parallel and len(pairs) > 4:
        workers = max(1, int(max_workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_work, p) for p in pairs]
            for fut in as_completed(futures):
                hints.extend(fut.result())
    else:
        for p in pairs:
            hints.extend(_work(p))

    hints.sort(key=lambda h: (-float(h.confidence or 0), h.left_file, h.right_file, h.left_field, h.right_field))
    return _dedupe_hints(hints)[:500]


def _build_summary_index(file_summaries: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fs in file_summaries:
        path = str(getattr(fs, "path", "") or "")
        if not path:
            continue
        profiles = {}
        for item in getattr(fs, "column_profiles", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "")
            if name:
                profiles[_norm_field(name)] = item
        semantics = {
            _norm_field(str(k)): str(v)
            for k, v in (getattr(fs, "column_semantics", {}) or {}).items()
            if str(k).strip()
        }
        out[path] = {
            "path": path,
            "columns": [str(x) for x in (getattr(fs, "columns", []) or [])],
            "profiles": profiles,
            "semantics": semantics,
            "source_metadata": getattr(fs, "source_metadata", {}) or {},
        }
    return out


def _candidate_field_pairs(
    left_cols: list[str],
    right_cols: list[str],
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str, float]] = []
    right_by_norm = {_norm_field(c): str(c) for c in right_cols}
    right_tokens = {_norm_field(c): _field_tokens(c, right_summary) for c in right_cols}
    for left in left_cols:
        left_norm = _norm_field(left)
        if not left_norm:
            continue
        if left_norm in right_by_norm:
            candidates.append((str(left), right_by_norm[left_norm], "字段名完全匹配", 1.0))
            continue
        left_tok = _field_tokens(left, left_summary)
        for right, rtok in right_tokens.items():
            score = _token_similarity(left_tok, rtok)
            if score >= 0.72:
                candidates.append((str(left), right_by_norm.get(right, right), "字段名/语义近似匹配", score))
    candidates.sort(key=lambda x: -x[3])
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for left, right, reason, _score in candidates:
        key = (_norm_field(left), _norm_field(right))
        if key in seen:
            continue
        seen.add(key)
        out.append((left, right, reason))
    return out


def _build_relation_hint(
    left_file: str,
    left_field: str,
    right_file: str,
    right_field: str,
    *,
    match_reason: str,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> RelationHint:
    left_profile = _profile_for(left_summary, left_field)
    right_profile = _profile_for(right_summary, right_field)
    left_unique = _unique_ratio(left_profile)
    right_unique = _unique_ratio(right_profile)
    left_repeats = _repeats(left_profile)
    right_repeats = _repeats(right_profile)
    type_consistent = _type_consistent(left_profile, right_profile)
    value_overlap = _top_value_overlap(left_profile, right_profile)
    relation_type = _relation_type(left_unique, right_unique, left_repeats, right_repeats)

    confidence = 0.42
    if _norm_field(left_field) == _norm_field(right_field):
        confidence += 0.22
    elif match_reason:
        confidence += 0.10
    if type_consistent:
        confidence += 0.10
    if value_overlap is not None:
        confidence += min(0.18, value_overlap * 0.18)
    if relation_type != "shared_attribute":
        confidence += 0.10
    if _same_workbook_or_filename_family(left_file, right_file, left_summary, right_summary):
        confidence += 0.06
    confidence = round(max(0.0, min(0.98, confidence)), 3)

    evidence = _short_evidence(
        left_file,
        left_field,
        right_file,
        right_field,
        match_reason=match_reason,
        relation_type=relation_type,
        left_unique=left_unique,
        right_unique=right_unique,
        left_repeats=left_repeats,
        right_repeats=right_repeats,
        type_consistent=type_consistent,
        value_overlap=value_overlap,
        same_family=_same_workbook_or_filename_family(left_file, right_file, left_summary, right_summary),
    )
    return RelationHint(
        left_file=left_file,
        right_file=right_file,
        shared_columns=sorted({_norm_field(left_field), _norm_field(right_field)}),
        reason=evidence,
        left_field=left_field,
        right_field=right_field,
        relation_type=relation_type,
        confidence=confidence,
        short_evidence=evidence,
    )


def _relation_type(
    left_unique: float | None,
    right_unique: float | None,
    left_repeats: bool | None,
    right_repeats: bool | None,
) -> str:
    left_is_unique = bool(left_unique is not None and left_unique >= 0.98)
    right_is_unique = bool(right_unique is not None and right_unique >= 0.98)
    left_is_repeat = bool(left_repeats) if left_repeats is not None else False
    right_is_repeat = bool(right_repeats) if right_repeats is not None else False
    if left_is_unique and right_is_unique:
        return "one_to_one"
    if left_is_unique and (right_is_repeat or (right_unique is not None and right_unique < 0.98)):
        return "one_to_many"
    if right_is_unique and (left_is_repeat or (left_unique is not None and left_unique < 0.98)):
        return "many_to_one"
    if left_is_repeat and right_is_repeat:
        return "many_to_many"
    return "shared_attribute"


def _short_evidence(
    left_file: str,
    left_field: str,
    right_file: str,
    right_field: str,
    *,
    match_reason: str,
    relation_type: str,
    left_unique: float | None,
    right_unique: float | None,
    left_repeats: bool | None,
    right_repeats: bool | None,
    type_consistent: bool,
    value_overlap: float | None,
    same_family: bool,
) -> str:
    parts = [f"`{left_file}`.`{left_field}` 与 `{right_file}`.`{right_field}`候选关联：{match_reason}。"]
    if left_unique is not None:
        parts.append(f"左侧唯一率约 {left_unique:.3g}" + ("，近似唯一" if left_unique >= 0.98 else "，存在重复"))
    if right_unique is not None:
        parts.append(f"右侧唯一率约 {right_unique:.3g}" + ("，近似唯一" if right_unique >= 0.98 else "，存在重复"))
    if left_repeats and right_repeats:
        parts.append("两侧都出现重复值")
    elif left_repeats:
        parts.append("左侧出现重复值")
    elif right_repeats:
        parts.append("右侧出现重复值")
    if value_overlap is not None:
        parts.append(f"top 值重叠率约 {value_overlap:.2g}")
    if type_consistent:
        parts.append("字段类型一致")
    if same_family:
        parts.append("文件名模式或 workbook/sheet 来源支持该关系")
    parts.append(f"推断关系类型为 {relation_type}")
    return "；".join(parts)[:800]


def _profile_for(summary: dict[str, Any], field_name: str) -> dict[str, Any]:
    return dict((summary.get("profiles") or {}).get(_norm_field(field_name), {}) or {})


def _unique_ratio(profile: dict[str, Any]) -> float | None:
    try:
        unique = float(profile.get("unique_count"))
        non_null = float(profile.get("non_null_count") or 0)
        row_count = float(profile.get("row_count") or 0)
        denom = non_null or row_count
        if denom > 0:
            return max(0.0, min(1.0, unique / denom))
    except Exception:
        return None
    return None


def _repeats(profile: dict[str, Any]) -> bool | None:
    ratio = _unique_ratio(profile)
    if ratio is None:
        return None
    try:
        unique = float(profile.get("unique_count"))
        non_null = float(profile.get("non_null_count") or profile.get("row_count") or 0)
        return bool(non_null - unique > 0.5)
    except Exception:
        return ratio < 0.98


def _type_consistent(left_profile: dict[str, Any], right_profile: dict[str, Any]) -> bool:
    left_type = str(left_profile.get("logical_type") or left_profile.get("dtype") or "").lower()
    right_type = str(right_profile.get("logical_type") or right_profile.get("dtype") or "").lower()
    if not left_type or not right_type:
        return False
    if left_type == right_type:
        return True
    text_markers = ["text", "string", "object", "category"]
    num_markers = ["numeric", "int", "float", "number"]
    return (
        any(x in left_type for x in text_markers)
        and any(x in right_type for x in text_markers)
    ) or (
        any(x in left_type for x in num_markers)
        and any(x in right_type for x in num_markers)
    )


def _top_value_overlap(left_profile: dict[str, Any], right_profile: dict[str, Any]) -> float | None:
    left_values = _top_values(left_profile)
    right_values = _top_values(right_profile)
    if not left_values or not right_values:
        return None
    union = left_values | right_values
    return len(left_values & right_values) / max(1, len(union))


def _top_values(profile: dict[str, Any]) -> set[str]:
    raw = profile.get("top_values") or profile.get("sample_values") or []
    values: set[str] = set()
    if isinstance(raw, list):
        for item in raw[:20]:
            text = str(item)
            if ":" in text:
                text = text.split(":", 1)[0]
            text = text.strip()
            if text:
                values.add(text.lower())
    return values


def _same_workbook_or_filename_family(
    left_file: str,
    right_file: str,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> bool:
    if left_file == right_file:
        return True
    left_meta = left_summary.get("source_metadata") or {}
    right_meta = right_summary.get("source_metadata") or {}
    if left_meta.get("filename_sample_id") and left_meta.get("filename_sample_id") == right_meta.get("filename_sample_id"):
        return True
    return _filename_pattern(left_file) == _filename_pattern(right_file)


def _filename_pattern(path: str) -> str:
    name = str(path or "").replace("\\", "/").lower()
    name = re.sub(r"\d+", "{num}", name)
    name = re.sub(r"[0-9a-f]{6,}", "{id}", name)
    return name


def _field_tokens(name: str, summary: dict[str, Any]) -> set[str]:
    text = f"{name} {(summary.get('semantics') or {}).get(_norm_field(name), '')}"
    return {x for x in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", text.lower()) if x}


def _token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _norm_field(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _dedupe_hints(hints: list[RelationHint]) -> list[RelationHint]:
    out: list[RelationHint] = []
    seen: set[tuple[str, str, str, str]] = set()
    for hint in hints:
        key = (
            str(hint.left_file),
            _norm_field(hint.left_field),
            str(hint.right_file),
            _norm_field(hint.right_field),
        )
        reverse = (key[2], key[3], key[0], key[1])
        if key in seen or reverse in seen:
            continue
        seen.add(key)
        out.append(hint)
    return out
