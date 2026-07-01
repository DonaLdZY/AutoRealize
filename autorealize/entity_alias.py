from __future__ import annotations

from typing import Any


CARRIER_SETTLEMENT_ALIASES = {
    "承运商代码": {"alias_family": "carrier_code", "value_kind": "code"},
    "结算方代码": {"alias_family": "settlement_party_code", "value_kind": "code"},
    "承运商名称": {"alias_family": "carrier_name", "value_kind": "name"},
    "结算方名称": {"alias_family": "settlement_party_name", "value_kind": "name"},
}


def build_entity_alias_candidates(
    file_summaries: list[Any],
    *,
    limit: int = 80,
    filename_sample_groups: list[Any] | None = None,
    source_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build candidate entity-alias groups without asserting equivalence."""
    source_display = _build_source_display_map(filename_sample_groups, source_aliases)
    fields: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for fs in file_summaries or []:
        for table in _iter_tables(fs):
            source_file = source_display.get(table["source_file"], table["source_file"])
            sheet_name = table["sheet_name"]
            for column in table["columns"]:
                match = _match_carrier_settlement_alias(column)
                if not match:
                    continue
                key = (source_file, sheet_name, column)
                if key in seen:
                    continue
                seen.add(key)
                fields.append(
                    {
                        "source_file": source_file,
                        "sheet_name": sheet_name,
                        "field": column,
                        "alias_family": match["alias_family"],
                        "value_kind": match["value_kind"],
                        "status": "candidate",
                    }
                )
                if len(fields) >= max(1, int(limit)):
                    break
    if not fields:
        return []

    families = sorted({str(item.get("alias_family", "")) for item in fields if item.get("alias_family")})
    value_kinds = sorted({str(item.get("value_kind", "")) for item in fields if item.get("value_kind")})
    return [
        {
            "concept_id": "carrier_settlement_entity",
            "label": "承运商/结算方实体候选字段",
            "status": "candidate_not_equivalent",
            "candidate_fields": fields,
            "alias_families": families,
            "value_kinds": value_kinds,
            "caution": (
                "这些字段只因业务命名相近而被归为候选实体键；系统不得直接断言等价，"
                "必须通过唯一值交集、覆盖率、join coverage 或权威说明验证。"
            ),
            "recommended_qdi_checks": [
                "比较 code 类字段之间的 unique value 交集、left/right coverage 和缺失值比例。",
                "若存在成本/合同表与车辆/订单/资源表，验证合同实体键能否覆盖可用资源或订单侧实体键。",
                "若存在起点、终点、车型、线路、成本字段，进一步验证线路/车型合同覆盖率。",
                "覆盖率不足时记录 unresolved alias，不要把结算方代码和承运商代码强行等价。",
            ],
        }
    ]


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


def _match_carrier_settlement_alias(column: str) -> dict[str, str] | None:
    text = str(column or "").strip()
    if text in CARRIER_SETTLEMENT_ALIASES:
        return CARRIER_SETTLEMENT_ALIASES[text]
    # Keep this conservative: only catch columns that contain the exact business
    # alias phrase, and never infer equivalence from partial tokens alone.
    for alias, meta in CARRIER_SETTLEMENT_ALIASES.items():
        if alias in text:
            return meta
    return None
