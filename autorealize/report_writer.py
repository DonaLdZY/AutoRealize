from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import (
    AutoMLContextPack,
    DataAccessFileProtocol,
    DataAccessProtocol,
    DescriptionProtocolBundle,
    EvaluationContractReview,
    FileSummary,
    FileRole,
    PipelinePlan,
)
from .entity_alias import build_entity_alias_candidates
from .profiling.relations import RelationHint


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "任务概述": ("任务概述", "Overview"),
    "数据与文件说明": ("数据与文件说明", "Data Inventory"),
    "数据与读取方式": ("数据与读取方式", "数据读取", "Data Access"),
    "字段说明": ("字段说明", "数据字段说明", "Data Fields"),
    "任务定义": ("任务定义", "Task Definition"),
    "评估协议": ("评估协议", "Evaluation"),
    "输出或提交格式": ("输出或提交格式", "提交格式", "Submission Format", "Output Protocol"),
    "建模边界": ("建模边界", "Modeling Boundary"),
    "原始需求覆盖": ("原始需求覆盖", "Original Requirement Coverage"),
    "约束与风险": ("约束与风险", "Constraints & Risks"),
    "关键约束与注意事项": ("关键约束与注意事项", "约束与风险", "Constraints & Risks"),
}


def _section_aliases(header: str) -> tuple[str, ...]:
    h = str(header or "").strip()
    for canonical, aliases in SECTION_ALIASES.items():
        if h == canonical or h in aliases:
            return aliases
    return (h,)


def _has_h2(text: str, header: str) -> bool:
    aliases = "|".join(re.escape(x) for x in _section_aliases(header))
    return re.search(rf"^##\s+(\d+\.\s+)?(?:{aliases})\s*$", text, flags=re.M) is not None


def _has_h3(text: str, *headers: str) -> bool:
    aliases = "|".join(re.escape(x) for x in headers if str(x).strip())
    if not aliases:
        return False
    return re.search(rf"^###\s+(?:{aliases})\s*$", text, flags=re.M) is not None


def _section_text(text: str, header: str) -> str:
    aliases = set(_section_aliases(header))
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("## ") and line[3:].strip() in aliases:
            start = idx
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return "\n".join(lines[start:end]).strip()


def _text_before_section(text: str, header: str) -> str:
    aliases = set(_section_aliases(header))
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("## ") and line[3:].strip() in aliases:
            return "\n".join(lines[:idx])
    return text


def _task_type_label(task_type: str) -> str:
    tt = str(task_type or "").lower()
    if "time" in tt and "regression" in tt:
        return "时序回归"
    if "time" in tt:
        return "时序预测"
    if "class" in tt:
        return "分类"
    if "regression" in tt:
        return "回归"
    if "recommendation" in tt or "ranking" in tt:
        return "排序/推荐"
    if "reinforcement" in tt:
        return "强化学习/序贯决策"
    if "optimization" in tt:
        return "优化/调度"
    return str(task_type or "").strip() or "未明确"


def _role_label(role: FileRole | str) -> str:
    value = role.value if isinstance(role, FileRole) else str(role)
    labels = {
        "task_requirement": "任务需求文档",
        "data_description": "数据说明文档",
        "raw_data_table": "原始数据表",
        "code_or_config": "代码或配置",
        "image_or_media": "图片或媒体",
        "unknown": "未明确",
    }
    return labels.get(value, value)


def _metadata_label(key: str) -> str:
    labels = {
        "pages": "页数",
        "chars": "字符数",
        "archive_type": "压缩包类型",
        "entries": "条目数",
        "lines": "行数",
    }
    return labels.get(str(key), str(key))


def _direction_label_zh(direction: str) -> str:
    d = _direction_label(direction)
    if d == "minimize":
        return "越小越好"
    if d == "maximize":
        return "越大越好"
    return d or "未明确"


def _is_document_like(fs: FileSummary) -> bool:
    parsed_kind = str((fs.source_metadata or {}).get("kind", (fs.source_metadata or {}).get("parsed_kind", ""))).strip().lower()
    return fs.role in {FileRole.task_requirement, FileRole.data_description} or parsed_kind in {"document", "structured_document"}


def _profile_map(fs: FileSummary) -> dict[str, dict]:
    return {str(p.get("name", "")): p for p in (fs.column_profiles or []) if str(p.get("name", "")).strip()}


def _has_data_field_profiles(fs: FileSummary) -> bool:
    return bool(_profile_map(fs)) and not _is_document_like(fs)


def _llm_field_description(fs: FileSummary, col: str) -> str:
    meta = fs.column_semantic_meta.get(col, {}) if hasattr(fs, "column_semantic_meta") else {}
    if meta.get("source") not in {"llm_field_description", "heuristic", "deterministic"}:
        return ""
    return str(fs.column_semantics.get(col, "")).strip()


def _looks_like_sample_constant(value: str) -> bool:
    token = str(value or "").strip().strip("`'\"()（）[]【】")
    if not token:
        return False
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-./]{1,40}", token):
        return True
    return False


def _generic_field_description(col: str) -> str:
    name = str(col or "").strip()
    compact = name.lower()
    if not name:
        return ""
    if any(x in name for x in ["代码", "编码", "编号"]) or compact.endswith("id") or "_id" in compact:
        subject = re.sub(r"(代码|编码|编号|ID|Id|id)$", "", name).strip() or "业务实体"
        return f"用于标识{subject}的编码字段。"
    if "名称" in name or compact.endswith("name"):
        subject = name.replace("名称", "").strip() or "业务实体"
        return f"记录{subject}的名称。"
    if any(x in name for x in ["成本", "费用", "价格", "金额", "运费"]):
        return f"记录与{name}相关的费用或计价数值。"
    if any(x in name for x in ["重量", "体积", "数量", "件数", "车辆数"]):
        return f"记录{name}对应的数量型属性。"
    if any(x in name for x in ["日期", "时间", "时刻"]):
        return f"记录{name}对应的时间信息。"
    if any(x in name for x in ["类型", "类别", "车型"]):
        return f"记录{name}对应的分类或枚举属性。"
    return ""


def _sanitize_group_field_description(text: str, col: str) -> str:
    """Remove single-file sample constants before rendering grouped schemas."""
    value = str(text or "").strip()
    if not value:
        return ""

    def _replace_contextual_constant(match: re.Match) -> str:
        token = str(match.group("value") or "")
        return "" if _looks_like_sample_constant(token) else match.group(0)

    value = re.sub(
        r"(?:[，,；;。]\s*)?(?:此处|当前文件|该文件|本文件|代表文件|样例中|示例中|当前样例|该样例)"
        r"(?:的)?(?:取值)?(?:为|是|=|：|:)\s*(?P<value>[^，。；;、\n]+)",
        _replace_contextual_constant,
        value,
    )
    value = re.sub(
        r"(?:[，,；;]\s*)?(?:固定为|取值为|值为)\s*(?P<value>[A-Za-z0-9][A-Za-z0-9_\-./]{1,40})(?=$|[，。；;、\s])",
        _replace_contextual_constant,
        value,
    )
    value = re.sub(r"\s+", " ", value).strip(" ，,；;。")
    if not value or value == col or value == f"{col}。":
        return _generic_field_description(col)
    return value


def _parse_top_value(raw: str) -> tuple[str, int | None]:
    text = str(raw or "").strip()
    match = re.match(r"^(?P<value>.*?)\((?P<count>\d+)\)$", text)
    if not match:
        return text, None
    return match.group("value").strip(), int(match.group("count"))


def _profile_primary_value(profile: dict) -> tuple[str, float]:
    top_values = profile.get("top_values") or []
    if top_values:
        value, count = _parse_top_value(str(top_values[0]))
        try:
            denom = int(float(profile.get("non_null_count") or profile.get("row_count") or 0))
        except Exception:
            denom = 0
        ratio = float(count / denom) if count is not None and denom > 0 else 0.0
        return value, ratio
    sample_values = [str(x).strip() for x in (profile.get("sample_values") or []) if str(x).strip()]
    if sample_values and len(set(sample_values)) == 1:
        return sample_values[0], 1.0
    return "", 0.0


def _norm_identifier(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", str(value or "")).lower()


def _filename_variable_segments(paths: list[str]) -> list[str]:
    names = [Path(str(p).replace("\\", "/")).name for p in paths if str(p).strip()]
    if not names:
        return []
    suffixes = [Path(name).suffix for name in names]
    suffix = suffixes[0] if len(set(suffixes)) == 1 else ""
    stems = [name[: -len(suffix)] if suffix and name.endswith(suffix) else Path(name).stem for name in names]
    if len(stems) <= 1:
        return stems
    prefix = _common_prefix(stems)
    common_suffix = _common_suffix(stems)
    while prefix and prefix[-1].isdigit():
        prefix = prefix[:-1]
    segments: list[str] = []
    for stem in stems:
        end = len(stem) - len(common_suffix) if common_suffix else len(stem)
        segment = stem[len(prefix):end].strip()
        segments.append(segment or stem)
    return segments


def _identifier_matches_filename(value: str, segment: str) -> bool:
    nv = _norm_identifier(value)
    ns = _norm_identifier(segment)
    if not nv or not ns:
        return False
    ns_without_sequence = re.sub(r"^\d+", "", ns)
    return nv == ns or nv in ns or ns_without_sequence == nv or nv in ns_without_sequence


def _subject_from_col(col: str) -> str:
    name = str(col or "").strip()
    subject = re.sub(r"(代码|编码|编号|ID|Id|id)$", "", name).strip()
    return subject or "业务实体"


def _entity_hint_from_pattern(pattern: str) -> str:
    text = str(pattern or "")
    for token in ["承运商", "客户", "订单", "车辆", "车型", "商品", "货品", "用户", "仓库", "门店", "站点", "样本", "井"]:
        if token in text:
            return token
    return ""


def _group_column_role_insight(group: list[FileSummary], col: str) -> dict[str, Any]:
    profiles = [_profile_map(fs).get(col) for fs in group]
    profiles = [p for p in profiles if isinstance(p, dict)]
    if len(profiles) < 2:
        return {}
    primary_values: list[str] = []
    stable_count = 0
    for profile in profiles:
        value, ratio = _profile_primary_value(profile)
        if value:
            primary_values.append(value)
        try:
            unique_count = int(float(profile.get("unique_count", 0) or 0))
        except Exception:
            unique_count = 0
        if value and (ratio >= 0.85 or unique_count <= 1):
            stable_count += 1
    if len(primary_values) < 2:
        return {}
    distinct_values = {str(v).strip() for v in primary_values if str(v).strip()}
    stable_by_file = stable_count >= max(2, int(len(profiles) * 0.7))
    varies_across_files = len(distinct_values) >= 2
    paths = [str(fs.path) for fs in group]
    segments = _filename_variable_segments(paths)
    match_count = 0
    for value, segment in zip(primary_values, segments):
        if _identifier_matches_filename(value, segment):
            match_count += 1
    filename_linked = match_count >= max(2, int(min(len(primary_values), len(segments)) * 0.6))
    if not stable_by_file:
        return {}
    pattern = _common_path_pattern(paths)
    subject = _subject_from_col(col)
    entity = _entity_hint_from_pattern(pattern)
    if varies_across_files and filename_linked:
        if entity and entity not in subject:
            meaning = f"{subject}/{entity}标识字段；在单个文件内通常保持稳定，跨文件随文件名中的 `{{id}}` 部分变化，用于区分该数据表所属的{entity}或结算主体。"
        else:
            meaning = f"{subject}标识字段；在单个文件内通常保持稳定，跨文件随文件名中的 `{{id}}` 部分变化，用于区分该文件组中的不同{subject}。"
        return {
            "meaning": meaning,
            "profile_note": "文件内主值稳定，跨文件取值不同，并与文件名 `{id}` 部分对应",
        }
    if varies_across_files:
        return {
            "meaning": f"{subject}或文件来源标识字段；在单个文件内通常保持稳定，但不同文件之间取值不同，建模时应结合 `source_file` 保留来源差异。",
            "profile_note": "文件内主值稳定，跨文件取值不同",
        }
    return {
        "meaning": f"文件组内基本固定的{subject}字段；可作为公共属性或数据来源一致性校验字段。",
        "profile_note": "文件内和跨文件主值均较稳定",
    }


def _fmt_profile_value(value: Any) -> str:
    if value is None:
        return "无"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_profile_ratio(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return _fmt_profile_value(value)


def _logical_type_label(value: Any) -> str:
    labels = {
        "integer": "整数",
        "float": "浮点数",
        "numeric": "数值",
        "datetime": "日期时间",
        "date": "日期",
        "categorical": "类别",
        "boolean": "布尔/开关",
        "text": "文本",
        "string": "字符串",
        "unknown": "未明确",
    }
    raw = str(value or "unknown").strip()
    return labels.get(raw.lower(), raw)


def format_column_profile_inline(profile: dict) -> str:
    """Render a field profile as reader-facing Chinese prose for data files."""
    null_count = profile.get("null_count", "NA")
    row_count = profile.get("row_count", "NA")
    unique_count = profile.get("unique_count", "NA")
    parts = [
        f"存储类型：{_fmt_profile_value(profile.get('dtype'))}",
        f"逻辑类型：{_logical_type_label(profile.get('logical_type', 'unknown'))}",
        f"缺失：{null_count}/{row_count}（{_fmt_profile_ratio(profile.get('null_ratio'))}）",
        f"唯一值数：{unique_count}",
    ]
    if profile.get("numeric_parse_ratio") is not None:
        parts.append(f"数值可解析比例：{_fmt_profile_ratio(profile.get('numeric_parse_ratio'))}")
    if profile.get("datetime_parse_ratio") is not None:
        parts.append(f"日期时间可解析比例：{_fmt_profile_ratio(profile.get('datetime_parse_ratio'))}")
    hints = profile.get("format_hints") or []
    if hints:
        parts.append("格式线索：" + "、".join([str(x) for x in hints[:8]]))
    ns = profile.get("numeric_stats") or {}
    if ns:
        parts.append(
            "数值统计："
            f"范围 {_fmt_profile_value(ns.get('min'))} 至 {_fmt_profile_value(ns.get('max'))}，"
            f"均值 {_fmt_profile_value(ns.get('mean'))}，"
            f"标准差 {_fmt_profile_value(ns.get('std'))}，"
            f"方差 {_fmt_profile_value(ns.get('var'))}"
        )
    qs = profile.get("quantiles") or {}
    if qs:
        parts.append(
            "分位数："
            f"5%={_fmt_profile_value(qs.get('p05'))}，"
            f"25%={_fmt_profile_value(qs.get('q1'))}，"
            f"75%={_fmt_profile_value(qs.get('q3'))}，"
            f"95%={_fmt_profile_value(qs.get('p95'))}"
        )
    ds = profile.get("datetime_stats") or {}
    if ds:
        parts.append(
            "时间范围："
            f"{_fmt_profile_value(ds.get('min'))} 至 {_fmt_profile_value(ds.get('max'))}"
            f"（跨度 {_fmt_profile_value(ds.get('range_days'))} 天）"
        )
    bad = profile.get("abnormal_tokens") or []
    if bad:
        parts.append("异常取值样例：" + "、".join([str(x) for x in bad[:8]]))
    top_values = profile.get("top_values") or []
    if top_values and not ns:
        parts.append("高频值：" + "、".join([str(x) for x in top_values[:8]]))
    return "；".join(parts)


def write_data_description(
    path: Path,
    file_summaries: list[FileSummary],
    dir_summaries: list[str],
    relations: list[RelationHint],
) -> None:
    lines: list[str] = ["# 数据认知文档", ""]
    lines.append("## 文件夹概览")
    lines.extend([f"- {x}" for x in dir_summaries] or ["- 暂无"])
    lines.append("")
    lines.append("## 文件级认知")
    for fs in file_summaries:
        lines.append(f"### {fs.path}")
        lines.append(f"- 角色: `{fs.role.value}`")
        lines.append(f"- 摘要: {fs.summary}")
        if str(fs.detailed_report or "").strip():
            lines.append("")
            lines.append("#### 详细认知报告")
            lines.append(str(fs.detailed_report).strip())
        data_profile_map = _profile_map(fs) if _has_data_field_profiles(fs) else {}
        if fs.columns and data_profile_map:
            lines.append(f"- 字段: {', '.join(fs.columns[:30])}")
        llm_descriptions = {col: _llm_field_description(fs, col) for col in fs.columns}
        llm_descriptions = {k: v for k, v in llm_descriptions.items() if v}
        if llm_descriptions and data_profile_map:
            lines.append("- 字段说明（LLM自然语言）:")
            for col, meaning in llm_descriptions.items():
                lines.append(f"  - `{col}`: {meaning}")
        if fs.extracted_knowledge:
            lines.append("- 文档/字段关键知识明细:")
            for item in fs.extracted_knowledge[:30]:
                lines.append(f"  - {item}")
        if data_profile_map:
            lines.append("- 字段结构与质量（全部数据字段）:")
            for p in fs.column_profiles:
                name = str(p.get("name", "")).strip()
                if not name:
                    continue
                lines.append(f"  - `{name}` | {format_column_profile_inline(p)}")
        if fs.related_files:
            lines.append(f"- 可能关联: {', '.join(fs.related_files[:12])}")
        if fs.warnings:
            lines.append(f"- 风险: {'; '.join(fs.warnings[:8])}")
        lines.append("")
    lines.append("## 跨文件关系")
    if relations:
        for r in relations:
            left_field = str(getattr(r, "left_field", "") or "").strip()
            right_field = str(getattr(r, "right_field", "") or "").strip()
            relation_type = str(getattr(r, "relation_type", "") or "").strip()
            confidence = getattr(r, "confidence", "")
            evidence = str(getattr(r, "short_evidence", "") or getattr(r, "reason", "") or "").strip()
            if left_field or right_field:
                lines.append(
                    f"- {r.left_file}.{left_field or '?'} <-> {r.right_file}.{right_field or '?'}"
                    f": {relation_type or 'shared_attribute'} confidence={confidence} ({evidence})"
                )
            else:
                lines.append(f"- {r.left_file} <-> {r.right_file}: {', '.join(r.shared_columns)} ({r.reason})")
    else:
        lines.append("- 暂未发现明显同名字段关系。")
    path.write_text("\n".join(lines), encoding="utf-8")


def _quote_input_path(path: str) -> str:
    rel_path = str(path or "").replace("\\", "/").lstrip("/")
    return f"./input/{rel_path}"


def _excel_sheet_names_from_metadata(fs: FileSummary) -> list[str]:
    meta = fs.source_metadata or {}
    raw = (
        meta.get("excel_sheet_names")
        or meta.get("sheet_names")
        or meta.get("sheets")
        or []
    )
    if isinstance(raw, str):
        raw = [raw]
    return _dedupe_any(raw, limit=20)


def _read_example_for_file(fs: FileSummary) -> tuple[str, str, list[str]]:
    """Return read method, executable example, and parsing notes for a file."""
    path = str(fs.path or "")
    suffix = Path(path).suffix.lower()
    notes: list[str] = []
    input_path = _quote_input_path(path)
    if suffix == ".csv":
        dialect = (fs.source_metadata or {}).get("csv_dialect") or {}
        sep = str(dialect.get("sep", ",") or ",")
        engine = dialect.get("engine")
        inferred = bool(dialect.get("inferred"))
        reason = str(dialect.get("reason", "") or "")
        encoding = str((fs.source_metadata or {}).get("csv_encoding", "") or "utf-8-sig")
        kwargs = [f"{input_path!r}"]
        if sep == r"\s+":
            kwargs.append(r"sep=r'\s+'")
        elif sep != ",":
            kwargs.append(f"sep={sep!r}")
        if engine:
            kwargs.append(f"engine={engine!r}")
        kwargs.append(f"encoding={encoding!r}")
        example = f"pd.read_csv({', '.join(kwargs)})"
        if inferred:
            notes.append(f"检测到 CSV 需要显式读取参数：sep={sep!r}；原因：{reason or '自动探测'}。")
        if encoding != "utf-8-sig":
            notes.append(f"检测到编码候选为 {encoding!r}。")
        notes.append("如果出现编码错误，可使用 gb18030 作为备选编码重试。")
        return "pandas.read_csv", example, notes
    if suffix in {".xlsx", ".xls"}:
        sheet_names = _excel_sheet_names_from_metadata(fs)
        sheet_profiles = _excel_sheet_profiles(fs)
        abnormal = [
            sheet
            for sheet in sheet_profiles
            if isinstance(sheet, dict) and str(sheet.get("layout_kind", "") or "") not in {"", "standard_table"}
        ]
        if len(sheet_names) > 1:
            notes.append("检测到多工作表 Excel，必须显式读取需要的 sheet；不要依赖 `pd.read_excel(path)` 默认读取第一个工作表。")
            shown = sheet_names[:8]
            notes.append("已识别工作表：" + "、".join(f"`{x}`" for x in shown) + (" 等。" if len(sheet_names) > len(shown) else "。"))
            notes.append("建议先用 `sheet_name=None` 读取为 dict，再按 automl_context 中的 sheet_groups 判断是否逐 sheet 使用或合并同结构 sheet。")
            if abnormal:
                notes.append(
                    "存在非标准布局 sheet，需按 sheet 级 read_example/header 策略读取："
                    + "；".join(
                        f"`{sheet.get('sheet_name')}`={sheet.get('layout_kind')} read={sheet.get('recommended_read') or 'inspect header=None'}"
                        for sheet in abnormal[:6]
                    )
                    + "。"
                )
            return "pandas.read_excel", f"pd.read_excel({input_path!r}, sheet_name=None)", notes
        if abnormal:
            sheet = abnormal[0]
            example = str(sheet.get("recommended_read") or f"pd.read_excel({input_path!r}, header=None)")
            notes.append(
                f"检测到 sheet `{sheet.get('sheet_name')}` 为 {sheet.get('layout_kind')}，不要盲信默认表头；"
                f"建议读取方式：{example}。"
            )
            return "pandas.read_excel", example, notes
        return "pandas.read_excel", f"pd.read_excel({input_path!r})", notes
    if suffix == ".json":
        json_strategy = str((fs.source_metadata or {}).get("json_strategy", "") or "")
        if json_strategy:
            notes.append(f"JSON 已被识别为可表格化结构：{json_strategy}。")
        else:
            notes.append("若 JSON 为嵌套结构，需先按键路径展开或抽取记录数组后再建模。")
        return "pandas.read_json/json.load", f"pd.read_json({input_path!r})", notes
    if suffix in {".parquet", ".pq"}:
        return "pandas.read_parquet", f"pd.read_parquet({input_path!r})", notes
    if suffix in {".txt", ".md"}:
        return "Path.read_text", f"Path({input_path!r}).read_text(encoding='utf-8', errors='ignore')", notes
    return "按文件类型读取", f"# inspect {input_path!r} according to its file type", notes

def build_data_access_protocol(file_summaries: list[FileSummary]) -> DataAccessProtocol:
    """Build deterministic file reading hints from parser metadata."""
    files: list[DataAccessFileProtocol] = []
    global_notes: list[str] = [
        "下游代码应从 `./input` 目录读取数据；不要使用裸文件名读取，避免工作目录变化导致找不到文件。",
        "读取后必须核对行数、列名和关键字段是否与本说明一致。",
    ]
    for fs in file_summaries:
        if _is_document_like(fs):
            continue
        suffix = Path(fs.path).suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq", ".txt", ".md"} and not fs.columns:
            continue
        read_method, read_example, notes = _read_example_for_file(fs)
        profiles = _profile_map(fs)
        key_fields: list[str] = []
        target_fields: list[str] = []
        important_fields: list[str] = []
        for col in fs.columns:
            c = str(col)
            lower = c.lower()
            meaning = _llm_field_description(fs, c).lower()
            if any(k in lower for k in ["id", "编号", "订单号", "用户", "user", "key"]):
                key_fields.append(c)
            if any(k in lower for k in ["target", "label", "标签", "结果", "y", "class", "category"]) or "目标" in meaning or "标签" in meaning:
                target_fields.append(c)
            if c in profiles or c in key_fields or c in target_fields:
                important_fields.append(c)
        # Keep a bounded physical-column list so repeated-file groups can show
        # both shared and variant columns instead of only task-scored fields.
        for col in fs.columns[:30]:
            text = str(col)
            if text and text not in important_fields:
                important_fields.append(text)
        files.append(
            DataAccessFileProtocol(
                path=str(fs.path),
                file_role=_role_label(fs.role),
                read_method=read_method,
                read_example=read_example,
                row_grain=str(fs.summary or "").strip()[:240],
                key_fields=key_fields[:12],
                target_fields=target_fields[:8],
                relation_keys=[str(x) for x in fs.key_entities[:12]],
                important_fields=important_fields[:30],
                parsing_notes=notes + [str(x) for x in fs.warnings[:4]],
            )
        )
    return DataAccessProtocol(files=files, global_notes=global_notes)


def _common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        while prefix and not value.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _common_suffix(values: list[str]) -> str:
    if not values:
        return ""
    reversed_suffix = _common_prefix([value[::-1] for value in values])
    return reversed_suffix[::-1]


def _common_path_pattern(paths: list[str]) -> str:
    cleaned = [str(p).replace("\\", "/") for p in paths if str(p).strip()]
    if not cleaned:
        return "同结构文件组"
    parents = [str(Path(p).parent).replace("\\", "/") for p in cleaned]
    same_parent = len(set(parents)) == 1
    parent = "" if parents[0] in {"", "."} else parents[0]
    names = [Path(p).name for p in cleaned]
    suffixes = [Path(name).suffix for name in names]
    suffix = suffixes[0] if len(set(suffixes)) == 1 else ""
    stems = [name[: -len(suffix)] if suffix and name.endswith(suffix) else Path(name).stem for name in names]
    if len(stems) == 1:
        pattern_name = names[0]
    else:
        prefix = _common_prefix(stems)
        common_suffix = _common_suffix(stems)
        while prefix and prefix[-1].isdigit():
            prefix = prefix[:-1]
        min_len = min(len(x) for x in stems)
        if len(prefix) + len(common_suffix) >= min_len:
            common_suffix = common_suffix[: max(0, min_len - len(prefix) - 1)]
        if len(prefix) >= 2 or len(common_suffix) >= 4:
            pattern_name = f"{prefix}{{id}}{common_suffix}{suffix}"
        else:
            normalized = re.sub(r"[A-Za-z0-9]+", "{id}", names[0])
            pattern_name = normalized if "{id}" in normalized else f"同结构*{suffix or Path(names[0]).suffix}"
    return f"{parent}/{pattern_name}" if same_parent and parent else pattern_name


def _filename_pattern_signature(path_value: str) -> tuple[str, str, str] | None:
    """Return a conservative repeated-file signature based on filename literals.

    This catches domains such as `成本/承运商01BZWL01 承运商成本.xlsx` even when
    sampled files have slightly different columns. Pure-English names like
    `train1.csv` are intentionally left to the stricter schema signature because
    replacing every English token would over-group unrelated Kaggle files.
    """
    normalized = str(path_value or "").replace("\\", "/").strip()
    if not normalized:
        return None
    path = Path(normalized)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}:
        return None
    stem = path.stem
    pattern = re.sub(r"[A-Za-z0-9]+", "{id}", stem)
    pattern = re.sub(r"(?:\{id\})+", "{id}", pattern)
    literal = pattern.replace("{id}", "")
    literal = re.sub(r"[\s_\-()（）\[\]【】.]+", "", literal)
    if "{id}" not in pattern or len(literal) < 2:
        return None
    parent = str(path.parent).replace("\\", "/")
    return (parent, suffix, pattern)


def _data_access_group_signature(item: DataAccessFileProtocol) -> tuple | None:
    path = Path(str(item.path or ""))
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}:
        return None
    fields = tuple(str(x) for x in (item.important_fields or item.key_fields or item.target_fields or []) if str(x).strip())
    if not fields:
        return None
    parent = str(path.parent).replace("\\", "/")
    return (parent, suffix, str(item.file_role), str(item.read_method), fields)


def _data_access_pattern_signature(item: DataAccessFileProtocol) -> tuple | None:
    sig = _filename_pattern_signature(str(item.path or ""))
    if sig is None:
        return None
    parent, suffix, pattern = sig
    return (parent, suffix, pattern, str(item.file_role), str(item.read_method))


def _group_data_access_items(files: list[DataAccessFileProtocol]) -> list[tuple[bool, list[DataAccessFileProtocol]]]:
    pattern_grouped: dict[tuple, list[DataAccessFileProtocol]] = {}
    for item in files:
        sig = _data_access_pattern_signature(item)
        if sig is not None:
            pattern_grouped.setdefault(sig, []).append(item)

    grouped: dict[tuple, list[DataAccessFileProtocol]] = {}
    for item in files:
        sig = _data_access_group_signature(item)
        if sig is not None:
            grouped.setdefault(sig, []).append(item)
    emitted: set[int] = set()
    emitted_pattern_sigs: set[tuple] = set()
    blocks: list[tuple[bool, list[DataAccessFileProtocol]]] = []
    for item in files:
        item_id = id(item)
        if item_id in emitted:
            continue
        pattern_sig = _data_access_pattern_signature(item)
        if pattern_sig is not None and pattern_sig in emitted_pattern_sigs:
            emitted.add(item_id)
            continue
        pattern_group = pattern_grouped.get(pattern_sig, []) if pattern_sig is not None else []
        if len(pattern_group) >= 3:
            blocks.append((True, pattern_group))
            emitted.update(id(x) for x in pattern_group)
            if pattern_sig is not None:
                emitted_pattern_sigs.add(pattern_sig)
            continue
        sig = _data_access_group_signature(item)
        group = grouped.get(sig, []) if sig is not None else []
        if len(group) >= 3:
            filtered = [
                x
                for x in group
                if (ps := _data_access_pattern_signature(x)) is None or ps not in emitted_pattern_sigs
            ]
            if len(filtered) >= 3:
                blocks.append((True, filtered))
                emitted.update(id(x) for x in filtered)
            else:
                blocks.append((False, [item]))
                emitted.add(item_id)
        else:
            blocks.append((False, [item]))
            emitted.add(item_id)
    return blocks


def _data_access_group_field_profile(items: list[DataAccessFileProtocol]) -> dict[str, Any]:
    observed: list[tuple[str, list[str]]] = []
    for item in items:
        fields: list[str] = []
        for value in list(item.important_fields or []) + list(item.key_fields or []) + list(item.target_fields or []):
            text = str(value).strip()
            if text and text not in fields:
                fields.append(text)
        if item.path and fields:
            observed.append((str(item.path), fields))
    if not observed:
        return {}

    sets = [set(fields) for _, fields in observed]
    common_set = set.intersection(*sets) if sets else set()
    union: list[str] = []
    for _, fields in observed:
        for field in fields:
            if field not in union:
                union.append(field)

    shared_fields = [field for field in union if field in common_set]
    variants: list[dict[str, Any]] = []
    for path, fields in observed[:12]:
        only_fields = [field for field in fields if field not in common_set]
        if only_fields:
            variants.append(
                {
                    "file": path,
                    "fields": only_fields[:18],
                    "omitted": max(0, len(only_fields) - 18),
                }
            )

    return {
        "shared_fields": shared_fields,
        "variant_fields_by_file": variants,
    }


def _dedupe_nonempty(values: list[str], limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = str(value or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _dedupe_any(values, limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        s = str(value or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _reader_expr_with_path_var(read_example: str) -> str:
    expr = str(read_example or "").strip()
    if not expr:
        return "pd.read_csv(path)"
    return re.sub(r"(['\"])\./input/[^'\"]+\1", "path", expr, count=1)


def _render_group_read_code(pattern: str, representative: DataAccessFileProtocol) -> list[str]:
    glob_pattern = pattern.replace("{id}", "*").replace("\\", "/")
    reader_expr = _reader_expr_with_path_var(representative.read_example)
    if "sheet_name=None" in reader_expr and _path_suffix(representative.path) in {".xlsx", ".xls"}:
        return [
            "```python",
            "from pathlib import Path",
            "import pandas as pd",
            "",
            "input_dir = Path('./input')",
            "workbooks = {}",
            "sheets = {}",
            f"for path in sorted(input_dir.glob({glob_pattern!r})):",
            f"    book = {reader_expr}",
            "    workbooks[path.name] = book",
            "    for sheet_name, df_one in book.items():",
            "        df_one = df_one.copy()",
            "        df_one['source_file'] = path.name",
            "        df_one['source_sheet'] = sheet_name",
            "        sheets.setdefault(sheet_name, []).append(df_one)",
            "sheet_frames = {",
            "    sheet_name: pd.concat(frames, ignore_index=True)",
            "    for sheet_name, frames in sheets.items()",
            "}",
            "# Example: contract rate rows are often in sheet_frames['导出信息'];",
            "# rule/algorithm notes are often in sheet_frames['Sheet1'] or another explanatory sheet.",
            "```",
        ]
    return [
        "```python",
        "from pathlib import Path",
        "import pandas as pd",
        "",
        "input_dir = Path('./input')",
        "frames = []",
        f"for path in sorted(input_dir.glob({glob_pattern!r})):",
        f"    df_one = {reader_expr}",
        "    df_one['source_file'] = path.name",
        "    frames.append(df_one)",
        "df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()",
        "```",
    ]


def _path_suffix(path_value: str) -> str:
    return Path(str(path_value or "")).suffix.lower()


def _has_non_default_csv_read(item: DataAccessFileProtocol) -> bool:
    if _path_suffix(item.path) != ".csv":
        return False
    example = str(item.read_example or "")
    notes = "\n".join(str(x) for x in item.parsing_notes)
    return "sep=" in example or "engine=" in example or "分隔符" in notes


def _has_multi_sheet_excel_read(item: DataAccessFileProtocol) -> bool:
    if _path_suffix(item.path) not in {".xlsx", ".xls"}:
        return False
    example = str(item.read_example or "")
    notes = "\n".join(str(x) for x in item.parsing_notes)
    return "sheet_name" in example or "多工作表" in notes or "工作表" in notes


def _needs_explicit_read_example(item: DataAccessFileProtocol, *, is_group: bool = False) -> bool:
    if is_group:
        return True
    suffix = _path_suffix(item.path)
    if suffix == ".json":
        return True
    if suffix == ".csv" and _has_non_default_csv_read(item):
        return True
    if suffix in {".xlsx", ".xls"} and _has_multi_sheet_excel_read(item):
        return True
    return False


def _compact_profile_for_automl(profile: dict[str, Any]) -> dict[str, Any]:
    name = str(profile.get("name", "") or "").strip()
    out: dict[str, Any] = {
        "name": name,
        "logical_type": profile.get("logical_type") or profile.get("dtype"),
        "row_count": profile.get("row_count"),
        "null_ratio": profile.get("null_ratio"),
        "unique_count": profile.get("unique_count"),
    }
    numeric = profile.get("numeric_stats") or {}
    if numeric:
        out["numeric_range"] = {
            "min": numeric.get("min"),
            "max": numeric.get("max"),
            "mean": numeric.get("mean"),
            "std": numeric.get("std"),
        }
    datetime = profile.get("datetime_stats") or {}
    if datetime:
        out["datetime_range"] = {
            "min": datetime.get("min"),
            "max": datetime.get("max"),
            "granularity": datetime.get("granularity"),
        }
    top_values = profile.get("top_values") if isinstance(profile.get("top_values"), list) else []
    if top_values:
        out["top_values"] = [str(x) for x in top_values[:8]]
    hints = profile.get("format_hints") if isinstance(profile.get("format_hints"), list) else []
    if hints:
        out["format_hints"] = [str(x) for x in hints[:6]]
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def _compact_file_profiles_for_automl(fs: FileSummary, *, limit: int = 40) -> list[dict[str, Any]]:
    profiles = []
    for profile in (fs.column_profiles or [])[:limit]:
        if isinstance(profile, dict) and str(profile.get("name", "")).strip():
            item = _compact_profile_for_automl(profile)
            meaning = _llm_field_description(fs, str(profile.get("name", "")))
            if meaning:
                item["meaning"] = meaning[:300]
            profiles.append(item)
    return profiles


def _sheet_field_descriptions(fs: FileSummary, sheet_name: str) -> dict[str, str]:
    meta = fs.source_metadata or {}
    sheet_fields = meta.get("sheet_field_descriptions") if isinstance(meta.get("sheet_field_descriptions"), dict) else {}
    fields = sheet_fields.get(sheet_name) if isinstance(sheet_fields.get(sheet_name), dict) else {}
    return {str(k): str(v) for k, v in fields.items() if str(k).strip() and str(v).strip()}


def _sheet_profile_items(sheet: dict[str, Any], fs: FileSummary, *, max_cols: int = 24) -> list[dict[str, Any]]:
    descriptions = _sheet_field_descriptions(fs, str(sheet.get("sheet_name", "")))
    out: list[dict[str, Any]] = []
    for profile in (sheet.get("column_profiles") or [])[:max_cols]:
        if not isinstance(profile, dict):
            continue
        item = _compact_profile_for_automl(profile)
        name = str(profile.get("name", "") or "")
        meaning = descriptions.get(name) or _llm_field_description(fs, name)
        if meaning:
            item["meaning"] = meaning[:300]
        out.append(item)
    if not out and descriptions:
        for name, meaning in list(descriptions.items())[:max_cols]:
            out.append({"name": name, "meaning": meaning[:300]})
    return out


def _compact_excel_sheet_profiles_for_automl(fs: FileSummary, *, max_sheets: int = 80, max_cols: int = 24) -> list[dict[str, Any]]:
    meta = fs.source_metadata or {}
    sheets = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
    out: list[dict[str, Any]] = []
    for sheet in sheets[:max_sheets]:
        if not isinstance(sheet, dict):
            continue
        sheet_name = str(sheet.get("sheet_name", "") or "")
        descriptions = _sheet_field_descriptions(fs, sheet_name)
        item = {
            "sheet_name": sheet_name,
            "shape": sheet.get("shape", []),
            "shape_profiled": sheet.get("shape_profiled", sheet.get("shape_sampled", [])),
            "shape_estimated": sheet.get("shape_estimated", False),
            "preview_rows_used": sheet.get("preview_rows_used", 0),
            "columns": [str(x) for x in (sheet.get("columns") or [])[:max_cols]],
            "layout_kind": sheet.get("layout_kind", ""),
            "header_confidence": sheet.get("header_confidence", None),
            "detected_header_row": sheet.get("detected_header_row", None),
            "read_strategy_kind": sheet.get("read_strategy_kind", ""),
            "reading_risks": [str(x) for x in (sheet.get("reading_risks") or [])[:4]],
            "is_deep_profiled": sheet.get("is_deep_profiled", False),
            "profile_policy": sheet.get("profile_policy", ""),
            "profile_rows_limit": sheet.get("profile_rows_limit", None),
            "sheet_group_id": sheet.get("sheet_group_id", ""),
            "sheet_group_size": sheet.get("sheet_group_size", 1),
            "sheet_group_representative": sheet.get("sheet_group_representative", ""),
            "profiled_column_count": sheet.get("profiled_column_count", 0),
            "read_example": sheet.get("recommended_read") or f"pd.read_excel(path, sheet_name={str(sheet.get('sheet_name', ''))!r})",
            "raw_preview_note": "If opening notes or non-default headers matter, inspect this sheet directly with header=None before modeling.",
            "field_descriptions": descriptions,
            "column_profiles": _sheet_profile_items(sheet, fs, max_cols=max_cols),
        }
        out.append({k: v for k, v in item.items() if v not in (None, "", [], {})})
    return out


def _compact_group_excel_sheet_profiles_for_automl(group: list[FileSummary], *, max_sheets: int = 20, max_cols: int = 32) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sheet_title, sheet, members in _merged_group_sheet_profiles(group, max_sheets=max_sheets):
        field_profiles = []
        descriptions_by_name: dict[str, list[str]] = {}
        for fs in members:
            for name, meaning in _sheet_field_descriptions(fs, sheet_title).items():
                descriptions_by_name.setdefault(name, []).append(meaning)
        for profile in (sheet.get("column_profiles") or [])[:max_cols]:
            if not isinstance(profile, dict):
                continue
            item = _compact_profile_for_automl(profile)
            name = str(profile.get("name", "") or "")
            desc = _dedupe_nonempty(descriptions_by_name.get(name, []), limit=2)
            meaning = "；".join(desc) if desc else _merge_group_field_description(group, name)
            if meaning:
                item["meaning"] = meaning[:300]
            field_profiles.append(item)
        for name, values in descriptions_by_name.items():
            if any(str(p.get("name", "")) == name for p in field_profiles if isinstance(p, dict)):
                continue
            desc = _dedupe_nonempty(values, limit=2)
            if desc:
                field_profiles.append({"name": name, "meaning": "；".join(desc)[:300]})
        item = {
            "sheet_name": sheet_title,
            "source_file_count": len(members),
            "shape": sheet.get("shape", []),
            "shape_profiled": sheet.get("shape_profiled", sheet.get("shape_sampled", [])),
            "columns": [str(x) for x in (sheet.get("columns") or [])[:max_cols]],
            "layout_kind": sheet.get("layout_kind", ""),
            "detected_header_row": sheet.get("detected_header_row", None),
            "read_strategy_kind": sheet.get("read_strategy_kind", ""),
            "reading_risks": [str(x) for x in (sheet.get("reading_risks") or [])[:4]],
            "read_example": sheet.get("recommended_read") or f"pd.read_excel(path, sheet_name={sheet_title!r})",
            "raw_preview_note": "If opening notes or non-default headers matter, inspect representative workbooks directly with header=None before modeling.",
            "column_profiles": field_profiles[:max_cols],
        }
        out.append({k: v for k, v in item.items() if v not in (None, "", [], {})})
    return out


def _compact_preview_records(rows: Any, *, max_rows: int = 2, max_cols: int = 8) -> list[dict[str, str]]:
    if not isinstance(rows, list):
        return []
    compact: list[dict[str, str]] = []
    for row in rows[:max_rows]:
        if not isinstance(row, dict):
            continue
        item: dict[str, str] = {}
        for idx, (key, value) in enumerate(row.items()):
            if idx >= max_cols:
                break
            text = str(value)
            item[str(key)] = text[:120] + ("..." if len(text) > 120 else "")
        if item:
            compact.append(item)
    return compact


def _compact_raw_preview_rows(rows: Any, *, max_rows: int = 10, max_cols: int = 12) -> list[list[str | None]]:
    if not isinstance(rows, list):
        return []
    compact: list[list[str | None]] = []
    for row in rows[:max_rows]:
        if not isinstance(row, list):
            continue
        item: list[str | None] = []
        for value in row[:max_cols]:
            if value is None:
                item.append(None)
                continue
            text = str(value)
            item.append(text[:120] + ("..." if len(text) > 120 else ""))
        if len(row) > max_cols:
            item.append(f"... {len(row) - max_cols} more cells")
        if item:
            compact.append(item)
    return compact


def _compact_group_profiles_for_automl(group: list[FileSummary], *, limit: int = 40) -> list[dict[str, Any]]:
    if not group:
        return []
    out: list[dict[str, Any]] = []
    for col in _merged_group_columns(group, limit=limit):
        item = {
            "name": col,
            "meaning": _merge_group_field_description(group, col),
            "group_profile": _merge_group_profiles(group, col),
        }
        out.append({k: v for k, v in item.items() if str(v or "").strip()})
    return out


def _compact_file_metadata_for_automl(fs: FileSummary) -> dict[str, Any]:
    meta = fs.source_metadata or {}
    out: dict[str, Any] = {}
    for key in [
        "source_format",
        "kind",
        "shape",
        "shape_estimated",
        "preview_rows_used",
        "dtypes",
        "csv_dialect",
        "csv_encoding",
        "json_strategy",
        "json_root_type",
        "json_first_level_schema",
        "excel_sheet_names",
        "excel_default_sheet",
        "excel_sheet_groups",
        "profile_sampling",
    ]:
        value = meta.get(key)
        if value not in (None, "", [], {}):
            if key == "excel_sheet_groups" and isinstance(value, list):
                out[key] = [
                    {
                        "group_id": item.get("group_id", ""),
                        "sheet_name_pattern": item.get("sheet_name_pattern", ""),
                        "representative": item.get("representative", ""),
                        "sheet_count": item.get("sheet_count", 0),
                        "sheets": [str(x) for x in (item.get("sheets") or [])[:20]],
                        "columns": [str(x) for x in (item.get("columns") or [])[:24]],
                    }
                    for item in value
                    if isinstance(item, dict)
                ][:12]
            if key == "dtypes" and isinstance(value, dict):
                out[key] = {str(k): str(v) for k, v in list(value.items())[:80]}
            else:
                out[key] = value
    return out


def _schema_table_kind(path: str, *, sheet_name: str = "") -> str:
    suffix = Path(str(path or "")).suffix.lower()
    if sheet_name:
        return "excel_sheet"
    if suffix in {".xlsx", ".xls"}:
        return "excel_workbook_or_single_sheet"
    if suffix == ".csv":
        return "csv_table"
    if suffix in {".json", ".jsonl"}:
        return "json_table"
    if suffix in {".parquet", ".pq"}:
        return "parquet_table"
    return "table"


def _schema_profile_by_name(profiles: Any) -> dict[str, dict[str, Any]]:
    return {
        str(p.get("name", "")): p
        for p in (profiles or [])
        if isinstance(p, dict) and str(p.get("name", "")).strip()
    }


def _profile_row_counts(profiles: Any) -> list[int]:
    out: list[int] = []
    for profile in profiles or []:
        if not isinstance(profile, dict):
            continue
        value = profile.get("row_count")
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out.append(int(value))
    return sorted(set(out))


def _shape_warning(shape: Any, row_counts: list[int]) -> str:
    if not isinstance(shape, (list, tuple)) or not shape or not row_counts:
        return ""
    first = shape[0]
    if isinstance(first, bool) or not isinstance(first, (int, float)):
        return ""
    shaped_rows = int(first)
    if shaped_rows in row_counts:
        return ""
    if len(row_counts) == 1:
        return (
            f"profiled row_count={row_counts[0]} differs from reported shape rows={shaped_rows}; "
            "code should inspect the runtime dataframe shape before relying on row counts."
        )
    return (
        f"profiled row_counts={row_counts[:4]} differ from reported shape rows={shaped_rows}; "
        "code should inspect the runtime dataframe shape before relying on row counts."
    )


def _schema_field_score(name: str, meaning: str, profile: dict[str, Any]) -> int:
    text = f"{name} {meaning}".lower()
    score = 0
    priority_terms = [
        "id",
        "key",
        "target",
        "label",
        "score",
        "metric",
        "cost",
        "price",
        "amount",
        "fee",
        "date",
        "time",
        "order",
        "vehicle",
        "carrier",
        "capacity",
        "weight",
        "volume",
        "订单",
        "单号",
        "编号",
        "代码",
        "日期",
        "时间",
        "交付",
        "交货",
        "提货",
        "车辆",
        "车牌",
        "承运商",
        "车型",
        "成本",
        "费用",
        "价格",
        "重量",
        "体积",
        "数量",
        "装载",
        "容量",
        "限制",
        "约束",
    ]
    if any(term in text for term in priority_terms):
        score += 20
    logical_type = str(profile.get("logical_type") or profile.get("dtype") or "").lower()
    if logical_type in {"datetime", "date", "numeric", "integer", "float"}:
        score += 4
    if profile.get("unique_count") not in (None, "", [], {}):
        score += 2
    if profile.get("numeric_stats") or profile.get("datetime_stats"):
        score += 3
    return score


def _schema_field_summaries(
    *,
    columns: list[str],
    profiles: Any,
    meanings: dict[str, str],
    max_fields: int = 18,
) -> list[dict[str, Any]]:
    profile_by_name = _schema_profile_by_name(profiles)
    scored: list[tuple[int, int, str]] = []
    for idx, col in enumerate(columns):
        meaning = str(meanings.get(col, "") or "")
        scored.append((-_schema_field_score(col, meaning, profile_by_name.get(col, {})), idx, col))
    selected: list[str] = []
    for _score, _idx, col in sorted(scored):
        if col not in selected:
            selected.append(col)
        if len(selected) >= max(1, int(max_fields)):
            break
    out: list[dict[str, Any]] = []
    for col in selected:
        profile = profile_by_name.get(col, {})
        meaning = str(meanings.get(col, "") or "")
        numeric = profile.get("numeric_stats") if isinstance(profile.get("numeric_stats"), dict) else {}
        datetime_stats = profile.get("datetime_stats") if isinstance(profile.get("datetime_stats"), dict) else {}
        item: dict[str, Any] = {
            "name": col,
            "meaning": meaning[:160] if meaning else "",
            "logical_type": profile.get("logical_type") or profile.get("dtype"),
            "null_ratio": profile.get("null_ratio"),
            "unique_count": profile.get("unique_count"),
        }
        if numeric:
            item["numeric_range"] = {
                k: numeric.get(k)
                for k in ["min", "max"]
                if numeric.get(k) not in (None, "", [], {})
            }
        if datetime_stats:
            item["datetime_range"] = {
                k: datetime_stats.get(k)
                for k in ["min", "max"]
                if datetime_stats.get(k) not in (None, "", [], {})
            }
        out.append({k: v for k, v in item.items() if v not in (None, "", [], {})})
    return out


def _source_schema_entries_from_file(
    fs: FileSummary,
    *,
    max_columns: int = 180,
    max_field_summaries: int = 18,
) -> list[dict[str, Any]]:
    path = str(fs.path or "")
    role = fs.role.value if isinstance(fs.role, FileRole) else str(fs.role)
    meta = fs.source_metadata or {}
    entries: list[dict[str, Any]] = []
    sheet_profiles = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
    if sheet_profiles:
        for sheet in sheet_profiles:
            if not isinstance(sheet, dict):
                continue
            sheet_name = str(sheet.get("sheet_name", "") or "")
            columns = [str(x) for x in (sheet.get("columns") or []) if str(x).strip()]
            profiles = sheet.get("column_profiles") if isinstance(sheet.get("column_profiles"), list) else []
            meanings = _sheet_field_descriptions(fs, sheet_name)
            shape = sheet.get("shape") or sheet.get("shape_profiled") or sheet.get("shape_sampled")
            row_counts = _profile_row_counts(profiles)
            warnings = [x for x in [_shape_warning(shape, row_counts)] if x]
            entries.append(
                {
                    "table_id": f"{path}::{sheet_name}" if sheet_name else path,
                    "source_file": path,
                    "sheet_name": sheet_name,
                    "table_kind": "excel_sheet",
                    "file_role": role,
                    "shape": shape,
                    "row_count_from_profiles": row_counts[0] if len(row_counts) == 1 else row_counts[:4],
                    "column_count": len(columns),
                    "physical_columns_exact": columns[:max_columns],
                    "physical_columns_omitted": max(0, len(columns) - max_columns),
                    "field_summaries": _schema_field_summaries(
                        columns=columns,
                        profiles=profiles,
                        meanings=meanings,
                        max_fields=max_field_summaries,
                    ),
                    "read_example": f"pd.read_excel('./input/{path}', sheet_name={sheet_name!r})",
                    "warnings": warnings,
                }
            )
        return entries

    columns = [str(x) for x in (fs.columns or []) if str(x).strip()]
    profiles = fs.column_profiles or []
    meanings = {str(k): str(v) for k, v in (fs.column_semantics or {}).items() if str(k).strip()}
    shape = meta.get("shape") or meta.get("shape_estimated")
    row_counts = _profile_row_counts(profiles)
    warnings = [x for x in [_shape_warning(shape, row_counts)] if x]
    entries.append(
        {
            "table_id": path,
            "source_file": path,
            "sheet_name": "",
            "table_kind": _schema_table_kind(path),
            "file_role": role,
            "shape": shape,
            "row_count_from_profiles": row_counts[0] if len(row_counts) == 1 else row_counts[:4],
            "column_count": len(columns),
            "physical_columns_exact": columns[:max_columns],
            "physical_columns_omitted": max(0, len(columns) - max_columns),
            "field_summaries": _schema_field_summaries(
                columns=columns,
                profiles=profiles,
                meanings=meanings,
                max_fields=max_field_summaries,
            ),
            "read_example": _default_schema_read_example(path),
            "warnings": warnings,
        }
    )
    return entries


def _default_schema_read_example(path: str) -> str:
    suffix = Path(str(path or "")).suffix.lower()
    quoted = f"./input/{path}"
    if suffix == ".csv":
        return f"pd.read_csv({quoted!r})"
    if suffix in {".xlsx", ".xls"}:
        return f"pd.read_excel({quoted!r})"
    if suffix in {".json", ".jsonl"}:
        return f"pd.read_json({quoted!r})"
    if suffix in {".parquet", ".pq"}:
        return f"pd.read_parquet({quoted!r})"
    return f"# inspect and load {quoted!r} with the appropriate reader"


def _schema_entry_pattern_signature(entry: dict[str, Any]) -> tuple[Any, ...] | None:
    sig = _filename_pattern_signature(str(entry.get("source_file") or ""))
    if sig is None:
        return None
    parent, suffix, pattern = sig
    return (
        parent,
        suffix,
        pattern,
        entry.get("table_kind"),
        entry.get("sheet_name"),
        entry.get("file_role"),
    )


def _schema_group_column_profile(items: list[dict[str, Any]]) -> dict[str, Any]:
    observed: list[tuple[str, list[str]]] = []
    for item in items:
        source_file = str(item.get("source_file") or item.get("table_id") or "").strip()
        cols = [str(x) for x in (item.get("physical_columns_exact") or []) if str(x).strip()]
        if source_file and cols:
            observed.append((source_file, cols))
    if not observed:
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
    for source_file, cols in observed[:16]:
        only_fields = [col for col in cols if col not in common_set]
        if only_fields:
            variant_fields_by_file.append(
                {
                    "file": source_file,
                    "fields": only_fields[:24],
                    "omitted": max(0, len(only_fields) - 24),
                }
            )

    field_presence: list[dict[str, Any]] = []
    for col in union:
        if col in common_set:
            continue
        present = [source_file for source_file, cols in observed if col in set(cols)]
        field_presence.append(
            {
                "field": col,
                "present_in_count": len(present),
                "example_files": present[:3],
            }
        )

    return {
        "observed_file_count": len(observed),
        "shared_fields": shared_fields,
        "variant_fields_by_file": variant_fields_by_file,
        "field_presence": field_presence,
    }


def _merge_schema_group_entries(items: list[dict[str, Any]]) -> dict[str, Any]:
    rep = dict(items[0])
    paths = [str(x.get("source_file", "")) for x in items if str(x.get("source_file", "")).strip()]
    pattern = _common_path_pattern(paths)
    columns: list[str] = []
    for item in items:
        for col in item.get("physical_columns_exact") or []:
            text = str(col).strip()
            if text and text not in columns:
                columns.append(text)
    field_by_name: dict[str, dict[str, Any]] = {}
    for item in items:
        for field in item.get("field_summaries") or []:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name") or "").strip()
            if name and name not in field_by_name:
                field_by_name[name] = dict(field)
    column_variants = {
        tuple(str(x) for x in (item.get("physical_columns_exact") or []))
        for item in items
    }
    column_profile = _schema_group_column_profile(items)
    rep.update(
        {
            "table_id": f"{pattern}::{rep.get('sheet_name')}" if rep.get("sheet_name") else pattern,
            "column_count": len(columns),
            "physical_columns_exact": columns,
            "physical_columns_omitted": 0,
            "field_summaries": list(field_by_name.values())[:18],
            "schema_group": {
                "file_count": len(items),
                "schema_consistent": len(column_variants) == 1,
                "column_variant_count": len(column_variants),
                "representative_files": paths[:5],
                "shared_physical_columns_exact": column_profile.get("shared_fields", [])[:80],
                "variant_fields_by_file": column_profile.get("variant_fields_by_file", [])[:12],
                "field_presence": column_profile.get("field_presence", [])[:24],
                "note": (
                    "Repeated filename-pattern group; physical_columns_exact is the union of observed "
                    "columns, not a guarantee that every file has every column. Use shared_physical_columns_exact "
                    "and variant_fields_by_file before per-file dataframe access."
                ),
            },
            "read_example": (
                "For each file matching this filename pattern, use the same reader, "
                "then concatenate and keep a source_file column."
            ),
        }
    )
    rep.pop("source_file", None)
    return {k: v for k, v in rep.items() if v not in (None, "", [], {})}


def _group_schema_entries(entries: list[dict[str, Any]], *, max_tables: int = 80) -> list[dict[str, Any]]:
    pattern_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in entries:
        sig = _schema_entry_pattern_signature(entry)
        if sig is not None:
            pattern_grouped.setdefault(sig, []).append(entry)

    emitted_ids: set[int] = set()
    out: list[dict[str, Any]] = []
    for entry in entries:
        sig = _schema_entry_pattern_signature(entry)
        items = pattern_grouped.get(sig, []) if sig is not None else []
        if len(items) >= 3 and id(entry) not in emitted_ids:
            out.append(_merge_schema_group_entries(items))
            emitted_ids.update(id(x) for x in items)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in entries:
        if id(entry) in emitted_ids:
            continue
        columns = tuple(str(x) for x in (entry.get("physical_columns_exact") or []))
        key = (
            entry.get("table_kind"),
            entry.get("sheet_name"),
            entry.get("file_role"),
            entry.get("column_count"),
            columns,
        )
        grouped.setdefault(key, []).append(entry)

    for items in grouped.values():
        rep = dict(items[0])
        paths = [str(x.get("source_file", "")) for x in items if str(x.get("source_file", "")).strip()]
        if len(items) > 1:
            column_profile = _schema_group_column_profile(items)
            rep["table_id"] = (
                f"{_common_path_pattern(paths)}::{rep.get('sheet_name')}"
                if rep.get("sheet_name")
                else _common_path_pattern(paths)
            )
            rep["schema_group"] = {
                "file_count": len(items),
                "representative_files": paths[:3],
                "schema_consistent": True,
                "column_variant_count": 1,
                "shared_physical_columns_exact": column_profile.get("shared_fields", [])[:80],
                "variant_fields_by_file": column_profile.get("variant_fields_by_file", [])[:12],
                "field_presence": column_profile.get("field_presence", [])[:24],
            }
            rep.pop("source_file", None)
            read = rep.get("read_example")
            if read:
                rep["read_example"] = (
                    "For each matching file, use the same reader as the representative example, "
                    "then concatenate and keep a source_file column."
                )
        out.append({k: v for k, v in rep.items() if v not in (None, "", [], {})})

    out.sort(
        key=lambda item: (
            0 if str(item.get("file_role", "")) == "raw_data_table" else 1,
            str(item.get("table_id", "")),
        )
    )
    return out[: max(1, int(max_tables))]


def _build_data_schema_contract(file_summaries: list[FileSummary], *, max_tables: int = 80) -> dict[str, Any]:
    raw_entries: list[dict[str, Any]] = []
    workbook_sheets: list[dict[str, Any]] = []
    structured_files = [
        fs
        for fs in file_summaries or []
        if str(fs.path or "").lower().endswith((".csv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet", ".pq"))
    ]
    data_files = [fs for fs in structured_files if not _is_document_like(fs)] or structured_files
    for fs in data_files:
        meta = fs.source_metadata or {}
        sheet_names = meta.get("excel_sheet_names") if isinstance(meta.get("excel_sheet_names"), list) else []
        if sheet_names:
            workbook_sheets.append(
                {
                    "source_file": str(fs.path or ""),
                    "valid_sheet_names_exact": [str(x) for x in sheet_names if str(x).strip()],
                    "rule": "Only these exact strings are valid for pandas.read_excel(..., sheet_name=...). File roles or business labels are not sheet names.",
                }
            )
        raw_entries.extend(_source_schema_entries_from_file(fs))

    tables = _group_schema_entries(raw_entries, max_tables=max_tables)
    return {
        "purpose": "Authoritative physical schema map for downstream code. Use this before writing pandas column or sheet access.",
        "rules": [
            "Physical source column names are exactly the strings in `physical_columns_exact`; do not normalize, translate, or silently rename them before checking they exist.",
            "For repeated filename-pattern groups, `physical_columns_exact` may be the union of observed columns; use `shared_physical_columns_exact`, `variant_fields_by_file`, and runtime `df.columns` checks before assuming a column exists in every file.",
            "Business concepts such as delivery day, target, capacity, route, vehicle, or cost are derived/code-local variables unless the exact same string appears in `physical_columns_exact`.",
            "For Excel files, only `valid_sheet_names_exact` / table `sheet_name` values are legal sheet names. Do not use table roles such as cost contract or daily vehicle pool as sheet names unless they are listed exactly.",
            "If code wants English variable names such as `delivery_day`, `max_weight_kg`, or `carrier_code`, create them by explicit mapping from exact source columns and keep the mapping near the load code.",
            "Before hard-coding a rename or merge key, inspect `pd.ExcelFile(path).sheet_names` and `df.columns.tolist()` at runtime and fail with a diagnostic listing available names if the expected exact name is absent.",
            "Natural-language field meanings in description.md or field_summaries are explanatory only; they are not dataframe column names.",
        ],
        "runtime_inspection_snippet": (
            "For Excel: xls = pd.ExcelFile(path); print(xls.sheet_names); "
            "df = pd.read_excel(path, sheet_name=exact_sheet); print(df.columns.tolist()). "
            "For CSV/JSON/parquet: load once, then print(df.columns.tolist()) before renaming."
        ),
        "workbooks": workbook_sheets[:40],
        "tables": tables,
        "omitted_table_count": max(0, len(raw_entries) - len(tables)),
    }


def build_data_schema_contract(file_summaries: list[FileSummary], *, max_tables: int = 80) -> dict[str, Any]:
    """Build the downstream code-facing exact source schema contract."""

    return _build_data_schema_contract(file_summaries, max_tables=max_tables)


def _paradigm_label(paradigm: str) -> str:
    labels = {
        "ml_dl_prediction": "机器学习/深度学习预测",
        "static_optimization": "静态优化/组合决策",
        "reinforcement_learning": "强化学习/序贯决策",
        "hybrid_ml_optimization": "预测与优化混合",
        "unknown_but_executable": "可执行建模任务",
    }
    return labels.get(str(paradigm or "").strip(), str(paradigm or "").strip() or "可执行建模任务")


def _render_data_access_section(protocol: DataAccessProtocol) -> str:
    lines = ["## 数据与读取方式"]
    lines.append("- 本节只列容易读错或需要特殊编排的输入：多工作表 Excel、非默认分隔符 CSV、作为数据使用的 JSON，以及需要按 glob 合并的重复文件组。")
    lines.append("- 未列出的普通 CSV/Excel 可按 pandas 默认方式读取，但路径仍应以 `./input` 为根目录。")
    if protocol.global_notes:
        for note in protocol.global_notes[:2]:
            lines.append(f"- {note}")
    if not protocol.files:
        lines.append("- 暂未识别需要直接读取的结构化数据文件；请以原始任务说明中的输入文件为准。")
        return "\n".join(lines)
    rendered_any = False
    for is_group, items in _group_data_access_items(list(protocol.files)):
        if is_group:
            representative = items[0]
            paths = [str(x.path) for x in items]
            pattern = _common_path_pattern(paths)
            rendered_any = True
            lines.append("")
            lines.append(f"### {pattern}")
            lines.append(f"- 文件角色：{representative.file_role or '数据文件'}。")
            row_grains = _dedupe_nonempty([str(x.row_grain) for x in items], limit=3)
            if row_grains:
                lines.append("- 数据粒度：" + "；".join(row_grains))
            lines.append(f"- 读取方法：`{representative.read_method or 'pandas'}`，按 glob 批量读取并保留 `source_file` 以区分来源。")
            lines.append("- 批量读取示例：")
            lines.extend(_render_group_read_code(pattern, representative))
            key_fields = _dedupe_nonempty([x for item in items for x in item.key_fields], limit=12)
            if key_fields:
                lines.append("- 关键实体键：" + "、".join(f"`{x}`" for x in key_fields))
            field_profile = _data_access_group_field_profile(items)
            shared_fields = _dedupe_nonempty([str(x) for x in field_profile.get("shared_fields", [])], limit=24)
            if shared_fields:
                lines.append("- 同组共通字段：" + "、".join(f"`{x}`" for x in shared_fields))
            variants = field_profile.get("variant_fields_by_file") if isinstance(field_profile.get("variant_fields_by_file"), list) else []
            if variants:
                lines.append("- 同组差异字段（读取时不要假设每个文件都有 union 中的全部字段）：")
                for idx, variant in enumerate(variants[:8], start=1):
                    if not isinstance(variant, dict):
                        continue
                    fields = _dedupe_nonempty([str(x) for x in (variant.get("fields") or [])], limit=18)
                    omitted = int(variant.get("omitted") or 0)
                    suffix = f"；另有 {omitted} 个字段省略" if omitted else ""
                    lines.append(
                        f"  - 变体 {idx} 独有/非共通字段："
                        + ("、".join(f"`{x}`" for x in fields) if fields else "（未识别）")
                        + suffix
                    )
            elif shared_fields:
                lines.append("- 同组差异字段：未在已观测字段中发现差异。")
            notes = _dedupe_nonempty([x for item in items for x in item.parsing_notes], limit=8)
            if notes:
                lines.append("- 读取注意事项：")
                for note in notes:
                    lines.append(f"  - {note}")
            continue

        item = items[0]
        if not _needs_explicit_read_example(item):
            continue
        rendered_any = True
        lines.append("")
        lines.append(f"### {item.path}")
        lines.append(f"- 文件角色：{item.file_role or '数据文件'}。")
        if item.row_grain:
            lines.append(f"- 数据粒度：{item.row_grain}")
        lines.append(f"- 读取方法：`{item.read_method or 'pandas'}`。")
        if item.read_example:
            lines.append("- 读取示例：")
            lines.append("```python")
            lines.append("import pandas as pd")
            if "Path(" in item.read_example:
                lines.append("from pathlib import Path")
            lines.append(f"df = {item.read_example}")
            lines.append("```")
        if item.key_fields:
            lines.append("- 关键实体键：" + "，".join(f"`{x}`" for x in item.key_fields))
        if item.parsing_notes:
            lines.append("- 读取注意事项：")
            for note in item.parsing_notes[:8]:
                lines.append(f"  - {note}")
    if not rendered_any:
        lines.append("- 当前未发现需要特殊读取说明的结构化文件。")
    return "\n".join(lines)


def build_automl_context_pack(
    bundle: DescriptionProtocolBundle,
    file_summaries: list[FileSummary],
    downstream_context: dict | None = None,
    evaluation_contract: EvaluationContractReview | dict[str, Any] | None = None,
    compiled_context: dict[str, Any] | None = None,
) -> AutoMLContextPack:
    """Build concise supplemental facts for the downstream fixed context.

    `description.md` remains the primary task document. This pack records
    important constraints, contracts, table facts, and data notes that would
    be too detailed or too machine-oriented for the human-facing description.
    It intentionally avoids retrieval mechanics: downstream code is generated
    as a whole solution, so important facts must be present here or in
    `description.md` rather than hidden behind an interactive fetch step.
    """
    ctx = downstream_context or {}
    if not isinstance(bundle, DescriptionProtocolBundle):
        bundle = DescriptionProtocolBundle.model_validate(bundle)
    contract = _as_evaluation_contract(evaluation_contract) if evaluation_contract is not None else None
    paradigm = str(bundle.problem_paradigm or ctx.get("problem_paradigm", "unknown_but_executable")).strip()
    problem_review = ctx.get("problem_paradigm_review") if isinstance(ctx.get("problem_paradigm_review"), dict) else {}
    explicit_rl_requested = bool(problem_review.get("explicit_rl_requested"))
    rl_as_required_paradigm = bool(problem_review.get("rl_as_required_paradigm")) or paradigm == "reinforcement_learning"
    recommended_solver_families = _dedupe_any(problem_review.get("recommended_solver_families", []), limit=10)
    method_routing_notes = _dedupe_any(problem_review.get("method_routing_notes", []), limit=10)
    summaries_by_path = {str(fs.path): fs for fs in file_summaries}
    compact_ctx = compiled_context if isinstance(compiled_context, dict) else {}
    compact_table_cards = compact_ctx.get("table_cards") if isinstance(compact_ctx.get("table_cards"), list) else []
    compact_relations = compact_ctx.get("relations") if isinstance(compact_ctx.get("relations"), list) else []
    compact_filename_groups = (
        compact_ctx.get("filename_sample_groups")
        if isinstance(compact_ctx.get("filename_sample_groups"), list)
        else []
    )
    data_schema_contract = build_data_schema_contract(file_summaries)

    data_entries: list[dict[str, Any]] = []
    entity_alias_source_aliases: dict[str, str] = {}
    if compact_table_cards:
        protocol_by_path = {str(item.path): item for item in list(bundle.data_access.files or [])}
        for card in compact_table_cards[:40]:
            if not isinstance(card, dict):
                continue
            path = str(card.get("source_file") or card.get("table_id") or "")
            protocol = protocol_by_path.get(path)
            raw_fields = card.get("fields") if isinstance(card.get("fields"), list) else None
            fields = [f for f in (raw_fields or card.get("field_index") or []) if isinstance(f, dict)]
            field_hints = [str(x) for x in (card.get("field_hints") or []) if str(x).strip()]
            fs = summaries_by_path.get(path)
            if fs is None and str(card.get("source_file", "")):
                fs = summaries_by_path.get(str(card.get("source_file", "")))
            if not fields and fs is not None:
                sheet_name = str(card.get("sheet_name", "") or "")
                if sheet_name:
                    sheet = _find_excel_sheet_profile(fs, sheet_name)
                    if sheet is not None:
                        fields = _sheet_profile_items(sheet, fs, max_cols=24)
                else:
                    fields = _compact_file_profiles_for_automl(fs, limit=24)
            columns = [str(f.get("name", "")) for f in fields if str(f.get("name", "")).strip()]
            if not columns:
                columns = field_hints or ([str(x) for x in (fs.columns or [])[:80]] if fs is not None else [])
            entry = {
                "kind": card.get("table_kind") or "table_card",
                "path": path,
                "table_id": card.get("table_id"),
                "sheet_name": card.get("sheet_name"),
                "file_role": card.get("role"),
                "summary": (card.get("file_cognition") or (str(fs.summary or "")[:400] if fs is not None else "")),
                "shape": card.get("shape"),
                "read_method": protocol.read_method if protocol is not None else "",
                "read_example": protocol.read_example if protocol is not None else "",
                "columns": columns[:80],
                "fields": fields[:24],
                "key_fields": protocol.key_fields if protocol is not None else [],
                "relation_keys": protocol.relation_keys if protocol is not None else [],
                "important_fields": protocol.important_fields if protocol is not None else [],
                "parsing_notes": _dedupe_any(
                    list(card.get("reading_notes") or [])
                    + (list(protocol.parsing_notes) if protocol is not None else []),
                    limit=16,
                ),
                "warnings": card.get("warnings", []),
            }
            data_entries.append({k: v for k, v in entry.items() if v not in (None, "", [], {})})
    else:
        for is_group, items in _group_data_access_items(list(bundle.data_access.files or [])):
            if not items:
                continue
            representative = items[0]
            paths = [str(x.path) for x in items]
            grouped_summaries = [summaries_by_path[p] for p in paths if p in summaries_by_path]
            if is_group:
                pattern = _common_path_pattern(paths)
                for path in paths:
                    entity_alias_source_aliases[path] = pattern
                read_example = "\n".join(_render_group_read_code(pattern, representative))
                entry = {
                    "kind": "repeated_file_group",
                    "pattern": pattern,
                    "file_count": len(items),
                    "read_method": representative.read_method,
                    "read_example": read_example,
                    "columns": _merged_group_columns(grouped_summaries, limit=80),
                    "row_grain": "; ".join(_dedupe_any([x.row_grain for x in items], limit=4)),
                    "key_fields": _dedupe_any([v for item in items for v in item.key_fields], limit=16),
                    "relation_keys": _dedupe_any([v for item in items for v in item.relation_keys], limit=16),
                    "important_fields": _dedupe_any([v for item in items for v in item.important_fields], limit=30),
                    "parsing_notes": _dedupe_any([v for item in items for v in item.parsing_notes], limit=12),
                    "orchestration_note": "Read every file matching the pattern, concatenate rows, and keep a `source_file` column to preserve file-level identity.",
                }
                group_profiles = _compact_group_profiles_for_automl(grouped_summaries)
                if group_profiles:
                    entry["field_profiles"] = group_profiles
                group_sheet_profiles = _compact_group_excel_sheet_profiles_for_automl(grouped_summaries)
                if group_sheet_profiles:
                    entry["excel_sheet_profiles"] = group_sheet_profiles
            else:
                entry = {
                    "kind": "single_file",
                    "path": representative.path,
                    "read_method": representative.read_method,
                    "read_example": representative.read_example,
                    "columns": [],
                    "row_grain": representative.row_grain,
                    "key_fields": _dedupe_any(representative.key_fields, limit=16),
                    "relation_keys": _dedupe_any(representative.relation_keys, limit=16),
                    "important_fields": _dedupe_any(representative.important_fields, limit=30),
                    "parsing_notes": _dedupe_any(representative.parsing_notes, limit=12),
                }
                fs = summaries_by_path.get(str(representative.path))
                if fs is not None:
                    entry["file_role"] = fs.role.value if isinstance(fs.role, FileRole) else str(fs.role)
                    entry["summary"] = str(fs.summary or "")[:600]
                    entry["columns"] = [str(x) for x in (fs.columns or [])[:120]]
                    metadata = _compact_file_metadata_for_automl(fs)
                    if metadata:
                        entry["source_metadata"] = metadata
                    profiles = _compact_file_profiles_for_automl(fs)
                    if profiles:
                        entry["field_profiles"] = profiles
                    sheet_profiles = _compact_excel_sheet_profiles_for_automl(fs)
                    if sheet_profiles:
                        entry["excel_sheet_profiles"] = sheet_profiles
            data_entries.append(entry)

    entity_alias_candidates = build_entity_alias_candidates(
        file_summaries,
        filename_sample_groups=compact_filename_groups,
        source_aliases=entity_alias_source_aliases,
    )

    output = bundle.output
    output_contract = {
        "output_kind": output.output_kind,
        "output_filename": output.output_filename,
        "sample_submission_required": bool(output.sample_submission_required),
        "columns": _dedupe_any(output.columns or ctx.get("submission_columns") or ctx.get("generated_submission_columns"), limit=40),
        "row_unit": output.row_unit,
        "format_rules": _dedupe_any(output.format_rules, limit=20),
        "no_sample_submission_reason": output.no_sample_submission_reason,
        "authoritative_submission_contract": ctx.get("authoritative_submission_contract", {}),
        "sample_submission_spec": ctx.get("sample_submission_spec", {}),
        "sample_submission_available": bool(ctx.get("sample_submission_available", False)),
        "sample_submission_generation_status": ctx.get("sample_submission_generation_status", ""),
        "sample_submission_generation_issues": ctx.get("sample_submission_generation_issues", []),
        "sample_submission_source_field_corrections": ctx.get("sample_submission_source_field_corrections", []),
        "generated_submission_path": ctx.get("generated_submission_path", ""),
        "generated_submission_columns": ctx.get("generated_submission_columns", []),
    }

    if contract is not None:
        evaluation = contract.model_dump()
        final_formula = _final_score_formula(contract)
        if final_formula:
            evaluation["final_score_formula"] = final_formula
        evaluation["single_scalar_score_required"] = True
        evaluation["final_validation_score_rule"] = (
            "The code must print one numeric `Final Validation Score`. "
            "Use `final_score_formula` / `metric_formula` as the single ranking score. "
            "`scalar_score_formula` is a legacy mirror only when identical; never optimize a second score."
        )
    else:
        evaluation = {
            "single_scalar_score_required": True,
            "primary_metric": bundle.evaluation_summary,
            "metric_direction": "",
            "metric_formula": bundle.evaluation_summary,
            "scalar_score_formula": "",
            "final_score_formula": bundle.evaluation_summary,
            "final_validation_score_rule": "Metric must evaluate to one numeric scalar.",
        }

    method_strategy: dict[str, Any] = {
        "problem_paradigm": paradigm or "unknown_but_executable",
        "explicit_rl_requested": explicit_rl_requested,
        "rl_as_required_paradigm": rl_as_required_paradigm,
        "recommended_solver_families": recommended_solver_families,
        "method_routing_notes": method_routing_notes,
    }
    if paradigm in {"static_optimization", "hybrid_ml_optimization"}:
        method_strategy["first_draft_policy"] = (
            "Build a deterministic evaluator plus a greedy/repair/local-search or OR baseline first. "
            "The first runnable node should produce a real, possibly partial, solution and a penalized scalar score."
        )
        method_strategy["rl_branch_policy"] = (
            "If RL is requested, treat it as a later comparable branch. It must reuse the same load_problem_data, "
            "validate_solution, score_solution, output schema, hard constraints, and final scalar score."
        )
    elif paradigm == "reinforcement_learning":
        method_strategy["first_draft_policy"] = (
            "Build the environment/evaluator contract first, then compare a simple policy or heuristic rollout before expensive RL training."
        )
    else:
        method_strategy["first_draft_policy"] = "Follow the task paradigm, but keep the first runnable solution simple and fully evaluable."

    modeling_boundary: list[str] = []
    if paradigm == "ml_dl_prediction":
        p = bundle.ml_dl
        modeling_boundary.extend(
            _dedupe_any(
                [
                    f"Train data: {p.train_data}",
                    f"Predict/test data: {p.predict_data}",
                    f"Target: {p.target}",
                    f"Prediction unit: {p.prediction_unit}",
                    f"Validation design: {p.validation_design}",
                    *p.feature_boundary,
                ],
                limit=24,
            )
        )
    elif paradigm == "static_optimization":
        p = bundle.optimization
        modeling_boundary.extend(
            _dedupe_any(
                [
                    f"Input instance: {p.input_instance}",
                    f"Objective: {p.objective}",
                    f"Solution representation: {p.solution_representation}",
                    *p.decision_variables,
                    *p.feasibility_checks,
                ],
                limit=24,
            )
        )
        if explicit_rl_requested:
            modeling_boundary.extend(
                [
                    "RL is a requested/allowed solver branch, not the task paradigm; do not replace the deterministic evaluator with a reward-only metric.",
                    "First implement a scorable static optimization baseline, then compare any RL policy against that same score_solution contract.",
                ]
            )
    elif paradigm == "reinforcement_learning":
        p = bundle.rl
        modeling_boundary.extend(
            _dedupe_any(
                [
                    f"Environment: {p.environment}",
                    f"State: {p.state}",
                    f"Action: {p.action}",
                    f"Transition: {p.transition}",
                    f"Reward: {p.reward}",
                    f"Terminal: {p.terminal_condition}",
                    f"Policy output: {p.policy_output}",
                    f"Evaluation episodes: {p.evaluation_episodes}",
                    *p.illegal_action_handling,
                ],
                limit=24,
            )
        )
    elif paradigm == "hybrid_ml_optimization":
        p = bundle.hybrid
        modeling_boundary.extend(
            _dedupe_any(
                [
                    f"Prediction subproblem: {p.prediction_subproblem}",
                    f"Decision subproblem: {p.decision_subproblem}",
                    f"Handoff: {p.handoff}",
                    f"Final objective: {p.final_objective}",
                    f"Validation design: {p.validation_design}",
                ],
                limit=20,
            )
        )

    data_orchestration = [
        "Runtime data root fact: input files are available under `./input` in downstream workspaces.",
        "Non-default CSV dialects, repeated-file groups, multi-sheet Excel, and JSON table extraction notes are recorded only when detected.",
        "This file is fixed-context supplemental facts; `description.md` remains the primary task statement.",
    ]
    if explicit_rl_requested and paradigm in {"static_optimization", "hybrid_ml_optimization"}:
        data_orchestration.append(
            "RL was requested in the task text, but AutoRealize classified the executable contract as static optimization because evaluation consumes a complete solution table/plan."
        )
    evidence_levels = ctx.get("evidence_levels") if isinstance(ctx.get("evidence_levels"), dict) else {}
    heuristic_fields = [str(x) for x in (ctx.get("heuristic_fields") or []) if str(x).strip()]
    if evidence_levels:
        data_orchestration.append(
            "Downstream context evidence levels: "
            + "; ".join(f"{k}={v}" for k, v in evidence_levels.items() if str(v).strip())
        )
    if heuristic_fields:
        data_orchestration.append(
            "Treat these downstream fields as non-authoritative heuristics unless confirmed by the main task protocol: "
            + ", ".join(heuristic_fields)
        )
    if len(data_entries) > 1:
        data_orchestration.append("Join/merge tables only through documented key fields, relation keys, filename IDs, or authoritative task descriptions.")
    if compact_table_cards:
        data_orchestration.append(
            "Large previews and raw source metadata are intentionally omitted from this fixed context."
        )
    if compact_relations:
        data_orchestration.append(
            "relation_cards are non-authoritative join hints unless confirmed by task requirements."
        )

    constraints = _dedupe_any(
        list(bundle.constraints or [])
        + list((ctx.get("authoritative_memory") or {}).get("constraints", []) if isinstance(ctx.get("authoritative_memory"), dict) else [])
        + list((ctx.get("constraint_memory") or {}).get("items", []) if isinstance(ctx.get("constraint_memory"), dict) else []),
        limit=30,
    )
    leakage_guards = _dedupe_any(
        (contract.leakage_guards if contract is not None else [])
        + list(bundle.ml_dl.leakage_guards or [])
        + list(bundle.warnings or []),
        limit=24,
    )
    pitfalls = _dedupe_any(
        [
            "Do not invent submission columns, target fields, row-count rules, random seeds, distance matrices, or cost formulas.",
            "Do not use future information or validation/test labels during training, preprocessing, feature engineering, policy construction, or optimization.",
            "Do not force `id,target` or `submission.csv` for optimization/RL tasks unless an authoritative contract requires it.",
            "The final search metric must be one scalar number so tree search can compare nodes consistently.",
            "If original requirements or description prose uses a business synonym that is not an exact source column, resolve it against the Exact Source Schema Contract before pandas access; prose aliases are never raw dataframe names.",
            *list(bundle.warnings or []),
        ],
        limit=24,
    )

    return AutoMLContextPack(
        priority_rules=[
            "Exact Source Schema Contract is authoritative for pandas sheet/column access; natural-language aliases are not raw dataframe names.",
            "Read the Source Alias Guard before coding: aliases listed there are business concepts or corrected names, not safe raw dataframe columns unless an exact_physical_column is provided.",
            "Authority priority: user task hint > existing input description.md > README/official/spec/other task documents > data statistics and LLM inference.",
            "If AutoRealize records source-field corrections, downstream code must use the corrected exact physical column names for pandas access.",
            "AutoRealize evaluation/output/data-access contracts override generic Kaggle templates and MLEvolve preview heuristics.",
            "Runtime path facts: input files under `./input`; configured outputs under `./submission` only when required.",
            "For static optimization/dispatch tasks, first produce a deterministic scorable baseline; RL is a later comparable branch unless the problem paradigm is explicitly reinforcement_learning.",
        ],
        problem_paradigm=paradigm or "unknown_but_executable",
        task_goal=bundle.task_goal or bundle.overview or str(ctx.get("task_hint", "")),
        data_orchestration=data_orchestration,
        data_access=data_entries,
        data_schema_contract=data_schema_contract,
        source_alias_guard=list(ctx.get("source_alias_guard", []) or []),
        entity_alias_candidates=entity_alias_candidates,
        output_contract=output_contract,
        evaluation_contract=evaluation,
        method_strategy=method_strategy,
        relation_cards=compact_relations[:80],
        filename_sample_groups=compact_filename_groups[:40],
        modeling_boundary=modeling_boundary,
        constraints=constraints,
        leakage_guards=leakage_guards,
        pitfalls=pitfalls,
        source_artifacts={
            "description": "description.md",
            "data_description": "realize_report/data_description.md",
            "problem_paradigm": "realize_report/problem_paradigm_report.json",
            "data_access_protocol": "realize_report/data_access_protocol.json",
            "description_protocol_bundle": "realize_report/description_protocol_bundle.json",
            "evaluation_contract": "realize_report/evaluation_contract_report.json",
        },
    )


def render_automl_context_markdown(pack: AutoMLContextPack | dict[str, Any]) -> str:
    """Render the AutoML context pack into a concise prompt-friendly markdown file."""
    if not isinstance(pack, AutoMLContextPack):
        pack = AutoMLContextPack.model_validate(pack)
    lines: list[str] = [
        "## AutoRealize Structured Context",
        "",
        "This file is a concise fixed-context supplement to `description.md`.",
        "It records important contracts, constraints, table facts, and data caveats that may be too detailed for the human-facing description.",
        "It must not rely on interactive retrieval: important supplemental facts should be visible here or in `description.md`.",
        "",
        "## Priority Rules",
    ]
    for item in _dedupe_any(pack.priority_rules, limit=12):
        lines.append(f"- {item}")

    schema_contract = pack.data_schema_contract or {}
    if schema_contract:
        lines.extend(["", "## Exact Source Schema Contract"])
        purpose = str(schema_contract.get("purpose", "") or "").strip()
        if purpose:
            lines.append(f"- purpose: {purpose}")
        rules = _dedupe_any(schema_contract.get("rules", []), limit=12)
        if rules:
            lines.append("- hard_rules:")
            lines.extend(f"  - {x}" for x in rules)
        snippet = str(schema_contract.get("runtime_inspection_snippet", "") or "").strip()
        if snippet:
            lines.append(f"- runtime_inspection_snippet: {snippet}")
        workbooks = schema_contract.get("workbooks") if isinstance(schema_contract.get("workbooks"), list) else []
        if workbooks:
            lines.append("- excel_workbook_sheet_names:")
            for item in workbooks[:24]:
                if not isinstance(item, dict):
                    continue
                sheets = [str(x) for x in (item.get("valid_sheet_names_exact") or [])[:24]]
                lines.append(
                    f"  - source_file={item.get('source_file')}; "
                    f"valid_sheet_names_exact={sheets}"
                )
        tables = schema_contract.get("tables") if isinstance(schema_contract.get("tables"), list) else []
        if tables:
            lines.append("- tables:")
            for table in tables[:40]:
                if not isinstance(table, dict):
                    continue
                lines.append(
                    f"  - table_id={table.get('table_id')}; "
                    f"kind={table.get('table_kind')}; "
                    f"source_file={table.get('source_file')}; "
                    f"sheet_name={table.get('sheet_name')}; "
                    f"shape={table.get('shape')}; "
                    f"row_count_from_profiles={table.get('row_count_from_profiles')}; "
                    f"column_count={table.get('column_count')}"
                )
                group = table.get("schema_group") if isinstance(table.get("schema_group"), dict) else {}
                if group:
                    representative_count = len(group.get("representative_files") or [])
                    lines.append(
                        f"    schema_group: file_count={group.get('file_count')}; "
                        f"representative_count={representative_count}; "
                        f"schema_consistent={group.get('schema_consistent')}; "
                        f"column_variant_count={group.get('column_variant_count')}"
                    )
                    shared = [str(x) for x in (group.get("shared_physical_columns_exact") or [])]
                    if shared:
                        lines.append(
                            "    shared_physical_columns_exact: "
                            + ", ".join(f"`{x}`" for x in shared[:80])
                        )
                    variants = group.get("variant_fields_by_file") if isinstance(group.get("variant_fields_by_file"), list) else []
                    if variants:
                        lines.append("    variant_fields_by_file:")
                        for idx, variant in enumerate(variants[:12], start=1):
                            if not isinstance(variant, dict):
                                continue
                            fields = [str(x) for x in (variant.get("fields") or [])]
                            suffix = f"; omitted={variant.get('omitted')}" if variant.get("omitted") else ""
                            lines.append(
                                f"      - variant=variant_{idx}; "
                                f"fields={fields[:24]}{suffix}"
                            )
                    presence = group.get("field_presence") if isinstance(group.get("field_presence"), list) else []
                    if presence:
                        lines.append("    non_shared_field_presence:")
                        for item in presence[:16]:
                            if not isinstance(item, dict):
                                continue
                            examples = item.get("example_files") if isinstance(item.get("example_files"), list) else []
                            lines.append(
                                f"      - field={item.get('field')}; "
                                f"present_in_count={item.get('present_in_count')}; "
                                f"example_file_count={len(examples)}"
                            )
                columns = [str(x) for x in (table.get("physical_columns_exact") or [])]
                if columns:
                    lines.append("    physical_columns_exact: " + ", ".join(f"`{x}`" for x in columns[:120]))
                    omitted = int(table.get("physical_columns_omitted") or 0)
                    if omitted:
                        lines.append(f"    physical_columns_omitted: {omitted}")
                fields = table.get("field_summaries") if isinstance(table.get("field_summaries"), list) else []
                if fields:
                    lines.append("    key_field_summaries:")
                    for field in fields[:16]:
                        if not isinstance(field, dict):
                            continue
                        parts = [
                            f"name={field.get('name')}",
                            f"meaning={field.get('meaning')}" if field.get("meaning") else "",
                            f"type={field.get('logical_type')}" if field.get("logical_type") else "",
                            f"null_ratio={field.get('null_ratio')}" if field.get("null_ratio") is not None else "",
                            f"unique={field.get('unique_count')}" if field.get("unique_count") is not None else "",
                        ]
                        if field.get("numeric_range"):
                            parts.append(f"numeric_range={field.get('numeric_range')}")
                        if field.get("datetime_range"):
                            parts.append(f"datetime_range={field.get('datetime_range')}")
                        lines.append("      - " + "; ".join(str(x) for x in parts if str(x).strip()))
                warnings = _dedupe_any(table.get("warnings", []), limit=4)
                if warnings:
                    lines.append("    warnings: " + "; ".join(warnings))

    if pack.source_alias_guard:
        lines.extend(["", "## Source Alias Guard"])
        lines.append("- purpose: Business/source-field aliases below appeared in requirements, description, constraints, or output specs, but they must be resolved before pandas access.")
        lines.append("- hard_rule: Never use an alias as `df[alias]`, `groupby(alias)`, `merge(on=alias)`, or `sheet_name=alias` unless `status=exact_physical_column` or `exact_physical_column` is provided.")
        lines.append("- hard_rule: If `status=unresolved_business_concept`, implement a conservative derived rule, mark the constraint unresolved in validation details, or avoid using it for raw filtering; do not invent a same-named column.")
        for item in pack.source_alias_guard[:40]:
            if not isinstance(item, dict):
                continue
            parts = [
                f"alias={item.get('alias')}",
                f"status={item.get('status')}",
                f"exact_physical_column={item.get('exact_physical_column')}" if item.get("exact_physical_column") else "",
                f"candidate_exact_columns={item.get('candidate_exact_columns')}" if item.get("candidate_exact_columns") else "",
                f"source={item.get('source')}" if item.get("source") else "",
                f"rule={item.get('rule')}" if item.get("rule") else "",
            ]
            lines.append("  - " + "; ".join(str(x) for x in parts if str(x).strip()))

    if pack.entity_alias_candidates:
        lines.extend(["", "## Entity Alias Candidates"])
        lines.append("- purpose: Candidate business-entity aliases for data investigation. These are not confirmed equivalent keys.")
        for group in pack.entity_alias_candidates[:12]:
            if not isinstance(group, dict):
                continue
            lines.append(
                f"- concept_id={group.get('concept_id')}; "
                f"label={group.get('label')}; "
                f"status={group.get('status')}; "
                f"alias_families={group.get('alias_families')}; "
                f"value_kinds={group.get('value_kinds')}"
            )
            fields = group.get("candidate_fields") if isinstance(group.get("candidate_fields"), list) else []
            if fields:
                lines.append("  candidate_fields:")
                for item in fields[:24]:
                    if not isinstance(item, dict):
                        continue
                    parts = [
                        f"source_file={item.get('source_file')}",
                        f"sheet_name={item.get('sheet_name')}" if item.get("sheet_name") else "",
                        f"field={item.get('field')}",
                        f"alias_family={item.get('alias_family')}",
                        f"value_kind={item.get('value_kind')}",
                        f"status={item.get('status')}",
                    ]
                    lines.append("    - " + "; ".join(str(x) for x in parts if str(x).strip()))
            checks = group.get("recommended_qdi_checks") if isinstance(group.get("recommended_qdi_checks"), list) else []
            if checks:
                lines.append("  recommended_qdi_checks:")
                lines.extend(f"    - {x}" for x in checks[:8])

    lines.extend(["", "## Minimal Task Reference", f"- Problem paradigm: `{pack.problem_paradigm}`"])
    if pack.task_goal:
        lines.append(f"- Task goal reference: {pack.task_goal}")

    method = pack.method_strategy or {}
    if method:
        lines.extend(["", "## Method Strategy"])
        for key in ["problem_paradigm", "explicit_rl_requested", "rl_as_required_paradigm"]:
            if key in method:
                lines.append(f"- {key}: `{method.get(key)}`")
        families = _dedupe_any(method.get("recommended_solver_families", []), limit=10)
        if families:
            lines.append("- recommended_solver_families: " + ", ".join(f"`{x}`" for x in families))
        for key in ["first_draft_policy", "rl_branch_policy"]:
            value = str(method.get(key, "") or "").strip()
            if value:
                lines.append(f"- {key}: {value}")
        notes = _dedupe_any(method.get("method_routing_notes", []), limit=8)
        if notes:
            lines.append("- method_routing_notes:")
            lines.extend(f"  - {x}" for x in notes)

    lines.extend(["", "## Evaluation Contract Reference"])
    evaluation = pack.evaluation_contract or {}
    final_formula = str(evaluation.get("final_score_formula") or evaluation.get("metric_formula") or "").strip()
    for key in [
        "primary_metric",
        "metric_direction",
        "prediction_unit",
        "computation_scope",
        "aggregation_rule",
        "validation_protocol",
    ]:
        value = str(evaluation.get(key, "") or "").strip()
        if value:
            lines.append(f"- {key}: {value}")
    if final_formula:
        lines.append(f"- final_score_formula: {final_formula}")
    lines.append("- final_validation_score_rule: validation should be comparable by exactly one numeric `Final Validation Score` derived from `final_score_formula`.")
    for key in ["submission_checks", "invalid_solution_rules", "tie_break_rules", "audit_metrics"]:
        values = _dedupe_any(evaluation.get(key, []), limit=12)
        if values:
            lines.append(f"- {key}:")
            lines.extend(f"  - {x}" for x in values)

    lines.extend(["", "## Output Contract Reference"])
    output = pack.output_contract or {}
    for key in ["output_kind", "output_filename", "sample_submission_required", "row_unit", "no_sample_submission_reason"]:
        value = output.get(key)
        if value not in (None, "", []):
            lines.append(f"- {key}: {value}")
    columns = _dedupe_any(output.get("columns", []), limit=60)
    if columns:
        lines.append("- columns: " + ", ".join(f"`{x}`" for x in columns))
        lines.extend(
            [
                "- output_schema_rules:",
                "  - Output/submission columns are generated result fields, not raw input dataframe columns.",
                "  - Do not select these columns from source tables unless an exact physical column with the same name is listed in `physical_columns_exact`.",
                "  - Define a code constant such as `OUTPUT_COLUMNS` from this column order.",
                "  - Build generated result tables with `pd.DataFrame(rows, columns=OUTPUT_COLUMNS)` so zero-row/no-feasible solutions still keep the required schema.",
                "  - Empty, all-unassigned, or no-feasible solutions must be handled by validation/scoring and reported in `Decision Validation Summary`; do not let output CSV construction fail with pandas `KeyError`.",
            ]
        )
    rules = _dedupe_any(output.get("format_rules", []), limit=16)
    if rules:
        lines.append("- format_rules:")
        lines.extend(f"  - {x}" for x in rules)
    sample_spec = output.get("sample_submission_spec") if isinstance(output.get("sample_submission_spec"), dict) else {}
    sample_source_fields = sample_spec.get("source_fields") if isinstance(sample_spec.get("source_fields"), dict) else {}
    if sample_source_fields:
        lines.append("- sample_submission_source_fields:")
        for col, source in list(sample_source_fields.items())[:20]:
            lines.append(f"  - `{col}`: {source}")
    sample_validation_rules = _dedupe_any(sample_spec.get("validation_rules", []), limit=16) if sample_spec else []
    if sample_validation_rules:
        lines.append("- sample_submission_validation_rules:")
        lines.extend(f"  - {x}" for x in sample_validation_rules)
    corrections = output.get("sample_submission_source_field_corrections")
    if isinstance(corrections, list) and corrections:
        lines.append("- source_field_corrections:")
        for item in corrections[:12]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                + "; ".join(
                    str(x)
                    for x in [
                        f"output_column={item.get('output_column')}",
                        f"from_alias={item.get('from')}",
                        f"to_physical_column={item.get('to')}",
                        f"reason={item.get('reason')}",
                    ]
                    if str(x).strip() and not str(x).endswith("=None")
                )
            )

    lines.extend(["", "## Supplemental Data Facts"])
    for item in _dedupe_any(pack.data_orchestration, limit=12):
        lines.append(f"- {item}")
    if pack.data_access:
        lines.append("- This section is intentionally more detailed than `description.md`, but remains bounded for fixed-context use.")
    for entry in pack.data_access[:20]:
        title = entry.get("pattern") or entry.get("table_id") or entry.get("path") or "data file"
        lines.extend(["", f"### {title}"])
        for key in ["kind", "file_count", "file_role", "sheet_name", "shape", "summary", "read_method", "row_grain", "orchestration_note"]:
            value = entry.get(key)
            if value not in (None, "", []):
                lines.append(f"- {key}: {value}")
        columns = _dedupe_any(entry.get("columns", []), limit=80)
        if columns:
            lines.append("- columns: " + ", ".join(f"`{x}`" for x in columns[:40]))
        fields = entry.get("fields") if isinstance(entry.get("fields"), list) else []
        if fields:
            lines.append("- fields:")
            for field in fields[:16]:
                if not isinstance(field, dict):
                    continue
                parts = [
                    f"name={field.get('name')}",
                    f"meaning={field.get('meaning')}" if field.get("meaning") else "",
                    f"role={field.get('role')}" if field.get("role") else "",
                    f"type={field.get('logical_type')}" if field.get("logical_type") else "",
                    f"row_count={field.get('row_count')}" if field.get("row_count") is not None else "",
                    f"non_null={field.get('non_null_count')}" if field.get("non_null_count") is not None else "",
                    f"null_ratio={field.get('null_ratio')}" if field.get("null_ratio") is not None else "",
                    f"unique={field.get('unique_count')}" if field.get("unique_count") is not None else "",
                ]
                numeric = field.get("numeric_stats") if isinstance(field.get("numeric_stats"), dict) else {}
                datetime_stats = field.get("datetime_stats") if isinstance(field.get("datetime_stats"), dict) else {}
                if numeric:
                    parts.append(f"numeric_stats={numeric}")
                if datetime_stats:
                    parts.append(f"datetime_stats={datetime_stats}")
                if field.get("top_values"):
                    parts.append(f"top_values={field.get('top_values')}")
                lines.append("  - " + "; ".join(str(x) for x in parts if str(x).strip()))
        source_metadata = entry.get("source_metadata") if isinstance(entry.get("source_metadata"), dict) else {}
        if source_metadata:
            shape = source_metadata.get("shape")
            if shape not in (None, "", []):
                lines.append(f"- shape: {shape}")
        for key in ["key_fields", "relation_keys", "important_fields", "parsing_notes"]:
            values = _dedupe_any(entry.get(key, []), limit=16)
            if values:
                lines.append(f"- {key}: " + ", ".join(f"`{x}`" for x in values))
        warnings = _dedupe_any(entry.get("warnings", []), limit=8)
        if warnings:
            lines.append("- warnings: " + "; ".join(str(x) for x in warnings))
        metadata = source_metadata
        sampling = metadata.get("profile_sampling") if isinstance(metadata.get("profile_sampling"), dict) else {}
        if sampling:
            lines.append(
                "- profile_sampling: "
                f"rows_read={sampling.get('rows_read')}, "
                f"max_rows={sampling.get('configured_max_rows')}, "
                f"reason={sampling.get('sampling_reason')}, "
                f"large_file={sampling.get('is_large_file')}"
            )
        sheet_names = metadata.get("excel_sheet_names") if isinstance(metadata.get("excel_sheet_names"), list) else []
        if sheet_names:
            lines.append("- excel_sheets: " + ", ".join(f"`{x}`" for x in sheet_names[:12]))
        sheet_groups = metadata.get("excel_sheet_groups") if isinstance(metadata.get("excel_sheet_groups"), list) else []
        if sheet_groups:
            lines.append("- excel_sheet_groups:")
            for group in sheet_groups[:8]:
                if not isinstance(group, dict):
                    continue
                sheets = [str(x) for x in (group.get("sheets") or [])[:8]]
                lines.append(
                    f"  - group={group.get('group_id')}; "
                    f"pattern={group.get('sheet_name_pattern')}; "
                    f"count={group.get('sheet_count')}; "
                    f"representative={group.get('representative')}; "
                    f"sheets={sheets}"
                )
        field_profiles = entry.get("field_profiles") if isinstance(entry.get("field_profiles"), list) else []
        if field_profiles:
            lines.append("- field_profiles:")
            for profile in field_profiles[:16]:
                if not isinstance(profile, dict):
                    continue
                parts = [
                    f"name={profile.get('name')}",
                    f"type={profile.get('logical_type')}",
                    f"null_ratio={profile.get('null_ratio')}",
                    f"unique={profile.get('unique_count')}",
                ]
                if profile.get("numeric_range"):
                    parts.append(f"numeric_range={profile.get('numeric_range')}")
                if profile.get("datetime_range"):
                    parts.append(f"datetime_range={profile.get('datetime_range')}")
                if profile.get("top_values"):
                    parts.append(f"top_values={profile.get('top_values')}")
                if profile.get("group_profile"):
                    parts.append(f"group_profile={profile.get('group_profile')}")
                lines.append("  - " + "; ".join(str(x) for x in parts if str(x).strip()))
        sheet_profiles = entry.get("excel_sheet_profiles") if isinstance(entry.get("excel_sheet_profiles"), list) else []
        if sheet_profiles:
            lines.append("- excel_sheet_profiles:")
            for sheet in sheet_profiles[:12]:
                if not isinstance(sheet, dict):
                    continue
                cols = [str(x) for x in (sheet.get("columns") or [])[:12]]
                lines.append(
                    f"  - sheet={sheet.get('sheet_name')}; "
                    f"shape={sheet.get('shape')}; "
                    f"shape_profiled={sheet.get('shape_profiled')}; "
                    f"source_file_count={sheet.get('source_file_count')}; "
                    f"deep_profiled={sheet.get('is_deep_profiled')}; "
                    f"group={sheet.get('sheet_group_id')}; "
                    f"representative={sheet.get('sheet_group_representative')}; "
                    f"layout={sheet.get('layout_kind')}; "
                    f"detected_header_row={sheet.get('detected_header_row')}; "
                    f"read_strategy={sheet.get('read_strategy_kind')}; "
                    f"columns={cols}; "
                    f"read={sheet.get('read_example')}"
                )
                risks = [str(x) for x in (sheet.get("reading_risks") or [])[:4]]
                if risks:
                    lines.append("    reading_risks: " + " | ".join(risks))
                sheet_desc = sheet.get("field_descriptions") if isinstance(sheet.get("field_descriptions"), dict) else {}
                if sheet_desc:
                    lines.append("    field_descriptions:")
                    for name, meaning in list(sheet_desc.items())[:12]:
                        lines.append(f"      - `{name}`: {meaning}")
                sheet_profiles = sheet.get("column_profiles") if isinstance(sheet.get("column_profiles"), list) else []
                if sheet_profiles:
                    lines.append("    column_profiles:")
                    for profile in sheet_profiles[:10]:
                        if not isinstance(profile, dict):
                            continue
                        parts = [f"name={profile.get('name')}"]
                        if profile.get("meaning"):
                            parts.append(f"meaning={profile.get('meaning')}")
                        if profile.get("logical_type"):
                            parts.append(f"type={profile.get('logical_type')}")
                        if profile.get("null_ratio") is not None:
                            parts.append(f"null_ratio={profile.get('null_ratio')}")
                        if profile.get("unique_count") is not None:
                            parts.append(f"unique={profile.get('unique_count')}")
                        if profile.get("numeric_range"):
                            parts.append(f"numeric_range={profile.get('numeric_range')}")
                        if profile.get("top_values"):
                            parts.append(f"top_values={profile.get('top_values')}")
                        lines.append("      - " + "; ".join(str(x) for x in parts if str(x).strip()))
        read_example = str(entry.get("read_example", "") or "").strip()
        if read_example:
            if "\n" in read_example or read_example.startswith("```"):
                lines.append("- read_example:")
                lines.append(read_example)
            else:
                lines.append("- read_example:")
                lines.append("```python")
                lines.append(f"df = {read_example}")
                lines.append("```")

    if pack.relation_cards:
        lines.extend(["", "## Relation Cards"])
        for rel in pack.relation_cards[:40]:
            if not isinstance(rel, dict):
                continue
            parts = [
                f"{rel.get('left_file')}.{rel.get('left_field')}",
                f"-> {rel.get('right_file')}.{rel.get('right_field')}",
                f"type={rel.get('relation_type')}",
                f"confidence={rel.get('confidence')}",
                f"evidence={rel.get('short_evidence')}",
            ]
            lines.append("- " + "; ".join(str(x) for x in parts if str(x).strip()))

    if pack.filename_sample_groups:
        lines.extend(["", "## Filename Sample Groups"])
        for group in pack.filename_sample_groups[:30]:
            if not isinstance(group, dict):
                continue
            lines.append(
                "- "
                + "; ".join(
                    str(x)
                    for x in [
                        f"template={group.get('template_path_or_sample_id') or group.get('template_path')}",
                        f"count={group.get('file_count') or group.get('count')}",
                        f"role={group.get('role')}",
                        f"representatives={group.get('representative_files')}",
                        f"shared_fields={group.get('shared_fields')}",
                        f"variant_fields_by_file={group.get('variant_fields_by_file')}",
                        f"field_presence={group.get('field_presence')}",
                        f"evidence={group.get('short_evidence')}",
                    ]
                    if str(x).strip() and not str(x).endswith("=None")
                )
            )

    if pack.modeling_boundary:
        lines.extend(["", "## Problem Boundary Reference"])
        lines.extend(f"- {x}" for x in _dedupe_any(pack.modeling_boundary, limit=30))
    if pack.constraints:
        lines.extend(["", "## Constraints Reference"])
        lines.extend(f"- {x}" for x in _dedupe_any(pack.constraints, limit=30))
    if pack.leakage_guards:
        lines.extend(["", "## Leakage Guards"])
        lines.extend(f"- {x}" for x in _dedupe_any(pack.leakage_guards, limit=24))
    if pack.pitfalls:
        lines.extend(["", "## Pitfalls"])
        lines.extend(f"- {x}" for x in _dedupe_any(pack.pitfalls, limit=24))
    return "\n".join(lines).strip() + "\n"


def _compact_field_profile(profile: dict) -> str:
    parts = [
        f"类型={_logical_type_label(profile.get('logical_type', 'unknown'))}",
        f"缺失={_fmt_profile_value(profile.get('null_count'))}/{_fmt_profile_value(profile.get('row_count'))}",
    ]
    ns = profile.get("numeric_stats") or {}
    if ns:
        parts.append(f"范围={_fmt_profile_value(ns.get('min'))}~{_fmt_profile_value(ns.get('max'))}")
    ds = profile.get("datetime_stats") or {}
    if ds:
        parts.append(f"时间范围={_fmt_profile_value(ds.get('min'))}~{_fmt_profile_value(ds.get('max'))}")
    top_values = profile.get("top_values") or []
    if top_values and not ns:
        parts.append("高频值=" + "，".join(str(x) for x in top_values[:5]))
    return "；".join(parts)


def _file_summary_group_signature(fs: FileSummary) -> tuple | None:
    path = Path(str(fs.path or ""))
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}:
        return None
    if not fs.columns:
        return None
    parent = str(path.parent).replace("\\", "/")
    return (parent, suffix, str(fs.role), tuple(str(x) for x in fs.columns))


def _file_summary_pattern_signature(fs: FileSummary) -> tuple | None:
    sig = _filename_pattern_signature(str(fs.path or ""))
    if sig is None:
        return None
    parent, suffix, pattern = sig
    return (parent, suffix, pattern, str(fs.role))


def _group_file_summaries(file_summaries: list[FileSummary]) -> list[tuple[bool, list[FileSummary]]]:
    pattern_grouped: dict[tuple, list[FileSummary]] = {}
    for fs in file_summaries:
        sig = _file_summary_pattern_signature(fs)
        if sig is not None:
            pattern_grouped.setdefault(sig, []).append(fs)

    grouped: dict[tuple, list[FileSummary]] = {}
    for fs in file_summaries:
        sig = _file_summary_group_signature(fs)
        if sig is not None:
            grouped.setdefault(sig, []).append(fs)
    emitted: set[int] = set()
    emitted_pattern_sigs: set[tuple] = set()
    blocks: list[tuple[bool, list[FileSummary]]] = []
    for fs in file_summaries:
        fs_id = id(fs)
        if fs_id in emitted:
            continue
        pattern_sig = _file_summary_pattern_signature(fs)
        if pattern_sig is not None and pattern_sig in emitted_pattern_sigs:
            emitted.add(fs_id)
            continue
        pattern_group = pattern_grouped.get(pattern_sig, []) if pattern_sig is not None else []
        if len(pattern_group) >= 3:
            blocks.append((True, pattern_group))
            emitted.update(id(x) for x in pattern_group)
            if pattern_sig is not None:
                emitted_pattern_sigs.add(pattern_sig)
            continue
        sig = _file_summary_group_signature(fs)
        group = grouped.get(sig, []) if sig is not None else []
        if len(group) >= 3:
            filtered = [
                x
                for x in group
                if (ps := _file_summary_pattern_signature(x)) is None or ps not in emitted_pattern_sigs
            ]
            if len(filtered) >= 3:
                blocks.append((True, filtered))
                emitted.update(id(x) for x in filtered)
            else:
                blocks.append((False, [fs]))
                emitted.add(fs_id)
        else:
            blocks.append((False, [fs]))
            emitted.add(fs_id)
    return blocks


def _render_one_field_line(
    fs: FileSummary,
    col: str,
    profiles: dict[str, dict],
    *,
    meaning_override: str = "",
    profile_override: str = "",
) -> str:
    meaning = str(meaning_override or "").strip() or _llm_field_description(fs, col)
    if meaning:
        return f"- `{col}`：{meaning}"
    generic = _generic_field_description(col)
    if generic:
        return f"- `{col}`：{generic}"
    profile = profiles.get(col) if isinstance(profiles, dict) else None
    logical_type = str((profile or {}).get("logical_type", "") or "").lower() if isinstance(profile, dict) else ""
    if logical_type in {"integer", "float", "numeric"}:
        return f"- `{col}`：数值型原始字段，可作为特征、约束计算或评估校验输入。"
    if logical_type in {"datetime", "date"}:
        return f"- `{col}`：时间型原始字段，用于时间切分、时序特征或时效约束。"
    if logical_type in {"categorical", "boolean", "text", "string"}:
        return f"- `{col}`：原始类别或文本字段，用于实体区分、分组、关联或约束校验。"
    return f"- `{col}`：原始数据字段，保留用于建模、关联或约束校验。"


def _merge_group_profiles(group: list[FileSummary], col: str) -> str:
    profiles = [_profile_map(fs).get(col) for fs in group]
    profiles = [p for p in profiles if isinstance(p, dict)]
    if not profiles:
        return ""
    first = profiles[0]
    logical_types = _dedupe_nonempty([str(p.get("logical_type", "")) for p in profiles], limit=3)
    row_total = 0
    null_total = 0
    has_row_counts = False
    numeric_mins: list[float] = []
    numeric_maxs: list[float] = []
    datetime_mins: list[str] = []
    datetime_maxs: list[str] = []
    for profile in profiles:
        try:
            row_total += int(float(profile.get("row_count", 0) or 0))
            null_total += int(float(profile.get("null_count", 0) or 0))
            has_row_counts = True
        except Exception:
            pass
        ns = profile.get("numeric_stats") or {}
        try:
            if ns.get("min") is not None:
                numeric_mins.append(float(ns.get("min")))
            if ns.get("max") is not None:
                numeric_maxs.append(float(ns.get("max")))
        except Exception:
            pass
        ds = profile.get("datetime_stats") or {}
        if ds.get("min") is not None:
            datetime_mins.append(str(ds.get("min")))
        if ds.get("max") is not None:
            datetime_maxs.append(str(ds.get("max")))

    parts: list[str] = []
    if logical_types:
        parts.append("类型=" + "/".join(logical_types))
    if has_row_counts:
        parts.append(f"组内缺失合计={null_total}/{row_total}")
    insight = _group_column_role_insight(group, col)
    if insight.get("profile_note"):
        parts.append(str(insight["profile_note"]))
    if numeric_mins and numeric_maxs:
        parts.append(f"组内范围={_fmt_profile_value(min(numeric_mins))}~{_fmt_profile_value(max(numeric_maxs))}")
    elif datetime_mins and datetime_maxs:
        parts.append(f"组内时间范围={min(datetime_mins)}~{max(datetime_maxs)}")
    elif not has_row_counts:
        return _compact_field_profile(first)
    return "；".join(parts)


def _merge_group_field_description(group: list[FileSummary], col: str) -> str:
    insight = _group_column_role_insight(group, col)
    if insight.get("meaning"):
        return str(insight["meaning"])
    descriptions = [
        _sanitize_group_field_description(_llm_field_description(fs, col), col)
        for fs in group
    ]
    descriptions = _dedupe_nonempty(descriptions, limit=4)
    generic = _generic_field_description(col)
    if not descriptions:
        return generic
    if len(descriptions) == 1:
        return descriptions[0]
    if generic:
        return generic
    shortest = min(descriptions, key=len)
    if all(shortest in item or item in shortest for item in descriptions):
        return shortest
    return "；".join(descriptions[:2])


def _render_one_group_field_line(group: list[FileSummary], col: str) -> str:
    fs = group[0]
    return _render_one_field_line(
        fs,
        col,
        _profile_map(fs),
        meaning_override=_merge_group_field_description(group, col),
        profile_override="",
    )


def _excel_sheet_profiles(fs: FileSummary) -> list[dict[str, Any]]:
    meta = fs.source_metadata or {}
    sheets = meta.get("excel_sheet_profiles") if isinstance(meta.get("excel_sheet_profiles"), list) else []
    return [x for x in sheets if isinstance(x, dict)]


def _find_excel_sheet_profile(fs: FileSummary, sheet_name: str) -> dict[str, Any] | None:
    wanted = str(sheet_name or "").strip()
    if not wanted:
        return None
    for sheet in _excel_sheet_profiles(fs):
        if str(sheet.get("sheet_name", "") or "").strip() == wanted:
            return sheet
    return None


def _has_multi_sheet_profiles(fs: FileSummary) -> bool:
    return len(_excel_sheet_profiles(fs)) > 1


def _render_sheet_field_lines(fs: FileSummary, sheet: dict[str, Any], *, max_cols: int = 40) -> list[str]:
    sheet_name = str(sheet.get("sheet_name", "") or "")
    columns = [str(x) for x in (sheet.get("columns") or []) if str(x).strip()]
    descriptions = _sheet_field_descriptions(fs, sheet_name)
    profiles = {
        str(p.get("name", "")): p
        for p in (sheet.get("column_profiles") or [])
        if isinstance(p, dict) and str(p.get("name", "")).strip()
    }
    names = _dedupe_any(columns + list(descriptions.keys()) + list(profiles.keys()), limit=max_cols)
    lines: list[str] = []
    if not names and sheet.get("raw_preview"):
        lines.append("- 该 sheet 更像说明/规则页；请优先读取 raw_preview 或原始 sheet 文本提取规则。")
        return lines
    for name in names:
        meaning = descriptions.get(name) or _llm_field_description(fs, name) or _generic_field_description(name)
        profile = profiles.get(name)
        if profile:
            schema = _compact_field_profile(profile)
            lines.append(f"- `{name}`：{meaning}（{schema}）")
        else:
            lines.append(f"- `{name}`：{meaning}")
    return lines


def _sheet_signature(sheet: dict[str, Any]) -> tuple:
    return (
        str(sheet.get("sheet_name", "") or "").lower(),
        str(sheet.get("sheet_group_id", "") or ""),
    )


def _merged_group_sheet_profiles(group: list[FileSummary], *, max_sheets: int = 12) -> list[tuple[str, dict[str, Any], list[FileSummary]]]:
    buckets: dict[tuple, dict[str, Any]] = {}
    for fs in group:
        for sheet in _excel_sheet_profiles(fs):
            key = _sheet_signature(sheet)
            title = str(sheet.get("sheet_name", "") or "sheet")
            if key not in buckets:
                buckets[key] = {"title": title, "sheet": dict(sheet), "members": [fs]}
            else:
                buckets[key]["members"].append(fs)
                merged = buckets[key]["sheet"]
                merged["columns"] = _dedupe_any(
                    list(merged.get("columns") or []) + list(sheet.get("columns") or []),
                    limit=120,
                )
                merged_profiles = {
                    str(p.get("name", "")): p
                    for p in (merged.get("column_profiles") or [])
                    if isinstance(p, dict) and str(p.get("name", "")).strip()
                }
                for profile in sheet.get("column_profiles") or []:
                    if not isinstance(profile, dict):
                        continue
                    name = str(profile.get("name", "") or "")
                    if name and name not in merged_profiles:
                        merged_profiles[name] = profile
                merged["column_profiles"] = list(merged_profiles.values())
                merged["source_file_count"] = len(buckets[key]["members"])
    out: list[tuple[str, dict[str, Any], list[FileSummary]]] = []
    for bucket in list(buckets.values())[:max_sheets]:
        out.append((str(bucket["title"]), bucket["sheet"], list(bucket["members"])))
    return out


def _render_group_sheet_field_lines(group: list[FileSummary], title: str, sheet: dict[str, Any], members: list[FileSummary], *, max_cols: int = 40) -> list[str]:
    columns = [str(x) for x in (sheet.get("columns") or []) if str(x).strip()]
    descriptions: dict[str, list[str]] = {}
    for fs in members:
        for name, meaning in _sheet_field_descriptions(fs, title).items():
            descriptions.setdefault(name, []).append(meaning)
    sheet_profiles = {
        str(p.get("name", "")): p
        for p in (sheet.get("column_profiles") or [])
        if isinstance(p, dict) and str(p.get("name", "")).strip()
    }
    names = _dedupe_any(columns + list(descriptions.keys()) + list(sheet_profiles.keys()), limit=max_cols)
    lines: list[str] = []
    if not names and sheet.get("raw_preview"):
        lines.append("- 该 sheet 更像说明/规则页；组内各 workbook 应读取该 sheet 提取计费、约束或字段口径说明。")
        return lines
    for name in names:
        merged_desc = _dedupe_nonempty(descriptions.get(name, []), limit=2)
        meaning = "；".join(merged_desc) if merged_desc else _merge_group_field_description(group, name)
        if not meaning:
            meaning = _generic_field_description(name)
        profile = sheet_profiles.get(name)
        if profile:
            lines.append(f"- `{name}`：{meaning}（{_compact_field_profile(profile)}）")
        else:
            lines.append(f"- `{name}`：{meaning}")
    return lines


def _merged_group_columns(group: list[FileSummary], *, limit: int = 80) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for fs in group:
        for col in fs.columns:
            s = str(col).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= limit:
                return out
    return out


def _render_field_section(file_summaries: list[FileSummary], protocol: DataAccessProtocol) -> str:
    lines = ["## 字段说明"]
    protocol_paths = {item.path for item in protocol.files}
    rendered = False
    candidates = [
        fs
        for fs in file_summaries
        if not _is_document_like(fs)
        and fs.columns
        and (not protocol_paths or fs.path in protocol_paths)
    ]
    for is_group, group in _group_file_summaries(candidates):
        fs = group[0]
        if _is_document_like(fs) or not fs.columns:
            continue
        if protocol_paths and fs.path not in protocol_paths:
            continue
        rendered = True
        lines.append("")
        if is_group:
            paths = [str(x.path) for x in group]
            pattern = _common_path_pattern(paths)
            lines.append(f"### {pattern}")
        else:
            lines.append(f"### {fs.path}")
        if is_group and any(_has_multi_sheet_profiles(x) for x in group):
            for sheet_title, sheet, members in _merged_group_sheet_profiles(group):
                shape = sheet.get("shape")
                source_count = sheet.get("source_file_count") or len(members)
                lines.append("")
                lines.append(f"#### sheet: {sheet_title}")
                read_hint = str(sheet.get("recommended_read", "") or sheet.get("read_example", "") or "").strip()
                layout = str(sheet.get("layout_kind", "") or "standard_table")
                lines.append(
                    f"- sheet用途：来自该重复 workbook 组的 `{sheet_title}` sheet，覆盖 {source_count} 个文件；"
                    f"shape 示例={shape}；layout={layout}"
                    + (f"；建议读取：`{read_hint}`。" if read_hint else "。")
                )
                lines.extend(_render_group_sheet_field_lines(group, sheet_title, sheet, members))
            continue
        if (not is_group) and _has_multi_sheet_profiles(fs):
            for sheet in _excel_sheet_profiles(fs):
                sheet_title = str(sheet.get("sheet_name", "") or "sheet")
                lines.append("")
                lines.append(f"#### sheet: {sheet_title}")
                read_hint = str(sheet.get("recommended_read", "") or "").strip()
                layout = str(sheet.get("layout_kind", "") or "standard_table")
                lines.append(
                    f"- sheet用途：`{fs.path}` 中的 `{sheet_title}` sheet；shape={sheet.get('shape')}；layout={layout}"
                    + (f"；建议读取：`{read_hint}`。" if read_hint else "。")
                )
                lines.extend(_render_sheet_field_lines(fs, sheet))
            continue
        profiles = _profile_map(fs)
        columns = _merged_group_columns(group, limit=80) if is_group else [str(x) for x in fs.columns[:80]]
        for col in columns:
            if is_group:
                lines.append(_render_one_group_field_line(group, col))
            else:
                lines.append(_render_one_field_line(fs, col, profiles))
    if not rendered:
        lines.append("- 暂未识别结构化字段；请参考原始任务文档中的数据说明。")
    return "\n".join(lines)


def _render_task_definition_from_bundle(bundle: DescriptionProtocolBundle, downstream_context: dict | None = None) -> str:
    ctx = downstream_context or {}
    paradigm = str(bundle.problem_paradigm or ctx.get("problem_paradigm", "unknown_but_executable"))
    lines = ["## 任务定义"]
    if paradigm == "ml_dl_prediction":
        p = bundle.ml_dl
        lines.extend(
            [
                f"- 问题范式：{_paradigm_label(paradigm)}。",
                f"- 训练数据：{p.train_data or ctx.get('train_table', '训练数据')}。",
                f"- 预测/测试数据：{p.predict_data or ctx.get('predict_table', '未提供独立预测文件，由训练数据切分验证')}。",
                f"- 预测单元：{p.prediction_unit or '数据表中的一行样本'}。",
                f"- 目标字段：`{p.target or ctx.get('target_column', '目标字段由任务说明确定')}`。",
            ]
        )
        if p.feature_boundary:
            lines.append("- 可用特征边界：")
            lines.extend(f"  - {x}" for x in p.feature_boundary)
        if p.validation_design:
            lines.append(f"- 验证设计：{p.validation_design}")
        if p.leakage_guards:
            lines.append("- 防泄漏要求：")
            lines.extend(f"  - {x}" for x in p.leakage_guards)
        return "\n".join(lines)
    if paradigm == "static_optimization":
        p = bundle.optimization
        lines.extend(
            [
                f"- 问题范式：{_paradigm_label(paradigm)}。",
                f"- 输入实例：{p.input_instance or '由订单、资源、成本、约束等数据表共同构成一个待求解实例'}。",
                f"- 目标函数：{p.objective or '在满足约束的前提下最小化成本或最大化收益'}。",
                f"- 方案表示：{p.solution_representation or '输出一套覆盖全部待决策对象的可执行方案'}。",
            ]
        )
        if p.decision_variables:
            lines.append("- 决策变量：")
            lines.extend(f"  - {x}" for x in p.decision_variables)
        if p.hard_constraints:
            lines.append("- 硬约束：")
            lines.extend(f"  - {x}" for x in p.hard_constraints)
        if p.soft_constraints:
            lines.append("- 软约束/惩罚项：")
            lines.extend(f"  - {x}" for x in p.soft_constraints)
        if p.feasibility_checks:
            lines.append("- 可行性检查：")
            lines.extend(f"  - {x}" for x in p.feasibility_checks)
        return "\n".join(lines)
    if paradigm == "reinforcement_learning":
        p = bundle.rl
        lines.extend(
            [
                f"- 问题范式：{_paradigm_label(paradigm)}。",
                f"- 环境：{p.environment or '由任务数据或仿真器定义的序贯决策环境'}。",
                f"- 状态：{p.state or '每个决策步可观测的信息'}。",
                f"- 动作：{p.action or '当前状态下允许选择的合法动作'}。",
                f"- 状态转移：{p.transition or '执行动作后环境更新到下一状态'}。",
                f"- 奖励：{p.reward or '与任务目标一致的即时收益或成本惩罚'}。",
                f"- 终止条件：{p.terminal_condition or '任务完成、达到步数上限或进入不可行状态'}。",
                f"- 策略输出：{p.policy_output or '策略根据当前状态输出动作或动作分数'}。",
                f"- 评估回合：{p.evaluation_episodes or '在固定回放集或固定环境 episode 上执行策略并汇总得分'}。",
            ]
        )
        if p.illegal_action_handling:
            lines.append("- 非法动作处理：")
            lines.extend(f"  - {x}" for x in p.illegal_action_handling)
        return "\n".join(lines)
    if paradigm == "hybrid_ml_optimization":
        p = bundle.hybrid
        lines.extend(
            [
                f"- 问题范式：{_paradigm_label(paradigm)}。",
                f"- 预测子问题：{p.prediction_subproblem or '先从历史数据估计需求、风险、成本或收益等中间量'}。",
                f"- 决策子问题：{p.decision_subproblem or '再基于预测结果生成最终分配、调度、排序或组合方案'}。",
                f"- 预测到决策的衔接：{p.handoff or '预测结果只能作为决策输入，不得覆盖硬约束'}。",
                f"- 最终目标：{p.final_objective or '以最终方案的目标函数或官方得分为准'}。",
                f"- 验证设计：{p.validation_design or '同时验证预测子任务质量和最终方案质量，最终排名以方案评估为准'}。",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            f"- 问题范式：{_paradigm_label(paradigm)}。",
            "- 下游系统必须先根据数据和任务说明固定输入、输出、评估和约束，再进行建模或搜索。",
        ]
    )
    return "\n".join(lines)


def _render_output_protocol(bundle: DescriptionProtocolBundle, downstream_context: dict | None = None) -> str:
    ctx = downstream_context or {}
    output = bundle.output
    cols = [str(x) for x in (output.columns or ctx.get("submission_columns", []) or ctx.get("generated_submission_columns", [])) if str(x).strip()]
    lines = ["## 输出或提交格式"]
    lines.append(f"- 输出类型：{output.output_kind or 'submission_table'}。")
    lines.append(f"- 输出文件名：`{output.output_filename or ctx.get('submission_output_filename', 'submission.csv')}`。")
    if cols:
        lines.append("- 列顺序：" + "，".join(f"`{x}`" for x in cols) + "。")
    if output.row_unit:
        lines.append(f"- 行粒度：{output.row_unit}")
    if output.sample_submission_required:
        sample_name = str(ctx.get("submission_sample_filename", "sample_submission.csv") or "sample_submission.csv")
        lines.append(f"- 必须遵循 `{sample_name}` 的列名、列顺序和行数规则。")
    elif output.no_sample_submission_reason:
        lines.append(f"- 本任务不强制生成 `sample_submission.csv`：{output.no_sample_submission_reason}")
    if not cols and not output.sample_submission_required:
        lines.append("- 没有权威提交样例时，不得把任务硬套成固定 `id,target` 两列表。")
    if output.format_rules:
        lines.append("- 格式与内容规则：")
        lines.extend(f"  - {x}" for x in output.format_rules)
    return "\n".join(lines)


def _render_constraints_section(bundle: DescriptionProtocolBundle, downstream_context: dict | None = None) -> str:
    ctx = downstream_context or {}
    constraint_memory = ctx.get("constraint_memory", {}) if isinstance(ctx, dict) else {}
    memory_items = []
    if isinstance(constraint_memory, dict):
        for item in constraint_memory.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description", "") or "").strip()
            name = str(item.get("name", "") or "").strip()
            if desc:
                memory_items.append(f"{name}: {desc}" if name else desc)
    values = [str(x).strip() for x in (bundle.constraints or []) if str(x).strip()]
    values.extend(memory_items[:12])
    warnings = [str(x).strip() for x in (bundle.warnings or []) if str(x).strip()]
    lines = ["## 关键约束与注意事项"]
    if values:
        lines.append("### 关键约束")
        lines.extend(f"- {x}" for x in values[:30])
    else:
        lines.append("- 未发现额外硬约束；仍需严格遵守任务定义、读取方式、输出格式和评估协议。")
    if warnings:
        lines.append("### 注意事项")
        lines.extend(f"- {x}" for x in warnings[:20])
    return "\n".join(lines)


def _render_leakage_section(
    bundle: DescriptionProtocolBundle,
    evaluation_contract: EvaluationContractReview | dict[str, Any] | None = None,
) -> str:
    guards: list[str] = []
    if evaluation_contract is not None:
        try:
            contract = _as_evaluation_contract(evaluation_contract)
            guards.extend(_nonempty_list(contract.submission_checks))
            guards.extend(_nonempty_list(contract.leakage_guards))
            guards.extend(_nonempty_list(contract.invalid_solution_rules))
        except Exception:
            pass
    guards.extend(_nonempty_list(bundle.ml_dl.leakage_guards))
    guards.extend([x for x in _nonempty_list(bundle.warnings) if any(k in x for k in ["泄漏", "未来", "作弊", "评估", "标签", "约束"])])
    guards = _dedupe_any(guards, limit=12)
    lines = ["## 防作弊与防泄漏"]
    if guards:
        lines.extend(f"- {x}" for x in guards)
    else:
        lines.extend(
            [
                "- 不得使用评估集真实标签、未来信息、评估反馈或人工查看评分结果来反向修改训练特征、策略或优化规则。",
                "- 不得通过缺行、重复行、非法默认值、NaN/Inf、伪造低成本或绕过约束校验来获得虚假分数。",
            ]
        )
    return "\n".join(lines)


def _render_pitfalls_section(bundle: DescriptionProtocolBundle) -> str:
    paradigm = str(bundle.problem_paradigm or "").strip()
    values = _nonempty_list(bundle.warnings)
    generic = [
        "不要发明原始材料中没有的提交列、目标字段、行数规则、随机种子、距离矩阵或成本公式。",
        "不要把文件名相似的多份表当成互不相关的单表，也不要只读一个样例文件代表整个重复文件组。",
    ]
    if paradigm == "ml_dl_prediction":
        generic.extend(
            [
                "不要把测试集或预测集标签当作特征；训练、验证和提交边界必须固定。",
                "不要在没有官方样例时硬套 `id,target` 两列表。",
            ]
        )
    elif paradigm == "static_optimization":
        generic.extend(
            [
                "不要把不可行方案用低成本掩盖；硬约束违规必须按评估协议处理。",
                "不要把优化任务硬改成普通监督预测任务。",
            ]
        )
    elif paradigm == "reinforcement_learning":
        generic.extend(
            [
                "不要只写贪心或局部搜索却声称已经训练强化学习策略；必须说明环境、动作、奖励和评估回合。",
                "不要让策略在评估时访问未来订单、真实最优动作或评估反馈。",
            ]
        )
    elif paradigm == "hybrid_ml_optimization":
        generic.extend(
            [
                "预测子任务只能为最终决策提供输入，不能覆盖硬约束或最终方案评估。",
                "最终排名以方案评估的单一数值主指标为准，不以中间预测指标替代。",
            ]
        )
    pitfalls = _dedupe_any(values + generic, limit=12)
    lines = ["## 最关键坑点"]
    lines.extend(f"- {x}" for x in pitfalls)
    return "\n".join(lines)


def render_description_protocol_markdown(
    bundle: DescriptionProtocolBundle,
    file_summaries: list[FileSummary],
    downstream_context: dict | None = None,
    evaluation_contract: EvaluationContractReview | dict[str, Any] | None = None,
) -> str:
    """Render the final Kaggle-style description from structured protocols."""
    ctx = downstream_context or {}
    if not bundle.data_access.files:
        bundle.data_access = build_data_access_protocol(file_summaries)
    paradigm = str(bundle.problem_paradigm or ctx.get("problem_paradigm", "unknown_but_executable"))
    overview = str(bundle.overview or bundle.task_goal or ctx.get("task_hint", "") or "本任务要求根据给定数据构建可执行方案。").strip()
    lines = [
        "# 赛题说明",
        "",
        "## 任务概述",
        f"- 任务目标：{overview}",
        f"- 问题范式：{_paradigm_label(paradigm)}。",
        "",
        "## 评估协议",
        "### 主指标",
        f"- {bundle.evaluation_summary or '以任务定义中的唯一主指标进行排序。'}",
        "### 计算公式",
        "- 由后续评估协议合同固化为唯一可计算公式。",
        "### 计算范围",
        "- 覆盖评估协议指定的全部样本、方案记录或 episode。",
        "### 验证协议",
        "- 使用固定验证协议比较所有候选方案；不得在调参过程中临时改变指标或切分口径。",
        "### 结果报告要求",
        "- 报告唯一主指标、样本/方案/episode 数量、非法输出数量和约束违约数量。",
        "",
        _render_output_protocol(bundle, ctx),
        "",
        _render_task_definition_from_bundle(bundle, ctx),
        "",
        _render_constraints_section(bundle, ctx),
        "",
        _render_leakage_section(bundle, evaluation_contract),
        "",
        _render_pitfalls_section(bundle),
        "",
        _render_field_section(file_summaries, bundle.data_access),
        "",
        _render_data_access_section(bundle.data_access),
        "",
    ]
    desc = "\n".join(lines).strip() + "\n"
    if evaluation_contract is not None:
        desc = apply_evaluation_contract(desc, evaluation_contract, ctx)
    return desc


def description_protocol_bundle_defects(
    bundle: DescriptionProtocolBundle | dict[str, Any],
    downstream_context: dict | None = None,
) -> list[str]:
    """Validate the structured description source before rendering markdown.

    Markdown section checks are too late and too weak for ML/DL/RL routing:
    the renderer can fill generic fallback prose even when the LLM left the
    actual task protocol empty. This validator keeps the structured bundle
    honest before the final human-facing description is rendered.
    """
    if not isinstance(bundle, DescriptionProtocolBundle):
        bundle = DescriptionProtocolBundle.model_validate(bundle)
    ctx = downstream_context or {}
    defects: list[str] = []
    paradigm = str(bundle.problem_paradigm or ctx.get("problem_paradigm", "")).strip()
    allowed = {
        "ml_dl_prediction",
        "static_optimization",
        "reinforcement_learning",
        "hybrid_ml_optimization",
        "unknown_but_executable",
    }
    if paradigm not in allowed:
        defects.append(f"description_protocol invalid problem_paradigm: {paradigm}")

    files = list(bundle.data_access.files or [])
    if not files:
        defects.append("description_protocol missing data_access.files")
    for item in files:
        if not str(item.path or "").strip():
            defects.append("description_protocol data_access file missing path")
        if not str(item.read_example or "").strip():
            defects.append(f"description_protocol data_access file missing read_example: {item.path}")
        if not str(item.read_method or "").strip():
            defects.append(f"description_protocol data_access file missing read_method: {item.path}")

    output = bundle.output
    authoritative_contract = ctx.get("authoritative_submission_contract") or {}
    authoritative = bool(isinstance(authoritative_contract, dict) and authoritative_contract.get("is_defined"))
    if paradigm in {"static_optimization", "reinforcement_learning"} and output.sample_submission_required and not authoritative:
        defects.append("description_protocol must not require sample_submission for optimization/RL without authoritative contract")
    if output.sample_submission_required and not _nonempty_list(output.columns):
        defects.append("description_protocol sample_submission_required but output.columns is empty")

    if paradigm == "ml_dl_prediction":
        p = bundle.ml_dl
        required = {
            "ml_dl.train_data": p.train_data,
            "ml_dl.prediction_unit": p.prediction_unit,
            "ml_dl.target": p.target or ctx.get("target_column", ""),
            "ml_dl.validation_design": p.validation_design,
        }
        if ctx.get("predict_table") or p.predict_data:
            required["ml_dl.predict_data"] = p.predict_data or ctx.get("predict_table", "")
        for name, value in required.items():
            if not str(value or "").strip():
                defects.append(f"description_protocol missing {name}")
        if not _nonempty_list(p.feature_boundary):
            defects.append("description_protocol missing ml_dl.feature_boundary")
        if not _nonempty_list(p.leakage_guards):
            defects.append("description_protocol missing ml_dl.leakage_guards")

    elif paradigm == "static_optimization":
        p = bundle.optimization
        required = {
            "optimization.input_instance": p.input_instance,
            "optimization.objective": p.objective,
            "optimization.solution_representation": p.solution_representation,
        }
        for name, value in required.items():
            if not str(value or "").strip():
                defects.append(f"description_protocol missing {name}")
        if not _nonempty_list(p.decision_variables):
            defects.append("description_protocol missing optimization.decision_variables")
        if not _nonempty_list(p.hard_constraints):
            defects.append("description_protocol missing optimization.hard_constraints")
        if not _nonempty_list(p.feasibility_checks):
            defects.append("description_protocol missing optimization.feasibility_checks")

    elif paradigm == "reinforcement_learning":
        p = bundle.rl
        required = {
            "rl.environment": p.environment,
            "rl.state": p.state,
            "rl.action": p.action,
            "rl.transition": p.transition,
            "rl.reward": p.reward,
            "rl.terminal_condition": p.terminal_condition,
            "rl.policy_output": p.policy_output,
            "rl.evaluation_episodes": p.evaluation_episodes,
        }
        for name, value in required.items():
            if not str(value or "").strip():
                defects.append(f"description_protocol missing {name}")
        if not _nonempty_list(p.illegal_action_handling):
            defects.append("description_protocol missing rl.illegal_action_handling")

    elif paradigm == "hybrid_ml_optimization":
        p = bundle.hybrid
        required = {
            "hybrid.prediction_subproblem": p.prediction_subproblem,
            "hybrid.decision_subproblem": p.decision_subproblem,
            "hybrid.handoff": p.handoff,
            "hybrid.final_objective": p.final_objective,
            "hybrid.validation_design": p.validation_design,
        }
        for name, value in required.items():
            if not str(value or "").strip():
                defects.append(f"description_protocol missing {name}")
        if output.sample_submission_required and not authoritative and not _nonempty_list(output.columns):
            defects.append("description_protocol hybrid requires sample_submission without authoritative columns")

    if not str(bundle.overview or bundle.task_goal or "").strip():
        defects.append("description_protocol missing overview/task_goal")
    if not str(bundle.evaluation_summary or "").strip():
        defects.append("description_protocol missing evaluation_summary")

    return list(dict.fromkeys(defects))


def append_constraint_memory_section(path: Path, constraint_memory: dict | None) -> None:
    """在数据认知文档末尾追加“关键约束记忆”小节。"""
    if not constraint_memory:
        return
    items = list(constraint_memory.get("items", []) or [])
    summary = str(constraint_memory.get("summary", "") or "").strip()
    if not summary and not items:
        return

    text = path.read_text(encoding="utf-8") if path.exists() else "# 数据认知文档\n"
    if "## 关键约束记忆" in text:
        return

    lines: list[str] = ["", "## 关键约束记忆"]
    if summary:
        lines.append(f"- 总结: {summary}")
    if items:
        for it in items[:30]:
            name = str(it.get("name", "")).strip() or "未命名约束"
            desc = str(it.get("description", "")).strip()
            pri = str(it.get("priority", "medium")).strip()
            fields = [str(x) for x in (it.get("related_fields", []) or []) if str(x).strip()]
            evidence = [str(x) for x in (it.get("evidence", []) or []) if str(x).strip()]
            lines.append(f"- `{name}` [{pri}]: {desc}")
            if fields:
                lines.append(f"  - 相关字段: {', '.join(fields[:12])}")
            if evidence:
                lines.append(f"  - 证据: {'; '.join(evidence[:4])}")

    path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _truncate_line(text: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _is_requirement_noise_line(line: str) -> bool:
    s = str(line or "").strip()
    if not s:
        return True
    if re.fullmatch(r"[0-9\s\-/_.]+", s):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9\s\-]{4,48}", s):
        return True
    if re.fullmatch(r"\[[0-9]+\].*", s):
        return True
    return False


def _normalize_requirement_lines(text: str) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        s = re.sub(r"\s+", " ", raw.strip())
        if _is_requirement_noise_line(s):
            continue
        if len(s) < 2:
            continue
        if s in seen:
            continue
        seen.add(s)
        lines.append(s)
    return lines


def _pick_requirement_items(lines: list[str], keywords: list[str], limit: int) -> list[str]:
    out: list[str] = []
    for ln in lines:
        lower = ln.lower()
        if any(k.lower() in lower for k in keywords):
            out.append(_truncate_line(ln))
        if len(out) >= limit:
            break
    return out


def _render_original_requirement_coverage(original_requirements: str) -> str:
    lines = _normalize_requirement_lines(original_requirements)
    if not lines:
        return "- 未提供可解析的原始需求文本。"

    numbered: list[str] = []
    for ln in lines:
        if re.match(r"^([0-9]+[\.、]|[（(]?[0-9]+[）)]|[-*•])\s*", ln):
            numbered.append(_truncate_line(ln))
        if len(numbered) >= 8:
            break

    goal_items = _pick_requirement_items(
        lines,
        ["目标", "预测", "训练", "建模", "problem", "task", "output"],
        limit=6,
    )
    deliverables = _pick_requirement_items(
        lines,
        ["输出", "脚本", "数据表", "submission", "description", "sample", "文件", "格式"],
        limit=8,
    )
    constraints = _pick_requirement_items(
        lines,
        ["必须", "禁止", "不得", "指标", "评估", "MAPE", "时间步", "未来", "泄漏", "loss"],
        limit=8,
    )

    if not goal_items:
        goal_items = [_truncate_line(x) for x in lines[:4]]
    if not deliverables:
        deliverables = [_truncate_line(x) for x in lines[:6]]

    summary: list[str] = []
    summary.append("### 需求摘要")
    summary.append("- 以下内容由原始需求自动提炼，保留可执行信息并去除版式噪声。")
    summary.append("")
    summary.append("### 核心目标")
    summary.extend([f"- {x}" for x in goal_items])
    if numbered:
        summary.append("")
        summary.append("### 明确列出的需求")
        summary.extend([f"- {x}" for x in numbered])
    summary.append("")
    summary.append("### 预期交付物")
    summary.extend([f"- {x}" for x in deliverables])
    if constraints:
        summary.append("")
        summary.append("### 约束与评估线索")
        summary.extend([f"- {x}" for x in constraints])
    return "\n".join(summary)


def build_description_markdown(
    plan: PipelinePlan,
    original_requirements: str,
    data_description_digest: str,
    file_summaries: list[FileSummary] | None = None,
    downstream_context: dict | None = None,
) -> str:
    ctx = downstream_context or {}
    task_hint = str(ctx.get("task_hint", "")).strip()
    submission_columns_ctx = [str(x) for x in ctx.get("submission_columns", []) if str(x).strip()]
    generated_submission_columns_ctx = [
        str(x) for x in ctx.get("generated_submission_columns", []) if str(x).strip()
    ]

    id_column = str(ctx.get("id_column", "id"))
    target_column = str(ctx.get("target_column", "target"))
    # Only official/confirmed columns are a hard schema contract. Planner
    # examples such as [id, target] must not lock custom tasks too early.
    # For description output, generated sample_submission columns should still
    # be rendered so the written contract matches the actual CSV artifact.
    spec_columns = submission_columns_ctx or generated_submission_columns_ctx

    task_type = str(ctx.get("task_type_hint", plan.task_type))
    is_multi_class_prob = "class" in task_type.lower() and len(spec_columns) >= 3

    if is_multi_class_prob:
        id_column = spec_columns[0]
        target_column = "probability_vector"
    elif len(spec_columns) >= 2:
        id_column = ",".join(spec_columns[:-1])
        target_column = spec_columns[-1]

    has_official_test_labels = bool(ctx.get("has_official_test_labels", False))
    y_true_field = str(ctx.get("y_true_field", target_column))

    sample_submission_enabled = bool(ctx.get("generate_sample_submission", True))
    if spec_columns:
        if sample_submission_enabled:
            effective_submission_spec = f"sample_submission.csv: [{', '.join(spec_columns)}]"
        else:
            effective_submission_spec = f"submission.csv 建议列: [{', '.join(spec_columns)}]"
    else:
        if sample_submission_enabled:
            effective_submission_spec = "sample_submission.csv: 由任务证据生成列结构。"
        else:
            effective_submission_spec = "未生成 sample_submission.csv；提交列结构以本节任务证据、评估协议和下游建模约定为准。"

    y_true_source = _y_true_source(task_type, target_column, has_official_test_labels, y_true_field)
    validation_protocol = _default_validation_protocol(task_type)
    validation_guardrail = _validation_guardrail(task_type)
    metric_details = _metric_details(task_type, plan.evaluation_metric, plan.evaluation_formula, target_column, spec_columns)
    prediction_unit = _prediction_unit(task_type, task_hint, id_column, target_column, spec_columns)
    input_boundary = _input_boundary(task_type)
    feature_alignment = _feature_alignment(ctx)
    output_boundary = _output_boundary(
        id_column,
        target_column,
        effective_submission_spec,
        spec_columns,
        task_type,
        sample_submission_enabled=sample_submission_enabled,
    )
    submission_contract = _submission_contract(
        id_column,
        target_column,
        effective_submission_spec,
        task_type,
        spec_columns,
        has_predict_table=bool(ctx.get("predict_table", "")),
        sample_submission_enabled=sample_submission_enabled,
    )
    data_inventory_text = _render_data_inventory(file_summaries or [], data_description_digest)
    constraint_memory = ctx.get("constraint_memory", {}) or {}
    cm_items = list(constraint_memory.get("items", []) or [])
    cm_lines: list[str] = []
    if cm_items:
        for it in cm_items[:12]:
            nm = str(it.get("name", "")).strip() or "未命名约束"
            ds = str(it.get("description", "")).strip()
            rf = [str(x) for x in (it.get("related_fields", []) or []) if str(x).strip()]
            if rf:
                cm_lines.append(f"- {nm}: {ds}；相关字段：{', '.join(rf[:8])}")
            else:
                cm_lines.append(f"- {nm}: {ds}")
    constraint_memory_text = "\n".join(cm_lines) if cm_lines else "- 无"
    requirement_coverage_text = _render_original_requirement_coverage(original_requirements)

    return (
        "# 赛题说明\n\n"
        "## 任务概述\n"
        f"- 任务类型：{_task_type_label(task_type)}。\n"
        f"- 任务目标：{'; '.join(plan.objectives)}\n\n"
        "## 任务定义\n"
        "### 学习目标\n"
        f"- {task_hint or '; '.join(plan.objectives)}\n"
        "### 预测或决策单元\n"
        f"{prediction_unit}\n"
        "### 可用输入边界\n"
        f"{input_boundary}\n"
        "### 训练与预测字段一致性\n"
        f"{feature_alignment}\n"
        "### 输出格式与约束\n"
        f"{output_boundary}\n"
        "### 数据切分与验证口径\n"
        f"{validation_protocol}\n\n"
        "## 评估协议\n"
        "### 主指标\n"
        f"- 指标名称：{plan.evaluation_metric}\n"
        f"{metric_details}\n"
        "### 计算公式\n"
        f"- {plan.evaluation_formula}\n"
        "### 计算范围\n"
        "- 在验证集/交叉验证上计算主指标，并报告均值与标准差；如有官方测试集评分，以官方评分为最终比较标准。\n"
        f"{y_true_source}\n"
        "### 验证协议\n"
        "- 复现同一切分策略；若评估过程包含随机性，随机种子必须来自原始需求、官方评估说明或下游实验配置，并写入实验记录，不得在任务定义阶段凭空指定。\n"
        f"{validation_guardrail}\n"
        "### 结果报告要求\n"
        "- 主指标保留至少 6 位小数；并同时报告样本量、切分方式、是否分层。\n"
        "- 若存在并列结果，以主指标优先；如仍并列，比较推理成本与稳定性（方差更小优先）。\n\n"
        "## 提交格式\n"
        f"- {effective_submission_spec}\n"
        f"{submission_contract}\n\n"
        "## 建模边界\n"
        "- 本文档不固定具体算法实现；模型选择、特征组合、超参数搜索由下游建模系统负责探索。\n"
        "- 本文档仅提供任务目标、数据约束、评估协议与提交格式，确保可执行与可比较。\n\n"
        "## 原始需求覆盖\n"
        f"{requirement_coverage_text}\n\n"
        "## 约束与风险\n"
        "### 关键约束记忆\n"
        f"{constraint_memory_text}\n"
        "- 假设：字段语义以数据统计与文档说明联合推断。\n"
        "- 风险：若业务口径存在隐含规则，需在下游建模前补充确认。\n\n"
        "## 数据与文件说明\n"
        f"{data_inventory_text}\n"
    )


def description_quality_check(text: str) -> list[str]:
    required_headers = [
        "任务概述",
        "数据与读取方式",
        "字段说明",
        "任务定义",
        "评估协议",
        "输出或提交格式",
        "关键约束与注意事项",
    ]
    defects: list[str] = []
    for h in required_headers:
        if not _has_h2(text, h):
            defects.append(f"缺少章节: {h}")

    if not _has_h3(text, "计算公式", "Formal Formula"):
        defects.append("缺少计算公式小节")
    if not _has_h3(text, "计算范围", "Computation Scope"):
        defects.append("缺少计算范围小节")
    if not _has_h3(text, "验证协议", "Validation Protocol"):
        defects.append("缺少验证协议小节")
    if not _has_h3(text, "结果报告要求", "Reporting Rules"):
        defects.append("缺少结果报告要求小节")

    lower = text.lower()
    for bad in ["unknown", "tbd", "待补充", "待确认"]:
        if bad in lower:
            defects.append(f"存在占位词: {bad}")

    scoped_text = text
    for bad in ["推荐", "可选", "通常", "视情况", "可以考虑"]:
        if bad in scoped_text:
            defects.append(f"存在评估歧义措辞: {bad}")

    for bad in ["p1 数据认知", "p2 任务定义", "autorealize"]:
        if bad in lower:
            defects.append(f"检测到面向系统内部的流程描述: {bad}")
    for bad in [
        "contract status",
        "unresolved evaluation gaps",
        "required fixes before final scoring",
        "blocked_by_evidence_gap",
        "reflection",
        "ambiguity",
        "output layout",
        "directory tree",
        "file & directory roles",
        "task definition",
        "submission format",
    ]:
        if bad in lower:
            defects.append(f"检测到中间审查痕迹: {bad}")

    return defects


def _replace_legacy_section_titles(text: str) -> str:
    h3_aliases = {
        "Primary Metric": "主指标",
        "Metric Direction": "主指标",
        "Formal Formula": "计算公式",
        "Computation Scope": "计算范围",
        "Validation Protocol": "验证协议",
        "Reporting Rules": "结果报告要求",
        "Submission & Anti-Gaming Checks": "提交校验与防作弊规则",
        "Leakage Guards": "防泄漏要求",
        "Invalid Solution Rules": "非法输出处理",
        "File Format Requirements": "文件格式要求",
        "Expected Output Contract": "输出格式与约束",
    }
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            raw = re.sub(r"^\d+\.\s+", "", line[3:].strip())
            for canonical, aliases in SECTION_ALIASES.items():
                if raw == canonical or raw in aliases:
                    line = f"## {canonical}"
                    break
        elif line.startswith("### "):
            raw = re.sub(r"^\d+\.\s+", "", line[4:].strip())
            if raw in h3_aliases:
                line = f"### {h3_aliases[raw]}"
        out.append(line)
    return "\n".join(out)


def _remove_markdown_section(text: str, level: int, headers: set[str]) -> str:
    marker = "#" * level + " "
    next_marker_re = re.compile(rf"^#{{1,{level}}}\s+")
    lines = text.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        if line.startswith(marker):
            raw = re.sub(r"^\d+\.\s+", "", line[level + 1 :].strip()).lower()
            skip = raw in {h.lower() for h in headers}
            if skip:
                continue
        elif skip and next_marker_re.match(line):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out)


def finalize_description_markdown(text: str) -> str:
    """Final reader-facing cleanup before writing description.md."""
    out = _replace_legacy_section_titles(text)
    out = _remove_markdown_section(
        out,
        2,
        {
            "Output Layout",
            "输出目录结构",
        },
    )
    out = _remove_markdown_section(
        out,
        3,
        {
            "Contract Status",
            "Unresolved Evaluation Gaps",
            "Required Fixes Before Final Scoring",
            "审查状态",
            "未解决评估缺口",
            "正式评分前的修复项",
        },
    )
    replacements = {
        "Submission Format": "输出或提交格式",
        "Task Definition": "任务定义",
        "Data Inventory": "数据与文件说明",
        "Modeling Boundary": "建模边界",
        "Constraints & Risks": "约束与风险",
        "Original Requirement Coverage": "原始需求覆盖",
        "Evaluation": "评估协议",
        "Overview": "任务概述",
        "submission 规范参考": "提交格式参考",
        "submission 要求列": "提交要求列",
        "submission 列名": "提交列名",
        "样例 submission": "提交样例",
        "AutoML": "建模系统",
        "AutoRealize": "系统",
        "rolling-window": "滚动窗口",
        "CV均值": "交叉验证均值",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)

    banned_line_tokens = [
        "contract status",
        "unresolved evaluation gaps",
        "required fixes before final scoring",
        "blocked_by_evidence_gap",
        "ambiguity_points",
        "reflection_log",
        "revision_log",
        "issues/fixes",
        "output layout",
        "directory tree",
        "file & directory roles",
    ]
    kept: list[str] = []
    for line in out.splitlines():
        low = line.lower()
        if any(token in low for token in banned_line_tokens):
            continue
        kept.append(line.rstrip())
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out + "\n"


def eval_ambiguity_defects(text: str) -> list[str]:
    """评估协议零上下文歧义检查（规则版）。"""
    defects: list[str] = []
    task_type = _extract_task_type(text)
    text_lower = text.lower()
    is_solution_or_policy = (
        "静态优化" in text
        or "组合决策" in text
        or "强化学习" in text
        or "序贯决策" in text
        or "方案/策略来源" in text
        or "solution" in text_lower
        or "policy" in text_lower
    )
    must_patterns = [
        r"(y_true|评估依据)",
        r"(Validation Protocol|验证协议)",
        r"(Computation Scope|计算范围)",
        r"(Reporting Rules|结果报告要求)",
    ]
    if is_solution_or_policy:
        must_patterns.append(r"(方案/策略来源|solution|policy|决策变量|动作)")
    else:
        must_patterns.append(r"(y_pred|预测)")

    if "time" in task_type:
        must_patterns.extend(
            [
                r"(按时间顺序|时间顺序|chronological|time order|rolling|walk-forward)",
                r"(验证窗口|validation window|时间切分|rolling-window|walk-forward|外部配置|external configuration)",
                r"(未来信息泄漏|禁止未来|no future|leakage)",
            ]
        )
    elif "class" in task_type or "regression" in task_type:
        must_patterns.extend([r"(K-Fold|holdout|交叉验证|验证集|时间切分|分层切分|训练/验证)"])

    for pattern in must_patterns:
        if not re.search(pattern, text, flags=re.I):
            defects.append(f"缺少关键评估约束: {pattern}")

    scoped = _text_before_section(text, "原始需求覆盖")
    for word in ["推荐", "可选", "通常", "视情况", "可以考虑"]:
        if word in scoped:
            defects.append(f"存在歧义措辞: {word}")

    return defects


def apply_eval_fixes(text: str, y_true_field: str) -> str:
    """当缺少关键约束时，做最小化程序化补丁。"""
    out = text

    if "y_true" not in out:
        if "### 计算范围" in out:
            out = out.replace(
                "### 计算范围\n",
                f"### 计算范围\n- `y_true` 来源：验证窗口真实标签 `{y_true_field}`。\n",
                1,
            )
        elif "### Computation Scope" in out:
            out = out.replace(
                "### Computation Scope\n",
                f"### Computation Scope\n- `y_true` 来源：验证窗口真实标签 `{y_true_field}`。\n",
                1,
            )

    out = out.replace("通常无公开标签", "默认不提供公开标签")
    out = out.replace(
        "- 若为时序任务，切分必须按时间顺序，禁止未来信息泄漏。\n",
        "- 时序任务切分严格按时间顺序，禁止未来信息泄漏。\n",
    )
    return out


def coverage_defects(text: str, original_requirements: str) -> list[str]:
    """Check whether generated description weakens the original requirement."""
    defects: list[str] = []
    original = original_requirements.strip()
    generated = text.strip()
    if original:
        if len(generated) < max(120, int(len(original) * 0.15)):
            defects.append("generated description is much shorter than original requirement")

        key_terms = _extract_key_terms(original_requirements)
        if key_terms:
            base_terms = key_terms[:60]
            hit = sum(1 for t in base_terms if t in text)
            ratio = hit / max(1, len(base_terms))
            if ratio < 0.2:
                missing = [t for t in base_terms if t not in text][:8]
                defects.append(f"original requirement keyword coverage is insufficient: {', '.join(missing)}")
    return defects

def _extract_key_terms(text: str) -> list[str]:
    # 混合中英文粗粒度术语抽取
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,8}", text)
    stop = {"我们", "可以", "进行", "如果", "一个", "这个", "那个", "系统", "数据", "任务", "说明"}
    uniq: list[str] = []
    seen = set()
    for c in candidates:
        c = c.strip()
        if c in stop:
            continue
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:80]


def _default_validation_protocol(task_type: str) -> str:
    tt = task_type.lower()
    if "recommendation" in tt or "ranking" in tt:
        return (
            "- 使用时间切分或用户分层切分构建验证集；若使用随机负采样或随机切分，随机种子由下游实验配置提供并写入记录。\n"
            "- 在每个用户上计算排序指标后取平均，最终分数=验证折均值。\n"
            "- 严禁使用验证/测试交互标签进行候选召回调参。"
        )
    if "reinforcement_learning" in tt or "optimization" in tt:
        return (
            "- 使用任务定义或原始材料中明确给出的验证/评估协议；若未明确给出，下游必须在训练数据上固定一个可复现协议。\n"
            "- 每次评估必须使用相同数据切分与相同约束/惩罚配置；若协议包含随机过程，随机种子必须由外部配置或官方说明提供并记录。\n"
            "- 主指标、约束校验与分数聚合方式一经确定，不得在调参过程中临时改动。"
        )
    if "time" in tt:
        return (
            "- 采用滚动窗口严格时间切分：训练窗口长度=180天，验证窗口长度=30天，步长=30天。\n"
            "- 从最早可用日期开始滚动，直到数据末端；每个窗口仅用历史训练预测未来验证。\n"
            "- 最终分数=所有验证窗口主指标的算术平均值。"
        )
    if "class" in tt:
        return (
            "- 使用训练数据构建分层验证协议，并在实验记录中写明折数、是否打散、随机种子来源与类别分布保持方式。\n"
            "- 各折保持类别分布一致，最终分数=各验证折主指标均值，次级比较=折间标准差（更低更优）。\n"
            "- 本任务默认官方测试集不提供公开标签；模型比较主排序依据为交叉验证均值。"
        )
    if "regression" in tt:
        return (
            "- 使用训练数据构建可复现验证协议，并在实验记录中写明折数或验证集比例、是否打散、随机种子来源。\n"
            "- 最终分数=各验证折或验证集主指标均值，次级比较=折间/重复评估标准差（更低更优）。"
        )
    return (
        "- 在可用历史数据上构建固定且可复现的验证协议。\n"
        "- 使用统一评估口径比较不同方案，按主指标排序并记录约束/数据假设。"
    )


def _metric_details(
    task_type: str,
    metric: str,
    formula: str,
    target_column: str,
    submission_columns: list[str] | None = None,
) -> str:
    tt = task_type.lower()
    submission_columns = submission_columns or []

    if "recommendation" in tt or "ranking" in tt:
        return (
            "- 每行代表“用户-候选对象”或“用户-推荐位次”预测单元。\n"
            "- `score` 列用于排序，分值越高表示推荐优先级越高。\n"
            "- 评估时仅按排序位置计算，不以固定阈值二分类。"
        )
    if "reinforcement_learning" in tt or "optimization" in tt:
        return (
            "- 指标基于策略执行结果计算（成本、收益、约束违约惩罚）。\n"
            "- 需同时报告可行率（满足约束的样本占比）作为审计指标。"
        )
    if "class" in tt:
        if len(submission_columns) >= 3:
            return (
                f"- 标识列: `{submission_columns[0]}`，概率列数: {len(submission_columns) - 1}。\n"
                "- 每个概率列取值范围必须在 [0, 1]。\n"
                "- 每行所有类别概率之和必须为 1（允许绝对误差 <= 1e-6）。"
            )
        return (
            f"- 目标列: `{target_column}`，标签空间必须在训练阶段显式定义。\n"
            "- 若输出为概率，必须给出固定阈值（默认0.5）将概率映射为类别后再计算主指标。\n"
            "- 正负类定义必须固定并写入实验记录；禁止按测试集调阈值。"
        )

    if "regression" in tt or "time" in tt:
        return (
            f"- 目标列: `{target_column}`。\n"
            "- 指标按样本级误差计算后求整体聚合，不得按文件分别归一后再平均。"
        )

    return (
        f"- 指标: `{metric}`。\n"
        f"- 公式: `{formula}`。\n"
        "- 若存在不可分配/不可行动样本，需在评估时按统一惩罚规则计入主指标。"
    )


def _prediction_unit(
    task_type: str,
    task_hint: str,
    id_column: str,
    target_column: str,
    submission_columns: list[str] | None = None,
) -> str:
    submission_columns = submission_columns or []
    t = task_type.lower()
    if "recommendation" in t or "ranking" in t:
        key = submission_columns[0] if submission_columns else id_column
        return (
            f"- 以 `{key}` 作为用户或请求主键。\n"
            "- 每个主键对应多个候选对象，输出用于排序的分数字段。\n"
            f"- 任务语义: {task_hint or task_type}。"
        )
    if "reinforcement_learning" in t or "optimization" in t:
        key = submission_columns[0] if submission_columns else id_column
        return (
            f"- 以 `{key}` 标识一条输出记录或待处理对象。\n"
            "- 模型输出的具体字段语义以任务定义、原始材料和提交列名为准，不套用固定二列模板。\n"
            f"- 任务语义: {task_hint or task_type}。"
        )
    if "class" in task_type.lower() and len(submission_columns) >= 3:
        return (
            f"- 以 `{submission_columns[0]}` 标识单条预测单元。\n"
            f"- 模型需为每个预测单元输出 `{len(submission_columns) - 1}` 个候选类别概率。\n"
            f"- 任务语义: {task_hint or task_type}。"
        )
    return (
        f"- 以 `{id_column}` 标识单条预测单元。\n"
        f"- 模型需为每个预测单元输出 `{target_column}`。\n"
        f"- 任务语义: {task_hint or task_type}。"
    )


def _input_boundary(task_type: str) -> str:
    lines = [
        "- 仅可使用训练阶段可获得的特征；禁止泄漏未来信息或使用测试标签。",
        "- 对缺失值、异常值、类别编码的处理必须在训练折内拟合并应用于验证/测试折。",
    ]
    if "time" in task_type.lower():
        lines.append("- 时序任务必须保留时间顺序，禁止随机打散后切分。")
    return "\n".join(lines)


def _feature_alignment(ctx: dict) -> str:
    train_table = str(ctx.get("train_table", "") or "")
    predict_table = str(ctx.get("predict_table", "") or "")
    target_col = str(ctx.get("target_column", "") or "")
    train_only = [str(x) for x in ctx.get("train_only_columns", []) if str(x).strip()]
    predict_only = [str(x) for x in ctx.get("predict_only_columns", []) if str(x).strip()]

    train_name = train_table or "（未识别）"
    pred_name = predict_table or "（未提供，默认由下游在训练数据上自行切分验证）"
    lines = [
        f"- 训练数据文件: `{train_name}`；预测数据文件: `{pred_name}`。",
        f"- 目标标签列: `{target_col}`（仅允许出现在训练侧；预测侧缺失该列视为正常）。",
    ]

    if train_only:
        lines.append(
            "- 仅训练侧可见字段: "
            + ", ".join([f"`{c}`" for c in train_only[:20]])
            + "。这些字段必须在建模时显式标注为“训练可用但预测不可直接输入”；"
            "若要使用，需通过训练阶段可复现的聚合/编码方式映射到测试侧。"
        )
    else:
        lines.append("- 未发现仅训练侧字段。")

    if predict_only:
        lines.append(
            "- 仅预测侧可见字段: "
            + ", ".join([f"`{c}`" for c in predict_only[:20]])
            + "。默认不纳入训练主特征，除非可在训练侧构造完全同构的字段。"
        )
    else:
        lines.append("- 未发现仅预测侧字段。")

    lines.append("- 严禁在验证/预测阶段引用测试集不存在的原始字段。")
    return "\n".join(lines)


def _output_boundary(
    id_column: str,
    target_column: str,
    submission_spec: str,
    submission_columns: list[str] | None = None,
    task_type: str = "",
    *,
    sample_submission_enabled: bool = True,
) -> str:
    submission_columns = submission_columns or []
    tt = task_type.lower()
    if not submission_columns:
        if not sample_submission_enabled:
            return (
                "- 当前配置未生成 `sample_submission.csv`，不要把任务强行套成固定 `id` + `target` 模板。\n"
                "- 下游应以本节任务定义、原始需求、数据认知和约束记忆确定正式 `submission.csv` 的列结构。\n"
                f"- 提交格式参考：{submission_spec}"
            )
        return (
            "- 当前未发现官方提交样例列合同；不要把任务强行套成固定 `id` + `target` 模板。\n"
            "- 下游应以实际生成的 `sample_submission.csv` 为格式样例，并结合原始需求、数据认知和约束记忆解释每一列。\n"
            f"- 提交格式参考：{submission_spec}"
        )
    if ("recommendation" in tt or "ranking" in tt) and len(submission_columns) >= 3:
        return (
            f"- 输出文件必须包含 `{submission_columns[0]}`、`{submission_columns[1]}` 与排序分数列（如 `{submission_columns[-1]}`）。\n"
            "- 对同一主键可出现多行候选对象，按排序分数降序用于评估。\n"
            f"- 提交格式参考：{submission_spec}"
        )
    if ("reinforcement_learning" in tt or "optimization" in tt) and len(submission_columns) >= 2:
        columns_text = ", ".join(f"`{c}`" for c in submission_columns)
        return (
            f"- 输出文件列顺序必须严格为: {columns_text}。\n"
            "- 每一行表示一个由任务定义确定的预测、决策或方案记录；每列语义以原始材料、字段名和任务定义为准。\n"
            "- 若任务定义包含业务约束或惩罚规则，正式评估必须使用同一套规则校验；`sample_submission.csv` 只表示格式样例。\n"
            f"- 提交格式参考：{submission_spec}"
        )
    if "class" in tt and len(submission_columns) >= 3:
        return (
            f"- 输出文件必须包含 `{submission_columns[0]}` 与其余全部类别概率列。\n"
            "- 概率列集合与列顺序必须与提交样例完全一致。\n"
            f"- 提交格式参考：{submission_spec}"
        )
    id_repr = id_column if "," not in id_column else f"复合主键({id_column})"
    return (
        f"- 输出文件必须包含且仅包含 `{id_repr}` 与 `{target_column}`（或提交要求列）。\n"
        "- 列顺序必须与提交格式规范一致。\n"
        f"- 提交格式参考：{submission_spec}"
    )


def _submission_contract(
    id_column: str,
    target_column: str,
    submission_spec: str,
    task_type: str,
    submission_columns: list[str] | None = None,
    has_predict_table: bool = True,
    *,
    sample_submission_enabled: bool = True,
) -> str:
    tt = task_type.lower()
    is_classification = "class" in tt
    spec_columns = submission_columns or []
    lines = ["### 文件格式要求", "- 文件名: `submission.csv`。"]
    if not spec_columns:
        if sample_submission_enabled:
            lines.append("- 当前未发现官方提交列合同；正式列定义以本次生成的 `sample_submission.csv` 和任务证据为准。")
        else:
            lines.append("- 当前配置未生成 `sample_submission.csv`；正式列定义以任务证据、评估协议和下游建模约定为准。")
        lines.append("- 禁止在没有证据时硬套固定两列模板；优化、调度、推荐等任务可以生成多列决策输出。")
        lines.append("- 每一列都必须能从原始需求、数据字段或约束记忆中解释其业务含义。")
        if has_predict_table:
            lines.append("- 若存在独立预测/待决策清单，正式输出记录数应覆盖该清单要求的对象。")
        else:
            if sample_submission_enabled:
                lines.append("- 当前未识别独立预测/待决策清单时，`sample_submission.csv` 仅为格式样例，不代表正式评测行数。")
            else:
                lines.append("- 当前未识别独立预测/待决策清单时，正式评测行数需由评估协议或下游评测系统明确。")
        return "\n".join(lines)

    if ("recommendation" in tt or "ranking" in tt) and len(spec_columns) >= 3:
        lines.append(f"- 用户键列: `{spec_columns[0]}`；候选对象列: `{spec_columns[1]}`；排序分数字段: `{spec_columns[-1]}`。")
        lines.append("- 对每个用户必须输出固定数量候选（如任务定义要求 TopK），不得重复候选对象。")
        lines.append("- 排序分数字段必须为可排序数值，禁止 NaN/Inf。")
        lines.append("- 行数按任务定义中的用户数、候选数或全候选对范围严格校验。")
        return "\n".join(lines)
    if ("reinforcement_learning" in tt or "optimization" in tt) and len(spec_columns) >= 2:
        lines.append(f"- 列顺序: `{', '.join(spec_columns)}`。")
        lines.append("- 各列含义必须来自任务定义、原始材料和字段名，不得按固定“主键+动作”模板自行改写。")
        lines.append("- 若存在业务约束、惩罚或可行性规则，正式提交需按任务定义统一校验。")
        lines.append("- 行数必须覆盖任务定义要求的全部输出记录；`sample_submission.csv` 仅用于说明格式。")
        return "\n".join(lines)

    has_explicit_multicolumn = len(spec_columns) >= 3
    if has_explicit_multicolumn:
        lines.append("- 列定义必须严格遵循“提交格式”章节中的列序与语义，不得省略或新增。")
        if is_classification and len(spec_columns) >= 3:
            lines.append(f"- 标识列: `{spec_columns[0]}`。")
            lines.append(f"- 概率列: `{', '.join(spec_columns[1:])}`。")
            lines.append("- 除标识列外其余列均为概率列，取值范围[0,1]，且每行概率和=1（容差1e-6）。")
        else:
            lines.append(f"- 标识列: `{', '.join(spec_columns[:-1])}`。")
            lines.append(f"- 目标列: `{spec_columns[-1]}`。")
    else:
        lines.append(f"- 第一列: `{id_column}`，类型 `string`，值域必须与测试集 ID 一一对应且无缺失。")
        if is_classification:
            lines.append(f"- 第二列: `{target_column}`，取值为 `True/False` 或 `0/1`（需与官方样例一致）。")
        else:
            lines.append(f"- 第二列: `{target_column}`，取值为浮点数或整型数值。")

    if has_predict_table:
        lines.append("- 行数必须等于预测/测试集样本数；不得增删行；不得重排列。")
    else:
        lines.append("- 当前数据未提供独立预测/测试集；`sample_submission.csv` 仅为格式样例，不代表正式评测行数。")
        lines.append("- 若下游评测系统提供待预测清单，正式 `submission.csv` 行数必须等于该清单行数；若无待预测清单，则按本文“评估协议”的滚动验证规则报告分数。")
    return "\n".join(lines)


def _parse_submission_columns(submission_spec: str) -> list[str]:
    left = submission_spec.find("[")
    right = submission_spec.find("]", left + 1) if left >= 0 else -1
    if left < 0 or right < 0:
        return []
    raw = submission_spec[left + 1 : right]
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    return cols


def _as_evaluation_contract(review: EvaluationContractReview | dict[str, Any]) -> EvaluationContractReview:
    if isinstance(review, EvaluationContractReview):
        return review
    return EvaluationContractReview.model_validate(review)


def _nonempty_list(values: list[str] | None) -> list[str]:
    return [str(x).strip() for x in (values or []) if str(x).strip()]


def _direction_label(direction: str) -> str:
    d = direction.strip().lower().replace("-", "_").replace(" ", "_")
    if d in {"minimize", "min", "lower_is_better", "smaller_is_better"}:
        return "minimize"
    if d in {"maximize", "max", "higher_is_better", "larger_is_better"}:
        return "maximize"
    return d


def _evaluation_mentions_multiple_objectives(contract: EvaluationContractReview) -> bool:
    packed = "\n".join(
        [
            str(contract.primary_metric),
            str(contract.metric_formula),
            str(contract.aggregation_rule),
            str(contract.validation_protocol),
            "\n".join(_nonempty_list(contract.tie_break_rules)),
            "\n".join(_nonempty_list(contract.audit_metrics)),
        ]
    ).lower()
    multi_markers = [
        "multi-objective",
        "multi objective",
        "secondary",
        "tie-break",
        "tie break",
        "lexicographic",
        "weighted",
        "weight",
        "多目标",
        "次要",
        "并列",
        "字典序",
        "权重",
        "加权",
    ]
    if any(marker in packed for marker in multi_markers):
        return True
    paired_markers = [
        ("fail", "cost"),
        ("failure", "cost"),
        ("violation", "cost"),
        ("constraint", "objective"),
        ("失败", "成本"),
        ("违规", "成本"),
        ("约束", "目标"),
    ]
    return any(a in packed and b in packed for a, b in paired_markers)


def _evaluation_has_scalarization(contract: EvaluationContractReview) -> bool:
    packed = "\n".join(
        [
            str(contract.scalar_score_formula),
            str(contract.metric_formula),
            str(contract.aggregation_rule),
            "\n".join(_nonempty_list(contract.tie_break_rules)),
        ]
    ).lower()
    if str(contract.scalar_score_formula or "").strip():
        return True
    scalar_markers = [
        "weighted",
        "weight",
        "penalty",
        "large constant",
        "score =",
        "scalar",
        "权重",
        "加权",
        "惩罚",
        "罚",
        "大常数",
        "单一数值",
        "标量",
    ]
    return any(marker in packed for marker in scalar_markers)


def _contains_any(text: str, markers: list[str]) -> bool:
    low = str(text or "").lower()
    return any(marker.lower() in low for marker in markers)


def _normalize_formula_for_compare(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", "", value)
    value = value.replace("：", ":").replace("＝", "=")
    value = re.sub(r"^(final_)?score=", "", value)
    return value


def _scalar_formula_duplicates_metric(contract: EvaluationContractReview) -> bool:
    metric = _normalize_formula_for_compare(contract.metric_formula)
    scalar = _normalize_formula_for_compare(contract.scalar_score_formula)
    if not scalar:
        return True
    if not metric:
        return False
    return metric == scalar or metric in scalar or scalar in metric


def _final_score_formula(contract: EvaluationContractReview) -> str:
    return str(contract.metric_formula or contract.scalar_score_formula or "").strip()


def _evaluation_scalarization_defects(contract: EvaluationContractReview) -> list[str]:
    """Ensure tie-break or multi-objective rules are present in the scalar score.

    Tree search can only compare one number. If the prose says "same success
    rate then lower cost wins", a score that only contains success rate is not
    operationally equivalent to the evaluation protocol.
    """
    defects: list[str] = []
    effective_formula = _final_score_formula(contract)
    tie_text = "\n".join(_nonempty_list(contract.tie_break_rules))
    aggregation_text = str(contract.aggregation_rule or "")
    objective_text = "\n".join(
        [
            str(contract.primary_metric or ""),
            str(contract.metric_formula or ""),
            aggregation_text,
            tie_text,
            "\n".join(_nonempty_list(contract.audit_metrics)),
        ]
    )
    has_multi = _evaluation_mentions_multiple_objectives(contract)
    if has_multi and not effective_formula.strip():
        defects.append("evaluation_contract multi-objective/tie-break metrics must define one final metric_formula with explicit scalarization")
        return defects
    if has_multi and not _evaluation_has_scalarization(contract):
        defects.append("evaluation_contract multi-objective/tie-break metrics must define one final metric_formula with explicit scalarization")

    groups = {
        "cost": ["成本", "费用", "运费", "cost", "expense", "fee"],
        "failure_or_unassigned": ["未分配", "失败数", "失败数量", "违约数", "违规数", "不可行数", "fail_count", "failed_count", "failure_count", "unassigned", "violation_count", "infeasible_count"],
        "trip_count": ["发车次数", "车次数", "车辆数", "派车数", "trip", "dispatch", "vehicle_count"],
        "utilization": ["装载率", "利用率", "载荷", "体积利用", "重量利用", "utilization", "load_rate"],
    }
    for group_name, markers in groups.items():
        if _contains_any(tie_text or objective_text, markers) and not _contains_any(effective_formula, markers):
            defects.append(
                f"evaluation_contract final metric_formula ignores tie-break/objective term: {group_name}"
            )

    if str(contract.scalar_score_formula or "").strip() and not _scalar_formula_duplicates_metric(contract):
        defects.append(
            "evaluation_contract metric_formula and scalar_score_formula define competing scores; put the single final ranking formula in metric_formula and leave scalar_score_formula empty or identical"
        )
    return defects


def evaluation_contract_defects(review: EvaluationContractReview | dict[str, Any]) -> list[str]:
    contract = _as_evaluation_contract(review)
    defects: list[str] = []
    required_text_fields = {
        "primary_metric": contract.primary_metric,
        "metric_direction": contract.metric_direction,
        "metric_formula": contract.metric_formula,
        "prediction_unit": contract.prediction_unit,
        "y_true_source": contract.y_true_source,
        "y_pred_source": contract.y_pred_source,
        "computation_scope": contract.computation_scope,
        "aggregation_rule": contract.aggregation_rule,
        "validation_protocol": contract.validation_protocol,
    }
    for name, value in required_text_fields.items():
        if not str(value).strip():
            defects.append(f"evaluation_contract missing field: {name}")

    direction = _direction_label(contract.metric_direction)
    if direction not in {"minimize", "maximize"}:
        defects.append("evaluation_contract metric_direction must be minimize or maximize")
    defects.extend(_evaluation_scalarization_defects(contract))

    required_lists = {
        "submission_checks": contract.submission_checks,
        "leakage_guards": contract.leakage_guards,
        "invalid_solution_rules": contract.invalid_solution_rules,
        "tie_break_rules": contract.tie_break_rules,
    }
    for name, values in required_lists.items():
        if not _nonempty_list(values):
            defects.append(f"evaluation_contract missing list: {name}")

    if not contract.passed and not (_nonempty_list(contract.issues) and _nonempty_list(contract.fixes)):
        defects.append("evaluation_contract not passed but issues/fixes are missing")

    packed = "\n".join(
        [
            str(contract.primary_metric),
            str(contract.metric_formula),
            str(contract.validation_protocol),
            "\n".join(_nonempty_list(contract.submission_checks)),
            "\n".join(_nonempty_list(contract.leakage_guards)),
            "\n".join(_nonempty_list(contract.invalid_solution_rules)),
        ]
    )
    for bad in ["推荐", "可选", "通常", "视情况", "可以考虑", "unknown", "tbd", "待补充", "待确认"]:
        if bad in packed.lower():
            defects.append(f"evaluation_contract contains ambiguous placeholder: {bad}")

    return defects


def _render_contract_list(title: str, values: list[str]) -> list[str]:
    lines = [f"### {title}"]
    for item in _nonempty_list(values):
        lines.append(f"- {item}")
    return lines


def apply_evaluation_contract(desc: str, review: EvaluationContractReview | dict[str, Any], downstream_context: dict | None = None) -> str:
    """Rewrite Evaluation from the strict structured contract.

    The LLM decides the contract, but this deterministic renderer prevents the
    final description from drifting back into vague metric or validation prose.
    """
    downstream_context = downstream_context or {}
    contract = _as_evaluation_contract(review)
    defects = evaluation_contract_defects(contract)
    if defects:
        raise RuntimeError(f"Evaluation contract is not strict enough: {defects[:8]}")

    direction = _direction_label(contract.metric_direction)
    audit_metrics = _nonempty_list(contract.audit_metrics) or ["无；只使用唯一主指标排序，审计项不得改变主排名。"]
    evidence = _nonempty_list(contract.evidence) or ["原始需求、数据认知与提交样例。"]
    paradigm = str((downstream_context or {}).get("problem_paradigm", "")).strip()
    if paradigm in {"static_optimization", "reinforcement_learning", "hybrid_ml_optimization"}:
        true_label = "评估依据"
        pred_label = "方案/策略来源"
    else:
        true_label = "`y_true` 来源"
        pred_label = "`y_pred` 来源"
    final_formula = _final_score_formula(contract)
    lines: list[str] = [
        "## 评估协议",
        "### 主指标",
        f"- 指标名称：{contract.primary_metric.strip()}",
        f"- 优化方向：{_direction_label_zh(direction)}。所有候选方案必须按该方向比较，禁止在实验过程中切换排序方向。",
        f"- 预测或决策单元：{contract.prediction_unit.strip()}",
        "### 计算公式",
        f"- 最终评分公式：{final_formula}",
        "### 计算范围",
        f"- {true_label}：{contract.y_true_source.strip()}",
        f"- {pred_label}：{contract.y_pred_source.strip()}",
        f"- 覆盖范围：{contract.computation_scope.strip()}",
        f"- 聚合方式：{contract.aggregation_rule.strip()}",
        "### 验证协议",
        f"- {contract.validation_protocol.strip()}",
    ]
    lines.extend(_render_contract_list("提交校验与防作弊规则", contract.submission_checks))
    lines.extend(_render_contract_list("防泄漏要求", contract.leakage_guards))
    lines.extend(_render_contract_list("非法输出处理", contract.invalid_solution_rules))
    lines.append("### 结果报告要求")
    lines.append("- 主指标必须至少保留 6 位小数，并同时记录样本量、切分协议、评估脚本版本与提交文件哈希。")
    lines.append("- 任何因异常、NaN、Inf、缺行、多行、重复主键或约束违约产生的修正，必须先按“非法输出处理”规则处理，再计算主指标。")
    for item in audit_metrics:
        lines.append(f"- 审计指标：{item}")
    for item in _nonempty_list(contract.tie_break_rules):
        lines.append(f"- 并列规则：{item}")
    for item in evidence:
        lines.append(f"- 依据：{item}")
    if not contract.passed:
        lines.append("### 评估前置条件")
        for item in _nonempty_list(contract.issues):
            lines.append(f"- 当前材料限制：{item}")
        for item in _nonempty_list(contract.fixes):
            lines.append(f"- 正式评分前需明确：{item}")
    lines.append("")
    return _replace_h2_section(desc, "评估协议", "\n".join(lines))


def sync_submission_format_with_context(desc: str, downstream_context: dict) -> str:
    """Force description.md's Submission Format to match sample_submission.csv.

    LLM rewrite/reflection steps may edit the Submission Format section after the
    sample schema is generated. This final deterministic pass prevents drift
    between the actual sample_submission.csv header and description.md.
    """
    ctx = downstream_context or {}
    if not bool(ctx.get("generate_sample_submission", True)):
        return desc
    confirmed_cols = [str(x) for x in ctx.get("submission_columns", []) if str(x).strip()]
    generated_cols = [str(x) for x in ctx.get("generated_submission_columns", []) if str(x).strip()]
    cols = confirmed_cols or generated_cols
    if not cols:
        return desc

    cols_text = ", ".join(f"`{c}`" for c in cols)
    spec_text = ", ".join(cols)
    source_text = "官方样例确认" if confirmed_cols else "本次生成"
    has_predict_table = bool(str(ctx.get("predict_table", "") or "").strip())
    row_rule = (
        "- 若存在独立预测/待决策清单，正式 `submission.csv` 的行数必须覆盖该清单要求的对象。"
        if has_predict_table
        else "- 当前未识别独立预测/待决策清单时，`sample_submission.csv` 仅为格式样例，不代表正式评测行数。"
    )
    section = "\n".join(
        [
            "## 输出或提交格式",
            f"- sample_submission.csv: [{spec_text}]",
            "### 文件格式要求",
            "- 文件名: `submission.csv`。",
            f"- 列定义来源: {source_text}的 `sample_submission.csv`。",
            f"- 列顺序必须与 `sample_submission.csv` 完全一致: {cols_text}。",
            "- `sample_submission.csv` 只表示格式样例；正式求解结果由下游建模系统生成 `submission.csv`。",
            row_rule,
            "",
        ]
    )
    return _replace_h2_section(desc, "提交格式", section)


def _replace_h2_section(text: str, header: str, replacement: str) -> str:
    lines = text.splitlines()
    aliases = set(_section_aliases(header))
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("## ") and line[3:].strip() in aliases:
            start = idx
            break
    if start is None:
        return text.rstrip() + "\n\n" + replacement.rstrip() + "\n"
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    new_lines = lines[:start] + replacement.rstrip("\n").splitlines() + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def _y_true_source(task_type: str, target_column: str, has_official_test_labels: bool, y_true_field: str) -> str:
    if has_official_test_labels:
        return f"- `y_true` 来源：官方测试集标签列 `{y_true_field}`。"

    tt = task_type.lower()
    if "time" in tt:
        return (
            f"- `y_true` 来源：历史数据按时间切分得到的验证窗口真实标签 `{y_true_field}`；"
            "本任务默认官方测试集不提供公开标签，仅用于提交评分。"
        )
    if "class" in tt or "regression" in tt:
        return (
            f"- `y_true` 来源：训练集标签 `{y_true_field}`，通过固定切分策略得到验证集真实标签；"
            "本任务默认官方测试集不提供公开标签。"
        )
    return "- `y_true` 来源：任务定义中的验证真值、历史执行结果或评估反馈字段；必须在下游实验记录中固定说明。"


def _validation_guardrail(task_type: str) -> str:
    tt = task_type.lower()
    if "time" in tt:
        return "- 时序任务切分严格按时间顺序，禁止未来信息泄漏与随机打散切分。"
    if "class" in tt:
        return "- 分类任务必须保持类别分布一致；若使用分层交叉验证，折数、打散策略与随机种子来源必须写入实验记录。"
    if "regression" in tt:
        return "- 回归任务必须使用可复现的训练/验证切分；折数或验证比例、打散策略与随机种子来源必须写入实验记录。"
    return "- 优化/强化学习任务必须固定评估协议、约束配置与随机种子，不允许在调参过程中临时改动。"


def _extract_task_type(text: str) -> str:
    match = re.search(r"^\s*-\s*(?:Task Type|任务类型)[：:]\s*(.+?)\s*[。.]?\s*$", text, flags=re.M)
    if not match:
        return ""
    return match.group(1).strip().lower()


def _render_data_inventory(file_summaries: list[FileSummary], inventory_digest: str) -> str:
    if not file_summaries:
        return inventory_digest

    def _clean_inline(text: str) -> str:
        s = str(text or "").replace("\r", " ").replace("\n", " | ").strip()
        s = re.sub(r"\s+", " ", s)
        return s[:800]

    def _is_tabular(fs: FileSummary) -> bool:
        is_json = str(fs.path).lower().endswith(".json")
        json_strategy = str((fs.source_metadata or {}).get("json_strategy", "")).strip()
        parsed_kind = str((fs.source_metadata or {}).get("kind", "")).strip().lower()
        role_tabular = fs.role == FileRole.raw_data_table
        doc_like = fs.role in {FileRole.task_requirement, FileRole.data_description}
        if _is_document_like(fs) or parsed_kind in {"document", "structured_document"} or doc_like:
            # 文档类文件即便 LLM 提取了 key_columns，也不应渲染为 data fields。
            return False
        return (
            role_tabular
            or bool(fs.columns)
            or (is_json and bool(json_strategy))
        )

    lines: list[str] = []
    for fs in file_summaries:
        lines.append(f"### {fs.path}")
        lines.append(f"- 文件角色：{_role_label(fs.role)}")
        lines.append(f"- 文件摘要：{_clean_inline(fs.summary)}")
        if str(fs.detailed_report or "").strip():
            lines.append("")
            lines.append("#### 详细认知报告")
            lines.append(str(fs.detailed_report).strip())
        is_json = str(fs.path).lower().endswith(".json")
        tabular_candidate = _is_tabular(fs)
        if is_json and not tabular_candidate:
            lines.append("- JSON结构：")
            meta = fs.source_metadata or {}
            root_type = str(meta.get("type", meta.get("json_root_type", "未明确")))
            lines.append(f"  - 根节点类型：`{root_type}`")
            paths = meta.get("json_paths_topk") or []
            if paths:
                lines.append("  - 常见嵌套路径：")
                for p in list(paths)[:40]:
                    lines.append(f"    - `{p}`")
            lines.append("  - 该 JSON 不适合直接表格化，建议按嵌套结构/配置语义使用。")
        elif tabular_candidate and fs.columns:
            lines.append("- 字段说明：")
            profiles = _profile_map(fs)
            for col in fs.columns:
                meaning = _llm_field_description(fs, col)
                if not meaning:
                    meaning = "本字段尚未形成稳定的自然语言解释；请结合字段名、样例值和结构画像使用。"
                schema = format_column_profile_inline(profiles[col]) if col in profiles else "结构画像暂缺"
                lines.append(f"  - `{col}`：{meaning}（{schema}）")
        else:
            lines.append("- 提取的关键信息：")
            if fs.key_entities:
                lines.append("  - 关键实体：" + "，".join([f"`{x}`" for x in fs.key_entities[:20]]))
            if fs.warnings:
                lines.append("  - 约束或风险：")
                for w in fs.warnings[:12]:
                    lines.append(f"    - {_clean_inline(w)}")
            meta = fs.source_metadata or {}
            meta_keys = [k for k in ["pages", "chars", "archive_type", "entries", "lines"] if k in meta]
            if meta_keys:
                lines.append("  - 源文件元信息：")
                for k in meta_keys:
                    lines.append(f"    - {_metadata_label(k)}：`{_clean_inline(meta.get(k))}`")
        lines.append("")
    return "\n".join(lines).strip()

