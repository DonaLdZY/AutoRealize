from __future__ import annotations

import re
from pathlib import Path

from .models import FileSummary, FileRole, PipelinePlan
from .profiling.relations import RelationHint


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
        if fs.columns:
            lines.append(f"- 字段: {', '.join(fs.columns[:30])}")
        if fs.column_semantics:
            lines.append("- 字段作用猜测:")
            for col, meaning in list(fs.column_semantics.items())[:80]:
                lines.append(f"  - `{col}`: {meaning}")
        if fs.column_profiles:
            lines.append("- 字段统计（关键列）:")
            for p in fs.column_profiles[:40]:
                line = (
                    f"  - `{p.get('name')}` | dtype={p.get('dtype')} | "
                    f"null_ratio={p.get('null_ratio')} | unique={p.get('unique_count')}"
                )
                ns = p.get("numeric_stats") or {}
                qs = p.get("quantiles") or {}
                ds = p.get("datetime_stats") or {}
                if ns:
                    line += (
                        f" | mean={ns.get('mean')} median={ns.get('median')} "
                        f"std={ns.get('std')} var={ns.get('var')} min={ns.get('min')} max={ns.get('max')}"
                    )
                if qs:
                    line += f" | q1={qs.get('q1')} q3={qs.get('q3')} p05={qs.get('p05')} p95={qs.get('p95')}"
                if ds:
                    line += f" | date_min={ds.get('min')} date_max={ds.get('max')} range_days={ds.get('range_days')}"
                lines.append(line)
        if fs.related_files:
            lines.append(f"- 可能关联: {', '.join(fs.related_files[:12])}")
        if fs.warnings:
            lines.append(f"- 风险: {'; '.join(fs.warnings[:8])}")
        lines.append("")
    lines.append("## 跨文件关系")
    if relations:
        for r in relations:
            lines.append(f"- {r.left_file} <-> {r.right_file}: {', '.join(r.shared_columns)} ({r.reason})")
    else:
        lines.append("- 暂未发现明显同名字段关系。")
    path.write_text("\n".join(lines), encoding="utf-8")


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

    id_column = str(ctx.get("id_column", "id"))
    target_column = str(ctx.get("target_column", "target"))
    spec_columns = _parse_submission_columns(plan.submission_spec)
    if submission_columns_ctx:
        spec_columns = submission_columns_ctx

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

    if spec_columns:
        effective_submission_spec = f"sample_submission.csv: [{', '.join(spec_columns)}]"
    else:
        effective_submission_spec = plan.submission_spec

    y_true_source = _y_true_source(task_type, target_column, has_official_test_labels, y_true_field)
    validation_protocol = _default_validation_protocol(task_type)
    validation_guardrail = _validation_guardrail(task_type)
    metric_details = _metric_details(task_type, plan.evaluation_metric, plan.evaluation_formula, target_column, spec_columns)
    prediction_unit = _prediction_unit(task_type, task_hint, id_column, target_column, spec_columns)
    input_boundary = _input_boundary(task_type)
    feature_alignment = _feature_alignment(ctx)
    output_boundary = _output_boundary(id_column, target_column, effective_submission_spec, spec_columns, task_type)
    submission_contract = _submission_contract(id_column, target_column, effective_submission_spec, task_type, spec_columns)
    data_inventory_text = _render_data_inventory(file_summaries or [], data_description_digest)

    return (
        "# description.md\n\n"
        "## Overview\n"
        f"- Task Type: {task_type}\n"
        f"- Objectives: {'; '.join(plan.objectives)}\n\n"
        "## Data Inventory\n"
        f"{data_inventory_text}\n\n"
        "## Task Definition\n"
        "### Learning Objective\n"
        f"- {task_hint or '; '.join(plan.objectives)}\n"
        "### Prediction Unit\n"
        f"{prediction_unit}\n"
        "### Model Input Boundary\n"
        f"{input_boundary}\n"
        "### Train/Test Feature Alignment\n"
        f"{feature_alignment}\n"
        "### Expected Output Contract\n"
        f"{output_boundary}\n"
        "### Data Split & Validation Protocol\n"
        f"{validation_protocol}\n\n"
        "## Evaluation\n"
        "### Metric Definition\n"
        f"- Primary Metric: {plan.evaluation_metric}\n"
        f"{metric_details}\n"
        "### Formal Formula\n"
        f"- {plan.evaluation_formula}\n"
        "### Computation Scope\n"
        "- 在验证集/交叉验证上计算主指标，并报告均值与标准差；如有官方测试集评分，以官方评分为最终比较标准。\n"
        f"{y_true_source}\n"
        "### Validation Protocol\n"
        "- 固定随机种子为 `20250430`，并复现相同切分策略；禁止使用测试标签参与训练、特征构造或阈值搜索。\n"
        f"{validation_guardrail}\n"
        "### Reporting Rules\n"
        "- 主指标保留至少 6 位小数；并同时报告样本量、切分方式、是否分层。\n"
        "- 若存在并列结果，以主指标优先；如仍并列，比较推理成本与稳定性（方差更小优先）。\n\n"
        "## Submission Format\n"
        f"- {effective_submission_spec}\n"
        f"{submission_contract}\n\n"
        "## Modeling Boundary\n"
        "- 本文档不固定具体算法实现；模型选择、特征组合、超参数搜索由下游 AutoML 系统负责探索。\n"
        "- 本文档仅提供任务目标、数据约束、评估协议与提交格式，确保可执行与可比较。\n\n"
        "## Original Requirement Coverage\n"
        f"{original_requirements[:20000]}\n\n"
        "## Constraints & Risks\n"
        "- 假设: 字段语义以数据统计与文档说明联合推断。\n"
        "- 风险: 若业务口径存在隐含规则，需在下游建模前补充确认。\n"
    )


def description_quality_check(text: str) -> list[str]:
    required_headers = [
        "Overview",
        "Data Inventory",
        "Task Definition",
        "Evaluation",
        "Submission Format",
        "Modeling Boundary",
        "Constraints & Risks",
    ]
    defects: list[str] = []
    for h in required_headers:
        if not re.search(rf"^##\s+(\d+\.\s+)?{re.escape(h)}\s*$", text, flags=re.M):
            defects.append(f"缺少章节: {h}")

    if "### Formal Formula" not in text:
        defects.append("缺少 Formal Formula 小节")
    if "### Computation Scope" not in text:
        defects.append("缺少 Computation Scope 小节")
    if "### Validation Protocol" not in text:
        defects.append("缺少 Validation Protocol 小节")
    if "### Reporting Rules" not in text:
        defects.append("缺少 Reporting Rules 小节")

    lower = text.lower()
    for bad in ["unknown", "tbd", "待补充", "待确认"]:
        if bad in lower:
            defects.append(f"存在占位词: {bad}")

    scoped_text = text.split("## Original Requirement Coverage")[0]
    for bad in ["推荐", "可选", "通常", "视情况", "可以考虑"]:
        if bad in scoped_text:
            defects.append(f"存在评估歧义措辞: {bad}")

    for bad in ["p1 数据认知", "p2 任务定义", "p3 数据清洗", "autorealize"]:
        if bad in lower:
            defects.append(f"检测到面向系统内部的流程描述: {bad}")

    return defects


def eval_ambiguity_defects(text: str) -> list[str]:
    """评估协议零上下文歧义检查（规则版）。"""
    defects: list[str] = []
    task_type = _extract_task_type(text)
    must_patterns = [
        r"20250430",
        r"y_true",
        r"Validation Protocol",
        r"Computation Scope",
        r"Reporting Rules",
    ]

    if "time" in task_type:
        must_patterns.extend(
            [
                r"训练窗口长度[：:=]?\\s*180\\s*天",
                r"验证窗口长度[：:=]?\\s*30\\s*天",
                r"步长[：:=]?\\s*30\\s*天",
                r"按时间顺序",
            ]
        )
    elif "class" in task_type:
        must_patterns.extend([r"Stratified K-Fold", r"k[=：:]?\\s*5", r"shuffle\\s*=\\s*True", r"random_state\\s*=\\s*20250430"])
    elif "regression" in task_type:
        must_patterns.extend([r"K-Fold", r"k[=：:]?\\s*5", r"shuffle\\s*=\\s*True", r"random_state\\s*=\\s*20250430"])

    for pattern in must_patterns:
        if not re.search(pattern, text):
            defects.append(f"缺少关键评估约束: {pattern}")

    scoped = text.split("## Original Requirement Coverage")[0]
    for word in ["推荐", "可选", "通常", "视情况", "可以考虑"]:
        if word in scoped:
            defects.append(f"存在歧义措辞: {word}")

    return defects


def apply_eval_fixes(text: str, y_true_field: str) -> str:
    """当缺少关键约束时，做最小化程序化补丁。"""
    out = text

    if "训练窗口长度" not in out and "### Data Split & Validation Protocol" in out:
        out = out.replace(
            "### Data Split & Validation Protocol\n",
            "### Data Split & Validation Protocol\n"
            "- 采用 rolling-window 严格时间切分：训练窗口长度=180天，验证窗口长度=30天，步长=30天。\n",
            1,
        )

    if "20250430" not in out and "### Validation Protocol" in out:
        out = out.replace(
            "### Validation Protocol\n",
            "### Validation Protocol\n- 固定随机种子为 `20250430`。\n",
            1,
        )

    if "y_true" not in out and "### Computation Scope" in out:
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
    """检查新文档是否显著弱化原始需求。"""
    defects: list[str] = []
    if original_requirements.strip():
        if len(text) < max(500, int(len(original_requirements) * 0.6)):
            defects.append("新 description 内容长度明显短于原始需求，可能存在信息丢失。")
        key_terms = _extract_key_terms(original_requirements)
        missing = [t for t in key_terms if t not in text]
        if len(missing) >= 8:
            defects.append(f"原始需求关键术语覆盖不足，缺失示例: {', '.join(missing[:8])}")
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


def write_cleaning_report(path: Path, action_lines: list[str]) -> None:
    lines = [
        "# 数据清洗文档",
        "",
        "## 清洗目标",
        "- 修复会导致下游 AutoML 训练/评估失败的数据问题（空值、INF、非法类型、异常格式）。",
        "- 在不破坏业务主键与时间字段语义的前提下做最小改动清洗。",
        "",
        "## 清洗计划",
        "- 先识别问题列与问题类型，再按列生成可回滚脚本。",
        "- 每次改动都经过契约检查、规则监控、检查智能体审查。",
        "",
        "## 执行摘要",
        *action_lines,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_validation_protocol(task_type: str) -> str:
    tt = task_type.lower()
    if "recommendation" in tt or "ranking" in tt:
        return (
            "- 使用时间切分或用户分层切分构建验证集，固定 random_state=20250430。\n"
            "- 在每个用户上计算排序指标后取平均，最终分数=验证折均值。\n"
            "- 严禁使用验证/测试交互标签进行候选召回调参。"
        )
    if "reinforcement_learning" in tt or "optimization" in tt:
        return (
            "- 使用固定离线回放集评估策略，回放样本集合与约束规则固定不变。\n"
            "- 每次评估必须在相同随机种子(20250430)与相同约束配置下执行。\n"
            "- 最终分数取多次回放均值，并报告标准差。"
        )
    if "time" in tt:
        return (
            "- 采用 rolling-window 严格时间切分：训练窗口长度=180天，验证窗口长度=30天，步长=30天。\n"
            "- 从最早可用日期开始滚动，直到数据末端；每个窗口仅用历史训练预测未来验证。\n"
            "- 最终分数=所有验证窗口主指标的算术平均值。"
        )
    if "class" in tt:
        return (
            "- 采用 Stratified K-Fold，k=5，shuffle=True，random_state=20250430。\n"
            "- 各折保持类别分布一致，最终分数=5折主指标均值，次级比较=5折标准差（更低更优）。\n"
            "- 本任务默认官方测试集不提供公开标签；模型比较主排序依据=CV均值。"
        )
    if "regression" in tt:
        return (
            "- 采用 K-Fold，k=5，shuffle=True，random_state=20250430。\n"
            "- 最终分数=5折主指标均值，次级比较=5折标准差（更低更优）。"
        )
    return (
        "- 在历史决策数据上构建离线回放评估集。\n"
        "- 使用统一约束集评估不同策略，按主指标比较优劣。"
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
            "- 每个主键对应多个候选对象，输出用于排序的 score。\n"
            f"- 任务语义: {task_hint or task_type}。"
        )
    if "reinforcement_learning" in t or "optimization" in t:
        key = submission_columns[0] if submission_columns else id_column
        return (
            f"- 以 `{key}` 标识决策单元（订单/时段/状态）。\n"
            "- 模型输出决策动作或分配结果，并在离线回放中计算收益/成本。\n"
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
) -> str:
    submission_columns = submission_columns or []
    tt = task_type.lower()
    if ("recommendation" in tt or "ranking" in tt) and len(submission_columns) >= 3:
        return (
            f"- 输出文件必须包含 `{submission_columns[0]}`、`{submission_columns[1]}` 与排序分数列（如 `{submission_columns[-1]}`）。\n"
            "- 对同一主键可出现多行候选对象，按 score 降序用于评估。\n"
            f"- submission 规范参考: {submission_spec}"
        )
    if ("reinforcement_learning" in tt or "optimization" in tt) and len(submission_columns) >= 2:
        return (
            f"- 输出文件必须包含决策主键列 `{submission_columns[0]}` 与动作/分配列 `{submission_columns[1]}`。\n"
            "- 必须满足业务约束（容量、时窗、唯一分配等），违规记录按评估协议惩罚。\n"
            f"- submission 规范参考: {submission_spec}"
        )
    if "class" in tt and len(submission_columns) >= 3:
        return (
            f"- 输出文件必须包含 `{submission_columns[0]}` 与其余全部类别概率列。\n"
            "- 概率列集合与列顺序必须与样例 submission 完全一致。\n"
            f"- submission 规范参考: {submission_spec}"
        )
    id_repr = id_column if "," not in id_column else f"复合主键({id_column})"
    return (
        f"- 输出文件必须包含且仅包含 `{id_repr}` 与 `{target_column}`（或 submission 要求列）。\n"
        "- 列顺序必须与 submission 规范一致。\n"
        f"- submission 规范参考: {submission_spec}"
    )


def _submission_contract(
    id_column: str,
    target_column: str,
    submission_spec: str,
    task_type: str,
    submission_columns: list[str] | None = None,
) -> str:
    tt = task_type.lower()
    is_classification = "class" in tt
    spec_columns = submission_columns or _parse_submission_columns(submission_spec)
    lines = ["### File Contract", "- 文件名: `submission.csv`。"]

    if ("recommendation" in tt or "ranking" in tt) and len(spec_columns) >= 3:
        lines.append(f"- 用户键列: `{spec_columns[0]}`；候选对象列: `{spec_columns[1]}`；排序分数字段: `{spec_columns[-1]}`。")
        lines.append("- 对每个用户必须输出固定数量候选（如任务定义要求 TopK），不得重复候选对象。")
        lines.append("- score 必须可排序（数值型），禁止 NaN/Inf。")
        lines.append("- 行数按任务定义（user_count * K 或全候选对）严格校验。")
        return "\n".join(lines)
    if ("reinforcement_learning" in tt or "optimization" in tt) and len(spec_columns) >= 2:
        lines.append(f"- 决策键列: `{spec_columns[0]}`；动作/分配列: `{spec_columns[1]}`。")
        if len(spec_columns) > 2:
            lines.append(f"- 其余列: `{', '.join(spec_columns[2:])}`。")
        lines.append("- 所有动作必须满足约束，不可行动作需显式输出回退策略编码。")
        lines.append("- 行数必须覆盖全部待决策单元。")
        return "\n".join(lines)

    has_explicit_multicolumn = len(spec_columns) >= 3
    if has_explicit_multicolumn:
        lines.append("- 列定义必须严格遵循 Submission Format 中的列序与语义，不得省略或新增。")
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

    lines.append("- 行数必须等于测试集样本数；不得增删行；不得重排列。")
    return "\n".join(lines)


def _parse_submission_columns(submission_spec: str) -> list[str]:
    left = submission_spec.find("[")
    right = submission_spec.find("]", left + 1) if left >= 0 else -1
    if left < 0 or right < 0:
        return []
    raw = submission_spec[left + 1 : right]
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    return cols


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
    return "- `y_true` 来源：离线回放评估集中的真实执行结果（订单履约与成本真值）。"


def _validation_guardrail(task_type: str) -> str:
    tt = task_type.lower()
    if "time" in tt:
        return "- 时序任务切分严格按时间顺序，禁止未来信息泄漏与随机打散切分。"
    if "class" in tt:
        return "- 分类任务切分仅允许使用 Stratified K-Fold(k=5, shuffle=True, random_state=20250430)。"
    if "regression" in tt:
        return "- 回归任务切分仅允许使用 K-Fold(k=5, shuffle=True, random_state=20250430)。"
    return "- 优化/强化学习任务必须使用文档中定义的离线回放评估切分协议，不允许临时改动。"


def _extract_task_type(text: str) -> str:
    match = re.search(r"^\s*-\s*Task Type:\s*(.+?)\s*$", text, flags=re.M)
    if not match:
        return ""
    return match.group(1).strip().lower()


def _render_data_inventory(file_summaries: list[FileSummary], fallback_digest: str) -> str:
    if not file_summaries:
        return fallback_digest
    lines: list[str] = []
    for fs in file_summaries:
        lines.append(f"### {fs.path}")
        lines.append(f"- role: `{fs.role.value}`")
        lines.append(f"- summary: {fs.summary}")
        is_json = str(fs.path).lower().endswith(".json")
        tabular_candidate = bool((fs.source_metadata or {}).get("json_strategy")) or bool(fs.columns)
        if is_json and not tabular_candidate:
            lines.append("- json structure:")
            meta = fs.source_metadata or {}
            root_type = str(meta.get("type", meta.get("json_root_type", "unknown")))
            lines.append(f"  - root_type: `{root_type}`")
            paths = meta.get("json_paths_topk") or []
            if paths:
                lines.append("  - nested paths top-k:")
                for p in list(paths)[:40]:
                    lines.append(f"    - `{p}`")
            lines.append("  - 该 JSON 不适合直接表格化，建议按嵌套结构/配置语义使用。")
        elif fs.columns:
            lines.append("- data fields:")
            for col in fs.columns[:200]:
                meaning = fs.column_semantics.get(col, "字段语义待结合上下文确认")
                lines.append(f"  - `{col}`: {meaning}")
            if len(fs.columns) > 200:
                lines.append(f"  - ... 其余 {len(fs.columns) - 200} 个字段省略（详见 realize_report/data_description.md）")
        lines.append("")
    return "\n".join(lines).strip()
