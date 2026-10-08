"""Detect explicit non-scalar contracts without executing model-authored formulas."""

from __future__ import annotations

import re
from typing import Any, Mapping


def scalar_score_shape_defects(evaluation: Mapping[str, Any]) -> list[str]:
    defects: list[str] = []
    if evaluation.get("score_representation", "scalar") != "scalar":
        defects.append("evaluation_contract score_representation must be scalar; compile all ranking objectives into one finite numeric metric_formula")
    # Legacy contracts have no shape field. Reject explicit refusals to scalarize,
    # including the phrase that previously matched the positive 'scalar' heuristic.
    formula = str(evaluation.get("metric_formula") or evaluation.get("scalar_score_formula") or "").strip()
    non_scalar_declaration = (
        r"(?:不合成|不合并|不转换为|不转化为|不采用|不使用|不引入|无需|拒绝)\s*(?:一个|任何)?\s*(?:单一|唯一)?\s*(?:标量|数值指标|数值分数)"
        r"|(?:no|without)\s+(?:a\s+|any\s+)?(?:single\s+)?scalar(?:ization|\s+score)?\b"
        r"|(?:do\s+not|never)\s+(?:scalarize|combine\s+(?:\w+\s+){0,4}into\s+(?:a\s+)?(?:single\s+)?scalar)"
    )
    tuple_return = r"(?:^|\b(?:return|score)\s*(?:\([^)]*\))?\s*[:=]?\s*)[\[(]\s*-?\w+\s*,\s*-?\w+(?:\s*,\s*-?\w+)*\s*[\])]\s*[.;。；]?$"
    primary = re.escape(str(evaluation.get("primary_metric") or "score"))
    final_assignment = rf"(?:\bscore|\bfinal_score|{primary})\s*(?:\([^)]*\))?\s*(?:=|＝|:=)(?!=)"
    layered = r"字典序|第一层.*第二层|先(?:最大化|最小化|比较).*再(?:最大化|最小化|比较)|\blexicographic\b|\bfirst\s+(?:maximize|minimize).*\bthen\b"
    ordering_without_score = re.search(layered, formula, re.IGNORECASE | re.DOTALL) and not re.search(final_assignment, formula, re.IGNORECASE)
    if (re.search(non_scalar_declaration, formula, re.IGNORECASE)
            or re.search(tuple_return, formula, re.IGNORECASE) or ordering_without_score):
        defects.append("evaluation_contract final metric_formula must return one numeric scalar, not a tuple, vector or refusal to scalarize")
    return defects
