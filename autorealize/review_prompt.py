"""Lossless review snapshots, with immutable evidence before changing feedback."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from .models import ArtifactConsistencyReview

from .prompt_cache import STABLE_CONTEXT_TITLE, json_block, lossless_json
from .utils.safe_json import json_safe


class ReviewDimension(BaseModel):
    status: Literal["consistent", "conflict", "not_applicable", "insufficient_evidence"]
    evidence_refs: list[str]
    reasoning: str


class ReviewCoverage(BaseModel):
    task_and_time_semantics: ReviewDimension
    population_arithmetic: ReviewDimension
    metric_and_direction: ReviewDimension
    output_keys_and_columns: ReviewDimension
    physical_schema_and_document_roles: ReviewDimension
    causality_and_holdout: ReviewDimension
    authority_and_unresolved: ReviewDimension
    mirrored_artifacts: ReviewDimension


class ArtifactConsistencyAudit(ArtifactConsistencyReview):
    coverage: ReviewCoverage

    @model_validator(mode="after")
    def validate_coverage(self):
        checks = self.coverage.model_dump()
        conflicts = [check for check in checks.values() if check["status"] == "conflict"]
        if conflicts and not self.issues:
            raise ValueError("Conflict coverage requires a locatable issue")
        if conflicts and self.passed:
            raise ValueError("An unresolved coverage conflict cannot pass the review")
        if any(issue.severity == "blocking" for issue in self.issues) and self.passed:
            raise ValueError("A blocking issue cannot pass the review")
        for check in checks.values():
            if not check["reasoning"].strip():
                raise ValueError("Each review dimension needs its evidence or applicability reasoning")
        return self


REVIEW_COVERAGE_INSTRUCTIONS = """
逐项完成 coverage 中的八个维度，不得因已找到若干明显冲突而结束审查。
coverage 的每项给出短推导和可定位证据；不适用时明确说明原因，不虚构文件或字段。
population_arithmetic 必须独立推导集合的首个和末个合法索引、端点包含关系、起点数及行数。
时间窗口、切分比例、分组数量须从公式实际计算，不能因为各文件重复同一个数就视为正确。
明确区分最后一个观测时刻、预测起点和目标区间的开闭性；恰好落在分区终点的窗口须代入公式判断。
physical_schema_and_document_roles 必须区分真实表列、文档提及的业务术语、输出列和派生特征；
同时比较机器可读的 columns_scope、read_contract、read_example 与程序核验的来源/Sheet/字段。
同一编码对象的引用意味着内容相同，不意味着规则有效或来源权威；仍需检查它本身的逻辑。
保留原有 issues 与 repair_target 契约。coverage 用于证明审查覆盖，不代替具体缺陷和修复指令。
"""


EVALUATION_REVIEW_INSTRUCTIONS = """
输出前交叉核对 primary_metric、metric_direction、ranking_scope、metric_formula、computation_scope、
aggregation_rule、audit_metrics 及 validation_protocol 的一致性，不能只检查字段是否填写。
最终排序必须返回唯一有限数值，不能输出分层/元组/Pareto 或额外业务 tie-break。
标量化权重和尺度必须固定并保留来源；无官方权重时可明确披露实现假设，不得改回分层。
计算公式只用一种完整编码；不能留下候选自行选择的权重、上界或精度失败时的替代公式。
缺失输入须保留对应全部硬约束并使当前业务评分不可用，不能删除约束来得到数值。
开发集指标用于候选选择和超参数搜索是合法操作，invalid_solution_rules 也不能将其写为违规；
严格区分这种选择与基础模型在正在评分的开发标签上拟合，以及利用最终留出反馈选择。
同一分区和同一评估人口上的主排序指标不能在其他字段中又被称为“仅审计、不参与排序”。
列举审计指标时逐项排除该主排序指标；若同名指标在另一分区仅作审计，必须明确分区与用途。
逐个代入端点推导索引数量与输出行数，保留所有未裁决口径、真实业务缺口和防泄漏规则。
"""


def decode_review_snapshot(text: str) -> Any:
    envelope = json.loads(text)
    if not (isinstance(envelope, dict) and isinstance(envelope.get("_prompt_encoding"), str)
            and envelope["_prompt_encoding"].startswith("Lossless input encoding:")
            and set(envelope) == {"_prompt_encoding", "data"}):
        return envelope

    def resolve(value: Any, active: frozenset[str] = frozenset()) -> Any:
        if isinstance(value, dict) and set(value) == {"$prompt_ref"}:
            pointer = value["$prompt_ref"]
            if not isinstance(pointer, str) or not pointer.startswith("/data/") or pointer in active:
                raise ValueError("Invalid or cyclic review reference")
            target = envelope
            for part in pointer.split("/")[1:]:
                key = part.replace("~1", "/").replace("~0", "~")
                target = target[int(key)] if isinstance(target, list) else target[key]
            return resolve(target, active | {pointer})
        if ("$review_text" in envelope["_prompt_encoding"] and isinstance(value, dict)
                and set(value) == {"$review_text"}):
            return "".join(resolve(item, active) for item in value["$review_text"])
        if isinstance(value, dict):
            return {key: resolve(item, active) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item, active) for item in value]
        return value

    return resolve(envelope["data"])


def encode_review_snapshot(snapshot: Any) -> str:
    encoded = lossless_json(snapshot, sort_keys=False, canonical_matches=True, min_shared_chars=160)
    literal = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    if '"$review_text"' in literal or '"$prompt_ref"' in literal:
        return encoded

    def lines(value: Any) -> Any:
        if isinstance(value, str) and len(value) >= 512 and ("\n" in value or "\r" in value):
            # Preserve all line separators, including code indentation and CRLF.
            return {"$review_text": re.split(r"(\r\n|\r|\n)", value)}
        if isinstance(value, dict):
            return {key: lines(item) for key, item in value.items()}
        if isinstance(value, list):
            return [lines(item) for item in value]
        return value

    factored = lossless_json(lines(snapshot), sort_keys=False, canonical_matches=True, min_shared_chars=160)
    candidate = json.loads(factored)
    if not (isinstance(candidate, dict) and "_prompt_encoding" in candidate):
        return encoded
    candidate["_prompt_encoding"] += (
        " A sole $review_text object is the original complete string: concatenate its list entries"
        " in order without adding characters, resolving $prompt_ref aliases first. Newlines and code"
        " indentation are preserved. All referenced text is present in this message; no excerpts are omitted."
    )
    factored = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
    return factored if len(factored) < len(encoded) else encoded


def review_prompt_parts(*, authority: Any, stage_evidence: Any, latest_request: dict,
                        dynamic_title: str) -> tuple[str, str]:
    # Only control fields move to the tail. Every evidence value remains in the
    # snapshot, including full descriptions, nonidentical contracts and unknowns.
    control_keys = {
        "instruction", "round", "deterministic_defects", "reflection_instruction",
        "reflection_feedback", "repair_instruction", "previous_defects", "previous_issues",
        "previous_fixes", "finalizer_instruction",
    }
    snapshot = json_safe({
        "authority": authority,
        "stage_evidence": stage_evidence,
        "latest_request": {key: value for key, value in latest_request.items() if key not in control_keys},
    })
    encoded = encode_review_snapshot(snapshot)
    decoded = decode_review_snapshot(encoded)
    canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if canonical(decoded) != canonical(snapshot):
        raise ValueError("Review prompt round-trip failed; refusing to send incomplete evidence")
    controls = {key: value for key, value in latest_request.items() if key in control_keys}
    return f"{STABLE_CONTEXT_TITLE}\n{encoded}", json_block(dynamic_title, controls, sort_keys=False)
