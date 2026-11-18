"""Lossless review snapshots, with immutable evidence before changing feedback."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from .models import ArtifactConsistencyReview

from .prompt_cache import STABLE_CONTEXT_TITLE, json_block, lossless_json, estimate_text_tokens
from .utils.safe_json import json_safe


class ReviewDimension(BaseModel):
    status: Literal["consistent", "conflict", "not_applicable", "insufficient_evidence"]
    evidence_refs: list[str]
    reasoning: str

    @field_validator("status", mode="before")
    @classmethod
    def normalize_review_status(cls, value: Any) -> Any:
        """Accept common reviewer synonyms without weakening the gate.

        The audit contract deliberately has a small status vocabulary. Models
        occasionally emit ``warning``/``unknown`` for a dimension that cannot
        be proven either way; treating that as insufficient evidence preserves
        the review finding and keeps the overall ``passed``/issue gate intact.
        """
        normalized = str(value or "").strip().lower()
        aliases = {
            "warning": "insufficient_evidence",
            "warn": "insufficient_evidence",
            "unknown": "insufficient_evidence",
            "uncertain": "insufficient_evidence",
            "not sure": "insufficient_evidence",
            "confident": "insufficient_evidence",
            "inconsistent_but_not_confirmed_by_numbers": "insufficient_evidence",
            "ok": "consistent",
            "pass": "consistent",
            "passed": "consistent",
        }
        return aliases.get(normalized, value)


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
        insufficient = [check for check in checks.values() if check["status"] == "insufficient_evidence"]
        if not self.passed and not self.issues:
            raise ValueError("A failed review must include actionable issues")
        if conflicts and not self.issues:
            raise ValueError("Conflict coverage requires a locatable issue")
        if conflicts and self.passed:
            raise ValueError("An unresolved coverage conflict cannot pass the review")
        if insufficient and self.passed:
            raise ValueError("Insufficient coverage evidence cannot pass the review")
        if any(issue.severity == "blocking" for issue in self.issues) and self.passed:
            raise ValueError("A blocking issue cannot pass the review")
        for check in checks.values():
            if not check["reasoning"].strip():
                raise ValueError("Each review dimension needs its evidence or applicability reasoning")
        return self


REVIEW_COVERAGE_INSTRUCTIONS = """
逐项完成 coverage 中的八个维度，不得因已找到若干明显冲突而结束审查。
每个 status 必须严格使用 consistent、conflict、not_applicable、insufficient_evidence 之一；warning、confident、unknown 等不确定状态一律写 insufficient_evidence。
存在 conflict 或 insufficient_evidence 时 passed 必须为 false，并在 issues 中给出可定位、可修复的问题；不得只把问题写进 summary。
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

输出完整但紧凑的机器合同：每条规则只在最合适的字段定义一次，避免把用户原文、约束、证据和缺口逐段复制。
公式保留可执行定义；其余字段用短句和精确文件/Sheet/列名/路径表达，不写长篇推理。
evidence 仅列可定位来源，不重述规则；issues/fixes 只写当前阻塞及对应处理，assumptions 写实现默认值，
residual_issues 只保留仍未裁决的业务事实，不在多个列表重复同一条内容。
列表保留全部实质规则并合并重复项；submission_checks、leakage_guards、invalid_solution_rules、
audit_metrics、assumptions、residual_issues 和 evidence 均以每项一个可验证要点为限。
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
                        dynamic_title: str, dynamic_evidence: Any | None = None) -> tuple[str, str]:
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
    if dynamic_evidence is not None:
        controls["batch_evidence"] = dynamic_evidence
    return f"{STABLE_CONTEXT_TITLE}\n{encoded}", json_block(dynamic_title, controls, sort_keys=False)


def bounded_review_batches(*, authority: dict, stage_evidence: Any, latest_request: dict,
                           dynamic_title: str, input_budget: int) -> list[dict]:
    """Partition oversized audits without clipping any evidence values.

    Every batch repeats the original requirements and current description;
    other complete evidence branches are distributed by exact JSON paths.
    Full source files remain unchanged. A too-large indivisible value fails
    explicitly; it is never shortened to make an audit appear successful.
    """
    common_authority_keys = {'original_requirements_full', 'task_hint',
        'authoritative_submission_contract', 'authority_order', 'context_policy',
        'output_language_policy', 'schema_version'}
    common_stage_keys = {'full_scan_evidence'}
    common_request_keys = {'description_markdown', 'instruction', 'round', 'deterministic_defects'}
    common_authority = {k: v for k, v in authority.items() if k in common_authority_keys}
    common_stage = {k: v for k, v in stage_evidence.items() if k in common_stage_keys}
    common_request = {k: v for k, v in latest_request.items() if k in common_request_keys}

    def _path_label(path: tuple[Any, ...]) -> str:
        return "/".join(str(part) for part in path)

    def _assign_path(root: dict[str, Any], origin: str, path: tuple[Any, ...], value: Any) -> None:
        """Rebuild a lossless dict/list projection from a mixed JSON path."""
        if not path:
            root[origin] = value
            return
        current: Any = root[origin]
        for index, part in enumerate(path):
            last = index == len(path) - 1
            next_part = None if last else path[index + 1]
            if isinstance(current, dict):
                key = str(part) if not isinstance(part, str) else part
                if last:
                    current[key] = value
                    return
                if key not in current:
                    current[key] = [] if isinstance(next_part, int) else {}
                current = current[key]
                continue
            if not isinstance(current, list) or not isinstance(part, int):
                raise ValueError(f"Invalid mixed audit evidence path: {origin}/{_path_label(path)}")
            while len(current) <= part:
                current.append(None)
            if last:
                current[part] = value
                return
            if current[part] is None:
                current[part] = [] if isinstance(next_part, int) else {}
            current = current[part]

    def render(extra):
        partial_authority = dict(common_authority)
        batch_evidence: dict[str, Any] = {
            "authority": {},
            "stage_evidence": dict(common_stage),
            "latest_request": {},
        }
        for (origin, path), value in extra:
            _assign_path(batch_evidence, origin, path, value)
        shared_paths = (
            ['authority/' + key for key in common_authority]
            + ['stage_evidence/' + key for key in common_stage]
            + ['latest_request/' + key for key in common_request]
        )
        scope = {
            'policy': '本轮是完整审查的一个无损分片。每批重复原始需求与当前 description；batch_evidence 按 included_paths 映射原始 authority/stage_evidence/latest_request 路径。只核对本批可见证据，不因兄弟分片内容未出现而判为原文件缺失。所有分片合并通过才通过总审查。',
            'shared_paths': shared_paths,
            'included_paths': shared_paths + [
                origin + '/' + _path_label(path) for (origin, path), _ in extra
            ]}
        stable, dynamic = review_prompt_parts(authority=partial_authority, stage_evidence={},
            latest_request=common_request, dynamic_title=dynamic_title,
            dynamic_evidence={'review_batch_scope': scope, 'evidence': batch_evidence})
        tokens = estimate_text_tokens(json.dumps({'messages':[{'role':'user','content':stable},
            {'role':'user','content':dynamic}]}, ensure_ascii=False))
        return {'stable':stable, 'dynamic':dynamic, 'input_tokens':tokens,
            'included_paths':scope['included_paths']}

    if render([])['input_tokens'] > input_budget:
        raise ValueError('Common authoritative audit evidence exceeds context budget; refusing to truncate')
    pending = []
    for origin, values, common in (
        ('authority', authority, common_authority_keys),
        ('stage_evidence', stage_evidence, common_stage_keys),
        ('latest_request', latest_request, common_request_keys),
    ):
        if isinstance(values, dict):
            pending.extend(
                ((origin, (str(key),)), value)
                for key, value in values.items()
                if key not in common
            )
        elif isinstance(values, (list, tuple)):
            pending.extend(
                ((origin, (index,)), value)
                for index, value in enumerate(values)
            )
        else:
            pending.append(((origin, ()), values))
    batches, current = [], []
    while pending:
        item = pending.pop(0)
        if render(current + [item])['input_tokens'] <= input_budget:
            current.append(item)
            continue
        if current:
            batches.append(render(current));current=[]
            pending.insert(0,item)
            continue
        (origin, path), value = item
        if isinstance(value, dict) and value:
            pending[0:0] = [
                ((origin, path + (str(key),)), child)
                for key, child in value.items()
            ]
            continue
        if isinstance(value, list) and value:
            pending[0:0] = [
                ((origin, path + (index,)), child)
                for index, child in enumerate(value)
            ]
            continue
        raise ValueError(
            'Indivisible audit evidence exceeds context budget: '
            + origin + '/' + _path_label(path)
        )
    if current or not batches:
        batches.append(render(current))
    return batches
