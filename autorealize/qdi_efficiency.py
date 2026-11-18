"""Evidence-preserving QDI cost controls; no provider calls in this module."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import pandas as pd

from .models import InvestigationStepResult, InvestigationToolRequest, ReadonlyPythonRequest


def request_signature(request: InvestigationToolRequest) -> str:
    # Question IDs and prose describe consumers, not the computation identity.
    script = request.custom_python
    payload = {
        "tool": request.tool_name, "params": request.params,
        "code": script.python_code, "files": sorted(script.input_files),
        "sheets": script.focus_sheets, "columns": script.focus_columns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class QDIEvidenceCache:
    """Cache successful sealed checks by exact code, runtime, and input contents.

    Custom code may be nondeterministic and is not cached. For sealed compiled
    checks, hash every actual input, or decline caching above the hashing budget.
    Never trust file size/mtime alone or a script's claim of determinism.
    """

    SCHEMA_VERSION = "autorealize.qdi_evidence_cache.v1"
    TTL_SECONDS = 30 * 24 * 60 * 60
    MAX_ENTRIES = 1000

    def __init__(
        self,
        root: Path,
        max_bytes: int = 64 * 1024 * 1024,
        shared_cache_dir: Path | None = None,
    ):
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.shared_cache_dir = (
            Path(shared_cache_dir).resolve() / "qdi-evidence-v1"
            if shared_cache_dir is not None else None
        )
        self.results: dict[str, Any] = {}
        self.hits = 0
        self.local_hits = 0
        self.shared_hits = 0

    def key(self, request: InvestigationToolRequest) -> str | None:
        # Arbitrary Python may depend on time, randomness, files outside its
        # declared list, or filesystem metadata. Cache only our sealed templates.
        if not is_compiled_check(request):
            return None
        try:
            files = []
            for name in sorted(set(request.custom_python.input_files)):
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root) or not path.is_file():
                    return None
                files.append(path)
            if not files or sum(p.stat().st_size for p in files) > self.max_bytes:
                return None
            input_fingerprints = []
            for path in files:
                before = path.stat()
                file_digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        file_digest.update(chunk)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    return None
                input_fingerprints.append([
                    path.relative_to(self.root).as_posix(), after.st_size, file_digest.hexdigest(),
                ])
            runtime = {
                "python": sys.version,
                "platform": sys.platform,
                "pandas": pd.__version__,
                "openpyxl": _package_version("openpyxl"),
                "xlrd": _package_version("xlrd"),
                "pyxlsb": _package_version("pyxlsb"),
                "odfpy": _package_version("odfpy"),
            }
            payload = {
                "schema": self.SCHEMA_VERSION,
                "request": request_signature(request),
                "runtime": runtime,
                "inputs": input_fingerprints,
            }
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        except (OSError, ValueError):
            return None

    def get(self, key: str | None, request: InvestigationToolRequest):
        if key is None:
            return None
        original = self.results.get(key)
        tier = "run"
        if original is None and self.shared_cache_dir is not None:
            original = self._read_shared(key)
            if original is not None:
                tier = "shared_project"
                self.shared_hits += 1
        elif original is not None:
            self.local_hits += 1
        if original is None:
            return None
        result = original.model_copy(deep=True, update={
            "request_id": request.request_id, "question_id": request.question_id,
            "reason": request.reason,
        })
        result.result["_evidence_reuse"] = {
            "source_request_id": original.request_id, "source_question_id": original.question_id,
            "content_fingerprint": key, "scope": tier,
        }
        self.hits += 1
        return result

    def put(self, key: str | None, result: Any):
        if key is not None and result.status == "completed":
            # Per-check errors are useful evidence, but must remain retryable.
            checks = result.result.get("checks", [])
            if checks and all(item.get("status") == "completed" for item in checks):
                self.results[key] = result.model_copy(deep=True)
                self._write_shared(key, result)

    def _cache_path(self, key: str) -> Path:
        assert self.shared_cache_dir is not None
        return self.shared_cache_dir / f"{key}.json"

    def _read_shared(self, key: str) -> InvestigationStepResult | None:
        if self.shared_cache_dir is None:
            return None
        path = self._cache_path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            created_at = float(payload.get("created_at") or 0)
            if (payload.get("schema_version") != self.SCHEMA_VERSION
                    or payload.get("key") != key
                    or created_at <= 0
                    or time.time() - created_at > self.TTL_SECONDS):
                path.unlink(missing_ok=True)
                return None
            result = InvestigationStepResult.model_validate(payload.get("result"))
            checks = result.result.get("checks", [])
            if result.status != "completed" or not checks or not all(
                item.get("status") == "completed" for item in checks
            ):
                return None
            return result
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def _write_shared(self, key: str, result: InvestigationStepResult) -> None:
        if self.shared_cache_dir is None:
            return
        temporary = None
        try:
            self.shared_cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.shared_cache_dir,
                prefix="qdi-check-", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump({
                    "schema_version": self.SCHEMA_VERSION,
                    "key": key,
                    "created_at": time.time(),
                    "result": result.model_dump(mode="json"),
                }, stream, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, self._cache_path(key))
            self._prune_shared_cache()
        except (OSError, TypeError, ValueError):
            return
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _prune_shared_cache(self) -> None:
        if self.shared_cache_dir is None:
            return
        entries = sorted(
            self.shared_cache_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in entries[self.MAX_ENTRIES:]:
            stale.unlink(missing_ok=True)


class QDINodeResultCache:
    """Reuse only exact, validated QDI decisions; never cache tool execution."""

    SCHEMA_VERSION = "autorealize.qdi_node_result.v2"
    TTL_SECONDS = 7 * 24 * 60 * 60
    MAX_ENTRIES = 256

    def __init__(self, shared_cache_dir: Path | None = None) -> None:
        self.shared_cache_dir = (
            Path(shared_cache_dir).resolve() / "qdi-nodes-v1"
            if shared_cache_dir is not None else None
        )
        self._memory: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.run_hits = 0
        self.shared_hits = 0
        self.misses = 0
        self.writes = 0
        self.skipped_unconfirmed = 0
        self.invalidations = 0

    @staticmethod
    def key(*, model_cls: Any, llm_client: Any, request: dict[str, Any]) -> str:
        config = getattr(llm_client, "config", None)
        llm = getattr(config, "llm", None)
        payload = {
            "cache_schema": QDINodeResultCache.SCHEMA_VERSION,
            "model_type": f"{model_cls.__module__}.{model_cls.__qualname__}",
            "output_schema": model_cls.model_json_schema(),
            "request": request,
            "inference": {
                "base_url": getattr(llm_client, "base_url", ""),
                "model": getattr(llm, "model_name", ""),
                "temperature": getattr(llm, "temperature", None),
                "enable_thinking": getattr(llm, "enable_thinking", None),
                "reasoning_effort": getattr(llm, "reasoning_effort", None),
                "structured_disable_thinking": getattr(llm, "structured_disable_thinking", None),
                "control_language": getattr(llm, "control_language", ""),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key: str, model_cls: Any) -> Any | None:
        memory_payload = self._memory.get(key)
        if memory_payload is not None:
            try:
                result = model_cls.model_validate(memory_payload)
                self.hits += 1
                self.run_hits += 1
                return result
            except (ValueError, TypeError, AttributeError):
                self._memory.pop(key, None)
                self.invalidations += 1

        if self.shared_cache_dir is not None:
            path = self._cache_path(key)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                created_at = float(payload.get("created_at") or 0)
                if (payload.get("schema_version") != self.SCHEMA_VERSION
                        or payload.get("key") != key
                        or payload.get("model_type") != f"{model_cls.__module__}.{model_cls.__qualname__}"
                        or created_at <= 0
                        or time.time() - created_at > self.TTL_SECONDS):
                    path.unlink(missing_ok=True)
                    self.invalidations += 1
                else:
                    result = model_cls.model_validate(payload.get("result"))
                    serialized = result.model_dump(mode="json")
                    self._memory[key] = serialized
                    self.hits += 1
                    self.shared_hits += 1
                    return result
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, AttributeError):
                path.unlink(missing_ok=True)
                self.invalidations += 1
        self.misses += 1
        return None

    def put(self, key: str, result: Any, *, cacheable: bool) -> bool:
        if not cacheable:
            self.skipped_unconfirmed += 1
            return False
        try:
            model_type = f"{result.__class__.__module__}.{result.__class__.__qualname__}"
            payload_result = result.model_dump(mode="json")
            self._memory[key] = payload_result
            if self.shared_cache_dir is not None:
                self.shared_cache_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "schema_version": self.SCHEMA_VERSION,
                    "key": key,
                    "model_type": model_type,
                    "created_at": time.time(),
                    "result": payload_result,
                }
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", dir=self.shared_cache_dir,
                        prefix="qdi-node-", suffix=".tmp", delete=False,
                    ) as stream:
                        temporary = Path(stream.name)
                        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
                    os.replace(temporary, self._cache_path(key))
                    self._prune_shared_cache()
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            self.writes += 1
            return True
        except (OSError, TypeError, ValueError, AttributeError):
            return False

    def _cache_path(self, key: str) -> Path:
        assert self.shared_cache_dir is not None
        return self.shared_cache_dir / f"{key}.json"

    def _prune_shared_cache(self) -> None:
        if self.shared_cache_dir is None:
            return
        entries = sorted(
            self.shared_cache_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in entries[self.MAX_ENTRIES:]:
            stale.unlink(missing_ok=True)

    def stats(self) -> dict[str, int]:
        return {
            "qdi_node_cache_hits": self.hits,
            "qdi_node_cache_run_hits": self.run_hits,
            "qdi_node_cache_project_hits": self.shared_hits,
            "qdi_node_cache_misses": self.misses,
            "qdi_node_cache_writes": self.writes,
            "qdi_node_cache_skipped_unconfirmed": self.skipped_unconfirmed,
            "qdi_node_cache_invalidations": self.invalidations,
        }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def extractive_answer(context: dict, proposal: Any) -> str | None:
    """Accept only a complete, exact scalar authority quote; never a paraphrase."""
    pointer = proposal.source_pointer
    parts = pointer.split("/")
    if len(parts) < 3 or parts[0] or parts[1] not in {"authoritative_memory", "constraint_memory"}:
        return None
    value: Any = context
    try:
        for part in parts[1:]:
            key = part.replace("~1", "/").replace("~0", "~")
            if any(word in key.lower() for word in ("unresolved", "uncertain", "warning", "risk", "candidate")):
                return None
            value = value[int(key)] if isinstance(value, list) else value[key]
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 3000:
        return None
    if any(word in value.lower() for word in (
        "unknown", "uncertain", "unresolved", "不确定", "未知", "未确定", "待确认", "无法确定",
    )):
        return None
    # Keep the full source sentence, including units, qualifications and negation.
    return value if proposal.quote == value else None


def question_context(context: dict, question: dict) -> dict:
    """Project table details only with explicit sources; retain all global rules.

    Keep the entire connected relation component. Unknown/omitted routes fall
    back to full context. All table IDs remain discoverable for request_context.
    """
    cards = context.get("table_cards", [])
    wanted = {str(x).replace("\\", "/") for x in question.get("candidate_files", [])}
    if not wanted or not isinstance(cards, list) or len(cards) < 3:
        return context
    if any(not isinstance(c, dict) or str(c.get("table_id", "")).startswith("__omitted_") for c in cards):
        return context
    def names(card):
        return {str(card.get(k, "")).replace("\\", "/") for k in ("table_id", "source_file")} - {""}
    selected = [c for c in cards if names(c) & wanted]
    matched = set().union(*(names(c) for c in selected)) if selected else set()
    if not wanted.issubset(matched):
        return context
    # Only recognized file-addressed relation records are safe to project.
    for relation in context.get("relations", []):
        if not isinstance(relation, dict) or not relation.get("left_file") or not relation.get("right_file"):
            return context
    while True:
        before = len(matched)
        for relation in context.get("relations", []):
            endpoints = {str(relation[k]).replace("\\", "/") for k in ("left_file", "right_file")}
            if matched & endpoints:
                matched |= endpoints
        for card in cards:
            if names(card) & matched:
                matched |= names(card)
        if len(matched) == before:
            break
    selected = [c for c in cards if names(c) & matched]
    if len(selected) == len(cards):
        return context
    result = dict(context)
    result["table_cards"] = selected
    result["table_index"] = [{k: c[k] for k in ("table_id", "source_file", "sheet_name") if k in c}
                             for c in cards]
    result["question_context_policy"] = {
        "table_details": "Explicit sources plus their connected relation component; other details are retrievable.",
        "global_authority_and_constraints": "unchanged",
        "missing_details_are_not_negative_evidence": True,
    }
    return result


def scoped_timeline(timeline: list[dict], question_id: str) -> list[dict]:
    return [entry for entry in timeline if entry.get("question_id") == question_id]


def compact_live_window(actions: list[dict], count: int) -> list[dict]:
    """Keep exact observations, refer to successful source code stored locally."""
    result = deepcopy(actions[-count:]) if count > 0 else []
    for entry in result:
        request = entry.get("request", {})
        observation = entry.get("observation", {})
        if request.get("source_artifact") and observation.get("status") == "completed":
            request.pop("python_code", None)
            request["python_code_visibility"] = "Full source is retrievable via source_artifact; execution succeeded."
    return result


def repeated_action_count(actions: list[dict]) -> int:
    """Detect exact request loops, not semantic similarity or uncertain facts."""
    if not actions:
        return 0
    def identity(entry):
        kind = entry.get("action")
        if kind not in {"request_checks", "request_script", "request_context", "search_document",
                        "read_document_chunks", "read_qdi_artifact_excerpt", "refine_current_question"}:
            return None
        request = deepcopy(entry.get("request", {}))
        for key in ("source_artifact", "notes", "question_id"):
            request.pop(key, None)
        if kind == "request_script":
            # Stored source reference identifies the complete code, including
            # parts beyond the displayed excerpt. Source ID includes sequence,
            # so use the explicit complete-code hash instead.
            code_hash = request.pop("python_code_sha256", None)
            if not code_hash:
                return None
            request = {"code_sha256": code_hash, "input_files": request.get("input_files", [])}
        return json.dumps([kind, request], sort_keys=True, ensure_ascii=False)
    target = identity(actions[-1])
    if target is None:
        return 0
    count = 0
    for action in reversed(actions):
        if identity(action) != target:
            break
        count += 1
    return count


# Fixed implementation; params are serialized as Python literals, never executed
# as code. Execution still uses the original subprocess timeout and IO guards.
_CHECK_BODY = '''
def analyze(input_dir, scratch_dir):
    import pandas as pd
    import re
    from datetime import datetime
    from pathlib import Path
    cache = {}
    def read(spec, limit, key_columns=()):
        key = str(spec) + str(limit) + str(key_columns)
        if key in cache:
            return cache[key]
        path = (Path(input_dir) / spec['file']).resolve()
        if not path.is_relative_to(Path(input_dir).resolve()):
            raise ValueError('input path outside data root')
        options = dict(header=spec['header'], nrows=limit + 1)
        if key_columns:
            # Identifiers such as 001 and NA must not silently become 1 or null.
            options.update(dtype={col: 'string' for col in key_columns}, keep_default_na=False)
        if path.suffix.lower() == '.csv':
            frame = pd.read_csv(path, encoding=spec['encoding'], sep=spec['delimiter'], skip_blank_lines=False, **options)
        elif path.suffix.lower() in ('.xlsx', '.xls'):
            frame = pd.read_excel(path, sheet_name=spec['sheet'], **options)
        else:
            raise ValueError('typed checks support CSV/XLSX/XLS; use a custom script for other formats')
        scope = dict(source=spec, rows_observed=min(len(frame), limit),
                     complete=len(frame) <= limit, row_limit=limit,
                     population='full table' if len(frame) <= limit else 'first rows only')
        frame = frame.iloc[:limit]
        cache[key] = (frame, scope)
        return frame, scope
    results = []
    for index, spec in enumerate(CHECKS):
        try:
            cols = spec['columns']
            key_columns = cols if spec['kind'] in ('key_integrity', 'join_coverage') else ()
            frame, scope = read(spec['left'], spec['row_limit'], key_columns)
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise ValueError('missing columns: ' + str(missing))
            facts = dict(columns=list(frame.columns), dtypes={str(c): str(frame[c].dtype) for c in cols})
            if spec['kind'] == 'table_profile':
                chosen = cols or list(frame.columns)[:12]
                facts['column_profile_scope'] = dict(selected=chosen, total_columns=len(frame.columns))
                facts['null_counts'] = {str(c): int(frame[c].isna().sum()) for c in chosen}
            elif spec['kind'] in ('key_integrity', 'join_coverage'):
                valid = (frame[cols].notna() & frame[cols].ne('')).all(axis=1)
                tuples = list(frame.loc[valid, cols].itertuples(index=False, name=None))
                keys = set(tuples)
                facts.update(null_key_rows=int((~valid).sum()), nonnull_key_rows=len(tuples),
                             unique_keys=len(keys), duplicate_key_rows=len(tuples)-len(keys),
                             key_policy='Literal strings; no trimming/case/numeric normalization; empty cells are missing')
                if spec['kind'] == 'join_coverage':
                    rcols = spec['right_columns']
                    right, right_scope = read(spec['right'], spec['row_limit'], rcols)
                    rvalid = (right[rcols].notna() & right[rcols].ne('')).all(axis=1)
                    rtuples = list(right.loc[rvalid, rcols].itertuples(index=False, name=None))
                    rkeys = set(rtuples)
                    matched_rows = sum(key in rkeys for key in tuples)
                    right_matched_rows = sum(key in keys for key in rtuples)
                    common = keys & rkeys
                    facts.update(right_scope=right_scope, right_null_key_rows=int((~rvalid).sum()),
                                 right_unique_keys=len(rkeys), right_duplicate_key_rows=len(rtuples)-len(rkeys),
                                 matched_left_nonnull_rows=matched_rows,
                                 left_nonnull_row_coverage=matched_rows/len(tuples) if tuples else None,
                                 matched_right_nonnull_rows=right_matched_rows,
                                 right_nonnull_row_coverage=right_matched_rows/len(rtuples) if rtuples else None,
                                 intersection_unique_keys=len(common),
                                 left_unique_key_coverage=len(common)/len(keys) if keys else None,
                                 right_unique_key_coverage=len(common)/len(rkeys) if rkeys else None,
                                 exact_key_comparison=True, null_keys_never_match=True)
            elif spec['kind'] == 'time_range':
                facts['time_columns'] = {}
                for col in cols:
                    ranges = {
                        'naive': {'minimum': None, 'maximum': None, 'rows': 0},
                        'aware_utc': {'minimum': None, 'maximum': None, 'rows': 0},
                    }
                    parse_failures = 0
                    for raw in frame[col].tolist():
                        if pd.isna(raw):
                            continue
                        raw_text = str(raw).strip()
                        if raw_text.endswith('z'):
                            raw_text = raw_text[:-1] + '+00:00'
                        try:
                            timestamp = datetime.fromisoformat(raw_text)
                        except (TypeError, ValueError):
                            has_timezone_marker = bool(
                                re.search(r'(?:Z|[+-]\\d{2}:?\\d{2}|\\b(?:UTC|GMT))$', raw_text, flags=re.IGNORECASE)
                            )
                            if has_timezone_marker:
                                parse_failures += 1
                                continue
                            parsed = pd.to_datetime(raw, format='mixed', errors='coerce')
                            if pd.isna(parsed):
                                parse_failures += 1
                                continue
                            timestamp = pd.Timestamp(parsed)
                        if timestamp.tzinfo is None:
                            group = ranges['naive']
                        else:
                            group = ranges['aware_utc']
                            # Shift the wall-clock value by its explicit offset.
                            # Named-zone conversion can read tzdata outside the sandbox.
                            offset = timestamp.utcoffset()
                            timestamp = datetime(
                                timestamp.year,
                                timestamp.month,
                                timestamp.day,
                                timestamp.hour,
                                timestamp.minute,
                                timestamp.second,
                                timestamp.microsecond,
                                fold=timestamp.fold,
                            )
                            timestamp = pd.Timestamp(timestamp) - pd.Timedelta(offset)
                        group['minimum'] = (
                            timestamp
                            if group['minimum'] is None or timestamp < group['minimum']
                            else group['minimum']
                        )
                        group['maximum'] = (
                            timestamp
                            if group['maximum'] is None or timestamp > group['maximum']
                            else group['maximum']
                        )
                        group['rows'] += 1

                    observed_groups = [key for key, value in ranges.items() if value['rows']]
                    mixed_awareness = len(observed_groups) > 1
                    range_details = {
                        key: {
                            'rows': value['rows'],
                            'min': (
                                value['minimum'].isoformat() + ('+00:00' if key == 'aware_utc' else '')
                                if value['minimum'] is not None else None
                            ),
                            'max': (
                                value['maximum'].isoformat() + ('+00:00' if key == 'aware_utc' else '')
                                if value['maximum'] is not None else None
                            ),
                        }
                        for key, value in ranges.items()
                    }
                    sole_group_key = observed_groups[0] if len(observed_groups) == 1 else None
                    sole_group = ranges[sole_group_key] if sole_group_key else None
                    facts['time_columns'][col] = {
                        'source_nulls': int(frame[col].isna().sum()),
                        'valid_timestamp_rows': sum(value['rows'] for value in ranges.values()),
                        'parse_failures': parse_failures,
                        'min': (
                            sole_group['minimum'].isoformat() + ('+00:00' if sole_group_key == 'aware_utc' else '')
                            if sole_group and sole_group['minimum'] is not None else None
                        ),
                        'max': (
                            sole_group['maximum'].isoformat() + ('+00:00' if sole_group_key == 'aware_utc' else '')
                            if sole_group and sole_group['maximum'] is not None else None
                        ),
                        'timestamp_ranges_by_timezone_awareness': range_details,
                        'mixed_timezone_awareness': mixed_awareness,
                        'timezone_policy': (
                            'preserve naive wall-clock timestamps; normalize explicit offsets to UTC; '
                            'report separate ranges when awareness is mixed'
                        ),
                    }
            results.append(dict(check_index=index, kind=spec['kind'], status='completed', scope=scope, facts=facts))
        except Exception as exc:
            results.append(dict(check_index=index, kind=spec['kind'], status='failed', error=str(exc)))
    failed_indices = [item['check_index'] for item in results if item['status'] == 'failed']
    return dict(
        checks=results,
        evidence_policy='Sampled checks cannot establish whole-population claims.',
        error=('typed checks failed at indexes ' + str(failed_indices)) if failed_indices else '',
    )
'''


def compile_checks(checks: list[Any], question_id: str) -> InvestigationToolRequest:
    if not 1 <= len(checks) <= 4:
        raise ValueError("request_checks requires 1-4 independent checks")
    payload = [check.model_dump() for check in checks]
    files = sorted({item[side]["file"] for item in payload for side in ("left", "right") if item.get(side)})
    code = "CHECKS = " + repr(payload) + "\n" + _CHECK_BODY
    return InvestigationToolRequest(
        question_id=question_id, tool_name="custom_readonly_python",
        params={"qdi_compiled_checks": payload}, reason="Deterministic bounded evidence checks",
        custom_python=ReadonlyPythonRequest(question_id=question_id, input_files=files,
            goal="Run independent typed checks; preserve each check's sampling scope", python_code=code),
    )


def is_compiled_check(request: InvestigationToolRequest) -> bool:
    from .models import QDIDeterministicCheck
    try:
        checks = [QDIDeterministicCheck.model_validate(x) for x in request.params["qdi_compiled_checks"]]
        expected = compile_checks(checks, request.question_id)
        return (request.custom_python.python_code == expected.custom_python.python_code
                and request.custom_python.input_files == expected.custom_python.input_files)
    except (KeyError, TypeError, ValueError):
        return False
