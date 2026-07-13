from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import AutoRealizeConfig, DEFAULT_CONFIG_PATH, ServiceConfig


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WORKDIR = str(ROOT_DIR)


def now_ts() -> float:
    return time.time()


class StartAutoRealizeRequest(BaseModel):
    task_id: str
    input_root: str
    output_root: str
    run_name: str
    task_hint: str = ""
    config_path: str = ""
    python_executable: str = "python"
    working_dir: str = DEFAULT_WORKDIR
    auto_generate_predict_split: bool = False
    env_overrides: dict[str, str] = Field(default_factory=dict)


class StopRequest(BaseModel):
    job_id: str


class SnapshotRequest(BaseModel):
    run_dir: str


class JobStatus(BaseModel):
    job_id: str
    task_id: str
    status: str
    started_at: float
    updated_at: float
    run_dir: str | None = None
    exit_code: int | None = None
    last_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class JobRuntime:
    job_id: str
    task_id: str
    run_dir: str
    process: subprocess.Popen[str] | None
    status: str
    started_at: float
    updated_at: float
    exit_code: int | None = None
    last_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    job_status_tail_chars: int = 60000
    stop_wait_seconds: float = 15.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRuntime] = {}

    def create(self, task_id: str, run_dir: str) -> JobRuntime:
        with self._lock:
            for j in self._jobs.values():
                if j.task_id == task_id and j.status == "running":
                    proc = j.process
                    if proc is not None and proc.poll() is not None:
                        j.status = "failed" if (proc.returncode or 0) != 0 else "completed"
                        j.exit_code = proc.returncode
                        j.updated_at = now_ts()
                        continue
                    raise HTTPException(status_code=400, detail="task already running in AutoRealize service")
            job_id = uuid.uuid4().hex
            ts = now_ts()
            job = JobRuntime(
                job_id=job_id,
                task_id=task_id,
                run_dir=run_dir,
                process=None,
                status="pending",
                started_at=ts,
                updated_at=ts,
            )
            self._jobs[job_id] = job
            return job

    def _get_unlocked(self, job_id: str) -> JobRuntime:
        job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    def get(self, job_id: str) -> JobRuntime:
        with self._lock:
            return self._get_unlocked(job_id)

    def set_process(self, job_id: str, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            job = self._get_unlocked(job_id)
            job.process = proc
            job.status = "running"
            job.updated_at = now_ts()

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._get_unlocked(job_id)
            for k, v in kwargs.items():
                setattr(job, k, v)
            job.updated_at = now_ts()

    def status(self, job_id: str) -> JobStatus:
        job = self.get(job_id)
        return JobStatus(
            job_id=job.job_id,
            task_id=job.task_id,
            status=job.status,
            started_at=job.started_at,
            updated_at=job.updated_at,
            run_dir=job.run_dir,
            exit_code=job.exit_code,
            last_error=job.last_error,
            stdout_tail=_tail_text(job.stdout_tail, job.job_status_tail_chars),
            stderr_tail=_tail_text(job.stderr_tail, job.job_status_tail_chars),
        )


store = JobStore()
app = FastAPI(title="AutoRealize Service API", version="0.1.0")


def _tail_text(text: str, limit: int = 200000) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _extract_last_error_from_output(text: str, *, limit: int = 1200) -> str:
    """Pick a useful failure line instead of a trailing docs URL."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    noisy_fragments = [
        "for further information visit",
        "https://errors.pydantic.dev",
    ]
    signal_fragments = [
        "RuntimeError:",
        "ValidationError",
        "Pydantic validation failed",
        "LLM structured output failed",
        "Traceback",
        "Error:",
        "error:",
    ]
    for line in reversed(lines):
        lower = line.lower()
        if any(noise in lower for noise in noisy_fragments):
            continue
        if any(sig.lower() in lower for sig in signal_fragments):
            return line[-limit:]
    for line in reversed(lines):
        lower = line.lower()
        if not any(noise in lower for noise in noisy_fragments):
            return line[-limit:]
    return lines[-1][-limit:]


def _safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _parse_jsonl(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return rows


def _render_directory_tree(root: Path, max_nodes: int = 6000) -> str:
    if not root.exists() or not root.is_dir():
        return ""
    lines = [root.name or str(root)]
    count = 0

    def walk(node: Path, depth: int) -> None:
        nonlocal count
        if count >= max_nodes:
            return
        try:
            children = sorted(node.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except Exception:
            return
        for c in children:
            if count >= max_nodes:
                break
            suffix = "/" if c.is_dir() else ""
            lines.append(f"{'  ' * depth}- {c.name}{suffix}")
            count += 1
            if c.is_dir():
                walk(c, depth + 1)

    walk(root, 0)
    if count >= max_nodes:
        lines.append("... (truncated)")
    return "\n".join(lines)


def _load_file_cognition_index(
    report_dir: Path,
    max_items: int = 400,
    markdown_chars: int = 50000,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    folder = report_dir / "file_cognition"
    if not folder.exists() or not folder.is_dir():
        return out
    files = sorted(folder.glob("*.json"), key=lambda x: x.name.lower())
    for f in files[:max_items]:
        payload = _safe_read_json(f, {})
        if not isinstance(payload, dict):
            continue
        source = str(payload.get("path", "")).replace("\\", "/").strip()
        if not source:
            continue
        md_text = ""
        md_path = f.with_suffix(".md")
        if md_path.exists():
            md_text = md_path.read_text(encoding="utf-8", errors="ignore")
        out[source] = {
            "json": payload,
            "markdown": md_text[: max(0, markdown_chars)],
        }
    return out


def _load_run_config(report_dir: Path) -> AutoRealizeConfig:
    candidates = [report_dir / "final_config.yaml", report_dir / "final_config.json"]
    manifest = _safe_read_json(report_dir / "frontend_manifest.json", {})
    snapshot_rel = str((manifest.get("config_entrypoints") or {}).get("snapshot") or "")
    if snapshot_rel:
        candidates.insert(0, report_dir.parent / snapshot_rel)
    candidates.extend(sorted(report_dir.glob("*config*.yaml")))
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists() or path.name == "config_schema.json":
            continue
        seen.add(path)
        try:
            return AutoRealizeConfig.from_file(path)
        except Exception:
            continue
    return AutoRealizeConfig.from_env()


def _build_snapshot(run_dir_raw: str) -> dict[str, Any]:
    run_dir = Path(run_dir_raw).expanduser().resolve()
    report_dir = run_dir / "autorealize" / "realize_report"
    if not report_dir.exists():
        report_dir = run_dir / "realize_report"

    out: dict[str, Any] = {"report_dir": str(report_dir)}
    cfg = _load_run_config(report_dir)
    service_cfg = cfg.service
    autorealize_dir = report_dir.parent
    directory_tree_file = report_dir / "directory_tree.txt"

    out["current_state"] = _safe_read_json(
        report_dir / cfg.telemetry.current_state_filename,
        {},
    )
    out["frontend_manifest"] = _safe_read_json(report_dir / "frontend_manifest.json", {})
    out["run_summary"] = _safe_read_json(report_dir / "run_summary.json", {})
    out["data_cognition_report"] = _safe_read_json(report_dir / "data_cognition_report.json", {})
    out["task_definition_report"] = _safe_read_json(report_dir / "task_definition_report.json", {})
    out["evaluation_contract_report"] = _safe_read_json(report_dir / "evaluation_contract_report.json", {})
    out["main_task_protocol"] = _safe_read_json(report_dir / "main_task_protocol.json", {})
    out["automl_context_pack"] = _safe_read_json(report_dir / "automl_context_pack.json", {})
    out["authoritative_task_memory"] = _safe_read_json(report_dir / "authoritative_task_memory.json", {})
    out["agent_context_pack"] = _safe_read_json(report_dir / "agent_context_pack.json", {})
    out["retrieved_knowledge"] = _safe_read_json(report_dir / "retrieved_knowledge.json", [])
    out["events"] = _parse_jsonl(
        report_dir / cfg.telemetry.event_stream_filename,
        limit=max(1, int(service_cfg.snapshot_event_limit)),
    )
    out["directory_tree_text"] = directory_tree_file.read_text(encoding="utf-8", errors="ignore") if directory_tree_file.exists() else ""
    out["output_tree_text"] = _render_directory_tree(
        autorealize_dir,
        max_nodes=max(1, int(service_cfg.snapshot_tree_max_nodes)),
    )
    desc_file = autorealize_dir / "description.md"
    out["description_text"] = desc_file.read_text(encoding="utf-8", errors="ignore") if desc_file.exists() else ""
    data_desc_file = report_dir / "data_description.md"
    out["data_description_text"] = data_desc_file.read_text(encoding="utf-8", errors="ignore") if data_desc_file.exists() else ""
    automl_context_file = report_dir / "automl_context.md"
    out["automl_context_text"] = automl_context_file.read_text(encoding="utf-8", errors="ignore") if automl_context_file.exists() else ""
    original_file = report_dir / "original_requirements.txt"
    out["original_requirements_text"] = original_file.read_text(encoding="utf-8", errors="ignore") if original_file.exists() else ""
    out["file_cognition_index"] = _load_file_cognition_index(
        report_dir,
        max_items=max(0, int(service_cfg.snapshot_file_cognition_limit)),
        markdown_chars=max(0, int(service_cfg.snapshot_file_markdown_chars)),
    )
    return out


def _run_job(job_id: str, req: StartAutoRealizeRequest) -> None:
    run_dir = str((Path(req.output_root).expanduser().resolve() / req.run_name))
    workdir = req.working_dir.strip() or DEFAULT_WORKDIR
    config_path = Path(req.config_path).expanduser() if req.config_path.strip() else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = Path(workdir) / config_path
    try:
        service_cfg = AutoRealizeConfig.from_file(config_path).service
    except Exception:
        service_cfg = ServiceConfig()
    store.update(
        job_id,
        job_status_tail_chars=max(0, int(service_cfg.job_status_tail_chars)),
        stop_wait_seconds=max(0.0, float(service_cfg.stop_wait_seconds)),
    )
    cmd = [
        req.python_executable or "python",
        "-m",
        "autorealize.cli",
        "--input-root",
        req.input_root,
        "--output-root",
        req.output_root,
        "--task",
        req.task_hint,
        "--run-name",
        req.run_name,
        "--config",
        str(config_path),
    ]
    if req.auto_generate_predict_split:
        cmd.append("--auto-generate-predict-split")

    env = os.environ.copy()
    env.update(req.env_overrides or {})

    try:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except Exception as e:
        store.update(job_id, status="failed", last_error=f"start failed: {e}")
        return

    store.set_process(job_id, proc)
    out, err = proc.communicate()
    exit_code = proc.returncode
    status = "completed" if exit_code == 0 else "failed"
    last_error = None
    if exit_code != 0:
        tail = (err or out or "").strip()
        if tail:
            last_error = _extract_last_error_from_output(
                tail,
                limit=max(1, int(service_cfg.last_error_chars)),
            )
        else:
            last_error = f"AutoRealize exited with code {exit_code}"

    # Best effort: write service-captured output for debugging.
    try:
        report_dir = Path(run_dir) / "realize_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        if out:
            (report_dir / service_cfg.stdout_filename).write_text(
                _tail_text(out, max(0, int(service_cfg.captured_log_tail_chars))),
                encoding="utf-8",
                errors="ignore",
            )
        if err:
            (report_dir / service_cfg.stderr_filename).write_text(
                _tail_text(err, max(0, int(service_cfg.captured_log_tail_chars))),
                encoding="utf-8",
                errors="ignore",
            )
    except Exception:
        pass

    store.update(
        job_id,
        status=status,
        exit_code=exit_code,
        last_error=last_error,
        stdout_tail=_tail_text(out or "", max(0, int(service_cfg.captured_log_tail_chars))),
        stderr_tail=_tail_text(err or "", max(0, int(service_cfg.captured_log_tail_chars))),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs/start")
def start_job(req: StartAutoRealizeRequest) -> dict[str, Any]:
    run_dir = str((Path(req.output_root).expanduser().resolve() / req.run_name))
    job = store.create(task_id=req.task_id, run_dir=run_dir)
    thread = threading.Thread(target=_run_job, args=(job.job_id, req), daemon=True)
    thread.start()
    return {"job_id": job.job_id, "status": "started", "run_dir": run_dir}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return store.status(job_id).model_dump()


@app.post("/jobs/stop")
def stop_job(req: StopRequest) -> dict[str, Any]:
    job = store.get(req.job_id)
    proc = job.process
    if proc is None or proc.poll() is not None:
        return {"status": "not_running", "job_id": req.job_id}
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[arg-type]
            try:
                proc.wait(timeout=max(0.0, float(job.stop_wait_seconds)))
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            proc.terminate()
    except Exception:
        try:
            proc.kill()
        except Exception:
            proc.terminate()
    store.update(req.job_id, status="stopped", last_error="stopped by user")
    return {"status": "stopping", "job_id": req.job_id}


@app.post("/snapshot")
def snapshot(req: SnapshotRequest) -> dict[str, Any]:
    try:
        return _build_snapshot(req.run_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"snapshot failed: {e}")
