import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import autorealize.pipeline as legacy
from autorealize.config import AutoRealizeConfig
from autorealize.models import FileRole, FileSummary
from autorealize.modules import data_cognition
from autorealize.modules.data_cognition import DataCognitionModule
from autorealize.modules.types import RuntimeServices


def test_completion_order_cannot_change_constraint_prefix_or_drop_evidence(tmp_path, monkeypatch):
    root = tmp_path / "data"
    sources = {
        "tables/z_orders.csv": ("volume,date,order_id\n5,2026-09-08,1\n", ["volume", "date", "order_id"]),
        "docs/b_rules.md": ("Keep historical order decisions as observations only.\nTAIL_HISTORY_RULE", []),
        "tables/a_capacity.csv": ("weight,vehicle_id,date\n1000,1,\n", ["weight", "vehicle_id", "date"]),
        "docs/a_rules.md": ("Do not assume dates for undated capacity.\nTAIL_NO_ASSUMED_DATES", []),
    }
    for relative_path, (text, _) in sources.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    ordered_paths = sorted(sources)
    results = []
    requests = []
    completions = []

    class RecordingClient:
        def __init__(self, config):
            self.config = config
            self.requests = {}

        def ask_structured(self, **kwargs):
            self.requests[kwargs["prompt_name"]] = kwargs
            return kwargs["model_cls"]()

    schedules = [
        (True, list(reversed(ordered_paths))),
        (True, [ordered_paths[index] for index in (2, 0, 3, 1)]),
        (False, ordered_paths),
    ]
    for run_index, (parallel, completion_order) in enumerate(schedules):
        config = AutoRealizeConfig()
        config.parallel.enable_parallel_cognition = parallel
        config.parallel.cognition_max_workers = len(sources)
        config.investigation.enabled = False
        client = RecordingClient(config)
        services = RuntimeServices(
            llm_client=client, prompt_mgr=None, registry=None,
            trajectory=SimpleNamespace(log=lambda *args: None),
        )
        report_dir = tmp_path / f"report-{run_index}"
        report_dir.mkdir()
        module = DataCognitionModule(config, services, report_dir)
        selected = [root / path for path in completion_order]
        monkeypatch.setattr(module, "_select_files_with_pattern_sampling", lambda _: (selected, {}, [], []))

        barrier = Barrier(len(sources), timeout=10)
        releases = {path: Event() for path in sources}
        releases[completion_order[0]].set()
        lock = Lock()
        state = {"active": 0, "max_active": 0}
        started = []
        submitted = []
        completed = []

        def cognize(*, file, **kwargs):
            relative_path = file.relative_to(root).as_posix()
            with lock:
                started.append(relative_path)
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                if parallel:
                    barrier.wait()
                    assert releases[relative_path].wait(10), f"File never released: {relative_path}"
                text, columns = sources[relative_path]
                requirement = not columns
                summary = FileSummary(
                    path=relative_path,
                    role=FileRole.task_requirement if requirement else FileRole.raw_data_table,
                    summary=text,
                    columns=columns,
                    extracted_knowledge=[f"Evidence from {relative_path}", text],
                    column_semantics={column: f"Observed {column}" for column in columns},
                )
                return {
                    "fs": summary, "rpath": relative_path, "columns": list(columns),
                    "is_requirement": requirement, "summary_text": text,
                }
            finally:
                with lock:
                    state["active"] -= 1
                    if not parallel:
                        completed.append(relative_path)

        class ScheduledExecutor(ThreadPoolExecutor):
            def submit(self, fn, file):
                relative_path = file.relative_to(root).as_posix()
                submitted.append(relative_path)
                future = super().submit(fn, file)

                def after_completion(_):
                    with lock:
                        completed.append(relative_path)
                        index = completion_order.index(relative_path) + 1
                        if index < len(completion_order):
                            releases[completion_order[index]].set()

                # Release the next file only after this actual future finishes.
                future.add_done_callback(after_completion)
                return future

        monkeypatch.setattr(legacy, "_cognize_one_file", cognize)
        monkeypatch.setattr(data_cognition, "ThreadPoolExecutor", ScheduledExecutor)
        result = module.run(root, "Build a model without assuming missing capacity dates.")

        assert sorted(started) == ordered_paths
        assert completed == completion_order
        if parallel:
            assert submitted == ordered_paths
            assert state["max_active"] == len(sources)
        else:
            assert submitted == []
            assert state["max_active"] == 1
        assert [summary.path for summary in result.file_summaries] == ordered_paths
        assert list(result.table_columns) == [path for path in ordered_paths if sources[path][1]]
        assert result.original_requirement_texts == [sources[path][0] for path in ordered_paths if not sources[path][1]]
        for summary in result.file_summaries:
            text, columns = sources[summary.path]
            assert summary.columns == columns
            assert summary.extracted_knowledge == [f"Evidence from {summary.path}", text]
            if columns:
                assert result.table_columns[summary.path] == columns
        artifacts = list((report_dir / "file_cognition").glob("*.json"))
        assert len(artifacts) == len(sources)
        assert sorted(json.loads(path.read_text(encoding="utf-8"))["path"] for path in artifacts) == ordered_paths
        results.append(result)
        requests.append(client.requests)
        completions.append(completed)

    assert completions[0] != completions[1]
    for result in results[1:]:
        assert result.file_summaries == results[0].file_summaries
        assert result.constraint_memory == results[0].constraint_memory
        assert result.authoritative_memory == results[0].authoritative_memory
        assert result.agent_context_pack == results[0].agent_context_pack
    for request in requests[1:]:
        assert request == requests[0]
    constraint_request = requests[0]["constraint_memory_extractor"]
    prefix = constraint_request["static_context_prompt"]
    assert prefix
    assert "TAIL_HISTORY_RULE" in prefix
    assert "TAIL_NO_ASSUMED_DATES" in prefix
    assert constraint_request["max_tokens"] == 32768
