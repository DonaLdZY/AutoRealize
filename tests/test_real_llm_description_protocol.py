from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autorealize.config import AutoRealizeConfig
from autorealize.llm.client import LLMClient
from autorealize.models import DescriptionProtocolBundle, FileRole, FileSummary, ProblemParadigmReview
from autorealize.pipeline import AutoRealizePipeline
from autorealize.prompts.manager import PromptManager
from autorealize.report_writer import build_data_access_protocol, render_description_protocol_markdown


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


pytestmark = [
    pytest.mark.real_llm,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (
            _truthy_env("AUTOREALIZE_REAL_LLM_TESTS")
            or _truthy_env("AUTOREALIZE_REAL_LLM_FULL_PIPELINE")
        ),
        reason="set AUTOREALIZE_REAL_LLM_TESTS=1 or AUTOREALIZE_REAL_LLM_FULL_PIPELINE=1 to run real LLM integration tests",
    ),
]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _real_cfg(tmp_path: Path) -> AutoRealizeConfig:
    cfg = AutoRealizeConfig.from_env()
    if not cfg.llm.api_key:
        pytest.skip("DEEPSEEK_API_KEY is required for real LLM integration tests")
    cfg.run_root = tmp_path / "runs"
    cfg.llm.enable_cache = False
    cfg.llm.trace_cache_hits = False
    cfg.llm.max_concurrent_requests = 1
    cfg.llm.max_retries = 3
    cfg.llm.temperature = 0.0
    cfg.llm.max_tokens = 6000
    cfg.llm.request_timeout_seconds = 90.0
    cfg.switches.enable_fewshot = False
    cfg.parallel.enable_parallel_cognition = False
    cfg.parallel.enable_parallel_relations = False
    cfg.parallel.enable_parallel_probe_actions = False
    cfg.data.table_profile_sample_rows = 200
    cfg.vllm.enabled = False
    return cfg


@pytest.fixture(scope="module")
def real_llm_services(tmp_path_factory: pytest.TempPathFactory):
    cfg = AutoRealizeConfig.from_env()
    if not cfg.llm.api_key:
        pytest.skip("DEEPSEEK_API_KEY is required for real LLM integration tests")
    cfg.llm.enable_cache = False
    cfg.llm.trace_cache_hits = False
    cfg.llm.max_concurrent_requests = 1
    cfg.llm.max_retries = 3
    cfg.llm.temperature = 0.0
    cfg.llm.max_tokens = 5000
    cfg.llm.request_timeout_seconds = 90.0
    cfg.switches.enable_fewshot = False
    report_dir = tmp_path_factory.mktemp("real_llm_protocol_smoke")
    client = LLMClient(cfg, report_dir)
    client.health_check()
    return cfg, client, PromptManager(cfg), report_dir


def _classify_problem(
    *,
    services,
    task_hint: str,
    original_requirements: str,
    data_digest: str,
    downstream_context: dict,
    prompt_name: str,
) -> ProblemParadigmReview:
    _cfg, client, prompt_mgr, _report_dir = services
    payload = {
        "task_hint": task_hint,
        "original_requirements": original_requirements,
        "data_cognition_digest": data_digest,
        "downstream_context": downstream_context,
    }
    return client.ask_structured(
        model_cls=ProblemParadigmReview,
        system_prompt=prompt_mgr.load("system/problem_paradigm_classifier.md"),
        user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        prompt_name=prompt_name,
    )


def _protocol_bundle(
    *,
    services,
    paradigm: str,
    review: ProblemParadigmReview,
    original_requirements: str,
    data_digest: str,
    data_access,
    downstream_context: dict,
    prompt_name: str,
) -> DescriptionProtocolBundle:
    _cfg, client, prompt_mgr, _report_dir = services
    prompt_by_paradigm = {
        "ml_dl_prediction": "system/ml_dl_description_protocol.md",
        "static_optimization": "system/optimization_description_protocol.md",
        "reinforcement_learning": "system/rl_description_protocol.md",
        "hybrid_ml_optimization": "system/hybrid_description_protocol.md",
    }
    payload = {
        "problem_paradigm_review": review.model_dump(),
        "authoritative_context": {
            "priority_order": ["original task documents", "official sample/output contract", "data profiles"],
            "do_not_invent": [
                "Do not invent sample_submission columns.",
                "Do not force id,target for optimization or RL.",
            ],
        },
        "original_requirements": original_requirements,
        "data_cognition_digest": data_digest,
        "deterministic_data_access": data_access.model_dump(),
        "downstream_context": downstream_context,
    }
    bundle = client.ask_structured(
        model_cls=DescriptionProtocolBundle,
        system_prompt=prompt_mgr.load(prompt_by_paradigm[paradigm]),
        user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        prompt_name=prompt_name,
    )
    if not bundle.data_access.files:
        bundle.data_access = data_access
    return bundle


def test_real_llm_protocol_smoke_classifies_and_renders_ml_optimization_rl(real_llm_services) -> None:
    ml_files = [
        FileSummary(
            path="trainset.csv",
            role=FileRole.raw_data_table,
            summary="Whitespace-separated training table with session-level click labels.",
            columns=["session_id", "user_id", "exposed_items", "clicked_items", "dwell_seconds", "click_label"],
            source_metadata={
                "csv_dialect": {
                    "sep": r"\s+",
                    "engine": "python",
                    "inferred": True,
                    "reason": "whitespace_columns_with_comma_lists",
                }
            },
        ),
        FileSummary(
            path="testset.csv",
            role=FileRole.raw_data_table,
            summary="Whitespace-separated prediction table without click_label.",
            columns=["session_id", "user_id", "exposed_items", "dwell_seconds"],
            source_metadata={
                "csv_dialect": {
                    "sep": r"\s+",
                    "engine": "python",
                    "inferred": True,
                    "reason": "whitespace_columns_with_comma_lists",
                }
            },
        ),
    ]
    ml_original = (
        "Train a supervised binary classifier. trainset.csv is whitespace-separated and contains "
        "click_label. testset.csv has the same feature columns without click_label. Official output "
        "columns are session_id, click_label. Evaluate ROC AUC; larger is better."
    )
    ml_access = build_data_access_protocol(ml_files)
    ml_review = _classify_problem(
        services=real_llm_services,
        task_hint="supervised click_label prediction",
        original_requirements=ml_original,
        data_digest="trainset.csv has labels; testset.csv is unlabeled; sample columns are session_id, click_label.",
        downstream_context={"submission_columns": ["session_id", "click_label"], "target_column": "click_label"},
        prompt_name="real_llm_smoke_problem_paradigm_ml",
    )
    assert ml_review.problem_paradigm == "ml_dl_prediction"
    ml_bundle = _protocol_bundle(
        services=real_llm_services,
        paradigm="ml_dl_prediction",
        review=ml_review,
        original_requirements=ml_original,
        data_digest="trainset/testset click prediction tables.",
        data_access=ml_access,
        downstream_context={"submission_columns": ["session_id", "click_label"], "target_column": "click_label"},
        prompt_name="real_llm_smoke_protocol_ml",
    )
    ml_text = render_description_protocol_markdown(ml_bundle, ml_files)
    assert "pd.read_csv" in ml_text
    assert "sep=r'\\s+'" in ml_text
    assert ml_bundle.ml_dl.target or ml_bundle.output.columns

    opt_files = [
        FileSummary(
            path="orders.csv",
            role=FileRole.raw_data_table,
            summary="Orders requiring carrier assignment.",
            columns=["order_id", "zone", "weight_kg", "volume_m3"],
        ),
        FileSummary(
            path="carrier_costs.csv",
            role=FileRole.raw_data_table,
            summary="Carrier cost and feasibility table.",
            columns=["carrier_id", "vehicle_type", "served_zone", "max_weight_kg", "base_cost", "cost_per_kg"],
        ),
    ]
    opt_original = (
        "This is a static optimization task. Assign every order to one feasible carrier and vehicle. "
        "Minimize total transport cost plus penalties for missing, duplicated, or infeasible assignments. "
        "There is no official sample_submission.csv; document a solution protocol instead of inventing id,target."
    )
    opt_access = build_data_access_protocol(opt_files)
    opt_review = _classify_problem(
        services=real_llm_services,
        task_hint="static carrier assignment optimization minimizing cost",
        original_requirements=opt_original,
        data_digest="orders.csv and carrier_costs.csv define a one-shot constrained assignment instance.",
        downstream_context={"generate_sample_submission": False, "submission_columns": []},
        prompt_name="real_llm_smoke_problem_paradigm_opt",
    )
    assert opt_review.problem_paradigm == "static_optimization"
    opt_bundle = _protocol_bundle(
        services=real_llm_services,
        paradigm="static_optimization",
        review=opt_review,
        original_requirements=opt_original,
        data_digest="Orders, carrier feasibility, and cost tables.",
        data_access=opt_access,
        downstream_context={"generate_sample_submission": False, "submission_columns": []},
        prompt_name="real_llm_smoke_protocol_opt",
    )
    assert opt_bundle.optimization.objective.strip()
    assert opt_bundle.optimization.hard_constraints
    assert not opt_bundle.output.sample_submission_required

    rl_files = [
        FileSummary(
            path="demand_sequences.csv",
            role=FileRole.raw_data_table,
            summary="Offline replay episodes for inventory control.",
            columns=[
                "episode_id",
                "day",
                "inventory_on_hand",
                "demand_forecast",
                "realized_demand",
                "reward_inputs",
            ],
        )
    ]
    rl_original = (
        "This is a reinforcement learning inventory-control task. State includes inventory, demand forecast, "
        "and day of week. Action is order quantity 0..5. Transition advances the episode one day. Reward is "
        "revenue minus holding, stockout, and order costs. Terminal after 14 days. Evaluate average cumulative "
        "reward over fixed replay episodes. No official sample_submission.csv; expose policy(state)->action."
    )
    rl_access = build_data_access_protocol(rl_files)
    rl_review = _classify_problem(
        services=real_llm_services,
        task_hint="reinforcement learning state action transition reward episode policy",
        original_requirements=rl_original,
        data_digest="demand_sequences.csv contains replay episodes for a policy evaluation environment.",
        downstream_context={"generate_sample_submission": False, "submission_columns": []},
        prompt_name="real_llm_smoke_problem_paradigm_rl",
    )
    assert rl_review.problem_paradigm == "reinforcement_learning"
    rl_bundle = _protocol_bundle(
        services=real_llm_services,
        paradigm="reinforcement_learning",
        review=rl_review,
        original_requirements=rl_original,
        data_digest="Offline replay episodes for inventory-control policy evaluation.",
        data_access=rl_access,
        downstream_context={"generate_sample_submission": False, "submission_columns": []},
        prompt_name="real_llm_smoke_protocol_rl",
    )
    assert rl_bundle.rl.state.strip()
    assert rl_bundle.rl.action.strip()
    assert rl_bundle.rl.reward.strip()
    assert rl_bundle.rl.terminal_condition.strip()
    assert not rl_bundle.output.sample_submission_required

    hybrid_files = [
        FileSummary(
            path="history.csv",
            role=FileRole.raw_data_table,
            summary="Historical demand and delivery outcomes for forecasting.",
            columns=["day", "zone", "orders", "late_rate", "cost"],
        ),
        FileSummary(
            path="capacity.csv",
            role=FileRole.raw_data_table,
            summary="Vehicle and carrier capacity constraints for final allocation.",
            columns=["carrier_id", "zone", "vehicle_count", "max_orders"],
        ),
    ]
    hybrid_original = (
        "This is a hybrid prediction plus optimization task. First forecast next-day demand and delay risk "
        "by zone from history.csv. Then use those predictions with capacity.csv to produce a feasible carrier "
        "allocation plan. The final score is total allocation cost plus penalties for unmet demand and capacity "
        "violations; lower is better. Prediction error is diagnostic only."
    )
    hybrid_access = build_data_access_protocol(hybrid_files)
    hybrid_review = _classify_problem(
        services=real_llm_services,
        task_hint="先预测需求和延迟风险，再优化承运商分配方案",
        original_requirements=hybrid_original,
        data_digest="history.csv provides prediction history; capacity.csv provides optimization constraints.",
        downstream_context={"generate_sample_submission": False, "submission_columns": []},
        prompt_name="real_llm_smoke_problem_paradigm_hybrid",
    )
    assert hybrid_review.problem_paradigm == "hybrid_ml_optimization"
    hybrid_bundle = _protocol_bundle(
        services=real_llm_services,
        paradigm="hybrid_ml_optimization",
        review=hybrid_review,
        original_requirements=hybrid_original,
        data_digest="Forecast demand/risk and then optimize final allocation.",
        data_access=hybrid_access,
        downstream_context={"generate_sample_submission": False, "submission_columns": []},
        prompt_name="real_llm_smoke_protocol_hybrid",
    )
    assert hybrid_bundle.hybrid.prediction_subproblem.strip()
    assert hybrid_bundle.hybrid.decision_subproblem.strip()
    assert hybrid_bundle.hybrid.handoff.strip()
    assert hybrid_bundle.hybrid.final_objective.strip()


def _run_real_pipeline(tmp_path: Path, input_root: Path, *, task_hint: str, run_name: str) -> Path:
    cfg = _real_cfg(tmp_path)
    run_dir = AutoRealizePipeline(cfg).run(
        input_root=input_root,
        output_root=tmp_path / "runs",
        task_hint=task_hint,
        run_name=run_name,
    )
    print(f"\n[real-llm] run_dir={run_dir}")
    return run_dir


def _load_report(run_dir: Path, name: str) -> dict:
    return json.loads((run_dir / "realize_report" / name).read_text(encoding="utf-8"))


@pytest.mark.full_pipeline
@pytest.mark.skipif(
    not _truthy_env("AUTOREALIZE_REAL_LLM_FULL_PIPELINE"),
    reason="set AUTOREALIZE_REAL_LLM_FULL_PIPELINE=1 for full pipeline real LLM tests",
)
def test_real_llm_ml_dl_whitespace_csv_description_protocol(tmp_path: Path) -> None:
    input_root = tmp_path / "ml_input"
    input_root.mkdir()
    _write(
        input_root / "description.md",
        """
        # Click Prediction Task

        Train a supervised binary prediction model. `trainset.csv` is a whitespace-separated
        table; the `exposed_items` and `clicked_items` fields contain comma-separated item ids
        inside one cell. Use `trainset.csv` for training and predict `click_label` for every row
        in `testset.csv`. The official `sample_submission.csv` has columns
        `session_id, click_label`. Evaluate with ROC AUC over all test sessions; larger is better.
        Never use any future or test labels as features.
        """,
    )
    _write(
        input_root / "trainset.csv",
        """
        session_id user_id exposed_items clicked_items dwell_seconds click_label
        s001 u001 10,11,12 11 23 1
        s002 u001 11,12,13 0 5 0
        s003 u002 20,21,22 20,22 42 1
        s004 u003 30,31,32 0 4 0
        """,
    )
    _write(
        input_root / "testset.csv",
        """
        session_id user_id exposed_items clicked_items dwell_seconds
        s101 u011 10,15,18 0 7
        s102 u012 20,23,24 20 31
        """,
    )
    _write(
        input_root / "sample_submission.csv",
        """
        session_id,click_label
        s101,0
        s102,0
        """,
    )

    run_dir = _run_real_pipeline(
        tmp_path,
        input_root,
        task_hint="supervised click_label prediction with whitespace CSV input",
        run_name="real_llm_ml_dl",
    )

    paradigm = _load_report(run_dir, "problem_paradigm_report.json")
    assert paradigm["problem_paradigm"] == "ml_dl_prediction"

    data_access = _load_report(run_dir, "data_access_protocol.json")
    read_examples = "\n".join(item.get("read_example", "") for item in data_access.get("files", []))
    assert "pd.read_csv" in read_examples
    assert "sep=r'\\s+'" in read_examples

    desc = (run_dir / "description.md").read_text(encoding="utf-8")
    assert "pd.read_csv" in desc
    assert "sep=r'\\s+'" in desc
    assert (run_dir / "sample_submission.csv").exists()


@pytest.mark.full_pipeline
@pytest.mark.skipif(
    not _truthy_env("AUTOREALIZE_REAL_LLM_FULL_PIPELINE"),
    reason="set AUTOREALIZE_REAL_LLM_FULL_PIPELINE=1 for full pipeline real LLM tests",
)
def test_real_llm_static_optimization_description_protocol(tmp_path: Path) -> None:
    input_root = tmp_path / "opt_input"
    input_root.mkdir()
    _write(
        input_root / "description.md",
        """
        # Daily Carrier Assignment Optimization

        This is a static optimization problem, not a supervised prediction benchmark.
        Given one day's `orders.csv` and `carrier_costs.csv`, construct a feasible assignment
        plan for all orders. Each order must be assigned to exactly one carrier and one vehicle
        type. Feasibility depends on weight, volume, destination zone, and carrier capability.
        The objective is to minimize total transportation cost plus a penalty of 100000 for
        each missing, duplicated, or infeasible order assignment. There is no official
        sample_submission.csv in this task package; AutoRealize should document a solution
        protocol instead of inventing an id,target sample file.
        """,
    )
    _write(
        input_root / "orders.csv",
        """
        order_id,zone,weight_kg,volume_m3,delivery_day
        o001,A,12,0.8,2026-01-01
        o002,B,28,1.7,2026-01-01
        o003,A,5,0.2,2026-01-01
        """,
    )
    _write(
        input_root / "carrier_costs.csv",
        """
        carrier_id,vehicle_type,served_zone,max_weight_kg,max_volume_m3,base_cost,cost_per_kg
        c01,van,A,20,1.2,40,1.5
        c01,truck,B,60,4.0,80,1.1
        c02,van,A,18,1.0,35,1.7
        c02,truck,B,55,3.5,75,1.3
        """,
    )

    run_dir = _run_real_pipeline(
        tmp_path,
        input_root,
        task_hint="static carrier assignment optimization minimizing cost with feasibility constraints",
        run_name="real_llm_static_optimization",
    )

    paradigm = _load_report(run_dir, "problem_paradigm_report.json")
    assert paradigm["problem_paradigm"] == "static_optimization"
    assert not (run_dir / "sample_submission.csv").exists()

    bundle = _load_report(run_dir, "description_protocol_bundle.json")
    assert bundle["problem_paradigm"] == "static_optimization"
    assert bundle["optimization"]["objective"].strip()
    assert bundle["optimization"]["hard_constraints"]
    assert bundle["output"]["output_kind"] in {"solution_table", "solution", "plan", "policy"}


@pytest.mark.full_pipeline
@pytest.mark.skipif(
    not _truthy_env("AUTOREALIZE_REAL_LLM_FULL_PIPELINE"),
    reason="set AUTOREALIZE_REAL_LLM_FULL_PIPELINE=1 for full pipeline real LLM tests",
)
def test_real_llm_reinforcement_learning_description_protocol(tmp_path: Path) -> None:
    input_root = tmp_path / "rl_input"
    input_root.mkdir()
    _write(
        input_root / "description.md",
        """
        # Offline Inventory Control RL

        Learn a sequential decision policy for inventory control. Each episode lasts 14 days.
        At each step the observable state includes inventory_on_hand, demand_forecast, and
        day_of_week. The action is an integer order quantity from 0 to 5. The transition consumes
        realized demand, receives the previous replenishment, and advances to the next day.
        Reward equals sales revenue minus holding cost, stockout penalty, and order cost.
        Evaluation replays fixed demand sequences and maximizes average cumulative reward.
        Illegal actions outside 0..5 receive a -100 penalty and are clipped to the nearest legal
        action. There is no official sample_submission.csv; downstream code should expose a
        policy(state) -> action interface or equivalent saved policy.
        """,
    )
    _write(
        input_root / "demand_sequences.csv",
        """
        episode_id,day,inventory_on_hand,demand_forecast,realized_demand,price,holding_cost,stockout_penalty,order_cost
        e001,1,3,2,2,10,1,8,2
        e001,2,2,3,4,10,1,8,2
        e002,1,1,1,1,10,1,8,2
        e002,2,2,2,1,10,1,8,2
        """,
    )

    run_dir = _run_real_pipeline(
        tmp_path,
        input_root,
        task_hint="reinforcement learning inventory control with state action transition reward episode",
        run_name="real_llm_reinforcement_learning",
    )

    paradigm = _load_report(run_dir, "problem_paradigm_report.json")
    assert paradigm["problem_paradigm"] == "reinforcement_learning"
    assert not (run_dir / "sample_submission.csv").exists()

    bundle = _load_report(run_dir, "description_protocol_bundle.json")
    rl = bundle["rl"]
    assert rl["state"].strip()
    assert rl["action"].strip()
    assert rl["reward"].strip()
    assert rl["terminal_condition"].strip()
