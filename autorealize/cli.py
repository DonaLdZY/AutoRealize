from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .config import AutoRealizeConfig, DEFAULT_CONFIG_PATH
from .logging_utils import setup_logging
from .pipeline import AutoRealizePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoRealize CLI")
    parser.add_argument("--input-root", help="原始数据目录 / Source data directory")
    parser.add_argument("--output-root", help="运行输出根目录 / Run output root")
    parser.add_argument("--task", default="", help="可选业务需求 / Optional human task requirement")
    parser.add_argument("--run-name", help="本次运行名称 / Run name")
    parser.add_argument(
        "--config",
        help="YAML 配置文件路径；兼容 JSON/TOML / YAML config path; JSON/TOML remain supported",
    )
    parser.add_argument("--print-default-config", action="store_true", help="打印默认 YAML / Print default YAML")
    parser.add_argument("--write-default-config", help="写出默认 YAML / Write default YAML to path")

    # Compatibility overrides. The YAML file remains the primary source of truth.
    parser.add_argument("--no-cognition", action="store_true", help="跳过数据认知 / Skip data cognition")
    parser.add_argument("--no-task-definition", action="store_true", help="跳过任务定义 / Skip task definition")
    parser.add_argument("--no-knowledge", action="store_true", help="关闭知识库 / Disable knowledge store")
    parser.add_argument("--no-telemetry", action="store_true", help="关闭结构化遥测 / Disable telemetry")
    parser.add_argument("--no-llm-cache", action="store_true", help="关闭 LLM 缓存 / Disable LLM cache")
    parser.add_argument("--llm-timeout", type=float, help="LLM 请求超时秒数 / LLM timeout seconds")
    parser.add_argument("--llm-concurrency", type=int, help="LLM 最大并发 / Max concurrent LLM calls")
    parser.add_argument("--cognition-workers", type=int, help="认知 worker 数 / Cognition worker count")
    parser.add_argument(
        "--auto-generate-predict-split",
        action="store_true",
        help="自动生成预测演练切分 / Generate a synthetic prediction split",
    )
    return parser


def _load_config(args: argparse.Namespace) -> AutoRealizeConfig:
    config_path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIG_PATH
    cfg = AutoRealizeConfig.from_file(config_path) if config_path.exists() else AutoRealizeConfig.from_env()
    if args.no_cognition:
        cfg.switches.run_data_cognition = False
    if args.no_task_definition:
        cfg.switches.run_task_definition = False
    if args.no_knowledge:
        cfg.knowledge.enabled = False
    if args.no_telemetry:
        cfg.telemetry.enabled = False
    if args.no_llm_cache:
        cfg.llm.enable_cache = False
    if args.llm_timeout is not None:
        cfg.llm.request_timeout_seconds = max(1.0, float(args.llm_timeout))
    if args.llm_concurrency is not None:
        cfg.llm.max_concurrent_requests = max(1, int(args.llm_concurrency))
    if args.cognition_workers is not None:
        cfg.parallel.cognition_max_workers = max(1, int(args.cognition_workers))
    if args.auto_generate_predict_split:
        cfg.data.auto_generate_predict_split = True
    return cfg


def main() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args()

    if args.print_default_config:
        default_cfg = (
            AutoRealizeConfig.from_file(DEFAULT_CONFIG_PATH)
            if DEFAULT_CONFIG_PATH.exists()
            else AutoRealizeConfig.from_env()
        )
        print(yaml.safe_dump(default_cfg.to_dict(), allow_unicode=True, sort_keys=False))
        return
    if args.write_default_config:
        default_cfg = (
            AutoRealizeConfig.from_file(DEFAULT_CONFIG_PATH)
            if DEFAULT_CONFIG_PATH.exists()
            else AutoRealizeConfig.from_env()
        )
        default_cfg.write_yaml(args.write_default_config)
        print(f"[AutoRealize] Default YAML config written to: {args.write_default_config}")
        return

    missing = [name for name in ("input_root", "output_root", "run_name") if not getattr(args, name)]
    if missing:
        parser.error("Missing required arguments: " + ", ".join("--" + x.replace("_", "-") for x in missing))

    cfg = _load_config(args)
    setup_logging(
        cfg.logging.level,
        raw_event_log=cfg.logging.raw_event_log,
        noisy_logger_level=cfg.logging.noisy_logger_level,
        noisy_loggers=cfg.logging.noisy_loggers,
    )
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        task_hint=args.task,
        run_name=args.run_name,
    )
    print(f"[AutoRealize] Run completed: {run_dir}")


if __name__ == "__main__":
    main()
