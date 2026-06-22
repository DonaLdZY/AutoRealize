from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import AutoRealizeConfig
from .logging_utils import setup_logging
from .pipeline import AutoRealizePipeline
from .utils.safe_json import dumps_json_safe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AutoRealize CLI")
    p.add_argument("--input-root", help="原始数据目录")
    p.add_argument("--output-root", help="输出 runs 根目录")
    p.add_argument("--task", default="", help="可选自然语言业务需求；若为空则尽量从数据文档中识别")
    p.add_argument("--run-name", help="本次运行名称，不建议使用时间戳，建议使用编号+测试内容")
    p.add_argument("--config", help="JSON 配置文件路径，可覆盖默认配置")
    p.add_argument("--print-default-config", action="store_true", help="打印默认 JSON 配置")
    p.add_argument("--write-default-config", help="把默认 JSON 配置写入指定路径")
    p.add_argument("--no-cognition", action="store_true", help="跳过数据认知阶段")
    p.add_argument("--no-task-definition", action="store_true", help="跳过任务定义阶段")
    p.add_argument("--no-knowledge", action="store_true", help="关闭本地知识库写入与检索")
    p.add_argument("--no-telemetry", action="store_true", help="关闭 event_stream/current_state 遥测输出")
    p.add_argument("--no-llm-cache", action="store_true", help="关闭 LLM 响应缓存")
    p.add_argument("--llm-timeout", type=float, help="单次 LLM 请求超时时间（秒）")
    p.add_argument("--llm-concurrency", type=int, help="LLM API 最大并发请求数")
    p.add_argument("--cognition-workers", type=int, help="数据认知并行 worker 数")
    p.add_argument(
        "--auto-generate-predict-split",
        action="store_true",
        help="当没有独立预测集时自动生成 predict_split 演练集，默认关闭",
    )
    return p


def _load_config(args: argparse.Namespace) -> AutoRealizeConfig:
    cfg = AutoRealizeConfig.from_file(args.config) if args.config else AutoRealizeConfig.from_env()
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
    # Windows 控制台常见默认编码不是 UTF-8；强制重配可避免中文日志和 CLI 帮助乱码。
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(logging.INFO)

    if args.print_default_config:
        print(dumps_json_safe(AutoRealizeConfig.from_env().to_dict(), indent=2))
        return
    if args.write_default_config:
        AutoRealizeConfig.from_env().write_json(args.write_default_config)
        print(f"[AutoRealize] 默认配置已写入: {args.write_default_config}")
        return

    missing = [name for name in ["input_root", "output_root", "run_name"] if not getattr(args, name)]
    if missing:
        parser.error("缺少必要参数: " + ", ".join(["--" + x.replace("_", "-") for x in missing]))

    cfg = _load_config(args)
    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        task_hint=args.task,
        run_name=args.run_name,
    )
    print(f"[AutoRealize] 运行完成，输出目录: {run_dir}")


if __name__ == "__main__":
    main()
