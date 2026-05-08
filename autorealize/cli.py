from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import AutoRealizeConfig
from .logging_utils import setup_logging
from .pipeline import AutoRealizePipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AutoRealize CLI")
    p.add_argument("--input-root", required=True, help="鍘熷鏁版嵁鐩綍")
    p.add_argument("--output-root", required=True, help="杈撳嚭鏍圭洰褰?)
    p.add_argument("--task", required=True, help="鐢ㄦ埛浠诲姟鎻忚堪锛屼緥濡傦細棰勬祴涓嬩釜鏈堥攢閲?)
    p.add_argument("--run-name", required=True, help="杩愯鍚嶇О锛堝缓璁紪鍙?淇鐐癸級")
    p.add_argument("--no-cleaning", action="store_true", help="鍏抽棴鏁版嵁娓呮礂")
    p.add_argument("--offline", action="store_true", help="绂荤嚎妯″紡锛堜笉璋冪敤 LLM API锛?)
    p.add_argument(
        "--auto-generate-predict-split",
        action="store_true",
        help="鑻ョ己灏戠嫭绔嬮娴嬮泦锛岃嚜鍔ㄤ粠璁粌鏁版嵁鐢熸垚 predict_split锛堥粯璁ゅ叧闂級",
    )
    p.add_argument(
        "--parallel-cleaning",
        action="store_true",
        help="鍚敤鏂囦欢绾у苟琛屾竻娲楋紙榛樿鍏抽棴锛?,
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(logging.INFO)

    cfg = AutoRealizeConfig.from_env()
    if args.no_cleaning:
        cfg.switches.run_data_cleaning = False
    if args.offline:
        cfg.llm.api_key = None
    if args.auto_generate_predict_split:
        cfg.data.auto_generate_predict_split = True
    if args.parallel_cleaning:
        cfg.parallel.enable_parallel_cleaning = True

    pipeline = AutoRealizePipeline(cfg)
    run_dir = pipeline.run(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        task_hint=args.task,
        run_name=args.run_name,
    )
    print(f"[AutoRealize] 瀹屾垚锛岃緭鍑虹洰褰? {run_dir}")


if __name__ == "__main__":
    main()

