from __future__ import annotations

from dataclasses import dataclass, field
import logging

import pandas as pd

from ..config import AutoRealizeConfig
from ..logging_utils import log_event
from .sandbox import SandboxResult, run_code

logger = logging.getLogger(__name__)

@dataclass
class LevelRecord:
    level: str
    rows: int
    success: bool
    error: str = ""
    exec_seconds: float = 0.0
    output_shape: tuple[int, int] = (0, 0)


@dataclass
class ProgressiveResult:
    success: bool
    final_df: pd.DataFrame | None
    records: list[LevelRecord] = field(default_factory=list)
    refinements: int = 0
    confidence_score: float = 0.0


class ProgressiveSampler:
    def __init__(self, config: AutoRealizeConfig) -> None:
        self.config = config

    def _levels(self, full_rows: int) -> list[tuple[str, int]]:
        return [
            ("XS", min(self.config.sampling.xs_rows, full_rows)),
            ("S", min(self.config.sampling.s_rows, full_rows)),
            ("M", min(self.config.sampling.m_rows, full_rows)),
            ("FULL", full_rows),
        ]

    def _confidence(self, records: list[LevelRecord]) -> float:
        ok_records = [r for r in records if r.success]
        if len(ok_records) < 2:
            return 0.0
        # schema stability
        schemas = {r.output_shape[1] for r in ok_records}
        schema_stability = 1.0 if len(schemas) == 1 else 0.6
        # row pattern stability
        row_sizes = [r.output_shape[0] for r in ok_records]
        monotonic = all(row_sizes[i] <= row_sizes[i + 1] for i in range(len(row_sizes) - 1))
        row_stability = 0.95 if monotonic else 0.65
        # time scaling (粗粒度)
        times = [max(r.exec_seconds, 1e-6) for r in ok_records]
        time_scaling = 0.95 if all(times[i] <= times[i + 1] * 2.5 for i in range(len(times) - 1)) else 0.7
        # level depth
        depth = len(ok_records)
        depth_score = 0.95 if depth >= 3 else (0.6 if depth == 2 else 0.3)
        return float(schema_stability * row_stability * time_scaling * depth_score)

    def run(self, code: str, df: pd.DataFrame) -> ProgressiveResult:
        levels = self._levels(len(df))
        records: list[LevelRecord] = []
        refinements = 0
        final_df: pd.DataFrame | None = None

        for level_name, nrows in levels:
            log_event(logger, "progressive_sampler", "LEVEL_STARTED", level=level_name, rows=nrows)
            sample_df = df.head(nrows).copy()
            result: SandboxResult = run_code(code, sample_df)
            records.append(
                LevelRecord(
                    level=level_name,
                    rows=nrows,
                    success=result.success,
                    error=result.error[:1200],
                    exec_seconds=result.exec_seconds,
                    output_shape=result.output_df.shape if result.output_df is not None else (0, 0),
                )
            )
            if not result.success:
                log_event(
                    logger,
                    "progressive_sampler",
                    "LEVEL_FAILED",
                    level=level_name,
                    rows=nrows,
                    error=result.error[:160],
                )
                return ProgressiveResult(
                    success=False,
                    final_df=None,
                    records=records,
                    refinements=refinements,
                    confidence_score=self._confidence(records),
                )
            final_df = result.output_df
            score = self._confidence(records)
            log_event(
                logger,
                "progressive_sampler",
                "LEVEL_COMPLETED",
                level=level_name,
                rows=nrows,
                confidence=f"{score:.4f}",
            )
            if (
                self.config.switches.enable_confidence_early_stop
                and level_name != "FULL"
                and score >= self.config.sampling.confidence_commit_threshold
            ):
                # 置信度足够时直接全量执行一次确认
                log_event(
                    logger,
                    "progressive_sampler",
                    "EARLY_STOP_TRIGGERED",
                    level=level_name,
                    confidence=f"{score:.4f}",
                )
                full_result = run_code(code, df)
                records.append(
                    LevelRecord(
                        level="FULL_COMMIT",
                        rows=len(df),
                        success=full_result.success,
                        error=full_result.error[:1200],
                        exec_seconds=full_result.exec_seconds,
                        output_shape=full_result.output_df.shape if full_result.output_df is not None else (0, 0),
                    )
                )
                log_event(
                    logger,
                    "progressive_sampler",
                    "FULL_COMMIT_COMPLETED",
                    success=full_result.success,
                    rows=len(df),
                )
                return ProgressiveResult(
                    success=full_result.success,
                    final_df=full_result.output_df,
                    records=records,
                    refinements=refinements,
                    confidence_score=self._confidence(records),
                )
        log_event(logger, "progressive_sampler", "RUN_COMPLETED", success=bool(final_df is not None), levels=len(records))
        return ProgressiveResult(
            success=final_df is not None,
            final_df=final_df,
            records=records,
            refinements=refinements,
            confidence_score=self._confidence(records),
        )
