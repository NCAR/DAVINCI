"""Inspection pipeline stage."""

from __future__ import annotations

import time
from pathlib import Path

from davinci_monet.config.schema import InspectionConfig, MonetConfig
from davinci_monet.inspection import inspect_run_directory
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)


class InspectionStage(BaseStage):
    """Run deterministic inspection checks on final plot products."""

    def __init__(self) -> None:
        super().__init__("inspection")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        cfg = _inspection_config(context)
        if cfg is None or not cfg.enabled:
            return self._create_result(
                StageStatus.SKIPPED,
                data={"skipped": "inspection disabled"},
                duration=time.time() - start,
            )

        output_dir = Path(context.analysis_config().output_dir or ".")
        plot_paths = None
        plotting = context.results.get("plotting")
        if plotting and isinstance(plotting.data, dict) and "plots_generated" in plotting.data:
            plot_paths = plotting.data["plots_generated"]
        result = inspect_run_directory(
            output_dir,
            presets=cfg.presets,
            preview_format=cfg.preview_format,
            plot_paths=plot_paths,
        )
        data = {
            "passed": result.passed,
            "checks": result.checks,
            "inspection_json": str(result.json_path),
            "inspection_markdown": str(result.markdown_path),
            "inspection_previews": [str(path) for path in result.preview_paths],
        }
        if result.passed:
            return self._create_result(
                StageStatus.COMPLETED,
                data=data,
                duration=time.time() - start,
            )
        if cfg.required:
            return self._create_result(
                StageStatus.FAILED,
                data=data,
                error="required inspection failed",
                duration=time.time() - start,
            )
        return self._create_result(
            StageStatus.COMPLETED,
            data=data,
            duration=time.time() - start,
        )


def _inspection_config(context: PipelineContext) -> InspectionConfig | None:
    if isinstance(context.config, MonetConfig):
        return context.config.inspection
    if not isinstance(context.config, dict):
        return None

    section = context.config.get("inspection")
    if section is None or isinstance(section, InspectionConfig):
        return section
    return InspectionConfig(**section)
