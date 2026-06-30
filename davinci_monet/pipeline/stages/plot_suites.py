from __future__ import annotations

from typing import Any

from davinci_monet.config.schema import MonetConfig
from davinci_monet.core.schema_utils import dump_schema, is_schema_object
from davinci_monet.pipeline.stages.base import BaseStage, PipelineContext, StageResult, StageStatus


def _as_dict(value: Any) -> dict[str, Any]:
    if is_schema_object(value):
        return dump_schema(value, exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _plot_dicts(config: MonetConfig | dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(config, MonetConfig):
        return {name: _as_dict(plot) for name, plot in config.plots.items()}
    plots = config.get("plots", {})
    return {str(name): _as_dict(plot) for name, plot in plots.items()}


def _plot_suite_dicts(config: MonetConfig | dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(config, MonetConfig):
        return {name: _as_dict(suite) for name, suite in config.plot_suites.items()}
    suites = config.get("plot_suites", {})
    return {str(name): _as_dict(suite) for name, suite in suites.items()}


class PlotSuiteStage(BaseStage):
    """Expand configured plot suites into concrete plot specs."""

    def __init__(self) -> None:
        super().__init__("plot_suites")

    def validate(self, context: PipelineContext) -> bool:
        return bool(context.config_dict().get("plot_suites"))

    def execute(self, context: PipelineContext) -> StageResult:
        import time

        from davinci_monet.plots.suites import expand_plot_suite

        start = time.time()
        typed = context._typed_config()
        config = typed if typed is not None else context.config_dict()
        suites = _plot_suite_dicts(config)
        plots = _plot_dicts(config)
        generated: list[str] = []
        for suite_name, suite in suites.items():
            source = suite["source"]
            if source not in context.sources:
                return self._create_result(
                    StageStatus.FAILED,
                    error=f"plot suite '{suite_name}' references missing source '{source}'",
                    duration=time.time() - start,
                )
            ds = context.get_source_dataset(source)
            field_metadata = {str(name): dict(ds[name].attrs) for name in ds.data_vars}
            expanded = expand_plot_suite(
                suite_name,
                suite,
                available_fields=list(ds.data_vars),
                field_metadata=field_metadata,
            )
            plots.update(expanded)
            generated.extend(expanded)
        if typed is not None:
            payload = dump_schema(typed, exclude_none=True)
            payload["plots"] = plots
            context.config = MonetConfig(**payload)
        else:
            raw_config = context.config_dict()
            raw_config["plots"] = plots
            context.config = raw_config
        return self._create_result(
            StageStatus.COMPLETED,
            data={"expanded_plots": generated},
            duration=time.time() - start,
        )
