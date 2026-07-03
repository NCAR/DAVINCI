"""Pipeline runner for orchestrating analysis workflows.

This module provides the PipelineRunner class that executes a sequence
of analysis stages, managing state and handling errors.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TextIO

from tqdm import tqdm

from davinci_monet.config.schema import MonetConfig, PlotStyleConfig
from davinci_monet.core.exceptions import ConfigurationError, PipelineError
from davinci_monet.pipeline.display import ProgressFormatter
from davinci_monet.pipeline.lifecycle import PipelineResourcePolicy
from davinci_monet.pipeline.progress import create_progress_callback
from davinci_monet.pipeline.reporting import LogCollector, LogEntry
from davinci_monet.pipeline.stages import (
    BaseStage,
    PipelineContext,
    Stage,
    StageResult,
    StageStatus,
    create_standard_pipeline,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-export progress helpers for direct imports from this module.
# ---------------------------------------------------------------------------
__all__ = [
    "LogCollector",
    "LogEntry",
    "ProgressFormatter",
    "PipelineResult",
    "PipelineRunner",
    "PipelineBuilder",
    "run_analysis",
]


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution.

    Attributes
    ----------
    success
        True if all stages completed successfully.
    stage_results
        Results from each stage in execution order.
    context
        Final pipeline context with all data.
    start_time
        Pipeline start time.
    end_time
        Pipeline end time.
    total_duration_seconds
        Total execution time.
    """

    success: bool
    stage_results: list[StageResult] = field(default_factory=list)
    context: PipelineContext | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_duration_seconds: float = 0.0

    @property
    def failed_stages(self) -> list[StageResult]:
        """Get list of failed stage results."""
        return [r for r in self.stage_results if r.status == StageStatus.FAILED]

    @property
    def completed_stages(self) -> list[str]:
        """Get names of completed stages."""
        return [r.stage_name for r in self.stage_results if r.status == StageStatus.COMPLETED]

    def get_stage_result(self, stage_name: str) -> StageResult | None:
        """Get result for a specific stage."""
        for result in self.stage_results:
            if result.stage_name == stage_name:
                return result
        return None

    @property
    def stage_errors(self) -> dict[str, list[Any]]:
        """Collect per-item errors from all stages.

        Stages stash per-item error lists in ``context.metadata`` under keys
        such as ``pairing_errors``, ``stats_errors``, ``plot_errors``, and
        ``analysis_errors``.
        This property aggregates those lists alongside any stage-level
        failures so that all errors are discoverable in one place without
        changing the ``success`` flag semantics.

        Returns
        -------
        dict[str, list[Any]]
            Mapping of error-list key (e.g. ``"pairing_errors"``) or stage
            name to a non-empty list of error descriptions.  Only entries
            with at least one error are included.
        """
        errors: dict[str, list[Any]] = {}

        # Per-item errors stashed in context.metadata by stages
        if self.context is not None:
            _METADATA_ERROR_KEYS = (
                "pairing_errors",
                "stats_errors",
                "plot_errors",
                "analysis_errors",
            )
            for key in _METADATA_ERROR_KEYS:
                value = self.context.metadata.get(key)
                if value:
                    errors[key] = list(value)

        # Stage-level failures from StageResult.error
        for sr in self.stage_results:
            if sr.status == StageStatus.FAILED and sr.error:
                stage_key = f"stage:{sr.stage_name}"
                errors[stage_key] = [sr.error]

        return errors


class PipelineRunner:
    """Orchestrates execution of analysis pipeline stages.

    The runner manages the flow of data through stages, handles errors,
    and provides hooks for monitoring and logging.

    Parameters
    ----------
    stages
        List of stages to execute. If None, uses standard pipeline.
    fail_fast
        If True, stop on first stage failure.
    hooks
        Optional callback hooks for pipeline events.

    Examples
    --------
    >>> from davinci_monet.pipeline import PipelineRunner, PipelineContext
    >>> runner = PipelineRunner()
    >>> context = PipelineContext(config=my_config)
    >>> result = runner.run(context)
    >>> print(f"Success: {result.success}")
    """

    def __init__(
        self,
        stages: Sequence[Stage] | None = None,
        fail_fast: bool = True,
        hooks: dict[str, Callable[..., None]] | None = None,
        show_progress: bool = True,
        show_plots: bool = False,
        preview_format: Literal["pdf", "png"] = "pdf",
        close_datasets_after_run: bool = True,
    ) -> None:
        """Initialize pipeline runner.

        Parameters
        ----------
        stages
            Stages to execute. If None, uses standard pipeline.
        fail_fast
            Stop execution on first failure.
        hooks
            Event callbacks: on_start, on_stage_start, on_stage_end, on_end.
        show_progress
            Display progress bar and stage status to stdout.
        show_plots
            Display interactive plot preview after completion (requires display).
        preview_format
            Format for plot preview: "pdf" opens in system viewer, "png" in matplotlib.
        close_datasets_after_run
            Close source datasets before returning. Set False when programmatic
            callers need to inspect data in ``PipelineResult.context``.
        """
        self._stages: list[Stage] = (
            list(stages) if stages is not None else list(create_standard_pipeline())
        )
        self._fail_fast = fail_fast
        self._hooks = hooks or {}
        self._show_progress = show_progress
        self._show_plots = show_plots
        self._preview_format = preview_format
        self._resource_policy = PipelineResourcePolicy(
            close_datasets_after_run=close_datasets_after_run
        )

    @property
    def stages(self) -> list[Stage]:
        """Get the list of stages."""
        return list(self._stages)

    def add_stage(self, stage: Stage, position: int | None = None) -> None:
        """Add a stage to the pipeline.

        Parameters
        ----------
        stage
            Stage to add.
        position
            Position to insert at. If None, appends to end.
        """
        if position is None:
            self._stages.append(stage)
        else:
            self._stages.insert(position, stage)

    def remove_stage(self, stage_name: str) -> bool:
        """Remove a stage by name.

        Parameters
        ----------
        stage_name
            Name of stage to remove.

        Returns
        -------
        bool
            True if stage was found and removed.
        """
        for i, stage in enumerate(self._stages):
            if stage.name == stage_name:
                self._stages.pop(i)
                return True
        return False

    def _apply_plot_style(self, context: PipelineContext) -> None:
        """Apply plot styling from configuration.

        Parameters
        ----------
        context
            Pipeline context containing configuration.
        """
        style_config = context.analysis_config().style

        if style_config is None:
            return

        # Handle both dict and PlotStyleConfig object
        if isinstance(style_config, PlotStyleConfig):
            theme = style_config.theme
            style_context = style_config.context
            use_seaborn = style_config.use_seaborn
            seaborn_style = style_config.seaborn_style
        else:
            theme = style_config.get("theme")
            style_context = style_config.get("context", "default")
            use_seaborn = style_config.get("use_seaborn", True)
            seaborn_style = style_config.get("seaborn_style", "whitegrid")

        if theme == "ncar":
            from davinci_monet.plots.style import apply_ncar_style

            apply_ncar_style(
                context=style_context,
                use_seaborn=use_seaborn,
                seaborn_style=seaborn_style,
            )
            logger.info(f"Applied NCAR plot style (context={style_context})")
        elif theme == "default":
            from davinci_monet.plots.style import reset_style

            reset_style()
            logger.info("Reset to default matplotlib style")

    def _cleanup_hdf5_state(self) -> None:
        """Clear HDF5/NetCDF state to avoid transient file handle errors.

        This helps prevent "invalid location identifier" errors that can occur
        when HDF5 has stale file handles from previous runs.
        """
        self._resource_policy.cleanup_hdf5_state()

    def _cleanup_context_datasets(self, context: PipelineContext) -> None:
        """Close all datasets in context to avoid transient file handle errors.

        This prevents crashes that can occur when Python's garbage collector
        tries to close stale NetCDF file handles after the pipeline completes.
        Should be called after log data extraction but before preview/exit.

        Note: Does NOT clear the dictionaries, as other code may still geometry them.
        """
        self._resource_policy.cleanup_context_datasets(context)

    def run(self, context: PipelineContext | None = None) -> PipelineResult:
        """Execute the pipeline.

        Parameters
        ----------
        context
            Pipeline context. If None, creates empty context.

        Returns
        -------
        PipelineResult
            Result of pipeline execution.
        """
        if context is None:
            context = PipelineContext()

        self._resource_policy.prepare_before_run()

        # Apply plot styling from config if specified
        self._apply_plot_style(context)

        # Set up logging and formatting
        log_path: Path | None = None
        log_collector: LogCollector | None = None
        formatter = ProgressFormatter(show_output=self._show_progress)

        analysis_config = context.config_dict().get("analysis", {})
        log_dir = analysis_config.get("log_dir")
        config_path = context.metadata.get("config_path")

        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            # Create timestamped log file with .md extension
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir_path / f"pipeline_{timestamp}.md"

            # Initialize log collector
            log_collector = LogCollector()
            log_collector.start_pipeline(config_path=config_path)

        # Print header
        formatter.header(config_path=config_path, analysis_config=analysis_config)

        context.progress_callback = create_progress_callback(formatter, log_collector)

        result = PipelineResult(
            success=True,
            context=context,
            start_time=datetime.now(),
        )

        start_time = time.time()
        self._call_hook("on_start", context)

        def run_stage(stage: Stage) -> StageResult:
            formatter.stage_start(stage.name)
            if log_collector:
                log_collector.start_stage(stage.name)

            stage_result = self._execute_stage(stage, context)
            result.stage_results.append(stage_result)
            context.results[stage.name] = stage_result

            if log_collector:
                log_collector.finalize_items()

            if stage_result.status == StageStatus.FAILED:
                result.success = False
                formatter.stage_end(stage.name, False, stage_result.duration_seconds)
                if log_collector:
                    log_collector.end_stage(stage.name, "failed", stage_result.duration_seconds)
                    if stage_result.error:
                        log_collector.log_error(
                            stage_name=stage.name,
                            error_type=stage_result.error_type or "Exception",
                            error_message=stage_result.error,
                            traceback_str=stage_result.traceback_str,
                        )
            elif stage_result.status == StageStatus.SKIPPED:
                formatter.stage_end(stage.name, True, stage_result.duration_seconds)
                if log_collector:
                    log_collector.end_stage(stage.name, "skipped", stage_result.duration_seconds)
            elif stage_result.status == StageStatus.COMPLETED:
                formatter.stage_end(stage.name, True, stage_result.duration_seconds)
                if log_collector:
                    log_collector.end_stage(stage.name, "completed", stage_result.duration_seconds)
            return stage_result

        try:
            for stage in self._stages:
                stage_result = run_stage(stage)
                if stage_result.status == StageStatus.FAILED and self._fail_fast:
                    break

            if self._fail_fast and result.failed_stages:
                for stage in self._stages:
                    if stage.name == "manifest" and stage.name not in context.results:
                        run_stage(stage)

        finally:
            # Print footer
            total_duration = time.time() - start_time
            failed_stage = None
            error_message = None
            if result.failed_stages:
                failed = result.failed_stages[0]
                failed_stage = failed.stage_name
                error_message = failed.error
            formatter.footer(
                result.success,
                total_duration,
                log_path,
                failed_stage=failed_stage,
                error_message=error_message,
            )

            # Surface non-fatal per-item errors that stages collected in
            # metadata. These do not flip success, but were previously silent —
            # a run could "succeed" while dropping items.
            item_errors = {
                key: context.metadata.get(key)
                for key in (
                    "pairing_errors",
                    "stats_errors",
                    "plot_errors",
                    "analysis_errors",
                )
                if context.metadata.get(key)
            }
            if item_errors:
                formatter.print_item_errors(item_errors, pipeline_success=result.success)

            # Display the AI summary brief (if produced) to the terminal. The
            # summary stage cannot print durably itself (its log_progress is
            # transient), so the runner renders it here at end of run.
            summary_result = context.results.get("summary")
            if (
                summary_result is not None
                and summary_result.status == StageStatus.COMPLETED
                and isinstance(summary_result.data, dict)
                and summary_result.data.get("bullets")
            ):
                formatter.print_summary(
                    summary_result.data["bullets"],
                    summary_result.data.get("summary_file"),
                    usage=summary_result.data.get("usage"),
                    credits_remaining=summary_result.data.get("credits_remaining"),
                )

            # Write Markdown log file
            if log_collector and log_path:
                log_collector.end_pipeline(result.success)
                # Extract detailed data from context for the report
                log_collector.extract_context_data(context)
                try:
                    log_path.write_text(log_collector.to_markdown())
                except Exception as e:
                    logger.warning(f"Failed to write log file: {e}")

            self._resource_policy.cleanup_after_run(context)

            # Preview generated plots if pipeline succeeded and show_plots is enabled
            if self._show_plots and result.success:
                plot_paths: list[str] = []
                if "plotting" in context.results:
                    stage_result = context.results["plotting"]
                    if stage_result.data and "plots_generated" in stage_result.data:
                        plot_paths.extend(stage_result.data["plots_generated"])
                if plot_paths:
                    formatter.preview_plots(
                        plot_paths, duration=1.0, preview_format=self._preview_format
                    )

        result.end_time = datetime.now()
        result.total_duration_seconds = time.time() - start_time

        self._call_hook("on_end", result)

        return result

    def run_from_config(self, config: dict[str, Any] | str | MonetConfig) -> PipelineResult:
        """Execute pipeline from configuration.

        Parameters
        ----------
        config
            Configuration dictionary or path to YAML file.

        Returns
        -------
        PipelineResult
            Result of pipeline execution.

        Raises
        ------
        ConfigurationError
            If configuration is empty or missing required sections.
        """
        from davinci_monet.config import load_config, validate_config

        config_path: str | None = None
        if isinstance(config, str):
            config_path = config
            config_model = load_config(config)
        elif isinstance(config, MonetConfig):
            config_model = config
        else:
            config_model = validate_config(config)

        sources_config = config_model.sources

        if not sources_config:
            raise ConfigurationError(
                "Configuration is empty or incomplete. "
                "At least one source must be defined under 'sources:'."
            )

        # The unified standard pipeline handles both paired-source and
        # single-source runs: pairing skips when there are no pairs, while
        # statistics/plotting dispatch on the available source state.

        context = PipelineContext(config=config_model)
        if config_path:
            context.metadata["config_path"] = config_path
        return self.run(context)

    def _execute_stage(self, stage: Stage, context: PipelineContext) -> StageResult:
        """Execute a single stage.

        Parameters
        ----------
        stage
            Stage to execute.
        context
            Pipeline context.

        Returns
        -------
        StageResult
            Result of stage execution.
        """
        self._call_hook("on_stage_start", stage, context)

        start_time = time.time()

        try:
            # Validate stage
            if not stage.validate(context):
                # A stage that opts out of running for this configuration is a
                # benign skip (e.g. an optional stage without input), not a
                # failure — log at debug so it does not read as an error.
                logger.debug(f"Stage '{stage.name}' not applicable for this run, skipping")
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                    error="Not applicable for this run",
                    duration_seconds=time.time() - start_time,
                )
            else:
                # Execute stage
                logger.info(f"Executing stage: {stage.name}")
                result = stage.execute(context)
                result.duration_seconds = time.time() - start_time
                logger.info(f"Stage '{stage.name}' completed in " f"{result.duration_seconds:.2f}s")

        except Exception as e:
            # Don't use logger.exception() - it prints traceback to console
            # We capture the traceback and store it in the result for the log file
            tb_str = traceback.format_exc()
            logger.error(f"Stage '{stage.name}' failed: {e}")
            result = StageResult(
                stage_name=stage.name,
                status=StageStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                traceback_str=tb_str,
                duration_seconds=time.time() - start_time,
            )

        self._call_hook("on_stage_end", stage, result, context)

        return result

    def _call_hook(self, hook_name: str, *args: Any) -> None:
        """Call a hook if registered."""
        if hook_name in self._hooks:
            try:
                self._hooks[hook_name](*args)
            except Exception as e:
                logger.warning(f"Hook '{hook_name}' raised exception: {e}")


class PipelineBuilder:
    """Fluent builder for constructing pipelines.

    Examples
    --------
    >>> pipeline = (
    ...     PipelineBuilder()
    ...     .add_sources()
    ...     .add_pairing()
    ...     .add_statistics()
    ...     .build()
    ... )
    """

    def __init__(self) -> None:
        self._stages: list[Stage] = []
        self._fail_fast = True
        self._hooks: dict[str, Callable[..., None]] = {}
        self._show_progress = True
        self._show_plots = False
        self._preview_format: Literal["pdf", "png"] = "pdf"

    def add_stage(self, stage: Stage) -> PipelineBuilder:
        """Add a custom stage."""
        self._stages.append(stage)
        return self

    def add_sources(self) -> PipelineBuilder:
        """Add the unified data-source loading stage.

        Loads all data sources (native ``sources:`` configs) into
        ``context.sources``. Replaces the removed ``add_datasets``/
        registered source readers.
        """
        from davinci_monet.pipeline.stages import LoadSourcesStage

        self._stages.append(LoadSourcesStage())
        return self

    def add_pairing(self) -> PipelineBuilder:
        """Add pairing stage."""
        from davinci_monet.pipeline.stages import PairingStage

        self._stages.append(PairingStage())
        return self

    def add_statistics(self) -> PipelineBuilder:
        """Add statistics stage."""
        from davinci_monet.pipeline.stages import StatisticsStage

        self._stages.append(StatisticsStage())
        return self

    def add_plotting(self) -> PipelineBuilder:
        """Add plotting stage."""
        from davinci_monet.pipeline.stages import PlottingStage

        self._stages.append(PlottingStage())
        return self

    def add_save(self) -> PipelineBuilder:
        """Add save results stage."""
        from davinci_monet.pipeline.stages import SaveResultsStage

        self._stages.append(SaveResultsStage())
        return self

    def fail_fast(self, enabled: bool = True) -> PipelineBuilder:
        """Set fail-fast mode."""
        self._fail_fast = enabled
        return self

    def with_hook(self, event: str, callback: Callable[..., None]) -> PipelineBuilder:
        """Add an event hook."""
        self._hooks[event] = callback
        return self

    def show_progress(self, enabled: bool = True) -> PipelineBuilder:
        """Set progress display mode."""
        self._show_progress = enabled
        return self

    def show_plots(
        self, enabled: bool = True, preview_format: Literal["pdf", "png"] = "pdf"
    ) -> PipelineBuilder:
        """Set interactive plot preview mode."""
        self._show_plots = enabled
        self._preview_format = preview_format
        return self

    def build(self) -> PipelineRunner:
        """Build the pipeline runner."""
        return PipelineRunner(
            stages=self._stages,
            fail_fast=self._fail_fast,
            hooks=self._hooks,
            show_progress=self._show_progress,
            show_plots=self._show_plots,
            preview_format=self._preview_format,
        )


def run_analysis(
    config: dict[str, Any] | str | MonetConfig,
    show_progress: bool = True,
    show_plots: bool = False,
    preview_format: Literal["pdf", "png"] = "pdf",
) -> PipelineResult:
    """Convenience function to run a complete analysis.

    Parameters
    ----------
    config
        Configuration dictionary or path to YAML file.
    show_progress
        Display progress bar and stage timing to stdout.
    show_plots
        Display interactive plot preview after completion (requires display).
    preview_format
        Format for plot preview: "pdf" opens in system viewer, "png" in matplotlib.

    Returns
    -------
    PipelineResult
        Result of pipeline execution.

    Examples
    --------
    >>> result = run_analysis("config.yaml")
    >>> if result.success:
    ...     print("Analysis complete!")
    """
    runner = PipelineRunner(
        show_progress=show_progress,
        show_plots=show_plots,
        preview_format=preview_format,
    )
    return runner.run_from_config(config)
