"""Pipeline reporting — structured log collection and Markdown report generation.

Contains :class:`LogEntry` and :class:`LogCollector`, which gather timing
and metadata during a pipeline run and emit a Markdown summary on completion.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from davinci_monet.pipeline.stages import PipelineContext


@dataclass
class LogEntry:
    """A single log entry with timing information."""

    name: str
    category: str  # 'source', 'pair', 'stage'
    start_time: float
    end_time: float | None = None
    status: str = "running"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class LogCollector:
    """Collects structured log data for Markdown report generation."""

    def __init__(self) -> None:
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.config_path: str | None = None
        self.entries: list[LogEntry] = []
        self._current_stage: LogEntry | None = None
        self._item_start_times: dict[str, float] = {}
        # Additional data extracted from context
        self.source_details: dict[str, dict[str, Any]] = {}
        self.pair_details: dict[str, dict[str, Any]] = {}
        self.statistics: dict[str, dict[str, Any]] = {}
        # Error tracking
        self.errors: list[dict[str, Any]] = []

    def start_pipeline(self, config_path: str | None = None) -> None:
        """Record pipeline start."""
        self.start_time = datetime.now()
        self.config_path = config_path

    def end_pipeline(self, success: bool) -> None:
        """Record pipeline end."""
        self.end_time = datetime.now()
        self.success = success

    def log_error(
        self,
        stage_name: str,
        error_type: str,
        error_message: str,
        traceback_str: str | None = None,
    ) -> None:
        """Record an error that occurred during pipeline execution.

        Parameters
        ----------
        stage_name
            Name of the stage where the error occurred.
        error_type
            The exception class name (e.g., 'ValueError', 'DataNotFoundError').
        error_message
            The error message string.
        traceback_str
            Optional formatted traceback string.
        """
        self.errors.append(
            {
                "stage": stage_name,
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback_str,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def start_stage(self, name: str) -> None:
        """Record stage start."""
        self._current_stage = LogEntry(
            name=name,
            category="stage",
            start_time=time.time(),
        )

    def end_stage(
        self,
        name: str,
        status: str,
        duration: float,
        disposition: str | None = None,
    ) -> None:
        """Record stage completion."""
        if self._current_stage and self._current_stage.name == name:
            self._current_stage.end_time = time.time()
            self._current_stage.status = status
            if disposition is not None:
                self._current_stage.details["disposition"] = disposition
            self.entries.append(self._current_stage)
            self._current_stage = None

    def log_item(self, message: str) -> None:
        """Parse and log a progress message."""
        # Parse patterns like "Loading dataset: cesm_asiaq (1/2)"
        source_match = re.match(r"\s*Loading source: (\S+)", message)
        y_match = re.match(r"\s*Loading dataset: (\S+)", message)
        x_match = re.match(r"\s*Loading geometry: (\S+)", message)
        # Pairing pattern
        pair_match = re.match(r"\s*Pairing: (\S+)", message)
        # New parallel pairing patterns
        parallel_started_match = re.match(r"\s*parallel_started: (\S+)", message)
        parallel_completed_match = re.match(r"\s*parallel_completed: (\S+)(.*)", message)

        if source_match:
            name = source_match.group(1)
            self._start_item(name, "source")
        elif y_match:
            name = y_match.group(1)
            self._start_item(name, "dataset")
        elif x_match:
            name = x_match.group(1)
            self._start_item(name, "dataset")
        elif pair_match:
            name = pair_match.group(1)
            self._start_item(name, "pair")
        elif parallel_started_match:
            name = parallel_started_match.group(1)
            self._start_item(name, "pair")
        elif parallel_completed_match:
            name = parallel_completed_match.group(1)
            # Close this specific pair's timing
            key = f"pair:{name}"
            if key in self._item_start_times:
                start = self._item_start_times[key]
                self.entries.append(
                    LogEntry(
                        name=name,
                        category="pair",
                        start_time=start,
                        end_time=time.time(),
                        status="completed",
                    )
                )
                del self._item_start_times[key]

    def _start_item(self, name: str, category: str) -> None:
        """Start timing an item and close previous in same category."""
        # Close any previous item in the same category
        for entry in self.entries:
            if entry.category == category and entry.end_time is None:
                entry.end_time = time.time()

        # Also check _item_start_times for any unclosed items
        for key, start in list(self._item_start_times.items()):
            if key.startswith(f"{category}:"):
                old_name = key.split(":", 1)[1]
                self.entries.append(
                    LogEntry(
                        name=old_name,
                        category=category,
                        start_time=start,
                        end_time=time.time(),
                        status="completed",
                    )
                )
                del self._item_start_times[key]

        self._item_start_times[f"{category}:{name}"] = time.time()

    def finalize_items(self) -> None:
        """Close any open items when stage ends."""
        now = time.time()
        for key, start in list(self._item_start_times.items()):
            category, name = key.split(":", 1)
            self.entries.append(
                LogEntry(
                    name=name,
                    category=category,
                    start_time=start,
                    end_time=now,
                    status="completed",
                )
            )
        self._item_start_times.clear()

    def extract_context_data(self, context: "PipelineContext") -> None:
        """Extract detailed data from pipeline context after execution."""

        def _source_detail(label: str, source_data: Any) -> dict[str, Any]:
            details: dict[str, Any] = {}
            ds = source_data.data if hasattr(source_data, "data") else source_data
            if hasattr(source_data, "source_type"):
                details["type"] = source_data.source_type
            if hasattr(ds, "data_vars"):
                details["variables"] = len(ds.data_vars)
                if "time" in ds.sizes:
                    details["time_steps"] = ds.sizes["time"]
                elif "x_time" in ds.sizes:
                    details["time_steps"] = ds.sizes["x_time"]
                if "site" in ds.sizes:
                    details["sites"] = ds.sizes["site"]
                elif "x" in ds.sizes:
                    details["points"] = ds.sizes["x"]
                total_size = sum(ds[v].size for v in ds.data_vars)
                details["data_points"] = total_size
            return details

        for label, source_data in context.sources.items():
            self.source_details[label] = _source_detail(label, source_data)

        # Extract pair details
        for pair_key, paired_data in context.paired.items():
            details = {}
            ds = paired_data.data if hasattr(paired_data, "data") else paired_data
            if ds is not None:
                # Count paired points
                total_points = 1
                for dim in ds.sizes:
                    total_points *= ds.sizes[dim]
                details["paired_points"] = total_points
                # Count paired x/y variable pairs.
                from davinci_monet.core.base import iter_paired_variable_xy

                details["variables"] = len(iter_paired_variable_xy(ds))
            self.pair_details[pair_key] = details

        # Extract statistics
        stats_result = context.results.get("statistics")
        if stats_result and stats_result.data:
            self.statistics = stats_result.data

        # Capture per-item (non-fatal) errors that stages stashed in metadata so
        # they appear in the report. These do not flip pipeline success, but
        # silently dropping them hides real failures (a pair/stat/plot that
        # errored while the run still "succeeded").
        _ITEM_ERROR_STAGES = {
            "pairing_errors": "pairing",
            "stats_errors": "statistics",
            "plot_errors": "plotting",
            "analysis_errors": "analyses",
        }
        for meta_key, stage_name in _ITEM_ERROR_STAGES.items():
            for message in context.metadata.get(meta_key, []) or []:
                self.log_error(
                    stage_name=stage_name,
                    error_type="ItemError",
                    error_message=str(message),
                )

    def _format_number(self, n: int) -> str:
        """Format large numbers with K/M suffix."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines: list[str] = []

        # Header
        lines.append("# DAVINCI Pipeline Log")
        lines.append("")

        # Metadata table
        lines.append("## Run Information")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        if self.config_path:
            lines.append(f"| Config | `{self.config_path}` |")
        if self.start_time:
            lines.append(f"| Started | {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} |")
        if self.end_time:
            lines.append(f"| Finished | {self.end_time.strftime('%Y-%m-%d %H:%M:%S')} |")
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            lines.append(f"| Duration | {duration:.1f}s |")
        if hasattr(self, "success"):
            status = "Success" if self.success else "Failed"
            lines.append(f"| Status | **{status}** |")
        lines.append("")

        # Stage summary table
        stages = [e for e in self.entries if e.category == "stage"]
        if stages:
            lines.append("## Stage Summary")
            lines.append("")
            lines.append("| Stage | Status | Checkpoint | Duration |")
            lines.append("|-------|--------|------------|----------|")
            for entry in stages:
                status_icon = "✓" if entry.status == "completed" else "✗"
                disposition = entry.details.get("disposition", "-")
                lines.append(
                    f"| {entry.name} | {status_icon} {entry.status.title()} | "
                    f"{disposition} | {entry.duration:.1f}s |"
                )
            lines.append("")

        # Sources table with details
        if self.source_details:
            lines.append("## Sources Loaded")
            lines.append("")
            lines.append("| Source | Type | Variables | Records | Data Points |")
            lines.append("|--------|------|-----------|---------|-------------|")
            for name, details in self.source_details.items():
                source_type = details.get("type") or "-"
                vars_count = details.get("variables", "-")
                records = (
                    details.get("sites")
                    or details.get("points")
                    or details.get("time_steps")
                    or "-"
                )
                if isinstance(records, int):
                    records = self._format_number(records)
                data_points = details.get("data_points")
                data_str = self._format_number(data_points) if data_points else "-"
                lines.append(f"| {name} | {source_type} | {vars_count} | {records} | {data_str} |")
            lines.append("")

        # Pairings table with details
        pairs = [e for e in self.entries if e.category == "pair"]
        if pairs:
            lines.append("## Pairings")
            lines.append("")
            lines.append("| Pair | Variables | Paired Points | Duration |")
            lines.append("|------|-----------|---------------|----------|")
            for entry in pairs:
                details = self.pair_details.get(entry.name, {})
                vars_count = details.get("variables", "-")
                paired_points = details.get("paired_points")
                points_str = self._format_number(paired_points) if paired_points else "-"
                lines.append(
                    f"| {entry.name} | {vars_count} | {points_str} | {entry.duration:.1f}s |"
                )
            lines.append("")

        # Statistics summary
        if self.statistics:
            lines.append("## Statistics Summary")
            lines.append("")
            for pair_key, pair_stats in self.statistics.items():
                if not pair_stats:
                    continue
                lines.append(f"### {pair_key}")
                lines.append("")
                lines.append("| Variable | N | Mean X | Mean Y | MB | RMSE | R |")
                lines.append("|----------|---|--------|--------|-----|------|---|")
                for var_name, stats in pair_stats.items():
                    if not isinstance(stats, dict):
                        continue

                    def _get_metric(*keys: str, default: Any = None) -> Any:
                        for key in keys:
                            if key in stats and stats[key] is not None:
                                return stats[key]
                        return default

                    n = _get_metric("N", "n", default="-")
                    x_mean = _get_metric("MX", "x_mean")
                    y_mean = _get_metric("MY", "y_mean")
                    mb = _get_metric("MB", "mean_bias")
                    rmse = _get_metric("RMSE", "rmse")
                    r = _get_metric("R", "correlation")
                    # Format values
                    x_str = f"{x_mean:.2f}" if x_mean is not None else "-"
                    y_str = f"{y_mean:.2f}" if y_mean is not None else "-"
                    mb_str = f"{mb:+.2f}" if mb is not None else "-"
                    rmse_str = f"{rmse:.2f}" if rmse is not None else "-"
                    r_str = (
                        f"{r:.2f}"
                        if r is not None and not (isinstance(r, float) and r != r)
                        else "-"
                    )
                    lines.append(
                        f"| {var_name} | {n} | {x_str} | {y_str} | {mb_str} | {rmse_str} | {r_str} |"
                    )
                lines.append("")

        # Errors section (if any)
        if self.errors:
            lines.append("## Errors")
            lines.append("")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"### Error {i}: {error['error_type']} in `{error['stage']}`")
                lines.append("")
                lines.append(f"**Time:** {error['timestamp']}")
                lines.append("")
                lines.append(f"**Message:** {error['error_message']}")
                lines.append("")
                if error.get("traceback"):
                    lines.append("**Traceback:**")
                    lines.append("```")
                    lines.append(error["traceback"])
                    lines.append("```")
                    lines.append("")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by DAVINCI*")
        lines.append("")

        return "\n".join(lines)
