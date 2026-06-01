"""Pipeline runner for orchestrating analysis workflows.

This module provides the PipelineRunner class that executes a sequence
of analysis stages, managing state and handling errors.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TextIO

from tqdm import tqdm

from davinci_monet.core.exceptions import PipelineError
from davinci_monet.pipeline.stages import (
    BaseStage,
    PipelineContext,
    Stage,
    StageResult,
    StageStatus,
    create_standard_pipeline,
)

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """A single log entry with timing information."""

    name: str
    category: str  # 'model', 'observation', 'pair', 'stage'
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
        self.model_details: dict[str, dict[str, Any]] = {}
        self.obs_details: dict[str, dict[str, Any]] = {}
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

    def end_stage(self, name: str, status: str, duration: float) -> None:
        """Record stage completion."""
        if self._current_stage and self._current_stage.name == name:
            self._current_stage.end_time = time.time()
            self._current_stage.status = status
            self.entries.append(self._current_stage)
            self._current_stage = None

    def log_item(self, message: str) -> None:
        """Parse and log a progress message."""
        # Parse patterns like "Loading model: cesm_asiaq (1/2)"
        model_match = re.match(r"\s*Loading model: (\S+)", message)
        obs_match = re.match(r"\s*Loading obs: (\S+)", message)
        # Legacy pairing pattern
        pair_match = re.match(r"\s*Pairing: (\S+)", message)
        # New parallel pairing patterns
        parallel_started_match = re.match(r"\s*parallel_started: (\S+)", message)
        parallel_completed_match = re.match(r"\s*parallel_completed: (\S+)(.*)", message)

        if model_match:
            name = model_match.group(1)
            self._start_item(name, "model")
        elif obs_match:
            name = obs_match.group(1)
            self._start_item(name, "observation")
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
        # Extract model details
        for label, model_data in context.models.items():
            details: dict[str, Any] = {}
            if hasattr(model_data, "data") and model_data.data is not None:
                ds = model_data.data
                details["variables"] = len(ds.data_vars)
                if "time" in ds.sizes:
                    details["time_steps"] = ds.sizes["time"]
                # Calculate approximate size
                total_size = sum(ds[v].size for v in ds.data_vars)
                details["data_points"] = total_size
            self.model_details[label] = details

        # Extract observation details
        for label, obs_data in context.observations.items():
            details = {}
            if hasattr(obs_data, "data") and obs_data.data is not None:
                ds = obs_data.data
                details["variables"] = len(ds.data_vars)
                if "time" in ds.sizes:
                    details["time_steps"] = ds.sizes["time"]
                elif "obs_time" in ds.sizes:
                    details["time_steps"] = ds.sizes["obs_time"]
                # Get observation type
                if hasattr(obs_data, "obs_type"):
                    details["type"] = obs_data.obs_type
                # Count sites/points
                if "site" in ds.sizes:
                    details["sites"] = ds.sizes["site"]
                elif "x" in ds.sizes:
                    details["points"] = ds.sizes["x"]
            self.obs_details[label] = details

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
                # Count paired (obs, model) variable pairs (role-based; R-5)
                from davinci_monet.core.base import iter_paired_variable_pairs

                details["variables"] = len(iter_paired_variable_pairs(ds))
            self.pair_details[pair_key] = details

        # Extract statistics
        stats_result = context.results.get("statistics")
        if stats_result and stats_result.data:
            self.statistics = stats_result.data

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
            lines.append("| Stage | Status | Duration |")
            lines.append("|-------|--------|----------|")
            for entry in stages:
                status_icon = "✓" if entry.status == "completed" else "✗"
                lines.append(
                    f"| {entry.name} | {status_icon} {entry.status.title()} | {entry.duration:.1f}s |"
                )
            lines.append("")

        # Models table with details
        models = [e for e in self.entries if e.category == "model"]
        if models:
            lines.append("## Models Loaded")
            lines.append("")
            lines.append("| Model | Variables | Time Steps | Data Points | Duration |")
            lines.append("|-------|-----------|------------|-------------|----------|")
            for entry in models:
                details = self.model_details.get(entry.name, {})
                vars_count = details.get("variables", "-")
                time_steps = details.get("time_steps", "-")
                data_points = details.get("data_points")
                data_str = self._format_number(data_points) if data_points else "-"
                lines.append(
                    f"| {entry.name} | {vars_count} | {time_steps} | {data_str} | {entry.duration:.1f}s |"
                )
            lines.append("")

        # Observations table with details
        observations = [e for e in self.entries if e.category == "observation"]
        if observations:
            lines.append("## Observations Loaded")
            lines.append("")
            lines.append("| Observation | Type | Variables | Records | Duration |")
            lines.append("|-------------|------|-----------|---------|----------|")
            for entry in observations:
                details = self.obs_details.get(entry.name, {})
                obs_type = details.get("type", "-")
                vars_count = details.get("variables", "-")
                # Get record count (sites, points, or time steps)
                records = (
                    details.get("sites")
                    or details.get("points")
                    or details.get("time_steps")
                    or "-"
                )
                if isinstance(records, int):
                    records = self._format_number(records)
                lines.append(
                    f"| {entry.name} | {obs_type} | {vars_count} | {records} | {entry.duration:.1f}s |"
                )
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
                lines.append("| Variable | N | Mean Obs | Mean Model | MB | RMSE | R |")
                lines.append("|----------|---|----------|------------|-----|------|---|")
                for var_name, stats in pair_stats.items():
                    if not isinstance(stats, dict):
                        continue

                    def _get_metric(*keys: str, default: Any = None) -> Any:
                        for key in keys:
                            if key in stats and stats[key] is not None:
                                return stats[key]
                        return default

                    n = _get_metric("N", "n", default="-")
                    obs_mean = _get_metric("MO", "obs_mean")
                    model_mean = _get_metric("MP", "model_mean")
                    mb = _get_metric("MB", "mean_bias")
                    rmse = _get_metric("RMSE", "rmse")
                    r = _get_metric("R", "correlation")
                    # Format values
                    obs_str = f"{obs_mean:.2f}" if obs_mean is not None else "-"
                    model_str = f"{model_mean:.2f}" if model_mean is not None else "-"
                    mb_str = f"{mb:+.2f}" if mb is not None else "-"
                    rmse_str = f"{rmse:.2f}" if rmse is not None else "-"
                    r_str = (
                        f"{r:.2f}"
                        if r is not None and not (isinstance(r, float) and r != r)
                        else "-"
                    )
                    lines.append(
                        f"| {var_name} | {n} | {obs_str} | {model_str} | {mb_str} | {rmse_str} | {r_str} |"
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


class TeeWriter:
    """Write to multiple file-like objects simultaneously."""

    def __init__(self, *writers: TextIO) -> None:
        self.writers = writers

    def write(self, message: str) -> None:
        for writer in self.writers:
            writer.write(message)
            writer.flush()

    def flush(self) -> None:
        for writer in self.writers:
            writer.flush()


class ProgressFormatter:
    """Formats pipeline progress output with rich animated styling.

    Uses rich library for pulsing text, panels, and color-coded output.
    Features a pulsing "Da Vinci" animation with NCAR blue color palette
    and an elapsed time counter during stage execution.
    """

    # NCAR blue color palette for brightening effect (dark to bright)
    # Based on NCAR brand colors: space -> dark_blue -> ncar_blue -> light_blue
    NCAR_BLUE_PALETTE = [
        "#011837",  # NCAR Space (darkest)
        "#00357A",  # NCAR Dark Blue
        "#0A5DDA",  # NCAR Blue (primary)
        "#22A7F0",  # Bright blue (interpolated)
        "#67C8F9",  # Light blue
        "#CEDFF8",  # NCAR Light Blue (brightest)
    ]

    # NCAR accent colors for status
    NCAR_AQUA = "#00A2B4"  # UCAR Aqua - for accents
    NCAR_ORANGE = "#FF8C00"  # For warnings/stage names
    NCAR_GREEN = "#2E8B57"  # For success
    NCAR_RED = "#D62839"  # For errors

    def __init__(self, show_output: bool = True) -> None:
        from rich.console import Console

        self.show_output = show_output
        self.console = Console(force_terminal=show_output, no_color=not show_output)
        self._current_stage: str | None = None
        self._stage_start: float | None = None
        self._lines: list[str] = []  # For log file
        self._live: Any = None  # Rich Live context
        self._current_status: str = ""
        self._stage_items: list[tuple[str, str]] = []  # (category, name) pairs for current stage
        self._animation_frame: int = 0
        self._current_item: str | None = None
        self._current_progress: tuple[int, int] | None = None  # (index, total)
        # For parallel stages: track completed count separately from current item
        self._parallel_total: int = 0
        self._parallel_completed: int = 0
        self._parallel_mode: bool = False
        self._parallel_loading_msg: str | None = None  # Message to show during [0/N]

    def _log(self, line: str) -> None:
        """Store a line for log file."""
        self._lines.append(line)

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print via rich console if output enabled."""
        if self.show_output:
            self.console.print(*args, **kwargs)

    def _create_pulsing_davinci(self) -> "Text":  # type: ignore[name-defined]
        """Create 'DAVINCI' text with left-to-right brightening effect."""
        from rich.text import Text

        text = "DAVINCI"
        result = Text()
        text_len = len(text)
        palette_len = len(self.NCAR_BLUE_PALETTE)

        # Bright spot moves from left to right
        # Animation frame determines position of the bright spot
        bright_pos = self._animation_frame % (text_len + 4)  # +4 for trail off

        for i, char in enumerate(text):
            # Calculate distance from bright spot
            distance = bright_pos - i

            if distance < 0:
                # Bright spot hasn't reached this char yet - use darkest
                color_idx = 0
            elif distance >= palette_len:
                # Bright spot has passed - use darkest
                color_idx = 0
            else:
                # In the brightening zone - brighter as distance decreases
                color_idx = palette_len - 1 - distance

            result.append(char, style=f"bold {self.NCAR_BLUE_PALETTE[color_idx]}")

        return result

    def _create_stage_display(self) -> "Text":  # type: ignore[name-defined]
        """Create the animated stage display with pulsing text and timer."""
        from rich.text import Text

        result = Text()

        # Pulsing "Da Vinci" animation
        result.append("  ")
        result.append_text(self._create_pulsing_davinci())
        result.append(" ")

        # Elapsed time counter
        if self._stage_start is not None:
            elapsed = time.time() - self._stage_start
            result.append(f"[{elapsed:5.1f}s] ", style=f"bold {self.NCAR_AQUA}")

        # Stage name
        if self._current_stage:
            result.append(self._current_stage, style=f"bold {self.NCAR_ORANGE}")

        # Parallel mode: show completion progress
        if self._parallel_mode and self._parallel_total > 0:
            result.append(" › ", style="dim")
            result.append(
                f"[{self._parallel_completed}/{self._parallel_total}] ",
                style=f"dim {self.NCAR_AQUA}" if self._parallel_completed > 0 else "dim",
            )
            if self._current_item:
                # After completions start, show the completed item name
                result.append(self._current_item, style="white")
            elif self._parallel_loading_msg and self._parallel_completed == 0:
                # During [0/N] phase, show what's being loaded
                result.append(self._parallel_loading_msg, style="dim italic")
        # Sequential mode: show current item
        elif self._current_item:
            result.append(" › ", style="dim")
            if self._current_progress:
                idx, total = self._current_progress
                result.append(f"[{idx}/{total}] ", style="dim")
            result.append(self._current_item, style="white")

        return result

    def _start_animation_loop(self) -> None:
        """Start background thread for animation updates."""
        import threading

        text_len = len("DAVINCI")

        def animate() -> None:
            while self._live is not None:
                # Cycle through positions for left-to-right sweep
                self._animation_frame = (self._animation_frame + 1) % (text_len + 4)
                if self._live is not None:
                    try:
                        self._live.update(self._create_stage_display())
                    except Exception:
                        break
                time.sleep(0.15)  # Speed for smooth left-to-right sweep

        self._animation_thread = threading.Thread(target=animate, daemon=True)
        self._animation_thread.start()

    def header(
        self,
        config_path: str | None = None,
        analysis_config: dict[str, Any] | None = None,
        clear_screen: bool = True,
    ) -> None:
        """Print pipeline header with NSF NCAR UCAR logo.

        Parameters
        ----------
        config_path
            Path to configuration file to display.
        analysis_config
            Analysis configuration dict with start_time, end_time, etc.
        clear_screen
            If True, clear the terminal before displaying header.
        """
        from rich.panel import Panel
        from rich.text import Text

        from davinci_monet.assets.logo import get_colored_logo

        # Log file header (plain text)
        self._log("DAVINCI Pipeline")
        if config_path:
            self._log(f"Config: {config_path}")
        self._log("")

        # Clear screen if requested
        if clear_screen and self.show_output:
            self.console.clear()

        # Show NSF NCAR UCAR logo
        if self.show_output:
            self._print()
            self._print(get_colored_logo())

        # Rich console output - pipeline title with date and system info
        content = Text()
        content.append("DAVINCI Pipeline", style=f"bold {self.NCAR_AQUA}")
        content.append("  ")
        content.append(datetime.now().strftime("%a %b %-d, %Y %H:%M"), style="dim")
        content.append("  ")
        system_info = self._get_system_info()
        content.append(system_info, style="dim")
        self._print(Panel(content, border_style=self.NCAR_AQUA, padding=(0, 2)))

        # Config path below the panel
        if config_path:
            # Truncate path if too long
            max_path_len = 70
            display_path = config_path
            if len(config_path) > max_path_len:
                display_path = "..." + config_path[-(max_path_len - 3) :]
            self._print(f"  [dim]Config:[/dim] {display_path}")

        # Display analysis info
        if analysis_config:
            start_time = analysis_config.get("start_time")
            end_time = analysis_config.get("end_time")
            if start_time and end_time:
                # Format dates nicely
                start_str = self._format_datetime(start_time)
                end_str = self._format_datetime(end_time)
                self._print(f"  [dim]Period:[/dim] {start_str} → {end_str}")

        self._print()

    def _format_datetime(self, dt: Any) -> str:
        """Format a datetime for display.

        Parameters
        ----------
        dt
            Datetime object or string.

        Returns
        -------
        str
            Formatted date string (e.g., "Feb 1, 2024").
        """
        if isinstance(dt, str):
            # Parse ISO format string
            try:
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            except ValueError:
                return dt  # Return as-is if parsing fails
        if isinstance(dt, datetime):
            return dt.strftime("%b %-d, %Y")
        return str(dt)

    def _get_system_info(self) -> str:
        """Get system information for display.

        Returns
        -------
        str
            Formatted system info string.
        """
        import os
        import platform
        import subprocess

        parts = []

        # Hostname
        hostname = platform.node()
        if hostname:
            # Remove .local suffix if present
            hostname = hostname.removesuffix(".local")
            parts.append(hostname)

        # CPU type - try to get a friendly name
        cpu_name = None
        if platform.system() == "Darwin":
            # macOS: use sysctl
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    cpu_name = result.stdout.strip()
            except Exception:
                pass
        elif platform.system() == "Linux":
            # Linux: parse /proc/cpuinfo
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_name = line.split(":")[1].strip()
                            break
            except Exception:
                pass

        if cpu_name:
            # Shorten common prefixes
            cpu_name = cpu_name.replace("Intel(R) Core(TM) ", "Intel ")
            cpu_name = cpu_name.replace("AMD Ryzen ", "Ryzen ")

        # CPU cores
        cpu_count = os.cpu_count()

        # GPU info (macOS only for now)
        gpu_cores = None
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    import json

                    data = json.loads(result.stdout)
                    displays = data.get("SPDisplaysDataType", [])
                    for display in displays:
                        gpu_cores = display.get("sppci_cores")
                        if gpu_cores:
                            break
            except Exception:
                pass

        # Combine CPU name with core counts
        if cpu_name:
            core_info = []
            if cpu_count:
                core_info.append(f"{cpu_count} CPU")
            if gpu_cores:
                core_info.append(f"{gpu_cores} GPU")
            if core_info:
                parts.append(f"{cpu_name} ({', '.join(core_info)})")
            else:
                parts.append(cpu_name)
        elif cpu_count:
            parts.append(f"{cpu_count} cores")

        # RAM
        ram_gb = None
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    ram_bytes = int(result.stdout.strip())
                    ram_gb = ram_bytes // (1024**3)
            except Exception:
                pass
        elif platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            # Format: "MemTotal:       16384000 kB"
                            kb = int(line.split()[1])
                            ram_gb = kb // (1024**2)
                            break
            except Exception:
                pass

        if ram_gb:
            parts.append(f"{ram_gb} GB")

        return " | ".join(parts)

    def stage_start(self, name: str) -> None:
        """Print stage start with pulsing Da Vinci animation and timer."""
        from rich.live import Live

        self._current_stage = name
        self._stage_start = time.time()
        self._stage_items = []  # Reset items for this stage
        self._current_item = None
        self._current_progress = None
        self._animation_frame = 0
        self._log(f"[{name}]")

        if self.show_output:
            self._live = Live(
                self._create_stage_display(),
                console=self.console,
                refresh_per_second=8,  # Smooth animation
                transient=True,  # Clear when done
            )
            self._live.start()
            self._start_animation_loop()

    def stage_end(self, name: str, success: bool, duration: float) -> None:
        """Print stage end with status and summary of items processed."""
        # Stop the live animation
        if self._live is not None:
            self._live.stop()
            self._live = None  # This stops the animation thread too

        # Clear animation state
        self._current_item = None
        self._current_progress = None

        if success:
            icon = "✓"
            style = f"bold {self.NCAR_GREEN}"
            status = "completed"
        else:
            icon = "✗"
            style = f"bold {self.NCAR_RED}"
            status = "FAILED"

        self._log(f"  {icon} {status} ({duration:.1f}s)")

        # Show stage completion
        self._print(f"  [{style}]{icon} {name}[/{style}] [dim]({duration:.1f}s)[/dim]")

        # Show summary of items processed in this stage (exclude plots - shown in preview)
        if self._stage_items and success:
            # Group items by category
            categories: dict[str, list[str]] = {}
            for category, item_name in self._stage_items:
                # Skip plot items - they'll be shown in the preview slideshow
                if category == "plot":
                    continue
                if category not in categories:
                    categories[category] = []
                categories[category].append(item_name)

            # Display each category
            for category, items in categories.items():
                items_str = ", ".join(items)
                self._log(f"    {category}: {items_str}")
                self._print(f"    [dim]{category}:[/dim] [white]{items_str}[/white]")

        self._print()
        self._current_stage = None
        self._stage_items = []

    def start_parallel(self, total: int, loading_msg: str | None = None) -> None:
        """Enter parallel mode for tracking completion of multiple items.

        In parallel mode, the display shows "[completed/total]" instead of
        "[current/total]", which is more meaningful for parallel execution.

        Parameters
        ----------
        total
            Total number of items to process in parallel.
        loading_msg
            Optional message to show during [0/N] phase (e.g., "loading cesm → obs1, obs2").
        """
        self._parallel_mode = True
        self._parallel_total = total
        self._parallel_completed = 0
        self._parallel_loading_msg = loading_msg

    def end_parallel(self) -> None:
        """Exit parallel mode."""
        self._parallel_mode = False
        self._parallel_total = 0
        self._parallel_completed = 0
        self._parallel_loading_msg = None

    def parallel_item_started(self, name: str) -> None:
        """Record that a parallel item has started (for logging only).

        In parallel mode, we log the start but don't update the display.
        The display will show just "[0/N]" until items complete, which
        accurately reflects that shared work (e.g., Dask model loading)
        is happening across all items.

        Parameters
        ----------
        name
            Name of the item that started.
        """
        self._log(f"  → {name} (started)")
        # Don't update _current_item - display shows just [0/N] until completions

    def parallel_item_completed(self, category: str, name: str, details: str = "") -> None:
        """Record that a parallel item has completed.

        Parameters
        ----------
        category
            Category of item (e.g., "pair").
        name
            Name of the completed item.
        details
            Optional details string.
        """
        self._parallel_completed += 1
        detail_str = f" - {details}" if details else ""
        self._log(f"  ✓ {name} ({self._parallel_completed}/{self._parallel_total}){detail_str}")
        self._stage_items.append((category, name))

        # Update display and ensure it's visible
        # When completions happen in rapid succession, pause so user can see each one
        self._current_item = name
        if self.show_output and self._live:
            self._live.update(self._create_stage_display())
            time.sleep(1.0)  # Pause so each completion is clearly visible

    def item_start(
        self, category: str, name: str, index: int, total: int, track: bool = True
    ) -> None:
        """Print item start (model, observation, pair).

        Parameters
        ----------
        track
            If True, add to stage_items for summary display. Set False for
            "in progress" messages where completion is tracked separately.
        """
        self._log(f"  → {name} ({index}/{total})")
        if track:
            self._stage_items.append((category, name))

        # Update the current item for animation display
        self._current_item = name
        self._current_progress = (index, total)

        # Force display update so each item is visible
        if self.show_output and self._live:
            self._live.update(self._create_stage_display())

    def step(self, message: str) -> None:
        """Print a step within an item."""
        self._log(f"      • {message}")
        # Steps are logged but not displayed during animation

    def item_done(self, summary: str) -> None:
        """Print item completion with summary."""
        self._log(f"      ✓ {summary}")
        # Completion is logged but animation continues

    def item_complete(
        self, category: str, name: str, index: int, total: int, details: str = ""
    ) -> None:
        """Print item completion with visible output (not just animation update)."""
        detail_str = f" - {details}" if details else ""
        self._log(f"  ✓ {name} ({index}/{total}){detail_str}")
        self._stage_items.append((category, name))

        # Update animation state
        self._current_item = name
        self._current_progress = (index, total)

        # Print visible output
        if self.show_output and self._live:
            self._live.update(self._create_stage_display())

    def item_fail(self, error: str) -> None:
        """Print item failure."""
        # Truncate error if too long
        max_len = 60
        if len(error) > max_len:
            error = error[: max_len - 3] + "..."
        self._log(f"      ✗ {error}")

        # Stop animation and show failure immediately
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._current_item = None
        self._current_progress = None
        self._print(f"    [{self.NCAR_RED}]✗ {error}[/{self.NCAR_RED}]")

    def footer(
        self,
        success: bool,
        duration: float,
        log_path: Path | None = None,
        failed_stage: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Print pipeline footer."""
        from rich.panel import Panel
        from rich.text import Text

        # Lighter red for error details
        NCAR_RED_LIGHT = "#E8788A"

        if success:
            msg = f"✓ Pipeline completed successfully in {duration:.1f}s"
            style = f"bold {self.NCAR_GREEN}"
            border_style = self.NCAR_GREEN
        else:
            msg = f"✗ Pipeline failed after {duration:.1f}s"
            style = f"bold {self.NCAR_RED}"
            border_style = self.NCAR_RED

        self._log("")
        self._log(msg)
        if log_path:
            self._log(f"Log: {log_path}")

        text = Text(msg, style=style)
        self._print(Panel(text, border_style=border_style, padding=(0, 2)))

        # Show error details for failed pipelines
        if not success and failed_stage and error_message:
            self._print(f"  [bold {self.NCAR_RED}]{failed_stage}[/bold {self.NCAR_RED}]")
            self._print(f"  [{NCAR_RED_LIGHT}]{error_message}[/{NCAR_RED_LIGHT}]")

        if log_path:
            self._print(f"  [dim]Log:[/dim] [white]{log_path}[/white]")
        self._print()

    def preview_plots(
        self,
        plot_paths: list[str],
        duration: float = 1.0,
        preview_format: Literal["pdf", "png"] = "pdf",
    ) -> None:
        """Show a slideshow preview of generated plots.

        Parameters
        ----------
        plot_paths
            List of paths to plot files to preview.
        duration
            How long to show each plot in seconds (for png format).
        preview_format
            Format to preview: "pdf" opens in system viewer, "png" shows in matplotlib.
        """
        from rich.live import Live
        from rich.text import Text

        if preview_format == "pdf":
            self._preview_pdfs(plot_paths, duration)
        else:
            self._preview_pngs(plot_paths, duration)

    def _preview_pdfs(self, plot_paths: list[str], duration: float = 1.0) -> None:
        """Show PDF plots one at a time using Quick Look.

        Parameters
        ----------
        plot_paths
            List of paths to PDF files.
        duration
            Seconds to display each plot before moving to the next.
        """
        import re
        import subprocess

        from rich.live import Live
        from rich.text import Text

        preview_files = sorted(p for p in plot_paths if p.endswith(".pdf") or p.endswith(".png"))

        if not preview_files:
            return

        n_files = len(preview_files)
        self._log(f"Previewing {n_files} plots...")
        self._print(f"  [dim]Previewing {n_files} plots...[/dim]")

        # Countdown before slideshow starts
        if self.show_output:
            with Live(console=self.console, refresh_per_second=4, transient=True) as live:
                for countdown in range(5, 0, -1):
                    text = Text()
                    text.append(f"  Starting slideshow in ", style="dim")
                    text.append(str(countdown), style=f"bold {self.NCAR_AQUA}")
                    live.update(text)
                    time.sleep(1.0)

        for i, file_path in enumerate(preview_files):
            try:
                # Strip date and index prefixes from filename for display
                # Format: {flight_date}_{index}_{plot_name} -> {plot_name}
                plot_name = Path(file_path).stem
                plot_name = re.sub(r"^\d+_\d+_", "", plot_name)
                self._print(f"  [dim][{i + 1}/{n_files}] {plot_name}[/dim]")

                # Open with Quick Look (handles both PDF and PNG on macOS)
                ql_proc = subprocess.Popen(
                    ["qlmanage", "-p", file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Wait for display duration
                time.sleep(duration)

                # Close Quick Look
                ql_proc.terminate()
                ql_proc.wait()

            except Exception as e:
                self._log(f"  Error previewing {file_path}: {e}")

    def _preview_pngs(self, plot_paths: list[str], duration: float = 1.0) -> None:
        """Show PNG plots in matplotlib window."""
        import re

        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        from rich.live import Live
        from rich.text import Text

        png_files = [p for p in plot_paths if p.endswith(".png")]

        if not png_files:
            return

        self._log(f"Previewing {len(png_files)} PNG plots...")

        # Countdown before preview
        if self.show_output:
            with Live(console=self.console, refresh_per_second=4, transient=True) as live:
                for countdown in range(5, 0, -1):
                    text = Text()
                    text.append(f"  Preparing to preview {len(png_files)} plots ... ", style="dim")
                    text.append(str(countdown), style=f"bold {self.NCAR_AQUA}")
                    live.update(text)
                    time.sleep(1.0)

        self._print(f"  [dim]Previewing {len(png_files)} plots...[/dim]")

        # Close any existing figures to start fresh
        plt.close("all")

        # Set up matplotlib for non-blocking display
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.manager.set_window_title("DAVINCI Plot Preview")  # type: ignore[union-attr]

        for i, png_path in enumerate(png_files):
            try:
                # Load and display image
                img = mpimg.imread(png_path)
                ax.clear()
                ax.imshow(img)
                ax.axis("off")

                # Show plot name in window (strip date and index prefixes)
                plot_name = Path(png_path).stem
                plot_name = re.sub(r"^\d+_\d+_", "", plot_name)
                ax.set_title(f"[{i + 1}/{len(png_files)}] {plot_name}", fontsize=10)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(duration)

            except Exception as e:
                self._log(f"  Error previewing {png_path}: {e}")

        plt.close("all")
        plt.ioff()

    def get_log_lines(self) -> list[str]:
        """Get all output lines for logging."""
        return self._lines.copy()


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
        """
        self._stages = list(stages) if stages is not None else create_standard_pipeline()
        self._fail_fast = fail_fast
        self._hooks = hooks or {}
        self._show_progress = show_progress
        self._show_plots = show_plots
        self._preview_format = preview_format

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
            self._stages.append(stage)  # type: ignore[arg-type]
        else:
            self._stages.insert(position, stage)  # type: ignore[arg-type]

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
        analysis_config = context.config.get("analysis", {})
        style_config = analysis_config.get("style")

        if style_config is None:
            return

        # Handle both dict and PlotStyleConfig object
        if hasattr(style_config, "theme"):
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
        import gc

        # Force garbage collection to close any dangling file handles
        gc.collect()

        # Clear xarray's file manager cache if available
        try:
            from xarray.backends.file_manager import FILE_CACHE

            FILE_CACHE.clear()
        except (ImportError, AttributeError):
            pass

        # Clear netCDF4's file cache if available
        try:
            import netCDF4

            # netCDF4 doesn't have a public cache clear, but gc.collect handles it
        except ImportError:
            pass

        logger.debug("Cleared HDF5/NetCDF file state")

    def _cleanup_context_datasets(self, context: PipelineContext) -> None:
        """Close all datasets in context to avoid transient file handle errors.

        This prevents crashes that can occur when Python's garbage collector
        tries to close stale NetCDF file handles after the pipeline completes.
        Should be called after log data extraction but before preview/exit.

        Note: Does NOT clear the dictionaries, as other code may still reference them.
        """
        import gc

        # Close model datasets
        for label, model_data in list(context.models.items()):
            try:
                if hasattr(model_data, "data") and hasattr(model_data.data, "close"):
                    model_data.data.close()
                elif hasattr(model_data, "close"):
                    model_data.close()
            except Exception:
                pass  # Ignore errors during cleanup

        # Close observation datasets
        for label, obs_data in list(context.observations.items()):
            try:
                if hasattr(obs_data, "data") and hasattr(obs_data.data, "close"):
                    obs_data.data.close()
                elif hasattr(obs_data, "close"):
                    obs_data.close()
            except Exception:
                pass  # Ignore errors during cleanup

        # Force garbage collection and clear file caches
        gc.collect()

        try:
            from xarray.backends.file_manager import FILE_CACHE

            FILE_CACHE.clear()
        except (ImportError, AttributeError):
            pass

        logger.debug("Closed all context datasets")

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

        # Clear HDF5/NetCDF state to avoid transient file handle errors
        self._cleanup_hdf5_state()

        # Apply plot styling from config if specified
        self._apply_plot_style(context)

        # Set up logging and formatting
        log_path: Path | None = None
        log_collector: LogCollector | None = None
        formatter = ProgressFormatter(show_output=self._show_progress)

        analysis_config = context.config.get("analysis", {})
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

        # Capture any externally-supplied progress callback (e.g. from the daemon
        # worker) before we overwrite it below.  The new internal callback will
        # forward every message to this outer callback *after* the formatter/log
        # logic, so both the display/log pipeline and the caller's callback are
        # notified.
        _outer_progress_callback: Callable[[str], None] | None = context.progress_callback

        # Set up progress callback that uses formatter and collects data
        def _make_progress_callback(
            fmt: ProgressFormatter,
            collector: LogCollector | None,
            outer: Callable[[str], None] | None = None,
        ) -> Callable[[str], None]:
            """Create progress callback for formatted output and log collection."""

            def callback(msg: str) -> None:
                # Collect structured data for Markdown log
                if collector:
                    collector.log_item(msg)
                # Parse message type and format appropriately
                if msg.strip().startswith("Loading model:"):
                    match = re.match(r"\s*Loading model: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("model", name, int(idx), int(total))
                elif msg.strip().startswith("Loading obs:"):
                    match = re.match(r"\s*Loading obs: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("obs", name, int(idx), int(total))
                # Parallel mode control messages
                elif msg.strip().startswith("parallel_start:"):
                    # Format: "parallel_start: N" or "parallel_start: N | loading_msg"
                    match = re.match(r"\s*parallel_start: (\d+)(?:\s*\|\s*(.+))?", msg)
                    if match:
                        total = int(match.group(1))
                        loading_msg = match.group(2)  # May be None
                        fmt.start_parallel(total, loading_msg)
                elif msg.strip().startswith("parallel_end"):
                    fmt.end_parallel()
                elif msg.strip().startswith("parallel_started:"):
                    # Item started in parallel mode (for logging, minimal display update)
                    match = re.match(r"\s*parallel_started: (\S+)", msg)
                    if match:
                        name = match.group(1)
                        fmt.parallel_item_started(name)
                elif msg.strip().startswith("parallel_completed:"):
                    # Item completed in parallel mode
                    match = re.match(r"\s*parallel_completed: (\S+)(.*)", msg)
                    if match:
                        name = match.group(1)
                        details = match.group(2).strip(" -") if match.group(2) else ""
                        fmt.parallel_item_completed("pair", name, details)
                # Legacy sequential pairing messages (for backward compatibility)
                elif msg.strip().startswith("Pairing:"):
                    # "Pairing:" = start, just update animation (don't track)
                    match = re.match(r"\s*Pairing: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("pair", name, int(idx), int(total), track=False)
                elif msg.strip().startswith("Paired:"):
                    # "Paired:" = completion, show progress and track for summary
                    match = re.match(r"\s*Paired: (\S+) \((\d+)/(\d+)\)(.*)", msg)
                    if match:
                        name, idx, total, details = match.groups()
                        details = details.strip(" -") if details else ""
                        fmt.item_complete("pair", name, int(idx), int(total), details)
                elif msg.strip().startswith("Stats:"):
                    match = re.match(r"\s*Stats: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("stats", name, int(idx), int(total))
                elif msg.strip().startswith("Plot:"):
                    match = re.match(r"\s*Plot: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("plot", name, int(idx), int(total))
                elif msg.strip().startswith("step:"):
                    # Step messages from stages
                    step_msg = msg.strip()[5:].strip()
                    fmt.step(step_msg)
                elif msg.strip().startswith("done:"):
                    # Completion messages from stages
                    done_msg = msg.strip()[5:].strip()
                    fmt.item_done(done_msg)

                # Forward to caller-supplied outer callback (e.g. daemon worker).
                if outer is not None:
                    outer(msg)

            return callback

        context.progress_callback = _make_progress_callback(
            formatter, log_collector, _outer_progress_callback
        )

        result = PipelineResult(
            success=True,
            context=context,
            start_time=datetime.now(),
        )

        start_time = time.time()
        self._call_hook("on_start", context)

        try:
            for stage in self._stages:
                # Start stage in formatter and collector
                formatter.stage_start(stage.name)
                if log_collector:
                    log_collector.start_stage(stage.name)

                stage_result = self._execute_stage(stage, context)
                result.stage_results.append(stage_result)

                # Store result in context
                context.results[stage.name] = stage_result

                # Finalize any open items before ending stage
                if log_collector:
                    log_collector.finalize_items()

                if stage_result.status == StageStatus.FAILED:
                    result.success = False
                    formatter.stage_end(stage.name, False, stage_result.duration_seconds)
                    if log_collector:
                        log_collector.end_stage(stage.name, "failed", stage_result.duration_seconds)
                        # Log the error with traceback for the Markdown report
                        if stage_result.error:
                            log_collector.log_error(
                                stage_name=stage.name,
                                error_type=stage_result.error_type or "Exception",
                                error_message=stage_result.error,
                                traceback_str=stage_result.traceback_str,
                            )
                    if self._fail_fast:
                        break
                elif stage_result.status == StageStatus.SKIPPED:
                    formatter.stage_end(stage.name, True, stage_result.duration_seconds)
                    if log_collector:
                        log_collector.end_stage(
                            stage.name, "skipped", stage_result.duration_seconds
                        )
                elif stage_result.status == StageStatus.COMPLETED:
                    formatter.stage_end(stage.name, True, stage_result.duration_seconds)
                    if log_collector:
                        log_collector.end_stage(
                            stage.name, "completed", stage_result.duration_seconds
                        )

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

            # Write Markdown log file
            if log_collector and log_path:
                log_collector.end_pipeline(result.success)
                # Extract detailed data from context for the report
                log_collector.extract_context_data(context)
                try:
                    log_path.write_text(log_collector.to_markdown())
                except Exception as e:
                    logger.warning(f"Failed to write log file: {e}")

            # Close all datasets to prevent transient file handle errors during preview/exit
            self._cleanup_context_datasets(context)

            # Preview generated plots if pipeline succeeded and show_plots is enabled
            if self._show_plots and result.success:
                plot_paths: list[str] = []
                for stage_name in ("plotting", "obs_plotting"):
                    if stage_name in context.results:
                        stage_result = context.results[stage_name]
                        if stage_result.data and "plots_generated" in stage_result.data:
                            plot_paths.extend(stage_result.data["plots_generated"])
                if plot_paths:
                    formatter.preview_plots(
                        plot_paths, duration=1.0, preview_format=self._preview_format
                    )

        # Store the log_path in context metadata so callers (e.g. the daemon
        # worker) can retrieve the real log file path without having to mirror
        # the runner's path construction logic.
        if log_path is not None and context is not None:
            context.metadata["log_path"] = str(log_path)

        result.end_time = datetime.now()
        result.total_duration_seconds = time.time() - start_time

        self._call_hook("on_end", result)

        return result

    def run_from_config(self, config: dict[str, Any] | str) -> PipelineResult:
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
        from davinci_monet.core.exceptions import ConfigurationError

        config_path: str | None = None
        if isinstance(config, str):
            config_path = config
            from davinci_monet.config import load_config

            config = load_config(config).model_dump()

        used_sources = bool(config.get("sources"))

        # Validate that config has something to process
        model_config = config.get("model") or {}
        obs_config = config.get("obs") or {}
        sources_config = config.get("sources") or {}

        # Deprecation: a legacy model:/obs: config still works (auto-converted),
        # but the unified `sources:` schema is the going-forward format.
        if not used_sources and (model_config or obs_config):
            import warnings

            from davinci_monet.config.migration import LegacyConfigWarning

            warnings.warn(
                "The 'model:'/'obs:' config schema is deprecated; use the unified "
                "'sources:' schema. Convert with: davinci-monet migrate-config "
                "<config.yaml> -o <new.yaml>.",
                LegacyConfigWarning,
                stacklevel=2,
            )

        if not model_config and not obs_config and not sources_config:
            raise ConfigurationError(
                "Configuration is empty or incomplete. "
                "At least one source, model, or observation must be defined."
            )

        # The unified standard pipeline handles both model-vs-obs and obs-only
        # runs: the pairing/statistics/plotting stages skip when there are no
        # pairs, and the obs-only stages skip when pairs exist. No special-case
        # pipeline swap is needed.

        context = PipelineContext(config=config)
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
                logger.warning(f"Stage '{stage.name}' validation failed, skipping")
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                    error="Validation failed",
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
    ...     .add_models()
    ...     .add_observations()
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

    def add_models(self) -> PipelineBuilder:
        """Add model loading stage."""
        from davinci_monet.pipeline.stages import LoadModelsStage

        self._stages.append(LoadModelsStage())
        return self

    def add_observations(self) -> PipelineBuilder:
        """Add observation loading stage."""
        from davinci_monet.pipeline.stages import LoadObservationsStage

        self._stages.append(LoadObservationsStage())
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
    config: dict[str, Any] | str,
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
