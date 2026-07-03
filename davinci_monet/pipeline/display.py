"""Pipeline display — rich/terminal progress UI.

Contains :class:`ProgressFormatter`, which renders animated stage progress,
a pulsing DAVINCI header, elapsed-time counters, and plot previews to the
terminal during pipeline execution.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping


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
        # Transient step detail — rendered as '› detail' when no per-item display active
        self._current_step: str | None = None

    def _log(self, line: str) -> None:
        """Store a line for log file."""
        self._lines.append(line)

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print via rich console if output enabled."""
        if self.show_output:
            self.console.print(*args, **kwargs)

    def print_summary(
        self,
        items: list[str],
        summary_file: str | None = None,
        usage: dict[str, Any] | None = None,
        credits_remaining: float | None = None,
    ) -> None:
        """Render an itemized AI summary to the terminal at end of run.

        Shows the condensed bullet list, then (dim) tokens used, OpenRouter
        credits remaining (when available), and a pointer to the full-brief
        file. No-op when output is disabled or there are no items.
        """
        if not self.show_output or not items:
            return
        from rich.panel import Panel

        body_lines = [f"• {item}" for item in items]
        meta: list[str] = []
        if (
            usage
            and usage.get("input_tokens") is not None
            and usage.get("output_tokens") is not None
        ):
            tin = int(usage["input_tokens"])
            tout = int(usage["output_tokens"])
            meta.append(f"Tokens: {tin:,} in / {tout:,} out ({tin + tout:,} total)")
        if credits_remaining is not None:
            meta.append(f"OpenRouter credits: ${credits_remaining:.2f} remaining")
        if summary_file:
            meta.append(f"Full brief → {summary_file}")
        if meta:
            body_lines.append("")
            body_lines.extend(f"[dim]{line}[/dim]" for line in meta)

        self._print()
        self._print(
            Panel(
                "\n".join(body_lines),
                title="AI Summary",
                border_style=self.NCAR_AQUA,
                padding=(1, 2),
            )
        )
        self._print()

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
        # Sequential mode: show current item or step detail
        elif self._current_item:
            result.append(" › ", style="dim")
            if self._current_progress:
                idx, total = self._current_progress
                result.append(f"[{idx}/{total}] ", style="dim")
            result.append(self._current_item, style="white")
        elif self._current_step:
            result.append(" › ", style="dim")
            result.append(self._current_step, style="dim italic")

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
            from davinci_monet.util.system import _cpu_name_from_cpuinfo

            try:
                with open("/proc/cpuinfo") as f:
                    cpu_name = _cpu_name_from_cpuinfo(f.read())
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
        self._current_step = None
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
        self._current_step = None

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
            Optional message to show during [0/N] phase (e.g., "loading source-a → source-b").
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
        accurately reflects that shared work (e.g., Dask dataset loading)
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
        """Print item start (dataset, dataset, pair).

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
        self._current_step = message
        # Force an immediate display refresh so the detail appears without
        # waiting for the next 0.15 s animation tick.
        if self.show_output and self._live:
            self._live.update(self._create_stage_display())

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

    def print_item_errors(
        self,
        item_errors: Mapping[str, list[Any] | None],
        *,
        pipeline_success: bool = False,
    ) -> None:
        """Surface non-fatal per-item errors collected during the run.

        Stages continue past an individual item failure and stash the error in
        ``context.metadata`` rather than failing the whole pipeline. Those
        errors used to be collected but never shown, so a run could report
        success while silently dropping pairs, stats, plots, or analyses. This
        renders a concise amber summary; full detail (with the count) is in the
        Markdown log.
        """
        stage_labels = {
            "pairing_errors": "pairing",
            "stats_errors": "statistics",
            "plot_errors": "plotting",
            "analysis_errors": "analyses",
        }
        total = sum(len(v or []) for v in item_errors.values())
        if not total:
            return

        suffix = " (pipeline still succeeded)" if pipeline_success else ""
        self._log("")
        self._log(f"{total} non-fatal error(s) occurred{suffix}:")
        self._print(
            f"  [bold {self.NCAR_ORANGE}]⚠ {total} non-fatal error(s) "
            f"occurred{suffix}:[/bold {self.NCAR_ORANGE}]"
        )
        for key, errors in item_errors.items():
            stage = stage_labels.get(key, key)
            for message in errors or []:
                self._log(f"  [{stage}] {message}")
                self._print(f"    [{self.NCAR_ORANGE}]• [{stage}] {message}[/{self.NCAR_ORANGE}]")
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
