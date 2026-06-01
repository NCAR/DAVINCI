"""Main CLI application for DAVINCI.

This module provides the command-line interface using Typer.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

try:
    import typer
except ImportError as e:
    print(
        "The DAVINCI CLI requires the module 'typer'. "
        "You can install it with `conda install -c conda-forge typer` or "
        "`pip install typer`. "
        f"The error message was: {e}"
    )
    raise SystemExit(1)


# Global debug flag
DEBUG = False

# Color scheme
INFO_COLOR = typer.colors.CYAN
ERROR_COLOR = typer.colors.BRIGHT_RED
SUCCESS_COLOR = typer.colors.GREEN
WARNING_COLOR = typer.colors.YELLOW

# Rich color constants (for styled error display)
NCAR_BLUE = "#0A5DDA"
NCAR_AQUA = "#00A2B4"
NCAR_RED = "#D62839"
NCAR_RED_LIGHT = "#E8788A"  # Lighter red for error details


def _get_system_info() -> str:
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


def display_error(title: str, message: str, config_path: str | None = None) -> None:
    """Display a styled error message with the DAVINCI branding.

    Shows the logo, a styled panel (matching pipeline header style), and
    the error message in red. Used for early errors (YAML parsing, validation)
    before the pipeline starts.

    Parameters
    ----------
    title
        Error title (e.g., "Configuration Error", "File Not Found").
    message
        The error message to display.
    config_path
        Optional path to the config file that caused the error.
    """
    from datetime import datetime

    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from davinci_monet.assets.logo import get_colored_logo

    console = Console()

    # Clear screen and show logo
    console.clear()
    console.print()
    console.print(get_colored_logo())

    # Panel content - same style as pipeline header (no error type in panel)
    content = Text()
    content.append("DAVINCI", style=f"bold {NCAR_AQUA}")
    content.append("  ")
    content.append(datetime.now().strftime("%a %b %-d, %Y %H:%M"), style="dim")
    content.append("  ")
    system_info = _get_system_info()
    content.append(system_info, style="dim")
    console.print(Panel(content, border_style=NCAR_AQUA, padding=(0, 2)))

    # Show config path if provided
    if config_path:
        # Truncate path if too long
        max_path_len = 70
        display_path = config_path
        if len(config_path) > max_path_len:
            display_path = "..." + config_path[-(max_path_len - 3) :]
        console.print(f"  [dim]Config:[/dim] {display_path}")

    console.print()

    # Error type in bold red
    console.print(f"  [bold {NCAR_RED}]{title}[/bold {NCAR_RED}]")

    # Error details in lighter red
    console.print(f"  [{NCAR_RED_LIGHT}]{message}[/{NCAR_RED_LIGHT}]")
    console.print()


def _get_full_name(obj: object) -> str:
    """Get the full name of a function or type, including the module name."""
    import builtins
    import inspect

    mod = inspect.getmodule(obj)
    name = getattr(obj, "__qualname__", str(obj))
    if mod is None or mod is builtins:
        return name
    else:
        return f"{mod.__name__}.{name}"


@contextmanager
def timer(desc: str = "") -> Generator[None, None, None]:
    """Context manager for timing operations with colored output.

    Parameters
    ----------
    desc
        Description of the operation being timed.

    Yields
    ------
    None
    """
    start = time.perf_counter()
    tpl = f"{desc} {{status}} in {{elapsed:.3g}} seconds"

    typer.secho(f"{desc} ...", fg=INFO_COLOR)
    try:
        yield
    except Exception as e:
        typer.secho(
            tpl.format(status="failed", elapsed=time.perf_counter() - start),
            fg=ERROR_COLOR,
        )
        typer.secho(f"Error message (type: {_get_full_name(type(e))}): {e}", fg=ERROR_COLOR)
        if DEBUG:
            raise
        else:
            typer.echo("(Use the '--debug' flag to see more info.)")
            raise typer.Exit(1)
    else:
        typer.secho(
            tpl.format(status="succeeded", elapsed=time.perf_counter() - start),
            fg=SUCCESS_COLOR,
        )


def _version_callback(value: bool) -> None:
    """Display version information."""
    from davinci_monet import __version__

    if value:
        typer.echo(f"davinci-monet {__version__}")
        raise typer.Exit()


# Create the main application
app = typer.Typer(
    name="davinci-monet",
    help="DAVINCI: Data Analysis and Validation Infrastructure for Chemistry",
    add_completion=False,
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """DAVINCI: Data Analysis and Validation Infrastructure for Chemistry.

    A modern tool for evaluating atmospheric chemistry and air quality
    models against observations.
    """


# Import and register commands
def register_commands() -> None:
    """Register all CLI commands."""
    # Import command modules
    from davinci_monet.cli.commands import daemon, get_data, run, validate

    # Register subcommands
    app.add_typer(get_data.app, name="get")
    app.add_typer(daemon.app, name="daemon")


# Register commands when module loads
register_commands()


# Direct commands are added via decorators
@app.command()
def run(
    control: str = typer.Argument(
        ...,
        help="Path to the control file (YAML configuration).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Print more messages (including full tracebacks).",
    ),
    show_plots: bool = typer.Option(
        False,
        "--show-plots",
        help="Display interactive plot preview after pipeline completes (requires display).",
    ),
    preview_format: str = typer.Option(
        "pdf",
        "--preview-format",
        "-p",
        help="Format for plot preview: 'pdf' (opens in system viewer) or 'png' (matplotlib window).",
    ),
) -> None:
    """Run DAVINCI analysis as described in the control file."""
    from davinci_monet.cli.commands.run import run_analysis

    global DEBUG
    DEBUG = debug

    run_analysis(control, debug=debug, show_plots=show_plots, preview_format=preview_format)


@app.command()
def validate(
    control: str = typer.Argument(
        ...,
        help="Path to the control file (YAML configuration) to validate.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Use strict validation (no extra fields allowed).",
    ),
    show_config: bool = typer.Option(
        False,
        "--show",
        help="Print the parsed configuration.",
    ),
) -> None:
    """Validate a DAVINCI configuration file."""
    from davinci_monet.cli.commands.validate import validate_config_command

    validate_config_command(control, strict=strict, show_config=show_config)


@app.command("migrate-config")
def migrate_config(
    control: str = typer.Argument(
        ...,
        help="Path to the legacy control file (model:/obs: form) to migrate.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to write the migrated control file (unified sources: form).",
    ),
) -> None:
    """Migrate a legacy model/obs control file to the unified sources schema."""
    from davinci_monet.cli.commands.migrate import migrate_config_command

    migrate_config_command(control, output)


# CLI entry point
cli = app

# For sphinx-click documentation
_typer_click_object = typer.main.get_command(app)


if __name__ == "__main__":
    cli()
