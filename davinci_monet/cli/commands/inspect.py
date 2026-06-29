"""Inspect command for DAVINCI CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from davinci_monet.cli.app import ERROR_COLOR, SUCCESS_COLOR
from davinci_monet.inspection import inspect_run_directory


def inspect_command(run_dir: Path, presets: list[str]) -> None:
    """Run deterministic inspection checks for a DAVINCI run directory."""
    try:
        result = inspect_run_directory(run_dir, presets=presets)
    except (FileNotFoundError, NotADirectoryError) as exc:
        typer.secho(str(exc), fg=ERROR_COLOR)
        raise typer.Exit(1) from exc

    if result.passed:
        typer.secho(f"Inspection passed: {result.markdown_path}", fg=SUCCESS_COLOR)
        return

    typer.secho(f"Inspection failed: {result.markdown_path}", fg=ERROR_COLOR)
    raise typer.Exit(1)
