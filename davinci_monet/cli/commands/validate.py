"""Validate command for DAVINCI CLI.

This module implements the configuration validation command.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable

import typer

from davinci_monet.cli.app import (
    ERROR_COLOR,
    INFO_COLOR,
    SUCCESS_COLOR,
    display_error,
)
from davinci_monet.core.exceptions import ConfigurationError


def _iter_source_paths(value: str | Path | list[str | Path] | None) -> Iterable[Path]:
    """Yield configured source paths, preserving glob patterns."""
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            yield Path(item)
        return
    yield Path(value)


def _resolve_config_path(path: Path, config_dir: Path) -> Path:
    """Resolve relative source paths against the config file directory."""
    if path.is_absolute():
        return path
    return config_dir / path


def _path_has_glob(path: Path) -> bool:
    return any(char in path.as_posix() for char in "*?[]")


def _validate_declared_source_paths(config: object, config_dir: Path) -> None:
    """Reject configured source paths that do not exist."""
    sources = getattr(config, "sources", {})
    for source_name, source_cfg in sources.items():
        for field_name in ("files", "filename"):
            raw_value = getattr(source_cfg, field_name, None)
            for raw_path in _iter_source_paths(raw_value):
                path = _resolve_config_path(raw_path.expanduser(), config_dir)
                if _path_has_glob(path):
                    if not glob(path.as_posix()):
                        raise ConfigurationError(
                            f"sources.{source_name}.{field_name} matched no files: {path}"
                        )
                elif not path.exists():
                    raise ConfigurationError(
                        f"sources.{source_name}.{field_name} does not exist: {path}"
                    )


def _validate_registered_source_types(config: object) -> None:
    """Reject source types not registered with DAVINCI."""
    import davinci_monet.datasets  # noqa: F401  # ensure built-in readers register
    from davinci_monet.core.registry import source_registry

    sources = getattr(config, "sources", {})
    for source_name, source_cfg in sources.items():
        source_type = getattr(source_cfg, "type", None)
        if not source_type:
            raise ConfigurationError(f"sources.{source_name}.type is required")
        if source_type not in source_registry:
            available = ", ".join(sorted(source_registry))
            raise ConfigurationError(
                f"sources.{source_name}.type {source_type!r} is not registered. "
                f"Available source types: {available}"
            )


def _validate_analysis_window(config: object) -> None:
    """Reject missing or inverted analysis windows in CLI control files."""
    analysis = getattr(config, "analysis", None)
    if analysis is None:
        raise ConfigurationError("analysis section is required")
    start_time = getattr(analysis, "start_time", None)
    end_time = getattr(analysis, "end_time", None)
    if start_time is None:
        raise ConfigurationError("analysis.start_time is required")
    if end_time is None:
        raise ConfigurationError("analysis.end_time is required")
    if end_time <= start_time:
        raise ConfigurationError("analysis.end_time must be after analysis.start_time")


def _validate_control_file_semantics(config: object, config_path: Path) -> None:
    """Run semantic checks expected from ``davinci validate``."""
    _validate_analysis_window(config)
    _validate_registered_source_types(config)
    _validate_declared_source_paths(config, config_path.parent)


def validate_config_command(
    control_path: str,
    strict: bool = False,
    show_config: bool = False,
    readiness: bool = False,
    json_output: bool = False,
) -> None:
    """Validate a DAVINCI configuration file.

    Parameters
    ----------
    control_path
        Path to the YAML control file.
    strict
        If True, reject unknown fields in core config sections.
    show_config
        If True, print the parsed configuration.
    readiness
        If True, evaluate scheduled-run readiness after semantic validation.
    json_output
        If True, print only the machine-readable readiness report.
    """
    p = Path(control_path)
    if not p.is_file():
        if json_output:
            import json

            typer.echo(json.dumps({"ready": False, "error": "control file does not exist"}))
            raise typer.Exit(2)
        typer.secho(f"Error: control file {control_path!r} does not exist", fg=ERROR_COLOR)
        raise typer.Exit(2)

    if json_output and not readiness:
        typer.echo('{"ready": false, "error": "--json requires --readiness"}')
        raise typer.Exit(2)

    if not json_output:
        typer.secho(f"Validating: {control_path!r}", fg=INFO_COLOR)
        typer.secho(f"Full path: {p.absolute().as_posix()}", fg=INFO_COLOR)
        typer.secho(f"Mode: {'strict' if strict else 'flexible'}", fg=INFO_COLOR)
        typer.echo()

    try:
        from davinci_monet.config import load_config

        # Parse and validate.
        config = load_config(p, strict=strict)
        _validate_control_file_semantics(config, p)
        readiness_report = None
        if readiness:
            from davinci_monet.validation import evaluate_run_readiness

            readiness_report = evaluate_run_readiness(config, p)

        if json_output:
            import json

            assert readiness_report is not None
            typer.echo(json.dumps(readiness_report.to_dict(), indent=2, sort_keys=True))
            if not readiness_report.ready:
                raise typer.Exit(1)
            return

        # Report what was found
        typer.echo()
        typer.secho("Configuration summary:", fg=INFO_COLOR)

        # Analysis section
        if config.analysis:
            typer.echo(f"  Analysis:")
            typer.echo(f"    Start: {config.analysis.start_time}")
            typer.echo(f"    End: {config.analysis.end_time}")
            if config.analysis.output_dir:
                typer.echo(f"    Output dir: {config.analysis.output_dir}")

        # Unified sources
        if config.sources:
            typer.echo(f"  Sources: {len(config.sources)} defined")
            for name, source_cfg in config.sources.items():
                typer.echo(f"    - {name}: {source_cfg.type}")

        if config.analyses:
            typer.echo(f"  Derived analyses: {len(config.analyses)} defined")
            for name, analysis_cfg in config.analyses.items():
                typer.echo(f"    - {name}: {analysis_cfg.type}")

        if config.run is not None:
            typer.echo(f"  Run: {config.run.id} ({config.run.kind})")
            if config.run.completion is not None:
                typer.echo(
                    "    Required analyses: " + ", ".join(config.run.completion.required_analyses)
                )
                typer.echo("    Required plots: " + ", ".join(config.run.completion.required_plots))

        # Unified pairs
        if config.pairs:
            typer.echo(f"  Pairs: {len(config.pairs)} defined")
            for name, pair_cfg in config.pairs.items():
                axes = (
                    f"x={pair_cfg.x.source}:{pair_cfg.x.variable}, "
                    f"y={pair_cfg.y.source}:{pair_cfg.y.variable}"
                )
                typer.echo(f"    - {name}: {axes}")

        # Plots
        if config.plots:
            typer.echo(f"  Plots: {len(config.plots)} defined")
            for name, plot_cfg in config.plots.items():
                typer.echo(f"    - {name}: {plot_cfg.type}")

        # Stats
        if config.stats:
            typer.echo(f"  Statistics: configured")

        typer.echo()
        typer.secho("Validation passed!", fg=SUCCESS_COLOR)

        if readiness_report is not None:
            typer.echo()
            typer.secho("Readiness:", fg=INFO_COLOR)
            for check in readiness_report.checks:
                marker = {
                    "passed": "PASS",
                    "failed": "FAIL",
                    "skipped": "SKIP",
                }[check.status]
                typer.echo(f"  {marker} {check.name}: {check.detail}")
            if readiness_report.ready:
                typer.secho("Run readiness passed!", fg=SUCCESS_COLOR)
            else:
                typer.secho("Run readiness failed!", fg=ERROR_COLOR)
                raise typer.Exit(1)

        # Show full config if requested
        if show_config:
            typer.echo()
            typer.secho("Parsed configuration:", fg=INFO_COLOR)
            typer.echo("-" * 40)

            import json

            # Convert to dict and display
            from davinci_monet.core.schema_utils import dump_schema

            config_dict = dump_schema(config, exclude_none=True)
            typer.echo(json.dumps(config_dict, indent=2, default=str))

    except ConfigurationError as e:
        if json_output:
            import json

            typer.echo(json.dumps({"ready": False, "error": str(e)}, sort_keys=True))
            raise typer.Exit(1)
        # Styled display for configuration/YAML errors
        display_error("Validation Error", str(e), config_path=control_path)
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        if json_output:
            import json

            typer.echo(json.dumps({"ready": False, "error": str(e)}, sort_keys=True))
            raise typer.Exit(1)
        # Styled display for unexpected errors
        display_error("Error", str(e), config_path=control_path)
        raise typer.Exit(1)
