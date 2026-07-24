"""Command-line interface for DAVINCI.

This module provides the CLI using Typer for running analyses,
downloading data, and validating configurations.

Quick Start
-----------
>>> # From command line:
>>> davinci run config.yaml
>>> davinci validate config.yaml
>>> davinci get aeronet -s 2024-01-01 -e 2024-01-31

Programmatic Usage
------------------
>>> from davinci_monet.cli import app, cli
>>> # Run the CLI programmatically
>>> cli(["run", "config.yaml"])
"""

from davinci_monet.cli.app import (
    DEBUG,
    ERROR_COLOR,
    INFO_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    app,
    cli,
    timer,
)

__all__ = [
    "app",
    "cli",
    "timer",
    "INFO_COLOR",
    "ERROR_COLOR",
    "SUCCESS_COLOR",
    "WARNING_COLOR",
    "DEBUG",
]
