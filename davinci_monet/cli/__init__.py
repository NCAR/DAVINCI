"""Command-line interface for DAVINCI.

This module provides the CLI using Typer for running analyses,
downloading data, and validating configurations.

Quick Start
-----------
>>> # From command line:
>>> davinci-monet run config.yaml
>>> davinci-monet validate config.yaml
>>> davinci-monet get aeronet -s 2024-01-01 -e 2024-01-31

Programmatic Usage
------------------
>>> from davinci_monet.cli import app, cli
>>> # Run the CLI programmatically
>>> cli(["run", "config.yaml"])
"""

from __future__ import annotations

# Lazy re-exports so that importing a subpackage of davinci_monet.cli
# (e.g. davinci_monet.cli.commands.daemon) does NOT trigger cli.app at
# module-load time.  cli.app calls register_commands() at import, which
# pulls in the full pipeline (xarray / matplotlib / monet), violating the
# daemon isolation contract.
#
# Any existing code that does `from davinci_monet.cli import app` still
# works — Python calls __getattr__ on the first attribute access.

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

_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> object:
    if name in _ATTRS:
        from davinci_monet.cli import app as _app_module  # noqa: PLC0415

        return getattr(_app_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
