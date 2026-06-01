"""CLI command modules for DAVINCI."""

# Submodules are imported lazily by cli.app.register_commands() and directly
# by callers (e.g. `from davinci_monet.cli.commands.get_data import ...`).
# Eager top-level imports here would create a circular import cycle:
#   commands/__init__ -> get_data.py -> cli.app -> register_commands()
#   -> `from davinci_monet.cli.commands import get_data` (partially initialised)
# and would also pull in xarray/matplotlib/monet at package load time,
# violating the daemon isolation contract.

__all__ = [
    "daemon",
    "get_data",
    "run",
    "validate",
]
