"""CLI command modules for DAVINCI."""

# Intentionally empty: submodules are imported lazily by cli.app.register_commands()
# and directly by callers (e.g. `from davinci_monet.cli.commands.get_data import ...`).
# Eager imports here would pull in the full pipeline at module load time, violating
# the daemon isolation contract (importing davinci_monet.cli.commands.daemon must
# not load xarray / matplotlib / monet).
