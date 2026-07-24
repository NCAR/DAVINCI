# Contributing to DAVINCI

Thank you for your interest in contributing. This guide covers the essentials
for getting started.

## Development Setup

1. Clone the repository and create the conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate davinci
   python -m pip install --no-deps pycwt==0.4.0b0
   python -m pip install --no-deps -e ".[dev]"
   davinci --version
   ```

2. Verify your setup:

   ```bash
   pytest
   mypy davinci_monet
   ```

## Running Checks

Before submitting changes, run all three:

```bash
pytest                  # Tests
mypy davinci_monet      # Type checking
black davinci_monet && isort davinci_monet  # Formatting
```

## Git Workflow

- **`develop`** is the primary working branch. Create feature branches from it
  and open PRs back to `develop`.
- **`main`** is for releases only. Do not open PRs directly to `main` unless
  cutting a release.
- Keep commits focused. Use descriptive commit messages that explain *why*,
  not just *what*.

## Code Style

- **Formatting**: black + isort (configured in `pyproject.toml`).
- **Type hints**: Required on all public functions and methods.
- **Module size**: Keep modules under 500 lines. If a file is growing past
  that, split it.
- **Data dataset**: Use xarray throughout pairing and analysis. Pandas is
  acceptable for I/O adapters and statistics output tables only.
- **Variable naming**: Paired datasets from the pipeline use source-label prefix
  format (`<source_label>_<var>`, e.g. `cam_pm25`/`airnow_pm25`), with `pair_axis` and
  `source_label` attrs on each variable; the low-level pairing API still emits
  `dataset_<var>`/`geometry_<var>` when no source labels are supplied. Either way it is
  prefix, not suffix. See the Variable Naming Convention section in CLAUDE.md.

## Project Conventions

See `CLAUDE.md` for detailed conventions including:

- Pipeline stage contracts and YAML configuration patterns
- Plot styling (NCAR branding, color conventions)
- CESM vertical coordinate handling
- The pre-implementation audit checklist (required before building new
  components)

## Cross-Dataset AI Review

This project uses AI-assisted development with cross-dataset code reviews.
If you find `REVIEW_*.md` or `HANDOFF_*.md` files in the repo root, these
are structured handoff documents for AI review workflows. See the
"Cross-Dataset Handoff Convention" section in `CLAUDE.md` for the format.

## Reporting Issues

Open a GitHub issue with:

- What you expected to happen
- What actually happened
- Steps to reproduce (include config YAML if relevant)
- Python and conda environment versions (`conda list`)

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0. See `LICENSE` for the full text.
