> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or an equivalent task-by-task workflow. Steps use checkbox (`- [ ]`) syntax for tracking. Do not commit or push without explicit user approval.

# Dependency Upgrade Remediation

**Status:** Planned after clean environment solve; implementation not started.

**Branch target:** `develop`

**Scope:** Make DAVINCI source and package metadata compatible with the new clean conda-first `davinci` environment.

**Out of scope:** Restoring `wrf-python`, WRF/WRF-Chem reader support, moving beyond NumPy 2.4, or migrating to Python 3.14.

## Implementation Readiness

Prepared on `develop`. Working tree already contains the `environment.yml` upgrade and this untracked plan file; do not overwrite unrelated user work.

Verified baseline:

- `mamba run -n davinci python` reports Python 3.13.14, NumPy 2.4.6, pandas 3.0.3, xarray 2026.4.0, SciPy 1.18.0, Matplotlib 3.11.0, and numba 0.65.1.
- `mamba run -n davinci davinci-monet --help` passes.
- `mamba run -n davinci python -m pip check` fails only on the expected metadata conflicts:
  - `davinci-monet` requires `matplotlib<3.9,>=3.5`.
  - `davinci-monet` requires `pandas<2,>=1.5`.
  - `pycwt` requires `numpy<2,>=1.24`.

First implementation pass should touch only:

- `pyproject.toml`: update Python classifiers/tool targets, pandas and Matplotlib dependency ranges, and decide the `pycwt` dependency location.
- `README.md`: update user-facing environment setup from `pip install -e ".[dev]"` to conda-first plus explicit `--no-deps` pip installs.
- `CLAUDE.md`: update agent-facing environment setup and keep the warning about using the `davinci` conda environment.

Defer source compatibility edits until a focused gate fails. Current pre-audit found only a few likely `Dataset.dims` size-mapping assertions in tests (`davinci_monet/tests/unit/pairing/test_swath_grid_strategy.py`) and one implementation size lookup candidate (`davinci_monet/datasets/satellite/generic_l2.py`). Most `.dims` usage is dimension-name tuple logic and should not be changed mechanically.

Recommended implementation order:

1. Update package metadata and docs.
2. Reinstall editable with `mamba run -n davinci python -m pip install --no-deps -e .`.
3. Run `mamba run -n davinci python -m pip check`; expected result is only the PyCWT NumPy metadata conflict unless PyCWT is moved out of core metadata.
4. Run import, CLI, PyCWT, pandas/xarray, plotting, and numba gates below.
5. Make targeted source/test changes only for observed failures.

## Context

The `davinci` environment has been rebuilt from `environment.yml` with a conda-owned scientific stack:

- Python 3.13.14
- NumPy 2.4.6
- numba 0.65.1
- pandas 3.0.3
- xarray 2026.4.0
- SciPy 1.18.0
- Matplotlib 3.11.0
- Cartopy 0.25.0
- MONET 2.3.1 / MONETIO 0.3.2

Only two packages are installed by pip:

- `pycwt==0.4.0b0`
- local editable `davinci-monet`

Both are installed manually with `--no-deps` after `mamba env create -f environment.yml`, because the `pip:` section in conda environment files does not support the global `--no-deps` option reliably.

Current smoke checks:

- Core imports pass.
- `davinci-monet --help` passes.
- `pip check` fails only on metadata:
  - `davinci-monet` still declares `pandas<2`
  - `davinci-monet` still declares `matplotlib<3.9`
  - `pycwt` declares `numpy<2`

## Decisions & Rationale

1. **Python 3.13 is the upgrade target.**
   Python 3.14 does not solve cleanly with the current geo/science stack. Python 3.13 gives modern package support without taking the newest interpreter risk.

2. **NumPy 2.4 is the NumPy target.**
   NumPy 2.5 currently conflicts with `numba`, and numba is essential for `davinci_monet/pairing/grid_binning.py`.

3. **Conda-forge owns all compiled/scientific packages.**
   Pip must not install or upgrade NumPy, pandas, xarray, Matplotlib, Cartopy, MONET, MONETIO, netCDF4, numba, or related compiled packages.

4. **PyCWT stays as `0.4.0b0` for now.**
   Runtime probes passed under NumPy 2.4 and 2.5. The NumPy `<2` cap appears to be metadata, not an immediate runtime failure. Long-term fix is a patched/forked PyCWT package or upstream metadata update.

5. **WRF support remains dropped from the environment.**
   `wrf-python` is intentionally absent.

## Task 1: Fix DAVINCI Package Metadata

- [ ] Update `pyproject.toml` runtime requirements:
  - [ ] Replace `pandas>=1.5,<2` with a pandas 3-compatible range.
  - [ ] Replace `matplotlib>=3.5,<3.9` with a Matplotlib 3.11-compatible range.
  - [ ] Decide whether to keep `pycwt==0.4.0b0` in core dependencies or move it to an optional extra while environment installation remains manual/no-deps.
- [ ] Update classifiers and Python metadata for Python 3.13 support.
- [ ] Reinstall editable with:
  ```bash
  mamba run -n davinci python -m pip install --no-deps -e .
  ```
- [ ] Run:
  ```bash
  mamba run -n davinci python -m pip check
  ```
  Expected after this task: only the PyCWT NumPy metadata conflict may remain.

## Task 2: Add Environment Installation Documentation

- [ ] Update `CLAUDE.md` and/or `README.md` environment instructions:
  - [ ] `mamba env create -f environment.yml`
  - [ ] `mamba activate davinci`
  - [ ] `python -m pip install --no-deps pycwt==0.4.0b0`
  - [ ] `python -m pip install --no-deps -e .`
- [ ] Document why pip dependencies are disabled: avoid wheel overlays of the conda scientific stack.
- [ ] Note that `pycwt` metadata is temporarily incompatible with NumPy 2 even though the runtime smoke test passes.

## Task 3: Run Focused Import And CLI Gates

- [ ] Run import smoke:
  ```bash
  mamba run -n davinci python - <<'PY'
  import numpy, numba, pandas, xarray, scipy, matplotlib, cartopy
  import dask, distributed, netCDF4, h5py, h5netcdf, pyresample, xesmf
  import monet, monetio, statsmodels, pycwt, davinci_monet
  print("imports ok")
  PY
  ```
- [ ] Run CLI smoke:
  ```bash
  mamba run -n davinci davinci-monet --help
  ```
- [ ] Run a minimal PyCWT smoke test using `cwt`, `icwt`, and `significance`.

## Task 4: Remediate pandas 3 / xarray 2026 API Changes

- [ ] Search for `Dataset.dims` size lookups and replace with `Dataset.sizes` where size mapping semantics are required.
- [ ] Audit pandas usage in:
  - [ ] `davinci_monet/stats`
  - [ ] `davinci_monet/pairing`
  - [ ] `davinci_monet/pipeline`
  - [ ] dataset readers that create or consume DataFrames.
- [ ] Run focused tests:
  ```bash
  mamba run -n davinci pytest davinci_monet/tests/test_dataset_readers.py
  mamba run -n davinci pytest davinci_monet/tests/unit/pairing
  mamba run -n davinci pytest davinci_monet/tests/unit/config
  ```

## Task 5: Remediate Matplotlib 3.11 / Plotting Changes

- [ ] Run plotting tests first:
  ```bash
  mamba run -n davinci pytest davinci_monet/tests/unit/plots
  ```
- [ ] Fix API removals/deprecations in renderers, focusing on:
  - [ ] axes/layout APIs
  - [ ] colorbar/legend behavior
  - [ ] colormap access
  - [ ] Cartopy integration edge cases
- [ ] Keep visual changes minimal unless tests reveal real behavior changes.

## Task 6: Remediate numba / Grid Binning Compatibility

- [ ] Run grid-binning and pairing tests that exercise JIT paths.
- [ ] Confirm `numba.jit(nopython=True)` still compiles on Python 3.13 / NumPy 2.4.
- [ ] If compile failures occur, make minimal type/signature changes in `davinci_monet/pairing/grid_binning.py`.

## Task 7: Full Validation

- [ ] Run:
  ```bash
  mamba run -n davinci pytest
  mamba run -n davinci mypy davinci_monet
  mamba run -n davinci black --check davinci_monet
  mamba run -n davinci isort --check davinci_monet
  ```
- [ ] Record actual failures; do not assume existing gate status.
- [ ] If failures are unrelated pre-existing style/type issues, separate them from dependency-upgrade regressions in the final report.

## Open Questions

- Should DAVINCI vendor/fork PyCWT metadata so `pip check` can be fully clean?
- Should wavelet support become an optional extra until PyCWT publishes NumPy 2-compatible metadata?
- Should CI include a Python 3.13 / NumPy 2.4 job before changing the default development environment?

## Acceptance Criteria

- `environment.yml` creates a Python 3.13 / NumPy 2.4 conda-first environment.
- No pip wheel overlays of the scientific stack.
- Core imports and `davinci-monet --help` pass.
- `pip check` is clean except, at most, the known PyCWT metadata cap.
- DAVINCI tests pass or remaining failures are documented as pre-existing/non-upgrade blockers.
