# DAVINCI

**Data Analysis and Visual Intelligence for Climate/Chemistry**

A modern, type-safe Python toolkit for evaluating atmospheric chemistry and air quality models against observations.

---

**Session Reminder**: Agentic AI coding is cognitively intense - you're managing a fast-moving collaborator while maintaining your own mental model. Remember to:
- Take breaks between major tasks (stretch, move around)
- Step away when stuck - insights often come when you're not at the keyboard
- You control the pace - slow down when needed

---

## ⚠️ Output Display Constraint

**The user cannot scroll up in their terminal** — only the most recent output is visible. Keep responses short and chunked:
- Be very brief; sentence fragments are fine. At most ~three points per message.
- Do NOT dump long multi-section answers, large menus, or many options at once; earlier content is lost.
- When presenting a design, plan, or list, deliver it incrementally and wait for "ok"/confirmation before continuing.

---

## Cross-Model Handoff Convention

This repo uses **cross-model code reviews and hand-offs** (e.g., Claude Opus writes implementation, Codex reviews, then back). Handoff files follow a consistent structure so any agent can pick up context quickly.

### Handoff File Format

Use `REVIEW_<MODEL>.md` or `HANDOFF_<TOPIC>.md` in the repo root. Structure:

```markdown
## Context
Branch, task description, which files are involved

## Changes Made
What was done, with file paths and line links

## Decisions & Rationale
Why choices were made — prevents the next model from undoing or redoing work

## Open Questions / Concerns
Things the next model should investigate or address

## Suggested Next Steps
Specific actionable items
```

### Rules

- **One file per task/feature** — scoped context, not a running log
- **Always include Decisions & Rationale** — this is the highest-value section
- **Give file paths and line numbers** — so the next agent can verify without searching
- **Check for handoff files at session start** — look for `REVIEW_*.md` or `HANDOFF_*.md` in repo root
- **Git diff supplements the handoff** — the file gives intent, `git diff` gives the changes
- **Do NOT track handoff files in git** — these are ephemeral working artifacts, not permanent records. Delete them once the handoff is complete.

---

## Git Workflow

- **NEVER auto commit or push**: Do NOT commit or push changes until the user explicitly confirms
- **Hold off on merge to main**: Do NOT merge to `main` until the user has verified the runs and explicitly asks for a merge
- **After merge, return to develop**: Always switch back to `develop` branch after merging to `main`

---

## Execution Environment & Output Conventions

These are standing directives for any agent or contributor running this project:

- **Run tests/regression in the `davinci` conda env.** The full suite depends on cartopy, monet, monetio, netCDF4, and other geo packages that are only present in this environment. Do not run the suite in a generic/sandbox Python that lacks these deps.
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate davinci
  HDF5_USE_FILE_LOCKING=FALSE python -m pytest
  ```

- **Send generated plots to the iCloud Claude folder.** All generated plots should land in:
  ```
  ~/Library/Mobile Documents/com~apple~CloudDocs/Claude
  ```
  Either point a run's output directory there, or copy the plot files there after generation. (Example plots are produced by `examples/run_all_examples.py` into `examples/output/plots`; copy the resulting `*.png`/`*.pdf` to the iCloud folder.)

---

## ⚠️ CRITICAL: CESM Vertical Coordinate Convention

**This issue has been rediscovered 4+ times. READ THIS FIRST when working with CESM/CAM-chem data.**

CESM uses hybrid sigma-pressure coordinates where **pressure increases with level index**:
- `lev=0` → **Top of Atmosphere** (stratosphere, ~3 hPa)
- `lev=-1` (last index) → **Surface** (highest pressure, ~1000 hPa)

**Common symptom**: Model O3 values of 5000-10000 ppb (stratospheric) instead of 30-80 ppb (surface).

**The fix** is single-sourced in `_extract_surface()` in `pairing/strategies/base.py` (the live pairing path). Spatial renderers that slice a vertical level use the matching `surface_level_index()` helper in `plots/renderers/spatial/base.py` so map overlays default to the surface, not the top of atmosphere:
```python
# Auto-detect if pressure increases with index
if vert_vals[-1] > vert_vals[0]:
    surface_idx = -1  # CESM convention: last level is surface
else:
    surface_idx = 0   # Other conventions: first level is surface
```

**If you see impossibly high trace gas values, check vertical level extraction FIRST.**

---

## Pre-Implementation Audit (REQUIRED)

**Before building any new component**, audit existing code for patterns it must conform to. This has caused repeated rework — new features were built in isolation, ignoring existing conventions, styles, renderers, CLI features, and naming patterns.

**Audit checklist** (search the codebase for each before writing code):

1. **Existing renderers/implementations** — Does a similar component already exist? (e.g., 3D flight track plotter already existed when 2D was introduced; should have extended, not reinvented)
2. **Style and theming** — Check `plots/style.py` for color conventions, fonts, palettes. New plotters must use the correct colors for their context (see Plot Styling section)
3. **CLI integration** — Check `cli/app.py` and `pipeline/runner.py` for flags like `--show-plots`. New pipeline stages must work with existing CLI features
4. **Config naming conventions** — Check `analyses/*/configs/` for naming patterns (e.g., `*-gemini.yaml` for machine-specific configs with absolute paths)
5. **Pipeline stage contracts** — Check what data keys existing stages return (e.g., `plots_generated`) and ensure new stages follow the same contracts
6. **Data conventions** — Check dataset readers for coordinate aliases, unit conventions, and standardization patterns already in place

**The audit is not optional.** Every missed convention costs a debugging + fix cycle that's slower than reading the code upfront.

---

## Testing Rules

1. **Integration tests must run through the pipeline.** Tests labeled as "integration" must exercise `PipelineRunner.run_from_config()` — the same code path a user takes with `davinci-monet run config.yaml`. Tests that call renderer APIs directly are **unit tests**, not integration tests. Do not label a test as integration if it bypasses the pipeline.

2. **Present test design before implementation.** Before writing tests, describe which code paths each test exercises and get approval. The list of assertions is not the design — the design is which entry points are called and what data flows through them.

3. **No shortcuts for green checkmarks.** If the full pipeline path has unknowns, investigate them rather than falling back to a simpler path that skips the code under test. A test that passes by avoiding the hard part is worse than no test — it creates false confidence.

---



```bash
# Activate conda environment
conda activate davinci

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy davinci_monet

# Format code
black davinci_monet && isort davinci_monet
```

## Conda Environment

**Name**: `davinci`

**Create from environment.yml**:
```bash
conda env create -f environment.yml
conda activate davinci
```

**Key packages**:
- monet, monetio - atmospheric data I/O
- xarray, numpy, pandas - data structures
- matplotlib, cartopy - plotting
- netCDF4 - file I/O

**Additional I/O support**:
- pyhdf - HDF4/HDF-EOS for MODIS satellite data
- tqdm - Progress bars for pipeline execution

**Development tools**:
- pydantic - configuration validation
- mypy - static type checking
- pytest, pytest-cov - testing
- black, isort - formatting

## Related Repositories

- **Wiki**: https://github.com/NCAR/DAVINCI/wiki

## Project Goals

- **Maintainability**: Small, focused modules (<500 lines each)
- **Type Safety**: Full type hints, mypy strict mode
- **Performance**: Parallel processing, lazy loading
- **Extensibility**: Plugin architecture for data sources and plotters

## Key Design Principles

1. **Uniform Pairing Logic**: Strategy based on data geometry (point, track, profile, swath, grid) not data source

2. **xarray-Only Data Model**: All data as `xr.Dataset` throughout pairing/analysis. Pandas only for I/O adapters and stats output tables.

3. **Synthetic Data for Testing**: Generate test data programmatically - no external dataset dependencies

## Architecture Overview

```
davinci_monet/
├── core/           # Protocols, registry, base classes, exceptions
├── config/         # Pydantic schemas, YAML parsing
├── analysis/       # Derived analysis products (EOF, wavelet, gridded analysis)
├── assets/         # Logos and bundled visual assets
├── datasets/       # Model and observation readers
├── geography/      # Geographic helpers and map assets
├── inspection/     # Output inspection presets and preview generation
├── pairing/        # Unified pairing engine + strategies
│   └── strategies/ # point, track, profile, swath, grid
├── plots/          # Modular plotting system
│   └── renderers/  # Individual plot types
├── stats/          # Statistics calculation
├── pipeline/       # Execution orchestration
├── ai/             # AI analysis summary (optional; Anthropic/OpenRouter)
├── io/             # File readers/writers
│   └── download/   # External data staging helpers
├── cli/            # Command-line interface
├── logging/        # Structured logging
├── util/           # Shared utilities
└── tests/
    └── synthetic/  # Test data generators
```

## Implementation Status

**Gate reality** — do not assume the full gate is green. `FABLE_REVIEW.md` recorded pytest passing on 2026-07-02 (`1,657 passed`, `6 skipped`) but mypy, black, and isort failures on `develop`. A CI workflow is defined (`.github/workflows/ci.yml`: pytest + mypy + black/isort on a 3.11/3.12 matrix), but **GitHub Actions is currently disabled for the repository**, so rerun the local gates in the `davinci` conda env and report actual output before relying on gate status.

## Running Analyses

**ALL analysis scripts MUST use DAVINCI pipelines.** Do not write custom pairing/plotting scripts.

### Pipeline Stages

The standard pipeline executes these stages in order (optional stages may skip):
1. `load_sources` - Load source data, apply unit conversions
2. `analyses` - Generate derived analysis products
3. `plot_suites` - Expand configured plot suites into plot specs
4. `pairing` - Pair sources using geometry-specific strategies
5. `statistics` - Calculate evaluation metrics (N, MB, RMSE, R, NMB, NME, IOA)
6. `save_results` - Write statistics to CSV
7. `plotting` - Generate configured plots
8. `summary` - Generate optional AI summary output
9. `inspection` - Inspect generated products when configured
10. `manifest` - Write the run manifest

### Running a Pipeline

```python
from davinci_monet.pipeline.runner import run_analysis

result = run_analysis("path/to/config.yaml")
if result.success:
    print(f"Completed in {result.total_duration_seconds:.1f}s")
```

Or via CLI:
```bash
davinci-monet run path/to/config.yaml
```

### YAML Configuration Pattern

```yaml
analysis:
  start_time: "2024-02-01"
  end_time: "2024-02-03"
  output_dir: ${MY_ANALYSIS}/output  # Supports env var expansion
  log_dir: ${MY_ANALYSIS}/logs       # Pipeline logs with timestamps

sources:
  cam:
    type: cesm_fv  # or cmaq, wrfchem, ufs, generic
    files: ${MY_DATA}/dataset/*.nc     # Env vars in paths
    radius_of_influence: 15000
    variables:
      PM25:
        unit_scale: 1.2e9  # kg/kg to µg/m³
      O3:
        unit_scale: 1.0e9  # mol/mol to ppb

  airnow:
    type: pt_sfc
    filename: ${MY_ANALYSIS}/data/datasets.nc
    variables:
      pm25:
        valid_min: 0
      o3:
        valid_min: 0
        valid_max: 500

pairs:
  cam_vs_airnow_o3:
    x: {source: airnow, variable: o3}   # horizontal axis; the "− x" in diffs
    y: {source: cam,    variable: O3}   # vertical axis;   the "y −" in diffs

  cam_vs_airnow_pm25:
    x: {source: airnow, variable: pm25}
    y: {source: cam,    variable: PM25}

plots:
  pm25_scatter:
    type: scatter
    pairs: [cam_vs_airnow_pm25]
    title: "PM2.5 CAM vs AirNow"

stats:
  metrics: [N, MB, RMSE, R, NMB, NME, IOA]
  # Per-axis mean/median/std metric keys are x/y-named (output-facing):
  #   MX/MY (mean), MdnX/MdnY (median), STDX/STDY (std). The CSV columns are
  #   Mean_X/Mean_Y. (Formerly MG/MD, MdnG/MdnD, STDG/STDD, Mean_Geometry/Mean_Dataset.)

summary:
  enabled: true
  model: claude-haiku-4-5  # cheapest vision model
  max_images: 8
```

**Note**: The old pair shape (`sources: [a, b]` + `geometry:` + `variables:`) is rejected by validation. Use the nested `x:`/`y:` shape above.

### Config Naming Convention

Tracked configs use `{campaign}-{variant}.example.yaml` with environment variables for portability. Examples:
- `asia-aq-airnow.example.yaml` — ASIA-AQ AirNow surface evaluation
- `dc3-geometry-dc8.example.yaml` — DC3 geometry-only DC-8 analysis
- `modis-aod-cam6.example.yaml` — MODIS AOD vs CAM6

Machine-specific configs (`*-gemini.yaml`, `*-derecho.yaml`) are gitignored — keep them local, don't commit.

### Environment Variable Expansion

YAML config paths support `${VAR}` syntax for environment variables, but **machine-specific configs should use absolute paths instead**. Env var configs are for portable/template use only:
```bash
export MY_DATA=~/Data/campaign
export MY_ANALYSIS=/path/to/analysis
```

### Variable Naming Convention

Paired datasets produced by the pipeline use **source-label prefix** format —
`<source_label>_<var>`, where `<source_label>` is the source's key in the
`sources:`/`pairs:` config:
- `cam_pm25` - the `cam` source's values (y axis in the pair)
- `airnow_pm25` - the `airnow` source's values (x axis in the pair)

Each paired variable carries `axis` (`"x"`/`"y"`) and `source_label` attrs, so
consumers select series by axis/source rather than by a name prefix. This is the
going-forward naming after the renderer rewire clean break (R-5).

Pipeline and `PairingEngine.pair_sources()` output is source-label named. Strategy
implementations emit `x_`/`y_` prefixed variable names, which the engine then
relabels to the public source-label convention; the `x_`/`y_` prefixes are also
recognized as a fallback by the axis/canonical helpers in `core/base.py`.

Either way it is **prefix** format, NOT suffix (`pm25_cam`, `pm25_source`).

### Plot Arity & Single-Source Spatial Plots

Plots differ by how many sources they take:
- **Time series** overlays **N** source series on a shared value axis.
- **Scatter** and **spatial bias** are intrinsically **pairwise** (x vs y; bias = `y − x`).
- **Single-source** plots take **one** source via a `source:` + `variable:` plot spec
  (no `data:`/`pairs:`). The pipeline's single-source path renders these.

`type: spatial` is the **general single-source spatial map** (`plots/renderers/spatial/field.py`,
`SpatialPlotter`). It takes one source and chooses the mark from the source's **shape**
(the `geometry` attr `point`/`track`/`profile`/`swath`/`grid`, with coordinate-based fallback):
point/track/profile → **scatter**, grid/swath → **pcolormesh**. 3-D fields slice the **surface**
via `surface_level_index` (CESM convention); point/grid/swath time-average by default (track/profile
keep time as the sampling path). The shared scatter/pcolormesh primitive is `draw_spatial_field()`
in `plots/renderers/spatial/base.py` (used by `spatial`; `spatial_distribution` has been removed — use `type: spatial` instead).

```yaml
plots:
  cam_o3_map:
    type: spatial
    source: cam        # one source — NOT a pair
    variable: O3
```

When adding/altering spatial render logic, **verify the render mark programmatically**
(QuadMesh for grid/swath, PathCollection for point/track) — not by eye. See
`tests/test_spatial_single_source.py`.

### Derived Analyses (`analyses:` block)

The optional top-level `analyses:` block runs after `load_sources` and before `pairing`. Each entry is a named derived-analysis spec; its output is registered as a **pseudo-source** referenceable by `source:` + `variable:` in `plots:`. Derived sources are **not pairable** — they cannot appear in `pairs:`.

**EOF analysis** (`type: eof`) decomposes a gridded field into empirical orthogonal functions via in-repo truncated SVD (no external EOF library). 3-D fields are decomposed as a coupled state vector with area × layer-mass weighting.

| Field | Default | Description |
|---|---|---|
| `source` | required | Source key from `sources:` |
| `variable` | required | Variable name in that source |
| `n_modes` | `10` | Number of EOF modes to retain |
| `standardize` | `false` | `true` → correlation EOF; `false` → covariance EOF |
| `remove_seasonal_cycle` | `false` | Subtract monthly climatology before decomposition |
| `rotation` | `none` | `none` or `varimax` |
| `level` | — | Restrict a 3-D field to one level index before decomposing |

**Output variables** (all `mode`-indexed, 1-based):

- `eofs(mode, [lev,] lat, lon)` — spatial patterns (unit-variance)
- `pc(time, mode)` — principal component time series (unit variance)
- `explained_variance(mode)` — fraction of total variance (%)
- `explained_variance_error(mode)` — North's rule sampling error (unrotated only)

**Plot types for EOF output**:

- `type: eof_pattern` — signed spatial maps, one panel per mode; use `display_level:` to slice a 3-D mode
- `type: eof_scree` — bar chart of `explained_variance` (%)
- `type: timeseries` with `mode: N` — PC time series for mode N

```yaml
analyses:
  cam_O3_eof: { type: eof, source: cam, variable: O3, n_modes: 6 }

plots:
  eof_maps: { type: eof_pattern, source: cam_O3_eof, variable: eofs }
  eof_var:  { type: eof_scree,   source: cam_O3_eof, variable: explained_variance }
  pc1:      { type: timeseries,  source: cam_O3_eof, variable: pc, mode: 1 }
```

**Wavelet analysis** (`type: wavelet`) computes a Morlet continuous wavelet transform (CWT) via `pycwt` (Torrence & Compo 1998). The input must be (or reduce to) a 1-D time series; the AR(1) red-noise background is estimated automatically for significance testing.

| Field | Default | Description |
|---|---|---|
| `source` | required | Source key from `sources:` or `analyses:` (e.g. an EOF key) |
| `variable` | required | Variable name in that source |
| `mode` | — | Integer (1-based): select one PC when `variable` has a `mode` dim |
| `reduce` | `area_mean` | `area_mean` — spatial mean over all grid points; `{point: [lat, lon]}` — nearest-neighbor; `null`/omit — input is already 1-D |
| `omega0` | `6.0` | Morlet parameter ω₀ (higher → better frequency resolution) |
| `significance_level` | `0.95` | Confidence level for the AR(1) red-noise significance test |
| `dj` | `0.25` | Scale resolution (smaller → more scales) |
| `s0` | auto | Smallest scale; defaults to `2 × dt` |
| `j` | auto | Number of scales; defaults to spanning the full time series |

**Output** (SPECTRUM geometry — `power(time, period)` is the primary variable):

- `power(time, period)` — local wavelet power spectrum
- `power_significance(time, period)` — power normalised by the local significance threshold; values > 1 are significant
- `coi(time)` — cone of influence period (edge-effects boundary); units match `period`
- `global_power(period)` — time-averaged power spectrum
- `global_significance(period)` — global significance threshold

The `period` coordinate carries units derived from the input time step (e.g. `"days"`).

**Plot type for wavelet output**:

- `type: wavelet_scalogram` — Torrence & Compo–style figure: time × log-period filled power, COI overlay, significance contour, and a global-spectrum side panel.

**Headline use-case — wavelet of an EOF PC**: chain an `eof` entry with a `wavelet` entry whose `source:` points at the EOF key and `mode:` selects the PC:

```yaml
analyses:
  cam_O3_eof: { type: eof, source: cam, variable: O3, n_modes: 3 }
  pc1_wav:
    type: wavelet
    source: cam_O3_eof   # pseudo-source produced by the eof step
    variable: pc
    mode: 1              # wavelet of PC-1

plots:
  eof_maps:    { type: eof_pattern,        source: cam_O3_eof, variable: eofs }
  eof_var:     { type: eof_scree,          source: cam_O3_eof, variable: explained_variance }
  pc1_scal:    { type: wavelet_scalogram,  source: pc1_wav,    variable: power }
```

## Key Design Patterns

1. **Plugin Registry**: Components register via decorators
   ```python
   @source_registry.register('cmaq')
   class CMAQReader(BaseModel): ...
   ```

2. **Protocol-based Interfaces**: Python Protocols define contracts

3. **Pydantic Configuration**: Type-safe YAML parsing with validation

4. **Pipeline Architecture**: Composable stages replace monolithic methods

## Data Model (xarray-only)

```
Dataset:   xr.Dataset with dims (time, level, lat, lon)
Point:   xr.Dataset with dims (time, site) + lat/lon coords
Track:   xr.Dataset with dims (time,) + lat/lon/alt coords
Profile: xr.Dataset with dims (time, level) + lat/lon coords
Swath:   xr.Dataset with dims (time, scanline, pixel)
Grid:    xr.Dataset with dims (time, lat, lon)
Paired:  xr.Dataset with aligned x/y source variables
```

## External Dependencies

- Continues using monet/monetio libraries for data I/O
- YAML control files must use the unified `sources:` schema with nested `x:`/`y:` pairs.
  Old pair shapes (`sources: [a, b]` + `geometry:` + `variables:`) are rejected by validation.

## Unified Data-Source Config (`sources:`)

The going-forward config format is a single `sources:` block plus binary `pairs:`
with nested `x:`/`y:` keys. All data sources (model, satellite, surface network)
are defined in `sources:`; the pair specifies which variable from each source goes
on each axis. The old shape (`sources: [a, b]` + `geometry:` + `variables:`) is
rejected by validation.

```yaml
sources:
  cam:
    type: cesm_fv
    files: ${DATA}/cam/*.nc
    variables: { O3: { unit_scale: 1.0e9 } }
  airnow:
    type: pt_sfc
    filename: ${DATA}/airnow.nc
    variables: { o3: { valid_min: 0, valid_max: 500 } }

pairs:
  cam_vs_airnow_o3:
    x: {source: airnow, variable: o3}   # horizontal axis; the "− x" in diffs
    y: {source: cam,    variable: O3}   # vertical axis;   the "y −" in diffs
```

**Pairing direction** is decided by spatial geometry (shape) precedence: irregular
geometries (point/track/profile/swath) outrank GRID, so a gridded source is
sampled onto them. x/y assignment is plot-axis labeling only and does not affect
which source drives the spatial sampling.

**Intermediate gridding** (`method: grid`) bins both sources onto a common uniform grid instead of
sampling one onto the other — useful when both sources have irregular/incompatible geometries:

```yaml
pairs:
  a_vs_b:
    x: {source: aeronet, variable: aod_500nm}
    y: {source: cam,     variable: AODVISdn}
    method: grid          # intermediate gridding: bin BOTH sources onto a common grid
    grid:
      horizontal_res: 0.5
      time_resolution: 1D
      min_sample_count: 1
      vertical: { res: 500, units: m }   # presence -> 3-D (time, lon, lat, alt) grid
```

`method: grid` bins both sources onto a uniform `(time, lon, lat)` grid and pairs cell-to-cell (symmetric); default `method: auto` keeps geometry-based pairing. Add a `grid.vertical:` block for a 3-D altitude grid; each source must supply a usable vertical (geometric altitude, geopotential height, or pressure) or the strategy errors.

## Working Example: ASIA-AQ Analysis

Geometry implementation in `analyses/asia-aq/`:

```
analyses/asia-aq/
├── configs/
│   └── asia-aq-airnow.example.yaml # Portable template config
├── scripts/
│   ├── download_airnow.py          # Data download
│   └── run_evaluation.py           # Pipeline execution
├── data/                           # Model and observation data (gitignored)
├── output/                         # Plots and statistics (gitignored)
└── logs/                           # Pipeline logs (gitignored)
```

**Run the analysis**:
```bash
cd analyses/asia-aq
export ASIA_AQ_DATA=~/Data/ASIA-AQ
export ASIA_AQ_ANALYSIS=$(pwd)
davinci-monet run configs/asia-aq-airnow.example.yaml
```

## Plot Styling (NCAR Branding)

The plotting system uses NSF NCAR brand colors and fonts. Enable via YAML config or Python:

### YAML Configuration (Recommended)

```yaml
analysis:
  start_time: "2024-02-01"
  end_time: "2024-02-03"
  output_dir: ${MY_ANALYSIS}/output
  style:
    theme: ncar                # Apply NCAR branding
    context: default           # or: presentation, publication
    use_seaborn: true          # Apply seaborn whitegrid theme
```

### Python API

```python
from davinci_monet.plots import PlotConfig, TimeSeriesPlotter, apply_ncar_style, build_series

# Apply NCAR styling globally (call once at start of script)
apply_ncar_style()

# Direct plotting uses plotter classes and render(build_series(...))
plotter = TimeSeriesPlotter(PlotConfig(title="O3 Time Series"))
fig = plotter.render(build_series(paired_data, "airnow_o3", "cam_o3"))
```

**Key colors** (`davinci_monet.plots.style`):
- `NCAR_PRIMARY`: NCAR Blue (#0A5DDA) — brand color, used for single-source plots
- `x_color` / `DATASET_A_COLOR`: Gray (#58595B) — x-axis source in paired plots (conventional reference)
- `y_color` / `DATASET_B_COLOR`: NCAR Blue (#0A5DDA) — y-axis source in paired plots
- `NCAR_PALETTE`: 8-color palette for multiple sources / per-flight coloring

**Context presets**:
- `default`: Standard sizes for general use
- `presentation`: Larger fonts for slides
- `publication`: Smaller fonts for journal figures

**Font**: Poppins (with Helvetica/Arial fallbacks)

### Plot Label & Title Conventions (publication quality)

All plot text is composed by one module — **`davinci_monet/plots/labeling.py`**
(pure functions: `format_units`, `source_display_name`, `quantity_label`,
`axis_label`, `legend_label`, `bias_label`, `title_text`, `subtitle_text`).
Renderers MUST call these, never build label strings ad-hoc. `labels.py` keeps
the lookup tables. Design spec: `docs/superpowers/specs/2026-06-15-plot-labeling-system-design.md`.

Rules (enforced by `tests/unit/plots/test_labeling.py` + `test_labels_rendered.py`):
- **Titles are terse** — the quantity only (chem-formatted), plus an operation
  word only when it changes meaning (`… Bias`, `… Vertical Profile`). NEVER put
  dates, stats, `Mean ± Std`, or `vs <source>` in a title; those go in the date
  **subtitle** and the in-axes **stats box**.
- **Never expose `x`/`y` to viewers.** Bias labels name the sources:
  `Bias, <Ysrc> − <Xsrc> (units)` (U+2212 minus). Pass the quantity to
  `bias_label` so it factors out (→ `Bias, CESM − Pandora`).
- **Source names are auto-cleaned** from the config key (acronym map: CESM,
  MERRA-2, AirNow, AERONET, CERES, MODIS, WRF-Chem…); never the raw key, never
  ALL-CAPS. Source appears on an axis only where no legend carries it (scatter);
  timeseries/single-source carry it in the legend/title.
- **De-dup**: when a source key embeds the quantity (`cesm_no2_column`), only its
  distinctive token (`CESM`) is kept next to the quantity — no repetition.
- **Units are negative-exponent SI** via `format_units`: `mol m⁻²`, `W m⁻²`,
  `µg m⁻³`; ratios `mol mol⁻¹`; dimensionless (`1`/`none`) omitted; `ppb`/`%`/`K`
  left as-is. **Build axis labels from `quantity + format_units(units)` — do NOT
  bake units into config `ylabel_plot` strings.** A variable's `units` (config or
  attr) drives this; aircraft/ICARTT variables that lack a `units` attr show no
  unit until one is supplied (config `units:` field, applied at `load.py`).
- **iCloud delivery is PDF-only**; the optional synthetic gallery
  (`analyses/_gallery/make_gallery.py`) renders one figure per plot type for
  multimodal label inspection.
- **Rasterize dense data layers in vector (PDF) output.** `pcolormesh`/`imshow`/
  filled (`fill_between`, `contourf`) and large scatter layers must set
  `rasterized=True` (or `mappable.set_rasterized(True)`) so PDFs stay small and
  render cleanly; keep axes, text, and thin contour LINES vector (matplotlib
  ignores — and warns on — rasterizing a `contour` line set, which trips the
  `error::UserWarning` gate, so do NOT rasterize contour lines).
- **Date/time x-axes**: rotate tick labels with `ax.tick_params(axis="x", rotation=45)`
  (the convention in `timeseries`/`curtain`) to avoid overlap.

## Common Gotchas

1. **Unit conversions**: Model variables often need `unit_scale` in config:
   - CESM mixing ratios (mol/mol) → ppb: `unit_scale: 1.0e9`
   - CESM PM mass (kg/kg) → µg/m³: `unit_scale: 1.2e9`

2. **AERONET wavelengths**: Use `aod_500nm` or `aod_440nm` for Asia (not `aod_551nm`)

3. **CESM vertical levels**: Surface is `lev=-1` (last index), NOT `lev=0`. See CRITICAL warning above. This has caused bugs 4+ times.

4. **Source coordinates**: Must have `latitude`, `longitude` as coordinates or variables

5. **Time alignment**: Pipeline uses nearest-neighbor interpolation for source-to-source times

6. **High-frequency observations**: Use `resample` to average sub-hourly data to match model resolution:
   ```yaml
   sources:
     pandora:
       type: pt_sfc
       filename: /data/pandora/*.nc
       resample: "h"           # Average to hourly
       min_sample_count: 3     # Require ≥3 samples per hour (reject sparse hours)
       track_sample_count: true  # Add sample_count variable for diagnostics
   ```

7. **Scatter plot density**: For busy scatter plots with many points, enable density coloring:
   ```yaml
   plots:
     my_scatter:
       type: scatter
       show_density: true      # Color points by local density
   ```

8. **HDF5 thread safety segfaults**: If you see HDF5 errors mentioning "thread 1/thread 2" followed by a segmentation fault, this is an HDF5 thread safety issue. The segfault happens at the C level before Python can catch it, so retry logic won't help. Fix by disabling HDF5 file locking:
   ```bash
   HDF5_USE_FILE_LOCKING=FALSE davinci-monet run config.yaml
   ```
   If it persists, also limit Dask workers:
   ```bash
   DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE davinci-monet run config.yaml
   ```

9. **AI summary stage**: The `summary:` block enables an opt-in final stage that
   sends stats + plot images to the Claude API (`pip install -e ".[ai]"`,
   `ANTHROPIC_API_KEY`). It is always non-fatal — missing key/network just skips
   it. Default model `claude-haiku-4-5`. Vision images are downscaled to ≤1568px.
   The provider can be `anthropic` (default, `ANTHROPIC_API_KEY`) or `openrouter`
   (`provider: openrouter`, key via `api_key_file:` or `OPENROUTER_API_KEY`,
   default model `anthropic/claude-haiku-4.5`).
