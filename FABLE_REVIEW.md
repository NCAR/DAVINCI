# DAVINCI — Complete Design & Code Review

**Reviewer:** Claude Fable 5 · **Date:** 2026-07-02 · **Branch:** `develop` @ `9e5ebdb` · **Scope:** all of `davinci_monet/` (339 files, ~67k lines incl. tests), packaging, examples, docs

---

## The prompt (verbatim)

> hi Claude, do a deep dive of this code base, and a complete design and code review.  write your review to FABLE_REVIEW.md.  we want to reach a near perfect architecture (granted that is an ideal state that will never be fully realized).  also search for bugs and recommend fixes.  include this prompt verbatim at the top of your review.

---

## 1. Executive summary

DAVINCI is in fundamentally good shape. The bones are right: a registry-plus-protocol plugin core, geometry-driven pairing strategies, a single-sourced labeling module, pydantic-validated config, a stage pipeline that is genuinely the one path users and tests share, and a test suite (1,657 passing) that mostly earns its green — integration tests really do go through `PipelineRunner.run_from_config()`, and the CESM vertical convention is exercised in both orderings. That is a far better starting point than most scientific-Python codebases.

Three problems stand between the current state and the "near-perfect architecture" ambition:

1. **The quality gates are red on `develop` and nothing enforces them.** mypy reports **42 errors in 15 files**, black and isort each flag 3 files — all introduced by the recently merged `codex/ceres-davinci-analysis-program` branch. CLAUDE.md still claims all gates pass. With GitHub Actions disabled, local discipline is the only gate, and it was skipped for a whole merge.

2. **There is no explicit coordinate-convention contract at the load boundary.** Readers emit whatever latitude ordering and longitude convention the source uses; pairing and plotting then each *guess*, differently, and several guesses are wrong. This one architectural gap is the root cause of the two highest-severity bugs found (descending-latitude grids silently mis-pair or produce all-NaN output) and at least three more medium ones (longitude-convention mismatches, the square-grid transpose, mixed-convention grid extents).

3. **A repo-wide find/replace ("model" → "dataset") corrupted documentation and leaked into code.** CLAUDE.md now contains a config example that fails validation (`summary.dataset:` — the real key is `model:`), nonsense phrases ("evaluating … datasets against datasets"), and a stale architecture description; ~10 code docstrings say "dataset output"; and one *functional* casualty: Linux CPU detection reads `"dataset name"` from `/proc/cpuinfo`, whose field is `model name`.

**Bug count:** 4 high, 19 medium, 23 low (plus test-suite defects in §5.4). The most user-visible: one failed plot aborts the pipeline *before* the stats CSV is written (while the UI prints "pipeline still succeeded"); the headline wavelet-of-a-monthly-EOF-PC path throws a `TypeError` that is silently swallowed, so the scalogram simply never appears; and the tracked example config `cmaq_airnow.yaml` is malformed YAML that fails for every user. Full list with fixes in §5.

Everything here is a recommendation; no fixes have been applied.

---

## 2. Method

- Full local gate run in the `davinci` conda env (pytest, mypy, black, isort) — results in §3.
- Eight parallel subsystem deep-reads (core/config, datasets, pairing, plots, pipeline/CLI/inspection, analysis/stats/ai, io/util/logging/packaging, test suite), each combining design assessment with bug hunting. Candidate bugs were verified by tracing full code paths and, where possible, **runtime repro in the davinci env** (marked CONFIRMED); untraced-but-suspicious items are marked PLAUSIBLE.
- An independent severity-gated design-review workflow (70 agents: dimension reviewers → adversarial verification, majority-vote survival) ran in parallel; its five surviving medium-severity findings are folded in below (they independently re-derived A1/A3/A4/A6/A10, with additional evidence cited in those sections).
- My own passes: dependency-graph mapping, config parser/schema, surface-extraction logic, docs-vs-code consistency.

---

## 3. Ground truth: gate status on `develop`

| Gate | Documented | Actual (2026-07-02) |
|---|---|---|
| pytest | "1,262 tests passing" | ✅ **1,657 passed**, 6 skipped, 48.7 s |
| mypy | passing | ❌ **42 errors in 15 files** |
| black | passing | ❌ 3 files (`cli/commands/inspect.py`, `inspection/core.py`, `tests/unit/inspection/test_inspection_core.py`) |
| isort | passing | ❌ 3 files (`datasets/__init__.py`, 2 test files) |

The mypy errors cluster in the merged gridded-analysis/inspection work: `analysis/formula.py` (3), `plots/renderers/spatial/field.py` (3), `pipeline/stages/load.py` (2, incl. untyped `cftime` imports), `pipeline/runner.py:216-218` (`Stage` vs `BaseStage` list invariance), `cli/commands/inspect.py:22` (`preview_format` Literal mismatch), and ~30 in new test files constructing schema objects from raw dicts. None look hard; a half-day clears them. Add `cftime` to mypy's ignored-imports config rather than sprinkling `# type: ignore`.

**Process recommendation (P0):** treat "the local gate is the CI" seriously while Actions is disabled — run the full four-gate check before any merge into `develop`, and record gate output in the merge commit or a `RUNS.md`. Better: re-enable GitHub Actions; the workflow file already exists.

---

## 4. Architecture review

### 4.1 What is genuinely strong (keep and protect)

- **Registry + protocols core.** `core/registry.py` is a clean generic registry (alias chains with cycle guard, decorator + programmatic registration). `core/protocols.py` is honest — it documents which contracts are real Protocols and which are concrete base classes, instead of faking universality. No circular imports in the foundation.
- **The pairing engine's x/y contract.** Strategies emit `x_`/`y_`; relabeling to source-label names happens in exactly one place (`engine._assemble_paired_dataset`), which hard-raises on unresolved variables instead of silently dropping. `bias = y − x` survives even when shape precedence swaps the sampling direction (`_normalize_config_axes` restores config axes by `source_label`) — verified both paths.
- **Labeling centralization is real, not aspirational.** Every renderer routes text through `plots/labeling.py`; `format_units` verified correct. This is the pattern the rest of the codebase should imitate (see 4.4).
- **`plots/contracts.py` as a lightweight seam** — no matplotlib import, single source of plot arity for config validation, registry, and dispatch. Right idea; its *content* has drifted (see A5).
- **`io/reader_utils.py`** — exemplary extraction of reader boilerplate (validation, transient-open retry, dim standardization, geometry tagging), well used by the model and newer satellite readers.
- **The download layer's testability** — all network calls behind injectable hooks; products and tests run fully offline; lazy `earthaccess` import.
- **The AI summary stage's non-fatal guarantee is real** (whole body wrapped, per-image failures caught, no key logging), matching the documented contract.
- **Test-suite integrity**: integration tests genuinely exercise `run_from_config`; synthetic data sets `geometry` attrs and realistic physics; CERES tests pin exact `N=144` to catch binning regressions; registry mutations are restored via teardown. Formula-evaluator escape attempts (`__import__`, dunders, subscripts, lambdas) are tested and blocked.
- **Stats formulas are correct.** MB, RMSE, NMB, NME, Pearson R, Willmott IOA all verified against hand-computed values; pairwise NaN deletion and Σx=0 guards behave.

### 4.2 Dependency structure

Measured import graph (package → packages it imports):

```
core        → (nothing)                     ✅ clean foundation
config      → core, plots                   ⚠ upward edge
pairing     → core                          ✅
datasets    → core, io, pairing             ⚠ pairing edge
stats       → core                          ✅
plots       → core, geography, stats        ~ acceptable
analysis    → config, core                  ✅
io          → core                          ✅
pipeline    → (everything)                  ✅ top orchestrator
cli         → config…pipeline               ✅ top
ai          → config, core, pipeline*       * TYPE_CHECKING only
```

Three edges deserve attention (A6 below): `config → plots` (schema imports `plots.contracts` at module load and `list_plotters` at validation time), `datasets → pairing` (`modis_l2.py` reaching into `pairing.grid_binning` — moot if that dead module is deleted, but `grid_binning` itself is regridding, not pairing), and the type-only `ai ↔ pipeline` cycle (the `PipelineContext` *type* is wanted below the pipeline layer — move the type to `core`).

### 4.3 Design findings

**A1 · HIGH — The stage error-handling contract contradicts itself, and users lose data.**
`plot.py:746-753`, `stats.py:144-151`, `pair.py:577-585` collect per-item errors and continue (partial-success semantics) — then return **FAILED** if any item errored. The runner defaults to `fail_fast=True` (`runner.py:163`) and `SaveResultsStage` runs *after* `PlottingStage` (`factory.py:36-37`). Net effect: **one bad plot aborts the run and the statistics CSV is never written**, while `display.py:695-699` simultaneously prints "N non-fatal error(s) occurred **(pipeline still succeeded)**" under a red "✗ Pipeline failed" footer. Comments in `runner.py:416-418` calling these errors "non-fatal" contradict the code.
*Recommendation:* decide the contract once — per-item failures ⇒ stage returns COMPLETED-with-warnings (and the run reports partial success), stage-level crashes ⇒ FAILED. Move `save_results` before `plotting` (stats exist by then; there is no dependency), and gate the display wording on actual `result.success`.

**A2 · HIGH — No coordinate-convention contract at the load boundary.**
Readers emit native coordinate conventions; only the two CERES readers normalize (sort lat ascending, wrap lon to [-180,180)). Downstream, `pairing/strategies/base.py:209` assumes ascending lat (`searchsorted`), `grid_binning.py` assumes ascending edges, the nearest-neighbor path never reconciles −180..180 obs against 0..360 models (`base.py:167`), `intermediate_grid.py:599` computes extents across mixed conventions, and `plots/.../spatial/base.py:211` guesses orientation from array shape. Each consumer guesses; four bugs result (B1, B2, B5, B8, B17).
*Recommendation:* make the post-load dataset contract explicit and enforced in **one** place — the load stage: lat ascending, lon canonicalized to one convention (pick [-180,180)), dims/coords standardized, `geometry` attr set. Then delete the per-consumer guessing. This is the single highest-leverage architectural change in this review.

**A3 · MEDIUM-HIGH — Vertical/surface logic is neither single-sourced nor CF-aware.**
CLAUDE.md says surface extraction is "single-sourced in `_extract_surface()`", but reality is: two mirrored copies (`pairing/strategies/base.py:379` and `plots/renderers/spatial/base.py:121`, synced only by docstring), one re-derivation (`track.py:425`), and one bypass (`swath.py:287` averages the vertical). The adversarial verify pass additionally confirmed the **vertical-dim candidate lists have already diverged across 6+ sites**: pairing detects `["lev","z","level","altitude","height"]`, while `spatial/field.py:48` and `spatial/overlay.py:142` use `"vertical"` instead of altitude/height, and `intermediate_grid.py:138` omits both — so a field whose vertical dim is `height` is surfaced correctly in pairing but mishandled by the spatial renderers. Worse, the shared heuristic — "values increase with index ⇒ surface at last index" — is pressure logic applied to *any* auto-detected vertical dim including `z`/`altitude`/`height`; an ascending altitude coordinate in metres gets its **top** level selected as "surface" (B9). This is the same class of bug the CESM warning exists for, in the opposite direction.
*Recommendation:* one helper in `core`/`util`, CF-aware (use `positive: up|down` attr and `units` before falling back to the ordering heuristic), consumed by pairing, plots, and track; migrate swath to it or document why swath averages.

**A4 · MEDIUM — Open plugin registry vs. closed hand-maintained enumerations.**
The registry says "plotters are plugins", but plot types live in two hand-synchronized catalogs: the decorator registry (`plots/registry.py:40-63`) and the arity/category frozensets in `plots/contracts.py:23-78`. Config validation enforces **both** — registry membership at `schema.py:500-508`, then `plot_arity` against the frozensets at `schema.py:927-935`, re-checked at `pipeline/stages/plot.py:676` — so a plotter registered through the public API but absent from the frozensets is rejected at config-validation time (verified by the adversarial pass). Out-of-tree plot types are unusable through the pipeline, and every in-tree addition needs three coordinated edits; the plugin registry is effectively decorative. Same pattern: `StatMetric` (schema.py:538-563) is a 24-member Literal **no field uses**, already drifted from the documented metric names.
*Recommendation:* arity/capability should be declared *on the plotter class* (class attribute) and `contracts.py` derived from the registry at runtime; type `stats.metrics` with the Literal or delete it. A plugin architecture where the plugin list is hand-maintained elsewhere is not a plugin architecture.

**A5 · MEDIUM — Pydantic is underused where it's strongest.**
`AnalysisSpec` (EOF/Wavelet/Gridded) is dispatched by hand in `build_analysis_spec` despite every member carrying `type: Literal[...]` — a textbook discriminated union (`Field(discriminator="type")`), which would also produce better error messages. ~13 `mode="before"` validators re-implement dict→model coercion pydantic does natively. `SourcePairConfig` is `extra="allow"` although its shape is fully known, so `methd: grid` (typo) silently degrades to `method: auto` unless the non-default `--strict` is passed (B16). `SourceConfig` legitimately needs passthrough; the pair schema does not.

**A6 · MEDIUM — Layering nits (see 4.2).** Move `PipelineContext`'s type to `core`; relocate `grid_binning` out of `pairing` (it is regridding used by readers and strategies alike — `util` or a small `regrid` package); keep `plots.contracts` matplotlib-free (it is) but stop importing the full `plots` package inside schema validation (`schema.py:264-269`). Because `plots/__init__.py:35-118` eagerly imports every renderer — pulling matplotlib *and cartopy* via `lma_density.py:11-12` — merely parsing a YAML config (and every pipeline stage via `stages/base.py:18-28`) transitively loads the full plotting stack (**~0.56 s measured** by the adversarial pass). Fix by lazy/registry-side type listing or lazy renderer imports in `plots/__init__`.

**A7 · MEDIUM — Duplication catalogue** (each a drift risk; none hard to fix):
- Model readers: cesm/cmaq/ufs/wrfchem/merra2 each re-implement the monetio-try/xarray-fallback/standardize dance → extract a `MonetioModelReader` base.
- `latitude→lat`/`longitude→lon` rename + QA-filter blocks copy-pasted across ~12 obs/satellite readers → `reader_utils` already has `alias_coord`; use it.
- `intermediate_grid.py` (852 lines) carries three near-parallel bin pipelines (2-D match, 2-D symmetric, 3-D symmetric) → one parameterized `_bin()`.
- `pipeline/stages/load.py:127 _file_list` duplicates `io/reader_utils.py:98 resolve_file_list` (which calls itself "the canonical glob resolver").
- ~120 lines of `_get_system_info` duplicated verbatim in `display.py:306-427` and `cli/app.py:41-162` — both copies share the `"dataset name"` bug (B-low).
- Longitude re-sort duplicated in `plots/.../field.py:458-475` and `bias.py:136-145`; spatial mark-selection logic differs across `field.py` (geometry attr), `bias.py` (coord sniffing), `overlay.py` (always contourf) → single-source the shape→mark decision in `spatial/base.py`.

**A8 · MEDIUM — Module-size budget vs. reality.** The project's own goal is <500-line modules. Over budget: `config/schema.py` 1003, `pipeline/display.py` 853, `pairing/strategies/intermediate_grid.py` 852, `plots/renderers/timeseries.py` 833, `stats/metrics.py` 812, `pipeline/stages/plot.py` 759, `pipeline/runner.py` 724, `pairing/strategies/track.py` 707, `core/exceptions.py` 668, `plots/renderers/_track3d.py` 609, `stats/calculator.py` 586, `stats/output.py` 510. Most are cohesive; schema.py and exceptions.py split naturally (see A9/A10). Treat the budget as a review trigger, not a hard law — but 12 files over it means the law is currently fiction.

**A9 · MEDIUM — Dead code.** `datasets/satellite/modis_l2.py` (356 lines, zero callers, unregistered, untested; its one referencer is a docstring). ~8 of 19 exception classes in `core/exceptions.py` are never raised or caught in production (`ConfigValidationError`, `ConfigParseError`, `ConfigMigrationError`, `InterpolationError`, `PlotConfigError`, `InsufficientDataError`, `StageExecutionError`, `PipelineAbortError`) — each with elaborate structured `__init__`s nothing constructs; the config layer raises base `ConfigurationError`/plain `ValueError` instead of its own purpose-built subclasses. Also: `curtain.py:176-190` (computed then unused), `cli/app.py:314` dead imports, `surf_only` kwarg documented but silently discarded (cmaq.py:69, ufs.py:68), `writers.py` pickle output with no callers. Delete or start using; either is fine, limbo is not.

**A10 · MEDIUM — Stringly-typed inter-stage contracts, and the convention has already broken.** `StageResult.data: Any` and `PipelineContext.sources/paired: dict[str, Any]` (`stages/base.py:69,146-147`) make undocumented string keys ("plots_generated", "statistics_kind", "*_errors") the real cross-stage API, duplicated across `runner.py:117-121,459-462`, `reporting.py:245-249`, `manifest.py:30-31`, and `ai/payload.py:38-51,100-108`. The drift is not hypothetical: `AnalysesStage` emits `analysis_errors` (`analyses.py:67`) that **all three error-aggregation points silently ignore**. Relatedly, the typed config layer is abandoned at the stage boundary: the three largest stages flatten `MonetConfig` back to raw dicts via `config_dict()` (`load.py:65`, `pair.py:104-140`, `plot.py:646-673`) and re-supply schema defaults as duplicated literals (`time_tolerance "1h"`/`time_method "nearest"` at `pair.py:412-413` duplicating `schema.py:358-359` — the same duplication behind L8), leaving two competing config-access conventions. *Fix:* TypedDicts/dataclasses per stage output; wire `analysis_errors` into the itemization maps; pick the typed accessors as the one convention and migrate the three big stages.

**A11 · LOW — Config parser robustness.** `load_yaml` falls back to parsing the *path string as YAML* when the file doesn't exist, so a typo'd path yields "YAML root must be a mapping, got <class 'str'>" instead of file-not-found (the `except FileNotFoundError` branch is effectively dead). `os.path.expandvars` leaves unset `${VAR}` as literal text — stages then `mkdir` a directory literally named `${MY_ANALYSIS}` — and `~` is expanded for `files`/`filename` but **not** for `output_dir`/`log_dir`. *Fix:* split path-vs-string entry points (or require `Path` for files), raise on residual `${…}` after expansion, `.expanduser()` uniformly.

**A12 · Performance notes (real but not urgent).** `_find_nearest_2d` is an O(n_obs × n_grid) pure-Python loop (docstring admits it; use cKDTree). EOF is eager full-matrix SVD and the pattern step broadcasts to `(time, space, mode)` — ~1.6 GB for a daily-year 192×288 field at 10 modes; compute patterns as `anomᵀ @ pc / nt` instead (B21). `PlotSuiteStage` re-validates the entire `MonetConfig` mid-pipeline (`plot_suites.py:70-77`) to merge a plot dict. Config validation imports the full plots package (matplotlib) even for `davinci-monet validate`.

### 4.4 The road to "near-perfect" — what that would actually mean here

The codebase already has three exemplary patterns: **labeling.py** (all text through pure functions in one module), **reader_utils.py** (boilerplate hoisted once), and **contracts.py** (a dependency-free seam between layers). "Near-perfect," concretely, is: *every cross-cutting concern gets the labeling.py treatment*. The current gaps are exactly the concerns that never got theirs:

1. **Coordinate conventions** → one load-boundary normalizer (A2). Kills 5 bugs.
2. **Vertical/surface selection** → one CF-aware core helper (A3). Kills 2 bugs, closes the "rediscovered 4+ times" loop for good.
3. **Error semantics** → one written stage contract (A1). Kills the data-loss path and the lying banner.
4. **Plot capability metadata** → declared on the class, derived in contracts (A4). Makes the plugin story true.
5. **Schema** → discriminated unions + strict pair schema (A5). Deletes ~200 lines of manual dispatch and closes the typo hole.

Then hold the line with the gates (mypy/black/isort clean, Actions re-enabled) and the <500-line trigger. That is an achievable steady state, and nothing in the codebase structurally resists it.

---

## 5. Bugs

Severity ordered. CONFIRMED = traced and/or reproduced at runtime during this review; PLAUSIBLE = code-level analysis, repro not run. Line numbers are from `develop` @ `9e5ebdb`.

### 5.1 High

**B1 · CONFIRMED — Descending-latitude grids silently mis-pair (then mask everything).**
`pairing/strategies/base.py:209` `_find_nearest_1d` uses `np.searchsorted`, which requires ascending input. Repro: grid lat [90…−90], point at 25° matched to −60°; point at −55° matched to 60°. The radius-of-influence check then measures a huge distance and masks the pair → **all sites silently NaN** for any descending-lat source (ERA5-style) via point/track/profile pairing.
*Fix:* normalize lat ascending at the load boundary (A2), or make `_find_nearest_1d` order-aware (`searchsorted` on a flipped view + index remap).

**B2 · CONFIRMED — `method: grid` (match_dataset) yields an all-NaN grid for descending-lat models.**
`pairing/grid_binning.py:246` (`edges_from_centers`) + `:58` (bin gate): descending centers produce inverted edges (dy<0) and a `lat>=105 and lat<=−105` gate nothing satisfies → swath-vs-model runs bin zero points, output all-NaN, no error.
*Fix:* sort centers ascending before edge construction (or make the binner order-agnostic); raise if zero points bin.

**B3 · CONFIRMED — The headline wavelet path fails silently on monthly (or any irregular) data.**
`analysis/reductions.py:79-80` builds `freq = pd.Timedelta(seconds=med)` and calls `series.resample(time=freq)`; xarray 2024.3.0 raises `TypeError` (expects a frequency *string*). Monthly data is always classified "irregular" (28-day February deviates >5% from the median), so **wavelet-of-an-EOF-PC on monthly CESM output — the documented headline use case — always dies here**, and `analyses.py:123` soft-catches it, so the scalogram just never renders.
*Fix:* pass a string (`f"{int(med)}s"`); better, interpolate to a uniform axis rather than fixed-timedelta resampling of monthly bins (the current approach would inject NaN gaps that flow unhandled into detrend/AR(1)/CWT).

**B4 · CONFIRMED — One failed plot loses the stats CSV (and the UI claims success).**
The A1 mechanism: per-item plot error → `PlottingStage` returns FAILED → `fail_fast` aborts → `SaveResultsStage` never runs → `statistics_summary.csv` missing; `display.py:695-699` prints "(pipeline still succeeded)".
*Fix:* reorder `save_results` before `plotting` in `factory.py`; per-item failures → COMPLETED-with-warnings; wording gated on `result.success`.

### 5.2 Medium

**B5 · PLAUSIBLE (live for several sources) — Longitude conventions never reconciled in nearest-neighbor pairing.** `pairing/strategies/base.py:167`: obs at lon −100° against a 0..360 model grid (CESM, MERRA-2) matches lon 0 instead of 260; RoI masks it → **western-hemisphere sites dropped wholesale** for readers that don't normalize (most don't; only CERES wraps). *Fix:* A2, or wrap both to a common convention pre-lookup.

**B6 · CONFIRMED — Config dump/reload round-trip is broken.** `config/parser.py:273-303`: `dump_schema` (python mode) leaves `PosixPath` objects; `yaml.dump` emits `!!python/object/apply:pathlib.PosixPath` tags; `load_yaml` uses `safe_load`, which rejects them. `load_config(config_to_yaml(cfg))` fails for any config with an output/log dir. *Fix:* dump `mode="json"` (Path→str, datetime→ISO) and/or `yaml.safe_dump`.

**B7 · CONFIRMED — Formula-evaluator integer-exponent DoS.** `analysis/formula.py:106-107`: `left ** right` unguarded; `10**10**8` (right-associative) builds a ~100M-digit int and hangs the pipeline; the DataArray type check runs *after* the compute. `gridded.py:46` feeds user YAML straight in. *Fix:* reject int**int beyond a small bit-length cap, or coerce numeric constants to float (`10.0**1e8 → inf` instantly). The sandbox is otherwise genuinely tight (attribute/subscript/dunder/import all blocked, verified).

**B8 · PLAUSIBLE — Square-grid maps render transposed.** `plots/renderers/spatial/base.py:211`: `data.T if data.shape[0] == len(lons) else data` — for any N×N (lat,lon) field the predicate fires and transposes a *correct* array → lat/lon-swapped map for square grids (EOF patterns, regional AOD). *Fix:* orient by dim names, not lengths.

**B9 · CONFIRMED (by inspection) — "Surface" heuristic inverts for altitude coordinates.** `pairing/strategies/base.py:394-414` auto-detects `["lev","z","level","altitude","height"]` then applies pressure logic (ascending values ⇒ surface at last index). For ascending altitude in metres this selects the **top of atmosphere**. Mirrored in `plots/renderers/spatial/base.py:121-136`. *Fix:* A3 — CF-aware (`positive`/`units`) single helper.

**B10 · CONFIRMED — ICARTT fallback parser: all timestamps collapse to 1970.** `datasets/aircraft/icartt.py:227,386`: seconds-past-midnight is fed to `pd.to_datetime` (interpreted as nanoseconds) with no flight-date anchor; triggers whenever monetio is unavailable. Derived `flight` coord bogus; pairing aligns aircraft to 1970-01-01. *Fix:* read the base date from ICARTT header line 7 and add the seconds offset.

**B11 · CONFIRMED — Fallback text parsers leak sentinel fills as data.** `icartt.py:252`, `ozonesonde.py:205,281`: only ""/"NaN" are masked; header-declared missing codes (−9999, 9000) enter as real values. *Fix:* parse the header missing-value declarations and mask.

**B12 · CONFIRMED — Inspection gate looks in a directory the plot stage doesn't write to.** `inspection/core.py:39-44` rglobs `<output_dir>/plots/**.pdf`, but paired plots go to `<output_dir>/<x_source>/` (`plot.py:618`) and single-source plots to `<output_dir>[/<output_subdir>]` (`plot.py:196-200`). Any paired run with `inspection.required=true` fails `final_pdf_products_exist` despite producing PDFs. *Fix:* consume `plotting.data["plots_generated"]` (already the manifest's source of truth) instead of re-globbing.

**B13 · CONFIRMED (reproduced) — Log files get doubled prefixes and duplicated extras.** `logging/config.py:286-298`: the color console formatter **mutates the shared LogRecord** (`record.msg`, `record.args`), so the file handler re-formats an already-formatted line: `…12:07:23 | INFO | … | 2026-07-02 12:07:23 | INFO | … hello [extras] [extras]`. Also `asctime` missing from `STANDARD_FIELDS` (:102-125) leaks the timestamp into `[extras]` on every line. *Fix:* format a local copy, never assign to `record.msg`; add `"asctime"` to `STANDARD_FIELDS`.

**B14 · CONFIRMED — Tracked example config is malformed YAML.** `examples/configs/cmaq_airnow.yaml:15`: `radius_of_influence: 12000    variables:` on one line → `ScannerError` for every user following the docs. *Fix:* newline + correct indentation.

**B15 · CONFIRMED — `uxarray` is an unpinned phantom dependency that endangers the pandas<2 pin.** `pyproject.toml:36`, `environment.yml:15`: imported nowhere (0 hits incl. tests), not installed in the canonical env — yet a hard dependency. Fresh `pip install davinci-monet` resolves unpinned uxarray → likely drags pandas≥2 → breaks the pinned env and (via the `error::UserWarning` gate) ~dozens of tests. *Fix:* remove until actually used, or make it an optional extra with an explicit pandas-compatible pin.

**B16 · CONFIRMED — Pair-config typos silently accepted.** `config/schema.py:364` `SourcePairConfig` is `extra="allow"`; `methd: grid` validates, `method` stays `auto`, user silently gets geometry pairing instead of intermediate gridding. Only the non-default `--strict` rejects. *Fix:* `extra="forbid"` on the pair schema (it has no passthrough need).

**B17 · PLAUSIBLE — `method: grid` with mixed lon conventions builds a bogus extent.** `intermediate_grid.py:599`: extent spans −180..360 (~538°) when sources disagree; the one-sided shift at :587 doesn't fire. *Fix:* normalize both sources to one convention before computing extent (A2).

**B18 · PLAUSIBLE — Profile vertical interpolation is a no-op for `lev`-named models.** `profile.py:131` hardcodes `level_coord="z"`; `_interpolate_vertical` returns data unchanged when `z` absent (`base.py:362`). Sonde vs CESM (which keeps `lev`) never interpolates — mismatched levels pair positionally. *Fix:* auto-detect the vertical dim with the same candidate list as `_extract_surface` (A3).

**B19 · CONFIRMED — North's-rule error bars can exceed 100%.** `analysis/eof.py:89-101,254`: `n_eff = n(1−r1)/(1+r1)` (single domain-mean r1 for all modes) can fall below 1 — measured mode-1 EV 0.95 with error 1.53 (153%). *Fix:* floor `n_eff`, per-mode r1 (or plain N), clamp display.

**B20 · CONFIRMED — `min_samples` NaNs the sample count itself, and the two stats entry points disagree.** `stats/calculator.py:412-415` NaNs **every** metric including `N` below the floor (verified: 2 pairs → N=NaN); `quick_stats` (:537) applies no floor at all. *Fix:* always report true N; unify the paths.

**B21 · CONFIRMED — EOF memory blow-up.** `analysis/eof.py:214`: eager `np.asarray` + pattern step broadcasting `(time, space, mode)` → ~1.6 GB for a modest daily field. *Fix:* `anomᵀ @ pc / nt`; document the in-memory requirement or add a dask path.

**B22 · PLAUSIBLE — Nested time-concat trusts lexical file order with no monotonic/dedup guard.** `datasets/generic.py:165` (commit e610406): `combine="nested"` concatenates in given order; unpadded names (`out_10.nc` before `out_2.nc`) → scrambled time axis; overlapping files → duplicated timestamps double-counted by nearest-neighbor pairing. Downstream `sel(time=slice(…))` may raise or silently misbehave. *Fix:* `sortby(concat_dim)` + duplicate check after concat.

**B23 · CONFIRMED — `resample_dataset` masks every variable by the FIRST variable's sample count.** `datasets/base.py:41-48`: multi-variable sources with differing validity get under-sampled bins retained (or valid bins dropped) for all but var[0]; `sample_count` describes only var[0]. *Fix:* per-variable count masks.

### 5.3 Low

| # | Where | Defect | Fix |
|---|---|---|---|
| L1 | `display.py:347`, `cli/app.py:82` | Reads `"dataset name"` from `/proc/cpuinfo` (field is `model name`) — rename artifact; Linux CPU never detected | restore `"model name"` |
| L2 | `display.py:695-699` | "(pipeline still succeeded)" printed on failed runs | gate wording on `result.success` |
| L3 | `plots/renderers/scorecard.py:109,178,355` | `center` accepted, passed for bias metrics, never applied → white ≠ zero bias on RdBu_r | wire into `TwoSlopeNorm` |
| L4 | `plots/renderers/taylor.py:153-160` | div-by-zero/NaN-arccos for constant series → points silently vanish | guard `x_std>0`, `isfinite(r)` |
| L5 | `plots/renderers/spatial/overlay.py:154,159-186` | `y_lats/y_lons` can be unbound → NameError; KeyError when only lat present | else-branch raise with clear message |
| L6 | `datasets/satellite/tropomi.py:203-207` | `ground_pixel` aliased to `scanline` (it's the across-track pixel) — dormant | reorder alias |
| L7 | `datasets/satellite/modis_l2_aod.py:176` | epoch arithmetic assumes numeric seconds; datetime64 input would be off by 1e9 | dtype guard |
| L8 | `pipeline/stages/pair.py:412` | injects default `time_tolerance="1h"` though config default is None → coarse-vs-hourly silently NaNs most steps | log the NaN'd count or default None |
| L9 | `pairing/strategies/track.py:55` | `altitude_to_pressure` NaN above ~44 km (negative base, fractional exponent) | clamp base ≥0 |
| L10 | `analysis/reductions.py:49-51` | PointReduce flat `argmin` mis-indexes 2-D curvilinear lat/lon | require 1-D or stack-nearest |
| L11 | `stats/calculator.py:275` | grouped stats zip two groupbys positionally; asymmetric keys misalign silently | assert key equality |
| L12 | `core/base.py:384-387` | POINT dims check tautologically dead (`or "time" in dims` absorbs the site clause); test-only path | fix boolean |
| L13 | `config/parser.py:46-70` | missing file falls through to "parse path as YAML" → cryptic error; `FileNotFoundError` branch dead | separate path/string entry |
| L14 | `config/parser.py:97` + stages | unset `${VAR}` kept literal then `mkdir`'d; `~` unexpanded for output/log dirs (but expanded for input files) | raise on residual `${…}`; `.expanduser()` uniformly |
| L15 | `io/writers.py:58,74,99` | non-atomic writes; crash leaves truncated file treated as valid later | temp + `os.replace` |
| L16 | `logging/config.py:3,220,313` | docstrings promise rotation; plain `FileHandler` | `RotatingFileHandler` or drop claim |
| L17 | `pairing/strategies/point.py:330` | y assigned into x dims positionally (`dims=x_ref.dims` on raw `.values`) — correct only while both are (time,site) | `.transpose(*x_ref.dims)` first |
| L18 | `pairing/grid_binning.py:268` | `edges_from_centers` assumes uniform spacing; Gaussian/stretched grids mis-bin in match_dataset | `searchsorted` on true edges |
| L19 | `analysis/gridded_reductions.py:39-42` vs `stats/calculator.py:342-357` | two different "season" definitions (calendar quarters vs DJF/MAM/JJA/SON) | unify (meteorological) |
| L20 | `cli/app.py:314`; `scripts/run_analysis.py:77` | dead imports; F541 f-string | delete |
| L21 | root `OpenRouter.api` | live API key in working tree — properly gitignored (`*.api`), never committed; still better outside the repo | move to `~/.config/…` or env var |
| L22 | root `REPOS` | untracked scratch file of related-repo URLs | move into `docs/` or delete |
| L23 | `pairing/engine.py:366-371` | bare-name `y` fallback candidate: when x and y variables share a name and a strategy violates the `y_` prefix contract, the same array silently resolves as both axes instead of raising (defeats the guard the docstring promises) | drop the bare `y_name` candidate, or reject it when it equals the resolved `x_key` |

### 5.4 Test-suite defects (from the dedicated pass)

- `integration/test_ceres_readers_pipeline.py:144,257,369`, `test_merra2_reader_pipeline.py:102`, `test_merra2_modis_aod_pipeline.py:160` — real-data smokes that call readers directly but carry `pytest.mark.integration` (violates Testing Rule 1's labeling); use a `real_data`/`smoke` marker. Conversely `test_gridded_aod_product_pipeline.py` is a true pipeline test *missing* the marker.
- **Coverage gap:** paired `spatial_bias` on GRID is never mark-verified (`QuadMesh` vs `PathCollection`) anywhere — all assertions are "PNG > 1 KB". A regression rendering grid bias as scatter would pass. Same for `flight_track`/`lma_density`/`track_map_3d`. CLAUDE.md's own rule demands the programmatic check; only single-source `spatial` has it.
- `unit/core/test_base.py:112-142` — `get_x`/`get_y`/`get_pair` tests assert only `is not None`; an axis-swap bug would pass. Assert values + `source_label`.
- `test_cli_e2e.py:216` — `exit_code != 0 or "error" in stdout` weakens all 15 reject-config cases; `:198` glob-at-import silently collects zero if the dir moves (add a count guard).
- `pyproject.toml:155` — the `error::UserWarning` gate is load-bearing on the pandas<2 pin; document that coupling next to the pin.

---

## 6. Documentation & convention integrity

**The find/replace incident.** A repo-wide "model→dataset" (and apparently "obs→geometry") rename was applied to prose, not just identifiers. Damage:

- **CLAUDE.md gives a config example that fails validation**: `summary: dataset: claude-haiku-4-5` — the actual field is `model:` (schema.py:638). `StrictSchema` rejects `dataset:` outright.
- CLAUDE.md nonsense: "evaluating … datasets against datasets", "Cross-Dataset Handoff Convention", "Data Dataset (xarray-only)", "any dataset can pick up context", "Geometry file paths"; the architecture tree lists `datasets/` twice and omits `analysis/`, `geography/`, `inspection/`, `assets/`, `io/download/`.
- ~10 code docstrings now read "dataset output" (`core/protocols.py:35,51`, `datasets/{generic,cmaq,ufs,wrfchem}.py`, `plots/renderers/{diurnal,timeseries,spatial/base}.py`), plus "Base dataset with strict validation settings" (`schema.py:24`).
- One functional casualty: L1 (`"dataset name"` vs `/proc/cpuinfo`).

*Fix:* one sweep commit restoring "model output"/"model name"/"Cross-Model" in prose and docstrings; correct the CLAUDE.md `summary:` example; then add a cheap guard (grep for `dataset output`/`dataset name` in CI/pre-commit) so a future rename can't silently do this again.

**Stale CLAUDE.md facts:** "1,262 tests" (now 1,657); "pipeline executes 5 stages" (standard pipeline is 10: load, analyses, plot_suites, pairing, statistics, plotting, save_results, summary, inspection, manifest); "all gates passing" (§3). Also "rejected with a migration error" — the code raises plain `ValueError` (wrapped as generic `ConfigurationError`); the purpose-built `ConfigMigrationError` is exported but never used.

**Other:** stale "DAVINCI-MONET" branding in `scripts/run_analysis.py`, `examples/run_all_examples.py`, `examples/configs/cmaq_airnow.yaml`; stale `examples/output/plots/08_spatial_distribution.*` artifacts for a removed plot type; the rasterization convention is enforced in eof/wavelet renderers but **missing from the shared `draw_spatial_field` primitive** (`spatial/base.py:198,208,217`) and from overlay/curtain/lma_density/scatter/scorecard/vertical_profile — the single highest-value fix is `rasterized=True` once inside `draw_spatial_field`, which covers the PDF-only AOD/SARB deliverables; and `lma_density.py:116` appends the hour window to the *title* (belongs in the subtitle per the labeling rules).

---

## 7. Test suite assessment

Strong overall — see §4.1 for what's working. The suite's structure mirrors the package; synthetic generators set geometry attrs, realistic units, and exercise the CESM vertical convention in both orderings; registry mutations are restored; real-file I/O is env-gated. The defects worth acting on are in §5.4; the theme is **assertion strength at the plot layer** (PNG-size gates instead of artist-type/content checks) and **marker hygiene**. One structural risk to keep in view: the whole suite's green depends on the pandas 1.5.3 pin via the `error::UserWarning` gate — any dependency change that upgrades pandas flips ~dozens of tests red for reasons unrelated to the change (this is also why B15/uxarray matters).

---

## 8. Prioritized action plan

**P0 — stop the bleeding (≈1–2 days)**
1. Clear the gate: fix 42 mypy errors, 3 black, 3 isort files; add `cftime` to mypy overrides. Re-enable Actions or adopt a written pre-merge gate ritual.
2. B4/A1: reorder `save_results` before `plotting`; per-item errors → COMPLETED-with-warnings; fix the success banner (L2).
3. B3: wavelet resample string fix (one line) — restores the headline feature.
4. B14: fix the broken example config. B15: remove/pin uxarray.
5. Docs sweep: repair CLAUDE.md (summary key, stage list, test count, tree) + the ~10 mangled docstrings + L1.

**P1 — correctness architecture (≈1 week)**
6. A2: load-boundary coordinate normalization (lat ascending, lon canonical) → closes B1, B2, B5, B17; add regression tests with descending-lat and 0..360 fixtures.
7. A3: single CF-aware surface/vertical helper → closes B9, B18; migrate track/swath.
8. B6, B7, B10, B11, B12, B13, B16, B22, B23 individually (each is small and independent).
9. Add the missing mark-verification tests (spatial_bias QuadMesh; axis-correctness for `get_x`/`get_y`); wire `analysis_errors` into the error itemization maps (A10).

**P2 — architecture consolidation (≈2–3 weeks, incremental)**
10. A4: plotter-declared arity, registry-derived contracts; resolve the taylor drift; StatMetric decision.
11. A5: discriminated unions; delete manual dispatch/coercion validators; split schema.py.
12. A7: dedup catalogue (reader base class, standardize/QA helpers, intermediate_grid `_bin()`, system-info, file-list resolver, lon-resort, mark-selection).
13. A9: delete dead code (modis_l2.py, 8 exception classes, misc).
14. A10: typed stage outputs.
15. Rasterization: `rasterized=True` in `draw_spatial_field` + the per-renderer stragglers; B8 transpose fix by dim names.

**P3 — polish & performance**
16. A12 items (cKDTree, EOF memory, plot-suite revalidation, lazy plots import for validate).
17. Remaining lows (L3–L22); B19–B21 numerics refinements; season-definition unification (L19).
18. Module-size splits where cohesion allows (display.py, timeseries.py, metrics.py).

---

## Appendix — subsystem scorecards

| Subsystem | Design | Correctness | Notes |
|---|---|---|---|
| core/ | A− | A | Registry/protocols excellent; exceptions.py 40% dead |
| config/ | B+ | B | Validation thorough; round-trip broken (B6); pydantic underused |
| datasets/ | B+ | B+ | reader_utils strong; fallback parsers weak (B10/B11); dup across readers |
| pairing/ | B+ | C+ | Engine contract excellent; coordinate-convention bugs (B1/B2/B5); intermediate_grid oversized |
| plots/ | A− | B | Labeling/contract exemplary; rasterization inconsistent; B8 latent |
| stats/ | A− | A− | Formulas verified correct; min_samples quirk (B20) |
| analysis/ | B | B− | EOF math sound but eager (B21); wavelet path broken (B3); formula sandbox tight minus B7 |
| pipeline/ | B+ | B− | Clean stage framework; error contract self-contradicting (A1/B4); inspection gate misaimed (B12) |
| cli/ | A− | A− | Idiomatic Typer, correct exit codes; minor dups/dead imports |
| io/, util/, logging/ | B+ | B | Download layer excellent; logging mutation bug (B13); non-atomic writes |
| ai/ | A− | A | Non-fatal guarantee real; key hygiene good |
| tests/ | A− | A− | Integration-through-pipeline honored; plot-layer assertions weak |

*Generated by Claude Fable 5 with eight parallel subsystem review agents plus an adversarially-verified design-review workflow; all CONFIRMED items were traced in code and, where marked, reproduced at runtime in the `davinci` conda env on 2026-07-02.*
