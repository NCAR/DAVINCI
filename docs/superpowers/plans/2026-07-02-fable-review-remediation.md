# FABLE Review Remediation (P0–P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For the Opus 4.8 orchestrator:** this plan is designed for a dynamic Workflow. Map **one task → one subagent**. Waves define the dependency order; tasks inside Wave 1 are file-disjoint and may run as parallel agents (use `isolation: "worktree"` and merge each task's commit back to `develop` as it completes; on conflict, rebase the task branch — conflicts should not occur if the file lists below are respected). Waves 2–4 are sequential. Run the full gate (Global Constraints) after every wave. Every CONFIRMED/PLAUSIBLE label, B#/L#/A# reference, and file:line below comes from `FABLE_REVIEW.md` (repo root, commit `fafb39c`) — the spec for this plan. Subagents should read the referenced FABLE_REVIEW section before their task.

**Goal:** Clear the quality gates and fix all high- plus medium-severity correctness bugs from FABLE_REVIEW.md (P0+P1), including the two root-cause architectural fixes (coordinate-convention contract, CF-aware surface selection).

**Architecture:** Three root-cause fixes carry most of the value: (1) a load-boundary horizontal-coordinate normalizer that makes descending-lat/0..360 sources safe everywhere downstream; (2) a single CF-aware vertical/surface helper in `core/` replacing four divergent copies; (3) a rewritten stage error contract (per-item failures are warnings, `save_results` runs before `plotting`). Everything else is small, independent bug fixes done TDD.

**Tech Stack:** Python 3.11, xarray, pydantic v2, pytest, mypy, black/isort. All commands run in the `davinci` conda env.

## Global Constraints

- **Environment (every test/gate command):** `source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE ...`
- **The full gate** (run after each task's commit, and after each wave): `HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet && black --check davinci_monet && isort --check-only davinci_monet` — all four must pass.
- **pandas must remain 1.5.3.** After ANY dependency change run `python -c "import pandas; print(pandas.__version__)"` and abort if ≠ 1.5.3 (the `error::UserWarning` pytest gate is load-bearing on this pin).
- **Git:** work on `develop` only. Commit per task. **NEVER push and NEVER merge to `main`** — the user pushes/merges after verifying. Commit messages end with `Co-Authored-By: Claude <model> <noreply@anthropic.com>` (use the executing model's name).
- **Testing Rule 1 (CLAUDE.md):** tests labeled integration MUST go through `PipelineRunner.run_from_config()` / `run_analysis()`. Renderer/stage-direct tests are unit tests.
- **Testing Rule 3:** no shortcuts for green checkmarks — if a pipeline-path test is hard, investigate, don't bypass.
- **Plot text** only via `davinci_monet/plots/labeling.py` functions; never rasterize `contour` LINE sets (trips the `error::UserWarning` gate); dense mesh/fill/scatter layers in PDF must set `rasterized=True`.
- **Paired-variable naming:** source-label prefix (`cam_o3`), attrs `axis` + `source_label`; strategies emit `x_`/`y_`, engine relabels.
- Do not modify `FABLE_REVIEW.md`. Do not touch the `OpenRouter.api` file. Do not add new hard dependencies.

---

## Wave 0 — restore the gate (sequential, must land first)

### Task 1: Make mypy/black/isort green on develop

**Files:**
- Modify: `pyproject.toml:110-140` (mypy overrides list)
- Modify: `davinci_monet/analysis/formula.py:161,175,177`
- Modify: `davinci_monet/plots/renderers/spatial/field.py:171,217,282`
- Modify: `davinci_monet/pipeline/stages/load.py:276,282`
- Modify: `davinci_monet/pipeline/runner.py:216-218`
- Modify: `davinci_monet/cli/commands/inspect.py:22` (+ reformat)
- Modify: `davinci_monet/inspection/core.py` (reformat only)
- Modify: `davinci_monet/datasets/__init__.py` (isort only)
- Modify: test files flagged by mypy: `tests/unit/config/test_product_config_schema.py`, `tests/unit/analysis/test_gridded_analysis_schema.py`, `tests/unit/analysis/test_gridded_analysis.py`, `tests/unit/pipeline/test_plot_suite_stage.py`, `tests/unit/pipeline/test_single_source_stages.py`, `tests/unit/pipeline/test_inspection_stage.py`, `tests/unit/plots/test_robust_clim.py`, `tests/unit/plots/test_timeseries_renderer.py`, `tests/unit/pairing/test_grid_binning.py` + `tests/unit/datasets/satellite/test_modis_l2_aod.py` (isort), `tests/unit/inspection/test_inspection_core.py` (black)

**Interfaces:**
- Consumes: nothing.
- Produces: a green 4-gate baseline every later task builds on. `PipelineRunner.__init__`/`add_stage` keep their public signatures but consistently use the `Stage` protocol type.

- [ ] **Step 1: Reproduce the failures** — run the full gate; capture the 42 mypy errors, 3 black files, 3 isort files. Expected: exactly the FABLE_REVIEW §3 list.
- [ ] **Step 2: Silence untyped `cftime`** — in `pyproject.toml`, add `"cftime.*"` to the existing `[[tool.mypy.overrides]]` module list (the one containing `"monetio.*"`).
- [ ] **Step 3: Fix production-code errors (no `# type: ignore` unless a comment justifies it):**
  - `runner.py`: type the stage list with the protocol — `self._stages: list[Stage] = list(stages) if stages is not None else list(create_standard_pipeline())` (list invariance: `list[BaseStage]` is not `list[Stage]`).
  - `cli/commands/inspect.py:22`: change the CLI param to `preview_format: str = typer.Option("png", ...)` → validate/narrow before the call: `fmt: Literal["png"] | None = "png" if preview_format == "png" else None` (match `inspect_run_directory`'s accepted type; if it should accept "pdf", widen the callee's Literal instead — read its signature first and pick the one true type).
  - `spatial/field.py:171`: assign the `Figure | None` to a temp and raise/narrow (`if fig is None: raise PlottingError(...)`) before use; `:217` keep `cmap` as `str | Colormap | None` in the local annotation; `:282` pass the colorbar kwargs with the correct types (read `add_colorbar`'s signature; don't `**` a `dict[str, str]` into a float param).
  - `load.py:282`: `values = np.asarray(maybe_list).ravel()` instead of calling `.ravel()` on a `Any | list` union.
  - `formula.py:161,175,177`: annotate `eval_env: dict[str, Any]` and build it with `str(name)` keys: `eval_env = {str(name): env[name] for name in env.data_vars}`; for `:161` guard `func = _FUNCTIONS[node.func.id]` (typed as `Callable[..., Any]` in the dict annotation: `_FUNCTIONS: dict[str, Callable[..., Any]]`).
- [ ] **Step 4: Fix test-file errors** — construct schema objects properly instead of raw dicts (e.g. `MonetConfig(analysis=AnalysisConfig(...), inspection=InspectionConfig(enabled=True, required=False), sources={"cam": SourceConfig(type=..., ...)})`), or where a test intentionally feeds dicts through pydantic coercion, `model_validate({...})` (typed as accepting `Any`) instead of kwargs. For `test_robust_clim.py`: the helper returns `Figure | list[tuple[str, Figure]]` — isolate with `assert isinstance(fig, Figure)` before `plt.close(fig)`/`.axes`.
- [ ] **Step 5: Run `black davinci_monet && isort davinci_monet`** to fix the 3+3 formatting files.
- [ ] **Step 6: Full gate.** Expected: `1657 passed, 6 skipped` / `Success: no issues found in 339 source files` / black+isort clean.
- [ ] **Step 7: Commit** — `git add -A && git commit -m "chore: restore mypy/black/isort gate on develop"`

---

## Wave 1 — independent fixes (file-disjoint; parallelizable as worktree agents)

### Task 2: B3 — wavelet resample TypeError (headline monthly path)

**Files:**
- Modify: `davinci_monet/analysis/reductions.py:79-80`
- Test: `davinci_monet/tests/integration/test_wavelet_pipeline.py` (add one test)

**Interfaces:** Produces a working `reduce_to_series` for irregular time axes; no signature changes.

- [ ] **Step 1: Failing test (integration — through the pipeline, per Testing Rule 1).** Add to `test_wavelet_pipeline.py` a config-driven run whose source has a **monthly** (`freq="MS"`, ≥36 steps) synthetic grid, `analyses: {eof: ..., wav: {type: wavelet, source: <eof>, variable: pc, mode: 1}}`, and a `wavelet_scalogram` plot. Assert the run succeeds AND the scalogram file exists AND `"wav"` appears in `context.sources` (i.e. the analysis was not soft-skipped). Model the config/bootstrap on the existing `test_areamean_wavelet`.
- [ ] **Step 2: Run it.** Expected: FAIL — analysis silently missing (the `TypeError: ... got 'Timedelta'` is swallowed at `analyses.py:123`; the assert on the pseudo-source/plot catches it).
- [ ] **Step 3: Fix.** In `reductions.py:79-80` replace the Timedelta object with a frequency string and interpolate rather than bin-resample an irregular axis:
  ```python
  med = float(np.median(np.diff(times).astype("timedelta64[s]").astype(float)))
  freq = f"{int(med)}s"
  uniform = series.resample(time=freq).interpolate("linear")
  ```
  Keep the regular-axis fast path unchanged. If NaNs remain at the edges, `.dropna("time")` before returning (pycwt cannot take NaNs).
- [ ] **Step 4: Run the new test + the whole wavelet test file.** Expected: PASS, no NaN-related pycwt warnings.
- [ ] **Step 5: Full gate, then commit** — `fix(analysis): wavelet reduction works on monthly/irregular time axes (B3)`

### Task 3: B4/A1 — stage error contract + save order + honest banner (+ `analysis_errors` wiring)

**Files:**
- Modify: `davinci_monet/pipeline/stages/factory.py:30-41` (order), `davinci_monet/pipeline/stages/plot.py:746-753`, `davinci_monet/pipeline/stages/stats.py:144-151`, `davinci_monet/pipeline/stages/pair.py:577-585`, `davinci_monet/pipeline/runner.py:117-121,459-462`, `davinci_monet/pipeline/display.py:685-699`, `davinci_monet/pipeline/reporting.py:242-249`
- Test: `davinci_monet/tests/unit/pipeline/test_pipeline_core.py` (amend `test_stats_item_error_fails_stage` and add), `davinci_monet/tests/test_integration.py` (add one pipeline test)

**Interfaces:**
- Produces the contract every later stage obeys: *per-item* failures → stage returns `StageStatus.COMPLETED` with the item errors in `result.data["<stage>_errors"]` and a `warnings` count; stage-level exceptions → `FAILED`. `save_results` executes before `plotting` in both factory pipelines. `print_item_errors(errors, success: bool)` gains the success flag.

- [ ] **Step 1: Failing integration test.** Config with 2 pairs, 1 valid plot + 1 plot spec whose renderer will raise (e.g. `type: scatter` referencing a pair variable that exists but with `pairs:` pointing at a valid pair and a monkeypatched renderer that raises — prefer a config-level trigger: a `spatial` plot with `source:` valid and `variable:` present but a monkeypatched `SpatialPlotter.render` raising RuntimeError). Assert: `result.success is True`, `statistics_summary.csv` exists, the good plot exists, and `plot_errors` has 1 entry. Run: FAIL (today the run aborts, no CSV, success False).
- [ ] **Step 2: Reorder** `create_standard_pipeline` and `create_geometry_pipeline`: `SaveResultsStage()` immediately **before** `PlottingStage()`.
- [ ] **Step 3: Change the three stages** to return `StageStatus.COMPLETED` when only per-item errors occurred (keep the error lists in `result.data` exactly as-is — keys unchanged). A stage still returns FAILED when it produced **zero** outputs because *every* item failed (empty pairing = FAILED; 1-of-2 plots = COMPLETED).
- [ ] **Step 4: Fix messaging.** `display.print_item_errors(..., success: bool)` — wording: success → `"N non-fatal error(s) occurred (run completed with warnings)"`; failure → `"N error(s) occurred before the pipeline stopped"`. Update the two runner call sites. Add `"analyses": "analysis_errors"` to the `_ITEM_ERROR_STAGES`-style maps in `runner.py`, `reporting.py`, and `display.py` so AnalysesStage errors are itemized like the others (A10 drift).
- [ ] **Step 5: Amend `test_stats_item_error_fails_stage`** to the new contract (item error → COMPLETED + `stats_errors` non-empty; all-items-failed → FAILED). Run unit + new integration test: PASS.
- [ ] **Step 6: Full gate, then commit** — `fix(pipeline): per-item errors are warnings; save_results precedes plotting (B4/A1)`

### Task 4: B13 — logging formatter mutation + asctime leak

**Files:**
- Modify: `davinci_monet/logging/config.py:102-125,286-298`
- Test: `davinci_monet/tests/unit/logging/` (add `test_formatter_isolation.py`; create dir with `__init__.py` if absent)

- [ ] **Step 1: Failing test** — build a logger with BOTH the color console formatter and the file `StructuredFormatter` (two handlers, one record), log `logger.info("hello", extra={"var": "O3"})`, read the file line. Assert the line contains exactly one timestamp, one `INFO`, one `[var='O3']`, and no `asctime=`. Run: FAIL (doubled prefix + `[asctime=...]`).
- [ ] **Step 2: Fix** — in the color formatter, never assign to `record.msg`/`record.args`; format a copy: `record_copy = logging.makeLogRecord(record.__dict__)` then operate on `record_copy` (or build the colored string purely from `super().format(record)`'s return). Add `"asctime"` to `STANDARD_FIELDS`.
- [ ] **Step 3: Test passes; full gate; commit** — `fix(logging): formatters no longer mutate shared records; asctime excluded from extras (B13)`

### Task 5: B6 — config dump/reload round-trip

**Files:**
- Modify: `davinci_monet/config/parser.py:273-303` (`dump_config`, `config_to_yaml`)
- Test: `davinci_monet/tests/unit/config/test_schema.py` (add round-trip tests)

**Interfaces:** `config_to_yaml`/`dump_config` emit `yaml.safe_load`-able YAML (paths as strings, datetimes as ISO strings). `dump_schema` callers elsewhere are untouched.

- [ ] **Step 1: Failing tests** — `load_config(config_to_yaml(cfg))` and `dump_config(cfg, p); load_config(p)` for a config with `analysis.output_dir`/`log_dir` set. Run: FAIL with `could not determine a constructor for the tag ... pathlib.PosixPath`.
- [ ] **Step 2: Fix** — in both functions dump JSON-mode: `data = dump_schema(config, mode="json", exclude_none=True, exclude_unset=True)` (add a `mode` passthrough param to `core/schema_utils.dump_schema` if it lacks one, defaulting to `"python"` so other callers are unchanged) and write with `yaml.safe_dump(data, ..., default_flow_style=False, sort_keys=False)`.
- [ ] **Step 3: Tests pass; full gate; commit** — `fix(config): YAML dump round-trips through safe_load (B6)`

### Task 6: B16 — pair-config typos must fail validation

**Files:**
- Modify: `davinci_monet/config/schema.py:364` (`SourcePairConfig` base class), `davinci_monet/config/parser.py:164-171` (drop now-redundant pair extra checks)
- Test: `davinci_monet/tests/unit/config/test_schema.py`

- [ ] **Step 1: Failing test** — `load_config` (non-strict) of a pair with `methd: grid` must raise `ConfigurationError` mentioning `methd`. Run: FAIL (currently accepted, `method == "auto"`).
- [ ] **Step 2: Fix** — change `class SourcePairConfig(FlexibleSchema)` → `(StrictSchema)`. Keep `AxisRef`/`GridConfig` strict as they are. Remove the manual `_raise_for_extra(pair, ...)` block for pairs in `_validate_strict_core_fields` (now automatic). **Check fallout:** grep `analyses/*/configs/*.yaml`, `examples/configs/*.yaml`, `tests/` for pair keys not in the schema; if a legit key surfaces (e.g. `time_tolerance` on a pair), add it to the schema as a typed field instead of re-loosening.
- [ ] **Step 3: Full suite (this is the riskiest wave-1 change — any test feeding stray pair keys will surface); fix surfaced configs; full gate; commit** — `fix(config): reject unknown keys in pair specs (B16)`

### Task 7: B7 — formula evaluator exponent guard

**Files:**
- Modify: `davinci_monet/analysis/formula.py:106-107`
- Test: `davinci_monet/tests/unit/analysis/test_formula.py`

- [ ] **Step 1: Failing test** — `evaluate_formula("10**10**8", {"a": xr.DataArray([1.0])})` must raise `FormulaError` in <1s (wrap with a 5s `signal.alarm` guard or just assert it raises promptly). Run: FAIL (hangs / long compute).
- [ ] **Step 2: Fix** — in `visit_BinOp`, before computing `Pow`:
  ```python
  if isinstance(node.op, ast.Pow):
      if isinstance(left, int) and isinstance(right, int) and (
          abs(right) > 64 or left.bit_length() * max(abs(right), 1) > 1024
      ):
          raise FormulaError("integer exponent too large in formula")
      return left**right
  ```
  (DataArray/float operands are unaffected.)
- [ ] **Step 3: Existing escape tests + new test pass; full gate; commit** — `fix(analysis): guard integer-exponent blowup in formula evaluator (B7)`

### Task 8: B10/B11 — ICARTT & sonde fallback parsers (time anchor, sentinel fills)

**Files:**
- Modify: `davinci_monet/datasets/aircraft/icartt.py:227-260,380-395`, `davinci_monet/datasets/sonde/ozonesonde.py:205,281`
- Test: `davinci_monet/tests/unit/datasets/` (add `aircraft/test_icartt_fallback.py`, `sonde/test_ozonesonde_fallback.py`)

**Interfaces:** fallback parsers produce `time` as `datetime64[ns]` anchored to the header flight date, and mask header-declared missing values to NaN. No public signature changes.

- [ ] **Step 1: Failing tests.** Write a minimal in-test ICARTT header + 3 data rows to `tmp_path` (header line 7 = `2024 02 01 2024 03 15`, one variable with missing code `-9999`, times `43200,43260,43320`). Parse via the module's fallback function directly (unit test). Assert `time[0] == np.datetime64("2024-02-01T12:00:00")` and the `-9999` row is NaN. Similarly a SHADOZ block with `9000`/`-9999` fills. Run: FAIL (1970 timestamps; fills present).
- [ ] **Step 2: Fix ICARTT** — parse the date from header line index 6 (`YYYY MM DD` first triplet); after building the frame: `base = pd.Timestamp(year, month, day, tz=None)`; `df.index = base + pd.to_timedelta(df.index.astype(float), unit="s")`. Parse the missing-value line (per ICARTT 1001: line 12 region, one code per dependent variable, aligned with the variable list) and `df[col] = df[col].replace(code, np.nan)` per column; also apply declared scale factors if the header carries non-1 scales.
- [ ] **Step 3: Fix sondes** — after numeric parse, mask each column's declared missing codes; for SHADOZ additionally mask values ≥ 9000 in ozone columns (documented SHADOZ convention).
- [ ] **Step 4: Tests pass; full gate; commit** — `fix(datasets): ICARTT/sonde fallback parsers anchor time and mask sentinel fills (B10, B11)`

### Task 9: B12 — inspection gate consumes real plot outputs

**Files:**
- Modify: `davinci_monet/inspection/core.py:39-44` + its caller `davinci_monet/pipeline/stages/inspection.py`
- Test: `davinci_monet/tests/unit/pipeline/test_inspection_stage.py`

**Interfaces:** `inspect_run_directory(...)` gains an optional `plot_paths: Sequence[Path] | None = None`; when given, PDF discovery uses it instead of rglobbing `<root>/plots`. `InspectionStage` passes `context`'s `plots_generated` list (the same source the manifest uses).

- [ ] **Step 1: Failing unit test** — build a context whose plotting stage recorded PDFs under `output_dir/<x_source>/` (paired-plot layout), `inspection.required=True`; assert the stage passes `final_pdf_products_exist`. Run: FAIL (gate rglobs `output_dir/plots`).
- [ ] **Step 2: Fix** as per Interfaces; keep the rglob as fallback when `plot_paths is None` (CLI `inspect` command path).
- [ ] **Step 3: Test + `tests/unit/inspection/` + gridded-AOD integration test pass; full gate; commit** — `fix(inspection): gate discovers PDFs from plots_generated, not a hardcoded dir (B12)`

### Task 10: B22/B23 — reader-side data integrity (concat order, per-var resample counts)

**Files:**
- Modify: `davinci_monet/datasets/generic.py:160-175`, `davinci_monet/datasets/base.py:41-48`
- Test: `davinci_monet/tests/test_dataset_readers.py` (generic concat), `davinci_monet/tests/unit/datasets/test_resample.py` (create if absent)

- [ ] **Step 1: Failing tests.** (a) Write three tiny NetCDFs named `out_1.nc`, `out_10.nc`, `out_2.nc` with disjoint times; open via the generic reader's nested-concat path; assert the time axis is strictly monotonic increasing. (b) `resample_dataset` on a 2-var dataset where var A has 5 samples/hour and var B has 1: with `min_sample_count=3`, assert A's hour survives and B's hour is NaN (today B survives because A's count masks both).
- [ ] **Step 2: Fix (a)** — after the nested `open_mfdataset`/concat: `ds = ds.sortby(concat_dim)`; then `if pd.Index(ds[concat_dim].values).has_duplicates: raise DataFormatError(f"duplicate {concat_dim} values across input files")`.
- [ ] **Step 3: Fix (b)** — compute `counts = resampler.count()` once (Dataset), then per variable `out[v] = out[v].where(counts[v] >= min_sample_count)`; `sample_count` becomes per-variable (`sample_count_<var>`) when variables differ, else keep the single variable (preserve the existing name when len(data_vars)==1 for backward compat).
- [ ] **Step 4: Tests pass; full gate; commit** — `fix(datasets): monotonic nested concat; per-variable resample count masks (B22, B23)`

### Task 11: B14 + branding — repair example config, stale names

**Files:**
- Modify: `examples/configs/cmaq_airnow.yaml:15`, `scripts/run_analysis.py:2,29,63,77`, `examples/run_all_examples.py:2,41`
- Delete: `examples/output/plots/08_spatial_distribution.pdf`, `examples/output/plots/08_spatial_distribution.png`

- [ ] **Step 1: Failing check** — `python -c "from davinci_monet.config.parser import load_config; load_config('examples/configs/cmaq_airnow.yaml')"`. Expected today: `ScannerError` line 15.
- [ ] **Step 2: Fix the YAML** — split line 15 so `variables:` starts its own properly-indented block under the source; verify `O3`/`PM25_TOT` indentation. Re-run Step 1: must raise nothing OR only a data-file-not-found style error (parse succeeds).
- [ ] **Step 3:** Replace "DAVINCI-MONET" → "DAVINCI" in the three scripts; fix the F541 f-string (`print("Analysis completed successfully!")`). Delete the two stale `08_spatial_distribution.*` artifacts.
- [ ] **Step 4: Full gate; commit** — `fix(examples): repair cmaq_airnow.yaml; refresh stale branding (B14, L20)`

### Task 12: B15 — remove phantom uxarray dependency

**Files:**
- Modify: `pyproject.toml:36`, `environment.yml:15`

- [ ] **Step 1: Verify unused** — `grep -rn uxarray davinci_monet/ examples/ scripts/` → expect only the two dependency entries, zero imports.
- [ ] **Step 2: Remove** the `uxarray` line from both files. (If the CERES follow-on work needs it later, it returns as an optional extra with an explicit pandas<2-compatible pin — record that in the commit body.)
- [ ] **Step 3:** `pip install -e ".[dev]"` in the davinci env, then `python -c "import pandas; print(pandas.__version__)"` → must print `1.5.3`. Full gate. Commit — `chore(deps): drop unused uxarray dep that endangers the pandas<2 pin (B15)`

---

## Wave 2 — root-cause architecture fixes (sequential; each is one agent)

### Task 13: A2/B1/B2/B5/B17 — load-boundary horizontal coordinate contract

**Files:**
- Modify: `davinci_monet/io/reader_utils.py` (add `normalize_horizontal_coords`), `davinci_monet/pipeline/stages/load.py` (apply post-open), `davinci_monet/pairing/strategies/base.py:209` (defensive), `davinci_monet/pairing/grid_binning.py:246` (defensive), `davinci_monet/pairing/strategies/intermediate_grid.py:580-600` (extent)
- Test: `davinci_monet/tests/unit/io/test_reader_utils.py`, `davinci_monet/tests/unit/pairing/test_engine_and_strategies.py`, `davinci_monet/tests/test_integration.py` (one pipeline test)

**Interfaces (the contract — copy into the function docstring):**
```python
def normalize_horizontal_coords(ds: xr.Dataset) -> xr.Dataset:
    """Canonicalize horizontal coordinates after a reader opens a source.

    Guarantees on return, for 1-D 'lat'/'lon' coordinate variables:
      * lat strictly ascending (sortby when needed)
      * lon in [-180, 180) (wrapped ((lon + 180) % 360) - 180, then sorted
        ascending with the data rolled accordingly)
    2-D (curvilinear) lat/lon and point/track/profile geometries are returned
    unchanged except lon wrapping of coordinate values (no reordering).
    Attributes and non-horizontal dims are untouched.
    """
```
The load stage calls this for **every** source right after `reader.open(...)` (single call site — `_load_single_source` or equivalent). Downstream code may assume the contract; the two defensive fixes below turn violations into loud errors instead of silent wrong answers.

- [ ] **Step 1: Failing unit tests** for the normalizer: descending-lat grid → ascending with data remapped (check a marker value moves with its coordinate); 0..360 lon grid → [-180,180) sorted, marker value follows; already-canonical dataset → identical (use `xr.testing.assert_identical`); 2-D lat/lon → values wrapped, order untouched; point dataset with lon 240 → −120.
- [ ] **Step 2: Implement; unit tests pass.**
- [ ] **Step 3: Failing regression tests** (these encode B1/B2/B5): (a) point pairing against a **descending-lat** gridded source through `PairingEngine.pair_sources` → paired values must match the analytically-correct cell (assert exact value, not just non-NaN); (b) same via a `method: grid` match_dataset pair → non-empty grid with correct cell means; (c) obs at lon −100 vs a 0..360 model → pairs within RoI (B5). Written against the pipeline load path (build tiny configs; run `run_analysis`) so the normalizer is exercised where it lives — Testing Rule 1.
- [ ] **Step 4: Wire the normalizer into the load stage.** Regression tests pass.
- [ ] **Step 5: Defensive hardening** — `_find_nearest_1d`: `if vals[0] > vals[-1]: raise PairingError("latitude must be ascending — normalize_horizontal_coords was bypassed")`; `edges_from_centers`: same check; `intermediate_grid` extent: compute after asserting both sources' lons are within [-180, 180] (raise otherwise). Unit tests for each raise.
- [ ] **Step 5b (L18): non-uniform grids bin correctly in match_dataset** — failing unit test: lat centers `[0, 1, 2, 4, 8]` (non-uniform), points at 3.9 and 4.1 must land in the cell centered at 4. Fix `bin_swath_to_grid`'s index computation to `np.searchsorted(edges, values, side="right") - 1` (bounds-checked) instead of `(value - edge0) / d` arithmetic; keep the uniform fast path only if it stays exactly equivalent (else delete it).
- [ ] **Step 6: Full gate (watch CERES tests — those readers already normalized; the new pass must be idempotent). Commit** — `fix(pairing,io): canonical horizontal-coordinate contract at load boundary (A2: B1,B2,B5,B17)`

### Task 14: A3/B9/B18 — one CF-aware vertical/surface helper

**Files:**
- Create: `davinci_monet/core/vertical.py`
- Modify: `davinci_monet/pairing/strategies/base.py:379-414` (delegate), `davinci_monet/plots/renderers/spatial/base.py:121-136` (delegate), `davinci_monet/plots/renderers/spatial/field.py:48`, `davinci_monet/plots/renderers/spatial/overlay.py:142`, `davinci_monet/pairing/strategies/intermediate_grid.py:138`, `davinci_monet/pairing/strategies/track.py:185,425`, `davinci_monet/pairing/strategies/profile.py:125-135`
- Test: `davinci_monet/tests/unit/core/test_vertical.py` (create), plus existing `test_extract_surface_*` and `test_spatial_surface_level.py` must keep passing

**Interfaces:**
```python
# davinci_monet/core/vertical.py
VERTICAL_DIM_CANDIDATES: tuple[str, ...] = ("lev", "z", "level", "altitude", "height", "vertical")

def find_vertical_dim(obj: xr.Dataset | xr.DataArray) -> str | None:
    """First VERTICAL_DIM_CANDIDATES member present in obj.dims, else None."""

def surface_level_index(obj: xr.Dataset | xr.DataArray, level_dim: str) -> int:
    """Index of the surface along level_dim. Decision order:
    1. CF 'positive' attr on the coord: 'down' -> surface at values.argmax();
       'up' -> surface at values.argmin().
    2. Units on the coord: pressure units (Pa, hPa, mb, millibar) -> argmax;
       length units (m, km, meter*, kilometer*) -> argmin.
    3. Unambiguous dim name: 'altitude'/'height' -> argmin (height-like).
       ('z', 'lev', 'level' stay ambiguous: CESM renames lev->z with
       pressure-like values, so name alone must NOT imply height.)
    4. Fallback (preserves current CESM behavior): values[-1] > values[0]
       -> -1 (ascending == pressure increasing toward surface), else 0.
    Coord missing or <2 values -> 0."""
```

- [ ] **Step 1: Failing unit tests** covering every decision row: `positive: "down"` descending values; `positive: "up"`; `units: "hPa"` ascending; `units: "m"` **ascending → expect index 0** (this is B9 — fails today); dim named `height` ascending → 0; dim `z` with pressure-like unattributed values ascending → −1 (CESM back-compat); single-level → 0.
- [ ] **Step 2: Implement `core/vertical.py`; unit tests pass.**
- [ ] **Step 3: Delegate all six call sites** to the new helper; delete the local copies and the divergent candidate lists (pairing's `["lev","z","level","altitude","height"]`, renderers' `"vertical"` variants, `intermediate_grid`'s subset) in favor of `VERTICAL_DIM_CANDIDATES`/`find_vertical_dim`. `track.py:425`'s re-derivation delegates too.
- [ ] **Step 4: B18** — `profile.py:131`: replace hardcoded `level_coord="z"` with `find_vertical_dim(y_data)`; add a unit test: synthetic sonde (dim `level`) vs model with dim `lev` → interpolation actually runs (output levels == obs levels; fails today as a silent no-op).
- [ ] **Step 5: Renderer-side regression test:** grid field with vertical dim `height` (ascending, `units: "m"`) through `type: spatial` → sliced value equals the **lowest** level's value (mark-verified per CLAUDE.md; extend `test_spatial_surface_level.py`).
- [ ] **Step 6: Full gate. Commit** — `refactor(core): single CF-aware surface/vertical helper; fix altitude-coordinate surface pick (A3: B9,B18)`

### Task 15: B8 + rasterization — spatial primitive correctness

**Files:**
- Modify: `davinci_monet/plots/renderers/spatial/base.py:198-217` (orientation + rasterized), `davinci_monet/plots/renderers/spatial/overlay.py:193`, `davinci_monet/plots/renderers/curtain.py:278,359`, `davinci_monet/plots/renderers/lma_density.py:244`, `davinci_monet/plots/renderers/scatter.py:154-178`, `davinci_monet/plots/renderers/scorecard.py:178`, `davinci_monet/plots/renderers/vertical_profile.py:88`
- Test: `davinci_monet/tests/unit/plots/test_spatial_single_source.py` (square-grid case), `davinci_monet/tests/unit/plots/test_rasterization.py` (create)

- [ ] **Step 1: Failing test (B8)** — render a **3×3** (square) grid via `type: spatial` where `field[lat=0, lon=2] = 9.0` uniquely; pull the QuadMesh array back and assert 9.0 sits at the (row=lat-index 0, col=lon-index 2) position. Run: FAIL (the `data.shape[0] == len(lons)` predicate transposes a correct square array).
- [ ] **Step 2: Fix orientation by names, not lengths** — `draw_spatial_field` gains the DataArray's dim order (pass `dims: tuple[str, str]` from callers): transpose only when `dims[0]` is the lon-like dim. Where only raw arrays exist, thread the flag from the caller that still has the DataArray.
- [ ] **Step 3: Failing rasterization test** — render each affected plot type to PDF; walk `fig` artists asserting every `QuadMesh`/dense `PathCollection`/`contourf` collection has `get_rasterized() is True`, and (guard) that no `contour` LINE collection is rasterized. Run: FAIL for spatial/overlay/curtain/lma/scatter/scorecard/vertical_profile.
- [ ] **Step 4: Fix** — `rasterized=True` inside `draw_spatial_field` (pcolormesh + scatter branches) and at the six renderer-local sites listed above. NOT on any `ax.contour(...)` line sets.
- [ ] **Step 5: Add the missing paired-grid mark test (test-gap from FABLE_REVIEW §5.4):** `spatial_bias` on a GRID pair through the pipeline → figure contains a `QuadMesh` (not `PathCollection`).
- [ ] **Step 6: Full gate. Commit** — `fix(plots): dim-name orientation for square grids; enforce rasterized dense layers (B8)`

### Task 16: Documentation integrity sweep (find/replace corruption + stale facts)

**Files:**
- Modify: `CLAUDE.md`, `davinci_monet/core/protocols.py:35,51`, `davinci_monet/datasets/generic.py:3,122`, `davinci_monet/datasets/cmaq.py:4,29`, `davinci_monet/datasets/ufs.py:28`, `davinci_monet/datasets/wrfchem.py:33`, `davinci_monet/plots/renderers/diurnal.py:4`, `davinci_monet/plots/renderers/timeseries.py:4`, `davinci_monet/plots/renderers/spatial/base.py:42`, `davinci_monet/config/schema.py:24`, `davinci_monet/pipeline/display.py:347`, `davinci_monet/cli/app.py:82`
- Test: `davinci_monet/tests/unit/test_doc_integrity.py` (create)

- [ ] **Step 1: The functional fix first (L1), TDD** — unit test: feed a fake `/proc/cpuinfo` text containing `model name : Apple M2` through the parsing helper (refactor the 3-line loop into a testable `_cpu_name_from_cpuinfo(text: str) -> str | None` used by both copies); assert it returns "Apple M2". Run: FAIL (`"dataset name"`). Fix both `display.py:347` and `cli/app.py:82`.
- [ ] **Step 2: Docstring sweep** — restore "model output" (etc.) at the 10 code sites listed above (`dataset output` → `model output`; "Base dataset with strict validation settings" → "Base model with strict validation settings"; "match datasets with dataset output" → "match observations with model output").
- [ ] **Step 3: CLAUDE.md repair** (surgical edits, keep structure): `summary.dataset:` → `summary.model:` (and "cheapest vision dataset" → "cheapest vision model"); "evaluating atmospheric chemistry and air quality datasets against datasets" → "…model output against observations"; "Cross-Dataset Handoff" → "Cross-Model Handoff" (and "any dataset can pick up context" → "any model"); "Data Dataset (xarray-only)" → "Data Shapes (xarray-only)" with first row `Grid (model): …`; pipeline stage list → the actual 10-stage order from `factory.py`; "1,262 tests" → "1,65x tests (see pytest)"; architecture tree → add `analysis/`, `geography/`, `inspection/`, `assets/`, `io/download/`, single `datasets/` entry.
- [ ] **Step 4: Regression guard** — `test_doc_integrity.py`: grep the package for the corruption markers: assert zero matches for `"dataset output"`, `"dataset name"`, `"datasets against datasets"` in `davinci_monet/**/*.py` + `CLAUDE.md` (read files with pathlib; no subprocess).
- [ ] **Step 5: Full gate. Commit** — `docs: repair model->dataset find/replace corruption; sync CLAUDE.md facts (L1, §6)`

---

## Wave 3 — numerics & remaining mediums (parallelizable, file-disjoint)

### Task 17: B19/B20/B21 + L11/L19 — stats & EOF numerics

**Files:**
- Modify: `davinci_monet/analysis/eof.py:89-101,214,248-260`, `davinci_monet/stats/calculator.py:275,342-357,412-415,537`, `davinci_monet/analysis/gridded_reductions.py:39-42`
- Test: `davinci_monet/tests/unit/analysis/test_eof.py`, `davinci_monet/tests/unit/stats/test_metrics_calculator_output.py`, `davinci_monet/tests/unit/analysis/test_gridded_reductions.py`

(L11 lives here, not in Task 18, so `calculator.py` has exactly one Wave-3 owner.)

- [ ] **Step 1: Failing tests** — (a) N below `min_samples`: `calculate(...)` with 2 valid pairs and `min_samples=3` → all metrics NaN **except** `N == 2` (true count survives); (b) `quick_stats` with the same input agrees with `calculate` metric-for-metric; (c) EOF North error: strongly autocorrelated synthetic PC → `explained_variance_error <= explained_variance` (clamped) and `n_eff >= 2`; (d) EOF memory: monkeypatch-free — assert the patterns step never materializes a `(time, space, mode)` array by checking peak shape via the implementation (test the new `_patterns` helper directly: `patterns = anom.T @ pc / nt` equals old result on a small case, `np.allclose`).
- [ ] **Step 2: Fixes** — calculator: compute `N` before the floor mask and exempt it; route `quick_stats` through the same floor logic (share one `_apply_min_samples(metrics, n, floor)`); eof: `n_eff = max(n * (1 - r1) / (1 + r1), 2.0)`, clamp error to the EV value; replace the broadcast pattern computation with `(anom_matrix.T @ pcs) / n_time` reshaped to the spatial dims.
- [ ] **Step 3 (L11):** failing test — grouped stats where the y groupby is missing one key present in x → must raise `StatisticsError` naming the key (today: silent positional misalignment). Fix `calculator.py:275`: iterate one groupby and `.get()` the other by key, raising on mismatch.
- [ ] **Step 4 (L19):** unify the two season definitions — `gridded_reductions.py:39-42` switches from calendar quarters to the meteorological seasons already used by `calculator.py:342-357` (DJF/MAM/JJA/SON via `time.dt.season`), labels `f"{year}-{season}"` with the December-rolls-forward convention; update `test_gridded_reductions.py` expectations accordingly and note the output-label change in the commit body.
- [ ] **Step 5: Full gate. Commit** — `fix(stats,analysis): honest N; bounded North errors; EOF memory; season unification (B19-B21, L11, L19)`

### Task 18: Low-severity batch (independent one-liners)

**Files:**
- Modify: `davinci_monet/plots/renderers/scorecard.py:109,178,355` (L3), `davinci_monet/plots/renderers/taylor.py:153-160` (L4), `davinci_monet/plots/renderers/spatial/overlay.py:154-186` (L5), `davinci_monet/datasets/satellite/tropomi.py:203-207` (L6), `davinci_monet/datasets/satellite/modis_l2_aod.py:176` (L7), `davinci_monet/pipeline/stages/pair.py:412` (L8), `davinci_monet/pairing/strategies/track.py:55` (L9), `davinci_monet/analysis/reductions.py:49-51` (L10), `davinci_monet/core/base.py:384-387` (L12), `davinci_monet/config/parser.py:46-105` (L13/L14), `davinci_monet/io/writers.py:58-99` (L15), `davinci_monet/logging/config.py:3,220,313` (L16), `davinci_monet/pairing/strategies/point.py:330` (L17), `davinci_monet/pairing/engine.py:366-371` (L23), `davinci_monet/cli/app.py:314` (L20)
- Test: colocated unit-test files per fix (one test per L-item; grouped in the nearest existing test module)

Each item: write the one failing test exactly as its FABLE_REVIEW row's "Defect" column describes, apply the row's "Fix" column, test passes. Non-obvious specifics:
- [ ] **L3:** pass `center` into `render_from_dataframe`; when set, `norm = matplotlib.colors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)` on the imshow.
- [ ] **L8:** keep the "1h" default (behavior change is out of scope) but after gating, `logger.warning("time tolerance %s masked %d of %d target times", tol, n_masked, n_total)` when `n_masked > 0`.
- [ ] **L13:** `load_yaml`: if `isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.endswith((".yaml", ".yml")))` and the path does not exist → raise `ConfigurationError(f"Configuration file not found: {source}")` — never YAML-parse a path-looking string.
- [ ] **L14:** after `expand_env_vars`, walk string values: any residual `"${"` in `analysis.output_dir`/`log_dir`/`sources.*.files`/`filename` → raise `ConfigurationError` naming the variable; apply `Path(...).expanduser()` where `output_dir`/`log_dir` are consumed.
- [ ] **L15:** each writer: write to `path.with_suffix(path.suffix + ".tmp")`, then `os.replace(tmp, path)`.
- [ ] **L17:** `y_var = y_var.transpose(*x_ref.dims, missing_dims="raise")` before `.values`.
- [ ] **L23:** drop the bare `y_name` candidate in `engine._select_var`'s y list **unless** `y_name != x_key`; add a unit test where x and y share a variable name and a fake strategy emits only the bare name → expect `PairingError`, not a duplicated series.
- [ ] Full gate; one commit — `fix: low-severity batch from FABLE_REVIEW (L3-L10, L12-L17, L20, L23)`

### Task 19: Test hardening (FABLE_REVIEW §5.4)

**Files:**
- Modify: `davinci_monet/tests/unit/core/test_base.py:112-142`, `davinci_monet/tests/test_cli_e2e.py:198,216`, `davinci_monet/tests/integration/test_ceres_readers_pipeline.py:22`, `davinci_monet/tests/integration/test_merra2_reader_pipeline.py:19`, `davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py:24`, `davinci_monet/tests/integration/test_gridded_aod_product_pipeline.py:11`, `pyproject.toml` (markers)
- Test: this task IS tests.

- [ ] **Step 1:** `get_x`/`get_y`/`get_pair` tests assert values and attrs: `assert result.attrs["axis"] == "x"` and `xr.testing.assert_equal(result, paired["x_ozone"] ...)` (adapt to the real fixture names).
- [ ] **Step 2:** CLI e2e: reject-configs assert `result.exit_code != 0` (drop the `or "error" in stdout`); add `assert len(ERROR_CONFIGS) >= 15, "error_configs dir moved?"` at module import.
- [ ] **Step 3:** add `real_data` to `[tool.pytest.ini_options] markers`; switch the five real-data smokes from `pytest.mark.integration` to `pytest.mark.real_data`; add `pytest.mark.integration` to `test_gridded_aod_product_pipeline.py`.
- [ ] **Step 4:** `pytest -m integration -q` collects only true pipeline tests (spot-check list); full gate; commit — `test: honest markers; strengthen axis-accessor and CLI reject assertions (§5.4)`

---

## Wave 4 — wrap-up

### Task 20: Final verification + review handoff

- [ ] **Step 1:** Full gate from a clean shell. Expected: pytest green (count will exceed 1,657 — record the new number), mypy/black/isort clean.
- [ ] **Step 2:** `pip install -e ".[dev]"` fresh-resolve check + pandas version still 1.5.3.
- [ ] **Step 3:** Run one real end-to-end smoke if data is present locally (any `analyses/*/configs/*-gemini.yaml` the user has): confirm CSV + plots + manifest appear and the summary stage skips gracefully without a key. Skip this step if no local data — say so in the report.
- [ ] **Step 4:** Write `HANDOFF_FABLE_REMEDIATION.md` (repo root, **untracked** per CLAUDE.md): Context / Changes Made (per task, with commits) / Decisions & Rationale (the contract decisions from Tasks 3, 13, 14) / Open Questions / Suggested Next Steps (the P2/P3 seeds below). Request a fresh code review (superpowers:requesting-code-review) before telling the user the program is done. **Do not push; do not merge to main** — report to the user for verification.

---

## Follow-on plans (P2/P3 — NOT in scope here; each needs its own plan doc)

Decisions locked now so the follow-on plans don't relitigate:
1. **Plugin-true plot contracts (A4):** arity declared as a class attribute on each plotter (`arity: PlotArity`); `plots/contracts.py` derives its sets from the registry at call time; config validation consults only the registry. Kills the triple-edit requirement.
2. **Pydantic modernization (A5):** `AnalysisSpec` → `Annotated[Union[...], Field(discriminator="type")]`; delete `build_analysis_spec` + the ~13 redundant `mode="before"` validators; split `schema.py` into `schema/{analysis,sources,plots,stats,root}.py` (<500 lines each); `StatMetric`: type `stats.metrics: list[StatMetric]` and reconcile the member list with `stats/metrics.py` registry names.
3. **Dedup catalogue (A7):** `MonetioModelReader` base; `reader_utils.alias_coord` adoption across the 12 copy-paste readers; `intermediate_grid` single `_bin()`; shared `_get_system_info`; `load._file_list` → `resolve_file_list`; single mark-selection + lon-resort helpers in `plots/spatial/base.py`.
4. **Dead-code removal (A9):** delete `datasets/satellite/modis_l2.py`, the 8 never-raised exception classes, `write_pickle`, `curtain.py:176-190`, `surf_only` doc-or-implement.
5. **Typed stage contracts (A10 remainder):** TypedDict per stage output; `PipelineContext.sources/paired` typed; migrate the three `config_dict()` stages to the typed accessors; delete the duplicated `"1h"`/`"nearest"` literals in `pair.py:412-413` in favor of `schema.py:358-359` defaults.
6. **Performance (A12):** cKDTree in `_find_nearest_2d`; lazy renderer imports in `plots/__init__` (kills the 0.56 s config-parse matplotlib tax); `PlotSuiteStage` targeted plot-dict update instead of full re-validation.
