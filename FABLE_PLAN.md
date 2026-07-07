# FABLE_PLAN — Aerosol Mode-Space Tuning: MERRA-2/GEOS-IT AOD → MODIS/VIIRS

> Tracked planning document (user-requested exception to the untracked-handoff convention).
> Status: **approved design, pre-implementation**. Written 2026-07-07 after interactive brainstorming.
> Author: Claude Fable 5 (planning session with D. Fillmore). No code has been written.

---

## 1. Context

**Goal.** Tune MERRA-2 (later GEOS-IT) aerosol mass mixing ratios (MMRs) so that model AOD
better tracks MODIS/VIIRS AOD. The corrected MMRs feed a radiative transfer model.

**Method (user's concept, refined).** Compute EOFs of the analysis (model) AOD field; project
satellite AOD *innovations* (obs − model) onto that basis with a missing-data-aware regularized
least-squares (reduced-space optimal interpolation); wavelet-filter the resulting obs-space PC
time series (denoise + band-select + gap-bridge); reconstruct a correction field from the
filtered coefficients and the analysis EOFs; apply it as a **uniform per-column scale factor to
all aerosol species MMRs** (valid because AOD is linear in MMR under external mixing with
unchanged RH — speciation and vertical shape are untouched, total AOD scales exactly by `r`).

**Scope decisions (locked during brainstorming):**

| Decision | Choice | Rationale |
|---|---|---|
| Projection under missing data | Ridge/OI innovation projection (reduced-space OI, Kaplan et al. 1998) | EOFs are not orthogonal on the observed subdomain; naive inner products alias the seasonal mask into the PCs. Innovation projection makes "mode unobserved → zero correction" the graceful default. Multi-sensor blending falls out of the same solve. |
| Pipeline integration | **(A)** chained derived analyses in the existing `analyses:` block | Follows repo rule "all analyses through DAVINCI pipelines"; every intermediate is a pseudo-source → QA plots free; pipeline-level integration tests possible. |
| Working space | **log-AOD** (`ln(AOD + ε)`) | `r = exp(Δ)` positive by construction (no clamp hacks); AOD ~ lognormal → more Gaussian anomalies (better EOFs, better AR(1) significance); variance homogenized (dust/smoke maxima don't monopolize leading modes). |
| Correction structure | `r = r_clim(month, x) × r_anom(t, x)` | Separates systematic (seasonal-mean) bias from variability transfer; both terms default to "no correction" where the mask never sees. |
| Mode-space grid | Everything at **1°** (the MODIS L3 grid) | Matches obs grid for free; SVD shrinks to ~11k × 65k; `r` is smooth by construction (truncated basis), so 1° → native interpolation loses nothing. |
| Analysis product | **MERRA-2 first**; GEOS-IT is a near-clone follow-on | `merra2` reader exists (`datasets/merra2.py`); both systems assimilate AOD; GEOS-IT needs reader work only. |
| Obs | **MODIS Aqua daily L3 (`MYD08_D3`) first**, but the projection accepts a *list* of obs sources from day one | Multi-sensor blending is just a sum of per-sensor terms in the K×K solve; Terra (`MOD08_D3`) and VIIRS (`AERDB_D3`) drop in later as more innovation sources. |
| Wavelet role | (a) significance-gated denoise + (b) band selection + (c) temporal gap-bridging | Requires retaining complex CWT coefficients and adding inverse CWT — the current `wavelet` analysis keeps only `|W|²`. |
| Domain / period | Global; **decades-long daily** EOF training | Stable modes; seasonal masks need full annual cycles. |
| Cadence | Daily correction, applied to 3-hourly `inst3_3d_aer_Nv` via log-linear time interpolation | Once-daily sun-synchronous obs cannot constrain the diurnal cycle; MERRA-2's diurnal shape is accepted by design. |

---

## 2. Method — complete mathematical specification

### 2.1 Notation & preprocessing

- Grid cells `x` on the 1° L3 grid; area weight `w(x) = cos(lat)` (the same metric the EOF SVD
  uses via its `sqrt(cos lat)` field weighting, `analysis/eof.py:45`).
- Model AOD `A_m(t, x)`: MERRA-2 `TOTEXTTAU` from hourly `tavg1_2d_aer_Nx`, **sampled at the
  Aqua overpass** (~13:30 local solar time; pick the hour index nearest `UTC = 13.5 − lon/15`
  per longitude column), then area-weight coarsened 0.5°×0.625° → 1°. Overpass sampling is used
  for **both** basis training and innovation so model and obs sampling are identical
  (single-sourced helper — see §4.7).
- Log transform: `y = ln(A + ε)`, `ε = 0.01` (config `log_epsilon`). Applied identically to
  model and obs.
- Obs `y_o^s(t, x)` for sensor `s`, defined on the observed set `Ω_s(t)` (the day's valid L3
  cells, QA-filtered). Time alignment is by calendar day.

### 2.2 Basis (EOF training)

Anomalies about the monthly climatology of the training period (`remove_seasonal_cycle: true`):
`y'_m(t,x) = y_m(t,x) − C(month(t), x)`. Weighted SVD (existing `_svd_decompose`,
`analysis/eof.py:122`) yields:

- unit-variance, mutually uncorrelated PCs `p_k(t)` (unrotated), and
- regression patterns `E_k(x)` in log-AOD units (`_patterns_from_pc`, `analysis/eof.py:164`),
  so that `y'_m ≈ Σ_k p_k(t) E_k(x)`.

`n_modes` K ≈ 50 to start; truncation chosen from the scree/North's-rule output.
**Constraint: `rotation: none`** for any basis used by projection (varimax-rotated PCs are not
uncorrelated, so the identity prior in §2.4 would be wrong; validated with a clear config error).

### 2.3 Innovation and climatological bias term

Per sensor, day, and observed cell:

```
d_s(t, x) = y_o^s(t, x) − y_m(t, x),   x ∈ Ω_s(t)        # log(obs/model)
```

Split into systematic + anomaly parts, `d_s = b(month, x) + d'_s`:

- `b̂(m, x)` = mean of `d` over all days of month `m` (all training-overlap years) at cells with
  at least `N_min` samples (default: ≥ 20 % of that month's days across years); cells below
  `N_min` get `b̂ = 0` (**no correction**). Then a mask-aware 3×3 boxcar smoothing (2 iterations),
  and cap `|b̂| ≤ ln(r_max)`.
- The projection (§2.4) acts on the anomaly innovation `d' = d − b̂`.

Note the basis climatology `C` cancels in `d` (both obs and model would subtract the same `C`),
so the innovation needs no climatology handling beyond `b̂`.

### 2.4 Per-day projection (reduced-space OI / ridge)

For each day `t`, solve the small K×K system over observed cells only:

```
â(t) = argmin_a  Σ_s σ_s⁻² Σ_{x∈Ω_s(t)} w(x) [ d'_s(t,x) − Σ_k a_k E_k(x) ]²  +  λ ‖a‖²

⇔  ( H(t) + λ I ) â(t) = g(t),
    H(t) = Σ_s σ_s⁻² E_sᵀ W_s(t) E_s          (K×K)
    g(t) = Σ_s σ_s⁻² E_sᵀ W_s(t) d'_s(t)      (K)
```

with `W_s(t) = diag(w(x) · 1[x ∈ Ω_s(t)])` and `E_s` the patterns on sensor s's grid (all 1°
here, so `E_s = E`). Since PCs are unit-variance and uncorrelated, `λ = 1` **is** the Bayesian
MAP with prior `a ~ N(0, I)` and per-cell obs error `σ_s²/w(x)`; `λ` and `σ_s` (default 0.15 in
log-AOD units, roughly the MODIS DT/DB uncertainty envelope) are config-tunable. Multi-sensor
blending is precision-weighted automatically; each sensor contributes only where it has data.

Properties that answer the seasonal-mask question:

- Cross-mode leakage from mask-broken orthogonality is handled exactly (full `H`, not its diagonal).
- A mode invisible under that day's mask has `H_kk ≈ 0` → `â_k → 0` → **zero correction**
  (keep the analysis), never extrapolation.
- Cost: K=50 → a 50×50 solve per day; decades ≈ 11 k days → seconds, batched with einsum.

### 2.5 Observability diagnostics (stored with the PCs)

- `resolution(t, mode)` = `diag[(H+λI)⁻¹ H]` ∈ [0, 1) — recovered fraction of a unit true amplitude.
- `coverage(t, mode)` = `Σ_{x∈Ω} w E_k² / Σ_x w E_k²` — mask coverage of mode k's variance.
- `n_obs(t, sensor)` — observed-cell counts.
- Flag `(t, k)` when `resolution < ρ_min` (default 0.3, config `min_resolution`); flagged entries
  are treated as **gaps** by the wavelet filter rather than trusted as shrunk-to-zero values.

### 2.6 Wavelet filter (per mode k)

1. Set flagged entries of `â_k` to NaN; fill by linear interpolation (record synthesized fraction).
2. Remove linear trend (store it — obs−model drift is *signal* here, re-added at step 6).
3. Morlet CWT (shared core, §4.4) with the mode's AR(1) `α` estimated from the filled series.
4. Zero coefficients failing the pointwise AR(1) significance test (`keep_significant: true`).
5. Zero coefficients outside the configured period band `[T_min, T_max]` (`band:`); COI is kept
   but the per-mode fraction of retained power inside the COI is recorded.
6. Inverse CWT (`pycwt.icwt`, Torrence & Compo 1998 eq. 11) → `ã_k(t)`; add trend (and mean) back.
7. Record per mode: retained variance fraction, and the unfiltered icwt round-trip relative error
   (Morlet reconstruction is approximate, ~few %); warn above 5 %.

### 2.7 Reconstruction, scale factor, application

```
Δ(t, x)   = b̂(month(t), x) + Σ_k ã_k(t) E_k(x)          # log-ratio correction, 1° grid
r_1°      = clip( exp(Δ), r_min, r_max )                  # default [0.2, 5]; clip counts logged
r_native  = bilinear(r_1° → 0.5°×0.625°)                  # with periodic-longitude padding
r_3hr     = exp( linear-in-time interp of Δ_native )      # day centers at 12 UTC; edges held
q̃_i      = r_3hr · q_i    for every aerosol species i    # DU×5, SS×5, SO4, BC×2, OC×2
```

Under external mixing with unchanged RH, per-species AOD is linear in `q_i`, so total column AOD
scales by exactly `r`. Speciation fractions and vertical profile shape are preserved. Gas-phase
tracers in the collection (SO2, DMS, MSA) are **not** scaled.

### 2.8 Success criteria

- **OSSE (§7.1):** mode-recovery correlation `corr(ã_k, p_k^true) > 0.9` in retained bands under
  realistic seasonal masks; `|Δ| < tol` in never-observed regions; reconstructed-field RMSE vs
  withheld truth quantified per season and latitude band.
- **Real data:** corrected AOD vs MODIS improves RMSE / |NMB| / R over baseline MERRA-2 vs MODIS,
  globally and per season/region, with no degradation where obs are sparse (r ≈ 1 there by
  construction). *Honesty note:* v1 metrics are **in-sample** (the correction is fit to the same
  obs it is scored against) — methodological validity comes from the OSSE. Clean out-of-sample
  scoring (basis + `b̂` from window A applied to held-out window B) needs a small follow-on:
  `basis:`/`coefficients:` accepting a saved artifact path instead of an in-run analysis key
  (deferred; noted in P8).

---

## 3. Architecture & data flow

```
sources:   merra2 (TOTEXTTAU, hourly Nx)          modis_aqua (MYD08_D3 AOD, 1°)
                │  overpass-sample → 1° → log            │  log, QA mask
analyses:  1. eof            basis {eofs, pc, explained_variance, climatology, ...}
           2. eof_projection â(t,mode) + resolution/coverage diagnostics
           3. wavelet_filter ã(t,mode) + retained-variance / recon-error diags
           4. aod_scaling    r(t,lat,lon) [1° + native] + aod_target diagnostic
           5. mmr_writer     corrected inst3_3d_aer_Nv files (original layout) + manifest
plots:     eof_pattern, eof_scree, pc timeseries, wavelet_scalogram (QA on 2–3),
           spatial maps of r / b̂ / aod_target
follow-up: second, standard eval pipeline: {corrected AOD artifact, original} vs MODIS
```

Steps 1–4 are ordinary derived analyses (pseudo-sources; plottable; not pairable). Step 5 writes
model-format files as a side effect and returns a small manifest dataset (files written, clip
stats, mean r) as its pseudo-source.

### 3.1 Reference YAML

```yaml
analysis:
  start_time: "1995-01-01"        # training window
  end_time:   "2024-12-31"
  output_dir: ${TUNE}/output
  log_dir:    ${TUNE}/logs

sources:
  merra2:
    type: merra2
    files: ${DATA}/MERRA2/tavg1_2d_aer_Nx/*.nc4
    variables: { TOTEXTTAU: {} }
  modis_aqua:
    type: modis_viirs
    product: MYD08_D3
    files: ${DATA}/MODIS/MYD08_D3/*.hdf
    variables: { AOD_550_Dark_Target_Deep_Blue_Combined_Mean: {} }

analyses:
  aod_basis:
    type: eof
    source: merra2
    variable: TOTEXTTAU
    n_modes: 50
    remove_seasonal_cycle: true
    rotation: none                 # REQUIRED for projection use
    log_space: true                # NEW
    log_epsilon: 0.01              # NEW
    target_grid: 1.0               # NEW: coarsen to 1° before decomposing
    sample_local_time: 13.5        # NEW: Aqua overpass sampling

  obs_pcs:
    type: eof_projection           # NEW analysis type
    basis: aod_basis
    obs:
      - source: modis_aqua
        variable: AOD_550_Dark_Target_Deep_Blue_Combined_Mean
        error: 0.15                # σ_s, log-AOD units
    ridge: 1.0
    clim_bias: true                # estimate + subtract b̂(month, x)
    clim_min_samples: 0.2          # fraction of month-days required per cell
    min_resolution: 0.3            # gap-flag threshold

  filtered_pcs:
    type: wavelet_filter           # NEW analysis type
    source: obs_pcs
    variable: pc
    keep_significant: true
    significance_level: 0.95
    band: { min: 4, max: null, units: days }
    omega0: 6.0

  scaling:
    type: aod_scaling              # NEW analysis type
    basis: aod_basis
    coefficients: filtered_pcs
    r_bounds: [0.2, 5.0]
    native_grid_from: merra2       # target grid for r_native

  corrected:
    type: mmr_writer               # NEW analysis type (writes files)
    scaling: scaling
    files: ${DATA}/MERRA2/inst3_3d_aer_Nv/*.nc4
    species: null                  # default: 15 GOCART aerosol tracers
    output_dir: ${TUNE}/corrected
    time_interp: log_linear

plots:
  basis_maps:  { type: eof_pattern, source: aod_basis, variable: eofs }
  basis_scree: { type: eof_scree,   source: aod_basis, variable: explained_variance }
  pc1:         { type: timeseries,  source: filtered_pcs, variable: pc, mode: 1 }
  pc1_scal:    { type: wavelet_scalogram, source: filtered_pcs, variable: power, mode: 1 }
  r_map:       { type: spatial, source: scaling, variable: r }
```

(The `plots:` wiring for per-mode scalograms of `wavelet_filter` output and `mode:`-selected
spatial maps reuses existing renderers; any small renderer-side gaps surface in P3/P4 tests.)

---

## 4. New components (file by file)

All new modules stay < 500 lines (project goal); pure math lives in plain functions for unit
testability, analysis classes are thin adapters — mirroring `eof.py`'s structure.

### 4.1 `davinci_monet/analysis/projection.py` — `eof_projection`

- `@analysis_registry.register("eof_projection")`, `output_geometry = GRID` (mixed-shape derived
  source like EOF: `pc(time, mode)`, `resolution(time, mode)`, `coverage(time, mode)`,
  `clim_bias(month, lat, lon)`, `n_obs(time, sensor)`).
- Pure functions: `innovation(y_obs, y_model) -> masked DataArray`,
  `clim_bias(d, min_frac) -> (b̂, counts)`, `solve_day(E_w, d_w, mask, sigma, ridge) ->
  (a, resolution)` (vectorized over batched days), `project(...)` orchestrator.
- Consumes: basis pseudo-source (patterns + metadata incl. `log_epsilon`, `sample_local_time`,
  training grid) and N obs sources. Applies the **identical** log/overpass/regrid preprocessing
  to model and obs via the shared helpers (§4.7) — single-sourced, never duplicated.
- Validates: basis `rotation == "none"`; obs sources exist and are GRID geometry on (or
  regriddable to) the basis grid; time overlap non-empty.
- **Time-window semantics** (applies to the whole chain): `eof` trains on the full `analysis:`
  window (model-only, e.g. 1995–2024); `eof_projection` and everything downstream operate on the
  **intersection** of that window with obs availability (e.g. Aqua ⇒ 2002+). Days with no obs at
  all inside the intersection still produce `â` (shrunk to 0 ⇒ `r → r_clim`).

### 4.2 `davinci_monet/analysis/wavelet_filter.py` — `wavelet_filter`

- `@analysis_registry.register("wavelet_filter")`, output: `pc(time, mode)` (filtered) +
  `power(time, period, mode)`-compatible QA variables (per selected mode), `retained_variance
  (mode)`, `recon_error(mode)`, `synth_fraction(mode)`.
- Loops modes internally (`variable: pc` without `mode:` = all modes; with `mode: N` = one).
- Steps exactly as §2.6; consumes `resolution` from the projection source to identify gaps
  (attr-linked; falls back to no-gaps when absent so it also works on plain series).

### 4.3 `davinci_monet/analysis/scaling.py` — `aod_scaling` (+ writer in 4.5)

- `@analysis_registry.register("aod_scaling")`, output: `r(time, lat, lon)` on the 1° grid,
  `r_native(time, lat, lon)` on the model grid, `delta_log(time, lat, lon)`,
  `aod_target(time, lat, lon)` (= model AOD × r, for evaluation), `clip_fraction(time)`.
- Pure functions: `reconstruct_delta(E, a, b̂, month_index)`, `to_ratio(delta, bounds)`,
  `interp_to_native(r, native_lat, native_lon)` with periodic-longitude padding.
- Saves `r` + `aod_target` as NetCDF artifacts via the existing artifact helper
  (`analysis/artifacts.py`) so the follow-up evaluation pipeline can read them as a plain
  `gridded` source.

### 4.4 `davinci_monet/analysis/cwt_core.py` — shared CWT (refactor)

- Extract from `WaveletAnalysis.analyze()` (`analysis/wavelet.py:36-111`) a function
  `cwt_decompose(y, dt, spec-ish params) -> CWTResult` returning **complex** `wave(scale, time)`,
  `scales`, `periods`, `coi`, `alpha`, significance arrays; plus `cwt_reconstruct(wave_filtered,
  scales, dt, dj, mother) -> y` wrapping `pycwt.icwt`.
- `WaveletAnalysis` becomes a thin consumer producing its exact current outputs (its tests are
  the regression guard); `wavelet_filter` is the second consumer. One CWT implementation.

### 4.5 `davinci_monet/analysis/mmr_writer.py` — `mmr_writer`

- `@analysis_registry.register("mmr_writer")`. Takes the file glob **directly** (not through
  `sources:`) because decades of 3-D MMR must be streamed file-by-file, never loaded into the
  pipeline context.
- Per input file (8 timesteps for `inst3`): open, time-interpolate `delta_log_native` to the
  file's timestamps (linear in Δ, i.e. log-linear in r; edges held), multiply the species list
  (default: `DU001–005, SS001–005, SO4, BCPHOBIC, BCPHILIC, OCPHOBIC, OCPHILIC` — confirm the
  exact tracer list against the files at implementation; config `species:` overrides; a clear
  error lists available variables when a requested species is missing), write to `output_dir`
  preserving filename, variable names/dtypes/attrs, dimension layout, and compression; append
  provenance global attrs (davinci version, config hash, basis/obs identifiers, r statistics).
- Returns a manifest dataset (files written, per-file clip counts, mean/min/max r) as its
  pseudo-source; also records paths into the run manifest stage.

### 4.6 Config schema (`config/schema.py`)

- New spec classes appended to the union at `config/schema.py:753` and dispatcher
  `build_analysis_spec` (`:756`): `EOFProjectionSpec` (with nested `ObsEntry {source, variable,
  error}` list), `WaveletFilterSpec`, `AODScalingSpec`, `MMRWriterSpec` — all `StrictSchema`,
  `type: Literal[...]` discriminators, defaults as in §3.1.
- `EOFSpec` (`config/schema.py:675`) gains `log_space: bool = False`, `log_epsilon: float =
  0.01`, `target_grid: float | None = None`, `sample_local_time: float | None = None`.
- **Dependency resolution generalization**: `pipeline/stages/analyses.py` toposort currently
  reads the single `spec.source` (`analyses.py:35`, source lookup `:76`). Add one function
  `analysis_deps(spec) -> list[str]` mapping each spec type to its analysis-key references
  (`eof/wavelet/gridded_analysis → [source]`, `eof_projection → [basis]`, `wavelet_filter →
  [source]`, `aod_scaling → [basis, coefficients]`, `mmr_writer → [scaling]`); obs entries
  reference **raw** sources only (validated present, not toposorted). Cycle/unknown-ref errors
  keep their current clear messages.

### 4.7 Shared preprocessing utilities

- `davinci_monet/util/regrid.py`: `coarsen_to_degrees(da, res)` — area-weighted bin-average
  (`groupby_bins` on lat/lon with cos-lat weights; handles the non-commensurate 0.625° → 1°
  ratio); pass-through + alignment check when already on target grid.
- `davinci_monet/util/local_time.py`: `sample_local_solar_time(ds, hour)` — per-longitude
  nearest-hour selection from an hourly dataset, returning a daily field stamped at 12 UTC.
- `davinci_monet/util/logspace.py` (or fold into regrid module if tiny): `to_log(da, eps)` /
  `from_log(...)`. All three used by `eof` **and** `eof_projection` so model/obs preprocessing
  can never diverge.

### 4.8 Satellite catalog entries

- `datasets/satellite/catalog/data/modis_viirs_atmosphere.yaml` gains `MOD08_D3` and `MYD08_D3`
  (daily L3, same `.AYYYYDDD.` date token as the existing `_M3` entries at lines 2/18, 1° grid,
  AOD SDS names verified against C6.1 files at implementation) and, later, `AERDB_D3_VIIRS_SNPP`.
  Reader machinery (`modis_viirs.py`) is catalog-driven and should need no code changes for D3.

### 4.9 EOF output extension (`analysis/eof.py`)

- Store what reconstruction/projection need, currently discarded at `eof.py:212-218`:
  `climatology(month, lat, lon)` (when `remove_seasonal_cycle`), `time_mean(lat, lon)`,
  `std(lat, lon)` (when `standardize`), plus attrs: `log_space`, `log_epsilon`,
  `sample_local_time`, `target_grid`, training period. Existing outputs unchanged.

---

## 5. Changes to existing code — summary table

| File | Change | Size |
|---|---|---|
| `config/schema.py` | 4 new spec classes + `ObsEntry`; extend `EOFSpec`; extend union + dispatcher | ~120 lines |
| `pipeline/stages/analyses.py` | `analysis_deps()` generalization of toposort + validation | ~30 lines |
| `analysis/eof.py` | `log_space`/`target_grid`/`sample_local_time` preprocessing (via §4.7 helpers); store climatology/mean/std + attrs | ~50 lines |
| `analysis/wavelet.py` | refactor guts into `cwt_core.py`; behavior-identical outputs | net ~0 |
| `catalog/.../modis_viirs_atmosphere.yaml` | + `MOD08_D3`, `MYD08_D3` entries | ~30 lines |
| `analysis/__init__.py`, docs, CLAUDE.md | register new modules; document new analyses | small |

New modules: `projection.py`, `wavelet_filter.py`, `scaling.py`, `mmr_writer.py`, `cwt_core.py`,
`util/regrid.py`, `util/local_time.py` (each < 500 lines).

---

## 6. Performance & memory

- **Training matrix**: 30 yr × 365 d ≈ 11 k times × 64.8 k cells (1°). float32 ≈ 2.9 GB +
  LAPACK workspace — fine on Casper/Derecho large-mem. Escape hatches (config): `target_grid: 2.0`,
  or a training-time stride. Load as float32; SVD in float32 is adequate for K ≤ 100.
- **Projection**: batched einsum over days; K=50 → negligible (seconds for decades).
- **CWT**: 50 modes × 11 k points, FFT-based → seconds.
- **Writer**: I/O-bound streaming, one file at a time; parallelizable later by date-range subsets
  of the config (YAGNI now).
- Run environment note: on this machine the conda env is **`davinci-monet`** (miniforge3), not
  `davinci` as in CLAUDE.md; `HDF5_USE_FILE_LOCKING=FALSE` for runs.

---

## 7. Validation & testing design (per repo testing rules — this section is the test design to approve)

### 7.1 OSSE self-test (the centerpiece; real data, not CI)

Config + script under `analyses/aerosol-tuning/`: take MERRA-2 itself as "truth", hide it behind
**real MODIS daily masks**, run the full chain (project → filter → reconstruct), compare against
the withheld complete field. Measures mask-induced error per mode/season/latitude **before real
obs enter**, and shakes down the not-well-tested EOF/wavelet code on real-scale data. Deliver:
recovery-statistics CSV + maps/scalograms via standard plots.

### 7.2 Pipeline integration tests (CI; synthetic NetCDF via `tmp_path`; each through `PipelineRunner.run_from_config()`)

- **T1 `test_tuning_chain_known_modes`**: synthetic "model" = 3 prescribed orthogonal spatial
  patterns × sinusoids (10 / 30 / 90 d) + noise, 2 yr daily on a coarse global grid; synthetic
  "obs" = truth × known ratio field + noise + **seasonal mask** (poleward-of-50° winter gaps +
  40 % random clouds). Full `eof → eof_projection → wavelet_filter → aod_scaling` config.
  Assert: `corr(ã_k, truth) > 0.9` for retained bands; `|Δ| < tol` where never observed;
  RMSE(aod_target, obs-truth) < RMSE(model, obs-truth).
- **T2 `test_mmr_writer_roundtrip`**: tiny synthetic `inst3`-like file (2 species, 8 timesteps)
  + prescribed `r`. Assert: species scaled exactly, gas tracers untouched, layout/attrs/dtypes
  preserved, provenance attrs present, manifest correct.
- **T3 `test_multi_sensor_blend`**: two synthetic sensors with complementary masks and different
  errors → blended `â` beats either alone (RMSE), and precision weighting is honored.
- Runtime target for the three: < ~60 s total.

### 7.3 Unit tests (pure functions)

- Solver: complete-mask + λ→0 recovers exact coefficients of a constructed field; masked solve
  bounded shrinkage; `resolution` diagonal ∈ [0,1) and → 1 as mask → complete; two-sensor
  precision weighting; identity-prior equivalence (λ=1 vs explicit Bayesian form).
- `clim_bias`: min-sample gating → 0; smoothing mask-aware; cap applied.
- Wavelet filter: in-band sinusoid survives, out-of-band removed, trend preserved, gap-fill
  fraction recorded; icwt round-trip error < 5 % on white+red noise.
- `coarsen_to_degrees`: global cos-lat mean conserved to tolerance; non-commensurate ratios.
- `sample_local_solar_time`: known hour selection at test longitudes; dateline behavior.
- `interp_to_native`: periodic-longitude padding correct at the seam.
- Schema: new specs parse/validate; rotation≠none basis rejected; dep resolution (chain order,
  unknown ref, cycle).

### 7.4 Follow-up evaluation pipeline (real data)

Second standard config: load the saved `aod_target` artifact and original MERRA-2 AOD as two
`gridded` sources vs `modis_aqua`; standard pairs/stats/plots produce the headline
before/after table (N, MB, RMSE, R, NMB, NME, IOA) globally and per season. This reuses the
entire existing evaluation machinery untouched — and doubles as the acceptance gate (§2.8).

---

## 8. Implementation phases

| Phase | Deliverable | Gate |
|---|---|---|
| **P0** | `util/regrid.py`, `util/local_time.py`, log helpers; catalog `MOD08_D3`/`MYD08_D3`; unit tests | pytest, mypy, black/isort green (env `davinci-monet`) |
| **P1** | `eof` extensions (log_space, target_grid, sample_local_time, stored climatology/mean/std) | existing EOF tests still green + new unit tests |
| **P2** | `cwt_core.py` refactor; `wavelet` regression-identical | existing wavelet tests green |
| **P3** | `eof_projection` (+ schema, dep-resolution generalization) | unit + T3 |
| **P4** | `wavelet_filter` | unit tests |
| **P5** | `aod_scaling` (+ artifacts) | T1 end-to-end green |
| **P6** | `mmr_writer` | T2 |
| **P7** | OSSE self-test on real MERRA-2 + MODIS masks (`analyses/aerosol-tuning/`) | recovery stats reviewed by user |
| **P8** | Real-data run + follow-up evaluation config (§7.4); tune K, band, σ, ρ_min; optional basis-artifact reload for out-of-sample scoring (§2.8) | before/after stats reviewed |
| **P9** | GEOS-IT: reader (grid/collection audit first — lat-lon vs cubed-sphere distribution), then clone configs | same gates |

Each phase TDD (repo convention), no commits/pushes without explicit user approval.

---

## 9. Risks & open questions

1. **MODIS DT/DB regional biases** leak into `b̂` (it will faithfully "correct" toward biased
   retrievals, e.g. DB over bright surfaces). Mitigations later: per-region σ inflation,
   AERONET cross-check of `b̂` maps before trusting them.
2. **Basis stationarity over decades** — major eruptions (Pinatubo) distort covariance. Option:
   `exclude_periods:` on the EOF training window; decide after inspecting the scree/patterns.
3. **icwt fidelity** (~few % for Morlet) — measured and logged per mode; acceptable for a
   correction field, but verify in T1/OSSE.
4. **Daily → 3-hourly application** assumes MERRA-2's diurnal AOD shape — accepted by design
   (we correct daily-and-slower scales only).
5. **Overpass sampling approximation** — daily `r` stamped at 12 UTC though estimated at 13:30
   LST; consistent between training and innovation, so it cancels to first order.
6. **Exact product details to verify at implementation**: MYD08_D3 SDS name + QA variable
   choice; `inst3_3d_aer_Nv` tracer list; MODIS collection (C6.1); GEOS-IT collection grids.
7. **Memory of the training SVD** on smaller nodes — mitigations in §6.
8. **Pandora/renderer edge cases**: per-mode scalogram plots of `wavelet_filter` output may need
   a small renderer accommodation (surface in P4 tests, fix then).

---

## 10. Decisions & rationale (conversation distillate)

1. **Innovation projection, not raw-obs projection** — zero-observability limit = "keep the
   analysis", which is the only safe default under seasonally vanishing coverage.
2. **Ridge = Bayesian prior, λ=1 natural** because DAVINCI PCs are unit-variance — the identity
   prior is exact, not a hack; regularization strength still exposed for tuning.
3. **Log space** — positivity of `r` by construction; near-lognormal AOD; homogenized variance.
4. **Two-term correction (`r_clim × r_anom`)** — systematic vs variability separated; both
   degrade gracefully to 1 under missing data.
5. **1° mode space** — obs-grid match, tractable SVD, no information loss for a truncated-basis
   (smooth) correction; `r` interpolated to native grid at the end.
6. **Chained analyses (option A), not a bespoke CLI** — repo architecture rules; QA plots and
   pipeline integration tests come free; `mmr_writer` is the one file-writing analysis, taking
   its file glob directly to allow streaming.
7. **Multi-sensor from day one** — obs is a list; blending is additive in the OI solve; no merged
   obs product needed.
8. **Uniform per-column species scaling** — AOD linear in MMR under external mixing; RT model
   recomputes optics from scaled MMRs; speciation/vertical structure deliberately untouched.
9. **Wavelet = filter (significance gate + band + gap-bridge)**, requiring complex-coefficient
   retention + `pycwt.icwt`; trend is preserved as signal.
10. **MERRA-2 first, Aqua first**; GEOS-IT and Terra/VIIRS are drop-in extensions by design.

## 11. References

- Kaplan, A., et al. (1998): Analyses of global sea surface temperature 1856–1991 — reduced-space
  optimal interpolation of gappy obs onto complete-field EOFs (the method template).
- Torrence, C. & Compo, G. P. (1998): A practical guide to wavelet analysis — CWT, significance,
  reconstruction (eq. 11).
- North, G. R., et al. (1982): Sampling errors in the estimation of empirical orthogonal functions.
- Beckers, J.-M. & Rixen, M. (2003): DINEOF — EOF-based infilling (context; our fixed-basis ridge
  solve supersedes iteration).
- Existing in-repo spec: `docs/superpowers/specs/2026-06-17-eof-and-wavelet-analysis-design.md`
  (derived-analysis layer this plan builds on).
