# FABLE_PLAN — Aerosol Mode-Space Tuning: MERRA-2/GEOS-IT AOD → MODIS/VIIRS

> Tracked planning document (user-requested exception to the untracked-handoff convention).
> Status: **pre-flight revised, awaiting approval**. Original design written 2026-07-07;
> synthetic-first pre-flight revision completed 2026-07-10.
> Authors: Claude Fable 5 (original planning session with D. Fillmore); Codex (pre-flight review).
> No FABLE implementation code has been written.

---

## 0. Pre-flight disposition (2026-07-10)

**Development boundary.** Phases P0-P8 are strictly synthetic: no MERRA-2, GEOS-IT,
MODIS/VIIRS files, real retrieval masks, product downloads, or real-data-derived tuning choices.
Real readers/catalog entries and real-data experiments start only after the `SYNTHETIC_READY`
gate in §8.5 is approved. Generated NetCDF/Zarr and plots are untracked run artifacts.

**Blocking design corrections incorporated by this revision:**

1. `ln(A + epsilon)` does not imply `r = exp(delta)`. Section 2.7 now uses the exact shifted-log
   inverse and an explicit low-AOD policy before applying physical ratio bounds.
2. A global EOF can reconstruct into a cell that was never observed. Section 2.3 now defines a
   monthly spatial support field that is applied during reconstruction, making the promised
   no-observation/no-correction behavior explicit rather than assumed.
3. The live derived-analysis API accepts exactly one `source`, but projection/scaling require
   several named inputs. Section 4.1 defines one named-input contract used by schema validation,
   topological ordering, and execution, plus required/fatal-chain semantics.
4. Projection previously had no time-varying model field. A first-class `aod_preprocess` analysis
   (§4.2) now produces the one daily AOD/log-AOD pseudo-source consumed by both EOF training and
   innovation projection.
5. The live full, float64 SVD and monolithic eager artifact path do not scale to the proposed
   real problem. Truncated/chunk-aware EOF and artifact benchmark gates now precede real data.
6. Daily MODIS support is not catalog-only: timestamp cadence, canonical variable naming, and QA
   behavior require reader work. That work is explicitly deferred until after `SYNTHETIC_READY`.

**Repository baseline (not caused by this document change):** on `develop` at `2b62d7f`, in the
active `davinci` environment, pytest reported 1,749 passed, 10 skipped, and 5 failed because
`pycwt` is not installed; mypy reported 2 errors; Black flagged 1 file; isort passed. Restore or
explicitly disposition those baseline gates before implementation. The required environment name
throughout this plan is `davinci`.

---

## 1. Context

**Goal.** Tune MERRA-2 (later GEOS-IT) aerosol mass mixing ratios (MMRs) so that model AOD
better tracks MODIS/VIIRS AOD. The corrected MMRs feed a radiative transfer model.

**Method (user's concept, refined).** Compute EOFs of the analysis (model) AOD field; project
satellite AOD *innovations* (obs − model) onto that basis with a missing-data-aware regularized
least-squares (reduced-space optimal interpolation); wavelet-filter the resulting obs-space PC
time series (denoise + band-select + gap-bridge); reconstruct a correction field from the
filtered coefficients and the analysis EOFs; apply it as a **uniform per-column scale factor to
all configured aerosol species MMRs. Exact optical scaling is conditional on a fixed homogeneous
forward operator (species/size/composition/RH/meteorology unchanged); the synthetic optical oracle
tests that condition before any claim is transferred to a real radiative-transfer model.

**Scope decisions (original choices plus pre-flight corrections):**

| Decision | Choice | Rationale |
|---|---|---|
| Initial evidence | **Synthetic only through `SYNTHETIC_READY`** | Separates algorithm/software correctness from reader quirks and prevents tuning against real observations before the recovery limits are known. |
| Projection under missing data | Ridge/OI innovation projection (reduced-space OI, Kaplan et al. 1998) | EOFs are not orthogonal on the observed subdomain; naive inner products alias the seasonal mask into the PCs. Innovation projection makes "mode unobserved → zero correction" the graceful default. Multi-sensor blending falls out of the same solve. |
| Pipeline integration | **(A)** chained derived analyses in the existing `analyses:` block | Follows repo rule "all analyses through DAVINCI pipelines"; every intermediate is a pseudo-source → QA plots free; pipeline-level integration tests possible. |
| Working space | **shifted log-AOD** (`ln(AOD + ε)`) | AOD is closer to Gaussian and variance is homogenized, but the shift requires the exact inverse in §2.7; `exp(Δ)` alone is not the physical MMR scale factor. |
| Correction structure | `Δ = Δ_clim(month, x) + Δ_anom(t, x)`, then exact physical `r` | Separates systematic bias from variability transfer in transformed space; a stored monthly support field gates both terms where synthetic observations provide no evidence. |
| Mode-space grid | Everything at **1°** for the real target (coarser in CI) | Matches the MODIS L3 grid and bounds the state size. Treating the truncated correction as smooth enough for native-grid interpolation is a tested modeling assumption, not a no-loss theorem. |
| Analysis product | Synthetic MERRA-like first; **MERRA-2 first real product**; GEOS-IT follow-on | Core behavior is proven without real files. MERRA-2 then exercises the existing reader; GEOS-IT begins with a grid/collection audit, not an assumed clone. |
| Obs | Synthetic daily L3 sensors first; **MODIS Aqua (`MYD08_D3`) first real sensor**; projection accepts a list from day one | Complementary masks and precision weighting are tested with known truth before Terra/VIIRS or retrieval-specific QA is introduced. |
| Wavelet role | (a) significance-gated denoise + (b) band selection + (c) temporal gap-bridging | Requires retaining complex CWT coefficients and adding inverse CWT — the current `wavelet` analysis keeps only `|W|²`. |
| Domain / period | Small global synthetic cases first; later **decades-long daily** real EOF training | CI and synthetic OSSE establish correctness; truncated/chunked solver benchmarks gate the decades-long run. |
| Cadence | Daily correction, applied to 3-hourly `inst3_3d_aer_Nv` via log-linear time interpolation | Once-daily sun-synchronous obs cannot constrain the diurnal cycle; MERRA-2's diurnal shape is accepted by design. |

---

## 2. Method — complete mathematical specification

### 2.1 Notation & preprocessing

- Grid cells `x` on the 1° L3 grid; area weight `w(x) = cos(lat)` (the same metric the EOF SVD
  uses via its `sqrt(cos lat)` field weighting, `analysis/eof.py:45`).
- Model AOD `A_m(t, x)`: produced once by `aod_preprocess` (§4.2), then consumed unchanged by
  the EOF and projection analyses. For the real follow-on this is MERRA-2 `TOTEXTTAU` from hourly
  `tavg1_2d_aer_Nx`, sampled at ~13:30 local solar time (nearest `UTC = 13.5 - lon/15` per
  longitude column), area-weight coarsened 0.5°×0.625° → 1°, and stamped by calendar day.
  Synthetic sources reproduce the same cadence and longitude-dependent selection. The raw model
  source loads one adjacent UTC day on each side; the preprocessor clips its daily output back to
  the requested window so dateline columns and date-only boundaries cannot lose an edge day.
- Log transform: `y = ln(A + ε)`, `ε = 0.01` (config `log_epsilon`). Applied identically to
  model and obs **after** linear-space coarsening. Its inverse and the MMR ratio are defined in
  §2.7; no code may substitute `exp(Δ)` for that ratio.
- Obs `y_o^s(t, x)` for sensor `s`, defined on the observed set `Ω_s(t)` (the day's valid L3
  cells, QA-filtered). Time alignment is by calendar day.

### 2.2 Basis (EOF training)

Anomalies about the monthly climatology of the training period (`remove_seasonal_cycle: true`):
`y'_m(t,x) = y_m(t,x) − C(month(t), x)`. Weighted SVD (existing `_svd_decompose`,
`analysis/eof.py:122`) yields:

- unit-variance, mutually uncorrelated PCs `p_k(t)` (unrotated), and
- regression patterns `E_k(x)` in log-AOD units (`_patterns_from_pc`, `analysis/eof.py:164`),
  so that `y'_m ≈ Σ_k p_k(t) E_k(x)`.

`n_modes` K ≈ 50 to start; truncation is selected on the synthetic validation split, then frozen
for the synthetic test ensemble. Real selection later uses scree/North's-rule plus sensitivity
tests. The real-size path must use a truncated/randomized, chunk-aware solver; the live full
float64 SVD is retained only as a small-case reference oracle.

**Projection-basis constraints:** `rotation: none` and `standardize: false`. Rotated PCs are not
uncorrelated, and standardized EOF patterns are not in log-AOD units unless explicitly
de-standardized. Violations are clear cross-spec validation errors.

### 2.3 Innovation and climatological bias term

Per sensor, day, and observed cell:

```
d_s(t, x) = y_o^s(t, x) − y_m(t, x),   x ∈ Ω_s(t)        # log(obs/model)
```

Split into systematic + anomaly parts, `d_s = b(month, x) + d'_s`:

- `b_mean(m,x)` is the precision-weighted mean of `d` over the **bias-fit window only**, retaining
  per-sensor counts and standard error. Within cells meeting `f_min`, form `b_hat` with two
  mask-aware 3x3 passes (cyclic longitude, clipped latitude); restore unsupported cells afterward.
  The baseline synthetic sensors share one physical bias; a stress scenario adds sensor offsets
  to quantify the limitation of a common term.
- Let `f(m,x)` be the number of unique `bias_fit` days in month `m` with at least one valid sensor,
  divided by the total `bias_fit` calendar days in that month. With defaults `f_min=0.20` and
  `f_full=0.50`, define `S0=0` below `f_min`, `S0=(f-f_min)/(f_full-f_min)` between the bounds,
  and `S0=1` at/above `f_full`. Smooth `S0` with two mask-aware 3x3 passes (cyclic longitude,
  clipped latitude) independently for each calendar month, then restore `S=0` wherever
  `f<f_min` and clip to `[0,1]`. Define `b_applied=S*b_hat`; the reconstructed anomaly is also
  multiplied by `S` in §2.7. This is an explicit post-estimation confidence taper. Observations at
  `S=0` are excluded from projection so a newly appearing unsupported cell cannot alter global
  coefficients elsewhere.
- `b_hat` is bounded in transformed space by explicit `delta_bounds`; asymmetric physical
  `r_bounds` are enforced only by the exact conversion in §2.7.
- The projection (§2.4) acts on `d' = d-b_hat`; tapering occurs only at reconstruction, so the
  un-applied fraction of a partial-support bias cannot leak into anomaly coefficients.

Note the basis climatology `C` cancels in `d` (both obs and model would subtract the same `C`),
so the innovation needs no climatology handling beyond `b_hat`.

### 2.4 Per-day projection (reduced-space OI / ridge)

For each day `t`, stack all valid supported sensor/cell observations into `d(t)` and the matching
basis rows into `G(t)`. Define `C_obs(t)` as the **effective joint covariance** of that stacked
innovation vector, including the intended area representation and any cross-sensor blocks:

```
a_hat(t) = argmin_a [d' - G a]^T C_obs^-1 [d' - G a] + lambda ||a||^2

<=> (H(t) + lambda I) a_hat(t) = g(t),
    H(t) = G(t)^T C_obs(t)^-1 G(t)               (KxK)
    g(t) = G(t)^T C_obs(t)^-1 d'(t)              (K)
```

For independent errors the default is `C_obs,ii = sigma_i^2 / cos(lat_i)`, exactly reproducing the
original area-weighted objective; thus `sigma_i` is the unweighted log-error scale and `C_obs` is
the covariance used by the solve. Structured v1 covariance is block-diagonal plus configured
low-rank common sensor modes, never an assembled dense global matrix. `lambda=1` is a baseline
prior assumption, not a physical identity: it is MAP only under `a~N(0,I)` and `C_obs`. Absolute
covariance scale and ridge strength are confounded, so one is frozen while the other tunes only on
calibration. Overlapping sensors are not counted independent when a common-mode block is configured.

Properties that answer the seasonal-mask question:

- Cross-mode leakage from mask-broken orthogonality is handled exactly (full `H`, not its diagonal).
- An exact null eigen-direction of `H` is set to zero by the prior. Near-null mixed directions are
  shrunk according to their eigenvalues and may cross-talk across named modes; geographic
  extrapolation is controlled separately by `S`, not inferred from `H_kk` alone.
- Cost includes assembly of `H`, nominally `O(T Ncell K^2)`, not only the K×K solve. The plan
  requires a benchmark of chunked/batched assembly before any real-data claim (§6).

### 2.5 Observability diagnostics (stored with the PCs)

- `resolution(t, mode)` = `diag[(H+λI)⁻¹ H]` ∈ [0, 1) — recovered fraction of a unit true amplitude.
- `coverage(t, mode)` = `Σ_{x∈Ω} w E_k² / Σ_x w E_k²` — mask coverage of mode k's variance.
- `n_obs(t, sensor)` — observed-cell counts.
- `posterior_variance(t, mode)`, minimum/maximum resolution eigenvalue, condition number, and
  effective observed rank — diagonal resolution alone cannot expose poorly observed combinations.
- `spatial_support(month, lat, lon)`, support counts, and climatological-bias standard error.
- Flag `(t, k)` when `resolution < ρ_min` (default 0.3, config `min_resolution`); flagged entries
  are treated as **gaps** by the wavelet filter rather than trusted as shrunk-to-zero values.

### 2.6 Wavelet filter (per mode k)

1. Set flagged entries of `a_hat_k` to NaN. Interpolate only interior gaps no longer than
   `max_bridge_days`; record `bridged(time, mode)` and synthesized fraction. Long gaps split the
   series into independent segments; outside an accepted segment the anomaly correction is zero,
   not edge-held or linearly extrapolated.
2. Fix one `dt,dj,s0,J` and period grid from the full requested axis and finite configured `T_max`.
   Require `min_segment_days >= 2*T_max`; shorter segments emit zero anomaly and invalid QA.
3. For each accepted segment fit/remove `mu + beta*(t-t_bar)`, estimate AR(1), and compute the CWT
   on the common scale grid. Mean/trend are re-added only inside that segment and intentionally
   bypass the configured wavelet band.
4. Zero coefficients failing the pointwise AR(1) significance test (`keep_significant: true`).
5. Zero coefficients outside the configured period band `[T_min, T_max]` (`band:`); COI is kept
   for reconstruction but is excluded from acceptance scoring; the retained COI fraction is stored.
6. Inverse CWT (`pycwt.icwt`, Torrence & Compo 1998 eq. 11), restore segment mean/trend, then apply
   a cosine taper over `min(T_max, segment_length/4)` at each segment edge so correction approaches
   identity continuously. Power is NaN outside segments; global spectra/significance aggregate
   segment values with valid-sample weighting on the shared period grid.
7. Record per mode: retained variance fraction, and the unfiltered icwt round-trip relative error
   (Morlet reconstruction is approximate, ~few %); warn above 5 %.

Pointwise AR(1) significance is initially a filtering heuristic because ridge shrinkage and gap
fill make its nominal probability imperfect. A zero-correction synthetic ensemble measures the
false-positive rate. If its frozen threshold fails, FDR or synthetic Monte Carlo significance is
required before `SYNTHETIC_READY`; the test may not be weakened to preserve the heuristic.

### 2.7 Reconstruction, scale factor, application

```
Δ_requested  = b_applied(month(t),x) + S(month(t),x) * Σ_k a_tilde_k(t) E_k(x)
Δ_rmin       = ln([A_m*r_min + ε] / [A_m + ε])
Δ_rmax       = ln([A_m*r_max + ε] / [A_m + ε])
Δ_safe       = clip(Δ_requested, Δ_rmin, Δ_rmax)
Δ_safe       = 0 where A_m < aod_floor or S = 0
A*_raw       = [A_m + ε] * exp(Δ_safe) - ε
r_1deg       = clip(A*_raw / A_m, r_min, r_max)           # only A_m>=floor, S>0; roundoff clip
r_1deg       = 1 elsewhere
A_target     = A_m * r_1deg
Δ_applied    = ln(A_target + ε) - ln(A_m + ε)
ln_r_native  = periodic_bilinear(ln(r_1deg) -> native grid)
S_native     = periodic_bilinear(S -> native grid)
ln_r_native  = 0 where S_native = 0
ln_r_3hr     = linear-in-time interp(ln_r_native)         # only inside correction coverage
r_3hr        = exp(ln_r_3hr); outside coverage r_3hr = 1 (or file is skipped by config)
q_tilde_i    = r_3hr * q_i                                # configured aerosol species only
```

The AOD-dependent transformed bounds are applied **before** exponentiation (with stable
`log1p`/`expm1` forms where appropriate), so an extreme reconstructed anomaly cannot overflow.
This ordering keeps the physical MMR multiplier positive and bounded, preserves exact
`A_target = A_m * r_1deg`, and makes daily-to-3-hourly interpolation multiplicative. Exact identity
holds at analysis-grid `S=0` cells and native cells whose interpolated support is zero; boundary
cells taper continuously. The writer does not hold the first/last correction into pre-observation
or post-observation files. Counts are stored
for low-AOD identity, spatial-support identity, lower/upper clipping, and outside-coverage identity.

For a fixed homogeneous optical operator (same species list, size/composition, RH, meteorology,
and optical coefficients), uniform scaling makes its column AOD scale by exactly `r`. The
synthetic generator includes that forward-operator oracle (§7.4). This is an optical correction,
not a chemically or mass-balanced aerosol analysis. Speciation fractions and vertical profile
shape are preserved; configured gas-phase tracers and fill values are not scaled.

### 2.8 Success criteria

- **Synthetic algebra gates:** noiseless full-mask projection, shifted-log round trip, native/time
  interpolation, and MMR optical closure meet numerical tolerances.
- **Synthetic recovery gates (§8.1):** on untouched seeded test cases, corrected AOD and `Δ_applied`
  improve field error against the latent nature state, not noisy observations. Report coefficient
  correlation **and** slope/bias/NRMSE after weighted mode matching, plus field metrics by season,
  latitude, observation support, and resolution bin. Include multiple seeds/confidence intervals,
  clipping rate, null false-positive rate, and the representable-subspace error ceiling.
- Cells with `S = 0` have `r = 1` exactly. With the support gate disabled in an explicit research
  scenario, global EOF extrapolation is allowed and no geographic no-correction claim is made.
- **Real data (deferred):** corrected AOD vs assimilated Aqua is an assimilation diagnostic;
  external success is improvement against predeclared unassimilated AERONET and/or a wholly
  withheld sensor. Basis, bias/support, and hyperparameters freeze before external evaluation.

---

## 3. Architecture & data flow

```
sources:   model_hourly             sensor_a_raw + sensor_b_raw       native MMR files
                 │                            │ QA
analyses:  1. aod_preprocess ────────────────┘
              model_daily + sensor_daily {aod, log_aod, valid/support metadata}
           2. eof              basis {eofs, pc, variance, climatology, ...}
           3. eof_projection   a_hat + bias/support/posterior diagnostics
           4. wavelet_filter   a_tilde + bridge/COI/reconstruction diagnostics
           5. aod_scaling      mode-grid r/A_target + chunked analysis-grid artifact
           6. mmr_writer       corrected native files + checksummed manifest
plots:     eof_pattern, eof_scree, pc timeseries, wavelet_scalogram (QA on 2–3),
           spatial maps of r / b_hat / aod_target
oracle:    truth sidecar is absent from fitting sources; only the post-fit evaluation loads it
```

Steps 1–5 are required derived analyses with named inputs (§4.1). Large outputs are persisted by
an analysis-declared, chunked artifact policy and remain lazy in pipeline context. Step 6 is a
side-effect-capable result with atomic per-file writes and a checksummed manifest; it is fatal on
partial failure unless an explicit resume policy proves each existing output complete.

### 3.1 Synthetic development reference YAML

This is the first executable target. Paths are created under `tmp_path` in CI or an ignored
synthetic run directory. The oracle file is deliberately absent from the config.

```yaml
analysis:
  start_time: "2001-01-01 00:00:00"
  end_time:   "2006-12-31 23:59:59"
  output_dir: ${FABLE_SYNTH}/output
  log_dir:    ${FABLE_SYNTH}/logs

sources:
  model_hourly:
    type: generic
    files: ${FABLE_SYNTH}/inputs/model/MERRA2_SYNTH.tavg1_2d_aer_Nx.nc4
    variables: { TOTEXTTAU: { units: "1" } }
    time_padding: "1D"             # NEW: retained through load; output clips to analysis window
  sensor_a_raw:
    type: satellite_l3
    files: ${FABLE_SYNTH}/inputs/obs/sensor_a.nc
    variables: { aod_550nm: { units: "1" }, reported_sigma_log: {}, QA: {} }
    qa_variable: QA
    qa_values: [3]
  sensor_b_raw:
    type: satellite_l3
    files: ${FABLE_SYNTH}/inputs/obs/sensor_b.nc
    variables: { aod_550nm: { units: "1" }, reported_sigma_log: {}, QA: {} }
    qa_variable: QA
    qa_values: [3]

analyses:
  model_daily:
    type: aod_preprocess          # NEW
    source: model_hourly
    variable: TOTEXTTAU
    sample_local_time: 13.5
    day_anchor_hour: 12.0
    target_grid: 30.0             # 6x12 CI grid; real config uses 1.0
    log_epsilon: 0.01
    required: true

  sensor_a_daily:
    type: aod_preprocess
    source: sensor_a_raw
    variable: aod_550nm
    uncertainty_variable: reported_sigma_log
    day_anchor_hour: 12.0
    target_grid_from: model_daily
    log_epsilon: 0.01
    required: true

  sensor_b_daily:
    type: aod_preprocess
    source: sensor_b_raw
    variable: aod_550nm
    uncertainty_variable: reported_sigma_log
    day_anchor_hour: 12.0
    target_grid_from: model_daily
    log_epsilon: 0.01
    required: true

  aod_basis:
    type: eof
    source: model_daily
    variable: log_aod
    n_modes: 3
    remove_seasonal_cycle: true
    standardize: false
    rotation: none
    solver: full                    # reference oracle for small CI; real path uses randomized
    fit_window: { start: "2001-01-01", end: "2002-12-31 23:59:59" }
    required: true

  obs_pcs:
    type: eof_projection
    basis: aod_basis
    model: model_daily
    model_variable: log_aod
    obs:
      - { source: sensor_a_daily, variable: log_aod, error_variable: obs_error_std }
      - { source: sensor_b_daily, variable: log_aod, error_variable: obs_error_std }
    ridge: 1.0
    bias_fit_window: { start: "2003-01-01", end: "2004-12-31 23:59:59" }
    clim_bias: true
    spatial_support: monthly_taper
    support_min_fraction: 0.2
    support_full_fraction: 0.5
    support_smoothing_passes: 2
    delta_bounds: [-1.6094379, 1.6094379]
    min_resolution: 0.3
    required: true

  filtered_pcs:
    type: wavelet_filter
    source: obs_pcs
    variable: pc
    keep_significant: true
    significance_level: 0.95
    band: { min: 4, max: 180, units: days }
    max_bridge_days: 7
    min_segment_days: 360
    omega0: 6.0
    required: true

  scaling:
    type: aod_scaling
    basis: aod_basis
    projection: obs_pcs             # bias, support, diagnostics
    coefficients: filtered_pcs
    model: model_daily              # A_m for exact shifted-log inverse
    r_bounds: [0.2, 5.0]
    aod_floor: 0.001
    required: true

  corrected:
    type: mmr_writer
    scaling: scaling
    files: ${FABLE_SYNTH}/inputs/mmr/*.nc4
    species: null                  # synthetic files contain the complete default list
    output_dir: ${FABLE_SYNTH}/corrected
    time_interp: log_linear
    outside_coverage: identity
    overwrite: false
    required: true

plots:
  basis_maps:  { type: eof_pattern, source: aod_basis, variable: eofs }
  basis_scree: { type: eof_scree,   source: aod_basis, variable: explained_variance }
  pc1:         { type: timeseries,  source: filtered_pcs, variable: pc, mode: 1 }
  pc1_scal:    { type: wavelet_scalogram, source: filtered_pcs, variable: power, mode: 1 }
  r_map:       { type: spatial, source: scaling, variable: r }
```

The generator's immutable schedule is 2001-02 `basis_train`, 2003-04 `bias_fit`, 2005
`calibration`, and 2006 `development_test`; the explicit config windows must hash-match that
schedule. Production saved-fit semantics use explicit basis/bias artifact references and are
implemented/tested synthetically rather than deferred to real-data evaluation.

### 3.2 Deferred real-data mapping

After `SYNTHETIC_READY`, replace the three synthetic raw sources with MERRA-2 hourly AOD and
QA-filtered daily MODIS sources, change `target_grid` to 1°, choose the randomized solver, and use
saved train/calibration artifacts. The real config uses canonical reader output `aod_550nm`, not
an HDF SDS name. It uses explicit end-of-day timestamps and source padding. No mathematical or
pipeline contract is allowed to change merely to make the real run pass; reader/metadata defects
are fixed at their boundary and covered by reader integration tests.

---

## 4. New components (file by file)

All new modules stay < 500 lines (project goal); pure math lives in plain functions for unit
testability, analysis classes are thin adapters — mirroring `eof.py`'s structure.

### 4.1 Named-input execution and result contract

- Add `AnalysisSpecBase.input_refs() -> dict[str, str]` and `required: bool = false`. Existing
  single-source specs return `{"source": spec.source}`. New specs return named roles such as
  `basis`, `model`, `projection`, `coefficients`, and `obs[0]`; raw and derived inputs use the
  same resolver. Schema validation, DAG construction, and runtime resolution all call this one
  method rather than maintaining type switches that can drift.
- Make legacy `DerivedAnalysis.analyze()` a concrete compatibility hook (remove `@abstractmethod`;
  its default raises `NotImplementedError`) and make `analyze_inputs(inputs, spec, runtime)` the
  stage entry point, where immutable
  `AnalysisRuntime` supplies the requested analysis window and artifact service without exposing
  mutable pipeline context. Its default adapter calls the existing `analyze(inputs["source"],
  spec)`, preserving current analyses. Multi-input analyses override the named-input method.
- Add `AnalysisResult(dataset, artifacts, manifest_entries)`; plain `xr.Dataset` returns are
  adapted for backward compatibility. Artifact-producing analyses declare policy in their result,
  not via `if spec.type == "gridded_analysis"` in the stage.
- If a required analysis fails, the stage is `FAILED`, descendants are recorded as dependency
  blocked, and no pipeline success is possible. Optional independent analyses retain soft-failure
  behavior. A writer error is always fatal after cleaning its temporary file; already finalized
  outputs remain listed for explicit resume, never silently treated as a complete run.
- Add `DataGeometry.ARTIFACT` for manifest-only pseudo-sources; it is excluded from pairing and
  plotting unless a renderer explicitly declares support.

### 4.2 `davinci_monet/analysis/aod_preprocess.py` — `aod_preprocess`

- Named inputs are `source` and optional derived `target_grid_from`; the latter participates in
  schema validation and the DAG even though only its coordinates are used. Output is
  `aod(time,lat,lon)`, `log_aod`, `valid`, and optional standardized `obs_error_std`, with attrs and
  source hashes. Both model and sensor paths use the same ordered operations: QA/finite screening
  (`AOD >= 0` valid; negative invalid), optional local-solar-time sampling, area-weighted regrid,
  then shifted log. `aod_floor` controls scaling identity separately and does not invalidate zero.
- Initial sensor errors already share the target grid and pass through exactly. If uncertainty is
  coarsened, use the declared covariance of a weighted mean (`var=sum_ij alpha_i alpha_j C_ij`),
  never an ordinary mean of standard deviations; reject missing covariance assumptions. Optional
  `common_factor_variables` are propagated through the same linear regrid weights and keep their
  `(time, common_mode, lat, lon)` contract.
- `time_padding` is consumed by `LoadSourcesStage` and added to `SOURCE_LOADER_CONFIG_KEYS` so it
  is never forwarded to `xr.open_dataset`. Output is clipped to the
  requested calendar-day window only after longitude-dependent sampling. Synthetic tests cover
  dateline selection, a date-only last day, adjacent UTC days, and non-commensurate grids.

### 4.3 `davinci_monet/analysis/projection.py` — `eof_projection`

- Named inputs: one basis, the preprocessed daily model field, and one or more preprocessed obs
  fields. Output geometry is GRID with `pc(time, mode)`, `resolution`, `coverage`, posterior
  diagnostics, `clim_bias`, `clim_bias_applied`, `spatial_support`, counts/standard errors, and
  `n_obs(time, sensor)`.
- The output time axis is the complete daily `model` axis. Every sensor is reindexed to it before
  masking; a missing file/day and a present-but-all-invalid day both remain explicit rows with
  `n_obs=0`, zero coefficients/resolution, and gap QA.
- Pure functions cover innovation, precision-weighted bias/support, `C_obs` construction, one-day
  solve, posterior diagnostics, and chunked orchestration. The small exact solver and the batched
  path are compared on identical synthetic inputs.
- Validate basis metadata (`rotation=none`, `standardize=false`, log epsilon/grid identity), input
  coordinates, error positivity/covariance shape, nonempty time overlap, and frozen fit artifacts.
- Days with no usable obs emit zero projected anomaly and zero resolution. They may be short-gap
  bridged only under §2.6; long gaps remain zero anomaly. Bias/support are saved as reproducible
  artifacts so validation/test runs cannot refit them.

### 4.4 `davinci_monet/analysis/wavelet_filter.py` — `wavelet_filter`

- Output geometry is SPECTRUM. It emits filtered `pc(time, mode)`, `power`, normalized
  `power_significance`, `coi`, `global_power`, `global_significance`, period units and
  `wavelet_quantity`, plus `bridged`, `valid_segment`, `retained_variance`, `recon_error`, and
  `synth_fraction`. All segments share the configured period coordinate; outside-segment values
  are NaN/invalid and global quantities are valid-sample weighted. This is the exact existing
  `wavelet_scalogram` contract, extended by `mode`.
- It implements the bounded-gap/segment rules in §2.6. Mode-selected renderer tests verify the
  actual artist/data contract; PNG-size-only assertions are insufficient.

### 4.5 `davinci_monet/analysis/scaling.py` — `aod_scaling`

- Named inputs: basis, unfiltered projection (bias/support), filtered coefficients, and daily
  model AOD. Output geometry is GRID and stays on the analysis grid only:
  `r`, `delta_log_requested`, `delta_log_applied`, `aod_target`, support/clip/low-AOD masks, and
  per-time fractions. This avoids the invalid attempt to put different mode/native grids on the
  same `(time, lat, lon)` dimensions.
- Pure functions implement reconstruction and the exact shifted-log conversion in §2.7. The
  scaling artifact is time-chunked and lazily reopened. The writer reads the needed daily chunks,
  interpolates `ln(r)` to each file's native grid with periodic longitude, then interpolates in
  time; a decades-long native-grid ratio is never retained in memory or one monolithic file.

### 4.6 `davinci_monet/analysis/cwt_core.py` — shared CWT

- Extract from `WaveletAnalysis.analyze()` a typed `CWTResult` holding complex coefficients,
  scales, periods, COI, AR(1), and significance arrays, plus `cwt_reconstruct(...)` around
  `pycwt.icwt`.
- `WaveletAnalysis` remains behavior-identical and `wavelet_filter` is the second consumer. Pin
  and install `pycwt==0.4.0b0` without dependency resolution in `davinci` before this phase.

### 4.7 `davinci_monet/analysis/mmr_writer.py` — `mmr_writer`

- Streams the direct file glob one file at a time. It rejects input/output aliasing, validates the
  complete configured species set before writing, reads only bracketing daily ratio chunks,
  applies periodic spatial plus log-linear temporal interpolation, and follows the explicit
  outside-coverage policy (`identity`, `skip`, or `error`; never edge-hold by accident).
- Default aerosols are `DU001-005, SS001-005, SO4, BCPHOBIC, BCPHILIC, OCPHOBIC, OCPHILIC`;
  real names are audited only in the real phase. Gas tracers, fill values, coordinates, unrelated
  fields, dtype/dim order/attrs/compression/chunks are preserved.
- Write to a same-filesystem temporary path, fsync/close/validate, then `os.replace`. `overwrite`
  defaults false. Resume accepts an existing file only when its input/config/scaling hashes match.
- Return an ARTIFACT pseudo-source and manifest entries with final checksums, coverage/clip stats,
  config/code/scenario hashes, and file status. The run manifest consumes these entries directly.

### 4.8 Config schema (`config/schema.py`)

- Add `AnalysisSpecBase`, `AODPreprocessSpec`, `EOFProjectionSpec` with nested obs/covariance
  entries, `WaveletFilterSpec`, `AODScalingSpec`, `MMRWriterSpec`, and evaluation-only
  `KnownTruthSpec`; extend the union/dispatcher.
- `EOFSpec` gains `solver: full|randomized`, solver seed/oversampling/iterations, fit-window or
  fit-artifact selection, and stored preprocessing metadata. A projection basis rejects
  `standardize=true` and non-`none` rotation.
- Constrain positive epsilon/error/ridge/grid values, ordered ratio/delta/band bounds, fractions in
  `[0,1]`, local time in `[0,24)`, nonnegative gap length, valid covariance dimensions, disjoint
  output paths, and every named dependency. `monthly_taper` defaults are exactly §2.3; segmented
  wavelets require finite `band.max` and `min_segment_days >= 2*band.max`. Add source-level
  `time_padding` duration validation.

**Structured covariance config.** V1 represents `C_obs = D + U U^T`: `D` comes from each obs
entry's `error_variable` after area scaling; optional `common_factor_variables` have dimensions
`(time, common_mode, lat, lon)` in shifted-log effective-error units. Stack rows deterministically
in config sensor order then C-order `(lat,lon)`, concatenate common-mode columns by declared name,
and subset the same rows for each day's valid mask. Apply `C_obs^-1` with the Woodbury identity
using diagonal `D`; never materialize dense `C_obs`. Schema rejects mismatched common-mode names,
grids, units, or missing factors. The generator serializes both factors and realized common errors.

### 4.9 Shared preprocessing utilities

- `util/regrid.py`: conservative/area-weighted coarsening and periodic bilinear interpolation,
  with explicit center/edge convention and longitude normalization.
- `util/local_time.py`: calendar-day local-solar sampling with adjacent-day input and an output
  validity mask; nearest/tie behavior is deterministic.
- `util/logspace.py`: shifted-log forward/inverse plus `delta_to_ratio` with low-AOD and bounds
  policies. Generator oracles do **not** import these production functions.

### 4.10 Daily satellite support (real-data phase only)

- Add `MOD08_D3`/`MYD08_D3` and later VIIRS catalog entries, but also make the reader cadence-aware:
  daily files retain their parsed day rather than being snapped to month start. Return canonical
  `aod_550nm` plus explicit QA/support fields; tests use representative synthetic HDF/NetCDF before
  any real file. Verify C6.1 SDS/QA details against real metadata only after `SYNTHETIC_READY`.

### 4.11 EOF output and solver (`analysis/eof.py`)

- Store reconstruction metadata currently discarded: climatology, time mean, optional std,
  training split/window, solver/seed, source/preprocess hashes, and log epsilon/grid attrs.
- Keep the full SVD as a small-array reference. Add a deterministic truncated/randomized solver
  that never constructs full right-singular factors, reports approximation/subspace error against
  the reference on synthetic cases, operates on float32/chunks at real scale, and is benchmarked
  before a real run. Explained variance denominator/accuracy must remain well-defined.

### 4.12 Artifact persistence (`analysis/artifacts.py` and pipeline manifest)

- Replace the gridded-analysis type check with analysis-declared artifacts. Migrate existing
  `GriddedAnalysis` and its regression tests to return the new policy. Use atomic, yearly/monthly
  time-chunked **NetCDF4 collections** (the existing supported stack), lazy `open_mfdataset`, scientific
  content hashes, and lightweight summaries that do not eagerly scan multi-GB arrays.
- Artifact manifests record role, dimensions/chunks, source/config/code hashes, split/fit identity,
  and checksums. Wall-clock timestamps are metadata but excluded from scientific hashes.

### 4.13 `davinci_monet/analysis/known_truth.py` — evaluation-only `known_truth`

- A generic named-input ARTIFACT analysis compares an estimate artifact with an explicitly configured truth
  source after fitting is frozen. It computes the weighted field/subspace/coefficient and strata
  metrics in §8.1 and emits a small recovery dataset/CSV. It cannot create or replace basis,
  bias/support, coefficient, scaling, or corrected-MMR artifacts.
- Synthetic evaluation config validation allows `oracle/` only when every fit input is a finalized,
  hash-validated artifact and rejects `known_truth` in a fitting config.

---

## 5. Changes to existing code — summary table

| File | Change | Size |
|---|---|---|
| `analysis/base.py`, `config/schema.py` | Named-input/result contract, required semantics, 6 new specs + nested obs/covariance entries, extend `EOFSpec` | scoped; split schema if module limit requires |
| `pipeline/stages/analyses.py` | Single resolver for validation/DAG/execution; required failure and artifact handling | ~100 lines |
| `pipeline/stages/load.py` | Loader-only per-source time padding before analysis-window clipping | small |
| `analysis/eof.py` | Stored fit metadata + deterministic full/reference and randomized/chunked solvers | substantial, keep helpers isolated |
| `analysis/artifacts.py`, `pipeline/stages/manifest.py` | Analysis-declared chunked/atomic artifacts and checksummed manifest entries | scoped |
| `analysis/gridded.py` | Migrate existing product artifacts to the new result policy without regression | small |
| `analysis/wavelet.py` | refactor guts into `cwt_core.py`; behavior-identical outputs | net ~0 |
| `core/protocols.py` | Add non-pairable `ARTIFACT` geometry | small |
| `pipeline/stages/plot.py` | Replace whole-array finite check with chunk-aware/lazy sampling before real-scale plots | small, gated in P7 |
| `datasets/satellite/modis_viirs.py` + catalog | Post-gate D3 cadence, canonical variable, QA contract | deferred real phase |
| `analysis/__init__.py`, docs, CLAUDE.md | register new modules; document new analyses | small |

New production modules: `aod_preprocess.py`, `projection.py`, `wavelet_filter.py`, `scaling.py`,
`mmr_writer.py`, `cwt_core.py`, `known_truth.py`, `util/regrid.py`, `util/local_time.py`, and
`util/logspace.py`.
Synthetic modules/files are specified in §7.1. Each module remains cohesive and under the project
500-line target; split orchestration from pure math when needed rather than compressing behavior.

---

## 6. Performance & memory

- **CI cases:** coarse global grids and six synthetic years; full SVD is the deterministic oracle.
  `masked_chain_ci` is fixed at 12x24 native cells, 6x12 analysis cells, hourly 2001-2006 model
  input (plus padding), daily sensors, and float32 fields: model input < 75 MiB, complete FABLE CI
  < 60 seconds and peak RSS < 2 GiB on the development machine. A documented environment-based
  skip is allowed only for the opt-in OSSE, never for CI.
- **Real training matrix:** 30 yr x 365 d by 64.8 k cells is ~2.9 GB at float32 and ~5.7 GB at
  float64 before factors/workspace. The current eager float64 full SVD is not viable. The real gate
  requires bounded-memory randomized/truncated SVD, no full `Vt`, deterministic seed, and measured
  subspace/explained-variance error against the full solver on graduated synthetic matrices.
- **Projection:** assembling daily `H` is `O(T Ncell K^2)`. Benchmark independent-diagonal and
  structured-covariance paths, peak RSS, chunk size, and wall time on a synthetic 1°/multi-year
  stress case before estimating real cost. A 50x50 solve is not the dominant-cost argument.
- **CWT:** benchmark 50 x 11k series including segmentation and significance, not only the raw FFT.
- **Artifacts/scaling:** never eagerly summarize/reload a whole decades-long field or retain both
  mode/native grids. Use time chunks, lazy reopen, and per-file native interpolation in the writer.
- **Writer:** one input file at a time with bounded memory and atomic finalization; parallel writing
  remains deferred until deterministic serial behavior and resume semantics pass.
- **Opt-in synthetic OSSE budget:** 8 years, 36x72 native and 18x36 analysis cells, peak RSS < 8 GiB
  and wall time < 30 minutes. The pre-real 30-year/1° solver-artifact benchmark must stay below
  16 GiB peak RSS; wall time is measured and approved by the user rather than guessed in advance.
- Run in conda env **`davinci`**. Install `pycwt==0.4.0b0` with `--no-deps` as documented and use
  `HDF5_USE_FILE_LOCKING=FALSE` for test/runs where needed.

---

## 7. Synthetic data design and generation

### 7.1 Files, API, and ownership

Add these tracked, text-only components:

- `davinci_monet/tests/synthetic/aerosol_tuning.py`: coupled, pure generator and serialization API.
- `davinci_monet/tests/unit/synthetic/test_aerosol_tuning.py`: generator/oracle identities.
- `davinci_monet/tests/integration/test_aerosol_tuning_pipeline.py`: full pipeline tests.
- `analyses/aerosol-tuning/scripts/generate_synthetic.py`: thin CLI over the same generator.
- `analyses/aerosol-tuning/configs/fable-synthetic.example.yaml`: portable config matching §3.1.
- `analyses/aerosol-tuning/configs/fable-synthetic-eval.example.yaml`: post-fit-only artifact vs
  oracle pairs/stats plus `known_truth`; it is never a fitting input.
- `analyses/aerosol-tuning/.gitignore`: generated inputs, oracle, output, plots, logs, and corrected
  files. No generated binary or machine-specific path is committed.

`generate_aerosol_tuning_bundle(root, spec) -> SyntheticTuningBundle` returns in-memory datasets
and, through a separate writer adapter, creates:

```
root/
  inputs/model/MERRA2_SYNTH.tavg1_2d_aer_Nx.nc4
  inputs/obs/sensor_a.nc
  inputs/obs/sensor_b.nc
  inputs/mmr/MERRA2_SYNTH.inst3_3d_aer_Nv.YYYYMMDD.nc4
  oracle/truth.nc
  scenario.json
```

**Fitting/tuning configs** may reference only `inputs/`. Tests fail if an `oracle/` path or truth
variable appears there. A separate post-fit evaluation config may load oracle fields after all fit
artifacts and parameters are frozen; it cannot write or replace fit artifacts.

### 7.2 Determinism and provenance

- `SyntheticTuningSpec` is a frozen, validated dataclass containing grids, periods, split dates,
  mask/error settings, species, bounds, and a root seed. Reuse existing `Domain`/`TimeConfig`, but
  not the independent random-field scenarios, because this feature requires coupled latent truth.
- Use `np.random.Generator(PCG64)` with stable, named stream IDs derived from `(master_seed,
  stream_name)`, not order-dependent sequential draws. Streams include model residual, correction
  residual, each cloud mask/noise source, common error, outages, and MMR perturbations.
- `scenario.json` records normalized spec, schema/root seed/stream map, serializer and NumPy/
  xarray/netCDF versions, roles, and two hashes: byte SHA-256 for file integrity plus a canonical
  coordinate/array/attribute hash for scientific reproducibility across serializer versions.
  Dataset attrs contain `synthetic=true`, scenario/schema/seed/spec hash. Wall-clock time is not
  part of scientific content.

### 7.3 Coupled latent process (independent of production code)

Create analytic spatial patterns, then weighted-orthonormalize them with an independently written
`cos(lat)` Gram-Schmidt implementation. Model variability and correction variability have
different periods, phases, amplitudes, and random streams:

```
y_m(t,x) = mu(x) + C_m(month,x)
           + sum_j gamma_j p_model,j(t) F_j(x) + eta_m(t,x)

delta_true(t,x) = b_true(month,x)
                  + sum_k a_true,k(t) F_k(x)
                  + delta_perp(t,x)

y_nature(t,x) = ln(A_m_overpass(t,x) + epsilon) + delta_true(t,x)
y_obs,s(t,x)  = y_nature + sensor_bias_s + error_common + error_s
```

`delta_perp` is generated independently and orthogonalized against the retained model subspace.
`exact_micro` sets it to zero; acceptance/stress cases do not. This exposes the representable
error floor and avoids the inverse crime of generating truth by calling the same EOF/projection
functions being tested. Generator code may not import FABLE preprocessing, projection, scaling,
wavelet-filter, interpolation, or writer helpers.

Model files are hourly on a MERRA-like native grid with a known longitude-dependent diurnal term.
Observations are formed from the independent local-time/regrid oracle in linear AOD, converted via
the exact shifted-log equations, corrupted in log space, and finally masked/QA-screened. Every
daily product is anchored at **12:00 UTC** for interpolation; local overpass selection remains
longitude dependent but maps to that labeled calendar day.

### 7.4 Truth and MMR schemas

`oracle/truth.nc` uses distinct dimensions for distinct grids:

| Variable | Dimensions | Meaning |
|---|---|---|
| `pattern_true` | `(truth_mode, mode_lat, mode_lon)` | prescribed weighted-orthogonal model patterns |
| `model_pc_true`, `correction_pc_true` | `(time, truth_mode)` | independent model/correction coefficients |
| `clim_bias_raw_true`, `spatial_support_true` | `(month, mode_lat, mode_lon)` | pre-taper bias and frozen support policy |
| `delta_in_span_true`, `delta_perp_true`, `delta_requested_true` | `(time, mode_lat, mode_lon)` | recoverable, irreducible, and requested correction |
| `delta_supported_true`, `delta_applied_true` | `(time, mode_lat, mode_lon)` | post-support and post-floor/bounds corrections |
| `delta_best_representable_true` | `(time, mode_lat, mode_lon)` | independent policy-adjusted retained-subspace ceiling |
| `delta_filter_target_true` | `(time, mode_lat, mode_lon)` | independent spatial+temporal passband/segment/policy target |
| `model_aod_overpass_true` | `(time, mode_lat, mode_lon)` | independent local-time/regrid oracle |
| `r_requested_true`, `r_applied_true`, `clip_mask_true` | `(time, mode_lat, mode_lon)` | pre-policy ratio, applied ratio, clip reason |
| `aod_target_requested_true`, `aod_target_applied_true` | `(time, mode_lat, mode_lon)` | noise-free nature before/after configured policy |
| `aod_filter_target_true` | `(time, mode_lat, mode_lon)` | AOD implied by `delta_filter_target_true` |
| `r_native_true` | `(time, native_lat, native_lon)` | independent support-aware periodic interpolation |
| `r_3hour_true` | `(mmr_time, native_lat, native_lon)` | independent log-time interpolation |
| `valid_mask`, `qa_flag`, `mask_reason` | `(sensor, time, mode_lat, mode_lon)` | support and decomposed mask causes |
| `obs_error_log`, `reported_sigma_log` | sensor/time/grid | realized and reported errors |
| `innovation_noise_true` | `(time, mode_lat, mode_lon)` | precision-combined noise field for null-energy scoring |
| `obs_holdout_aod`, `obs_holdout_error_log` | `(time, mode_lat, mode_lon)` | independent unassimilated replicate |
| `kappa`, `layer_weight`, `baseline_optical_aod`, `scaled_optical_aod` | species/level/time/grid | serialized optical operator and closure oracle |
| `split` | `(time)` | immutable basis/bias/calibration/development-test schedule |

Generate at least two daily native MMR files with eight 3-hourly samples, four CESM-style pressure
levels (pressure increases with index; surface last), all 15 default aerosols, SO2/DMS/MSA, RH,
pressure thickness, and an unrelated meteorological field. Species have distinct normalized
fractions and vertical profiles. An independent optical oracle
`AOD = sum_i,z kappa_i(RH,z) q_i dp/g` normalizes model AOD and proves `AOD_out = r*AOD_in` under
the fixed synthetic operator. Files exercise float32, `_FillValue`, NaNs, attrs, zlib compression,
chunking, and dimension order. The holdout observation and optical fields use their own named RNG
streams and never appear in fitting inputs.

### 7.5 Masks, errors, and scenario ladder

Store each mask component and form `valid_s = footprint & seasonal_visibility & cloud &
day_available & qa_pass`. Include hemispheric winter gaps, correlated synthetic clouds, whole
absent days, present-but-all-invalid days, a permanent unsupported region, a wholly unobservable
mode, complementary sensors, and overlap. Invalid raw AOD remains finite and extreme with `QA=0`
so a skipped QA filter is detectable; valid data use `QA=3`.

| Scenario | Purpose | Characteristics |
|---|---|---|
| `exact_micro` | algebra/unit oracle | full mask, no noise/clipping/off-basis term; ridge-zero and analytic-ridge variants |
| `masked_chain_ci` | full preprocess-to-scaling CI | six years with the §3.1 schedule, 12x24 native/6x12 analysis grids, separated modes, seasonal/cloud masks, high SNR |
| `multi_sensor_ci` | QA/precision/covariance | complementary footprints, controlled overlap, distinct known errors |
| `writer_ci` | file and optical closure | reuse `masked_chain_ci` analysis inputs; write two selected MMR days with full species/gas/static/fill cases |
| `null_ci` | false-positive/gap control | zero true bias and anomaly, noise, short and long gaps |
| `low_aod_ci` | shifted-log closure | zero/near-zero model AOD, both ratio clips, support-zero cells |
| `synthetic_osse` | opt-in acceptance | 8 years, 36x72 native/18x36 analysis grids, off-basis truth, drift/correlated errors/MNAR stress |

Canonical recovery cases use independent Gaussian log errors with known diagonal `C_obs`. Stress cases
add common/correlated error, heteroscedasticity, sensor bias, missing-not-at-random cloud masks,
and basis drift. Stress cases diagnose assumption failure; they are not mislabeled exact recovery.

### 7.6 Leakage controls and comparison oracles

- Basis and climatological support/bias fit only their explicit windows. Hyperparameters (`K`,
  covariance simplification, band, ridge, resolution, gap length) tune only on 2005 calibration;
  2006 is a repeatable development test. After config/thresholds freeze, the user or gate runner
  supplies three acceptance seeds not used or hard-coded during development; they are recorded and
  run once for `SYNTHETIC_READY`.
- Daily test observations may enter projection because this is retrospective assimilation; their
  latent truth and scores never enter fitting/tuning. Score against `aod_target_applied_true` and
  an unused independent observation replicate, not the noisy observations assimilated.
- Compare learned bases using weighted subspace angles/projector error. For separated modes, use
  weighted Hungarian sign/permutation matching before coefficient scores; never assume EOF labels.
- Primary reconstruction comparison uses `delta_filter_target_true`; also report full-policy
  `delta_applied_true`, pre-policy `delta_in_span_true`, `delta_perp`, and distance from
  `delta_best_representable_true` so spatial versus temporal/policy error is not conflated. The
  independent temporal oracle is assembled analytically from known in-band/out-of-band components,
  configured mean/trend, segment taper, support, floor, and clipping; it does not call pycwt or
  production filters. Exclude CWT COI/segment edges from primary scores but report excluded fraction
  and full-domain results so exclusion cannot hide failure.

---

## 8. Validation & testing design (approval required before implementation)

### 8.1 Synthetic OSSE (centerpiece; opt-in, no real data)

Run the fitting config through `PipelineRunner.run_from_config()`, freeze its artifacts, then run a
separate evaluation-only pipeline that loads scaling output and oracle AOD as ordinary raw gridded
sources for standard pairs/stats/plots. A read-only `known_truth` analysis in that evaluation
pipeline produces subspace/coefficient metrics; it has no artifact-write access to fitting roles.

Primary hard gates score `delta_log_applied` against `delta_filter_target_true` on development/acceptance
times, supported cells, and non-COI valid segments. Each day is equally weighted and cells use
normalized `cos(lat)` weights. Required: weighted correlation >= 0.90, origin-constrained slope
0.8-1.2, and NRMSE <= 0.35 where NRMSE divides by weighted RMS oracle correction. Separately,
weighted AOD RMSE against `aod_filter_target_true` must be <= 70% of uncorrected model RMSE; RMSE
against the full `aod_target_applied_true` must also improve over the uncorrected model.
Report full-domain and support/resolution/season/latitude strata, excluded fraction, coefficient
metrics, and distance from `delta_best_representable_true`; raw AOD correlation is diagnostic only.

For `null_ci`, define false-positive energy as
`sum(w*delta_log_applied^2) / sum(w*innovation_noise_true^2)` over the same non-COI test domain; it
must be <= 0.10. The fraction of significant coefficients among valid, non-COI coefficients inside
the configured band must also be <= 0.10.
Exact policy/IO cases use numerical closure tolerances rather than statistical targets.

### 8.2 Pipeline integration tests (CI; all enter through `PipelineRunner.run_from_config()`)

- **T1 `test_aerosol_tuning_known_mode_chain`**: generate `masked_chain_ci`; run preprocess → EOF
  → projection → filter → scaling. Assert required pseudo-sources/artifacts, no analysis errors,
  subspace recovery, shifted-log closure, support-zero identity, and latent-target improvement.
- **T2 `test_aerosol_tuning_multi_sensor_projection`**: produce A-only, B-only, and blended analyses
  from one basis. Assert exact QA counts, analytic precision weighting in controlled overlap,
  absent-sensor zero contribution, and lower analytic posterior variance. Empirical RMSE advantage
  is assessed only as a locked multi-seed aggregate, not required from one random realization.
- **T3 `test_aerosol_tuning_writer_pipeline`**: run the complete chain through `mmr_writer`; inspect
  both files, optical closure, metadata/fill preservation, atomic outputs, checksums, and run manifest.
- **T4 `test_aerosol_tuning_null_and_gap_behavior`**: projection is zero on all-missing days and a
  wholly unobservable mode; only bounded gaps bridge; long gaps/outside coverage remain identity;
  null retained anomaly energy/false-positive rate stays below the frozen 10% ceiling.
- **T5 `test_aerosol_tuning_required_failure_is_fatal`**: inject a required descendant failure and
  assert failed stage/run and dependency-blocked descendants. Hash-validated files finalized before
  the failure may remain resumable, but the run/manifest is explicitly incomplete and never success.
- **T6 `test_aerosol_tuning_saved_fit_fresh_runner`**: write basis/bias/support artifacts, start a
  fresh runner using only those fits but the **same complete 2001-2006 application axis** (so CWT
  detrending/AR(1)/COI/segment context is identical), prove no refit occurred, compare 2006 output
  with the same-run result, and reject source/config hash mismatch.

Total synthetic CI target: under 60 seconds. Large `synthetic_osse` is developer-run, not CI.

### 8.3 Unit and generator-contract tests

- Generator: deterministic named streams, positivity, mask composition, QA rejection, split
  immutability, truth hidden from configs, shifted-log identities, and independent optical closure.
- Projection: exact full-mask coefficients; analytic ridge shrinkage; masked/correlated `C_obs` cases;
  posterior/resolution bounds and rank; support preservation after smoothing; common-bias precision.
- Wavelet: band pass/reject, mean/trend policy, bounded gaps/segments, masks/COI, reconstruction
  error, and null ensemble. If the false-positive gate fails, implement calibrated FDR/Monte Carlo.
- Scaling: exact shifted-log inverse including low AOD, asymmetric bounds, support identity,
  pre-exponential AOD-dependent bounds (including extreme finite anomalies), periodic seam, and
  log-time interpolation. Float64 pure math uses tight `rtol <= 1e-10`; writer
  float32 closure uses `rtol <= 5e-6` unless the independent oracle proves a stricter bound.
- Writer: prescribed-ratio pure unit test (separate from T3), full species coverage, unchanged gas/
  fill/static fields, dtype/dims/attrs/compression/chunks, collision/overwrite/resume, atomic failure.
- Pipeline/schema: named DAG order, raw+derived refs, unknown/cycle errors, numeric bounds, required
  failures, artifact laziness/manifest, and exact renderer output contracts.

### 8.4 Deferred real evaluation

Only after §8.5: enable daily readers/catalogs and generate frozen real fit artifacts. A second
standard pipeline evaluates saved `aod_target` and MERRA-2. Aqua observations used in projection,
including a later time window, yield **assimilation diagnostics only**. Headline external validation
uses a predeclared unassimilated source (prefer AERONET; optionally a completely withheld satellite)
and reports N/MB/RMSE/R/NMB/NME/IOA globally and by season/support.

### 8.5 `SYNTHETIC_READY` gate

The gate requires: repository tests/mypy/Black/isort green; all T1-T6 and pure oracles green;
three user-supplied, then locked, `synthetic_osse` seeds meet frozen aggregate/per-seed targets;
null and off-basis floors
are honestly reported; peak memory/runtime benchmarks satisfy documented limits; artifacts/manifests
reproduce from hashes; and the user reviews the recovery report. No real-data phase begins earlier.

---

## 9. Implementation phases

| Phase | Deliverable | Gate |
|---|---|---|
| **P0** | Restore/disposition baseline gates in `davinci`; install pinned pycwt; approve this test design. Build named-input/result/required-failure/artifact foundations and migrate existing `gridded_analysis`. | existing behavior/artifact regressions green; T5 foundation green |
| **P1** | Coupled generator, independent truth/MMR oracle, serializers, provenance, scenario unit tests, synthetic example skeleton. | generator identities and no-truth-leak checks green |
| **P2** | Regrid/local-time/log utilities, source time padding, and `aod_preprocess`; no real reader changes. | exact/date-edge/dateline/low-AOD unit tests + pipeline preprocess test |
| **P3** | EOF metadata + full reference and deterministic randomized solver; graduated synthetic benchmark. | existing EOF tests + subspace/variance/memory gates |
| **P4** | Shared `cwt_core.py` refactor with current wavelet output regression. | existing wavelet tests behavior-identical |
| **P5** | `eof_projection`, saved bias/support fit artifacts, covariance/posterior diagnostics. | pure solver tests + T2 |
| **P6** | Bounded-gap `wavelet_filter` with exact scalogram contract and null calibration. | wavelet unit/null tests + renderer contract |
| **P7** | Exact `aod_scaling`, chunked/lazy artifacts, and chunk-aware plot finite checks. | T1 + low-AOD/support/clip oracles |
| **P8** | Atomic `mmr_writer`, `known_truth` evaluation, full-species optical closure, manifests; run user-supplied acceptance seeds. | T3-T6 + user-approved `SYNTHETIC_READY` report |
| **P9** | **Real-data enablement begins:** MERRA-2 audit, MODIS D3 cadence/catalog/canonical variable/QA readers, real configs. | reader tests with synthetic representative files, then controlled real-file smoke |
| **P10** | Frozen-fit MERRA-2/Aqua run; assimilation diagnostics plus external unassimilated evaluation; tune nothing on evaluation data. | before/after and sensitivity report reviewed by user |
| **P11** | GEOS-IT grid/collection/species audit, reader work, then configs; Terra/VIIRS expansion separately. | same synthetic regression + product-specific gates |

P0-P8 remain strictly synthetic. Each phase is TDD, but repository rules require presenting that
phase's concrete test entry points/data flow and receiving approval before writing tests. No commits
or pushes occur without explicit user approval.

---

## 10. Risks & open questions

1. **MODIS DT/DB regional biases** leak into `b_hat` (it will faithfully "correct" toward biased
   retrievals, e.g. DB over bright surfaces). Mitigations later: per-region σ inflation,
   AERONET cross-check of `b_hat` maps before trusting them.
2. **Basis stationarity over decades** — major eruptions (Pinatubo) distort covariance. Option:
   `exclude_periods:` on the EOF training window; decide after inspecting the scree/patterns.
3. **Correction-subspace mismatch** is structural. Synthetic `delta_perp` and basis-drift cases
   quantify the irreducible floor; real claims must not imply EOF span completeness.
4. **Wavelet significance after shrinkage/gap fill** is heuristic until the null ensemble passes.
   FDR/Monte Carlo calibration is mandatory if the frozen false-positive gate fails.
5. **icwt fidelity** (~few % for Morlet) is measured per mode and included in recovery error.
6. **Support gating trades covariance extrapolation for conservative identity.** The default honors
   the stated no-evidence/no-correction policy; an ungated research sensitivity is reported, not
   silently substituted.
7. **Observation covariance misspecification** affects shrinkage and sensor weighting. Synthetic
   correlated-error cases bound the consequence; real `C_obs` remains an effective covariance model.
8. **Artifact scale and restart integrity** require benchmarked chunks, hashes, atomic writes, and
   bounded summaries before real decades are attempted.
9. **Optical closure is conditional**, not chemical closure. The real RT operator/species/RH audit
   must confirm homogeneity and complete aerosol coverage before corrected MMRs are trusted.
10. **Daily → 3-hourly application** assumes MERRA-2's diurnal AOD shape — accepted by design
   (we correct daily-and-slower scales only).
11. **Overpass sampling approximation** — daily `r` stamped by calendar day though estimated at 13:30
   LST; consistent between training and innovation, so it cancels to first order.
12. **Exact real product details remain deliberately unverified until P9:** D3 SDS/QA/cadence,
   C6.1 choice, MMR tracer list/encoding, and GEOS-IT grids/collections.
13. **Renderer compatibility** is an explicit data/artist contract in P6 rather than a late PNG
   smoke-test discovery.

---

## 11. Decisions & rationale

1. **Innovation projection, not raw-obs projection** — zero-observability limit = "keep the
   analysis", which is the only safe default under seasonally vanishing coverage.
2. **Ridge is an explicit prior model.** `lambda=1` is the baseline for unit-PC covariance only
   when paired with configured `C_obs`; synthetic calibration tests this assumption.
3. **Shifted log space with exact inverse** — preserve statistical benefits without pretending
   `exp(delta)` is the physical ratio at low AOD.
4. **Systematic + anomaly correction in transformed space**, both multiplied by stored monthly
   support; physical `r` is derived once after their sum.
5. **Synthetic truth is independent and coupled** — model/nature/obs/MMR share a known physical
   story, while oracle code never calls production implementations.
6. **Frozen train/calibration/test evidence** precedes all real data; real masks are not synthetic.
7. **1° real mode space** remains the target, but only a benchmarked truncated solver can reach it.
8. **Chained named-input analyses** preserve DAVINCI architecture; required/fatal and artifact
   results make a correction chain safer than today's single-source soft-failure behavior.
9. **Multi-sensor from day one** — obs is a list and covariance-aware contributions are additive;
   no merged observation product is required.
10. **Uniform per-column aerosol scaling** is an optical diagnostic under a fixed homogeneous
    operator; speciation/vertical shape are deliberately untouched and gases are excluded.
11. **Wavelet filtering bridges bounded gaps only.** Long gaps/outside coverage revert anomaly
    correction to zero; trend/mean preservation is explicit and null behavior is calibrated.
12. **MERRA-2/Aqua are first only after `SYNTHETIC_READY`**; GEOS-IT and Terra/VIIRS require their
    own reader/grid/QA audits rather than being assumed drop-in.

## 12. References

- Kaplan, A., et al. (1998): Analyses of global sea surface temperature 1856–1991 — reduced-space
  optimal interpolation of gappy obs onto complete-field EOFs (the method template).
- Torrence, C. & Compo, G. P. (1998): A practical guide to wavelet analysis — CWT, significance,
  reconstruction (eq. 11).
- North, G. R., et al. (1982): Sampling errors in the estimation of empirical orthogonal functions.
- Beckers, J.-M. & Rixen, M. (2003): DINEOF — EOF-based infilling (context; our fixed-basis ridge
  solve supersedes iteration).
- Existing in-repo spec: `docs/superpowers/specs/2026-06-17-eof-and-wavelet-analysis-design.md`
  (derived-analysis layer this plan builds on).
