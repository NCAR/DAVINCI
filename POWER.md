# NASA POWER Integration — Design & Plan

**Status**: Design approved 2026-07-14; **rescoped 2026-07-15** against the talk abstract.
**Implementation active** — build before the talk. No code written yet.
**Scope**: New `power` data source (point + regional), REST client with disk cache, staging CLI,
MERRA-2 demo evaluation campaign.

---

## Context

[NASA POWER](https://power.larc.nasa.gov/) (Prediction Of Worldwide Energy Resources, NASA LaRC) serves 300+
analysis-ready solar and meteorological parameters through a free REST API — no login, no granule staging:

- **Solar** parameters derive from CERES SYN1deg + FLASHFlux (native 1° × 1°).
- **Meteorology** derives from MERRA-2 / GEOS 5.12.4 (native 0.5° × ⅝°; GEOS covers the near-real-time tail).
- **Temporal levels**: hourly, daily, monthly, climatology; record from 1981 to within days of present.
- **Endpoints**: `https://power.larc.nasa.gov/api/temporal/{hourly|daily|monthly|climatology}/{point|regional}`.
  Point requests allow ≤ 20 parameters; regional requests allow **1 parameter** on a 0.5° × 0.5° grid.
  Output formats include JSON, CSV, ASCII, and **NetCDF**.

**Why DAVINCI wants it**: a long, gap-free, globally complete evaluation reference for model surface
meteorology and radiation, obtainable in seconds at exactly the sites/regions/windows an analysis needs.
DAVINCI already reads POWER's parents natively (`merra2`, `ceres_syn1deg` readers), but those require
Earthdata staging of full granules; POWER is the cheap targeted extract. It is also a NASA LaRC product in
the same family as the CERES-SARB-CAM7 work featured in the JOSS paper — a natural story extension.

**Caveat to document for users**: POWER met *is* reanalysis and POWER solar *is* satellite-derived analysis.
Neither half is ground truth — POWER is not a pyranometer or a thermometer. Evaluating a model against POWER
is evaluation against MERRA-2/CERES. Whether that is *informative* or *circular* depends entirely on which
half of POWER and which model, which is what the table below settles for the v1 campaign.

**Talk driver**: `abstract.txt` (repo root) is a talk abstract announcing this integration. It drove the
2026-07-15 rescope below and is a **guide, not a contract** — it is not authoritative for later scope
changes, and it names capabilities (MPAS-GOCART2G-JEDI, CheMPAS-A) that DAVINCI does not have.

### Provenance is not symmetric — this splits the campaign in two

POWER's two halves have different parents, so pairing a source against POWER means different things per
variable. This is the single most important thing to get right in the demo and in the talk:

| POWER variable | Parent | Pairing MERRA-2 against it is… |
|---|---|---|
| `T2M`, `RH2M`, `WS10M`, `PS`, `PRECTOTCORR` | **MERRA-2**, regridded to 0.5° × 0.5° | **circular — a traceability check.** Must agree to regridding error. Disagreement means our reader is wrong, so this is the strongest correctness test available. |
| `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `ALLSKY_SFC_LW_DWN` | **CERES SYN1deg + FLASHFlux** | **a genuine evaluation.** CERES is independent of GEOS's radiation scheme, so MERRA-2 `SWGDN` vs POWER `ALLSKY_SFC_SW_DWN` is a real result, not a tautology. |

Verify the exact per-parameter source attribution against the POWER methodology docs at implementation
(especially `ALLSKY_SFC_LW_DWN`, and which parameters GEOS 5.12.4 FP — not MERRA-2 — supplies in the
near-real-time tail). Never present a circular leg as an evaluation.

## Decisions & Rationale (2026-07-14 brainstorm)

| Decision | Choice | Rationale |
|---|---|---|
| Primary purpose | **Model evaluation reference** | Evaluate model surface met + radiation against a long gap-free record. |
| Geometry | **Both point and regional from day one** | One `power` source type: `sites:` → POINT, `bbox:` → GRID. Endpoints share one query grammar. |
| Data flow | **Live fetch + disk cache** | Reader fetches at pipeline runtime, writes each response as NetCDF to a cache; reruns hit cache offline. `davinci-stage-power` CLI wraps the *same* fetch path for pre-staging. Mirrors AirNow/OpenAQ `dates`-mode precedent. |
| ~~First campaign: CAM radiation + met~~ | **Superseded 2026-07-15** → MERRA-2 | No CAM data on the build machine. See rescope below. |
| Temporal levels | **Hourly + daily + monthly** | Same response schema, near-free together. Climatology (no time axis) deferred — needs special pairing. |

## Rescope & Rationale (2026-07-15, driven by `abstract.txt`)

| Decision | Choice | Rationale |
|---|---|---|
| First campaign | **MERRA-2, not CAM** | CAM output is not on the build machine; MERRA-2 is stageable here today via the existing `davinci-stage-merra2` + Earthdata. MERRA-2 also yields both a self-validating traceability leg *and* an honest evaluation leg (see provenance table above). |
| Parent cross-check | **Promoted backlog → v1** | The abstract headlines traceability. `merra2` and `ceres_syn1deg` readers already exist, so it is mostly config — and it validates the new reader against known-good data. Supporting leg, not the lead. |
| Long-record demo | **POWER single-source monthly, 1981→present** | The abstract's most-repeated selling point is the four-decade record, which a one-month demo never shows. POWER-only, so it needs no staged data and runs anywhere. **POWER leads it, not MERRA-2** — leading with parent cross-checks frames DAVINCI as POWER QA rather than a model-evaluation toolkit. |
| Regional-model leg | **Deferred** | MPAS support does not exist (no reader; abstract names it anyway). `wrfchem`/`cmaq` readers exist and POWER is source-agnostic, so nothing about the reader needs proving with them. MPAS becomes the regional exemplar later. |
| Build vs run split | **Build local, run model campaigns where data lives** | The client and reader are fully buildable + testable here: POWER is a live API needing no staging, and tests are synthetic-first. Machine-bound legs get gitignored `*-<machine>.yaml` configs per the repo naming convention. |
| AI summary | **Enabled in the demo config** | The abstract cites LLM interpretation as a DAVINCI capability; the `summary:` stage already exists and is non-fatal. |

## Architecture

Three components, each following an existing repo pattern:

### 1. `davinci_monet/io/download/power.py` — REST client + cache + staging CLI

Mirrors `io/download/merra2.py` structure (module-level collection/catalog constants, injectable network
hooks, `stage_*()` function, argparse `main()`, `console_scripts` entry `davinci-stage-power`).

- **URL builder** (pure function, golden-tested):
  `build_power_url(temporal, mode, params, *, lat/lon or bbox, start, end, community, fmt="NETCDF")`.
  Example (from POWER docs):
  `…/api/temporal/daily/point?parameters=T2M,PS,WS10M&community=AG&longitude=0&latitude=0&start=20170101&end=20170201&format=CSV`
- **Fetcher**: stdlib `urllib` isolated in a lazy module-local function (the `earthaccess` lazy-import
  pattern) — zero new dependencies, monkeypatchable seam for tests. Bounded retries with backoff on
  HTTP 429/5xx; HTTP 422 (validation) raised immediately with the offending URL in the message.
- **Request planner**: splits an ask into API-legal requests and reassembles:
  - point, N sites, ≤20 params → one request per site (chunk params if > 20); concat along `site`.
  - regional, M params → M single-param requests; merge variables.
  - long windows chunked by calendar year; concat along `time`. (Hourly per-request window limits:
    verify at implementation; the year-chunker is the mitigation either way.)
- **Cache**: each API response saved as NetCDF at
  `<cache_dir>/<temporal>/<community>/<point|regional>/<key>.nc`, where `key` is a deterministic slug of
  (sorted params, rounded coords/bbox, start, end). Fetch checks cache first; `--force` refetches.
  Default `cache_dir`: `~/.cache/davinci/power`, overridable in config/CLI.
- `stage_power(...)` + CLI with `--temporal --params --site LAT,LON[,NAME] / --bbox ... --start --end
  --cache-dir --dry-run` (dry-run prints the planned request list, mirroring `DryRunReport`).

### 2. `davinci_monet/datasets/power.py` — `@source_registry.register("power")`

- **Config modes** (exactly one of):
  - `sites:` list → live/cached fetch → **POINT** dataset `(time, site)` with `latitude(site)`,
    `longitude(site)`, site-name coord — same shape `pt_sfc`/AirNow produce; pairs with model grids via
    the existing point strategy.
  - `bbox:` → live/cached fetch → **GRID** dataset `(time, lat, lon)` on POWER's 0.5° grid.
  - `files:` glob → pure file mode: open previously staged/cached POWER NetCDF, no network (offline
    reruns, CI).
- Dates come from `analysis.start_time/end_time` (AirNow `dates`-mode precedent).
- Standardization: coordinate/attr cleanup to DAVINCI conventions (`latitude`/`longitude` coords,
  `geometry` attr, canonical units — below), fill values (−999 per response attrs) → NaN.
- Reader delegates all fetching to the `io/download/power.py` client — one fetch path for live mode,
  staging CLI, and cache.

### 3. `analyses/power/` — demo evaluation campaign

Standard campaign layout (`configs/`, `data/`, `output/`, `logs/`) with portable env-var-path configs:
`power-longrecord.example.yaml` (POWER-only, runs anywhere) and `power-merra2.example.yaml` (needs staged
MERRA-2). See "Demo campaign" below.

## Config surface (target YAML)

### `power-merra2.example.yaml` — the v1 campaign

```yaml
analysis:
  start_time: "2026-05-01"
  end_time: "2026-05-31"          # MERRA-2 lags ~2-3 weeks behind present; dry-run staging to confirm
  output_dir: ${POWER_ANALYSIS}/output
  log_dir: ${POWER_ANALYSIS}/logs

sources:
  # --- MERRA-2: staged locally, `davinci-stage-merra2 <collection> --root ${POWER_DATA}` ---
  merra2_rad:
    type: merra2
    files: ${POWER_DATA}/MERRA2_tavg1/rad_Nx/*.nc4     # collection tavg1_2d_rad_Nx (M2T1NXRAD)
    variables:
      SWGDN: {}           # surface incoming shortwave, W m⁻² — GEOS radiation scheme's own answer

  merra2_slv:
    type: merra2
    files: ${POWER_DATA}/MERRA2_tavg1/slv_Nx/*.nc4     # collection tavg1_2d_slv_Nx (M2T1NXSLV)
    variables:
      T2M: {}             # K

  # --- POWER: fetched live + cached; no staging, no login ---
  power:
    type: power
    temporal: hourly      # matches MERRA-2 tavg1 natively — no aggregation on either side
    community: RE         # affects native units; normalized away by the catalog
    cache_dir: ${POWER_CACHE}   # optional; default ~/.cache/davinci/power
    sites:                # POINT mode ("virtual stations")
      - {name: boulder,   latitude: 40.02,  longitude: -105.27}
      - {name: table_mtn, latitude: 40.125, longitude: -105.24}
    variables:
      ALLSKY_SFC_SW_DWN: {}    # CERES-derived -> INDEPENDENT of MERRA-2. Evaluation.
      T2M: {}                  # MERRA-2-derived -> CIRCULAR. Traceability only.

  power_grid:
    type: power
    temporal: hourly
    bbox: {lat_min: 30, lat_max: 50, lon_min: -125, lon_max: -100}   # GRID mode
    variables:
      ALLSKY_SFC_SW_DWN: {}

pairs:
  # Leg A - EVALUATION: CERES-derived POWER solar vs MERRA-2's model-computed radiation.
  # Genuinely independent; the headline result.
  merra2_vs_power_swdn:
    x: {source: power,      variable: ALLSKY_SFC_SW_DWN}
    y: {source: merra2_rad, variable: SWGDN}

  # Leg B - TRACEABILITY: circular by construction (POWER T2M *is* MERRA-2 T2M regridded).
  # Must agree to regridding error. This is the reader's correctness test, NOT a result.
  merra2_vs_power_t2m:
    x: {source: power,      variable: T2M}
    y: {source: merra2_slv, variable: T2M}

  # Leg A regional - spatial bias map. Both sides are GRID on DIFFERENT grids
  # (POWER 0.5x0.5 vs MERRA-2 0.5x0.625), so bin both onto a common grid.
  # NOTE: unlike the point legs above, this one DOES aggregate - time_resolution
  # daily-averages both sides. That is deliberate for a bias map (a diurnal cycle
  # has no business in one), but it means the regional and point legs answer
  # slightly different questions. Say so if both appear in the talk.
  merra2_vs_power_grid_swdn:
    x: {source: power_grid, variable: ALLSKY_SFC_SW_DWN}
    y: {source: merra2_rad, variable: SWGDN}
    method: grid
    grid:
      horizontal_res: 0.5
      time_resolution: 1D
      min_sample_count: 1

summary:
  enabled: true
  model: claude-haiku-4-5
```

### `power-longrecord.example.yaml` — the four-decade leg (POWER only, runs anywhere)

Single-source: no `pairs:`, no staged data, no login. This is the local acceptance vehicle.

```yaml
analysis:
  start_time: "1981-01-01"
  end_time: "2026-06-30"
  output_dir: ${POWER_ANALYSIS}/output

sources:
  power_monthly:
    type: power
    temporal: monthly
    sites: [{name: boulder, latitude: 40.02, longitude: -105.27}]
    variables:
      ALLSKY_SFC_SW_DWN: {}
      T2M: {}

plots:
  swdn_record: {type: timeseries, source: power_monthly, variable: ALLSKY_SFC_SW_DWN}
  t2m_record:  {type: timeseries, source: power_monthly, variable: T2M}
```

## Variable catalog & unit normalization

Module-level `POWER_CATALOG` in `datasets/power.py` (pattern: `SSF_CATALOG` in `ceres_ssf.py`), keyed by
**POWER's official parameter names** (they are community-standard; the labeling system supplies display
names). Each entry: expected native units per temporal level, normalization scale/offset, canonical units.

- **Normalize to SI on read**: POWER's native units vary by community and temporal level (e.g. daily solar
  arrives as kWh m⁻² day⁻¹ for RE, MJ m⁻² day⁻¹ for AG; hourly as W m⁻²). The catalog converts everything
  to fixed canonical units: radiative fluxes → **W m⁻²** (kWh m⁻² day⁻¹ × 41.667; MJ m⁻² day⁻¹ × 11.574),
  **T2M °C → K (+273.15)** to match model conventions (offset handled inside the reader — config
  `unit_scale` has no offset support and shouldn't grow one for this).
- Reader asserts the response's `units` attr matches the catalog's expectation before scaling — a unit
  drift in the upstream API fails loudly instead of silently corrupting stats.
- v1 catalog (demo needs + obvious neighbors): `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`,
  `ALLSKY_SFC_LW_DWN`, `T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS10M`, `WS50M`, `PS`, `PRECTOTCORR`.
  Parameters outside the catalog pass through with response-attr units and a logged warning.

## Demo campaign (acceptance vehicle)

`analyses/power/` — every leg runs **through the pipeline** (CLAUDE.md rule), never a bespoke script.

| # | Leg | Pairing | Story | Runs where |
|---|---|---|---|---|
| C | **Long record** *(leads)* | POWER monthly, single-source, 1981→present at the demo sites | The four-decade record — POWER's headline selling point, which a one-month demo never shows | **Anywhere.** Live API, zero staging, no login |
| A | **Radiation evaluation** | POWER `ALLSKY_SFC_SW_DWN` (CERES) vs MERRA-2 `SWGDN`; point + regional | **Genuine result.** CERES is independent of GEOS's radiation scheme | Build machine, once MERRA-2 is staged |
| B | **T2M traceability** | POWER `T2M` vs MERRA-2 `T2M` | **Not a result — a correctness test.** Circular by construction; must agree to regridding error | Build machine, once MERRA-2 is staged |
| D | **Solar parent cross-check** *(supporting)* | POWER `ALLSKY_SFC_SW_DWN` vs `ceres_syn1deg` | Closes traceability on the solar side, as the abstract's impact claim describes | Wherever CERES SYN1deg is staged (see `[[ceres-staged-samples]]` — Io, currently unmounted) |

**Leg C is the local acceptance vehicle**: it is the only leg needing no staged data, and it still exercises
the whole path — URL build → fetch → cache → reader → pipeline → plots. Get it green first.

**Ordering matters for the talk.** POWER leads (leg C), then the evaluation (leg A), then traceability
(legs B, D) as support. Leading with parent cross-checks frames DAVINCI as POWER QA rather than a
model-evaluation toolkit.

Plots follow labeling conventions (terse titles, `bias_label` names sources, PDF to the iCloud per-analysis
subdir). Legs B and D must be **labeled as consistency checks wherever they appear** — in plot titles, in the
stats CSV commentary, and in the talk. A circular pairing presented as an evaluation is a false result.

### Deferred: the CAM campaign

`power-cam.example.yaml` (FSDS vs `ALLSKY_SFC_SW_DWN`, TREFHT vs `T2M`) is unchanged in intent but blocked
on data: **CAM output is not on the build machine.** It runs later, wherever CAM lives, via a gitignored
machine-specific `power-cam-<machine>.yaml`. Nothing in the reader build depends on it.

## Testing plan (synthetic-first, per repo rules)

- **Unit** — URL builder golden strings; request planner (site/param/year chunking); cache-key determinism;
  catalog normalization scales/offsets incl. the units-attr assertion; fill-value masking; POINT/GRID
  standardization from synthetic POWER-shaped NetCDF fixtures (generated programmatically, no network).
- **Download helper** — mirror `tests/test_download_merra2.py`: injected fetch hooks, dry-run report, cache
  hit/miss/force paths. No live HTTP in tests.
- **Integration** — `PipelineRunner.run_from_config()` on a demo-shaped config with the fetch seam
  monkeypatched to serve synthetic POWER NetCDF; assert pairing/stats/plotting stages run and the spatial
  render mark is verified **programmatically** (QuadMesh for the grid leg, PathCollection for points).
- **Campaign acceptance** (live data, run by hand — not CI, which must stay offline):
  - Leg C green end-to-end against the live API — the gate before anything else.
  - **Leg B is the reader's live correctness test**: POWER `T2M` vs MERRA-2 `T2M` is circular, so it must
    agree to regridding error. Decide the pass threshold *before* looking at the numbers (proposed: MB
    within a few tenths K, R > 0.99) and treat a miss as a reader bug — wrong grid, wrong time convention,
    wrong unit offset — not as a finding. This is the highest-value check in the plan: it is the one place
    where the right answer is known in advance.
- Present the test design (entry points + data flow) for approval before writing tests (repo rule).

## Error handling

- Source load failures are **fatal** (unlike the AI-summary stage): a missing evaluation reference must
  fail the run, not skip silently.
- 422 → raise immediately with full request URL and API error body (bad param name, bad bbox…).
- 429/5xx → bounded exponential backoff (e.g. 3 tries), then raise.
- Offline + cache miss → actionable error: the exact `davinci-stage-power` command that would populate it.
- All-fill responses (ocean-only bbox for a land-only param, etc.) surface as all-NaN with a logged warning;
  standard `valid_min/max` machinery applies downstream.

## Implementation checklist (ordered for the talk deadline)

**Start step 0 first — it runs unattended while the build proceeds.**

0. **Kick off MERRA-2 staging** (long pole; needs Earthdata creds + network + disk):
   `davinci-stage-merra2 tavg1_2d_rad_Nx --start 2026-05-01 --end 2026-05-31 --root ${POWER_DATA} --dry-run`
   first to size it, then the real run; repeat for `tavg1_2d_slv_Nx`. Note `DEFAULT_ROOT` is `/Volumes/Io`,
   which is **not mounted on the build machine** — pass `--root` explicitly to local disk.
1. Pre-implementation audit refresh (CLAUDE.md): re-read `io/download/merra2.py`, `datasets/surface/airnow.py`
   (dates mode), `datasets/satellite/ceres_ssf.py` (catalog pattern), `pt_sfc` standardization.
2. Verify against live API (small manual probes, then encode in golden tests): NETCDF format availability
   per endpoint/level; **monthly API date format** (blocking — leg C depends on it); hourly per-request
   window limit; regional bbox size limit; exact native unit strings per community/level; **per-parameter
   source attribution** (which parameters are CERES vs MERRA-2 vs GEOS-FP — the campaign's whole framing
   rests on this).
3. `io/download/power.py` (URL builder → planner → fetcher → cache → CLI) with tests.
4. `datasets/power.py` (config schema, three modes, catalog, standardization) with tests.
5. Wire `console_scripts` entry; docs (CLAUDE.md source-type table, README).
6. Integration test through `run_from_config` (test design approved first).
7. **Leg C** (`power-longrecord.example.yaml`) — no staged data required. **This is the go/no-go gate for
   the talk**: green here means the client, reader, cache, and pipeline wiring all work.
8. **Legs A + B** (`power-merra2.example.yaml`) once step 0 lands. Treat a leg-B disagreement as a reader
   bug, per the testing plan.
9. **Leg D** (CERES SYN1deg cross-check) — needs Io mounted or CERES re-staged. Cut first if time runs short.
10. Gates in the `davinci` conda env: pytest, mypy, black, isort. PDF plot copies to the iCloud
    per-analysis subdir.

**Cut order if time runs short**: leg D → leg A regional → *never* leg C.

**Legs A and B are not separable.** They read the same staged MERRA-2 and the same reader path, so B costs
almost nothing once A runs — and A's radiation bias is uninterpretable until B has passed, because an
unvalidated reader turns every bias into "is this GEOS, or is this our bug?" **Do not present leg A in the
talk unless leg B passed.** If B fails and there is no time to fix it, cut A too and give the POWER-only
talk (leg C), which stands on its own.

## Open questions (resolve at implementation; proposed defaults stated)

**Blocking** — the rescope made these load-bearing:

1. **Talk date** — not yet recorded. It sets the cut line in the checklist above. Fill this in.
2. **Monthly endpoint date format** — likely year-only `start`/`end`; verify and encode in URL builder
   tests. **Leg C, the go/no-go gate, is monthly** — this must work.
3. **Hourly request window limit** — legs A and B are hourly across a month at several sites, so the
   planner's chunking is exercised on day one, not deferred. Verify the exact limit; chunk, don't cap.
4. **GRID↔GRID pairing** — the regional leg pairs POWER (0.5° × 0.5°) against MERRA-2 (0.5° × ⅝°), two
   different grids. Confirm whether the geometry precedence rules sample one onto the other sensibly, or
   whether `method: grid` (intermediate gridding) is required. The config above assumes `method: grid`.

**Non-blocking**:

5. **Regional bbox size limit** — verify; tile if needed.
6. **NETCDF for every endpoint** — if an endpoint lacks NETCDF, fall back to JSON → xarray construction in
   the client (response schema is stable).
7. **T2M in K** — the rescope settles this: leg B compares against MERRA-2 `T2M`, which is in K, so K it is.
8. **Community default `RE`** — any reason to prefer `AG`/`SB`? Normalization makes it mostly moot.
9. **MERRA-2 monthly collections absent** — `MERRA2_COLLECTIONS` has `tavgM` for aerosol only, not rad/slv.
   Not needed for v1 (leg C is POWER-only), but a monthly MERRA-2 cross-check would first need `M2TMNXRAD` /
   `M2TMNXSLV` added to the collection map.

## Deferred ideas backlog (not in v1)

*(The parent cross-validation campaign was promoted out of this list into v1 — it is now legs B and D.)*

- **CAM campaign** (`power-cam.example.yaml`) — designed, blocked only on data: no CAM output on the build
  machine. Runs later wherever CAM lives, via a gitignored `power-cam-<machine>.yaml`.
- **MPAS-GOCART2G-JEDI / CheMPAS-A readers** — **do not exist.** `abstract.txt` names them twice as models
  DAVINCI reads; the registry has no MPAS reader and the repo's only "GOCART" hits are MERRA-2's aerosols.
  Deferred 2026-07-15. Until a reader lands, do not claim MPAS support.
- **Regional-model leg** — POWER vs `wrfchem`/`cmaq` over a bbox, demonstrating the "global to regional"
  span concretely rather than claiming it. Cheap once suitable output is staged; MPAS is the preferred
  exemplar when its reader exists.
- **Climatology level** (`temporal: climatology`) — needs time-axis-less pairing semantics.
- **Applications angle (JOSS impact)**: renewable-energy / agroclimatology derived analyses in the
  `analyses:` block (capacity-factor-style solar resource, degree-days) benchmarked against POWER.
- **Virtual-station generator**: auto-derive `sites:` from another source's coordinates (e.g. put POWER
  met context at every AERONET/Pandora site in a campaign).
- **AWS cloud datastore (zarr)** backend for bulk global/hourly work where the REST API is too slow.
- **Uncertainty**: surface POWER's parameter-uncertainty products as auxiliary variables.
- **Upstream**: contribute the POWER client to monetio once stable in DAVINCI.

## References

- Site: https://power.larc.nasa.gov/ · Docs: https://power.larc.nasa.gov/docs/
- API: https://power.larc.nasa.gov/docs/services/api/ (temporal point/regional; JSON/CSV/ASCII/NetCDF;
  point ≤ 20 params, regional 1 param on 0.5° grid; 429 rate limiting)
- Methodology (sources/resolutions/latency): https://power.larc.nasa.gov/docs/methodology/
- Parameter dictionary: https://power.larc.nasa.gov/parameters/
