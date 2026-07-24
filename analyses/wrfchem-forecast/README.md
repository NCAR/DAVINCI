# Daily WRF-Chem AQ Forecast Evaluation

Daily evaluation of the operational AQ_WATCH WRF-Chem forecast against AirNow
surface datasets, replacing the current `melodies-scripts/wrfchem.yaml` +
`run_wrfchem.py` pipeline with a pure DAVINCI configuration.

## What it does

1. `fetch_airnow.sh` downloads the previous UTC day's AirNow datasets into
   `/glade/work/fillmore/Data/AirNow/AirNow_YYYYMMDD.nc`.
2. `qsub_wrfchem_daily.sh` exports `YYYY/MM/DD` (yesterday by default), submits
   `davinci run` to PBS on Casper. The pipeline pairs the dated WRF-Chem
   forecast with that day's AirNow + AERONET files and writes CONUS scatter,
   timeseries, and spatial bias plots for PM2.5 / OZONE / AOD to
   `/glade/campaign/acom/acom-da/fillmore/DAVINCI/WRF-Chem/YYYY/MM/DD/`.

No Python wrapper — date substitution is done entirely via DAVINCI's `${VAR}`
env-var expansion in the YAML config.

## Data sources

| Source | Path |
|---|---|
| WRF-Chem forecast (06/12/18Z) | `/glade/campaign/acom/acom-da/shawnh/AQ_WATCH/YYYYMMDD/wrf/wrfout_d01_YYYY-MM-DD_{06,12,18}:00:00` |
| WRF-Chem forecast (00Z) | `/glade/campaign/acom/acom-da/shawnh/AQ_WATCH/PREV_YYYYMMDD/wrf/wrfout_d01_YYYY-MM-DD_00:00:00` (prev cycle +24h) |
| AirNow datasets | `/glade/work/fillmore/Data/AirNow/AirNow_YYYYMMDD.nc` |
| AERONET datasets | `/glade/work/fillmore/Data/AeroNet/AeroNet_YYYYMMDD.nc` |
| Plot output | `/glade/campaign/acom/acom-da/fillmore/DAVINCI/WRF-Chem/YYYY/MM/DD/` |

The WRF-Chem forecast is the operational AQ-WATCH run (CONUS d01, every
6 hours per init day, multi-day forecast horizon), owned by `shawnh` in
the acom-da group. Plots from this run are published at
<https://acom.ucar.edu/firex-aq/forecast.html>.

Note that the WRF-Chem files are named `wrfout_d01_YYYY-MM-DD_HH:MM:SS`
(no `_hourly_` infix, no `.nc` extension) — DAVINCI's WRF-Chem reader
opens them via netCDF4 regardless of extension.

### Hour-0 from the previous cycle

The hour-0 file of each cycle (`wrfout_d01_YYYY-MM-DD_00:00:00`) is the IC
dump — written before any chemistry tendency step has run — so the
`PM2_5_DRY` diagnostic is exactly zero across the entire grid. The other
snapshots are populated normally. For the daily evaluation, hour 00 comes
from the **previous** cycle's +24h forecast (file at the same
`YYYY-MM-DD_00:00:00` timestamp, but under `AQ_WATCH/${PREV_YYYYMMDD}/`),
where the diagnostic is populated. The `qsub_wrfchem_daily.sh` script
exports `PREV_YYYYMMDD` for this purpose. The WRF-Chem reader also drops
any all-zero PM2.5 step it sees as a defense-in-depth backstop.

## Files

| File | Purpose |
|---|---|
| `configs/wrfchem-forecast.example.yaml` | Pipeline config (env-var dates) |
| `scripts/fetch_airnow.sh` | Download yesterday's AirNow file |
| `scripts/qsub_fetch_airnow.sh` | PBS wrapper around `fetch_airnow.sh` |
| `scripts/qsub_wrfchem_daily.sh` | PBS wrapper around `davinci run` |

## Usage

**Date replay (for development/testing):**
```bash
analyses/wrfchem-forecast/scripts/qsub_wrfchem_daily.sh 20250801
```

**Yesterday's forecast (operational):**
```bash
analyses/wrfchem-forecast/scripts/qsub_wrfchem_daily.sh
```

**Manual run without qsub:**
```bash
export YYYY=2025 MM=08 DD=01
export PREV_YYYYMMDD=20250731     # previous-day cycle for hour-0 file
davinci run analyses/wrfchem-forecast/configs/wrfchem-forecast.example.yaml
```

## Cron setup (cron.hpc.ucar.edu)

Install the crontab on NCAR's high-availability cron server, **not** on a
Casper login node:

```bash
ssh fillmore@cron.hpc.ucar.edu
crontab -e
```

Two entries replace the current `airnow.sh` + `wrfchem.yaml` chain:

```
20 08 * * * /glade/work/fillmore/DAVINCI/analyses/wrfchem-forecast/scripts/qsub_fetch_airnow.sh
30 09 * * * /glade/work/fillmore/DAVINCI/analyses/wrfchem-forecast/scripts/qsub_wrfchem_daily.sh
```

The cron server's environment is intentionally sparse (no modules, 1 GB
user memory cap), so the scripts:

- use fully-qualified queue names (`-q casper@casper-pbs`), as required
  when submitting from the cron server to Casper PBS
- run a small `activate_env.sh` to source conda and activate the
  `davinci` env before invoking the CLI on the compute node, so
  nothing depends on the user's interactive shell init

Override env paths if needed:
```bash
export DAVINCI_CONDA_BASE=/path/to/miniforge3   # default: /glade/work/fillmore/miniforge3
export DAVINCI_CONDA_ENV=davinci
```

Geometry: [NCAR HPC Documentation — Cron services](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/additional-resources/cron/).

## Notes

- **Domain**: shawnh's operational d01 covers CONUS (lat 23.7→51.7,
  lon -129.8→-64.2), 6-hourly snapshots over a multi-day forecast horizon.
- **Mechanism**: `mech: racm_esrl_vcp` is forwarded to monetio's
  WRF-Chem reader for proper variable resolution.
- **wrf-python compatibility**: monetio's WRF-Chem reader requires `wrf-python`,
  which in turn breaks against `netCDF4 >= 1.7` (raises
  `NotImplementedError: Dataset is not picklable` from wrf-python's internal
  `copy.copy()` on a `netCDF4.Dataset`). The DAVINCI WRF-Chem reader catches
  this and falls back to a plain xarray open with a loud warning. In the
  fallback path, raw WRF variables are returned without mech-aware decoding —
  notably `o3` stays in **ppmv** instead of being converted to ppb. The
  example config keeps `o3.unit_scale: 1.0` for the monetio path; if the
  fallback fires (the warning will tell you), switch it to `1.0e3` so the
  AirNow comparison is unit-consistent. To restore the monetio path,
  pin `netCDF4<1.7` or an older `wrf-python` build in `environment.yml`.
