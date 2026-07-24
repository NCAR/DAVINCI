# Ancillary Surface Observation Notes

Working notes for AERONET and AirNow observations downloaded for the DAVINCI
campaign subsets.

## Access Method

Downloads were made with the DAVINCI CLI in the `davinci` conda
environment:

```bash
conda activate davinci
davinci get aeronet ...
davinci get airnow ...
```

The CLI uses `monetio` to fetch source data and writes DAVINCI-readable NetCDF
files.

## Campaign Windows

| Campaign | Window | Reason |
|---|---|---|
| ASIA-AQ | 2024-01-29 through 2024-04-01 | Full ASIA-AQ analysis window already used by the ASIA-AQ AirNow/AERONET examples. |
| DC3 | 2012-05-29 through 2012-05-30 | Oklahoma / Kingfisher supercell canonical case. |
| FIREX-AQ | 2019-08-06 through 2019-08-08 | Williams Flats sequence, including the 7 August focused flight and 8 August pyroCb. |

## AERONET Inventory

Campaign-specific AERONET files live under each campaign data root:

```text
/glade/work/fillmore/Data/<CAMPAIGN>/AERONET/<date-range>/<filename>
```

Downloaded files:

| Campaign | File | Size | Verification |
|---|---|---:|---|
| ASIA-AQ | `/glade/work/fillmore/Data/ASIA-AQ/AERONET/20240129_20240401/AERONET_L15_20240129_20240401.nc` | 51M | opens with `time=1512`, `site=473` |
| DC3 | `/glade/work/fillmore/Data/DC3/AERONET/20120529_20120530/AERONET_L15_20120529_20120530.nc` | 833K | opens with `time=24`, `site=198` |
| FIREX-AQ | `/glade/work/fillmore/Data/FIREX-AQ/AERONET/20190806_20190808/AERONET_L15_20190806_20190808.nc` | 2.1M | opens with `time=48`, `site=341` |

The downloaded AERONET product is level 1.5, hourly-resampled by the DAVINCI
CLI default.

## AirNow Inventory

Campaign-specific AirNow files live under each campaign data root:

```text
/glade/work/fillmore/Data/<CAMPAIGN>/AIRNOW/<YYYYMMDD>/<filename>
```

The full ASIA-AQ AirNow range could not be written as one file because the
process was killed while combining the full-range dataframe. AirNow was
therefore downloaded one day at a time.

Downloaded files:

| Campaign | Layout | Files | Size | Status |
|---|---|---:|---:|---|
| ASIA-AQ | `/glade/work/fillmore/Data/ASIA-AQ/AIRNOW/YYYYMMDD/AirNow_YYYYMMDD.nc` | 64 | 110M | Complete for 2024-01-29 through 2024-04-01 |
| FIREX-AQ | `/glade/work/fillmore/Data/FIREX-AQ/AIRNOW/YYYYMMDD/AirNow_YYYYMMDD.nc` | 3 | 7.2M | Complete for 2019-08-06 through 2019-08-08 |
| DC3 | `/glade/work/fillmore/Data/DC3/AIRNOW/YYYYMMDD/AirNow_YYYYMMDD.nc` | 0 | 0 | Public `monetio` AirNow path returned empty data for 2012-05-29 and 2012-05-30 |

The DC3 AirNow failures are recorded in:

```text
/glade/work/fillmore/Data/DC3/AIRNOW/download_failures.tsv
```

Representative AirNow files open successfully. The DAVINCI/monetio converter
produced sub-hourly same-day time coordinates for the checked daily files; for
example, `AirNow_20240206.nc` and `AirNow_20190806.nc` both span `00:00`
through `23:30` UTC with 72 time coordinates.

## Notes / Caveats

- The AirNow public `monetio` path did not provide DC3-era 2012 data. Use
  AirNowTech credentials if true AirNow data are required for DC3, or use an
  AQS/EPA surface-monitor fallback if that is scientifically acceptable.
- The root-level flat daily stores under `/glade/work/fillmore/Data/AeroNet`
  and `/glade/work/fillmore/Data/AirNow` still exist for other workflows. The
  campaign-specific ancillary files above have been moved into the campaign
  roots and are the intended inputs for these campaign analyses.
- Existing unrelated files in the repository were not modified.
