# FIREX-AQ Dataset Notes for DAVINCI

Working notes for evaluating the NASA/NOAA Fire Influence on Regional to
Global Environments and Air Quality (FIREX-AQ) campaign data in DAVINCI.

## Access

- Mission page: https://www-air.larc.nasa.gov/missions/firex-aq/
- Public archive page: https://www-air.larc.nasa.gov/cgi-bin/ArcView/firexaq
- NASA Earthdata project page: https://www.earthdata.nasa.gov/data/projects/firex-aq
- Williams Flats WRF-Chem / GOES FRP case study:
  https://acp.copernicus.org/articles/22/10195/2022/
- Williams Flats MISR / aircraft aerosol case study:
  https://www.mdpi.com/2072-4292/12/22/3823

The NASA LaRC FIREX-AQ archive is reachable from this environment. Public files
are downloadable through `/cgi-bin/enzFile?...` links.

The archive exposes buttons for `DC8`, `ER2`, `N48`, `N46`, `C130`,
`NightFOX`, `MERGE`, `MOBILE`, `GROUND`, `SATELLITE`, `MODEL`, `TRAJECTORY`,
`ANALYSIS`, and `OTHER`.

## Visible Archive Volume

The visible archive rows report file size in `Size (KB)`. Summing listed file
rows by archive family gives:

| Family | Files | Archive MiB |
|---|---:|---:|
| DC8 | 2,771 | 14,574.6 |
| ER2 | 126 | 37,013.1 |
| N48 | 184 | 294.4 |
| N46 | 81 | 8,640.4 |
| C130 | 0 | 0.0 |
| NightFOX | 0 | 0.0 |
| MERGE | 309 | 11,401.2 |
| MOBILE | 82 | 1,454.4 |
| GROUND | 15 | 10.1 |
| SATELLITE | 147 | 85,513.6 |
| MODEL | 0 | 0.0 |
| TRAJECTORY | 0 | 0.0 |
| ANALYSIS | 382 | 12,164.2 |
| OTHER | 0 | 0.0 |

Visible LaRC archive total is about **167.1 GiB** across **4,097** files.
Satellite and ER-2 products dominate the volume.

## Recommended DAVINCI Starter Subset

Start with **2019-08-06, 2019-08-07, and 2019-08-08**, the Williams Flats
wildfire sequence.

Rationale:

- The ACP WRF-Chem / GOES FRP case study treats Williams Flats as the flagship
  Boise-phase fire and uses DC-8 flights on 3, 6, 7, and 8 August 2019.
- The 6 August flight sampled Williams Flats first, then Horsefly. It is a good
  lead-in day for fresh Williams Flats smoke plus a second wildfire contrast.
- The 7 August flight focused exclusively on Williams Flats, with aged smoke
  and fresh plume sampling phases.
- NASA Earthdata notes that the 8 August DC-8 flight captured the Williams
  Flats pyroCb event.
- The MDPI MISR / aircraft aerosol study focuses on Williams Flats observations
  on 6 August 2019, which makes 6 August useful for satellite/aircraft pairing.

Optional later addition:

| Date | Reason |
|---|---|
| 2019-08-03 | Earlier Williams Flats DC-8 sampling day used in the ACP WRF-Chem case study. |

## Local Download Inventory

Local data root:

```text
/glade/work/fillmore/Data/FIREX-AQ
```

Directory convention:

```text
/glade/work/fillmore/Data/FIREX-AQ/<family>/<YYYYMMDD>/<group>/<filename>
```

For these downloads, `<family>` is the first manifest column, such as `DC8`,
`ER2`, `MERGE`, `SATELLITE`, `ANALYSIS`, `MOBILE`, `N46`, or `GROUND`.
The `group` column is usually `FIREXAQ`, so a typical path is:

```text
/glade/work/fillmore/Data/FIREX-AQ/DC8/<YYYYMMDD>/FIREXAQ/<filename>
```

The direct LaRC archive URLs encode the archive dataset key as `XFS`; the files
were initially downloaded under `XFS/<YYYYMMDD>/FIREXAQ/` and then reorganized
by manifest family for easier discovery.

Each selected day has:

- `manifest_all_YYYYMMDD.tsv`: dated manifest across all archive families.
- `download_YYYYMMDD.log`: one-file-at-a-time download history.

Completed on 2026-06-24:

| Date | Families | Files | Archive MiB | On-Disk Size |
|---|---|---:|---:|---:|
| 2019-08-06 | analysis 13, DC-8 118, ER-2 13, merge 10, mobile 1, N46 3, satellite 4 | 162 | 11,562.1 | 12G |
| 2019-08-07 | analysis 7, DC-8 119, ER-2 11, ground 2, merge 10, mobile 9, N46 6, satellite 4 | 168 | 10,399.7 | 11G |
| 2019-08-08 | analysis 8, DC-8 118, ER-2 12, merge 10, mobile 1, N46 1, satellite 4 | 154 | 9,871.6 | 9.7G |

The local subset currently occupies about **32G** and completed cleanly with
zero `.part` files remaining. A manifest-to-file audit found zero missing
files.

Current family-level local inventory:

| Family | Files | On-Disk Size |
|---|---:|---:|
| ANALYSIS | 28 | 153M |
| DC8 | 355 | 2.0G |
| ER2 | 36 | 16G |
| GROUND | 2 | 5.1M |
| MERGE | 30 | 1.5G |
| MOBILE | 11 | 453M |
| N46 | 10 | 1.2G |
| SATELLITE | 12 | 11G |

Satellite products included for each selected day are:

- `firexaq-GOES16-data-WesternUS_Satellite_YYYYMMDD_R0.zip`
- `firexaq-GOES16-images-DC8_Satellite_YYYYMMDD_R0.zip`
- `firexaq-GOES17-data-WesternUS_Satellite_YYYYMMDD_R0.zip`
- `firexaq-GOES17-images-DC8_Satellite_YYYYMMDD_R0.zip`

## Suggested Products To Use First

Prioritize merged and flag products for initial DAVINCI wiring:

- `FIREXAQ-mrg60-DC8_merge_YYYYMMDD_R3.ict`
- `FIREXAQ-mrg60-DC8-NC_merge_YYYYMMDD_R3.nc`
- `firexaq-fire-Flags-1HZ_DC8_YYYYMMDD_R9.ict`
- `firexaq-FSU-smokeage_dc8_YYYYMMDD_R1.ict`
- `firexaq-Fuel2Fire-FlightTimeAlignedEmissions_DC8_YYYYMMDD_R0.ict`

Then add the core instrument-level products needed for smoke chemistry and
aerosol analysis:

- DC-8 MetNav / MMS for position and aircraft state.
- CO and CO2 from DACOM / CO2-7000.
- O3 and NOx from ROZE / NOyO3 / LIF / CANOE.
- CH2O from ISAF and ACES.
- Aerosol composition and optical products from AMS, SP2, LAS/neph, AOP, and
  LARGE.
- ER-2 CPL, GCAS, NAST-I, and SHIS products for remote-sensing context.
- GOES16/GOES17 satellite ZIPs for fire evolution, FRP, and imagery context.

## Practical DAVINCI Plan

1. Start with the 2019-08-07 one-minute DC-8 merge plus fire flags and smoke
   age. This is the cleanest Williams Flats-only aircraft day.
2. Add 2019-08-08 to exercise the pyroCb case and plume-rise-sensitive
   behavior.
3. Add 2019-08-06 for MISR/aircraft pairing and a Williams Flats plus Horsefly
   contrast.
4. Add ER-2 CPL/GCAS and GOES products once the aircraft-only path works.
5. Add instrument-level aerosol and trace-gas products only after the merge
   products are ingested and plotted correctly.

## Notes / Caveats

- File counts and sizes are based on the archive listing as accessed on
  2026-06-24. NASA may update archive contents and revisions.
- The direct archive links are session-like encoded `enzFile` URLs. Prefer
  discovering current links from the archive page rather than hard-coding the
  exact encoded URLs.
- The local subset includes LaRC archive products for the selected dates,
  including satellite products. It does not include non-LaRC external products
  unless they appear in the LaRC FIREX-AQ archive listing.
