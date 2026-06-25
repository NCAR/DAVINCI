# DC3 Dataset Notes for DAVINCI

Working notes for evaluating the NASA/NSF Deep Convective Clouds and Chemistry
Project (DC3) field campaign data in DAVINCI.

## Access

- Mission page: https://www-air.larc.nasa.gov/missions/dc3-seac4rs/
- Public archive page: https://www-air.larc.nasa.gov/cgi-bin/ArcView/dc3-seac4rs
- NCAR/EOL project page: https://www.eol.ucar.edu/field_projects/dc3
- NASA NTRS 29-30 May WRF-Chem case: https://ntrs.nasa.gov/citations/20160005306
- 29 May Kingfisher supercell overview: https://repository.library.noaa.gov/view/noaa/32177/noaa_32177_DS1.pdf

The NASA LaRC DC3 archive is reachable from this environment. Public files are
downloadable through `/cgi-bin/enzFile?...` links.

The archive exposes buttons for `DC8`, `GV`, `FALCON`, `MERGE`, `O3SONDES`,
`TRAJECTORY`, `LIGHTNING`, `MODEL`, `ANALYSIS`, and `SATELLITE`. As accessed on
2026-06-24, only `DC8`, `GV`, `FALCON`, and `MERGE` returned file rows through
this archive endpoint. `SATELLITE` currently returned zero file rows, so there
is no LaRC satellite product family analogous to ASIA-AQ/GEMS to pull from this
interface.

## Visible Archive Volume

The visible archive rows report file size in `Size (KB)`. Summing listed file
rows by archive family gives:

| Family | Files | Archive MiB |
|---|---:|---:|
| DC8 | 1,935 | 15,706.6 |
| GV | 485 | 1,107.0 |
| FALCON | 187 | 108.3 |
| MERGE | 214 | 1,094.0 |
| O3SONDES | 0 | 0.0 |
| TRAJECTORY | 0 | 0.0 |
| LIGHTNING | 0 | 0.0 |
| MODEL | 0 | 0.0 |
| ANALYSIS | 0 | 0.0 |
| SATELLITE | 0 | 0.0 |

Visible LaRC archive total is about **18.0 GiB** across **2,821** files.

## Recommended DAVINCI Starter Subset

Start with **2012-05-29 and 2012-05-30**, the Oklahoma severe convection /
Kingfisher supercell case.

Rationale:

- The DC3 overview paper identifies the 29 May Oklahoma case as the severe
  convection case: a line of isolated supercells from northern through central
  Oklahoma that produced strong winds, large hail, and an EF1 tornado. It also
  notes that the GV sampled upper-tropospheric convective outflow, the DC-8
  sampled inflow and then upper-tropospheric outflow, and the DLR Falcon sampled
  outflow.
- The same overview notes that on 30 May, the GV and DC-8 sampled aged
  convective outflow from the 29 May storm over the southern Appalachian region.
- DiGangi et al. describe the 29 May Kingfisher storm as a DC3 intensive
  analysis case with multiple aircraft, mobile Doppler radars, soundings, and
  Oklahoma Lightning Mapping Array coverage.
- NASA NTRS has a WRF-Chem analysis focused specifically on the 29-30 May 2012
  Oklahoma convective event, lightning NOx production, and downwind chemistry,
  which makes this a natural DAVINCI/WRF-Chem bridge case.

Use 2012-05-29 for direct storm inflow/outflow and 2012-05-30 for aged plume
evolution.

Optional later additions:

| Date | Reason |
|---|---|
| 2012-06-06 | Colorado strong convection case in the DC3 overview paper. |
| 2012-06-22 | Colorado smoke ingestion case, useful once the storm pipeline works. |
| 2012-06-30 | No-storm / boundary-layer VOC sampling, useful as a contrast case. |

## Local Download Inventory

Local data root:

```text
/glade/work/fillmore/Data/DC3
```

Directory convention:

```text
/glade/work/fillmore/Data/DC3/<dataset>/<YYYYMMDD>/<PI-or-product>/<filename>
```

Each selected day has manifests and logs in the data root:

- `manifest_YYYYMMDD.tsv`: original DC-8/GV aircraft manifest.
- `manifest_all_YYYYMMDD.tsv`: dated manifest across all archive families.
- `manifest_extra_YYYYMMDD.tsv`: Falcon and merge additions pulled after the
  first aircraft pass.
- `download_YYYYMMDD.log`: one-file-at-a-time DC-8/GV download history.
- `download_extra_YYYYMMDD.log`: one-file-at-a-time Falcon/merge download
  history.

Completed on 2026-06-24:

| Date | Families | Files | Archive MiB | On-Disk Size |
|---|---|---:|---:|---:|
| 2012-05-29 | DC-8 103, GV 20, Falcon 17, merge 15 | 155 | 964.8 | 820M DC-8, 40M GV, 11M Falcon, 96M merge |
| 2012-05-30 | DC-8 93, GV 20, Falcon 20, merge 11 | 144 | 773.9 | 615M DC-8, 69M GV, 12M Falcon, 80M merge |

The local subset currently occupies about **1.8G** and completed cleanly with
zero `.part` files remaining.

## Suggested Products To Use First

Prioritize merged products for initial DAVINCI wiring:

- `MERGES/1_MINUTE.*_MRG` for quick time-series ingestion and plotting.
- `MERGES/10_SECOND.*_MRG` where higher time resolution is needed.
- `MERGES/1_SECOND.*_MRG` after the pipeline works on the smaller products.

Then add instrument-level files by platform:

- DC-8: MetNav/state, NO/NO2/NOx, O3, CO, CO2/CH4, CH2O, peroxides, aerosol
  products, and cloud/remote-sensing files as needed.
- GV: high-altitude outflow chemistry, water vapor, O3/NOx, peroxides, and
  VOC support products.
- DLR Falcon: outflow trace species and aerosol/cloud support products.

For storm-context datasets beyond the LaRC archive, look next at NCAR/EOL for
ground-based radar, sounding, and lightning products. The LaRC archive exposes
`LIGHTNING`, `MODEL`, `TRAJECTORY`, `O3SONDES`, `ANALYSIS`, and `SATELLITE`
buttons, but those views returned no file rows through the current endpoint.

## Practical DAVINCI Plan

1. Start with the 2012-05-29 `MERGES/1_MINUTE.DC8_MRG` and
   `MERGES/1_MINUTE.GV_MRG` products to validate ICARTT parsing, time handling,
   and aircraft track plotting.
2. Add 2012-05-30 merge files to test aged outflow continuity and downwind
   chemistry.
3. Add 10-second merge files once the one-minute pass is stable.
4. Add instrument-level DC-8, GV, and Falcon files only for variables needed by
   the first DAVINCI analysis.
5. Pull NCAR/EOL radar, sounding, and lightning products for the Kingfisher
   supercell if the analysis needs storm kinematics or flash-rate constraints.

## Notes / Caveats

- File counts and sizes are based on the archive listing as accessed on
  2026-06-24. NASA may update archive contents and revisions.
- The direct archive links are session-like encoded `enzFile` URLs. Prefer
  discovering current links from the archive page rather than hard-coding the
  exact encoded URLs.
- This local download currently includes the LaRC aircraft and merge products
  for the selected dates. It does not include external NCAR/EOL ground-based
  radar, sounding, or lightning products.
