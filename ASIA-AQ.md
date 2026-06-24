# ASIA-AQ Dataset Notes for DAVINCI

Working notes for evaluating the NASA ASIA-AQ field campaign data in DAVINCI.

## Access

- Mission page: https://www-air.larc.nasa.gov/missions/asia-aq/
- Public archive page: https://www-air.larc.nasa.gov/cgi-bin/ArcView/asiaaq
- Flight calendar PDF: https://www-air.larc.nasa.gov/missions/asia-aq/docs/ASIAAQCalendar.pdf
- AWS registry page: https://registry.opendata.aws/nasa-asia-aq/

The NASA LaRC ASIA-AQ archive is reachable from this environment. Public
archive files are downloadable through `/cgi-bin/enzFile?...` links. A spot
check of an `.ict` file returned readable ICARTT header content, and a spot
check of a ZIP file returned `HTTP 200 OK` with `Content-Disposition:
attachment`.

The Custom Data Merging Tool is also reachable, but it is gated by a
User ID/password form. Use direct archive downloads for unauthenticated access,
or provide credentials if the merge workflow is needed.

The AWS registry notes that ASIA-AQ data on AWS requires an Earthdata Login
account and controlled-access AWS credentials.

## Total Archive Volume

The visible archive rows report file size in `Size (KB)`. Summing the listed
file rows across platform views gives:

| Platform | Files | Archive KB | GiB |
|---|---:|---:|---:|
| DC8_AIRCRAFT | 1,388 | 3,648,615.5 | 3.480 |
| KINGAIR-1900D | 90 | 81,864.6 | 0.078 |
| KINGAIR-G90GT | 54 | 45,106.6 | 0.043 |
| KINGAIR-350HW | 25 | 16,209.0 | 0.015 |
| LARC-G3-AIRCRAFT | 244 | 11,490,974.3 | 10.959 |
| RV-GISANG | 35 | 2,674.8 | 0.003 |
| GROUND | 42 | 259,165.9 | 0.247 |
| SONDES | 0 | 0.0 | 0.000 |
| MODEL | 79 | 6,218,324.9 | 5.930 |
| ANALYSIS | 1 | 1,613.7 | 0.002 |
| SATELLITE | 610 | 702,305,732.2 | 669.771 |
| **Total** | **2,568** | **724,070,281.5** | **690.527** |

Equivalent total volume is about **741.4 GB decimal** or **0.674 TiB**.
Satellite products dominate the volume.

## Recommended DAVINCI Starter Subset

Start with a four-day cross-region subset that covers several campaign regimes
while keeping the first pass manageable. Aircraft and G-III files are small
enough for quick iteration. Add satellite products only after the aircraft
pipeline path is working.

| Date | Region / Case | Platform Coverage | Approx Volume |
|---|---|---|---:|
| 2024-02-06 | Philippines, weak NE monsoon | DC-8 + LaRC G-III + satellite | 0.75 GiB non-sat, 8.58 GiB with satellite |
| 2024-02-17 | Korea, local pollution; richest multi-platform day | DC-8 + LaRC G-III + King Airs + RV-GISANG + satellite | 0.80 GiB non-sat, 8.99 GiB with satellite |
| 2024-03-13 | Taiwan, frontal case; possible pollution transport aloft | DC-8 + LaRC G-III + RV-GISANG + satellite | 0.72 GiB non-sat, 8.95 GiB with satellite |
| 2024-03-18 | Thailand, Bangkok/Chiang Mai smoke case | DC-8 + LaRC G-III + satellite | 0.85 GiB non-sat, 9.09 GiB with satellite |

Estimated four-day total:

- Aircraft/G-III/non-satellite only: about **3.1 GiB**
- Including all listed satellite products for those dates: about **35.6 GiB**

## Local Download Inventory

Local data root:

```text
/glade/work/fillmore/Data/ASIA-AQ
```

Directory convention:

```text
/glade/work/fillmore/Data/ASIA-AQ/<platform>/<YYYYMMDD>/<PI-or-product>/<filename>
```

Each downloaded day also has a manifest and log in the data root:

- `manifest_YYYYMMDD.tsv` with platform, date, product/group, filename,
  archive size, and current archive URL.
- `download_YYYYMMDD.log` with the one-file-at-a-time download history.

Completed on 2026-06-24:

| Date | Files | Platforms | On-Disk Size |
|---|---:|---|---:|
| 2024-02-06 | 103 | DC-8 84, LaRC G-III 12, satellite 7 | 221M DC-8, 550M G-III, 7.9G satellite |
| 2024-02-17 | 125 | DC-8 84, LaRC G-III 13, King Air 18, RV-GISANG 2, satellite 8 | 208M DC-8, 595M G-III, 14M King Air, 178K RV, 8.2G satellite |
| 2024-03-13 | 102 | DC-8 82, LaRC G-III 10, RV-GISANG 2, satellite 8 | 179M DC-8, 564M G-III, 178K RV, 8.3G satellite |
| 2024-03-18 | 102 | DC-8 81, LaRC G-III 13, satellite 8 | 219M DC-8, 657M G-III, 8.3G satellite |

The local subset currently occupies about **36G**. All four downloaded days
completed cleanly with zero `.part` files remaining.

Satellite products included for the selected dates are the listed archive
products for each day: `FUSION.PRODUCT`, `GEMS.AEH`, `GEMS.AOD`, `GEMS.CH2O`,
`GEMS.NO2`, `GEMS.O3P`, and `GEMS.SO2`. The 2024-02-17, 2024-03-13, and
2024-03-18 manifests also include `GOCI-II.AOP`; 2024-02-06 did not expose that
product in the archive listing used for this download.

## Optional Expanded Six-Day Subset

If the first four days work cleanly, add:

| Date | Reason | Approx Volume |
|---|---|---:|
| 2024-02-13 | Philippines, strong NE monsoon follow-up with DC-8 + G-III overlap | 0.85 GiB non-sat, 8.06 GiB with satellite |
| 2024-03-25 | Thailand, later smoke / flow evolution case with DC-8 + G-III overlap | 0.80 GiB non-sat, 9.05 GiB with satellite |

Estimated six-day total:

- Non-satellite only: about **4.8 GiB**
- Including all listed satellite products: about **52.7 GiB**

## Suggested Products To Pull First

Prioritize these before adding the larger satellite volume:

- DC-8 MetNav for time, location, altitude, and aircraft state.
- DC-8 trace gases:
  - O3 from ROZE / NOxO3-related products.
  - NO2 from CANOE.
  - CO, CH4, CO2 from DACOM.
  - CH2O from ISAF.
  - VOCs from TOGA/WAS/PTR-MS only after the core pipeline is working.
- DC-8 analysis flags, especially smoke, combustion, and air-mass-layer flags.
- LaRC G-III GCAS and HSRL2 for remote-sensing context.
- Korea 2024-02-17 only: King Air and RV-GISANG files for multi-platform tests.

For satellite workflows, add a small set of GEMS products on the same dates:

- GEMS NO2
- GEMS AOD
- GEMS CH2O

Avoid starting with every satellite product. The full satellite archive is
about 670 GiB, and each selected day contributes roughly 7-8+ GiB.

## Flight Calendar Context

Key notes from the NASA ASIA-AQ flight calendar:

- 2024-02-06: Philippines DC-8 flight and G-III sorties during weak NE monsoon.
- 2024-02-17: Korea DC-8 flight, two G-III sorties, and Hanseo aircraft flight
  during local pollution.
- 2024-03-13: Taiwan transit/flight case, behind a frontal system, easterly
  surface winds, westerly winds aloft, possible pollution transport aloft.
- 2024-03-18: Thailand DC-8 flight with two G-III sorties/three rasters; Bangkok
  and Chiang Mai smoke conditions noted around AOD approximately 1.8.
- 2024-03-25: Thailand DC-8 flight with two G-III sorties/three rasters; Bangkok
  afternoon clouds and Chiang Mai clear sky.

## Practical DAVINCI Plan

1. Build the first test configuration using only DC-8 MetNav plus one or two
   core trace gases for 2024-02-06.
2. Add the remaining three recommended dates once ICARTT ingestion and flight
   track plotting are stable.
3. Add LaRC G-III files for the same dates to test remote-sensing integration.
4. Add GEMS NO2/AOD/CH2O for the same dates only after aircraft analysis works.
5. Use 2024-02-17 for multi-platform pairing tests because it has the richest
   overlap across DC-8, G-III, King Airs, and RV-GISANG.

## Notes / Caveats

- File counts and sizes above are based on the archive listing as accessed on
  2026-06-24. NASA may update archive contents and revisions.
- Preliminary files use alphabetic revisions such as `RA` or `RB`; numeric
  revisions such as `R0`, `R1`, and `R2` indicate QA/QC-reviewed files per the
  archive guidance.
- The direct archive links are session-like encoded `enzFile` URLs. Prefer
  discovering current links from the archive page rather than hard-coding the
  exact encoded URLs.
