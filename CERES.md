# CERES Dataset Notes for DAVINCI

Working notes for adding CERES Single Scanner Footprint (SSF) data to the
ASIA-AQ, DC3, and FIREX-AQ campaign data stores.

## Access

- CERES data products: https://ceres.larc.nasa.gov/data/
- CERES documentation: https://ceres.larc.nasa.gov/data/documentation/
- NASA Earthdata / CMR granule metadata:
  https://cmr.earthdata.nasa.gov/search/
- ASDC protected data host:
  https://data.asdc.earthdata.nasa.gov/

Downloads used the existing `~/.netrc` Earthdata credentials to create an
Earthdata bearer token through the User Tokens API, then streamed protected
ASDC URLs returned by CMR.

## Product Scope

This first CERES pull stages SSF products for the existing campaign dates of
interest:

| Campaign | Dates | SSF Product |
|---|---|---|
| ASIA-AQ | 2024-02-06, 2024-02-17, 2024-03-13, 2024-03-18 | `CER_SSF_NOAA20-FM6-VIIRS` Edition1C |
| DC3 | 2012-05-29, 2012-05-30 | `CER_SSF_Terra-FM1-MODIS` Edition4A and `CER_SSF_Aqua-FM3-MODIS` Edition4A |
| FIREX-AQ | 2019-08-06, 2019-08-07, 2019-08-08 | `CER_SSF_Terra-FM1-MODIS` Edition4A and `CER_SSF_Aqua-FM3-MODIS` Edition4A |

NPP-FM5 VIIRS was not included in this first SSF pass. The scope follows the
initial product recommendation: use modern NOAA-20 VIIRS SSF for ASIA-AQ and
Terra/Aqua MODIS SSF for the older DC3 and FIREX-AQ cases.

## Local Layout

Campaign-specific SSF files live under each campaign root:

```text
/glade/work/fillmore/Data/<CAMPAIGN>/CERES_SSF/<dataset>/<YYYYMMDD>/<filename>
```

Each daily directory contains a manifest:

```text
/glade/work/fillmore/Data/<CAMPAIGN>/CERES_SSF/<dataset>/<YYYYMMDD>/manifest_<short-name>_<YYYYMMDD>.tsv
```

Combined manifests:

```text
/glade/work/fillmore/Data/ASIA-AQ/CERES_SSF/manifest_all_selected_dates.tsv
/glade/work/fillmore/Data/DC3/CERES_SSF/manifest_all_selected_dates.tsv
/glade/work/fillmore/Data/FIREX-AQ/CERES_SSF/manifest_all_selected_dates.tsv
/glade/work/fillmore/Data/CERES_SSF_manifest_summary.tsv
```

Download log:

```text
/glade/work/fillmore/Data/CERES_SSF_download.log
```

## Local Inventory

Completed on 2026-06-24:

| Campaign | Files | On-Disk Size |
|---|---:|---:|
| ASIA-AQ | 92 | 5.3G |
| DC3 | 96 | 5.8G |
| FIREX-AQ | 144 | 8.8G |

Total local SSF inventory is **332 files** and about **19.7 GiB** on disk.
The CMR size sum is **19.8 GiB**.

By daily product:

| Campaign | Date | Dataset | Files | CMR Size |
|---|---|---|---:|---:|
| ASIA-AQ | 2024-02-06 | CERES_SSF_NOAA20_FM6_VIIRS | 23 | 1.31 GiB |
| ASIA-AQ | 2024-02-17 | CERES_SSF_NOAA20_FM6_VIIRS | 23 | 1.30 GiB |
| ASIA-AQ | 2024-03-13 | CERES_SSF_NOAA20_FM6_VIIRS | 23 | 1.31 GiB |
| ASIA-AQ | 2024-03-18 | CERES_SSF_NOAA20_FM6_VIIRS | 23 | 1.31 GiB |
| DC3 | 2012-05-29 | CERES_SSF_TERRA_FM1_MODIS | 24 | 1.45 GiB |
| DC3 | 2012-05-29 | CERES_SSF_AQUA_FM3_MODIS | 24 | 1.44 GiB |
| DC3 | 2012-05-30 | CERES_SSF_TERRA_FM1_MODIS | 24 | 1.43 GiB |
| DC3 | 2012-05-30 | CERES_SSF_AQUA_FM3_MODIS | 24 | 1.50 GiB |
| FIREX-AQ | 2019-08-06 | CERES_SSF_TERRA_FM1_MODIS | 24 | 1.44 GiB |
| FIREX-AQ | 2019-08-06 | CERES_SSF_AQUA_FM3_MODIS | 24 | 1.53 GiB |
| FIREX-AQ | 2019-08-07 | CERES_SSF_TERRA_FM1_MODIS | 24 | 1.45 GiB |
| FIREX-AQ | 2019-08-07 | CERES_SSF_AQUA_FM3_MODIS | 24 | 1.46 GiB |
| FIREX-AQ | 2019-08-08 | CERES_SSF_TERRA_FM1_MODIS | 24 | 1.44 GiB |
| FIREX-AQ | 2019-08-08 | CERES_SSF_AQUA_FM3_MODIS | 24 | 1.45 GiB |

## Audit

The final manifest audit found:

| Check | Result |
|---|---:|
| Manifest rows | 332 |
| Missing files | 0 |
| Empty files | 0 |
| Unexpected size mismatches | 0 |
| Remaining `.part` files | 0 |

Two Aqua-FM3 granules have CMR `granule_size` values roughly double the ASDC
HTTP `Content-Length`. The local files match ASDC `Content-Length` and open as
HDF4:

```text
/glade/work/fillmore/Data/DC3/CERES_SSF/CERES_SSF_AQUA_FM3_MODIS/20120530/CER_SSF_Aqua-FM3-MODIS_Edition4A_400403.2012053002
/glade/work/fillmore/Data/FIREX-AQ/CERES_SSF/CERES_SSF_AQUA_FM3_MODIS/20190806/CER_SSF_Aqua-FM3-MODIS_Edition4A_404405.2019080602
```

The mismatch details are saved in:

```text
/glade/work/fillmore/Data/CERES_SSF_audit_known_cmr_size_mismatch.tsv
```

## DAVINCI Reader Notes

DAVINCI already has a `ceres_ssf` reader. Useful canonical variables include:

- `toa_lw_up`
- `toa_sw_up`
- `toa_solar_in`
- `sfc_sw_down`
- `sfc_sw_down_clr`
- `sfc_lw_down`
- `sfc_lw_down_clr`
- `sfc_sw_net`
- `sfc_lw_net`

Terra/Aqua Edition4A files are HDF4. NOAA-20 Edition1C files are NetCDF/HDF5.
The reader handles both formats, but one `open()` call should not mix formats;
open each SSF product family separately.

Reader validation was run against NOAA-20, Terra, Aqua, and the two Aqua CMR
size-mismatch granules listed above. The reader now handles the real SSF naming
variants found in these files: HDF4 `Time of observation`, HDF4 `Model B`
surface flux SDS names, and netCDF `model_b_*` surface flux variables.
