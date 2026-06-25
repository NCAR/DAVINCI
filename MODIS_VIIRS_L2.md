# MODIS and VIIRS L2 Cloud and Aerosol Notes for DAVINCI

Working notes for adding MODIS and VIIRS Level-2 cloud and aerosol satellite
products to the ASIA-AQ, DC3, and FIREX-AQ campaign data stores.

## Access

- LAADS DAAC: https://ladsweb.modaps.eosdis.nasa.gov/
- LAADS API-V2 guide:
  https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/api-v2/quick-start-guide/
- LAADS API-V2 content spec:
  https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/api-v2/specs/content

LAADS search and metadata queries are reachable from this environment. Direct
file downloads require either a browser login or a bearer token. A direct test
download without authentication returned an HTML login page rather than an HDF
file.

After creating `~/.netrc` for `urs.earthdata.nasa.gov`, the direct OAuth
redirect path still ended with a LAADS OAuth callback error. The working
non-interactive path is:

1. Use the Earthdata User Tokens API with `.netrc` credentials to get a bearer
   token.
2. Use the bearer token in the LAADS download request:

```bash
curl -L -C - \
  --header "X-Requested-With: XMLHttpRequest" \
  --header "Authorization: Bearer ${EARTHDATA_TOKEN}" \
  -o /path/to/output/file \
  "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/<filename-or-archive-path>"
```

## Product Set

Selected products are the standard L2 aerosol and cloud products with the
broadest campaign utility:

| Local Dataset | LAADS Short Name | Platform | Product Type | Collection / ArchiveSet | Format |
|---|---|---|---|---|---|
| MODIS_TERRA_AEROSOL | MOD04_L2 | Terra MODIS | Aerosol | 061 / 61 | HDF-EOS |
| MODIS_AQUA_AEROSOL | MYD04_L2 | Aqua MODIS | Aerosol | 061 / 61 | HDF-EOS |
| MODIS_TERRA_CLOUD | MOD06_L2 | Terra MODIS | Cloud | 061 / 61 | HDF-EOS |
| MODIS_AQUA_CLOUD | MYD06_L2 | Aqua MODIS | Cloud | 061 / 61 | HDF-EOS |
| VIIRS_SNPP_AEROSOL_DEEPBLUE | AERDB_L2_VIIRS_SNPP | Suomi-NPP VIIRS | Deep Blue aerosol | 002 / 5200 | NetCDF4 |
| VIIRS_NOAA20_AEROSOL_DEEPBLUE | AERDB_L2_VIIRS_NOAA20 | NOAA-20 VIIRS | Deep Blue aerosol | 002 / 5200 | NetCDF4 |
| VIIRS_SNPP_AEROSOL_DARKTARGET | AERDT_L2_VIIRS_SNPP | Suomi-NPP VIIRS | Dark Target aerosol | 021 / 5201 | NetCDF4 |
| VIIRS_NOAA20_AEROSOL_DARKTARGET | AERDT_L2_VIIRS_NOAA20 | NOAA-20 VIIRS | Dark Target aerosol | 021 / 5201 | NetCDF4 |
| VIIRS_SNPP_CLOUD | CLDPROP_L2_VIIRS_SNPP | Suomi-NPP VIIRS | Cloud properties | 011 / 5111 | NetCDF4 |
| VIIRS_NOAA20_CLOUD | CLDPROP_L2_VIIRS_NOAA20 | NOAA-20 VIIRS | Cloud properties | 011 / 5111 | NetCDF4 |

NOAA-20 products are expected to be absent for DC3 because NOAA-20 did not
exist in 2012. Suomi-NPP VIIRS spans all three campaign windows.

## Search Windows

Manifests were generated for the existing campaign dates of interest rather
than full campaign windows, because full-window L2 cloud products would be very
large.

| Campaign | Dates | LAADS Region |
|---|---|---|
| ASIA-AQ | 2024-02-06, 2024-02-17, 2024-03-13, 2024-03-18 | `[BBOX]W95 E150 S0 N45` |
| DC3 | 2012-05-29, 2012-05-30 | `[BBOX]W-105 E-88 S25 N42` |
| FIREX-AQ | 2019-08-06, 2019-08-07, 2019-08-08 | `[BBOX]W-126 E-105 S35 N53` |

The boxes are intentionally broad campaign-region filters. They select granules
intersecting the analysis areas without pulling whole global days.

## Local Layout

Manifests live under each campaign data root:

```text
/glade/work/fillmore/Data/<CAMPAIGN>/MODIS_VIIRS_L2/<dataset>/<YYYYMMDD>/manifest_<product>_<YYYYMMDD>.tsv
```

When downloads are enabled, data files should land beside their daily manifest:

```text
/glade/work/fillmore/Data/<CAMPAIGN>/MODIS_VIIRS_L2/<dataset>/<YYYYMMDD>/<filename>
```

Combined campaign manifests:

```text
/glade/work/fillmore/Data/ASIA-AQ/MODIS_VIIRS_L2/manifest_all_selected_dates.tsv
/glade/work/fillmore/Data/DC3/MODIS_VIIRS_L2/manifest_all_selected_dates.tsv
/glade/work/fillmore/Data/FIREX-AQ/MODIS_VIIRS_L2/manifest_all_selected_dates.tsv
```

Cross-campaign summary:

```text
/glade/work/fillmore/Data/MODIS_VIIRS_L2_manifest_summary.tsv
```

## Manifest Inventory

The manifest total is **1,795 files** and about **103.6 GiB** for the selected
dates and domains. Aerosol products are modest; cloud products dominate.

| Campaign | Files | Manifest Size |
|---|---:|---:|
| ASIA-AQ | 1,050 | 66.16 GiB |
| DC3 | 184 | 7.82 GiB |
| FIREX-AQ | 561 | 29.59 GiB |

By product family:

| Dataset | Files | Manifest Size |
|---|---:|---:|
| MODIS_AQUA_AEROSOL | 106 | 0.41 GiB |
| MODIS_TERRA_AEROSOL | 110 | 0.43 GiB |
| VIIRS_NOAA20_AEROSOL_DEEPBLUE | 74 | 0.66 GiB |
| VIIRS_NOAA20_AEROSOL_DARKTARGET | 146 | 1.43 GiB |
| VIIRS_SNPP_AEROSOL_DEEPBLUE | 285 | 2.60 GiB |
| VIIRS_SNPP_AEROSOL_DARKTARGET | 364 | 3.70 GiB |
| MODIS_AQUA_CLOUD | 207 | 11.65 GiB |
| MODIS_TERRA_CLOUD | 202 | 12.44 GiB |
| VIIRS_NOAA20_CLOUD | 138 | 31.17 GiB |
| VIIRS_SNPP_CLOUD | 163 | 39.07 GiB |

## Download Status

Completed on 2026-06-24:

| Product Block | Files | Size | Status |
|---|---:|---:|---|
| Aerosol products | 1,085 | 9.23 GiB | Downloaded and audited with zero missing files and zero size mismatches |
| MODIS cloud products | 409 | 24.09 GiB | Not bulk-downloaded; one MODIS cloud HDF4 probe downloaded successfully |
| VIIRS `CLDPROP` cloud products | 301 | 70.24 GiB | Not bulk-downloaded; one VIIRS cloud NetCDF/HDF5 probe downloaded successfully |

Downloaded aerosol files live beside their manifests:

```text
/glade/work/fillmore/Data/<CAMPAIGN>/MODIS_VIIRS_L2/<aerosol-dataset>/<YYYYMMDD>/<filename>
```

Aerosol download log:

```text
/glade/work/fillmore/Data/MODIS_VIIRS_L2_aerosol_download.log
```

## Download Plan

Recommended first tranche once a token is available:

1. Download all aerosol products first. This has been completed.
2. Download MODIS cloud products next. They add about **24.09 GiB** and use
   the same HDF-EOS family as the existing MODIS AOD reader path.
3. Download VIIRS `CLDPROP` last. It adds about **70.24 GiB** and is the
   largest block by far.

The manifest `download_url` column already contains the API archive URL for
each granule.

## DAVINCI Reader Notes

- The repo already has a working MODIS L2 AOD path for `MOD04_L2` and
  `MYD04_L2` through `modis_l2_aod`.
- The catalog-driven `modis_viirs` reader currently covers MODIS monthly L3 AOD
  (`MOD08_M3` / `MYD08_M3`), not these L2 products.
- MODIS cloud and VIIRS L2 cloud/aerosol products will need reader/catalog work
  or an adapter before they are first-class DAVINCI pipeline sources.
