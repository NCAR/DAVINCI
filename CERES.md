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

## ASDC Archive Staging

The read-only ASDC archive roots are:

```text
/ASDC_archive/GMAO/GEOSIT
/ASDC_archive/MODIS
```

Both are backed by `/CERES_prd` (`GMAO` and `MODIS` are symbolic links under
`/ASDC_archive`). The source files must remain unmodified.

The resolved production paths are:

```text
/ASDC_archive/GMAO/GEOSIT -> /CERES_prd/GMAO/GEOSIT
/ASDC_archive/GMAO/MERRA2 -> /CERES_prd/GMAO/MERRA2
/ASDC_archive/MODIS        -> /CERES_prd/MODIS
```

`/CERES_prd` is the target of the ASDC aliases, not an alternate warm copy.
Changing only the path prefix does not bypass cold archive storage.

### Observed Source Hierarchy

```text
/ASDC_archive/
├── GMAO/
│   └── GEOSIT/<YYYY>/<MM>/
│       ├── GEOS*.nc4
│       └── GEOS*.nc4.xml
└── MODIS/
    ├── Aqua/C7/<YYYY>/<DDD>/
    │   ├── MYD0203_SS*.nc
    │   └── MYD04_L2*.nc
    ├── Terra/C7/<YYYY>/<DDD>/
    │   ├── MOD0203_SS*.nc
    │   └── MOD04_L2*.nc
    └── LAND/
        └── C6|C61/<YYYY>/MCD43C1*.hdf
```

GEOSIT spans yearly directories with zero-padded monthly subdirectories and
stores native NetCDF4 (`.nc4`) files with XML sidecars. MODIS Aqua and Terra
Collection 7 data use a three-digit day-of-year directory; MODIS LAND has
Collection 6 and 6.1 year directories, with observed `MCD43C1` HDF files.

The observed Aqua/Terra Collection 7 files are Level 2. The daily one-degree
MODIS AOD subset scripts use the C6.1 daily Level 3 inputs staged under the
same platform/year/day convention:

```text
/ASDC_archive/MODIS/Terra/C61/<YYYY>/<DDD>/MOD08_D3.*.hdf
/ASDC_archive/MODIS/Aqua/C61/<YYYY>/<DDD>/MYD08_D3.*.hdf
```

### Subset Output Layout

The writable subset root is:

```text
/CERES/sarb/dfillmor/DAVINCI
```

The subset scripts preserve each source-relative path below this root:

```text
/CERES/sarb/dfillmor/DAVINCI/
├── GMAO/GEOSIT/<YYYY>/<MM>/GEOSIT_TOTEXTTAU550_daily.<YYYYMMDD>.nc
└── MODIS/
    ├── Aqua/C61/<YYYY>/<DDD>/MYD08_D3.*.nc
    ├── Terra/C61/<YYYY>/<DDD>/MOD08_D3.*.nc
    └── LAND/C6|C61/<YYYY>/<subset-file>.nc
```

The scripts run with `conda activate nco` and use NCO operators for the
subsets. The GEOS-IT product contains only total 550-nm aerosol extinction
optical thickness (`TOTEXTTAU`). For C6.1 MODIS HDF4 reads, the MODIS script
defaults to `/usr/local/bin/ncks` (`NCKS_HDF4_BIN` can override it), which
writes only the final NetCDF4 subset to the mirrored output tree.

Known unreadable source files are treated as missing data before an NCO reader
opens them. The default lists are
`/CERES/sarb/dfillmor/DAVINCI/GEOSIT_SKIPPED_SOURCE_FILES.txt` and
`/CERES/sarb/dfillmor/DAVINCI/MODIS_SKIPPED_HDF_FILES.txt`. When a day has
multiple unlisted MODIS production files, the script selects the
lexicographically latest filename, which is the latest production timestamp in
the C6.1 naming convention.

### Archive Access Policy

Do not process cold ASDC archive files from this system. A July 2008
full-field test completed promptly for GEOS-IT, Terra, and Aqua, while sampled
2000 and 2009 GEOS-IT files entered uninterruptible filesystem I/O. A sampled MERRA-2
`tavg1_2d_aer_Nx` file from 1980 behaved the same way. This behavior is
file-specific; there is no assumed warm-date cutoff or separate warm
`/CERES_prd` path.

Before starting a production year, perform a temporary full-field NCO preflight
for one GEOS-IT `TOTEXTTAU` file and one Terra/Aqua D3 file when available. A
source that does not complete within the preflight limit (currently 60 seconds)
is unavailable through the local archive. Do not repeatedly retry it here. Add
the exact path to the appropriate skip list and treat that data as missing.

Run only preflight-approved years, one calendar year per batch. Do not launch a
single multi-year archive command:

```bash
scripts/subset_geosit_aod_daily.sh \
  --start <YYYY>-01-01 --end <YYYY>-12-31 --allow-missing-day
scripts/subset_modis_d3_aod_daily.sh \
  --start <YYYY>-01-01 --end <YYYY>-12-31
```

### External Acquisition

Cold-source acquisition will run on a different system. That system downloads
inputs to local temporary storage, runs NCO against local files, and transfers
only final NetCDF subsets to the DAVINCI output tree.

- MERRA-2 aerosol inputs (`M2T1NXAER` / `tavg1_2d_aer_Nx`) are acquired from
  NASA Goddard's GES DISC using Earthdata Login and `earthaccess`.
  See [MERRA-2 data access](https://gmao.gsfc.nasa.gov/gmao-products/merra-2/data-access_merra-2/)
  and the [earthaccess access guide](https://earthaccess.readthedocs.io/en/v0.15.1/howto/access-data/).
- MODIS C6.1 `MOD08_D3` and `MYD08_D3` inputs are acquired from LAADS DAAC.
  Use a LAADS token for its download endpoint; an Earthdata Download Token also
  works but may be slower. See the [LAADS MOD08_D3 archive](https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD08_D3/).
- Do not stream MODIS HDF4 directly into NCO. Convert/subset the locally staged
  file in temporary storage, then remove the temporary source and intermediate
  files after validation.

### Derecho Earthaccess Queue

DAVINCI includes a PBS-array acquisition path for the daily AOD inputs:

```bash
scripts/qsub_aod_earthdata.sh \
  --start 2019-08-01 --end 2019-08-31 \
  --products merra2,terra,aqua
```

The submitter creates one array task per calendar month and defaults to at most
two simultaneous tasks on `casper@casper-pbs`. Each worker authenticates from
`~/.netrc`; credentials and bearer tokens are not passed through PBS variables
or written to logs. Use `--search-only` for a queued CMR inventory and size
estimate before downloading, or `--print-only` to inspect the `qsub` command.

Raw files default to purgeable Derecho scratch storage:

```text
/glade/derecho/scratch/<user>/DAVINCI-AOD/raw/
├── GMAO/MERRA2/<YYYY>/<MM>/MERRA2_*.tavg1_2d_aer_Nx.<YYYYMMDD>.nc4
└── MODIS/
    ├── Terra/C61/<YYYY>/<DDD>/MOD08_D3*.hdf
    └── Aqua/C61/<YYYY>/<DDD>/MYD08_D3*.hdf
```

Terra/Aqua Level-2 aerosol swaths are optional because they add roughly 300
granules per day across both platforms. Include them explicitly when needed:

```bash
scripts/qsub_aod_earthdata.sh \
  --start 2019-08-01 --end 2019-08-31 \
  --products merra2,terra,aqua,terra-l2,aqua-l2
```

The L2 keys stage `MOD04_L2` and `MYD04_L2` Collection 6.1 files alongside the
daily products under the same Terra/Aqua year/day hierarchy. The downloader
retains every acquisition but selects only the latest production revision for
each platform, date, and acquisition time. The default `all` selection excludes
L2 to prevent an unintended high-volume swath download.

Every task writes a TSV under `<root>/manifests/` recording CMR size, local
size, source URL, destination, and whether each file was downloaded, already
present, or missing. The downloader filters out adjacent-day granules that CMR
MODIS searches can return. Re-running a task skips existing non-empty files.

### Derecho AOD Subsetting

Create compact NetCDF4 subsets from the purgeable scratch inputs with:

```bash
scripts/subset_aod_earthdata.py \
  --start 2008-07-01 --end 2008-07-31 \
  --products merra2,terra,aqua,terra-l2,aqua-l2
```

The direct script defaults to the Earthaccess staging root above and writes to:

```text
/glade/work/fillmore/Data/CERES-SARB-CAM7/AOD_SUBSETS/
├── GMAO/MERRA2/<YYYY>/<MM>/MERRA2_TOTEXTTAU_daily.<YYYYMMDD>.nc4
└── MODIS/<Terra|Aqua>/C61/<YYYY>/<DDD>/*.AOD550.nc4
```

MERRA-2 outputs contain the daily mean of the 24 hourly `TOTEXTTAU` samples.
MODIS D3 outputs contain the combined Dark Target/Deep Blue 550-nm daily mean.
MODIS L2 outputs contain the combined 550-nm AOD, QA and algorithm flags,
latitude, longitude, and scan start time. Optional latitude/longitude bounds
apply to all selected products; L2 granules outside the requested region are
recorded as `outside-bounds` and are not written.

For the full L2 inventory, submit one throttled PBS array task per UTC day:

```bash
scripts/qsub_aod_subsets.sh \
  --start 2008-07-01 --end 2008-07-31 \
  --products merra2,terra,aqua,terra-l2,aqua-l2
```

The subsetter's `all` selection includes all five products. Outputs are written
atomically, existing non-empty outputs are skipped unless `--overwrite` is
used, and each task writes a TSV under `<output-root>/manifests/`.

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
