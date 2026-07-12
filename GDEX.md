# GDEX Datasets for DAVINCI on Derecho

Last checked: 2026-07-12 on NCAR Derecho.

## Local GDEX Roots

On Derecho, GDEX data is mounted locally. Prefer the short path in configs and
notes:

```text
/gdex/data
```

Equivalent paths observed on Derecho:

```text
/gdex/data
/glade/campaign/collections/gdex/data
/glade/campaign/collections/rda/data
```

For ERA5, all three resolved to the same collection directory during this check:

```text
/gdex/data/d633000
-> /glade/campaign/collections/gdex/data/d633000
```

This means DAVINCI can usually read staged GDEX NetCDF directly on Derecho
without downloading ERA5 from CDS first.

## ERA5: `d633000`

Primary dataset for DAVINCI:

- GDEX ID: `d633000`
- Product: ERA5 Reanalysis on a 0.25 degree latitude-longitude grid
- Public page: <https://gdex.ucar.edu/datasets/d633000/>
- DOI listed by GDEX: `10.5065/BH6N-5N20`
- Format on Derecho: NetCDF4 files, plus zarr/catalog/kerchunk products
- Time coverage reported by GDEX when checked: 1940 through 2026-03-31, with
  monthly updates

High-value local subdirectories:

```text
/gdex/data/d633000/e5.oper.an.sfc
/gdex/data/d633000/e5.oper.an.pl
/gdex/data/d633000/e5.oper.fc.sfc.accumu
/gdex/data/d633000/e5.oper.fc.sfc.meanflux
/gdex/data/d633000/e5.oper.an.sfc.zarr
/gdex/data/d633000/catalogs
/gdex/data/d633000/kerchunk
```

### Surface Analysis

Surface analysis files are hourly, monthly, and variable-specific.

Example:

```text
/gdex/data/d633000/e5.oper.an.sfc/202501/e5.oper.an.sfc.128_167_2t.ll025sc.2025010100_2025013123.nc
```

Observed xarray metadata for the sample `2t` file:

```text
dims: time=744, latitude=721, longitude=1440
variable: VAR_2T
long_name: 2 metre temperature
units: K
coords: latitude, longitude, time
```

Useful surface-analysis variables for DAVINCI:

| ERA5 short name | NetCDF variable | Meaning | Units |
| --- | --- | --- | --- |
| `2t` | `VAR_2T` | 2 metre temperature | K |
| `10u` | `VAR_10U` | 10 metre U wind component | m s**-1 |
| `10v` | `VAR_10V` | 10 metre V wind component | m s**-1 |
| `blh` | `BLH` | Boundary layer height | m |
| `msl` | `MSL` | Mean sea level pressure | Pa |
| `sp` | `SP` | Surface pressure | Pa |
| `tcwv` | `TCWV` | Total column water vapour | kg m**-2 |
| `tco3` | `TCO3` | Total column ozone | kg m**-2 |

### Pressure Levels

Pressure-level analysis files are hourly, daily, and variable-specific.

Example:

```text
/gdex/data/d633000/e5.oper.an.pl/202501/e5.oper.an.pl.128_203_o3.ll025sc.2025010100_2025010123.nc
```

Observed xarray metadata for the sample `o3` pressure-level file:

```text
dims: time=24, level=37, latitude=721, longitude=1440
variable: O3
long_name: Ozone mass mixing ratio
units: kg kg**-1
coords: latitude, level, longitude, time
```

Useful pressure-level variables for DAVINCI:

| ERA5 short name | NetCDF variable | Meaning | Units |
| --- | --- | --- | --- |
| `t` | `T` | Temperature | K |
| `u` | `U` | U component of wind | m s**-1 |
| `v` | `V` | V component of wind | m s**-1 |
| `q` | `Q` | Specific humidity | kg kg**-1 |
| `o3` | `O3` | Ozone mass mixing ratio | kg kg**-1 |
| `z` | `Z` | Geopotential | m**2 s**-2 |

Reader note: DAVINCI's ERA5 reader should rename the pressure-level dimension
from `level` to `z` for consistency with existing vertical extraction logic. If
the geopotential variable is named `Z` or `z`, avoid colliding with the vertical
dimension name.

### Forecast Accumulations And Fluxes

Precipitation and flux variables live in ERA5 forecast products, not surface
analysis.

Observed local directories:

```text
/gdex/data/d633000/e5.oper.fc.sfc.accumu
/gdex/data/d633000/e5.oper.fc.sfc.meanflux
```

Examples from January 2025:

```text
/gdex/data/d633000/e5.oper.fc.sfc.accumu/202501/e5.oper.fc.sfc.accumu.128_143_cp.ll025sc.2025010106_2025011606.nc
/gdex/data/d633000/e5.oper.fc.sfc.meanflux/202501/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc.2025010106_2025011606.nc
```

Useful forecast/flux variables:

| ERA5 short name | NetCDF variable | Meaning | Units |
| --- | --- | --- | --- |
| `cp` | `CP` | Convective precipitation | m |
| `sf` | `SF` | Snowfall | m of water equivalent |
| `mtpr` | `MTPR` | Mean total precipitation rate | kg m**-2 s**-1 |
| `mslhf` | `MSLHF` | Mean surface latent heat flux | W m**-2 |
| `msshf` | `MSSHF` | Mean surface sensible heat flux | W m**-2 |

## ERA5 Related GDEX Datasets

These are present or relevant to the ERA5 family:

| GDEX ID | Product | Local status checked | Notes |
| --- | --- | --- | --- |
| `d633000` | ERA5 hourly pressure/surface products | present | Best first target for DAVINCI |
| `d633001` | ERA5 monthly means | present | Useful for climatology and long-period summaries |
| `d633006` | ERA5 model levels | present | Larger and out of first-pass DAVINCI scope |
| `d633008` | ERA5-Land hourly | public GDEX page checked | Land-focused 0.1 degree product |

Links:

- ERA5 hourly: <https://gdex.ucar.edu/datasets/d633000/>
- ERA5 monthly means: <https://gdex.ucar.edu/datasets/d633001/>
- ERA5 model levels: <https://gdex.ucar.edu/datasets/d633006/>
- ERA5-Land: <https://gdex.ucar.edu/datasets/d633008/>

## Other GDEX Datasets Relevant To DAVINCI

### NCAR/MOPITT CO Reanalysis

- GDEX ID: `d342000`
- Public page: <https://gdex.ucar.edu/datasets/d342000/>
- Relevance: chemistry and atmospheric composition, especially CO evaluation

### MERRA-2

MERRA-2 is available in GDEX, but the mounted products do not include the daily
aerosol collection needed for DAVINCI AOD evaluation (`M2T1NXAER` /
`tavg1_2d_aer_Nx`). In particular, a search of the mounted archive found no
`TOTEXTTAU`, `M2T1NXAER`, or `tavg1_2d_aer_Nx` files. Continue to acquire that
collection from NASA GES DISC with Earthdata authentication.

MERRA-2 datasets currently mounted on Derecho include:

| GDEX ID | Product | Derecho path | DAVINCI relevance |
| --- | --- | --- | --- |
| `d927358` | Native-resolution `M2TMNXSLV` monthly single-level diagnostics, reorganized by variable | `/gdex/data/d927358` | Monthly meteorology; no aerosol AOD |
| `d313003` | MERRA-2 global atmospheric forcing at native and remapped resolutions | `/gdex/data/d313003` | CESM/WRF meteorological forcing; no aerosol fields |
| `d313002` | MERRA-2 forcing interpolated to the CESM2 grid and vertical levels | `/gdex/data/d313002` | CESM2 meteorological forcing; no aerosol fields |
| `d651012` | ARTMIP Tier 1 MERRA-2 source data and atmospheric-river catalogs | `/gdex/data/d651012` | IVT/IWV and related AR diagnostics |
| `d651018` | ARTMIP Tier 2 reanalysis source data and catalogs | `/gdex/data/d651018` | MERRA-2 IVT/IWV, pressure, and winds; no aerosol AOD |

GDEX pages:

- <https://gdex.ucar.edu/datasets/d927358/>
- <https://gdex.ucar.edu/datasets/d313003/>
- <https://gdex.ucar.edu/datasets/d313002/>
- <https://gdex.ucar.edu/datasets/d651012/>
- <https://gdex.ucar.edu/datasets/d651018/>

The `d927358` convenience copy is small enough for direct use (about 16 GB) and
contains 1980-present variable-specific NetCDF4 files. The forcing and ARTMIP
datasets are much larger derived products and should not be treated as a
substitute for the native daily aerosol collection.

Current DAVINCI MERRA-2 collection names in `davinci_monet/io/download/merra2.py`:

| Friendly name | Earthdata short name | DAVINCI staging subpath |
| --- | --- | --- |
| `tavgM_2d_aer_Nx` | `M2TMNXAER` | `MERRA2_tavgM/aer_Nx` |
| `inst3_3d_aer_Nv` | `M2I3NVAER` | `MERRA2_inst3/aer_Nv` |
| `tavg1_2d_slv_Nx` | `M2T1NXSLV` | `MERRA2_tavg1/slv_Nx` |
| `inst3_3d_asm_Np` | `M2I3NPASM` | `MERRA2_inst3/asm_Np` |
| `tavg1_2d_rad_Nx` | `M2T1NXRAD` | `MERRA2_tavg1/rad_Nx` |
| `tavg3_3d_cld_Np` | `M2T3NPCLD` | `MERRA2_tavg3/cld_Np` |
| `tavg3_3d_rad_Np` | `M2T3NPRAD` | `MERRA2_tavg3/rad_Np` |

## DAVINCI Implementation Notes

1. Prefer local GDEX ERA5 paths on Derecho over CDS downloads for reader testing
   and example configs.
2. Use `type: era5` once a dedicated ERA5 reader exists. The generic reader is
   likely insufficient for pressure-level files because DAVINCI expects a
   standardized vertical dimension (`z`) and because ERA5 variable/dimension
   names need normalization.
3. For direct NetCDF reads, point configs at variable-specific files or globs.
   Avoid broad archive-wide globs; ERA5 files are split by product, variable,
   and month/day.
4. For large multi-year workflows, consider the GDEX zarr or kerchunk catalogs
   under `/gdex/data/d633000/catalogs` and `/gdex/data/d633000/kerchunk`, but
   DAVINCI's current local-file reader path should start with direct NetCDF.
5. ERA5 precipitation and flux diagnostics require forecast product directories
   (`fc.sfc.accumu` or `fc.sfc.meanflux`), not `an.sfc`.

## Commands Used For Local Verification

```bash
ls -ld /gdex/data /glade/campaign/collections/gdex/data /glade/campaign/collections/rda/data
find /gdex/data /glade/campaign/collections/gdex/data /glade/campaign/collections/rda/data \
  -maxdepth 2 -type d \( -name 'd633000' -o -name 'd633001' -o -name 'd633006' \)
readlink -f /gdex/data/d633000 /glade/campaign/collections/gdex/data/d633000 \
  /glade/campaign/collections/rda/data/d633000
find /gdex/data/d633000 -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
find /gdex/data/d633000/e5.oper.an.sfc/202501 -maxdepth 1 -type f | sed -n '1,80p'
find /gdex/data/d633000/e5.oper.an.pl/202501 -maxdepth 1 -type f | sed -n '1,80p'
head -5 /gdex/data/d633000/catalogs/d633000-posix.csv
```
