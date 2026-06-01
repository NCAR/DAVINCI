# MODIS/VIIRS Catalog-Driven Readers — Design

**Date:** 2026-06-01
**Status:** Design approved, pending spec review
**Branch target:** `develop` (planning only; no implementation started)

## Context

DAVINCI already has useful satellite foundations:

- Generic L2 and L3 satellite readers in `davinci_monet/observations/satellite/`.
- Dedicated readers for MODIS L2 AOD, TROPOMI, TEMPO, GOES, MOPITT, and OMPS.
- Swath and grid geometries in the pairing engine.
- A production MODIS L2 gridding/caching path, currently tied to legacy
  `obs:` configuration through `sat_type: modis_l2`.
- Synthetic swath/grid examples and tests.

The current implementation is not ready to scale to broad EOS successor support.
Adding one reader per product would duplicate file-format handling,
geolocation, QA metadata, scale/fill rules, and variable discovery across a very
large product surface. MODIS and VIIRS include Level 1, Level 2, and Level 3
products across several DAACs and science domains, and some Level 3 atmosphere
products contain hundreds of SDSs. The first MODIS/VIIRS milestone therefore
needs a metadata-centered design.

**Goal:** add catalog-complete and reader-complete local-file support for MODIS
and VIIRS products across L1, L2, and L3, exposing all parameters/SDSs through a
small set of generic readers. Science-curated defaults, full QA recipes, DAAC
discovery, and download are explicitly separate follow-on capabilities.

## Decisions locked during brainstorming

1. **Reference family:** MODIS -> VIIRS is the first EOS/successor family.
2. **Scope:** Level 1, Level 2, and Level 3, all parameters/SDSs.
3. **Milestone definition:** catalog-complete plus reader-complete. Fully
   curated QA/default mappings for every parameter are out of the first
   milestone.
4. **Data access:** local files only. Catalog fields preserve DAAC/product IDs so
   Earthdata discovery/download can be added in a separate future milestone.
5. **Architecture:** catalog-driven generic readers, not product-specific reader
   classes for every product.
6. **Pipeline path:** use the existing `sources:`/`SourceData`/`PairingEngine`
   architecture; keep geometry-based pairing.
7. **Compatibility:** keep the current legacy MODIS L2 config path working while
   documenting `sources:` as the forward path.

## Architecture

Add a MODIS/VIIRS subsystem under the existing satellite observation package:

```text
davinci_monet/observations/satellite/
├── catalog/
│   ├── __init__.py
│   ├── schema.py
│   ├── registry.py
│   └── data/
│       ├── modis_viirs_core.yaml
│       ├── modis_viirs_atmosphere.yaml
│       ├── modis_viirs_land.yaml
│       ├── modis_viirs_snow_ice.yaml
│       └── modis_viirs_ocean.yaml
├── geolocation.py
├── metadata.py
└── modis_viirs.py
```

The initial reader registration should be one source type:

```yaml
sources:
  viirs_aod:
    type: modis_viirs
    product: AERDB_L2_VIIRS_SNPP
    level: L2
    files: /data/viirs/*.nc
    variables: "*"
    geolocation: auto
    qa: metadata_only
```

Optional aliases such as `modis` and `viirs` can point to the same reader only if
they do not obscure the fact that MODIS/VIIRS products are resolved by catalog
`product`, not by one class per instrument.

Existing pairing stays unchanged at the conceptual level:

```text
local files -> MODIS/VIIRS reader -> xr.Dataset -> SourceData -> PairingEngine -> stats/plots
```

The reader returns standard DAVINCI geometry:

- L1: readable/inspectable swath-like radiance/geolocation data. Not
  automatically model-pairable unless a user selects or derives a geophysical
  quantity.
- L2: `SWATH` geometry with standardized 2-D `lat`/`lon`, `time`, and
  scanline/pixel dimensions.
- L3: `GRID` geometry for rectilinear products with standardized `time`, `lat`,
  and `lon`; projected products load with projection metadata and initially raise
  a clear error if model pairing requires unsupported reprojection.

## Product Catalog

The catalog is the source of truth for known MODIS/VIIRS products. It should be
data, not code, so adding or correcting a product does not require a new reader
class.

Each product entry includes:

- Product identity: `product_id`, aliases, instrument, platform, collection or
  version, DAAC, short name, DOI when known.
- Continuity metadata: MODIS Terra/Aqua product links to VIIRS SNPP/NOAA-20/
  NOAA-21 equivalents where they exist.
- Level and geometry: `L1`, `L2`, or `L3`; `SWATH` or `GRID`; expected
  dimensions and geolocation strategy.
- File format: HDF4, HDF5/netCDF4, HDF-EOS, NetCDF.
- Variables: every cataloged SDS/data variable with path/group, display name when
  known, units, scale factor, offset, fill values, valid range, and optional QA
  companion field.
- QA metadata: QA variable names, bitfield names/descriptions when known, and
  recommended masks only as metadata in the first milestone.
- Reader hints: group names, coordinate names, dimension aliases, time parsing,
  and separate geolocation product requirements.

The catalog should seed known product families from NASA sources while also
supporting runtime variable discovery. When `variables: "*"` is used, the reader
loads all cataloged variables and any additional variables discovered in the
file. Known catalog metadata is attached to variables; discovered variables are
kept with file-derived attrs.

## Reader Behavior

`MODISVIIRSReader` is generic and driven by catalog metadata plus file
inspection.

### Level 1

The L1 reader opens calibrated radiance/reflectance and geolocation inputs,
standardizes scan/pixel dimensions, exposes bands as variables, and attaches
geolocation coordinates when available. It handles embedded geolocation and
separate geolocation files through `geolocation: auto | embedded | file`.

L1 output is useful for inspection, plotting, true-color workflows, and separate
derived-product processing. It is not automatically paired with model output
unless a transform produces a geophysical quantity with a supported
geometry.

### Level 2

The L2 reader returns `SWATH` geometry:

- Standard dimensions: `time`, `scanline`, `pixel` where possible.
- Standard coordinates: 2-D `lat` and `lon`, plus time/overpass metadata.
- All requested SDSs, or all discoverable SDSs when `variables: "*"` is used.
- Scale/fill metadata preserved and optionally applied through existing
  variable-config mechanisms.
- QA fields exposed as variables and tagged in attrs; no default science mask
  unless explicitly configured.

The current MODIS L2 gridding/caching path should be generalized into reusable
swath-to-grid utilities so MODIS and VIIRS L2 products can be binned onto a
model grid or explicit grid using the same machinery.

### Level 3

The L3 reader returns `GRID` geometry for regular lat/lon products and preserves
projection metadata for projected grids. Regular grids pair through the existing
grid strategy. Projected products load and can be inspected or plotted where
possible; model pairing that requires reprojection raises an explicit
unsupported-projection error until reprojection support is added.

## Config Semantics

Recommended forward config:

```yaml
sources:
  modis_l3_atmos:
    type: modis_viirs
    product: MYD08_M3
    files: /data/modis/MYD08_M3.*.hdf
    variables: "*"
    qa: metadata_only

  viirs_l2_fire:
    type: modis_viirs
    product: VNP14IMG
    files: /data/viirs/VNP14IMG*.nc
    variables: "*"
    geolocation: auto
```

Config rules:

- `product` is required for catalog lookup. `product: auto` is out of scope for
  this milestone.
- `level` is optional when the catalog product entry defines it; if supplied, it
  must match the catalog.
- `variables: "*"` means all cataloged variables plus discovered variables.
- `variables: [A, B]` means explicit variables; missing explicit variables fail.
- `qa: metadata_only` exposes QA variables and QA metadata but does not mask
  science variables.
- `qa: none` leaves QA variables as normal data variables with no special attrs.
- `qa: <named-mask>` is reserved for future science-curated masks and should
  fail clearly until a named mask exists for the product.
- `geolocation: auto` tries embedded coordinates first, then catalog-declared
  geolocation associations.

Legacy support:

- Existing `obs: sat_type: modis_l2` configs continue working.
- A future migration should translate that form to `sources:` with
  `type: modis_viirs`, the appropriate `product`, and equivalent gridding/cache
  options.

## Data Flow And Metadata

Reader output is an `xr.Dataset` with standard DAVINCI attrs:

- `geometry`
- `source_label`
- `product_id`
- `instrument`
- `platform`
- `level`
- `daac`
- `collection`
- `catalog_version`
- `continuity_family`

Each variable gets attrs where known:

- `source_path`
- `units`
- `scale_factor`
- `add_offset`
- `_FillValue` or `fill_value`
- `valid_min` / `valid_max`
- `qa_variable`
- `qa_meaning`
- `catalog_status` (`cataloged` or `discovered`)

This makes plots, statistics, logs, and future provenance reports product-aware
without coupling those layers to MODIS/VIIRS-specific code.

## Error Handling

- Unknown product: fail with close product-id matches and suggest
  `type: satellite_l2` or `type: satellite_l3` as a low-level fallback.
- Level mismatch: fail and report both configured and catalog levels.
- Missing geolocation: load data variables when possible but mark geometry as
  incomplete; fail only when pairing or gridding needs geolocation.
- Unsupported projection: load and expose the dataset; raise a clear pairing
  error if model comparison requires reprojection.
- Missing SDS under `variables: "*"`: warn and continue.
- Missing explicit SDS: fail.
- Mixed products in one source: fail unless the catalog marks them as a valid
  data/geolocation pair.
- Unsupported file format engine: fail with the tried engines and required
  dependency hint.

## Testing And Validation

Testing is layered:

- **Catalog schema tests:** every product entry validates; aliases resolve; level,
  geometry, DAAC, file format, and geolocation strategy are present.
- **Continuity tests:** known MODIS/VIIRS continuity links are bidirectional where
  expected.
- **Synthetic file tests:** tiny HDF5/netCDF files represent L1, L2, and L3
  layouts. Verify `variables: "*"` exposes all parameters, dimensions are
  standardized, metadata is preserved, and QA fields are discoverable.
- **Existing MODIS regression:** current MODIS AOD config and cached-binned
  workflow still work.
- **Pipeline integration tests:** use `PipelineRunner.run_from_config()` for at
  least one synthetic L2 swath and one synthetic L3 grid, following the repo rule
  that integration tests use the real pipeline path.
- **Real-data smoke tests:** optional and skipped unless local env vars point to
  sample files. Cover one MODIS L1/geolocation case, one MODIS L2, one MODIS L3,
  one VIIRS L1/geolocation case, one VIIRS L2, and one VIIRS L3.
- **Docs/config tests:** example local-file configs validate without requiring
  real data.

Validation command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest
mypy davinci_monet
black --check davinci_monet && isort --check davinci_monet
```

## Phasing

Each phase should be independently testable and keep the suite green.

1. **Reader foundation:** ensure satellite readers expose reader-level
   `geometry`; add catalog schema and lookup API; add the `modis_viirs` source
   registration.
2. **Variable discovery:** implement file inspection for HDF4/HDF5/netCDF,
   variable path discovery, scale/fill metadata extraction, and `variables: "*"` behavior.
3. **L2 swath support:** implement generic L2 swath loading, geolocation
   attachment, and all-parameter exposure for representative MODIS and VIIRS L2
   files.
4. **Reusable gridding/cache:** extract the current MODIS-specific gridding path
   into reusable swath-to-grid utilities and route MODIS/VIIRS L2 through it.
5. **L3 grid support:** implement regular-grid L3 loading, metadata preservation,
   and grid-pairing integration.
6. **L1 support:** implement local L1 radiance/reflectance plus geolocation
   loading for MODIS and VIIRS.
7. **Catalog expansion:** seed atmosphere, land, snow/ice, and ocean product
   families with product identity, variables, QA metadata, and continuity links.
8. **Docs/examples:** add local-file examples for MODIS/VIIRS L1, L2, and L3,
   plus a migration note from legacy MODIS L2 config.

## Out of scope for the first milestone

- Earthdata/DAAC search, authentication, download, or cloud access.
- Fully curated QA masks for every product variable.
- Full reprojection support for projected L3 grids.
- Physical retrieval derivation from L1 radiances.
- Product-specific reader classes for every product.
- Non-MODIS/VIIRS EOS families such as CERES/Libera, AIRS/CrIS, or
  OMI/TROPOMI/TEMPO/GEMS.

## Source References

- LAADS DAAC mission and product scope:
  https://ladsweb.modaps.eosdis.nasa.gov/about/
- MODIS to VIIRS transition:
  https://ladsweb.modaps.eosdis.nasa.gov/learn/modis-to-viirs-transition/
- LAADS MODIS/VIIRS transition PDF:
  https://ladsweb.modaps.eosdis.nasa.gov/learn/modis-to-viirs-transition/MODIS-VIIRS_Transition.pdf
- NASA snow/sea ice MODIS/VIIRS data:
  https://snow.nasa.gov/data/modisviirs-snowsea-ice-data
- NASA Earthdata LP DAAC center:
  https://www.earthdata.nasa.gov/centers/lp-daac
- NASA Ocean Color:
  https://oceancolor.gsfc.nasa.gov/
