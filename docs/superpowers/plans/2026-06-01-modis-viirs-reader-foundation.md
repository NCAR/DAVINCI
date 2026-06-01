# MODIS/VIIRS Reader Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first usable MODIS/VIIRS catalog-reader slice: source config supports `variables: "*"`, a product catalog can be loaded and queried, and a registered `modis_viirs` reader can open local xarray-readable L2/L3 files through the existing `sources:` pipeline.

**Architecture:** This is the first implementation slice of `docs/superpowers/specs/2026-06-01-modis-viirs-catalog-readers-design.md`. It does not attempt full MODIS/VIIRS coverage. It creates the catalog and reader foundation, proves the pipeline contract with synthetic local files, and leaves HDF4-specific handling, L1 geolocation pairs, reusable gridding/cache extraction, and full catalog expansion for separate follow-on plans.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, `importlib.resources`, xarray, pytest, mypy, black, isort.

---

## Scope Check

The approved design covers multiple implementation phases. This plan implements the first independently shippable foundation:

- Config support for `variables: "*"` and explicit variable lists in unified `sources:`.
- Catalog schema and lookup API.
- Seed catalog entries for representative MODIS/VIIRS L2/L3 products.
- `MODISVIIRSReader` registered as `type: modis_viirs`.
- Synthetic-file tests proving `variables: "*"` exposes all parameters and pipeline loading works.

This plan does not implement real HDF4 parsing, L1 geolocation pairing, science QA masks, DAAC discovery/download, or reusable MODIS/VIIRS swath gridding.

Repo policy override: `AGENTS.md` says never auto commit. The task steps include `git status` plus suggested commit messages, but workers must not run `git commit` unless the user explicitly authorizes it.

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `davinci_monet/config/schema.py` | Pydantic config models | Allow `SourceConfig.variables` to be a mapping, list, or `"*"` |
| `davinci_monet/pipeline/stages.py` | Source loading stage | Preserve wildcard/list variable selection when calling readers |
| `davinci_monet/observations/satellite/catalog/__init__.py` | Catalog public API | Create |
| `davinci_monet/observations/satellite/catalog/schema.py` | Product/variable metadata models | Create |
| `davinci_monet/observations/satellite/catalog/registry.py` | Load/query product catalog YAML | Create |
| `davinci_monet/observations/satellite/catalog/data/__init__.py` | Package marker for bundled catalog data | Create |
| `davinci_monet/observations/satellite/catalog/data/modis_viirs_core.yaml` | Seed product catalog | Create |
| `davinci_monet/observations/satellite/modis_viirs.py` | Catalog-driven local reader | Create |
| `davinci_monet/observations/satellite/__init__.py` | Satellite exports/registration import | Export `MODISVIIRSReader` |
| `davinci_monet/observations/__init__.py` | Observation package imports | Import/export `MODISVIIRSReader` so `source_registry` is populated |
| `pyproject.toml` | Package data | Include catalog YAML files |
| `davinci_monet/tests/unit/config/test_schema.py` | Config schema tests | Add wildcard/list source variable tests |
| `davinci_monet/tests/test_load_sources_stage.py` | Source-stage tests | Add variable-selection and pipeline source tests |
| `davinci_monet/tests/test_modis_viirs_catalog.py` | Catalog tests | Create |
| `davinci_monet/tests/test_modis_viirs_reader.py` | Reader tests | Create |

## Task 1: Support `variables: "*"` In Unified Sources

**Files:**
- Modify: `davinci_monet/config/schema.py`
- Modify: `davinci_monet/pipeline/stages.py`
- Modify: `davinci_monet/tests/unit/config/test_schema.py`
- Modify: `davinci_monet/tests/test_load_sources_stage.py`

- [ ] **Step 1: Add failing schema tests**

In `davinci_monet/tests/unit/config/test_schema.py`, add `SourceConfig` to the import list from `davinci_monet.config.schema`, then append this test class after `TestObservationConfig`:

```python
class TestSourceConfig:
    """Tests for unified source configuration."""

    def test_variables_accepts_wildcard(self) -> None:
        config = SourceConfig.model_validate(
            {"type": "modis_viirs", "files": "/data/*.nc", "variables": "*"}
        )

        assert config.variables == "*"

    def test_variables_accepts_explicit_list(self) -> None:
        config = SourceConfig.model_validate(
            {"type": "modis_viirs", "files": "/data/*.nc", "variables": ["AOD", "QA"]}
        )

        assert config.variables == ["AOD", "QA"]

    def test_variables_accepts_mapping(self) -> None:
        config = SourceConfig.model_validate(
            {
                "type": "modis_viirs",
                "files": "/data/*.nc",
                "variables": {"AOD": {"units": "1", "obs_min": 0.0}},
            }
        )

        assert isinstance(config.variables, dict)
        assert config.variables["AOD"].units == "1"
        assert config.variables["AOD"].obs_min == 0.0
```

- [ ] **Step 2: Add failing variable-selection tests**

In `davinci_monet/tests/test_load_sources_stage.py`, add this class before `TestPipelineContextSources`:

```python
class TestSourceVariableSelection:
    def test_wildcard_variables_request_load_all(self) -> None:
        names, configs, load_all = LoadSourcesStage._source_variable_selection("*")

        assert names is None
        assert configs == {}
        assert load_all is True

    def test_list_variables_request_explicit_names_without_configs(self) -> None:
        names, configs, load_all = LoadSourcesStage._source_variable_selection(["AOD", "QA"])

        assert names == ["AOD", "QA"]
        assert configs == {}
        assert load_all is False

    def test_mapping_variables_request_names_and_configs(self) -> None:
        names, configs, load_all = LoadSourcesStage._source_variable_selection(
            {"AOD": {"units": "1", "obs_min": 0.0}}
        )

        assert names == ["AOD"]
        assert configs == {"AOD": {"units": "1", "obs_min": 0.0}}
        assert load_all is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest davinci_monet/tests/unit/config/test_schema.py::TestSourceConfig davinci_monet/tests/test_load_sources_stage.py::TestSourceVariableSelection -v`

Expected:
- `TestSourceConfig` fails because `SourceConfig.variables` does not accept `"*"` or lists.
- `TestSourceVariableSelection` fails because `LoadSourcesStage._source_variable_selection` does not exist.

- [ ] **Step 4: Update `SourceConfig.variables`**

In `davinci_monet/config/schema.py`, change `SourceConfig.variables` and `_parse_variables` to:

```python
    variables: dict[str, VariableConfig] | list[str] | Literal["*"] = Field(default_factory=dict)
```

```python
    @field_validator("variables", mode="before")
    @classmethod
    def _parse_variables(
        cls, v: Any
    ) -> dict[str, VariableConfig] | list[str] | Literal["*"]:
        if v is None:
            return {}
        if v == "*":
            return "*"
        if isinstance(v, (list, tuple)):
            return [str(item) for item in v]
        if isinstance(v, dict):
            return {
                str(name): VariableConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in v.items()
            }
        return dict(v)
```

- [ ] **Step 5: Add source variable selection helper**

In `davinci_monet/pipeline/stages.py`, inside `LoadSourcesStage`, add this static method immediately after `_normalize_var_configs`:

```python
    @staticmethod
    def _source_variable_selection(
        raw: Any,
    ) -> tuple[list[str] | None, dict[str, dict[str, Any]], bool]:
        """Return reader variable names, variable configs, and wildcard flag.

        ``variables: "*"`` means the reader should expose every discoverable
        variable without applying per-variable config. A list requests explicit
        names without configs. A mapping requests names and common transforms.
        """
        if raw == "*":
            return None, {}, True
        if isinstance(raw, (list, tuple)):
            return [str(item) for item in raw], {}, False
        configs = LoadSourcesStage._normalize_var_configs(raw)
        return list(configs) or None, configs, False
```

- [ ] **Step 6: Use the helper in unified source loading**

In `LoadSourcesStage._load_unified_source`, replace:

```python
        variables = self._normalize_var_configs(cfg.get("variables", {}))
        variable_names = list(variables) or None
```

with:

```python
        raw_variables = cfg.get("variables", {})
        variable_names, variables, load_all_variables = self._source_variable_selection(
            raw_variables
        )
```

Then before `data = reader.open(...)`, add:

```python
        if load_all_variables:
            open_kwargs["load_all_variables"] = True
```

Keep the existing call shape:

```python
        data = reader.open(file_paths, variables=variable_names, **open_kwargs)
```

- [ ] **Step 7: Run focused tests**

Run: `pytest davinci_monet/tests/unit/config/test_schema.py::TestSourceConfig davinci_monet/tests/test_load_sources_stage.py::TestSourceVariableSelection -v`

Expected: all new tests pass.

- [ ] **Step 8: Record status for authorized commit**

Run: `git status --short`

Suggested commit message when explicitly authorized:

```bash
git add davinci_monet/config/schema.py davinci_monet/pipeline/stages.py \
  davinci_monet/tests/unit/config/test_schema.py davinci_monet/tests/test_load_sources_stage.py
git commit -m "feat(config): support wildcard source variables"
```

## Task 2: Add Catalog Schema Models

**Files:**
- Create: `davinci_monet/observations/satellite/catalog/__init__.py`
- Create: `davinci_monet/observations/satellite/catalog/schema.py`
- Create: `davinci_monet/tests/test_modis_viirs_catalog.py`

- [ ] **Step 1: Write failing catalog schema tests**

Create `davinci_monet/tests/test_modis_viirs_catalog.py`:

```python
"""Tests for the MODIS/VIIRS product catalog."""

from __future__ import annotations

import pytest

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.observations.satellite.catalog.schema import (
    ProductGeometry,
    ProductLevel,
    ProductSpec,
    VariableSpec,
)


def test_variable_spec_defaults() -> None:
    spec = VariableSpec(name="AOD", path="ScienceData/AOD")

    assert spec.name == "AOD"
    assert spec.path == "ScienceData/AOD"
    assert spec.catalog_status == "cataloged"
    assert spec.units is None


def test_product_spec_derives_data_geometry() -> None:
    product = ProductSpec(
        product_id="MOD08_M3",
        aliases=["MODIS_L3_ATMOS_MONTHLY"],
        instrument="MODIS",
        platforms=["Aqua"],
        level=ProductLevel.L3,
        geometry=ProductGeometry.GRID,
        daac="LAADS",
        file_format="HDF4",
        variables={"AOD": VariableSpec(name="AOD", path="AOD")},
    )

    assert product.product_id == "MOD08_M3"
    assert product.data_geometry is DataGeometry.GRID
    assert product.variables["AOD"].path == "AOD"


def test_product_spec_requires_variables() -> None:
    with pytest.raises(ValueError):
        ProductSpec(
            product_id="EMPTY",
            instrument="MODIS",
            platforms=["Terra"],
            level=ProductLevel.L2,
            geometry=ProductGeometry.SWATH,
            daac="LAADS",
            file_format="HDF4",
            variables={},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_modis_viirs_catalog.py -v`

Expected: FAIL at import because `davinci_monet.observations.satellite.catalog.schema` does not exist.

- [ ] **Step 3: Create catalog package init**

Create `davinci_monet/observations/satellite/catalog/__init__.py`:

```python
"""MODIS/VIIRS satellite product catalog."""

from davinci_monet.observations.satellite.catalog.schema import (
    ProductGeometry,
    ProductLevel,
    ProductSpec,
    VariableSpec,
)

__all__ = [
    "ProductGeometry",
    "ProductLevel",
    "ProductSpec",
    "VariableSpec",
]
```

- [ ] **Step 4: Create schema models**

Create `davinci_monet/observations/satellite/catalog/schema.py`:

```python
"""Schema models for MODIS/VIIRS product catalog entries."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from davinci_monet.core.protocols import DataGeometry


class CatalogModel(BaseModel):
    """Base catalog model with strict fields."""

    model_config = ConfigDict(extra="forbid", validate_default=True, str_strip_whitespace=True)


class ProductLevel(str, Enum):
    """Supported MODIS/VIIRS processing levels for this reader family."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class ProductGeometry(str, Enum):
    """Catalog geometry values that map to DAVINCI DataGeometry."""

    SWATH = "SWATH"
    GRID = "GRID"

    @property
    def data_geometry(self) -> DataGeometry:
        if self is ProductGeometry.SWATH:
            return DataGeometry.SWATH
        return DataGeometry.GRID


class VariableSpec(CatalogModel):
    """Metadata for one cataloged SDS or data variable."""

    name: str
    path: str
    display_name: str | None = None
    units: str | None = None
    scale_factor: float | None = None
    add_offset: float | None = None
    fill_values: list[float | int | str] = Field(default_factory=list)
    valid_min: float | None = None
    valid_max: float | None = None
    qa_variable: str | None = None
    qa_meaning: str | None = None
    catalog_status: Literal["cataloged", "discovered"] = "cataloged"


class ProductSpec(CatalogModel):
    """Catalog metadata for one MODIS/VIIRS product."""

    product_id: str
    aliases: list[str] = Field(default_factory=list)
    instrument: str
    platforms: list[str]
    level: ProductLevel
    geometry: ProductGeometry
    daac: str
    file_format: str
    collection: str | None = None
    short_name: str | None = None
    doi: str | None = None
    continuity_family: str | None = None
    continuity_products: list[str] = Field(default_factory=list)
    geolocation_strategy: Literal["embedded", "separate", "none", "auto"] = "auto"
    dimension_aliases: dict[str, str] = Field(default_factory=dict)
    coordinate_aliases: dict[str, str] = Field(default_factory=dict)
    variables: dict[str, VariableSpec]

    @field_validator("variables")
    @classmethod
    def _require_variables(cls, value: dict[str, VariableSpec]) -> dict[str, VariableSpec]:
        if not value:
            raise ValueError("ProductSpec.variables must contain at least one variable")
        return value

    @property
    def data_geometry(self) -> DataGeometry:
        return self.geometry.data_geometry

    def matches(self, key: str) -> bool:
        normalized = key.lower()
        return normalized == self.product_id.lower() or normalized in {
            alias.lower() for alias in self.aliases
        }
```

- [ ] **Step 5: Run catalog schema tests**

Run: `pytest davinci_monet/tests/test_modis_viirs_catalog.py -v`

Expected: 3 passed.

- [ ] **Step 6: Run mypy on new schema**

Run: `mypy davinci_monet/observations/satellite/catalog/schema.py`

Expected: no errors.

- [ ] **Step 7: Record status for authorized commit**

Run: `git status --short`

Suggested commit message when explicitly authorized:

```bash
git add davinci_monet/observations/satellite/catalog/__init__.py \
  davinci_monet/observations/satellite/catalog/schema.py \
  davinci_monet/tests/test_modis_viirs_catalog.py
git commit -m "feat(satellite): add MODIS/VIIRS catalog schema"
```

## Task 3: Add Catalog Loader And Seed Data

**Files:**
- Create: `davinci_monet/observations/satellite/catalog/registry.py`
- Create: `davinci_monet/observations/satellite/catalog/data/__init__.py`
- Create: `davinci_monet/observations/satellite/catalog/data/modis_viirs_core.yaml`
- Modify: `davinci_monet/observations/satellite/catalog/__init__.py`
- Modify: `pyproject.toml`
- Modify: `davinci_monet/tests/test_modis_viirs_catalog.py`

- [ ] **Step 1: Add failing catalog registry tests**

Append to `davinci_monet/tests/test_modis_viirs_catalog.py`:

```python
from davinci_monet.observations.satellite.catalog.registry import (
    CatalogProductNotFoundError,
    closest_product_ids,
    get_product,
    iter_products,
)


def test_iter_products_loads_seed_catalog() -> None:
    product_ids = {product.product_id for product in iter_products()}

    assert "MOD04_L2" in product_ids
    assert "MYD04_L2" in product_ids
    assert "AERDB_L2_VIIRS_SNPP" in product_ids
    assert "MOD08_M3" in product_ids


def test_get_product_resolves_alias_case_insensitive() -> None:
    product = get_product("modis_terra_l2_aod")

    assert product.product_id == "MOD04_L2"
    assert product.level is ProductLevel.L2
    assert product.data_geometry is DataGeometry.SWATH


def test_get_product_unknown_reports_close_matches() -> None:
    with pytest.raises(CatalogProductNotFoundError) as excinfo:
        get_product("MOD04")

    assert "MOD04_L2" in str(excinfo.value)


def test_closest_product_ids() -> None:
    assert "MOD04_L2" in closest_product_ids("MOD04", limit=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_modis_viirs_catalog.py -v`

Expected: FAIL at import because `catalog.registry` does not exist.

- [ ] **Step 3: Create seed catalog YAML**

Create `davinci_monet/observations/satellite/catalog/data/__init__.py`:

```python
"""Bundled MODIS/VIIRS catalog data."""
```

Create `davinci_monet/observations/satellite/catalog/data/modis_viirs_core.yaml`:

```yaml
products:
  - product_id: MOD04_L2
    aliases: [modis_terra_l2_aod, terra_modis_aod]
    instrument: MODIS
    platforms: [Terra]
    level: L2
    geometry: SWATH
    daac: LAADS
    file_format: HDF4
    collection: "061"
    short_name: MOD04_L2
    continuity_family: modis_viirs_aerosol
    continuity_products: [AERDB_L2_VIIRS_SNPP]
    geolocation_strategy: embedded
    dimension_aliases: {Cell_Along_Swath_10km: scanline, Cell_Across_Swath_10km: pixel}
    coordinate_aliases: {Latitude: lat, Longitude: lon}
    variables:
      AOD_550_Dark_Target_Deep_Blue_Combined:
        name: AOD_550_Dark_Target_Deep_Blue_Combined
        path: AOD_550_Dark_Target_Deep_Blue_Combined
        display_name: AOD 550 nm
        units: "1"
        scale_factor: 0.001
        valid_min: 0.0
        valid_max: 10.0
        qa_variable: Land_Ocean_Quality_Flag
      Land_Ocean_Quality_Flag:
        name: Land_Ocean_Quality_Flag
        path: Land_Ocean_Quality_Flag
        display_name: Land/Ocean Quality Flag

  - product_id: MYD04_L2
    aliases: [modis_aqua_l2_aod, aqua_modis_aod]
    instrument: MODIS
    platforms: [Aqua]
    level: L2
    geometry: SWATH
    daac: LAADS
    file_format: HDF4
    collection: "061"
    short_name: MYD04_L2
    continuity_family: modis_viirs_aerosol
    continuity_products: [AERDB_L2_VIIRS_SNPP]
    geolocation_strategy: embedded
    coordinate_aliases: {Latitude: lat, Longitude: lon}
    variables:
      AOD_550_Dark_Target_Deep_Blue_Combined:
        name: AOD_550_Dark_Target_Deep_Blue_Combined
        path: AOD_550_Dark_Target_Deep_Blue_Combined
        display_name: AOD 550 nm
        units: "1"
        scale_factor: 0.001
        valid_min: 0.0
        valid_max: 10.0
        qa_variable: Land_Ocean_Quality_Flag
      Land_Ocean_Quality_Flag:
        name: Land_Ocean_Quality_Flag
        path: Land_Ocean_Quality_Flag
        display_name: Land/Ocean Quality Flag

  - product_id: AERDB_L2_VIIRS_SNPP
    aliases: [viirs_snpp_l2_aod, snpp_viirs_aod]
    instrument: VIIRS
    platforms: [SNPP]
    level: L2
    geometry: SWATH
    daac: LAADS
    file_format: NetCDF4
    short_name: AERDB_L2_VIIRS_SNPP
    continuity_family: modis_viirs_aerosol
    continuity_products: [MOD04_L2, MYD04_L2]
    geolocation_strategy: embedded
    coordinate_aliases: {Latitude: lat, Longitude: lon, latitude: lat, longitude: lon}
    variables:
      Aerosol_Optical_Thickness_550_Land_Ocean:
        name: Aerosol_Optical_Thickness_550_Land_Ocean
        path: Aerosol_Optical_Thickness_550_Land_Ocean
        display_name: AOD 550 nm
        units: "1"
        valid_min: 0.0
        valid_max: 10.0
        qa_variable: QF1_VIIRSAERO
      QF1_VIIRSAERO:
        name: QF1_VIIRSAERO
        path: QF1_VIIRSAERO
        display_name: VIIRS Aerosol Quality Flags

  - product_id: VNP14IMG
    aliases: [viirs_snpp_l2_fire, viirs_i_band_fire]
    instrument: VIIRS
    platforms: [SNPP]
    level: L2
    geometry: SWATH
    daac: LAADS
    file_format: NetCDF4
    short_name: VNP14IMG
    continuity_family: modis_viirs_fire
    geolocation_strategy: embedded
    coordinate_aliases: {latitude: lat, longitude: lon, Latitude: lat, Longitude: lon}
    variables:
      FireMask:
        name: FireMask
        path: FireMask
        display_name: Fire Mask
      FP_power:
        name: FP_power
        path: FP_power
        display_name: Fire Radiative Power
        units: MW

  - product_id: MOD08_M3
    aliases: [modis_terra_l3_atmos_monthly]
    instrument: MODIS
    platforms: [Terra]
    level: L3
    geometry: GRID
    daac: LAADS
    file_format: HDF4
    collection: "061"
    short_name: MOD08_M3
    continuity_family: modis_viirs_atmosphere_l3
    continuity_products: [VNP08_M3]
    geolocation_strategy: none
    coordinate_aliases: {Latitude: lat, Longitude: lon}
    variables:
      AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean:
        name: AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean
        path: AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean
        display_name: Monthly mean AOD 550 nm
        units: "1"

  - product_id: VNP08_M3
    aliases: [viirs_snpp_l3_atmos_monthly]
    instrument: VIIRS
    platforms: [SNPP]
    level: L3
    geometry: GRID
    daac: LAADS
    file_format: NetCDF4
    short_name: VNP08_M3
    continuity_family: modis_viirs_atmosphere_l3
    continuity_products: [MOD08_M3]
    geolocation_strategy: none
    coordinate_aliases: {Latitude: lat, Longitude: lon}
    variables:
      AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean:
        name: AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean
        path: AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean
        display_name: Monthly mean AOD 550 nm
        units: "1"
```

- [ ] **Step 4: Create catalog registry**

Create `davinci_monet/observations/satellite/catalog/registry.py`:

```python
"""Load and query the MODIS/VIIRS product catalog."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Iterable

import yaml

from davinci_monet.observations.satellite.catalog.schema import ProductSpec


class CatalogProductNotFoundError(KeyError):
    """Raised when a product id or alias is not present in the catalog."""

    def __init__(self, product: str, suggestions: list[str]) -> None:
        suffix = f" Close matches: {', '.join(suggestions)}." if suggestions else ""
        super().__init__(f"Unknown MODIS/VIIRS product {product!r}.{suffix}")
        self.product = product
        self.suggestions = suggestions


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, ProductSpec]:
    """Load bundled MODIS/VIIRS catalog YAML into ProductSpec objects."""
    package = resources.files("davinci_monet.observations.satellite.catalog.data")
    products: dict[str, ProductSpec] = {}
    for path in sorted(package.iterdir()):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        raw = yaml.safe_load(path.read_text()) or {}
        for entry in raw.get("products", []):
            product = ProductSpec.model_validate(entry)
            products[product.product_id] = product
    return products


def iter_products() -> Iterable[ProductSpec]:
    """Iterate over catalog products sorted by product id."""
    yield from (load_catalog()[key] for key in sorted(load_catalog()))


def closest_product_ids(product: str, limit: int = 5) -> list[str]:
    """Return simple substring/prefix product-id suggestions."""
    needle = product.lower()
    scored: list[tuple[int, str]] = []
    for candidate in load_catalog():
        lower = candidate.lower()
        if lower.startswith(needle):
            scored.append((0, candidate))
        elif needle in lower:
            scored.append((1, candidate))
    return [candidate for _, candidate in sorted(scored)[:limit]]


def get_product(product: str) -> ProductSpec:
    """Return a ProductSpec by product id or alias, case-insensitive."""
    catalog = load_catalog()
    normalized = product.lower()
    for spec in catalog.values():
        if spec.matches(normalized):
            return spec
    raise CatalogProductNotFoundError(product, closest_product_ids(product))
```

- [ ] **Step 5: Export registry API**

Update `davinci_monet/observations/satellite/catalog/__init__.py` to:

```python
"""MODIS/VIIRS satellite product catalog."""

from davinci_monet.observations.satellite.catalog.registry import (
    CatalogProductNotFoundError,
    closest_product_ids,
    get_product,
    iter_products,
    load_catalog,
)
from davinci_monet.observations.satellite.catalog.schema import (
    ProductGeometry,
    ProductLevel,
    ProductSpec,
    VariableSpec,
)

__all__ = [
    "CatalogProductNotFoundError",
    "ProductGeometry",
    "ProductLevel",
    "ProductSpec",
    "VariableSpec",
    "closest_product_ids",
    "get_product",
    "iter_products",
    "load_catalog",
]
```

- [ ] **Step 6: Ensure package data includes catalog YAML**

In `pyproject.toml`, update package data:

```toml
[tool.setuptools.package-data]
davinci_monet = ["py.typed", "observations/satellite/catalog/data/*.yaml"]
```

- [ ] **Step 7: Run catalog tests**

Run: `pytest davinci_monet/tests/test_modis_viirs_catalog.py -v`

Expected: all catalog tests pass.

- [ ] **Step 8: Run mypy on catalog package**

Run: `mypy davinci_monet/observations/satellite/catalog`

Expected: no errors.

- [ ] **Step 9: Record status for authorized commit**

Run: `git status --short`

Suggested commit message when explicitly authorized:

```bash
git add pyproject.toml davinci_monet/observations/satellite/catalog \
  davinci_monet/tests/test_modis_viirs_catalog.py
git commit -m "feat(satellite): add seed MODIS/VIIRS product catalog"
```

## Task 4: Add `MODISVIIRSReader`

**Files:**
- Create: `davinci_monet/observations/satellite/modis_viirs.py`
- Modify: `davinci_monet/observations/satellite/__init__.py`
- Modify: `davinci_monet/observations/__init__.py`
- Create: `davinci_monet/tests/test_modis_viirs_reader.py`

- [ ] **Step 1: Write failing reader tests**

Create `davinci_monet/tests/test_modis_viirs_reader.py`:

```python
"""Tests for the catalog-driven MODIS/VIIRS reader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import davinci_monet.observations  # noqa: F401 - imports reader registrations
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader


def _write_l3_file(path) -> None:
    ds = xr.Dataset(
        {
            "AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean": (
                ("time", "lat", "lon"),
                np.ones((1, 3, 4), dtype=np.float32),
            ),
            "Extra_Parameter": (("time", "lat", "lon"), np.full((1, 3, 4), 2.0)),
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=1),
            "lat": np.linspace(-10, 10, 3),
            "lon": np.linspace(100, 103, 4),
        },
    )
    ds.to_netcdf(path)


def _write_l2_file(path) -> None:
    lat = np.arange(6, dtype=np.float32).reshape(2, 3)
    lon = 100.0 + lat
    ds = xr.Dataset(
        {
            "Aerosol_Optical_Thickness_550_Land_Ocean": (
                ("scanline", "pixel"),
                np.ones((2, 3), dtype=np.float32),
            ),
            "QF1_VIIRSAERO": (("scanline", "pixel"), np.zeros((2, 3), dtype=np.int16)),
            "Extra_Swath_Parameter": (("scanline", "pixel"), np.full((2, 3), 4.0)),
        },
        coords={
            "scanline": [0, 1],
            "pixel": [0, 1, 2],
            "Latitude": (("scanline", "pixel"), lat),
            "Longitude": (("scanline", "pixel"), lon),
        },
    )
    ds.to_netcdf(path)


def test_reader_registered() -> None:
    assert "modis_viirs" in source_registry
    assert source_registry.get("modis_viirs") is MODISVIIRSReader


def test_l3_wildcard_loads_all_variables(tmp_path) -> None:
    path = tmp_path / "mod08_m3.nc"
    _write_l3_file(path)
    reader = MODISVIIRSReader()

    ds = reader.open([path], product="MOD08_M3", load_all_variables=True)

    assert reader.geometry is DataGeometry.GRID
    assert ds.attrs["product_id"] == "MOD08_M3"
    assert ds.attrs["geometry"] == "grid"
    assert "AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean" in ds.data_vars
    assert "Extra_Parameter" in ds.data_vars
    assert ds["AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean"].attrs["catalog_status"] == "cataloged"
    assert ds["Extra_Parameter"].attrs["catalog_status"] == "discovered"


def test_l2_standardizes_lat_lon_and_geometry(tmp_path) -> None:
    path = tmp_path / "aerdb_l2.nc"
    _write_l2_file(path)
    reader = MODISVIIRSReader()

    ds = reader.open([path], product="AERDB_L2_VIIRS_SNPP", load_all_variables=True)

    assert reader.geometry is DataGeometry.SWATH
    assert ds.attrs["geometry"] == "swath"
    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert "Aerosol_Optical_Thickness_550_Land_Ocean" in ds.data_vars
    assert "Extra_Swath_Parameter" in ds.data_vars


def test_explicit_missing_variable_fails(tmp_path) -> None:
    path = tmp_path / "mod08_m3.nc"
    _write_l3_file(path)
    reader = MODISVIIRSReader()

    with pytest.raises(KeyError, match="Missing requested variable"):
        reader.open([path], product="MOD08_M3", variables=["not_present"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_modis_viirs_reader.py -v`

Expected: FAIL at import because `davinci_monet.observations.satellite.modis_viirs` does not exist.

- [ ] **Step 3: Create `MODISVIIRSReader`**

Create `davinci_monet/observations/satellite/modis_viirs.py`:

```python
"""Catalog-driven MODIS/VIIRS local-file reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.observations.satellite.catalog import ProductSpec, get_product


@source_registry.register("modis_viirs")
class MODISVIIRSReader:
    """Open local MODIS/VIIRS files using catalog product metadata."""

    def __init__(self) -> None:
        self._geometry = DataGeometry.SWATH

    @property
    def name(self) -> str:
        return "modis_viirs"

    @property
    def geometry(self) -> DataGeometry:
        return self._geometry

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        product: str,
        load_all_variables: bool = False,
        group: str | None = None,
        engine: str | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open local files for a cataloged MODIS/VIIRS product."""
        product_spec = get_product(product)
        self._geometry = product_spec.data_geometry
        files = [Path(path).expanduser() for path in file_paths]
        if not files:
            raise DataNotFoundError("No MODIS/VIIRS files provided")
        missing = [path for path in files if not path.exists()]
        if missing:
            raise DataNotFoundError(f"MODIS/VIIRS files not found: {missing}")

        ds = self._open_xarray(files, group=group, engine=engine, **kwargs)
        ds = self._standardize_dimensions_and_coords(ds, product_spec)
        ds = self._select_variables(ds, product_spec, variables, load_all_variables)
        ds = self._attach_metadata(ds, product_spec)
        return ds

    def _open_xarray(
        self,
        files: list[Path],
        *,
        group: str | None,
        engine: str | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        open_kwargs = dict(kwargs)
        if group is not None:
            open_kwargs["group"] = group
        if engine is not None:
            open_kwargs["engine"] = engine
        if len(files) == 1:
            return xr.open_dataset(files[0], **open_kwargs)
        return xr.open_mfdataset([str(path) for path in files], combine="by_coords", **open_kwargs)

    def _standardize_dimensions_and_coords(
        self, ds: xr.Dataset, product: ProductSpec
    ) -> xr.Dataset:
        renames: dict[str, str] = {}
        for old, new in product.dimension_aliases.items():
            if old in ds.dims and new not in ds.dims:
                renames[old] = new
        for old, new in product.coordinate_aliases.items():
            if old in ds.coords and new not in ds.coords:
                renames[old] = new
            elif old in ds.data_vars and new not in ds.coords and new not in ds.data_vars:
                renames[old] = new
        if renames:
            ds = ds.rename(renames)
        return ds

    def _select_variables(
        self,
        ds: xr.Dataset,
        product: ProductSpec,
        variables: Sequence[str] | None,
        load_all_variables: bool,
    ) -> xr.Dataset:
        if load_all_variables or variables is None:
            return ds

        selected: dict[str, xr.DataArray] = {}
        missing: list[str] = []
        for requested in variables:
            resolved = self._resolve_variable_name(ds, product, str(requested))
            if resolved is None:
                missing.append(str(requested))
            else:
                selected[resolved] = ds[resolved]
        if missing:
            raise KeyError(f"Missing requested variable(s): {', '.join(missing)}")
        return xr.Dataset(selected, coords=ds.coords, attrs=ds.attrs)

    @staticmethod
    def _resolve_variable_name(
        ds: xr.Dataset, product: ProductSpec, requested: str
    ) -> str | None:
        if requested in ds.data_vars:
            return requested
        if requested in product.variables:
            path = product.variables[requested].path
            if path in ds.data_vars:
                return path
        for name, spec in product.variables.items():
            if requested == spec.path and name in ds.data_vars:
                return name
        return None

    def _attach_metadata(self, ds: xr.Dataset, product: ProductSpec) -> xr.Dataset:
        ds = ds.copy()
        ds.attrs.update(
            {
                "geometry": product.data_geometry.name.lower(),
                "product_id": product.product_id,
                "instrument": product.instrument,
                "platform": ",".join(product.platforms),
                "level": product.level.value,
                "daac": product.daac,
                "collection": product.collection or "",
                "catalog_version": "seed",
                "continuity_family": product.continuity_family or "",
            }
        )
        known_by_name = product.variables
        known_by_path = {spec.path: spec for spec in product.variables.values()}
        for name in ds.data_vars:
            spec = known_by_name.get(str(name)) or known_by_path.get(str(name))
            if spec is None:
                ds[name].attrs.setdefault("catalog_status", "discovered")
                ds[name].attrs.setdefault("source_path", str(name))
                continue
            ds[name].attrs.setdefault("catalog_status", "cataloged")
            ds[name].attrs.setdefault("source_path", spec.path)
            if spec.units is not None:
                ds[name].attrs.setdefault("units", spec.units)
            if spec.scale_factor is not None:
                ds[name].attrs.setdefault("scale_factor", spec.scale_factor)
            if spec.add_offset is not None:
                ds[name].attrs.setdefault("add_offset", spec.add_offset)
            if spec.valid_min is not None:
                ds[name].attrs.setdefault("valid_min", spec.valid_min)
            if spec.valid_max is not None:
                ds[name].attrs.setdefault("valid_max", spec.valid_max)
            if spec.qa_variable is not None:
                ds[name].attrs.setdefault("qa_variable", spec.qa_variable)
            if spec.qa_meaning is not None:
                ds[name].attrs.setdefault("qa_meaning", spec.qa_meaning)
        return ds

    def get_variable_mapping(self) -> dict[str, str]:
        return {}
```

- [ ] **Step 4: Export and import the reader**

In `davinci_monet/observations/satellite/__init__.py`, import the reader:

```python
from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader
```

Add `"MODISVIIRSReader"` to `__all__`.

In `davinci_monet/observations/__init__.py`, import the reader near the other satellite imports:

```python
from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader
```

Add `MODISVIIRSReader: DataGeometry.SWATH` to the `_geometry_property` mapping only if the reader does not implement its own `geometry` property. Because `MODISVIIRSReader` has a dynamic `geometry` property, do not add it to that mapping.

Add `"MODISVIIRSReader"` to `__all__`.

- [ ] **Step 5: Run reader tests**

Run: `pytest davinci_monet/tests/test_modis_viirs_reader.py -v`

Expected: all reader tests pass.

- [ ] **Step 6: Run registration smoke test**

Run: `pytest davinci_monet/tests/test_observation_readers.py::TestObservationRegistry::test_satellite_readers_registered -v`

Expected: pass. If adding a dedicated assertion for `"modis_viirs"`, update the test and verify it passes.

- [ ] **Step 7: Run mypy on the reader**

Run: `mypy davinci_monet/observations/satellite/modis_viirs.py`

Expected: no errors.

- [ ] **Step 8: Record status for authorized commit**

Run: `git status --short`

Suggested commit message when explicitly authorized:

```bash
git add davinci_monet/observations/satellite/modis_viirs.py \
  davinci_monet/observations/satellite/__init__.py davinci_monet/observations/__init__.py \
  davinci_monet/tests/test_modis_viirs_reader.py
git commit -m "feat(satellite): add catalog-driven MODIS/VIIRS reader foundation"
```

## Task 5: Prove Unified Pipeline Loading

**Files:**
- Modify: `davinci_monet/tests/test_load_sources_stage.py`

- [ ] **Step 1: Add failing pipeline source test**

Append to `TestLoadSourcesStage` in `davinci_monet/tests/test_load_sources_stage.py`:

```python
    def test_unified_modis_viirs_l3_source_loads_all_variables(self, tmp_path) -> None:
        path = tmp_path / "mod08_m3.nc"
        ds = xr.Dataset(
            {
                "AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean": (
                    ("time", "lat", "lon"),
                    np.ones((1, 2, 3), dtype=np.float32),
                ),
                "Extra_Parameter": (("time", "lat", "lon"), np.full((1, 2, 3), 2.0)),
            },
            coords={
                "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                "lat": np.array([10.0, 11.0]),
                "lon": np.array([100.0, 101.0, 102.0]),
            },
        )
        ds.to_netcdf(path)
        ctx = PipelineContext(
            config={
                "sources": {
                    "modis_l3": {
                        "type": "modis_viirs",
                        "product": "MOD08_M3",
                        "files": str(path),
                        "variables": "*",
                    }
                }
            }
        )

        result = LoadSourcesStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        assert set(ctx.sources) == {"modis_l3"}
        source = ctx.sources["modis_l3"]
        assert source.geometry is DataGeometry.GRID
        assert source.role == "obs"
        assert source.data.attrs["source_label"] == "modis_l3"
        assert source.data.attrs["geometry"] == "grid"
        assert "AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean" in source.data
        assert "Extra_Parameter" in source.data
```

- [ ] **Step 2: Run test to verify it fails before integration is complete**

Run: `pytest davinci_monet/tests/test_load_sources_stage.py::TestLoadSourcesStage::test_unified_modis_viirs_l3_source_loads_all_variables -v`

Expected before Task 4 is complete: FAIL because `type: modis_viirs` is not registered. Expected after Task 4 is complete: PASS.

- [ ] **Step 3: Run source-stage tests**

Run: `pytest davinci_monet/tests/test_load_sources_stage.py -v`

Expected: all source-stage tests pass.

- [ ] **Step 4: Run pipeline-adjacent focused suite**

Run:

```bash
pytest \
  davinci_monet/tests/test_load_sources_stage.py \
  davinci_monet/tests/test_modis_viirs_catalog.py \
  davinci_monet/tests/test_modis_viirs_reader.py \
  davinci_monet/tests/unit/config/test_schema.py::TestSourceConfig \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Record status for authorized commit**

Run: `git status --short`

Suggested commit message when explicitly authorized:

```bash
git add davinci_monet/tests/test_load_sources_stage.py
git commit -m "test(pipeline): cover MODIS/VIIRS source loading"
```

## Task 6: Verification And Formatting

**Files:**
- No code edits unless verification surfaces a concrete defect.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest \
  davinci_monet/tests/test_modis_viirs_catalog.py \
  davinci_monet/tests/test_modis_viirs_reader.py \
  davinci_monet/tests/test_load_sources_stage.py \
  davinci_monet/tests/unit/config/test_schema.py::TestSourceConfig \
  -v
```

Expected: all tests pass.

- [ ] **Step 2: Run broader observation/source regression**

Run:

```bash
pytest \
  davinci_monet/tests/test_observation_readers.py \
  davinci_monet/tests/test_load_sources_stage.py \
  davinci_monet/tests/unit/core/test_source_abstraction.py \
  -v
```

Expected: all tests pass.

- [ ] **Step 3: Run mypy on changed modules**

Run:

```bash
mypy \
  davinci_monet/config/schema.py \
  davinci_monet/pipeline/stages.py \
  davinci_monet/observations/satellite/catalog \
  davinci_monet/observations/satellite/modis_viirs.py
```

Expected: no new type errors.

- [ ] **Step 4: Run formatting checks on changed Python files**

Run:

```bash
black --check \
  davinci_monet/config/schema.py \
  davinci_monet/pipeline/stages.py \
  davinci_monet/observations/satellite/catalog \
  davinci_monet/observations/satellite/modis_viirs.py \
  davinci_monet/observations/satellite/__init__.py \
  davinci_monet/observations/__init__.py \
  davinci_monet/tests/test_modis_viirs_catalog.py \
  davinci_monet/tests/test_modis_viirs_reader.py \
  davinci_monet/tests/test_load_sources_stage.py \
  davinci_monet/tests/unit/config/test_schema.py
isort --check \
  davinci_monet/config/schema.py \
  davinci_monet/pipeline/stages.py \
  davinci_monet/observations/satellite/catalog \
  davinci_monet/observations/satellite/modis_viirs.py \
  davinci_monet/observations/satellite/__init__.py \
  davinci_monet/observations/__init__.py \
  davinci_monet/tests/test_modis_viirs_catalog.py \
  davinci_monet/tests/test_modis_viirs_reader.py \
  davinci_monet/tests/test_load_sources_stage.py \
  davinci_monet/tests/unit/config/test_schema.py
```

Expected: both commands pass. If either fails, run `black ...` or `isort ...` on the same paths, then rerun the checks.

- [ ] **Step 5: Run final status check**

Run: `git status --short`

Expected: only intended implementation files are modified or created. Existing unrelated untracked files such as `OpenRouter.api` remain untouched.

## Follow-On Plans

Write separate implementation plans for these approved design phases after this foundation lands:

1. HDF4/HDF-EOS variable discovery and file-engine selection.
2. Generic L2 geolocation attachment for separate geolocation files.
3. Reusable swath-to-grid gridding/cache extraction from the existing MODIS L2 path.
4. L3 projected-grid loading and explicit unsupported-reprojection errors.
5. L1 radiance/reflectance plus geolocation loading.
6. Full catalog expansion across atmosphere, land, snow/ice, and ocean product families.
