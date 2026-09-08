"""Satellite dataset readers.

This subpackage provides readers for satellite-based datasets.

Satellite-Specific Readers
--------------------------
These readers are optimized for specific satellite products and rely on
monetio for full functionality:

- **TROPOMIReader**: TROPOMI L2 products (NO2, O3, CO, HCHO, SO2)
  Requires: monetio.sat._tropomi_l2_no2_mm

- **TEMPOL2NO2Reader**: TEMPO L2 NO2 products
  Requires: monetio.sat._tempo_l2_no2_mm

- **MODISL2AODReader**: MODIS L2 AOD products
  Requires: monetio.sat.modis_l2

- **GOESL3AODReader**: GOES-ABI L3 AOD products
  Requires: monetio.sat.goes

- **MOPITTL3COReader**: MOPITT L3 CO products
  Requires: monetio.sat._mopitt_l3_mm

- **OMPSL3O3Reader**: OMPS L3 total ozone products
  Requires: monetio.sat._omps_l3_mm

- **MODISVIIRSReader**: Catalog-driven MODIS/VIIRS L3 grid products (MOD08_M3, MYD08_M3)

Generic Readers
---------------
These readers work with any satellite product but lack satellite-specific
features like projection handling and specialized QA filtering:

- **GenericL2Reader**: Generic L2 swath products
- **GenericL3Reader**: Generic L3 gridded products

Note
----
The satellite-specific readers fall back to basic xarray reading if monetio
is not available, but some features (projections, swath geometry) may not
work correctly without monetio.
"""

from davinci_monet.datasets.satellite.ceres_l3 import (  # noqa: F401  (registers both ceres_ebaf and ceres_syn1deg)
    CERESEBAFReader,
    CERESSYN1degReader,
)
from davinci_monet.datasets.satellite.ceres_ssf import (  # noqa: F401  (registers "ceres_ssf")
    CERESSSFReader,
)

# Generic readers (pure xarray, no monetio dependency)
from davinci_monet.datasets.satellite.generic_l2 import GenericL2Reader
from davinci_monet.datasets.satellite.generic_l3 import GenericL3Reader

# Satellite-specific L3 readers (require monetio for full functionality)
from davinci_monet.datasets.satellite.goes_l3_aod import (
    GOES_AOD_VARIABLE_MAPPING,
    GOES_VARIABLE_MAPPING,
    GOESL3AODReader,
    GOESReader,
)
from davinci_monet.datasets.satellite.modis_l2_aod import (
    MODIS_AOD_VARIABLE_MAPPING,
    MODISL2AODReader,
)
from davinci_monet.datasets.satellite.modis_viirs import (  # noqa: F401  (registers "modis_viirs")
    MODISVIIRSReader,
)
from davinci_monet.datasets.satellite.mopitt_l3_co import (
    MOPITT_CO_VARIABLE_MAPPING,
    MOPITTL3COReader,
)
from davinci_monet.datasets.satellite.omps_l3_o3 import (
    OMPS_O3_VARIABLE_MAPPING,
    OMPSL3O3Reader,
)
from davinci_monet.datasets.satellite.tempo_l2_no2 import (
    TEMPO_NO2_VARIABLE_MAPPING,
    TEMPOL2NO2Reader,
)

# Satellite-specific L2 readers (require monetio for full functionality)
from davinci_monet.datasets.satellite.tropomi import (
    TROPOMI_VARIABLE_MAPPING,
    TROPOMIReader,
)

__all__ = [
    # TROPOMI L2
    "TROPOMIReader",
    "TROPOMI_VARIABLE_MAPPING",
    # TEMPO L2 NO2
    "TEMPOL2NO2Reader",
    "TEMPO_NO2_VARIABLE_MAPPING",
    # MODIS L2 AOD
    "MODISL2AODReader",
    "MODIS_AOD_VARIABLE_MAPPING",
    # CERES L3
    "CERESEBAFReader",
    "CERESSYN1degReader",
    # CERES SSF L2
    "CERESSSFReader",
    # MODIS/VIIRS L3 catalog reader
    "MODISVIIRSReader",
    # GOES L3 AOD
    "GOESL3AODReader",
    "GOESReader",
    "GOES_AOD_VARIABLE_MAPPING",
    "GOES_VARIABLE_MAPPING",
    # MOPITT L3 CO
    "MOPITTL3COReader",
    "MOPITT_CO_VARIABLE_MAPPING",
    # OMPS L3 O3
    "OMPSL3O3Reader",
    "OMPS_O3_VARIABLE_MAPPING",
    # Generic L2
    "GenericL2Reader",
    # Generic L3
    "GenericL3Reader",
]
