"""Dataset reader package."""

# Import subpackages for source_registry side effects.
from davinci_monet.datasets import base as _base  # noqa: F401
from davinci_monet.datasets.aircraft import icartt as _icartt  # noqa: F401
from davinci_monet.datasets.cesm import CESMFVReader, CESMSEReader
from davinci_monet.datasets.cmaq import CMAQReader
from davinci_monet.datasets.generic import GenericReader
from davinci_monet.datasets.lightning import lma as _lma  # noqa: F401
from davinci_monet.datasets.merra2 import MERRA2Reader
from davinci_monet.datasets.power import POWERReader
from davinci_monet.datasets.satellite import ceres_l3 as _ceres_l3
from davinci_monet.datasets.satellite import ceres_ssf as _ceres_ssf
from davinci_monet.datasets.satellite import generic_l2 as _generic_l2
from davinci_monet.datasets.satellite import generic_l3 as _generic_l3
from davinci_monet.datasets.satellite import goes_l3_aod as _goes_l3_aod
from davinci_monet.datasets.satellite import modis_l2_aod as _modis_l2_aod
from davinci_monet.datasets.satellite import mopitt_l3_co as _mopitt_l3_co
from davinci_monet.datasets.satellite import omps_l3_o3 as _omps_l3_o3
from davinci_monet.datasets.satellite import tempo_l2_no2 as _tempo_l2_no2
from davinci_monet.datasets.satellite import tropomi as _tropomi
from davinci_monet.datasets.satellite import viirs_l2_aerdb as _viirs_l2_aerdb
from davinci_monet.datasets.sonde import ozonesonde as _ozonesonde  # noqa: F401
from davinci_monet.datasets.surface import aeronet as _aeronet
from davinci_monet.datasets.surface import airnow as _airnow
from davinci_monet.datasets.surface import aqs as _aqs
from davinci_monet.datasets.surface import openaq as _openaq
from davinci_monet.datasets.surface import pandora as _pandora
from davinci_monet.datasets.ufs import RRFSReader, UFSReader
from davinci_monet.datasets.wrfchem import WRFChemReader

__all__ = [
    # Readers
    "CESMFVReader",
    "CESMSEReader",
    "CMAQReader",
    "GenericReader",
    "MERRA2Reader",
    "POWERReader",
    "RRFSReader",
    "UFSReader",
    "WRFChemReader",
]
