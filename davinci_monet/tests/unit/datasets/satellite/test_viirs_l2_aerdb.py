import os
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.datasets.satellite.viirs_l2_aerdb import VIIRSL2AERDBReader

REAL = (
    "/glade/work/fillmore/DAVINCI/test-data/VIIRS_original/"
    "AERDB_L2_VIIRS_NOAA20.A2023362.0000.002.2023362124926.nc"
)


def _write_aerdb_like(path, *, na=2, nx=2):
    lat = np.array([[10.0, 10.0], [11.0, 11.0]], np.float32)
    lon = np.array([[20.0, 21.0], [20.0, 21.0]], np.float32)
    aod_land = np.array([[0.30, -999.0], [0.50, 0.40]], np.float32)
    qa_land = np.array([[3.0, 0.0], [2.0, 1.0]], np.float64)
    eu_land = np.array([[0.10, 0.10], [0.12, 0.08]], np.float32)
    ds = xr.Dataset(
        {
            "Aerosol_Optical_Thickness_550_Land": (
                ("Idx_Atrack", "Idx_Xtrack"),
                aod_land,
            ),
            "Aerosol_Optical_Thickness_550_Ocean": (
                ("Idx_Atrack", "Idx_Xtrack"),
                np.full((na, nx), -999.0, np.float32),
            ),
            "Aerosol_Optical_Thickness_550_Expected_Uncertainty_Land": (
                ("Idx_Atrack", "Idx_Xtrack"),
                eu_land,
            ),
            "Aerosol_Optical_Thickness_550_Expected_Uncertainty_Ocean": (
                ("Idx_Atrack", "Idx_Xtrack"),
                np.full((na, nx), -999.0, np.float32),
            ),
            "Aerosol_Optical_Thickness_QA_Flag_Land": (
                ("Idx_Atrack", "Idx_Xtrack"),
                qa_land,
            ),
            "Aerosol_Optical_Thickness_QA_Flag_Ocean": (
                ("Idx_Atrack", "Idx_Xtrack"),
                np.zeros((na, nx)),
            ),
            "Latitude": (("Idx_Atrack", "Idx_Xtrack"), lat),
            "Longitude": (("Idx_Atrack", "Idx_Xtrack"), lon),
            "Scan_Start_Time": (
                ("Idx_Atrack", "Idx_Xtrack"),
                np.zeros((na, nx), np.float64),
            ),
        }
    )
    ds["Aerosol_Optical_Thickness_550_Land"].attrs["_FillValue"] = -999.0
    ds.to_netcdf(path)


def test_reader_basic_swath(tmp_path):
    path = str(tmp_path / "g.nc")
    _write_aerdb_like(path)
    reader = VIIRSL2AERDBReader()
    ds = reader.open([path])
    assert reader.name == "viirs_l2_aerdb"
    assert reader.geometry == DataGeometry.SWATH
    assert {"lat", "lon"}.issubset(set(ds.coords))
    assert ds["Aerosol_Optical_Thickness_550_Land"].dims == (
        "Idx_Atrack",
        "Idx_Xtrack",
    )
    assert np.isnan(float(ds["Aerosol_Optical_Thickness_550_Land"].values[0, 1]))
    assert ds.attrs["geometry"] == "swath"


def test_reader_qa_threshold_masks_land_aod(tmp_path):
    path = str(tmp_path / "g.nc")
    _write_aerdb_like(path)
    ds = VIIRSL2AERDBReader().open([path], qa_threshold=2)
    land = ds["Aerosol_Optical_Thickness_550_Land"].values
    assert not np.isnan(land[0, 0])
    assert np.isnan(land[1, 1])


def test_reader_registered_via_package_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import davinci_monet.datasets; "
                "from davinci_monet.core.registry import source_registry; "
                "assert 'viirs_l2_aerdb' in source_registry.list()"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_reader_real_sample_invariants():
    if not os.path.exists(REAL):
        pytest.skip(f"real VIIRS sample not present: {REAL}")
    ds = VIIRSL2AERDBReader().open([REAL])
    assert ds["lat"].shape == (402, 400)
    assert {"lat", "lon"}.issubset(set(ds.coords))
    assert "Aerosol_Optical_Thickness_550_Land" in ds.data_vars
    assert "Scan_Start_Time" in ds.variables
