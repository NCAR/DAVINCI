import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.datasets.satellite.catalog import UnknownProductError
from davinci_monet.datasets.satellite.modis_viirs import MODISVIIRSReader

_D3_AOD = "AOD_550_Dark_Target_Deep_Blue_Combined_Mean"
_D3_AOD_STDDEV = "AOD_550_Dark_Target_Deep_Blue_Combined_Standard_Deviation"


def _write_mod08_like(path, fname):
    # 4x8 grid mirroring MOD08_M3 dim/coord layout
    lat = np.linspace(89.5, -89.5, 4)
    lon = np.linspace(-179.5, 179.5, 8)
    aod = np.random.default_rng(0).uniform(0.0, 1.0, size=(4, 8)).astype("float32")
    ds = xr.Dataset(
        {"Aerosol_Optical_Depth_Land_Ocean_Mean_Mean": (("YDim:mod08", "XDim:mod08"), aod)},
        coords={"YDim": ("YDim:mod08", lat), "XDim": ("XDim:mod08", lon)},
    )
    fpath = path / fname
    ds.to_netcdf(fpath)
    return str(fpath)


def test_reader_returns_grid_with_time_and_display_name(tmp_path):
    # 2024-032 = day-of-year 32 = 2024-02-01
    f = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()
    ds = reader.open([f], variables=["aod_550nm"], product="MOD08_M3")

    assert reader.geometry == DataGeometry.GRID
    assert "aod_550nm" in ds.data_vars  # SDS renamed to display name
    assert {"lat", "lon"}.issubset(set(ds.coords))  # XDim/YDim -> lon/lat
    assert "time" in ds.coords
    assert str(ds["time"].values[0])[:7] == "2024-02"  # filename month parsed
    assert ds["aod_550nm"].attrs.get("wavelength_nm") == 550


def test_reader_concatenates_months_sorted(tmp_path):
    f2 = _write_mod08_like(
        tmp_path, "MOD08_M3.A2024061.061.0000.nc"
    )  # Mar 1 (day 61 in leap year 2024)
    f1 = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")  # Feb
    reader = MODISVIIRSReader()
    ds = reader.open([f2, f1], variables=["aod_550nm"], product="MOD08_M3")
    months = [str(t)[:7] for t in ds["time"].values]
    assert months == ["2024-02", "2024-03"]  # sorted ascending


def test_monthly_reader_still_snaps_midmonth_token_to_month_start(tmp_path):
    f = _write_mod08_like(tmp_path, "MOD08_M3.A2024045.061.0000.nc")

    ds = MODISVIIRSReader().open([f], variables=["aod_550nm"], product="MOD08_M3")

    assert ds["time"].values[0] == np.datetime64("2024-02-01T00:00:00.000000000")


def _write_d3_subset(
    path: Path,
    fname: str,
    timestamp: str,
    *,
    source_product: str = "MYD08_D3",
    values: np.ndarray | None = None,
) -> str:
    lat = np.array([-0.5, 0.5], dtype=np.float32)
    lon = np.array([-1.5, -0.5, 0.5], dtype=np.float32)
    if values is None:
        values = np.array([[0.1, 0.2, np.nan], [0.3, 0.4, 0.5]], dtype=np.float32)
    ds = xr.Dataset(
        {_D3_AOD: (("time", "lat", "lon"), values[np.newaxis, ...])},
        coords={
            "time": np.array([timestamp], dtype="datetime64[ns]"),
            "lat": lat,
            "lon": lon,
        },
        attrs={"source_product": source_product},
    )
    ds[_D3_AOD].attrs.update(units="1", valid_min=0.0, valid_max=5.0)
    fpath = path / fname
    ds.to_netcdf(fpath)
    return str(fpath)


def _write_d3_hdf4(path: Path, fname: str, *, qa_screened: bool = True) -> str:
    hdf_api = pytest.importorskip("pyhdf.SD", reason="pyhdf required for MODIS D3 HDF4 tests")
    SD, SDC = hdf_api.SD, hdf_api.SDC
    fpath = path / fname
    hdf = SD(str(fpath), SDC.WRITE | SDC.CREATE)

    lon = hdf.create("XDim", SDC.FLOAT32, 3)
    lon.dim(0).setname("XDim:mod08")
    lon[:] = np.array([-1.5, -0.5, 0.5], dtype=np.float32)
    lon.endaccess()
    lat = hdf.create("YDim", SDC.FLOAT32, 2)
    lat.dim(0).setname("YDim:mod08")
    lat[:] = np.array([0.5, -0.5], dtype=np.float32)
    lat.endaccess()

    aod = hdf.create(_D3_AOD, SDC.INT16, (2, 3))
    aod.dim(0).setname("YDim:mod08")
    aod.dim(1).setname("XDim:mod08")
    aod[:] = np.array([[100, 200, -9999], [300, 5001, 500]], dtype=np.int16)
    aod.attr("_FillValue").set(SDC.INT16, -9999)
    aod.attr("valid_range").set(SDC.INT16, [0, 5000])
    aod.attr("scale_factor").set(SDC.FLOAT32, 0.001)
    aod.attr("add_offset").set(SDC.FLOAT32, 0.0)
    if qa_screened:
        aod.attr("Masked_With_QA_Usefulness_Flag").set(SDC.CHAR, "True")
    aod.endaccess()

    stddev = hdf.create(_D3_AOD_STDDEV, SDC.INT16, (2, 3))
    stddev.dim(0).setname("YDim:mod08")
    stddev.dim(1).setname("XDim:mod08")
    stddev[:] = np.array([[20, 30, -9999], [40, 5001, 50]], dtype=np.int16)
    stddev.attr("_FillValue").set(SDC.INT16, -9999)
    stddev.attr("valid_range").set(SDC.INT16, [0, 5000])
    stddev.attr("scale_factor").set(SDC.FLOAT32, 0.001)
    stddev.attr("add_offset").set(SDC.FLOAT32, 0.0)
    if qa_screened:
        stddev.attr("Masked_With_QA_Usefulness_Flag").set(SDC.CHAR, "True")
    stddev.endaccess()
    hdf.end()
    return str(fpath)


def test_daily_d3_hdf4_returns_canonical_qa_screened_aod(tmp_path):
    f = _write_d3_hdf4(tmp_path, "MYD08_D3.A2008184.061.0000.hdf")

    ds = MODISVIIRSReader().open(
        [f],
        variables=["aod_550nm", "aod_550nm_stddev"],
        product="MYD08_D3",
    )

    assert ds["time"].values[0] == np.datetime64("2008-07-02T00:00:00.000000000")
    assert ds.attrs["cadence"] == "daily"
    assert ds["aod_550nm"].dims == ("time", "lat", "lon")
    np.testing.assert_allclose(ds["aod_550nm"].isel(time=0, lat=0, lon=0), 0.1)
    np.testing.assert_allclose(ds["aod_550nm_stddev"].isel(time=0, lat=0, lon=0), 0.02)
    assert int(ds["aod_550nm"].isnull().sum()) == 2
    assert int(ds["aod_550nm_stddev"].isnull().sum()) == 2
    assert ds["aod_550nm"].attrs["qa_screening"] == "level2_usefulness_flag"
    assert "finite cells" in ds["aod_550nm"].attrs["support_definition"]


def test_daily_d3_hdf4_rejects_unverified_qa_contract(tmp_path):
    f = _write_d3_hdf4(
        tmp_path,
        "MYD08_D3.A2008184.061.0000.hdf",
        qa_screened=False,
    )

    with pytest.raises(ValueError, match="QA usefulness screening"):
        MODISVIIRSReader().open([f], variables=["aod_550nm"], product="MYD08_D3")


def test_daily_d3_staged_netcdf_preserves_days_and_masks_range(tmp_path):
    values = np.array([[-0.1, 0.2, np.nan], [0.3, 5.1, 0.5]], dtype=np.float32)
    f2 = _write_d3_subset(
        tmp_path,
        "MYD08_D3.A2008184.061.0000.AOD550.nc4",
        "2008-07-02T12:00:00",
        values=values,
    )
    f1 = _write_d3_subset(
        tmp_path,
        "MYD08_D3.A2008183.061.0000.AOD550.nc4",
        "2008-07-01T12:00:00",
    )

    ds = MODISVIIRSReader().open([f2, f1], variables=["aod_550nm"], product="MYD08_D3")

    np.testing.assert_array_equal(
        ds["time"].values,
        np.array(["2008-07-01", "2008-07-02"], dtype="datetime64[ns]"),
    )
    assert ds["aod_550nm"].shape == (2, 2, 3)
    assert int(ds["aod_550nm"].isel(time=1).isnull().sum()) == 3


def test_daily_d3_rejects_filename_time_mismatch(tmp_path):
    f = _write_d3_subset(
        tmp_path,
        "MYD08_D3.A2008183.061.0000.AOD550.nc4",
        "2008-07-02T12:00:00",
    )

    with pytest.raises(ValueError, match="does not match filename date"):
        MODISVIIRSReader().open([f], variables=["aod_550nm"], product="MYD08_D3")


def test_daily_d3_rejects_duplicate_calendar_days(tmp_path):
    f1 = _write_d3_subset(
        tmp_path,
        "MYD08_D3.A2008183.061.0001.AOD550.nc4",
        "2008-07-01T12:00:00",
    )
    f2 = _write_d3_subset(
        tmp_path,
        "MYD08_D3.A2008183.061.0002.AOD550.nc4",
        "2008-07-01T12:00:00",
    )

    with pytest.raises(ValueError, match="duplicate daily timestamps"):
        MODISVIIRSReader().open([f1, f2], variables=["aod_550nm"], product="MYD08_D3")


def _write_mod08_hdf4_like(path, fname):
    """Fixture where XDim/YDim are plain data variables (not coords).

    Real MOD08_M3 HDF4 files expose XDim/YDim as 1-D SDS data variables on
    dims named ``XDim:mod08`` / ``YDim:mod08``.  When the reader selects only
    the AOD SDS, those axis variables would be dropped unless explicitly
    promoted to coords first.  This fixture reproduces that layout so the
    coord-attachment fix can be exercised without a real HDF4 file.
    """
    lat = np.linspace(89.5, -89.5, 4)
    lon = np.linspace(-179.5, 179.5, 8)
    aod = np.random.default_rng(1).uniform(0.0, 1.0, size=(4, 8)).astype("float32")
    # XDim and YDim are data_vars on their own dims, NOT in coords={}.
    ds = xr.Dataset(
        {
            "Aerosol_Optical_Depth_Land_Ocean_Mean_Mean": (
                ("YDim:mod08", "XDim:mod08"),
                aod,
            ),
            "YDim": ("YDim:mod08", lat),
            "XDim": ("XDim:mod08", lon),
        }
    )
    fpath = path / fname
    ds.to_netcdf(fpath)
    return str(fpath)


def test_reader_attaches_lat_lon_coords_when_axes_are_data_vars(tmp_path):
    """Reader must attach lat/lon coords even when XDim/YDim are data vars.

    In real MOD08_M3 HDF4 files the grid-axis arrays (XDim, YDim) are plain
    data variables, not xarray coordinates.  Before the fix, selecting the AOD
    SDS would drop them, leaving lon/lat as dimension names with no coordinate
    values attached.  This test exercises the coord-promotion path.
    """
    f = _write_mod08_hdf4_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()
    ds = reader.open([f], variables=["aod_550nm"], product="MOD08_M3")

    assert {"lat", "lon"}.issubset(ds.coords), (
        "lat/lon must be coordinate arrays, not bare dimension names. "
        "Check that XDim/YDim are promoted to coords before variable selection."
    )
    assert ds["lat"].shape == (4,)
    assert ds["lon"].shape == (8,)
    # Coordinate values should be actual degree values, not indices.
    assert float(ds["lat"].max()) > 80.0
    assert float(ds["lon"].min()) < -170.0


def test_reader_unknown_product_raises(tmp_path):
    f = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()
    with pytest.raises(UnknownProductError):
        reader.open([f], variables=["aod_550nm"], product="NOPE_M3")


def test_reader_is_registered_via_package_import():
    """Registration must happen via the datasets package, not a direct module import."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import davinci_monet.datasets; "
                "from davinci_monet.core.registry import source_registry; "
                "assert 'modis_viirs' in source_registry.list(), "
                "'modis_viirs not registered after importing davinci_monet.datasets'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Part-1 correctness: junk-var exclusion + no duplicate-dim warning
# ---------------------------------------------------------------------------


def _write_mod08_with_junk_vars(path: Path, fname: str) -> str:
    """Write a MOD08-like .nc fixture with the AOD SDS + XDim/YDim + junk variables.

    Some junk variables use the duplicate-dimension pattern (same dim name on
    two axes) that real MOD08_M3 histogram SDS exhibit, to exercise the
    duplicate-dim warning suppression path.
    """
    lat = np.linspace(89.5, -89.5, 4)
    lon = np.linspace(-179.5, 179.5, 8)
    aod = np.random.default_rng(42).uniform(0.0, 1.0, size=(4, 8)).astype("float32")
    junk1 = np.zeros((4, 8), dtype="float32")
    junk2 = np.ones((4, 8), dtype="float32")
    # Junk variable on identical dim names (simulates histogram SDS in MOD08)
    hist = np.zeros((5, 5), dtype="float32")

    ds = xr.Dataset(
        {
            "Aerosol_Optical_Depth_Land_Ocean_Mean_Mean": (("YDim:mod08", "XDim:mod08"), aod),
            "Junk_Variable_One": (("YDim:mod08", "XDim:mod08"), junk1),
            "Junk_Variable_Two": (("YDim:mod08", "XDim:mod08"), junk2),
            # Histogram-like: two different dims but both named similarly
            "Histogram_Junk": (("bin_dim", "bin_dim2"), hist),
            "YDim": ("YDim:mod08", lat),
            "XDim": ("XDim:mod08", lon),
        }
    )
    fpath = path / fname
    ds.to_netcdf(fpath)
    return str(fpath)


def test_reader_loads_only_aod_and_no_junk_vars(tmp_path):
    """Reader must return only the requested SDS (aod_550nm) + lat/lon/time.

    Junk variables in the file must not appear in the returned dataset.
    No duplicate-dim UserWarning should be raised during the open.
    """
    from pathlib import Path

    f = _write_mod08_with_junk_vars(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = reader.open([f], variables=["aod_550nm"], product="MOD08_M3")

    # No duplicate-dim warning should escape from the reader
    dup_warns = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "Duplicate dimension" in str(w.message)
    ]
    assert not dup_warns, f"Unexpected duplicate-dim warnings: {dup_warns}"

    # Dataset must contain aod_550nm and NOT the junk variables
    assert "aod_550nm" in ds.data_vars, "aod_550nm missing from result"
    for junk in ("Junk_Variable_One", "Junk_Variable_Two", "Histogram_Junk"):
        assert junk not in ds.data_vars, f"Junk variable {junk!r} leaked into result"
    # Axis variables must not appear as data variables (they should be coords or absent)
    assert "XDim" not in ds.data_vars, "XDim should be a coord, not data_var"
    assert "YDim" not in ds.data_vars, "YDim should be a coord, not data_var"

    # Structural checks
    assert {"lat", "lon"}.issubset(ds.coords)
    assert "time" in ds.coords
    assert len(ds.data_vars) == 1, f"Expected 1 data_var, got: {list(ds.data_vars)}"


# ---------------------------------------------------------------------------
# Part-2: progress_callback
# ---------------------------------------------------------------------------


def test_progress_callback_called_per_file(tmp_path):
    """progress_callback must be called once per file with (idx, total, name)."""
    f1 = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    f2 = _write_mod08_like(tmp_path, "MOD08_M3.A2024061.061.0000.nc")

    calls: list[tuple[int, int, str]] = []

    def callback(idx: int, total: int, name: str) -> None:
        calls.append((idx, total, name))

    reader = MODISVIIRSReader()
    reader.open([f1, f2], variables=["aod_550nm"], product="MOD08_M3", progress_callback=callback)

    assert len(calls) == 2, f"Expected 2 callback calls, got {len(calls)}: {calls}"
    assert calls[0] == (1, 2, "MOD08_M3.A2024032.061.0000.nc"), f"First call wrong: {calls[0]}"
    assert calls[1] == (2, 2, "MOD08_M3.A2024061.061.0000.nc"), f"Second call wrong: {calls[1]}"


def test_progress_callback_none_does_not_raise(tmp_path):
    """Calling open() without progress_callback (default None) must not raise."""
    f = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()
    # Should not raise TypeError or AttributeError
    ds = reader.open([f], variables=["aod_550nm"], product="MOD08_M3")
    assert "aod_550nm" in ds.data_vars
