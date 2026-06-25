"""Unit tests for the CERES SSF (L2 footprint) reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.datasets.satellite.ceres_ssf import (  # noqa: E402
    SSF_CATALOG,
    CERESSSFReader,
)

_FILL32 = 3.4028234663852886e38
_JD_EPOCH = 2440587.5  # Julian Date of 1970-01-01T00


def _jd(iso: str) -> float:
    """ISO timestamp -> Julian Date (float days)."""
    ns = (np.datetime64(iso) - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return _JD_EPOCH + float(ns) / 86400.0


def _pyhdf_sd():
    return pytest.importorskip("pyhdf.SD", reason="pyhdf required for SSF HDF4 tests")


def _write_ssf_hdf4(
    path: Path,
    n: int = 6,
    base_iso: str = "2026-04-01T00:30:00",
    flux_name: str = "CERES LW TOA flux - upwards",
    time_name: str = "Time of dataset",
    fill_flux_idx: int | None = None,
    fill_coord_idx: int | None = None,
) -> Path:
    """Write a minimal SSF-like HDF4 file: 1-D footprint SDS with real names."""
    pyhdf_SD = _pyhdf_sd()
    SD, SDC = pyhdf_SD.SD, pyhdf_SD.SDC
    times = np.array([_jd(base_iso) + i * (10.0 / 86400.0) for i in range(n)])  # 10 s apart
    colat = np.linspace(60.0, 120.0, n).astype(np.float32)  # lat +30 .. -30
    lon = np.linspace(10.0, 350.0, n).astype(np.float32)  # crosses the 180 wrap
    flux = np.linspace(150.0, 300.0, n).astype(np.float32)
    if fill_flux_idx is not None:
        flux[fill_flux_idx] = _FILL32
    if fill_coord_idx is not None:
        colat[fill_coord_idx] = _FILL32

    f = SD(str(path), SDC.WRITE | SDC.CREATE)

    def _sds(name: str, data: np.ndarray, typ: int, fill: float, vr: list[float]) -> None:
        s = f.create(name, typ, n)
        s.dim(0).setname("Footprints")
        s[:] = data
        s.attr("_FillValue").set(typ, fill)
        s.attr("valid_range").set(typ, vr)
        s.endaccess()

    _sds(time_name, times, SDC.FLOAT64, 1.7976931348623157e308, [2440000.0, 2480000.0])
    _sds("Colatitude of CERES FOV at surface", colat, SDC.FLOAT32, _FILL32, [0.0, 180.0])
    _sds("Longitude of CERES FOV at surface", lon, SDC.FLOAT32, _FILL32, [0.0, 360.0])
    _sds(flux_name, flux, SDC.FLOAT32, _FILL32, [0.0, 500.0])
    f.end()
    return path


def test_reader_registered_and_swath_geometry() -> None:
    reader_cls = source_registry.get("ceres_ssf")
    reader = reader_cls()
    assert reader.name == "ceres_ssf"
    assert reader.geometry is DataGeometry.SWATH


def test_catalog_covers_spec_canonical_set() -> None:
    assert {
        "toa_sw_up",
        "toa_lw_up",
        "toa_solar_in",
        "sfc_sw_down",
        "sfc_sw_down_clr",
        "sfc_lw_down",
        "sfc_lw_down_clr",
        "sfc_sw_net",
        "sfc_lw_net",
    } == set(SSF_CATALOG)


def test_hdf4_canonical_open_and_coords(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100")

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.attrs["geometry"] == "swath"
    assert set(ds.data_vars) == {"toa_lw_up"}
    assert ds["toa_lw_up"].dims == ("time",)
    assert ds.sizes["time"] == 6
    # colat 60..120 -> lat +30..-30
    np.testing.assert_allclose(ds["lat"].values[[0, -1]], [30.0, -30.0])
    # lon 0-360 -> wrapped to [-180, 180): 350 -> -10
    assert float(ds["lon"].values[-1]) == pytest.approx(-10.0)
    assert float(ds["lon"].values[0]) == pytest.approx(10.0)
    # Julian time decoded: first footprint at base time (10 s spacing after)
    assert ds["time"].values[0] == np.datetime64("2026-04-01T00:30:00")
    assert ds["time"].values[1] - ds["time"].values[0] == np.timedelta64(10, "s")


def test_hdf4_accepts_time_of_observation_alias(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        time_name="Time of observation",
    )

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.sizes["time"] == 6
    assert ds["time"].values[0] == np.datetime64("2026-04-01T00:30:00")


def test_hdf4_resolves_model_b_surface_flux_alias(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        flux_name="CERES downward SW surface flux - Model B",
    )

    ds = CERESSSFReader().open([p], variables=["sfc_sw_down"])

    assert set(ds.data_vars) == {"sfc_sw_down"}
    np.testing.assert_allclose(ds["sfc_sw_down"].values, np.linspace(150.0, 300.0, 6))


def test_hdf4_flux_fill_masked_but_footprint_kept(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        fill_flux_idx=2,
    )

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.sizes["time"] == 6  # footprint kept; value masked
    assert bool(np.isnan(ds["toa_lw_up"].values[2]))
    assert int(np.isnan(ds["toa_lw_up"].values).sum()) == 1


def test_hdf4_invalid_coord_footprint_dropped(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        fill_coord_idx=1,
    )

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.sizes["time"] == 5  # footprint without a valid position is unusable
    assert not np.isnan(ds["lat"].values).any()


def test_hdf4_multigranule_concat_sorted(tmp_path: Path) -> None:
    _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040101",
        base_iso="2026-04-01T01:30:00",
    )
    _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        base_iso="2026-04-01T00:30:00",
    )

    ds = CERESSSFReader().open(sorted(tmp_path.glob("CER_SSF_*")), variables=["toa_lw_up"])

    assert ds.sizes["time"] == 12
    t = ds["time"].values
    assert (np.diff(t) > np.timedelta64(0, "s")).all()


def test_time_range_prunes_files_before_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_file = tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100"
    keep_file = tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040200"
    old_file.write_bytes(b"\x0e\x03\x13\x01")
    keep_file.write_bytes(b"\x0e\x03\x13\x01")
    opened: list[str] = []

    def _fake_open_one_hdf4(
        self: CERESSSFReader, path: Path, variables: list[str] | None
    ) -> xr.Dataset:
        if path == old_file:
            raise AssertionError("out-of-window file was opened")
        opened.append(path.name)
        return xr.Dataset(
            {"toa_lw_up": ("time", np.array([250.0]))},
            coords={
                "time": np.array(["2026-04-02T00:30:00"], dtype="datetime64[ns]"),
                "lat": ("time", np.array([0.0])),
                "lon": ("time", np.array([0.0])),
            },
        )

    monkeypatch.setattr(CERESSSFReader, "_open_one_hdf4", _fake_open_one_hdf4)

    ds = CERESSSFReader().open(
        [old_file, keep_file],
        variables=["toa_lw_up"],
        time_range=("2026-04-02T00:00:00", "2026-04-02T23:59:59"),
    )

    assert opened == [keep_file.name]
    assert ds.sizes["time"] == 1
    assert ds["time"].values[0] == np.datetime64("2026-04-02T00:30:00")


def test_hdf4_raw_source_name_escape(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(
        tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100",
        flux_name="CERES SW TOA flux - upwards",
    )

    ds = CERESSSFReader().open([p], variables=["CERES SW TOA flux - upwards"])

    assert "CERES SW TOA flux - upwards" in ds.data_vars


def test_hdf4_missing_variable_raises(tmp_path: Path) -> None:
    p = _write_ssf_hdf4(tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100")

    with pytest.raises(ValueError, match="not found"):
        CERESSSFReader().open([p], variables=["toa_solar_in"])


# ---------------------------------------------------------------------------
# netCDF (Edition1C) path
# ---------------------------------------------------------------------------


def _write_ssf_netcdf(
    path: Path,
    n: int = 6,
    base_iso: str = "2026-04-01T00:30:00",
    nc_var: str = "toa_longwave_flux",
    fill_flux_idx: int | None = None,
    fill_coord_idx: int | None = None,
) -> Path:
    """Write a minimal Edition1C-like grouped netCDF granule."""
    base = np.datetime64(base_iso)
    times = base + np.arange(n) * np.timedelta64(10, "s")
    lat = np.linspace(30.0, -30.0, n)
    lon = np.linspace(10.0, 350.0, n)  # 0-360 in-file; reader wraps
    flux = np.linspace(150.0, 300.0, n)
    if fill_coord_idx is not None:
        lat[fill_coord_idx] = np.nan  # xarray-decoded fill arrives as NaN

    pos = xr.Dataset(
        {
            "time": ("Footprints", times),
            "instrument_fov_latitude": ("Footprints", lat),
            "instrument_fov_longitude": ("Footprints", lon),
        }
    )
    flux_da = xr.DataArray(flux, dims=("Footprints",))
    if fill_flux_idx is not None:
        flux_vals = flux.copy()
        flux_vals[fill_flux_idx] = np.nan
        flux_da = xr.DataArray(flux_vals, dims=("Footprints",))
    fluxes = xr.Dataset({nc_var: flux_da})

    pos.to_netcdf(path, group="Time_and_Position", mode="w")
    fluxes.to_netcdf(path, group="TOA_and_Surface_Fluxes", mode="a")
    return path


def test_netcdf_canonical_open_matches_hdf4_semantics(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc")

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.attrs["geometry"] == "swath"
    assert set(ds.data_vars) == {"toa_lw_up"}
    assert ds["toa_lw_up"].dims == ("time",)
    assert ds.sizes["time"] == 6
    np.testing.assert_allclose(ds["lat"].values[[0, -1]], [30.0, -30.0])
    assert float(ds["lon"].values[-1]) == pytest.approx(-10.0)
    assert ds["time"].values[0] == np.datetime64("2026-04-01T00:30:00")


def test_netcdf_resolves_model_b_surface_flux_alias(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(
        tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc",
        nc_var="model_b_surface_shortwave_downward_flux",
    )

    ds = CERESSSFReader().open([p], variables=["sfc_sw_down"])

    assert set(ds.data_vars) == {"sfc_sw_down"}
    np.testing.assert_allclose(ds["sfc_sw_down"].values, np.linspace(150.0, 300.0, 6))


def test_netcdf_invalid_coord_footprint_dropped(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(
        tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc",
        fill_coord_idx=1,
    )

    ds = CERESSSFReader().open([p], variables=["toa_lw_up"])

    assert ds.sizes["time"] == 5
    assert not np.isnan(ds["lat"].values).any()


def test_netcdf_group_path_escape(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(
        tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc",
        nc_var="toa_longwave_channel_flux",
    )

    ds = CERESSSFReader().open([p], variables=["TOA_and_Surface_Fluxes/toa_longwave_channel_flux"])

    assert "TOA_and_Surface_Fluxes/toa_longwave_channel_flux" in ds.data_vars


def test_netcdf_missing_variable_raises(tmp_path: Path) -> None:
    p = _write_ssf_netcdf(tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc")

    with pytest.raises(ValueError, match="not found"):
        CERESSSFReader().open([p], variables=["toa_solar_in"])


def test_mixed_formats_rejected(tmp_path: Path) -> None:
    h4 = _write_ssf_hdf4(tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100")
    nc = _write_ssf_netcdf(tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc")

    with pytest.raises(ValueError, match="mixed"):
        CERESSSFReader().open([h4, nc], variables=["toa_lw_up"])


# ---------------------------------------------------------------------------
# Grid-aligned helper for pipeline integration tests
# ---------------------------------------------------------------------------


def _write_ssf_hdf4_grid(
    path: Path,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    base_iso: str,
) -> Path:
    """Write an SSF-like HDF4 granule with one footprint per (lat, lon) center."""
    pyhdf_SD = _pyhdf_sd()
    SD, SDC = pyhdf_SD.SD, pyhdf_SD.SDC
    lat2d, lon2d = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    lat = lat2d.ravel().astype(np.float32)
    lon = lon2d.ravel().astype(np.float32)
    n = lat.size
    times = np.array([_jd(base_iso) + i * (1.0 / 86400.0) for i in range(n)])
    flux = np.linspace(150.0, 300.0, n).astype(np.float32)

    f = SD(str(path), SDC.WRITE | SDC.CREATE)

    def _sds(name: str, data: np.ndarray, typ: int, fill: float, vr: list) -> None:
        s = f.create(name, typ, n)
        s.dim(0).setname("Footprints")
        s[:] = data
        s.attr("_FillValue").set(typ, fill)
        s.attr("valid_range").set(typ, vr)
        s.endaccess()

    _sds("Time of dataset", times, SDC.FLOAT64, 1.7976931348623157e308, [2440000.0, 2480000.0])
    _sds(
        "Colatitude of CERES FOV at surface",
        (90.0 - lat).astype(np.float32),
        SDC.FLOAT32,
        _FILL32,
        [0.0, 180.0],
    )
    _sds(
        "Longitude of CERES FOV at surface",
        np.where(lon < 0, lon + 360.0, lon).astype(np.float32),
        SDC.FLOAT32,
        _FILL32,
        [0.0, 360.0],
    )
    _sds("CERES LW TOA flux - upwards", flux, SDC.FLOAT32, _FILL32, [0.0, 500.0])
    f.end()
    return path


# ---------------------------------------------------------------------------
# Cross-format identity test
# ---------------------------------------------------------------------------


def test_hdf4_and_netcdf_paths_produce_identical_output(tmp_path: Path) -> None:
    h4 = _write_ssf_hdf4(tmp_path / "CER_SSF_Terra-FM1-MODIS_Edition4A_410406.2026040100")
    nc = _write_ssf_netcdf(tmp_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026040100.nc")

    ds_h4 = CERESSSFReader().open([h4], variables=["toa_lw_up"])
    ds_nc = CERESSSFReader().open([nc], variables=["toa_lw_up"])

    np.testing.assert_array_equal(ds_h4["time"].values, ds_nc["time"].values)
    np.testing.assert_array_equal(ds_h4["lat"].values, ds_nc["lat"].values)
    np.testing.assert_array_equal(ds_h4["lon"].values, ds_nc["lon"].values)
    np.testing.assert_array_equal(ds_h4["toa_lw_up"].values, ds_nc["toa_lw_up"].values)
