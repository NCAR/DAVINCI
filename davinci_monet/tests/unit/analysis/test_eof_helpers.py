"""EOF preprocessing/weighting helpers behave correctly (and log, not warn)."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from davinci_monet.analysis.eof import (
    _area_weight,
    _effective_n,
    _fix_sign,
    _lat_coord,
    _layer_mass_weight,
    _vertical_dim,
)


def _grid(nt=10, nlat=4, nlon=5, nlev=0) -> xr.DataArray:
    lat = np.linspace(10, 40, nlat)
    lon = np.linspace(-120, -90, nlon)
    if nlev:
        dims: tuple[str, ...] = ("time", "lev", "lat", "lon")
        shape: tuple[int, ...] = (nt, nlev, nlat, nlon)
        coords = {
            "time": np.arange(nt),
            "lev": np.arange(nlev),
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        }
    else:
        dims = ("time", "lat", "lon")
        shape = (nt, nlat, nlon)
        coords = {
            "time": np.arange(nt),
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        }
    return xr.DataArray(np.ones(shape), dims=dims, coords=coords, name="O3")


def test_lat_and_area_weight() -> None:
    da = _grid()
    lat = _lat_coord(da)
    w = _area_weight(da, lat)
    assert float(w.isel(lat=0)) > float(w.isel(lat=-1))


def test_vertical_dim_detection() -> None:
    assert _vertical_dim(_grid(nlev=0), _lat_coord(_grid()), _grid()["longitude"]) is None
    da3 = _grid(nlev=3)
    assert _vertical_dim(da3, da3["latitude"], da3["longitude"]) == "lev"


def test_layer_mass_weight_fallback_logs_not_warns(caplog) -> None:
    da3 = _grid(nlev=3)
    with caplog.at_level(logging.WARNING):
        mw = _layer_mass_weight(da3.to_dataset(), "lev")
    assert mw is None


def test_fix_sign_makes_max_loading_positive() -> None:
    mode = xr.DataArray(
        np.array([[-3.0, 1.0], [2.0, -0.5]]),
        dims=("mode", "lat"),
        coords={"mode": [1, 2], "lat": [0, 1]},
    )
    pc = xr.DataArray(
        np.ones((2, 4)), dims=("mode", "time"), coords={"mode": [1, 2], "time": np.arange(4)}
    )
    m2, p2 = _fix_sign(mode, pc)
    assert float(m2.sel(mode=1).max()) == 3.0
    assert float(p2.sel(mode=1).isel(time=0)) == -1.0


def test_patterns_from_pc_uses_spatial_matrix_multiplication() -> None:
    from davinci_monet.analysis.eof import _patterns_from_pc

    time = np.arange(4)
    lat = np.array([10.0, 20.0])
    lon = np.array([100.0, 110.0, 120.0])
    values = np.arange(24, dtype=float).reshape(4, 2, 3)
    anom = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    )
    pc = xr.DataArray(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]),
        dims=("time", "mode"),
        coords={"time": time, "mode": [1, 2]},
    )

    out = _patterns_from_pc(anom, pc)

    matrix = values.reshape(4, 6)
    expected = (matrix.T @ pc.values / 4).T.reshape(2, 2, 3)
    assert out.dims == ("mode", "lat", "lon")
    np.testing.assert_allclose(out.values, expected)


def test_effective_n_is_floored_for_highly_autocorrelated_series() -> None:
    lat = np.array([0.0, 10.0])
    lon = np.array([100.0, 110.0])
    series = np.arange(20, dtype=float)
    anom = xr.DataArray(
        series[:, None, None] * np.ones((20, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(20),
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    )

    assert _effective_n(anom, anom["latitude"]) >= 2.0
