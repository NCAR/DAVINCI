"""Series selection/reduction + preprocessing helpers for wavelet input."""

from __future__ import annotations

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis.reductions import (
    ar1_alpha,
    detrend_series,
    normalize_series,
    regularize,
    select_series,
)
from davinci_monet.config.schema import PointReduce, WaveletSpec


def _grid(nt=20, nlat=3, nlon=4) -> xr.Dataset:
    lat = np.linspace(-5, 5, nlat)
    lon = np.linspace(0, 9, nlon)
    times = pd.date_range("2024-01-01", periods=nt, freq="D")
    data = np.random.default_rng(0).normal(size=(nt, nlat, nlon))
    return xr.Dataset(
        {"O3": (("time", "lat", "lon"), data, {"units": "ppb"})},
        coords={
            "time": times,
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    )


def _pc(nt=20) -> xr.Dataset:
    times = pd.date_range("2024-01-01", periods=nt, freq="D")
    pc = np.stack([np.arange(nt, dtype=float), np.full(nt, 9.0)], axis=1)
    return xr.Dataset({"pc": (("time", "mode"), pc)}, coords={"time": times, "mode": [1, 2]})


def _curvilinear_grid(nt=5) -> xr.Dataset:
    yy, xx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    lat2d = yy * 10.0 + xx
    lon2d = xx * 10.0 + yy
    times = pd.date_range("2024-01-01", periods=nt, freq="D")
    data = 100.0 * (yy * 4 + xx)[None, :, :] + np.arange(nt, dtype=float)[:, None, None]
    return xr.Dataset(
        {"O3": (("time", "y", "x"), data, {"units": "ppb"})},
        coords={
            "time": times,
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
        },
    )


def test_select_area_mean_reduces_to_1d() -> None:
    spec = WaveletSpec(type="wavelet", source="cam", variable="O3")
    s = select_series(_grid(), spec)
    assert s.dims == ("time",)


def test_select_area_mean_warns_when_averaging_nonhorizontal_dims(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ds = _grid(nt=5)
    expanded = xr.concat(
        [ds["O3"], ds["O3"] + 10.0],
        dim=xr.IndexVariable("lev", [100.0, 1000.0]),
    ).transpose("time", "lev", "lat", "lon")
    ds["O3"] = expanded
    spec = WaveletSpec(type="wavelet", source="cam", variable="O3")

    with caplog.at_level("WARNING", logger="davinci_monet.analysis.reductions"):
        selected = select_series(ds, spec)

    latitude = ds["latitude"]
    weights = xr.DataArray(
        np.cos(np.deg2rad(latitude.values)).clip(min=0.0),
        dims=latitude.dims,
        coords=latitude.coords,
    )
    expected = ds["O3"].weighted(weights).mean(("lev", "lat", "lon"))
    xr.testing.assert_identical(selected, expected)
    assert "non-horizontal dimensions without weights: lev" in caplog.text


def test_select_point() -> None:
    spec = WaveletSpec(
        type="wavelet", source="cam", variable="O3", reduce=PointReduce(point=(0.0, 3.0))
    )
    s = select_series(_grid(), spec)
    assert s.dims == ("time",)


def test_select_point_wraps_longitude_on_rectilinear_grid() -> None:
    longitudes = np.array([0.0, 90.0, 181.0, 270.0])
    ds = _grid().assign_coords(
        lon=("lon", longitudes),
        longitude=("lon", longitudes),
    )
    spec = WaveletSpec(
        type="wavelet",
        source="cam",
        variable="O3",
        reduce=PointReduce(point=(0.0, -179.0)),
    )

    selected = select_series(ds, spec)

    xr.testing.assert_identical(selected, ds["O3"].isel(lat=1, lon=2))


def test_select_point_uses_spherical_distance_at_high_latitude() -> None:
    times = pd.date_range("2024-01-01", periods=3, freq="D")
    latitudes = np.array([65.0, 89.0])
    longitudes = np.array([0.0])
    values = np.arange(6, dtype=float).reshape(3, 2, 1)
    ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), values)},
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes,
            "latitude": ("lat", latitudes),
            "longitude": ("lon", longitudes),
        },
    )
    spec = WaveletSpec(
        type="wavelet",
        source="cam",
        variable="O3",
        reduce=PointReduce(point=(70.0, 60.0)),
    )

    selected = select_series(ds, spec)

    xr.testing.assert_identical(selected, ds["O3"].isel(lat=1, lon=0))


def test_select_point_curvilinear_uses_combined_nearest() -> None:
    ds = _curvilinear_grid()
    spec = WaveletSpec(
        type="wavelet",
        source="cam",
        variable="O3",
        reduce=PointReduce(point=(1.2, 10.3)),
    )
    s = select_series(ds, spec)
    assert s.dims == ("time",)
    expected = ds["O3"].isel(y=0, x=1)
    assert np.allclose(s.values, expected.values)


def test_select_point_wraps_longitude_on_curvilinear_grid() -> None:
    ds = _curvilinear_grid()
    longitudes = np.zeros((4, 4), dtype=float)
    longitudes[0, 1] = 181.0
    ds = ds.assign_coords(longitude=(("y", "x"), longitudes))
    spec = WaveletSpec(
        type="wavelet",
        source="cam",
        variable="O3",
        reduce=PointReduce(point=(1.0, -179.0)),
    )

    selected = select_series(ds, spec)

    xr.testing.assert_identical(selected, ds["O3"].isel(y=0, x=1))


def test_select_point_warns_when_averaging_remaining_dims(caplog: pytest.LogCaptureFixture) -> None:
    ds = _grid(nt=5)
    expanded = xr.concat(
        [ds["O3"], ds["O3"] + 10.0],
        dim=xr.IndexVariable("lev", [100.0, 1000.0]),
    ).transpose("time", "lev", "lat", "lon")
    ds["O3"] = expanded
    spec = WaveletSpec(
        type="wavelet",
        source="cam",
        variable="O3",
        reduce=PointReduce(point=(0.0, 3.0)),
    )

    with caplog.at_level("WARNING", logger="davinci_monet.analysis.reductions"):
        selected = select_series(ds, spec)

    expected = ds["O3"].isel(lat=1, lon=1).mean("lev")
    xr.testing.assert_identical(selected, expected)
    assert "remaining dimensions without weights: lev" in caplog.text


def test_select_pc_mode_is_already_1d() -> None:
    spec = WaveletSpec(type="wavelet", source="eof", variable="pc", mode=1)
    s = select_series(_pc(), spec)
    assert s.dims == ("time",)
    assert list(s.values[:3]) == [0.0, 1.0, 2.0]


def test_pc_without_mode_errors() -> None:
    spec = WaveletSpec(type="wavelet", source="eof", variable="pc")
    with pytest.raises(ValueError, match="requires mode"):
        select_series(_pc(), spec)


def test_point_reduce_on_1d_series_errors() -> None:
    spec = WaveletSpec(
        type="wavelet", source="eof", variable="pc", mode=1, reduce=PointReduce(point=(0.0, 0.0))
    )
    with pytest.raises(ValueError, match="point.*1-D"):
        select_series(_pc(), spec)


def test_regularize_regular_series() -> None:
    s = select_series(_grid(), WaveletSpec(type="wavelet", source="c", variable="O3"))
    reg, dt, unit, frac = regularize(s)
    assert dt == pytest.approx(1.0)
    assert unit == "days"
    assert frac == 0.0


def test_regularize_irregular_monthly_interpolates_to_uniform_axis() -> None:
    times = pd.date_range("2018-01-01", periods=48, freq="MS")
    s = xr.DataArray(np.arange(48, dtype=float), dims=("time",), coords={"time": times})

    reg, dt, unit, frac = regularize(s)

    assert reg.dims == ("time",)
    assert dt == pytest.approx(31.0)
    assert unit == "days"
    assert frac > 0.9
    assert not np.isnan(reg.values).any()
    steps = np.diff(reg["time"].values).astype("timedelta64[s]").astype(float)
    assert np.allclose(steps, steps[0])


def test_regularize_noleap_monthly_axis() -> None:
    times = xr.date_range(
        "2018-01-01",
        periods=48,
        freq="MS",
        calendar="noleap",
        use_cftime=True,
    )
    s = xr.DataArray(np.arange(48, dtype=float), dims=("time",), coords={"time": times})

    reg, dt, unit, frac = regularize(s)

    assert isinstance(reg["time"].values[0], cftime.DatetimeNoLeap)
    assert dt == pytest.approx(31.0)
    assert unit == "days"
    assert frac > 0.9
    steps = np.asarray(
        [
            (right - left).total_seconds()
            for left, right in zip(reg["time"].values[:-1], reg["time"].values[1:])
        ]
    )
    assert np.allclose(steps, steps[0])


def test_regularize_irregular_axis_with_nan_raises_actionable_error() -> None:
    times = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-07", "2024-01-08"])
    s = xr.DataArray(
        [0.0, 1.0, np.nan, 6.0, 7.0],
        dims=("time",),
        coords={"time": times},
        name="O3",
    )

    with pytest.raises(
        ValueError,
        match=r"wavelet input 'O3' has 1 non-finite samples .*2024-01-03.*fill or subset",
    ):
        regularize(s, variable="O3")


def test_regularize_gappy_axis_is_uniform_and_nan_free() -> None:
    first = pd.date_range("2024-01-01", periods=10, freq="D")
    second = pd.date_range("2024-01-21", periods=10, freq="D")
    times = first.append(second)
    s = xr.DataArray(np.arange(20, dtype=float), dims=("time",), coords={"time": times})

    reg, dt, unit, frac = regularize(s)

    assert dt == pytest.approx(1.0)
    assert unit == "days"
    assert not np.isnan(reg.values).any()
    steps = np.diff(reg["time"].values).astype("timedelta64[s]").astype(float)
    assert np.allclose(steps, steps[0])
    assert reg.sizes["time"] > s.sizes["time"]
    assert frac > 0.0


def test_detrend_and_normalize() -> None:
    y = np.arange(50, dtype=float) + 5.0
    d = detrend_series(y)
    assert abs(float(np.mean(d))) < 1e-9
    n, std, mean = normalize_series(d)
    assert float(np.std(n)) == pytest.approx(1.0, abs=1e-6)


def test_ar1_alpha_on_white_noise_is_small() -> None:
    y = np.random.default_rng(1).normal(size=500)
    assert abs(ar1_alpha(y)) < 0.2
