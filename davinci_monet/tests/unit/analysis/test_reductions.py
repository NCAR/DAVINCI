"""Series selection/reduction + preprocessing helpers for wavelet input."""

from __future__ import annotations

from typing import Any

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


def test_select_area_mean_reduces_to_1d() -> None:
    spec = WaveletSpec(type="wavelet", source="cam", variable="O3")
    s = select_series(_grid(), spec)
    assert s.dims == ("time",)


def test_select_point() -> None:
    spec = WaveletSpec(
        type="wavelet", source="cam", variable="O3", reduce=PointReduce(point=(0.0, 3.0))
    )
    s = select_series(_grid(), spec)
    assert s.dims == ("time",)


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


def test_regularize_irregular_datetime_resamples_with_frequency_string(monkeypatch) -> None:
    times = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    s = xr.DataArray(np.arange(4, dtype=float), dims=("time",), coords={"time": times})
    seen: dict[str, object] = {}
    original_resample: Any = xr.DataArray.resample

    def wrapped_resample(self: xr.DataArray, *args: Any, **kwargs: Any) -> Any:
        seen["time"] = kwargs.get("time")
        return original_resample(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "resample", wrapped_resample)

    reg, dt, unit, frac = regularize(s)

    assert isinstance(seen["time"], str)
    assert reg.dims == ("time",)
    assert dt == pytest.approx(31.0)
    assert unit == "days"
    assert frac >= 0.0


def test_detrend_and_normalize() -> None:
    y = np.arange(50, dtype=float) + 5.0
    d = detrend_series(y)
    assert abs(float(np.mean(d))) < 1e-9
    n, std, mean = normalize_series(d)
    assert float(np.std(n)) == pytest.approx(1.0, abs=1e-6)


def test_ar1_alpha_on_white_noise_is_small() -> None:
    y = np.random.default_rng(1).normal(size=500)
    assert abs(ar1_alpha(y)) < 0.2
