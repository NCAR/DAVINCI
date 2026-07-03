"""Reduce a source variable to a 1-D time series and prepare it for the CWT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from davinci_monet.config.schema import WaveletSpec

_LAT_NAMES = ("latitude", "lat", "LAT", "Latitude")
_LON_NAMES = ("longitude", "lon", "LON", "Longitude")


def _coord(da: xr.DataArray, names: tuple[str, ...], kind: str) -> xr.DataArray:
    for name in names:
        if name in da.coords:
            return da.coords[name]
    raise ValueError(f"wavelet reduction requires a {kind} coordinate (one of {names})")


def select_series(data: xr.Dataset, spec: "WaveletSpec") -> xr.DataArray:
    """Resolve spec.variable (+ mode + reduce) to a 1-D (time,) series."""
    from davinci_monet.config.schema import PointReduce

    da = data[spec.variable]
    if "mode" in da.dims:
        if spec.mode is None:
            raise ValueError(f"wavelet on '{spec.variable}' with a 'mode' dim requires mode: N")
        da = da.sel(mode=spec.mode)

    spatial = [d for d in da.dims if d != "time"]
    if not spatial:
        if isinstance(spec.reduce, PointReduce):
            raise ValueError("reduce: point is invalid for an already-1-D series")
        return da

    reduce = spec.reduce
    if reduce is None or reduce == "area_mean":
        lat = _coord(da, _LAT_NAMES, "latitude")
        w = np.cos(np.deg2rad(lat)).clip(min=0.0)
        return da.weighted(w).mean(dim=spatial)
    if isinstance(reduce, PointReduce):
        lat = _coord(da, _LAT_NAMES, "latitude")
        lon = _coord(da, _LON_NAMES, "longitude")
        i = int(np.abs(np.asarray(lat.values) - reduce.point[0]).argmin())
        j = int(np.abs(np.asarray(lon.values) - reduce.point[1]).argmin())
        da = da.isel({lat.dims[0]: i, lon.dims[0]: j})
        rem = [d for d in da.dims if d != "time"]
        return da.mean(rem) if rem else da
    raise ValueError(f"unknown reduce: {reduce!r}")


def _step_and_unit(time_values: np.ndarray) -> tuple[float, str, np.ndarray]:
    arr = np.asarray(time_values)
    if np.issubdtype(arr.dtype, np.datetime64):
        deltas = np.diff(arr).astype("timedelta64[s]").astype(float)
        med = float(np.median(deltas)) if deltas.size else 86400.0
        if med >= 86400.0:
            return med / 86400.0, "days", deltas
        return med / 3600.0, "hours", deltas
    deltas = np.diff(arr.astype(float))
    return (float(np.median(deltas)) if deltas.size else 1.0), "steps", deltas


def regularize(series: xr.DataArray) -> tuple[xr.DataArray, float, str, float]:
    """Return (regular series, dt, period-unit, fraction of synthesized samples)."""
    dt, unit, deltas = _step_and_unit(series["time"].values)
    if deltas.size == 0:
        return series, dt, unit, 0.0
    med = float(np.median(deltas))
    irregular = bool(np.any(np.abs(deltas - med) > 0.05 * med))
    if not irregular or unit == "steps":
        return series, dt, unit, 0.0
    n_before = int(series.sizes["time"])
    freq = pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=med)).freqstr
    regular = series.resample(time=freq).mean()
    n_after = int(regular.sizes["time"])
    frac = max(0.0, (n_after - n_before) / max(n_after, 1))
    return regular, dt, unit, frac


def detrend_series(y: np.ndarray) -> np.ndarray:
    """Remove a linear trend and center the series at zero."""
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size)
    coef = np.polyfit(x, y, 1)
    return y - np.polyval(coef, x)


def ar1_alpha(y: np.ndarray) -> float:
    """Lag-1 autocorrelation (red-noise parameter) of the (detrended) series."""
    try:
        import pycwt

        return float(pycwt.ar1(np.asarray(y, dtype=float))[0])
    except Exception:  # noqa: BLE001 - robust fallback
        y = np.asarray(y, dtype=float)
        if y.size < 3:
            return 0.0
        return float(np.clip(np.corrcoef(y[:-1], y[1:])[0, 1], -0.99, 0.99))


def normalize_series(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (unit-variance series, std, mean)."""
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y))
    std = float(np.std(y))
    std = std if std > 0 else 1.0
    return (y - mean) / std, std, mean
