"""Reduce a source variable to a 1-D time series and prepare it for the CWT."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import cftime
import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from davinci_monet.config.schema import WaveletSpec

_LAT_NAMES = ("latitude", "lat", "LAT", "Latitude")
_LON_NAMES = ("longitude", "lon", "LON", "Longitude")
logger = logging.getLogger(__name__)


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
        try:
            lon = _coord(da, _LON_NAMES, "longitude")
        except ValueError:
            lon = None
        horizontal_dims = set(lat.dims) | (set(lon.dims) if lon is not None else set())
        unweighted_dims = [dim for dim in spatial if dim not in horizontal_dims]
        if unweighted_dims:
            logger.warning(
                "wavelet area_mean reduction is averaging non-horizontal dimensions "
                "without weights: %s",
                ", ".join(map(str, unweighted_dims)),
            )
        w = xr.DataArray(np.cos(np.deg2rad(lat)).clip(min=0.0), dims=lat.dims, coords=lat.coords)
        return da.weighted(w).mean(dim=spatial)
    if isinstance(reduce, PointReduce):
        lat = _coord(da, _LAT_NAMES, "latitude")
        lon = _coord(da, _LON_NAMES, "longitude")
        lat_grid, lon_grid = xr.broadcast(lat, lon)
        lat_values = np.asarray(lat_grid.values, dtype=float)
        lon_delta = _wrapped_longitude_delta(
            np.asarray(lon_grid.values, dtype=float), reduce.point[1]
        )
        target_latitude = np.deg2rad(reduce.point[0])
        latitude = np.deg2rad(lat_values)
        latitude_delta = latitude - target_latitude
        longitude_delta = np.deg2rad(lon_delta)
        haversine = (
            np.sin(latitude_delta / 2.0) ** 2
            + np.cos(latitude) * np.cos(target_latitude) * np.sin(longitude_delta / 2.0) ** 2
        )
        if not np.isfinite(haversine).any():
            raise ValueError("wavelet point reduction found no finite latitude/longitude pair")
        flat = int(np.nanargmin(haversine))
        idx = np.unravel_index(flat, haversine.shape)
        da = da.isel(dict(zip(lat_grid.dims, (int(k) for k in idx))))
        rem = [d for d in da.dims if d != "time"]
        if rem:
            logger.warning(
                "wavelet point reduction is averaging remaining dimensions without weights: %s",
                ", ".join(map(str, rem)),
            )
        return da.mean(rem) if rem else da
    raise ValueError(f"unknown reduce: {reduce!r}")


def _wrapped_longitude_delta(longitudes: np.ndarray, target: float) -> np.ndarray:
    """Return signed shortest-path longitude differences in degrees."""
    return (np.asarray(longitudes, dtype=float) - float(target) + 180.0) % 360.0 - 180.0


def _is_object_datetime(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return isinstance(values.flat[0], (datetime, cftime.datetime))


def _step_and_unit(time_values: np.ndarray) -> tuple[float, str, np.ndarray]:
    arr = np.asarray(time_values)
    if np.issubdtype(arr.dtype, np.datetime64):
        if np.isnat(arr).any():
            raise ValueError("wavelet time coordinate must not contain NaT values")
        deltas = np.asarray(np.diff(arr) / np.timedelta64(1, "s"), dtype=float)
        med = float(np.median(deltas)) if deltas.size else 86400.0
        _validate_time_deltas(deltas)
        if med >= 86400.0:
            return med / 86400.0, "days", deltas
        return med / 3600.0, "hours", deltas
    if _is_object_datetime(arr):
        deltas = np.asarray(
            [(right - left).total_seconds() for left, right in zip(arr[:-1], arr[1:])],
            dtype=float,
        )
        med = float(np.median(deltas)) if deltas.size else 86400.0
        _validate_time_deltas(deltas)
        if med >= 86400.0:
            return med / 86400.0, "days", deltas
        return med / 3600.0, "hours", deltas
    deltas = np.diff(arr.astype(float))
    _validate_time_deltas(deltas)
    return (float(np.median(deltas)) if deltas.size else 1.0), "steps", deltas


def _validate_time_deltas(deltas: np.ndarray) -> None:
    if deltas.size and (not np.isfinite(deltas).all() or np.any(deltas <= 0.0)):
        raise ValueError("wavelet time coordinate must be finite and strictly increasing")


def _regular_target(time_values: np.ndarray, step: float) -> np.ndarray:
    arr = np.asarray(time_values)
    if np.issubdtype(arr.dtype, np.datetime64):
        start = pd.Timestamp(arr[0])
        end = pd.Timestamp(arr[-1])
        return np.asarray(pd.date_range(start, end, freq=pd.Timedelta(seconds=step)).values)
    if _is_object_datetime(arr):
        duration = float((arr[-1] - arr[0]).total_seconds())
        count = int(np.floor(duration / step + 1.0e-12)) + 1
        interval = timedelta(seconds=step)
        return np.asarray([arr[0] + index * interval for index in range(count)], dtype=object)
    numeric = arr.astype(float)
    count = int(np.floor((numeric[-1] - numeric[0]) / step + 1.0e-12)) + 1
    return numeric[0] + np.arange(count, dtype=float) * step


def _relative_time_values(time_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(time_values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.asarray((arr - arr[0]) / np.timedelta64(1, "s"), dtype=float)
    if _is_object_datetime(arr):
        return np.asarray([(value - arr[0]).total_seconds() for value in arr], dtype=float)
    numeric = arr.astype(float)
    return numeric - numeric[0]


def _matching_timestamp_count(
    original_times: np.ndarray, target_times: np.ndarray, step: float
) -> int:
    original = _relative_time_values(original_times)
    target = _relative_time_values(target_times)
    if original.size == 0 or target.size == 0:
        return 0
    positions = np.searchsorted(original, target)
    right = np.clip(positions, 0, original.size - 1)
    left = np.clip(positions - 1, 0, original.size - 1)
    distance = np.minimum(np.abs(target - original[left]), np.abs(target - original[right]))
    tolerance = max(abs(step) * 1.0e-9, 1.0e-12)
    return int(np.count_nonzero(distance <= tolerance))


def _raise_for_nonfinite(series: xr.DataArray, variable: str | None) -> None:
    values = np.asarray(series.values)
    try:
        finite = np.isfinite(values)
    except TypeError as exc:
        label = variable or series.name or "series"
        raise ValueError(f"wavelet input '{label}' must contain numeric values") from exc
    invalid = np.flatnonzero(~finite.reshape(-1))
    if invalid.size == 0:
        return
    label = variable or series.name or "series"
    time_index = int(invalid[0])
    first_time = np.asarray(series["time"].values).reshape(-1)[time_index]
    raise ValueError(
        f"wavelet input '{label}' has {invalid.size} non-finite samples "
        f"(first at {first_time}); fill or subset the source"
    )


def regularize(
    series: xr.DataArray, *, variable: str | None = None
) -> tuple[xr.DataArray, float, str, float]:
    """Return (regular series, dt, period-unit, fraction of synthesized samples)."""
    _raise_for_nonfinite(series, variable)
    dt, unit, deltas = _step_and_unit(series["time"].values)
    if deltas.size == 0:
        return series, dt, unit, 0.0
    med = float(np.median(deltas))
    irregular = bool(np.any(np.abs(deltas - med) > 0.05 * med))
    if not irregular:
        return series, dt, unit, 0.0
    original_times = np.asarray(series["time"].values)
    target = _regular_target(original_times, med)
    regular = series.interp(time=target)
    _raise_for_nonfinite(regular, variable)
    _, _, regular_deltas = _step_and_unit(regular["time"].values)
    regular_med = float(np.median(regular_deltas)) if regular_deltas.size else med
    if regular_deltas.size and np.any(np.abs(regular_deltas - regular_med) > 0.05 * regular_med):
        raise ValueError("wavelet time-axis regularization did not produce a uniform axis")
    n_after = int(regular.sizes["time"])
    matching = _matching_timestamp_count(original_times, target, med)
    frac = 1.0 - matching / max(n_after, 1)
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
