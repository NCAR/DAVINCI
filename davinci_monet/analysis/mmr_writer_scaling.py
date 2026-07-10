"""Validation and bounded native interpolation for MMR scaling inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from davinci_monet.core.coordinates import wrap_longitudes
from davinci_monet.util.regrid import periodic_bilinear


@dataclass(frozen=True)
class ValidatedScaling:
    """Aligned scaling fields with one validated nanosecond time axis."""

    ratio: xr.DataArray
    support: xr.DataArray
    time_ns: NDArray[np.int64]


def validate_scaling(scaling: xr.Dataset) -> ValidatedScaling:
    """Validate one complete scaling collection and return its aligned view."""
    missing = [name for name in ("r", "spatial_support") if name not in scaling]
    if missing:
        raise ValueError(f"scaling input is missing variables: {', '.join(missing)}")
    ratio = scaling["r"]
    support = scaling["spatial_support"]
    expected = ("time", "lat", "lon")
    if set(ratio.dims) != set(expected) or set(support.dims) != set(expected):
        raise ValueError("scaling ratio and support must have time, lat, and lon dimensions")
    ratio = ratio.transpose(*expected)
    support = support.transpose(*expected)
    try:
        ratio, support = xr.align(ratio, support, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("scaling ratio and support coordinates are not aligned") from exc
    times = np.asarray(ratio["time"].values).astype("datetime64[ns]")
    if times.ndim != 1 or times.size == 0 or np.isnat(times).any():
        raise ValueError("scaling time coordinate must be nonempty and finite")
    time_ns = times.astype(np.int64)
    if np.any(np.diff(time_ns) <= 0):
        raise ValueError("scaling time coordinate must be unique and strictly increasing")
    if _array_any(cast(xr.DataArray, ~np.isfinite(ratio))) or _array_any(ratio <= 0.0):
        raise ValueError("scaling ratio must be finite and positive")
    if _array_any(cast(xr.DataArray, ~np.isfinite(support))) or _array_any(
        (support < 0.0) | (support > 1.0)
    ):
        raise ValueError("scaling support must be finite and between zero and one")
    return ValidatedScaling(ratio=ratio, support=support, time_ns=time_ns)


def interpolate_validated_native_ratio(
    validated: ValidatedScaling,
    target_time: xr.DataArray | NDArray[np.datetime64] | Sequence[np.datetime64],
    target_lat: xr.DataArray | NDArray[np.floating] | Sequence[float],
    target_lon: xr.DataArray | NDArray[np.floating] | Sequence[float],
    *,
    outside_coverage: str,
) -> xr.Dataset:
    """Interpolate a validated scaling view using only needed time brackets."""
    target_time_values = np.asarray(target_time).astype("datetime64[ns]")
    latitude_values = np.asarray(target_lat, dtype=np.float64)
    longitude_values = np.asarray(target_lon, dtype=np.float64)
    if target_time_values.ndim != 1 or np.isnat(target_time_values).any():
        raise ValueError("native time coordinate must be one-dimensional and finite")
    if latitude_values.ndim != 1 or longitude_values.ndim != 1:
        raise ValueError("native latitude and longitude coordinates must be one-dimensional")
    target_ns = target_time_values.astype(np.int64)
    inside = (target_ns >= validated.time_ns[0]) & (target_ns <= validated.time_ns[-1])
    if outside_coverage not in {"identity", "skip", "error"}:
        raise ValueError("outside_coverage must be identity, skip, or error")
    if outside_coverage == "error" and not inside.all():
        raise ValueError("native file contains times outside scaling coverage")

    shape = (target_ns.size, latitude_values.size, longitude_values.size)
    log_native = np.zeros(shape, dtype=np.float64)
    support_native = np.zeros(shape, dtype=np.float64)
    if inside.any():
        lower, upper, fractions = _time_brackets(validated.time_ns, target_ns, inside)
        needed = np.unique(np.concatenate((lower[inside], upper[inside]))).astype(np.int64)
        selected_ratio = validated.ratio.isel(time=needed).load()
        selected_support = validated.support.isel(time=needed).load()
        spatial_log = _restore_native_order(
            periodic_bilinear(
                cast(xr.DataArray, np.log(selected_ratio)),
                latitude_values,
                longitude_values,
            ),
            latitude_values,
            longitude_values,
        )
        spatial_support = _restore_native_order(
            periodic_bilinear(selected_support, latitude_values, longitude_values),
            latitude_values,
            longitude_values,
        )
        log_values = np.asarray(spatial_log.values, dtype=np.float64)
        support_values = np.asarray(spatial_support.values, dtype=np.float64)
        local_index = {int(source_index): index for index, source_index in enumerate(needed)}
        for target_index in np.flatnonzero(inside):
            lower_index = local_index[int(lower[target_index])]
            upper_index = local_index[int(upper[target_index])]
            fraction = float(fractions[target_index])
            log_native[target_index] = (1.0 - fraction) * log_values[
                lower_index
            ] + fraction * log_values[upper_index]
            support_native[target_index] = (1.0 - fraction) * support_values[
                lower_index
            ] + fraction * support_values[upper_index]
    log_native[support_native <= 0.0] = 0.0
    ratio_native = np.exp(log_native)
    ratio_native[~inside, :, :] = 1.0
    coordinates = {
        "time": target_time_values,
        "lat": latitude_values,
        "lon": longitude_values,
    }
    return xr.Dataset(
        {
            "ratio": (("time", "lat", "lon"), ratio_native),
            "support": (("time", "lat", "lon"), support_native),
            "inside_coverage": (("time",), inside),
        },
        coords=coordinates,
        attrs={"time_interpolation": "log_linear", "outside_coverage": outside_coverage},
    )


def _time_brackets(
    source_ns: NDArray[np.int64], target_ns: NDArray[np.int64], inside: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    positions = np.searchsorted(source_ns, target_ns, side="left")
    positions = np.clip(positions, 0, source_ns.size - 1)
    exact = source_ns[positions] == target_ns
    upper = positions.astype(np.int64)
    lower = np.where(exact, upper, upper - 1).astype(np.int64)
    lower[~inside] = 0
    upper[~inside] = 0
    denominator = source_ns[upper] - source_ns[lower]
    fractions = np.divide(
        target_ns - source_ns[lower],
        denominator,
        out=np.zeros(target_ns.size, dtype=np.float64),
        where=denominator != 0,
    )
    fractions[~inside] = 0.0
    return lower, upper, fractions


def _restore_native_order(
    data: xr.DataArray,
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
) -> xr.DataArray:
    latitude_inverse = np.argsort(np.argsort(latitude))
    normalized_longitude = np.asarray(wrap_longitudes(longitude), dtype=np.float64)
    longitude_inverse = np.argsort(np.argsort(normalized_longitude))
    restored = data.isel(lat=latitude_inverse, lon=longitude_inverse)
    return restored.assign_coords(lat=latitude, lon=longitude)


def _array_any(condition: xr.DataArray) -> bool:
    value = condition.any()
    if value.chunks is not None:
        value = value.compute()
    return bool(value.item())


__all__ = [
    "ValidatedScaling",
    "interpolate_validated_native_ratio",
    "validate_scaling",
]
