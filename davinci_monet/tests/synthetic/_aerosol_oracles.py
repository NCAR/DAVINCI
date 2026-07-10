"""Numerical comparison oracles independent of production FABLE helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from davinci_monet.tests.synthetic._aerosol_contracts import (
    DEFAULT_AEROSOL_SPECIES,
    GRAVITY,
    ShiftedLogResult,
)


def periodic_bilinear_oracle(
    values: np.ndarray,
    source_lat: npt.ArrayLike,
    source_lon: npt.ArrayLike,
    target_lat: npt.ArrayLike,
    target_lon: npt.ArrayLike,
) -> np.ndarray:
    """Bilinearly interpolate trailing ``(lat, lon)`` axes across a periodic seam."""
    data = np.asarray(values, dtype=np.float64)
    src_lat = np.asarray(source_lat, dtype=np.float64)
    src_lon = np.mod(np.asarray(source_lon, dtype=np.float64), 360.0)
    dst_lat = np.asarray(target_lat, dtype=np.float64)
    dst_lon = np.mod(np.asarray(target_lon, dtype=np.float64), 360.0)
    if data.shape[-2:] != (src_lat.size, src_lon.size):
        raise ValueError("values trailing dimensions must match source_lat/source_lon")
    if src_lat.size < 2 or src_lon.size < 2:
        raise ValueError("source grid must contain at least two points per axis")

    lat_order = np.argsort(src_lat)
    lon_order = np.argsort(src_lon)
    src_lat = src_lat[lat_order]
    src_lon = src_lon[lon_order]
    data = data[..., lat_order, :][..., lon_order]
    if np.any(np.diff(src_lat) <= 0.0) or np.any(np.diff(src_lon) <= 0.0):
        raise ValueError("source coordinates must be unique")
    extended_lon = np.concatenate((src_lon, src_lon[:1] + 360.0))
    extended_data = np.concatenate((data, data[..., :1]), axis=-1)
    query_lon = np.mod(dst_lon - src_lon[0], 360.0) + src_lon[0]

    lat_hi = np.clip(np.searchsorted(src_lat, dst_lat, side="right"), 1, src_lat.size - 1)
    lat_lo = lat_hi - 1
    lat_fraction = np.clip(
        (dst_lat - src_lat[lat_lo]) / (src_lat[lat_hi] - src_lat[lat_lo]), 0.0, 1.0
    )
    lon_hi = np.clip(
        np.searchsorted(extended_lon, query_lon, side="right"), 1, extended_lon.size - 1
    )
    lon_lo = lon_hi - 1
    lon_fraction = (query_lon - extended_lon[lon_lo]) / (
        extended_lon[lon_hi] - extended_lon[lon_lo]
    )

    v00 = extended_data[..., lat_lo[:, None], lon_lo[None, :]]
    v01 = extended_data[..., lat_lo[:, None], lon_hi[None, :]]
    v10 = extended_data[..., lat_hi[:, None], lon_lo[None, :]]
    v11 = extended_data[..., lat_hi[:, None], lon_hi[None, :]]
    wx = lon_fraction.reshape((1,) * (data.ndim - 2) + (1, -1))
    wy = lat_fraction.reshape((1,) * (data.ndim - 2) + (-1, 1))
    lower = v00 * (1.0 - wx) + v01 * wx
    upper = v10 * (1.0 - wx) + v11 * wx
    return lower * (1.0 - wy) + upper * wy


def area_weighted_regrid_oracle(
    values: np.ndarray,
    source_lat: npt.ArrayLike,
    source_lon: npt.ArrayLike,
    target_lat: npt.ArrayLike,
    target_lon: npt.ArrayLike,
) -> np.ndarray:
    """Conservatively coarsen trailing regular global axes by spherical cell overlap."""
    data = np.asarray(values, dtype=np.float64)
    src_lat = np.asarray(source_lat, dtype=np.float64)
    src_lon = np.mod(np.asarray(source_lon, dtype=np.float64), 360.0)
    dst_lat = np.asarray(target_lat, dtype=np.float64)
    dst_lon = np.mod(np.asarray(target_lon, dtype=np.float64), 360.0)
    if data.shape[-2:] != (src_lat.size, src_lon.size):
        raise ValueError("values trailing dimensions must match source_lat/source_lon")

    src_lat_order = np.argsort(src_lat)
    src_lon_order = np.argsort(src_lon)
    dst_lat_order = np.argsort(dst_lat)
    dst_lon_order = np.argsort(dst_lon)
    src_lat_sorted = src_lat[src_lat_order]
    src_lon_sorted = src_lon[src_lon_order]
    dst_lat_sorted = dst_lat[dst_lat_order]
    dst_lon_sorted = dst_lon[dst_lon_order]
    data = data[..., src_lat_order, :][..., src_lon_order]
    src_lat_edges = _global_latitude_edges(src_lat_sorted)
    dst_lat_edges = _global_latitude_edges(dst_lat_sorted)
    src_lon_width = _regular_longitude_width(src_lon_sorted)
    dst_lon_width = _regular_longitude_width(dst_lon_sorted)

    lat_overlap = np.zeros((dst_lat.size, src_lat.size), dtype=np.float64)
    for target_index in range(dst_lat.size):
        lower = np.maximum(dst_lat_edges[target_index], src_lat_edges[:-1])
        upper = np.minimum(dst_lat_edges[target_index + 1], src_lat_edges[1:])
        valid = upper > lower
        lat_overlap[target_index, valid] = np.sin(np.deg2rad(upper[valid])) - np.sin(
            np.deg2rad(lower[valid])
        )
    lon_overlap = np.empty((dst_lon.size, src_lon.size), dtype=np.float64)
    for target_index, target_center in enumerate(dst_lon_sorted):
        for source_index, source_center in enumerate(src_lon_sorted):
            lon_overlap[target_index, source_index] = np.deg2rad(
                _periodic_interval_overlap(
                    target_center, dst_lon_width, source_center, src_lon_width
                )
            )
    finite = np.isfinite(data)
    numerator = np.einsum(
        "...ij,ai,bj->...ab", np.where(finite, data, 0.0), lat_overlap, lon_overlap
    )
    denominator = np.einsum(
        "...ij,ai,bj->...ab", finite.astype(np.float64), lat_overlap, lon_overlap
    )
    output = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    return output[..., np.argsort(dst_lat_order), :][..., np.argsort(dst_lon_order)]


def _global_latitude_edges(centers: np.ndarray) -> np.ndarray:
    if centers.size < 2 or np.any(np.diff(centers) <= 0.0):
        raise ValueError("latitude centers must be unique and increasing")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    if not np.allclose(edges[[0, -1]], [-90.0, 90.0], rtol=0.0, atol=1.0e-8):
        raise ValueError("latitude centers must cover the globe from -90 to 90 degree edges")
    edges[[0, -1]] = (-90.0, 90.0)
    return edges


def _regular_longitude_width(centers: np.ndarray) -> float:
    if centers.size < 2:
        raise ValueError("longitude must contain at least two centers")
    spacing = np.diff(np.concatenate((centers, centers[:1] + 360.0)))
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-8):
        raise ValueError("longitude centers must form a regular global grid")
    return float(spacing[0])


def _periodic_interval_overlap(
    first_center: float, first_width: float, second_center: float, second_width: float
) -> float:
    distance = abs((first_center - second_center + 180.0) % 360.0 - 180.0)
    first_half = 0.5 * first_width
    second_half = 0.5 * second_width
    return max(
        0.0,
        min(first_half, distance + second_half) - max(-first_half, distance - second_half),
    )


def local_overpass_oracle(
    hourly: xr.DataArray,
    daily_time: Sequence[np.datetime64] | pd.DatetimeIndex,
    local_hour: float = 13.5,
) -> xr.DataArray:
    """Select each longitude column nearest a fixed local-solar overpass hour."""
    if hourly.dims != ("time", "lat", "lon"):
        raise ValueError("hourly input must have dimensions ('time', 'lat', 'lon')")
    source_time = pd.DatetimeIndex(hourly["time"].values)
    days = pd.DatetimeIndex(daily_time)
    lon = np.asarray(hourly["lon"].values, dtype=np.float64)
    nearest_utc = np.ceil(local_hour - lon / 15.0 - 0.5).astype(int)
    day_shift, utc_hour = np.divmod(nearest_utc, 24)
    output = np.empty((days.size, hourly.sizes["lat"], hourly.sizes["lon"]), dtype=np.float64)
    for lon_index in range(lon.size):
        wanted = days.normalize() + pd.to_timedelta(day_shift[lon_index], unit="D")
        wanted += pd.to_timedelta(utc_hour[lon_index], unit="h")
        indices = source_time.get_indexer(wanted)
        if np.any(indices < 0):
            raise ValueError("hourly input does not include the padded overpass times")
        output[:, :, lon_index] = np.asarray(hourly.values)[indices, :, lon_index]
    return xr.DataArray(
        output,
        dims=("time", "lat", "lon"),
        coords={"time": days.values, "lat": hourly["lat"], "lon": hourly["lon"]},
        name=hourly.name,
        attrs=dict(hourly.attrs),
    )


def log_time_interpolation_oracle(
    daily_ratio: np.ndarray,
    daily_time: Sequence[np.datetime64] | pd.DatetimeIndex,
    target_time: Sequence[np.datetime64] | pd.DatetimeIndex,
) -> np.ndarray:
    """Interpolate a positive field in log time; use identity outside coverage."""
    ratio = np.asarray(daily_ratio, dtype=np.float64)
    if ratio.ndim < 1 or ratio.shape[0] != len(daily_time):
        raise ValueError("daily_ratio first dimension must match daily_time")
    if np.any(ratio <= 0.0):
        raise ValueError("daily_ratio must be positive")
    source = pd.DatetimeIndex(daily_time).asi8
    target = pd.DatetimeIndex(target_time).asi8
    if np.any(np.diff(source) <= 0):
        raise ValueError("daily_time must be strictly increasing")
    high = np.clip(np.searchsorted(source, target, side="right"), 1, source.size - 1)
    low = high - 1
    fraction = (target - source[low]) / (source[high] - source[low])
    shaped = fraction.reshape((-1,) + (1,) * (ratio.ndim - 1))
    result = (1.0 - shaped) * np.log(ratio)[low] + shaped * np.log(ratio)[high]
    result[(target < source[0]) | (target > source[-1])] = 0.0
    return np.exp(result)


def shifted_log_ratio_oracle(
    model_aod: np.ndarray,
    delta_requested: np.ndarray,
    *,
    epsilon: float,
    r_bounds: tuple[float, float],
    aod_floor: float,
    support: np.ndarray | float = 1.0,
) -> ShiftedLogResult:
    """Apply exact shifted-log inversion, physical ratio bounds, and identity gates."""
    model, delta, support_array = np.broadcast_arrays(
        np.asarray(model_aod, dtype=np.float64),
        np.asarray(delta_requested, dtype=np.float64),
        np.asarray(support, dtype=np.float64),
    )
    if np.any(model < 0.0) or epsilon <= 0.0:
        raise ValueError("model_aod must be nonnegative and epsilon must be positive")
    r_min, r_max = r_bounds
    if not 0.0 < r_min <= 1.0 <= r_max:
        raise ValueError("r_bounds must be positive and contain identity")

    lower_delta = np.log((model * r_min + epsilon) / (model + epsilon))
    upper_delta = np.log((model * r_max + epsilon) / (model + epsilon))
    low_aod = (model < aod_floor) | (model <= 0.0)
    unsupported = support_array <= 0.0
    safe_delta = np.clip(delta, lower_delta, upper_delta)
    safe_delta = np.where(low_aod | unsupported, 0.0, safe_delta)

    applied_ratio = np.ones_like(model)
    active = ~(low_aod | unsupported)
    applied_ratio[active] += (
        (model[active] + epsilon) / model[active] * np.expm1(safe_delta[active])
    )
    applied_ratio[active] = np.clip(applied_ratio[active], r_min, r_max)
    applied_aod = model * applied_ratio
    applied_delta = np.log((applied_aod + epsilon) / (model + epsilon))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        requested_aod = model + (model + epsilon) * np.expm1(delta)
        requested_ratio = requested_aod / model
    clip_mask = (delta < lower_delta).astype(np.uint8)
    clip_mask |= (delta > upper_delta).astype(np.uint8) << 1
    clip_mask |= low_aod.astype(np.uint8) << 2
    clip_mask |= unsupported.astype(np.uint8) << 3
    return ShiftedLogResult(
        requested_ratio=requested_ratio,
        applied_ratio=applied_ratio,
        requested_aod=requested_aod,
        applied_aod=applied_aod,
        applied_delta=applied_delta,
        clip_mask=clip_mask,
    )


def extinction_coefficients(rh: np.ndarray, n_species: int) -> np.ndarray:
    base = np.linspace(3200.0, 6800.0, n_species, dtype=np.float64)
    level_factor = np.linspace(0.94, 1.06, rh.shape[1], dtype=np.float64)
    return (
        base[:, None, None, None, None]
        * level_factor[None, None, :, None, None]
        * (1.0 + 0.35 * rh[None, ...])
    )


def optical_aod_oracle(
    mmr: xr.Dataset, species: Sequence[str] = DEFAULT_AEROSOL_SPECIES
) -> xr.DataArray:
    """Evaluate the fixed synthetic optical operator without writer code."""
    required = {"RH", "DELP", *species}
    missing = required - set(mmr.data_vars)
    if missing:
        raise ValueError(f"MMR dataset is missing optical inputs: {sorted(missing)}")
    rh = np.asarray(mmr["RH"].transpose("time", "lev", "lat", "lon"), dtype=np.float64)
    dp = np.asarray(mmr["DELP"].transpose("time", "lev", "lat", "lon"), dtype=np.float64)
    mixing_ratio = np.stack(
        [
            np.asarray(mmr[name].transpose("time", "lev", "lat", "lon"), dtype=np.float64)
            for name in species
        ],
        axis=0,
    )
    aod = np.sum(
        extinction_coefficients(rh, len(species)) * mixing_ratio * dp[None, ...] / GRAVITY,
        axis=(0, 2),
    )
    return xr.DataArray(
        aod,
        dims=("time", "lat", "lon"),
        coords={"time": mmr["time"], "lat": mmr["lat"], "lon": mmr["lon"]},
        name="optical_aod",
        attrs={"units": "1", "operator": "fixed synthetic kappa*q*dp/g"},
    )
