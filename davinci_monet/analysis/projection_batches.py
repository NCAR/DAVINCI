"""Bounded-memory orchestration for daily EOF projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from numpy.typing import NDArray

from davinci_monet.analysis.projection_core import (
    EffectiveCovariance,
    MonthlyBiasFit,
    build_effective_covariance,
    innovation,
    masked_boxcar_smooth,
    mode_coverage,
    solve_one_day,
)
from davinci_monet.analysis.projection_inputs import ProjectionObservation


@dataclass(frozen=True)
class LoadedObservationChunk:
    """One bounded time slice of a validated observation source."""

    values: NDArray[np.float64]
    errors: NDArray[np.float64]
    valid: NDArray[np.bool_]
    factors: NDArray[np.float64]


@dataclass(frozen=True)
class ProjectionArrays:
    """Reduced diagnostics plus lazy grid-sized innovation fields."""

    coefficients: NDArray[np.float64]
    resolution: NDArray[np.float64]
    coverage: NDArray[np.float64]
    posterior_variance: NDArray[np.float64]
    resolution_eigenvalue: NDArray[np.float64]
    posterior_eigenvalue: NDArray[np.float64]
    condition_number: NDArray[np.float64]
    effective_rank: NDArray[np.int64]
    n_obs: NDArray[np.int64]
    innovation_mean: da.Array
    innovation_count: da.Array


def _load_chunk(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    start: int,
    stop: int,
) -> tuple[NDArray[np.float64], tuple[LoadedObservationChunk, ...]]:
    """Load and validate only one contiguous time batch."""
    selection = slice(start, stop)
    model_values = np.asarray(model.isel(time=selection).values, dtype=np.float64)
    loaded: list[LoadedObservationChunk] = []
    for observation in observations:
        chunk = xr.Dataset(
            {
                "values": observation.values.isel(time=selection),
                "errors": observation.errors.isel(time=selection),
                "valid": observation.valid.isel(time=selection),
                "factors": observation.factors.isel(time=selection),
            }
        ).load()
        values = np.asarray(chunk["values"].values, dtype=np.float64)
        errors = np.asarray(chunk["errors"].values, dtype=np.float64)
        valid = np.asarray(chunk["valid"].values, dtype=bool)
        factors = np.asarray(chunk["factors"].values, dtype=np.float64)
        if np.any(valid & (~np.isfinite(errors) | (errors <= 0.0))):
            raise ValueError(
                f"projection observation {observation.name!r} has non-positive valid errors"
            )
        if factors.shape[1] and np.any(valid[:, None] & ~np.isfinite(factors)):
            raise ValueError(
                f"projection observation {observation.name!r} has missing common factors"
            )
        loaded.append(LoadedObservationChunk(values, errors, valid, factors))
    return model_values, tuple(loaded)


def fit_monthly_bias_batched(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    months: NDArray[np.int64],
    fit_mask: NDArray[np.bool_],
    *,
    support_min_fraction: float,
    support_full_fraction: float,
    smoothing_passes: int,
    delta_bounds: tuple[float, float],
    time_chunk_size: int,
) -> MonthlyBiasFit:
    """Fit monthly bias/support from bounded batches without a sensor-time cube."""
    if time_chunk_size < 1:
        raise ValueError("projection time_chunk_size must be positive")
    time_count = model.sizes["time"]
    if months.shape != (time_count,) or fit_mask.shape != (time_count,):
        raise ValueError("projection month and fit masks must match the model time axis")
    if not 0.0 <= support_min_fraction < support_full_fraction <= 1.0:
        raise ValueError("support fractions must satisfy 0 <= min < full <= 1")
    lower, upper = map(float, delta_bounds)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("delta_bounds must be finite and strictly ordered")

    sensor_count_value = len(observations)
    lat_count = model.sizes["lat"]
    lon_count = model.sizes["lon"]
    spatial_shape = (12, lat_count, lon_count)
    precision_sum = np.zeros(spatial_shape, dtype=np.float64)
    weighted_sum = np.zeros(spatial_shape, dtype=np.float64)
    support_count = np.zeros(spatial_shape, dtype=np.int64)
    sensor_count = np.zeros((12, sensor_count_value, lat_count, lon_count), dtype=np.int64)
    support_day_total = np.bincount(
        months[np.asarray(fit_mask, dtype=bool)] - 1, minlength=12
    ).astype(np.int64)
    any_valid = False

    for start in range(0, time_count, time_chunk_size):
        stop = min(start + time_chunk_size, time_count)
        selected = np.flatnonzero(fit_mask[start:stop])
        if selected.size == 0:
            continue
        model_values, chunks = _load_chunk(model, observations, start, stop)
        for local_day in selected:
            month = int(months[start + local_day]) - 1
            observed_union = np.zeros((lat_count, lon_count), dtype=bool)
            for sensor, chunk in enumerate(chunks):
                usable = (
                    chunk.valid[local_day]
                    & np.isfinite(chunk.values[local_day])
                    & np.isfinite(model_values[local_day])
                    & np.isfinite(chunk.errors[local_day])
                    & (chunk.errors[local_day] > 0.0)
                )
                if not np.any(usable):
                    continue
                any_valid = True
                observed_union |= usable
                sensor_count[month, sensor] += usable
                precision = np.zeros((lat_count, lon_count), dtype=np.float64)
                np.divide(
                    1.0,
                    np.square(chunk.errors[local_day]),
                    out=precision,
                    where=usable,
                )
                raw = chunk.values[local_day] - model_values[local_day]
                precision_sum[month] += precision
                weighted_sum[month] += np.where(usable, raw * precision, 0.0)
            support_count[month] += observed_union

    if not any_valid:
        raise ValueError("bias fit window contains no valid observations")

    raw_mean = np.divide(
        weighted_sum,
        precision_sum,
        out=np.zeros_like(weighted_sum),
        where=precision_sum > 0.0,
    )
    standard_error = np.sqrt(
        np.divide(
            1.0,
            precision_sum,
            out=np.zeros_like(precision_sum),
            where=precision_sum > 0.0,
        )
    )
    bias = np.zeros(spatial_shape, dtype=np.float64)
    support = np.zeros(spatial_shape, dtype=np.float64)
    support_fraction = np.zeros(spatial_shape, dtype=np.float64)
    for month in range(12):
        total = support_day_total[month]
        if total == 0:
            continue
        fraction = support_count[month] / float(total)
        support_fraction[month] = fraction
        supported = (fraction >= support_min_fraction) & (precision_sum[month] > 0.0)
        smoothed_bias = masked_boxcar_smooth(raw_mean[month], supported, smoothing_passes)
        bias[month] = np.where(supported, np.clip(smoothed_bias, lower, upper), 0.0)
        raw_support = np.clip(
            (fraction - support_min_fraction) / (support_full_fraction - support_min_fraction),
            0.0,
            1.0,
        )
        smoothed_support = masked_boxcar_smooth(raw_support, supported, smoothing_passes)
        support[month] = np.where(
            fraction >= support_min_fraction,
            np.clip(smoothed_support, 0.0, 1.0),
            0.0,
        )
    return MonthlyBiasFit(
        raw_mean=raw_mean,
        bias=bias,
        bias_applied=support * bias,
        support=support,
        support_fraction=support_fraction,
        support_count=support_count,
        support_day_total=support_day_total,
        sensor_count=sensor_count,
        standard_error=standard_error,
    )


def _innovation_diagnostics_chunk(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    fit: MonthlyBiasFit,
    months: NDArray[np.int64],
    apply_bias: bool,
    sensor_offsets: NDArray[np.float64],
    start: int,
    stop: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    model_values, chunks = _load_chunk(model, observations, start, stop)
    shape = model_values.shape
    means = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    for local_day, absolute_day in enumerate(range(start, stop)):
        month = int(months[absolute_day]) - 1
        bias = fit.bias[month] if apply_bias else np.zeros(shape[1:], dtype=np.float64)
        support = fit.support[month]
        precision_sum = np.zeros(shape[1:], dtype=np.float64)
        weighted = np.zeros(shape[1:], dtype=np.float64)
        for sensor, chunk in enumerate(chunks):
            usable = chunk.valid[local_day] & np.isfinite(model_values[local_day]) & (support > 0.0)
            daily = innovation(
                chunk.values[local_day],
                model_values[local_day],
                bias + sensor_offsets[sensor],
            )
            precision = np.zeros(shape[1:], dtype=np.float64)
            np.divide(
                1.0,
                np.square(chunk.errors[local_day]),
                out=precision,
                where=usable,
            )
            precision_sum += precision
            weighted += np.where(usable, daily * precision, 0.0)
            counts[local_day] += usable
        means[local_day] = np.divide(
            weighted,
            precision_sum,
            out=np.zeros_like(weighted),
            where=precision_sum > 0.0,
        )
    return means, counts


def _lazy_innovation_diagnostics(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    fit: MonthlyBiasFit,
    months: NDArray[np.int64],
    apply_bias: bool,
    sensor_offsets: NDArray[np.float64],
    time_chunk_size: int,
) -> tuple[da.Array, da.Array]:
    mean_chunks: list[da.Array] = []
    count_chunks: list[da.Array] = []
    lat_count = model.sizes["lat"]
    lon_count = model.sizes["lon"]
    for start in range(0, model.sizes["time"], time_chunk_size):
        stop = min(start + time_chunk_size, model.sizes["time"])
        delayed_mean, delayed_count = delayed(_innovation_diagnostics_chunk, nout=2)(
            model,
            observations,
            fit,
            months,
            apply_bias,
            sensor_offsets,
            start,
            stop,
        )
        shape = (stop - start, lat_count, lon_count)
        mean_chunks.append(da.from_delayed(delayed_mean, shape=shape, dtype=np.float64))
        count_chunks.append(da.from_delayed(delayed_count, shape=shape, dtype=np.int64))
    return da.concatenate(mean_chunks, axis=0), da.concatenate(count_chunks, axis=0)


def solve_projection_batched(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    patterns: NDArray[np.float64],
    fit: MonthlyBiasFit,
    months: NDArray[np.int64],
    *,
    apply_bias: bool,
    ridge: float,
    time_chunk_size: int,
    sensor_offsets: NDArray[np.float64] | None = None,
) -> ProjectionArrays:
    """Solve the complete daily axis while holding one time batch of grids."""
    time_count = model.sizes["time"]
    mode_count, lat_count, lon_count = patterns.shape
    shape_tm = (time_count, mode_count)
    coefficients = np.zeros(shape_tm, dtype=np.float64)
    resolution = np.zeros(shape_tm, dtype=np.float64)
    coverage = np.zeros(shape_tm, dtype=np.float64)
    posterior_variance = np.zeros(shape_tm, dtype=np.float64)
    resolution_eigenvalue = np.zeros(shape_tm, dtype=np.float64)
    posterior_eigenvalue = np.zeros(shape_tm, dtype=np.float64)
    condition_number = np.zeros(time_count, dtype=np.float64)
    effective_rank = np.zeros(time_count, dtype=np.int64)
    n_obs = np.zeros((time_count, len(observations)), dtype=np.int64)
    offsets = (
        np.zeros(len(observations), dtype=np.float64)
        if sensor_offsets is None
        else np.asarray(sensor_offsets, dtype=np.float64)
    )
    if offsets.shape != (len(observations),) or np.any(~np.isfinite(offsets)):
        raise ValueError("projection sensor offsets must be one finite value per sensor")
    latitude_rows = np.broadcast_to(
        np.asarray(model["lat"].values)[:, None], (lat_count, lon_count)
    ).reshape(-1)
    flat_patterns = patterns.reshape(mode_count, -1)
    common_mode_count = len(observations[0].factor_names)

    for start in range(0, time_count, time_chunk_size):
        stop = min(start + time_chunk_size, time_count)
        model_values, chunks = _load_chunk(model, observations, start, stop)
        for local_day, day in enumerate(range(start, stop)):
            month = int(months[day]) - 1
            bias = fit.bias[month] if apply_bias else np.zeros((lat_count, lon_count))
            support = fit.support[month]
            row_design: list[NDArray[np.float64]] = []
            row_innovation: list[NDArray[np.float64]] = []
            row_error: list[NDArray[np.float64]] = []
            row_latitude: list[NDArray[np.float64]] = []
            row_factors: list[NDArray[np.float64]] = []
            observed_union = np.zeros((lat_count, lon_count), dtype=bool)
            for sensor, chunk in enumerate(chunks):
                usable = (
                    chunk.valid[local_day] & np.isfinite(model_values[local_day]) & (support > 0.0)
                )
                flat = np.flatnonzero(usable.reshape(-1))
                n_obs[day, sensor] = flat.size
                if flat.size == 0:
                    continue
                daily = innovation(
                    chunk.values[local_day],
                    model_values[local_day],
                    bias + offsets[sensor],
                )
                row_design.append(flat_patterns[:, flat].T)
                row_innovation.append(daily.reshape(-1)[flat])
                row_error.append(chunk.errors[local_day].reshape(-1)[flat])
                row_latitude.append(latitude_rows[flat])
                factor_rows = chunk.factors[local_day].reshape(
                    common_mode_count, lat_count * lon_count
                )
                row_factors.append(factor_rows[:, flat].T)
                observed_union |= usable
            coverage[day] = mode_coverage(patterns, observed_union, model["lat"].values)
            if row_design:
                covariance = build_effective_covariance(
                    np.concatenate(row_error),
                    np.concatenate(row_latitude),
                    np.concatenate(row_factors),
                )
                solution = solve_one_day(
                    np.concatenate(row_design),
                    np.concatenate(row_innovation),
                    covariance,
                    ridge,
                )
            else:
                covariance = EffectiveCovariance(
                    np.empty(0, dtype=np.float64),
                    np.empty((0, common_mode_count), dtype=np.float64),
                )
                solution = solve_one_day(np.empty((0, mode_count)), np.empty(0), covariance, ridge)
            coefficients[day] = solution.coefficients
            resolution[day] = solution.resolution
            posterior_variance[day] = solution.posterior_variance
            resolution_eigenvalue[day] = solution.resolution_eigenvalues
            posterior_eigenvalue[day] = solution.posterior_eigenvalues
            condition_number[day] = solution.condition_number
            effective_rank[day] = solution.effective_rank

    innovation_mean, innovation_count = _lazy_innovation_diagnostics(
        model,
        observations,
        fit,
        months,
        apply_bias,
        offsets,
        time_chunk_size,
    )
    return ProjectionArrays(
        coefficients=coefficients,
        resolution=resolution,
        coverage=coverage,
        posterior_variance=posterior_variance,
        resolution_eigenvalue=resolution_eigenvalue,
        posterior_eigenvalue=posterior_eigenvalue,
        condition_number=condition_number,
        effective_rank=effective_rank,
        n_obs=n_obs,
        innovation_mean=innovation_mean,
        innovation_count=innovation_count,
    )


__all__ = [
    "ProjectionArrays",
    "fit_monthly_bias_batched",
    "solve_projection_batched",
]
