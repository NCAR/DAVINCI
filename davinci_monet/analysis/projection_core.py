"""Pure numerical kernels for FABLE innovation projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MonthlyBiasFit:
    """Monthly bias and support fields fitted without validation/test leakage."""

    raw_mean: FloatArray
    bias: FloatArray
    bias_applied: FloatArray
    support: FloatArray
    support_fraction: FloatArray
    support_count: NDArray[np.int64]
    support_day_total: NDArray[np.int64]
    sensor_count: NDArray[np.int64]
    standard_error: FloatArray


@dataclass(frozen=True)
class EffectiveCovariance:
    """Diagonal-plus-low-rank representation of an effective covariance."""

    diagonal: FloatArray
    factors: FloatArray


@dataclass(frozen=True)
class ProjectionSolution:
    """One-day ridge solution and its observability diagnostics."""

    coefficients: FloatArray
    resolution: FloatArray
    posterior_variance: FloatArray
    resolution_eigenvalues: FloatArray
    posterior_eigenvalues: FloatArray
    condition_number: float
    effective_rank: int
    information: FloatArray


def masked_boxcar_smooth(values: FloatArray, valid: BoolArray, passes: int) -> FloatArray:
    """Apply mask-aware 3x3 smoothing with cyclic lon and clipped lat."""
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if data.shape != mask.shape or data.ndim < 2:
        raise ValueError("smoothing values and mask must share at least two dimensions")
    if passes < 0:
        raise ValueError("smoothing pass count must be non-negative")

    current = np.where(mask & np.isfinite(data), data, 0.0)
    prefix_pad = [(0, 0)] * (current.ndim - 2)
    for _ in range(passes):
        lon_data = np.concatenate((current[..., -1:], current, current[..., :1]), axis=-1)
        lon_mask = np.concatenate((mask[..., -1:], mask, mask[..., :1]), axis=-1)
        padded_data = np.pad(
            lon_data,
            [*prefix_pad, (1, 1), (0, 0)],
            mode="constant",
            constant_values=0.0,
        )
        padded_mask = np.pad(
            lon_mask,
            [*prefix_pad, (1, 1), (0, 0)],
            mode="constant",
            constant_values=False,
        )
        total = np.zeros_like(current)
        count = np.zeros_like(current)
        nlat, nlon = current.shape[-2:]
        for lat_offset in range(3):
            for lon_offset in range(3):
                neighbor = padded_data[
                    ..., lat_offset : lat_offset + nlat, lon_offset : lon_offset + nlon
                ]
                neighbor_valid = padded_mask[
                    ..., lat_offset : lat_offset + nlat, lon_offset : lon_offset + nlon
                ]
                total += np.where(neighbor_valid, neighbor, 0.0)
                count += neighbor_valid
        current = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
        current = np.where(mask, current, 0.0)
    return current


def fit_monthly_bias(
    innovations: FloatArray,
    error_std: FloatArray,
    valid: BoolArray,
    months: NDArray[np.integer],
    fit_mask: BoolArray,
    *,
    support_min_fraction: float,
    support_full_fraction: float,
    smoothing_passes: int,
    delta_bounds: tuple[float, float],
) -> MonthlyBiasFit:
    """Fit a precision-weighted common monthly bias and support taper."""
    values = np.asarray(innovations, dtype=np.float64)
    errors = np.asarray(error_std, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool)
    month_index = np.asarray(months, dtype=np.int64)
    selected = np.asarray(fit_mask, dtype=bool)
    if values.ndim != 4:
        raise ValueError("innovations must have (sensor, time, lat, lon) dimensions")
    if errors.shape != values.shape or usable.shape != values.shape:
        raise ValueError("innovation, error, and validity arrays must have identical shapes")
    if month_index.shape != (values.shape[1],) or selected.shape != (values.shape[1],):
        raise ValueError("month and fit masks must match the time dimension")
    if np.any((month_index < 1) | (month_index > 12)):
        raise ValueError("calendar months must be integers in [1, 12]")
    if not 0.0 <= support_min_fraction < support_full_fraction <= 1.0:
        raise ValueError("support fractions must satisfy 0 <= min < full <= 1")
    lower, upper = map(float, delta_bounds)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("delta_bounds must be finite and strictly ordered")
    invalid_error = usable & (~np.isfinite(errors) | (errors <= 0.0))
    if np.any(invalid_error):
        raise ValueError("observation errors must be finite and positive where data are valid")

    nsensor, _, nlat, nlon = values.shape
    shape = (12, nlat, nlon)
    raw_mean = np.zeros(shape, dtype=np.float64)
    bias = np.zeros(shape, dtype=np.float64)
    support = np.zeros(shape, dtype=np.float64)
    support_fraction = np.zeros(shape, dtype=np.float64)
    support_count = np.zeros(shape, dtype=np.int64)
    sensor_count = np.zeros((12, nsensor, nlat, nlon), dtype=np.int64)
    standard_error = np.zeros(shape, dtype=np.float64)
    support_day_total = np.zeros(12, dtype=np.int64)

    finite = usable & np.isfinite(values) & np.isfinite(errors) & (errors > 0.0)
    for month in range(1, 13):
        day_mask = selected & (month_index == month)
        index = month - 1
        support_day_total[index] = int(np.count_nonzero(day_mask))
        if not np.any(day_mask):
            continue
        month_valid = finite[:, day_mask]
        month_values = values[:, day_mask]
        month_errors = errors[:, day_mask]
        precision = np.zeros_like(month_errors, dtype=np.float64)
        np.divide(1.0, np.square(month_errors), out=precision, where=month_valid)
        precision_sum = precision.sum(axis=(0, 1), dtype=np.float64)
        weighted_sum = np.where(month_valid, month_values * precision, 0.0).sum(
            axis=(0, 1), dtype=np.float64
        )
        raw_mean[index] = np.divide(
            weighted_sum,
            precision_sum,
            out=np.zeros_like(weighted_sum),
            where=precision_sum > 0.0,
        )
        standard_error[index] = np.sqrt(
            np.divide(
                1.0,
                precision_sum,
                out=np.zeros_like(precision_sum),
                where=precision_sum > 0.0,
            )
        )
        sensor_count[index] = month_valid.sum(axis=1, dtype=np.int64)
        observed_days = np.any(month_valid, axis=0)
        support_count[index] = observed_days.sum(axis=0, dtype=np.int64)
        fraction = support_count[index] / float(support_day_total[index])
        support_fraction[index] = fraction
        supported = (fraction >= support_min_fraction) & (precision_sum > 0.0)

        smoothed_bias = masked_boxcar_smooth(raw_mean[index], supported, smoothing_passes)
        bias[index] = np.where(supported, np.clip(smoothed_bias, lower, upper), 0.0)
        raw_support = np.clip(
            (fraction - support_min_fraction) / (support_full_fraction - support_min_fraction),
            0.0,
            1.0,
        )
        smoothed_support = masked_boxcar_smooth(raw_support, supported, smoothing_passes)
        support[index] = np.where(
            fraction >= support_min_fraction, np.clip(smoothed_support, 0.0, 1.0), 0.0
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


def innovation(observation: FloatArray, model: FloatArray, monthly_bias: FloatArray) -> FloatArray:
    """Return transformed-space observation-minus-model-minus-bias."""
    return (
        np.asarray(observation, dtype=np.float64)
        - np.asarray(model, dtype=np.float64)
        - np.asarray(monthly_bias, dtype=np.float64)
    )


def build_effective_covariance(
    error_std: FloatArray,
    latitudes: FloatArray,
    common_factors: FloatArray | None = None,
) -> EffectiveCovariance:
    """Build ``D + U U.T`` after applying the cosine-area representation."""
    sigma = np.asarray(error_std, dtype=np.float64).reshape(-1)
    latitude = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    if sigma.shape != latitude.shape:
        raise ValueError("error and latitude vectors must have identical shapes")
    if np.any(~np.isfinite(sigma) | (sigma <= 0.0)):
        raise ValueError("observation errors must be finite and positive")
    area = np.cos(np.deg2rad(latitude))
    if np.any(~np.isfinite(area) | (area <= 0.0)):
        raise ValueError("observation latitude must have positive cosine area weight")
    diagonal = np.square(sigma) / area
    if common_factors is None:
        factors = np.empty((sigma.size, 0), dtype=np.float64)
    else:
        raw_factors = np.asarray(common_factors, dtype=np.float64)
        if raw_factors.ndim != 2 or raw_factors.shape[0] != sigma.size:
            raise ValueError("common factors must have shape (observation, common_mode)")
        if np.any(~np.isfinite(raw_factors)):
            raise ValueError("common factors must be finite")
        factors = raw_factors / np.sqrt(area)[:, None]
    return EffectiveCovariance(diagonal=diagonal, factors=factors)


def apply_inverse_covariance(covariance: EffectiveCovariance, rhs: FloatArray) -> FloatArray:
    """Apply ``(D + U U.T)^-1`` using the Woodbury identity."""
    diagonal = np.asarray(covariance.diagonal, dtype=np.float64).reshape(-1)
    factors = np.asarray(covariance.factors, dtype=np.float64)
    values = np.asarray(rhs, dtype=np.float64)
    vector = values.ndim == 1
    matrix = values[:, None] if vector else values
    if matrix.ndim != 2 or matrix.shape[0] != diagonal.size:
        raise ValueError("covariance right-hand side has an incompatible row count")
    if factors.ndim != 2 or factors.shape[0] != diagonal.size:
        raise ValueError("covariance factors have an incompatible row count")
    if np.any(~np.isfinite(diagonal) | (diagonal <= 0.0)):
        raise ValueError("covariance diagonal must be finite and positive")
    diagonal_inverse_rhs = matrix / diagonal[:, None]
    if factors.shape[1] == 0:
        result = diagonal_inverse_rhs
    else:
        diagonal_inverse_factors = factors / diagonal[:, None]
        middle = np.eye(factors.shape[1]) + factors.T @ diagonal_inverse_factors
        correction = diagonal_inverse_factors @ np.linalg.solve(
            middle, factors.T @ diagonal_inverse_rhs
        )
        result = diagonal_inverse_rhs - correction
    return result[:, 0] if vector else result


def solve_one_day(
    patterns: FloatArray,
    innovations: FloatArray,
    covariance: EffectiveCovariance,
    ridge: float,
) -> ProjectionSolution:
    """Solve one complete reduced-space ridge system using the full information matrix."""
    design = np.asarray(patterns, dtype=np.float64)
    residual = np.asarray(innovations, dtype=np.float64).reshape(-1)
    if design.ndim != 2 or design.shape[0] != residual.size:
        raise ValueError("patterns must have shape (observation, mode)")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    mode_count = design.shape[1]
    zeros = np.zeros(mode_count, dtype=np.float64)
    zero_matrix = np.zeros((mode_count, mode_count), dtype=np.float64)
    if residual.size == 0:
        if ridge == 0.0:
            raise ValueError("ridge=0 requires full-rank observation information")
        prior_variance = np.full(mode_count, 1.0 / ridge, dtype=np.float64)
        return ProjectionSolution(
            zeros,
            zeros,
            prior_variance,
            zeros,
            prior_variance,
            1.0,
            0,
            zero_matrix,
        )
    if np.any(~np.isfinite(design)) or np.any(~np.isfinite(residual)):
        raise ValueError("projection design and innovation must be finite")

    inverse_design = apply_inverse_covariance(covariance, design)
    inverse_residual = apply_inverse_covariance(covariance, residual)
    information = design.T @ inverse_design
    information = 0.5 * (information + information.T)
    information_eigenvalues = np.clip(np.linalg.eigvalsh(information), 0.0, None)
    tolerance = (
        np.finfo(np.float64).eps
        * max(information.shape, default=1)
        * max(float(information_eigenvalues.max(initial=0.0)), 1.0)
    )
    effective_rank = int(np.count_nonzero(information_eigenvalues > tolerance))
    if ridge == 0.0 and effective_rank < mode_count:
        raise ValueError("ridge=0 requires full-rank observation information")
    normal = information + ridge * np.eye(mode_count)
    coefficients = np.asarray(
        np.linalg.solve(normal, design.T @ inverse_residual), dtype=np.float64
    )
    posterior = np.linalg.inv(normal)
    resolution_matrix = np.linalg.solve(normal, information)
    resolution = np.clip(np.diag(resolution_matrix), 0.0, 1.0)
    resolution_eigenvalues = np.divide(
        information_eigenvalues,
        information_eigenvalues + ridge,
        out=np.zeros_like(information_eigenvalues),
        where=(information_eigenvalues + ridge) > 0.0,
    )
    posterior_eigenvalues = np.asarray(np.linalg.eigvalsh(posterior), dtype=np.float64)
    return ProjectionSolution(
        coefficients=coefficients,
        resolution=resolution,
        posterior_variance=np.clip(np.diag(posterior), 0.0, None),
        resolution_eigenvalues=resolution_eigenvalues,
        posterior_eigenvalues=posterior_eigenvalues,
        condition_number=float(np.linalg.cond(normal)),
        effective_rank=effective_rank,
        information=information,
    )


def mode_coverage(patterns: FloatArray, observed: BoolArray, latitudes: FloatArray) -> FloatArray:
    """Return union-mask coverage of each mode's cosine-weighted variance."""
    fields = np.asarray(patterns, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    latitude = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    if fields.ndim != 3 or mask.shape != fields.shape[1:]:
        raise ValueError("patterns and observed mask must have (mode, lat, lon)/(lat, lon)")
    if fields.shape[1] != latitude.size:
        raise ValueError("latitude size does not match pattern grid")
    weight = np.cos(np.deg2rad(latitude))[:, None]
    energy = np.square(fields) * weight[None, :, :]
    denominator = energy.sum(axis=(1, 2), dtype=np.float64)
    numerator = np.where(mask[None, :, :], energy, 0.0).sum(axis=(1, 2), dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


__all__ = [
    "EffectiveCovariance",
    "MonthlyBiasFit",
    "ProjectionSolution",
    "apply_inverse_covariance",
    "build_effective_covariance",
    "fit_monthly_bias",
    "innovation",
    "masked_boxcar_smooth",
    "mode_coverage",
    "solve_one_day",
]
