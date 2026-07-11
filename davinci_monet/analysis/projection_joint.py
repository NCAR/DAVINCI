"""Joint seasonal-bias and EOF-anomaly fit for innovation projection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from davinci_monet.analysis.projection_batches import fit_monthly_bias_batched
from davinci_monet.analysis.projection_core import (
    MonthlyBiasFit,
    apply_inverse_covariance,
    solve_one_day,
)
from davinci_monet.analysis.projection_inputs import ProjectionObservation
from davinci_monet.analysis.projection_joint_core import (
    MonthNormal,
    laplacian_edges,
    seasonal_design,
    solve_month_bias,
)
from davinci_monet.analysis.projection_joint_inputs import (
    iter_joint_fit_days,
    require_connected_sensor_overlap,
    sensor_overlap_counts,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def _seasonal_gauge(
    temporal_design: FloatArray,
    coefficients: FloatArray,
    observed: BoolArray,
    ridge: float,
) -> FloatArray:
    """Fit theta under the coefficient-ridge precision gauge."""
    weight = np.where(observed, ridge if ridge > 0.0 else 1.0, 0.0)
    weighted_design = temporal_design * weight[:, None]
    normal = temporal_design.T @ weighted_design
    if np.linalg.matrix_rank(normal) < temporal_design.shape[1]:
        raise ValueError("joint seasonal bias requires three independent temporal design columns")
    return np.asarray(
        np.linalg.solve(normal, temporal_design.T @ (coefficients * weight[:, None])),
        dtype=np.float64,
    )


def _update_coefficients(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    patterns: FloatArray,
    support: FloatArray,
    months: IntArray,
    fit_mask: BoolArray,
    bias: FloatArray,
    offsets: FloatArray,
    theta: FloatArray,
    ridge: float,
    time_chunk_size: int,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    temporal_basis = seasonal_design()
    seasonal = temporal_basis @ theta
    coefficients = np.zeros((model.sizes["time"], patterns.shape[0]), dtype=np.float64)
    observed = np.zeros(model.sizes["time"], dtype=bool)
    for day, month, rows in iter_joint_fit_days(
        model, observations, patterns, support, months, fit_mask, time_chunk_size
    ):
        center = seasonal[month]
        coefficients[day] = center
        if not rows.values.size:
            continue
        residual = (
            rows.values
            - bias[month].reshape(-1)[rows.cell]
            - offsets[rows.sensor]
            - rows.design @ center
        )
        coefficients[day] += solve_one_day(
            rows.design, residual, rows.covariance, ridge
        ).coefficients
        observed[day] = True
    temporal_design = temporal_basis[months - 1]
    selected = fit_mask & observed
    theta = _seasonal_gauge(temporal_design, coefficients, selected, ridge)
    return coefficients, observed, theta


def _spatial_normals(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    patterns: FloatArray,
    support: FloatArray,
    months: IntArray,
    fit_mask: BoolArray,
    coefficients: FloatArray,
    offsets: FloatArray,
    time_chunk_size: int,
) -> list[MonthNormal]:
    ncell = patterns.shape[1]
    normals = [MonthNormal(np.zeros(ncell), [], np.zeros(ncell)) for _ in range(12)]
    for day, month, rows in iter_joint_fit_days(
        model, observations, patterns, support, months, fit_mask, time_chunk_size
    ):
        if not rows.values.size:
            continue
        residual = rows.values - rows.design @ coefficients[day] - offsets[rows.sensor]
        normal = normals[month]
        inverse_diagonal = 1.0 / rows.covariance.diagonal
        np.add.at(normal.diagonal, rows.cell, inverse_diagonal)
        weighted = apply_inverse_covariance(rows.covariance, residual)
        np.add.at(normal.rhs, rows.cell, weighted)
        if rows.covariance.factors.shape[1]:
            weighted_factor = rows.covariance.factors * inverse_diagonal[:, None]
            collapsed = np.zeros((ncell, weighted_factor.shape[1]))
            np.add.at(collapsed, rows.cell, weighted_factor)
            middle = np.eye(weighted_factor.shape[1]) + (
                rows.covariance.factors.T @ weighted_factor
            )
            normal.factors.append(np.linalg.solve(np.linalg.cholesky(middle), collapsed.T).T)
    return normals


def _update_bias(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    patterns: FloatArray,
    support: FloatArray,
    months: IntArray,
    fit_mask: BoolArray,
    coefficients: FloatArray,
    offsets: FloatArray,
    laplacian_strength: float,
    time_chunk_size: int,
) -> tuple[FloatArray, IntArray, FloatArray, FloatArray]:
    normals = _spatial_normals(
        model,
        observations,
        patterns,
        support,
        months,
        fit_mask,
        coefficients,
        offsets,
        time_chunk_size,
    )
    nlat, nlon = model.sizes["lat"], model.sizes["lon"]
    bias = np.zeros((12, nlat * nlon))
    rank = np.zeros(12, dtype=np.int64)
    eigenvalue = np.zeros((12, patterns.shape[0]))
    precision_scale = np.zeros(12)
    for month, normal in enumerate(normals):
        active = support[month].reshape(-1) > 0.0
        bias[month], rank[month], eigenvalue[month], precision_scale[month] = solve_month_bias(
            normal,
            patterns,
            active,
            nlat,
            nlon,
            laplacian_strength,
        )
    return bias.reshape(12, nlat, nlon), rank, eigenvalue, precision_scale


def _update_offsets(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    patterns: FloatArray,
    support: FloatArray,
    months: IntArray,
    fit_mask: BoolArray,
    coefficients: FloatArray,
    bias: FloatArray,
    time_chunk_size: int,
) -> tuple[FloatArray, FloatArray]:
    sensor_count = len(observations)
    if sensor_count == 1:
        return np.zeros(1), np.zeros(1)
    normal = np.zeros((sensor_count, sensor_count))
    rhs = np.zeros(sensor_count)
    identity = np.eye(sensor_count)
    for day, month, rows in iter_joint_fit_days(
        model, observations, patterns, support, months, fit_mask, time_chunk_size
    ):
        if not rows.values.size:
            continue
        design = identity[rows.sensor]
        inverse_design = apply_inverse_covariance(rows.covariance, design)
        residual = (
            rows.values - rows.design @ coefficients[day] - bias[month].reshape(-1)[rows.cell]
        )
        normal += design.T @ inverse_design
        rhs += design.T @ apply_inverse_covariance(rows.covariance, residual)
    contrast = np.vstack((np.eye(sensor_count - 1), -np.ones(sensor_count - 1)))
    reduced = contrast.T @ normal @ contrast
    covariance = contrast @ np.linalg.inv(reduced) @ contrast.T
    offsets = contrast @ np.linalg.solve(reduced, contrast.T @ rhs)
    return offsets, np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def _objective(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    patterns: FloatArray,
    support: FloatArray,
    months: IntArray,
    fit_mask: BoolArray,
    coefficients: FloatArray,
    theta: FloatArray,
    bias: FloatArray,
    offsets: FloatArray,
    ridge: float,
    laplacian_strength: float,
    precision_scale: FloatArray,
    time_chunk_size: int,
) -> float:
    total = 0.0
    seasonal = seasonal_design() @ theta
    for day, month, rows in iter_joint_fit_days(
        model, observations, patterns, support, months, fit_mask, time_chunk_size
    ):
        anomaly = coefficients[day] - seasonal[month]
        total += ridge * float(anomaly @ anomaly)
        if rows.values.size:
            residual = (
                rows.values
                - rows.design @ coefficients[day]
                - bias[month].reshape(-1)[rows.cell]
                - offsets[rows.sensor]
            )
            total += float(residual @ apply_inverse_covariance(rows.covariance, residual))
    nlat, nlon = model.sizes["lat"], model.sizes["lon"]
    for month in range(12):
        active = support[month].reshape(-1) > 0.0
        left, right = laplacian_edges(active, nlat, nlon)
        values = bias[month].reshape(-1)[active]
        total += (
            laplacian_strength
            * precision_scale[month]
            * float(np.square(values[left] - values[right]).sum())
        )
    return total


def fit_joint_projection_bias_batched(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    patterns: FloatArray,
    months: IntArray,
    fit_mask: BoolArray,
    *,
    support_min_fraction: float,
    support_full_fraction: float,
    smoothing_passes: int,
    delta_bounds: tuple[float, float],
    ridge: float,
    sensor_offset_method: str,
    laplacian_strength: float,
    tolerance: float,
    max_iterations: int,
    time_chunk_size: int,
) -> MonthlyBiasFit:
    """Fit the v2 joint decomposition from bounded observation batches."""
    pattern = np.asarray(patterns, dtype=np.float64)
    if pattern.ndim != 3 or pattern.shape[1:] != (
        model.sizes["lat"],
        model.sizes["lon"],
    ):
        raise ValueError("joint projection patterns must match the model grid")
    if sensor_offset_method not in {"none", "overlap_zero_sum"}:
        raise ValueError("unknown joint sensor offset method")
    if not np.isfinite(laplacian_strength) or laplacian_strength < 0.0:
        raise ValueError("joint bias Laplacian strength must be finite and nonnegative")
    if not np.isfinite(tolerance) or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("joint bias convergence controls are invalid")
    base = fit_monthly_bias_batched(
        model,
        observations,
        months,
        fit_mask,
        support_min_fraction=support_min_fraction,
        support_full_fraction=support_full_fraction,
        smoothing_passes=smoothing_passes,
        delta_bounds=delta_bounds,
        time_chunk_size=time_chunk_size,
    )
    overlap = sensor_overlap_counts(
        model, observations, base.support, months, fit_mask, time_chunk_size
    )
    if sensor_offset_method == "overlap_zero_sum":
        require_connected_sensor_overlap(overlap)

    flat_pattern = pattern.reshape(pattern.shape[0], -1)
    coefficient = np.zeros((model.sizes["time"], pattern.shape[0]))
    theta = np.zeros((3, pattern.shape[0]))
    perpendicular = np.zeros_like(base.bias)
    offset = np.zeros(len(observations))
    offset_error = np.zeros(len(observations))
    precision_scale = np.zeros(12)
    history = [
        _objective(
            model,
            observations,
            flat_pattern,
            base.support,
            months,
            fit_mask,
            coefficient,
            theta,
            perpendicular,
            offset,
            ridge,
            laplacian_strength,
            precision_scale,
            time_chunk_size,
        )
    ]
    converged = False
    rank = np.zeros(12, dtype=np.int64)
    eigenvalue = np.zeros((12, pattern.shape[0]))
    for iteration in range(1, max_iterations + 1):
        coefficient, _observed, theta = _update_coefficients(
            model,
            observations,
            flat_pattern,
            base.support,
            months,
            fit_mask,
            perpendicular,
            offset,
            theta,
            ridge,
            time_chunk_size,
        )
        perpendicular, rank, eigenvalue, precision_scale = _update_bias(
            model,
            observations,
            flat_pattern,
            base.support,
            months,
            fit_mask,
            coefficient,
            offset,
            laplacian_strength,
            time_chunk_size,
        )
        if sensor_offset_method == "overlap_zero_sum":
            offset, offset_error = _update_offsets(
                model,
                observations,
                flat_pattern,
                base.support,
                months,
                fit_mask,
                coefficient,
                perpendicular,
                time_chunk_size,
            )
        objective = _objective(
            model,
            observations,
            flat_pattern,
            base.support,
            months,
            fit_mask,
            coefficient,
            theta,
            perpendicular,
            offset,
            ridge,
            laplacian_strength,
            precision_scale,
            time_chunk_size,
        )
        allowed = 1.0e-10 * max(abs(history[-1]), 1.0)
        if objective > history[-1] + allowed:
            raise ValueError("joint bias objective increased during block-coordinate fit")
        history.append(objective)
        decrease = (history[-2] - history[-1]) / max(abs(history[-2]), 1.0)
        if decrease < tolerance:
            converged = True
            break
    if not converged:
        raise ValueError("joint seasonal bias fit did not converge within max_iterations")

    mode_coefficient = seasonal_design() @ theta
    modal_bias = np.einsum("mk,kij->mij", mode_coefficient, pattern)
    combined = perpendicular + modal_bias
    lower, upper = delta_bounds
    combined = np.where(
        base.support > 0.0,
        np.clip(combined, float(lower), float(upper)),
        0.0,
    )
    return MonthlyBiasFit(
        raw_mean=base.raw_mean,
        bias=combined,
        bias_applied=base.support * combined,
        support=base.support,
        support_fraction=base.support_fraction,
        support_count=base.support_count,
        support_day_total=base.support_day_total,
        sensor_count=base.sensor_count,
        standard_error=base.standard_error,
        perpendicular_bias=perpendicular,
        mode_coefficient=mode_coefficient,
        sensor_offset=offset,
        sensor_offset_standard_error=offset_error,
        sensor_overlap_count=overlap,
        pooled_observable_rank=rank,
        pooled_observable_eigenvalue=eigenvalue,
        objective_history=np.asarray(history),
        converged=True,
        iterations=iteration,
    )


__all__ = ["fit_joint_projection_bias_batched"]
