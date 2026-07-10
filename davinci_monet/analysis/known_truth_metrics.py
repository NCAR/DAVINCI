"""Pure weighted field, subspace, and mode-matching recovery metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class FieldMetrics:
    """Cosine-latitude-weighted metrics for one estimate/truth pair."""

    correlation: float
    origin_slope: float
    bias: float
    rmse: float
    truth_rms: float
    nrmse: float
    valid_count: int
    candidate_count: int
    excluded_fraction: float


@dataclass(frozen=True)
class SubspaceMetrics:
    """Principal-angle diagnostics for two weighted spatial subspaces."""

    angles_degrees: np.ndarray
    projector_error: float
    estimate_rank: int
    truth_rank: int


@dataclass(frozen=True)
class ModeMatch:
    """One weighted Hungarian basis-mode match."""

    estimate_index: int
    truth_index: int
    sign: float
    similarity: float
    scale: float


def canonical_dimensions(data: xr.DataArray) -> xr.DataArray:
    """Normalize the synthetic oracle's distinct grid/mode dimension names."""
    rename: dict[str, str] = {}
    for old, new in (
        ("mode_lat", "lat"),
        ("mode_lon", "lon"),
        ("truth_mode", "mode"),
    ):
        if old in data.dims and new not in data.dims:
            rename[old] = new
    return data.rename(rename)


def align_fields(
    estimate: xr.DataArray, truth: xr.DataArray, mask: xr.DataArray | None
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None]:
    """Require identical estimate/truth axes and broadcast a compatible mask."""
    left = canonical_dimensions(estimate)
    right = canonical_dimensions(truth)
    if set(left.dims) != set(right.dims):
        raise ValueError(
            "estimate and truth fields must have identical dimensions; "
            f"got {left.dims} and {right.dims}"
        )
    right = right.transpose(*left.dims)
    try:
        left, right = xr.align(left, right, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("estimate and truth coordinates must match exactly") from exc
    selected = None
    if mask is not None:
        selected = canonical_dimensions(mask)
        extra = [dim for dim in selected.dims if dim not in left.dims]
        for dim in extra:
            dim_name = str(dim)
            selected = selected.any(dim_name) if dim_name == "sensor" else selected.all(dim_name)
        unexpected = set(selected.dims).difference(left.dims)
        if unexpected:
            raise ValueError(
                f"metric mask has incompatible dimensions: {sorted(map(str, unexpected))}"
            )
        try:
            selected, _ = xr.align(selected, left, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError("metric mask coordinates must match the scored field exactly") from exc
        selected, _ = xr.broadcast(selected, left)
        selected = selected.transpose(*left.dims)
    return left, right, selected


def _normalized_weights(
    template: xr.DataArray, valid: xr.DataArray
) -> tuple[np.ndarray, np.ndarray]:
    if "lat" not in template.coords:
        raise ValueError("weighted field metrics require a latitude coordinate")
    latitude = template["lat"]
    if latitude.ndim != 1:
        raise ValueError("weighted field metrics require one-dimensional latitude")
    coslat = xr.DataArray(
        np.clip(np.cos(np.deg2rad(latitude)), 0.0, None),
        dims=latitude.dims,
        coords=latitude.coords,
    )
    raw, _ = xr.broadcast(coslat, template)
    raw = raw.transpose(*template.dims).where(valid, 0.0)
    spatial_dims = [dim for dim in template.dims if dim != "time"]
    if "time" in template.dims:
        denominator = raw.sum(spatial_dims)
        day_valid = denominator > 0.0
        weights = (raw / denominator.where(day_valid)).fillna(0.0)
        day_count = _scalar_int(day_valid.sum())
        if day_count:
            weights = weights / day_count
    else:
        denominator = raw.sum()
        weights = raw / denominator if _scalar_float(denominator) > 0.0 else raw
    return np.asarray(weights.values, dtype=np.float64), np.asarray(valid.values, dtype=bool)


def _scalar_float(value: xr.DataArray) -> float:
    if value.chunks is not None:
        value = value.compute()
    return float(value.item())


def _scalar_int(value: xr.DataArray) -> int:
    return int(_scalar_float(value))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide with useful exact-zero behavior for closure diagnostics."""
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else float("inf")


def weighted_field_metrics(
    estimate: xr.DataArray,
    truth: xr.DataArray,
    mask: xr.DataArray | None = None,
) -> FieldMetrics:
    """Score fields with equal day weights and normalized cosine-latitude cell weights."""
    left, right, selected = align_fields(estimate, truth, mask)
    candidate = cast(xr.DataArray, np.isfinite(right))
    valid = candidate & cast(xr.DataArray, np.isfinite(left))
    if selected is not None:
        valid = valid & selected.astype(bool)
    candidate_count = _scalar_int(candidate.sum())
    valid_count = _scalar_int(valid.sum())
    excluded = 1.0 - valid_count / candidate_count if candidate_count else 1.0
    if valid_count == 0:
        return FieldMetrics(*(float("nan"),) * 6, 0, candidate_count, excluded)

    weights, valid_values = _normalized_weights(left, valid)
    x = np.asarray(left.values, dtype=np.float64)
    y = np.asarray(right.values, dtype=np.float64)
    x = np.where(valid_values, x, 0.0)
    y = np.where(valid_values, y, 0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return FieldMetrics(*(float("nan"),) * 6, valid_count, candidate_count, excluded)
    weights /= weight_sum
    x_mean = float(np.sum(weights * x))
    y_mean = float(np.sum(weights * y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = float(np.sum(weights * x_centered * y_centered))
    x_variance = float(np.sum(weights * x_centered**2))
    y_variance = float(np.sum(weights * y_centered**2))
    correlation = safe_ratio(covariance, np.sqrt(x_variance * y_variance))
    error = x - y
    rmse = float(np.sqrt(np.sum(weights * error**2)))
    truth_rms = float(np.sqrt(np.sum(weights * y**2)))
    return FieldMetrics(
        correlation=correlation,
        origin_slope=safe_ratio(float(np.sum(weights * x * y)), float(np.sum(weights * y**2))),
        bias=float(np.sum(weights * error)),
        rmse=rmse,
        truth_rms=truth_rms,
        nrmse=safe_ratio(rmse, truth_rms),
        valid_count=valid_count,
        candidate_count=candidate_count,
        excluded_fraction=excluded,
    )


def _basis_arrays(
    estimate: xr.DataArray, truth: xr.DataArray, mask: xr.DataArray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    left = canonical_dimensions(estimate)
    right = canonical_dimensions(truth)
    if "mode" not in left.dims or "mode" not in right.dims:
        raise ValueError("subspace metrics require a mode dimension")
    left = left.transpose("mode", "lat", "lon")
    right = right.transpose("mode", "lat", "lon")
    try:
        left, right = xr.align(
            left,
            right,
            join="exact",
            copy=False,
            exclude={"mode"},
        )
    except ValueError as exc:
        raise ValueError("estimate and truth basis coordinates must match exactly") from exc
    left_finite = cast(xr.DataArray, np.isfinite(left))
    right_finite = cast(xr.DataArray, np.isfinite(right))
    spatial_valid = left_finite.all("mode") & right_finite.all("mode")
    if mask is not None:
        selected = canonical_dimensions(mask)
        for dim in list(selected.dims):
            if dim not in ("lat", "lon"):
                selected = selected.any(str(dim))
        try:
            selected, spatial_valid = xr.align(
                selected.astype(bool), spatial_valid, join="exact", copy=False
            )
        except ValueError as exc:
            raise ValueError("subspace mask coordinates must match the basis exactly") from exc
        spatial_valid = spatial_valid & selected
    latitude = left["lat"]
    weights = xr.DataArray(
        np.clip(np.cos(np.deg2rad(latitude)), 0.0, None),
        dims=("lat",),
        coords={"lat": latitude},
    )
    weights, spatial_valid = xr.broadcast(weights, spatial_valid)
    keep = np.asarray(spatial_valid.values, dtype=bool).reshape(-1)
    if not np.any(keep):
        raise ValueError("subspace metrics have no valid spatial cells")
    sqrt_weight = np.sqrt(np.asarray(weights.values, dtype=np.float64).reshape(-1)[keep])
    left_values = np.asarray(left.values, dtype=np.float64).reshape(left.sizes["mode"], -1)
    right_values = np.asarray(right.values, dtype=np.float64).reshape(right.sizes["mode"], -1)
    return left_values[:, keep] * sqrt_weight, right_values[:, keep] * sqrt_weight


def weighted_subspace_metrics(
    estimate: xr.DataArray, truth: xr.DataArray, mask: xr.DataArray | None = None
) -> SubspaceMetrics:
    """Return weighted principal angles and normalized projector distance."""
    left, right = _basis_arrays(estimate, truth, mask)
    left_q, _ = np.linalg.qr(left.T)
    right_q, _ = np.linalg.qr(right.T)
    left_rank = int(np.linalg.matrix_rank(left))
    right_rank = int(np.linalg.matrix_rank(right))
    left_q = left_q[:, :left_rank]
    right_q = right_q[:, :right_rank]
    cosines = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    cosines[np.isclose(cosines, 1.0, rtol=0.0, atol=1.0e-12)] = 1.0
    angles = np.rad2deg(np.arccos(cosines))
    denominator = left_rank + right_rank
    distance_sq = left_rank + right_rank - 2.0 * float(np.sum(cosines**2))
    projector_error = np.sqrt(max(0.0, distance_sq) / denominator) if denominator else np.nan
    return SubspaceMetrics(angles, float(projector_error), left_rank, right_rank)


def match_weighted_modes(
    estimate: xr.DataArray, truth: xr.DataArray, mask: xr.DataArray | None = None
) -> tuple[ModeMatch, ...]:
    """Match individual spatial modes by absolute weighted correlation."""
    left, right = _basis_arrays(estimate, truth, mask)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm[:, None] * right_norm[None, :]
    similarity = np.divide(
        left @ right.T,
        denominator,
        out=np.zeros((left.shape[0], right.shape[0]), dtype=np.float64),
        where=denominator > 0.0,
    )
    left_index, right_index = linear_sum_assignment(-np.abs(similarity))
    return tuple(
        ModeMatch(
            int(i),
            int(j),
            -1.0 if similarity[i, j] < 0.0 else 1.0,
            float(abs(similarity[i, j])),
            safe_ratio(float(left[i] @ right[j]), float(right[j] @ right[j])),
        )
        for i, j in zip(left_index, right_index, strict=True)
    )


__all__ = [
    "FieldMetrics",
    "ModeMatch",
    "SubspaceMetrics",
    "align_fields",
    "canonical_dimensions",
    "match_weighted_modes",
    "safe_ratio",
    "weighted_field_metrics",
    "weighted_subspace_metrics",
]
