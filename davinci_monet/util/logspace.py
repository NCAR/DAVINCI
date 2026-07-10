"""Stable shifted-log transforms and bounded AOD ratio conversion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import xarray as xr


def shifted_log(aod: xr.DataArray, epsilon: float) -> xr.DataArray:
    """Return ``log(aod + epsilon)`` with negative AOD masked."""
    _validate_epsilon(epsilon)
    valid = np.isfinite(aod) & (aod >= 0.0)
    out = cast(xr.DataArray, np.log(aod.where(valid) + epsilon))
    out.attrs = dict(aod.attrs)
    out.attrs["transform"] = "log(aod + epsilon)"
    out.attrs["log_epsilon"] = float(epsilon)
    return out


def shifted_exp(values: xr.DataArray, epsilon: float) -> xr.DataArray:
    """Invert :func:`shifted_log`."""
    _validate_epsilon(epsilon)
    out = cast(xr.DataArray, np.exp(values) - epsilon)
    out.attrs = {key: value for key, value in values.attrs.items() if key != "transform"}
    return out


def ratio_delta_bounds(
    model_aod: xr.DataArray,
    r_bounds: Sequence[float],
    epsilon: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return AOD-dependent shifted-log deltas for physical ratio bounds."""
    r_min, r_max = _validate_ratio_bounds(r_bounds)
    _validate_epsilon(epsilon)
    if _any(model_aod < 0.0):
        raise ValueError("model AOD must be non-negative")

    denominator = model_aod + epsilon
    # log1p is stable when AOD is small relative to epsilon.
    lower = cast(xr.DataArray, np.log1p((r_min - 1.0) * model_aod / denominator))
    upper = cast(xr.DataArray, np.log1p((r_max - 1.0) * model_aod / denominator))
    return lower, upper


def apply_shifted_log_correction(
    model_aod: xr.DataArray,
    delta: xr.DataArray,
    *,
    epsilon: float,
    r_bounds: Sequence[float] = (0.2, 5.0),
    aod_floor: float = 0.001,
    support: xr.DataArray | None = None,
) -> xr.Dataset:
    """Convert a requested shifted-log correction to a bounded physical ratio.

    The physical ratio bounds are converted into model-AOD-dependent bounds in
    transformed space before exponentiation. Cells below ``aod_floor`` or with
    zero support receive the identity ratio.
    """
    r_min, r_max = _validate_ratio_bounds(r_bounds)
    _validate_epsilon(epsilon)
    if aod_floor < 0.0:
        raise ValueError("aod_floor must be non-negative")

    model, requested = xr.align(model_aod, delta, join="exact")
    if support is None:
        support_arr = xr.ones_like(model, dtype=float)
    else:
        model, requested, support_arr = xr.align(model, requested, support, join="exact")
        if _any((support_arr < 0.0) | (support_arr > 1.0)):
            raise ValueError("support must be between 0 and 1")

    finite = np.isfinite(model) & np.isfinite(requested) & np.isfinite(support_arr)
    if _any((model < 0.0) & np.isfinite(model)):
        raise ValueError("model AOD must be non-negative")
    # A physical ratio is undefined at exactly zero AOD, even when the
    # configured floor is zero. Keep those cells at identity.
    active = finite & (model > 0.0) & (model >= aod_floor) & (support_arr > 0.0)

    lower, upper = ratio_delta_bounds(model.fillna(0.0), (r_min, r_max), epsilon)
    safe_delta = requested.clip(min=lower, max=upper).where(active, 0.0)

    # safe_delta is bounded before exponentiation, preventing overflow.
    target_raw = (model.fillna(0.0) + epsilon) * np.exp(safe_delta) - epsilon
    denominator = model.where(active, 1.0)
    ratio = (target_raw / denominator).clip(min=r_min, max=r_max).where(active, 1.0)
    target = model * ratio
    applied_delta = cast(
        xr.DataArray,
        np.log(target + epsilon) - np.log(model + epsilon),
    ).where(finite)

    clip_reason = xr.zeros_like(requested, dtype=np.int8)
    clip_reason = xr.where(active & (requested < lower), np.int8(-1), clip_reason)
    clip_reason = xr.where(active & (requested > upper), np.int8(1), clip_reason)
    clip_reason = xr.where(finite & ~active, np.int8(2), clip_reason)
    clip_reason = clip_reason.where(finite)

    ratio.attrs.update(units="1", long_name="Applied physical AOD ratio")
    safe_delta.attrs.update(units="1", long_name="Bounded shifted-log correction")
    applied_delta.attrs.update(units="1", long_name="Applied shifted-log correction")
    clip_reason.attrs.update(
        long_name="Correction policy reason",
        flag_values=[-1, 0, 1, 2],
        flag_meanings="lower_clip unchanged upper_clip identity_low_aod_or_support",
    )
    return xr.Dataset(
        {
            "ratio": ratio,
            "delta_requested": requested,
            "delta_safe": safe_delta,
            "delta_applied": applied_delta,
            "aod_target": target,
            "clip_reason": clip_reason,
        },
        attrs={
            "log_epsilon": float(epsilon),
            "aod_floor": float(aod_floor),
            "r_min": r_min,
            "r_max": r_max,
        },
    )


def _validate_epsilon(epsilon: float) -> None:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")


def _any(condition: xr.DataArray) -> bool:
    value = condition.any()
    if value.chunks is not None:
        value = value.compute()
    return bool(value.item())


def _validate_ratio_bounds(bounds: Sequence[float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("r_bounds must contain exactly two values")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or lower >= upper:
        raise ValueError("r_bounds must be finite, positive, and increasing")
    return lower, upper
