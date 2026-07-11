"""Projection-fit policy identity and saved-artifact loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.analysis.projection_core import MonthlyBiasFit
from davinci_monet.analysis.projection_inputs import field_value, same_coordinate
from davinci_monet.analysis.provenance import consistent_spec_hash

SupportPolicy = tuple[float, float, int, tuple[float, float]]


def select_bias_fit_window(time: pd.DatetimeIndex, spec: Any) -> np.ndarray:
    """Return the explicit, nonempty model-time selection used for fitting."""
    window = field_value(spec, "bias_fit_window", None)
    if window is None:
        raise ValueError("eof_projection requires bias_fit_window or bias_fit_artifact")
    start = pd.Timestamp(field_value(window, "start"))
    end = pd.Timestamp(field_value(window, "end"))
    if start > end:
        raise ValueError("bias_fit_window start must be at or before end")
    selected = np.asarray((time >= start) & (time <= end), dtype=bool)
    if not np.any(selected):
        raise ValueError("bias_fit_window has no overlap with the model time axis")
    return selected


def digest_projection_fit_policy(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 identity of a projection-fit policy."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def projection_fit_policy(
    spec: Any,
    sensor_names: Sequence[str],
    support_policy: SupportPolicy,
) -> dict[str, Any]:
    """Collect every setting that can alter the saved projection fit."""
    minimum, full, passes, bounds = support_policy
    return {
        "bias_fit_method": str(field_value(spec, "bias_fit_method", "monthly_mean")),
        "sensor_offset_method": str(field_value(spec, "sensor_offset_method", "none")),
        "support_min_fraction": minimum,
        "support_full_fraction": full,
        "support_smoothing_passes": passes,
        "delta_bounds": [bounds[0], bounds[1]],
        "ridge": float(field_value(spec, "ridge", 1.0)),
        "joint_bias_laplacian_strength": float(
            field_value(spec, "joint_bias_laplacian_strength", 1.0)
        ),
        "joint_bias_tolerance": float(field_value(spec, "joint_bias_tolerance", 1.0e-6)),
        "joint_bias_max_iterations": int(field_value(spec, "joint_bias_max_iterations", 20)),
        "sensor_names": list(sensor_names),
    }


def projection_fit_window_signature(time: pd.DatetimeIndex, selected: np.ndarray) -> str:
    """Hash selected timestamps after portable nanosecond normalization."""
    digest = hashlib.sha256()
    nanoseconds = time.to_numpy(dtype="datetime64[ns]").view("<i8")
    values = np.ascontiguousarray(nanoseconds[selected], dtype="<i8")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _validate_common_artifact(
    artifact: xr.Dataset,
    patterns: xr.DataArray,
    sensors: Sequence[str],
    basis_signature: str,
    grid_signature: str,
    epsilon: float,
    spec_hash: str | None,
    support_policy: SupportPolicy,
    fit_policy: Mapping[str, Any],
) -> str:
    required = {
        "clim_bias_raw_mean",
        "clim_bias",
        "clim_bias_applied",
        "spatial_support",
        "support_fraction",
        "support_count",
        "support_day_total",
        "clim_bias_sensor_count",
        "clim_bias_standard_error",
    }
    missing = sorted(required.difference(artifact.data_vars))
    if missing:
        raise ValueError(f"bias fit artifact is missing variables: {missing}")
    artifact_spec_hash = consistent_spec_hash([artifact])
    if artifact_spec_hash != spec_hash and (
        artifact_spec_hash is not None or spec_hash is not None
    ):
        raise ValueError("bias fit artifact scientific spec hash does not match current inputs")
    if artifact.attrs.get("projection_basis_signature") != basis_signature:
        raise ValueError("bias fit artifact basis signature does not match")
    if artifact.attrs.get("projection_grid_signature") != grid_signature:
        raise ValueError("bias fit artifact grid signature does not match")
    artifact_epsilon = float(artifact.attrs.get("projection_log_epsilon", np.nan))
    if not np.isclose(artifact_epsilon, epsilon, rtol=0.0, atol=1.0e-15):
        raise ValueError("bias fit artifact log epsilon does not match")
    minimum, full, passes, bounds = support_policy
    expected_policy = {
        "projection_support_min_fraction": minimum,
        "projection_support_full_fraction": full,
        "projection_support_smoothing_passes": passes,
        "projection_delta_lower": bounds[0],
        "projection_delta_upper": bounds[1],
    }
    for attr, expected in expected_policy.items():
        actual = artifact.attrs.get(attr)
        if actual is None or not np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-15):
            raise ValueError(f"bias fit artifact policy {attr!r} does not match")
    expected_method = str(fit_policy["bias_fit_method"])
    artifact_method = str(artifact.attrs.get("projection_bias_fit_method", "monthly_mean"))
    if artifact_method != expected_method:
        raise ValueError("bias fit artifact method does not match")
    expected_offset_method = str(fit_policy["sensor_offset_method"])
    artifact_offset_method = str(artifact.attrs.get("projection_sensor_offset_method", "none"))
    if artifact_offset_method != expected_offset_method:
        raise ValueError("bias fit artifact sensor-offset method does not match")
    same_coordinate(artifact["lat"], patterns["lat"], "artifact latitude")
    same_coordinate(artifact["lon"], patterns["lon"], "artifact longitude")
    if not np.array_equal(artifact["month"].values, np.arange(1, 13)):
        raise ValueError("bias fit artifact must contain all calendar months in order")
    if [str(value) for value in artifact["sensor"].values] != list(sensors):
        raise ValueError("bias fit artifact sensor order does not match")
    expected_dims = {
        "clim_bias_raw_mean": ("month", "lat", "lon"),
        "clim_bias": ("month", "lat", "lon"),
        "clim_bias_applied": ("month", "lat", "lon"),
        "spatial_support": ("month", "lat", "lon"),
        "support_fraction": ("month", "lat", "lon"),
        "support_count": ("month", "lat", "lon"),
        "support_day_total": ("month",),
        "clim_bias_sensor_count": ("month", "sensor", "lat", "lon"),
        "clim_bias_standard_error": ("month", "lat", "lon"),
    }
    for variable, dimensions in expected_dims.items():
        if artifact[variable].dims != dimensions:
            raise ValueError(f"bias fit artifact {variable!r} must have dimensions {dimensions}")
    return expected_method


def _load_joint_values(
    artifact: xr.Dataset, sensors: Sequence[str], fit_policy: Mapping[str, Any]
) -> dict[str, np.ndarray | bool | int | None]:
    expected_signature = digest_projection_fit_policy(fit_policy)
    if artifact.attrs.get("projection_bias_fit_policy_signature") != expected_signature:
        raise ValueError("bias fit artifact joint policy signature does not match")
    window_signature = str(artifact.attrs.get("projection_bias_fit_window_signature", ""))
    if len(window_signature) != 64:
        raise ValueError("bias fit artifact is missing its fit-window signature")
    try:
        fit_start = pd.Timestamp(artifact.attrs["projection_bias_fit_start"])
        fit_end = pd.Timestamp(artifact.attrs["projection_bias_fit_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("bias fit artifact has invalid fit-window provenance") from exc
    if fit_start > fit_end:
        raise ValueError("bias fit artifact has invalid fit-window provenance")
    fit_time = pd.date_range(fit_start, fit_end, freq="1D")
    selected = np.ones(fit_time.size, dtype=bool)
    if not selected.size or projection_fit_window_signature(fit_time, selected) != window_signature:
        raise ValueError("bias fit artifact fit-window signature does not match")
    joint_dims = {
        "clim_bias_perpendicular": ("month", "lat", "lon"),
        "clim_bias_mode_coefficient": ("month", "mode"),
        "sensor_offset": ("sensor",),
        "sensor_offset_standard_error": ("sensor",),
        "sensor_overlap_count": ("sensor", "sensor_pair"),
        "pooled_observable_rank": ("month",),
        "pooled_observable_eigenvalue": ("month", "eigen"),
        "joint_objective": ("joint_iteration",),
    }
    missing = sorted(set(joint_dims).difference(artifact.data_vars))
    if missing:
        raise ValueError(f"joint bias fit artifact is missing variables: {missing}")
    for variable, dimensions in joint_dims.items():
        if artifact[variable].dims != dimensions:
            raise ValueError(
                f"joint bias fit artifact {variable!r} must have dimensions {dimensions}"
            )
    if [str(value) for value in artifact["sensor_pair"].values] != list(sensors):
        raise ValueError("bias fit artifact paired-sensor order does not match")
    converged = str(artifact.attrs.get("projection_joint_bias_converged", "")).lower()
    if converged != "true":
        raise ValueError("joint bias fit artifact is not converged")
    return {
        "perpendicular_bias": np.asarray(
            artifact["clim_bias_perpendicular"].values, dtype=np.float64
        ),
        "mode_coefficient": np.asarray(
            artifact["clim_bias_mode_coefficient"].values, dtype=np.float64
        ),
        "sensor_offset": np.asarray(artifact["sensor_offset"].values, dtype=np.float64),
        "sensor_offset_standard_error": np.asarray(
            artifact["sensor_offset_standard_error"].values, dtype=np.float64
        ),
        "sensor_overlap_count": np.asarray(artifact["sensor_overlap_count"].values, dtype=np.int64),
        "pooled_observable_rank": np.asarray(
            artifact["pooled_observable_rank"].values, dtype=np.int64
        ),
        "pooled_observable_eigenvalue": np.asarray(
            artifact["pooled_observable_eigenvalue"].values, dtype=np.float64
        ),
        "objective_history": np.asarray(artifact["joint_objective"].values, dtype=np.float64),
        "converged": True,
        "iterations": int(artifact.attrs["projection_joint_bias_iterations"]),
    }


def load_projection_fit_artifact(
    artifact: xr.Dataset,
    patterns: xr.DataArray,
    sensors: Sequence[str],
    basis_signature: str,
    grid_signature: str,
    epsilon: float,
    spec_hash: str | None,
    support_policy: SupportPolicy,
    fit_policy: Mapping[str, Any],
) -> MonthlyBiasFit:
    """Validate and load a sequential or joint immutable projection fit."""
    expected_method = _validate_common_artifact(
        artifact,
        patterns,
        sensors,
        basis_signature,
        grid_signature,
        epsilon,
        spec_hash,
        support_policy,
        fit_policy,
    )
    joint_values: dict[str, np.ndarray | bool | int | None] = {
        "perpendicular_bias": None,
        "mode_coefficient": None,
        "sensor_offset": None,
        "sensor_offset_standard_error": None,
        "sensor_overlap_count": None,
        "pooled_observable_rank": None,
        "pooled_observable_eigenvalue": None,
        "objective_history": None,
        "converged": None,
        "iterations": None,
    }
    if expected_method == "joint_seasonal":
        joint_values = _load_joint_values(artifact, sensors, fit_policy)
    return MonthlyBiasFit(
        raw_mean=np.asarray(artifact["clim_bias_raw_mean"].values, dtype=np.float64),
        bias=np.asarray(artifact["clim_bias"].values, dtype=np.float64),
        bias_applied=np.asarray(artifact["clim_bias_applied"].values, dtype=np.float64),
        support=np.asarray(artifact["spatial_support"].values, dtype=np.float64),
        support_fraction=np.asarray(artifact["support_fraction"].values, dtype=np.float64),
        support_count=np.asarray(artifact["support_count"].values, dtype=np.int64),
        support_day_total=np.asarray(artifact["support_day_total"].values, dtype=np.int64),
        sensor_count=np.asarray(artifact["clim_bias_sensor_count"].values, dtype=np.int64),
        standard_error=np.asarray(artifact["clim_bias_standard_error"].values, dtype=np.float64),
        perpendicular_bias=joint_values["perpendicular_bias"],  # type: ignore[arg-type]
        mode_coefficient=joint_values["mode_coefficient"],  # type: ignore[arg-type]
        sensor_offset=joint_values["sensor_offset"],  # type: ignore[arg-type]
        sensor_offset_standard_error=joint_values[
            "sensor_offset_standard_error"
        ],  # type: ignore[arg-type]
        sensor_overlap_count=joint_values["sensor_overlap_count"],  # type: ignore[arg-type]
        pooled_observable_rank=joint_values["pooled_observable_rank"],  # type: ignore[arg-type]
        pooled_observable_eigenvalue=joint_values[
            "pooled_observable_eigenvalue"
        ],  # type: ignore[arg-type]
        objective_history=joint_values["objective_history"],  # type: ignore[arg-type]
        converged=joint_values["converged"],  # type: ignore[arg-type]
        iterations=joint_values["iterations"],  # type: ignore[arg-type]
    )


__all__ = [
    "SupportPolicy",
    "digest_projection_fit_policy",
    "load_projection_fit_artifact",
    "projection_fit_policy",
    "projection_fit_window_signature",
    "select_bias_fit_window",
]
