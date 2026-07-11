"""Matched production projection solve for FABLE v2 oracle diagnostics."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.analysis.known_truth_metrics import canonical_dimensions
from davinci_monet.analysis.projection_core import build_effective_covariance, solve_one_day


def _require(dataset: xr.Dataset, name: str) -> xr.DataArray:
    if name not in dataset:
        raise ValueError(f"fable_v2_diagnostics truth is missing {name!r}")
    return canonical_dimensions(dataset[name])


def masked_projection_coefficients(
    target: xr.DataArray,
    basis: xr.DataArray,
    oracle_bias: xr.DataArray,
    support: xr.DataArray,
    truth: xr.Dataset,
    *,
    noisy: bool,
    common_factor_amplitude: float,
    sensor_offsets: xr.DataArray | None = None,
    ridge: float = 1.0,
) -> xr.DataArray:
    """Repeat the production daily ridge solve with oracle-selected observations."""
    valid = _require(truth, "valid_mask")
    sigma = _require(truth, "reported_sigma_log")
    error = _require(truth, "obs_error_log")
    expected = ("sensor", "time", "lat", "lon")
    if valid.dims != expected or sigma.dims != expected or error.dims != expected:
        raise ValueError(
            "v2 masked projection truth arrays must have sensor/time/lat/lon dimensions"
        )
    try:
        valid, sigma, error, target, oracle_bias, support, basis = xr.align(
            valid,
            sigma,
            error,
            target,
            oracle_bias,
            support,
            basis,
            join="exact",
            copy=False,
            exclude={"sensor", "mode"},
        )
    except ValueError as exc:
        raise ValueError("v2 masked projection inputs must have exact coordinates") from exc

    basis_values = np.asarray(basis.values, dtype=np.float64)
    support_values = np.asarray(support.values, dtype=np.float64)
    applied_residual = np.asarray(target.values - oracle_bias.values, dtype=np.float64)
    # Production solves the untapered innovation and applies support only in reconstruction.
    residual = np.divide(
        applied_residual,
        support_values,
        out=np.zeros_like(applied_residual),
        where=np.isfinite(support_values) & (support_values > 0.0),
    )
    valid_values = np.asarray(valid.values, dtype=bool)
    sigma_values = np.asarray(sigma.values, dtype=np.float64)
    error_values = np.asarray(error.values, dtype=np.float64)
    if not np.isfinite(common_factor_amplitude) or common_factor_amplitude < 0.0:
        raise ValueError("v2 reported common-factor amplitude must be finite and nonnegative")
    latitude = np.broadcast_to(
        np.asarray(target["lat"].values, dtype=np.float64)[:, None],
        (target.sizes["lat"], target.sizes["lon"]),
    )
    offset_values = np.zeros(valid.sizes["sensor"], dtype=np.float64)
    if sensor_offsets is not None:
        offset = canonical_dimensions(sensor_offsets)
        if offset.dims != ("sensor",):
            raise ValueError("v2 fitted sensor offsets must have the ('sensor',) dimension")
        try:
            offset, _ = xr.align(offset, valid["sensor"], join="exact", copy=False)
        except ValueError as exc:
            raise ValueError("v2 fitted sensor offsets must match truth sensors exactly") from exc
        offset_values = np.asarray(offset.values, dtype=np.float64)

    coefficients = np.zeros((target.sizes["time"], basis.sizes["mode"]), dtype=np.float64)
    for day in range(target.sizes["time"]):
        rows: list[np.ndarray] = []
        values: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        latitudes: list[np.ndarray] = []
        design = basis_values
        for sensor in range(valid.sizes["sensor"]):
            usable = (
                valid_values[sensor, day]
                & np.isfinite(sigma_values[sensor, day])
                & (sigma_values[sensor, day] > 0.0)
                & np.isfinite(residual[day])
                & np.isfinite(design).all(axis=0)
                & (support_values[day] > 0.0)
            )
            if not np.any(usable):
                continue
            rows.append(design[:, usable].T)
            observed = residual[day, usable].copy()
            if noisy:
                observed += error_values[sensor, day, usable] - offset_values[sensor]
            values.append(observed)
            errors.append(sigma_values[sensor, day, usable])
            latitudes.append(latitude[usable])
        if not rows:
            continue
        matrix = np.concatenate(rows)
        vector = np.concatenate(values)
        covariance = build_effective_covariance(
            np.concatenate(errors),
            np.concatenate(latitudes),
            np.full((vector.size, 1), common_factor_amplitude, dtype=np.float64),
        )
        coefficients[day] = solve_one_day(matrix, vector, covariance, ridge).coefficients
    return xr.DataArray(
        coefficients,
        dims=("time", "mode"),
        coords={"time": target["time"], "mode": basis["mode"]},
        name="diagnostic_projection_pc",
    )


__all__ = ["masked_projection_coefficients"]
