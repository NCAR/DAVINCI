"""FABLE innovation projection analysis and xarray orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.analysis.base import (
    AnalysisResult,
    AnalysisRuntime,
    ArtifactDeclaration,
    DerivedAnalysis,
)
from davinci_monet.analysis.projection_batches import (
    fit_monthly_bias_batched,
    solve_projection_batched,
)
from davinci_monet.analysis.projection_core import MonthlyBiasFit
from davinci_monet.analysis.projection_inputs import (
    field_value,
    prepare_projection_inputs,
    same_coordinate,
)
from davinci_monet.analysis.provenance import consistent_spec_hash
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry


def _fit_mask(time: pd.DatetimeIndex, spec: Any) -> np.ndarray:
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


def _fit_from_artifact(
    artifact: xr.Dataset,
    patterns: xr.DataArray,
    sensors: Sequence[str],
    basis_signature: str,
    grid_signature: str,
    epsilon: float,
    spec_hash: str | None,
    support_policy: tuple[float, float, int, tuple[float, float]],
) -> MonthlyBiasFit:
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
    )


def project_eof(
    basis: xr.Dataset,
    model: xr.Dataset,
    observation_inputs: Sequence[tuple[Any, xr.Dataset]],
    spec: Any,
    *,
    bias_fit_artifact: xr.Dataset | None = None,
) -> xr.Dataset:
    """Project daily multi-sensor innovations onto an unrotated EOF basis."""
    spec_hash = consistent_spec_hash(
        [basis, model, *(dataset for _entry, dataset in observation_inputs)]
    )
    prepared = prepare_projection_inputs(
        basis,
        model,
        observation_inputs,
        str(field_value(spec, "model_variable", "log_aod")),
    )
    patterns_da = prepared.patterns
    model_da = prepared.model
    time = prepared.time
    epsilon = prepared.epsilon
    basis_signature = prepared.basis_signature
    grid_signature = prepared.grid_signature
    observations = prepared.observations
    names = [observation.name for observation in observations]
    if len(names) != len(set(names)):
        raise ValueError("eof_projection sensor source names must be unique")
    if field_value(spec, "spatial_support", "monthly_taper") != "monthly_taper":
        raise ValueError("eof_projection currently requires spatial_support='monthly_taper'")

    months = time.month.to_numpy(dtype=np.int64)
    support_minimum = float(field_value(spec, "support_min_fraction", 0.2))
    support_full = float(field_value(spec, "support_full_fraction", 0.5))
    smoothing_passes = int(field_value(spec, "support_smoothing_passes", 2))
    raw_delta_bounds = tuple(field_value(spec, "delta_bounds", (-1.6094379, 1.6094379)))
    delta_bounds = (float(raw_delta_bounds[0]), float(raw_delta_bounds[1]))
    time_chunk_size = int(field_value(spec, "time_chunk_size", 31))
    support_policy = (support_minimum, support_full, smoothing_passes, delta_bounds)
    if bias_fit_artifact is None:
        selected = _fit_mask(time, spec)
        fit = fit_monthly_bias_batched(
            model_da,
            observations,
            months,
            selected,
            support_min_fraction=support_minimum,
            support_full_fraction=support_full,
            smoothing_passes=smoothing_passes,
            delta_bounds=delta_bounds,
            time_chunk_size=time_chunk_size,
        )
        fit_selection = "window"
        fit_start = str(time[selected][0])
        fit_end = str(time[selected][-1])
    else:
        fit = _fit_from_artifact(
            bias_fit_artifact,
            patterns_da,
            names,
            basis_signature,
            grid_signature,
            epsilon,
            spec_hash,
            support_policy,
        )
        fit_selection = "artifact"
        fit_start = str(bias_fit_artifact.attrs.get("projection_bias_fit_start", ""))
        fit_end = str(bias_fit_artifact.attrs.get("projection_bias_fit_end", ""))
        if not fit_start or not fit_end:
            raise ValueError("bias fit artifact is missing fit-window provenance")

    apply_bias = bool(field_value(spec, "clim_bias", True))
    bias_applied = fit.support * fit.bias if apply_bias else np.zeros_like(fit.bias)
    ridge = float(field_value(spec, "ridge", 1.0))
    min_resolution = float(field_value(spec, "min_resolution", 0.3))
    pattern_values = np.asarray(patterns_da.values, dtype=np.float64)
    mode_count = pattern_values.shape[0]
    solved = solve_projection_batched(
        model_da,
        observations,
        pattern_values,
        fit,
        months,
        apply_bias=apply_bias,
        ridge=ridge,
        time_chunk_size=time_chunk_size,
    )

    coords = {
        "time": model_da["time"],
        "mode": patterns_da["mode"],
        "eigen": np.arange(1, mode_count + 1),
        "sensor": names,
        "month": np.arange(1, 13),
        "lat": model_da["lat"],
        "lon": model_da["lon"],
    }
    output = xr.Dataset(
        {
            "pc": (("time", "mode"), solved.coefficients),
            "resolution": (("time", "mode"), solved.resolution),
            "coverage": (("time", "mode"), solved.coverage),
            "posterior_variance": (("time", "mode"), solved.posterior_variance),
            "resolution_eigenvalue": (("time", "eigen"), solved.resolution_eigenvalue),
            "posterior_eigenvalue": (("time", "eigen"), solved.posterior_eigenvalue),
            "resolution_eigen_min": ("time", solved.resolution_eigenvalue.min(axis=1)),
            "resolution_eigen_max": ("time", solved.resolution_eigenvalue.max(axis=1)),
            "condition_number": ("time", solved.condition_number),
            "effective_rank": ("time", solved.effective_rank),
            "n_obs": (("time", "sensor"), solved.n_obs),
            "low_resolution": (("time", "mode"), solved.resolution < min_resolution),
            "innovation_mean": (("time", "lat", "lon"), solved.innovation_mean),
            "innovation_count": (("time", "lat", "lon"), solved.innovation_count),
            "clim_bias_raw_mean": (("month", "lat", "lon"), fit.raw_mean),
            "clim_bias": (("month", "lat", "lon"), fit.bias),
            "clim_bias_applied": (("month", "lat", "lon"), bias_applied),
            "spatial_support": (("month", "lat", "lon"), fit.support),
            "support_fraction": (("month", "lat", "lon"), fit.support_fraction),
            "support_count": (("month", "lat", "lon"), fit.support_count),
            "support_day_total": ("month", fit.support_day_total),
            "clim_bias_sensor_count": (
                ("month", "sensor", "lat", "lon"),
                fit.sensor_count,
            ),
            "clim_bias_standard_error": (
                ("month", "lat", "lon"),
                fit.standard_error,
            ),
        },
        coords=coords,
    )
    output["pc"].attrs.update(units="1", kind="pc")
    output["clim_bias"].attrs.update(units="1", space="shifted_log")
    output["spatial_support"].attrs.update(units="1", valid_range=[0.0, 1.0])
    output.attrs.update(
        analysis_type="eof_projection",
        projection_method="innovation_ridge_woodbury",
        projection_time_chunk_size=time_chunk_size,
        projection_basis_signature=basis_signature,
        projection_grid_signature=grid_signature,
        projection_log_epsilon=epsilon,
        projection_ridge=ridge,
        projection_min_resolution=min_resolution,
        projection_bias_fit_selection=fit_selection,
        projection_bias_fit_start=fit_start,
        projection_bias_fit_end=fit_end,
        projection_common_modes=",".join(observations[0].factor_names),
        projection_clim_bias=str(apply_bias).lower(),
        projection_support_min_fraction=support_minimum,
        projection_support_full_fraction=support_full,
        projection_support_smoothing_passes=smoothing_passes,
        projection_delta_lower=delta_bounds[0],
        projection_delta_upper=delta_bounds[1],
    )
    if spec_hash is not None:
        output.attrs["source_spec_hash"] = spec_hash
    return output


@analysis_registry.register("eof_projection")
class EOFProjectionAnalysis(DerivedAnalysis):
    """Named-input pipeline adapter for multi-sensor EOF projection."""

    name = "eof_projection"
    long_name = "EOF Innovation Projection"
    output_geometry = DataGeometry.GRID

    def analyze_inputs(
        self,
        inputs: Mapping[str, xr.Dataset],
        spec: Any,
        runtime: AnalysisRuntime,
    ) -> AnalysisResult:
        del runtime
        try:
            basis = inputs["basis"]
            model = inputs["model"]
        except KeyError as exc:
            raise ValueError("eof_projection requires named basis and model inputs") from exc
        configured = list(field_value(spec, "obs", []))
        observations: list[tuple[Any, xr.Dataset]] = []
        for index, entry in enumerate(configured):
            role = f"obs[{index}]"
            if role not in inputs:
                raise ValueError(f"eof_projection input {role!r} was not resolved")
            observations.append((entry, inputs[role]))
        artifact = inputs.get("bias_fit_artifact")
        dataset = project_eof(
            basis,
            model,
            observations,
            spec,
            bias_fit_artifact=artifact,
        )
        artifacts: tuple[ArtifactDeclaration, ...] = ()
        if artifact is None:
            artifacts = (
                ArtifactDeclaration(
                    kind="netcdf_collection",
                    role="projection_fit",
                    reload=True,
                    options={"time_chunk_size": 31},
                ),
            )
        return AnalysisResult(
            dataset=dataset,
            artifacts=artifacts,
        )


__all__ = ["EOFProjectionAnalysis", "project_eof"]
