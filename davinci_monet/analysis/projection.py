"""FABLE innovation projection analysis and xarray orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
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
from davinci_monet.analysis.projection_fit_artifact import (
    digest_projection_fit_policy,
    load_projection_fit_artifact,
    projection_fit_policy,
    projection_fit_window_signature,
    select_bias_fit_window,
)
from davinci_monet.analysis.projection_inputs import (
    field_value,
    prepare_projection_inputs,
)
from davinci_monet.analysis.projection_joint import fit_joint_projection_bias_batched
from davinci_monet.analysis.provenance import consistent_spec_hash
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry


def _fit_mask(time: Any, spec: Any) -> np.ndarray:
    return select_bias_fit_window(time, spec)


def _digest_payload(payload: Mapping[str, Any]) -> str:
    return digest_projection_fit_policy(payload)


def _fit_policy(
    spec: Any,
    sensor_names: Sequence[str],
    support_policy: tuple[float, float, int, tuple[float, float]],
) -> dict[str, Any]:
    return projection_fit_policy(spec, sensor_names, support_policy)


def _window_signature(time: Any, selected: np.ndarray) -> str:
    return projection_fit_window_signature(time, selected)


_fit_from_artifact = load_projection_fit_artifact


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
    fit_method = str(field_value(spec, "bias_fit_method", "monthly_mean"))
    sensor_offset_method = str(field_value(spec, "sensor_offset_method", "none"))
    ridge = float(field_value(spec, "ridge", 1.0))
    laplacian_strength = float(field_value(spec, "joint_bias_laplacian_strength", 1.0))
    joint_tolerance = float(field_value(spec, "joint_bias_tolerance", 1.0e-6))
    joint_max_iterations = int(field_value(spec, "joint_bias_max_iterations", 20))
    fit_policy = _fit_policy(spec, names, support_policy)
    fit_policy_signature = _digest_payload(fit_policy)
    if bias_fit_artifact is not None and field_value(spec, "bias_fit_window", None) is not None:
        raise ValueError("bias_fit_window and bias_fit_artifact are mutually exclusive")
    if bias_fit_artifact is None:
        selected = _fit_mask(time, spec)
        if fit_method == "monthly_mean":
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
        elif fit_method == "joint_seasonal":
            fit = fit_joint_projection_bias_batched(
                model_da,
                observations,
                np.asarray(patterns_da.values, dtype=np.float64),
                months,
                selected,
                support_min_fraction=support_minimum,
                support_full_fraction=support_full,
                smoothing_passes=smoothing_passes,
                delta_bounds=delta_bounds,
                ridge=ridge,
                sensor_offset_method=sensor_offset_method,
                laplacian_strength=laplacian_strength,
                tolerance=joint_tolerance,
                max_iterations=joint_max_iterations,
                time_chunk_size=time_chunk_size,
            )
        else:
            raise ValueError(f"unknown projection bias fit method {fit_method!r}")
        fit_selection = "window"
        fit_start = str(time[selected][0])
        fit_end = str(time[selected][-1])
        fit_window_signature = _window_signature(time, selected)
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
            fit_policy,
        )
        fit_selection = "artifact"
        fit_start = str(bias_fit_artifact.attrs.get("projection_bias_fit_start", ""))
        fit_end = str(bias_fit_artifact.attrs.get("projection_bias_fit_end", ""))
        if not fit_start or not fit_end:
            raise ValueError("bias fit artifact is missing fit-window provenance")
        fit_window_signature = str(
            bias_fit_artifact.attrs.get("projection_bias_fit_window_signature", "")
        )

    apply_bias = bool(field_value(spec, "clim_bias", True))
    bias_applied = fit.support * fit.bias if apply_bias else np.zeros_like(fit.bias)
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
        sensor_offsets=fit.sensor_offset,
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
    if fit_method == "joint_seasonal":
        joint_fields = (
            fit.perpendicular_bias,
            fit.mode_coefficient,
            fit.sensor_offset,
            fit.sensor_offset_standard_error,
            fit.sensor_overlap_count,
            fit.pooled_observable_rank,
            fit.pooled_observable_eigenvalue,
            fit.objective_history,
        )
        if any(value is None for value in joint_fields):
            raise ValueError("joint seasonal bias fit did not produce complete diagnostics")
        output = output.assign_coords(
            sensor_pair=("sensor_pair", names),
            joint_iteration=(
                "joint_iteration",
                np.arange(len(fit.objective_history), dtype=np.int64),  # type: ignore[arg-type]
            ),
        )
        output["clim_bias_perpendicular"] = (
            ("month", "lat", "lon"),
            fit.perpendicular_bias,
        )
        output["clim_bias_mode_coefficient"] = (
            ("month", "mode"),
            fit.mode_coefficient,
        )
        output["sensor_offset"] = (("sensor",), fit.sensor_offset)
        output["sensor_offset_standard_error"] = (
            ("sensor",),
            fit.sensor_offset_standard_error,
        )
        output["sensor_overlap_count"] = (
            ("sensor", "sensor_pair"),
            fit.sensor_overlap_count,
        )
        output["pooled_observable_rank"] = (
            ("month",),
            fit.pooled_observable_rank,
        )
        output["pooled_observable_eigenvalue"] = (
            ("month", "eigen"),
            fit.pooled_observable_eigenvalue,
        )
        output["joint_objective"] = (
            ("joint_iteration",),
            fit.objective_history,
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
        projection_bias_fit_window_signature=fit_window_signature,
        projection_bias_fit_method=fit_method,
        projection_sensor_offset_method=sensor_offset_method,
        projection_bias_fit_policy_signature=fit_policy_signature,
        projection_joint_bias_laplacian_strength=laplacian_strength,
        projection_joint_bias_tolerance=joint_tolerance,
        projection_joint_bias_max_iterations=joint_max_iterations,
        projection_common_modes=",".join(observations[0].factor_names),
        projection_clim_bias=str(apply_bias).lower(),
        projection_support_min_fraction=support_minimum,
        projection_support_full_fraction=support_full,
        projection_support_smoothing_passes=smoothing_passes,
        projection_delta_lower=delta_bounds[0],
        projection_delta_upper=delta_bounds[1],
    )
    if fit_method == "joint_seasonal":
        output.attrs.update(
            projection_joint_bias_converged=str(bool(fit.converged)).lower(),
            projection_joint_bias_iterations=int(fit.iterations or 0),
            projection_absolute_sensor_offset_identifiable="false",
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
