"""Pure, evaluation-only diagnostics for the FABLE v2 synthetic cycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import xarray as xr

from davinci_monet.analysis.known_truth_metrics import (
    FieldMetrics,
    align_fields,
    canonical_dimensions,
    safe_ratio,
    weighted_field_metrics,
)

V2_STAGE_NAMES = (
    "learned_basis_oracle",
    "oracle_bias_noiseless_projection",
    "oracle_bias_noisy_projection",
    "fitted_bias_oracle_coefficients",
    "fitted_bias_unfiltered_projection",
    "post_wavelet",
    "final_policy",
)

_DIAGNOSTIC_PROVENANCE = {
    "diagnostic_only": "true",
    "eligible_for_calibration": "false",
    "calibration_use": "prohibited",
    "oracle_truth_used": "true",
}


def _require_target(field: xr.DataArray, description: str) -> xr.DataArray:
    result = canonical_dimensions(field)
    if result.dims != ("time", "lat", "lon"):
        raise ValueError(f"{description} must have exactly ('time', 'lat', 'lon') dimensions")
    return result


def _require_basis(basis: xr.DataArray) -> xr.DataArray:
    result = canonical_dimensions(basis)
    if result.dims != ("mode", "lat", "lon"):
        raise ValueError("learned basis must have exactly ('mode', 'lat', 'lon') dimensions")
    if result.sizes["mode"] == 0:
        raise ValueError("learned basis must contain at least one mode")
    return result


def _broadcast_field(
    field: xr.DataArray | None,
    template: xr.DataArray,
    description: str,
    *,
    default: float,
) -> xr.DataArray:
    if field is None:
        return xr.full_like(template, default, dtype=np.float64)
    value = canonical_dimensions(field)
    unexpected = set(value.dims).difference(template.dims)
    if unexpected:
        raise ValueError(
            f"{description} has incompatible dimensions: {sorted(map(str, unexpected))}"
        )
    try:
        value, _ = xr.align(value, template, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(f"{description} coordinates must match the target exactly") from exc
    value, _ = xr.broadcast(value, template)
    return value.transpose(*template.dims)


def reconstruct_stage_field(
    learned_basis: xr.DataArray,
    coefficients: xr.DataArray,
    *,
    fixed_component: xr.DataArray | None = None,
    spatial_multiplier: xr.DataArray | None = None,
) -> xr.DataArray:
    """Reconstruct one diagnostic stage from a fixed field and learned EOF coefficients."""
    basis = _require_basis(learned_basis)
    pc = canonical_dimensions(coefficients)
    if pc.dims != ("time", "mode"):
        raise ValueError("coefficients must have exactly ('time', 'mode') dimensions")
    try:
        pc, basis = xr.align(pc, basis, join="exact", copy=False, exclude={"time", "lat", "lon"})
    except ValueError as exc:
        raise ValueError("coefficient modes must match the learned basis exactly") from exc

    reconstructed = xr.dot(pc, basis, dim="mode").transpose("time", "lat", "lon")
    multiplier = _broadcast_field(
        spatial_multiplier,
        reconstructed,
        "spatial multiplier",
        default=1.0,
    )
    fixed = _broadcast_field(
        fixed_component,
        reconstructed,
        "fixed component",
        default=0.0,
    )
    result = fixed + multiplier * reconstructed
    result.name = "diagnostic_stage_delta"
    result.attrs.update(_DIAGNOSTIC_PROVENANCE)
    return result


def learned_basis_filtered_target_oracle(
    filtered_target: xr.DataArray,
    learned_basis: xr.DataArray,
    *,
    oracle_bias: xr.DataArray | None = None,
    spatial_multiplier: xr.DataArray | None = None,
    score_mask: xr.DataArray | None = None,
) -> xr.Dataset:
    """Project a filtered target onto learned EOFs with daily full-grid weighted least squares.

    ``oracle_bias`` is the already policy-matched true bias contribution. The optional spatial
    multiplier applies the same support/policy taper to each learned EOF before fitting and
    reconstruction. Neither field is inferred from the target by this diagnostic.
    """
    target = _require_target(filtered_target, "filtered target")
    basis = _require_basis(learned_basis)
    try:
        basis, _ = xr.align(basis, target, join="exact", copy=False, exclude={"mode", "time"})
    except ValueError as exc:
        raise ValueError("learned basis grid must match the filtered target exactly") from exc
    fixed = _broadcast_field(oracle_bias, target, "oracle bias", default=0.0)
    multiplier = _broadcast_field(
        spatial_multiplier,
        target,
        "spatial multiplier",
        default=1.0,
    )
    target_values = np.asarray(target.values, dtype=np.float64)
    fixed_values = np.asarray(fixed.values, dtype=np.float64)
    multiplier_values = np.asarray(multiplier.values, dtype=np.float64)
    basis_values = np.asarray(basis.values, dtype=np.float64)
    latitude_weight = np.broadcast_to(
        np.clip(
            np.cos(np.deg2rad(np.asarray(target["lat"].values, dtype=np.float64))),
            0.0,
            None,
        )[:, None],
        (target.sizes["lat"], target.sizes["lon"]),
    )

    coefficient_values = np.empty((target.sizes["time"], basis.sizes["mode"]), dtype=np.float64)
    ranks = np.empty(target.sizes["time"], dtype=np.int64)
    for index in range(target.sizes["time"]):
        design = multiplier_values[index, None, ...] * basis_values
        residual = target_values[index] - fixed_values[index]
        valid = (
            np.isfinite(residual)
            & np.isfinite(design).all(axis=0)
            & np.isfinite(latitude_weight)
            & (latitude_weight > 0.0)
        )
        sqrt_weight = np.sqrt(latitude_weight[valid])
        weighted_design = design[:, valid].T * sqrt_weight[:, None]
        weighted_target = residual[valid] * sqrt_weight
        if weighted_design.shape[0] < weighted_design.shape[1]:
            raise ValueError(
                "learned-basis oracle has fewer valid cells than modes " f"at time index {index}"
            )
        rank = int(np.linalg.matrix_rank(weighted_design))
        ranks[index] = rank
        if rank != basis.sizes["mode"]:
            raise ValueError(
                "learned-basis oracle requires full column rank "
                f"at time index {index}; got {rank} of {basis.sizes['mode']}"
            )
        coefficient_values[index] = np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )[0]

    coefficients = xr.DataArray(
        coefficient_values,
        dims=("time", "mode"),
        coords={"time": target["time"], "mode": basis["mode"]},
        name="learned_basis_oracle_pc",
    )
    reconstruction = reconstruct_stage_field(
        basis,
        coefficients,
        fixed_component=fixed,
        spatial_multiplier=multiplier,
    ).rename("learned_basis_oracle_delta")
    metrics = weighted_field_metrics(reconstruction, target, score_mask)
    output = xr.Dataset(
        {
            "learned_basis_oracle_pc": coefficients,
            "learned_basis_oracle_delta": reconstruction,
            "learned_basis_oracle_fit_rank": (
                "time",
                ranks,
            ),
            **_scalar_metric_variables("learned_basis_oracle", metrics),
        }
    )
    output.attrs.update(
        {
            "analysis_type": "fable_v2_learned_basis_oracle",
            "geometry": "artifact",
            "evaluation_only": "true",
            "field_weighting": "equal_day_cosine_latitude",
            "fit_method": "daily_full_grid_cosine_latitude_weighted_least_squares",
            "oracle_bias_supplied": str(oracle_bias is not None).lower(),
            "spatial_multiplier_supplied": str(spatial_multiplier is not None).lower(),
            "score_mask_supplied": str(score_mask is not None).lower(),
            **_DIAGNOSTIC_PROVENANCE,
        }
    )
    return output


def score_stagewise_fields(
    stages: Mapping[str, xr.DataArray],
    filtered_target: xr.DataArray,
    mask: xr.DataArray | None = None,
) -> xr.Dataset:
    """Score ordered stage fields on one common finite mask.

    Changes between stages are descriptive and non-additive. They are not causal error
    attribution because errors from adjacent stages can covary.
    """
    if not stages:
        raise ValueError("stagewise diagnostics require at least one stage")
    target = _require_target(filtered_target, "filtered target")
    aligned_stages: list[xr.DataArray] = []
    common = cast(xr.DataArray, np.isfinite(target))
    if mask is not None:
        common = common & _broadcast_field(mask, target, "stagewise mask", default=1.0).astype(bool)
    for name, stage in stages.items():
        if not name:
            raise ValueError("stage names must be nonempty")
        aligned, _, _ = align_fields(stage, target, None)
        aligned_stages.append(aligned)
        common = common & cast(xr.DataArray, np.isfinite(aligned))
    common_count = common.sum()
    if common_count.chunks is not None:
        common_count = common_count.compute()
    if int(common_count.item()) == 0:
        raise ValueError("stagewise diagnostic mask selects no common finite values")

    metrics = [weighted_field_metrics(stage, target, common) for stage in aligned_stages]
    truth_rms = metrics[0].truth_rms
    nrmse_change = [np.nan]
    transition_rmse = [np.nan]
    transition_nrmse = [np.nan]
    for previous, current, previous_metrics, current_metrics in zip(
        aligned_stages[:-1],
        aligned_stages[1:],
        metrics[:-1],
        metrics[1:],
        strict=True,
    ):
        difference = current - previous
        magnitude = weighted_field_metrics(difference, xr.zeros_like(difference), common).rmse
        nrmse_change.append(current_metrics.nrmse - previous_metrics.nrmse)
        transition_rmse.append(magnitude)
        transition_nrmse.append(safe_ratio(magnitude, truth_rms))

    names = list(stages)
    output = xr.Dataset(
        {
            "stage_field_correlation": (
                "stage",
                [value.correlation for value in metrics],
            ),
            "stage_field_origin_slope": (
                "stage",
                [value.origin_slope for value in metrics],
            ),
            "stage_field_bias": ("stage", [value.bias for value in metrics]),
            "stage_field_rmse": ("stage", [value.rmse for value in metrics]),
            "stage_field_truth_rms": (
                "stage",
                [value.truth_rms for value in metrics],
            ),
            "stage_field_nrmse": ("stage", [value.nrmse for value in metrics]),
            "stage_valid_count": ("stage", [value.valid_count for value in metrics]),
            "stage_candidate_count": (
                "stage",
                [value.candidate_count for value in metrics],
            ),
            "stage_excluded_fraction": (
                "stage",
                [value.excluded_fraction for value in metrics],
            ),
            "stage_nrmse_change_from_previous": ("stage", nrmse_change),
            "stage_transition_rmse": ("stage", transition_rmse),
            "stage_transition_nrmse": ("stage", transition_nrmse),
        },
        coords={"stage": names},
    )
    output["stage_nrmse_change_from_previous"].attrs["additivity"] = "non_additive"
    output["stage_transition_nrmse"].attrs["additivity"] = "non_additive"
    output.attrs.update(
        {
            "analysis_type": "fable_v2_stagewise_field_scores",
            "geometry": "artifact",
            "evaluation_only": "true",
            "field_weighting": "equal_day_cosine_latitude",
            "stage_mask_policy": "intersection_of_requested_mask_and_all_finite_stage_values",
            "stage_increment_interpretation": "descriptive_non_additive_not_causal_attribution",
            **_DIAGNOSTIC_PROVENANCE,
        }
    )
    return output


def build_fable_v2_diagnostic_report(
    stages: Mapping[str, xr.DataArray],
    filtered_target: xr.DataArray,
    *,
    unfiltered_in_span: xr.DataArray,
    mask: xr.DataArray | None = None,
) -> xr.Dataset:
    """Build the fixed seven-stage v2 report with corrected metric semantics."""
    supplied = tuple(stages)
    if supplied != V2_STAGE_NAMES:
        raise ValueError(
            "v2 diagnostic stages must match the preregistered order exactly; "
            f"expected {V2_STAGE_NAMES}, got {supplied}"
        )
    target = _require_target(filtered_target, "filtered target")
    output = score_stagewise_fields(stages, target, mask)
    comparator, _, _ = align_fields(unfiltered_in_span, target, None)
    comparison_mask = cast(xr.DataArray, np.isfinite(target)) & cast(
        xr.DataArray, np.isfinite(comparator)
    )
    if mask is not None:
        comparison_mask = comparison_mask & _broadcast_field(
            mask, target, "stagewise mask", default=1.0
        ).astype(bool)
    aligned_stages: dict[str, xr.DataArray] = {}
    for name, stage in stages.items():
        aligned, _, _ = align_fields(stage, target, None)
        aligned_stages[name] = aligned
        comparison_mask = comparison_mask & cast(xr.DataArray, np.isfinite(aligned))
    comparison = weighted_field_metrics(
        aligned_stages["final_policy"],
        comparator,
        comparison_mask,
    )
    output["learned_basis_oracle_nrmse"] = output["stage_field_nrmse"].sel(
        stage="learned_basis_oracle",
        drop=True,
    )
    output["estimate_vs_unfiltered_in_span_nrmse"] = comparison.nrmse
    output["estimate_vs_unfiltered_in_span_rmse"] = comparison.rmse
    output["estimate_vs_unfiltered_in_span_truth_rms"] = comparison.truth_rms
    output.attrs.update(
        {
            "analysis_type": "fable_v2_stage_diagnostics",
            "geometry": "artifact",
            "evaluation_only": "true",
            "metric_semantics_version": "fable-recovery-v2",
            "unfiltered_comparison_interpretation": (
                "estimate comparison only; not an optimum, ceiling, or acceptance gate"
            ),
            **_DIAGNOSTIC_PROVENANCE,
        }
    )
    return output


def _scalar_metric_variables(prefix: str, metrics: FieldMetrics) -> dict[str, xr.DataArray]:
    values: dict[str, float | int] = {
        f"{prefix}_correlation": metrics.correlation,
        f"{prefix}_origin_slope": metrics.origin_slope,
        f"{prefix}_bias": metrics.bias,
        f"{prefix}_rmse": metrics.rmse,
        f"{prefix}_truth_rms": metrics.truth_rms,
        f"{prefix}_nrmse": metrics.nrmse,
        f"{prefix}_valid_count": metrics.valid_count,
        f"{prefix}_candidate_count": metrics.candidate_count,
        f"{prefix}_excluded_fraction": metrics.excluded_fraction,
    }
    return {name: xr.DataArray(value) for name, value in values.items()}


__all__ = [
    "V2_STAGE_NAMES",
    "build_fable_v2_diagnostic_report",
    "learned_basis_filtered_target_oracle",
    "reconstruct_stage_field",
    "score_stagewise_fields",
]
