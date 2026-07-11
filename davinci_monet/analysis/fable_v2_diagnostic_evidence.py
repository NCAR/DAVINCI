"""Pure supplemental evidence for the evaluation-only FABLE v2 report."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import xarray as xr

from davinci_monet.analysis.known_truth_metrics import (
    FieldMetrics,
    align_fields,
    canonical_dimensions,
    match_weighted_modes,
    safe_ratio,
    weighted_field_metrics,
)

COEFFICIENT_STAGE_NAMES = ("unfiltered_projection", "post_wavelet")

_PROVENANCE = {
    "diagnostic_only": "true",
    "eligible_for_calibration": "false",
    "calibration_use": "prohibited",
    "oracle_truth_used": "true",
}


def _metric_variables(prefix: str, metrics: FieldMetrics) -> dict[str, xr.DataArray]:
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


def _diagnostic_field(value: xr.DataArray, name: str, long_name: str) -> xr.DataArray:
    result = value.rename(name)
    result.attrs.update(long_name=long_name, units="1", **_PROVENANCE)
    return result


def common_bias_evidence(
    fitted_bias: xr.DataArray,
    true_bias: xr.DataArray,
    mask: xr.DataArray | None = None,
) -> xr.Dataset:
    """Report fitted and physical true common-bias fields on a matched mask."""
    fitted, truth, selected = align_fields(fitted_bias, true_bias, mask)
    if fitted.dims != ("time", "lat", "lon"):
        raise ValueError("common-bias fields must have time/lat/lon dimensions")
    metrics = weighted_field_metrics(fitted, truth, selected)
    output = xr.Dataset(
        {
            "fitted_common_bias": _diagnostic_field(
                fitted,
                "fitted_common_bias",
                "Fitted applied common shifted-log AOD bias",
            ),
            "true_common_bias": _diagnostic_field(
                truth,
                "true_common_bias",
                "Physical true applied common shifted-log AOD bias",
            ),
            "common_bias_error": _diagnostic_field(
                fitted - truth,
                "common_bias_error",
                "Fitted minus physical true common shifted-log AOD bias",
            ),
            **_metric_variables("common_bias", metrics),
        }
    )
    output.attrs.update(
        analysis_type="fable_v2_common_bias_diagnostics",
        common_bias_truth_reference="physical_common_correction_bias_after_true_support",
        common_bias_gauge_caveat=(
            "fitted common bias may absorb the unidentifiable mean absolute sensor offset"
        ),
        field_weighting="equal_day_cosine_latitude",
        **_PROVENANCE,
    )
    return output


def align_sensor_offset_coordinates(
    fitted_offset: xr.DataArray,
    true_offset: xr.DataArray,
    projection_to_truth_sensor: Mapping[str, str],
    *,
    standard_error: xr.DataArray | None = None,
    overlap_count: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None, xr.DataArray | None]:
    """Apply one explicit bijection from projection sensor labels to oracle labels."""
    fitted = canonical_dimensions(fitted_offset)
    truth = canonical_dimensions(true_offset)
    if fitted.dims != ("sensor",) or truth.dims != ("sensor",):
        raise ValueError("sensor offsets must have exactly the ('sensor',) dimension")
    projection_names = [str(value) for value in fitted["sensor"].values]
    truth_names = [str(value) for value in truth["sensor"].values]
    mapping = {str(source): str(target) for source, target in projection_to_truth_sensor.items()}
    if projection_names == truth_names and not mapping:
        return fitted, standard_error, overlap_count
    if set(mapping) != set(projection_names):
        raise ValueError(
            "projection_to_truth_sensor keys must exactly cover projection sensor coordinates"
        )
    if len(set(mapping.values())) != len(mapping) or set(mapping.values()) != set(truth_names):
        raise ValueError(
            "projection_to_truth_sensor values must bijectively cover truth sensor coordinates"
        )

    mapped_names = [mapping[name] for name in projection_names]
    aligned_fitted = fitted.assign_coords(sensor=mapped_names).sel(sensor=truth_names)
    aligned_error = None
    if standard_error is not None:
        error = canonical_dimensions(standard_error)
        if error.dims != ("sensor",):
            raise ValueError("sensor-offset standard error must have the ('sensor',) dimension")
        if [str(value) for value in error["sensor"].values] != projection_names:
            raise ValueError("sensor-offset standard error uses different projection sensors")
        aligned_error = error.assign_coords(sensor=mapped_names).sel(sensor=truth_names)
    aligned_overlap = None
    if overlap_count is not None:
        overlap = canonical_dimensions(overlap_count)
        if "sensor" not in overlap.dims or "sensor_pair" not in overlap.dims:
            raise ValueError("sensor overlap count must have sensor and sensor_pair dimensions")
        overlap_sensors = [str(value) for value in overlap["sensor"].values]
        overlap_pairs = [str(value) for value in overlap["sensor_pair"].values]
        if overlap_sensors != projection_names or set(overlap_pairs) != set(projection_names):
            raise ValueError("sensor overlap count uses different projection sensors")
        mapped_pairs = [mapping[name] for name in overlap_pairs]
        aligned_overlap = overlap.assign_coords(
            sensor=mapped_names,
            sensor_pair=mapped_pairs,
        ).sel(sensor=truth_names, sensor_pair=truth_names)
    return aligned_fitted, aligned_error, aligned_overlap


def relative_sensor_offset_evidence(
    fitted_offset: xr.DataArray,
    true_offset: xr.DataArray,
    *,
    standard_error: xr.DataArray | None = None,
    overlap_count: xr.DataArray | None = None,
) -> xr.Dataset:
    """Score only identifiable zero-mean sensor-offset contrasts."""
    fitted = canonical_dimensions(fitted_offset)
    truth = canonical_dimensions(true_offset)
    if fitted.dims != ("sensor",) or truth.dims != ("sensor",):
        raise ValueError("sensor offsets must have exactly the ('sensor',) dimension")
    try:
        fitted, truth = xr.align(fitted, truth, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("fitted and true sensor coordinates must match exactly") from exc
    fitted_values = np.asarray(fitted.values, dtype=np.float64)
    truth_values = np.asarray(truth.values, dtype=np.float64)
    if not np.all(np.isfinite(fitted_values)) or not np.all(np.isfinite(truth_values)):
        raise ValueError("sensor offsets must be finite")
    fitted_relative = fitted_values - fitted_values.mean()
    truth_relative = truth_values - truth_values.mean()
    error = fitted_relative - truth_relative
    cross = float(fitted_relative @ truth_relative)
    fitted_energy = float(fitted_relative @ fitted_relative)
    truth_energy = float(truth_relative @ truth_relative)
    rmse = float(np.sqrt(np.mean(np.square(error))))
    truth_rms = float(np.sqrt(np.mean(np.square(truth_relative))))
    output = xr.Dataset(
        {
            "fitted_relative_sensor_offset": (
                "sensor",
                fitted_relative,
            ),
            "true_relative_sensor_offset": (
                "sensor",
                truth_relative,
            ),
            "relative_sensor_offset_error": ("sensor", error),
            "true_absolute_sensor_offset_mean": xr.DataArray(float(truth_values.mean())),
            "relative_sensor_offset_correlation": xr.DataArray(
                safe_ratio(cross, np.sqrt(fitted_energy * truth_energy))
            ),
            "relative_sensor_offset_origin_slope": xr.DataArray(safe_ratio(cross, truth_energy)),
            "relative_sensor_offset_rmse": xr.DataArray(rmse),
            "relative_sensor_offset_truth_rms": xr.DataArray(truth_rms),
            "relative_sensor_offset_nrmse": xr.DataArray(safe_ratio(rmse, truth_rms)),
        },
        coords={"sensor": fitted["sensor"]},
    )
    for name in (
        "fitted_relative_sensor_offset",
        "true_relative_sensor_offset",
        "relative_sensor_offset_error",
    ):
        output[name].attrs.update(units="1", **_PROVENANCE)
    if standard_error is not None:
        uncertainty = canonical_dimensions(standard_error)
        if uncertainty.dims != ("sensor",):
            raise ValueError("sensor-offset standard error must have the ('sensor',) dimension")
        try:
            uncertainty, _ = xr.align(uncertainty, fitted, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError("sensor-offset uncertainty coordinates must match exactly") from exc
        output["relative_sensor_offset_standard_error"] = uncertainty
    if overlap_count is not None:
        overlap = canonical_dimensions(overlap_count)
        if "sensor" not in overlap.dims:
            raise ValueError("sensor overlap count must have a sensor dimension")
        try:
            overlap, _ = xr.align(
                overlap,
                fitted,
                join="exact",
                copy=False,
                exclude=set(overlap.dims).difference({"sensor"}),
            )
        except ValueError as exc:
            raise ValueError("sensor overlap coordinates must match exactly") from exc
        output["sensor_offset_overlap_count"] = overlap
    output.attrs.update(
        analysis_type="fable_v2_relative_sensor_offset_diagnostics",
        sensor_offset_gauge="zero_sensor_mean",
        absolute_sensor_offset_identifiable="false",
        absolute_sensor_offset_fit_status="not_scored_scientifically_unidentifiable",
        **_PROVENANCE,
    )
    return output


def _temporal_mask(mask: xr.DataArray, time: xr.DataArray) -> xr.DataArray:
    selected = canonical_dimensions(mask).astype(bool)
    for dim in list(selected.dims):
        if dim != "time":
            selected = selected.any(str(dim))
    if selected.dims != ("time",):
        raise ValueError("coefficient mask must contain a time dimension")
    try:
        selected, _ = xr.align(selected, time, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("coefficient mask time must match coefficients exactly") from exc
    return selected


def _coefficient_pair_metrics(
    estimate: xr.DataArray,
    truth: xr.DataArray,
    mask: xr.DataArray,
) -> FieldMetrics:
    latitude = xr.DataArray([0.0], dims=("lat",), coords={"lat": [0.0]})
    left, _ = xr.broadcast(estimate, latitude)
    right, _ = xr.broadcast(truth, latitude)
    selected, _ = xr.broadcast(mask, latitude)
    return weighted_field_metrics(
        left.transpose("time", "lat"),
        right.transpose("time", "lat"),
        selected.transpose("time", "lat"),
    )


def coefficient_stage_evidence(
    learned_basis: xr.DataArray,
    truth_basis: xr.DataArray,
    estimate_stages: Mapping[str, xr.DataArray],
    truth_stages: Mapping[str, xr.DataArray],
    temporal_mask: xr.DataArray,
    *,
    observable_modes: xr.DataArray | None = None,
) -> xr.Dataset:
    """Score matched learned/true coefficients before and after filtering."""
    if tuple(estimate_stages) != COEFFICIENT_STAGE_NAMES:
        raise ValueError("estimate coefficient stages do not match the preregistered order")
    if tuple(truth_stages) != COEFFICIENT_STAGE_NAMES:
        raise ValueError("truth coefficient stages do not match the preregistered order")
    learned = canonical_dimensions(learned_basis).transpose("mode", "lat", "lon")
    truth_patterns = canonical_dimensions(truth_basis).transpose("mode", "lat", "lon")
    matches = match_weighted_modes(learned, truth_patterns)
    observable = np.ones(truth_patterns.sizes["mode"], dtype=bool)
    if observable_modes is not None:
        values = canonical_dimensions(observable_modes)
        if values.dims != ("mode",):
            raise ValueError("observable modes must have exactly the ('mode',) dimension")
        try:
            values, _ = xr.align(values, truth_patterns["mode"], join="exact", copy=False)
        except ValueError as exc:
            raise ValueError("observable mode coordinates must match the truth basis") from exc
        observable = np.asarray(values.values, dtype=bool)

    stage_metrics: list[list[FieldMetrics | None]] = []
    for stage in COEFFICIENT_STAGE_NAMES:
        estimate_pc = canonical_dimensions(estimate_stages[stage]).transpose("time", "mode")
        truth_pc = canonical_dimensions(truth_stages[stage]).transpose("time", "mode")
        try:
            estimate_pc, _ = xr.align(estimate_pc, learned["mode"], join="exact", copy=False)
            truth_pc, _ = xr.align(truth_pc, truth_patterns["mode"], join="exact", copy=False)
            estimate_pc, truth_pc = xr.align(
                estimate_pc,
                truth_pc,
                join="exact",
                copy=False,
                exclude={"mode"},
            )
        except ValueError as exc:
            raise ValueError("coefficient stage coordinates do not match their bases") from exc
        selected = _temporal_mask(temporal_mask, estimate_pc["time"])
        row: list[FieldMetrics | None] = []
        for match in matches:
            if not observable[match.truth_index]:
                row.append(None)
                continue
            estimate_mode = match.scale * estimate_pc.isel(mode=match.estimate_index)
            truth_mode = truth_pc.isel(mode=match.truth_index)
            row.append(_coefficient_pair_metrics(estimate_mode, truth_mode, selected))
        stage_metrics.append(row)

    def metric_values(attribute: str, missing: float | int) -> list[list[float | int]]:
        return [
            [getattr(metric, attribute) if metric is not None else missing for metric in row]
            for row in stage_metrics
        ]

    output = xr.Dataset(
        {
            "estimate_mode_index": (
                "matched_mode",
                [value.estimate_index for value in matches],
            ),
            "truth_mode_index": ("matched_mode", [value.truth_index for value in matches]),
            "mode_sign": ("matched_mode", [value.sign for value in matches]),
            "basis_mode_similarity": (
                "matched_mode",
                [value.similarity for value in matches],
            ),
            "basis_mode_scale_to_truth": (
                "matched_mode",
                [value.scale for value in matches],
            ),
            "matched_mode_observable": (
                "matched_mode",
                [observable[value.truth_index] for value in matches],
            ),
            "coefficient_correlation": (
                ("coefficient_stage", "matched_mode"),
                metric_values("correlation", np.nan),
            ),
            "coefficient_origin_slope": (
                ("coefficient_stage", "matched_mode"),
                metric_values("origin_slope", np.nan),
            ),
            "coefficient_nrmse": (
                ("coefficient_stage", "matched_mode"),
                metric_values("nrmse", np.nan),
            ),
            "coefficient_valid_count": (
                ("coefficient_stage", "matched_mode"),
                metric_values("valid_count", 0),
            ),
            "coefficient_candidate_count": (
                ("coefficient_stage", "matched_mode"),
                metric_values("candidate_count", 0),
            ),
        },
        coords={
            "coefficient_stage": list(COEFFICIENT_STAGE_NAMES),
            "matched_mode": np.arange(1, len(matches) + 1),
        },
    )
    output.attrs.update(
        analysis_type="fable_v2_coefficient_stage_diagnostics",
        coefficient_matching="cosine_latitude_weighted_Hungarian_sign_and_scale",
        coefficient_mask="common_primary_non_COI_time_mask",
        **_PROVENANCE,
    )
    return output


def score_stagewise_strata(
    stages: Mapping[str, xr.DataArray],
    target: xr.DataArray,
    strata: Mapping[str, tuple[str, xr.DataArray]],
) -> xr.Dataset:
    """Score every diagnostic stage for explicit support and COI strata."""
    if not stages or not strata:
        raise ValueError("stagewise strata require nonempty stages and strata")
    truth = canonical_dimensions(target).transpose("time", "lat", "lon")
    aligned: dict[str, xr.DataArray] = {}
    common = cast(xr.DataArray, np.isfinite(truth))
    for name, stage in stages.items():
        value, _, _ = align_fields(stage, truth, None)
        aligned[name] = value
        common = common & cast(xr.DataArray, np.isfinite(value))
    rows: list[list[FieldMetrics]] = []
    for _name, (_kind, mask) in strata.items():
        _, _, selected = align_fields(truth, truth, mask)
        if selected is None:
            raise ValueError("diagnostic stratum mask was not resolved")
        keep = common & selected.astype(bool)
        rows.append(
            [
                weighted_field_metrics(value.where(keep), truth.where(keep))
                for value in aligned.values()
            ]
        )

    def metric_values(attribute: str) -> list[list[float | int]]:
        return [[getattr(metric, attribute) for metric in row] for row in rows]

    output = xr.Dataset(
        {
            "stratum_stage_field_correlation": (
                ("diagnostic_stratum", "stage"),
                metric_values("correlation"),
            ),
            "stratum_stage_field_origin_slope": (
                ("diagnostic_stratum", "stage"),
                metric_values("origin_slope"),
            ),
            "stratum_stage_field_nrmse": (
                ("diagnostic_stratum", "stage"),
                metric_values("nrmse"),
            ),
            "stratum_stage_valid_count": (
                ("diagnostic_stratum", "stage"),
                metric_values("valid_count"),
            ),
            "stratum_stage_candidate_count": (
                ("diagnostic_stratum", "stage"),
                metric_values("candidate_count"),
            ),
            "diagnostic_stratum_kind": (
                "diagnostic_stratum",
                [kind for kind, _mask in strata.values()],
            ),
        },
        coords={
            "diagnostic_stratum": list(strata),
            "stage": list(stages),
        },
    )
    output.attrs.update(
        analysis_type="fable_v2_stagewise_strata",
        stratum_field_weighting="equal_day_cosine_latitude",
        stratum_stage_mask="intersection_with_all_finite_stage_values",
        **_PROVENANCE,
    )
    return output


__all__ = [
    "COEFFICIENT_STAGE_NAMES",
    "align_sensor_offset_coordinates",
    "coefficient_stage_evidence",
    "common_bias_evidence",
    "relative_sensor_offset_evidence",
    "score_stagewise_strata",
]
