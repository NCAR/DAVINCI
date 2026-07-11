"""Pipeline adapter for evaluation-only FABLE v2 stage diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.analysis.base import (
    AnalysisResult,
    AnalysisRuntime,
    ArtifactDeclaration,
    DerivedAnalysis,
)
from davinci_monet.analysis.fable_v2_diagnostic_evidence import (
    COEFFICIENT_STAGE_NAMES,
    align_sensor_offset_coordinates,
    coefficient_stage_evidence,
    common_bias_evidence,
    relative_sensor_offset_evidence,
    score_stagewise_strata,
)
from davinci_monet.analysis.fable_v2_diagnostics import (
    V2_STAGE_NAMES,
    build_fable_v2_diagnostic_report,
    learned_basis_filtered_target_oracle,
    reconstruct_stage_field,
)
from davinci_monet.analysis.fable_v2_projection_diagnostics import (
    masked_projection_coefficients,
)
from davinci_monet.analysis.known_truth import _base_masks
from davinci_monet.analysis.known_truth_metrics import canonical_dimensions
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry


def _require(dataset: xr.Dataset, name: str, role: str) -> xr.DataArray:
    if name not in dataset:
        raise ValueError(f"fable_v2_diagnostics {role} is missing {name!r}")
    return canonical_dimensions(dataset[name])


def _on_target_time(dataset: xr.Dataset, target: xr.DataArray, role: str) -> xr.Dataset:
    if "time" not in dataset.coords:
        raise ValueError(f"fable_v2_diagnostics {role} has no time coordinate")
    try:
        selected = dataset.sel(time=target["time"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"fable_v2_diagnostics {role} does not cover target time") from exc
    if not np.array_equal(selected["time"].values, target["time"].values):
        raise ValueError(f"fable_v2_diagnostics {role} time does not match target")
    return selected


def _daily_monthly(field: xr.DataArray, time: xr.DataArray, description: str) -> xr.DataArray:
    value = canonical_dimensions(field)
    if value.dims != ("month", "lat", "lon"):
        raise ValueError(f"{description} must have exactly ('month', 'lat', 'lon') dimensions")
    month = xr.DataArray(
        time.dt.month.data,
        dims=("time",),
        coords={"time": time},
    )
    try:
        return value.sel(month=month).transpose("time", "lat", "lon")
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{description} does not cover all target months") from exc


def _broadcast_to_target(
    field: xr.DataArray,
    target: xr.DataArray,
    description: str,
) -> xr.DataArray:
    value = canonical_dimensions(field)
    unexpected = set(value.dims).difference(target.dims)
    if unexpected:
        raise ValueError(
            f"{description} has incompatible dimensions: {sorted(map(str, unexpected))}"
        )
    try:
        value, _ = xr.align(value, target, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(f"{description} coordinates must match the target exactly") from exc
    value, _ = xr.broadcast(value, target)
    return value.transpose(*target.dims)


def _observable_time_field(
    estimate: xr.Dataset,
    truth: xr.Dataset,
    target: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    coi = _require(estimate, "coi", "estimate")
    valid_segment = _require(estimate, "valid_segment", "estimate")
    observable = _require(truth, "mode_observable_true", "truth").astype(bool)
    if coi.dims != ("time", "mode") or valid_segment.dims != ("time", "mode"):
        raise ValueError("v2 COI and valid-segment arrays must have time/mode dimensions")
    if observable.dims != ("mode",):
        raise ValueError("v2 observable-mode truth must have the ('mode',) dimension")
    try:
        coi, valid_segment, observable = xr.align(
            coi,
            valid_segment,
            observable,
            join="exact",
            copy=False,
        )
    except ValueError as exc:
        raise ValueError("v2 COI, segment, and observable-mode coordinates must match") from exc
    observed_modes = observable["mode"].where(observable, drop=True)
    if observed_modes.size == 0:
        raise ValueError("v2 diagnostics require at least one observable truth mode")
    if "band_max" not in estimate.attrs:
        raise ValueError("v2 diagnostic estimate with COI data is missing band_max metadata")
    selected_coi = coi.sel(mode=observed_modes)
    selected_segment = valid_segment.sel(mode=observed_modes).astype(bool)
    coi_interior = (selected_coi >= float(estimate.attrs["band_max"])).all("mode")
    segment_available = selected_segment.all("mode")
    return (
        _broadcast_to_target(coi_interior, target, "v2 COI interior"),
        _broadcast_to_target(segment_available, target, "v2 valid segment"),
    )


def _diagnostic_strata(
    estimate: xr.Dataset,
    truth: xr.Dataset,
    target: xr.DataArray,
    domain: xr.DataArray,
    primary: xr.DataArray,
    support: xr.DataArray | None,
) -> dict[str, tuple[str, xr.DataArray]]:
    if support is None:
        raise ValueError("v2 diagnostics require immutable fitted spatial support")
    coi_interior, segment_available = _observable_time_field(estimate, truth, target)
    coefficient_available = _require(estimate, "coefficient_available", "estimate")
    coefficient_available = _broadcast_to_target(
        coefficient_available.astype(bool),
        target,
        "v2 coefficient availability",
    )
    supported = support > 0.0
    available = coefficient_available & segment_available
    return {
        "primary": ("primary", primary),
        "full_domain": ("domain", domain),
        "support_zero": ("support", domain & (support <= 0.0)),
        "support_partial": ("support", domain & supported & (support < 1.0)),
        "support_full": ("support", domain & (support >= 1.0)),
        "coi_interior": ("coi", domain & supported & available & coi_interior),
        "coi_edge": ("coi", domain & supported & available & ~coi_interior),
        "segment_unavailable": ("segment", domain & supported & ~available),
    }


def _fitted_sensor_offsets(
    projection: xr.Dataset,
    truth_offset: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray | None, xr.DataArray | None, str]:
    method = str(projection.attrs.get("projection_sensor_offset_method", "none"))
    if "sensor_offset" in projection:
        fitted = canonical_dimensions(projection["sensor_offset"])
        standard_error = (
            canonical_dimensions(projection["sensor_offset_standard_error"])
            if "sensor_offset_standard_error" in projection
            else None
        )
        overlap = (
            canonical_dimensions(projection["sensor_overlap_count"])
            if "sensor_overlap_count" in projection
            else None
        )
        return fitted, standard_error, overlap, method
    if method != "none":
        raise ValueError(
            "v2 projection configured sensor offsets but its immutable artifact omits them"
        )
    if "sensor" in projection.coords:
        return (
            xr.DataArray(
                np.zeros(projection.sizes["sensor"], dtype=np.float64),
                dims=("sensor",),
                coords={"sensor": projection["sensor"]},
            ),
            None,
            None,
            method,
        )
    return xr.zeros_like(truth_offset, dtype=np.float64), None, None, method


def evaluate_fable_v2_diagnostics(
    estimate: xr.Dataset,
    projection: xr.Dataset,
    truth: xr.Dataset,
    spec: Any,
) -> xr.Dataset:
    """Build the fixed seven-stage report from finalized fit artifacts and oracle truth."""
    target = _require(truth, "delta_filter_target_true", "truth")
    if target.dims != ("time", "lat", "lon"):
        raise ValueError("v2 filtered target must have time/lat/lon dimensions")
    # Materialize each bounded evaluation subset once. The stage/stratum metrics
    # perform many scalar reductions; leaving these arrays lazy would replay the
    # same multi-file Dask graph for every metric.
    estimate = _on_target_time(estimate, target, "estimate").copy(deep=False).load()
    projection = _on_target_time(projection, target, "projection").copy(deep=False).load()
    truth = _on_target_time(truth, target, "truth").copy(deep=False).load()
    target = _require(truth, "delta_filter_target_true", "truth")
    basis = _require(estimate, "eofs", "estimate")
    support = _require(estimate, "spatial_support", "estimate")
    fitted_bias = _require(estimate, "clim_bias_applied", "estimate")
    oracle_bias = _daily_monthly(
        _require(truth, "clim_bias_raw_true", "truth"),
        target["time"],
        "oracle climatological bias",
    )
    oracle_support = _daily_monthly(
        _require(truth, "spatial_support_true", "truth"),
        target["time"],
        "oracle spatial support",
    )
    oracle_bias = oracle_support * oracle_bias
    domain, primary, fitted_support, _ = _base_masks(estimate, truth, target, spec)
    true_offset = _require(truth, "sensor_bias_log_true", "truth")
    fitted_offset, offset_error, overlap_count, offset_method = _fitted_sensor_offsets(
        projection,
        true_offset,
    )
    raw_sensor_mapping = (
        spec.get("projection_to_truth_sensor", {})
        if isinstance(spec, Mapping)
        else getattr(spec, "projection_to_truth_sensor", {})
    )
    sensor_mapping = {str(source): str(target) for source, target in raw_sensor_mapping.items()}
    fitted_offset, offset_error, overlap_count = align_sensor_offset_coordinates(
        fitted_offset,
        true_offset,
        sensor_mapping,
        standard_error=offset_error,
        overlap_count=overlap_count,
    )
    raw_common_factor = (
        spec.get("reported_common_factor_amplitude")
        if isinstance(spec, Mapping)
        else getattr(spec, "reported_common_factor_amplitude")
    )
    if raw_common_factor is None:
        raise ValueError("v2 diagnostics require reported_common_factor_amplitude")
    common_factor_amplitude = float(raw_common_factor)

    oracle = learned_basis_filtered_target_oracle(
        target,
        basis,
        oracle_bias=oracle_bias,
        spatial_multiplier=oracle_support,
        score_mask=primary,
    )
    noiseless_pc = masked_projection_coefficients(
        target,
        basis,
        oracle_bias,
        oracle_support,
        truth,
        noisy=False,
        common_factor_amplitude=common_factor_amplitude,
    )
    noisy_pc = masked_projection_coefficients(
        target,
        basis,
        oracle_bias,
        oracle_support,
        truth,
        noisy=True,
        common_factor_amplitude=common_factor_amplitude,
        sensor_offsets=fitted_offset,
    )
    stages = {
        V2_STAGE_NAMES[0]: oracle["learned_basis_oracle_delta"],
        V2_STAGE_NAMES[1]: reconstruct_stage_field(
            basis,
            noiseless_pc,
            fixed_component=oracle_bias,
            spatial_multiplier=oracle_support,
        ),
        V2_STAGE_NAMES[2]: reconstruct_stage_field(
            basis,
            noisy_pc,
            fixed_component=oracle_bias,
            spatial_multiplier=oracle_support,
        ),
        V2_STAGE_NAMES[3]: reconstruct_stage_field(
            basis,
            oracle["learned_basis_oracle_pc"],
            fixed_component=fitted_bias,
            spatial_multiplier=support,
        ),
        V2_STAGE_NAMES[4]: reconstruct_stage_field(
            basis,
            _require(projection, "pc", "projection"),
            fixed_component=fitted_bias,
            spatial_multiplier=support,
        ),
        V2_STAGE_NAMES[5]: _require(estimate, "delta_log_requested", "estimate"),
        V2_STAGE_NAMES[6]: _require(estimate, "delta_log_applied", "estimate"),
    }
    report = build_fable_v2_diagnostic_report(
        stages,
        target,
        unfiltered_in_span=_require(truth, "delta_best_representable_true", "truth"),
        mask=primary,
    )
    report["learned_basis_oracle_fit_rank"] = oracle["learned_basis_oracle_fit_rank"]
    bias_mask = domain & (oracle_support > 0.0)
    if fitted_support is not None:
        bias_mask = bias_mask & (fitted_support > 0.0)
    bias_report = common_bias_evidence(fitted_bias, oracle_bias, bias_mask)

    offset_report = relative_sensor_offset_evidence(
        fitted_offset,
        true_offset,
        standard_error=offset_error,
        overlap_count=overlap_count,
    )
    coefficient_report = coefficient_stage_evidence(
        basis,
        _require(truth, "pattern_true", "truth"),
        {
            COEFFICIENT_STAGE_NAMES[0]: _require(projection, "pc", "projection"),
            COEFFICIENT_STAGE_NAMES[1]: _require(estimate, "pc", "estimate"),
        },
        {
            COEFFICIENT_STAGE_NAMES[0]: _require(truth, "correction_pc_true", "truth"),
            COEFFICIENT_STAGE_NAMES[1]: _require(
                truth,
                "correction_pc_filter_target_true",
                "truth",
            ),
        },
        primary,
        observable_modes=_require(truth, "mode_observable_true", "truth"),
    )
    strata_report = score_stagewise_strata(
        stages,
        target,
        _diagnostic_strata(
            estimate,
            truth,
            target,
            domain,
            primary,
            fitted_support,
        ),
    )
    report = xr.merge(
        (report, bias_report, offset_report, coefficient_report, strata_report),
        compat="no_conflicts",
    )
    report.attrs.update(
        projection_noise_covariance="reported_diagonal_plus_shared_constant_low_rank_factor",
        reported_common_factor_amplitude=common_factor_amplitude,
        masked_projection_support_policy="binary_row_selection_then_reconstruction_taper",
        masked_projection_target_policy="untaper_applied_anomaly_on_positive_support",
        production_fit_artifacts_reused="true",
        sensor_offset_method=offset_method,
        sensor_coordinate_mapping=",".join(
            f"{source}->{target}" for source, target in sorted(sensor_mapping.items())
        ),
        sensor_offset_gauge="zero_sensor_mean",
        masked_noisy_sensor_offset_application="fitted_relative_offsets_subtracted",
        absolute_sensor_offset_identifiable="false",
        absolute_sensor_offset_fit_status="not_scored_scientifically_unidentifiable",
        common_bias_truth_reference="physical_common_correction_bias_after_true_support",
        common_bias_gauge_caveat=(
            "fitted common bias may absorb the unidentifiable mean absolute sensor offset"
        ),
        coefficient_matching="cosine_latitude_weighted_Hungarian_sign_and_scale",
        coefficient_mask="common_primary_non_COI_time_mask",
        support_stratum_reference="immutable_fitted_spatial_support",
        coi_stratum_definition="all_observable_modes_coi_at_least_band_max",
        stratum_stage_mask="intersection_with_all_finite_stage_values",
        supplemental_evidence_use="diagnostic_only_nonranking",
    )
    return report


@analysis_registry.register("fable_v2_diagnostics")
class FableV2DiagnosticsAnalysis(DerivedAnalysis):
    """Registered, read-only adapter for the v2 diagnostic report."""

    name = "fable_v2_diagnostics"
    long_name = "FABLE V2 Stage Diagnostics"
    output_geometry = DataGeometry.ARTIFACT

    def analyze_inputs(
        self,
        inputs: Mapping[str, xr.Dataset],
        spec: Any,
        runtime: AnalysisRuntime,
    ) -> AnalysisResult:
        del runtime
        missing = [role for role in ("estimate", "projection", "truth") if role not in inputs]
        if missing:
            raise ValueError(f"fable_v2_diagnostics is missing named inputs: {', '.join(missing)}")
        report = evaluate_fable_v2_diagnostics(
            inputs["estimate"],
            inputs["projection"],
            inputs["truth"],
            spec,
        )
        return AnalysisResult(
            dataset=report,
            artifacts=(
                ArtifactDeclaration(
                    kind="netcdf_collection",
                    role="v2_diagnostic_report",
                    reload=True,
                    options={"time_chunk_size": 31},
                ),
            ),
        )


__all__ = [
    "FableV2DiagnosticsAnalysis",
    "evaluate_fable_v2_diagnostics",
]
