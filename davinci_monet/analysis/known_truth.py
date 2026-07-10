"""Read-only recovery metrics for synthetic known-truth evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.analysis._known_truth_report import (
    add_basis_and_coefficients,
    add_policy_diagnostics,
    evaluation_strata,
    metric_variables,
    selected_evaluation_time,
)
from davinci_monet.analysis.base import (
    AnalysisResult,
    AnalysisRuntime,
    ArtifactDeclaration,
    DerivedAnalysis,
)
from davinci_monet.analysis.known_truth_metrics import (
    FieldMetrics,
    ModeMatch,
    SubspaceMetrics,
    align_fields,
    canonical_dimensions,
    match_weighted_modes,
    safe_ratio,
    weighted_field_metrics,
    weighted_subspace_metrics,
)
from davinci_monet.analysis.provenance import consistent_spec_hash
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry

_DEFAULTS: dict[str, Any] = {
    "estimate_delta_variable": "delta_log_applied",
    "truth_delta_variable": "delta_filter_target_true",
    "full_truth_delta_variable": "delta_applied_true",
    "in_span_truth_delta_variable": "delta_in_span_true",
    "perpendicular_truth_delta_variable": "delta_perp_true",
    "estimate_aod_variable": "aod_target",
    "truth_aod_variable": "aod_filter_target_true",
    "full_truth_aod_variable": "aod_target_applied_true",
    "model_aod_variable": "model_aod_overpass_true",
    "estimate_basis_variable": "eofs",
    "truth_basis_variable": "pattern_true",
    "estimate_coefficient_variable": "pc",
    "truth_coefficient_variable": "correction_pc_filter_target_true",
    "support_variable": "spatial_support",
    "resolution_variable": "resolution",
    "observable_mode_variable": "mode_observable_true",
    "primary_mask_variable": None,
    "split_variable": "split",
    "evaluation_splits": (),
    "best_representable_variable": "delta_best_representable_true",
}


def _field(spec: Any, name: str) -> Any:
    default = _DEFAULTS[name]
    if isinstance(spec, Mapping):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _lookup(dataset: xr.Dataset, name: str | None) -> xr.DataArray | None:
    return dataset[name] if name is not None and name in dataset else None


def _optional_field_name(spec: Any, name: str) -> str | None:
    value = _field(spec, name)
    return None if value is None else str(value)


def _loaded_subset(dataset: xr.Dataset, variable_names: Sequence[str | None]) -> xr.Dataset:
    names = list(
        dict.fromkeys(name for name in variable_names if name is not None and name in dataset)
    )
    return dataset[names].copy(deep=False).load()


def _load_evaluation_inputs(
    estimate: xr.Dataset, truth: xr.Dataset, spec: Any
) -> tuple[xr.Dataset, xr.Dataset]:
    primary_mask = _optional_field_name(spec, "primary_mask_variable")
    split_variable = (
        _optional_field_name(spec, "split_variable")
        if tuple(_field(spec, "evaluation_splits"))
        else None
    )
    estimate_variables = (
        _optional_field_name(spec, "estimate_delta_variable"),
        _optional_field_name(spec, "estimate_aod_variable"),
        _optional_field_name(spec, "estimate_basis_variable"),
        _optional_field_name(spec, "estimate_coefficient_variable"),
        _optional_field_name(spec, "support_variable"),
        _optional_field_name(spec, "resolution_variable"),
        primary_mask,
        "coefficient_available",
        "valid_segment",
        "coi",
        "clip_reason",
    )
    truth_variables = (
        _optional_field_name(spec, "truth_delta_variable"),
        _optional_field_name(spec, "full_truth_delta_variable"),
        _optional_field_name(spec, "in_span_truth_delta_variable"),
        _optional_field_name(spec, "perpendicular_truth_delta_variable"),
        _optional_field_name(spec, "truth_aod_variable"),
        _optional_field_name(spec, "full_truth_aod_variable"),
        _optional_field_name(spec, "model_aod_variable"),
        _optional_field_name(spec, "truth_basis_variable"),
        _optional_field_name(spec, "truth_coefficient_variable"),
        _optional_field_name(spec, "observable_mode_variable"),
        _optional_field_name(spec, "best_representable_variable"),
        primary_mask,
        split_variable,
        "spatial_support_true",
        "innovation_noise_true",
        "obs_holdout_aod",
    )
    selected_time = selected_evaluation_time(
        truth,
        _optional_field_name(spec, "split_variable"),
        tuple(str(value) for value in _field(spec, "evaluation_splits")),
    )
    if selected_time is not None:
        try:
            estimate = estimate.sel(time=selected_time)
            truth = truth.sel(time=selected_time)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "known_truth evaluation-split times must match estimate and truth exactly"
            ) from exc
    return (
        _loaded_subset(estimate, estimate_variables),
        _loaded_subset(truth, truth_variables),
    )


def _daily_support(support: xr.DataArray, time: xr.DataArray) -> xr.DataArray:
    support = canonical_dimensions(support)
    if "month" not in support.dims:
        return support
    month = xr.DataArray(time.dt.month, dims=("time",), coords={"time": time})
    return support.sel(month=month).transpose("time", "lat", "lon")


def _broadcast_exact(data: xr.DataArray, field: xr.DataArray, description: str) -> xr.DataArray:
    """Broadcast a lower-dimensional field only when shared coordinates are exact."""
    aligned = canonical_dimensions(data)
    unexpected = set(aligned.dims).difference(field.dims)
    if unexpected:
        raise ValueError(
            f"{description} has incompatible dimensions: {sorted(map(str, unexpected))}"
        )
    try:
        aligned, _ = xr.align(aligned, field, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(f"{description} coordinates must match the estimate exactly") from exc
    aligned, _ = xr.broadcast(aligned, field)
    return aligned.transpose(*field.dims)


def _observable_modes(estimate: xr.Dataset, truth: xr.Dataset, spec: Any) -> xr.DataArray | None:
    """Return the explicit truth mask for mode-indexed QA fields."""
    mode_fields = [
        canonical_dimensions(estimate[name])
        for name in ("valid_segment", "coi", str(_field(spec, "resolution_variable")))
        if name in estimate and "mode" in canonical_dimensions(estimate[name]).dims
    ]
    if not mode_fields:
        return None

    mode_template = mode_fields[0]["mode"]
    for data in mode_fields[1:]:
        try:
            xr.align(data["mode"], mode_template, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError("known_truth mode-indexed estimate fields must align exactly") from exc

    variable = _field(spec, "observable_mode_variable")
    if variable is None:
        return xr.ones_like(mode_template, dtype=bool)
    observable = _lookup(truth, str(variable))
    if observable is None:
        raise ValueError(
            f"known_truth requires observable mode variable {str(variable)!r} "
            "when estimate QA is mode-indexed"
        )
    observable = canonical_dimensions(observable)
    if observable.dims != ("mode",):
        raise ValueError(f"known_truth observable mode variable {str(variable)!r} must be 1-D")
    observable = observable.astype(bool)
    any_observable = observable.any()
    if any_observable.chunks is not None:
        any_observable = any_observable.compute()
    if not bool(any_observable.item()):
        raise ValueError("known_truth observable mode mask selects no modes")

    estimate_basis = _lookup(estimate, _optional_field_name(spec, "estimate_basis_variable"))
    truth_basis = _lookup(truth, _optional_field_name(spec, "truth_basis_variable"))
    if estimate_basis is not None and truth_basis is not None:
        truth_modes = canonical_dimensions(truth_basis)["mode"]
        try:
            observable, _ = xr.align(observable, truth_modes, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(
                "known_truth observable mode coordinates must match the truth basis exactly"
            ) from exc
        mapped = np.zeros(mode_template.size, dtype=bool)
        for match in match_weighted_modes(estimate_basis, truth_basis):
            mapped[match.estimate_index] = bool(observable.values[match.truth_index])
        return xr.DataArray(mapped, dims=("mode",), coords={"mode": mode_template})

    try:
        observable, _ = xr.align(observable, mode_template, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(
            "known_truth observable mode coordinates must match the estimate exactly"
        ) from exc
    return observable


def _reduce_observable_modes(
    data: xr.DataArray,
    observable: xr.DataArray | None,
    *,
    reduction: str,
    description: str,
) -> xr.DataArray:
    """Reduce a mode QA field over only explicitly observable truth modes."""
    selected = canonical_dimensions(data)
    if "mode" not in selected.dims:
        return selected
    if observable is not None:
        try:
            selected, mode_mask = xr.align(selected, observable, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(f"{description} mode coordinates must match exactly") from exc
        indices = np.flatnonzero(np.asarray(mode_mask.values, dtype=bool))
        selected = selected.isel(mode=indices)
    if selected.sizes["mode"] == 0:
        raise ValueError(f"{description} has no observable modes")
    if reduction == "all":
        return selected.astype(bool).all("mode")
    if reduction == "mean":
        return selected.mean("mode")
    raise ValueError(f"unknown mode reduction {reduction!r}")


def _base_masks(
    estimate: xr.Dataset, truth: xr.Dataset, field: xr.DataArray, spec: Any
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None, xr.DataArray | None]:
    domain = xr.ones_like(field, dtype=bool)
    split_variable = _field(spec, "split_variable")
    splits = tuple(str(value) for value in _field(spec, "evaluation_splits"))
    if splits:
        if split_variable is None or str(split_variable) not in truth:
            raise ValueError(
                f"known_truth requires split variable {str(split_variable)!r} "
                "for requested evaluation_splits"
            )
        split = canonical_dimensions(truth[str(split_variable)])
        if split.dims != ("time",):
            raise ValueError("known_truth split variable must have exactly the ('time',) dimension")
        available = set(np.asarray(split.values).astype(str).tolist())
        missing_splits = [value for value in splits if value not in available]
        if missing_splits:
            raise ValueError(
                "known_truth requested split value(s) are absent: " + ", ".join(missing_splits)
            )
        split_mask = xr.zeros_like(split, dtype=bool)
        for value in splits:
            split_mask = split_mask | (split.astype(str) == value)
        domain = domain & _broadcast_exact(split_mask, field, "known_truth split variable")

    primary = domain
    primary_name = _field(spec, "primary_mask_variable")
    explicit = _lookup(estimate, primary_name)
    if explicit is None:
        explicit = _lookup(truth, primary_name)
    if explicit is not None:
        primary = primary & _broadcast_exact(
            canonical_dimensions(explicit).astype(bool), field, "known_truth primary mask"
        )
    if "coefficient_available" in estimate:
        primary = primary & _broadcast_exact(
            estimate["coefficient_available"].astype(bool),
            field,
            "known_truth coefficient availability",
        )

    observable = _observable_modes(estimate, truth, spec)
    if "valid_segment" in estimate:
        valid_segment = _reduce_observable_modes(
            estimate["valid_segment"],
            observable,
            reduction="all",
            description="known_truth valid_segment",
        )
        primary = primary & _broadcast_exact(valid_segment, field, "known_truth valid_segment")
    if "coi" in estimate:
        if "band_max" not in estimate.attrs:
            raise ValueError("known_truth estimate with COI data is missing band_max metadata")
        coi_safe = _reduce_observable_modes(
            estimate["coi"] >= float(estimate.attrs["band_max"]),
            observable,
            reduction="all",
            description="known_truth COI",
        )
        primary = primary & _broadcast_exact(coi_safe, field, "known_truth COI")

    support_variable = _field(spec, "support_variable")
    support = _lookup(estimate, str(support_variable)) if support_variable is not None else None
    if support is None:
        support = _lookup(truth, "spatial_support_true")
    if support is not None:
        support = _broadcast_exact(
            _daily_support(support, field["time"]), field, "known_truth spatial support"
        )
        primary = primary & (support > 0.0)

    resolution_variable = _field(spec, "resolution_variable")
    resolution = (
        _lookup(estimate, str(resolution_variable)) if resolution_variable is not None else None
    )
    if resolution is not None:
        resolution = _reduce_observable_modes(
            resolution,
            observable,
            reduction="mean",
            description="known_truth resolution",
        )
        resolution = _broadcast_exact(resolution, field, "known_truth resolution")
    return domain, primary, support, resolution


def evaluate_known_truth(estimate: xr.Dataset, truth: xr.Dataset, spec: Any) -> xr.Dataset:
    """Build a small recovery report without mutating either input."""
    spec_hash = consistent_spec_hash([estimate, truth])
    required = {
        "estimate delta": (estimate, str(_field(spec, "estimate_delta_variable"))),
        "truth delta": (truth, str(_field(spec, "truth_delta_variable"))),
        "estimate AOD": (estimate, str(_field(spec, "estimate_aod_variable"))),
        "truth AOD": (truth, str(_field(spec, "truth_aod_variable"))),
        "model AOD": (truth, str(_field(spec, "model_aod_variable"))),
    }
    missing = [label for label, (dataset, variable) in required.items() if variable not in dataset]
    if missing:
        raise ValueError(f"known_truth is missing required variables: {', '.join(missing)}")
    estimate, truth = _load_evaluation_inputs(estimate, truth, spec)
    estimate_delta = canonical_dimensions(estimate[required["estimate delta"][1]])
    truth_delta = canonical_dimensions(truth[required["truth delta"][1]])
    estimate_delta, truth_delta, _ = align_fields(estimate_delta, truth_delta, None)
    base, primary, support, resolution = _base_masks(estimate, truth, estimate_delta, spec)
    rows = evaluation_strata(estimate_delta, base, primary, support, resolution)
    representable = _lookup(truth, str(_field(spec, "best_representable_variable")))
    full_truth_delta = _lookup(truth, _field(spec, "full_truth_delta_variable"))
    in_span_truth_delta = _lookup(truth, _field(spec, "in_span_truth_delta_variable"))
    perpendicular_truth_delta = _lookup(truth, _field(spec, "perpendicular_truth_delta_variable"))
    full_truth_aod = _lookup(truth, _field(spec, "full_truth_aod_variable"))
    output = metric_variables(
        estimate_delta,
        truth_delta,
        estimate[required["estimate AOD"][1]],
        truth[required["truth AOD"][1]],
        truth[required["model AOD"][1]],
        rows,
        representable,
        full_truth_aod,
    )
    add_policy_diagnostics(
        output,
        rows,
        estimate_delta,
        truth_delta,
        full_truth_delta,
        in_span_truth_delta,
        perpendicular_truth_delta,
        representable,
        _lookup(estimate, "clip_reason"),
        estimate[required["estimate AOD"][1]],
        truth[required["model AOD"][1]],
        _lookup(truth, "obs_holdout_aod"),
    )
    add_basis_and_coefficients(
        output,
        _lookup(estimate, str(_field(spec, "estimate_basis_variable"))),
        _lookup(truth, str(_field(spec, "truth_basis_variable"))),
        _lookup(estimate, str(_field(spec, "estimate_coefficient_variable"))),
        _lookup(truth, str(_field(spec, "truth_coefficient_variable"))),
        primary,
        _lookup(truth, str(_field(spec, "observable_mode_variable"))),
    )
    noise = _lookup(truth, "innovation_noise_true")
    if noise is not None:
        applied_energy = weighted_field_metrics(
            estimate_delta, xr.zeros_like(estimate_delta), primary
        )
        noise_energy = weighted_field_metrics(noise, xr.zeros_like(noise), primary)
        output["false_positive_energy_ratio"] = safe_ratio(
            applied_energy.rmse**2, noise_energy.rmse**2
        )
    output.attrs.update(
        analysis_type="known_truth",
        geometry="artifact",
        evaluation_only="true",
        persistence_policy="immutable_netcdf_collection",
        field_weighting="equal_day_cosine_latitude",
        mode_matching="cosine_latitude_weighted_hungarian_sign_permutation",
        estimate_delta_variable=required["estimate delta"][1],
        truth_delta_variable=required["truth delta"][1],
    )
    if spec_hash is not None:
        output.attrs["spec_hash"] = spec_hash
    return output


@analysis_registry.register("known_truth")
class KnownTruthAnalysis(DerivedAnalysis):
    """Pipeline adapter for evaluation-only synthetic recovery scoring."""

    name = "known_truth"
    long_name = "Known Truth Recovery Evaluation"
    output_geometry = DataGeometry.ARTIFACT

    def analyze_inputs(
        self,
        inputs: Mapping[str, xr.Dataset],
        spec: Any,
        runtime: AnalysisRuntime,
    ) -> AnalysisResult:
        del runtime
        missing = [role for role in ("estimate", "truth") if role not in inputs]
        if missing:
            raise ValueError(f"known_truth is missing named inputs: {', '.join(missing)}")
        return AnalysisResult(
            dataset=evaluate_known_truth(inputs["estimate"], inputs["truth"], spec),
            artifacts=(
                ArtifactDeclaration(
                    kind="netcdf_collection",
                    role="recovery_report",
                    reload=True,
                    options={"time_chunk_size": 1},
                ),
            ),
        )


__all__ = [
    "FieldMetrics",
    "KnownTruthAnalysis",
    "ModeMatch",
    "SubspaceMetrics",
    "evaluate_known_truth",
    "match_weighted_modes",
    "weighted_field_metrics",
    "weighted_subspace_metrics",
]
