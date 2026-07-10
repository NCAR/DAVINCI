"""Private helpers for assembling known-truth recovery reports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.analysis.known_truth_metrics import (
    FieldMetrics,
    canonical_dimensions,
    match_weighted_modes,
    safe_ratio,
    weighted_field_metrics,
    weighted_subspace_metrics,
)


def add_policy_diagnostics(
    output: xr.Dataset,
    rows: Sequence[tuple[str, str, xr.DataArray]],
    estimate_delta: xr.DataArray,
    truth_delta: xr.DataArray,
    full_truth_delta: xr.DataArray | None,
    in_span_truth_delta: xr.DataArray | None,
    perpendicular_truth_delta: xr.DataArray | None,
    representable: xr.DataArray | None,
    clip_reason: xr.DataArray | None,
    estimate_aod: xr.DataArray,
    model_aod: xr.DataArray,
    holdout_aod: xr.DataArray | None,
) -> None:
    """Add irreducible-floor, clipping, and independent-holdout diagnostics."""
    if full_truth_delta is not None:
        full_metrics = [
            weighted_field_metrics(estimate_delta, full_truth_delta, mask) for _, _, mask in rows
        ]
        output["full_delta_correlation"] = (
            "stratum",
            [value.correlation for value in full_metrics],
        )
        output["full_delta_origin_slope"] = (
            "stratum",
            [value.origin_slope for value in full_metrics],
        )
        output["full_delta_nrmse"] = (
            "stratum",
            [value.nrmse for value in full_metrics],
        )
        output["full_delta_truth_rms"] = (
            "stratum",
            [value.truth_rms for value in full_metrics],
        )
        _add_component_rms(
            output,
            rows,
            full_metrics,
            in_span_truth_delta,
            perpendicular_truth_delta,
        )
    if representable is not None and full_truth_delta is not None:
        floor = [
            weighted_field_metrics(representable, full_truth_delta, mask) for _, _, mask in rows
        ]
        output["off_basis_floor_nrmse"] = (
            "stratum",
            [value.nrmse for value in floor],
        )
        output["off_basis_floor_rmse"] = (
            "stratum",
            [value.rmse for value in floor],
        )
    if clip_reason is not None:
        fractions: list[float] = []
        clip = canonical_dimensions(clip_reason)
        for _, _, mask in rows:
            aligned_clip, aligned_truth, selected = _align_policy_field(clip, truth_delta, mask)
            valid = np.isfinite(aligned_clip) & np.isfinite(aligned_truth)
            if selected is not None:
                valid = valid & selected.astype(bool)
            clipped = valid & (abs(aligned_clip) == 1)
            denominator = _scalar_float(valid.sum())
            fractions.append(_scalar_float(clipped.sum()) / denominator if denominator else np.nan)
        output["clip_fraction"] = ("stratum", fractions)
    if holdout_aod is not None:
        holdout = canonical_dimensions(holdout_aod)
        estimate_metrics = [
            weighted_field_metrics(estimate_aod, holdout, mask) for _, _, mask in rows
        ]
        model_metrics = [weighted_field_metrics(model_aod, holdout, mask) for _, _, mask in rows]
        output["holdout_aod_rmse"] = (
            "stratum",
            [value.rmse for value in estimate_metrics],
        )
        output["model_holdout_aod_rmse"] = (
            "stratum",
            [value.rmse for value in model_metrics],
        )
        output["holdout_aod_rmse_ratio"] = (
            "stratum",
            [
                safe_ratio(estimate.rmse, model.rmse)
                for estimate, model in zip(estimate_metrics, model_metrics, strict=True)
            ],
        )


def _align_policy_field(
    field: xr.DataArray, truth: xr.DataArray, mask: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None]:
    from davinci_monet.analysis.known_truth_metrics import align_fields

    return align_fields(field, truth, mask)


def _add_component_rms(
    output: xr.Dataset,
    rows: Sequence[tuple[str, str, xr.DataArray]],
    full_metrics: Sequence[FieldMetrics],
    in_span: xr.DataArray | None,
    perpendicular: xr.DataArray | None,
) -> None:
    for name, component in (
        ("in_span_truth_rms", in_span),
        ("perpendicular_truth_rms", perpendicular),
    ):
        if component is None:
            continue
        metrics = [
            weighted_field_metrics(xr.zeros_like(component), component, mask) for _, _, mask in rows
        ]
        output[name] = ("stratum", [value.truth_rms for value in metrics])
        if name == "perpendicular_truth_rms":
            output["perpendicular_to_full_rms_ratio"] = (
                "stratum",
                [
                    safe_ratio(value.truth_rms, full.truth_rms)
                    for value, full in zip(metrics, full_metrics, strict=True)
                ],
            )


def _scalar_float(value: xr.DataArray) -> float:
    if value.chunks is not None:
        value = value.compute()
    return float(value.item())


def evaluation_strata(
    field: xr.DataArray,
    base: xr.DataArray,
    primary: xr.DataArray,
    support: xr.DataArray | None,
    resolution: xr.DataArray | None,
) -> list[tuple[str, str, xr.DataArray]]:
    """Return the fixed primary, policy, seasonal, and latitude report masks."""
    rows = [("primary", "primary", primary), ("full_domain", "domain", base)]
    if support is not None:
        rows.extend(
            [
                ("support_zero", "support", base & (support <= 0.0)),
                ("support_partial", "support", base & (support > 0.0) & (support < 1.0)),
                ("support_full", "support", base & (support >= 1.0)),
            ]
        )
    if resolution is not None:
        rows.extend(
            [
                ("resolution_low", "resolution", primary & (resolution < 0.3)),
                (
                    "resolution_medium",
                    "resolution",
                    primary & (resolution >= 0.3) & (resolution < 0.7),
                ),
                ("resolution_high", "resolution", primary & (resolution >= 0.7)),
            ]
        )
    month = field["time"].dt.month
    for name, months in (
        ("season_DJF", (12, 1, 2)),
        ("season_MAM", (3, 4, 5)),
        ("season_JJA", (6, 7, 8)),
        ("season_SON", (9, 10, 11)),
    ):
        rows.append((name, "season", primary & month.isin(months)))
    latitude = field["lat"]
    for name, lower, upper in (
        ("latitude_south_high", -90.0, -60.0),
        ("latitude_south_mid", -60.0, -23.5),
        ("latitude_tropical", -23.5, 23.5),
        ("latitude_north_mid", 23.5, 60.0),
        ("latitude_north_high", 60.0, 90.1),
    ):
        rows.append((name, "latitude", primary & (latitude >= lower) & (latitude < upper)))
    return rows


def selected_evaluation_time(
    truth: xr.Dataset,
    split_variable: str | None,
    splits: Sequence[str],
) -> xr.DataArray | None:
    """Load only split labels before slicing scientific evaluation arrays."""
    if not splits:
        return None
    if split_variable is None or split_variable not in truth:
        raise ValueError(
            f"known_truth requires split variable {split_variable!r} "
            "for requested evaluation_splits"
        )
    split = canonical_dimensions(truth[split_variable])
    if split.dims != ("time",):
        raise ValueError("known_truth split variable must have exactly the ('time',) dimension")
    split = split.load()
    available = set(np.asarray(split.values).astype(str).tolist())
    missing = [value for value in splits if value not in available]
    if missing:
        raise ValueError("known_truth requested split value(s) are absent: " + ", ".join(missing))
    selected = xr.zeros_like(split, dtype=bool)
    for value in splits:
        selected = selected | (split.astype(str) == value)
    return split["time"].where(selected, drop=True)


def metric_variables(
    estimate_delta: xr.DataArray,
    truth_delta: xr.DataArray,
    estimate_aod: xr.DataArray,
    truth_aod: xr.DataArray,
    model_aod: xr.DataArray,
    rows: Sequence[tuple[str, str, xr.DataArray]],
    representable: xr.DataArray | None,
    full_truth_aod: xr.DataArray | None,
) -> xr.Dataset:
    """Build the stratum-indexed field metric variables."""
    field_metrics = [
        weighted_field_metrics(estimate_delta, truth_delta, mask) for _, _, mask in rows
    ]
    aod_metrics = [weighted_field_metrics(estimate_aod, truth_aod, mask) for _, _, mask in rows]
    model_metrics = [weighted_field_metrics(model_aod, truth_aod, mask) for _, _, mask in rows]
    representable_metrics = (
        [weighted_field_metrics(estimate_delta, representable, mask) for _, _, mask in rows]
        if representable is not None
        else None
    )
    full_aod_metrics = (
        [weighted_field_metrics(estimate_aod, full_truth_aod, mask) for _, _, mask in rows]
        if full_truth_aod is not None
        else None
    )
    full_model_metrics = (
        [weighted_field_metrics(model_aod, full_truth_aod, mask) for _, _, mask in rows]
        if full_truth_aod is not None
        else None
    )
    coords = {
        "stratum": [name for name, _, _ in rows],
        "stratum_kind": ("stratum", [kind for _, kind, _ in rows]),
    }
    data: dict[str, Any] = {
        "field_correlation": ("stratum", [value.correlation for value in field_metrics]),
        "field_origin_slope": ("stratum", [value.origin_slope for value in field_metrics]),
        "field_bias": ("stratum", [value.bias for value in field_metrics]),
        "field_rmse": ("stratum", [value.rmse for value in field_metrics]),
        "field_nrmse": ("stratum", [value.nrmse for value in field_metrics]),
        "field_truth_rms": ("stratum", [value.truth_rms for value in field_metrics]),
        "valid_count": ("stratum", [value.valid_count for value in field_metrics]),
        "candidate_count": ("stratum", [value.candidate_count for value in field_metrics]),
        "excluded_fraction": ("stratum", [value.excluded_fraction for value in field_metrics]),
        "aod_rmse": ("stratum", [value.rmse for value in aod_metrics]),
        "aod_correlation": ("stratum", [value.correlation for value in aod_metrics]),
        "model_aod_rmse": ("stratum", [value.rmse for value in model_metrics]),
    }
    ratios = [
        safe_ratio(aod.rmse, model.rmse)
        for aod, model in zip(aod_metrics, model_metrics, strict=True)
    ]
    data["aod_rmse_ratio"] = ("stratum", ratios)
    data["aod_rmse_improvement"] = ("stratum", [1.0 - value for value in ratios])
    if representable_metrics is not None:
        data["best_representable_nrmse"] = (
            "stratum",
            [value.nrmse for value in representable_metrics],
        )
    if full_aod_metrics is not None and full_model_metrics is not None:
        full_ratios = [
            safe_ratio(aod.rmse, model.rmse)
            for aod, model in zip(full_aod_metrics, full_model_metrics, strict=True)
        ]
        data["full_target_aod_rmse"] = (
            "stratum",
            [value.rmse for value in full_aod_metrics],
        )
        data["model_full_target_aod_rmse"] = (
            "stratum",
            [value.rmse for value in full_model_metrics],
        )
        data["full_target_aod_rmse_ratio"] = ("stratum", full_ratios)
        data["full_target_aod_rmse_improvement"] = (
            "stratum",
            [1.0 - value for value in full_ratios],
        )
    return xr.Dataset(data, coords=coords)


def add_basis_and_coefficients(
    output: xr.Dataset,
    estimate_basis: xr.DataArray | None,
    truth_basis: xr.DataArray | None,
    estimate_pc: xr.DataArray | None,
    truth_pc: xr.DataArray | None,
    temporal_mask: xr.DataArray,
    observable_modes: xr.DataArray | None = None,
) -> None:
    """Add optional basis and coefficient diagnostics to a report."""
    if estimate_basis is None or truth_basis is None:
        return
    subspace = weighted_subspace_metrics(estimate_basis, truth_basis)
    output["subspace_angle_mean_degrees"] = float(np.mean(subspace.angles_degrees))
    output["subspace_angle_max_degrees"] = float(np.max(subspace.angles_degrees))
    output["subspace_projector_error"] = subspace.projector_error
    output["subspace_estimate_rank"] = subspace.estimate_rank
    output["subspace_truth_rank"] = subspace.truth_rank
    matches = match_weighted_modes(estimate_basis, truth_basis)
    if estimate_pc is None or truth_pc is None:
        return
    estimate_pc = canonical_dimensions(estimate_pc).transpose("time", "mode")
    truth_pc = canonical_dimensions(truth_pc).transpose("time", "mode")
    try:
        estimate_pc, truth_pc = xr.align(
            estimate_pc,
            truth_pc,
            join="exact",
            copy=False,
            exclude={"mode"},
        )
    except ValueError as exc:
        raise ValueError("estimate and truth coefficient times must match exactly") from exc
    temporal = canonical_dimensions(temporal_mask)
    for dim in list(temporal.dims):
        if dim != "time":
            temporal = temporal.any(str(dim))
    try:
        temporal, _ = xr.align(temporal, estimate_pc["time"], join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("known_truth primary-mask time must match coefficients exactly") from exc
    observable = np.ones(truth_pc.sizes["mode"], dtype=bool)
    if observable_modes is not None:
        observable_data = canonical_dimensions(observable_modes)
        if observable_data.dims != ("mode",):
            raise ValueError("known_truth observable mode variable must be one-dimensional")
        try:
            observable_data, _ = xr.align(
                observable_data,
                truth_pc["mode"],
                join="exact",
                copy=False,
            )
        except ValueError as exc:
            raise ValueError(
                "known_truth observable mode coordinates must match truth coefficients exactly"
            ) from exc
        observable = np.asarray(observable_data.values, dtype=bool)

    metrics: list[FieldMetrics | None] = []
    for match in matches:
        if not observable[match.truth_index]:
            metrics.append(None)
            continue
        left = match.scale * estimate_pc.isel(mode=match.estimate_index)
        right = truth_pc.isel(mode=match.truth_index)
        dummy_lat = [0.0]
        left = left.expand_dims(lat=dummy_lat)
        right = right.expand_dims(lat=dummy_lat)
        metrics.append(weighted_field_metrics(left, right, temporal))
    output.coords["matched_mode"] = np.arange(1, len(matches) + 1)
    output["estimate_mode_index"] = ("matched_mode", [value.estimate_index for value in matches])
    output["truth_mode_index"] = ("matched_mode", [value.truth_index for value in matches])
    output["mode_sign"] = ("matched_mode", [value.sign for value in matches])
    output["basis_mode_similarity"] = ("matched_mode", [value.similarity for value in matches])
    output["basis_mode_scale_to_truth"] = ("matched_mode", [value.scale for value in matches])
    output["matched_mode_observable"] = (
        "matched_mode",
        [observable[value.truth_index] for value in matches],
    )
    output["coefficient_correlation"] = (
        "matched_mode",
        [value.correlation if value is not None else np.nan for value in metrics],
    )
    output["coefficient_origin_slope"] = (
        "matched_mode",
        [value.origin_slope if value is not None else np.nan for value in metrics],
    )
    output["coefficient_bias"] = (
        "matched_mode",
        [value.bias if value is not None else np.nan for value in metrics],
    )
    output["coefficient_nrmse"] = (
        "matched_mode",
        [value.nrmse if value is not None else np.nan for value in metrics],
    )
