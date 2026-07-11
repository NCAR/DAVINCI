"""Read-only plot sources for frozen FABLE synthetic acceptance results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.analysis.known_truth_metrics import (
    canonical_dimensions,
    weighted_field_metrics,
)
from davinci_monet.tests.synthetic.fable_acceptance_gate import evidence_gate
from davinci_monet.tests.synthetic.fable_acceptance_provenance import (
    recovery_artifact_path,
    validate_seed_receipts,
)


@dataclass(frozen=True)
class AcceptanceDiagnosticSource:
    """A compact diagnostic dataset and the immutable inputs behind it."""

    dataset: xr.Dataset
    acceptance_root: Path
    seeds: tuple[int, int, int]
    snapshot_time: np.datetime64


@dataclass(frozen=True)
class _SeedInputs:
    seed: int
    basis: xr.Dataset
    projection: xr.Dataset
    scaling: xr.Dataset
    truth: xr.Dataset
    recovery: xr.Dataset


def build_acceptance_diagnostic_source(
    acceptance_root: str | Path,
) -> AcceptanceDiagnosticSource:
    """Build matched, acceptance-consistent diagnostics without changing run artifacts."""
    root = Path(acceptance_root).expanduser().resolve()
    record_path = root / "acceptance.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in record.get("seeds", ()))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("diagnostics require one completed three-seed acceptance record")
    if record.get("mode") != "full":
        raise ValueError("diagnostics require a full acceptance run")
    runs = {int(run["seed"]): run for run in record.get("runs", ())}
    if set(runs) != set(seeds):
        raise ValueError("acceptance record runs do not match its locked seeds")
    for seed in seeds:
        run = runs[seed]
        recomputed = evidence_gate(run.get("fitting", {}), run.get("evaluation", {}))
        if not recomputed.get("passed") or recomputed != run.get("evidence_gate"):
            raise ValueError(f"seed {seed} does not have valid acceptance evidence")
        validate_seed_receipts(root, seed, run)

    inputs = [_load_seed_inputs(root, seed, runs[seed]) for seed in seeds]
    try:
        prepared = [_prepare_seed(seed_inputs) for seed_inputs in inputs]
    finally:
        for seed_inputs in inputs:
            _close_seed_inputs(seed_inputs)

    common_score_day = prepared[0]["score_day"].astype(bool)
    for prepared_seed in prepared[1:]:
        common_score_day = common_score_day & prepared_seed["score_day"].astype(bool)
    common_times = common_score_day["time"].where(common_score_day, drop=True)
    if not common_times.size:
        raise ValueError("acceptance seeds have no common primary-safe diagnostic day")
    snapshot_time = np.datetime64(common_times.values[common_times.size // 2], "ns")

    per_seed = [
        _finalize_seed(prepared_seed, snapshot_time).expand_dims(
            seed=[int(prepared_seed.attrs["root_seed"])]
        )
        for prepared_seed in prepared
    ]
    dataset = xr.concat(per_seed, dim="seed", join="exact", combine_attrs="drop_conflicts")
    dataset.attrs.update(
        {
            "geometry": "grid",
            "source_label": "fable_acceptance_diagnostics",
            "analysis_type": "fable_acceptance_diagnostics",
            "acceptance_status": str(record.get("status")),
            "acceptance_record_sha256": _sha256(record_path),
            "seed_lock_sha256": str(record["seed_lock"]["file"]["sha256"]),
            "selected_policy_id": str(record["calibration"]["selected_policy_id"]),
            "snapshot_time": str(snapshot_time),
            "snapshot_selection": "temporal median of common primary-safe evaluation days",
            "primary_mask": (
                "development_test and coefficient_available and observable-mode valid_segment "
                "and coi>=band_max and spatial_support>0 and finite estimate/truth"
            ),
            "diagnostic_disposition": "diagnostic only; frozen acceptance remains rejected",
        }
    )
    return AcceptanceDiagnosticSource(dataset, root, seeds, snapshot_time)


def write_acceptance_diagnostic_source(
    source: AcceptanceDiagnosticSource, destination: str | Path
) -> Path:
    """Write the compact source used by the pipeline diagnostic renderers."""
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    encoding = {
        name: {"zlib": True, "complevel": 1, "shuffle": True}
        for name, value in source.dataset.data_vars.items()
        if value.ndim and value.dtype.kind in "fiu"
    }
    source.dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


def acceptance_collection_config(
    acceptance_root: str | Path,
    seed: int,
    artifact: str,
    variables: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Return one manifest-validated source config for a frozen collection."""
    root = Path(acceptance_root).expanduser().resolve()
    details = {
        "projection": ("obs_pcs", "projection_fit"),
        "scaling": ("scaling", "scaling"),
    }
    try:
        analysis, role = details[artifact]
    except KeyError as exc:
        raise ValueError(f"unsupported acceptance collection: {artifact}") from exc
    seed_root = root / f"seed-{seed}"
    return {
        "type": "generic",
        "files": str(seed_root / f"output/artifacts/{analysis}/chunk-*.nc"),
        "artifact_manifest": str(seed_root / "output/manifest.json"),
        "artifact_role": role,
        "artifact_analysis": analysis,
        "combine": "nested",
        "concat_dim": "time",
        "data_vars": "minimal",
        "coords": "minimal",
        "compat": "override",
        "join": "exact",
        "variables": variables,
    }


def verify_wavelet_replay(
    replayed: xr.Dataset,
    acceptance_root: str | Path,
    seed: int,
) -> dict[str, list[float]]:
    """Require a diagnostic wavelet replay to match the persisted reconstruction."""
    root = Path(acceptance_root).expanduser().resolve()
    persisted = _open_collection(root / f"seed-{seed}/output/artifacts/scaling")
    try:
        expected = persisted[["pc", "coi", "valid_segment"]].load()
    finally:
        persisted.close()
    for coord in ("time", "mode"):
        if not np.array_equal(replayed[coord].values, expected[coord].values):
            raise ValueError(f"seed {seed} replayed wavelet {coord} coordinate changed")
    for name in ("pc", "coi"):
        if not np.array_equal(replayed[name].values, expected[name].values, equal_nan=True):
            raise ValueError(f"seed {seed} replayed wavelet {name} changed")
    if not np.array_equal(replayed["valid_segment"].values, expected["valid_segment"].values):
        raise ValueError(f"seed {seed} replayed wavelet valid_segment changed")
    return {
        name: [float(value) for value in np.asarray(replayed[name].values).reshape(-1)]
        for name in (
            "retained_variance",
            "recon_error",
            "synth_fraction",
            "coi_valid_fraction",
        )
    }


def _load_seed_inputs(root: Path, seed: int, run: Mapping[str, Any]) -> _SeedInputs:
    seed_root = root / f"seed-{seed}"
    basis = xr.open_dataset(seed_root / "output/artifacts/aod_basis/chunk-00000.nc")
    projection = _open_collection(seed_root / "output/artifacts/obs_pcs")
    scaling = _open_collection(seed_root / "output/artifacts/scaling")
    truth = xr.open_dataset(seed_root / "oracle/truth.nc")
    recovery = xr.open_dataset(recovery_artifact_path(seed_root, run))
    return _SeedInputs(seed, basis, projection, scaling, truth, recovery)


def _open_collection(root: Path) -> xr.Dataset:
    paths = sorted(root.glob("chunk-*.nc"))
    if not paths:
        raise FileNotFoundError(f"acceptance artifact collection is empty: {root}")
    return xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    )


def _close_seed_inputs(inputs: _SeedInputs) -> None:
    for dataset in (
        inputs.basis,
        inputs.projection,
        inputs.scaling,
        inputs.truth,
        inputs.recovery,
    ):
        dataset.close()


def _prepare_seed(inputs: _SeedInputs) -> xr.Dataset:
    split = inputs.truth["split"].load().astype(str)
    evaluation_time = split["time"].where(split == "development_test", drop=True)
    truth = inputs.truth[
        [
            "pattern_true",
            "mode_observable_true",
            "correction_pc_true",
            "correction_pc_filter_target_true",
            "delta_filter_target_true",
            "delta_best_representable_true",
        ]
    ].sel(time=evaluation_time)
    projection = inputs.projection[["pc", "resolution"]].sel(time=evaluation_time)
    scaling = inputs.scaling[
        [
            "pc",
            "valid_segment",
            "coi",
            "coefficient_available",
            "spatial_support",
            "delta_log_applied",
        ]
    ].sel(time=evaluation_time)
    truth, projection, scaling = xr.align(truth, projection, scaling, join="exact", copy=False)
    truth = truth.load()
    projection = projection.load()
    scaling = scaling.load()
    basis = inputs.basis[["eofs", "explained_variance"]].load()
    recovery = inputs.recovery.load()

    estimate_indices = np.asarray(recovery["estimate_mode_index"].values, dtype=int)
    truth_indices = np.asarray(recovery["truth_mode_index"].values, dtype=int)
    scales = np.asarray(recovery["basis_mode_scale_to_truth"].values, dtype=float)
    order = np.argsort(truth_indices)
    estimate_indices = estimate_indices[order]
    truth_indices = truth_indices[order]
    scales = scales[order]
    if np.any(scales == 0.0) or len(set(truth_indices.tolist())) != truth_indices.size:
        raise ValueError(f"seed {inputs.seed} has an invalid recovery mode match")

    truth_basis, aligned_basis = _align_matched_basis(
        basis["eofs"],
        canonical_dimensions(truth["pattern_true"]),
        estimate_indices,
        truth_indices,
        scales,
    )
    basis_residual = aligned_basis - truth_basis

    observable_truth = np.asarray(truth["mode_observable_true"].values, dtype=bool)
    observable = observable_truth[truth_indices]
    observable_estimate_indices = estimate_indices[observable]
    valid_segment = scaling["valid_segment"].isel(mode=observable_estimate_indices).all("mode")
    coi_safe = (
        scaling["coi"].isel(mode=observable_estimate_indices)
        >= float(inputs.scaling.attrs["band_max"])
    ).all("mode")
    score_day = scaling["coefficient_available"].astype(bool) & valid_segment & coi_safe

    truth_delta = canonical_dimensions(truth["delta_filter_target_true"])
    estimate_delta = scaling["delta_log_applied"]
    representable_delta = canonical_dimensions(truth["delta_best_representable_true"])
    truth_delta, estimate_delta, representable_delta = xr.align(
        truth_delta, estimate_delta, representable_delta, join="exact", copy=False
    )
    primary = _acceptance_primary_mask(
        score_day,
        scaling["spatial_support"],
        truth_delta,
        estimate_delta,
    )
    primary_count = int(primary.sum().item())
    primary_report = recovery.sel(stratum="primary")
    if primary_count != int(primary_report["valid_count"].item()):
        raise ValueError(f"seed {inputs.seed} diagnostic primary mask does not match acceptance")
    if truth_delta.size != int(primary_report["candidate_count"].item()):
        raise ValueError(f"seed {inputs.seed} diagnostic candidate count does not match acceptance")

    mode_coord = truth_basis["mode"]
    matched_scale = xr.DataArray(scales, dims="mode", coords={"mode": mode_coord})
    raw_pc = _matched_estimate_modes(projection["pc"], estimate_indices, mode_coord) * matched_scale
    filtered_pc = (
        _matched_estimate_modes(scaling["pc"], estimate_indices, mode_coord) * matched_scale
    )
    truth_pc = canonical_dimensions(truth["correction_pc_true"]).isel(mode=truth_indices)
    target_pc = canonical_dimensions(truth["correction_pc_filter_target_true"]).isel(
        mode=truth_indices
    )
    for value in (truth_pc, target_pc):
        value.coords["mode"] = mode_coord
    raw_eligible = _matched_estimate_modes(
        projection["pc"], estimate_indices, mode_coord
    ).notnull() & (
        _matched_estimate_modes(projection["resolution"], estimate_indices, mode_coord) >= 0.3
    )
    matched_valid = _matched_estimate_modes(scaling["valid_segment"], estimate_indices, mode_coord)
    matched_coi_safe = _matched_estimate_modes(
        scaling["coi"], estimate_indices, mode_coord
    ) >= float(inputs.scaling.attrs["band_max"])

    dataset = xr.Dataset(
        {
            "truth_eof": truth_basis,
            "learned_eof_aligned": aligned_basis,
            "eof_residual": basis_residual,
            "truth_correction_rms": _masked_rms(truth_delta, primary),
            "estimate_correction_rms": _masked_rms(estimate_delta, primary),
            "residual_correction_rms": _masked_rms(estimate_delta - truth_delta, primary),
            "best_representable_rms": _masked_rms(representable_delta, primary),
            "primary_valid_days": primary.sum("time"),
            "raw_projected_pc": raw_pc,
            "raw_truth_pc": truth_pc,
            "wavelet_reconstruction_pc": filtered_pc,
            "wavelet_truth_target_pc": target_pc,
            "raw_eligible": raw_eligible,
            "wavelet_valid_segment": matched_valid,
            "wavelet_coi_safe": matched_coi_safe,
            "score_day": score_day,
            "mode_observable": ("mode", observable),
            "mode_similarity": (
                "mode",
                np.asarray(recovery["basis_mode_similarity"].values, dtype=float)[order],
            ),
            "basis_scale_to_truth": ("mode", scales),
            "explained_variance": _matched_estimate_modes(
                basis["explained_variance"], estimate_indices, mode_coord
            ),
            "coefficient_correlation": (
                "mode",
                np.asarray(recovery["coefficient_correlation"].values, dtype=float)[order],
            ),
            "coefficient_origin_slope": (
                "mode",
                np.asarray(recovery["coefficient_origin_slope"].values, dtype=float)[order],
            ),
            "coefficient_nrmse": (
                "mode",
                np.asarray(recovery["coefficient_nrmse"].values, dtype=float)[order],
            ),
            "field_nrmse": primary_report["field_nrmse"],
            "primary_valid_count": primary_report["valid_count"],
            "primary_candidate_count": primary_report["candidate_count"],
            "subspace_angle_mean_degrees": recovery["subspace_angle_mean_degrees"],
            "subspace_angle_max_degrees": recovery["subspace_angle_max_degrees"],
            "subspace_projector_error": recovery["subspace_projector_error"],
            "_truth_delta": truth_delta,
            "_estimate_delta": estimate_delta,
            "_primary_mask": primary,
        }
    )
    dataset.attrs.update(
        {
            "root_seed": inputs.seed,
            "band_max_days": float(inputs.scaling.attrs["band_max"]),
        }
    )
    return dataset


def _finalize_seed(dataset: xr.Dataset, snapshot_time: np.datetime64) -> xr.Dataset:
    seed = int(dataset.attrs["root_seed"])
    score_day = bool(dataset["score_day"].sel(time=snapshot_time).item())
    if not score_day:
        raise ValueError(f"seed {seed} is not primary-safe on the common snapshot day")

    for private_name in ("_truth_delta", "_estimate_delta", "_primary_mask"):
        if private_name not in dataset:
            raise ValueError(f"seed {seed} diagnostic staging is missing {private_name}")
    truth_snapshot = dataset["_truth_delta"].sel(time=snapshot_time)
    estimate_snapshot = dataset["_estimate_delta"].sel(time=snapshot_time)
    primary_snapshot = dataset["_primary_mask"].sel(time=snapshot_time).astype(bool)
    truth_snapshot = truth_snapshot.where(primary_snapshot)
    estimate_snapshot = estimate_snapshot.where(primary_snapshot)
    metrics = weighted_field_metrics(
        estimate_snapshot.expand_dims(time=[snapshot_time]),
        truth_snapshot.expand_dims(time=[snapshot_time]),
        primary_snapshot.expand_dims(time=[snapshot_time]),
    )
    output = dataset.drop_vars(["_truth_delta", "_estimate_delta", "_primary_mask"])
    output["truth_snapshot"] = truth_snapshot
    output["estimate_snapshot"] = estimate_snapshot
    output["residual_snapshot"] = estimate_snapshot - truth_snapshot
    output["snapshot_valid_count"] = xr.DataArray(metrics.valid_count)
    output["snapshot_rmse"] = xr.DataArray(metrics.rmse)
    output["snapshot_nrmse"] = xr.DataArray(metrics.nrmse)
    output["snapshot_active_domain"] = primary_snapshot
    return output


def _masked_rms(field: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    valid = field.where(mask)
    rms = (valid * valid).mean("time", skipna=True) ** 0.5
    return rms.where(mask.any("time"))


def _align_matched_basis(
    estimate_basis: xr.DataArray,
    truth_basis: xr.DataArray,
    estimate_indices: np.ndarray,
    truth_indices: np.ndarray,
    scales: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return truth modes and learned modes in the oracle's signed normalization."""
    selected_truth = canonical_dimensions(truth_basis).isel(mode=truth_indices)
    selected_estimate = canonical_dimensions(estimate_basis).isel(mode=estimate_indices)
    scale = xr.DataArray(
        scales,
        dims=("mode",),
        coords={"mode": selected_estimate["mode"]},
    )
    aligned = (selected_estimate / scale).assign_coords(mode=selected_truth["mode"])
    return selected_truth, aligned


def _matched_estimate_modes(
    values: xr.DataArray,
    estimate_indices: np.ndarray,
    truth_mode: xr.DataArray,
) -> xr.DataArray:
    """Select estimate modes in match order and label them with truth modes."""
    return values.isel(mode=estimate_indices).assign_coords(mode=truth_mode)


def _acceptance_primary_mask(
    score_day: xr.DataArray,
    support: xr.DataArray,
    truth: xr.DataArray,
    estimate: xr.DataArray,
) -> xr.DataArray:
    """Apply the spatial and finite portions of the frozen primary mask."""
    return score_day.broadcast_like(truth) & (support > 0.0) & truth.notnull() & estimate.notnull()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "AcceptanceDiagnosticSource",
    "acceptance_collection_config",
    "build_acceptance_diagnostic_source",
    "verify_wavelet_replay",
    "write_acceptance_diagnostic_source",
]
