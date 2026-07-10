"""Tests for read-only synthetic known-truth recovery metrics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import davinci_monet.analysis.known_truth as known_truth_module
from davinci_monet.analysis.artifacts import ArtifactService
from davinci_monet.analysis.base import AnalysisRuntime
from davinci_monet.analysis.known_truth import (
    KnownTruthAnalysis,
    evaluate_known_truth,
    match_weighted_modes,
    weighted_field_metrics,
    weighted_subspace_metrics,
)
from davinci_monet.core.protocols import DataGeometry


def _field(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange("2006-01-01", "2006-01-03", dtype="datetime64[D]"),
            "lat": [-60.0, 0.0],
            "lon": [0.0, 180.0],
        },
    )


def test_weighted_field_metrics_are_exact_for_identical_fields() -> None:
    truth = _field(np.arange(8, dtype=float).reshape(2, 2, 2))

    metrics = weighted_field_metrics(truth.copy(), truth)

    assert metrics.correlation == 1.0
    assert metrics.origin_slope == 1.0
    assert metrics.bias == 0.0
    assert metrics.rmse == 0.0
    assert metrics.nrmse == 0.0
    assert metrics.valid_count == 8
    assert metrics.excluded_fraction == 0.0


def test_weighted_field_metrics_report_offset_and_mask_exclusions() -> None:
    truth = _field(np.arange(1, 9, dtype=float).reshape(2, 2, 2))
    estimate = truth + 2.0
    estimate.values[0, 0, 0] = 10_000.0
    mask = xr.ones_like(truth, dtype=bool)
    mask.values[0, 0, 0] = False

    metrics = weighted_field_metrics(estimate, truth, mask)

    assert np.isclose(metrics.correlation, 1.0)
    assert np.isclose(metrics.bias, 2.0)
    assert np.isclose(metrics.rmse, 2.0)
    assert metrics.valid_count == 7
    assert metrics.candidate_count == 8
    assert metrics.excluded_fraction == 1.0 / 8.0
    assert metrics.origin_slope > 1.0
    assert metrics.nrmse > 0.0


def test_nonfinite_estimate_counts_as_an_exclusion_from_finite_truth() -> None:
    truth = _field(np.arange(1, 9, dtype=float).reshape(2, 2, 2))
    estimate = truth.copy()
    estimate.values[0, 0, 0] = np.nan

    metrics = weighted_field_metrics(estimate, truth)

    assert metrics.candidate_count == 8
    assert metrics.valid_count == 7
    assert metrics.excluded_fraction == 1.0 / 8.0


def _bases() -> tuple[xr.DataArray, xr.DataArray]:
    coords = {"lat": [-45.0, 45.0], "lon": [0.0, 180.0]}
    truth = xr.DataArray(
        [[[1.0, -1.0], [1.0, -1.0]], [[1.0, 1.0], [-1.0, -1.0]]],
        dims=("truth_mode", "mode_lat", "mode_lon"),
        coords={"truth_mode": [1, 2], "mode_lat": coords["lat"], "mode_lon": coords["lon"]},
    )
    estimate = xr.DataArray(
        np.stack((-truth.values[1], truth.values[0])),
        dims=("mode", "lat", "lon"),
        coords={"mode": [1, 2], **coords},
    )
    return estimate, truth


def test_subspace_and_mode_matching_ignore_sign_and_permutation() -> None:
    estimate, truth = _bases()

    subspace = weighted_subspace_metrics(estimate, truth)
    matches = match_weighted_modes(estimate, truth)

    np.testing.assert_allclose(subspace.angles_degrees, 0.0, atol=1.0e-6)
    assert subspace.projector_error < 1.0e-12
    assert [(item.estimate_index, item.truth_index, item.sign) for item in matches] == [
        (0, 1, -1.0),
        (1, 0, 1.0),
    ]
    np.testing.assert_allclose([item.similarity for item in matches], 1.0)
    np.testing.assert_allclose([item.scale for item in matches], [-1.0, 1.0])


def test_subspace_mask_coordinates_must_match_basis_exactly() -> None:
    estimate, truth = _bases()
    mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("lat", "lon"),
        coords={"lat": estimate["lat"], "lon": [0.0, 179.0]},
    )

    with pytest.raises(ValueError, match="subspace mask coordinates must match"):
        weighted_subspace_metrics(estimate, truth, mask)


def _evaluation_inputs() -> tuple[xr.Dataset, xr.Dataset, SimpleNamespace]:
    estimate_basis, truth_basis = _bases()
    delta = _field(np.arange(1, 9, dtype=float).reshape(2, 2, 2) / 20.0)
    target_aod = _field(np.arange(1, 9, dtype=float).reshape(2, 2, 2) / 10.0)
    support = xr.ones_like(delta)
    support.values[:, 0, 0] = 0.0
    truth_pc = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("time", "truth_mode"),
        coords={"time": delta["time"], "truth_mode": [1, 2]},
    )
    estimate_pc = xr.DataArray(
        np.column_stack((-truth_pc.values[:, 1], truth_pc.values[:, 0])),
        dims=("time", "mode"),
        coords={"time": delta["time"], "mode": [1, 2]},
    )
    estimate = xr.Dataset(
        {
            "delta_log_applied": delta,
            "aod_target": target_aod,
            "eofs": estimate_basis,
            "pc": estimate_pc,
            "spatial_support": support,
            "resolution": xr.DataArray(
                [[0.2, 0.4], [0.8, 0.9]],
                dims=("time", "mode"),
                coords={"time": delta["time"], "mode": [1, 2]},
            ),
            "coefficient_available": xr.DataArray(
                [True, True], dims=("time",), coords={"time": delta["time"]}
            ),
            "clip_reason": xr.zeros_like(delta, dtype=np.int8),
        }
    )
    estimate["clip_reason"].loc[{"time": delta["time"].values[1], "lat": 0.0, "lon": 180.0}] = 1
    truth = xr.Dataset(
        {
            "delta_filter_target_true": delta.rename({"lat": "mode_lat", "lon": "mode_lon"}),
            "delta_applied_true": (2.0 * delta).rename({"lat": "mode_lat", "lon": "mode_lon"}),
            "delta_in_span_true": (2.0 * delta).rename({"lat": "mode_lat", "lon": "mode_lon"}),
            "delta_perp_true": xr.zeros_like(delta).rename({"lat": "mode_lat", "lon": "mode_lon"}),
            "delta_best_representable_true": delta.rename({"lat": "mode_lat", "lon": "mode_lon"}),
            "aod_filter_target_true": target_aod.rename({"lat": "mode_lat", "lon": "mode_lon"}),
            "aod_target_applied_true": target_aod.rename({"lat": "mode_lat", "lon": "mode_lon"}),
            "model_aod_overpass_true": (target_aod + 0.1).rename(
                {"lat": "mode_lat", "lon": "mode_lon"}
            ),
            "obs_holdout_aod": target_aod.rename({"lat": "mode_lat", "lon": "mode_lon"}),
            "pattern_true": truth_basis,
            "correction_pc_filter_target_true": truth_pc,
            "mode_observable_true": xr.DataArray(
                [1, 1], dims=("truth_mode",), coords={"truth_mode": [1, 2]}
            ),
            "split": xr.DataArray(
                ["development_test", "development_test"],
                dims=("time",),
                coords={"time": delta["time"]},
            ),
        }
    )
    spec = SimpleNamespace(evaluation_splits=["development_test"])
    return estimate, truth, spec


def test_evaluator_emits_strata_basis_and_coefficient_metrics() -> None:
    estimate, truth, spec = _evaluation_inputs()

    report = evaluate_known_truth(estimate, truth, spec)

    assert report.attrs["evaluation_only"] == "true"
    assert report.sel(stratum="primary")["field_nrmse"].item() == 0.0
    assert report.sel(stratum="primary")["aod_rmse_ratio"].item() == 0.0
    assert report.sel(stratum="primary")["full_target_aod_rmse_ratio"].item() == 0.0
    assert report.sel(stratum="primary")["excluded_fraction"].item() == 0.25
    assert report.sel(stratum="primary")["off_basis_floor_nrmse"].item() == pytest.approx(0.5)
    assert report.sel(stratum="primary")["full_delta_nrmse"].item() == pytest.approx(0.5)
    assert report.sel(stratum="primary")["perpendicular_truth_rms"].item() == 0.0
    assert report.sel(stratum="primary")["perpendicular_to_full_rms_ratio"].item() == 0.0
    assert report.sel(stratum="primary")["clip_fraction"].item() == pytest.approx(1.0 / 6.0)
    assert report.sel(stratum="primary")["holdout_aod_rmse_ratio"].item() == 0.0
    assert "resolution_low" in report["stratum"].values
    assert "season_DJF" in report["stratum"].values
    assert report["subspace_projector_error"].item() < 1.0e-12
    np.testing.assert_allclose(report["coefficient_correlation"], 1.0)
    np.testing.assert_allclose(report["coefficient_origin_slope"], 1.0)
    np.testing.assert_allclose(report["coefficient_bias"], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(report["coefficient_nrmse"], 0.0, atol=1.0e-14)


def test_coefficient_metrics_crosswalk_eof_loading_amplitude() -> None:
    estimate, truth, spec = _evaluation_inputs()
    loading_scale = xr.DataArray([2.0, 4.0], dims=("mode",), coords={"mode": [1, 2]})
    estimate["eofs"] = estimate["eofs"] * loading_scale
    estimate["pc"] = estimate["pc"] / loading_scale

    report = evaluate_known_truth(estimate, truth, spec)

    np.testing.assert_allclose(np.abs(report["basis_mode_scale_to_truth"]), [2.0, 4.0])
    np.testing.assert_allclose(report["coefficient_origin_slope"], 1.0)
    np.testing.assert_allclose(report["coefficient_nrmse"], 0.0, atol=1.0e-14)


def test_evaluator_accepts_lazy_artifact_inputs() -> None:
    estimate, truth, spec = _evaluation_inputs()
    lazy_estimate = estimate.chunk({"time": 1})
    lazy_truth = truth.chunk({"time": 1})
    estimate_backing = {name: variable.data for name, variable in lazy_estimate.data_vars.items()}
    truth_backing = {name: variable.data for name, variable in lazy_truth.data_vars.items()}

    report = evaluate_known_truth(
        lazy_estimate,
        lazy_truth,
        spec,
    )

    assert report.sel(stratum="primary")["field_nrmse"].item() == 0.0
    assert lazy_estimate["delta_log_applied"].chunks is not None
    assert lazy_truth["delta_filter_target_true"].chunks is not None
    assert all(
        lazy_estimate.data_vars[name].data is data for name, data in estimate_backing.items()
    )
    assert all(lazy_truth.data_vars[name].data is data for name, data in truth_backing.items())


def test_evaluator_slices_requested_split_before_loading_scientific_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimate, truth, spec = _evaluation_inputs()
    truth["split"][0] = "calibration"
    loaded_time_sizes: list[int] = []
    original = known_truth_module._loaded_subset

    def observed_subset(dataset, variable_names):  # noqa: ANN001
        loaded_time_sizes.append(dataset.sizes["time"])
        return original(dataset, variable_names)

    monkeypatch.setattr(known_truth_module, "_loaded_subset", observed_subset)

    report = evaluate_known_truth(estimate.chunk({"time": 1}), truth.chunk({"time": 1}), spec)

    assert loaded_time_sizes == [1, 1]
    assert report.sel(stratum="primary")["valid_count"].item() == 3


def test_primary_uses_only_observable_modes_and_full_domain_keeps_exclusions() -> None:
    estimate, truth, spec = _evaluation_inputs()
    estimate["valid_segment"] = xr.DataArray(
        [[False, False], [False, True]],
        dims=("time", "mode"),
        coords={"time": estimate["time"], "mode": estimate["mode"]},
    )
    estimate["coi"] = xr.DataArray(
        [[np.nan, 32.0], [np.nan, 32.0]],
        dims=("time", "mode"),
        coords={"time": estimate["time"], "mode": estimate["mode"]},
    )
    estimate.attrs["band_max"] = 16.0
    truth["mode_observable_true"][:] = [1, 0]

    report = evaluate_known_truth(estimate, truth, spec)

    primary = report.sel(stratum="primary")
    full_domain = report.sel(stratum="full_domain")
    assert primary["valid_count"].item() == 3
    assert full_domain["valid_count"].item() == 8
    assert primary["excluded_fraction"].item() == 0.625
    assert full_domain["excluded_fraction"].item() == 0.0
    assert np.isfinite(primary["field_nrmse"].item())
    assert np.isfinite(full_domain["field_nrmse"].item())
    np.testing.assert_array_equal(report["matched_mode_observable"], [False, True])
    assert np.isnan(report["coefficient_nrmse"].isel(matched_mode=0).item())
    assert report["coefficient_nrmse"].isel(matched_mode=1).item() == pytest.approx(0.0)


def test_weighted_metrics_reject_missing_or_changed_axes() -> None:
    truth = _field(np.arange(8, dtype=float).reshape(2, 2, 2))

    with pytest.raises(ValueError, match="coordinates must match exactly"):
        weighted_field_metrics(truth.isel(time=[0]), truth)

    changed_longitude = truth.assign_coords(lon=[0.0, 179.0])
    with pytest.raises(ValueError, match="coordinates must match exactly"):
        weighted_field_metrics(changed_longitude, truth)


def test_evaluator_requires_requested_split_variable_and_values() -> None:
    estimate, truth, spec = _evaluation_inputs()
    missing_variable = truth.drop_vars("split")

    with pytest.raises(ValueError, match="requires split variable"):
        evaluate_known_truth(estimate, missing_variable, spec)

    unavailable = truth.copy()
    unavailable["split"][:] = "calibration"
    with pytest.raises(ValueError, match="requested split value.*development_test"):
        evaluate_known_truth(estimate, unavailable, spec)


def test_evaluator_rejects_truth_from_another_scenario() -> None:
    estimate, truth, spec = _evaluation_inputs()
    estimate.attrs["spec_hash"] = "scenario-a"
    truth.attrs["spec_hash"] = "scenario-b"

    with pytest.raises(ValueError, match="inconsistent scientific spec hashes"):
        evaluate_known_truth(estimate, truth, spec)


def test_pipeline_adapter_preserves_inputs_and_declares_immutable_report(tmp_path: Path) -> None:
    estimate, truth, spec = _evaluation_inputs()
    estimate_before = estimate.copy(deep=True)
    truth_before = truth.copy(deep=True)
    runtime = AnalysisRuntime(
        start_time=datetime(2006, 1, 1),
        end_time=datetime(2006, 1, 2),
        artifact_service=ArtifactService(tmp_path),
    )

    result = KnownTruthAnalysis().analyze_inputs(
        {"estimate": estimate, "truth": truth}, spec, runtime
    )

    assert KnownTruthAnalysis.output_geometry is DataGeometry.ARTIFACT
    assert len(result.artifacts) == 1
    assert result.artifacts[0].kind == "netcdf_collection"
    assert result.artifacts[0].role == "recovery_report"
    assert result.artifacts[0].reload is True
    assert result.manifest_entries == ()
    xr.testing.assert_identical(estimate, estimate_before)
    xr.testing.assert_identical(truth, truth_before)

    persisted = runtime.artifact_service.materialize("recovery", result.dataset, result.artifacts)
    artifact_path = tmp_path / "artifacts" / "recovery" / "chunk-00000.nc"
    assert artifact_path.is_file()
    assert persisted.dataset["valid_count"].chunks is not None
    assert persisted.manifest_entries[0]["role"] == "recovery_report"
    assert persisted.manifest_entries[0]["artifact_dir"] == str(artifact_path.parent)
    with xr.open_dataset(artifact_path) as artifact:
        assert artifact.attrs["persistence_policy"] == "immutable_netcdf_collection"
        assert artifact.sel(stratum="primary")["valid_count"].item() == 6

    repeated = runtime.artifact_service.materialize("recovery", result.dataset, result.artifacts)
    assert repeated.manifest_entries[0]["publication"] == "reused"
    assert repeated.manifest_entries[0]["checksums"] == persisted.manifest_entries[0]["checksums"]
