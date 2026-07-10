"""Synthetic-only integration coverage for the FABLE aerosol tuning chain."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.runner import PipelineResult, PipelineRunner
from davinci_monet.tests.synthetic.aerosol_tuning import (
    DEFAULT_AEROSOL_SPECIES,
    SyntheticTuningSpec,
    generate_aerosol_tuning_bundle,
    log_time_interpolation_oracle,
    optical_aod_oracle,
    periodic_bilinear_oracle,
)


@dataclass(frozen=True)
class _SyntheticRun:
    root: Path
    truth: xr.Dataset
    result: PipelineResult
    manifest: dict[str, Any]


@pytest.fixture(scope="module")
def synthetic_writer_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_SyntheticRun]:
    """Generate and run the expensive serialized six-year chain once."""
    root = tmp_path_factory.mktemp("fable-writer")
    bundle = generate_aerosol_tuning_bundle(root, SyntheticTuningSpec.writer_ci())
    truth = bundle.truth
    del bundle

    config_path = (
        Path(__file__).parents[3]
        / "analyses"
        / "aerosol-tuning"
        / "configs"
        / "fable-synthetic.example.yaml"
    )
    previous_root = os.environ.get("FABLE_SYNTH")
    os.environ["FABLE_SYNTH"] = str(root)
    try:
        result = PipelineRunner(
            show_progress=False,
            close_datasets_after_run=False,
        ).run_from_config(str(config_path))
    finally:
        if previous_root is None:
            os.environ.pop("FABLE_SYNTH", None)
        else:
            os.environ["FABLE_SYNTH"] = previous_root

    manifest_path = root / "output" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = _SyntheticRun(root=root, truth=truth, result=result, manifest=manifest)
    yield run

    if result.context is not None:
        for source in result.context.sources.values():
            source.data.close()
    truth.close()


def _dataset(run: _SyntheticRun, source: str) -> xr.Dataset:
    assert run.result.context is not None
    return run.result.context.sources[source].data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_collection_checksums(entry: dict[str, Any]) -> None:
    root = Path(entry["artifact_dir"])
    expected_files = entry["checksums"]["files"]
    assert {path.name for path in root.glob("chunk-*.nc")} == set(expected_files)
    for filename, expected in expected_files.items():
        assert _sha256(root / filename) == expected
    summary = Path(entry["summary_path"])
    assert _sha256(summary) == entry["checksums"]["summary_sha256"]

    collection = hashlib.sha256()
    for filename, checksum in sorted(expected_files.items()):
        collection.update(filename.encode("utf-8"))
        collection.update(checksum.encode("ascii"))
    assert collection.hexdigest() == entry["checksums"]["collection_sha256"]


_PROJECTION_FIT_VARIABLES = (
    "clim_bias_raw_mean",
    "clim_bias",
    "clim_bias_applied",
    "spatial_support",
    "support_fraction",
    "support_count",
    "support_day_total",
    "clim_bias_sensor_count",
    "clim_bias_standard_error",
)


def _fit_artifact(run: _SyntheticRun, role: str) -> tuple[Path, dict[str, Any]]:
    entries = [entry for entry in run.manifest["analysis_artifacts"] if entry["role"] == role]
    assert len(entries) == 1
    entry = entries[0]
    filenames = sorted(entry["checksums"]["files"])
    assert filenames
    path = Path(entry["artifact_dir"]) / filenames[0]
    assert _sha256(path) == entry["checksums"]["files"][path.name]
    return path, entry


def _saved_fit_config(
    run: _SyntheticRun,
    output_name: str,
    *,
    projection_fit: Path,
) -> dict[str, Any]:
    assert run.result.context is not None
    config = deepcopy(run.result.context.config_dict())
    config["analysis"]["output_dir"] = run.root / output_name
    config["analysis"]["log_dir"] = run.root / f"{output_name}-logs"
    basis_path, _basis_entry = _fit_artifact(run, "basis_fit")
    manifest_path = run.root / "output" / "manifest.json"
    config["sources"]["frozen_aod_basis"] = {
        "type": "generic",
        "files": str(basis_path),
        "artifact_manifest": str(manifest_path),
        "artifact_role": "basis_fit",
        "artifact_analysis": "aod_basis",
        "variables": {"eofs": {}},
    }
    config["sources"]["frozen_projection_fit"] = {
        "type": "generic",
        "files": str(projection_fit),
        "artifact_manifest": str(manifest_path),
        "artifact_role": "projection_fit",
        "artifact_analysis": "obs_pcs",
        "variables": {name: {} for name in _PROJECTION_FIT_VARIABLES},
    }

    analyses = {
        name: spec
        for name, spec in config["analyses"].items()
        if name not in {"aod_basis", "corrected"}
    }
    projection = analyses["obs_pcs"]
    projection["basis"] = "frozen_aod_basis"
    projection.pop("bias_fit_window")
    projection["bias_fit_artifact"] = "frozen_projection_fit"
    analyses["scaling"]["basis"] = "frozen_aod_basis"
    config["analyses"] = analyses
    return config


def _close_result(result: PipelineResult) -> None:
    if result.context is None:
        return
    for source in result.context.sources.values():
        source.data.close()


def _minimum_subspace_cosine(
    estimate: xr.DataArray, truth: xr.DataArray, latitude: xr.DataArray
) -> float:
    estimated_values = np.asarray(estimate.transpose("mode", "lat", "lon"), dtype=np.float64)
    truth_values = np.asarray(
        truth.transpose("truth_mode", "mode_lat", "mode_lon"), dtype=np.float64
    )
    weights = np.cos(np.deg2rad(np.asarray(latitude, dtype=np.float64)))[:, None]
    weights = np.broadcast_to(weights, truth_values.shape[1:]).reshape(-1)
    sqrt_weight = np.sqrt(weights)

    def orthonormal_columns(patterns: np.ndarray) -> np.ndarray:
        weighted = patterns.reshape(patterns.shape[0], -1) * sqrt_weight[None, :]
        return np.linalg.qr(weighted.T)[0]

    overlap = orthonormal_columns(estimated_values).T @ orthonormal_columns(truth_values)
    return float(np.linalg.svd(overlap, compute_uv=False).min())


def _weighted_rmse(estimate: xr.DataArray, truth: xr.DataArray, mask: xr.DataArray) -> float:
    estimated_values = np.asarray(estimate, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    valid &= np.isfinite(estimated_values) & np.isfinite(truth_values)
    weights = np.cos(np.deg2rad(np.asarray(truth["lat"], dtype=np.float64)))
    weights = np.broadcast_to(weights[None, :, None], truth_values.shape)
    return float(
        np.sqrt(
            np.sum(weights[valid] * np.square(estimated_values[valid] - truth_values[valid]))
            / np.sum(weights[valid])
        )
    )


@pytest.mark.integration
def test_aerosol_tuning_known_mode_chain(synthetic_writer_run: _SyntheticRun) -> None:
    """T1: run preprocess through scaling with real serialization boundaries."""
    run = synthetic_writer_run
    result = run.result
    assert result.success, result.stage_errors
    assert result.stage_errors == {}
    assert result.context is not None

    expected_geometry = {
        "model_daily": DataGeometry.GRID,
        "sensor_a_daily": DataGeometry.GRID,
        "sensor_b_daily": DataGeometry.GRID,
        "aod_basis": DataGeometry.GRID,
        "obs_pcs": DataGeometry.GRID,
        "filtered_pcs": DataGeometry.SPECTRUM,
        "scaling": DataGeometry.GRID,
        "corrected": DataGeometry.ARTIFACT,
    }
    assert result.context.metadata["analysis_status"] == {
        name: "completed" for name in expected_geometry
    }
    for name, geometry in expected_geometry.items():
        source = result.context.sources[name]
        assert source.geometry is geometry
        assert source.data.attrs["derived"] is True

    assert run.manifest["status"] == "completed"
    assert run.manifest["failed_stages"] == []
    assert run.manifest["errors"] == {}
    assert run.manifest["analysis_dependency_blocked"] == []
    assert run.manifest["analysis_status"] == result.context.metadata["analysis_status"]

    collection_entries = [
        entry
        for entry in run.manifest["analysis_artifacts"]
        if entry["kind"] == "netcdf_collection"
    ]
    assert {entry["role"] for entry in collection_entries} >= {
        "projection_fit",
        "scaling",
    }
    for entry in collection_entries:
        _assert_collection_checksums(entry)
        assert entry["status"] == "finalized"
        for bucket in ("source_hashes", "config_hashes", "code_hashes"):
            assert entry["identity"][bucket]
            assert all(entry["identity"][bucket].values())

    scaling = _dataset(run, "scaling")
    model = _dataset(run, "model_daily")["aod"]
    ratio = np.asarray(scaling["r"], dtype=np.float64)
    target = np.asarray(scaling["aod_target"], dtype=np.float64)
    model_values = np.asarray(model, dtype=np.float64)
    np.testing.assert_array_equal(target, model_values * ratio)

    epsilon = float(scaling.attrs["log_epsilon"])
    expected_delta = np.log((target + epsilon) / (model_values + epsilon))
    np.testing.assert_allclose(scaling["delta_log_applied"], expected_delta, rtol=0.0, atol=5.0e-15)

    support = np.asarray(scaling["spatial_support"], dtype=np.float64)
    unsupported = support <= 0.0
    assert np.count_nonzero(unsupported) > 0
    np.testing.assert_array_equal(ratio[unsupported], np.ones(np.count_nonzero(unsupported)))
    np.testing.assert_array_equal(
        np.asarray(scaling["delta_log_applied"])[unsupported],
        np.zeros(np.count_nonzero(unsupported)),
    )
    np.testing.assert_array_equal(target[unsupported], model_values[unsupported])
    np.testing.assert_array_equal(scaling["support_identity_mask"], unsupported)

    basis = _dataset(run, "aod_basis")["eofs"]
    minimum_cosine = _minimum_subspace_cosine(
        basis, run.truth["pattern_true"], run.truth["mode_lat"]
    )
    assert minimum_cosine > 0.995

    development_time = run.truth["time"].where(run.truth["split"] == "development_test", drop=True)
    development_scaling = scaling.sel(time=development_time)
    truth_target = (
        run.truth["aod_filter_target_true"]
        .rename(mode_lat="lat", mode_lon="lon")
        .sel(time=development_time)
    )
    development_model = model.sel(time=development_time)
    supported = development_scaling["spatial_support"] > 0.0
    corrected_rmse = _weighted_rmse(development_scaling["aod_target"], truth_target, supported)
    baseline_rmse = _weighted_rmse(development_model, truth_target, supported)
    assert corrected_rmse < 0.8 * baseline_rmse


@pytest.mark.integration
def test_aerosol_tuning_serialized_evaluation_persists_recovery_report(
    synthetic_writer_run: _SyntheticRun,
) -> None:
    """Evaluate frozen serialized outputs without losing full-domain recovery metrics."""
    run = synthetic_writer_run
    fit_paths = sorted((run.root / "output" / "artifacts").glob("**/*"))
    fit_checksums = {path: _sha256(path) for path in fit_paths if path.is_file()}
    config_path = (
        Path(__file__).parents[3]
        / "analyses"
        / "aerosol-tuning"
        / "configs"
        / "fable-synthetic-eval.example.yaml"
    )
    previous_root = os.environ.get("FABLE_SYNTH")
    os.environ["FABLE_SYNTH"] = str(run.root)
    try:
        evaluation = PipelineRunner(
            show_progress=False,
            close_datasets_after_run=False,
        ).run_from_config(str(config_path))
    finally:
        if previous_root is None:
            os.environ.pop("FABLE_SYNTH", None)
        else:
            os.environ["FABLE_SYNTH"] = previous_root

    try:
        assert evaluation.success, evaluation.stage_errors
        assert evaluation.context is not None
        report = evaluation.context.sources["recovery"].data
        primary = report.sel(stratum="primary").compute()
        full_domain = report.sel(stratum="full_domain").compute()
        assert primary["valid_count"].item() > 0
        assert full_domain["valid_count"].item() > primary["valid_count"].item()
        assert np.isfinite(primary["field_nrmse"].item())
        assert np.isfinite(full_domain["field_nrmse"].item())

        entries = [
            entry
            for entry in evaluation.context.metadata["analysis_artifacts"]
            if entry["role"] == "recovery_report"
        ]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["analysis"] == "recovery"
        assert entry["kind"] == "netcdf_collection"
        _assert_collection_checksums(entry)

        manifest = json.loads((run.root / "evaluation" / "manifest.json").read_text())
        manifest_entries = [
            candidate
            for candidate in manifest["analysis_artifacts"]
            if candidate["role"] == "recovery_report"
        ]
        assert manifest_entries == entries
        artifact_path = Path(entry["artifact_dir"]) / "chunk-00000.nc"
        with xr.open_dataset(artifact_path) as persisted:
            assert persisted.attrs["persistence_policy"] == "immutable_netcdf_collection"
            assert persisted.sel(stratum="primary")["valid_count"].item() > 0

        assert {path: _sha256(path) for path in fit_checksums} == fit_checksums
    finally:
        _close_result(evaluation)


def _native_ratio_oracle(scaling: xr.Dataset, native: xr.Dataset) -> np.ndarray:
    daily_log_ratio = periodic_bilinear_oracle(
        np.log(np.asarray(scaling["r"], dtype=np.float64)),
        scaling["lat"],
        scaling["lon"],
        native["lat"],
        native["lon"],
    )
    native_support = periodic_bilinear_oracle(
        np.asarray(scaling["spatial_support"], dtype=np.float64),
        scaling["lat"],
        scaling["lon"],
        native["lat"],
        native["lon"],
    )
    daily_log_ratio[native_support <= 0.0] = 0.0
    return log_time_interpolation_oracle(
        np.exp(daily_log_ratio), scaling["time"].values, native["time"].values
    )


def _raw_variable(variable: netCDF4.Variable) -> np.ndarray:
    variable.set_auto_maskandscale(False)
    return np.asarray(variable[:]).copy()


def _assert_attribute_equal(left: Any, right: Any) -> None:
    np.testing.assert_equal(np.asarray(left), np.asarray(right))


@pytest.mark.integration
def test_aerosol_tuning_writer_pipeline(synthetic_writer_run: _SyntheticRun) -> None:
    """T3: inspect complete-chain native outputs, provenance, and optical closure."""
    run = synthetic_writer_run
    scaling = _dataset(run, "scaling")
    corrected = _dataset(run, "corrected")
    output_paths = sorted(Path(value) for value in corrected["output_path"].values.tolist())
    dataset_hashes = {
        Path(path): checksum
        for path, checksum in zip(
            corrected["output_path"].values.tolist(),
            corrected["sha256"].values.tolist(),
        )
    }
    assert len(output_paths) == 2
    assert all(path.is_file() for path in output_paths)
    assert set(corrected["status"].values.tolist()) == {"written"}
    assert list((run.root / "corrected").glob(".*.tmp")) == []

    writer_entries = [
        entry for entry in run.manifest["analysis_artifacts"] if entry["role"] == "corrected_mmr"
    ]
    assert len(writer_entries) == 2
    entries_by_path = {Path(entry["path"]): entry for entry in writer_entries}
    assert set(entries_by_path) == set(output_paths)
    assert all(entry["status"] == "written" for entry in writer_entries)
    assert all(entry["statistics"]["active_correction_count"] > 0 for entry in writer_entries)
    assert all(
        entry["statistics"]["spatial_support_identity_count"] > 0 for entry in writer_entries
    )
    assert all(
        entry["statistics"]["outside_coverage_identity_count"] == 0 for entry in writer_entries
    )

    invalid_aerosol_values = 0
    for output_path in output_paths:
        entry = entries_by_path[output_path]
        input_path = Path(entry["input_path"])
        assert _sha256(input_path) == entry["checksums"]["input_sha256"]
        assert _sha256(output_path) == entry["checksums"]["output_sha256"]
        assert _sha256(output_path) == dataset_hashes[output_path]
        assert entry["checksums"]["scaling_sha256"] == corrected.attrs["scaling_sha256"]
        assert entry["checksums"]["config_sha256"] == corrected.attrs["config_sha256"]
        assert entry["checksums"]["code_sha256"] == corrected.attrs["code_sha256"]
        assert entry["scenario_hash"] == run.truth.attrs["spec_hash"]

        with xr.open_dataset(input_path) as opened:
            source = opened.load()
        with xr.open_dataset(output_path) as opened:
            output = opened.load()
        expected_ratio = _native_ratio_oracle(scaling, source)
        baseline_aod = np.asarray(optical_aod_oracle(source), dtype=np.float64)
        corrected_aod = np.asarray(optical_aod_oracle(output), dtype=np.float64)
        expected_aod = baseline_aod * expected_ratio
        finite_aod = np.isfinite(corrected_aod) & np.isfinite(expected_aod)
        np.testing.assert_allclose(
            corrected_aod[finite_aod],
            expected_aod[finite_aod],
            rtol=5.0e-6,
            atol=1.0e-10,
        )

        with netCDF4.Dataset(input_path) as source_nc, netCDF4.Dataset(output_path) as output_nc:
            assert set(source_nc.variables) == set(output_nc.variables)
            for attribute in source_nc.ncattrs():
                _assert_attribute_equal(
                    source_nc.getncattr(attribute), output_nc.getncattr(attribute)
                )
            for key in ("input_sha256", "scaling_sha256", "config_sha256", "code_sha256"):
                assert output_nc.getncattr(f"davinci_{key}") == entry["checksums"][key]
            assert output_nc.getncattr("davinci_scenario_hash") == entry["scenario_hash"]

            for name, source_variable in source_nc.variables.items():
                output_variable = output_nc.variables[name]
                assert output_variable.dtype == source_variable.dtype
                assert output_variable.dimensions == source_variable.dimensions
                assert output_variable.chunking() == source_variable.chunking()
                assert output_variable.filters() == source_variable.filters()
                assert set(output_variable.ncattrs()) == set(source_variable.ncattrs())
                for attribute in source_variable.ncattrs():
                    _assert_attribute_equal(
                        source_variable.getncattr(attribute),
                        output_variable.getncattr(attribute),
                    )

                source_values = _raw_variable(source_variable)
                output_values = _raw_variable(output_variable)
                if name not in DEFAULT_AEROSOL_SPECIES:
                    np.testing.assert_array_equal(output_values, source_values)
                    continue

                fill_value = source_variable.getncattr("_FillValue")
                invalid = ~np.isfinite(source_values) | (source_values == fill_value)
                invalid_aerosol_values += int(invalid.sum())
                np.testing.assert_array_equal(output_values[invalid], source_values[invalid])
                expected_values = source_values * expected_ratio[:, None, :, :]
                np.testing.assert_allclose(
                    output_values[~invalid],
                    expected_values[~invalid],
                    rtol=5.0e-6,
                    atol=0.0,
                )

    assert invalid_aerosol_values > 0


@pytest.mark.integration
def test_aerosol_tuning_saved_fit_fresh_runner(
    synthetic_writer_run: _SyntheticRun,
) -> None:
    """T6: reuse finalized fits on the full axis without refitting."""
    run = synthetic_writer_run
    basis_path, _basis_entry = _fit_artifact(run, "basis_fit")
    projection_path, _projection_entry = _fit_artifact(run, "projection_fit")
    assert basis_path.name == "chunk-00000.nc"
    assert projection_path.name == "chunk-00000.nc"

    fresh_config = _saved_fit_config(
        run,
        "saved-fit-output",
        projection_fit=projection_path,
    )
    fresh = PipelineRunner(
        show_progress=False,
        close_datasets_after_run=False,
    ).run_from_config(fresh_config)
    try:
        assert fresh.success, fresh.stage_errors
        assert fresh.context is not None
        assert fresh.context.metadata["analysis_status"] == {
            "model_daily": "completed",
            "sensor_a_daily": "completed",
            "sensor_b_daily": "completed",
            "obs_pcs": "completed",
            "filtered_pcs": "completed",
            "scaling": "completed",
        }
        assert "aod_basis" not in fresh.context.metadata["analysis_status"]
        assert fresh.context.sources["frozen_aod_basis"].source_type == "generic"

        projection = fresh.context.sources["obs_pcs"].data
        assert projection.attrs["projection_bias_fit_selection"] == "artifact"
        np.testing.assert_array_equal(
            projection["time"],
            _dataset(run, "model_daily")["time"],
        )
        assert projection.sizes["time"] == 2191
        fresh_fit_roles = {entry["role"] for entry in fresh.context.metadata["analysis_artifacts"]}
        assert "basis_fit" not in fresh_fit_roles
        assert "projection_fit" not in fresh_fit_roles

        expected = _dataset(run, "scaling").sel(time=slice("2006-01-01", "2006-12-31"))
        actual = fresh.context.sources["scaling"].data.sel(time=slice("2006-01-01", "2006-12-31"))
        for variable in ("r", "delta_log_applied", "aod_target", "spatial_support"):
            np.testing.assert_array_equal(actual[variable], expected[variable])
    finally:
        _close_result(fresh)

    with xr.open_dataset(projection_path) as opened:
        mismatched_fit = opened[list(_PROJECTION_FIT_VARIABLES)].load()
    mismatched_fit.attrs["source_spec_hash"] = "mismatched-synthetic-scenario"
    mismatched_path = run.root / "mismatched-projection-fit.nc"
    mismatched_fit.to_netcdf(mismatched_path, engine="netcdf4")
    mismatch_config = _saved_fit_config(
        run,
        "saved-fit-mismatch-output",
        projection_fit=mismatched_path,
    )
    mismatch = PipelineRunner(show_progress=False).run_from_config(mismatch_config)

    assert not mismatch.success
    assert mismatch.context is not None
    assert "analysis_status" not in mismatch.context.metadata
    assert "configured artifact files do not match" in str(
        mismatch.stage_errors.get("stage:load_sources", "")
    )
