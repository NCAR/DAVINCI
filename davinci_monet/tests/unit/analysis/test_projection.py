"""Tests for complete-axis multi-sensor EOF innovation projection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import davinci_monet.analysis.projection_batches as projection_batches
from davinci_monet.analysis.artifacts import ArtifactService
from davinci_monet.analysis.base import AnalysisRuntime, ArtifactDeclaration
from davinci_monet.analysis.projection import EOFProjectionAnalysis, project_eof
from davinci_monet.analysis.projection_batches import fit_monthly_bias_batched
from davinci_monet.analysis.projection_core import fit_monthly_bias
from davinci_monet.analysis.projection_inputs import prepare_projection_inputs
from davinci_monet.config.schema import EOFProjectionSpec


def _inputs() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    time = pd.date_range("2001-01-01T12:00:00", periods=4, freq="1D")
    lat = np.array([-30.0, 30.0])
    lon = np.array([-90.0, 90.0])
    patterns = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, -1.0]],
        ]
    )
    basis = xr.Dataset(
        {"eofs": (("mode", "lat", "lon"), patterns)},
        coords={"mode": [1, 2], "lat": lat, "lon": lon},
        attrs={
            "eof_rotation": "none",
            "eof_standardize": "false",
            "eof_input_log_epsilon": 0.01,
        },
    )
    model = xr.Dataset(
        {"log_aod": (("time", "lat", "lon"), np.zeros((4, 2, 2)))},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"log_epsilon": 0.01},
    )
    coefficients = np.array([0.5, -0.25])
    day_two = 0.2 + np.einsum("k,kij->ij", coefficients, patterns)
    values = np.stack(
        [np.full((2, 2), 0.2), np.full((2, 2), 0.2), day_two, np.full((2, 2), np.nan)]
    )
    mask_a = np.zeros((4, 2, 2), dtype=bool)
    mask_b = np.zeros((4, 2, 2), dtype=bool)
    mask_a[:3, :, 0] = True
    mask_b[:3, :, 1] = True

    def sensor(mask: np.ndarray, stop: int) -> xr.Dataset:
        error = xr.DataArray(
            np.full((stop, 2, 2), 0.01),
            dims=("time", "lat", "lon"),
            attrs={"units": "1", "space": "shifted_log"},
        )
        return xr.Dataset(
            {
                "log_aod": (("time", "lat", "lon"), np.where(mask[:stop], values[:stop], np.nan)),
                "obs_error_std": error,
                "valid": (("time", "lat", "lon"), mask[:stop]),
            },
            coords={"time": time[:stop], "lat": lat, "lon": lon},
            attrs={"log_epsilon": 0.01},
        )

    return basis, model, sensor(mask_a, 3), sensor(mask_b, 4)


def _spec(**updates: object) -> EOFProjectionSpec:
    values: dict[str, object] = {
        "type": "eof_projection",
        "basis": "basis",
        "model": "model",
        "model_variable": "log_aod",
        "obs": [
            {"source": "sensor_a", "variable": "log_aod", "error_variable": "obs_error_std"},
            {"source": "sensor_b", "variable": "log_aod", "error_variable": "obs_error_std"},
        ],
        "ridge": 1.0,
        "bias_fit_window": {"start": "2001-01-01", "end": "2001-01-02T23:59:59"},
        "support_min_fraction": 0.2,
        "support_full_fraction": 0.5,
        "support_smoothing_passes": 0,
        "min_resolution": 0.3,
    }
    values.update(updates)
    return EOFProjectionSpec.model_validate(values)


def _run(
    spec: EOFProjectionSpec | None = None, *, artifact: xr.Dataset | None = None
) -> xr.Dataset:
    basis, model, sensor_a, sensor_b = _inputs()
    selected = spec or _spec()
    return project_eof(
        basis,
        model,
        [(selected.obs[0], sensor_a), (selected.obs[1], sensor_b)],
        selected,
        bias_fit_artifact=artifact,
    )


def test_projection_preserves_model_axis_and_zeroes_all_missing_day() -> None:
    output = _run()

    expected_time = pd.date_range("2001-01-01T12:00:00", periods=4, freq="1D")
    np.testing.assert_array_equal(output["time"], expected_time.values)
    np.testing.assert_allclose(output["clim_bias"].sel(month=1), 0.2)
    np.testing.assert_allclose(output["spatial_support"].sel(month=1), 1.0)
    np.testing.assert_allclose(output["pc"].isel(time=2), [0.5, -0.25], atol=1.0e-4)
    np.testing.assert_allclose(output["coverage"].isel(time=2), 1.0)
    np.testing.assert_array_equal(output["n_obs"].isel(time=2), [2, 2])

    for variable in (
        "pc",
        "resolution",
        "coverage",
        "resolution_eigenvalue",
    ):
        np.testing.assert_array_equal(output[variable].isel(time=3), [0.0, 0.0])
    np.testing.assert_array_equal(output["posterior_variance"].isel(time=3), [1.0, 1.0])
    np.testing.assert_array_equal(output["posterior_eigenvalue"].isel(time=3), [1.0, 1.0])
    np.testing.assert_array_equal(output["n_obs"].isel(time=3), [0, 0])
    assert output["effective_rank"].isel(time=3).item() == 0
    assert output["condition_number"].isel(time=3).item() == 1.0
    assert output["low_resolution"].isel(time=3).all().item()


def test_projection_schema_allows_zero_ridge_for_full_rank_algebra_oracles() -> None:
    assert _spec(ridge=0.0).ridge == 0.0


def test_projection_batches_grid_inputs_and_keeps_diagnostics_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basis, model, sensor_a, sensor_b = _inputs()
    model = model.chunk({"time": 2})
    sensor_a = sensor_a.chunk({"time": 2})
    sensor_b = sensor_b.chunk({"time": 2})
    spec = _spec(time_chunk_size=2)
    loaded_sizes: list[int] = []
    original = projection_batches._load_chunk

    def tracked_load(*args: object, **kwargs: object):
        start = args[2]
        stop = args[3]
        assert isinstance(start, int)
        assert isinstance(stop, int)
        loaded_sizes.append(stop - start)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(projection_batches, "_load_chunk", tracked_load)
    output = project_eof(
        basis,
        model,
        [(spec.obs[0], sensor_a), (spec.obs[1], sensor_b)],
        spec,
    )

    assert loaded_sizes and max(loaded_sizes) == 2
    assert isinstance(output["innovation_mean"].data, da.Array)
    assert output["innovation_mean"].chunks is not None
    assert output["innovation_mean"].chunks[0] == (2, 2)
    loads_before_diagnostic = len(loaded_sizes)
    output["innovation_mean"].isel(time=0).load()
    assert loaded_sizes[loads_before_diagnostic:] == [2]


def test_batched_bias_fit_matches_full_array_oracle() -> None:
    basis, model, sensor_a, sensor_b = _inputs()
    spec = _spec(time_chunk_size=1)
    inputs = prepare_projection_inputs(
        basis,
        model,
        [(spec.obs[0], sensor_a), (spec.obs[1], sensor_b)],
        spec.model_variable,
    )
    selected = np.array([True, True, False, False])
    months = inputs.time.month.to_numpy(dtype=np.int64)
    model_values = np.asarray(inputs.model.values, dtype=np.float64)
    innovations = np.stack(
        [np.asarray(item.values.values) - model_values for item in inputs.observations]
    )
    errors = np.stack([np.asarray(item.errors.values) for item in inputs.observations])
    valid = np.stack([np.asarray(item.valid.values) for item in inputs.observations])
    expected = fit_monthly_bias(
        innovations,
        errors,
        valid,
        months,
        selected,
        support_min_fraction=spec.support_min_fraction,
        support_full_fraction=spec.support_full_fraction,
        smoothing_passes=spec.support_smoothing_passes,
        delta_bounds=spec.delta_bounds,
    )
    actual = fit_monthly_bias_batched(
        inputs.model,
        inputs.observations,
        months,
        selected,
        support_min_fraction=spec.support_min_fraction,
        support_full_fraction=spec.support_full_fraction,
        smoothing_passes=spec.support_smoothing_passes,
        delta_bounds=spec.delta_bounds,
        time_chunk_size=spec.time_chunk_size,
    )

    for field in (
        "raw_mean",
        "bias",
        "bias_applied",
        "support",
        "support_fraction",
        "support_count",
        "support_day_total",
        "sensor_count",
        "standard_error",
    ):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field))


def test_adapter_consumes_named_inputs_and_declares_fit_artifact(tmp_path: Path) -> None:
    basis, model, sensor_a, sensor_b = _inputs()
    spec = _spec()
    runtime = AnalysisRuntime(
        datetime(2001, 1, 1), datetime(2001, 1, 4, 23, 59), ArtifactService(tmp_path)
    )

    result = EOFProjectionAnalysis().analyze_inputs(
        {
            "basis": basis,
            "model": model,
            "obs[0]": sensor_a,
            "obs[1]": sensor_b,
        },
        spec,
        runtime,
    )

    assert result.dataset.sizes["time"] == 4
    assert len(result.artifacts) == 1
    assert result.artifacts[0].kind == "netcdf_collection"
    assert result.artifacts[0].role == "projection_fit"
    materialized = runtime.artifact_service.materialize(
        "projection", result.dataset, result.artifacts
    )
    assert materialized.source_config["artifact_glob"].endswith("artifacts/projection/chunk-*.nc")
    assert materialized.dataset["pc"].chunks is not None


def test_projection_reuses_frozen_bias_fit_without_refitting(tmp_path: Path) -> None:
    fitted = _run()
    persisted = (
        ArtifactService(tmp_path)
        .materialize(
            "frozen_fit",
            fitted,
            (ArtifactDeclaration(kind="product", reload=True),),
        )
        .dataset
    )
    artifact_spec = _spec(
        bias_fit_window=None,
        bias_fit_artifact="prior_projection",
    )

    reused = _run(artifact_spec, artifact=persisted)

    np.testing.assert_allclose(reused["clim_bias"], fitted["clim_bias"], atol=1.0e-7)
    np.testing.assert_allclose(reused["spatial_support"], fitted["spatial_support"], atol=1.0e-7)
    assert reused.attrs["projection_bias_fit_selection"] == "artifact"

    basis, model, sensor_a, sensor_b = _inputs()
    result = EOFProjectionAnalysis().analyze_inputs(
        {
            "basis": basis,
            "model": model,
            "obs[0]": sensor_a,
            "obs[1]": sensor_b,
            "bias_fit_artifact": persisted,
        },
        artifact_spec,
        AnalysisRuntime(None, None, ArtifactService(tmp_path / "reuse")),
    )
    assert result.artifacts == ()


def test_projection_rejects_frozen_fit_with_changed_support_policy() -> None:
    fitted = _run()
    fitted.attrs["projection_support_min_fraction"] = 0.1
    artifact_spec = _spec(bias_fit_window=None, bias_fit_artifact="prior_projection")

    with pytest.raises(ValueError, match="artifact policy"):
        _run(artifact_spec, artifact=fitted)


def test_projection_rejects_frozen_fit_with_changed_scientific_hash() -> None:
    basis, model, sensor_a, sensor_b = _inputs()
    for dataset in (model, sensor_a, sensor_b):
        dataset.attrs["source_spec_hash"] = "scenario-a"
    basis.attrs["eof_input_source_spec_hash"] = "scenario-a"
    fitted = project_eof(
        basis,
        model,
        [(_spec().obs[0], sensor_a), (_spec().obs[1], sensor_b)],
        _spec(),
    )
    fitted.attrs["source_spec_hash"] = "scenario-b"
    artifact_spec = _spec(bias_fit_window=None, bias_fit_artifact="prior_projection")

    with pytest.raises(ValueError, match="scientific spec hash"):
        project_eof(
            basis,
            model,
            [(artifact_spec.obs[0], sensor_a), (artifact_spec.obs[1], sensor_b)],
            artifact_spec,
            bias_fit_artifact=fitted,
        )


def test_projection_rejects_incompatible_basis_metadata() -> None:
    basis, model, sensor_a, sensor_b = _inputs()
    basis.attrs["eof_rotation"] = "varimax"
    spec = _spec()

    with pytest.raises(ValueError, match="eof_rotation='none'"):
        project_eof(
            basis,
            model,
            [(spec.obs[0], sensor_a), (spec.obs[1], sensor_b)],
            spec,
        )


def test_projection_propagates_and_validates_spec_hash() -> None:
    basis, model, sensor_a, sensor_b = _inputs()
    basis.attrs["eof_input_source_spec_hash"] = "scenario-a"
    model.attrs["source_spec_hash"] = "scenario-a"
    sensor_a.attrs["source_spec_hash"] = "scenario-a"
    sensor_b.attrs["source_spec_hash"] = "scenario-a"
    spec = _spec()

    output = project_eof(
        basis,
        model,
        [(spec.obs[0], sensor_a), (spec.obs[1], sensor_b)],
        spec,
    )

    assert output.attrs["source_spec_hash"] == "scenario-a"
    sensor_b.attrs["source_spec_hash"] = "scenario-b"
    with pytest.raises(ValueError, match="inconsistent scientific spec hashes"):
        project_eof(
            basis,
            model,
            [(spec.obs[0], sensor_a), (spec.obs[1], sensor_b)],
            spec,
        )


def test_shared_covariance_factor_reduces_observability() -> None:
    basis, model, sensor_a, sensor_b = _inputs()
    spec = _spec()
    independent = project_eof(
        basis,
        model,
        [(spec.obs[0], sensor_a), (spec.obs[1], sensor_b)],
        spec,
    )
    for sensor in (sensor_a, sensor_b):
        factor = xr.DataArray(
            np.full((sensor.sizes["time"], 1, 2, 2), 0.1),
            dims=("time", "common_mode", "lat", "lon"),
            coords={
                "time": sensor["time"],
                "common_mode": ["retrieval_common"],
                "lat": sensor["lat"],
                "lon": sensor["lon"],
            },
            attrs={"units": "1", "space": "shifted_log"},
        )
        sensor["common_error_factor"] = factor

    correlated = project_eof(
        basis,
        model,
        [(spec.obs[0], sensor_a), (spec.obs[1], sensor_b)],
        spec,
    )

    assert (
        (correlated["resolution"].isel(time=2) < independent["resolution"].isel(time=2))
        .any()
        .item()
    )
    assert correlated.attrs["projection_common_modes"] == "retrieval_common"
