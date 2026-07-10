"""Randomized EOF accuracy, fit isolation, and lazy-output regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from davinci_monet.analysis.artifacts import ArtifactService
from davinci_monet.analysis.base import AnalysisRuntime
from davinci_monet.analysis.eof import EOFAnalysis
from davinci_monet.config.schema import EOFSpec


def _modal_dataset(*, nt: int = 240, nlat: int = 10, nlon: int = 14, seed: int = 31) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    n_features = nlat * nlon
    spatial, _ = np.linalg.qr(rng.normal(size=(n_features, 4)))
    temporal, _ = np.linalg.qr(rng.normal(size=(nt, 4)))
    field = (temporal * np.array([24.0, 15.0, 9.0, 5.0])) @ spatial.T
    field += 0.002 * rng.normal(size=field.shape)
    lat = np.linspace(-75.0, 75.0, nlat)
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False)
    return xr.Dataset(
        {
            "log_aod": (
                ("time", "lat", "lon"),
                field.reshape(nt, nlat, nlon).astype(np.float32),
                {"units": "1", "preprocess_hash": "preprocess-123", "log_epsilon": 0.01},
            )
        },
        coords={
            "time": pd.date_range("2001-01-01", periods=nt, freq="D"),
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat, {"units": "degrees_north"}),
            "longitude": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"source_hash": "source-456"},
    )


def _subspace_cosines(left: xr.DataArray, right: xr.DataArray) -> np.ndarray:
    left_matrix = np.asarray(left).reshape(left.sizes["mode"], -1).T
    right_matrix = np.asarray(right).reshape(right.sizes["mode"], -1).T
    left_q, _ = np.linalg.qr(left_matrix)
    right_q, _ = np.linalg.qr(right_matrix)
    return np.linalg.svd(left_q.T @ right_q, compute_uv=False)


def test_randomized_solver_matches_full_subspace_and_variance() -> None:
    data = _modal_dataset()
    common = dict(type="eof", source="model", variable="log_aod", n_modes=4)
    full = EOFAnalysis().analyze(data, EOFSpec.model_validate({**common, "solver": "full"}))
    randomized = EOFAnalysis().analyze(
        data,
        EOFSpec.model_validate(
            {
                **common,
                "solver": "randomized",
                "solver_seed": 90210,
                "solver_oversampling": 8,
                "solver_iterations": 2,
            }
        ),
    )

    assert _subspace_cosines(full["eofs"], randomized["eofs"]).min() > 0.999
    np.testing.assert_allclose(
        randomized["explained_variance"], full["explained_variance"], rtol=2.0e-3, atol=2.0e-5
    )
    assert randomized.attrs["eof_solver"] == "randomized"
    assert randomized.attrs["eof_solver_matrix_dtype"] == "float32"
    assert randomized.attrs["eof_rotation"] == "none"
    assert randomized.attrs["eof_standardize"] == "false"
    assert randomized.attrs["eof_input_preprocess_hash"] == "preprocess-123"
    assert randomized.attrs["eof_input_source_hash"] == "source-456"
    assert randomized.attrs["eof_input_log_epsilon"] == 0.01


def test_randomized_solver_is_repeatable_for_fixed_seed() -> None:
    data = _modal_dataset(seed=87)
    spec = EOFSpec(
        type="eof",
        source="model",
        variable="log_aod",
        n_modes=4,
        solver="randomized",
        solver_seed=73,
    )

    first = EOFAnalysis().analyze(data, spec)
    second = EOFAnalysis().analyze(data, spec)

    np.testing.assert_array_equal(first["pc"], second["pc"])
    np.testing.assert_array_equal(first["eofs"], second["eofs"])
    np.testing.assert_array_equal(first["explained_variance"], second["explained_variance"])


def test_fit_window_metadata_and_preprocessing_do_not_leak() -> None:
    data = _modal_dataset(nt=48, nlat=6, nlon=8)
    data = data.assign_coords(time=pd.date_range("2001-01-01", periods=48, freq="MS"))
    rng = np.random.default_rng(99)
    leaked = data.copy(deep=True)
    leaked["log_aod"].values[24:] += rng.normal(0.0, 100.0, size=(24, 6, 8))
    fit_end = str(data["time"].values[23])
    spec = EOFSpec.model_validate(
        {
            "type": "eof",
            "source": "model",
            "variable": "log_aod",
            "n_modes": 4,
            "remove_seasonal_cycle": True,
            "fit_window": {"start": "2001-01-01", "end": fit_end},
        }
    )

    fitted = EOFAnalysis().analyze(leaked, spec)
    reference = EOFAnalysis().analyze(
        data.isel(time=slice(0, 24)),
        EOFSpec(
            type="eof",
            source="model",
            variable="log_aod",
            n_modes=4,
            remove_seasonal_cycle=True,
        ),
    )

    np.testing.assert_allclose(fitted["eofs"], reference["eofs"], atol=1.0e-12)
    np.testing.assert_allclose(fitted["time_mean"], reference["time_mean"], atol=1.0e-12)
    np.testing.assert_allclose(fitted["climatology"], reference["climatology"], atol=1.0e-12)
    assert fitted.sizes["time"] == 24
    assert fitted.attrs["eof_fit_selection"] == "window"
    assert fitted.attrs["eof_fit_count"] == 24
    assert fitted.attrs["eof_fit_start"].startswith("2001-01-01")
    assert fitted.attrs["eof_fit_end"].startswith("2002-12-01")
    assert fitted.attrs["eof_fit_requested_start"] == "2001-01-01"
    assert fitted.attrs["eof_fit_requested_end"] == fit_end

    standardized = EOFAnalysis().analyze(
        data.isel(time=slice(0, 24)),
        EOFSpec(type="eof", source="model", variable="log_aod", n_modes=2, standardize=True),
    )
    assert "standard_deviation" in standardized
    assert standardized["standard_deviation"].attrs["kind"] == "preprocessing"


def test_fit_artifact_selects_an_immutable_training_split(tmp_path) -> None:
    data = _modal_dataset(nt=80, nlat=6, nlon=8)
    split = xr.Dataset(
        {
            "split": (
                "time",
                np.where(np.arange(80) < 40, "basis_train", "development_test"),
            )
        },
        coords={"time": data["time"]},
        attrs={"scientific_hash": "frozen-split-789"},
    )
    spec = EOFSpec(
        type="eof",
        source="model",
        variable="log_aod",
        n_modes=3,
        fit_artifact="frozen_split",
        fit_split="basis_train",
    )
    runtime = AnalysisRuntime(None, None, ArtifactService(tmp_path))

    result = EOFAnalysis().analyze_inputs({"source": data, "fit_artifact": split}, spec, runtime)
    fitted = result.dataset
    reference = EOFAnalysis().analyze(
        data.isel(time=slice(0, 40)),
        EOFSpec(type="eof", source="model", variable="log_aod", n_modes=3),
    )

    np.testing.assert_allclose(fitted["eofs"], reference["eofs"])
    assert spec.input_refs() == {"source": "model", "fit_artifact": "frozen_split"}
    assert fitted.attrs["eof_fit_selection"] == "basis_train"
    assert fitted.attrs["eof_fit_artifact"] == "frozen_split"
    assert fitted.attrs["eof_fit_artifact_scientific_hash"] == "frozen-split-789"
    assert len(result.artifacts) == 1
    assert result.artifacts[0].role == "basis_fit"


def test_randomized_chunked_input_keeps_spatial_products_lazy() -> None:
    data = _modal_dataset(nt=192, nlat=24, nlon=32).chunk({"time": 32, "lat": 6, "lon": 8})
    spec = EOFSpec(
        type="eof",
        source="model",
        variable="log_aod",
        n_modes=4,
        solver="randomized",
        solver_seed=19,
        solver_oversampling=6,
        solver_iterations=1,
    )

    output = EOFAnalysis().analyze(data, spec)

    assert output["eofs"].chunks is not None
    assert output["time_mean"].chunks is not None
    assert output.attrs["eof_feature_count"] == 24 * 32
    assert output.attrs["eof_solver_rank"] == 4
    assert output.attrs["eof_right_vectors"] == "not_retained"
    assert np.isfinite(output["eofs"].compute()).all()


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"n_modes": 0}, "greater than or equal to 1"),
        (
            {"fit_window": {"start": "2002-01-01", "end": "2001-01-01"}},
            "start must be at or before end",
        ),
        (
            {
                "fit_window": {"start": "2001-01-01", "end": "2002-01-01"},
                "fit_artifact": "split",
            },
            "mutually exclusive",
        ),
    ],
)
def test_eof_solver_schema_rejects_invalid_config(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EOFSpec.model_validate({"type": "eof", "source": "model", "variable": "log_aod", **updates})
