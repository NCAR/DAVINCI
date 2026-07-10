"""Exact reconstruction, conversion, and named-input tests for AOD scaling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis.scaling import (
    build_aod_scaling,
    reconstruct_log_correction,
    scale_reconstructed_aod,
)
from davinci_monet.config.schema import AODScalingSpec
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages.analyses import AnalysesStage
from davinci_monet.pipeline.stages.base import (
    PipelineContext,
    SourceData,
    StageStatus,
)


def _model_time(days: int = 3) -> xr.DataArray:
    values = np.arange(
        np.datetime64("2005-01-01T12"),
        np.datetime64("2005-01-01T12") + np.timedelta64(days, "D"),
        np.timedelta64(1, "D"),
    ).astype("datetime64[ns]")
    return xr.DataArray(values, dims=("time",), coords={"time": values})


def _basis_signature(patterns: xr.DataArray) -> str:
    ordered = patterns.transpose("mode", "lat", "lon")
    digest = hashlib.sha256()
    for values in (
        ordered["mode"].values,
        ordered["lat"].values,
        ordered["lon"].values,
        np.asarray(ordered.values, dtype=np.float32),
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def test_reconstruction_combines_bias_supported_modes_and_model_time() -> None:
    time = _model_time()
    lat = [-30.0, 30.0]
    lon = [0.0, 180.0]
    mode = [1, 2]
    basis_values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [0.25, 2.0]],
        ]
    )
    coefficient_values = np.array([[2.0, -1.0], [0.5, 1.5], [-2.0, 0.25]])
    bias_values = np.full((1, 2, 2), 0.1)
    support_values = np.array([[[1.0, 0.5], [0.0, 0.25]]])
    basis = xr.DataArray(
        basis_values,
        dims=("mode", "lat", "lon"),
        coords={"mode": mode, "lat": lat, "lon": lon},
    )
    coefficients = xr.DataArray(
        coefficient_values,
        dims=("time", "mode"),
        coords={"time": time, "mode": mode},
    )
    bias = xr.DataArray(
        bias_values,
        dims=("month", "lat", "lon"),
        coords={"month": [1], "lat": lat, "lon": lon},
    )
    support = xr.DataArray(
        support_values,
        dims=("month", "lat", "lon"),
        coords=bias.coords,
    )

    output = reconstruct_log_correction(basis, coefficients, bias, support, time)

    anomaly = np.einsum("tm,mij->tij", coefficient_values, basis_values)
    expected = bias_values[0] + support_values[0] * anomaly
    expected[:, 1, 0] = 0.0
    np.testing.assert_allclose(output["delta_log_anomaly"], anomaly, rtol=1.0e-13)
    np.testing.assert_allclose(output["delta_log_requested"], expected, rtol=1.0e-13)
    np.testing.assert_array_equal(output["time"], time)
    assert bool(output["coefficient_available"].all())


def test_shifted_log_scaling_is_exact_bounded_and_support_aware() -> None:
    time = _model_time(1)
    lon = np.arange(6, dtype=float) * 60.0
    coords = {"time": time, "lat": [0.0], "lon": lon}
    model = xr.DataArray(
        [[[0.0, 0.0009, 0.1, 0.1, 0.1, 0.1]]],
        dims=("time", "lat", "lon"),
        coords=coords,
    )
    requested = xr.DataArray(
        [[[1.0e300, -1.0e300, 0.5, -1.0e300, 1.0e300, 0.2]]],
        dims=model.dims,
        coords=coords,
    )
    support = xr.DataArray(
        [[[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]],
        dims=model.dims,
        coords=coords,
    )
    epsilon = 0.01

    output = scale_reconstructed_aod(
        model,
        requested,
        support,
        epsilon=epsilon,
        r_bounds=(0.4, 3.2),
        aod_floor=0.001,
    )

    exact_unclipped = ((0.1 + epsilon) * np.exp(0.2) - epsilon) / 0.1
    expected_ratio = np.array([1.0, 1.0, 1.0, 0.4, 3.2, exact_unclipped])
    np.testing.assert_allclose(output["r"].values[0, 0], expected_ratio, rtol=1.0e-12)
    np.testing.assert_allclose(output["aod_target"], model * output["r"], rtol=1.0e-13)
    expected_delta = np.log(output["aod_target"] + epsilon) - np.log(model + epsilon)
    xr.testing.assert_allclose(output["delta_log_applied"], expected_delta)
    assert np.isfinite(output["delta_log_safe"]).all()
    assert np.isfinite(output["r"]).all()
    assert output["support_identity_count"].item() == 1
    assert output["low_aod_identity_count"].item() == 2
    assert output["lower_clip_count"].item() == 1
    assert output["upper_clip_count"].item() == 1
    assert output["support_identity_fraction"].item() == 1.0 / 6.0


def test_zero_aod_remains_identity_when_aod_floor_is_zero() -> None:
    time = _model_time(1)
    model = xr.DataArray(
        [[[0.0, 0.1]]],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0], "lon": [0.0, 180.0]},
    )
    requested = xr.full_like(model, 0.2)
    support = xr.ones_like(model)

    output = scale_reconstructed_aod(
        model,
        requested,
        support,
        epsilon=0.01,
        r_bounds=(0.2, 5.0),
        aod_floor=0.0,
    )

    assert output["r"].isel(lon=0).item() == 1.0
    assert output["aod_target"].isel(lon=0).item() == 0.0
    assert bool(output["low_aod_identity_mask"].isel(lon=0).item())
    assert np.isfinite(output["r"]).all()


def _pipeline_inputs() -> dict[str, SourceData]:
    time = _model_time()
    coefficient_time = time.isel(time=[0, 2])
    lat = [-30.0, 30.0]
    lon = [0.0, 180.0]
    mode = [1, 2]
    basis = xr.Dataset(
        {
            "eofs": (
                ("mode", "lat", "lon"),
                np.array(
                    [
                        [[0.1, 0.2], [0.3, 0.4]],
                        [[-0.2, 0.1], [0.05, 0.2]],
                    ]
                ),
            )
        },
        coords={"mode": mode, "lat": lat, "lon": lon},
        attrs={
            "eof_rotation": "none",
            "eof_standardize": "false",
            "eof_input_log_epsilon": 0.01,
            "eof_input_source_spec_hash": "scenario-a",
        },
    )
    basis_signature = _basis_signature(basis["eofs"])
    projection = xr.Dataset(
        {
            "clim_bias_applied": (
                ("month", "lat", "lon"),
                np.zeros((1, 2, 2)),
            ),
            "spatial_support": (
                ("month", "lat", "lon"),
                np.ones((1, 2, 2)),
            ),
            "resolution": (
                ("time", "mode"),
                np.full((3, 2), 0.8),
            ),
        },
        coords={"month": [1], "time": time, "mode": mode, "lat": lat, "lon": lon},
        attrs={
            "projection_basis_signature": basis_signature,
            "projection_log_epsilon": 0.01,
            "source_spec_hash": "scenario-a",
        },
    )
    coefficients = xr.Dataset(
        {
            "pc": (
                ("time", "mode"),
                np.array([[1.0, -0.5], [-0.25, 0.75]]),
            ),
            "valid_segment": (
                ("time", "mode"),
                np.ones((2, 2), dtype=bool),
            ),
            "coi": (("time", "mode"), np.full((2, 2), 180.0)),
        },
        coords={"time": coefficient_time, "mode": mode},
        attrs={
            "projection_basis_signature": basis_signature,
            "projection_log_epsilon": 0.01,
            "source_spec_hash": "scenario-a",
            "band_max": 180.0,
        },
    )
    model = xr.Dataset(
        {"aod": (("time", "lat", "lon"), np.full((3, 2, 2), 0.1))},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"log_epsilon": 0.01, "source_spec_hash": "scenario-a"},
    )
    return {
        "basis_src": SourceData(basis, "basis_src", "eof", DataGeometry.GRID),
        "projection_src": SourceData(
            projection, "projection_src", "eof_projection", DataGeometry.GRID
        ),
        "filtered_src": SourceData(
            coefficients, "filtered_src", "wavelet_filter", DataGeometry.SPECTRUM
        ),
        "model_src": SourceData(model, "model_src", "aod_preprocess", DataGeometry.GRID),
    }


def test_pipeline_resolves_four_named_inputs_and_keeps_daily_chunks(tmp_path: Path) -> None:
    context = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "analyses": {
                "scaling": {
                    "type": "aod_scaling",
                    "basis": "basis_src",
                    "projection": "projection_src",
                    "coefficients": "filtered_src",
                    "model": "model_src",
                    "r_bounds": [0.2, 5.0],
                    "aod_floor": 0.001,
                    "time_chunk_days": 2,
                    "required": True,
                }
            },
        },
        sources=_pipeline_inputs(),
    )

    result = AnalysesStage().execute(context)

    assert result.status is StageStatus.COMPLETED
    scaling = context.sources["scaling"]
    assert scaling.geometry is DataGeometry.GRID
    assert scaling.source_type == "aod_scaling"
    np.testing.assert_array_equal(scaling.data["time"], _model_time())
    np.testing.assert_array_equal(
        scaling.data["coefficient_available"], np.array([True, False, True])
    )
    np.testing.assert_allclose(scaling.data["r"].isel(time=1), 1.0, rtol=0.0, atol=0.0)
    assert scaling.data["r"].chunks is not None
    assert scaling.data["r"].chunks[0] == (2, 1)
    assert scaling.data.attrs["artifact_policy"] == "time_chunked_lazy"
    assert scaling.data.attrs["spec_hash"] == "scenario-a"
    assert scaling.data.attrs["band_max"] == 180.0
    assert {"eofs", "pc", "resolution", "valid_segment", "coi"} <= set(scaling.data.data_vars)
    assert scaling.config["artifact_glob"].endswith("artifacts/scaling/chunk-*.nc")
    artifact_entry = context.metadata["analysis_artifacts"][0]
    assert artifact_entry["role"] == "scaling"
    assert len(artifact_entry["checksums"]["collection_sha256"]) == 64


def _scaling_spec() -> AODScalingSpec:
    return AODScalingSpec(
        type="aod_scaling",
        basis="basis_src",
        projection="projection_src",
        coefficients="filtered_src",
        model="model_src",
    )


def test_scaling_rejects_cross_wired_basis_and_coefficients() -> None:
    sources = _pipeline_inputs()
    basis = sources["basis_src"].data
    projection = sources["projection_src"].data
    coefficients = sources["filtered_src"].data.copy()
    model = sources["model_src"].data
    coefficients.attrs["projection_basis_signature"] = "different-basis"

    with pytest.raises(ValueError, match="incompatible projection basis signature"):
        build_aod_scaling(basis, projection, coefficients, model, _scaling_spec())

    coefficients = sources["filtered_src"].data
    changed_basis = basis.copy(deep=True)
    changed_basis["eofs"].values[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="incompatible projection basis signature"):
        build_aod_scaling(changed_basis, projection, coefficients, model, _scaling_spec())


def test_scaling_rejects_cross_wired_projection_log_epsilon() -> None:
    sources = _pipeline_inputs()
    coefficients = sources["filtered_src"].data.copy()
    coefficients.attrs["projection_log_epsilon"] = 0.02

    with pytest.raises(ValueError, match="inconsistent shifted-log epsilon"):
        build_aod_scaling(
            sources["basis_src"].data,
            sources["projection_src"].data,
            coefficients,
            sources["model_src"].data,
            _scaling_spec(),
        )
