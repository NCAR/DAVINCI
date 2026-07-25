"""Tests for the shared synthetic-first AOD preprocessing analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis.aod_preprocess import preprocess_aod
from davinci_monet.analysis.artifacts import ArtifactService
from davinci_monet.analysis.base import AnalysisRuntime
from davinci_monet.config.schema import AODPreprocessSpec, LinearAODUncertaintySpec
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages.analyses import AnalysesStage
from davinci_monet.pipeline.stages.base import PipelineContext, SourceData, StageStatus


def _runtime(
    tmp_path: Path,
    start: datetime | None = datetime(2001, 1, 1),
    end: datetime | None = datetime(2001, 1, 2, 23, 59, 59),
) -> AnalysisRuntime:
    return AnalysisRuntime(start, end, ArtifactService(tmp_path))


def test_daily_preprocess_screens_before_log_and_anchors_at_noon(tmp_path: Path) -> None:
    source = xr.Dataset(
        {
            "raw_aod": (
                ("time", "lat", "lon"),
                np.array([[[0.0, -1.0, np.nan, np.inf]], [[0.2, 0.4, 0.8, 1.0]]]),
                {"units": "1"},
            )
        },
        coords={
            "time": np.array(["2001-01-01T03", "2001-01-02T21"], dtype="datetime64[h]"),
            "lat": [0.0],
            "lon": [-135.0, -45.0, 45.0, 135.0],
        },
        attrs={"spec_hash": "synthetic-source-hash"},
    )
    spec = AODPreprocessSpec(
        type="aod_preprocess", source="raw", variable="raw_aod", log_epsilon=0.01
    )

    output = preprocess_aod(source, spec, _runtime(tmp_path))

    np.testing.assert_array_equal(
        output["time"],
        np.array(["2001-01-01T12", "2001-01-02T12"], dtype="datetime64[h]"),
    )
    np.testing.assert_array_equal(output["valid"].isel(time=0), [[True, False, False, False]])
    assert output["aod"].isel(time=0, lat=0, lon=0).item() == 0.0
    assert np.isnan(output["log_aod"].isel(time=0, lat=0, lon=1).item())
    assert output["log_aod"].isel(time=0, lat=0, lon=0).item() == pytest.approx(np.log(0.01))
    assert output.attrs["source_spec_hash"] == "synthetic-source-hash"


def test_local_sampling_uses_padding_then_clips_calendar_window(tmp_path: Path) -> None:
    time = np.arange(
        np.datetime64("2000-12-31T00", "h"),
        np.datetime64("2001-01-04T00", "h"),
        np.timedelta64(1, "h"),
    )
    values = np.broadcast_to(np.arange(time.size)[:, None, None], (time.size, 1, 2))
    source = xr.Dataset(
        {"aod": (("time", "lat", "lon"), values.astype(float))},
        coords={"time": time, "lat": [0.0], "lon": [179.0, -179.0]},
    )
    spec = AODPreprocessSpec(
        type="aod_preprocess",
        source="model",
        variable="aod",
        sample_local_time=13.5,
        sample_tolerance="31min",
    )

    output = preprocess_aod(source, spec, _runtime(tmp_path))

    np.testing.assert_array_equal(
        output["time"],
        np.array(["2001-01-01T12", "2001-01-02T12"], dtype="datetime64[h]"),
    )
    np.testing.assert_array_equal(output["aod"].values[:, 0, :], [[26, 49], [50, 73]])
    np.testing.assert_array_equal(
        output["aod"]["sample_time"].values,
        np.array(
            [["2001-01-01T02", "2001-01-02T01"], ["2001-01-02T02", "2001-01-03T01"]],
            dtype="datetime64[h]",
        ),
    )


def test_exact_target_grid_preserves_uncertainty_and_common_factor(tmp_path: Path) -> None:
    coordinates = {
        "time": np.array(["2001-01-01"], dtype="datetime64[D]"),
        "lat": [-30.0, 30.0],
        "lon": [-90.0, 90.0],
    }
    source = xr.Dataset(
        {
            "aod": (("time", "lat", "lon"), np.full((1, 2, 2), 0.2)),
            "sigma": (("time", "lat", "lon"), np.full((1, 2, 2), 0.05)),
            "shared": (("time", "lat", "lon"), np.full((1, 2, 2), 0.02)),
        },
        coords=coordinates,
    )
    target = xr.Dataset(coords={"lat": coordinates["lat"], "lon": coordinates["lon"]})
    spec = AODPreprocessSpec(
        type="aod_preprocess",
        source="sensor",
        variable="aod",
        target_grid_from="model_daily",
        uncertainty_variable="sigma",
        common_factor_variables=["shared"],
    )

    output = preprocess_aod(
        source,
        spec,
        _runtime(tmp_path, end=datetime(2001, 1, 1, 23, 59, 59)),
        target_grid=target,
    )

    np.testing.assert_array_equal(output["obs_error_std"], source["sigma"])
    np.testing.assert_array_equal(
        output["common_error_factor"].isel(common_mode=0), source["shared"]
    )
    assert output["common_error_factor"]["common_mode"].item() == "shared"


def test_linear_aod_uncertainty_contract_transforms_and_records_provenance(
    tmp_path: Path,
) -> None:
    aod = np.array([[[0.0, 0.2]]])
    source_stddev = np.array([[[0.0, 0.04]]])
    source = xr.Dataset(
        {
            "aod": (("time", "lat", "lon"), aod),
            "aod_stddev": (("time", "lat", "lon"), source_stddev),
        },
        coords={
            "time": np.array(["2001-01-01"], dtype="datetime64[D]"),
            "lat": [0.0],
            "lon": [-0.5, 0.5],
        },
    )
    spec = AODPreprocessSpec(
        type="aod_preprocess",
        source="sensor",
        variable="aod",
        log_epsilon=0.01,
        uncertainty_model=LinearAODUncertaintySpec(
            type="linear_aod_rss",
            name="test-aod-error-v1",
            source_variable="aod_stddev",
            absolute_floor=0.05,
            relative_fraction=0.15,
            covariance="independent",
        ),
    )

    output = preprocess_aod(
        source,
        spec,
        _runtime(tmp_path, end=datetime(2001, 1, 1, 23, 59, 59)),
    )

    expected = np.sqrt(source_stddev**2 + 0.05**2 + (0.15 * aod) ** 2) / (aod + 0.01)
    np.testing.assert_allclose(output["obs_error_std"], expected)
    assert output["obs_error_std"].attrs["space"] == "shifted_log"
    assert output.attrs["uncertainty_contract"] == "test-aod-error-v1"
    assert output.attrs["uncertainty_transform"] == "delta_method"
    assert output.attrs["uncertainty_covariance"] == "independent"


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ("sigma", "obs_error_std must be finite and positive"),
        ("shared", "common factor 'shared' must be finite"),
    ],
)
def test_preprocess_rejects_infinite_uncertainty_inputs(
    tmp_path: Path, variable: str, message: str
) -> None:
    source = xr.Dataset(
        {
            "aod": (("time", "lat", "lon"), np.full((1, 1, 1), 0.2)),
            "sigma": (("time", "lat", "lon"), np.full((1, 1, 1), 0.05)),
            "shared": (("time", "lat", "lon"), np.full((1, 1, 1), 0.02)),
        },
        coords={
            "time": np.array(["2001-01-01"], dtype="datetime64[D]"),
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    source[variable].values[0, 0, 0] = np.inf
    spec = AODPreprocessSpec(
        type="aod_preprocess",
        source="sensor",
        variable="aod",
        uncertainty_variable="sigma",
        common_factor_variables=["shared"],
    )

    with pytest.raises(ValueError, match=message):
        preprocess_aod(
            source,
            spec,
            _runtime(tmp_path, end=datetime(2001, 1, 1, 23, 59, 59)),
        )


def test_preprocess_validates_chunked_uncertainty_inputs(tmp_path: Path) -> None:
    source = xr.Dataset(
        {
            "aod": (("time", "lat", "lon"), np.full((2, 1, 1), 0.2)),
            "sigma": (("time", "lat", "lon"), np.full((2, 1, 1), 0.05)),
            "shared": (("time", "lat", "lon"), np.full((2, 1, 1), 0.02)),
        },
        coords={
            "time": np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[D]"),
            "lat": [0.0],
            "lon": [0.0],
        },
    ).chunk({"time": 1})
    spec = AODPreprocessSpec(
        type="aod_preprocess",
        source="sensor",
        variable="aod",
        uncertainty_variable="sigma",
        common_factor_variables=["shared"],
    )

    output = preprocess_aod(source, spec, _runtime(tmp_path))

    assert output["aod"].chunks is not None
    np.testing.assert_allclose(output["obs_error_std"].compute(), 0.05)
    np.testing.assert_allclose(output["common_error_factor"].compute(), 0.02)

    invalid = source.copy()
    invalid["sigma"] = invalid["sigma"].where(invalid["time"] != invalid["time"][0], -1.0)
    with pytest.raises(ValueError, match="finite and positive at every valid"):
        preprocess_aod(invalid, spec, _runtime(tmp_path))


def test_uncertainty_coarsening_requires_declared_covariance(tmp_path: Path) -> None:
    source = xr.Dataset(
        {
            "aod": (("time", "lat", "lon"), np.full((1, 2, 2), 0.2)),
            "sigma": (("time", "lat", "lon"), np.ones((1, 2, 2))),
        },
        coords={
            "time": np.array(["2001-01-01"], dtype="datetime64[D]"),
            "lat": [-45.0, 45.0],
            "lon": [-90.0, 90.0],
        },
    )
    spec = AODPreprocessSpec(
        type="aod_preprocess",
        source="sensor",
        variable="aod",
        target_grid=180.0,
        uncertainty_variable="sigma",
    )

    with pytest.raises(ValueError, match="uncertainty_covariance='independent'"):
        preprocess_aod(
            source,
            spec,
            _runtime(tmp_path, end=datetime(2001, 1, 1, 23, 59, 59)),
        )

    independent = spec.model_copy(update={"uncertainty_covariance": "independent"})
    output = preprocess_aod(
        source,
        independent,
        _runtime(tmp_path, end=datetime(2001, 1, 1, 23, 59, 59)),
    )
    np.testing.assert_allclose(output["obs_error_std"], 1.0 / np.sqrt(2.0))

    invalid_source = source.copy(deep=True)
    invalid_source["sigma"].values[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="finite and positive at every valid"):
        preprocess_aod(
            invalid_source,
            independent,
            _runtime(tmp_path, end=datetime(2001, 1, 1, 23, 59, 59)),
        )


def test_analysis_stage_resolves_raw_and_derived_preprocess_inputs(tmp_path: Path) -> None:
    time = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[D]")
    lat = np.arange(-75.0, 90.0, 30.0)
    lon = np.arange(-165.0, 180.0, 30.0)
    model = xr.Dataset(
        {"TOTEXTTAU": (("time", "lat", "lon"), np.full((2, 6, 12), 0.2))},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    sensor = xr.Dataset(
        {
            "aod": (("time", "lat", "lon"), np.full((2, 6, 12), 0.25)),
            "sigma": (("time", "lat", "lon"), np.full((2, 6, 12), 0.05)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    context = PipelineContext(
        config={
            "analysis": {
                "start_time": "2001-01-01",
                "end_time": "2001-01-02 23:59:59",
                "output_dir": str(tmp_path),
            },
            "analyses": {
                "sensor_daily": {
                    "type": "aod_preprocess",
                    "source": "sensor_raw",
                    "variable": "aod",
                    "uncertainty_variable": "sigma",
                    "target_grid_from": "model_daily",
                    "required": True,
                },
                "model_daily": {
                    "type": "aod_preprocess",
                    "source": "model_raw",
                    "variable": "TOTEXTTAU",
                    "target_grid": 30.0,
                    "required": True,
                },
            },
        },
        sources={
            "model_raw": SourceData(model, "model_raw", "generic", DataGeometry.GRID),
            "sensor_raw": SourceData(sensor, "sensor_raw", "generic", DataGeometry.GRID),
        },
    )

    result = AnalysesStage().execute(context)

    assert result.status is StageStatus.COMPLETED
    assert context.metadata["analysis_status"] == {
        "model_daily": "completed",
        "sensor_daily": "completed",
    }
    np.testing.assert_array_equal(
        context.sources["sensor_daily"].data["lat"], context.sources["model_daily"].data["lat"]
    )
    assert context.sources["sensor_daily"].data["obs_error_std"].notnull().all()
