"""Tests for the unified data-source pipeline plumbing.

``LoadSourcesStage`` loads all data sources into a single ``context.sources``
view and tags each dataset with ``source_label`` and ``geometry`` metadata.

These are unit tests: they construct data containers directly and exercise the
context API and stage logic, per the repo's existing pipeline-stage test pattern
(see test_geometry_pipeline.py).
"""

from __future__ import annotations

from datetime import datetime

import cftime
import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.pipeline.stages import (
    LoadSourcesStage,
    PipelineContext,
    SourceData,
    StageStatus,
)


def _point_geometry_dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    n_t, n_s = 12, 4
    times = np.datetime64("2024-02-01") + np.arange(n_t) * np.timedelta64(1, "h")
    return xr.Dataset(
        {
            "o3": (("time", "site"), rng.uniform(10, 60, (n_t, n_s)), {"units": "ppb"}),
        },
        coords={
            "time": times,
            "site": np.arange(n_s),
            "latitude": ("site", rng.uniform(0, 40, n_s)),
            "longitude": ("site", rng.uniform(90, 140, n_s)),
        },
    )


def _grid_dataset_dataset() -> xr.Dataset:
    rng = np.random.default_rng(1)
    n_t, n_lat, n_lon = 12, 5, 6
    times = np.datetime64("2024-02-01") + np.arange(n_t) * np.timedelta64(1, "h")
    return xr.Dataset(
        {"O3": (("time", "lat", "lon"), rng.uniform(10, 60, (n_t, n_lat, n_lon)))},
        coords={
            "time": times,
            "lat": np.linspace(0, 40, n_lat),
            "lon": np.linspace(90, 140, n_lon),
        },
    )


@pytest.fixture
def y_data() -> SourceData:
    return SourceData(
        data=_grid_dataset_dataset(),
        label="cam",
        source_type="generic",
        geometry=DataGeometry.GRID,
    )


@pytest.fixture
def x_data() -> SourceData:
    return SourceData(
        data=_point_geometry_dataset(),
        label="airnow",
        source_type="pt_sfc",
        geometry=DataGeometry.POINT,
    )


class TestPipelineContextSources:
    def test_sources_defaults_to_empty_dict(self) -> None:
        ctx = PipelineContext()
        assert ctx.sources == {}

    def test_get_source_returns_registered(self, x_data: SourceData) -> None:
        ctx = PipelineContext(sources={"airnow": x_data})
        assert ctx.get_source("airnow") is x_data

    def test_get_source_missing_raises_keyerror(self) -> None:
        ctx = PipelineContext()
        with pytest.raises(KeyError):
            ctx.get_source("nope")


class TestLoadSourcesStage:
    def test_builtin_source_registration_modules_are_unique(self) -> None:
        """Built-in registration bootstrap should not carry duplicate module names."""
        from davinci_monet.io.source_registration import BUILTIN_SOURCE_READER_MODULES

        assert len(BUILTIN_SOURCE_READER_MODULES) == len(set(BUILTIN_SOURCE_READER_MODULES))

    def test_reader_kwargs_filters_loader_keys_for_kwargs_reader(self) -> None:
        """Loader/schema keys should not leak to readers even when they accept **kwargs."""

        class Reader:
            def open(self, file_paths, variables=None, **kwargs):  # noqa: ANN001
                raise AssertionError("not called")

        assert LoadSourcesStage._reader_kwargs(
            Reader(),
            {
                "type": "generic",
                "files": "data.nc",
                "variables": {"O3": {}},
                "display_name": "Display",
                "resample": "1D",
                "reader_option": "keep",
                "none_option": None,
            },
        ) == {"reader_option": "keep"}

    def test_reader_kwargs_filters_to_explicit_signature(self) -> None:
        """Readers without **kwargs receive only their declared reader options."""

        class Reader:
            def open(self, file_paths, variables=None, *, product=None):  # noqa: ANN001
                raise AssertionError("not called")

        assert LoadSourcesStage._reader_kwargs(
            Reader(),
            {
                "type": "generic",
                "files": "data.nc",
                "variables": {"O3": {}},
                "product": "NO2",
                "unexpected": "drop",
            },
        ) == {"product": "NO2"}

    def test_unifies_sources_with_dataset_labels_and_geometry(
        self, y_data: SourceData, x_data: SourceData
    ) -> None:
        # Pre-populated sources (no config) are tagged into the unified view.
        ctx = PipelineContext(
            sources={"cam": y_data, "airnow": x_data},
        )
        result = LoadSourcesStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        # Both sources exposed via the unified view.
        assert set(ctx.sources) == {"cam", "airnow"}
        assert ctx.get_source("cam") is y_data
        assert ctx.get_source("airnow") is x_data

        # Datasets tagged with source_label / geometry.
        cam_attrs = ctx.sources["cam"].data.attrs
        assert cam_attrs["source_label"] == "cam"
        assert cam_attrs["geometry"] == "grid"

        air_attrs = ctx.sources["airnow"].data.attrs
        assert air_attrs["source_label"] == "airnow"
        assert air_attrs["geometry"] == "point"

    def test_prepopulated_sources_get_post_load_coordinate_contract(self) -> None:
        grid = SourceData(
            data=xr.Dataset(
                {"O3": (("latitude", "longitude"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
                coords={"latitude": [40.0, 30.0], "longitude": [350.0, 10.0]},
            ),
            label="cam",
            source_type="generic",
            geometry=DataGeometry.GRID,
        )
        point = SourceData(
            data=xr.Dataset(
                {"o3": ("site", [11.0, 22.0])},
                coords={
                    "site": [0, 1],
                    "latitude": ("site", [35.0, 34.0]),
                    "longitude": ("site", [350.0, 20.0]),
                },
            ),
            label="airnow",
            source_type="pt_sfc",
            geometry=DataGeometry.POINT,
        )
        ctx = PipelineContext(sources={"cam": grid, "airnow": point})

        result = LoadSourcesStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        cam = ctx.sources["cam"].data
        assert "lat" in cam.coords and "lon" in cam.coords
        np.testing.assert_allclose(cam["latitude"].values, [30.0, 40.0])
        np.testing.assert_allclose(cam["longitude"].values, [-10.0, 10.0])
        np.testing.assert_allclose(cam["O3"].values, [[3.0, 4.0], [1.0, 2.0]])

        airnow = ctx.sources["airnow"].data
        assert "lat" in airnow.coords and "lon" in airnow.coords
        np.testing.assert_allclose(airnow["longitude"].values, [-10.0, 20.0])
        np.testing.assert_allclose(airnow["o3"].values, [11.0, 22.0])

    def test_prepopulated_rectilinear_grid_sorts_axis_coords_with_distinct_dims(
        self,
    ) -> None:
        grid = SourceData(
            data=xr.Dataset(
                {"O3": (("y", "x"), np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))},
                coords={
                    "y": [0, 1],
                    "x": [0, 1, 2, 3],
                    "latitude": ("y", [40.0, 30.0]),
                    "longitude": ("x", [0.0, 90.0, 180.0, 270.0]),
                },
            ),
            label="cam",
            source_type="generic",
            geometry=DataGeometry.GRID,
        )
        ctx = PipelineContext(sources={"cam": grid})

        result = LoadSourcesStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        cam = ctx.sources["cam"].data
        np.testing.assert_allclose(cam["latitude"].values, [30.0, 40.0])
        np.testing.assert_allclose(cam["longitude"].values, [-180.0, -90.0, 0.0, 90.0])
        np.testing.assert_allclose(
            cam["O3"].values,
            [[7.0, 8.0, 5.0, 6.0], [3.0, 4.0, 1.0, 2.0]],
        )

    def test_prepopulated_sources_resolve_via_get_source(
        self, y_data: SourceData, x_data: SourceData
    ) -> None:
        ctx = PipelineContext(
            sources={"cam": y_data, "airnow": x_data},
        )
        LoadSourcesStage().execute(ctx)
        assert ctx.get_source("cam") is y_data
        assert ctx.get_source("airnow") is x_data

    def test_unified_source_uses_reader_geometry(self, tmp_path) -> None:
        source_path = tmp_path / "cam.nc"
        _grid_dataset_dataset().to_netcdf(source_path)
        ctx = PipelineContext(
            config={
                "sources": {
                    "cam": {
                        "type": "generic",
                        "files": str(source_path),
                        "variables": {"O3": {"units": "ppb"}},
                    }
                }
            }
        )

        result = LoadSourcesStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        assert set(ctx.sources) == {"cam"}
        assert ctx.sources["cam"].geometry is DataGeometry.GRID
        assert ctx.sources["cam"].data.attrs["geometry"] == "grid"

    def test_source_time_filter_supports_cftime_calendar(self) -> None:
        class NoLeapReader:
            @property
            def name(self) -> str:
                return "noleap_probe"

            @property
            def geometry(self) -> DataGeometry:
                return DataGeometry.GRID

            def open(self, file_paths, variables=None):  # noqa: ANN001
                times = [
                    cftime.DatetimeNoLeap(2008, 7, 1),
                    cftime.DatetimeNoLeap(2008, 7, 2),
                    cftime.DatetimeNoLeap(2008, 7, 3),
                ]
                return xr.Dataset(
                    {"aod": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
                    coords={"time": times, "lat": [0.0], "lon": [0.0]},
                )

        source_registry.register("noleap_probe", NoLeapReader, replace=True)
        try:
            ctx = PipelineContext(
                config={
                    "analysis": {
                        "start_time": datetime(2008, 7, 2),
                        "end_time": datetime(2008, 7, 4),
                    },
                    "sources": {
                        "cam": {
                            "type": "noleap_probe",
                            "filename": "ignored.nc",
                            "variables": {"aod": {}},
                        }
                    },
                }
            )

            result = LoadSourcesStage().execute(ctx)

            assert result.status is StageStatus.COMPLETED
            ds = ctx.sources["cam"].data
            assert ds.sizes["time"] == 2
            assert list(ds["time"].values) == [
                cftime.DatetimeNoLeap(2008, 7, 2),
                cftime.DatetimeNoLeap(2008, 7, 3),
            ]
        finally:
            source_registry.unregister("noleap_probe")

    def test_stage_name(self) -> None:
        assert LoadSourcesStage().name == "load_sources"


class TestApplyVariableConfigValidRange:
    """valid_min/valid_max clamp configured source variables."""

    @staticmethod
    def _ds() -> xr.Dataset:
        return xr.Dataset({"o3": ("x", [-5.0, 10.0, 999.0])}, coords={"x": [0, 1, 2]})

    def test_valid_range_clamps_any_source(self) -> None:
        out = LoadSourcesStage._apply_variable_config(
            self._ds(), {"o3": {"valid_min": 0.0, "valid_max": 500.0}}
        )
        vals = out["o3"].values
        assert np.isnan(vals[0])  # below valid_min
        assert vals[1] == 10.0  # in range
        assert np.isnan(vals[2])  # above valid_max
