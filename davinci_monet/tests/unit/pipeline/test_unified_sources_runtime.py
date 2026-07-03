"""Runtime regressions for the unified data-source refactor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.integration


def _write_swath_source(path: Path, *, n_scan: int = 30, n_pix: int = 8) -> None:
    """Write a raw L2 swath NetCDF (2-D lat/lon on scanline/pixel dims).

    Mirrors a satellite L2 product: along-track scanlines, cross-track pixels,
    a single retrieved variable, and a per-scanline time coordinate. Lat/lon
    fall inside the GRID source's footprint so binning lands on real cells.
    """
    scan_lats = np.linspace(-48.0, -2.0, n_scan)
    scan_lons = np.linspace(102.0, 158.0, n_scan)
    pix_offsets = np.linspace(-3.0, 3.0, n_pix)
    lat2d = scan_lats[:, None] + pix_offsets[None, :] * 0.1
    lon2d = scan_lons[:, None] + pix_offsets[None, :]
    # Deterministic retrieved field: smooth gradient so binning is reproducible.
    aod = 0.1 + 0.002 * (lat2d - lat2d.min()) + 0.001 * (lon2d - lon2d.min())
    scan_times = np.array(["2019-12-21T12:00"], dtype="datetime64[m]").repeat(n_scan)
    ds = xr.Dataset(
        {"aod_550nm": (["scanline", "pixel"], aod.astype("float32"))},
        coords={
            "scanline": np.arange(n_scan),
            "pixel": np.arange(n_pix),
            "latitude": (["scanline", "pixel"], lat2d),
            "longitude": (["scanline", "pixel"], lon2d),
            "time": (["scanline"], scan_times),
        },
        attrs={"geometry": "swath"},
    )
    ds.to_netcdf(path)


def _write_grid_aod_source(path: Path, *, n_lat: int = 12, n_lon: int = 12) -> None:
    """Write a GRID NetCDF (1-D lat/lon) covering the swath footprint."""
    lat = np.linspace(-50.0, 0.0, n_lat)
    lon = np.linspace(100.0, 160.0, n_lon)
    times = np.array(["2019-12-21T00:00"], dtype="datetime64[m]")
    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 0.4, size=(1, n_lat, n_lon)).astype("float32")
    ds = xr.Dataset(
        {"AOD": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(path)


def _write_grid_source(path: Path, *, offset: float = 0.0) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    lat = np.array([40.0, 41.0])
    lon = np.array([-105.0, -104.0])
    values = np.arange(8, dtype=float).reshape(2, 2, 2) + offset
    ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(path)


def _write_point_source(path: Path) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    ds = xr.Dataset(
        {"o3": (("time", "site"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "time": times,
            "site": np.array([0, 1]),
            "latitude": ("site", np.array([40.0, 41.0])),
            "longitude": ("site", np.array([-105.0, -104.0])),
        },
        attrs={"geometry": "point"},
    )
    ds.to_netcdf(path)


def test_sources_config_pairs_from_pair_variables(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    y_path = tmp_path / "dataset.nc"
    x_path = tmp_path / "geometry.nc"
    _write_grid_source(y_path)
    _write_point_source(x_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "files": str(y_path),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "filename": str(x_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_airnow_o3": {
                "x": {"source": "airnow", "variable": "o3"},
                "y": {"source": "cam", "variable": "O3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    assert set(result.context.sources) == {"cam", "airnow"}
    assert set(result.context.paired) == {"cam_airnow_o3"}
    paired = result.context.paired["cam_airnow_o3"].data
    pair_info = result.context.paired["cam_airnow_o3"].pairing_info
    assert set(paired.data_vars) == {"cam_O3", "airnow_o3"}
    assert paired["cam_O3"].attrs["source_label"] == "cam"
    assert paired["cam_O3"].attrs["dataset_variable"] == "O3"
    assert paired["airnow_o3"].attrs["source_label"] == "airnow"
    assert paired["airnow_o3"].attrs["dataset_variable"] == "o3"
    assert pair_info["axis_variables"] == {"x": "o3", "y": "O3"}
    assert paired["cam_O3"].attrs["canonical_name"] == paired["airnow_o3"].attrs["canonical_name"]


def test_grid_x_point_y_uses_point_geometry_but_preserves_config_axes(tmp_path: Path) -> None:
    """x/y are plot axes; pairing direction follows geometry precedence."""
    from davinci_monet.pipeline.runner import PipelineRunner

    grid_path = tmp_path / "cam.nc"
    point_path = tmp_path / "airnow.nc"
    _write_grid_source(grid_path)
    _write_point_source(point_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "files": str(grid_path),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "filename": str(point_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_airnow_o3": {
                "x": {"source": "cam", "variable": "O3"},
                "y": {"source": "airnow", "variable": "o3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    paired_obj = result.context.paired["cam_airnow_o3"]
    paired = paired_obj.data
    assert paired_obj.x_source == "cam"
    assert paired_obj.y_source == "airnow"
    assert paired_obj.geometry.name == "POINT"
    assert paired["cam_O3"].attrs["axis"] == "x"
    assert paired["airnow_o3"].attrs["axis"] == "y"
    assert bool(np.isfinite(paired["cam_O3"].values).any())


def test_sources_config_without_pairs_loads_sources_only(tmp_path: Path) -> None:
    """A sources config without ``pairs:`` loads sources and produces no pairs."""
    from davinci_monet.pipeline.runner import PipelineRunner

    y_path = tmp_path / "dataset.nc"
    x_path = tmp_path / "geometry.nc"
    _write_grid_source(y_path)
    _write_point_source(x_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "files": str(y_path),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "filename": str(x_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    assert set(result.context.sources) == {"cam", "airnow"}
    assert result.context.paired == {}


def test_sources_config_supports_dataset_dataset_pair(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    x_path = tmp_path / "geometry_grid.nc"
    y_path = tmp_path / "dataset_grid.nc"
    _write_grid_source(x_path, offset=0.0)
    _write_grid_source(y_path, offset=1.0)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam_grid": {"type": "generic", "files": str(x_path)},
            "cam_offset": {"type": "generic", "files": str(y_path)},
        },
        "pairs": {
            "cam_grid_cam_offset_o3": {
                "x": {"source": "cam_grid", "variable": "O3"},
                "y": {"source": "cam_offset", "variable": "O3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    paired = result.context.paired["cam_grid_cam_offset_o3"].data
    assert set(paired.data_vars) == {"cam_grid_O3", "cam_offset_O3"}
    assert paired["cam_grid_O3"].attrs["axis"] == "x"
    assert paired["cam_offset_O3"].attrs["axis"] == "y"


def test_plot_data_geometry_without_pair_spec_uses_paired_dataset(
    tmp_path: Path,
) -> None:
    from davinci_monet.pipeline.stages import PipelineContext, PlottingStage, StageStatus

    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    paired = xr.Dataset(
        {
            "airnow_o3": ("time", np.array([1.0, 2.0])),
            "cam_O3": ("time", np.array([1.1, 2.2])),
        },
        coords={"time": times},
    )
    paired["airnow_o3"].attrs.update(
        {
            "axis": "x",
            "source_label": "airnow",
            "dataset_variable": "o3",
            "canonical_name": "o3",
        }
    )
    paired["cam_O3"].attrs.update(
        {
            "axis": "y",
            "source_label": "cam",
            "dataset_variable": "O3",
            "canonical_name": "o3",
        }
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "plots": {"scatter_o3": {"type": "scatter", "pairs": ["cam_airnow"]}},
        }
    )
    ctx.paired["cam_airnow"] = paired

    result = PlottingStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert len(result.data["plots_generated"]) == 2


def test_plot_sources_pair_spec_uses_geometry_and_dataset(
    tmp_path: Path,
) -> None:
    from davinci_monet.pipeline.stages import PipelineContext, PlottingStage, StageStatus

    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    paired = xr.Dataset(
        {
            "airnow_o3": ("time", np.array([1.0, 2.0])),
            "cam_O3": ("time", np.array([1.1, 2.2])),
        },
        coords={"time": times},
    )
    paired["airnow_o3"].attrs.update(
        {
            "axis": "x",
            "source_label": "airnow",
            "dataset_variable": "o3",
            "canonical_name": "o3",
        }
    )
    paired["cam_O3"].attrs.update(
        {
            "axis": "y",
            "source_label": "cam",
            "dataset_variable": "O3",
            "canonical_name": "o3",
        }
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "pairs": {
                "cam_airnow_o3": {
                    "x": {"source": "airnow", "variable": "o3"},
                    "y": {"source": "cam", "variable": "O3"},
                }
            },
            "plots": {"scatter_o3": {"type": "scatter", "pairs": ["cam_airnow_o3"]}},
        }
    )
    ctx.paired["cam_airnow_o3"] = paired

    result = PlottingStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert len(result.data["plots_generated"]) == 2


def test_plot_pair_spec_uses_configured_pair_name(
    tmp_path: Path,
) -> None:
    from davinci_monet.pipeline.stages import PipelineContext, PlottingStage, StageStatus

    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    paired = xr.Dataset(
        {
            "airnow_o3": ("time", np.array([1.0, 2.0])),
            "cam_o3": ("time", np.array([1.1, 2.2])),
        },
        coords={"time": times},
    )
    paired["airnow_o3"].attrs.update({"axis": "x", "source_label": "airnow"})
    paired["cam_o3"].attrs.update({"axis": "y", "source_label": "cam"})
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "pairs": {
                "cam_airnow_o3": {
                    "x": {"source": "airnow", "variable": "o3"},
                    "y": {"source": "cam", "variable": "O3"},
                }
            },
            "plots": {"scatter_o3": {"type": "scatter", "pairs": ["cam_airnow_o3"]}},
        }
    )
    ctx.paired["cam_airnow_o3"] = paired

    result = PlottingStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert len(result.data["plots_generated"]) == 2


def test_invalid_sources_pair_missing_variable_fails() -> None:
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    ctx = PipelineContext(
        config={
            "pairs": {
                "cam_airnow_o3": {
                    "x": {"source": "airnow"},
                    "y": {"source": "cam", "variable": "O3"},
                }
            }
        },
        sources={
            "cam": SourceData(
                data=xr.Dataset({"O3": ("time", np.array([1.0]))}),
                label="cam",
                source_type="generic",
                geometry=DataGeometry.GRID,
            ),
            "airnow": SourceData(
                data=xr.Dataset({"o3": ("time", np.array([1.0]))}),
                label="airnow",
                source_type="pt_sfc",
                geometry=DataGeometry.POINT,
            ),
        },
    )

    result = PairingStage().execute(ctx)

    assert result.status is StageStatus.FAILED
    assert "cam_airnow_o3" in str(result.error)
    assert "missing variable" in str(result.error)
    assert ctx.paired == {}


def test_invalid_sources_pair_unknown_source_fails() -> None:
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    ctx = PipelineContext(
        config={
            "pairs": {
                "cam_missing_o3": {
                    "x": {"source": "missing_geometry", "variable": "o3"},
                    "y": {"source": "cam", "variable": "O3"},
                }
            }
        },
        sources={
            "cam": SourceData(
                data=xr.Dataset({"O3": ("time", np.array([1.0]))}),
                label="cam",
                source_type="generic",
                geometry=DataGeometry.GRID,
            )
        },
    )

    stage = PairingStage()
    assert stage.validate(ctx)
    result = stage.execute(ctx)

    assert result.status is StageStatus.FAILED
    assert "cam_missing_o3" in str(result.error)
    assert "unknown source" in str(result.error)
    assert ctx.paired == {}


def test_single_dataset_source_gets_descriptive_stats(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    source_path = tmp_path / "cam.nc"
    _write_grid_source(source_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "files": str(source_path),
                "variables": {"O3": {"units": "ppb"}},
            }
        },
        "stats": {"metrics": ["N"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    stats = result.context.results["statistics"].data
    assert "cam" in stats
    assert "O3" in stats["cam"]
    assert stats["cam"]["O3"]["N"] == 8


def test_single_source_plot_uses_source_key_not_geometry_key(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    source_path = tmp_path / "cam.nc"
    _write_grid_source(source_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {
                "type": "generic",
                "files": str(source_path),
                "variables": {"O3": {"units": "ppb"}},
            }
        },
        "plots": {
            "hist_o3": {
                "type": "histogram",
                "source": "cam",
                "variable": "O3",
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    plots = result.context.results["plotting"].data["plots_generated"]
    assert len([p for p in plots if p.endswith(".png")]) == 1


def test_unsupported_source_pair_all_failed_fails_stage_with_warning() -> None:
    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.pipeline.stages import (
        PairingStage,
        PipelineContext,
        SourceData,
        StageStatus,
    )

    point_a = xr.Dataset(
        {"o3": ("site", np.array([1.0]))},
        coords={
            "site": [0],
            "latitude": ("site", [40.0]),
            "longitude": ("site", [-105.0]),
        },
        attrs={"geometry": "point"},
    )
    track_b = xr.Dataset(
        {"o3": ("time", np.array([1.2]))},
        coords={
            "time": np.array(["2024-01-01T00:00"], dtype="datetime64[m]"),
            "latitude": ("time", [40.0]),
            "longitude": ("time", [-105.0]),
        },
        attrs={"geometry": "track"},
    )
    ctx = PipelineContext(
        config={
            "pairs": {
                "a_b_o3": {
                    "x": {"source": "a", "variable": "o3"},
                    "y": {"source": "b", "variable": "o3"},
                }
            }
        },
        sources={
            "a": SourceData(point_a, "a", "pt_sfc", DataGeometry.POINT),
            "b": SourceData(track_b, "b", "icartt", DataGeometry.TRACK),
        },
    )

    result = PairingStage().execute(ctx)

    # Zero-output contract (FABLE remediation Gap 1): when the only configured
    # pair fails, the stage fails rather than silently reporting success. The
    # per-item warning is still recorded in metadata for diagnostics.
    assert result.status is StageStatus.FAILED
    assert result.error == "all 1 pairs failed"
    assert ctx.metadata["pairing_errors"]
    assert result.metadata["warnings"] == ctx.metadata["pairing_errors"]
    assert "a_b_o3" in result.metadata["warnings"][0]
    assert "Unsupported pairing combination" in result.metadata["warnings"][0]


def test_sources_config_supports_geometry_geometry_grid_pair(tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    x_path = tmp_path / "modis_grid.nc"
    y_path = tmp_path / "viirs_grid.nc"
    _write_grid_source(x_path, offset=0.0)
    _write_grid_source(y_path, offset=1.0)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "modis": {"type": "generic", "files": str(x_path)},
            "viirs": {"type": "generic", "files": str(y_path)},
        },
        "pairs": {
            "modis_viirs_o3": {
                "x": {"source": "modis", "variable": "O3"},
                "y": {"source": "viirs", "variable": "O3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    paired = result.context.paired["modis_viirs_o3"].data
    assert set(paired.data_vars) == {"modis_O3", "viirs_O3"}
    assert paired["modis_O3"].attrs["axis"] == "x"
    assert paired["viirs_O3"].attrs["axis"] == "y"


def test_unified_source_applies_resample(tmp_path: Path) -> None:
    """A `sources:` geometry with `resample` is averaged to the target frequency at load."""
    import pandas as pd

    from davinci_monet.pipeline.runner import PipelineRunner

    src = tmp_path / "hf.nc"
    times = pd.date_range("2024-01-01T00:00", periods=4, freq="15min")
    ds = xr.Dataset(
        {"o3": (("time", "site"), np.array([[10.0], [20.0], [30.0], [40.0]]))},
        coords={
            "time": times,
            "site": [0],
            "latitude": ("site", [40.0]),
            "longitude": ("site", [-105.0]),
        },
        attrs={"geometry": "point"},
    )
    ds.to_netcdf(src)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "hf": {
                "type": "generic",
                "files": str(src),
                "resample": "h",
                "track_sample_count": True,
                "variables": {"o3": {"units": "ppb"}},
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    loaded = result.context.sources["hf"].data
    assert loaded.sizes["time"] == 1
    assert float(loaded["o3"].isel(time=0, site=0)) == 25.0
    assert "sample_count" in loaded
    assert int(loaded["sample_count"].isel(time=0, site=0)) == 4


def test_sources_config_pairs_swath_onto_grid(tmp_path: Path) -> None:
    """A SWATH source pairs onto a GRID geometry end-to-end via run_from_config.

    Proves the production ``SwathGridStrategy`` (numba binning) is what the
    engine routes a swath-vs-grid pair through: the swath pixels are binned onto
    the grid, so the paired output is GRID-geometry ``(time, lon, lat)`` with
    geometry/dataset variables and axis tags. Before SwathGridStrategy was
    registered, the engine could route through a per-pixel SwathStrategy, whose
    output uses ``(y, x)``/``(pixel,)`` dimensions. These assertions require the
    binned grid path instead.
    """
    from davinci_monet.pipeline.runner import PipelineRunner

    swath_path = tmp_path / "modis_l2.nc"
    grid_path = tmp_path / "cam.nc"
    _write_swath_source(swath_path)
    _write_grid_aod_source(grid_path)

    config = {
        "analysis": {
            "output_dir": str(tmp_path / "out"),
        },
        "sources": {
            "cam": {
                "type": "generic",
                "files": str(grid_path),
                "variables": {"AOD": {"units": "1"}},
            },
            "modis": {
                "type": "satellite_l2",
                "files": str(swath_path),
                "variables": {"aod_550nm": {"units": "1"}},
            },
        },
        "pairs": {
            "cam_modis_aod": {
                "x": {"source": "modis", "variable": "aod_550nm"},
                "y": {"source": "cam", "variable": "AOD"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    failed = [
        f"{s.stage_name}: {s.error}" for s in result.stage_results if s.status.name == "FAILED"
    ]
    assert result.success, f"Pipeline failed. Errors: {failed}"
    assert result.context is not None
    assert set(result.context.sources) == {"cam", "modis"}
    assert set(result.context.paired) == {"cam_modis_aod"}

    paired = result.context.paired["cam_modis_aod"].data
    # SwathGridStrategy binned onto the grid: GRID-geometry output rather than
    # per-pixel (scanline, pixel)/(y, x) output.
    assert set(paired.dims) >= {"time", "lon", "lat"}
    assert not ({"scanline", "pixel", "y", "x"} & set(paired.dims))
    # Geometry and dataset share the canonical stem (aod_550nm) under their
    # source-label prefixes, with axis + axis tags.
    assert "modis_aod_550nm" in paired.data_vars
    assert "cam_AOD" in paired.data_vars
    assert paired["modis_aod_550nm"].attrs["axis"] == "x"
    assert paired["cam_AOD"].attrs["axis"] == "y"
    assert paired["modis_aod_550nm"].attrs["axis"] == "x"
    assert paired["cam_AOD"].attrs["axis"] == "y"
    # At least one grid cell received binned swath pixels (non-NaN), proving the
    # numba binning actually ran end-to-end.
    assert bool(np.isfinite(paired["modis_aod_550nm"].values).any())


def test_two_explicit_pairs_both_produced_via_executor(tmp_path: Path) -> None:
    """Two independent explicit pairs both land in ``context.paired``.

    Exercises the real user path (``PipelineRunner.run_from_config``) with two
    eager (in-memory, numpy-backed) source pairs. Proves the bounded concurrent
    executor in PairingStage runs *all* jobs. Both pair keys and their
    source-label-prefixed variables must be present.
    """
    from davinci_monet.pipeline.runner import PipelineRunner

    y_a = tmp_path / "cam_a.nc"
    y_b = tmp_path / "cam_b.nc"
    x_path = tmp_path / "airnow.nc"
    _write_grid_source(y_a, offset=0.0)
    _write_grid_source(y_b, offset=5.0)
    _write_point_source(x_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam_a": {
                "type": "generic",
                "files": str(y_a),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "cam_b": {
                "type": "generic",
                "files": str(y_b),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "filename": str(x_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_a_airnow_o3": {
                "x": {"source": "airnow", "variable": "o3"},
                "y": {"source": "cam_a", "variable": "O3"},
            },
            "cam_b_airnow_o3": {
                "x": {"source": "airnow", "variable": "o3"},
                "y": {"source": "cam_b", "variable": "O3"},
            },
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    # Both jobs ran: both keys present (the executor did not stop after one).
    assert set(result.context.paired) == {"cam_a_airnow_o3", "cam_b_airnow_o3"}
    paired_a = result.context.paired["cam_a_airnow_o3"].data
    paired_b = result.context.paired["cam_b_airnow_o3"].data
    assert set(paired_a.data_vars) == {"cam_a_O3", "airnow_o3"}
    assert set(paired_b.data_vars) == {"cam_b_O3", "airnow_o3"}


def test_two_explicit_pairs_with_max_pair_workers(tmp_path: Path) -> None:
    """``pairing.max_pair_workers: 2`` over 2 eager pairs still produces both.

    Smoke test that the ThreadPoolExecutor path (worker count > 1, > 1 eager job)
    runs all jobs and writes both into ``context.paired`` from the main thread.
    """
    from davinci_monet.pipeline.runner import PipelineRunner

    y_a = tmp_path / "cam_a.nc"
    y_b = tmp_path / "cam_b.nc"
    x_path = tmp_path / "airnow.nc"
    _write_grid_source(y_a, offset=0.0)
    _write_grid_source(y_b, offset=5.0)
    _write_point_source(x_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "pairing": {"max_pair_workers": 2},
        "sources": {
            "cam_a": {
                "type": "generic",
                "files": str(y_a),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "cam_b": {
                "type": "generic",
                "files": str(y_b),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "filename": str(x_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_a_airnow_o3": {
                "x": {"source": "airnow", "variable": "o3"},
                "y": {"source": "cam_a", "variable": "O3"},
            },
            "cam_b_airnow_o3": {
                "x": {"source": "airnow", "variable": "o3"},
                "y": {"source": "cam_b", "variable": "O3"},
            },
        },
        "stats": {"metrics": ["N", "MB"]},
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success
    assert result.context is not None
    assert set(result.context.paired) == {"cam_a_airnow_o3", "cam_b_airnow_o3"}
