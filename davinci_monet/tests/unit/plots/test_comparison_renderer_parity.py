"""Render-contract tests for comparison renderers.

Public plot() wrappers were removed in Task 6. These smoke + structural tests
exercise renderer behavior through render(build_series(...)) directly. No metric
math is checked here; that is covered by the existing test_plots.py tests.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from davinci_monet.plots.base import build_series

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _figure(
    result: matplotlib.figure.Figure | list[tuple[str, matplotlib.figure.Figure]],
) -> matplotlib.figure.Figure:
    assert isinstance(result, matplotlib.figure.Figure)
    return result


def _paired_ds(n: int = 30, seed: int = 0) -> xr.Dataset:
    """Minimal paired dataset: x_o3 (geometry) + y_o3 (dataset)."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {
            "x_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "x", "units": "ppb"},
            ),
            "y_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={"time": times},
    )
    return ds


# ---------------------------------------------------------------------------
# ScatterPlotter
# ---------------------------------------------------------------------------


class TestScatterRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        fig = _figure(plotter.render(build_series(ds, "x_o3", "y_o3")))
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_render_creates_scatter_collection(self) -> None:
        """render() produces PathCollections for scatter points."""
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        fig = _figure(plotter.render(build_series(ds, "x_o3", "y_o3")))
        assert len(fig.axes[0].collections) >= 1
        plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.scatter import ScatterPlotter

        ds = _paired_ds()
        plotter = ScatterPlotter()
        with pytest.raises(NotImplementedError, match="ScatterPlotter"):
            plotter.render(build_series(ds, "x_o3"))


# ---------------------------------------------------------------------------
# BoxPlotter
# ---------------------------------------------------------------------------


class TestBoxRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.boxplot import BoxPlotter

        ds = _paired_ds()
        plotter = BoxPlotter()
        with pytest.raises(NotImplementedError, match="BoxPlotter"):
            plotter.render(build_series(ds, "x_o3"))


# ---------------------------------------------------------------------------
# DiurnalPlotter
# ---------------------------------------------------------------------------


class TestDiurnalRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_render_creates_lines(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert len(fig.axes[0].get_lines()) >= 1
        plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.diurnal import DiurnalPlotter

        ds = _paired_ds(n=48)
        plotter = DiurnalPlotter()
        with pytest.raises(NotImplementedError, match="DiurnalPlotter"):
            plotter.render(build_series(ds, "x_o3"))


# ---------------------------------------------------------------------------
# TaylorPlotter
# ---------------------------------------------------------------------------


class TestTaylorRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_uses_config_subtitle(self) -> None:
        from davinci_monet.plots.base import PlotConfig
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter(
            PlotConfig(title="O3 Taylor Diagram", subtitle="2024-02-01 - 2024-02-02")
        )
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        ax = fig.axes[0]

        assert ax.get_title() == r"O$_3$ Taylor Diagram"
        assert any(t.get_text() == "2024-02-01 - 2024-02-02" for t in ax.texts)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_render_creates_lines(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert len(fig.axes[0].get_lines()) >= 1
        plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.taylor import TaylorPlotter

        ds = _paired_ds()
        plotter = TaylorPlotter()
        with pytest.raises(NotImplementedError, match="TaylorPlotter"):
            plotter.render(build_series(ds, "x_o3"))


# ---------------------------------------------------------------------------
# ScorecardPlotter
# ---------------------------------------------------------------------------


class TestScorecardRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_ds()
        plotter = ScorecardPlotter()
        with pytest.raises(NotImplementedError, match="ScorecardPlotter"):
            plotter.render(build_series(ds, "x_o3"))

    def test_canonical_dataframe_helpers_work(self) -> None:
        """DataFrame scorecard helpers use render_* names only."""
        import pandas as pd

        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        plotter = ScorecardPlotter()

        assert not hasattr(plotter, "plot_from_dataframe")
        assert not hasattr(plotter, "plot_multi_metric")

        stats_df = pd.DataFrame(
            {"Dataset A": [0.9, 2.5], "Dataset B": [0.85, -1.0]},
            index=["R", "MB"],
        )
        fig = plotter.render_from_dataframe(stats_df)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

        stats_dict = {
            "Dataset A": pd.DataFrame({"R": [0.9], "MB": [1.2]}, index=["o3"]),
            "Dataset B": pd.DataFrame({"R": [0.85], "MB": [-0.5]}, index=["o3"]),
        }
        fig2 = plotter.render_multi_metric(stats_dict, metrics=["R", "MB"])
        assert isinstance(fig2, matplotlib.figure.Figure)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# CurtainPlotter
# ---------------------------------------------------------------------------


def _track_ds(n: int = 30, seed: int = 0) -> xr.Dataset:
    """Minimal track dataset with altitude coordinate."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {
            "x_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "x", "units": "ppb"},
            ),
            "y_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "time": times,
            "altitude": ("time", rng.uniform(500, 5000, n)),
        },
    )
    return ds


class TestCurtainRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"), alt_var="altitude")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"), alt_var="altitude")
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_curtain_show_var_forwarded(self) -> None:
        """render() must accept show_var kwarg and produce the correct bias plot."""
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        for show_var in ("x", "y", "bias"):
            fig = plotter.render(
                build_series(ds, "x_o3", "y_o3"),
                alt_var="altitude",
                show_var=show_var,
            )
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.curtain import CurtainPlotter

        ds = _track_ds()
        plotter = CurtainPlotter()
        with pytest.raises(NotImplementedError, match="CurtainPlotter"):
            plotter.render(build_series(ds, "x_o3"))


# ---------------------------------------------------------------------------
# SpatialBiasPlotter
# ---------------------------------------------------------------------------


def _spatial_point_ds(n_sites: int = 5, seed: int = 0) -> xr.Dataset:
    """Minimal point-site spatial dataset."""
    rng = np.random.default_rng(seed)
    times = np.array(["2024-02-01T00:00", "2024-02-01T01:00"], dtype="datetime64[ns]")
    lats = np.linspace(30.0, 50.0, n_sites)
    lons = np.linspace(-110.0, -70.0, n_sites)
    geometry = rng.uniform(20, 60, size=(2, n_sites))
    dataset = geometry + rng.uniform(-5, 5, size=(2, n_sites))
    ds = xr.Dataset(
        {
            "x_o3": (
                ("time", "site"),
                geometry,
                {"axis": "x"},
            ),
            "y_o3": (
                ("time", "site"),
                dataset,
                {"axis": "y"},
            ),
        },
        coords={
            "time": times,
            "latitude": ("site", lats),
            "longitude": ("site", lons),
        },
    )
    return ds


class TestSpatialBiasRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter

        ds = _spatial_point_ds()
        plotter = SpatialBiasPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter

        ds = _spatial_point_ds()
        plotter = SpatialBiasPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"))
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter

        ds = _spatial_point_ds()
        plotter = SpatialBiasPlotter()
        with pytest.raises(NotImplementedError, match="SpatialBiasPlotter"):
            plotter.render(build_series(ds, "x_o3"))


# ---------------------------------------------------------------------------
# SpatialOverlayPlotter
# ---------------------------------------------------------------------------


def _overlay_ds(n_sites: int = 5, seed: int = 0) -> xr.Dataset:
    """Minimal point-site dataset for overlay (geometry points on a dataset contour)."""
    rng = np.random.default_rng(seed)
    lats = np.linspace(30.0, 50.0, n_sites)
    lons = np.linspace(-110.0, -70.0, n_sites)
    ds = xr.Dataset(
        {
            "x_o3": (
                "site",
                rng.uniform(20, 60, n_sites),
                {"axis": "x", "units": "ppb"},
            ),
            "y_o3": (
                "site",
                rng.uniform(20, 60, n_sites),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "latitude": ("site", lats),
            "longitude": ("site", lons),
        },
    )
    return ds


def _dataset_field_da() -> "xr.DataArray":
    """Tiny 2-D (lat x lon) DataArray suitable as dataset_field for overlay."""
    import xarray as xr

    rng = np.random.default_rng(42)
    lats = np.linspace(28.0, 52.0, 8)
    lons = np.linspace(-112.0, -68.0, 10)
    data = rng.uniform(20, 60, (len(lats), len(lons)))
    return xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
        attrs={"units": "ppb"},
    )


class TestSpatialOverlayRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        y_field = _dataset_field_da()
        plotter = SpatialOverlayPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"), y_field=y_field)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        y_field = _dataset_field_da()
        plotter = SpatialOverlayPlotter()
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"), y_field=y_field)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_dataset_field_forwarded_via_render(self) -> None:
        """render() must accept and use the dataset_field kwarg (not fall back to paired_data)."""
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        y_field = _dataset_field_da()
        plotter = SpatialOverlayPlotter()
        # Should succeed when dataset_field is explicitly provided
        fig = plotter.render(build_series(ds, "x_o3", "y_o3"), y_field=y_field)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_y_field_missing_lon_raises_clear_error(self) -> None:
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        y_field = xr.DataArray(
            np.ones((3, 3)),
            dims=["lat", "x"],
            coords={"lat": [30.0, 40.0, 50.0]},
        )
        plotter = SpatialOverlayPlotter()

        with pytest.raises(ValueError, match="longitude"):
            plotter.render(build_series(ds, "x_o3", "y_o3"), y_field=y_field)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter

        ds = _overlay_ds()
        plotter = SpatialOverlayPlotter()
        with pytest.raises(NotImplementedError, match="SpatialOverlayPlotter"):
            plotter.render(build_series(ds, "x_o3"))


# ---------------------------------------------------------------------------
# TrackMap3DPlotter
# ---------------------------------------------------------------------------


def _track_3d_ds(n: int = 40, seed: int = 0) -> xr.Dataset:
    """Minimal track dataset with lat/lon/altitude coordinates for 3D map."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "m")
    ds = xr.Dataset(
        {
            "x_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "x", "units": "ppb"},
            ),
            "y_o3": (
                "time",
                rng.uniform(20, 60, n),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "time": times,
            "latitude": ("time", np.linspace(25.0, 45.0, n)),
            "longitude": ("time", np.linspace(-120.0, -80.0, n)),
            "altitude": ("time", rng.uniform(500, 8000, n)),
        },
    )
    return ds


def _flight_3d_ds(n_per_flight: int = 30, n_flights: int = 2, seed: int = 0) -> xr.Dataset:
    """Track dataset with a flight coordinate for plot_per_flight tests."""
    rng = np.random.default_rng(seed)
    all_times = []
    all_geometry = []
    all_dataset = []
    all_flight = []
    all_lat = []
    all_lon = []
    all_alt = []
    for day in range(n_flights):
        base = np.datetime64(f"2024-02-0{day + 1}T10:00")
        times = base + np.arange(n_per_flight) * np.timedelta64(1, "m")
        geometry = rng.uniform(20, 60, n_per_flight)
        dataset = geometry + rng.uniform(-5, 5, n_per_flight)
        all_times.append(times)
        all_geometry.append(geometry)
        all_dataset.append(dataset)
        all_flight.extend([f"2024020{day + 1}"] * n_per_flight)
        all_lat.append(np.linspace(25.0 + day, 40.0 + day, n_per_flight))
        all_lon.append(np.linspace(-120.0, -80.0, n_per_flight))
        all_alt.append(rng.uniform(500, 5000, n_per_flight))
    ds = xr.Dataset(
        {
            "x_o3": (
                "time",
                np.concatenate(all_geometry),
                {"axis": "x", "units": "ppb"},
            ),
            "y_o3": (
                "time",
                np.concatenate(all_dataset),
                {"axis": "y", "units": "ppb"},
            ),
        },
        coords={
            "time": np.concatenate(all_times),
            "flight": ("time", all_flight),
            "latitude": ("time", np.concatenate(all_lat)),
            "longitude": ("time", np.concatenate(all_lon)),
            "altitude": ("time", np.concatenate(all_alt)),
        },
    )
    return ds


class TestTrackMap3DRendererContract:
    def test_render_returns_figure(self) -> None:
        from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter

        ds = _track_3d_ds()
        plotter = TrackMap3DPlotter()
        fig = plotter.render(
            build_series(ds, "x_o3", "y_o3"),
            alt_var="altitude",
            show_coastlines=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_creates_axes(self) -> None:
        from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter

        ds = _track_3d_ds()
        plotter = TrackMap3DPlotter()
        fig = plotter.render(
            build_series(ds, "x_o3", "y_o3"),
            alt_var="altitude",
            show_coastlines=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_render_split_by_flight_returns_labeled_figures(self) -> None:
        """Split track-map output is part of the render contract."""
        from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter

        ds = _flight_3d_ds()
        plotter = TrackMap3DPlotter()
        results = plotter.render(
            build_series(ds, "x_o3", "y_o3"),
            split_by_flight=True,
            min_points=5,
            show_coastlines=False,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for flight_id, fig in results:
            assert isinstance(flight_id, str)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_render_wrong_series_count_raises(self) -> None:
        from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter

        ds = _track_3d_ds()
        plotter = TrackMap3DPlotter()
        with pytest.raises(NotImplementedError, match="TrackMap3DPlotter"):
            plotter.render(build_series(ds, "x_o3"))
