"""Single-source spatial field renderer: per-shape render mode + surface slicing.

The mark must be chosen by data SHAPE and verified PROGRAMMATICALLY (per the
geometry-aware-rendering rule — never "scatter for everything"):
- grid / swath  -> QuadMesh (pcolormesh)
- point / track / profile -> PathCollection (scatter)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PathCollection, QuadMesh  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.plot_config import PlotConfig  # noqa: E402
from davinci_monet.plots.renderers.spatial.base import MapConfig  # noqa: E402
from davinci_monet.plots.renderers.spatial.field import SpatialPlotter  # noqa: E402
from davinci_monet.tests.synthetic.geometries import (  # noqa: E402
    create_gridded_geometries,
    create_point_geometries,
    create_profile_geometries,
    create_swath_geometries,
    create_track_geometries,
)


def _single_figure(result: plt.Figure | list[tuple[str, plt.Figure]]) -> plt.Figure:
    assert not isinstance(result, list)
    return result


def _render(ds: xr.Dataset, var: str) -> tuple[plt.Figure, plt.Axes]:
    fig = _single_figure(SpatialPlotter().render(build_series(ds, var)))
    return fig, fig.axes[0]  # GeoAxes is created first; colorbar is a later axes


def _has(ax: plt.Axes, cls: type) -> bool:
    return any(isinstance(c, cls) for c in ax.collections)


def _aod_grid() -> xr.Dataset:
    lat = np.array([-1.0, 1.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    aod = np.array([[0.0, 0.012, 0.4, 1.4], [0.005, 0.03, 0.8, 1.2]])
    return xr.Dataset(
        {"aod": (("latitude", "longitude"), aod, {"units": "1"})},
        coords={"latitude": lat, "longitude": lon},
        attrs={"geometry": "grid"},
    )


def test_spatial_save_uses_full_figure_bbox_by_default(monkeypatch, tmp_path):
    plotter = SpatialPlotter()
    fig = _single_figure(plotter.render(build_series(_aod_grid(), "aod")))
    captured: dict[str, object] = {}

    def fake_savefig(path, **kwargs):
        captured.update(kwargs)
        path.write_bytes(b"pdf")

    monkeypatch.setattr(fig, "savefig", fake_savefig)

    plotter.save(fig, tmp_path / "map.pdf")

    assert captured["bbox_inches"] is fig.bbox_inches
    plt.close(fig)


def test_spatial_title_position_is_finite_with_subtitle():
    plotter = SpatialPlotter(
        config=PlotConfig(title="SW05 Column AOD", subtitle="2008-07-01")
    )
    fig = _single_figure(
        plotter.render(
            build_series(_aod_grid(), "aod"),
            show_coastlines=False,
            show_countries=False,
            show_states=False,
            show_gridlines=False,
            land_color="none",
            ocean_color="none",
        )
    )

    fig.canvas.draw()
    ax = fig.axes[0]
    _, title_y = ax.title.get_position()
    title_bbox = ax.title.get_window_extent(fig.canvas.get_renderer())

    assert ax.get_title() == "SW05 Column AOD"
    assert np.isfinite(title_y)
    assert np.all(np.isfinite(title_bbox.bounds))
    assert title_y > 1.01
    assert any(text.get_text() == "2008-07-01" for text in ax.texts)
    plt.close(fig)


def test_domain_type_sets_fixed_map_extent():
    """A named domain pins the map extent so sparse-data maps aren't auto-clipped
    to their few sites — every map shares the same fixed extent."""
    ds = create_point_geometries()
    fig = _single_figure(SpatialPlotter().render(build_series(ds, "O3"), domain_type="asia_aq"))
    ax = fig.axes[0]
    # GeoAxes get_extent() -> (lon_min, lon_max, lat_min, lat_max) in PlateCarree.
    # cartopy may pad one axis to preserve aspect, so assert the box is pinned to
    # the ASIA-AQ domain (NOT auto-scaled to the synthetic data) with a margin.
    lon0, lon1, lat0, lat1 = getattr(ax, "get_extent")()
    assert lon0 == pytest.approx(90.0, abs=2.0) and lon1 == pytest.approx(140.0, abs=2.0)
    assert lat0 == pytest.approx(0.0, abs=4.0) and lat1 == pytest.approx(45.0, abs=4.0)
    plt.close(fig)


@pytest.mark.parametrize(
    "maker,var",
    [
        (create_point_geometries, "O3"),
        (create_track_geometries, "O3"),
        (create_profile_geometries, "O3"),
    ],
)
def test_scatter_shapes_render_as_pathcollection(maker, var):
    ds = maker()
    fig, ax = _render(ds, var)
    assert _has(ax, PathCollection), f"{ds.attrs['geometry']} should render as scatter"
    assert not _has(ax, QuadMesh), f"{ds.attrs['geometry']} must NOT pcolormesh"
    assert len(fig.axes) >= 2, "expected a colorbar axes"
    plt.close(fig)


@pytest.mark.parametrize(
    "maker,var",
    [
        (create_gridded_geometries, "NO2"),
        (create_swath_geometries, "NO2"),
    ],
)
def test_mesh_shapes_render_as_quadmesh(maker, var):
    ds = maker()
    fig, ax = _render(ds, var)
    assert _has(ax, QuadMesh), f"{ds.attrs['geometry']} should render as pcolormesh"
    assert not _has(ax, PathCollection), f"{ds.attrs['geometry']} must NOT scatter"
    assert len(fig.axes) >= 2, "expected a colorbar axes"
    plt.close(fig)


def test_single_series_required():
    series = build_series(create_point_geometries(), "O3")
    with pytest.raises(NotImplementedError, match="exactly 1 series"):
        SpatialPlotter().render(series + series)


def test_grid_slices_surface_not_toa():
    """A 3-D grid with pressure ascending in index must slice the SURFACE (last)."""
    nt, nlev, nlat, nlon = 3, 3, 4, 5
    lev = np.array([100.0, 500.0, 1000.0])  # hPa increasing with index (CESM-style)
    lat = np.linspace(20, 50, nlat)
    lon = np.linspace(-120, -90, nlon)
    # value == level index, broadcast over time/lat/lon
    data = np.broadcast_to(
        np.arange(nlev, dtype=float)[None, :, None, None], (nt, nlev, nlat, nlon)
    ).copy()
    ds = xr.Dataset(
        {"O3": (["time", "lev", "lat", "lon"], data, {"units": "ppb"})},
        coords={
            "time": np.arange(nt),
            "lev": ("lev", lev),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
        attrs={"geometry": "grid"},
    )
    fig, ax = _render(ds, "O3")
    qm = next(c for c in ax.collections if isinstance(c, QuadMesh))
    arr = np.asarray(qm.get_array(), dtype=float)
    # Surface level (index -1) == 2.0; TOA (index 0) == 0.0.
    assert np.nanmin(arr) == pytest.approx(2.0)
    assert np.nanmax(arr) == pytest.approx(2.0)
    plt.close(fig)


def test_square_grid_preserves_lat_lon_dim_orientation():
    lat = np.array([30.0, 40.0, 50.0])
    lon = np.array([-120.0, -110.0, -100.0])
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    ds = xr.Dataset(
        {"O3": (["lat", "lon"], data, {"units": "ppb"})},
        coords={"lat": lat, "lon": lon},
        attrs={"geometry": "grid"},
    )

    fig, ax = _render(ds, "O3")

    qm = next(c for c in ax.collections if isinstance(c, QuadMesh))
    rendered = np.asarray(qm.get_array(), dtype=float).reshape(data.shape)
    np.testing.assert_allclose(rendered, data)
    plt.close(fig)


def test_grid_squeezes_singleton_product_group_dimension():
    ds = xr.Dataset(
        {
            "aod": (
                ("group", "lat", "lon"),
                np.ones((1, 2, 2)),
                {"units": "1"},
            )
        },
        coords={
            "group": ["2008-07-01"],
            "lat": [-1.0, 1.0],
            "lon": [0.0, 90.0],
        },
        attrs={"geometry": "grid"},
    )

    fig, ax = _render(ds, "aod")

    assert _has(ax, QuadMesh)
    plt.close(fig)


def test_grid_splits_multiple_product_groups_into_labeled_figures():
    ds = xr.Dataset(
        {
            "aod": (
                ("group", "lat", "lon"),
                np.ones((2, 2, 2)),
                {"units": "1"},
            )
        },
        coords={
            "group": ["2008-07-01", "2008-07-02"],
            "lat": [-1.0, 1.0],
            "lon": [0.0, 90.0],
        },
        attrs={"geometry": "grid"},
    )

    result = SpatialPlotter().render(build_series(ds, "aod"))

    assert isinstance(result, list)
    assert [label for label, _fig in result] == ["2008-07-01", "2008-07-02"]
    for _label, fig in result:
        assert _has(fig.axes[0], QuadMesh)
        plt.close(fig)


def test_grid_split_group_label_is_rendered_as_subtitle():
    ds = xr.Dataset(
        {
            "aod": (
                ("group", "lat", "lon"),
                np.ones((2, 2, 2)),
                {"units": "1"},
            )
        },
        coords={
            "group": ["2008-07-01", "2008-07-02"],
            "lat": [-1.0, 1.0],
            "lon": [0.0, 90.0],
        },
        attrs={"geometry": "grid"},
    )

    result = SpatialPlotter(config=PlotConfig(title="CAM7 Analyzed AOD")).render(
        build_series(ds, "aod")
    )

    assert isinstance(result, list)
    for label, fig in result:
        ax = fig.axes[0]
        assert ax.get_title() == "CAM7 Analyzed AOD"
        assert any(text.get_text() == label for text in ax.texts)
        plt.close(fig)


def test_spatial_render_kwargs_control_map_features(monkeypatch):
    plotter = SpatialPlotter()
    fig, ax = plotter.create_map_figure()
    feature_names: list[str] = []

    def fake_add_feature(feature, **kwargs):
        feature_names.append(str(getattr(feature, "name", feature)))

    monkeypatch.setattr(ax, "add_feature", fake_add_feature)

    plotter.render(
        build_series(_aod_grid(), "aod"),
        ax=ax,
        show_countries=False,
        show_states=False,
        land_color="none",
        ocean_color="none",
    )

    assert not any("admin_0" in name for name in feature_names)
    assert not any("admin_1" in name for name in feature_names)
    assert not any(name == "land" for name in feature_names)
    assert not any(name == "ocean" for name in feature_names)
    plt.close(fig)


def test_geosit_aod_style_preset_uses_boundary_norm_and_turbo_colormap():
    fig = _single_figure(
        SpatialPlotter().render(build_series(_aod_grid(), "aod"), style_preset="geosit_aod")
    )
    ax = fig.axes[0]
    qm = next(c for c in ax.collections if isinstance(c, QuadMesh))

    assert isinstance(qm.norm, BoundaryNorm)
    assert qm.cmap is not None
    assert qm.cmap.name == "turbo"
    np.testing.assert_allclose(
        qm.norm.boundaries[:7],
        [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
    )
    assert qm.norm.boundaries[-1] == pytest.approx(1.0)
    assert len(fig.axes) >= 2, "expected a colorbar axes"
    plt.close(fig)


def test_map_config_controls_gridline_style(monkeypatch):
    plotter = SpatialPlotter(map_config=MapConfig(show_gridlines=True, gridline_style="-"))
    fig, ax = plotter.create_map_figure()
    captured = {}

    def fake_gridlines(**kwargs):
        captured.update(kwargs)

        class FakeGridliner:
            top_labels = True
            right_labels = True

        return FakeGridliner()

    monkeypatch.setattr(ax, "gridlines", fake_gridlines)

    plotter.add_map_features(ax)

    assert captured["linestyle"] == "-"
    plt.close(fig)


def test_spatial_plot_accepts_explicit_discrete_levels_and_colorbar_extend():
    levels = [0.0, 0.1, 0.5, 1.0]

    fig = _single_figure(
        SpatialPlotter().render(
            build_series(_aod_grid(), "aod"),
            levels=levels,
            cmap="turbo",
            extend="max",
        )
    )
    ax = fig.axes[0]
    qm = next(c for c in ax.collections if isinstance(c, QuadMesh))

    assert isinstance(qm.norm, BoundaryNorm)
    np.testing.assert_allclose(qm.norm.boundaries, levels)
    assert qm.cmap is not None
    assert qm.cmap.name == "turbo"
    assert len(fig.axes) >= 2, "expected a colorbar axes"
    plt.close(fig)


def test_unknown_spatial_style_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown spatial style_preset"):
        SpatialPlotter().render(build_series(_aod_grid(), "aod"), style_preset="not-a-style")


@pytest.mark.integration
def test_spatial_runs_through_pipeline(tmp_path):
    """A ``type: spatial`` single-source plot generates a PNG via run_from_config."""
    from davinci_monet.pipeline.runner import PipelineRunner

    ds = create_gridded_geometries(variables=["NO2"])
    source_path = tmp_path / "grid.nc"
    ds.to_netcdf(source_path)

    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "sat": {
                "type": "generic",
                "files": str(source_path),
                "variables": {"NO2": {"units": "molec/cm2"}},
            }
        },
        "plots": {
            "no2_map": {
                "type": "spatial",
                "source": "sat",
                "variable": "NO2",
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    plots = ctx.results["plotting"].data["plots_generated"]
    pngs = [p for p in plots if p.endswith(".png")]
    assert len(pngs) == 1
    assert "no2_map" in pngs[0]
