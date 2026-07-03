"""Rasterization sweep: dense mesh/fill/scatter layers must be rasterized so
PDF output stays small, per CLAUDE.md's PDF-only iCloud delivery convention.
Covers the renderers named in FABLE_REVIEW.md section 6 and remediation plan
Task 15 Steps 3-4: ``draw_spatial_field`` (spatial/base.py, exercised via
spatial_bias's grid and point paths), spatial_overlay, curtain, lma_density,
scatter, scorecard, and vertical_profile.

``ContourSet`` (both ``ax.contour`` and ``ax.contourf``) is excluded from the
"must be rasterized" requirement: in the pinned matplotlib (``<3.9``),
``ContourSet.draw`` does not support rasterization at all, so both the
constructor kwarg and ``set_rasterized()`` raise a ``UserWarning`` (escalated
to an error by this suite's ``error::UserWarning`` gate) regardless of
whether the contours are filled or line-only. Filled contour layers
(spatial_overlay, curtain) therefore stay vector in this environment; this
test guards against a future fix attempting to rasterize them and breaking
the gate.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.contour as mcontour  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PathCollection, PolyCollection, QuadMesh  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.curtain import CurtainPlotter  # noqa: E402
from davinci_monet.plots.renderers.lma_density import LMADensityPlotter  # noqa: E402
from davinci_monet.plots.renderers.scatter import ScatterPlotter  # noqa: E402
from davinci_monet.plots.renderers.scorecard import ScorecardPlotter  # noqa: E402
from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter  # noqa: E402
from davinci_monet.plots.renderers.spatial.overlay import SpatialOverlayPlotter  # noqa: E402
from davinci_monet.plots.renderers.vertical_profile import VerticalProfilePlotter  # noqa: E402


def _figure(
    result: plt.Figure | list[tuple[str, plt.Figure]],
) -> plt.Figure:
    """Narrow the unified ``render()`` union to a single Figure.

    ``ScatterPlotter.render`` keeps the base ``Figure | list[...]`` return
    annotation; the scatter tests below always produce a single figure.
    """
    assert isinstance(result, plt.Figure)
    return result


def _dense_layers(fig: plt.Figure) -> list:
    """Every dense mesh/fill/scatter/contour artist on the primary axes.

    Only ``fig.axes[0]`` (the GeoAxes/data axes created first) is scanned —
    a later colorbar axes carries its own gradient ``QuadMesh`` that mirrors
    the mappable's rasterized state and is not itself a data layer (mirrors
    the ``fig.axes[0]`` convention in test_spatial_single_source.py).
    Cartopy ``FeatureArtist`` map decorations (land/ocean/borders/gridlines)
    are also excluded — not the data layer the convention targets.
    """
    ax = fig.axes[0]
    layers: list = [
        artist
        for artist in ax.collections
        if isinstance(artist, (mcontour.ContourSet, QuadMesh, PathCollection, PolyCollection))
    ]
    layers.extend(ax.images)
    return layers


def _assert_dense_layers_rasterized(fig: plt.Figure) -> None:
    """Meshes/scatter/images are rasterized; ``ContourSet`` artists (filled
    or line) are not — the pinned matplotlib can't rasterize them at all."""
    layers = _dense_layers(fig)
    assert layers, "expected at least one dense data-layer artist"
    for artist in layers:
        if isinstance(artist, mcontour.ContourSet):
            assert artist.get_rasterized() is not True, f"contour set rasterized: {artist}"
        else:
            assert artist.get_rasterized() is True, f"dense layer not rasterized: {type(artist)}"


def _gridded_bias_dataset() -> xr.Dataset:
    lat = np.linspace(-89.5, 89.5, 6)
    lon = np.linspace(-179.5, 179.5, 8)
    rng = np.random.default_rng(0)
    x_field = xr.DataArray(
        rng.uniform(0, 1, (6, 8)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    y_field = xr.DataArray(
        rng.uniform(0, 1, (6, 8)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    return xr.Dataset({"x_aod": x_field, "y_aod": y_field})


def _point_bias_dataset() -> xr.Dataset:
    site = np.arange(8)
    lat = xr.DataArray(np.linspace(20, 50, 8), dims=("site",), coords={"site": site})
    lon = xr.DataArray(np.linspace(100, 140, 8), dims=("site",), coords={"site": site})
    rng = np.random.default_rng(1)
    x_field = xr.DataArray(rng.uniform(0, 1, 8), dims=("site",), coords={"site": site})
    y_field = xr.DataArray(rng.uniform(0, 1, 8), dims=("site",), coords={"site": site})
    return xr.Dataset({"x_v": x_field, "y_v": y_field}, coords={"lat": lat, "lon": lon})


def test_spatial_bias_grid_pcolormesh_rasterized():
    ds = _gridded_bias_dataset()
    fig = SpatialBiasPlotter().render(
        build_series(ds, "x_aod", "y_aod"), lat_var="lat", lon_var="lon"
    )
    assert any(isinstance(c, QuadMesh) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_spatial_bias_point_scatter_rasterized():
    ds = _point_bias_dataset()
    fig = SpatialBiasPlotter().render(build_series(ds, "x_v", "y_v"), lat_var="lat", lon_var="lon")
    assert any(isinstance(c, PathCollection) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_spatial_overlay_contourf_and_scatter_rasterized():
    n = 6
    lat_pt = np.linspace(30, 40, n)
    lon_pt = np.linspace(-120, -100, n)
    rng = np.random.default_rng(2)
    x_vals = rng.normal(40, 5, n)
    y_vals = x_vals + rng.normal(2, 1, n)
    x_ds = xr.Dataset(
        {
            "x_o3": (["site"], x_vals, {"units": "ppbv"}),
            "y_o3": (["site"], y_vals, {"units": "ppbv"}),
        },
        coords={"site": np.arange(n), "latitude": ("site", lat_pt), "longitude": ("site", lon_pt)},
    )
    lat_g = np.linspace(30, 40, 4)
    lon_g = np.linspace(-120, -100, 5)
    y_field = xr.DataArray(
        rng.normal(40, 5, (len(lat_g), len(lon_g))),
        dims=("lat", "lon"),
        coords={"lat": lat_g, "lon": lon_g},
        attrs={"units": "ppbv"},
    )

    fig = SpatialOverlayPlotter().render(build_series(x_ds, "x_o3", "y_o3"), y_field=y_field)
    layers = _dense_layers(fig)
    assert any(isinstance(a, mcontour.ContourSet) and a.filled for a in layers)
    assert any(isinstance(a, PathCollection) for a in layers)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_curtain_2d_contourf_rasterized():
    time = pd.date_range("2023-01-01", periods=5, freq="h")
    altitude = np.array([500.0, 1500.0, 3000.0, 6000.0])
    rng = np.random.default_rng(3)
    x_vals = rng.normal(40, 5, (5, 4))
    y_vals = x_vals + rng.normal(2, 1, (5, 4))
    ds = xr.Dataset(
        {
            "x_o3": (["time", "altitude"], x_vals, {"units": "ppbv"}),
            "y_o3": (["time", "altitude"], y_vals, {"units": "ppbv"}),
        },
        coords={"time": time, "altitude": altitude},
    )
    fig = CurtainPlotter().render(build_series(ds, "x_o3", "y_o3"), alt_var="altitude")
    layers = _dense_layers(fig)
    assert any(isinstance(a, mcontour.ContourSet) and a.filled for a in layers)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_curtain_1d_trajectory_scatter_rasterized():
    time = pd.date_range("2023-01-01", periods=20, freq="min")
    altitude = np.linspace(500, 6000, 20)
    rng = np.random.default_rng(4)
    x_vals = rng.normal(40, 5, 20)
    y_vals = x_vals + rng.normal(2, 1, 20)
    ds = xr.Dataset(
        {
            "x_o3": (["time"], x_vals, {"units": "ppbv"}),
            "y_o3": (["time"], y_vals, {"units": "ppbv"}),
        },
        coords={"time": time, "altitude": ("time", altitude)},
    )
    fig = CurtainPlotter().render(build_series(ds, "x_o3", "y_o3"), alt_var="altitude")
    assert any(isinstance(c, PathCollection) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_lma_density_pcolormesh_rasterized():
    time = pd.date_range("2023-01-01", periods=3, freq="h")
    lat = np.linspace(33, 36, 4)
    lon = np.linspace(-98, -95, 5)
    rng = np.random.default_rng(5)
    data = rng.integers(0, 5, (3, 4, 5)).astype(float)
    ds = xr.Dataset(
        {"flash_extent": (["time", "latitude", "longitude"], data)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    fig = LMADensityPlotter().render(build_series(ds, "flash_extent"))
    assert any(isinstance(c, QuadMesh) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def _scatter_dataset() -> xr.Dataset:
    n = 30
    rng = np.random.default_rng(6)
    x = rng.normal(40, 5, n)
    y = x + rng.normal(2, 1, n)
    z = rng.normal(0, 1, n)
    return xr.Dataset(
        {
            "x_o3": (["time"], x, {"units": "ppbv"}),
            "y_o3": (["time"], y, {"units": "ppbv"}),
            "z_o3": (["time"], z, {"units": "ppbv"}),
        },
        coords={"time": pd.date_range("2023-01-01", periods=n, freq="h")},
    )


def test_scatter_plain_rasterized():
    ds = _scatter_dataset()
    fig = _figure(ScatterPlotter().render(build_series(ds, "x_o3", "y_o3")))
    assert any(isinstance(c, PathCollection) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_scatter_density_rasterized():
    ds = _scatter_dataset()
    fig = _figure(ScatterPlotter().render(build_series(ds, "x_o3", "y_o3"), show_density=True))
    assert any(isinstance(c, PathCollection) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_scatter_color_by_rasterized():
    ds = _scatter_dataset()
    fig = _figure(ScatterPlotter().render(build_series(ds, "x_o3", "y_o3"), color_by="z_o3"))
    assert any(isinstance(c, PathCollection) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_scorecard_imshow_rasterized():
    df = pd.DataFrame({"R": [0.8], "MB": [1.2]}, index=["O3"])
    fig = ScorecardPlotter().render_from_dataframe(df)
    assert len(fig.axes[0].images) == 1
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def _vertical_profile_dataset() -> xr.Dataset:
    n = 40
    rng = np.random.default_rng(7)
    alt = np.linspace(500, 8000, n)
    o3 = rng.normal(40, 5, n)
    return xr.Dataset(
        {"o3": (["time"], o3, {"units": "ppbv"})},
        coords={"time": np.arange(n), "altitude": ("time", alt)},
    )


def test_vertical_profile_scatter_rasterized():
    ds = _vertical_profile_dataset()
    fig = VerticalProfilePlotter().render(build_series(ds, "o3"), mode="scatter")
    assert any(isinstance(c, PathCollection) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)


def test_vertical_profile_binned_fill_rasterized():
    ds = _vertical_profile_dataset()
    fig = VerticalProfilePlotter().render(build_series(ds, "o3"), mode="binned")
    assert any(isinstance(c, PolyCollection) for c in fig.axes[0].collections)
    _assert_dense_layers_rasterized(fig)
    plt.close(fig)
