"""eof_pattern renders one signed QuadMesh map per mode and slices the surface."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.eof_pattern import EOFPatternPlotter  # noqa: E402


def _eof_ds(nlev=0) -> xr.Dataset:
    lat = np.linspace(10, 40, 4)
    lon = np.linspace(-120, -90, 5)
    if nlev:
        arr = np.zeros((2, nlev, 4, 5))
        arr[:, -1] = 1.0  # surface level (last index) distinct
        da = xr.DataArray(
            arr,
            dims=("mode", "lev", "lat", "lon"),
            coords={
                "mode": [1, 2],
                "lev": np.array([100.0, 500.0, 1000.0])[:nlev],
                "lat": lat,
                "lon": lon,
                "latitude": ("lat", lat),
                "longitude": ("lon", lon),
            },
        )
    else:
        da = xr.DataArray(
            np.random.default_rng(0).normal(size=(2, 4, 5)),
            dims=("mode", "lat", "lon"),
            coords={
                "mode": [1, 2],
                "lat": lat,
                "lon": lon,
                "latitude": ("lat", lat),
                "longitude": ("lon", lon),
            },
        )
    ds = xr.Dataset({"eofs": da, "explained_variance": ("mode", np.array([0.7, 0.3]))})
    ds.attrs["eof_quantity"] = "O3"
    return ds


def test_eof_pattern_one_quadmesh_per_mode() -> None:
    figs = EOFPatternPlotter().render(build_series(_eof_ds(), "eofs"))
    assert isinstance(figs, list) and len(figs) == 2
    assert [lbl for lbl, _ in figs] == ["mode1", "mode2"]
    ax = figs[0][1].axes[0]
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert meshes, "expected a QuadMesh field"
    # Dense field layer is rasterized so the vector PDF stays small.
    assert meshes[0].get_rasterized() is True
    for _, f in figs:
        plt.close(f)


def test_eof_pattern_3d_defaults_to_surface() -> None:
    figs = EOFPatternPlotter().render(build_series(_eof_ds(nlev=3), "eofs"))
    qm = next(c for c in figs[0][1].axes[0].collections if isinstance(c, QuadMesh))
    arr = np.asarray(qm.get_array(), dtype=float)
    assert np.nanmax(arr) == pytest.approx(1.0)
    assert np.nanmin(arr) == pytest.approx(1.0)
    for _, f in figs:
        plt.close(f)


def test_eof_pattern_global_longitudes_keep_mesh_and_data_aligned() -> None:
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.arange(0.0, 360.0, 45.0)
    field = np.broadcast_to(np.arange(lon.size, dtype=float), (lat.size, lon.size)).copy()
    ds = xr.Dataset(
        {"eofs": (("mode", "lat", "lon"), field[None, ...])},
        coords={
            "mode": [1],
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    )

    figures = EOFPatternPlotter().render(build_series(ds, "eofs"))
    fig = figures[0][1]
    ax = fig.axes[0]
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))

    coordinates = np.asarray(mesh.get_coordinates(), dtype=float)
    x_edges = coordinates[0, :, 0]
    assert np.all(np.diff(x_edges) > 0.0)

    normalized_lon = np.where(lon > 180.0, lon - 360.0, lon)
    lon_order = np.argsort(normalized_lon)
    rendered = np.asarray(mesh.get_array(), dtype=float).reshape(field.shape)
    np.testing.assert_allclose(rendered, field[:, lon_order])

    lon_min, lon_max, _, _ = getattr(ax, "get_extent")(ccrs.PlateCarree())
    np.testing.assert_allclose([lon_min, lon_max], [-180.0, 180.0], atol=1e-6)
    plt.close(fig)


def test_eof_pattern_square_lon_lat_field_uses_dimension_orientation() -> None:
    lat = np.array([30.0, 40.0, 50.0])
    lon = np.array([-120.0, -110.0, -100.0])
    field_lon_lat = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    ds = xr.Dataset(
        {"eofs": (("mode", "lon", "lat"), field_lon_lat[None, ...])},
        coords={"mode": [1], "lat": lat, "lon": lon},
    )

    figures = EOFPatternPlotter().render(build_series(ds, "eofs"))
    fig = figures[0][1]
    mesh = next(c for c in fig.axes[0].collections if isinstance(c, QuadMesh))

    rendered = np.asarray(mesh.get_array(), dtype=float).reshape(field_lon_lat.shape)
    np.testing.assert_allclose(rendered, field_lon_lat.T)
    plt.close(fig)
