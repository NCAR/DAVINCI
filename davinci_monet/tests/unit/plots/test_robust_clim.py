from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.spatial.field import SpatialPlotter  # noqa: E402


def _heavy_tailed_grid() -> xr.Dataset:
    data = np.array([0.1] * 100 + [50.0], dtype=float).reshape(1, 101)
    return xr.Dataset(
        {"aod": (("latitude", "longitude"), data, {"units": "1"})},
        coords={
            "latitude": np.array([0.0]),
            "longitude": np.arange(101.0),
        },
        attrs={"geometry": "grid"},
    )


def _skewed_signed_grid() -> xr.Dataset:
    data = np.array([-2.0] * 50 + [0.25] * 50 + [25.0], dtype=float).reshape(1, 101)
    return xr.Dataset(
        {"increment": (("latitude", "longitude"), data, {"units": "1"})},
        coords={
            "latitude": np.array([0.0]),
            "longitude": np.arange(101.0),
        },
        attrs={"geometry": "grid"},
    )


def _quadmesh_clim(fig):
    ax = fig.axes[0]
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
    return mesh.get_clim()


def _render_spatial(*args, **kwargs) -> matplotlib.figure.Figure:
    fig = SpatialPlotter().render(*args, **kwargs)
    assert isinstance(fig, matplotlib.figure.Figure)
    return fig


def test_spatial_plotter_robust_color_limits_ignore_outlier():
    fig = _render_spatial(build_series(_heavy_tailed_grid(), "aod"), robust=True)
    _, vmax = _quadmesh_clim(fig)
    assert vmax < 1.0
    plt.close(fig)


def test_spatial_plotter_robust_symmetric_limits_center_zero():
    fig = _render_spatial(
        build_series(_skewed_signed_grid(), "increment"),
        robust=True,
        symmetric=True,
    )
    ax = fig.axes[0]
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
    vmin, vmax = mesh.get_clim()
    assert abs(vmin) == pytest.approx(vmax)
    assert isinstance(mesh.norm, TwoSlopeNorm)
    assert mesh.norm.vcenter == 0
    plt.close(fig)


def test_spatial_plotter_default_color_limits_use_full_data_range():
    fig = _render_spatial(build_series(_heavy_tailed_grid(), "aod"), robust=False)
    _, vmax = _quadmesh_clim(fig)
    assert vmax == pytest.approx(50.0)
    plt.close(fig)
