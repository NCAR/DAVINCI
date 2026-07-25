"""Paired ``spatial_bias`` on a GRID pair must render a filled field.

Coverage-gap test named in FABLE_REVIEW.md §5.4: "paired ``spatial_bias`` on
GRID is never mark-verified (``QuadMesh`` vs ``PathCollection``) anywhere".
Per CLAUDE.md's geometry-aware-rendering rule, gridded data must render as a
``QuadMesh`` (pcolormesh), never a point-scatter ``PathCollection`` — and this
must be verified programmatically, not by eye.

Unlike the existing ``x_``/``y_``-prefixed fixtures elsewhere in the plot
test suite, this dataset uses the real pairing-engine variable-naming and
attribute contract (see ``pairing/engine.py:381-395``): variables are named
``<source_label>_<canonical>`` and carry ``axis``/``source_label`` attrs
rather than relying on the ``x_``/``y_`` name-prefix fallback.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PathCollection, QuadMesh  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.spatial.bias import SpatialBiasPlotter  # noqa: E402


def _paired_grid_dataset() -> xr.Dataset:
    """A tiny paired GRID dataset with variables/attrs like real pipeline output."""
    lat = np.linspace(20.0, 50.0, 4)
    lon = np.linspace(-120.0, -80.0, 5)
    rng = np.random.default_rng(11)
    x_values = rng.uniform(10.0, 60.0, size=(4, 5))
    y_values = x_values + rng.normal(0.0, 3.0, size=(4, 5))

    x_da = xr.DataArray(
        x_values,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "axis": "x",
            "source_label": "airnow_grid",
            "canonical_name": "o3",
            "units": "ppb",
        },
    )
    y_da = xr.DataArray(
        y_values,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "axis": "y",
            "source_label": "cesm",
            "canonical_name": "o3",
            "units": "ppb",
        },
    )
    return xr.Dataset(
        {"airnow_grid_o3": x_da, "cesm_o3": y_da},
        attrs={"geometry": "grid"},
    )


@pytest.mark.skipif(
    not pytest.importorskip("cartopy", reason="cartopy not available"),
    reason="cartopy not available",
)
def test_spatial_bias_grid_pair_renders_quadmesh_not_scatter() -> None:
    """Source-label-named/attr'd GRID pair -> QuadMesh, never PathCollection."""
    ds = _paired_grid_dataset()

    fig = SpatialBiasPlotter().render(build_series(ds, "airnow_grid_o3", "cesm_o3"))
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]

    assert any(
        isinstance(c, QuadMesh) for c in ax.collections
    ), "gridded spatial_bias must render the data as a filled field (QuadMesh)"
    assert not any(
        isinstance(c, PathCollection) for c in ax.collections
    ), "gridded spatial_bias must not fall back to a point-scatter PathCollection"
    plt.close(fig)


def _global_grid_pair(finite_bbox: tuple[float, float, float, float] | None) -> xr.Dataset:
    """A GLOBAL grid whose bias is finite only inside ``finite_bbox``.

    Reproduces what `method: grid` produces: intermediate gridding always
    builds a global grid (intermediate_grid.py), so a regional pair lands as a
    global field that is NaN everywhere outside the region. ``finite_bbox`` is
    (lon_min, lon_max, lat_min, lat_max); None makes the whole grid finite.
    """
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(-179.0, 180.0, 2.0)
    rng = np.random.default_rng(3)
    x_values = rng.uniform(10.0, 60.0, size=(lat.size, lon.size))
    y_values = x_values + rng.normal(0.0, 3.0, size=(lat.size, lon.size))
    if finite_bbox is not None:
        lo_lon, hi_lon, lo_lat, hi_lat = finite_bbox
        inside = (
            (lon[None, :] >= lo_lon)
            & (lon[None, :] <= hi_lon)
            & (lat[:, None] >= lo_lat)
            & (lat[:, None] <= hi_lat)
        )
        x_values = np.where(inside, x_values, np.nan)
        y_values = np.where(inside, y_values, np.nan)

    def _da(values: np.ndarray, axis: str, src: str) -> xr.DataArray:
        return xr.DataArray(
            values,
            dims=("latitude", "longitude"),
            coords={"latitude": lat, "longitude": lon},
            attrs={"axis": axis, "source_label": src, "canonical_name": "o3", "units": "ppb"},
        )

    return xr.Dataset(
        {
            "airnow_grid_o3": _da(x_values, "x", "airnow_grid"),
            "cesm_o3": _da(y_values, "y", "cesm"),
        },
        attrs={"geometry": "grid"},
    )


@pytest.mark.skipif(
    not pytest.importorskip("cartopy", reason="cartopy not available"),
    reason="cartopy not available",
)
def test_regional_data_on_a_global_grid_fits_to_the_data() -> None:
    """method: grid always builds a GLOBAL grid, so regional data lands as a
    mostly-NaN global field. The bias map must fit the data, not the grid."""
    import cartopy.crs as ccrs

    ds = _global_grid_pair(finite_bbox=(-125.0, -100.0, 30.0, 50.0))
    fig = SpatialBiasPlotter().render(build_series(ds, "airnow_grid_o3", "cesm_o3"))
    ax = fig.axes[0]
    x0, x1, y0, y1 = cast(Any, ax).get_extent(crs=ccrs.PlateCarree())
    # Fits the finite CONUS box (small margin), not the -180..180 grid.
    assert -135 <= x0 <= -120 and -105 <= x1 <= -90, (x0, x1)
    assert 22 <= y0 <= 32 and 48 <= y1 <= 58, (y0, y1)
    plt.close(fig)


@pytest.mark.skipif(
    not pytest.importorskip("cartopy", reason="cartopy not available"),
    reason="cartopy not available",
)
def test_globally_finite_data_stays_global() -> None:
    """Safety property: fitting to finite data leaves global-coverage maps
    global. Only regional-on-global-grid maps change."""
    import cartopy.crs as ccrs

    ds = _global_grid_pair(finite_bbox=None)
    fig = SpatialBiasPlotter().render(build_series(ds, "airnow_grid_o3", "cesm_o3"))
    ax = fig.axes[0]
    x0, x1, _, _ = cast(Any, ax).get_extent(crs=ccrs.PlateCarree())
    assert (x1 - x0) > 300, "global-coverage bias map must stay ~global"
    plt.close(fig)
