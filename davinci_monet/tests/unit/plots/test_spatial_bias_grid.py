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
