"""Focused renderer regressions from FABLE review."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from davinci_monet.plots.base import build_series
from davinci_monet.plots.renderers.scorecard import ScorecardPlotter
from davinci_monet.plots.renderers.taylor import TaylorPlotter


def test_scorecard_center_uses_two_slope_norm() -> None:
    df = pd.DataFrame({"MB": [-2.0, 0.0, 5.0]}, index=["a", "b", "c"])

    fig = ScorecardPlotter().render_from_dataframe(df, cmap="RdBu_r", center=0.0)
    image = fig.axes[0].images[0]

    assert isinstance(image.norm, TwoSlopeNorm)
    assert image.norm.vcenter == 0.0


def test_taylor_skips_constant_comparison_series() -> None:
    ds = xr.Dataset(
        {
            "airnow_o3": ("time", np.array([1.0, 2.0, 3.0, 4.0])),
            "cam_o3": ("time", np.array([5.0, 5.0, 5.0, 5.0])),
        },
        coords={"time": np.arange(4)},
    )
    ds["airnow_o3"].attrs.update({"axis": "x", "source_label": "airnow"})
    ds["cam_o3"].attrs.update({"axis": "y", "source_label": "cam"})

    fig = TaylorPlotter().render(build_series(ds, "airnow_o3", "cam_o3"))

    for line in fig.axes[0].get_lines():
        assert np.all(np.isfinite(line.get_xdata()))
        assert np.all(np.isfinite(line.get_ydata()))
