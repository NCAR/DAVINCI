"""wavelet_scalogram draws a QuadMesh + a global-spectrum side panel."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402

from davinci_monet.analysis.wavelet_filter import filter_projected_coefficients  # noqa: E402
from davinci_monet.config.schema import PeriodBandSpec, WaveletFilterSpec  # noqa: E402
from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.wavelet_scalogram import WaveletScalogramPlotter  # noqa: E402


def _spectrum() -> xr.Dataset:
    nt, npd = 60, 6
    time = pd.date_range("2024-01-01", periods=nt, freq="D")
    period = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    rng = np.random.default_rng(0)
    power = rng.random((nt, npd))
    ds = xr.Dataset(
        {
            "power": (("time", "period"), power, {"kind": "power"}),
            "power_significance": (("time", "period"), power / power.mean(0), {"kind": "power"}),
            "coi": (("time",), np.full(nt, 16.0), {"kind": "coi", "units": "days"}),
            "global_power": (("period",), power.mean(0), {"kind": "global"}),
            "global_significance": (("period",), np.ones(npd), {"kind": "global"}),
        },
        coords={"time": time, "period": ("period", period, {"units": "days"})},
    )
    ds.attrs["wavelet_quantity"] = "O3"
    return ds


def test_scalogram_quadmesh_and_global_panel() -> None:
    fig = WaveletScalogramPlotter().render(build_series(_spectrum(), "power"))
    meshes = [c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh)]
    assert meshes, "expected a QuadMesh power layer"
    # Dense data layer is rasterized so the vector PDF stays small.
    assert meshes[0].get_rasterized() is True
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_scalogram_uses_mode_selected_filter_arrays() -> None:
    nt = 256
    time = pd.date_range("2001-01-01", periods=nt, freq="D")
    signal = np.sin(2.0 * np.pi * np.arange(nt) / 16.0)
    projection = xr.Dataset(
        {
            "pc": (("time", "mode"), np.stack((signal, 0.5 * signal), axis=1)),
            "resolution": (("time", "mode"), np.ones((nt, 2))),
        },
        coords={"time": time, "mode": [1, 2]},
        attrs={
            "projection_basis_signature": "plot-test-basis",
            "projection_log_epsilon": 0.01,
        },
    )
    spec = WaveletFilterSpec(
        type="wavelet_filter",
        source="projection",
        band=PeriodBandSpec(min=8.0, max=32.0),
        min_segment_days=64.0,
        keep_significant=False,
    )
    selected = filter_projected_coefficients(projection, spec).sel(mode=2)

    fig = WaveletScalogramPlotter().render(build_series(selected, "power"))

    meshes = [
        collection for collection in fig.axes[0].collections if isinstance(collection, QuadMesh)
    ]
    assert len(meshes) == 1
    expected_power = selected["power"].transpose("period", "time").values
    np.testing.assert_allclose(
        np.asarray(meshes[0].get_array()).reshape(expected_power.shape),
        expected_power,
    )
    global_lines = fig.axes[1].get_lines()
    np.testing.assert_allclose(
        np.asarray(global_lines[0].get_xdata(), dtype=float),
        selected["global_power"].values,
    )
    plt.close(fig)
