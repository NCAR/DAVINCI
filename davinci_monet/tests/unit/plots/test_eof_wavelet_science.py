"""Scientific EOF/wavelet suite joins immutable plotting inputs correctly."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402

from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.renderers.eof_wavelet_science import (  # noqa: E402
    EOFWaveletSciencePlotter,
)
from davinci_monet.plots.renderers.eof_wavelet_science_data import (  # noqa: E402
    seasonal_fields,
)


def science_inputs() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    time = pd.date_range("2008-01-01T12:00:00", "2008-12-31T12:00:00", freq="D")
    mode = np.arange(1, 4)
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    month = np.arange(1, 13)
    rng = np.random.default_rng(4)

    annual_aod = 0.15
    monthly_aod = 0.08 + 0.01 * month
    epsilon = 0.01
    time_mean = np.full((lat.size, lon.size), np.log(annual_aod + epsilon))
    climatology = np.stack(
        [np.full_like(time_mean, np.log(value + epsilon)) - time_mean for value in monthly_aod]
    )
    basis = xr.Dataset(
        {
            "eofs": (("mode", "lat", "lon"), rng.normal(scale=0.08, size=(3, 3, 4))),
            "pc": (("time", "mode"), rng.normal(size=(time.size, 3))),
            "explained_variance": ("mode", [0.20, 0.12, 0.08]),
            "explained_variance_error": ("mode", [0.03, 0.02, 0.01]),
            "time_mean": (("lat", "lon"), time_mean),
            "climatology": (("month", "lat", "lon"), climatology),
        },
        coords={"time": time, "mode": mode, "lat": lat, "lon": lon, "month": month},
        attrs={
            "eof_quantity": "log_aod",
            "eof_remove_seasonal_cycle": "true",
            "eof_input_log_epsilon": epsilon,
            "source_label": "basis",
        },
    )

    projected_pc = rng.normal(size=(time.size, 3))
    projection = xr.Dataset(
        {
            "pc": (("time", "mode"), projected_pc),
            "resolution": (("time", "mode"), np.full((time.size, 3), 0.9)),
            "posterior_variance": (("time", "mode"), np.full((time.size, 3), 0.04)),
            "spatial_support": (("month", "lat", "lon"), np.ones((12, 3, 4))),
            "clim_bias_applied": (
                ("month", "lat", "lon"),
                rng.normal(scale=0.03, size=(12, 3, 4)),
            ),
        },
        coords=basis.coords,
        attrs={
            "projection_basis_signature": "basis-identity",
            "projection_min_resolution": 0.3,
            "source_label": "projection",
        },
    )

    period = np.array([4.0, 8.0, 16.0, 32.0, 64.0])
    power = np.square(rng.normal(size=(time.size, 3, period.size)))
    filtered = xr.Dataset(
        {
            "pc": (("time", "mode"), 0.6 * projected_pc),
            "power": (("time", "mode", "period"), power),
            "power_significance": (
                ("time", "mode", "period"),
                power / np.maximum(power.mean(axis=0, keepdims=True), 1.0e-12),
            ),
            "coi": (("time", "mode"), np.full((time.size, 3), 40.0)),
            "global_power": (("mode", "period"), power.mean(axis=0)),
            "global_significance": (("mode", "period"), np.ones((3, period.size))),
            "retained_variance": ("mode", [0.55, 0.45, 0.35]),
            "recon_error": ("mode", [0.04, 0.06, 0.08]),
        },
        coords={"time": time, "mode": mode, "period": period},
        attrs={
            "projection_basis_signature": "basis-identity",
            "wavelet_quantity": "pc",
            "source_label": "filtered",
        },
    )
    filtered["period"].attrs["units"] = "days"
    return basis, projection, filtered


def _series() -> list:
    basis, projection, filtered = science_inputs()
    return [
        build_series(basis, "eofs")[0],
        build_series(projection, "pc")[0],
        build_series(filtered, "pc")[0],
    ]


def test_seasonal_aod_uses_calendar_day_weights() -> None:
    basis, projection, _filtered = science_inputs()
    seasonal, departures, biases = seasonal_fields(basis, projection)

    expected_djf = (0.20 * 31 + 0.09 * 31 + 0.10 * 29) / (31 + 31 + 29)
    assert float(seasonal[0].isel(lat=0, lon=0)) == pytest.approx(expected_djf)
    assert len(departures) == 4
    assert len(biases) == 4


def test_science_plotter_emits_complete_figure_set_and_rasterized_maps() -> None:
    figures = EOFWaveletSciencePlotter().render(
        _series(),
        modes=[1, 2, 3],
        pc_modes=[1, 2],
        wavelet_modes=[1, 2],
    )
    labels = [label for label, _figure in figures]
    assert labels == [
        "seasonal_aod",
        "seasonal_aod_departure",
        "seasonal_projection_bias",
        "eof_patterns",
        "variance_summary",
        "pc_comparison",
        "mode_quality",
        "spatial_wavelet_rms",
        "wavelet_mode1",
        "wavelet_mode2",
    ]
    seasonal = dict(figures)["seasonal_aod"]
    meshes = [
        collection
        for axis in seasonal.axes[:4]
        for collection in axis.collections
        if isinstance(collection, QuadMesh) and collection.get_rasterized() is True
    ]
    assert len(meshes) == 4
    wavelet_title = getattr(dict(figures)["wavelet_mode2"], "_suptitle", None)
    assert wavelet_title is not None
    assert wavelet_title.get_text() == "Projected PC 2 Wavelet Power"
    for _label, figure in figures:
        plt.close(figure)


def test_science_plotter_rejects_mismatched_mode_identity() -> None:
    basis, projection, filtered = science_inputs()
    filtered = filtered.assign_coords(mode=[1, 2, 4])
    series = [
        build_series(basis, "eofs")[0],
        build_series(projection, "pc")[0],
        build_series(filtered, "pc")[0],
    ]

    with pytest.raises(ValueError, match="mode coordinates"):
        EOFWaveletSciencePlotter().render(series)
