"""Corrected-AOD figures quantify the intended before/after product."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from davinci_monet.core.exceptions import PlottingError  # noqa: E402
from davinci_monet.plots.base import build_series  # noqa: E402
from davinci_monet.plots.plot_config import PlotConfig  # noqa: E402
from davinci_monet.plots.renderers.aod_correction_science import (  # noqa: E402
    AODCorrectionSciencePlotter,
)
from davinci_monet.plots.renderers.aod_correction_science_data import (  # noqa: E402
    correction_inputs,
    weighted_metrics,
)


def correction_series(*, aod_scale: float = 1.0) -> list:
    time = pd.date_range("2008-01-15T12:00:00", periods=12, freq="MS")
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    month = np.arange(12, dtype=float)[:, None, None]
    latitude = np.cos(np.deg2rad(lat))[None, :, None]
    longitude = (1.0 + np.cos(np.deg2rad(lon)))[None, None, :]
    observation_values = 0.08 + 0.015 * np.sin(2.0 * np.pi * month / 12.0)
    observation_values = observation_values + 0.02 * latitude + 0.01 * longitude
    observation_values = np.broadcast_to(observation_values, (12, 3, 4)).copy()
    model_values = observation_values + 0.08
    corrected_values = observation_values + 0.015
    observation_values[2, 0, 0] = np.nan
    observation_values *= aod_scale
    model_values *= aod_scale
    corrected_values *= aod_scale

    coords = {"time": time, "lat": lat, "lon": lon}
    model = xr.Dataset(
        {
            "aod": (
                ("time", "lat", "lon"),
                model_values,
                {"display_name": "AOD (550 nm)", "units": "1"},
            )
        },
        coords=coords,
        attrs={"source_label": "merra2"},
    )
    observation = xr.Dataset(
        {
            "aod": (
                ("time", "lat", "lon"),
                observation_values,
                {"display_name": "AOD (550 nm)", "units": "1"},
            )
        },
        coords=coords,
        attrs={"source_label": "modis_aqua"},
    )
    support = np.ones_like(model_values)
    support[:, 0, 0] = 0.0
    corrected = xr.Dataset(
        {
            "aod_target": (("time", "lat", "lon"), corrected_values),
            "r": (("time", "lat", "lon"), corrected_values / model_values),
            "spatial_support": (("time", "lat", "lon"), support),
            "lower_clip_mask": (
                ("time", "lat", "lon"),
                np.zeros_like(model_values, dtype=bool),
            ),
            "upper_clip_mask": (
                ("time", "lat", "lon"),
                np.zeros_like(model_values, dtype=bool),
            ),
        },
        coords=coords,
        attrs={"source_label": "corrected_merra2"},
    )
    return [
        build_series(model, "aod")[0],
        build_series(corrected, "aod_target")[0],
        build_series(observation, "aod")[0],
    ]


def test_corrected_metrics_are_closer_to_modis() -> None:
    inputs = correction_inputs(
        correction_series(),
        model_source="merra2",
        corrected_source="corrected_merra2",
        observation_source="modis_aqua",
    )
    original = weighted_metrics(inputs.model, inputs.observation, inputs.valid)
    corrected = weighted_metrics(inputs.corrected, inputs.observation, inputs.valid)

    assert corrected["RMSE"] < original["RMSE"]
    assert corrected["MAE"] < original["MAE"]
    assert corrected["MB"] == pytest.approx(0.015)


def test_correction_plotter_emits_before_after_science_suite() -> None:
    figures = AODCorrectionSciencePlotter().render(
        correction_series(),
        max_scatter_points=500,
    )

    assert [label for label, _figure in figures] == [
        "annual_aod",
        "annual_bias_improvement",
        "seasonal_aod",
        "seasonal_error_reduction",
        "agreement_metrics",
        "global_timeseries",
        "matched_scatter",
        "correction_diagnostics",
    ]
    annual = dict(figures)["annual_aod"]
    annual_title = getattr(annual, "_suptitle", None)
    assert annual_title is not None
    assert annual_title.get_text() == "AOD (550 nm) Annual Mean"
    metrics = dict(figures)["agreement_metrics"]
    metrics_title = getattr(metrics, "_suptitle", None)
    assert metrics_title is not None
    assert metrics_title.get_text() == "AOD (550 nm) Agreement"
    report = AODCorrectionSciencePlotter().validate_rendered_figures(figures)
    assert report["protocol"] == "davinci-aod-correction-v2"
    assert report["passed"] is True
    for label in ("annual_aod", "seasonal_aod"):
        artists = getattr(dict(figures)[label], "_davinci_absolute_aod_artists")
        assert artists
        assert all(isinstance(artist.norm, BoundaryNorm) for artist in artists)
        assert all(artist.cmap.name == "turbo" for artist in artists)
        assert all(artist.get_rasterized() is True for artist in artists)
    scatter_artists = getattr(dict(figures)["matched_scatter"], "_davinci_dense_artists")
    assert all(artist.get_rasterized() is True for artist in scatter_artists)
    rotations = [
        tick.get_rotation() for tick in dict(figures)["global_timeseries"].axes[1].get_xticklabels()
    ]
    assert rotations and all(rotation == pytest.approx(45.0) for rotation in rotations)
    for _label, figure in figures:
        plt.close(figure)


def test_correction_plotter_rejects_plot_protocol_violation() -> None:
    plotter = AODCorrectionSciencePlotter()
    figures = plotter.render(correction_series(), max_scatter_points=500)
    scatter = dict(figures)["matched_scatter"]
    getattr(scatter, "_davinci_dense_artists")[0].set_rasterized(False)

    try:
        with pytest.raises(PlottingError, match="unrasterized dense artist"):
            plotter.validate_rendered_figures(figures)
    finally:
        for _label, figure in figures:
            plt.close(figure)


def test_correction_plotter_enforces_rendered_publication_layout() -> None:
    series = correction_series(aod_scale=5.0)
    verbose_quantity = "Screened and Daily Sampled Aerosol Optical Depth"
    series[2].dataset[series[2].var_name].attrs["display_name"] = verbose_quantity
    plotter = AODCorrectionSciencePlotter(PlotConfig(subtitle="2008-01-01 – 2008-12-31"))
    figures = plotter.render(series, max_scatter_points=500)

    try:
        report = plotter.validate_rendered_figures(figures)
        assert report["checks"]["layout"] == (
            "rendered text and ticks are unclipped and nonoverlapping"
        )
        annual_colorbar = dict(figures)["annual_aod"].axes[-1]
        np.testing.assert_allclose(
            annual_colorbar.get_xticks(),
            [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.95],
        )
        timeseries = dict(figures)["global_timeseries"]
        assert timeseries.axes[0].get_ylabel() == "Matched-Grid Mean AOD"
        scatter = dict(figures)["matched_scatter"]
        assert scatter.axes[0].get_xlabel() == "MODIS Aqua Daily Sampled AOD"
    finally:
        for _label, figure in figures:
            plt.close(figure)


def test_correction_plotter_rejects_clipped_layout_text() -> None:
    plotter = AODCorrectionSciencePlotter()
    figures = plotter.render(correction_series(), max_scatter_points=500)
    scatter = dict(figures)["matched_scatter"]
    scatter.axes[0].set_xlabel("clipped " * 200)

    try:
        with pytest.raises(PlottingError, match="clipped by the figure"):
            plotter.validate_rendered_figures(figures)
    finally:
        for _label, figure in figures:
            plt.close(figure)


def test_correction_plotter_requires_explicit_source_labels() -> None:
    series = correction_series()
    series[0].source_label = "wrong_model"

    with pytest.raises(ValueError, match="missing labeled sources: merra2"):
        AODCorrectionSciencePlotter().render(series, max_scatter_points=10)
