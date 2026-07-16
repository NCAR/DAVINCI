"""Timeseries renderer behavior for one, two, and many source series.

The canonical TimeSeriesPlotter gains a render() that handles 1/2/N source
series: 1 → single aggregated line (the spaghetti fix), 2 → x-vs-y
(delegates to the paired plot), N → overlay. The unified PlottingStage routes
geometry-only specs through render() for migrated renderers.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.base import PlotConfig
from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter
from davinci_monet.plots.style import NCAR_PRIMARY


def _multisite_series(n_t: int = 12, n_s: int = 6, source_label: str = "airnow") -> PlotSeries:
    rng = np.random.default_rng(0)
    times = np.datetime64("2024-02-01") + np.arange(n_t) * np.timedelta64(1, "h")
    ds = xr.Dataset(
        {"o3": (("time", "site"), rng.uniform(10, 60, (n_t, n_s)), {"units": "ppb"})},
        coords={"time": times, "site": np.arange(n_s)},
    )
    ds["o3"].attrs["axis"] = "x"
    ds["o3"].attrs["source_label"] = source_label
    return PlotSeries(ds, "o3", "o3", "x", source_label, 0)


class TestTimeseriesRenderSingleSource:
    def test_single_multisite_aggregates_to_one_line(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series(n_s=6)])
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 1  # mean over sites, not 6 spaghetti lines
        plt.close(fig)

    def test_single_source_uses_brand_blue(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series()])
        assert fig.axes[0].get_lines()[0].get_color() == NCAR_PRIMARY
        plt.close(fig)

    def test_date_tick_labels_use_compact_size(self) -> None:
        """Rotated date tick labels use the compact annotation_small size — not the
        larger, context-driven default tick size (which overflows in presentation)."""
        from davinci_monet.plots.plot_config import TextConfig

        cfg = PlotConfig(text=TextConfig(annotation_small=7.0, tick_fontsize=20.0))
        fig = TimeSeriesPlotter(cfg).render([_multisite_series()])
        ax = fig.axes[0]
        fig.canvas.draw()
        sizes = {round(cast(float, t.get_fontsize()), 1) for t in ax.get_xticklabels()}
        assert sizes == {7.0}, f"x date ticks must use annotation_small (7.0), got {sizes}"
        plt.close(fig)

    def test_single_dataset_labelled_by_source(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series(source_label="pandora")])
        # labeling.legend_label("pandora") -> "Pandora" (friendly display name)
        assert fig.axes[0].get_lines()[0].get_label() == "Pandora"
        plt.close(fig)

    def test_show_individual_sites_opt_in(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series(n_s=6)], show_individual_sites=True)
        # One line per site when explicitly requested.
        assert len(fig.axes[0].get_lines()) == 6
        plt.close(fig)

    def test_show_uncertainty_adds_band(self) -> None:
        fig = TimeSeriesPlotter().render([_multisite_series(n_s=6)], show_uncertainty=True)
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 1
        assert len(ax.collections) >= 1  # +/-1 sigma PolyCollection
        plt.close(fig)


def _named_site_series(source_label: str = "power") -> PlotSeries:
    """Single-source multi-site series whose site coord carries NAMES.

    Mirrors what the POWER reader emits: a `site` dim whose coordinate values
    are station names, not integers.
    """
    rng = np.random.default_rng(1)
    times = np.datetime64("2024-02-01") + np.arange(12) * np.timedelta64(1, "h")
    names = ["boulder", "mauna_loa", "south_pole"]
    ds = xr.Dataset(
        {"swdn": (("time", "site"), rng.uniform(50, 300, (12, len(names))), {"units": "W m-2"})},
        coords={"time": times, "site": names},
    )
    ds["swdn"].attrs["source_label"] = source_label
    return PlotSeries(ds, "swdn", "swdn", None, source_label, 0)


class TestIndividualSitesLegend:
    """Per-site timeseries must be readable: a legend naming each site.

    Regression: the single-source per-site path drew unlabelled lines and no
    legend, so a four-site record was four indistinguishable coloured lines.
    """

    def test_per_site_lines_carry_a_legend_with_site_names(self) -> None:
        fig = TimeSeriesPlotter().render(
            [_named_site_series()], show_individual_sites=True, site_label_var="site"
        )
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None, "per-site timeseries must draw a legend"
        labels = {t.get_text() for t in legend.get_texts()}
        assert {"boulder", "mauna_loa", "south_pole"} <= labels, labels
        plt.close(fig)

    def test_site_names_are_used_even_without_an_explicit_label_var(self) -> None:
        """String coord values on the split dim are labels; never "Site 0"."""
        fig = TimeSeriesPlotter().render([_named_site_series()], show_individual_sites=True)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        labels = {t.get_text() for t in legend.get_texts()}
        assert "boulder" in labels, labels
        plt.close(fig)

    def test_single_source_uses_config_subtitle(self) -> None:
        plotter = TimeSeriesPlotter(
            PlotConfig(title="O3 Time Series", subtitle="2024-02-01 - 2024-02-02")
        )
        fig = plotter.render([_multisite_series()])
        ax = fig.axes[0]

        assert ax.get_title() == r"O$_3$ Time Series"
        assert any(t.get_text() == "2024-02-01 - 2024-02-02" for t in ax.texts)
        plt.close(fig)


class TestTimeseriesRenderPaired:
    def test_two_series_delegates_to_paired_plot(self) -> None:
        rng = np.random.default_rng(1)
        t = np.datetime64("2024-02-01") + np.arange(10) * np.timedelta64(1, "h")
        ds = xr.Dataset(
            {
                "airnow_o3": (
                    "time",
                    rng.uniform(10, 60, 10),
                    {"axis": "x", "source_label": "airnow"},
                ),
                "cam_o3": (
                    "time",
                    rng.uniform(10, 60, 10),
                    {"axis": "y", "source_label": "cam"},
                ),
            },
            coords={"time": t},
        )
        x_series = PlotSeries(ds, "airnow_o3", "o3", "x", "airnow", 0)
        y_series = PlotSeries(ds, "cam_o3", "o3", "y", "cam", 1)
        fig = TimeSeriesPlotter().render([x_series, y_series])
        # Two series (geometry + dataset) on the axes.
        assert len(fig.axes[0].get_lines()) == 2
        plt.close(fig)


class TestGeometryTimeseriesName:
    def test_geometry_timeseries_name_is_not_registered(self) -> None:
        import pytest

        from davinci_monet.plots.registry import get_plotter_class

        with pytest.raises(Exception):
            get_plotter_class("geometry_timeseries")


class TestUnifiedStageRoutesTimeseriesThroughRender:
    def test_geometry_timeseries_spec_renders_single_line(self, tmp_path: Any) -> None:
        from davinci_monet.core.protocols import DataGeometry
        from davinci_monet.pipeline.stages import (
            PipelineContext,
            PlottingStage,
            SourceData,
            StageStatus,
        )

        rng = np.random.default_rng(0)
        times = np.datetime64("2024-02-01") + np.arange(12) * np.timedelta64(1, "h")
        ds = xr.Dataset(
            {"o3": (("time", "site"), rng.uniform(10, 60, (12, 6)), {"units": "ppb"})},
            coords={"time": times, "site": np.arange(6)},
        )
        geometry = SourceData(
            data=ds,
            label="airnow",
            source_type="pt_sfc",
            geometry=DataGeometry.POINT,
        )
        ctx = PipelineContext(
            config={
                "analysis": {"output_dir": str(tmp_path / "out")},
                "plots": {
                    "o3_ts": {
                        "type": "timeseries",
                        "source": "airnow",
                        "variable": "o3",
                        "title": "O3",
                    }
                },
            },
            sources={"airnow": geometry},
        )
        res = PlottingStage().execute(ctx)
        assert res.status == StageStatus.COMPLETED
        assert any("o3_ts" in p.name for p in (tmp_path / "out").glob("*.png"))
