"""Distribution histogram renderer for one source series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    BasePlotter,
    get_variable_label,
    get_variable_units,
    series_colors,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


@register_plotter("histogram", arity="single_source", category="statistical")
class HistogramPlotter(BasePlotter):
    """Distribution histogram for one source series."""

    name: str = "histogram"
    default_figsize: tuple[float, float] = (8, 5)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        *,
        n_bins: int = 30,
        show_stats: bool = True,
        title: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a distribution histogram for one source series.

        The histogram includes a red median line and optional
        N/Mean/Median/Std/P10/P90 stats box.
        """
        if len(series) != 1:
            raise NotImplementedError(
                f"HistogramPlotter.render requires exactly 1 series; got {len(series)}."
            )

        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        colors = series_colors(series)

        for s, color in zip(series, colors):
            values = s.dataset[s.var_name].values.ravel()
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            label = s.source_label or get_variable_label(
                s.dataset, s.var_name, include_prefix=False
            )
            ax.hist(
                values,
                bins=n_bins,
                color=color,
                edgecolor="white",
                alpha=0.8,
                label=label,
                **kwargs,
            )
            median = float(np.median(values))
            ax.axvline(median, color="#D62839", linestyle="--", linewidth=1.5)
            if show_stats:
                stats_text = (
                    f"N={len(values)}\n"
                    f"Mean={float(np.mean(values)):.2f}\n"
                    f"Median={median:.2f}\n"
                    f"Std={float(np.std(values)):.2f}\n"
                    f"P10={float(np.percentile(values, 10)):.2f}\n"
                    f"P90={float(np.percentile(values, 90)):.2f}"
                )
                ax.text(
                    0.97,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=self.config.text.annotation,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="#CCCCCC",
                    ),
                )

        first = series[0]
        var_label = get_variable_label(first.dataset, first.var_name, include_prefix=False)
        units = get_variable_units(first.dataset, first.var_name)
        ax.set_xlabel(
            labeling.axis_label(var_label, units),
            fontsize=self.config.text.fontsize,
        )
        ax.set_ylabel("Count", fontsize=self.config.text.fontsize)
        self.set_title(ax, title if title else labeling.title_text(var_label))
        ax.grid(True, alpha=0.3, axis="y")
        return fig
