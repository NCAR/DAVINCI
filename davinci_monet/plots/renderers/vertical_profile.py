"""Vertical profile renderer for one source series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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


@register_plotter("vertical_profile", arity="single_source", category="specialized")
class VerticalProfilePlotter(BasePlotter):
    """Altitude vs. value profile for one source series."""

    name: str = "vertical_profile"
    default_figsize: tuple[float, float] = (6, 8)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        *,
        mode: Literal["scatter", "binned"] = "scatter",
        n_bins: int = 20,
        alt_coord: str = "altitude",
        title: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render one source's vertical profile."""
        if len(series) != 1:
            raise NotImplementedError(
                f"VerticalProfilePlotter.render requires exactly 1 series; got {len(series)}."
            )

        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        colors = series_colors(series)
        for s, color in zip(series, colors):
            label = s.source_label or get_variable_label(
                s.dataset, s.var_name, include_prefix=False
            )
            if mode == "binned":
                self._plot_binned(ax, s, alt_coord, n_bins, color, label)
            else:
                self._plot_scatter(ax, s, alt_coord, color, label)

        first = series[0]
        var_label = get_variable_label(first.dataset, first.var_name, include_prefix=False)
        units = get_variable_units(first.dataset, first.var_name)
        ax.set_xlabel(
            labeling.axis_label(var_label, units),
            fontsize=self.config.text.fontsize,
        )
        alt_units = (
            first.dataset[alt_coord].attrs.get("units", "")
            if alt_coord in first.dataset.coords
            else None
        )
        ax.set_ylabel(
            labeling.axis_label("Altitude", alt_units),
            fontsize=self.config.text.fontsize,
        )
        self.set_title(ax, title if title else labeling.title_text(var_label))
        ax.grid(True, alpha=0.3)
        return fig

    def _plot_scatter(self, ax: Any, s: PlotSeries, alt_coord: str, color: str, label: str) -> None:
        values = s.dataset[s.var_name].values.ravel()
        altitudes = s.dataset[alt_coord].values.ravel()
        valid = np.isfinite(values) & np.isfinite(altitudes)
        ax.scatter(
            values[valid], altitudes[valid], c=color, s=8, alpha=0.5, edgecolors="none", label=label
        )

    def _plot_binned(
        self, ax: Any, s: PlotSeries, alt_coord: str, n_bins: int, color: str, label: str
    ) -> None:
        values = s.dataset[s.var_name].values.ravel()
        altitudes = s.dataset[alt_coord].values.ravel()
        valid = np.isfinite(values) & np.isfinite(altitudes)
        values, altitudes = values[valid], altitudes[valid]
        if values.size == 0:
            return
        bin_edges = np.linspace(altitudes.min(), altitudes.max(), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_indices = np.clip(np.digitize(altitudes, bin_edges) - 1, 0, n_bins - 1)
        means = np.full(n_bins, np.nan)
        stds = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                means[i] = np.nanmean(values[mask])
                stds[i] = np.nanstd(values[mask])
        vb = np.isfinite(means)
        ax.plot(means[vb], bin_centers[vb], color=color, linewidth=1.5, label=label)
        ax.fill_betweenx(
            bin_centers[vb], (means - stds)[vb], (means + stds)[vb], color=color, alpha=0.2
        )
