"""EOF scree plot: explained variance (%) per mode with North error bars."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from davinci_monet.plots import labeling
from davinci_monet.plots.base import BasePlotter
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


@register_plotter("eof_scree", arity="single_source", category="statistical")
class EOFScreePlotter(BasePlotter):
    """Bar chart of explained variance (%) by EOF mode."""

    name: str = "eof_scree"
    default_figsize: tuple[float, float] = (8, 5)

    def render(
        self,
        series: list["PlotSeries"],
        ax: "matplotlib.axes.Axes | None" = None,
        *,
        title: str | None = None,
        **kwargs: Any,
    ) -> "matplotlib.figure.Figure":
        if len(series) != 1:
            raise NotImplementedError(
                f"EOFScreePlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        ds = s.dataset
        ev = ds[s.var_name]
        modes = [int(v) for v in ev["mode"].values]
        heights = np.asarray(ev.values, dtype=float) * 100.0
        err = None
        if "explained_variance_error" in ds:
            err = np.asarray(ds["explained_variance_error"].values, dtype=float) * 100.0

        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        ax.bar(modes, heights, color=self.config.style.y_color, yerr=err, capsize=3)
        ax.set_xticks(modes)
        ax.set_xlabel("Mode", fontsize=self.config.text.fontsize)
        ax.set_ylabel("Explained variance (%)", fontsize=self.config.text.fontsize)
        quantity = str(ds.attrs.get("eof_quantity", ""))
        self.set_title(
            ax, title or labeling.title_text(quantity, operation="EOF Explained Variance")
        )
        ax.grid(True, alpha=0.3, axis="y")
        return fig
