"""Wavelet scalogram: time x period power with COI, significance, global panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from davinci_monet.plots import labeling
from davinci_monet.plots.base import BasePlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import get_sequential_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


@register_plotter("wavelet_scalogram", arity="single_source", category="specialized")
class WaveletScalogramPlotter(BasePlotter):
    """Torrence & Compo style scalogram with a global-spectrum side panel."""

    name: str = "wavelet_scalogram"
    default_figsize: tuple[float, float] = (10, 6)

    def render(
        self,
        series: list["PlotSeries"],
        ax: "matplotlib.axes.Axes | None" = None,
        **kwargs: Any,
    ) -> "matplotlib.figure.Figure":
        if len(series) != 1:
            raise NotImplementedError(
                f"WaveletScalogramPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        ds = s.dataset
        power = ds[s.var_name]
        time = power["time"].values
        period = power["period"].values

        fig = plt.figure(
            figsize=self.config.figure.figsize,
            dpi=self.config.figure.dpi,
            facecolor=self.config.figure.facecolor,
        )
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.06)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_glob = fig.add_subplot(gs[0, 1], sharey=ax_main)

        # Rasterize the dense data layers so the vector PDF stays small and
        # renders cleanly; axes/text/contour lines stay vector.
        mesh = ax_main.pcolormesh(
            time,
            period,
            power.transpose("period", "time").values,
            cmap=get_sequential_cmap(),
            shading="auto",
            rasterized=True,
        )
        ax_main.set_yscale("log")
        ax_main.set_ylim(float(period.max()), float(period.min()))

        if "power_significance" in ds:
            sig = ds["power_significance"].transpose("period", "time").values
            # A single significance line: cheap vector, left unrasterized
            # (matplotlib ignores rasterizing contour lines).
            ax_main.contour(time, period, sig, levels=[1.0], colors="black", linewidths=1.0)

        if "coi" in ds:
            coi = ds["coi"].values
            ax_main.plot(time, coi, color="white", linestyle="--", linewidth=1.2)
            ax_main.fill_between(
                time,
                coi,
                float(period.max()),
                color="white",
                alpha=0.3,
                hatch="xx",
                rasterized=True,
            )

        unit = str(ds["period"].attrs.get("units", ""))
        unit_label = labeling.format_units(unit)
        ax_main.set_ylabel(
            f"Period ({unit_label})" if unit_label else "Period",
            fontsize=self.config.text.fontsize,
        )
        ax_main.set_xlabel("Time", fontsize=self.config.text.fontsize)
        ax_main.tick_params(axis="x", rotation=45)
        self.set_title(
            ax_main,
            labeling.title_text(
                str(ds.attrs.get("wavelet_quantity", "")), operation="Wavelet Power"
            ),
        )
        fig.colorbar(mesh, ax=ax_main, label="Power", shrink=0.85, pad=0.02)

        if "global_power" in ds:
            ax_glob.plot(ds["global_power"].values, period, color=self.config.style.y_color)
            if "global_significance" in ds:
                ax_glob.plot(
                    ds["global_significance"].values,
                    period,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                )
        ax_glob.set_xlabel("Power", fontsize=self.config.text.fontsize)
        plt.setp(ax_glob.get_yticklabels(), visible=False)
        return fig
