"""Temporal renderers and scoped registration for FABLE acceptance plots."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from davinci_monet.plots.base import BasePlotter, format_plot_title
from davinci_monet.plots.registry import plotter_registry, register_plotter
from davinci_monet.plots.renderers.wavelet_scalogram import WaveletScalogramPlotter
from davinci_monet.plots.style import NCAR_COLORS
from davinci_monet.tests.synthetic.fable_acceptance_spatial_plots import (
    FableEOFComparisonPlotter,
    FableSpatialRecoveryPlotter,
    _disposition,
    _finite_limit,
    _one_dataset,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


def _shade_mask(
    ax: matplotlib.axes.Axes,
    time: pd.DatetimeIndex,
    mask: np.ndarray,
    *,
    color: str,
    alpha: float,
    label: str,
) -> None:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        right = time[min(stop, len(time) - 1)]
        ax.axvspan(
            time[start],
            right,
            color=color,
            alpha=alpha,
            linewidth=0,
            label=label if index == 0 else None,
            zorder=0,
        )


class FablePCReconstructionPlotter(BasePlotter):
    """Compare projected and wavelet-reconstructed PCs with analytic truth."""

    name = "fable_pc_reconstruction"
    default_figsize = (15, 7.5)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, matplotlib.figure.Figure]]:
        del ax, kwargs
        dataset = _one_dataset(series, type(self).__name__)
        return [
            (f"seed_{int(seed)}", self._seed_figure(dataset, int(seed)))
            for seed in dataset["seed"].values
        ]

    def _seed_figure(self, dataset: Any, seed: int) -> matplotlib.figure.Figure:
        selected = dataset.sel(seed=seed)
        modes = selected["mode"].where(selected["mode_observable"], drop=True).values
        fig, axes = plt.subplots(
            len(modes),
            2,
            figsize=self.default_figsize,
            dpi=self.config.figure.dpi,
            sharex=True,
            layout="constrained",
        )
        axes = np.asarray(axes, dtype=object).reshape(len(modes), 2)
        time = pd.DatetimeIndex(pd.to_datetime(selected["time"].values))
        for row, mode in enumerate(modes):
            raw = selected["raw_projected_pc"].sel(mode=mode)
            raw_truth = selected["raw_truth_pc"].sel(mode=mode)
            reconstructed = selected["wavelet_reconstruction_pc"].sel(mode=mode)
            target = selected["wavelet_truth_target_pc"].sel(mode=mode)
            raw_eligible = np.asarray(selected["raw_eligible"].sel(mode=mode).values, dtype=bool)
            valid = np.asarray(selected["wavelet_valid_segment"].sel(mode=mode).values, dtype=bool)
            coi_safe = np.asarray(selected["wavelet_coi_safe"].sel(mode=mode).values, dtype=bool)
            values = [raw.values, raw_truth.values, reconstructed.values, target.values]
            ymin, ymax = _finite_limit(values, symmetric=True)

            left, right = axes[row]
            _shade_mask(
                left,
                time,
                ~raw_eligible,
                color=NCAR_COLORS["yellow"],
                alpha=0.22,
                label="Resolution < 0.3 / missing",
            )
            left.plot(
                time,
                raw.where(raw_eligible).values,
                color=NCAR_COLORS["gray"],
                linewidth=1.0,
                label="Projected PC",
            )
            left.plot(
                time,
                raw_truth.values,
                color=NCAR_COLORS["orange"],
                linewidth=1.35,
                label="Latent correction PC",
            )

            _shade_mask(
                right,
                time,
                ~valid,
                color=NCAR_COLORS["red"],
                alpha=0.15,
                label="Invalid segment",
            )
            _shade_mask(
                right,
                time,
                valid & ~coi_safe,
                color=NCAR_COLORS["gray"],
                alpha=0.16,
                label="Outside 180-day COI",
            )
            right.plot(
                time,
                reconstructed.where(valid).values,
                color=NCAR_COLORS["ncar_blue"],
                linewidth=1.35,
                label="Wavelet reconstruction",
            )
            right.plot(
                time,
                target.values,
                color=NCAR_COLORS["orange"],
                linewidth=1.15,
                label="Analytic filter target",
            )
            right.plot(
                time,
                (reconstructed - target).where(valid).values,
                color=NCAR_COLORS["red"],
                linewidth=0.8,
                alpha=0.8,
                label="Residual",
            )
            for axis in (left, right):
                axis.set_ylim(ymin, ymax)
                axis.set_xlim(time.min(), time.max())
                axis.grid(True, alpha=0.25)
                axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                axis.tick_params(axis="x", rotation=35)
            left.set_ylabel(f"Mode {int(mode)} amplitude")
            if row == 0:
                left.set_title("Projection versus latent correction", fontsize=13)
                right.set_title("Frozen 4-180 day reconstruction", fontsize=13)
                left.legend(loc="upper right", fontsize=8, frameon=True)
                right.legend(loc="upper right", fontsize=8, frameon=True)
            correlation = float(selected["coefficient_correlation"].sel(mode=mode).item())
            slope = float(selected["coefficient_origin_slope"].sel(mode=mode).item())
            nrmse = float(selected["coefficient_nrmse"].sel(mode=mode).item())
            right.text(
                0.012,
                0.97,
                f"R={correlation:.3f}  slope={slope:.3f}  NRMSE={nrmse:.3f}",
                transform=right.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
        fig.suptitle(
            f"FABLE acceptance coefficient recovery | seed {seed}\n"
            "Signed basis-scale alignment; shaded intervals are excluded diagnostics\n"
            + _disposition(dataset),
            fontsize=15,
        )
        return fig


class FableWaveletScalogramPlotter(WaveletScalogramPlotter):
    """Add acceptance seed and mode identity to the standard scalogram."""

    name = "fable_wavelet_scalogram"

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        figure = super().render(series, ax=ax, **kwargs)
        if self.config.title:
            figure.axes[0].set_title(
                format_plot_title(self.config.title),
                fontsize=self.config.text.title_fontsize,
                fontweight=self.config.text.fontweight,
                pad=22 if self.config.subtitle else None,
                wrap=True,
            )
        return figure


_ACCEPTANCE_PLOTTERS = (
    ("fable_spatial_recovery", FableSpatialRecoveryPlotter, "single_source", "spatial"),
    ("fable_eof_comparison", FableEOFComparisonPlotter, "single_source", "spatial"),
    ("fable_pc_reconstruction", FablePCReconstructionPlotter, "single_source", "temporal"),
    (
        "fable_wavelet_scalogram",
        FableWaveletScalogramPlotter,
        "single_source",
        "specialized",
    ),
)


@contextmanager
def registered_acceptance_plotters() -> Iterator[None]:
    """Register acceptance-only plotters for one pipeline invocation."""
    added: list[str] = []
    try:
        for name, plotter, arity, category in _ACCEPTANCE_PLOTTERS:
            if name in plotter_registry:
                if plotter_registry.get(name) is not plotter:
                    raise RuntimeError(f"plot type {name!r} is already registered")
                continue
            register_plotter(name, arity=arity, category=category)(plotter)
            added.append(name)
        yield
    finally:
        for name in reversed(added):
            plotter_registry.unregister(name)


__all__ = [
    "FableEOFComparisonPlotter",
    "FablePCReconstructionPlotter",
    "FableSpatialRecoveryPlotter",
    "FableWaveletScalogramPlotter",
    "registered_acceptance_plotters",
]
