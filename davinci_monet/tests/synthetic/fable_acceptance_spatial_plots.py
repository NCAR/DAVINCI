"""Spatial renderers for FABLE synthetic acceptance diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
    draw_spatial_field,
    get_projection,
)
from davinci_monet.plots.style import NCAR_COLORS, get_bias_cmap, get_sequential_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


_MAP_CONFIG = MapConfig(
    projection="Robinson",
    show_states=False,
    show_countries=True,
    show_coastlines=True,
    show_gridlines=True,
    resolution="110m",
    land_color="none",
    ocean_color="none",
)


def _one_dataset(series: list[PlotSeries], name: str) -> Any:
    if len(series) != 1:
        raise NotImplementedError(f"{name} requires exactly one diagnostic source")
    return series[0].dataset


def _finite_limit(values: list[np.ndarray], *, symmetric: bool) -> tuple[float, float]:
    finite = np.concatenate(
        [np.asarray(value, dtype=float)[np.isfinite(value)] for value in values]
    )
    if not finite.size:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), 99.5))
        return -max(limit, np.finfo(float).eps), max(limit, np.finfo(float).eps)
    upper = float(np.nanpercentile(finite, 99.5))
    return 0.0, max(upper, np.finfo(float).eps)


def _map_field(
    plotter: BaseSpatialPlotter,
    ax: matplotlib.axes.Axes,
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    cmap: Any,
    norm: Normalize,
) -> Any:
    ax.set_global()  # type: ignore[attr-defined]
    plotter.add_map_features(ax, _MAP_CONFIG)
    return draw_spatial_field(
        ax,
        values,
        lat,
        lon,
        plot_type="pcolormesh",
        cmap=cmap,
        vmin=None,
        vmax=None,
        norm=norm,
        marker_size=2.0,
        alpha=1.0,
    )


def _column_title(ax: matplotlib.axes.Axes, title: str) -> None:
    ax.text(
        0.5,
        1.025,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=NCAR_COLORS["space"],
        clip_on=False,
    )


def _disposition(dataset: Any) -> str:
    policy = dataset.attrs.get("selected_policy_id", "fable-v1-all-band")
    disposition = dataset.attrs.get(
        "diagnostic_disposition", "diagnostic only; frozen acceptance remains rejected"
    )
    return f"{policy} | {disposition}"


class FableSpatialRecoveryPlotter(BaseSpatialPlotter):
    """Render cross-seed snapshot and temporal-RMS recovery maps."""

    name = "fable_spatial_recovery"
    default_figsize = (15, 10)

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config, _MAP_CONFIG)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, matplotlib.figure.Figure]]:
        del ax, kwargs
        dataset = _one_dataset(series, type(self).__name__)
        return [
            ("snapshot", self._snapshot(dataset)),
            ("temporal_rms", self._temporal_rms(dataset)),
        ]

    def _axes(self, rows: int) -> tuple[matplotlib.figure.Figure, np.ndarray]:
        fig, axes = plt.subplots(
            rows,
            3,
            figsize=self.default_figsize,
            dpi=self.config.figure.dpi,
            subplot_kw={"projection": get_projection("Robinson")},
            layout="constrained",
        )
        return fig, np.asarray(axes, dtype=object).reshape(rows, 3)

    def _snapshot(self, dataset: Any) -> matplotlib.figure.Figure:
        names = ("truth_snapshot", "estimate_snapshot", "residual_snapshot")
        titles = ("Analytic filter target", "FABLE reconstruction", "Estimate - truth")
        shared = _finite_limit([dataset[name].values for name in names[:2]], symmetric=True)
        residual = _finite_limit([dataset[names[2]].values], symmetric=True)
        norms = (
            TwoSlopeNorm(vmin=shared[0], vcenter=0.0, vmax=shared[1]),
            TwoSlopeNorm(vmin=shared[0], vcenter=0.0, vmax=shared[1]),
            TwoSlopeNorm(vmin=residual[0], vcenter=0.0, vmax=residual[1]),
        )
        cmap = get_bias_cmap()
        fig, axes = self._axes(dataset.sizes["seed"])
        lat = dataset["lat"].values
        lon = dataset["lon"].values
        for row, seed in enumerate(dataset["seed"].values):
            for col, (name, norm) in enumerate(zip(names, norms, strict=True)):
                _map_field(
                    self,
                    axes[row, col],
                    dataset[name].sel(seed=seed).values,
                    lat,
                    lon,
                    cmap=cmap,
                    norm=norm,
                )
                if row == 0:
                    _column_title(axes[row, col], titles[col])
            count = int(dataset["snapshot_valid_count"].sel(seed=seed).item())
            nrmse = float(dataset["snapshot_nrmse"].sel(seed=seed).item())
            axes[row, 0].text(
                0.98,
                0.96,
                f"Seed {int(seed)}\nN={count:,}\nNRMSE={nrmse:.3f}",
                transform=axes[row, 0].transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=NCAR_COLORS["space"],
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
        fig.colorbar(
            ScalarMappable(norm=norms[0], cmap=cmap),
            ax=axes[:, :2].ravel().tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            label="Shifted-log AOD correction",
        )
        fig.colorbar(
            ScalarMappable(norm=norms[2], cmap=cmap),
            ax=axes[:, 2].ravel().tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            label="Residual correction",
        )
        timestamp = pd.Timestamp(dataset.attrs["snapshot_time"]).strftime("%Y-%m-%d %H:%M UTC")
        fig.suptitle(
            "FABLE acceptance: primary-mask spatial correction snapshot\n"
            f"{timestamp}\n{_disposition(dataset)}",
            fontsize=15,
        )
        return fig

    def _temporal_rms(self, dataset: Any) -> matplotlib.figure.Figure:
        names = (
            "truth_correction_rms",
            "estimate_correction_rms",
            "residual_correction_rms",
        )
        titles = ("Target temporal RMS", "Reconstruction temporal RMS", "Residual temporal RMS")
        shared = _finite_limit([dataset[name].values for name in names[:2]], symmetric=False)
        residual = _finite_limit([dataset[names[2]].values], symmetric=False)
        norms = (
            Normalize(vmin=shared[0], vmax=shared[1]),
            Normalize(vmin=shared[0], vmax=shared[1]),
            Normalize(vmin=residual[0], vmax=residual[1]),
        )
        cmap = get_sequential_cmap()
        fig, axes = self._axes(dataset.sizes["seed"])
        lat = dataset["lat"].values
        lon = dataset["lon"].values
        for row, seed in enumerate(dataset["seed"].values):
            for col, (name, norm) in enumerate(zip(names, norms, strict=True)):
                _map_field(
                    self,
                    axes[row, col],
                    dataset[name].sel(seed=seed).values,
                    lat,
                    lon,
                    cmap=cmap,
                    norm=norm,
                )
                if row == 0:
                    _column_title(axes[row, col], titles[col])
            count = int(dataset["primary_valid_count"].sel(seed=seed).item())
            nrmse = float(dataset["field_nrmse"].sel(seed=seed).item())
            axes[row, 0].text(
                0.98,
                0.96,
                f"Seed {int(seed)}\nN={count:,}\nNRMSE={nrmse:.3f}",
                transform=axes[row, 0].transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=NCAR_COLORS["space"],
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
        fig.colorbar(
            ScalarMappable(norm=norms[0], cmap=cmap),
            ax=axes[:, :2].ravel().tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            label="Temporal RMS of shifted-log correction",
        )
        fig.colorbar(
            ScalarMappable(norm=norms[2], cmap=cmap),
            ax=axes[:, 2].ravel().tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            label="Temporal RMS residual",
        )
        fig.suptitle(
            "FABLE acceptance: temporal RMS over the frozen primary domain\n"
            + _disposition(dataset),
            fontsize=15,
        )
        return fig


class FableEOFComparisonPlotter(BaseSpatialPlotter):
    """Render truth, aligned learned, and residual EOF patterns for each seed."""

    name = "fable_eof_comparison"
    default_figsize = (17, 10)

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config, _MAP_CONFIG)

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
        names = ("truth_eof", "learned_eof_aligned", "eof_residual")
        titles = ("True pattern", "Matched learned pattern", "Learned - truth")
        selected = dataset.sel(seed=seed)
        vmin, vmax = _finite_limit([selected[name].values for name in names], symmetric=True)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = get_bias_cmap()
        modes = selected["mode"].values
        fig = plt.figure(
            figsize=self.default_figsize,
            dpi=self.config.figure.dpi,
            layout="constrained",
        )
        grid = fig.add_gridspec(len(modes), 4, width_ratios=(3, 3, 3, 1.2))
        axes = np.empty((len(modes), 3), dtype=object)
        metrics_axes = []
        for row in range(len(modes)):
            for col in range(3):
                axes[row, col] = fig.add_subplot(
                    grid[row, col], projection=get_projection("Robinson")
                )
            metrics_axes.append(fig.add_subplot(grid[row, 3]))
        lat = selected["lat"].values
        lon = selected["lon"].values
        for row, mode in enumerate(modes):
            for col, name in enumerate(names):
                _map_field(
                    self,
                    axes[row, col],
                    selected[name].sel(mode=mode).values,
                    lat,
                    lon,
                    cmap=cmap,
                    norm=norm,
                )
                if row == 0:
                    _column_title(axes[row, col], titles[col])
            similarity = float(selected["mode_similarity"].sel(mode=mode).item())
            variance = 100.0 * float(selected["explained_variance"].sel(mode=mode).item())
            observable = bool(selected["mode_observable"].sel(mode=mode).item())
            label = "observable" if observable else "unobservable\nexcluded from PC scoring"
            metrics_ax = metrics_axes[row]
            metrics_ax.axis("off")
            metrics_ax.text(
                0.02,
                0.5,
                f"Mode {int(mode)}\nSimilarity={similarity:.3f}\nEV={variance:.1f}%\n{label}",
                transform=metrics_ax.transAxes,
                ha="left",
                va="center",
                fontsize=9,
                color=NCAR_COLORS["space"],
            )
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            label="Pattern amplitude in truth normalization",
        )
        angle = float(selected["subspace_angle_mean_degrees"].item())
        angle_max = float(selected["subspace_angle_max_degrees"].item())
        projector = float(selected["subspace_projector_error"].item())
        fig.suptitle(
            f"FABLE acceptance EOF recovery | seed {seed}\n"
            f"Mean/max subspace angle {angle:.2f}°/{angle_max:.2f}°; "
            f"projector error {projector:.3f}\n{_disposition(dataset)}",
            fontsize=15,
        )
        return fig


__all__ = ["FableEOFComparisonPlotter", "FableSpatialRecoveryPlotter"]
