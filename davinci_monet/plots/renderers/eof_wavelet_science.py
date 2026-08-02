"""Scientific summary figures for the aerosol EOF/projection/wavelet workflow."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.plots.base import build_series
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.eof_wavelet_science_data import (
    SEASONS,
    dataset_roles,
    finite_percentile,
    seasonal_fields,
    select_modes,
    spatial_rms,
    symmetric_limit,
    validate_identity,
)
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
    draw_spatial_field,
    get_projection,
    resolve_spatial_coords,
)
from davinci_monet.plots.renderers.wavelet_scalogram import WaveletScalogramPlotter
from davinci_monet.plots.style import (
    NCAR_COLORS,
    NCAR_PRIMARY,
    get_bias_cmap,
    get_sequential_cmap,
)

if TYPE_CHECKING:
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


_GLOBAL_MAP = MapConfig(
    resolution="110m",
    show_states=False,
    show_countries=False,
    show_gridlines=False,
    land_color="none",
    ocean_color="none",
)


@register_plotter("eof_wavelet_science", arity="multi_source", category="specialized")
class EOFWaveletSciencePlotter(BaseSpatialPlotter):
    """Render the connected seasonal, EOF, projection, and wavelet science story."""

    name = "eof_wavelet_science"
    default_figsize = (12, 8)

    def _map_panels(
        self,
        fields: list[xr.DataArray],
        titles: list[str],
        *,
        figure_title: str,
        colorbar_label: str,
        cmap: str,
        vmin: float,
        vmax: float,
        columns: int,
    ) -> matplotlib.figure.Figure:
        rows = math.ceil(len(fields) / columns)
        fig = plt.figure(
            figsize=(4.7 * columns, 3.2 * rows + 0.8),
            dpi=self.config.figure.dpi,
            facecolor=self.config.figure.facecolor,
            constrained_layout=True,
        )
        projection = get_projection(_GLOBAL_MAP.projection)
        axes = []
        mappable = None
        for index, (field, title) in enumerate(zip(fields, titles, strict=True)):
            ax = fig.add_subplot(rows, columns, index + 1, projection=projection)
            self.add_map_features(ax, _GLOBAL_MAP)
            ax.set_global()
            ds = field.to_dataset(name="field")
            lat_name, lon_name, lats, lons = resolve_spatial_coords(ds)
            mappable = draw_spatial_field(
                ax,
                field.values,
                lats,
                lons,
                plot_type="pcolormesh",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                marker_size=self.config.style.markersize,
                alpha=1.0,
                field_dims=tuple(field.dims),
                lat_dim=ds[lat_name].dims[0],
                lon_dim=ds[lon_name].dims[0],
            )
            ax.set_title(title, fontsize=self.config.text.fontsize)
            axes.append(ax)
        if mappable is None:
            raise ValueError("no map panels were requested")
        fig.colorbar(
            mappable,
            ax=axes,
            orientation="horizontal",
            fraction=0.045,
            pad=0.035,
            label=colorbar_label,
        )
        fig.suptitle(figure_title, fontsize=self.config.text.title_fontsize)
        return fig

    def _variance_figure(self, basis: xr.Dataset) -> matplotlib.figure.Figure:
        modes = np.asarray(basis["mode"].values, dtype=int)
        variance = np.asarray(basis["explained_variance"].values, dtype=float) * 100.0
        error = (
            np.asarray(basis["explained_variance_error"].values, dtype=float) * 100.0
            if "explained_variance_error" in basis
            else None
        )
        cumulative = np.cumsum(variance)
        fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
        bars = ax.bar(
            modes,
            variance,
            color=NCAR_PRIMARY,
            yerr=error,
            capsize=2,
            label="Individual mode",
        )
        ax.set_xlabel("EOF mode")
        ax.set_ylabel("Explained variance (%)")
        ax.grid(True, axis="y", alpha=0.3)
        step = 5 if modes.size <= 60 else 10
        ticks = np.unique(np.r_[modes[0], modes[step - 1 :: step], modes[-1]])
        ax.set_xticks(ticks)
        twin = ax.twinx()
        (line,) = twin.plot(
            modes,
            cumulative,
            color=NCAR_COLORS["orange"],
            linewidth=2.2,
            label="Cumulative",
        )
        twin.set_ylabel("Cumulative variance (%)")
        twin.set_ylim(0.0, 100.0)
        ax.legend([bars, line], ["Individual mode", "Cumulative"], loc="upper right")
        ax.set_title("De-seasonalized log-AOD EOF variance")
        ax.text(
            0.99,
            0.82,
            f"{modes.size} retained modes: {cumulative[-1]:.1f}%",
            transform=ax.transAxes,
            ha="right",
            fontsize=self.config.text.annotation_small,
        )
        return fig

    def _pc_figure(
        self,
        basis: xr.Dataset,
        projection: xr.Dataset,
        filtered: xr.Dataset,
        modes: list[int],
    ) -> matplotlib.figure.Figure:
        fig, axes = plt.subplots(
            len(modes),
            1,
            figsize=(12, 2.5 * len(modes) + 1.2),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        time = np.asarray(basis["time"].values)
        for index, mode in enumerate(modes):
            ax = axes[index, 0]
            model_pc = np.asarray(basis["pc"].sel(mode=mode).values, dtype=float)
            projected_pc = np.asarray(projection["pc"].sel(mode=mode).values, dtype=float)
            filtered_pc = np.asarray(filtered["pc"].sel(mode=mode).values, dtype=float)
            posterior_variance = np.asarray(
                projection["posterior_variance"].sel(mode=mode).values,
                dtype=float,
            )
            posterior_std = np.sqrt(np.clip(posterior_variance, 0.0, None))
            ax.fill_between(
                time,
                projected_pc - posterior_std,
                projected_pc + posterior_std,
                color=NCAR_PRIMARY,
                alpha=0.15,
                linewidth=0,
                label="Projected ±1σ" if index == 0 else None,
            )
            ax.plot(time, model_pc, color=NCAR_COLORS["gray"], linewidth=1.0, label="MERRA-2")
            ax.plot(time, projected_pc, color=NCAR_PRIMARY, linewidth=1.2, label="MODIS projection")
            ax.plot(
                time,
                filtered_pc,
                color=NCAR_COLORS["orange"],
                linewidth=1.4,
                label="4–180 day component",
            )
            explained = (
                float(np.asarray(basis["explained_variance"].sel(mode=mode).values).item()) * 100.0
            )
            resolution = float(
                np.nanmedian(np.asarray(projection["resolution"].sel(mode=mode).values))
            )
            retained = (
                float(np.asarray(filtered["retained_variance"].sel(mode=mode).values).item())
                * 100.0
            )
            ax.set_title(
                f"Mode {mode} · EOF {explained:.2f}% · median resolution {resolution:.3f} "
                f"· band variance {retained:.1f}%",
                fontsize=self.config.text.fontsize,
            )
            ax.set_ylabel("PC")
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(ncol=4, fontsize=self.config.text.annotation_small, loc="upper right")
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        axes[-1, 0].xaxis.set_major_locator(locator)
        axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[-1, 0].set_xlabel("Time")
        fig.suptitle("Model, MODIS-projected, and wavelet-filtered EOF coefficients")
        return fig

    def _mode_quality_figure(
        self,
        projection: xr.Dataset,
        filtered: xr.Dataset,
    ) -> matplotlib.figure.Figure:
        modes = np.asarray(projection["mode"].values, dtype=int)
        resolution = np.asarray(projection["resolution"].values, dtype=float)
        retained = np.asarray(filtered["retained_variance"].values, dtype=float) * 100.0
        error = np.asarray(filtered["recon_error"].values, dtype=float) * 100.0
        median = np.nanmedian(resolution, axis=0)
        low, high = np.nanpercentile(resolution, [5.0, 95.0], axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        axes[0].fill_between(modes, low, high, color=NCAR_PRIMARY, alpha=0.18, label="5–95%")
        axes[0].plot(modes, median, color=NCAR_PRIMARY, label="Median")
        threshold = float(projection.attrs.get("projection_min_resolution", 0.3))
        axes[0].axhline(threshold, color="black", linestyle="--", linewidth=1.0)
        axes[0].set_ylabel("Resolution")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].legend(loc="lower left")
        axes[1].bar(modes, retained, color=NCAR_COLORS["aqua"])
        axes[1].set_ylabel("Band variance (%)")
        axes[2].bar(modes, error, color=NCAR_COLORS["orange"])
        axes[2].axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="5% warning")
        axes[2].set_ylabel("CWT inverse error (%)")
        axes[2].set_xlabel("EOF mode")
        axes[2].legend(loc="upper left")
        for ax in axes:
            ax.grid(True, axis="y", alpha=0.3)
        step = 5 if modes.size <= 60 else 10
        axes[2].set_xticks(np.unique(np.r_[modes[0], modes[step - 1 :: step], modes[-1]]))
        fig.suptitle("Projection observability and wavelet-filter quality by mode")
        return fig

    def render(
        self,
        series: list[PlotSeries],
        ax: Any = None,
        *,
        modes: list[int] | None = None,
        pc_modes: list[int] | None = None,
        wavelet_modes: list[int] | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, matplotlib.figure.Figure]]:
        del ax, kwargs
        basis, projection, filtered = dataset_roles(series)
        validate_identity(basis, projection, filtered)
        eof_modes = select_modes(basis, modes, 6)
        pc_selected = select_modes(basis, pc_modes, min(4, len(eof_modes)))
        wavelet_selected = select_modes(basis, wavelet_modes, min(3, len(eof_modes)))

        seasonal, departures, biases = seasonal_fields(basis, projection)
        season_titles = [label for label, _months in SEASONS]
        seasonal_vmin = max(0.0, finite_percentile(seasonal, 1.0))
        seasonal_vmax = finite_percentile(seasonal, 99.0)
        departure_limit = symmetric_limit(departures)
        bias_limit = symmetric_limit(biases)

        patterns = [basis["eofs"].sel(mode=mode) for mode in eof_modes]
        pattern_limit = symmetric_limit(patterns)
        pattern_titles = [
            f"Mode {mode} · {100.0 * float(basis['explained_variance'].sel(mode=mode)):.2f}%"
            for mode in eof_modes
        ]
        rms_fields = spatial_rms(basis, projection, filtered)
        rms_limit = finite_percentile(rms_fields, 99.0)

        figures: list[tuple[str, matplotlib.figure.Figure]] = [
            (
                "seasonal_aod",
                self._map_panels(
                    seasonal,
                    season_titles,
                    figure_title="MERRA-2 seasonal AOD patterns removed before EOF fitting",
                    colorbar_label="Geometric-mean AOD at 550 nm",
                    cmap=get_sequential_cmap(),
                    vmin=seasonal_vmin,
                    vmax=seasonal_vmax,
                    columns=2,
                ),
            ),
            (
                "seasonal_aod_departure",
                self._map_panels(
                    departures,
                    season_titles,
                    figure_title="Seasonal AOD departure from the annual geometric mean",
                    colorbar_label="AOD departure",
                    cmap=get_bias_cmap(),
                    vmin=-departure_limit,
                    vmax=departure_limit,
                    columns=2,
                ),
            ),
            (
                "seasonal_projection_bias",
                self._map_panels(
                    biases,
                    season_titles,
                    figure_title="MODIS-projected seasonal climatological bias",
                    colorbar_label="Shifted-log AOD bias",
                    cmap=get_bias_cmap(),
                    vmin=-bias_limit,
                    vmax=bias_limit,
                    columns=2,
                ),
            ),
            (
                "eof_patterns",
                self._map_panels(
                    patterns,
                    pattern_titles,
                    figure_title="Leading de-seasonalized log-AOD EOF patterns",
                    colorbar_label="log(AOD + 0.01) per 1σ PC",
                    cmap=get_bias_cmap(),
                    vmin=-pattern_limit,
                    vmax=pattern_limit,
                    columns=3,
                ),
            ),
            ("variance_summary", self._variance_figure(basis)),
            ("pc_comparison", self._pc_figure(basis, projection, filtered, pc_selected)),
            ("mode_quality", self._mode_quality_figure(projection, filtered)),
            (
                "spatial_wavelet_rms",
                self._map_panels(
                    rms_fields,
                    [str(field.attrs["long_name"]) for field in rms_fields],
                    figure_title="Physical-space RMS of projected EOF anomalies",
                    colorbar_label="RMS shifted-log AOD anomaly",
                    cmap=get_sequential_cmap(),
                    vmin=0.0,
                    vmax=rms_limit,
                    columns=3,
                ),
            ),
        ]

        for mode in wavelet_selected:
            selected = filtered.sel(mode=mode).copy(deep=False)
            selected.attrs = {**filtered.attrs, "wavelet_quantity": f"Projected PC {mode}"}
            figure = WaveletScalogramPlotter().render(build_series(selected, "power"))
            title = figure.axes[0].get_title()
            figure.axes[0].set_title("")
            figure.suptitle(title, y=0.97, fontsize=self.config.text.title_fontsize)
            figure.subplots_adjust(top=0.88)
            figures.append((f"wavelet_mode{mode}", figure))
        return figures


__all__ = ["EOFWaveletSciencePlotter"]
