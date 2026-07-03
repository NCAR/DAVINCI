"""Scatter plot renderer for DAVINCI.

This module provides scatter plot functionality for paired source
comparisons, including density coloring and regression lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots import labeling
from davinci_monet.plots._stats import annotation_metrics
from davinci_monet.plots.base import (
    BasePlotter,
    build_series,
    extract_xy_series,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import NCAR_ACCENT
from davinci_monet.plots.titles import title_for_labeled_subset

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("scatter", arity="pairwise", category="statistical")
class ScatterPlotter(BasePlotter):
    """Plotter for scatter comparisons.

    Creates scatter plots comparing paired source values, with options
    for density coloring, regression lines, and
    statistical annotations.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = ScatterPlotter()
    >>> fig = plotter.render(
    ...     build_series(paired_data, "airnow_o3", "cam_o3"),
    ...     show_density=True,
    ...     show_regression=True,
    ... )
    """

    name: str = "scatter"
    default_figsize: tuple[float, float] = (6, 6)  # Square for x vs y

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure | list[tuple[str, matplotlib.figure.Figure]]:
        """Render a scatter plot from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the scatter rendering logic.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        paired_data, x_var, y_var = extract_xy_series(series, "ScatterPlotter.render")

        split_by_flight: bool = kwargs.pop("split_by_flight", False)
        flight_coord: str = kwargs.pop("flight_coord", "flight")
        min_points: int = kwargs.pop("min_points", 10)
        if split_by_flight:
            return self._render_by_flight(
                paired_data,
                x_var,
                y_var,
                flight_coord=flight_coord,
                min_points=min_points,
                **kwargs,
            )

        show_density: bool = kwargs.pop("show_density", False)
        density_cmap: str = kwargs.pop("density_cmap", "viridis")
        density_bins: int = kwargs.pop("density_bins", 50)
        show_regression: bool = kwargs.pop("show_regression", True)
        show_one_to_one: bool = kwargs.pop("show_one_to_one", True)
        show_stats: bool = kwargs.pop("show_stats", True)
        color_by: str | None = kwargs.pop("color_by", None)
        marker_size: float | None = kwargs.pop("marker_size", None)
        alpha: float | None = kwargs.pop("alpha", None)

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get data and flatten
        x_values = paired_data[x_var].values.flatten()
        y_values = paired_data[y_var].values.flatten()

        # Remove NaN values
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[mask]
        y_values = y_values[mask]

        if len(x_values) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.config.text.fontsize,
            )
            return fig

        # Calculate limits
        all_values = np.concatenate([x_values, y_values])
        vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(all_values)
        vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(all_values)

        # Add padding
        padding = (vmax - vmin) * 0.05
        vmin -= padding
        vmax += padding

        # Get style configuration
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize
        a = alpha if alpha is not None else (0.5 if len(x_values) > 1000 else style.alpha)

        # Scatter plot
        if show_density and len(x_values) > 10:
            # Density-colored scatter
            density = self._calculate_density(x_values, y_values, density_bins)
            scatter = ax.scatter(
                x_values,
                y_values,
                c=density,
                s=ms**2,
                alpha=a,
                cmap=density_cmap,
                norm=LogNorm() if density.max() / density.min() > 10 else None,
            )
            fig.colorbar(scatter, ax=ax, label="Point Density")
        elif color_by is not None and color_by in paired_data:
            # Color by another variable
            color_values = paired_data[color_by].values.flatten()[mask]
            scatter = ax.scatter(
                x_values,
                y_values,
                c=color_values,
                s=ms**2,
                alpha=a,
                cmap=density_cmap,
            )
            color_label = get_variable_label(paired_data, color_by)
            fig.colorbar(scatter, ax=ax, label=color_label)
        else:
            ax.scatter(
                x_values,
                y_values,
                c=style.y_color,
                s=ms**2,
                alpha=a,
                edgecolors="none",
            )

        # 1:1 line
        if show_one_to_one:
            ax.plot(
                [vmin, vmax],
                [vmin, vmax],
                "k--",
                linewidth=1,
                label="1:1",
                zorder=1,
            )

        # Regression line
        if show_regression:
            self._add_regression_line(ax, x_values, y_values, vmin, vmax)

        # Statistics annotation
        if show_stats:
            self._add_stats_annotation(ax, x_values, y_values)

        # Set equal aspect and limits
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")

        # Formatting
        self.apply_text_style(ax)

        # Set labels. Explicit config labels are complete overrides; otherwise
        # build per-axis labels via the central labeling module so source names
        # are clean and units are always SI-formatted.
        x_units = get_variable_units(paired_data, x_var)
        y_units = get_variable_units(paired_data, y_var)
        x_q = labeling.quantity_label(paired_data, x_var)
        y_q = labeling.quantity_label(paired_data, y_var)
        x_src = paired_data[x_var].attrs.get("source_label") if x_var in paired_data else None
        y_src = paired_data[y_var].attrs.get("source_label") if y_var in paired_data else None
        x_label = self.config.x_label or labeling.axis_label(x_q, x_units, source=x_src)
        y_label = self.config.y_label or labeling.axis_label(y_q, y_units, source=y_src)
        self.set_labels(ax, xlabel=x_label, ylabel=y_label)

        # Grid
        ax.grid(True, alpha=0.3)

        # Legend
        if show_one_to_one or show_regression:
            self.add_legend(ax, loc="upper left")

        return fig

    def _render_by_flight(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        flight_coord: str = "flight",
        min_points: int = 10,
        **kwargs: Any,
    ) -> list[tuple[str, matplotlib.figure.Figure]]:
        """Render one labeled scatter figure per flight."""
        if flight_coord not in paired_data.coords:
            raise ValueError(
                f"Flight coordinate '{flight_coord}' not found in paired data. "
                f"Available coordinates: {list(paired_data.coords)}"
            )

        results: list[tuple[str, matplotlib.figure.Figure]] = []
        flight_values = paired_data[flight_coord].values
        unique_flights = np.unique(flight_values)

        for flight in unique_flights:
            flight_str = str(flight)
            mask = flight_values == flight
            flight_data = paired_data.isel(time=mask)
            x_vals = flight_data[x_var].values.flatten()
            y_vals = flight_data[y_var].values.flatten()
            valid = np.isfinite(x_vals) & np.isfinite(y_vals)

            if valid.sum() < min_points:
                continue

            original_title = self.config.title
            original_subtitle = self.config.subtitle
            self.config.title, flight_subtitle = title_for_labeled_subset(
                original_title,
                flight_str,
                label_prefix="Flight",
            )
            if flight_subtitle:
                self.config.subtitle = flight_subtitle

            try:
                fig = self.render(build_series(flight_data, x_var, y_var), **kwargs)
            finally:
                self.config.title = original_title
                self.config.subtitle = original_subtitle

            flight_id = flight_str.replace("-", "")
            if isinstance(fig, list):
                results.extend((f"{flight_id}_{label}", subfig) for label, subfig in fig)
            else:
                results.append((flight_id, fig))

        return results

    def _calculate_density(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: int,
    ) -> np.ndarray:
        """Calculate point density for coloring.

        Parameters
        ----------
        x, y
            Data coordinates.
        bins
            Number of bins for histogram.

        Returns
        -------
        np.ndarray
            Density values for each point.
        """
        # 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

        # Find bin indices for each point
        xidx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
        yidx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)

        # Get density at each point
        density = hist[xidx, yidx]

        return density

    def _add_regression_line(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        """Add regression line to plot.

        Parameters
        ----------
        ax
            Axes to add line to.
        x, y
            Data arrays.
        vmin, vmax
            Axis limits.
        """
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        # Plot line
        x_line = np.array([vmin, vmax])
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color=NCAR_ACCENT,
            linestyle="-",
            linewidth=1.5,
            label=f"y = {slope:.2f}x + {intercept:.2f}",
            zorder=2,
        )

    def _add_stats_annotation(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Add statistics annotation to plot.

        Parameters
        ----------
        ax
            Axes to annotate.
        x, y
            Data arrays.
        """
        # Calculate statistics (via central metric registry)
        stats = annotation_metrics(x, y, ["N", "MB", "RMSE", "R", "NME"])
        n = int(stats["N"])
        mb = stats["MB"]
        rmse = stats["RMSE"]
        r = stats["R"]
        nme = stats["NME"]

        # Create annotation text
        stats_text = (
            f"N = {n:,}\n"
            f"MB = {mb:.3g}\n"
            f"RMSE = {rmse:.3g}\n"
            f"R = {r:.3f}\n"
            f"NME = {nme:.1f}%"
        )

        # Add text box
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.95,
            0.05,
            stats_text,
            transform=ax.transAxes,
            fontsize=self.config.text.annotation,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )
