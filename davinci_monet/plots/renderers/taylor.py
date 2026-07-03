"""Taylor diagram plot renderer for DAVINCI.

This module provides Taylor diagram plotting functionality for
visualizing x-vs-y statistical relationships.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    BasePlotter,
    clean_xy,
    extract_xy_series,
    get_axis_color,
    get_variable_label,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

logger = logging.getLogger(__name__)


@register_plotter("taylor", arity="pairwise", category="statistical")
class TaylorPlotter(BasePlotter):
    """Plotter for Taylor diagrams.

    Creates Taylor diagrams showing the statistical relationship
    between x and y data using correlation, standard
    deviation, and centered RMS difference.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = TaylorPlotter()
    >>> fig = plotter.render(build_series(paired_data, "x_o3", "y_o3"))
    """

    name: str = "taylor"
    default_figsize: tuple[float, float] = (6, 6)  # Square for polar diagram

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a Taylor diagram from two or more PlotSeries.

        Parameters
        ----------
        series
            At least 2 series: one x/reference series and one or more y series.
        ax
            Optional axes to plot on (must be polar). If None, creates new.
        **kwargs
            Forwarded to the Taylor rendering logic.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) < 2:
            raise NotImplementedError(
                f"TaylorPlotter.render requires at least 2 series; got {len(series)}."
            )

        normalize: bool = kwargs.pop("normalize", True)
        show_x: bool = kwargs.pop("show_x", True)
        x_label: str | None = kwargs.pop("x_label", None)
        y_label: str | None = kwargs.pop("y_label", None)
        marker: str | None = kwargs.pop("marker", None)
        color: str | None = kwargs.pop("color", None)
        colors: dict[str, str] = kwargs.pop("colors", {}) or {}
        markers: dict[str, str] = kwargs.pop("markers", {}) or {}

        if len(series) == 2:
            paired_data, x_var, y_var = extract_xy_series(series, "TaylorPlotter.render")
            x_series = next((s for s in series if s.axis == "x"), series[0])
            y_series = next((s for s in series if s is not x_series), series[1])
            y_series_list = [y_series]
        else:
            x_series = next((s for s in series if s.axis == "x"), series[0])
            y_series_list = [s for s in series if s is not x_series]
            paired_data = x_series.dataset
            x_var = x_series.var_name
            y_var = y_series_list[0].var_name

        # Use the first comparison to scale the Taylor axes, preserving the
        # historical one-reference/many-model diagram behavior.
        x_values, _ = clean_xy(paired_data[x_var].values, y_series_list[0].dataset[y_var].values)
        x_std = float(np.std(x_values)) if x_values.size else np.nan

        # Normalize if requested
        x_std_norm = 1.0 if normalize else x_std
        if not np.isfinite(x_std_norm) or x_std_norm <= 0:
            x_std_norm = 1.0

        # Create Taylor diagram
        if ax is None:
            fig, ax = self._create_taylor_axes(x_std_norm, normalize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get style
        style = self.config.style
        default_colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        for i, current_y in enumerate(y_series_list):
            # Route source labels through labeling.legend_label for friendly display.
            raw_src = (
                current_y.dataset[current_y.var_name].attrs.get("source_label")
                if current_y.var_name in current_y.dataset
                else None
            )
            friendly_label = (
                labeling.legend_label(raw_src)
                if raw_src
                else get_variable_label(current_y.dataset, current_y.var_name, include_prefix=False)
            )
            if len(y_series_list) == 1:
                label = y_label or self.config.y_label or friendly_label
            else:
                label = friendly_label
            key = current_y.source_label or current_y.var_name
            m = markers.get(key) or markers.get(label) or marker or style.y_marker
            if len(y_series_list) == 1:
                default_color = color or get_axis_color(
                    current_y.dataset,
                    current_y.var_name,
                    current_y.index,
                    x_color=style.x_color,
                    y_color=style.y_color,
                )
            else:
                default_color = default_colors[i % len(default_colors)]
            c = colors.get(key) or colors.get(label) or default_color

            x_values, y_values = clean_xy(
                x_series.dataset[x_series.var_name].values,
                current_y.dataset[current_y.var_name].values,
            )
            x_std = float(np.std(x_values)) if x_values.size else np.nan
            y_std = float(np.std(y_values)) if y_values.size else np.nan
            if (
                x_values.size < 2
                or not np.isfinite(x_std)
                or not np.isfinite(y_std)
                or x_std <= 0.0
                or y_std <= 0.0
            ):
                logger.warning(
                    "Skipping Taylor series '%s' because standard deviation is zero or invalid",
                    label,
                )
                continue
            correlation = float(np.corrcoef(x_values, y_values)[0, 1])
            if not np.isfinite(correlation):
                logger.warning(
                    "Skipping Taylor series '%s' because correlation is undefined",
                    label,
                )
                continue

            y_std_norm = y_std / x_std if normalize else y_std

            # Taylor diagram uses polar coordinates: theta=arccos(correlation), r=std
            theta = np.arccos(np.clip(correlation, -1.0, 1.0))
            ax.plot(
                theta,
                y_std_norm,
                marker=m,
                color=c,
                markersize=style.markersize * 1.5,
                label=label,
                linestyle="none",
            )

        # Plot the x (reference) point. Label it with a friendly source name
        # (R-3), keeping the conventional black star marker.
        if show_x:
            x_raw_src = (
                x_series.dataset[x_series.var_name].attrs.get("source_label")
                if x_series.var_name in x_series.dataset
                else None
            )
            x_default_label = (
                labeling.legend_label(x_raw_src)
                if x_raw_src
                else get_variable_label(x_series.dataset, x_series.var_name, include_prefix=False)
            )
            ax.plot(
                0,  # Perfect correlation
                x_std_norm,
                marker="*",
                color="k",
                markersize=style.markersize * 2,
                label=x_label or self.config.x_label or x_default_label,
                linestyle="none",
            )

        # Add legend
        self.add_legend(ax, loc="upper right")
        self.set_title(ax)

        return fig

    def _create_taylor_axes(
        self,
        ref_std: float,
        normalized: bool,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create axes for Taylor diagram.

        Parameters
        ----------
        ref_std
            X (reference) standard deviation for scaling.
        normalized
            Whether values are normalized.

        Returns
        -------
        tuple[Figure, Axes]
            Figure and polar axes.
        """
        fig = plt.figure(figsize=self.config.figure.figsize, dpi=self.config.figure.dpi)

        # Create polar axes for first quadrant only
        ax = fig.add_subplot(111, projection="polar")
        ax.set_thetamin(0)  # type: ignore[attr-defined]
        ax.set_thetamax(90)  # type: ignore[attr-defined]

        # Set radial limits
        max_std = ref_std * 1.5
        ax.set_ylim(0, max_std)

        # Correlation labels on angular axis
        correlation_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        ax.set_thetagrids(  # type: ignore[attr-defined]
            np.arccos(correlation_ticks) * 180 / np.pi,
            labels=[f"{c:.2g}" for c in correlation_ticks],
        )

        # Labels
        ax.set_xlabel("Standard Deviation" + (" (normalized)" if normalized else ""))
        ax.text(
            np.pi / 4,
            max_std * 1.2,
            "Correlation",
            ha="center",
            va="center",
            rotation=-45,
        )

        # Add centered RMS contours
        self._add_rms_contours(ax, ref_std, max_std)

        return fig, ax

    def _add_rms_contours(
        self,
        ax: matplotlib.axes.Axes,
        ref_std: float,
        max_std: float,
    ) -> None:
        """Add centered RMS difference contours.

        Parameters
        ----------
        ax
            Polar axes to add contours to.
        ref_std
            X (reference) standard deviation.
        max_std
            Maximum standard deviation for plot limits.
        """
        # RMS contours are circles centered at the x (reference) point
        # In polar coordinates centered at origin, these become more complex
        theta = np.linspace(0, np.pi / 2, 100)

        # Draw a few RMS contours
        rms_values = (
            [0.25, 0.5, 0.75, 1.0]
            if ref_std == 1.0
            else [ref_std * x for x in [0.25, 0.5, 0.75, 1.0]]
        )

        for rms in rms_values:
            if rms > max_std:
                continue
            # Circle centered at (ref_std, 0) with radius rms
            # Parametric: x = ref_std + rms*cos(t), y = rms*sin(t)
            t = np.linspace(0, 2 * np.pi, 100)
            x = ref_std + rms * np.cos(t)
            y = rms * np.sin(t)

            # Convert to polar
            r = np.sqrt(x**2 + y**2)
            theta_rms = np.arctan2(y, x)

            # Only keep first quadrant
            mask = (theta_rms >= 0) & (theta_rms <= np.pi / 2) & (r <= max_std)
            if np.any(mask):
                ax.plot(
                    theta_rms[mask],
                    r[mask],
                    "k:",
                    alpha=0.3,
                    linewidth=0.5,
                )
