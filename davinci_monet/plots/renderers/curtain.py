"""Curtain plot renderer for DAVINCI.

This module provides curtain plot functionality for visualizing
vertical cross-sections of aircraft or gridded data along a trajectory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    BasePlotter,
    calculate_symmetric_limits,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


@register_plotter("curtain", arity="pairwise", category="specialized")
class CurtainPlotter(BasePlotter):
    """Plotter for vertical curtain plots.

    Creates 2D plots showing values along a trajectory (time × altitude),
    useful for aircraft data and vertical cross-sections.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = CurtainPlotter()
    >>> fig = plotter.render(
    ...     build_series(paired_data, "x_o3", "y_o3"),
    ...     alt_var="altitude",
    ... )
    """

    name: str = "curtain"
    default_figsize: tuple[float, float] = (9, 4)  # Wide for distance/time extent

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a curtain plot from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the curtain rendering logic, including ``alt_var``,
            ``time_dim``, ``show_var``, ``cmap``, ``n_levels``,
            ``show_scatter``, ``scatter_size``, ``invert_yaxis``.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"CurtainPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        alt_var: str = kwargs.pop("alt_var", "altitude")
        time_dim: str = kwargs.pop("time_dim", "time")
        show_var: str = kwargs.pop("show_var", "bias")
        cmap: str | None = kwargs.pop("cmap", None)
        n_levels: int = kwargs.pop("n_levels", 20)
        show_scatter: bool = kwargs.pop("show_scatter", True)
        scatter_size: float | None = kwargs.pop("scatter_size", None)
        invert_yaxis: bool = kwargs.pop("invert_yaxis", False)

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get data
        x_data = paired_data[x_var]
        y_data = paired_data[y_var]

        # Calculate bias if needed
        if show_var == "bias":
            plot_data = y_data - x_data
            default_cmap = "RdBu_r"
        elif show_var == "x":
            plot_data = x_data
            default_cmap = "viridis"
        else:
            plot_data = y_data
            default_cmap = "viridis"

        cmap = cmap or default_cmap

        # Get coordinates
        time = paired_data[time_dim]
        if alt_var in paired_data.coords:
            altitude = paired_data[alt_var]
        elif alt_var in paired_data:
            altitude = paired_data[alt_var]
        else:
            raise ValueError(f"Could not find altitude variable {alt_var}")

        # Get values
        time_values = time.values
        alt_values = altitude.values
        data_values = plot_data.values

        # Build colorbar label before dispatching to sub-renderers.
        units = get_variable_units(paired_data, x_var)
        y_src = paired_data[y_var].attrs.get("source_label") or ""
        x_src = paired_data[x_var].attrs.get("source_label") or ""
        if show_var == "bias":
            cbar_label: str = labeling.bias_label(
                y_src, x_src, units, quantity=labeling.quantity_label(paired_data, x_var)
            )
        elif show_var == "x":
            # Colorbar carries units only; the title carries the quantity.
            cbar_label = labeling.format_units(get_variable_units(paired_data, x_var))
        else:
            cbar_label = labeling.format_units(get_variable_units(paired_data, y_var))

        # Handle different data shapes
        if data_values.ndim == 1:
            # 1D trajectory data - create 2D representation
            # This is a simple approach; more sophisticated binning could be used
            self._plot_1d_trajectory(
                ax,
                time_values,
                alt_values,
                data_values,
                x_data.values if show_scatter else None,
                cmap,
                show_var,
                scatter_size,
                cbar_label,
            )
        else:
            # 2D or higher - use contourf
            self._plot_2d_curtain(
                ax,
                time_values,
                alt_values,
                data_values,
                cmap,
                n_levels,
                show_var,
                cbar_label,
            )

        # Calculate limits
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(data_values)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            data_flat = data_values[np.isfinite(data_values)]
            vmin = (
                self.config.vmin if self.config.vmin is not None else np.nanpercentile(data_flat, 2)
            )
            vmax = (
                self.config.vmax
                if self.config.vmax is not None
                else np.nanpercentile(data_flat, 98)
            )
            norm = None

        # Formatting
        self.apply_text_style(ax)

        alt_units = get_variable_units(paired_data, alt_var) or "m"
        self.set_labels(
            ax,
            xlabel="Time",
            ylabel=labeling.axis_label("Altitude", alt_units),
        )

        # Title
        if self.config.title:
            self.set_title(ax, self.config.title)
        else:
            var_label = get_variable_label(paired_data, x_var)
            title = labeling.title_text(var_label, operation="Bias" if show_var == "bias" else None)
            self.set_title(ax, title)

        # Invert y-axis if needed (e.g., for pressure)
        if invert_yaxis:
            ax.invert_yaxis()

        # Rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

        return fig

    def _plot_1d_trajectory(
        self,
        ax: matplotlib.axes.Axes,
        time_values: np.ndarray,
        alt_values: np.ndarray,
        data_values: np.ndarray,
        x_values: np.ndarray | None,
        cmap: str,
        show_var: str,
        scatter_size: float | None,
        cbar_label: str = "",
    ) -> None:
        """Plot 1D trajectory data as colored scatter.

        Parameters
        ----------
        ax
            Axes to plot on.
        time_values
            Time coordinates.
        alt_values
            Altitude values.
        data_values
            Data to color by.
        x_values
            Optional x-source values for scatter.
        cmap
            Colormap.
        show_var
            Which variable is shown.
        scatter_size
            Scatter point size.
        cbar_label
            Label for the colorbar.
        """
        import pandas as pd

        # Convert time to numeric for plotting
        time_numeric = pd.to_datetime(time_values)

        # Remove NaN
        mask = np.isfinite(data_values) & np.isfinite(alt_values)
        time_plot = time_numeric[mask]
        alt_plot = alt_values[mask]
        data_plot = data_values[mask]

        # Calculate limits
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(data_plot)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            vmin = np.nanpercentile(data_plot, 2)
            vmax = np.nanpercentile(data_plot, 98)
            norm = None

        # Scatter plot
        style = self.config.style
        ms = scatter_size if scatter_size is not None else style.markersize * 1.5

        scatter = ax.scatter(
            time_plot,
            alt_plot,
            c=data_plot,
            s=ms**2,
            cmap=cmap,
            norm=norm,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            alpha=style.alpha,
            edgecolors="none",
            rasterized=True,
        )

        # Add colorbar
        cbar = ax.get_figure().colorbar(scatter, ax=ax)  # type: ignore[union-attr]
        cbar.set_label(cbar_label or show_var.title())

    def _plot_2d_curtain(
        self,
        ax: matplotlib.axes.Axes,
        time_values: np.ndarray,
        alt_values: np.ndarray,
        data_values: np.ndarray,
        cmap: str,
        n_levels: int,
        show_var: str,
        cbar_label: str = "",
    ) -> None:
        """Plot 2D curtain data as contourf.

        Parameters
        ----------
        ax
            Axes to plot on.
        time_values
            Time coordinates.
        alt_values
            Altitude coordinates.
        data_values
            2D data array.
        cmap
            Colormap.
        n_levels
            Number of contour levels.
        show_var
            Which variable is shown.
        cbar_label
            Label for the colorbar.
        """
        import pandas as pd

        # Convert time
        time_numeric = pd.to_datetime(time_values)

        # Create meshgrid if needed
        if alt_values.ndim == 1:
            time_grid, alt_grid = np.meshgrid(
                np.arange(len(time_numeric)),
                alt_values,
            )
            # Transpose data if needed
            if data_values.shape[0] == len(time_numeric):
                data_values = data_values.T
        else:
            time_grid = np.broadcast_to(
                np.arange(len(time_numeric)),
                alt_values.shape,
            )
            alt_grid = alt_values

        # Calculate limits
        data_flat = data_values[np.isfinite(data_values)]
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(data_flat)
        else:
            vmin = np.nanpercentile(data_flat, 2)
            vmax = np.nanpercentile(data_flat, 98)

        levels = np.linspace(vmin, vmax, n_levels)

        # Contourf plot. ContourSet does not support rasterization in the
        # pinned matplotlib (<3.9): both the constructor kwarg and
        # set_rasterized() raise a UserWarning, so this layer stays vector.
        contour = ax.contourf(
            time_grid,
            alt_grid,
            data_values,
            levels=levels,
            cmap=cmap,
            extend="both",
        )

        # Set x-axis to show times
        ax.set_xticks(np.linspace(0, len(time_numeric) - 1, 5).astype(int))
        ax.set_xticklabels(
            [
                time_numeric[int(i)].strftime("%H:%M")
                for i in np.linspace(0, len(time_numeric) - 1, 5)
            ]
        )

        # Add colorbar
        cbar = ax.get_figure().colorbar(contour, ax=ax)  # type: ignore[union-attr]
        cbar.set_label(cbar_label or show_var.title())
