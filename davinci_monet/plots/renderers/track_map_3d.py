"""3D track map renderer for DAVINCI.

This module provides 3D visualization of aircraft flight tracks,
showing longitude, latitude, and altitude with color-coded values.
Includes continent outlines and city markers on the surface plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    BasePlotter,
    build_series,
    calculate_symmetric_limits,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter

# Shared 3D-track drawing mechanics live in ``_track3d``.  The geometry helpers
# are re-exported here so existing imports
# (``from ...renderers.track_map_3d import _get_coastline_segments`` etc.)
# keep working.
from davinci_monet.plots.renderers._track3d import (
    _get_border_segments,
    _get_coastline_segments,
    _get_land_polygons,
    _render_surface_map,
    draw_track_3d,
    resolve_track_coords,
)
from davinci_monet.plots.titles import title_for_labeled_subset

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

__all__ = [
    "TrackMap3DPlotter",
    "_get_coastline_segments",
    "_get_land_polygons",
    "_get_border_segments",
    "_render_surface_map",
]


@register_plotter("track_map_3d", arity="pairwise", category="specialized")
class TrackMap3DPlotter(BasePlotter):
    """Plotter for 3D flight track visualization.

    Creates 3D plots showing aircraft trajectory with:
    - X-axis: Longitude
    - Y-axis: Latitude
    - Z-axis: Altitude
    - Color: Variable value (x, y, or bias)

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = TrackMap3DPlotter()
    >>> fig = plotter.render(
    ...     build_series(paired_data, "x_O3", "y_O3"),
    ...     show_var="bias",
    ... )
    """

    name: str = "track_map_3d"
    default_figsize: tuple[float, float] = (7, 6)  # Near-square for 3D viewing

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure | list[tuple[str, matplotlib.figure.Figure]]:
        """Render a 3D track map from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Ignored (creates new 3D axes).
        **kwargs
            Forwarded kwargs; renderer-specific ones:
            alt_var (str, default "altitude"),
            lat_var (str, default "latitude"),
            lon_var (str, default "longitude"),
            show_var (str, default "bias"),
            cmap (str|None, default None),
            marker_size (float, default 20),
            alpha (float, default 0.7),
            elev (float, default 25),
            azim (float, default -60),
            show_projection (bool, default True),
            projection_alpha (float, default 0.3),
            alt_scale (float, default 0.001),
            show_coastlines (bool, default True),
            coastline_color (str, default "black"),
            coastline_alpha (float, default 1.0),
            coastline_linewidth (float, default 0.8),
            coastline_scale (str, default "10m"),
            show_borders (bool, default False),
            border_color (str, default "gray"),
            border_alpha (float, default 0.5),
            border_linewidth (float, default 0.5),
            city_labels (dict|None, default None),
            city_marker_size (float, default 50),
            city_marker_color (str, default "red"),
            city_font_size (float, default 10),
            show_surface_map (bool, default False),
            surface_map_resolution (int, default 250),
            land_color (str, default "#E8E8E8"),
            ocean_color (str, default "#D4E9F7").

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"TrackMap3DPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

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

        alt_var: str = kwargs.pop("alt_var", "altitude")
        lat_var: str = kwargs.pop("lat_var", "latitude")
        lon_var: str = kwargs.pop("lon_var", "longitude")
        show_var: Literal["x", "y", "bias"] = kwargs.pop("show_var", "bias")
        cmap: str | None = kwargs.pop("cmap", None)
        marker_size: float = kwargs.pop("marker_size", 20)
        alpha: float = kwargs.pop("alpha", 0.7)
        elev: float = kwargs.pop("elev", 25)
        azim: float = kwargs.pop("azim", -60)
        show_projection: bool = kwargs.pop("show_projection", True)
        projection_alpha: float = kwargs.pop("projection_alpha", 0.3)
        alt_scale: float = kwargs.pop("alt_scale", 0.001)
        show_coastlines: bool = kwargs.pop("show_coastlines", True)
        coastline_color: str = kwargs.pop("coastline_color", "black")
        coastline_alpha: float = kwargs.pop("coastline_alpha", 1.0)
        coastline_linewidth: float = kwargs.pop("coastline_linewidth", 0.8)
        coastline_scale: str = kwargs.pop("coastline_scale", "10m")
        show_borders: bool = kwargs.pop("show_borders", False)
        border_color: str = kwargs.pop("border_color", "gray")
        border_alpha: float = kwargs.pop("border_alpha", 0.5)
        border_linewidth: float = kwargs.pop("border_linewidth", 0.5)
        city_labels: dict[str, list[float]] | None = kwargs.pop("city_labels", None)
        city_marker_size: float = kwargs.pop("city_marker_size", 50)
        city_marker_color: str = kwargs.pop("city_marker_color", "red")
        city_font_size: float = kwargs.pop("city_font_size", 10)
        show_surface_map: bool = kwargs.pop("show_surface_map", False)
        surface_map_resolution: int = kwargs.pop("surface_map_resolution", 250)
        land_color: str = kwargs.pop("land_color", "#E8E8E8")
        ocean_color: str = kwargs.pop("ocean_color", "#D4E9F7")

        # Validate coordinate presence (helper would raise a bare KeyError).
        if lat_var not in paired_data.coords and lat_var not in paired_data.data_vars:
            raise ValueError(f"Latitude variable '{lat_var}' not found")
        if lon_var not in paired_data.coords and lon_var not in paired_data.data_vars:
            raise ValueError(f"Longitude variable '{lon_var}' not found")
        if alt_var not in paired_data.coords and alt_var not in paired_data.data_vars:
            raise ValueError(f"Altitude variable '{alt_var}' not found")

        # Calculate what to show (derived values colored on the track)
        y_src = paired_data[y_var].attrs.get("source_label") or ""
        x_src = paired_data[x_var].attrs.get("source_label") or ""
        if show_var == "x":
            values = paired_data[x_var].values
            default_cmap = "viridis"
            # Colorbar carries units only; the title carries the quantity.
            label = labeling.format_units(get_variable_units(paired_data, x_var))
        elif show_var == "y":
            values = paired_data[y_var].values
            default_cmap = "viridis"
            label = labeling.format_units(get_variable_units(paired_data, y_var))
        else:  # bias
            values = paired_data[y_var].values - paired_data[x_var].values
            default_cmap = "RdBu_r"
            label = labeling.bias_label(
                y_src,
                x_src,
                get_variable_units(paired_data, x_var),
                quantity=labeling.quantity_label(paired_data, x_var),
            )

        cmap = cmap or default_cmap

        # Extract finite-filtered coordinates and values (alt scaled m -> km).
        # The colored ``values`` array is derived, so attach it under a temp name
        # so the shared helper applies the same joint finite mask.
        _tmp_values = "__track_map_3d_values__"
        lons, lats, alts, values = resolve_track_coords(
            paired_data.assign({_tmp_values: (paired_data[x_var].dims, values)}),
            _tmp_values,
            lat_var=lat_var,
            lon_var=lon_var,
            alt_var=alt_var,
            alt_scale=alt_scale,
        )

        if len(values) == 0:
            raise ValueError("No valid data points for 3D track plot")

        # Create figure with 3D axes
        fig = plt.figure(figsize=self.config.figure.figsize)
        ax3d = fig.add_subplot(111, projection="3d")

        # Text settings - use absolute point sizes from config
        text_cfg = self.config.text

        # Set color limits
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(values)
        else:
            vmin = self.config.vmin if self.config.vmin is not None else float(np.nanmin(values))
            vmax = self.config.vmax if self.config.vmax is not None else float(np.nanmax(values))

        # Colorbar label (label already built above with units; apply subscript formatting)
        cbar_label = format_plot_title(label)

        # Draw the shared 3D track body (scatter, projection, surface-plane map
        # features, city markers, axis setup, labels, colorbar).
        draw_track_3d(
            fig,
            ax3d,
            lons,
            lats,
            alts,
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            text_cfg=text_cfg,
            cbar_label=cbar_label,
            marker_size=marker_size,
            alpha=alpha,
            elev=elev,
            azim=azim,
            show_projection=show_projection,
            projection_alpha=projection_alpha,
            show_coastlines=show_coastlines,
            coastline_color=coastline_color,
            coastline_alpha=coastline_alpha,
            coastline_linewidth=coastline_linewidth,
            coastline_scale=coastline_scale,
            show_borders=show_borders,
            border_color=border_color,
            border_alpha=border_alpha,
            border_linewidth=border_linewidth,
            show_surface_map=show_surface_map,
            surface_map_resolution=surface_map_resolution,
            land_color=land_color,
            ocean_color=ocean_color,
            city_labels=city_labels,
            city_marker_size=city_marker_size,
            city_marker_color=city_marker_color,
            city_font_size=city_font_size,
        )

        # Title - centered above plot
        if self.config.title:
            self.set_figure_title(fig, self.config.title, y=0.85)

        plt.tight_layout(rect=(0, 0, 1, 0.95))  # Leave room at top for title
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
        """Render one labeled 3D track map per flight."""
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

            # Update title/subtitle to identify this flight.
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
            except ValueError:
                self.config.title = original_title
                self.config.subtitle = original_subtitle
                continue
            finally:
                self.config.title = original_title
                self.config.subtitle = original_subtitle

            flight_id = flight_str.replace("-", "")
            if isinstance(fig, list):
                results.extend((f"{flight_id}_{label}", subfig) for label, subfig in fig)
            else:
                results.append((flight_id, fig))

        return results
