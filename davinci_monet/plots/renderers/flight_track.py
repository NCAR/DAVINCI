"""3D flight track map renderer for single-source geometry data.

Renders a 3D plot showing an aircraft flight path with longitude, latitude,
and altitude axes, colored by a variable value (e.g., O3 concentration).
Includes coastline outlines on the surface plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    BasePlotter,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers._track3d import draw_track_3d, resolve_track_coords
from davinci_monet.plots.style import get_sequential_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

    from davinci_monet.core.base import PlotSeries


@register_plotter("flight_track", arity="single_source", category="specialized")
class FlightTrackPlotter(BasePlotter):
    """Plotter for 3D flight track maps colored by variable value.

    Creates a 3D plot with the flight path rendered as scatter points
    colored by the selected variable, with coastlines on the surface plane.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = FlightTrackPlotter()
    >>> fig = plotter.render(build_series(x_data, "O3"), title="DC3 Flight O3")
    """

    name: str = "flight_track"
    default_figsize: tuple[float, float] = (7, 6)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> Any:
        """Unified entry: render a single source's flight track."""
        if len(series) != 1:
            raise NotImplementedError(
                f"FlightTrackPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        return self._plot(s.dataset, s.var_name, **kwargs)

    def _plot(
        self,
        x_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        title: str | None = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        marker_size: float = 20.0,
        alpha: float = 0.7,
        lat_coord: str = "latitude",
        lon_coord: str = "longitude",
        alt_coord: str = "altitude",
        alt_scale: float = 0.001,
        elev: float = 25,
        azim: float = -60,
        show_projection: bool = True,
        projection_alpha: float = 0.3,
        show_coastlines: bool = True,
        coastline_color: str = "black",
        coastline_alpha: float = 1.0,
        coastline_linewidth: float = 0.8,
        coastline_scale: str = "10m",
        show_borders: bool = False,
        border_color: str = "gray",
        border_alpha: float = 0.5,
        border_linewidth: float = 0.5,
        city_labels: dict[str, list[float]] | None = None,
        city_marker_size: float = 50,
        city_marker_color: str = "red",
        city_font_size: float = 10,
        show_surface_map: bool = False,
        surface_map_resolution: int = 250,
        land_color: str = "#E8E8E8",
        ocean_color: str = "#D4E9F7",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a 3D flight track map.

        Parameters
        ----------
        x_data
            Dataset dataset with lat/lon/alt coordinates and the variable.
        variable
            Name of the variable to color by.
        ax
            Ignored (always creates new 3D axes).
        title
            Plot title. Defaults to "{variable} Flight Track".
        cmap
            Colormap name. Defaults to the sequential NCAR colormap.
        vmin, vmax
            Colorbar limits. If None, auto-determined from data.
        marker_size
            Size of scatter markers.
        alpha
            Transparency of markers.
        lat_coord, lon_coord
            Names of latitude/longitude coordinates.
        alt_coord
            Name of the altitude coordinate.
        alt_scale
            Scale factor for altitude (e.g., 0.001 to convert m to km).
        elev
            Elevation angle for 3D view.
        azim
            Azimuth angle for 3D view.
        show_projection
            If True, show 2D projection on the bottom (z=0) plane.
        projection_alpha
            Transparency of projection markers.
        show_coastlines
            If True, draw continent outlines on the surface plane.
        coastline_color
            Color for coastline lines.
        coastline_alpha
            Transparency of coastlines.
        coastline_linewidth
            Line width for coastlines.
        coastline_scale
            Natural Earth scale for coastlines ('110m', '50m', or '10m').
        show_borders
            If True, draw country borders on surface plane.
        border_color
            Color for border lines.
        border_alpha
            Transparency of borders.
        border_linewidth
            Line width for borders.
        city_labels
            Dictionary of city names to [lat, lon] coordinates.
            Cities will be plotted as markers on the surface plane.
        city_marker_size
            Size of city markers.
        city_marker_color
            Color for city markers.
        city_font_size
            Font size for city labels.
        show_surface_map
            If True, render a filled map image on the z=0 plane.
        surface_map_resolution
            Resolution of surface map image in pixels.
        land_color
            Color for land areas on surface map.
        ocean_color
            Color for ocean areas on surface map.
        **kwargs
            Additional arguments passed to ax.scatter.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        # Validate altitude presence (helper would raise a bare KeyError).
        if alt_coord not in x_data.coords and alt_coord not in x_data.data_vars:
            raise ValueError(
                f"Altitude coordinate '{alt_coord}' not found. "
                f"Available: {list(x_data.coords) + list(x_data.data_vars)}"
            )

        # Extract finite-filtered coordinates and values (alt scaled m -> km).
        lons, lats, alts, values = resolve_track_coords(
            x_data,
            variable,
            lat_var=lat_coord,
            lon_var=lon_coord,
            alt_var=alt_coord,
            alt_scale=alt_scale,
        )

        if len(values) == 0:
            raise ValueError("No valid data points for 3D track plot")

        # Determine colorbar limits
        if vmin is None:
            vmin = float(np.nanmin(values))
        if vmax is None:
            vmax = float(np.nanmax(values))

        # Create figure with 3D axes
        fig = plt.figure(
            figsize=self.config.figure.figsize,
            dpi=self.config.figure.dpi,
        )
        ax3d = fig.add_subplot(111, projection="3d")

        text_cfg = self.config.text
        cmap = cmap or get_sequential_cmap()

        # Colorbar carries units only; the title (below) carries the quantity.
        var_label = get_variable_label(x_data, variable, include_prefix=False)
        units = get_variable_units(x_data, variable)
        cbar_label = labeling.format_units(units)

        # Draw the shared 3D track body (scatter, projection, surface-plane map
        # features, axis setup, labels, colorbar). ``use_maxnlocator`` matches
        # the geometry-only flight-track convention of capping x/y tick counts.
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
            city_labels=city_labels,
            city_marker_size=city_marker_size,
            city_marker_color=city_marker_color,
            city_font_size=city_font_size,
            show_surface_map=show_surface_map,
            surface_map_resolution=surface_map_resolution,
            land_color=land_color,
            ocean_color=ocean_color,
            use_maxnlocator=True,
        )

        # Title
        if title is None:
            title = labeling.title_text(var_label)
        self.set_figure_title(fig, title, y=0.85)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        return fig
