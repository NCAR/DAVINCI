"""Spatial overlay plot renderer for DAVINCI.

This module provides overlay plotting functionality for
displaying gridded y-field contours with x point overlays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.base import PlotSeries
from davinci_monet.core.coordinates import vertical_dim_name
from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    maybe_time_average,
    resolve_spatial_coords,
    surface_level_index,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

_Y_LAT_CANDIDATES = ("lat", "latitude", "y")
_Y_LON_CANDIDATES = ("lon", "longitude", "x")


@register_plotter("spatial_overlay", arity="pairwise", category="spatial")
class SpatialOverlayPlotter(BaseSpatialPlotter):
    """Plotter for x-vs-y spatial overlays.

    Creates maps showing y fields as filled contours with
    x values overlaid as scatter points.

    Parameters
    ----------
    config
        Plot configuration.
    map_config
        Map-specific configuration.

    Examples
    --------
    >>> plotter = SpatialOverlayPlotter()
    >>> fig = plotter.render(
    ...     build_series(paired_data, "x_o3", "y_o3"),
    ...     y_field=y_data["o3"],
    ... )
    """

    name: str = "spatial_overlay"
    default_figsize: tuple[float, float] = (8, 5)  # Wide for geographic extent

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a spatial overlay from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Optional GeoAxes to plot on.
        **kwargs
            Forwarded kwargs; renderer-specific ones:
            y_field (xr.DataArray|None, default None),
            lat_var (str, default "latitude"),
            lon_var (str, default "longitude"),
            y_lat (str, default "lat"),
            y_lon (str, default "lon"),
            time_index (int, default 0),
            level_index (int|str|None, default "surface"; "surface" auto-detects
              the surface level — last index for CESM-style ascending-pressure
              coords, else first — an int selects that index, None skips slicing),
            cmap (str, default "viridis"),
            n_levels (int, default 15),
            marker_size (float|None, default None),
            x_edgecolor (str, default "black"),
            x_linewidth (float, default 0.5).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"SpatialOverlayPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        y_field: xr.DataArray | None = kwargs.pop("y_field", None)
        lat_var: str = kwargs.pop("lat_var", "latitude")
        lon_var: str = kwargs.pop("lon_var", "longitude")
        y_lat: str = kwargs.pop("y_lat", "lat")
        y_lon: str = kwargs.pop("y_lon", "lon")
        time_index: int = kwargs.pop("time_index", 0)
        level_index: int | str | None = kwargs.pop("level_index", "surface")
        cmap: str = kwargs.pop("cmap", "viridis")
        n_levels: int = kwargs.pop("n_levels", 15)
        marker_size: float | None = kwargs.pop("marker_size", None)
        x_edgecolor: str = kwargs.pop("x_edgecolor", "black")
        x_linewidth: float = kwargs.pop("x_linewidth", 0.5)

        import cartopy.crs as ccrs

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_map_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Add map features
        self.add_map_features(ax)

        # Get y field for contouring
        if y_field is None:
            if y_var in paired_data:
                y_field = paired_data[y_var]
            else:
                raise ValueError(f"No y field provided and {y_var} not in paired_data")

        # Select time/level slice if needed
        if "time" in y_field.dims and y_field.dims.index("time") >= 0:
            if len(y_field.time) > 1:
                y_field = y_field.isel(time=time_index)

        if level_index is not None:
            dim = vertical_dim_name(y_field)
            if dim is not None:
                idx = surface_level_index(y_field, dim) if level_index == "surface" else level_index
                y_field = y_field.isel({dim: idx})

        y_lats, y_lons = _resolve_y_field_coords(y_field, y_lat, y_lon)

        # Get value limits
        all_values = np.concatenate(
            [
                y_field.values.flatten(),
                paired_data[x_var].values.flatten(),
            ]
        )
        all_values = all_values[np.isfinite(all_values)]

        vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(all_values)
        vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(all_values)

        # Create contour levels
        levels = np.linspace(vmin, vmax, n_levels)

        # Plot y field as filled contours
        if y_lats.ndim == 1:
            # Regular grid
            lon_grid, lat_grid = np.meshgrid(y_lons, y_lats)
        else:
            # Already 2D
            lon_grid, lat_grid = y_lons, y_lats

        # ContourSet does not support rasterization in the pinned matplotlib
        # (<3.9): both the constructor kwarg and set_rasterized() raise a
        # UserWarning, so this filled layer stays vector.
        contour = ax.contourf(
            lon_grid,
            lat_grid,
            y_field.values,
            levels=levels,
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
        )

        # Get dataset data
        x_data = paired_data[x_var]
        x_data = maybe_time_average(x_data)

        # Get dataset coordinates (with 0..360 -> -180..180 lon normalization).
        _, _, x_lats, x_lons = resolve_spatial_coords(paired_data, lat_var, lon_var)

        x_values = x_data.values.flatten()
        x_lats_flat = (
            np.broadcast_to(x_lats, x_data.shape).flatten()
            if x_lats.ndim < x_data.ndim
            else x_lats.flatten()
        )
        x_lons_flat = (
            np.broadcast_to(x_lons, x_data.shape).flatten()
            if x_lons.ndim < x_data.ndim
            else x_lons.flatten()
        )

        # Remove NaN values
        mask = np.isfinite(x_values)
        x_values = x_values[mask]
        x_lats_flat = x_lats_flat[mask]
        x_lons_flat = x_lons_flat[mask]

        # Get marker size
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize * 2

        # Overlay x scatter
        scatter = ax.scatter(
            x_lons_flat,
            x_lats_flat,
            c=x_values,
            s=ms**2,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            edgecolors=x_edgecolor,
            linewidths=x_linewidth,
            zorder=5,
            rasterized=True,
        )

        # Add colorbar. The contour and scatter share this scale, so the label
        # is the chemistry variable itself.
        units = get_variable_units(paired_data, x_var)
        label = labeling.axis_label(
            labeling.quantity_label(paired_data, x_var),
            units,
        )
        self.add_colorbar(fig, contour, ax, label=label)

        # Title — terse quantity only, no "vs", no contour/points roles.
        if self.config.title:
            self.set_title(ax, self.config.title)
        else:
            self.set_title(ax, labeling.title_text(labeling.quantity_label(paired_data, x_var)))

        return fig


def _resolve_y_field_coords(
    y_field: xr.DataArray,
    preferred_lat: str,
    preferred_lon: str,
) -> tuple[np.ndarray, np.ndarray]:
    lat_name = _first_coord_name(y_field, (preferred_lat, *_Y_LAT_CANDIDATES))
    lon_name = _first_coord_name(y_field, (preferred_lon, *_Y_LON_CANDIDATES))
    if lat_name is None or lon_name is None:
        missing = []
        if lat_name is None:
            missing.append("latitude")
        if lon_name is None:
            missing.append("longitude")
        available = ", ".join(str(name) for name in y_field.coords) or "none"
        raise ValueError(
            "Spatial overlay y_field requires "
            f"{' and '.join(missing)} coordinate(s); available coords: {available}"
        )
    return y_field[lat_name].values, y_field[lon_name].values


def _first_coord_name(y_field: xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in y_field.coords:
            return name
    return None
