"""Spatial bias plot renderer for DAVINCI.

This module provides spatial bias plotting functionality for
visualizing the difference between x and y values
on a map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    calculate_symmetric_limits,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    detect_spatial_geometry,
    draw_spatial_field,
    maybe_time_average,
    resolve_spatial_coords,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("spatial_bias", arity="pairwise", category="spatial")
class SpatialBiasPlotter(BaseSpatialPlotter):
    """Plotter for spatial bias maps.

    Creates maps showing the spatial distribution of x-vs-y
    bias, with points colored by bias magnitude.

    Parameters
    ----------
    config
        Plot configuration.
    map_config
        Map-specific configuration.

    Examples
    --------
    >>> plotter = SpatialBiasPlotter()
    >>> fig = plotter.render(build_series(paired_data, "x_o3", "y_o3"))
    """

    name: str = "spatial_bias"
    default_figsize: tuple[float, float] = (8, 5)  # Wide for geographic extent

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a spatial bias map from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Optional GeoAxes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the internal bias rendering logic.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"SpatialBiasPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        lat_var: str = kwargs.pop("lat_var", "latitude")
        lon_var: str = kwargs.pop("lon_var", "longitude")
        time_average: bool = kwargs.pop("time_average", True)
        cmap: str = kwargs.pop("cmap", "RdBu_r")
        marker_size: float | None = kwargs.pop("marker_size", None)
        symmetric_cbar: bool = kwargs.pop("symmetric_cbar", True)
        show_zero_line: bool = kwargs.pop("show_zero_line", True)
        show_site_labels: bool = kwargs.pop("show_site_labels", False)
        site_label_var: str = kwargs.pop("site_label_var", "site_name")
        label_sites: list[str] | None = kwargs.pop("label_sites", None)
        city_labels: dict[str, tuple[float, float]] | None = kwargs.pop("city_labels", None)
        label_fontsize: int | None = kwargs.pop("label_fontsize", None)
        plot_type: str = kwargs.pop("plot_type", "auto")

        import cartopy.crs as ccrs

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_map_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Add map features
        self.add_map_features(ax)

        # Calculate bias
        x_data = paired_data[x_var]
        y_data = paired_data[y_var]
        bias = y_data - x_data

        # Time average if requested (both bias and x_data, to keep them aligned).
        bias = maybe_time_average(bias, time_average)
        x_data = maybe_time_average(x_data, time_average)

        # Resolve coordinates (with 0..360 -> -180..180 lon normalization).
        resolved_lat, resolved_lon, lats, lons = resolve_spatial_coords(
            paired_data, lat_var, lon_var
        )

        # Re-sort the lon axis so pcolormesh gets monotonic coords, reordering
        # the bias field (and x_data) along the lon dim to match. Only needed
        # when the 0..360 -> -180..180 shift left a 1-D lon grid axis
        # non-monotonic; gated on lon being a field dim so coords/data stay paired.
        if (
            lons.ndim == 1
            and lons.size > 1
            and resolved_lon in bias.dims
            and np.any(np.diff(lons) < 0)
        ):
            sort_idx = np.argsort(lons)
            lons = lons[sort_idx]
            bias = bias.isel({resolved_lon: sort_idx})
            x_data = x_data.isel({resolved_lon: sort_idx})

        # Detect data geometry from the *DataArray* dims (not just the numpy
        # arrays), since for point/site datasets lats and lons share a
        # single dim and must not be meshgridded as if they were grid axes.
        lat_da = paired_data[resolved_lat]
        lon_da = paired_data[resolved_lon]
        _geometry = detect_spatial_geometry(lat_da, lon_da, bias)
        is_point_data = _geometry == "point"

        if is_point_data:
            # Point/site data: drop singleton dims (e.g. AirNow y=1 residual) so
            # the field collapses to the site dim; draw_spatial_field broadcasts
            # the per-site lat/lon over any remaining dims.
            bias = bias.squeeze(drop=True)

        bias_values = bias.values
        finite = bias_values.flatten()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
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

        # Calculate color limits
        if symmetric_cbar:
            vmin, vmax = calculate_symmetric_limits(finite)
        else:
            vmin = self.config.vmin if self.config.vmin is not None else float(np.nanmin(finite))
            vmax = self.config.vmax if self.config.vmax is not None else float(np.nanmax(finite))

        # Override with config if set
        if self.config.vmin is not None:
            vmin = self.config.vmin
        if self.config.vmax is not None:
            vmax = self.config.vmax

        # Create normalization
        if symmetric_cbar and vmin < 0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = None

        # Get marker size
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize * 2

        # Resolve "auto" to a concrete method based on data geometry: gridded
        # data (1-D lat/lon axes with a 2-D+ field, or 2-D curvilinear coords)
        # renders as a filled pcolormesh field; point/site data uses scatter.
        effective_plot_type = plot_type
        if plot_type == "auto":
            effective_plot_type = (
                "pcolormesh" if _geometry in ("regular_grid", "curvilinear_grid") else "scatter"
            )

        # Draw the bias field via the shared primitive. A TwoSlopeNorm (when the
        # symmetric range straddles zero) is applied to the mappable afterwards,
        # since draw_spatial_field takes vmin/vmax rather than a norm.
        scatter = draw_spatial_field(
            ax,
            bias_values,
            lats,
            lons,
            plot_type=effective_plot_type,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker_size=ms,
            alpha=style.alpha,
        )
        if norm is not None:
            scatter.set_norm(norm)

        # Add colorbar
        units = get_variable_units(paired_data, x_var)
        y_src = paired_data[y_var].attrs.get("source_label") or ""
        x_src = paired_data[x_var].attrs.get("source_label") or ""
        label = labeling.bias_label(
            y_src, x_src, units, quantity=labeling.quantity_label(paired_data, x_var)
        )
        self.add_colorbar(fig, scatter, ax, label=label)

        # Use config site_label size if not specified
        if label_fontsize is None:
            label_fontsize = self.config.text.site_label  # type: ignore[assignment]

        # Add site labels if requested
        if show_site_labels and site_label_var in paired_data.coords:
            site_labels = paired_data[site_label_var].values
            # Recover flattened, NaN-pruned point coords (site labels are
            # point-data only, so the per-site lat/lon broadcast over the field).
            if lats.ndim < bias_values.ndim:
                lats_flat = np.broadcast_to(lats, bias_values.shape).flatten()
                lons_flat = np.broadcast_to(lons, bias_values.shape).flatten()
            else:
                lats_flat = lats.flatten()
                lons_flat = lons.flatten()
            finite_mask = np.isfinite(bias_values.flatten())
            lats_flat = lats_flat[finite_mask]
            lons_flat = lons_flat[finite_mask]
            # Get unique site locations (after time averaging, each site has one point)
            unique_lons, unique_idx = np.unique(lons_flat, return_index=True)
            for i, idx in enumerate(unique_idx):
                site_idx = idx % len(site_labels) if len(site_labels) > 0 else 0
                if site_idx < len(site_labels):
                    site_name = str(site_labels[site_idx])
                    # Filter to specific sites if label_sites is provided
                    if label_sites is not None and site_name not in label_sites:
                        continue
                    ax.annotate(
                        site_name,
                        (lons_flat[idx], lats_flat[idx]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=label_fontsize,
                        alpha=0.8,
                        transform=ccrs.PlateCarree(),
                    )

        # Add city labels if provided
        if city_labels:
            for city_name, (lat, lon) in city_labels.items():
                ax.annotate(
                    city_name,
                    (lon, lat),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=label_fontsize,
                    fontweight="bold",
                    alpha=0.9,
                    transform=ccrs.PlateCarree(),
                )
                # Add a small marker for the city location
                ax.plot(
                    lon,
                    lat,
                    marker="*",
                    markersize=6,
                    color="black",
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )

        # Title
        if self.config.title:
            self.set_title(ax, self.config.title)
        else:
            quantity = get_variable_label(paired_data, x_var, include_prefix=False)
            self.set_title(ax, labeling.title_text(quantity, operation="Bias"))

        return fig
