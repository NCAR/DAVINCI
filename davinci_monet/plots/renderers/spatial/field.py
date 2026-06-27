"""Single-source spatial field renderer for DAVINCI.

Renders ONE source's field on a map, choosing the mark from the source's data
*shape* (point/track/profile/swath/grid) — scatter for point/track/profile,
pcolormesh for grid/swath. No pairing, no x/y: this is the general single-source
spatial plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    detect_spatial_geometry,
    draw_spatial_field,
    get_domain_extent,
    resolve_spatial_coords,
    surface_level_index,
)
from davinci_monet.plots.style import geosit_aod_levels, get_geosit_aod_cmap, get_sequential_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

    from davinci_monet.core.base import PlotSeries

# Lat/lon coordinate name candidates (resolved in order).
_LAT_CANDIDATES = ["latitude", "lat", "LAT", "Latitude"]
_LON_CANDIDATES = ["longitude", "lon", "LON", "Longitude"]
# Vertical dimension name candidates to reduce for 3-D fields.
_LEVEL_DIMS = ["lev", "level", "z", "vertical"]
# Shapes drawn as a filled mesh; everything else is scatter.
_MESH_SHAPES = {"grid", "swath", "regular_grid", "curvilinear_grid"}
# Shapes whose ``time`` dim is the sampling path, not something to average over.
_PATH_SHAPES = {"track", "profile"}


@register_plotter("spatial")
class SpatialPlotter(BaseSpatialPlotter):
    """Single-source spatial field map (shape-aware).

    Renders a single source's variable on a map, dispatching on the source's
    geometry shape: point/track/profile → scatter, grid/swath → pcolormesh.
    Drives off the source's ``geometry`` attribute (set by readers), falling
    back to coordinate-based detection.

    Examples
    --------
    >>> plotter = SpatialPlotter()
    >>> fig = plotter.render(build_series(cam_ds, "O3"))
    """

    name: str = "spatial"
    default_figsize: tuple[float, float] = (8, 5)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a single source's field on a map (exactly 1 series)."""
        if len(series) != 1:
            raise NotImplementedError(
                f"SpatialPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        return self._plot(s.dataset, s.var_name, s.source_label, ax=ax, **kwargs)

    def _plot(
        self,
        ds: xr.Dataset,
        variable: str,
        source_label: str | None = None,
        ax: matplotlib.axes.Axes | None = None,
        *,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        levels: Sequence[float] | np.ndarray | None = None,
        extend: str | None = None,
        style_preset: str | None = None,
        marker_size: float | None = None,
        alpha: float | None = None,
        time_average: bool = True,
        time_index: int | None = None,
        level_index: int | str = "surface",
        connect_track: bool = True,
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        domain_type: str | list[str] | None = None,
        domain_name: str | list[str] | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        import cartopy.crs as ccrs

        field = ds[variable]
        shape = self._resolve_shape(ds, variable, field, lat_var, lon_var)

        # Reduce a vertical dim (grid/profile) to a single level (surface default).
        field = self._reduce_vertical(field, level_index)
        # Collapse time unless it is the sampling path (track/profile).
        field = self._reduce_time(field, shape, time_average, time_index)

        lats, lons, field = self._resolve_coords(ds, field, lat_var, lon_var)
        plot_type = "pcolormesh" if shape in _MESH_SHAPES else "scatter"

        if ax is None:
            fig, ax = self.create_map_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]
        self.add_map_features(ax)

        extent = self._resolve_extent(domain_type, domain_name)
        if extent is not None:
            ax.set_extent(extent)  # type: ignore[attr-defined]

        values = field.values
        finite = values[np.isfinite(values)]
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
        norm = None
        if style_preset is not None:
            if style_preset != "geosit_aod":
                raise ValueError(f"Unknown spatial style_preset: {style_preset}")
            if levels is None:
                levels = geosit_aod_levels(finite)
            if cmap is None:
                cmap = get_geosit_aod_cmap()
            if extend is None:
                extend = "max"

        if levels is not None:
            levels_arr = np.asarray(levels, dtype=float)
            if (
                levels_arr.ndim != 1
                or levels_arr.size < 2
                or not np.all(np.isfinite(levels_arr))
                or not np.all(np.diff(levels_arr) > 0)
            ):
                raise ValueError("levels must be a 1-D sequence of at least two increasing values")
            cmap_obj = plt.get_cmap(cmap or get_sequential_cmap())
            norm = mcolors.BoundaryNorm(
                levels_arr,
                ncolors=cmap_obj.N,
                extend=extend or "neither",
            )
            cmap = cmap_obj
            vmin = float(levels_arr[0])
            vmax = float(levels_arr[-1])
        elif vmin is None:
            vmin = float(np.nanmin(finite))
        if levels is None and vmax is None:
            vmax = float(np.nanmax(finite))

        cmap = cmap or get_sequential_cmap()
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize * 2
        a = alpha if alpha is not None else style.alpha

        # For tracks, draw a faint connecting line beneath the colored points so
        # the trajectory reads as a path.
        if shape == "track" and connect_track and lats.ndim == 1 and lons.ndim == 1:
            ax.plot(
                lons,
                lats,
                color="#999999",
                linewidth=0.6,
                alpha=0.6,
                zorder=1,
                transform=ccrs.PlateCarree(),
            )

        mappable = draw_spatial_field(
            ax,
            values,
            lats,
            lons,
            plot_type=plot_type,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            marker_size=ms,
            alpha=a,
        )

        units = get_variable_units(ds, variable)
        # Colorbar carries units only; the title carries the quantity (+ source),
        # so the quantity is not repeated in both places.
        colorbar_kwargs = {"extend": extend} if norm is not None and extend else {}
        self.add_colorbar(
            fig,
            mappable,
            ax,
            label=labeling.format_units(units),
            **colorbar_kwargs,
        )

        if self.config.title:
            self.set_title(ax, self.config.title)
        else:
            # Source name leads the title, de-duped against the quantity, e.g.
            # "AERONET AOD (500 nm)" / "CESM NO2 Column" (not "… (CESM)").
            title_q = labeling.quantity_label(ds, variable)
            self.set_title(ax, labeling.sourced_title(source_label, title_q))
        return fig

    # -- helpers ---------------------------------------------------------------

    def _resolve_shape(
        self,
        ds: xr.Dataset,
        variable: str,
        field: xr.DataArray,
        lat_var: str,
        lon_var: str,
    ) -> str:
        """Resolve the source's data shape.

        Prefers the authoritative ``geometry`` attr (set by readers); falls back
        to coordinate-based detection mapped onto the shape vocabulary.
        """
        declared = str(ds.attrs.get("geometry", "")).lower()
        if declared in {"point", "track", "profile", "swath", "grid"}:
            return declared
        # Fallback: detect from coordinate dimensionality.
        resolved_lat = self._resolve_coord_name(ds, _LAT_CANDIDATES, lat_var)
        resolved_lon = self._resolve_coord_name(ds, _LON_CANDIDATES, lon_var)
        if resolved_lat is None or resolved_lon is None:
            return "grid" if field.ndim >= 2 else "point"
        geom = detect_spatial_geometry(ds[resolved_lat], ds[resolved_lon], field)
        return {"regular_grid": "grid", "curvilinear_grid": "swath"}.get(geom, "point")

    def _reduce_vertical(self, field: xr.DataArray, level_index: int | str) -> xr.DataArray:
        """Slice a vertical dim to a single level (surface by default)."""
        for dim in _LEVEL_DIMS:
            if dim in field.dims:
                idx = (
                    surface_level_index(field, dim)
                    if level_index == "surface"
                    else int(level_index)
                )
                return field.isel({dim: idx})
        return field

    def _reduce_time(
        self,
        field: xr.DataArray,
        shape: str,
        time_average: bool,
        time_index: int | None,
    ) -> xr.DataArray:
        """Collapse the time dim unless it is the sampling path (track/profile)."""
        if shape in _PATH_SHAPES or "time" not in field.dims:
            return field
        if time_index is not None:
            return field.isel(time=time_index)
        if time_average:
            return field.mean(dim="time")
        return field

    def _resolve_coords(
        self,
        ds: xr.Dataset,
        field: xr.DataArray,
        lat_var: str,
        lon_var: str,
    ) -> tuple[np.ndarray, np.ndarray, xr.DataArray]:
        """Resolve lat/lon arrays, shifting 0..360 lon to -180..180 for cartopy."""
        _, resolved_lon, lats, lons = resolve_spatial_coords(ds, lat_var, lon_var)

        # A 1-D lon axis that the 0..360 -> -180..180 shift left non-monotonic
        # must be re-sorted, reordering the field along the lon dim to match so
        # pcolormesh receives ascending coords. Only do this when lon is a field
        # dim (grid axis) so coords and data stay paired.
        if (
            lons.ndim == 1
            and lons.size > 1
            and resolved_lon in field.dims
            and np.any(np.diff(lons) < 0)
        ):
            sort_idx = np.argsort(lons)
            lons = lons[sort_idx]
            field = field.isel({resolved_lon: sort_idx})

        return lats, lons, field

    @staticmethod
    def _resolve_coord_name(ds: xr.Dataset, candidates: list[str], preferred: str) -> str | None:
        for name in [preferred, *candidates]:
            if name in ds.coords or name in ds:
                return name
        return None

    def _resolve_extent(
        self,
        domain_type: str | list[str] | None,
        domain_name: str | list[str] | None,
    ) -> tuple[float, float, float, float] | None:
        if self.map_config.extent is not None:
            return self.map_config.extent
        if not domain_type:
            return None
        dtype = domain_type[0] if isinstance(domain_type, list) else domain_type
        dname = (
            (domain_name[0] if isinstance(domain_name, list) else domain_name)
            if domain_name
            else None
        )
        if dtype in (None, "all"):
            return None
        return get_domain_extent(dtype, dname)
