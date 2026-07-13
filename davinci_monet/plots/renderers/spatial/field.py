"""Single-source spatial field renderer for DAVINCI.

Renders ONE source's field on a map, choosing the mark from the source's data
*shape* (point/track/profile/swath/grid) — scatter for point/track/profile,
pcolormesh for grid/swath. No pairing, no x/y: this is the general single-source
spatial plot.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    calculate_symmetric_limits,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
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
_LEVEL_DIMS = ["lev", "level", "z", "altitude", "height", "vertical"]
# Shapes drawn as a filled mesh; everything else is scatter.
_MESH_SHAPES = {"grid", "swath", "regular_grid", "curvilinear_grid"}
# Shapes whose ``time`` dim is the sampling path, not something to average over.
_PATH_SHAPES = {"track", "profile"}
_ColorbarExtend = Literal["neither", "both", "min", "max"]


def _global_area_mean(field: xr.DataArray, latitude_name: str) -> float:
    """Return a finite-cell, latitude-area-weighted mean for a global field."""
    if latitude_name not in field.coords:
        finite = np.asarray(field.values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.mean(finite)) if finite.size else float("nan")

    latitude = field.coords[latitude_name]
    weights = np.cos(np.deg2rad(latitude)).clip(min=0.0)
    valid = field.notnull() & np.isfinite(field)
    numerator = (field.where(valid).fillna(0.0) * weights).sum(skipna=True)
    denominator = weights.where(valid).sum(skipna=True)
    if float(denominator) <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _subtitle_with_global_mean(
    field: xr.DataArray,
    latitude_name: str,
    subtitle: str | None,
    decimals: int | None = None,
) -> str:
    mean = _global_area_mean(field, latitude_name)
    if decimals is not None and decimals < 0:
        raise ValueError("global_mean_decimals must be nonnegative")
    if not np.isfinite(mean):
        mean_text = "Mean: no finite values"
    elif decimals is None:
        mean_text = f"Mean: {mean:.6g}"
    else:
        mean_text = f"Mean: {mean:.{decimals}f}"
    return f"{subtitle} | {mean_text}" if subtitle else mean_text


def _nice_robust_limits(
    finite: np.ndarray,
    robust_pct: Sequence[float],
    bounds: Sequence[float] | None,
    *,
    target_intervals: int = 5,
) -> tuple[float, float]:
    """Round robust data limits outward to compact human-friendly values."""
    if target_intervals < 1:
        raise ValueError("nice_range_target_intervals must be positive")
    percentiles = np.asarray(robust_pct, dtype=float)
    if percentiles.shape != (2,) or not np.all(np.isfinite(percentiles)):
        raise ValueError("robust_pct must contain two finite percentiles")
    low, high = (float(value) for value in np.nanpercentile(finite, percentiles))
    if low > high:
        low, high = high, low
    span = high - low
    if not np.isfinite(span) or span <= 0.0:
        return low, high

    raw_step = span / target_intervals
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    fraction = raw_step / magnitude
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    step = nice_fraction * magnitude
    nice_low = float(np.floor(low / step) * step)
    nice_high = float(np.ceil(high / step) * step)

    if bounds is not None:
        physical_bounds = np.asarray(bounds, dtype=float)
        if physical_bounds.shape != (2,) or not np.all(np.isfinite(physical_bounds)):
            raise ValueError("nice_range_bounds must contain two finite values")
        if physical_bounds[0] > physical_bounds[1]:
            raise ValueError("nice_range_bounds must be increasing")
        nice_low = max(nice_low, float(physical_bounds[0]))
        nice_high = min(nice_high, float(physical_bounds[1]))
    return nice_low, nice_high


@register_plotter("spatial", arity="single_source", category="spatial")
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
    ) -> matplotlib.figure.Figure | list[tuple[str, matplotlib.figure.Figure]]:
        """Render a single source's field on a map (exactly 1 series)."""
        if len(series) != 1:
            raise NotImplementedError(
                f"SpatialPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        split_dim = self._split_dim_for_field(s.dataset, s.var_name, **kwargs)
        if split_dim is not None:
            figures: list[tuple[str, matplotlib.figure.Figure]] = []
            for idx, label in self._iter_split_labels(s.dataset, split_dim):
                subset = s.dataset.isel({split_dim: idx}, drop=True)
                figures.append(
                    (
                        self._format_split_label(label),
                        self._plot(
                            subset,
                            s.var_name,
                            s.source_label,
                            ax=ax,
                            subtitle=self._format_split_label(label),
                            **kwargs,
                        ),
                    )
                )
            return figures
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
        robust: bool = False,
        robust_pct: Sequence[float] = (2.0, 98.0),
        symmetric: bool = False,
        subtitle: str | None = None,
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
        show_coastlines: bool | None = None,
        show_countries: bool | None = None,
        show_states: bool | None = None,
        show_gridlines: bool | None = None,
        gridline_style: str | None = None,
        resolution: str | None = None,
        land_color: str | None = None,
        ocean_color: str | None = None,
        show_global_mean: bool = False,
        global_mean_decimals: int | None = 3,
        nice_range: bool = False,
        nice_range_bounds: Sequence[float] | None = None,
        nice_range_target_intervals: int = 5,
        tight_layout: bool = False,
        tight_layout_pad: float = 0.5,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        import cartopy.crs as ccrs

        field = ds[variable]
        shape = self._resolve_shape(ds, variable, field, lat_var, lon_var)
        map_config = self._map_config_with_overrides(
            show_coastlines=show_coastlines,
            show_countries=show_countries,
            show_states=show_states,
            show_gridlines=show_gridlines,
            gridline_style=gridline_style,
            resolution=resolution,
            land_color=land_color,
            ocean_color=ocean_color,
        )

        # Reduce a vertical dim (grid/profile) to a single level (surface default).
        field = self._reduce_vertical(field, level_index)
        # Collapse time unless it is the sampling path (track/profile).
        field = self._reduce_time(field, shape, time_average, time_index)
        field = self._squeeze_singleton_non_spatial_dims(ds, field, lat_var, lon_var)

        lats, lons, field, resolved_lat, resolved_lon = self._resolve_coords(
            ds, field, lat_var, lon_var
        )
        plot_type = "pcolormesh" if shape in _MESH_SHAPES else "scatter"

        if ax is None:
            fig, ax = self.create_map_figure()
        else:
            maybe_fig = ax.get_figure()
            if maybe_fig is None:
                raise RuntimeError("Axes is not attached to a figure")
            fig = cast("matplotlib.figure.Figure", maybe_fig)
        self.add_map_features(ax, map_config)

        extent = self._resolve_extent(domain_type, domain_name)
        if extent is not None:
            getattr(ax, "set_extent")(extent)

        values = field.values
        finite = values[np.isfinite(values)]
        rendered_subtitle = subtitle if subtitle is not None else self.config.subtitle
        if show_global_mean:
            rendered_subtitle = _subtitle_with_global_mean(
                field,
                resolved_lat,
                rendered_subtitle,
                decimals=global_mean_decimals,
            )
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
            self.set_title(ax, self.config.title, subtitle=rendered_subtitle)
            if tight_layout:
                fig.tight_layout(pad=tight_layout_pad)
            return fig
        norm: mcolors.Normalize | None = None
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
                extend=cast(_ColorbarExtend, extend or "neither"),
            )
            vmin = float(levels_arr[0])
            vmax = float(levels_arr[-1])
        elif vmin is None and vmax is None and (robust or nice_range):
            if nice_range:
                vmin, vmax = _nice_robust_limits(
                    finite,
                    robust_pct,
                    nice_range_bounds,
                    target_intervals=nice_range_target_intervals,
                )
            elif symmetric:
                percentile = float(max(robust_pct)) if robust_pct else 98.0
                vmin, vmax = calculate_symmetric_limits(finite, percentile=percentile)
            else:
                robust_limits = np.asarray(np.nanpercentile(finite, robust_pct), dtype=float)
                vmin = float(robust_limits[0])
                vmax = float(robust_limits[1])
        elif vmin is None:
            vmin = float(np.nanmin(finite))
        if levels is None and vmax is None:
            vmax = float(np.nanmax(finite))
        if (
            levels is None
            and symmetric
            and vmin is not None
            and vmax is not None
            and vmin < 0 < vmax
        ):
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

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
            cmap=cmap_obj if levels is not None else cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            marker_size=ms,
            alpha=a,
            field_dims=tuple(field.dims),
            lat_dim=resolved_lat if resolved_lat in field.dims else None,
            lon_dim=resolved_lon if resolved_lon in field.dims else None,
        )

        units = get_variable_units(ds, variable)
        # Colorbar carries units only; the title carries the quantity (+ source),
        # so the quantity is not repeated in both places.
        colorbar_kwargs: dict[str, Any] = {"extend": extend} if norm is not None and extend else {}
        self.add_colorbar(
            fig,
            mappable,
            ax,
            label=labeling.format_units(units),
            **colorbar_kwargs,
        )

        if self.config.title:
            self.set_title(ax, self.config.title, subtitle=rendered_subtitle)
        else:
            # Source name leads the title, de-duped against the quantity, e.g.
            # "AERONET AOD (500 nm)" / "CESM NO2 Column" (not "… (CESM)").
            title_q = labeling.quantity_label(ds, variable)
            self.set_title(
                ax,
                labeling.sourced_title(source_label, title_q),
                subtitle=rendered_subtitle,
            )
        if tight_layout:
            fig.tight_layout(pad=tight_layout_pad)
        return fig

    # -- helpers ---------------------------------------------------------------

    def _map_config_with_overrides(
        self,
        *,
        show_coastlines: bool | None,
        show_countries: bool | None,
        show_states: bool | None,
        show_gridlines: bool | None,
        gridline_style: str | None,
        resolution: str | None,
        land_color: str | None,
        ocean_color: str | None,
    ) -> MapConfig:
        cfg = self.map_config
        return MapConfig(
            projection=cfg.projection,
            extent=cfg.extent,
            show_states=cfg.show_states if show_states is None else show_states,
            show_countries=cfg.show_countries if show_countries is None else show_countries,
            show_coastlines=cfg.show_coastlines if show_coastlines is None else show_coastlines,
            show_gridlines=cfg.show_gridlines if show_gridlines is None else show_gridlines,
            gridline_style=cfg.gridline_style if gridline_style is None else gridline_style,
            resolution=cfg.resolution if resolution is None else resolution,
            land_color=cfg.land_color if land_color is None else land_color,
            ocean_color=cfg.ocean_color if ocean_color is None else ocean_color,
        )

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

    def _squeeze_singleton_non_spatial_dims(
        self,
        ds: xr.Dataset,
        field: xr.DataArray,
        lat_var: str,
        lon_var: str,
    ) -> xr.DataArray:
        spatial_dims = self._spatial_dims(ds, field, lat_var, lon_var)
        indexers = {
            dim: 0 for dim in field.dims if dim not in spatial_dims and field.sizes.get(dim) == 1
        }
        if not indexers:
            return field
        return field.isel(indexers, drop=True)

    def _split_dim_for_field(
        self,
        ds: xr.Dataset,
        variable: str,
        *,
        time_average: bool = True,
        time_index: int | None = None,
        level_index: int | str = "surface",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        **_: Any,
    ) -> Hashable | None:
        field = ds[variable]
        shape = self._resolve_shape(ds, variable, field, lat_var, lon_var)
        field = self._reduce_vertical(field, level_index)
        field = self._reduce_time(field, shape, time_average, time_index)
        spatial_dims = self._spatial_dims(ds, field, lat_var, lon_var)
        split_dims = [
            dim for dim in field.dims if dim not in spatial_dims and field.sizes.get(dim, 0) > 1
        ]
        if not split_dims:
            return None
        if len(split_dims) > 1:
            dims = ", ".join(str(dim) for dim in split_dims)
            raise NotImplementedError(
                f"SpatialPlotter can split over one non-spatial dimension; got {dims}."
            )
        return split_dims[0]

    def _spatial_dims(
        self,
        ds: xr.Dataset,
        field: xr.DataArray,
        lat_var: str,
        lon_var: str,
    ) -> set[Hashable]:
        spatial_dims: set[Hashable] = set()
        for coord_name in (
            self._resolve_coord_name(ds, _LAT_CANDIDATES, lat_var),
            self._resolve_coord_name(ds, _LON_CANDIDATES, lon_var),
        ):
            if coord_name is None:
                continue
            if coord_name in field.dims:
                spatial_dims.add(coord_name)
            if coord_name in ds:
                spatial_dims.update(dim for dim in ds[coord_name].dims if dim in field.dims)
        return spatial_dims

    @staticmethod
    def _iter_split_labels(ds: xr.Dataset, dim: Hashable) -> list[tuple[int, Any]]:
        size = ds.sizes[dim]
        if dim in ds.coords:
            return list(enumerate(ds.coords[dim].values))
        return [(idx, idx) for idx in range(size)]

    @staticmethod
    def _format_split_label(label: Any) -> str:
        text = str(label)
        text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
        return text or "group"

    def _resolve_coords(
        self,
        ds: xr.Dataset,
        field: xr.DataArray,
        lat_var: str,
        lon_var: str,
    ) -> tuple[np.ndarray, np.ndarray, xr.DataArray, str, str]:
        """Resolve lat/lon arrays, shifting 0..360 lon to -180..180 for cartopy."""
        resolved_lat, resolved_lon, lats, lons = resolve_spatial_coords(ds, lat_var, lon_var)
        return lats, lons, field, resolved_lat, resolved_lon

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
