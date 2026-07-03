"""Base classes and utilities for spatial plotting.

This module provides common functionality for map-based plots
using cartopy projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.coordinates import surface_index
from davinci_monet.plots.base import BasePlotter, PlotConfig

if TYPE_CHECKING:
    import cartopy.crs
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


def detect_spatial_geometry(
    lat_da: xr.DataArray,
    lon_da: xr.DataArray,
    field_da: xr.DataArray,
) -> str:
    """Classify the spatial geometry of a paired dataset variable.

    Examines the DataArray dimensions of the lat/lon coordinates and the
    field to distinguish three mutually exclusive cases:

    - ``"point"``: lat and lon are both 1-D, share the **same single
      dimension**, and that dimension appears in the field.  This is the
      geometry of station/site datasets (AirNow, AERONET, aircraft
      tracks) where each element is an independent location.

    - ``"regular_grid"``: lat and lon are both 1-D but do **not** share a
      single site dimension (i.e. they are independent axis arrays), and
      the field has at least two dimensions.  This is the geometry of
      structured rectilinear model output where lat and lon define a
      Cartesian product.

    - ``"curvilinear_grid"``: lat and/or lon are 2-D arrays whose values
      vary in two dimensions.  This is the geometry of non-rectangular
      grids (e.g. rotated-pole or staggered grids).

    Parameters
    ----------
    lat_da:
        DataArray for latitude coordinate (from ``paired_data[resolved_lat]``).
    lon_da:
        DataArray for longitude coordinate (from ``paired_data[resolved_lon]``).
    field_da:
        DataArray for the field being classified (bias, x_data, etc.).

    Returns
    -------
    str
        One of ``"point"``, ``"regular_grid"``, or ``"curvilinear_grid"``.
    """
    # Curvilinear: lat (or lon) is 2-D.  Check this first so that 2-D
    # coords are never misclassified as point data.
    if lat_da.ndim == 2:
        return "curvilinear_grid"

    # Point: both 1-D, sharing the same single dimension, and that
    # dimension is present in the field (not just a standalone coord).
    if (
        lat_da.ndim == 1
        and lon_da.ndim == 1
        and lat_da.dims == lon_da.dims
        and lat_da.dims[0] in field_da.dims
    ):
        return "point"

    # Regular grid: 1-D lat/lon axes that are independent (not the same
    # site dim), with a field that spans at least two dimensions.
    if lat_da.ndim == 1 and lon_da.ndim == 1 and field_da.ndim >= 2:
        return "regular_grid"

    # Fallback — treat as point to preserve existing "else" branch
    # behaviour (scatter for anything not clearly gridded).
    return "point"


_LAT_CANDIDATES = ["latitude", "lat", "LAT", "Latitude"]
_LON_CANDIDATES = ["longitude", "lon", "LON", "Longitude"]


def resolve_spatial_coords(
    ds: xr.Dataset,
    lat_var: str = "latitude",
    lon_var: str = "longitude",
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Resolve (lat_name, lon_name, lat_values, lon_values) from a dataset, with
    0..360 -> -180..180 longitude normalization. Raises ValueError if absent."""
    import numpy as np

    lat_name = next((c for c in [lat_var, *_LAT_CANDIDATES] if c in ds.coords or c in ds), None)
    lon_name = next((c for c in [lon_var, *_LON_CANDIDATES] if c in ds.coords or c in ds), None)
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not find latitude/longitude coordinates. Available: {list(ds.coords)}"
        )
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    if lons.ndim >= 1 and np.any(lons > 180):
        lons = np.where(lons > 180, lons - 360, lons)
    return lat_name, lon_name, lats, lons


def maybe_time_average(data: Any, time_average: bool = True, time_dim: str = "time") -> Any:
    """Mean over the time dim when present and requested; else return data."""
    if time_average and time_dim in getattr(data, "dims", ()):
        return data.mean(dim=time_dim)
    return data


def surface_level_index(field_da: xr.DataArray, level_dim: str) -> int:
    """Return the index of the surface level along ``level_dim``.

    Mirrors the auto-detection in
    :meth:`davinci_monet.pairing.strategies.base.BasePairingStrategy._extract_surface`
    so map renderers slice the *surface*, not the top of atmosphere. For
    CESM-style hybrid sigma-pressure coordinates the vertical coordinate values
    increase with index (TOA first, surface last), so the surface is the last
    index; for other conventions it is the first. Falls back to ``0`` when the
    level coordinate is absent or has fewer than two values.
    """
    return surface_index(field_da, level_dim)


def draw_spatial_field(
    ax: matplotlib.axes.Axes,
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    plot_type: str,
    cmap: Any,
    vmin: float | None,
    vmax: float | None,
    norm: Any | None = None,
    marker_size: float,
    alpha: float,
    field_dims: tuple[Any, ...] | None = None,
    lat_dim: Any | None = None,
    lon_dim: Any | None = None,
) -> Any:
    """Draw a single spatial field on a GeoAxes as scatter or pcolormesh.

    The shared field-drawing primitive used by the single-source
    :class:`~davinci_monet.plots.renderers.spatial.field.SpatialPlotter`.

    ``plot_type`` is ``"scatter"`` for point/site/track data (lat/lon are
    per-datum and broadcast, not meshgridded) or ``"pcolormesh"`` for gridded
    data (1-D independent lat/lon axes with a 2-D+ field, or 2-D curvilinear
    coordinates). Returns the matplotlib mappable for a colorbar.
    """
    import cartopy.crs as ccrs

    data = np.asarray(values)
    data_flat = data.flatten()
    if plot_type != "scatter" and lats.ndim == 1 and lons.ndim == 1 and data.ndim >= 2:
        # Regular grid: lat/lon are independent axes — build a meshgrid so each
        # grid cell gets the correct coordinate. Only for pcolormesh; scatter
        # (point/site/track) lat/lon are already per-datum and must be broadcast.
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")
        if lon_grid.shape != data.shape:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        lats_flat = lat_grid.flatten()
        lons_flat = lon_grid.flatten()
    elif lats.ndim < data.ndim:
        lats_flat = np.broadcast_to(lats, data.shape).flatten()
        lons_flat = np.broadcast_to(lons, data.shape).flatten()
    else:
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()

    # Remove NaN values (scatter path)
    mask = np.isfinite(data_flat)
    data_flat = data_flat[mask]
    lats_flat = lats_flat[mask]
    lons_flat = lons_flat[mask]

    color_kwargs: dict[str, Any] = {"cmap": cmap}
    if norm is not None:
        color_kwargs["norm"] = norm
    else:
        color_kwargs["vmin"] = vmin
        color_kwargs["vmax"] = vmax

    if plot_type == "pcolormesh" and lats.ndim == 2:
        # Curvilinear grid (e.g. swath) — pcolormesh with 2-D coords
        return ax.pcolormesh(
            lons,
            lats,
            data,
            **color_kwargs,
            transform=ccrs.PlateCarree(),
            alpha=alpha,
            rasterized=True,
        )
    if plot_type == "pcolormesh" and lats.ndim == 1 and data.ndim >= 2:
        # Regular grid with 1-D coords — pcolormesh handles natively
        mesh_data = _orient_regular_grid_data(data, lats, lons, field_dims, lat_dim, lon_dim)
        return ax.pcolormesh(
            lons,
            lats,
            mesh_data,
            **color_kwargs,
            transform=ccrs.PlateCarree(),
            alpha=alpha,
            rasterized=True,
        )
    # Point/track/site data — scatter
    return ax.scatter(
        lons_flat,
        lats_flat,
        c=data_flat,
        s=marker_size**2,
        **color_kwargs,
        transform=ccrs.PlateCarree(),
        alpha=alpha,
        edgecolors="none",
        rasterized=True,
    )


def _orient_regular_grid_data(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    field_dims: tuple[Any, ...] | None,
    lat_dim: Any | None,
    lon_dim: Any | None,
) -> np.ndarray:
    """Orient 2-D regular-grid data as ``(lat, lon)`` for pcolormesh."""
    if data.ndim != 2:
        return data
    if field_dims is not None and lat_dim in field_dims and lon_dim in field_dims:
        lat_axis = field_dims.index(lat_dim)
        lon_axis = field_dims.index(lon_dim)
        if (lat_axis, lon_axis) == (0, 1):
            return data
        if (lat_axis, lon_axis) == (1, 0):
            return data.T

    if data.shape == (len(lats), len(lons)):
        return data
    if data.shape == (len(lons), len(lats)):
        return data.T
    return data


@dataclass
class MapConfig:
    """Configuration for map display.

    Attributes
    ----------
    projection : str
        Map projection name ('PlateCarree', 'LambertConformal', etc.).
    extent : tuple[float, float, float, float] | None
        Map extent (lon_min, lon_max, lat_min, lat_max).
    show_states : bool
        Show state/province boundaries.
    show_countries : bool
        Show country boundaries.
    show_coastlines : bool
        Show coastlines.
    show_gridlines : bool
        Show lat/lon gridlines.
    gridline_style : str
        Matplotlib linestyle for lat/lon gridlines.
    resolution : str
        Feature resolution ('10m', '50m', '110m').
    land_color : str
        Color for land areas.
    ocean_color : str
        Color for ocean areas.
    """

    projection: str = "PlateCarree"
    extent: tuple[float, float, float, float] | None = None
    show_states: bool = True
    show_countries: bool = True
    show_coastlines: bool = True
    show_gridlines: bool = True
    gridline_style: str = "-"
    resolution: str = "50m"
    land_color: str = "lightgray"
    ocean_color: str = "lightblue"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> MapConfig:
        """Create MapConfig from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)


def get_projection(name: str, **kwargs: Any) -> cartopy.crs.Projection:
    """Get a cartopy projection by name.

    Parameters
    ----------
    name
        Projection name.
    **kwargs
        Projection-specific arguments.

    Returns
    -------
    cartopy.crs.Projection
        The projection.
    """
    import cartopy.crs as ccrs

    projections = {
        "PlateCarree": ccrs.PlateCarree,
        "LambertConformal": ccrs.LambertConformal,
        "Mercator": ccrs.Mercator,
        "Robinson": ccrs.Robinson,
        "Orthographic": ccrs.Orthographic,
        "AlbersEqualArea": ccrs.AlbersEqualArea,
        "LambertCylindrical": ccrs.LambertCylindrical,
    }

    proj_cls = projections.get(name, ccrs.PlateCarree)
    return proj_cls(**kwargs)


class BaseSpatialPlotter(BasePlotter):
    """Abstract base class for spatial/map plotters.

    Provides common functionality for map creation, feature overlays,
    and colorbar handling.

    Parameters
    ----------
    config
        Plot configuration.
    map_config
        Map-specific configuration.
    """

    def __init__(
        self,
        config: PlotConfig | None = None,
        map_config: MapConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.map_config = map_config or MapConfig()

    def create_map_figure(
        self,
        projection: cartopy.crs.Projection | None = None,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create a figure with map axes.

        Parameters
        ----------
        projection
            Cartopy projection. If None, uses map_config.

        Returns
        -------
        tuple[Figure, Axes]
            Figure and GeoAxes.
        """
        import cartopy.crs as ccrs

        if projection is None:
            projection = get_projection(self.map_config.projection)

        fig = plt.figure(
            figsize=self.config.figure.figsize,
            dpi=self.config.figure.dpi,
            facecolor=self.config.figure.facecolor,
        )
        ax = fig.add_subplot(111, projection=projection)

        return fig, ax

    def add_map_features(
        self,
        ax: matplotlib.axes.Axes,
        map_config: MapConfig | None = None,
    ) -> None:
        """Add map features (coastlines, borders, etc.) to axes.

        Parameters
        ----------
        ax
            GeoAxes to add features to.
        map_config
            Map configuration. If None, uses self.map_config.
        """
        import cartopy.feature as cfeature

        cfg = map_config or self.map_config

        # Add land/ocean colors
        if cfg.land_color and str(cfg.land_color).lower() != "none":
            ax.add_feature(  # type: ignore[attr-defined]
                cfeature.LAND.with_scale(cfg.resolution),
                facecolor=cfg.land_color,
                zorder=0,
            )
        if cfg.ocean_color and str(cfg.ocean_color).lower() != "none":
            ax.add_feature(  # type: ignore[attr-defined]
                cfeature.OCEAN.with_scale(cfg.resolution),
                facecolor=cfg.ocean_color,
                zorder=0,
            )

        # Add boundaries
        if cfg.show_coastlines:
            ax.add_feature(  # type: ignore[attr-defined]
                cfeature.COASTLINE.with_scale(cfg.resolution),
                linewidth=0.5,
            )
        if cfg.show_countries:
            ax.add_feature(  # type: ignore[attr-defined]
                cfeature.BORDERS.with_scale(cfg.resolution),
                linewidth=0.5,
                linestyle=":",
            )
        if cfg.show_states:
            ax.add_feature(  # type: ignore[attr-defined]
                cfeature.STATES.with_scale(cfg.resolution),
                linewidth=0.3,
                linestyle=":",
            )

        # Add gridlines
        if cfg.show_gridlines:
            gl = ax.gridlines(  # type: ignore[attr-defined]
                draw_labels=True,
                linewidth=0.5,
                alpha=0.5,
                linestyle=cfg.gridline_style,
            )
            gl.top_labels = False
            gl.right_labels = False

        # Set extent if specified
        if cfg.extent:
            ax.set_extent(cfg.extent)  # type: ignore[attr-defined]

    def add_colorbar(
        self,
        fig: matplotlib.figure.Figure,
        mappable: Any,
        ax: matplotlib.axes.Axes,
        label: str | None = None,
        orientation: str = "vertical",
        shrink: float = 0.8,
        pad: float = 0.05,
        **kwargs: Any,
    ) -> Any:
        """Add a colorbar to the figure.

        Parameters
        ----------
        fig
            Figure to add colorbar to.
        mappable
            Mappable object (from scatter, pcolormesh, etc.).
        ax
            Axes the mappable is on.
        label
            Colorbar label.
        orientation
            'vertical' or 'horizontal'.
        shrink
            Shrink factor.
        pad
            Padding.
        **kwargs
            Additional colorbar arguments.

        Returns
        -------
        Colorbar
            The created colorbar.
        """
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            orientation=orientation,
            shrink=shrink,
            pad=pad,
            **kwargs,
        )
        if label:
            cbar.set_label(label, fontsize=self.config.text.fontsize)
        cbar.ax.tick_params(labelsize=self.config.text.tick_fontsize)
        return cbar


def get_domain_extent(
    domain_type: str,
    domain_name: str | None = None,
) -> tuple[float, float, float, float] | None:
    """Get geographic extent for a named domain.

    Parameters
    ----------
    domain_type
        Type of domain ('epa_region', 'conus', 'global', etc.).
    domain_name
        Specific domain name within type (e.g., 'R1' for EPA Region 1).

    Returns
    -------
    tuple[float, float, float, float] | None
        Extent (lon_min, lon_max, lat_min, lat_max) or None if unknown.
    """
    from davinci_monet.geography.domains import get_domain_extent as _get_domain_extent

    return _get_domain_extent(domain_type, domain_name)
