"""Shared 3D track-map drawing mechanics.

This private module holds the 3D-plotting body common to the geometry-only flight
track renderer (module `davinci_monet.plots.renderers.flight_track`) and the
paired 3D track renderer (module `davinci_monet.plots.renderers.track_map_3d`).

Both renderers differ only in how they derive the colored ``values`` array
(single dataset variable vs. x/y/bias) and in the colorbar/title
text they compute.  The actual 3D scatter, surface-plane map features
(coastlines, borders, or a rendered cartopy texture), axis limits, view angle,
tick formatting, axis labels, and colorbar are identical and live here so the
two renderers stay byte-for-byte equivalent in their output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

    from davinci_monet.plots.plot_config import TextConfig


def resolve_track_coords(
    dataset: xr.Dataset,
    variable: str,
    lat_var: str = "latitude",
    lon_var: str = "longitude",
    alt_var: str = "altitude",
    alt_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return finite (lons, lats, alts, values) flat arrays for a track variable."""
    lats = dataset[lat_var].values
    lons = dataset[lon_var].values
    alts = dataset[alt_var].values * alt_scale
    vals = dataset[variable].values
    valid = np.isfinite(vals) & np.isfinite(lats) & np.isfinite(lons) & np.isfinite(alts)
    return lons[valid], lats[valid], alts[valid], vals[valid]


def _get_coastline_segments(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    scale: str = "10m",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract coastline segments from Natural Earth data within bounds.

    Parameters
    ----------
    lon_min, lon_max
        Longitude bounds.
    lat_min, lat_max
        Latitude bounds.
    scale
        Natural Earth scale: '110m', '50m', or '10m'.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (lons, lats) arrays for each coastline segment.
    """
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
    except ImportError:
        return []

    segments = []
    coastline = cfeature.NaturalEarthFeature(
        category="physical",
        name="coastline",
        scale=scale,
        facecolor="none",
    )
    bbox = box(lon_min, lat_min, lon_max, lat_max)

    for geom in coastline.geometries():
        # Clip to bounding box
        try:
            clipped = geom.intersection(bbox)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        # Handle different geometry types
        if hasattr(clipped, "geoms"):
            # MultiLineString or GeometryCollection
            geoms = list(clipped.geoms)
        else:
            geoms = [clipped]

        for g in geoms:
            if hasattr(g, "coords"):
                coords = np.array(g.coords)
                if len(coords) > 1:
                    segments.append((coords[:, 0], coords[:, 1]))

    return segments


def _get_land_polygons(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    scale: str = "50m",
) -> list[np.ndarray]:
    """Extract land polygon vertices from Natural Earth data within bounds.

    Parameters
    ----------
    lon_min, lon_max
        Longitude bounds.
    lat_min, lat_max
        Latitude bounds.
    scale
        Natural Earth scale: '110m', '50m', or '10m'.

    Returns
    -------
    list[np.ndarray]
        List of polygon vertex arrays, each with shape (N, 2) for lon, lat.
    """
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
    except ImportError:
        return []

    polygons = []
    # Use specified resolution
    land = cfeature.NaturalEarthFeature(
        category="physical",
        name="land",
        scale=scale,
        facecolor="none",
    )
    bbox = box(lon_min, lat_min, lon_max, lat_max)

    for geom in land.geometries():
        # Clip to bounding box
        try:
            clipped = geom.intersection(bbox)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        # Handle different geometry types
        if hasattr(clipped, "geoms"):
            geoms = list(clipped.geoms)
        else:
            geoms = [clipped]

        for g in geoms:
            # Get exterior ring of polygon
            if hasattr(g, "exterior"):
                coords = np.array(g.exterior.coords)
                if len(coords) >= 3:
                    polygons.append(coords[:, :2])  # lon, lat only

    return polygons


def _get_border_segments(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract country border segments from Natural Earth data within bounds.

    Parameters
    ----------
    lon_min, lon_max
        Longitude bounds.
    lat_min, lat_max
        Latitude bounds.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (lons, lats) arrays for each border segment.
    """
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
    except ImportError:
        return []

    segments = []
    borders = cfeature.BORDERS
    bbox = box(lon_min, lat_min, lon_max, lat_max)

    for geom in borders.geometries():
        try:
            clipped = geom.intersection(bbox)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        if hasattr(clipped, "geoms"):
            geoms = list(clipped.geoms)
        else:
            geoms = [clipped]

        for g in geoms:
            if hasattr(g, "coords"):
                coords = np.array(g.coords)
                if len(coords) > 1:
                    segments.append((coords[:, 0], coords[:, 1]))

    return segments


def _render_surface_map(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution: int = 250,
    land_color: str = "#E8E8E8",
    ocean_color: str = "#D4E9F7",
    coastline_color: str = "black",
    coastline_linewidth: float = 0.5,
    show_borders: bool = True,
    border_color: str = "#888888",
    border_linewidth: float = 0.3,
) -> np.ndarray | None:
    """Render a map image using cartopy for use as 3D surface texture.

    Parameters
    ----------
    lon_min, lon_max
        Longitude bounds.
    lat_min, lat_max
        Latitude bounds.
    resolution
        Target image resolution in pixels (e.g., 250 = ~250x250 pixels).
    land_color
        Color for land areas.
    ocean_color
        Color for ocean areas.
    coastline_color
        Color for coastline lines.
    coastline_linewidth
        Width of coastline lines.
    show_borders
        If True, draw country borders.
    border_color
        Color for border lines.
    border_linewidth
        Width of border lines.

    Returns
    -------
    np.ndarray | None
        RGBA image array, or None if cartopy unavailable.
    """
    try:
        import io

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from PIL import Image
    except ImportError:
        return None

    # Create figure with explicit size control
    # resolution parameter controls target image size (e.g., 250 pixels)
    fig_dpi = 100
    fig_size = resolution / fig_dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=fig_dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Remove margins
    ax.set_frame_on(False)
    ax.patch.set_visible(False)
    fig.patch.set_facecolor(ocean_color)

    # Use 50m resolution Natural Earth features (balance of quality and speed)
    ocean = cfeature.NaturalEarthFeature(
        "physical", "ocean", "50m", facecolor=ocean_color, edgecolor="none"
    )
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "50m", facecolor=land_color, edgecolor="none"
    )
    coastline = cfeature.NaturalEarthFeature(
        "physical", "coastline", "50m", facecolor="none", edgecolor=coastline_color
    )

    ax.add_feature(ocean)
    ax.add_feature(land)
    ax.add_feature(coastline, linewidth=coastline_linewidth)

    if show_borders:
        borders = cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            "50m",
            facecolor="none",
            edgecolor=border_color,
        )
        ax.add_feature(borders, linewidth=border_linewidth)

    ax.set_xticks([])
    ax.set_yticks([])

    # Render to buffer
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=fig_dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)

    # Load as numpy array
    img = Image.open(buf)
    img_array = np.array(img) / 255.0

    return img_array


def draw_track_3d(
    fig: matplotlib.figure.Figure,
    ax3d: matplotlib.axes.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    alts: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    text_cfg: TextConfig,
    cbar_label: str,
    marker_size: float,
    alpha: float,
    elev: float,
    azim: float,
    show_projection: bool,
    projection_alpha: float,
    show_coastlines: bool,
    coastline_color: str,
    coastline_alpha: float,
    coastline_linewidth: float,
    coastline_scale: str,
    show_borders: bool,
    border_color: str,
    border_alpha: float,
    border_linewidth: float,
    show_surface_map: bool,
    surface_map_resolution: int,
    land_color: str,
    ocean_color: str,
    use_maxnlocator: bool = False,
    city_labels: dict[str, list[float]] | None = None,
    city_marker_size: float = 50,
    city_marker_color: str = "red",
    city_font_size: float = 10,
) -> None:
    """Draw the shared 3D track-map body onto an existing 3D axes.

    Performs the colored 3D scatter, the optional 2D projection on the ``z=0``
    plane, the surface-plane map features (a rendered cartopy texture, vector
    coastlines, and/or country borders), optional city markers, axis-limit and
    view-angle setup, tick formatting, axis labels, and the colorbar.  The two
    track renderers create their own figure/axes and derive ``values``,
    ``vmin``/``vmax``, ``cmap``, and ``cbar_label``; everything below is shared.

    Parameters
    ----------
    fig
        Figure that owns ``ax3d`` (used to attach the colorbar).
    ax3d
        The 3D axes to draw onto.
    lons, lats, alts, values
        Already-filtered (finite) flight-track coordinate and value arrays.
        ``alts`` is expected to be pre-scaled (e.g. m -> km).
    cmap
        Colormap name.
    vmin, vmax
        Colorbar limits.
    text_cfg
        Text configuration providing the font sizes.
    cbar_label
        Colorbar label text (already formatted by the caller).
    marker_size
        Size of scatter markers.
    alpha
        Transparency of markers.
    elev, azim
        3D view angles.
    show_projection
        If True, draw a faded 2D projection on the ``z=0`` plane.
    projection_alpha
        Transparency of the projection markers.
    show_coastlines
        If True (and ``show_surface_map`` is False), draw vector coastlines.
    coastline_color, coastline_alpha, coastline_linewidth, coastline_scale
        Coastline styling.
    show_borders
        If True, draw country borders on the surface plane.
    border_color, border_alpha, border_linewidth
        Border styling.
    show_surface_map
        If True, render a filled cartopy map image on the ``z=0`` plane.
    surface_map_resolution
        Resolution of the surface map image in pixels.
    land_color, ocean_color
        Colors for the surface map image.
    use_maxnlocator
        If True, limit x/y tick counts via ``MaxNLocator(nbins=5)`` before
        formatting.  (Used by the geometry-only flight-track renderer.)
    city_labels
        Optional mapping of city name -> ``[lat, lon]`` to mark on the surface
        plane.  (Used by the paired track renderer.)
    city_marker_size, city_marker_color, city_font_size
        City marker/label styling.
    """
    # Plot 3D scatter
    scatter = ax3d.scatter(  # type: ignore[misc]
        lons,
        lats,
        alts,
        c=values,
        cmap=cmap,
        s=marker_size,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        depthshade=True,
    )

    # Add 2D projection on bottom plane
    if show_projection:
        ax3d.scatter(  # type: ignore[misc]
            lons,
            lats,
            np.zeros_like(alts),
            c=values,
            cmap=cmap,
            s=marker_size * 0.3,
            alpha=projection_alpha,
            vmin=vmin,
            vmax=vmax,
        )

    # Calculate bounds for map features and set axis limits
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    # Add padding
    lon_pad = (lon_max - lon_min) * 0.1
    lat_pad = (lat_max - lat_min) * 0.1
    lon_min -= lon_pad
    lon_max += lon_pad
    lat_min -= lat_pad
    lat_max += lat_pad

    # Draw surface map or coastlines
    if show_surface_map:
        # Render cartopy map as texture on z=0 plane
        map_img = _render_surface_map(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            resolution=surface_map_resolution,
            land_color=land_color,
            ocean_color=ocean_color,
            coastline_color=coastline_color,
            coastline_linewidth=coastline_linewidth,
            show_borders=show_borders,
            border_color=border_color,
            border_linewidth=border_linewidth,
        )
        if map_img is not None:
            # Create mesh grid for surface
            img_h, img_w = map_img.shape[:2]
            lon_grid = np.linspace(lon_min, lon_max, img_w)
            lat_grid = np.linspace(lat_max, lat_min, img_h)  # Flip for image coords
            X, Y = np.meshgrid(lon_grid, lat_grid)
            Z = np.zeros_like(X)

            # Plot textured surface
            ax3d.plot_surface(  # type: ignore[attr-defined]
                X,
                Y,
                Z,
                facecolors=map_img,
                rstride=1,
                cstride=1,
                shade=False,
                zorder=1,
            )
    elif show_coastlines:
        # Fall back to vector coastlines
        coastline_segments = _get_coastline_segments(
            lon_min, lon_max, lat_min, lat_max, scale=coastline_scale
        )
        for seg_lons, seg_lats in coastline_segments:
            ax3d.plot(
                seg_lons,
                seg_lats,
                np.zeros(len(seg_lons)),
                color=coastline_color,
                alpha=coastline_alpha,
                linewidth=coastline_linewidth,
                zorder=2,
            )

    # Draw country borders on surface plane (z=0) - only if not using surface map
    if show_borders and not show_surface_map:
        border_segments = _get_border_segments(lon_min, lon_max, lat_min, lat_max)
        for seg_lons, seg_lats in border_segments:
            ax3d.plot(
                seg_lons,
                seg_lats,
                np.zeros(len(seg_lons)),
                color=border_color,
                alpha=border_alpha,
                linewidth=border_linewidth,
                linestyle="--",
                zorder=2,
            )

    # Draw city markers and labels on surface plane
    if city_labels:
        for city_name, coords in city_labels.items():
            city_lat, city_lon = coords[0], coords[1]
            # Only plot if within bounds
            if lon_min <= city_lon <= lon_max and lat_min <= city_lat <= lat_max:
                # Plot marker
                ax3d.scatter(  # type: ignore[misc]
                    [city_lon],
                    [city_lat],
                    [0],
                    s=city_marker_size,
                    c=city_marker_color,
                    marker="^",
                    alpha=0.9,
                    zorder=10,
                )
                # Add label
                ax3d.text(
                    city_lon,
                    city_lat,
                    0,  # type: ignore[arg-type]
                    f"  {city_name}",  # type: ignore[arg-type]
                    fontsize=city_font_size,
                    color="black",
                    ha="left",
                    va="bottom",
                    zorder=11,
                )

    # Set axis limits to include padding
    ax3d.set_xlim(lon_min, lon_max)
    ax3d.set_ylim(lat_min, lat_max)

    # Set view angle
    ax3d.view_init(elev=elev, azim=azim)  # type: ignore[attr-defined]

    # Limit tick count to prevent overlap (geometry-only flight track), then format labels
    if use_maxnlocator:
        ax3d.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax3d.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax3d.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax3d.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # Labels
    ax3d.set_xlabel("Longitude (°E)", fontsize=text_cfg.fontsize, labelpad=8)
    ax3d.set_ylabel("Latitude (°N)", fontsize=text_cfg.fontsize, labelpad=8)
    ax3d.set_zlabel("Altitude (km)", fontsize=text_cfg.fontsize, labelpad=8)  # type: ignore[attr-defined]

    # Tick label size
    ax3d.tick_params(axis="both", labelsize=text_cfg.tick_fontsize)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax3d, shrink=0.6, pad=0.1)
    cbar.set_label(cbar_label, fontsize=text_cfg.fontsize)
    cbar.ax.tick_params(labelsize=text_cfg.tick_fontsize)


__all__ = [
    "resolve_track_coords",
    "_get_coastline_segments",
    "_get_land_polygons",
    "_get_border_segments",
    "_render_surface_map",
    "draw_track_3d",
]
