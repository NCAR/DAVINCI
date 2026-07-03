"""LMA flash density map renderer.

Produces cartopy maps of Lightning Mapping Array gridded flash density data,
with optional aircraft flight track overlays.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.plots import labeling
from davinci_monet.plots.base import BasePlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import NCAR_PALETTE


@register_plotter("lma_density", arity="single_source", category="specialized")
class LMADensityPlotter(BasePlotter):
    """Plotter for LMA gridded flash density maps."""

    name: str = "lma_density"
    default_figsize: tuple[float, float] = (10, 8)

    def render(
        self,
        series: list[Any],
        ax: "matplotlib.axes.Axes | None" = None,
        **kwargs: Any,
    ) -> Any:
        """Unified entry: render a single LMA source (may return hourly list)."""
        if len(series) != 1:
            raise NotImplementedError(
                f"LMADensityPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        return self._plot(s.dataset, s.var_name, **kwargs)

    def _plot(
        self,
        x_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        title: str | None = None,
        cmap: str = "YlOrRd",
        vmin: float | None = None,
        vmax: float | None = None,
        time_agg: str | None = None,
        map: dict[str, Any] | None = None,
        flight_tracks: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure | list[tuple[str, matplotlib.figure.Figure]]:
        """Generate LMA density map(s).

        Parameters
        ----------
        x_data : xr.Dataset
            LMA gridded data with dims (time, latitude, longitude).
        variable : str
            Variable name to plot (e.g. 'flash_extent').
        title : str, optional
            Title template. Hour info is appended automatically.
        cmap : str
            Matplotlib colormap name.
        vmin, vmax : float, optional
            Colorbar limits. Auto-scaled if not provided.
        time_agg : str, optional
            If 'hourly', produce one figure per hour. Otherwise one figure
            summing all time steps.
        map : dict, optional
            Map configuration (projection, features).
        flight_tracks : dict, optional
            Mapping of {label: source_key} for flight track overlays.
            Track data is resolved from kwargs['x_datasets'].

        Returns
        -------
            Figure or list of (label, Figure) tuples for multi-hour output.
        """
        map_config = map or {}
        projection = self._get_projection(map_config)
        features = map_config.get("features", ["states"])

        lat = x_data["latitude"].values
        lon = x_data["longitude"].values

        if time_agg == "hourly":
            hourly_groups = self._aggregate_hourly(x_data, variable)
            if not hourly_groups:
                fig, _ = self.create_figure()
                return fig

            results = []
            if vmax is None:
                all_maxes = [float(data.max()) for _, data in hourly_groups]
                auto_vmax = max(all_maxes) if all_maxes else 1.0
            else:
                auto_vmax = vmax

            for hour_label, data_2d in hourly_groups:
                fig = self._render_map(
                    lat,
                    lon,
                    data_2d,
                    projection=projection,
                    features=features,
                    cmap=cmap,
                    vmin=vmin or 0,
                    vmax=auto_vmax,
                    title=f"{title} {hour_label}" if title else hour_label,
                    flight_tracks=flight_tracks,
                    hour_label=hour_label,
                    **kwargs,
                )
                cleaned = hour_label.replace(":", "").replace(" ", "_").replace("\u2013", "-")
                results.append((cleaned, fig))
            return results
        else:
            summed = x_data[variable].sum(dim="time")
            if "latitude" in summed.dims and "longitude" in summed.dims:
                summed = summed.transpose("latitude", "longitude")
            data_2d = summed.values
            fig = self._render_map(
                lat,
                lon,
                data_2d,
                projection=projection,
                features=features,
                cmap=cmap,
                vmin=vmin or 0,
                vmax=vmax or float(data_2d.max()),
                title=title or f"LMA {variable}",
                flight_tracks=flight_tracks,
                **kwargs,
            )
            return fig

    def _get_projection(self, map_config: dict[str, Any]) -> ccrs.Projection:
        """Create cartopy projection from config."""
        proj_name = map_config.get("projection", "LambertConformal")
        if proj_name == "LambertConformal":
            return ccrs.LambertConformal(
                central_longitude=-98.5,
                central_latitude=35.0,
            )
        elif proj_name == "PlateCarree":
            return ccrs.PlateCarree()
        else:
            return ccrs.PlateCarree()

    def _aggregate_hourly(
        self,
        ds: xr.Dataset,
        variable: str,
    ) -> list[tuple[str, np.ndarray]]:
        """Aggregate data into hourly sums, returning only hours with activity.

        Returns 2D arrays in (latitude, longitude) order for pcolormesh.
        """
        groups = ds[variable].resample(time="1h").sum()
        results = []
        for t in groups.time.values:
            data_slice = groups.sel(time=t)
            # Ensure (latitude, longitude) ordering for pcolormesh
            if "latitude" in data_slice.dims and "longitude" in data_slice.dims:
                data_slice = data_slice.transpose("latitude", "longitude")
            data_2d = data_slice.values
            if np.nansum(data_2d) > 0:
                ts = np.datetime_as_string(t, unit="h")
                hour = int(ts[-2:]) if len(ts) >= 2 else 0
                date_str = str(t)[:10]
                label = f"{date_str} {hour:02d}:00\u2013{(hour + 1) % 24:02d}:00 UTC"
                results.append((label, data_2d))
        return results

    def _render_map(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        data_2d: np.ndarray,
        *,
        projection: ccrs.Projection,
        features: list[str],
        cmap: str,
        vmin: float,
        vmax: float,
        title: str,
        flight_tracks: dict[str, str] | None = None,
        hour_label: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a single density map."""
        text_cfg = self.config.text
        fig_cfg = self.config.figure

        fig = plt.figure(figsize=fig_cfg.figsize, dpi=fig_cfg.dpi)
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        pad = 0.3
        ax.set_extent(  # type: ignore[attr-defined]
            [lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad],
            crs=ccrs.PlateCarree(),
        )

        ax.add_feature(cfeature.LAND, facecolor="#F0F0F0", zorder=0)  # type: ignore[attr-defined]
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)  # type: ignore[attr-defined]
        if "states" in features:
            ax.add_feature(  # type: ignore[attr-defined]
                cfeature.STATES,
                edgecolor="gray",
                linewidth=0.5,
                zorder=1,
            )
        if "counties" in features:
            try:
                ax.add_feature(  # type: ignore[attr-defined]
                    cfeature.NaturalEarthFeature(
                        "cultural",
                        "admin_2_counties_lakes_shp",
                        "10m",
                        edgecolor="lightgray",
                        facecolor="none",
                        linewidth=0.3,
                    ),
                    zorder=1,
                )
            except Exception:
                # 10m county shapefile may not be cached; fall back silently
                pass
        if "coastlines" in features:
            ax.add_feature(  # type: ignore[attr-defined]
                cfeature.COASTLINE,
                edgecolor="black",
                linewidth=0.5,
                zorder=1,
            )

        mesh = ax.pcolormesh(
            lon,
            lat,
            data_2d,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
            zorder=2,
            rasterized=True,
        )

        if flight_tracks:
            self._overlay_tracks(ax, flight_tracks, hour_label, **kwargs)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)  # type: ignore[attr-defined]
        gl.top_labels = False
        gl.right_labels = False

        cbar = fig.colorbar(
            mesh,
            ax=ax,
            orientation="horizontal",
            shrink=0.7,
            pad=0.06,
        )
        cbar.set_label(
            "Flash extent density (flashes per grid cell)",
            fontsize=text_cfg.fontsize,
        )
        cbar.ax.tick_params(labelsize=text_cfg.tick_fontsize)

        self.set_figure_title(fig, title, y=0.95)

        return fig

    def _overlay_tracks(
        self,
        ax: matplotlib.axes.Axes,
        flight_tracks: dict[str, str],
        hour_label: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Overlay aircraft flight tracks on the map."""
        x_datasets: dict[str, xr.Dataset] = kwargs.get("x_datasets", {})
        if not x_datasets:
            return

        track_colors = {
            label: NCAR_PALETTE[i % len(NCAR_PALETTE)]
            for i, label in enumerate(flight_tracks.keys())
        }

        for label, x_key in flight_tracks.items():
            if x_key not in x_datasets:
                continue
            track_ds = x_datasets[x_key]

            if "latitude" not in track_ds.coords or "longitude" not in track_ds.coords:
                continue
            track_lat = track_ds["latitude"].values
            track_lon = track_ds["longitude"].values

            ax.plot(
                track_lon,
                track_lat,
                color=track_colors[label],
                linewidth=1.5,
                transform=ccrs.PlateCarree(),
                label=labeling.source_display_name(label),
                zorder=5,
            )

        ax.legend(
            loc="upper right",
            fontsize=self.config.text.legend_small,
            framealpha=0.8,
        )
