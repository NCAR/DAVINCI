"""EOF spatial-pattern map renderer (one signed map per mode)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.colors import TwoSlopeNorm

from davinci_monet.plots import labeling
from davinci_monet.plots.base import calculate_symmetric_limits, get_variable_units
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    draw_spatial_field,
    resolve_spatial_coords,
    surface_level_index,
)
from davinci_monet.plots.style import get_bias_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


@register_plotter("eof_pattern", arity="single_source", category="spatial")
class EOFPatternPlotter(BaseSpatialPlotter):
    """Render each EOF mode as a signed spatial map (diverging, centered at 0)."""

    name: str = "eof_pattern"
    default_figsize: tuple[float, float] = (8, 6)

    def render(
        self,
        series: list["PlotSeries"],
        ax: "matplotlib.axes.Axes | None" = None,
        *,
        display_level: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, "matplotlib.figure.Figure"]]:
        if len(series) != 1:
            raise NotImplementedError(
                f"EOFPatternPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        ds = s.dataset
        field = ds[s.var_name]
        if "mode" not in field.dims:
            raise NotImplementedError("eof_pattern requires a 'mode' dimension on the variable.")

        lat_name, lon_name, lats, lons = resolve_spatial_coords(ds)
        horiz = set(ds[lat_name].dims) | set(ds[lon_name].dims)
        units = get_variable_units(ds, s.var_name)
        quantity = str(ds.attrs.get("eof_quantity", ""))

        figures: list[tuple[str, "matplotlib.figure.Figure"]] = []
        for m in [int(v) for v in field["mode"].values]:
            fld = field.sel(mode=m)
            vdims = [d for d in fld.dims if d not in horiz]
            if vdims:
                lev = str(vdims[0])
                idx = display_level if display_level is not None else surface_level_index(fld, lev)
                fld = fld.isel({lev: idx})

            vmin, vmax = calculate_symmetric_limits(fld.values)
            fig, axx = self.create_map_figure()
            self.add_map_features(axx)
            mappable = draw_spatial_field(
                axx,
                fld.values,
                lats,
                lons,
                plot_type="pcolormesh",
                cmap=get_bias_cmap(),
                vmin=vmin,
                vmax=vmax,
                marker_size=self.config.style.markersize * 2,
                alpha=1.0,
            )
            # Rasterize the dense field layer; axes/text/colorbar stay vector.
            mappable.set_rasterized(True)
            if vmin < 0 < vmax:
                mappable.set_norm(TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
            self.add_colorbar(fig, mappable, axx, label=labeling.format_units(units))

            ev_pct = None
            if "explained_variance" in ds:
                ev_pct = float(ds["explained_variance"].sel(mode=m).item()) * 100.0
            self.set_title(
                axx,
                labeling.title_text(quantity, operation=f"EOF Mode {m}"),
                subtitle=(f"{ev_pct:.1f}% variance" if ev_pct is not None else None),
            )
            figures.append((f"mode{m}", fig))
        return figures
