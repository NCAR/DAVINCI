"""Before/after science figures for MODIS-constrained MERRA-2 AOD."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from matplotlib.layout_engine import ConstrainedLayoutEngine

from davinci_monet.core.exceptions import PlottingError
from davinci_monet.plots import labeling
from davinci_monet.plots.contracts import AOD_CORRECTION_FIGURES, AOD_CORRECTION_PROTOCOL
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.aod_correction_science_data import (
    SEASONS,
    AODCorrectionInputs,
    area_weighted_rmse_timeseries,
    area_weighted_timeseries,
    correction_inputs,
    finite_percentile,
    scatter_sample,
    seasonal_mean,
    symmetric_limit,
    weighted_metrics,
)
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
    draw_spatial_field,
    get_projection,
    resolve_spatial_coords,
)
from davinci_monet.plots.style import (
    NCAR_COLORS,
    NCAR_PRIMARY,
    geosit_aod_levels,
    get_bias_cmap,
    get_density_cmap,
    get_geosit_aod_cmap,
    get_sequential_cmap,
)

if TYPE_CHECKING:
    import matplotlib.figure

    from davinci_monet.core.base import PlotSeries


_GLOBAL_MAP = MapConfig(
    resolution="110m",
    show_states=False,
    show_countries=False,
    show_gridlines=False,
    land_color="none",
    ocean_color="none",
)

_DENSE_ARTIST_COUNTS = {
    "annual_aod": 3,
    "annual_bias_improvement": 3,
    "seasonal_aod": 12,
    "seasonal_error_reduction": 4,
    "agreement_metrics": 0,
    "global_timeseries": 0,
    "matched_scatter": 2,
    "correction_diagnostics": 4,
}

_AOD_COLORBAR_TICKS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.95)


@dataclass(frozen=True)
class _AODLabels:
    quantity: str
    units: str | None
    model_source: str
    corrected_source: str
    observation_source: str

    @property
    def model(self) -> str:
        return labeling.legend_label(self.model_source)

    @property
    def corrected(self) -> str:
        return labeling.legend_label(self.corrected_source)

    @property
    def observation(self) -> str:
        return labeling.legend_label(self.observation_source)


def _labels_for_inputs(
    inputs: AODCorrectionInputs,
    *,
    model_source: str,
    corrected_source: str,
    observation_source: str,
) -> _AODLabels:
    field_name = str(inputs.observation.name or "aod")
    dataset = inputs.observation.to_dataset(name=field_name)
    return _AODLabels(
        quantity=labeling.quantity_label(dataset, field_name),
        units=inputs.observation.attrs.get("units"),
        model_source=model_source,
        corrected_source=corrected_source,
        observation_source=observation_source,
    )


def _record_dense_artists(
    figure: matplotlib.figure.Figure,
    artists: list[Any],
    *,
    absolute_aod: bool = False,
) -> None:
    setattr(figure, "_davinci_dense_artists", artists)
    setattr(figure, "_davinci_absolute_aod_artists", artists if absolute_aod else [])


def _reserve_publication_layout(
    figure: matplotlib.figure.Figure,
    *,
    footer: bool = False,
) -> None:
    """Reserve explicit header/footer bands outside constrained-layout axes."""
    bottom = 0.075 if footer else 0.015
    top = 0.84
    engine = figure.get_layout_engine()
    if not isinstance(engine, ConstrainedLayoutEngine):
        raise PlottingError("AOD correction figures require a constrained-layout engine")
    engine.set(rect=(0.01, bottom, 0.98, top - bottom))


def _visible_text(artist: Any) -> bool:
    return bool(artist.get_visible() and str(artist.get_text()).strip())


def _boxes_overlap(left: Any, right: Any, *, pad: float = 1.0) -> bool:
    return bool(
        left.x0 < right.x1 + pad
        and left.x1 + pad > right.x0
        and left.y0 < right.y1 + pad
        and left.y1 + pad > right.y0
    )


def _layout_failures(
    label: str,
    figure: matplotlib.figure.Figure,
) -> list[str]:
    """Return publication-layout failures using the rendered canvas geometry."""
    canvas: Any = figure.canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    figure_box = figure.bbox
    failures: list[str] = []

    items: list[tuple[str, Any]] = []
    for index, artist in enumerate(figure.texts):
        if _visible_text(artist):
            items.append((f"figure text {index}", artist))
    for index, axis in enumerate(figure.axes):
        for name, artist in (
            ("title", axis.title),
            ("x label", axis.xaxis.label),
            ("y label", axis.yaxis.label),
        ):
            if _visible_text(artist):
                items.append((f"axis {index} {name}", artist))
        legend = axis.get_legend()
        if legend is not None and legend.get_visible():
            items.append((f"axis {index} legend", legend))

        for tick_axis, tick_artists in (
            ("x", axis.get_xticklabels()),
            ("y", axis.get_yticklabels()),
        ):
            visible_ticks = [artist for artist in tick_artists if _visible_text(artist)]
            tick_boxes = [artist.get_window_extent(renderer) for artist in visible_ticks]
            tick_boxes.sort(key=lambda box: box.x0 if tick_axis == "x" else box.y0)
            for first, second in zip(tick_boxes, tick_boxes[1:]):
                if _boxes_overlap(first, second, pad=0.0):
                    failures.append(f"{label} axis {index} {tick_axis} tick labels overlap")
                    break

    unique_items: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for name, artist in items:
        if id(artist) not in seen:
            unique_items.append((name, artist))
            seen.add(id(artist))

    boxes: list[tuple[str, Any]] = []
    for name, artist in unique_items:
        box = artist.get_window_extent(renderer)
        boxes.append((name, box))
        if (
            box.x0 < figure_box.x0 - 1.0
            or box.x1 > figure_box.x1 + 1.0
            or box.y0 < figure_box.y0 - 1.0
            or box.y1 > figure_box.y1 + 1.0
        ):
            failures.append(f"{label} {name} is clipped by the figure")

    for index, (left_name, left_box) in enumerate(boxes):
        for right_name, right_box in boxes[index + 1 :]:
            if _boxes_overlap(left_box, right_box):
                failures.append(f"{label} {left_name} overlaps {right_name}")

    return failures


@register_plotter("aod_correction_science", arity="multi_source", category="specialized")
class AODCorrectionSciencePlotter(BaseSpatialPlotter):
    """Render and enforce the corrected-product story against MODIS observations."""

    name = "aod_correction_science"
    default_figsize = (12, 8)

    @staticmethod
    def _absolute_aod_style(fields: list[xr.DataArray]) -> tuple[Any, BoundaryNorm]:
        values = np.concatenate(
            [np.asarray(field.values, dtype=float).reshape(-1) for field in fields]
        )
        levels = geosit_aod_levels(values)
        cmap = plt.get_cmap(get_geosit_aod_cmap())
        return cmap, BoundaryNorm(levels, cmap.N, extend="max")

    def _map_panels(
        self,
        fields: list[xr.DataArray],
        titles: list[str],
        *,
        figure_title: str,
        colorbar_label: str,
        cmap: Any,
        columns: int,
        vmin: float | None = None,
        vmax: float | None = None,
        norm: Any = None,
        absolute_aod: bool = False,
    ) -> matplotlib.figure.Figure:
        rows = math.ceil(len(fields) / columns)
        figure = plt.figure(
            figsize=(4.9 * columns, 3.15 * rows + 1.0),
            dpi=self.config.figure.dpi,
            facecolor=self.config.figure.facecolor,
            constrained_layout=True,
        )
        projection = get_projection(_GLOBAL_MAP.projection)
        axes = []
        artists = []
        for index, (field, title) in enumerate(zip(fields, titles, strict=True)):
            axis = figure.add_subplot(rows, columns, index + 1, projection=projection)
            self.add_map_features(axis, _GLOBAL_MAP)
            axis.set_global()
            artist = self._draw_map(
                axis,
                field,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
            )
            axis.set_title(title, fontsize=self.config.text.fontsize)
            axes.append(axis)
            artists.append(artist)
        if not artists:
            raise ValueError("no AOD correction map panels were requested")
        colorbar = figure.colorbar(
            artists[-1],
            ax=axes,
            orientation="horizontal",
            fraction=0.04,
            pad=0.03,
            shrink=0.90,
            aspect=80,
            label=colorbar_label,
        )
        if absolute_aod and isinstance(norm, BoundaryNorm):
            upper = float(np.asarray(norm.boundaries)[-1])
            ticks = [tick for tick in _AOD_COLORBAR_TICKS if tick <= upper + np.finfo(float).eps]
            colorbar.set_ticks(ticks)
        _reserve_publication_layout(figure)
        self.set_figure_title(figure, figure_title, y=0.985, subtitle_y=0.91)
        _record_dense_artists(figure, artists, absolute_aod=absolute_aod)
        return figure

    def _draw_map(
        self,
        axis: Any,
        field: xr.DataArray,
        *,
        cmap: Any,
        vmin: float | None = None,
        vmax: float | None = None,
        norm: Any = None,
    ) -> Any:
        dataset = field.to_dataset(name="field")
        lat_name, lon_name, latitudes, longitudes = resolve_spatial_coords(dataset)
        return draw_spatial_field(
            axis,
            field.values,
            latitudes,
            longitudes,
            plot_type="pcolormesh",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            marker_size=self.config.style.markersize,
            alpha=1.0,
            field_dims=tuple(field.dims),
            lat_dim=dataset[lat_name].dims[0],
            lon_dim=dataset[lon_name].dims[0],
        )

    def _annual_aod_figure(
        self,
        inputs: AODCorrectionInputs,
        labels: _AODLabels,
    ) -> matplotlib.figure.Figure:
        valid = inputs.valid
        fields = [
            inputs.model.where(valid).mean("time", skipna=True),
            inputs.corrected.where(valid).mean("time", skipna=True),
            inputs.observation.where(valid).mean("time", skipna=True),
        ]
        cmap, norm = self._absolute_aod_style(fields)
        return self._map_panels(
            fields,
            [labels.model, labels.corrected, labels.observation],
            figure_title=labeling.title_text(labels.quantity, operation="Annual Mean"),
            colorbar_label=labeling.axis_label(labels.quantity, labels.units),
            cmap=cmap,
            norm=norm,
            columns=3,
            absolute_aod=True,
        )

    def _annual_bias_figure(
        self,
        inputs: AODCorrectionInputs,
        labels: _AODLabels,
    ) -> matplotlib.figure.Figure:
        valid = inputs.valid
        original_error = (inputs.model - inputs.observation).where(valid)
        corrected_error = (inputs.corrected - inputs.observation).where(valid)
        fields = [
            original_error.mean("time", skipna=True),
            corrected_error.mean("time", skipna=True),
            (abs(original_error) - abs(corrected_error)).mean("time", skipna=True),
        ]
        limit = symmetric_limit(fields)
        return self._map_panels(
            fields,
            [
                f"{labels.model} −\n{labels.observation}",
                f"{labels.corrected} −\n{labels.observation}",
                "Absolute-Error Reduction",
            ],
            figure_title=labeling.title_text(labels.quantity, operation="Bias and Error Reduction"),
            colorbar_label=labeling.axis_label("AOD Difference", labels.units),
            cmap=get_bias_cmap(),
            vmin=-limit,
            vmax=limit,
            columns=3,
        )

    def _seasonal_aod_figure(
        self,
        inputs: AODCorrectionInputs,
        labels: _AODLabels,
    ) -> matplotlib.figure.Figure:
        valid = inputs.valid
        masked = (
            inputs.model.where(valid),
            inputs.corrected.where(valid),
            inputs.observation.where(valid),
        )
        source_labels = (labels.model, labels.corrected, labels.observation)
        fields: list[xr.DataArray] = []
        titles: list[str] = []
        for season, months in SEASONS:
            for source_label, field in zip(source_labels, masked, strict=True):
                fields.append(seasonal_mean(field, months))
                titles.append(f"{season} · {source_label}")
        cmap, norm = self._absolute_aod_style(fields)
        return self._map_panels(
            fields,
            titles,
            figure_title=labeling.title_text(labels.quantity, operation="Seasonal Mean"),
            colorbar_label=labeling.axis_label(labels.quantity, labels.units),
            cmap=cmap,
            norm=norm,
            columns=3,
            absolute_aod=True,
        )

    def _seasonal_improvement_figure(
        self,
        inputs: AODCorrectionInputs,
        labels: _AODLabels,
    ) -> matplotlib.figure.Figure:
        original_error = (inputs.model - inputs.observation).where(inputs.valid)
        corrected_error = (inputs.corrected - inputs.observation).where(inputs.valid)
        improvement = abs(original_error) - abs(corrected_error)
        fields = [seasonal_mean(improvement, months) for _label, months in SEASONS]
        limit = symmetric_limit(fields)
        return self._map_panels(
            fields,
            [label for label, _months in SEASONS],
            figure_title=labeling.title_text(labels.quantity, operation="Seasonal Error Reduction"),
            colorbar_label=labeling.axis_label("Absolute-Error Reduction", labels.units),
            cmap="RdYlGn",
            vmin=-limit,
            vmax=limit,
            columns=2,
        )

    def _metrics_figure(
        self,
        inputs: AODCorrectionInputs,
        labels: _AODLabels,
    ) -> matplotlib.figure.Figure:
        masks = (
            ("All Matched Cells", inputs.valid),
            ("Correction-Supported Cells", inputs.valid & (inputs.support > 0.0)),
        )
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(13, 4.8),
            constrained_layout=True,
            dpi=self.config.figure.dpi,
        )
        for axis, (title, mask) in zip(axes, masks, strict=True):
            original = weighted_metrics(inputs.model, inputs.observation, mask)
            corrected = weighted_metrics(inputs.corrected, inputs.observation, mask)
            rows = []
            for metric in ("MB", "MAE", "RMSE", "R"):
                before = original[metric]
                after = corrected[metric]
                rows.append([metric, f"{before:.4f}", f"{after:.4f}", f"{after - before:+.4f}"])
            axis.axis("off")
            table = axis.table(
                cellText=rows,
                colLabels=["Metric", "Original", "Corrected", "Δ Corrected−Original"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.config.text.annotation_small)
            table.scale(1.0, 1.65)
            axis.set_title(title)
            axis.text(
                0.5,
                0.10,
                f"N = {int(original['N']):,}",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=self.config.text.annotation_small,
            )
        _reserve_publication_layout(figure, footer=True)
        self.set_figure_title(
            figure,
            labeling.title_text(labels.quantity, operation="Agreement"),
            y=0.985,
            subtitle_y=0.91,
        )
        figure.text(
            0.5,
            0.015,
            "Metrics use the matched observations that constrain the correction.",
            ha="center",
            fontsize=self.config.text.annotation_small,
            color=NCAR_COLORS["gray"],
        )
        _record_dense_artists(figure, [])
        return figure

    def _timeseries_figure(
        self,
        inputs: AODCorrectionInputs,
        labels: _AODLabels,
    ) -> matplotlib.figure.Figure:
        valid = inputs.valid
        model = area_weighted_timeseries(inputs.model, valid)
        corrected = area_weighted_timeseries(inputs.corrected, valid)
        observed = area_weighted_timeseries(inputs.observation, valid)
        original_rmse = area_weighted_rmse_timeseries(inputs.model, inputs.observation, valid)
        corrected_rmse = area_weighted_rmse_timeseries(inputs.corrected, inputs.observation, valid)
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            constrained_layout=True,
            dpi=self.config.figure.dpi,
        )
        time = np.asarray(inputs.model["time"].values)
        axes[0].plot(
            time,
            model.values,
            color=NCAR_COLORS["gray"],
            label=labeling.legend_label(labels.model_source),
        )
        axes[0].plot(
            time,
            corrected.values,
            color=NCAR_PRIMARY,
            label=labeling.legend_label(labels.corrected_source),
        )
        axes[0].plot(
            time,
            observed.values,
            color=NCAR_COLORS["orange"],
            label=labeling.legend_label(labels.observation_source),
        )
        axes[0].set_ylabel(labeling.axis_label("Matched-Grid Mean AOD", labels.units))
        axes[0].legend(ncol=3)
        axes[1].plot(
            time,
            original_rmse.values,
            color=NCAR_COLORS["gray"],
            label=labeling.legend_label(labels.model_source),
        )
        axes[1].plot(
            time,
            corrected_rmse.values,
            color=NCAR_PRIMARY,
            label=labeling.legend_label(labels.corrected_source),
        )
        axes[1].set_ylabel(labeling.axis_label("Daily Spatial AOD RMSE", labels.units))
        axes[1].legend(ncol=2)
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axes[1].tick_params(axis="x", labelrotation=45)
        for axis in axes:
            axis.grid(True, alpha=0.3)
        _reserve_publication_layout(figure)
        self.set_figure_title(
            figure,
            labeling.title_text(labels.quantity, operation="Time Series"),
            y=0.985,
            subtitle_y=0.91,
        )
        _record_dense_artists(figure, [])
        return figure

    def _scatter_figure(
        self,
        inputs: AODCorrectionInputs,
        labels: _AODLabels,
        *,
        max_points: int,
    ) -> matplotlib.figure.Figure:
        model, corrected, observed = scatter_sample(inputs, max_points=max_points)
        if observed.size == 0:
            raise ValueError("no finite MODIS-matched samples are available for scatter plots")
        combined = np.concatenate((model, corrected, observed))
        upper = float(np.percentile(combined, 99.5))
        upper = max(upper, np.finfo(float).eps)
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5.7),
            constrained_layout=True,
            dpi=self.config.figure.dpi,
        )
        dense_artists = []
        for axis, estimate, source in (
            (axes[0], model, labels.model_source),
            (axes[1], corrected, labels.corrected_source),
        ):
            density = axis.hexbin(
                observed,
                estimate,
                gridsize=80,
                mincnt=1,
                bins="log",
                cmap=get_density_cmap(),
                extent=(0.0, upper, 0.0, upper),
                rasterized=True,
            )
            dense_artists.append(density)
            axis.plot([0.0, upper], [0.0, upper], "k--", linewidth=1.0)
            axis.set(
                xlim=(0.0, upper),
                ylim=(0.0, upper),
                xlabel=labeling.axis_label(
                    "Daily Sampled AOD", labels.units, source=labels.observation_source
                ),
                ylabel=labeling.axis_label("Daily Sampled AOD", labels.units, source=source),
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(labeling.legend_label(source))
            axis.grid(True, alpha=0.2)
            figure.colorbar(
                density,
                ax=axis,
                label=labeling.axis_label("Sampled Cell Count", "1"),
            )
        _reserve_publication_layout(figure, footer=True)
        self.set_figure_title(
            figure,
            labeling.title_text(labels.quantity, operation="Matched Distribution"),
            y=0.985,
            subtitle_y=0.91,
        )
        figure.text(
            0.5,
            0.015,
            f"N = {observed.size:,} matched samples",
            ha="center",
            fontsize=self.config.text.annotation_small,
            color=NCAR_COLORS["gray"],
        )
        _record_dense_artists(figure, dense_artists)
        return figure

    def _diagnostics_figure(
        self,
        inputs: AODCorrectionInputs,
        labels: _AODLabels,
    ) -> matplotlib.figure.Figure:
        active = inputs.support > 0.0
        ratio = inputs.ratio.where(active).mean("time", skipna=True)
        adjustment = (inputs.corrected - inputs.model).where(active).mean("time", skipna=True)
        support = inputs.support.mean("time", skipna=True)
        clipping = (inputs.lower_clip | inputs.upper_clip).mean("time", skipna=True) * 100.0
        fields = [ratio, adjustment, support, clipping]
        titles = ["Mean Applied Ratio", "Mean AOD Adjustment", "Mean Support", "Clip Frequency"]
        cmaps = [
            get_bias_cmap(),
            get_bias_cmap(),
            get_sequential_cmap(),
            "magma",
        ]
        adjustment_limit = symmetric_limit([adjustment])
        ratio_limit = symmetric_limit([ratio - 1.0])
        limits = [
            (1.0 - ratio_limit, 1.0 + ratio_limit),
            (-adjustment_limit, adjustment_limit),
            (0.0, 1.0),
            (0.0, max(1.0, finite_percentile([clipping], 99.0))),
        ]
        colorbar_labels = [
            labeling.axis_label("Applied AOD Ratio", "1"),
            labeling.axis_label("AOD Adjustment", labels.units),
            labeling.axis_label("Support Fraction", "1"),
            labeling.axis_label("Clipped Days", "%"),
        ]
        figure = plt.figure(
            figsize=(12, 7.5),
            constrained_layout=True,
            dpi=self.config.figure.dpi,
        )
        projection = get_projection(_GLOBAL_MAP.projection)
        dense_artists = []
        for index, (field, title, cmap, (vmin, vmax), colorbar_label) in enumerate(
            zip(fields, titles, cmaps, limits, colorbar_labels, strict=True)
        ):
            axis = figure.add_subplot(2, 2, index + 1, projection=projection)
            self.add_map_features(axis, _GLOBAL_MAP)
            axis.set_global()
            norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax) if index == 0 else None
            artist = self._draw_map(
                axis,
                field,
                cmap=cmap,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
                norm=norm,
            )
            dense_artists.append(artist)
            axis.set_title(title)
            figure.colorbar(
                artist,
                ax=axis,
                orientation="horizontal",
                pad=0.04,
                label=colorbar_label,
            )
        _reserve_publication_layout(figure)
        self.set_figure_title(
            figure,
            labeling.title_text(labels.quantity, operation="Correction Diagnostics"),
            y=0.985,
            subtitle_y=0.91,
        )
        _record_dense_artists(figure, dense_artists)
        return figure

    def validate_rendered_figures(
        self,
        figures: Sequence[tuple[str | None, matplotlib.figure.Figure]],
    ) -> dict[str, Any]:
        """Reject an AOD science suite that violates DAVINCI plot protocols."""
        failures: list[str] = []
        labels = [str(label) for label, _figure in figures]
        if labels != list(AOD_CORRECTION_FIGURES):
            failures.append(f"figure labels are {labels}, expected {list(AOD_CORRECTION_FIGURES)}")

        expected_cmap = get_geosit_aod_cmap()
        for label, figure in figures:
            logical_label = str(label)
            expected_count = _DENSE_ARTIST_COUNTS.get(logical_label)
            dense_artists = list(getattr(figure, "_davinci_dense_artists", []))
            if expected_count is None:
                failures.append(f"unexpected figure label {logical_label!r}")
            elif len(dense_artists) != expected_count:
                failures.append(
                    f"{logical_label} registered {len(dense_artists)} dense artists; "
                    f"expected {expected_count}"
                )
            for artist in dense_artists:
                if artist.get_rasterized() is not True:
                    failures.append(f"{logical_label} contains an unrasterized dense artist")

            title_artist = getattr(figure, "_suptitle", None)
            title = title_artist.get_text().strip() if title_artist is not None else ""
            if not title:
                failures.append(f"{logical_label} has no figure title")
            if re.search(r"\b(?:19|20)\d{2}\b", title):
                failures.append(f"{logical_label} puts a date in its figure title")
            if re.search(r"\bn\s*=", title, flags=re.IGNORECASE):
                failures.append(f"{logical_label} puts statistics in its figure title")
            if self.config.subtitle:
                subtitles = [text.get_text() for text in figure.texts if text is not title_artist]
                if self.config.subtitle not in subtitles:
                    failures.append(f"{logical_label} is missing the standardized date subtitle")

            absolute_artists = list(getattr(figure, "_davinci_absolute_aod_artists", []))
            if logical_label in {"annual_aod", "seasonal_aod"} and not absolute_artists:
                failures.append(f"{logical_label} did not register absolute-AOD artists")
            figure_boundaries: list[np.ndarray] = []
            for artist in absolute_artists:
                norm = artist.norm
                if not isinstance(norm, BoundaryNorm):
                    failures.append(f"{logical_label} does not use BoundaryNorm")
                    continue
                boundaries = np.asarray(norm.boundaries, dtype=float)
                figure_boundaries.append(boundaries)
                if artist.cmap.name != expected_cmap:
                    failures.append(
                        f"{logical_label} uses {artist.cmap.name!r}, expected {expected_cmap!r}"
                    )
                required_low_levels = {0.0, 0.005, 0.01, 0.02, 0.03, 0.04}
                if boundaries[-1] >= 0.05 and not required_low_levels.issubset(
                    set(np.round(boundaries, 6))
                ):
                    failures.append(f"{logical_label} is missing GEOSIT low-AOD levels")
                if boundaries[-1] > 1.0 + np.finfo(float).eps:
                    failures.append(f"{logical_label} exceeds the GEOSIT AOD cap of 1.0")
            if figure_boundaries:
                reference = figure_boundaries[0]
                for boundaries in figure_boundaries[1:]:
                    if not np.array_equal(boundaries, reference):
                        failures.append(
                            f"{logical_label} absolute-AOD panels do not share one contour scale"
                        )
                        break

            failures.extend(_layout_failures(logical_label, figure))

        timeseries = dict(figures).get("global_timeseries")
        if timeseries is not None:
            time_axes = timeseries.axes[:2]
            tick_labels = time_axes[-1].get_xticklabels() if time_axes else []
            if tick_labels and any(
                not np.isclose(float(tick.get_rotation()), 45.0) for tick in tick_labels
            ):
                failures.append("global_timeseries date labels are not rotated 45 degrees")

        if failures:
            raise PlottingError("AOD correction plot protocol failed: " + "; ".join(failures))
        return {
            "protocol": AOD_CORRECTION_PROTOCOL,
            "passed": True,
            "figures": labels,
            "checks": {
                "aod_style": "GEOSIT levels + turbo + BoundaryNorm",
                "aod_colorbar_ticks": "sparse GEOSIT reference ticks",
                "labeling": "DAVINCI centralized labels and subtitles",
                "rasterization": "all registered dense artists rasterized",
                "date_ticks": "45 degrees",
                "layout": "rendered text and ticks are unclipped and nonoverlapping",
            },
        }

    def render(
        self,
        series: list[PlotSeries],
        ax: Any = None,
        *,
        model_source: str = "merra2",
        corrected_source: str = "corrected_merra2",
        observation_source: str = "modis_aqua",
        max_scatter_points: int = 300_000,
        **kwargs: Any,
    ) -> list[tuple[str, matplotlib.figure.Figure]]:
        del ax, kwargs
        if max_scatter_points < 1:
            raise ValueError("max_scatter_points must be positive")
        inputs = correction_inputs(
            series,
            model_source=model_source,
            corrected_source=corrected_source,
            observation_source=observation_source,
        )
        labels = _labels_for_inputs(
            inputs,
            model_source=model_source,
            corrected_source=corrected_source,
            observation_source=observation_source,
        )
        return [
            ("annual_aod", self._annual_aod_figure(inputs, labels)),
            ("annual_bias_improvement", self._annual_bias_figure(inputs, labels)),
            ("seasonal_aod", self._seasonal_aod_figure(inputs, labels)),
            (
                "seasonal_error_reduction",
                self._seasonal_improvement_figure(inputs, labels),
            ),
            ("agreement_metrics", self._metrics_figure(inputs, labels)),
            ("global_timeseries", self._timeseries_figure(inputs, labels)),
            (
                "matched_scatter",
                self._scatter_figure(inputs, labels, max_points=max_scatter_points),
            ),
            ("correction_diagnostics", self._diagnostics_figure(inputs, labels)),
        ]


__all__ = ["AODCorrectionSciencePlotter"]
