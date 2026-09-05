"""Base plotter classes and common utilities for DAVINCI.

This module provides:
- PlotConfig: Pydantic dataset for plot configuration
- BasePlotter: Abstract base class for all plotters
- Common utilities for figure creation, styling, and saving

Implementation note
-------------------
The config dataclasses live in ``plots.plot_config``, the text/label/unit
formatting functions and lookup tables live in ``plots.labels``, and the
series/color helpers live in ``plots.series``. Everything is re-exported here
for renderers, tests, pipeline stages, and examples.
"""

from __future__ import annotations

import math
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt

# Import NCAR style colors for defaults
from davinci_monet.core.base import PlotSeries

# Re-export label/formatting utilities
from davinci_monet.plots.labels import (
    TITLE_FORMULA_REPLACEMENTS,
    UNIT_REPLACEMENTS,
    VARIABLE_DISPLAY_NAMES,
    calculate_data_limits,
    calculate_symmetric_limits,
    canonical_variable_name,
    format_label_with_units,
    format_plot_title,
    format_units,
    format_variable_display_name,
    get_variable_label,
    get_variable_units,
    merge_config_dicts,
)

# Re-export config dataclasses
from davinci_monet.plots.plot_config import (
    DomainConfig,
    FigureConfig,
    PlotConfig,
    StyleConfig,
    TextConfig,
)

# Re-export series/color helpers
from davinci_monet.plots.series import (
    build_series,
    get_axis_color,
    get_series_label,
    resolve_source_variable,
    series_colors,
    source_label,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import numpy as np
    import xarray as xr


# =============================================================================
# Paired-series helpers (shared by statistical renderers)
# =============================================================================


def extract_xy_series(series: list[PlotSeries], who: str = "render") -> tuple[xr.Dataset, str, str]:
    """From a 2-element series list, return (paired_dataset, x_var, y_var) selecting
    by the 'x'/'y' axis attr (falling back to positional order).

    ``who`` is the caller's identifier (e.g. ``"ScatterPlotter.render"``) and is
    embedded in the error message raised when the series count is not 2, so the
    message matches each renderer's prior inline guard.
    """
    if len(series) != 2:
        raise NotImplementedError(f"{who} requires exactly 2 series; got {len(series)}.")
    x_series = next((s for s in series if s.axis == "x"), series[0])
    y_series = next((s for s in series if s.axis == "y"), series[1])
    return x_series.dataset, x_series.var_name, y_series.var_name


def clean_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Flatten and drop non-finite pairs; return (x_clean, y_clean)."""
    import numpy as np

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def set_ylabel_wrapped(
    ax: "matplotlib.axes.Axes",
    label: str,
    *,
    fontsize: float | None = None,
    fontweight: str | None = None,
    max_lines: int = 3,
) -> None:
    """Set a y-axis label, wrapping it if it is taller than the axes.

    A rotated y label longer than the axes it is centred on is silently
    truncated at BOTH ends by the Agg backend, and ``bbox_inches="tight"``
    does not rescue it -- the label simply loses characters, so
    ``"... Anomaly (W m^-2)"`` renders as ``"...Anomaly (W m"``. It is a
    quiet, publication-breaking failure: nothing errors and nothing warns.

    Long composed labels hit this routinely at ``context: presentation`` font
    sizes, where a quantity plus units can exceed the axes height. Wrapping
    onto extra lines preserves both the full text and the configured font
    size, which shrinking the font would not.
    """
    kwargs: dict[str, Any] = {}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    if fontweight is not None:
        kwargs["fontweight"] = fontweight

    ax.set_ylabel(label, **kwargs)
    if "\n" in label:
        return  # caller wrapped it deliberately; respect that

    available = _axes_height_px(ax)
    needed = _ylabel_height_px(ax)
    if available <= 0 or needed <= 0 or needed <= available:
        return

    import textwrap

    lines = min(max_lines, math.ceil(needed / available))
    width = max(1, math.ceil(len(label) / lines))
    ax.set_ylabel("\n".join(textwrap.wrap(label, width=width)), **kwargs)


def _axes_height_px(ax: "matplotlib.axes.Axes") -> float:
    try:
        return float(ax.get_window_extent().height)
    except Exception:  # noqa: BLE001 - measurement is best-effort
        return 0.0


def _ylabel_height_px(ax: "matplotlib.axes.Axes") -> float:
    """Rendered height of the current y label, or 0 if it cannot be measured."""
    fig = ax.get_figure()
    if fig is None or fig.canvas is None:
        return 0.0
    try:
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    except AttributeError:
        return 0.0  # non-raster backend: skip rather than guess
    try:
        return float(ax.yaxis.label.get_window_extent(renderer=renderer).height)
    except Exception:  # noqa: BLE001 - measurement is best-effort
        return 0.0


# =============================================================================
# Base Plotter
# =============================================================================


class BasePlotter(ABC):
    """Abstract base class for all plotters.

    Provides common functionality for figure management, styling,
    and saving. Subclasses implement specific plot types.

    Parameters
    ----------
    config
        Plot configuration. If None, uses defaults.

    Attributes
    ----------
    config : PlotConfig
        The plot configuration.
    default_figsize : tuple[float, float]
        Default figure size for this plotter type. Subclasses can override.
    """

    # Class-level registry name (override in subclasses)
    name: str = "base"

    # Default figure size - subclasses can override for optimal sizing
    default_figsize: tuple[float, float] = (8, 5)

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()
        # Apply plotter-specific default figsize if not explicitly set
        if self.config.figure.figsize == (8, 5):  # Original default
            self.config.figure.figsize = self.default_figsize

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure | list[tuple[str, matplotlib.figure.Figure]]:
        """Render a list of source series (unified renderer contract).

        Renderers implement this directly. Multi-figure renderers return
        ``[(label, figure), ...]`` so pipeline saving does not need
        renderer-specific split methods.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.render is not implemented for {len(series)} series."
        )

    def create_figure(
        self,
        config: FigureConfig | None = None,
        **kwargs: Any,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create a new figure and axes.

        Parameters
        ----------
        config
            Figure configuration. If None, uses self.config.figure.
        **kwargs
            Additional arguments passed to plt.subplots.

        Returns
        -------
        tuple[Figure, Axes]
            The created figure and axes.
        """
        cfg = config or self.config.figure
        fig_kwargs = {
            "figsize": cfg.figsize,
            "dpi": cfg.dpi,
            "facecolor": cfg.facecolor,
            "constrained_layout": cfg.constrained_layout,
            **kwargs,
        }
        fig, ax = plt.subplots(**fig_kwargs)
        return fig, ax

    def apply_text_style(
        self,
        ax: matplotlib.axes.Axes,
        config: TextConfig | None = None,
    ) -> None:
        """Apply text styling to axes.

        Parameters
        ----------
        ax
            Axes to style.
        config
            Text configuration. If None, uses self.config.text.
        """
        cfg = config or self.config.text
        ax.tick_params(axis="both", labelsize=cfg.tick_fontsize)

    def set_labels(
        self,
        ax: matplotlib.axes.Axes,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Set axis labels and title.

        Parameters
        ----------
        ax
            Axes to label.
        xlabel
            X-axis label. Uses config if None.
        ylabel
            Y-axis label. Uses config if None.
        title
            Plot title. Uses config if None.
        """
        cfg = self.config.text

        if xlabel or self.config.xlabel:
            ax.set_xlabel(
                xlabel or self.config.xlabel,  # type: ignore[arg-type]
                fontsize=cfg.fontsize,
                fontweight=cfg.fontweight,
            )

        if ylabel or self.config.ylabel:
            set_ylabel_wrapped(
                ax,
                ylabel or self.config.ylabel,  # type: ignore[arg-type]
                fontsize=cfg.fontsize,
                fontweight=cfg.fontweight,
            )

        self.set_title(ax, title=title)

    def set_title(
        self,
        ax: matplotlib.axes.Axes,
        title: str | None = None,
        subtitle: str | None = None,
        *,
        fontsize: float | None = None,
    ) -> None:
        """Set an axes title with an optional smaller subtitle below it."""
        cfg = self.config.text
        title_text = title or self.config.title
        subtitle_text = subtitle if subtitle is not None else self.config.subtitle

        if title or self.config.title:
            formatted_title = format_plot_title(title_text)  # type: ignore[arg-type]
            ax.set_title(
                formatted_title,
                fontsize=fontsize or cfg.title_fontsize,
                fontweight=cfg.fontweight,
                pad=22 if subtitle_text else None,
                wrap=True,
            )
        if subtitle_text:
            ax.text(
                0.5,
                1.01,
                subtitle_text,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=cfg.annotation_small,
                color="#58595B",
                clip_on=False,
            )

    def set_figure_title(
        self,
        fig: matplotlib.figure.Figure,
        title: str | None = None,
        subtitle: str | None = None,
        *,
        y: float = 0.95,
        fontsize: float | None = None,
    ) -> None:
        """Set a figure title with an optional smaller subtitle below it."""
        cfg = self.config.text
        title_text = title or self.config.title
        subtitle_text = subtitle if subtitle is not None else self.config.subtitle

        if title_text:
            fig.suptitle(
                format_plot_title(title_text),
                fontsize=fontsize or cfg.title_fontsize,
                y=y,
            )
        if subtitle_text:
            fig.text(
                0.5,
                y - 0.04,
                subtitle_text,
                ha="center",
                va="top",
                fontsize=cfg.annotation_small,
                color="#58595B",
            )

    def set_limits(
        self,
        ax: matplotlib.axes.Axes,
        vmin: float | None = None,
        vmax: float | None = None,
        axis: Literal["x", "y", "both"] = "y",
    ) -> None:
        """Set axis limits.

        Parameters
        ----------
        ax
            Axes to configure.
        vmin
            Minimum value. Uses config if None.
        vmax
            Maximum value. Uses config if None.
        axis
            Which axis to set ('x', 'y', or 'both').
        """
        vmin = vmin if vmin is not None else self.config.vmin
        vmax = vmax if vmax is not None else self.config.vmax

        if vmin is not None or vmax is not None:
            if axis in ("y", "both"):
                ax.set_ylim(vmin, vmax)
            if axis in ("x", "both"):
                ax.set_xlim(vmin, vmax)

    def add_legend(
        self,
        ax: matplotlib.axes.Axes,
        loc: str = "best",
        **kwargs: Any,
    ) -> None:
        """Add legend to axes.

        Parameters
        ----------
        ax
            Axes to add legend to.
        loc
            Legend location.
        **kwargs
            Additional legend arguments.
        """
        legend_kwargs = {
            "loc": loc,
            "fontsize": self.config.text.legend,
            "framealpha": 0.9,
            **kwargs,
        }
        ax.legend(**legend_kwargs)

    def save(
        self,
        fig: matplotlib.figure.Figure,
        output_path: str | Path,
        dpi: int | None = None,
        bbox_inches: str = "tight",
        **kwargs: Any,
    ) -> Path:
        """Save figure to file.

        Parameters
        ----------
        fig
            Figure to save.
        output_path
            Output file path.
        dpi
            Resolution. Uses config if None.
        bbox_inches
            Bounding box setting.
        **kwargs
            Additional savefig arguments.

        Returns
        -------
        Path
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {
            "dpi": dpi or self.config.figure.dpi,
            "bbox_inches": bbox_inches,
            "facecolor": fig.get_facecolor(),
            **kwargs,
        }
        fig.savefig(output_path, **save_kwargs)
        return output_path

    def close(self, fig: matplotlib.figure.Figure) -> None:
        """Close a figure to free memory.

        Parameters
        ----------
        fig
            Figure to close.
        """
        plt.close(fig)


# =============================================================================
# Public re-export surface  (keep __all__ in sync with what callers expect)
# =============================================================================

__all__ = [
    # Config dataclasses
    "FigureConfig",
    "TextConfig",
    "StyleConfig",
    "DomainConfig",
    "PlotConfig",
    # Base ABC
    "BasePlotter",
    # Paired-series helpers
    "extract_xy_series",
    "clean_xy",
    "set_ylabel_wrapped",
    # Label/formatting utilities
    "VARIABLE_DISPLAY_NAMES",
    "TITLE_FORMULA_REPLACEMENTS",
    "UNIT_REPLACEMENTS",
    "format_plot_title",
    "format_variable_display_name",
    "canonical_variable_name",
    "get_variable_label",
    "get_variable_units",
    "format_units",
    "format_label_with_units",
    "calculate_symmetric_limits",
    "calculate_data_limits",
    "merge_config_dicts",
    # Series/color helpers
    "build_series",
    "series_colors",
    "get_axis_color",
    "source_label",
    "get_series_label",
    "resolve_source_variable",
]
