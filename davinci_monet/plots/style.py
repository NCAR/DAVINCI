"""Plot styling configuration for DAVINCI.

This module provides a standardized styling system based on the official
NSF NCAR brand guidelines. It includes:

- NCAR brand color palette
- Font configuration (Poppins with fallbacks)
- Font size presets for presentations and publications
- Functions to apply styling globally

Usage
-----
>>> from davinci_monet.plots.style import apply_ncar_style
>>>
>>> # Apply NCAR styling globally (call once at start of script)
>>> apply_ncar_style()
>>>
>>> # Then create plots as usual
>>> from davinci_monet.plots import TimeSeriesPlotter, build_series
>>> fig = TimeSeriesPlotter().render(build_series(paired_data, "x_o3", "y_o3"))

The apply_ncar_style() function sets matplotlib rcParams globally, affecting
all subsequent plots. For individual plots, you can also pass style options
directly through PlotConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# NSF NCAR Brand Colors
# =============================================================================

# Official NSF NCAR brand colors from the style guide
# https://brand.ucar.edu/

NCAR_COLORS = {
    # Primary colors
    "space": "#011837",  # Dark backgrounds
    "dark_blue": "#00357A",  # Secondary blue
    "ncar_blue": "#0A5DDA",  # Primary brand color
    "aqua": "#00A2B4",  # UCAR Aqua - accent cyan
    # Light colors
    "light_blue": "#CEDFF8",  # Light accents
    "light_gray": "#F1F0EE",  # Backgrounds
    # Accent colors
    "orange": "#FF8C00",  # Accent (dark orange)
    "yellow": "#FFDD31",  # Accent
    "gray": "#58595B",  # Gray
    # Additional colors for data visualization
    "red": "#D62839",  # Error/negative bias
    "green": "#2E8B57",  # Positive/good
    "purple": "#7B68EE",  # Alternative accent
}

# Convenience aliases for common use cases
NCAR_PRIMARY = NCAR_COLORS["ncar_blue"]
NCAR_SECONDARY = NCAR_COLORS["aqua"]
NCAR_ACCENT = NCAR_COLORS["orange"]

# Color scheme for x-y comparisons.
X_COLOR = NCAR_COLORS["gray"]  # x-axis values in neutral gray
Y_COLOR = NCAR_COLORS["ncar_blue"]  # y-axis values in primary blue
BIAS_POSITIVE_COLOR = NCAR_COLORS["red"]  # y is higher than x
BIAS_NEGATIVE_COLOR = NCAR_COLORS["ncar_blue"]  # y is lower than x

# Sequential color palette for multiple plotted series.
NCAR_PALETTE = [
    NCAR_COLORS["ncar_blue"],
    NCAR_COLORS["aqua"],
    NCAR_COLORS["orange"],
    NCAR_COLORS["purple"],
    NCAR_COLORS["green"],
    NCAR_COLORS["red"],
    NCAR_COLORS["yellow"],
    NCAR_COLORS["dark_blue"],
]


# =============================================================================
# Font Configuration
# =============================================================================

# Official NSF NCAR brand font is Poppins (with fallbacks)
NCAR_FONT_FAMILY = "sans-serif"
NCAR_FONT_SANS_SERIF = [
    "Poppins",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
]


@dataclass
class FontSizes:
    """Font size configuration for different contexts.

    All sizes are in points (1/72 inch). These are absolute measurements
    that do not scale with figure size - 12pt text is always 12pt regardless
    of whether the figure is 6" or 12" wide.

    Attributes
    ----------
    figure_title : float
        Font size for figure suptitle (overall title).
    axes_title : float
        Font size for subplot/panel titles.
    axes_label : float
        Font size for axis labels (xlabel, ylabel).
    tick_label : float
        Font size for tick labels (numbers on axes).
    legend : float
        Font size for primary legend text.
    legend_small : float
        Font size for crowded multi-panel legends.
    annotation : float
        Font size for text annotations (stats boxes, etc.).
    annotation_small : float
        Font size for dense multi-panel annotations.
    site_label : float
        Font size for map site markers and city labels.
    """

    figure_title: float = 20.0
    axes_title: float = 16.0
    axes_label: float = 14.0
    tick_label: float = 12.0
    legend: float = 12.0
    legend_small: float = 10.0
    annotation: float = 12.0
    annotation_small: float = 10.0
    site_label: float = 10.0


# Preset font sizes for different contexts
FONT_SIZES_PRESENTATION = FontSizes(
    figure_title=24.0,
    axes_title=18.0,
    axes_label=16.0,
    tick_label=14.0,
    legend=14.0,
    legend_small=12.0,
    annotation=14.0,
    annotation_small=12.0,
    site_label=12.0,
)

FONT_SIZES_PUBLICATION = FontSizes(
    figure_title=18.0,
    axes_title=14.0,
    axes_label=12.0,
    tick_label=10.0,
    legend=10.0,
    legend_small=9.0,
    annotation=10.0,
    annotation_small=9.0,
    site_label=9.0,
)

FONT_SIZES_DEFAULT = FontSizes()


# =============================================================================
# Style Application Functions
# =============================================================================


def apply_ncar_style(
    context: Literal["default", "presentation", "publication"] = "default",
    use_seaborn: bool = True,
    seaborn_style: str = "whitegrid",
) -> None:
    """Apply NCAR brand styling to matplotlib globally.

    This function configures matplotlib rcParams with NCAR brand fonts,
    colors, and sizes. Call it once at the beginning of your script
    to apply consistent styling to all subsequent plots.

    Parameters
    ----------
    context
        Preset context for font sizes:
        - "default": Standard sizes suitable for most uses
        - "presentation": Larger sizes for slides
        - "publication": Smaller sizes for journal figures
    use_seaborn
        If True and seaborn is available, apply seaborn theme for
        cleaner grid styling.
    seaborn_style
        Seaborn style to apply if use_seaborn is True.
        Options: "whitegrid", "darkgrid", "white", "dark", "ticks"

    Examples
    --------
    >>> # Apply default NCAR styling
    >>> apply_ncar_style()
    >>>
    >>> # Apply styling for a presentation
    >>> apply_ncar_style(context="presentation")
    >>>
    >>> # Apply without seaborn
    >>> apply_ncar_style(use_seaborn=False)
    """
    # Select font sizes based on context
    if context == "presentation":
        sizes = FONT_SIZES_PRESENTATION
    elif context == "publication":
        sizes = FONT_SIZES_PUBLICATION
    else:
        sizes = FONT_SIZES_DEFAULT

    # Apply seaborn style if requested
    if use_seaborn:
        try:
            import seaborn as sns

            sns.set_theme(style=seaborn_style, palette="deep")
        except ImportError:
            pass  # seaborn not available, continue without it

    # Font family
    plt.rcParams["font.family"] = NCAR_FONT_FAMILY
    plt.rcParams["font.sans-serif"] = NCAR_FONT_SANS_SERIF
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    # Font sizes
    plt.rcParams["axes.labelsize"] = sizes.axes_label
    plt.rcParams["axes.titlesize"] = sizes.axes_title
    plt.rcParams["xtick.labelsize"] = sizes.tick_label
    plt.rcParams["ytick.labelsize"] = sizes.tick_label
    plt.rcParams["legend.fontsize"] = sizes.legend
    plt.rcParams["figure.titlesize"] = sizes.figure_title

    # Colors
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=NCAR_PALETTE)

    # Line and marker defaults
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["lines.markersize"] = 6

    # Grid styling
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "-"

    # Figure defaults
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def reset_style() -> None:
    """Reset matplotlib to default styling.

    Restores matplotlib rcParams to their default values, undoing
    any changes made by apply_ncar_style().
    """
    plt.rcdefaults()


def get_color_for_variable(variable: str) -> str:
    """Get a color appropriate for a variable type.

    Parameters
    ----------
    variable
        Variable name or type. Recognized prefixes:
        - "x_*": Returns x-source color
        - "y_*": Returns y-source color
        - "bias_*": Returns bias color

    Returns
    -------
    str
        Hex color code.
    """
    var_lower = variable.lower()
    if var_lower.startswith("x_"):
        return X_COLOR
    elif var_lower.startswith("y_"):
        return Y_COLOR
    elif var_lower.startswith("bias"):
        return NCAR_COLORS["red"]
    else:
        return NCAR_COLORS["ncar_blue"]


def get_palette(n_colors: int | None = None) -> list[str]:
    """Get a list of colors from the NCAR palette.

    Parameters
    ----------
    n_colors
        Number of colors to return. If None, returns all palette colors.
        If n_colors > len(NCAR_PALETTE), colors will cycle.

    Returns
    -------
    list[str]
        List of hex color codes.
    """
    if n_colors is None:
        return list(NCAR_PALETTE)

    colors = []
    for i in range(n_colors):
        colors.append(NCAR_PALETTE[i % len(NCAR_PALETTE)])
    return colors


# =============================================================================
# Colormap Utilities
# =============================================================================


def get_bias_cmap() -> str:
    """Get the recommended colormap for bias plots.

    Returns a diverging colormap centered on zero, with blue for
    negative bias (y lower than x) and red for positive bias
    (y higher than x).

    Returns
    -------
    str
        Matplotlib colormap name.
    """
    return "RdBu_r"


def get_sequential_cmap() -> str:
    """Get the recommended colormap for sequential data.

    Returns
    -------
    str
        Matplotlib colormap name.
    """
    return "viridis"


def geosit_aod_levels(
    field_values: object,
    *,
    cap: float = 1.0,
    base_step: float = 0.05,
) -> np.ndarray:
    """Return the GEOSIT AOD contour levels for a field.

    The reference GEOSIT AOD plots use 0.05 bins up to a plotted cap of 1.0,
    with extra low-end detail at 0.005, 0.01, 0.02, 0.03, and 0.04.
    """
    values = np.asarray(field_values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(0.0, cap, 11)

    vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= 0:
        return np.linspace(0.0, cap, 11)

    capped = min(vmax, cap)
    upper = np.ceil(capped / base_step) * base_step
    upper = min(upper, cap)
    levels = np.arange(0.0, upper + base_step * 0.5, base_step)
    augment = [
        level for level in (0.005, 0.01, 0.02, 0.03, 0.04) if level < base_step and level < upper
    ]
    if augment:
        levels = np.unique(np.concatenate([levels, np.asarray(augment, dtype=float)]))
        levels.sort()
    return levels


def get_geosit_aod_cmap() -> str:
    """Get the GEOSIT AOD color table name."""
    return "turbo"


def get_density_cmap() -> str:
    """Get the recommended colormap for density plots.

    Returns
    -------
    str
        Matplotlib colormap name.
    """
    return "viridis"
