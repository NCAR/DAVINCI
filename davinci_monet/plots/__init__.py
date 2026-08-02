"""Plotting module for DAVINCI.

This module provides a comprehensive plotting system for source comparison,
including time series, spatial maps, Taylor diagrams, and more.

Quick Start
-----------
>>> from davinci_monet.plots import build_series, get_plotter
>>>
>>> plotter = get_plotter("scatter")
>>> fig = plotter.render(build_series(paired_data, "x_o3", "y_o3"))

Available Plot Types
--------------------
Temporal:
    timeseries : Time series comparison
    diurnal : Diurnal cycle comparison

Statistical:
    scatter : Scatter plot with regression
    taylor : Taylor diagram
    boxplot : Box plot comparison

Spatial:
    spatial : Single-source spatial field map (shape-aware)
    spatial_bias : Bias map
    spatial_overlay : gridded field contour + point overlay

Specialized:
    curtain : Vertical cross-section
    scorecard : Multi-metric heatmap
"""

# Base classes and utilities
from davinci_monet.plots.base import (
    BasePlotter,
    DomainConfig,
    FigureConfig,
    PlotConfig,
    StyleConfig,
    TextConfig,
    build_series,
    calculate_data_limits,
    calculate_symmetric_limits,
    format_label_with_units,
    get_axis_color,
    get_series_label,
    get_variable_label,
    get_variable_units,
    merge_config_dicts,
    resolve_source_variable,
    series_colors,
    source_label,
)

# Registry and factory
from davinci_monet.plots.registry import (
    ALL_PLOT_TYPES,
    SPATIAL_PLOTS,
    SPECIALIZED_PLOTS,
    STATISTICAL_PLOTS,
    TEMPORAL_PLOTS,
    get_plot_category,
    get_plotter,
    get_plotter_class,
    has_plotter,
    list_plotters,
    plotter_registry,
    register_plotter,
)

# Import all renderers to ensure they register
from davinci_monet.plots.renderers import (  # Temporal; Statistical; Specialized; Spatial
    AODCorrectionSciencePlotter,
    BaseSpatialPlotter,
    BoxPlotter,
    CurtainPlotter,
    DiurnalPlotter,
    EOFPatternPlotter,
    EOFScreePlotter,
    EOFWaveletSciencePlotter,
    FlightTrackPlotter,
    HistogramPlotter,
    LMADensityPlotter,
    MapConfig,
    ScatterPlotter,
    ScorecardPlotter,
    SpatialBiasPlotter,
    SpatialOverlayPlotter,
    SpatialPlotter,
    TaylorPlotter,
    TimeSeriesPlotter,
    TrackMap3DPlotter,
    VerticalProfilePlotter,
    WaveletScalogramPlotter,
    get_domain_extent,
    get_projection,
)

# Style configuration (NCAR branding)
from davinci_monet.plots.style import (
    FONT_SIZES_DEFAULT,
    FONT_SIZES_PRESENTATION,
    FONT_SIZES_PUBLICATION,
    NCAR_ACCENT,
    NCAR_COLORS,
    NCAR_PALETTE,
    NCAR_PRIMARY,
    NCAR_SECONDARY,
    X_COLOR,
    Y_COLOR,
    FontSizes,
    apply_ncar_style,
    get_bias_cmap,
    get_color_for_variable,
    get_density_cmap,
    get_palette,
    get_sequential_cmap,
    reset_style,
)

__all__ = [
    # Base classes
    "BasePlotter",
    "PlotConfig",
    "FigureConfig",
    "TextConfig",
    "StyleConfig",
    "DomainConfig",
    # Utilities
    "merge_config_dicts",
    "get_variable_label",
    "resolve_source_variable",
    "get_variable_units",
    "format_label_with_units",
    "calculate_symmetric_limits",
    "calculate_data_limits",
    "build_series",
    "series_colors",
    "get_axis_color",
    "get_series_label",
    "source_label",
    # Style configuration (NCAR branding)
    "NCAR_COLORS",
    "NCAR_PALETTE",
    "NCAR_PRIMARY",
    "NCAR_SECONDARY",
    "NCAR_ACCENT",
    "X_COLOR",
    "Y_COLOR",
    "apply_ncar_style",
    "reset_style",
    "get_color_for_variable",
    "get_palette",
    "get_bias_cmap",
    "get_sequential_cmap",
    "get_density_cmap",
    "FontSizes",
    "FONT_SIZES_DEFAULT",
    "FONT_SIZES_PRESENTATION",
    "FONT_SIZES_PUBLICATION",
    # Registry
    "plotter_registry",
    "register_plotter",
    "get_plotter",
    "get_plotter_class",
    "list_plotters",
    "has_plotter",
    "get_plot_category",
    "TEMPORAL_PLOTS",
    "STATISTICAL_PLOTS",
    "SPATIAL_PLOTS",
    "SPECIALIZED_PLOTS",
    "ALL_PLOT_TYPES",
    # Temporal plotters
    "TimeSeriesPlotter",
    "DiurnalPlotter",
    # Statistical plotters
    "ScatterPlotter",
    "TaylorPlotter",
    "BoxPlotter",
    # Specialized plotters
    "AODCorrectionSciencePlotter",
    "CurtainPlotter",
    "EOFWaveletSciencePlotter",
    "ScorecardPlotter",
    "TrackMap3DPlotter",
    # Spatial plotters
    "BaseSpatialPlotter",
    "MapConfig",
    "SpatialPlotter",
    "SpatialBiasPlotter",
    "SpatialOverlayPlotter",
    "get_domain_extent",
    "get_projection",
    # Single-source / distribution plotters (unified onto BasePlotter)
    "HistogramPlotter",
    "VerticalProfilePlotter",
    "FlightTrackPlotter",
    "LMADensityPlotter",
    "EOFPatternPlotter",
    "EOFScreePlotter",
    # Wavelet analysis
    "WaveletScalogramPlotter",
]

# Importing plots.base loads the plots.plot_config submodule; keep it off the
# top-level package surface so dir(davinci_monet.plots) has no plot_* names.
globals().pop("plot_config", None)
