"""Plot renderers for DAVINCI.

This subpackage provides individual plot type implementations:

Temporal plots:
- TimeSeriesPlotter: Time series comparisons
- DiurnalPlotter: Diurnal cycle plots

Statistical plots:
- ScatterPlotter: Scatter plots with regression
- TaylorPlotter: Taylor diagrams
- BoxPlotter: Box plot comparisons

Spatial plots:
- SpatialBiasPlotter: Bias maps
- SpatialOverlayPlotter: y/x overlays
- SpatialPlotter: Single-source spatial field maps

Specialized plots:
- CurtainPlotter: Vertical cross-sections
- ScorecardPlotter: Multi-metric scorecards
"""

# Specialized plots
from davinci_monet.plots.renderers.aod_correction_science import AODCorrectionSciencePlotter
from davinci_monet.plots.renderers.boxplot import BoxPlotter
from davinci_monet.plots.renderers.curtain import CurtainPlotter
from davinci_monet.plots.renderers.diurnal import DiurnalPlotter
from davinci_monet.plots.renderers.eof_pattern import EOFPatternPlotter
from davinci_monet.plots.renderers.eof_scree import EOFScreePlotter
from davinci_monet.plots.renderers.eof_wavelet_science import EOFWaveletSciencePlotter
from davinci_monet.plots.renderers.flight_track import FlightTrackPlotter
from davinci_monet.plots.renderers.histogram import HistogramPlotter
from davinci_monet.plots.renderers.lma_density import LMADensityPlotter

# Statistical plots
from davinci_monet.plots.renderers.scatter import ScatterPlotter
from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

# Spatial plots (imported from subpackage)
from davinci_monet.plots.renderers.spatial import (
    BaseSpatialPlotter,
    MapConfig,
    SpatialBiasPlotter,
    SpatialOverlayPlotter,
    SpatialPlotter,
    get_domain_extent,
    get_projection,
)
from davinci_monet.plots.renderers.taylor import TaylorPlotter

# Temporal plots
from davinci_monet.plots.renderers.timeseries import TimeSeriesPlotter
from davinci_monet.plots.renderers.track_map_3d import TrackMap3DPlotter
from davinci_monet.plots.renderers.vertical_profile import VerticalProfilePlotter
from davinci_monet.plots.renderers.wavelet_scalogram import WaveletScalogramPlotter

__all__ = [
    # Temporal
    "TimeSeriesPlotter",
    "DiurnalPlotter",
    # Statistical
    "ScatterPlotter",
    "TaylorPlotter",
    "BoxPlotter",
    # Specialized
    "AODCorrectionSciencePlotter",
    "CurtainPlotter",
    "ScorecardPlotter",
    "TrackMap3DPlotter",
    # Spatial
    "BaseSpatialPlotter",
    "MapConfig",
    "SpatialBiasPlotter",
    "SpatialOverlayPlotter",
    "SpatialPlotter",
    "get_domain_extent",
    "get_projection",
    # Single-source / distribution plotters
    "HistogramPlotter",
    "VerticalProfilePlotter",
    "FlightTrackPlotter",
    "LMADensityPlotter",
    # EOF analysis
    "EOFPatternPlotter",
    "EOFScreePlotter",
    "EOFWaveletSciencePlotter",
    # Wavelet analysis
    "WaveletScalogramPlotter",
]
