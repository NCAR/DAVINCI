"""Protocol definitions for DAVINCI components.

This module defines the interfaces (Protocols) for the pluggable data-source and
pairing components. Using runtime_checkable Protocols enables static type
checking while maintaining flexibility for the plugin architecture.

Only the contracts that real implementations actually conform to live here.
Plotters, metrics, and the statistics calculator are concrete base classes in
their own subpackages (``plots.base.BasePlotter``, ``stats.metrics.BaseMetric``,
``stats.calculator.StatisticsCalculator``) and are not modelled as protocols.

Data Geometry Types:
    - point: Fixed locations (surface stations, ground sites)
    - track: 3D trajectories (aircraft, mobile platforms)
    - profile: Vertical profiles (sondes, lidar)
    - swath: 2D satellite footprints (L2 products)
    - grid: Regular gridded data (L3 products, reanalysis)
    - spectrum: Time-frequency analysis output (not pairable)
    - artifact: Manifest-only analysis output (not pairable)
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    import xarray as xr


class DataGeometry(Enum):
    """Enumeration of supported data geometry types.

    The geometry type determines which pairing strategy will be used
    to match observations with model output.
    """

    POINT = auto()
    """Fixed point locations (time, site) - surface stations, ground sites."""

    TRACK = auto()
    """3D trajectory (time,) with lat/lon/alt coords - aircraft, mobile platforms."""

    PROFILE = auto()
    """Vertical profile (time, level) with lat/lon coords - sondes, lidar."""

    SWATH = auto()
    """2D satellite swath (time, scanline, pixel) - L2 products."""

    GRID = auto()
    """Regular grid (time, lat, lon) - L3 products, reanalysis, model output."""

    SPECTRUM = auto()
    """Time-frequency spectrum (time, period) - wavelet power. Not pairable."""

    ARTIFACT = auto()
    """Manifest-only analysis output. Not pairable or implicitly plottable."""


# =============================================================================
# Unified Source Protocols
# =============================================================================
#
# A data source is just data of a given geometry (point, track, profile,
# swath, grid). All data sources are distinguished only by topology, not
# origin. An optional ``pair_axis`` tag may travel as metadata for labeling
# and styling, but it never appears in these contracts.


@runtime_checkable
class SourceReader(Protocol):
    """Protocol for data source readers.

    Every source reader declares the geometry it produces and loads files into
    a standardized xarray Dataset whose ``attrs['geometry']`` is set. This is
    the unified reader interface for all source geometries.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this source type (e.g. 'cesm_fv', 'pt_sfc')."""
        ...

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """The data geometry this reader produces."""
        ...

    @abstractmethod
    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open source files and return a standardized Dataset.

        Parameters
        ----------
        file_paths
            Paths to source files (can include glob patterns).
        variables
            Optional list of variables to load. If None, load all.
        time_range
            Optional (start, end) time range to subset.
        **kwargs
            Additional reader-specific options.

        Returns
        -------
        xr.Dataset
            Source data with geometry-appropriate dimensions and the
            ``geometry`` attribute set.
        """
        ...


@runtime_checkable
class SourceProcessor(Protocol):
    """Protocol for data source post-processing operations.

    Processors handle
    unit conversion, vertical-coordinate handling, resampling, QA/QC,
    subsetting, and aggregation, composed into one chain regardless of origin.
    """

    @abstractmethod
    def process(self, dataset: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply processing to a source Dataset and return the result."""
        ...


# =============================================================================
# Pairing Protocols
# =============================================================================


@runtime_checkable
class PairingStrategy(Protocol):
    """Protocol for source-pairing strategies.

    Concrete strategies expose ``pair_sources(x_data, y_data, **kwargs)`` and
    return a single paired ``xr.Dataset``. By contract a strategy emits the x
    side under its bare variable name and the y side under a ``y_`` prefix; it
    sets **no** ``axis``/``source_label`` attrs. The :class:`PairingEngine`
    (``pairing.engine``) is the sole writer of those attrs and the sole point
    that relabels variables to the public ``<source_label>_<var>`` form.

    Option contract
    ---------------
    The engine passes these options (some by name, some via ``**kwargs``); a
    strategy must accept all of them and may ignore any it does not use:

    - ``radius_of_influence``, ``time_tolerance``, ``vertical_method``,
      ``horizontal_method``, ``time_method`` (by name);
    - ``x_vars``/``y_vars`` and the single-variable ``x_var``/``y_var``
      (via ``**kwargs``).

    Caveats (currently true, not yet unified):

    - ``time_method`` is honored **only** by ``PointStrategy``; every other
      geometry uses nearest-time alignment regardless of its value.
    - ``method: grid`` is a separate symmetric-binning route in the engine
      (``IntermediateGridStrategy``) that does **not** flow through this protocol's
      option set or the temporal-overlap guard; it is driven by ``time_resolution``.

    Concrete strategy classes may keep an internal ``pair(dataset, geometry,
    ...)`` method that ``pair_sources`` delegates to, but it is not part of this
    public contract.
    """

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """The data geometry this strategy handles."""
        ...

    @abstractmethod
    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair two sources, returning a dataset of x (bare) and y_ vars."""
        ...
