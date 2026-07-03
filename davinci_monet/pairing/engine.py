"""Unified pairing engine for source matching.

This module provides the main pairing orchestrator that dispatches to
geometry-specific strategies based on the x and y source geometries.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.base import PairedData
from davinci_monet.core.exceptions import GeometryMismatchError, NoOverlapError, PairingError
from davinci_monet.core.protocols import DataGeometry, PairingStrategy
from davinci_monet.core.types import TimeDelta


@dataclass
class PairingConfig:
    """Configuration for pairing operations.

    Attributes
    ----------
    radius_of_influence : float
        Spatial search radius in meters.
    time_tolerance : TimeDelta | None
        Maximum time difference for matching.
    vertical_method : str
        Vertical interpolation method ('nearest', 'linear', 'log').
    horizontal_method : str
        Horizontal interpolation method ('nearest', 'bilinear').
    time_method : str
        Time interpolation method ('nearest', 'linear'). Use 'linear' for
        datasets with sparse time output (e.g. 6-hourly WRF-Chem) to avoid
        step-function artifacts in the paired time series.
    apply_averaging_kernel : bool
        Whether to apply satellite averaging kernels.
    require_overlap : bool
        Whether to require temporal overlap.
    """

    radius_of_influence: float = 12000.0
    time_tolerance: TimeDelta | None = None
    vertical_method: str = "nearest"
    horizontal_method: str = "nearest"
    time_method: str = "nearest"
    apply_averaging_kernel: bool = False
    require_overlap: bool = True


class PairingEngine:
    """Unified pairing engine that dispatches to geometry-specific strategies.

    The engine automatically selects the appropriate pairing strategy based
    on the geometry data's geometry attribute.

    Examples
    --------
    >>> engine = PairingEngine()
    >>> paired = engine.pair_sources(
    ...     x_data=x_data,
    ...     y_data=y_data,
    ...     x_vars=["O3"],
    ...     y_vars=["OZONE"],
    ... )
    """

    def __init__(self, register_defaults: bool = True) -> None:
        """Initialize pairing engine.

        Parameters
        ----------
        register_defaults
            If True, register default strategies for all geometries.
        """
        self._strategies: dict[DataGeometry, PairingStrategy] = {}

        if register_defaults:
            self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register the default strategies for each geometry type."""
        # Import here to avoid circular imports
        from davinci_monet.pairing.strategies import (
            GridStrategy,
            IntermediateGridStrategy,
            PointStrategy,
            ProfileStrategy,
            TrackStrategy,
        )

        self.register_strategy(PointStrategy())
        self.register_strategy(TrackStrategy())
        self.register_strategy(ProfileStrategy())
        # IntermediateGridStrategy (numba binning) is the production SWATH handler.
        # SwathStrategy (per-pixel nearest-neighbor) is preserved for direct use
        # but is intentionally NOT the engine default.
        self.register_strategy(IntermediateGridStrategy())
        self.register_strategy(GridStrategy())

    def register_strategy(self, strategy: PairingStrategy) -> None:
        """Register a pairing strategy for a geometry type.

        The registry maps each :class:`DataGeometry` to exactly one strategy.
        Registering a *different* strategy class for a geometry that is already
        mapped warns (and keeps last-wins) rather than silently shadowing, so a
        collision — e.g. ``SwathStrategy`` and ``IntermediateGridStrategy`` both
        claiming ``SWATH`` — is visible. Re-registering the same class is silent.

        Parameters
        ----------
        strategy
            Strategy instance implementing PairingStrategy protocol.
        """
        existing = self._strategies.get(strategy.geometry)
        if existing is not None and type(existing) is not type(strategy):
            warnings.warn(
                f"Overwriting pairing strategy for geometry {strategy.geometry.name}: "
                f"{type(existing).__name__} -> {type(strategy).__name__}. Each geometry "
                f"maps to exactly one strategy; the last registration wins.",
                UserWarning,
                stacklevel=2,
            )
        self._strategies[strategy.geometry] = strategy

    def get_strategy(self, geometry: DataGeometry) -> PairingStrategy:
        """Get the strategy for a given geometry.

        Parameters
        ----------
        geometry
            The data geometry type.

        Returns
        -------
        PairingStrategy
            The registered strategy.

        Raises
        ------
        PairingError
            If no strategy registered for geometry.
        """
        if geometry not in self._strategies:
            raise PairingError(
                f"No pairing strategy registered for geometry {geometry.name}. "
                f"Available: {[g.name for g in self._strategies.keys()]}"
            )
        return self._strategies[geometry]

    def supported_pairing_combinations(self) -> set[tuple[DataGeometry, DataGeometry]]:
        """Return supported ``(x, y)`` geometry combinations."""
        return {(geometry, DataGeometry.GRID) for geometry in self._strategies}

    def supports_pairing_combination(
        self,
        geometry: DataGeometry,
        y_geometry: DataGeometry,
    ) -> bool:
        """Return whether the engine can pair this geometry combination."""
        return (
            geometry,
            y_geometry,
        ) in self.supported_pairing_combinations()

    def get_strategy_for(
        self,
        geometry: DataGeometry,
        y_geometry: DataGeometry,
    ) -> PairingStrategy:
        """Get the strategy for an ``(x, y)`` geometry pair.

        The dataset is resampled onto the geometry's geometry. Supported
        combinations are a GRID dataset sampled onto any registered geometry
        geometry (POINT/TRACK/PROFILE/SWATH/GRID). The geometry geometry selects
        the strategy.

        Parameters
        ----------
        geometry
            Geometry of the x source (sampled *onto*).
        y_geometry
            Geometry of the y source (sampled *from*).

        Returns
        -------
        PairingStrategy
            The strategy handling this combination.

        Raises
        ------
        PairingError
            If the combination is not supported.
        """
        if not self.supports_pairing_combination(geometry, y_geometry):
            raise PairingError(
                f"Unsupported pairing combination "
                f"(geometry={geometry.name}, dataset={y_geometry.name}). "
                f"Supported combinations sample a GRID dataset onto a "
                f"{[g.name for g in self._strategies.keys()]} geometry."
            )
        return self.get_strategy(geometry)

    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        x_vars: Sequence[str],
        y_vars: Sequence[str],
        output_geometry: DataGeometry | None = None,
        y_geometry: DataGeometry | None = None,
        config: PairingConfig | None = None,
        x_source: str = "x",
        y_source: str = "y",
        method: str = "auto",
        **kwargs: Any,
    ) -> PairedData:
        """Pair two sources.

        ``y_data`` is sampled onto ``x_data``. The paired output uses
        ``<source_label>_<dataset_variable>`` variable names with ``axis``,
        ``source_label``, and ``dataset_variable`` attrs.

        When ``method == "grid"`` the engine bypasses geometry auto-selection
        and routes both sources through ``IntermediateGridStrategy``'s
        symmetric path, binning each onto a common ``(time, lon, lat)`` grid.
        """
        if config is None:
            config = PairingConfig()

        if method == "grid":
            from davinci_monet.pairing.strategies.intermediate_grid import (
                IntermediateGridStrategy,
            )

            grid_strategy = IntermediateGridStrategy()
            grid_paired_ds = grid_strategy.pair_sources(
                x_data=x_data,
                y_data=y_data,
                x_var=str(x_vars[0]) if x_vars else None,
                y_var=str(y_vars[0]) if y_vars else None,
                x_source=x_source,
                y_source=y_source,
                horizontal_res=kwargs.get("horizontal_res"),
                extent=kwargs.get("extent"),
                time_resolution=kwargs.get("time_resolution", "1D"),
                min_sample_count=kwargs.get("min_sample_count", 1),
                vertical=kwargs.get("vertical"),
            )
            # Route through the shared relabeler so grid pairs get the public
            # ``<source_label>_<var>`` names and axis/source_label/canonical_name
            # attrs, exactly like the geometry-based path. (Diagnostic
            # ``*_sample_count`` vars are intentionally not carried to output.)
            result_ds = self._assemble_paired_dataset(
                grid_paired_ds,
                x_vars=x_vars,
                y_vars=y_vars,
                x_source=x_source,
                y_source=y_source,
            )
            return PairedData.from_sources(
                data=result_ds,
                x_source=x_source,
                y_source=y_source,
                geometry=DataGeometry.GRID,
                pairing_info={
                    "x_source": x_source,
                    "source_label": y_source,
                    "geometry": DataGeometry.GRID.name,
                    "y_geometry": DataGeometry.GRID.name,
                    "method": "grid",
                    "strategy": grid_strategy.__class__.__name__,
                },
            )

        if output_geometry is None:
            output_geometry = self._detect_geometry(x_data)
        if y_geometry is None:
            y_geometry = self._detect_geometry(y_data)

        if config.require_overlap:
            self._check_temporal_overlap(y_data, x_data)

        strategy = self.get_strategy_for(output_geometry, y_geometry)
        paired_ds = strategy.pair_sources(
            x_data=x_data,
            y_data=y_data,
            radius_of_influence=config.radius_of_influence,
            time_tolerance=config.time_tolerance,
            vertical_method=config.vertical_method,
            horizontal_method=config.horizontal_method,
            time_method=config.time_method,
            x_vars=list(x_vars),
            y_vars=list(y_vars),
            x_var=str(x_vars[0]) if x_vars else None,
            y_var=str(y_vars[0]) if y_vars else None,
            **kwargs,
        )
        result_ds = self._assemble_paired_dataset(
            paired_ds,
            x_vars=x_vars,
            y_vars=y_vars,
            x_source=x_source,
            y_source=y_source,
        )
        return PairedData.from_sources(
            data=result_ds,
            x_source=x_source,
            y_source=y_source,
            geometry=output_geometry,
            pairing_info={
                "x_source": x_source,
                "source_label": y_source,
                "geometry": output_geometry.name,
                "y_geometry": y_geometry.name,
                "radius_of_influence": config.radius_of_influence,
                "time_tolerance": config.time_tolerance,
                "vertical_method": config.vertical_method,
                "horizontal_method": config.horizontal_method,
                "strategy": strategy.__class__.__name__,
            },
        )

    @staticmethod
    def _select_var(ds: xr.Dataset, candidates: Sequence[str]) -> str | None:
        """Return the first matching data variable name from candidates."""
        for name in candidates:
            if name in ds.data_vars:
                return name
        return None

    def _assemble_paired_dataset(
        self,
        paired_ds: xr.Dataset,
        x_vars: Sequence[str],
        y_vars: Sequence[str],
        x_source: str,
        y_source: str,
    ) -> xr.Dataset:
        """Build a source-label paired dataset from strategy output.

        Contract: strategies emit the x side under its bare variable name (or an
        ``x_`` prefix for gridded strategies) and the y side under a ``y_``
        prefix. The ``y_``-prefixed candidates are therefore resolved *before*
        the bare ``y_name`` fallback so that a same-named x variable is never
        mistaken for the y series. A pair whose x or y variable cannot be
        resolved is a strategy/contract violation and raises rather than being
        silently dropped.
        """
        data_vars: dict[str, xr.DataArray] = {}
        coords = dict(paired_ds.coords)

        for x_var, y_var in zip(x_vars, y_vars):
            x_name = str(x_var)
            y_name = str(y_var)
            x_key = self._select_var(
                paired_ds,
                [x_name, f"x_{x_name}"],
            )
            y_key = self._select_var(
                paired_ds,
                [
                    f"y_{y_name}",
                    f"y_{x_name}",
                ],
            )

            if x_key is None or y_key is None:
                raise PairingError(
                    f"Could not resolve paired variables for x={x_name!r}, "
                    f"y={y_name!r} in strategy output {list(paired_ds.data_vars)}. "
                    "Strategies must emit x bare (or x_-prefixed) and y as y_<var>."
                )

            x_output = f"{x_source}_{x_name}"
            y_output = f"{y_source}_{y_name}"
            x_da = paired_ds[x_key].copy()
            y_da = paired_ds[y_key].copy()
            x_da.attrs.update(
                {
                    "axis": "x",
                    "source_label": x_source,
                    "dataset_variable": x_name,
                    "canonical_name": x_name,
                }
            )
            y_da.attrs.update(
                {
                    "axis": "y",
                    "source_label": y_source,
                    "dataset_variable": y_name,
                    "canonical_name": x_name,
                }
            )
            data_vars[x_output] = x_da
            data_vars[y_output] = y_da

        result = xr.Dataset(data_vars, coords=coords)
        result.attrs = dict(paired_ds.attrs)
        result.attrs.update({"created_by": "davinci_monet", "paired": True})
        return result

    def _detect_geometry(self, data: xr.Dataset) -> DataGeometry:
        """Detect source geometry from dataset attributes or structure.

        Parameters
        ----------
        data
            Source dataset.

        Returns
        -------
        DataGeometry
            Detected geometry type.

        Raises
        ------
        GeometryMismatchError
            If geometry cannot be determined.
        """
        # Check explicit geometry attribute
        if "geometry" in data.attrs:
            geom = data.attrs["geometry"]
            if isinstance(geom, DataGeometry):
                return geom
            if isinstance(geom, str):
                try:
                    return DataGeometry[geom.upper()]
                except KeyError:
                    pass

        # Infer from dimensions
        dims = set(data.dims)

        # Grid: has lat/lon as dimensions
        if ("lat" in dims or "latitude" in dims) and ("lon" in dims or "longitude" in dims):
            return DataGeometry.GRID

        # Swath: has scanline/pixel or similar
        if "scanline" in dims or "pixel" in dims or "cross_track" in dims:
            return DataGeometry.SWATH

        # Profile: has level/z dimension with time
        if "time" in dims and ("level" in dims or "z" in dims or "altitude" in dims):
            return DataGeometry.PROFILE

        # Point: has site/station dimension
        if "site" in dims or "station" in dims or "x" in dims:
            return DataGeometry.POINT

        # Track: has time dimension with lat/lon as coordinates
        if "time" in dims:
            coords = set(data.coords)
            if ("lat" in coords or "latitude" in coords) and (
                "lon" in coords or "longitude" in coords
            ):
                return DataGeometry.TRACK

        raise GeometryMismatchError(
            f"Cannot determine source geometry from dims {dims}. "
            "Please set the 'geometry' attribute on the dataset."
        )

    def _check_temporal_overlap(self, y_data: xr.Dataset, x_data: xr.Dataset) -> None:
        """Check if the two sources have temporal overlap.

        Parameters
        ----------
        y_data
            Source sampled from.
        x_data
            Source sampled onto.

        Raises
        ------
        NoOverlapError
            If no temporal overlap exists.
        """
        if "time" not in y_data.dims or "time" not in x_data.dims:
            return

        y_times = y_data["time"].values
        x_times = x_data["time"].values

        if len(y_times) == 0 or len(x_times) == 0:
            return

        y_start = y_times.min()
        y_end = y_times.max()
        x_start = x_times.min()
        x_end = x_times.max()

        if y_end < x_start or x_end < y_start:
            raise NoOverlapError(
                f"No temporal overlap between y ({y_start} to {y_end}) "
                f"and x ({x_start} to {x_end})"
            )


def create_default_engine() -> PairingEngine:
    """Create a pairing engine with all default strategies registered.

    Returns
    -------
    PairingEngine
        Engine with point, track, profile, swath, and grid strategies.
    """
    from davinci_monet.pairing.strategies.grid import GridStrategy
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy
    from davinci_monet.pairing.strategies.point import PointStrategy
    from davinci_monet.pairing.strategies.profile import ProfileStrategy
    from davinci_monet.pairing.strategies.track import TrackStrategy

    engine = PairingEngine()
    engine.register_strategy(PointStrategy())
    engine.register_strategy(TrackStrategy())
    engine.register_strategy(ProfileStrategy())
    # Production SWATH handler: bin onto a grid (see _register_default_strategies).
    engine.register_strategy(IntermediateGridStrategy())
    engine.register_strategy(GridStrategy())

    return engine
