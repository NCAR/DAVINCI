"""Base class for pairing strategies.

This module provides the abstract base class that all pairing strategies
must inherit from, along with common utility functions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.coordinates import (
    extract_surface,
    match_longitude_convention,
    wrap_longitudes,
)
from davinci_monet.core.exceptions import InterpolationError, PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta

_logger = logging.getLogger(__name__)


class BasePairingStrategy(ABC):
    """Abstract base class for pairing strategies.

    Provides common functionality used by all geometry-specific strategies.
    """

    @property
    @abstractmethod
    def geometry(self) -> DataGeometry:
        """Return the geometry type this strategy handles."""
        ...

    @abstractmethod
    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Sample ``y_data`` onto ``x_data``.

        The engine passes ``radius_of_influence``, ``time_tolerance``,
        ``vertical_method`` and ``horizontal_method`` by name, plus
        ``time_method`` and ``x_vars``/``y_vars``/``x_var``/``y_var`` via
        ``**kwargs``. By contract a strategy emits the x side bare and the y side
        ``y_``-prefixed, sets no attrs (the engine writes them), and tolerates
        options it does not use. NOTE: ``time_method`` is currently honored only
        by :class:`~davinci_monet.pairing.strategies.point.PointStrategy`; all
        other geometries use nearest-time alignment regardless of its value.
        """
        ...

    def _get_y_coords(self, data: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract latitude and longitude coordinates from the y source.

        Parameters
        ----------
        data
            The y source dataset.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (latitude, longitude) coordinate arrays.

        Raises
        ------
        PairingError
            If coordinates not found.
        """
        lat = None
        lon = None

        for name in ["lat", "latitude", "LAT", "LATITUDE", "XLAT"]:
            if name in data.coords:
                lat = data.coords[name]
                break
            if name in data.data_vars:
                lat = data[name]
                break

        for name in ["lon", "longitude", "LON", "LONGITUDE", "XLONG"]:
            if name in data.coords:
                lon = data.coords[name]
                break
            if name in data.data_vars:
                lon = data[name]
                break

        if lat is None or lon is None:
            raise PairingError(
                f"Cannot find lat/lon coordinates in y source. "
                f"Available coords: {list(data.coords)}"
            )

        return lat, lon

    def _get_x_coords(self, data: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract latitude and longitude from the x source.

        Parameters
        ----------
        data
            The x source dataset.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (latitude, longitude) arrays.
        """
        lat = None
        lon = None

        for name in ["lat", "latitude", "LAT", "LATITUDE"]:
            if name in data.coords:
                lat = data.coords[name]
                break
            if name in data.data_vars:
                lat = data[name]
                break

        for name in ["lon", "longitude", "LON", "LONGITUDE"]:
            if name in data.coords:
                lon = data.coords[name]
                break
            if name in data.data_vars:
                lon = data[name]
                break

        if lat is None or lon is None:
            raise PairingError(
                f"Cannot find lat/lon coordinates in x source. "
                f"Available coords: {list(data.coords)}"
            )

        return lat, lon

    def _find_nearest_indices(
        self,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        x_lat: xr.DataArray,
        x_lon: xr.DataArray,
        radius_of_influence: float | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Find nearest y grid indices for x locations.

        Parameters
        ----------
        y_lat, y_lon
            y source coordinate arrays.
        x_lat, x_lon
            x source coordinate arrays.
        radius_of_influence
            Maximum distance in meters. Points beyond this are masked.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (lat_indices, lon_indices) for each x point.
        """
        # Handle 1D vs 2D y grids
        if y_lat.ndim == 1 and y_lon.ndim == 1:
            # Regular grid - find nearest indices directly
            lat_idx = self._find_nearest_1d(y_lat.values, x_lat.values)
            lon_idx = self._find_nearest_1d(y_lon.values, x_lon.values, longitude=True)
        else:
            # Curvilinear grid - need 2D search
            lat_idx, lon_idx = self._find_nearest_2d(
                y_lat.values, y_lon.values, x_lat.values, x_lon.values
            )

        # Apply radius of influence filter if specified
        if radius_of_influence is not None:
            distances = self._haversine_distance(
                x_lat.values,
                x_lon.values,
                (y_lat.values[lat_idx] if y_lat.ndim == 1 else y_lat.values[lat_idx, lon_idx]),
                (y_lon.values[lon_idx] if y_lon.ndim == 1 else y_lon.values[lat_idx, lon_idx]),
            )
            mask = distances > radius_of_influence
            lat_idx = np.where(mask, -1, lat_idx)
            lon_idx = np.where(mask, -1, lon_idx)

        return xr.DataArray(lat_idx), xr.DataArray(lon_idx)

    def _find_nearest_1d(
        self,
        grid: np.ndarray[Any, np.dtype[Any]],
        points: np.ndarray[Any, np.dtype[Any]],
        *,
        longitude: bool = False,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Find nearest indices in a 1D sorted array.

        Parameters
        ----------
        grid
            1D array of grid values.
        points
            Array of points to locate.

        Returns
        -------
        np.ndarray
            Indices of nearest grid points.
        """
        grid_values = np.asarray(grid, dtype=np.float64)
        point_values = np.asarray(points, dtype=np.float64)
        if grid_values.size == 0:
            return np.full(point_values.shape, -1, dtype=np.int64)
        if grid_values.size == 1:
            return np.zeros(point_values.shape, dtype=np.int64)

        if longitude:
            flat_points = wrap_longitudes(point_values).reshape(-1)
            grid_lons = wrap_longitudes(grid_values)
            deltas = np.abs(((flat_points[:, None] - grid_lons[None, :] + 180.0) % 360.0) - 180.0)
            return np.nanargmin(deltas, axis=1).reshape(point_values.shape)

        order = np.argsort(grid_values)
        sorted_grid = grid_values[order]

        idx = np.searchsorted(sorted_grid, point_values)
        idx = np.clip(idx, 1, len(sorted_grid) - 1)

        left_idx = idx - 1
        right_idx = idx
        left = sorted_grid[left_idx]
        right = sorted_grid[right_idx]
        nearest_sorted = np.where(
            np.abs(point_values - left) < np.abs(point_values - right),
            left_idx,
            right_idx,
        )

        return order[nearest_sorted]

    def _find_nearest_2d(
        self,
        y_lat: np.ndarray[Any, np.dtype[Any]],
        y_lon: np.ndarray[Any, np.dtype[Any]],
        x_lat: np.ndarray[Any, np.dtype[Any]],
        x_lon: np.ndarray[Any, np.dtype[Any]],
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        """Find nearest indices in a 2D curvilinear grid.

        Uses brute force distance calculation. For large grids,
        consider using scipy.spatial.cKDTree.

        Parameters
        ----------
        y_lat, y_lon
            2D y source coordinate arrays.
        x_lat, x_lon
            1D x source coordinate arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (i_indices, j_indices) in the 2D grid.
        """
        n_x = len(x_lat)
        i_idx = np.zeros(n_x, dtype=int)
        j_idx = np.zeros(n_x, dtype=int)

        for k in range(n_x):
            # Calculate distance to all grid points
            dist = self._haversine_distance(x_lat[k], x_lon[k], y_lat, y_lon)
            # Find minimum
            flat_idx = np.argmin(dist)
            i_idx[k], j_idx[k] = np.unravel_index(flat_idx, y_lat.shape)

        return i_idx, j_idx

    def _haversine_distance(
        self,
        lat1: np.ndarray[Any, np.dtype[Any]] | float,
        lon1: np.ndarray[Any, np.dtype[Any]] | float,
        lat2: np.ndarray[Any, np.dtype[Any]] | float,
        lon2: np.ndarray[Any, np.dtype[Any]] | float,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate haversine distance in meters.

        Parameters
        ----------
        lat1, lon1
            First point(s) in degrees.
        lat2, lon2
            Second point(s) in degrees.

        Returns
        -------
        np.ndarray
            Distance(s) in meters.
        """
        R = 6371000  # Earth radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(match_longitude_convention(lon2 - lon1, [-180.0, 180.0]))

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return np.asarray(R * c)

    def _interpolate_time(
        self,
        data: xr.Dataset,
        target_times: xr.DataArray,
        method: str = "nearest",
        time_tolerance: TimeDelta | None = None,
    ) -> xr.Dataset:
        """Interpolate data to target times.

        Parameters
        ----------
        data
            Dataset with time dimension.
        target_times
            Target time values.
        method
            Interpolation method ('nearest', 'linear').
        time_tolerance
            Maximum allowed gap between a target time and the matched source
            time. When set and ``method == "nearest"``, target times with no
            source time within the tolerance are filled with NaN instead of
            being silently snapped to a distant time. None (default) disables
            the gate and matches the nearest time regardless of distance.

        Returns
        -------
        xr.Dataset
            Data interpolated to target times.
        """
        if "time" not in data.dims:
            return data

        if method == "nearest":
            if time_tolerance is not None:
                # Reindex (not sel) so out-of-tolerance targets become NaN
                # rather than snapping to a far-away time.
                tolerance = pd.Timedelta(time_tolerance)
                target_index = pd.Index(np.asarray(target_times.values))
                matched = data.get_index("time").get_indexer(
                    target_index, method="nearest", tolerance=tolerance
                )
                masked = int(np.count_nonzero(matched == -1))
                if masked:
                    _logger.warning(
                        "time tolerance %s masked %d of %d target times",
                        time_tolerance,
                        masked,
                        target_index.size,
                    )
                return data.reindex(
                    time=target_times.values,
                    method="nearest",
                    tolerance=tolerance,
                )
            # Select nearest times and assign target times as coordinate.
            # This ensures alignment when combining sources.
            result = data.sel(time=target_times, method="nearest")
            return result.assign_coords(time=target_times.values)
        else:
            return data.interp(time=target_times, method=method)  # type: ignore[arg-type]

    def _interpolate_vertical(
        self,
        data: xr.Dataset,
        target_levels: xr.DataArray,
        level_coord: str = "z",
        method: str = "linear",
    ) -> xr.Dataset:
        """Interpolate data to target vertical levels.

        Parameters
        ----------
        data
            Dataset with vertical dimension.
        target_levels
            Target level values.
        level_coord
            Name of vertical coordinate.
        method
            Interpolation method ('nearest', 'linear', 'log').

        Returns
        -------
        xr.Dataset
            Data interpolated to target levels.
        """
        if level_coord not in data.dims:
            return data

        if method == "nearest":
            return data.sel({level_coord: target_levels}, method="nearest")
        elif method == "log":
            # Log-pressure interpolation
            y_log = data.assign_coords({level_coord: np.log(data[level_coord].values)})
            target_log = np.log(target_levels.values)
            result = y_log.interp({level_coord: target_log}, method="linear")
            return result.assign_coords({level_coord: target_levels.values})
        else:
            return data.interp(
                {level_coord: target_levels.values},
                method=method,  # type: ignore[arg-type]
            )

    def _extract_surface(self, data: xr.Dataset, level_dim: str | None = None) -> xr.Dataset:
        """Extract surface level from data.

        Parameters
        ----------
        data
            Dataset to extract from.
        level_dim
            Name of vertical dimension. If None, auto-detects from common names.

        Returns
        -------
        xr.Dataset
            Data at surface level only.
        """
        return extract_surface(data, level_dim)
