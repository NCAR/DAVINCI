"""Track-to-grid pairing strategy.

This module implements pairing for track x sources (aircraft, mobile
platforms) with a gridded y source, including 3D interpolation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Hashable

import dask
import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.coordinates import surface_index, vertical_dim_name
from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy

_logger = logging.getLogger(__name__)


def altitude_to_pressure(
    altitude_m: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert altitude (meters) to pressure (hPa) using standard atmosphere.

    Uses the barometric formula for the troposphere (valid up to ~11 km):
    P = P0 * (1 - L*h/T0)^(g*M/(R*L))

    Parameters
    ----------
    altitude_m
        Altitude in meters above sea level.

    Returns
    -------
    np.ndarray
        Pressure in hPa.
    """
    # Standard atmosphere constants
    P0 = 1013.25  # Sea level pressure (hPa)
    L = 0.0065  # Temperature lapse rate (K/m)
    T0 = 288.15  # Sea level temperature (K)
    g = 9.80665  # Gravitational acceleration (m/s²)
    M = 0.0289644  # Molar mass of air (kg/mol)
    R = 8.31447  # Gas constant (J/(mol·K))

    exponent = g * M / (R * L)  # ≈ 5.2559

    # Barometric formula
    pressure = P0 * (1 - L * altitude_m / T0) ** exponent

    return pressure


def pressure_to_altitude(
    pressure_hpa: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert pressure (hPa) to altitude (metres) — inverse of altitude_to_pressure.

    Inverts the US Standard Atmosphere barometric formula (troposphere,
    valid to ~11 km): h = (T0/L) * (1 - (P/P0)**(1/exponent)).
    """
    P0 = 1013.25
    L = 0.0065
    T0 = 288.15
    g = 9.80665
    M = 0.0289644
    R = 8.31447
    exponent = g * M / (R * L)  # ≈ 5.2559
    return (T0 / L) * (1.0 - (pressure_hpa / P0) ** (1.0 / exponent))


class TrackStrategy(BasePairingStrategy):
    """Pairing strategy for track sources.

    Matches moving x sources (aircraft, mobile platforms) to y
    grid cells along their trajectory, with 3D interpolation.

    The strategy:
    1. For each track point, finds nearest y grid cell
    2. Interpolates the y source temporally to track time
    3. Interpolates the y source vertically to track altitude
    4. Creates paired dataset with aligned values

    Examples
    --------
    >>> strategy = TrackStrategy()
    >>> paired = strategy.pair_sources(y_data, aircraft_data,
    ...                        vertical_method='linear')
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return TRACK geometry."""
        return DataGeometry.TRACK

    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "linear",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair track data with the y grid.

        Parameters
        ----------
        x_data
            The x source. Dataset with dims (time,) and lat/lon/alt coords.
        y_data
            The y source. Dataset with dims (time, z, lat, lon).
        radius_of_influence
            Maximum horizontal distance in meters.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method ('nearest', 'linear', 'log').
        horizontal_method
            Horizontal matching method ('nearest' only currently).
        **kwargs
            Additional options:
            - pressure_var: str, name of pressure variable for vertical interp
            - altitude_var: str, name of altitude variable in the x source

        Returns
        -------
        xr.Dataset
            Paired dataset with y values along track.
        """
        pressure_var = kwargs.get("pressure_var", "pressure")
        altitude_var = kwargs.get("altitude_var", "altitude")

        # Get coordinates
        y_lat, y_lon = self._get_y_coords(y_data)
        x_lat, x_lon = self._get_x_coords(x_data)

        # Get x times
        if "time" not in x_data.dims:
            raise PairingError("Track sources must have 'time' dimension")
        x_times = x_data["time"]
        n_points = len(x_times)

        # Find nearest y grid cell for each track point
        lat_idx, lon_idx = self._find_nearest_indices(
            y_lat,
            y_lon,
            x_lat,
            x_lon,
            radius_of_influence=radius_of_influence,
        )

        # Drop track points that fall outside radius_of_influence so the paired
        # output contains only matched points by construction. Without this,
        # _extract_along_track masks the y source to NaN at unpaired points but
        # leaves the x values intact, polluting any cross-time aggregate
        # that mixes paired and unpaired points on the x side.
        valid = (lat_idx.values >= 0) & (lon_idx.values >= 0)
        if not valid.all():
            keep = np.where(valid)[0]
            _logger.info(
                "TrackStrategy: dropping %d track point(s) outside %.0f m "
                "radius of influence (kept %d/%d).",
                int((~valid).sum()),
                radius_of_influence,
                int(valid.sum()),
                len(valid),
            )
            x_data = x_data.isel(time=keep)
            x_times = x_data["time"]
            n_points = len(x_times)
            lat_idx = xr.DataArray(lat_idx.values[keep])
            lon_idx = xr.DataArray(lon_idx.values[keep])

        # Determine vertical coordinate
        x_altitude = self._get_altitude(x_data, altitude_var)
        y_has_vertical = vertical_dim_name(y_data) is not None

        # Extract and interpolate y values along track
        y_along_track = self._extract_along_track(
            y_data,
            y_lat,
            y_lon,
            lat_idx.values,
            lon_idx.values,
            x_times,
            x_altitude,
            vertical_method=vertical_method,
            y_has_vertical=y_has_vertical,
            time_tolerance=time_tolerance,
        )

        # Create paired output
        paired = self._create_paired_output(x_data, y_along_track)

        return paired

    def _get_altitude(self, x_data: xr.Dataset, altitude_var: str) -> xr.DataArray | None:
        """Get altitude/pressure coordinate from the x source.

        Parameters
        ----------
        x_data
            The x source.
        altitude_var
            Name of altitude/pressure variable.

        Returns
        -------
        xr.DataArray | None
            Altitude values or None if not present.
        """
        # Try various altitude/pressure variable names
        for name in [altitude_var, "altitude", "alt", "z", "pressure"]:
            if name in x_data.coords:
                return x_data.coords[name]
            if name in x_data.data_vars:
                return x_data[name]
        return None

    def _extract_along_track(
        self,
        y_data: xr.Dataset,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        x_times: xr.DataArray,
        x_altitude: xr.DataArray | None,
        vertical_method: str,
        y_has_vertical: bool,
        time_tolerance: TimeDelta | None = None,
    ) -> xr.Dataset:
        """Extract y values along the track with 3D interpolation.

        Uses vectorized extraction for efficiency. For 3D y sources, interpolates
        vertically to aircraft altitude at each track point.

        Parameters
        ----------
        y_data
            The y source.
        y_lat, y_lon
            The y coordinate arrays.
        lat_idx, lon_idx
            Nearest grid indices for each track point.
        x_times
            Track times.
        x_altitude
            Track altitudes in meters (required for 3D interpolation).
        vertical_method
            Vertical interpolation method ('nearest', 'linear').
        y_has_vertical
            Whether the y source has a vertical dimension.

        Returns
        -------
        xr.Dataset
            The y values along track with time dimension.
        """
        n_points = len(x_times)

        # Determine lat/lon dimension names
        if y_lat.ndim == 1:
            lat_dim = str(y_lat.dims[0])
            lon_dim = str(y_lon.dims[0])
        else:
            lat_dim = str(y_lat.dims[0])
            lon_dim = str(y_lat.dims[1])

        # Handle invalid indices (outside radius of influence)
        valid_mask = (lat_idx >= 0) & (lon_idx >= 0)

        # Create DataArray indexers for vectorized spatial extraction
        lat_indexer = xr.DataArray(np.where(valid_mask, lat_idx, 0), dims=["track_point"])
        lon_indexer = xr.DataArray(np.where(valid_mask, lon_idx, 0), dims=["track_point"])

        # Detect vertical dimension
        level_dim = vertical_dim_name(y_data)

        # If the y source has a vertical dimension and we have altitude, do 3D interpolation
        # Otherwise fall back to surface extraction
        if level_dim and y_has_vertical and x_altitude is not None:
            extracted = self._extract_with_vertical_interp(
                y_data,
                lat_dim,
                lon_dim,
                level_dim,
                lat_indexer,
                lon_indexer,
                x_altitude,
                vertical_method,
                valid_mask,
                n_points,
            )
        else:
            # Fall back to surface extraction
            y_2d = self._extract_surface(y_data)
            extracted = y_2d.isel({lat_dim: lat_indexer, lon_dim: lon_indexer})

        # Load data with optimized parallel scheduler for file I/O
        n_workers = min(32, os.cpu_count() or 4)
        with dask.config.set(scheduler="threads", num_workers=n_workers):
            extracted = extracted.compute()

        # Interpolate in time: for each track point, find nearest y time
        # Use vectorized nearest-neighbor time matching
        y_times = extracted["time"].values.astype("datetime64[ns]").astype(np.int64)
        x_times_vals = x_times.values.astype("datetime64[ns]").astype(np.int64)

        # Vectorized nearest time matching using searchsorted
        insert_idx = np.searchsorted(y_times, x_times_vals)
        # Clamp to valid range
        insert_idx = np.clip(insert_idx, 1, len(y_times) - 1)
        # Check if left or right neighbor is closer
        left_idx = insert_idx - 1
        right_idx = insert_idx
        left_dist = np.abs(x_times_vals - y_times[left_idx])
        right_dist = np.abs(y_times[right_idx] - x_times_vals)
        time_idx = np.where(left_dist <= right_dist, left_idx, right_idx)

        # Gate on time tolerance: track points whose nearest y time is
        # farther than the tolerance are dropped (NaN) rather than snapped to a
        # distant y time. None disables the gate.
        if time_tolerance is not None:
            tol_ns = pd.Timedelta(time_tolerance).value
            time_valid = np.minimum(left_dist, right_dist) <= tol_ns
        else:
            time_valid = np.ones(n_points, dtype=bool)

        # Create time indexer
        time_indexer = xr.DataArray(time_idx, dims=["track_point"])

        # Extract at matched times
        result_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}
        for var in extracted.data_vars:
            var_data = extracted[var]
            if "time" in var_data.dims:
                # Select the appropriate time for each track point
                out_data = var_data.isel(time=time_indexer).values
            else:
                out_data = np.full(n_points, var_data.values)

            # Mask points invalid in space (outside radius) or time (outside tolerance)
            out_data = np.where(valid_mask & time_valid, out_data, np.nan)
            result_vars[str(var)] = (("time",), out_data)

        # Build output dataset
        coords = {"time": x_times.values}

        return xr.Dataset(result_vars, coords=coords)

    def _extract_with_vertical_interp(
        self,
        y_data: xr.Dataset,
        lat_dim: str,
        lon_dim: str,
        level_dim: str,
        lat_indexer: xr.DataArray,
        lon_indexer: xr.DataArray,
        x_altitude: xr.DataArray,
        vertical_method: str,
        valid_mask: np.ndarray[Any, np.dtype[Any]],
        n_points: int,
    ) -> xr.Dataset:
        """Extract y values with vertical interpolation to aircraft altitude.

        Parameters
        ----------
        y_data
            The y source with vertical dimension.
        lat_dim, lon_dim, level_dim
            Dimension names.
        lat_indexer, lon_indexer
            DataArray indexers for horizontal extraction.
        x_altitude
            Aircraft altitude in meters.
        vertical_method
            Interpolation method ('nearest', 'linear').
        valid_mask
            Boolean mask for valid track points.
        n_points
            Number of track points.

        Returns
        -------
        xr.Dataset
            The y values interpolated to aircraft altitude.
        """
        # Get y pressure levels (hPa)
        y_levels = y_data.coords[level_dim].values

        # Determine if levels are in pressure (hPa) or something else
        # CESM uses hPa with values like 3.6 to 992.5
        levels_are_pressure = y_levels.max() > 100  # hPa values > 100

        # Convert aircraft altitude (meters) to pressure (hPa)
        altitude_values = x_altitude.values
        if hasattr(altitude_values, "compute"):
            altitude_values = altitude_values.compute()
        altitude_values = np.asarray(altitude_values, dtype=np.float64)

        # Handle NaN altitudes - use surface pressure as fallback
        altitude_values = np.where(np.isnan(altitude_values), 0.0, altitude_values)
        aircraft_pressure = altitude_to_pressure(altitude_values)

        # Extract 3D columns at each horizontal location
        # This gives us (time, level, track_point) for each variable
        extracted_3d = y_data.isel({lat_dim: lat_indexer, lon_dim: lon_indexer})

        surface_idx = surface_index(y_data, level_dim)

        # Build result variables
        result_vars = {}
        for var in extracted_3d.data_vars:
            var_data = extracted_3d[var]
            if level_dim not in var_data.dims:
                # Variable has no vertical dimension, keep as-is
                result_vars[var] = var_data
                continue

            # Get dimension order
            dims = list(var_data.dims)
            level_axis = dims.index(level_dim)

            # Interpolate each track point to its aircraft pressure level
            if vertical_method == "nearest":
                # Nearest neighbor interpolation
                out_data = self._interp_nearest_vertical(
                    var_data.values,
                    y_levels,
                    aircraft_pressure,
                    level_axis,
                    n_points,
                    valid_mask,
                    surface_idx,
                )
            else:
                # Linear interpolation (in log-pressure space)
                out_data = self._interp_linear_vertical(
                    var_data.values,
                    y_levels,
                    aircraft_pressure,
                    level_axis,
                    n_points,
                    valid_mask,
                    surface_idx,
                )

            # Remove level dimension from dims
            new_dims = tuple(d for d in dims if d != level_dim)
            result_vars[var] = xr.DataArray(out_data, dims=new_dims)

        # Preserve coordinates except the vertical one
        coords = {k: v for k, v in extracted_3d.coords.items() if level_dim not in v.dims}

        return xr.Dataset(result_vars, coords=coords)

    def _interp_nearest_vertical(
        self,
        data: np.ndarray[Any, np.dtype[Any]],
        y_levels: np.ndarray[Any, np.dtype[Any]],
        aircraft_pressure: np.ndarray[Any, np.dtype[Any]],
        level_axis: int,
        n_points: int,
        valid_mask: np.ndarray[Any, np.dtype[Any]],
        surface_idx: int,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Nearest-neighbor vertical interpolation.

        Parameters
        ----------
        data
            The y data array with shape including level dimension.
        y_levels
            The y pressure levels (hPa).
        aircraft_pressure
            Target pressure levels for each track point (hPa).
        level_axis
            Axis index for the level dimension.
        n_points
            Number of track points.
        valid_mask
            Boolean mask for valid points.
        surface_idx
            Index of surface level for out-of-bounds fallback.

        Returns
        -------
        np.ndarray
            Interpolated data with level dimension removed.
        """
        # Find nearest level for each aircraft pressure
        # Dataset levels might be ordered differently, so handle both cases
        level_indices = np.zeros(n_points, dtype=np.int64)

        for i in range(n_points):
            if not valid_mask[i]:
                level_indices[i] = surface_idx
                continue

            target_p = aircraft_pressure[i]
            # Find nearest level by absolute difference
            diffs = np.abs(y_levels - target_p)
            level_indices[i] = int(np.argmin(diffs))

        # Use advanced indexing to extract values at each level
        # Move level axis to the end for easier indexing
        data_moved = np.moveaxis(data, level_axis, -1)
        original_shape = data_moved.shape[:-1]

        # Flatten all but last axis for indexing, then reshape
        n_other = int(np.prod(original_shape[:-1])) if len(original_shape) > 1 else 1
        n_track = original_shape[-1] if len(original_shape) > 0 else n_points

        # Extract using the level indices for each track point
        # Result shape will be original_shape without the level dimension
        result: np.ndarray[Any, np.dtype[Any]]
        if len(original_shape) == 1:
            # Shape is (track_point, level) -> (track_point,)
            result = data_moved[np.arange(n_track), level_indices[:n_track]]
        elif len(original_shape) == 2:
            # Shape is (time, track_point, level) -> (time, track_point)
            n_time = original_shape[0]
            result = np.zeros((n_time, n_track), dtype=data.dtype)
            for t in range(n_time):
                result[t, :] = data_moved[t, np.arange(n_track), level_indices[:n_track]]
        else:
            # General case - iterate over track points
            result = np.take_along_axis(
                data_moved,
                level_indices.reshape((1,) * (len(original_shape) - 1) + (n_track, 1)),
                axis=-1,
            ).squeeze(axis=-1)

        return result

    def _interp_linear_vertical(
        self,
        data: np.ndarray[Any, np.dtype[Any]],
        y_levels: np.ndarray[Any, np.dtype[Any]],
        aircraft_pressure: np.ndarray[Any, np.dtype[Any]],
        level_axis: int,
        n_points: int,
        valid_mask: np.ndarray[Any, np.dtype[Any]],
        surface_idx: int,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Linear vertical interpolation in log-pressure space.

        Parameters
        ----------
        data
            The y data array with shape including level dimension.
        y_levels
            The y pressure levels (hPa).
        aircraft_pressure
            Target pressure levels for each track point (hPa).
        level_axis
            Axis index for the level dimension.
        n_points
            Number of track points.
        valid_mask
            Boolean mask for valid points.
        surface_idx
            Index of surface level for out-of-bounds fallback.

        Returns
        -------
        np.ndarray
            Interpolated data with level dimension removed.
        """
        # Use log-pressure for interpolation (better for atmospheric profiles)
        # Guard against zero/negative pressure values
        safe_y_levels = np.maximum(y_levels, 1e-10)
        log_y_levels = np.log(safe_y_levels)

        # Move level axis to end
        data_moved = np.moveaxis(data, level_axis, -1)
        original_shape = data_moved.shape[:-1]
        n_levels = data_moved.shape[-1]

        # Determine shape of output
        if len(original_shape) == 1:
            n_track = original_shape[0]
            result = np.zeros(n_track, dtype=np.float64)
        elif len(original_shape) == 2:
            n_time, n_track = original_shape
            result = np.zeros((n_time, n_track), dtype=np.float64)
        else:
            result = np.zeros(original_shape, dtype=np.float64)
            n_track = original_shape[-1]

        # Ensure y levels are sorted for searchsorted
        if y_levels[0] > y_levels[-1]:
            # Decreasing pressure - reverse for interpolation
            sorted_indices = np.arange(n_levels - 1, -1, -1)
            log_levels_sorted = log_y_levels[sorted_indices]
        else:
            sorted_indices = np.arange(n_levels)
            log_levels_sorted = log_y_levels

        # Interpolate each track point
        for i in range(min(n_track, n_points)):
            if not valid_mask[i]:
                # Use surface value for invalid points
                if len(original_shape) == 1:
                    result[i] = data_moved[i, surface_idx]
                elif len(original_shape) == 2:
                    result[:, i] = data_moved[:, i, surface_idx]
                continue

            target_log_p = np.log(max(aircraft_pressure[i], 1e-10))

            # Find bracketing levels
            idx = np.searchsorted(log_levels_sorted, target_log_p)

            if idx == 0:
                # Below lowest level - use lowest
                level_idx = sorted_indices[0]
                if len(original_shape) == 1:
                    result[i] = data_moved[i, level_idx]
                elif len(original_shape) == 2:
                    result[:, i] = data_moved[:, i, level_idx]
            elif idx >= n_levels:
                # Above highest level - use highest
                level_idx = sorted_indices[-1]
                if len(original_shape) == 1:
                    result[i] = data_moved[i, level_idx]
                elif len(original_shape) == 2:
                    result[:, i] = data_moved[:, i, level_idx]
            else:
                # Interpolate between bracketing levels
                idx_lo = sorted_indices[idx - 1]
                idx_hi = sorted_indices[idx]
                log_p_lo = log_y_levels[idx_lo]
                log_p_hi = log_y_levels[idx_hi]

                # Linear weight in log-pressure space
                weight = (target_log_p - log_p_lo) / (log_p_hi - log_p_lo)
                weight = np.clip(weight, 0.0, 1.0)

                if len(original_shape) == 1:
                    result[i] = (1 - weight) * data_moved[i, idx_lo] + weight * data_moved[
                        i, idx_hi
                    ]
                elif len(original_shape) == 2:
                    result[:, i] = (1 - weight) * data_moved[:, i, idx_lo] + weight * data_moved[
                        :, i, idx_hi
                    ]

        return result

    def _create_paired_output(
        self,
        x_data: xr.Dataset,
        y_along_track: xr.Dataset,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        x_data
            The x source.
        y_along_track
            The y values along track.

        Returns
        -------
        xr.Dataset
            Combined dataset with x and y variables.
        """
        # Combine coordinates
        coords = dict(x_data.coords)

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add x variables (bare names)
        for var in x_data.data_vars:
            data_vars[str(var)] = x_data[var]

        # Add y variables under a ``y_`` prefix. Contract: strategies emit y as
        # ``y_<var>`` with no attrs; the engine is the sole attr writer and the
        # sole point that relabels to the public ``<source_label>_<var>`` form.
        for var in y_along_track.data_vars:
            data_vars[f"y_{var}"] = y_along_track[var]

        return xr.Dataset(data_vars, coords=coords)
