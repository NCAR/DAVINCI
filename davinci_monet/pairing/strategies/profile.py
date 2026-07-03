"""Profile-to-grid pairing strategy.

This module implements pairing for profile x sources (sondes, lidar)
with a gridded y source.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.core.coordinates import vertical_dim_name
from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy


class ProfileStrategy(BasePairingStrategy):
    """Pairing strategy for profile sources.

    Matches vertical profile x sources (sondes, lidar) to y
    columns at the profile location.

    The strategy:
    1. Finds nearest y grid column for profile location
    2. Interpolates the y source temporally to profile time
    3. Interpolates both x and y to common vertical grid
    4. Creates paired dataset with aligned vertical profiles

    Examples
    --------
    >>> strategy = ProfileStrategy()
    >>> paired = strategy.pair_sources(y_data, sonde_data,
    ...                        vertical_method='log')
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return PROFILE geometry."""
        return DataGeometry.PROFILE

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
        """Pair profile data with the y grid.

        Parameters
        ----------
        x_data
            The x source. Dataset with dims (time, level) or (level,).
        y_data
            The y source. Dataset with dims (time, z, lat, lon).
        radius_of_influence
            Maximum horizontal distance in meters.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method ('nearest', 'linear', 'log').
        horizontal_method
            Horizontal matching method.
        **kwargs
            Additional options:
            - level_coord: str, name of vertical coordinate in the x source
            - interp_to_geometry_levels: bool, whether to interp the y source to x levels

        Returns
        -------
        xr.Dataset
            Paired dataset with x and y profiles.
        """
        level_coord = kwargs.get("level_coord") or vertical_dim_name(x_data) or "level"
        interp_to_geometry_levels = kwargs.get("interp_to_geometry_levels", True)

        # Get coordinates
        y_lat, y_lon = self._get_y_coords(y_data)
        x_lat, x_lon = self._get_x_coords(x_data)

        # Get profile location (may be single point or time-varying)
        if x_lat.ndim == 0:
            # Single location
            profile_lat = float(x_lat.values)
            profile_lon = float(x_lon.values)
        else:
            # Take first value (assume profile at fixed location)
            profile_lat = float(x_lat.values.flat[0])
            profile_lon = float(x_lon.values.flat[0])

        # Find nearest y column
        lat_idx, lon_idx = self._find_nearest_indices(
            y_lat,
            y_lon,
            xr.DataArray([profile_lat]),
            xr.DataArray([profile_lon]),
            radius_of_influence=radius_of_influence,
        )

        if lat_idx.values[0] < 0:
            raise PairingError(
                f"Profile location ({profile_lat}, {profile_lon}) is outside "
                f"radius of influence ({radius_of_influence}m) from the y grid"
            )

        # Extract y column at profile location
        y_column = self._extract_column(
            y_data,
            y_lat,
            y_lon,
            int(lat_idx.values[0]),
            int(lon_idx.values[0]),
        )

        # Interpolate the y source to the x times if needed
        if "time" in y_column.dims and "time" in x_data.dims:
            x_times = x_data["time"]
            y_column = self._interpolate_time(
                y_column, x_times, "nearest", time_tolerance=time_tolerance
            )

        # Handle vertical interpolation
        if interp_to_geometry_levels and level_coord in x_data.dims:
            x_levels = x_data[level_coord]
            y_level_coord = vertical_dim_name(y_column)
            y_column = self._interpolate_vertical(
                y_column,
                x_levels,
                level_coord=y_level_coord or level_coord,
                method=vertical_method,
            )
            if y_level_coord and y_level_coord != level_coord and y_level_coord in y_column.dims:
                y_column = y_column.rename({y_level_coord: level_coord})
                y_column = y_column.assign_coords({level_coord: x_levels.values})

        # Create paired output
        paired = self._create_paired_output(x_data, y_column, level_coord)

        return paired

    def _extract_column(
        self,
        y_data: xr.Dataset,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        lat_idx: int,
        lon_idx: int,
    ) -> xr.Dataset:
        """Extract a single y column.

        Parameters
        ----------
        y_data
            The y source.
        y_lat, y_lon
            The y coordinate arrays.
        lat_idx, lon_idx
            Grid indices for column location.

        Returns
        -------
        xr.Dataset
            The y data at single column with (time, z) dims.
        """
        # Determine dimension names
        if y_lat.ndim == 1:
            lat_dim = y_lat.dims[0]
            lon_dim = y_lon.dims[0]
        else:
            lat_dim = y_lat.dims[0]
            lon_dim = y_lat.dims[1]

        # Extract column
        return y_data.isel({lat_dim: lat_idx, lon_dim: lon_idx})

    def _create_paired_output(
        self,
        x_data: xr.Dataset,
        y_column: xr.Dataset,
        level_coord: str,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        x_data
            The x source.
        y_column
            The y column data.
        level_coord
            Name of vertical coordinate.

        Returns
        -------
        xr.Dataset
            Combined dataset.
        """
        # Combine coordinates
        coords = dict(x_data.coords)
        for coord in y_column.coords:
            if coord not in coords:
                coords[coord] = y_column.coords[coord]

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add x variables
        for var in x_data.data_vars:
            data_vars[str(var)] = x_data[var]

        # Add y variables with prefix
        for var in y_column.data_vars:
            y_var_name = f"y_{var}"
            data_vars[y_var_name] = y_column[var]

        return xr.Dataset(data_vars, coords=coords)
