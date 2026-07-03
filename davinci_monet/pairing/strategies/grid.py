"""Grid-to-grid pairing strategy.

This module implements pairing for gridded x sources (L3 satellite
products, reanalysis) with a gridded y source.
"""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np
import xarray as xr

from davinci_monet.core.coordinates import vertical_dim_name
from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy


class GridStrategy(BasePairingStrategy):
    """Pairing strategy for gridded sources.

    Matches gridded x sources (L3 satellite products, reanalysis)
    to a gridded y source through regridding.

    The strategy:
    1. Determines common grid (x grid, y grid, or custom)
    2. Regrids the y source to the x grid (or vice versa)
    3. Aligns temporal dimensions
    4. Creates paired dataset on common grid

    Examples
    --------
    >>> strategy = GridStrategy()
    >>> paired = strategy.pair_sources(y_data, l3_satellite_data,
    ...                        regrid_to='x')
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return GRID geometry."""
        return DataGeometry.GRID

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
        """Pair gridded data with the y grid.

        Parameters
        ----------
        x_data
            The x source. Dataset with dims (time, lat, lon).
        y_data
            The y source. Dataset with dims (time, [z], lat, lon).
        radius_of_influence
            Not used for grid-to-grid.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method if needed.
        horizontal_method
            Horizontal interpolation method ('nearest', 'bilinear').
        **kwargs
            Additional options:
            - regrid_to: str, 'x' or 'y' (default 'x')
            - extract_surface: bool, whether to extract surface level

        Returns
        -------
        xr.Dataset
            Paired dataset on common grid.
        """
        regrid_to = kwargs.get("regrid_to", "x")
        extract_surface = kwargs.get("extract_surface", True)

        # Get coordinates
        y_lat, y_lon = self._get_y_coords(y_data)
        x_lat, x_lon = self._get_x_coords(x_data)

        # Extract surface if the y source is 3D
        if extract_surface and vertical_dim_name(y_data) is not None:
            y_proc = self._extract_surface(y_data)
        else:
            y_proc = y_data

        # Regrid to common grid
        if regrid_to == "x":
            # Regrid the y source to the x grid
            y_regridded = self._regrid_to_target(y_proc, x_lat, x_lon, method=horizontal_method)
            x_aligned = x_data
        elif regrid_to == "y":
            # Regrid the x source to the y grid
            y_regridded = y_proc
            x_aligned = self._regrid_to_target(x_data, y_lat, y_lon, method=horizontal_method)
        else:
            raise PairingError(f"Invalid regrid_to option: {regrid_to}")

        # Align temporal dimensions
        if "time" in y_regridded.dims and "time" in x_aligned.dims:
            y_regridded, x_aligned = self._align_times(y_regridded, x_aligned, time_tolerance)

        # Create paired output
        paired = self._create_paired_output(x_aligned, y_regridded)

        return paired

    def _regrid_to_target(
        self,
        data: xr.Dataset,
        target_lat: xr.DataArray,
        target_lon: xr.DataArray,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Regrid a source to target lat/lon grid.

        Parameters
        ----------
        data
            Dataset to regrid.
        target_lat, target_lon
            Target coordinate arrays (1D).
        method
            Interpolation method.

        Returns
        -------
        xr.Dataset
            Regridded dataset.
        """
        # Find source lat/lon dimension names
        source_lat, source_lon = self._get_y_coords(data)

        if source_lat.ndim != 1 or source_lon.ndim != 1:
            # Curvilinear grids need more complex regridding
            return self._regrid_curvilinear(data, target_lat, target_lon, method)

        lat_dim = source_lat.dims[0]
        lon_dim = source_lon.dims[0]

        # Use xarray interp for regular grids
        return data.interp(
            {lat_dim: target_lat.values, lon_dim: target_lon.values},
            method=method,  # type: ignore[arg-type]
        )

    def _regrid_curvilinear(
        self,
        data: xr.Dataset,
        target_lat: xr.DataArray,
        target_lon: xr.DataArray,
        method: str,
    ) -> xr.Dataset:
        """Regrid curvilinear grid to regular target grid.

        Parameters
        ----------
        data
            Dataset with curvilinear coordinates.
        target_lat, target_lon
            Target 1D coordinate arrays.
        method
            Interpolation method.

        Returns
        -------
        xr.Dataset
            Regridded dataset on regular grid.
        """
        # This is a simplified implementation using nearest neighbor
        # For production use, consider xesmf or other regridding libraries

        source_lat, source_lon = self._get_y_coords(data)
        source_lat_flat = source_lat.values.flatten()
        source_lon_flat = source_lon.values.flatten()

        # Build output grid
        n_lat = len(target_lat)
        n_lon = len(target_lon)

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}

        for var in data.data_vars:
            var_data = data[var]
            source_shape = source_lat.shape

            # Get non-spatial dimensions
            spatial_dims = source_lat.dims
            other_dims: list[str] = [str(d) for d in var_data.dims if d not in spatial_dims]

            # Build output shape
            out_shape: list[int] = [int(var_data.sizes[d]) for d in other_dims]
            out_shape.extend([n_lat, n_lon])

            out_data = np.full(out_shape, np.nan)

            # For each target point, find nearest source point
            for i, lat in enumerate(target_lat.values):
                for j, lon in enumerate(target_lon.values):
                    dist = self._haversine_distance(lat, lon, source_lat_flat, source_lon_flat)
                    nearest_idx = np.argmin(dist)
                    src_i, src_j = np.unravel_index(nearest_idx, source_shape)

                    selection = {spatial_dims[0]: src_i, spatial_dims[1]: src_j}
                    val = var_data.isel(selection).values
                    out_data[..., i, j] = val

            out_dims = tuple(other_dims) + ("lat", "lon")
            data_vars[str(var)] = (out_dims, out_data)

        coords = {
            "lat": target_lat.values,
            "lon": target_lon.values,
        }

        # Add time coordinate if present
        if "time" in data.coords:
            coords["time"] = data.coords["time"].values

        return xr.Dataset(data_vars, coords=coords)

    def _align_times(
        self,
        y_data: xr.Dataset,
        x_data: xr.Dataset,
        time_tolerance: TimeDelta | None,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Align x and y time dimensions.

        Parameters
        ----------
        y_data
            The y source.
        x_data
            The x source.
        time_tolerance
            Maximum time difference.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Temporally aligned datasets.
        """
        y_times = y_data["time"].values
        x_times = x_data["time"].values

        # Find common times (within tolerance)
        if time_tolerance is not None:
            # Match each x time to the nearest y time, then relabel the
            # matched y times to the x times so the two share identical
            # time labels. Without the relabel, _create_paired_output reindexes
            # the y source onto the x time coordinate and the y source becomes all
            # NaN whenever x/y times differ (e.g. MERRA2 00:30 vs MODIS 00:00).
            y_matched = y_data.sel(time=x_times, method="nearest")
            y_matched = y_matched.assign_coords(time=x_times)
            return y_matched, x_data
        else:
            # Interpolate the y source to the x times
            y_interp = y_data.interp(time=x_times)
            return y_interp, x_data

    def _create_paired_output(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        x_data
            The x source (on target grid).
        y_data
            The y source (on target grid).

        Returns
        -------
        xr.Dataset
            Combined dataset.
        """
        # Combine coordinates
        coords = dict(x_data.coords)
        for coord in y_data.coords:
            if coord not in coords:
                coords[coord] = y_data.coords[coord]

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add x variables
        for var in x_data.data_vars:
            data_vars[str(var)] = x_data[var]

        # Add y variables with prefix
        for var in y_data.data_vars:
            y_var_name = f"y_{var}"
            data_vars[y_var_name] = y_data[var]

        return xr.Dataset(data_vars, coords=coords)
