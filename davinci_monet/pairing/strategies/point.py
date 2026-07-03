"""Point-to-grid pairing strategy.

This module implements pairing for point x sources (surface stations,
ground sites) with a gridded y source.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Hashable

import dask
import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy

_logger = logging.getLogger(__name__)


class PointStrategy(BasePairingStrategy):
    """Pairing strategy for point sources.

    Matches fixed-location x sources (surface stations, ground sites)
    to the nearest y grid cell within the radius of influence.

    The strategy:
    1. Finds nearest y grid cell for each x site
    2. Extracts surface level from the y source (if 3D)
    3. Interpolates the y source to the x times
    4. Creates paired dataset with aligned values

    Examples
    --------
    >>> strategy = PointStrategy()
    >>> paired = strategy.pair_sources(y_data, x_data,
    ...                        radius_of_influence=12000)
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return POINT geometry."""
        return DataGeometry.POINT

    def pair_sources(
        self,
        x_data: xr.Dataset,
        y_data: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        time_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair point data with the y grid.

        Parameters
        ----------
        x_data
            The x source. Dataset with dims (time, site).
        y_data
            The y source. Dataset with dims (time, [z], lat, lon).
        radius_of_influence
            Maximum distance in meters for matching. Default 12000m.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Not used for surface datasets.
        horizontal_method
            Horizontal matching method ('nearest' only currently).
        time_method
            Time interpolation method ('nearest' or 'linear'). Use 'linear'
            when the y source has sparse time output relative to the x source
            (e.g. 6-hourly WRF-Chem snapshots vs hourly AirNow) to avoid the
            step-function artifact produced by 'nearest'. Default 'nearest'
            preserves prior behavior.
        **kwargs
            Additional options:
            - extract_surface: bool, whether to extract surface level (default True)

        Returns
        -------
        xr.Dataset
            Paired dataset with y values at x locations.
        """
        if radius_of_influence is None:
            radius_of_influence = 12000.0

        extract_surface = kwargs.get("extract_surface", True)
        dask_num_workers = kwargs.get("dask_num_workers")

        # Get coordinates
        y_lat, y_lon = self._get_y_coords(y_data)
        x_lat, x_lon = self._get_x_coords(x_data)

        # Extract surface level if the y source is 3D
        if extract_surface:
            y_surface = self._extract_surface(y_data)
        else:
            y_surface = y_data

        # Find site dimension in the x source
        site_dim = self._get_site_dim(x_data)

        # Get unique site locations
        if x_lat.dims == (site_dim,):
            site_lats = x_lat.values
            site_lons = x_lon.values
        else:
            # Lat/lon may be (time, site) - take first time
            site_lats = x_lat.isel(time=0).values if "time" in x_lat.dims else x_lat.values
            site_lons = x_lon.isel(time=0).values if "time" in x_lon.dims else x_lon.values

        # Find nearest y grid indices for each site
        lat_idx, lon_idx = self._find_nearest_indices(
            y_lat,
            y_lon,
            xr.DataArray(site_lats),
            xr.DataArray(site_lons),
            radius_of_influence=radius_of_influence,
        )

        # Drop x sites that fall outside radius_of_influence. Without this,
        # _extract_at_sites masks the y source to NaN at unpaired sites but leaves
        # the x values intact, so cross-site aggregates (timeseries domain-mean)
        # are polluted by sites with no y match.
        valid = (lat_idx.values >= 0) & (lon_idx.values >= 0)
        if not valid.all():
            keep = np.where(valid)[0]
            _logger.info(
                "PointStrategy: dropping %d x site(s) outside %.0f m "
                "radius of influence (kept %d/%d).",
                int((~valid).sum()),
                radius_of_influence,
                int(valid.sum()),
                len(valid),
            )
            x_data = x_data.isel({site_dim: keep})
            site_lats = site_lats[keep]
            site_lons = site_lons[keep]
            lat_idx = xr.DataArray(lat_idx.values[keep])
            lon_idx = xr.DataArray(lon_idx.values[keep])

        # Extract y values at the x sites
        y_at_sites = self._extract_at_sites(
            y_surface,
            y_lat,
            y_lon,
            lat_idx.values,
            lon_idx.values,
            site_dim,
            dask_num_workers=dask_num_workers,
        )

        # Interpolate the y source to the x times
        if "time" in y_at_sites.dims and "time" in x_data.dims:
            x_times = x_data["time"]
            y_at_sites = self._interpolate_time(
                y_at_sites, x_times, method=time_method, time_tolerance=time_tolerance
            )

        # Combine into paired dataset
        paired = self._create_paired_output(x_data, y_at_sites, site_dim)

        return paired

    def _get_site_dim(self, x_data: xr.Dataset) -> str:
        """Find the site dimension name in the x source.

        Parameters
        ----------
        x_data
            The x source.

        Returns
        -------
        str
            Site dimension name.
        """
        for dim in ["site", "station", "x", "location"]:
            if dim in x_data.dims:
                return dim

        raise PairingError(
            f"Cannot find site dimension in the x source. " f"Available dims: {list(x_data.dims)}"
        )

    def _extract_at_sites(
        self,
        y_data: xr.Dataset,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        site_dim: str,
        dask_num_workers: int | None = None,
    ) -> xr.Dataset:
        """Extract y values at the x site locations.

        Parameters
        ----------
        y_data
            The y source (surface level).
        y_lat, y_lon
            The y coordinate arrays.
        lat_idx, lon_idx
            Indices of nearest y grid cells.
        site_dim
            Name of site dimension.

        Returns
        -------
        xr.Dataset
            The y values at site locations with site dimension.
        """
        n_sites = len(lat_idx)

        # Determine lat/lon dimension names
        if y_lat.ndim == 1:
            lat_dim = y_lat.dims[0]
            lon_dim = y_lon.dims[0]
        else:
            # Curvilinear grid - assume (y, x) or similar
            lat_dim = y_lat.dims[0]
            lon_dim = y_lat.dims[1]

        # Create site coordinate
        site_coord = np.arange(n_sites)

        # Handle invalid indices (outside radius of influence)
        valid_mask = (lat_idx >= 0) & (lon_idx >= 0)

        # Create DataArray indexers for vectorized extraction
        lat_indexer = xr.DataArray(
            np.where(valid_mask, lat_idx, 0), dims=[site_dim]  # Use 0 for invalid, mask later
        )
        lon_indexer = xr.DataArray(np.where(valid_mask, lon_idx, 0), dims=[site_dim])

        # Extract all sites at once using advanced indexing
        extracted = y_data.isel({lat_dim: lat_indexer, lon_dim: lon_indexer})

        # Load data to numpy with optimized parallel scheduler
        # Use threaded scheduler with multiple workers for parallel file I/O
        if dask_num_workers is not None:
            n_workers = max(1, int(dask_num_workers))
        else:
            n_workers = min(32, os.cpu_count() or 4)
        with dask.config.set(scheduler="threads", num_workers=n_workers):
            extracted = extracted.compute()

        # Mask invalid sites with NaN
        if not valid_mask.all():
            for var in extracted.data_vars:
                extracted[var] = extracted[var].where(xr.DataArray(valid_mask, dims=[site_dim]))

        # Build output dataset with proper coordinates
        coords = {site_dim: site_coord}
        if "time" in y_data.coords:
            coords["time"] = y_data.coords["time"].values

        return xr.Dataset({str(var): extracted[var] for var in extracted.data_vars}, coords=coords)

    def _create_paired_output(
        self,
        x_data: xr.Dataset,
        y_at_sites: xr.Dataset,
        site_dim: str,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        x_data
            The x source.
        y_at_sites
            The y values at site locations.
        site_dim
            Site dimension name.

        Returns
        -------
        xr.Dataset
            Combined dataset with both x and y values.
        """
        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add x variables (bare names)
        for var in x_data.data_vars:
            data_vars[str(var)] = x_data[var]

        # Add y variables - reassign to x coordinates to ensure alignment
        # The y source was extracted at same site/time locations, just with integer indices
        for var in y_at_sites.data_vars:
            y_var = y_at_sites[var]
            x_ref = x_data[list(x_data.data_vars)[0]]

            # Handle dimension mismatch (e.g., the x source has extra y=1 dimension)
            if y_var.ndim != x_ref.ndim:
                # Find extra dims in the x source that aren't in y (usually singleton dims like y=1)
                extra_dims = [d for d in x_ref.dims if d not in y_var.dims]
                if all(x_ref.sizes[d] == 1 for d in extra_dims):
                    # Squeeze x to match y dims, then expand y to match x
                    target_dims = y_var.dims
                    target_coords = {d: x_data.coords[d] for d in target_dims if d in x_data.coords}
                    y_da = xr.DataArray(
                        y_var.values,
                        dims=target_dims,
                        coords=target_coords,
                        name=var,
                    )
                    # Expand to include the extra singleton dimensions
                    for ed in extra_dims:
                        y_da = y_da.expand_dims({ed: x_data.coords[ed].values})
                    # Reorder to match x dims
                    y_da = y_da.transpose(*x_ref.dims)
                else:
                    raise PairingError(
                        f"Dimension mismatch between y ({y_var.dims}) and x ({x_ref.dims}). "
                        f"Extra dimensions {extra_dims} are not singletons."
                    )
            else:
                # Dimensions match - use original logic. Transpose first: y's
                # dims can match x's as a set while differing in order, and
                # assigning raw .values positionally would silently mislabel
                # the data in that case.
                y_var = y_var.transpose(*x_ref.dims, missing_dims="raise")
                y_da = xr.DataArray(
                    y_var.values,
                    dims=x_ref.dims,
                    coords={d: x_data.coords[d] for d in x_ref.dims if d in x_data.coords},
                    name=var,
                )
            # Contract: emit every y variable under a ``y_`` prefix. The engine
            # is the sole writer of axis/source_label attrs and the sole point
            # that relabels to the public ``<source_label>_<var>`` form.
            data_vars[f"y_{var}"] = y_da

        return xr.Dataset(data_vars, coords=dict(x_data.coords))
