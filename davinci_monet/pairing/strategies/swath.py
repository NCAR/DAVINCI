"""Swath-to-grid pairing strategy.

This module implements pairing for satellite swath x sources (L2 products)
with a gridded y source via per-pixel nearest-neighbor matching.

Note
----
For production satellite analyses prefer :class:`IntermediateGridStrategy` (or the
external `bin_swath_to_grid` helper in ``pairing/grid_binning.py``). Real L2
swaths have 10^5-10^6 pixels and per-pixel nearest-neighbor matching is too
slow; the binning path collapses pixels onto a target grid once and then
pairs grid-to-grid. The registered MODIS L2 AOD reader
(``datasets/satellite/modis_l2_aod.py``) follows that pattern and emits
``geometry = "GRID"``.

This class is preserved for possible future use cases that genuinely need
direct per-pixel pairing (e.g. small swaths, sparse retrievals, debugging).
It is not on the current production path.
"""

from __future__ import annotations

import logging
from typing import Any, Hashable

import numpy as np
import xarray as xr

from davinci_monet.core.coordinates import surface_index, vertical_dim_name
from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy

_logger = logging.getLogger(__name__)


class SwathStrategy(BasePairingStrategy):
    """Pairing strategy for satellite swath sources.

    Matches satellite swath x sources (L2 products) to a y
    grid, optionally applying averaging kernels.

    The strategy:
    1. For each swath pixel, finds nearest y grid cell
    2. Optionally matches the y source to satellite overpass time
    3. Optionally applies averaging kernels to y profiles
    4. Creates paired dataset with collocated values

    .. note::
        ``SwathStrategy`` and :class:`IntermediateGridStrategy` both report
        ``DataGeometry.SWATH``. The engine registry maps one strategy per
        geometry, so they cannot both be registered:
        :class:`IntermediateGridStrategy` is the canonical engine-registered
        SWATH handler (see ``PairingEngine._register_default_strategies``), and
        this direct per-pixel class is intentionally **not** registered — it is
        preserved for direct use only. Registering both into one engine emits a
        collision warning (last registration wins).

        Production satellite analyses use :class:`IntermediateGridStrategy` or the
        external ``bin_swath_to_grid`` helper, which collapse pixels onto a
        target grid before pairing. See the module docstring.

    Examples
    --------
    >>> strategy = SwathStrategy()
    >>> paired = strategy.pair_sources(y_data, satellite_data,
    ...                        apply_averaging_kernel=True)
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return SWATH geometry."""
        return DataGeometry.SWATH

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
        """Pair swath data with the y grid.

        Parameters
        ----------
        x_data
            The x source. Dataset with dims (time, scanline, pixel) or similar.
        y_data
            The y source. Dataset with dims (time, [z], lat, lon).
        radius_of_influence
            Maximum distance in meters for matching.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method.
        horizontal_method
            Horizontal matching method.
        **kwargs
            Additional options:
            - apply_averaging_kernel: bool, whether to apply AK
            - ak_var: str, name of averaging kernel variable
            - match_overpass: bool, whether to match the y source to overpass time

        Returns
        -------
        xr.Dataset
            Paired dataset with y values at swath pixels.
        """
        apply_ak = kwargs.get("apply_averaging_kernel", False)
        match_overpass = kwargs.get("match_overpass", False)

        # Get coordinates
        y_lat, y_lon = self._get_y_coords(y_data)
        x_lat, x_lon = self._get_x_coords(x_data)

        # Flatten swath coordinates for processing
        x_lat_flat = x_lat.values.flatten()
        x_lon_flat = x_lon.values.flatten()
        n_pixels = len(x_lat_flat)

        # Find nearest y grid for each pixel
        lat_idx, lon_idx = self._find_nearest_indices(
            y_lat,
            y_lon,
            xr.DataArray(x_lat_flat),
            xr.DataArray(x_lon_flat),
            radius_of_influence=radius_of_influence,
        )

        # Mask x values at pixels outside radius_of_influence so the paired
        # output has x and y NaN at the same locations. Swath data is
        # inherently 2D (scanline x pixel), so we mask rather than drop —
        # preserving the swath geometry for downstream spatial plotting.
        # Without this, _extract_at_pixels NaNs the y side but the x
        # side retains valid values, polluting cross-pixel aggregates.
        valid_flat = (lat_idx.values >= 0) & (lon_idx.values >= 0)
        if not valid_flat.all():
            spatial_dims = x_lat.dims
            valid_2d = valid_flat.reshape(x_lat.shape)
            valid_da = xr.DataArray(valid_2d, dims=spatial_dims)
            _logger.info(
                "SwathStrategy: masking %d swath pixel(s) outside %.0f m "
                "radius of influence (kept %d/%d).",
                int((~valid_flat).sum()),
                radius_of_influence,
                int(valid_flat.sum()),
                len(valid_flat),
            )
            masked = x_data.copy()
            for var in x_data.data_vars:
                if all(d in x_data[var].dims for d in spatial_dims):
                    masked[var] = x_data[var].where(valid_da)
            x_data = masked

        # Handle time matching
        if match_overpass and "time" in x_data.coords:
            # Get overpass times and match the y source
            y_matched = self._match_to_overpass(y_data, x_data)
        else:
            y_matched = y_data

        # Extract y values at pixel locations
        y_at_pixels = self._extract_at_pixels(
            y_matched,
            y_lat,
            y_lon,
            lat_idx.values,
            lon_idx.values,
            x_data.shape if hasattr(x_data, "shape") else x_lat.shape,
        )

        # Apply averaging kernel if requested
        if apply_ak:
            ak_var = kwargs.get("ak_var", "averaging_kernel")
            if ak_var in x_data:
                y_at_pixels = self._apply_averaging_kernel(y_at_pixels, x_data, ak_var)

        # Create paired output
        paired = self._create_paired_output(x_data, y_at_pixels)

        return paired

    def _match_to_overpass(
        self,
        y_data: xr.Dataset,
        x_data: xr.Dataset,
    ) -> xr.Dataset:
        """Match the y source to satellite overpass times.

        Parameters
        ----------
        y_data
            The y source.
        x_data
            The x source with time info.

        Returns
        -------
        xr.Dataset
            The y data at overpass times.
        """
        if "time" not in y_data.dims:
            return y_data

        # Get x times
        if "time" in x_data.coords:
            x_times = x_data["time"]
            if x_times.ndim > 0:
                # Use median time as representative overpass
                x_time = x_times.values.flat[len(x_times.values.flat) // 2]
            else:
                x_time = x_times.values
            return y_data.sel(time=x_time, method="nearest")

        return y_data

    def _extract_at_pixels(
        self,
        y_data: xr.Dataset,
        y_lat: xr.DataArray,
        y_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        output_shape: tuple[int, ...],
    ) -> xr.Dataset:
        """Extract y values at pixel locations.

        Parameters
        ----------
        y_data
            The y source.
        y_lat, y_lon
            The y coordinate arrays.
        lat_idx, lon_idx
            Flat arrays of nearest grid indices.
        output_shape
            Shape to reshape output to match swath.

        Returns
        -------
        xr.Dataset
            The y values at pixel locations.
        """
        n_pixels = len(lat_idx)

        # Determine dimension names
        if y_lat.ndim == 1:
            lat_dim = y_lat.dims[0]
            lon_dim = y_lon.dims[0]
        else:
            lat_dim = y_lat.dims[0]
            lon_dim = y_lat.dims[1]

        # Build output for each variable
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}

        for var in y_data.data_vars:
            var_data = y_data[var]

            # Determine which dims to keep (excluding lat/lon)
            keep_dims = [d for d in var_data.dims if d not in (lat_dim, lon_dim)]

            # Build output shape
            out_shape = []
            for d in keep_dims:
                out_shape.append(var_data.sizes[d])
            out_shape.extend(output_shape)

            # Extract values
            out_data = np.full([n_pixels], np.nan)

            for i in range(n_pixels):
                if lat_idx[i] < 0 or lon_idx[i] < 0:
                    continue

                if y_lat.ndim == 1:
                    selection = {lat_dim: lat_idx[i], lon_dim: lon_idx[i]}
                else:
                    selection = {lat_dim: lat_idx[i], lon_dim: lon_idx[i]}

                point_val = var_data.isel(selection)

                level_dim = vertical_dim_name(point_val)
                if level_dim is not None:
                    point_val = point_val.isel({level_dim: surface_index(point_val, level_dim)})

                # If still has dimensions, take mean for non-spatial leftovers.
                if point_val.ndim > 0:
                    out_data[i] = float(point_val.mean().values)
                else:
                    out_data[i] = float(point_val.values)

            # Reshape to swath dimensions
            out_data_reshaped = out_data.reshape(output_shape)
            out_dims = ("y", "x") if len(output_shape) == 2 else ("pixel",)

            data_vars[str(var)] = (out_dims, out_data_reshaped)

        return xr.Dataset(data_vars)

    def _apply_averaging_kernel(
        self,
        y_data: xr.Dataset,
        x_data: xr.Dataset,
        ak_var: str,
    ) -> xr.Dataset:
        """Apply satellite averaging kernel to the y data.

        Parameters
        ----------
        y_data
            The y data at pixel locations.
        x_data
            The x source containing averaging kernel.
        ak_var
            Name of averaging kernel variable.

        Returns
        -------
        xr.Dataset
            The y data with averaging kernel applied.
        """
        # This is a placeholder - full AK application requires
        # knowledge of the specific satellite product
        # For now, just return the y data unchanged
        return y_data

    def _create_paired_output(
        self,
        x_data: xr.Dataset,
        y_at_pixels: xr.Dataset,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        x_data
            The x source.
        y_at_pixels
            The y values at pixel locations.

        Returns
        -------
        xr.Dataset
            Combined dataset.
        """
        # Combine coordinates
        coords = dict(x_data.coords)

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add x variables
        for var in x_data.data_vars:
            data_vars[str(var)] = x_data[var]

        # Add y variables with prefix
        for var in y_at_pixels.data_vars:
            y_var_name = f"y_{var}"
            data_vars[y_var_name] = y_at_pixels[var]

        return xr.Dataset(data_vars, coords=coords)
