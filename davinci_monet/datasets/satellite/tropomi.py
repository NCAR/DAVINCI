"""TROPOMI (TROPOspheric Monitoring Instrument) L2 dataset reader.

This module provides the TROPOMIReader class for reading TROPOMI L2
satellite products including NO2, O3, CO, HCHO, and SO2.

Note
----
This reader is optimized for TROPOMI L2 products and relies on
monetio.sat._tropomi_l2_no2_mm for full functionality. Without monetio,
it falls back to basic xarray reading which may not handle all
TROPOMI-specific features (nested HDF5 groups, swath geometry, etc.).
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    select_variables,
    set_geometry_attr,
    validate_file_list,
)

# Standard variable name mappings for TROPOMI
TROPOMI_VARIABLE_MAPPING: dict[str, str] = {
    "no2": "nitrogendioxide_tropospheric_column",
    "no2_total": "nitrogendioxide_total_column",
    "o3": "ozone_total_column",
    "co": "carbonmonoxide_total_column",
    "hcho": "formaldehyde_tropospheric_vertical_column",
    "so2": "sulfurdioxide_total_column",
    "qa_value": "qa_value",
}


@source_registry.register("tropomi")
class TROPOMIReader:
    """Reader for TROPOMI L2 satellite datasets.

    Reads TROPOMI data from NetCDF files or via monetio.
    Data is returned as swath geometry with (time, scanline, pixel) dimensions.

    Note
    ----
    Full functionality requires monetio. Without monetio, the reader falls
    back to basic xarray which may not handle TROPOMI-specific features like
    nested HDF5 groups and swath geometry correctly.

    Examples
    --------
    >>> reader = TROPOMIReader()
    >>> ds = reader.open(["S5P_OFFL_L2__NO2_*.nc"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "tropomi"

    @property
    def geometry(self) -> DataGeometry:
        """Return produced geometry."""
        return DataGeometry.SWATH

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        product: str = "NO2",
        qa_threshold: float | None = 0.75,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open TROPOMI dataset files.

        Parameters
        ----------
        file_paths
            Paths to TROPOMI L2 files.
        variables
            Variables to load. If None, loads main product variable.
        product
            Product type ('NO2', 'O3', 'CO', 'HCHO', 'SO2').
        qa_threshold
            Quality assurance threshold (0-1). Pixels below threshold
            are masked. Set to None to disable filtering.
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            TROPOMI datasets with swath dimensions.
        """
        file_list = validate_file_list(file_paths, source_label="TROPOMI")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, product=product, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "TROPOMI swath geometry handling may be incomplete.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        # Apply QA filtering
        if qa_threshold is not None:
            ds = self._apply_qa_filter(ds, qa_threshold)

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        product: str = "NO2",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open TROPOMI files using monetio."""
        import monetio.sat._tropomi_l2_no2_mm as tropomi_module

        mio_kwargs: dict[str, Any] = {}
        if variables is not None:
            mio_kwargs["var_dict"] = {v: {} for v in variables}
        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]
        ds: xr.Dataset = tropomi_module.read_trpdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open TROPOMI files using xarray."""
        # TROPOMI files have nested groups
        ds_list = []
        for fpath in file_paths:
            try:
                # Try opening the main PRODUCT group
                ds = xr.open_dataset(
                    str(fpath),
                    group="PRODUCT",
                    **kwargs,
                )
                ds_list.append(ds)
            except Exception:
                # Fall back to root group
                try:
                    ds = xr.open_dataset(str(fpath), **kwargs)
                    ds_list.append(ds)
                except Exception as e:
                    warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                    continue

        if not ds_list:
            raise DataNotFoundError("No valid TROPOMI data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _apply_qa_filter(self, ds: xr.Dataset, qa_threshold: float) -> xr.Dataset:
        """Apply QA value filtering to dataset."""
        qa_var = None
        for name in ["qa_value", "QA_value", "quality_flag"]:
            if name in ds.data_vars:
                qa_var = name
                break

        if qa_var is not None:
            mask = ds[qa_var] >= qa_threshold
            # Apply mask to all data variables
            for var in ds.data_vars:
                if var != qa_var:
                    ds[var] = ds[var].where(mask)

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize TROPOMI dataset dimensions and coordinates."""
        dim_renames: dict[str, str] = {}
        coord_renames: dict[str, str] = {}

        # Standardize dimension names
        if "scanline" not in ds.dims:
            for alias in ["scanline", "nscans"]:
                if alias in ds.dims:
                    dim_renames[alias] = "scanline"
                    break

        if "pixel" not in ds.dims:
            for alias in ["ground_pixel", "npixels"]:
                if alias in ds.dims and alias not in dim_renames:
                    dim_renames[alias] = "pixel"
                    break

        if dim_renames:
            ds = ds.rename(dim_renames)

        # Standardize coordinate names
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.SWATH)
