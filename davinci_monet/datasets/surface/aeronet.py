"""AERONET (Aerosol Robotic Network) dataset reader.

This module provides the AERONETReader class for reading AERONET AOD
and inversion products.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    retry_transient_open,
    select_variables,
    set_geometry_attr,
    validate_file_list,
)


@source_registry.register("aeronet")
class AERONETReader:
    """Reader for AERONET AOD datasets.

    Reads AERONET data from files or via monetio API queries.

    Examples
    --------
    >>> reader = AERONETReader()
    >>> ds = reader.open(dates=["2024-01-01", "2024-01-31"], product="AOD15")
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "aeronet"

    @property
    def geometry(self) -> DataGeometry:
        """Return produced geometry."""
        return DataGeometry.POINT

    def open(
        self,
        file_paths: Sequence[str | Path] | None = None,
        variables: Sequence[str] | None = None,
        *,
        dates: Sequence[datetime | str] | None = None,
        product: str = "AOD15",
        siteid: str | Sequence[str] | None = None,
        inv_type: str | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AERONET dataset data.

        Parameters
        ----------
        file_paths
            Paths to pre-downloaded AERONET files.
        variables
            Variables to load.
        dates
            Date range for API query [start, end].
        product
            AERONET product type ('AOD10', 'AOD15', 'AOD20', 'SDA10', etc.).
        siteid
            Site ID(s) to load. If None, loads all available.
        inv_type
            Inversion type for size distribution products.
        **kwargs
            Additional options passed to monetio.

        Returns
        -------
        xr.Dataset
            AERONET datasets with dimensions (time, x).
        """
        if file_paths is not None:
            return self._open_files(file_paths, variables, **kwargs)
        elif dates is not None:
            return self._open_with_monetio(
                dates, variables, product=product, siteid=siteid, inv_type=inv_type, **kwargs
            )
        else:
            raise DataNotFoundError("Either file_paths or dates must be provided")

    def _open_files(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AERONET data from files."""
        file_list = validate_file_list(file_paths, source_label="AERONET")

        def _open() -> xr.Dataset:
            if len(file_list) > 1:
                ds = xr.open_mfdataset(
                    [str(f) for f in file_list],
                    combine="by_coords",
                    parallel=True,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_list[0]), **kwargs)
            return select_variables(ds, variables)

        ds = retry_transient_open(_open, context="Opening AERONET files")

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        dates: Sequence[datetime | str],
        variables: Sequence[str] | None,
        product: str = "AOD15",
        siteid: str | Sequence[str] | None = None,
        inv_type: str | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AERONET data using monetio API."""
        try:
            import monetio.obs.aeronet as aeronet_module
        except ImportError as e:
            raise ImportError(
                "monetio is required for AERONET API queries. " "Install with: pip install monetio"
            ) from e

        # Parse dates
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])

        # Build kwargs for monetio
        mio_kwargs: dict[str, Any] = {}
        if siteid is not None:
            if isinstance(siteid, str):
                mio_kwargs["siteid"] = [siteid]
            else:
                mio_kwargs["siteid"] = list(siteid)
        if inv_type is not None:
            mio_kwargs["inv_type"] = inv_type
        mio_kwargs.update(kwargs)

        try:
            # monetio's add_data(dates, product, ...) takes a SINGLE array-like `dates`
            # (it uses its min/max for the query window). Passing start_date AND end_date
            # as two positionals binds end_date to `product` -> "multiple values for
            # 'product'". Pass them as one list instead.
            df: pd.DataFrame = aeronet_module.add_data(
                [start_date, end_date], product=product, **mio_kwargs
            )
        except Exception as e:
            raise DataFormatError(f"Failed to query AERONET data: {e}") from e

        if df.empty:
            raise DataNotFoundError(f"No AERONET data found for {start_date} to {end_date}")

        ds: xr.Dataset = self._dataframe_to_dataset(df)
        ds = select_variables(ds, variables)

        return self._standardize_dataset(ds)

    def _dataframe_to_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert AERONET DataFrame to xarray Dataset."""
        # AERONET data has time, site info, and AOD measurements
        if "time" in df.columns and "siteid" in df.columns:
            df = df.set_index(["time", "siteid"])
            ds: xr.Dataset = df.to_xarray()
            if "siteid" in ds.dims:
                ds = ds.rename({"siteid": "x"})
        else:
            ds = df.to_xarray()
        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize AERONET dataset dimensions and coordinates.

        Handles NetCDF files with dimensions (time, y, x) where y=1 is a dummy
        dimension. Squeezes out y and renames x to site for point geometry.
        """
        # Squeeze out y dimension if it has size 1 (dummy dimension from NetCDF)
        if "y" in ds.dims and ds.sizes["y"] == 1:
            ds = ds.squeeze("y", drop=True)

        # Rename x to site for point geometry consistency
        if "x" in ds.dims and "site" not in ds.dims:
            ds = ds.rename({"x": "site"})

        # Standardize coordinate names
        coord_renames: dict[str, str] = {}
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.POINT)


def _dataframe_to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """Convert AERONET DataFrame from monetio to xarray Dataset.

    This function is used by the CLI to convert monetio output to xarray format
    suitable for saving as NetCDF.

    Parameters
    ----------
    df
        DataFrame from monetio.obs.aeronet.add_data().

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (time, site) and coordinate variables.
    """
    if df is None or df.empty:
        raise DataFormatError("Empty DataFrame provided")

    # Ensure required columns exist
    required = ["time", "siteid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataFormatError(f"Missing required columns: {missing}")

    # Get unique sites with their coordinates
    coord_cols = ["latitude", "longitude", "elevation"]
    available_coords = [c for c in coord_cols if c in df.columns]

    sites_df = df.groupby("siteid")[available_coords].first().reset_index()
    site_ids = sites_df["siteid"].tolist()

    # Identify data columns (exclude metadata columns)
    meta_cols = [
        "siteid",
        "time",
        "latitude",
        "longitude",
        "elevation",
        "day_of_year",
        "day_of_year(fraction)",
    ]
    data_cols = [c for c in df.columns if c not in meta_cols]

    # Get common time index first
    times = pd.to_datetime(df["time"].unique()).sort_values()

    # Use vectorized pivot_table for fast conversion
    data_vars = {}
    for col in data_cols:
        if col not in df.columns:
            continue
        # Pivot: rows=time, columns=siteid, values=col
        try:
            pivoted = df.pivot_table(index="time", columns="siteid", values=col, aggfunc="first")
            # Reindex to ensure consistent time and site ordering
            pivoted = pivoted.reindex(index=times, columns=site_ids)
            # Only include if there's actual data
            if not pivoted.isna().all().all():
                data_vars[col] = (["time", "site"], pivoted.values)
        except Exception:
            # Skip columns that can't be pivoted (e.g., string columns)
            continue

    if not data_vars:
        raise DataFormatError("No numeric data columns found in DataFrame")

    # Create coordinates
    coords = {
        "time": times,
        "site": site_ids,
    }

    # Add spatial coordinates
    for coord in available_coords:
        coords[coord] = ("site", sites_df[coord].values)

    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)

    # Add attributes
    ds.attrs["source"] = "AERONET"
    set_geometry_attr(ds, DataGeometry.POINT)

    return ds
