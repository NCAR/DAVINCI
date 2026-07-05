"""AirNow real-time air quality dataset reader.

This module provides the AirNowReader class for reading AirNow surface
monitoring data.
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


@source_registry.register("airnow")
class AirNowReader:
    """Reader for AirNow real-time air quality data.

    Reads AirNow data from files or via monetio API queries.

    Examples
    --------
    >>> reader = AirNowReader()
    >>> ds = reader.open(dates=["2024-01-01", "2024-01-02"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "airnow"

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
        wide_fmt: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AirNow dataset data.

        Parameters
        ----------
        file_paths
            Paths to pre-downloaded AirNow files.
        variables
            Variables to load.
        dates
            Date range for monetio API query [start, end].
        wide_fmt
            If True, return wide format (one column per variable).
        **kwargs
            Additional options passed to monetio.

        Returns
        -------
        xr.Dataset
            AirNow datasets with dimensions (time, x).
        """
        if file_paths is not None:
            return self._open_files(file_paths, variables, **kwargs)
        elif dates is not None:
            return self._open_with_monetio(dates, variables, wide_fmt=wide_fmt, **kwargs)
        else:
            raise DataNotFoundError("Either file_paths or dates must be provided")

    def _open_files(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AirNow data from files."""
        file_list = validate_file_list(file_paths, source_label="AirNow")

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

        ds = retry_transient_open(_open, context="Opening AirNow files")

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        dates: Sequence[datetime | str],
        variables: Sequence[str] | None,
        wide_fmt: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AirNow data using monetio API."""
        try:
            import monetio.obs.airnow as airnow_module
        except ImportError as e:
            raise ImportError(
                "monetio is required for AirNow API queries. " "Install with: pip install monetio"
            ) from e

        # Parse dates
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])
        date_range = pd.date_range(start_date, end_date, freq="D")

        # Query AirNow data
        ds_list = []
        for date in date_range:
            try:
                df: pd.DataFrame = airnow_module.add_data(date, wide_fmt=wide_fmt, **kwargs)
                if not df.empty:
                    ds_list.append(df)
            except Exception:
                continue

        if not ds_list:
            raise DataNotFoundError(f"No AirNow data found for {start_date} to {end_date}")

        # Combine and convert to xarray
        combined_df = pd.concat(ds_list, ignore_index=True)
        ds: xr.Dataset = self._dataframe_to_dataset(combined_df)
        ds = select_variables(ds, variables)

        return self._standardize_dataset(ds)

    def _dataframe_to_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert AirNow DataFrame to xarray Dataset."""
        if "time" in df.columns and "siteid" in df.columns:
            df = df.set_index(["time", "siteid"])
            ds: xr.Dataset = df.to_xarray()
            if "siteid" in ds.dims:
                ds = ds.rename({"siteid": "x"})
        else:
            ds = df.to_xarray()
        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize AirNow dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.POINT)


def _dataframe_to_xarray(df: pd.DataFrame, daily: bool = False) -> xr.Dataset:
    """Convert wide-format monetio AirNow DataFrame to an xarray Dataset.

    Produces dims (time, y=1, x) with `latitude`/`longitude` coordinates,
    site metadata variables, per-
    species units propagated from `*_unit` columns, and (for non-daily data)
    a derived `time_local` variable computed from `utcoffset`.

    Parameters
    ----------
    df
        Wide-format DataFrame from `monetio.airnow.add_data(..., wide_fmt=True)`.
    daily
        If True, the data is daily and lacks `utcoffset`/`time_local`.

    Returns
    -------
    xr.Dataset
        Dataset with dims (time, y, x) and AirNow species as data variables.
    """
    if df is None or df.empty:
        raise DataFormatError("Empty AirNow DataFrame provided")

    df = df.dropna(subset=["latitude", "longitude"])

    site_vns = [
        "site",
        "siteid",
        "utcoffset",
        "latitude",
        "longitude",
        "cmsa_name",
        "msa_code",
        "msa_name",
        "state_name",
        "epa_region",
    ]
    if daily:
        site_vns.remove("utcoffset")

    site_vns = [c for c in site_vns if c in df.columns]

    ds_site = df[site_vns].groupby("siteid").first().to_xarray().swap_dims(siteid="x")

    unit_suff = "_unit"
    unit_cols = [n for n in df.columns if n.endswith(unit_suff)]
    units: dict[str, str] = {}
    if unit_cols:
        non_null = df[unit_cols].dropna(how="all")
        if not non_null.empty:
            units = non_null.iloc[0].dropna().to_dict()

    cols = [n for n in df.columns if not n.endswith(unit_suff)]
    ds = (
        df[cols]
        .set_index(["time", "siteid"])
        .to_xarray()
        .swap_dims(siteid="x")
        .drop_vars(site_vns)
        .merge(ds_site)
        .set_coords(["latitude", "longitude"])
        .assign(x=range(ds_site.sizes["x"]))
    )

    for k, u in units.items():
        vn = k[: -len(unit_suff)]
        if vn in ds.data_vars:
            ds[vn].attrs["units"] = u

    if not daily and "utcoffset" in ds.data_vars:
        ds["time_local"] = (ds.time + ds.utcoffset.astype("timedelta64[h]")).astype(
            "datetime64[ns]"
        )

    ds = ds.expand_dims("y").transpose("time", "y", "x")

    return ds
