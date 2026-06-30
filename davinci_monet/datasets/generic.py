"""Generic dataset reader.

This module provides a fallback reader for dataset output that doesn't match
any specific dataset type. It uses xarray's generic NetCDF/grib readers.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    retry_transient_open,
    select_variables,
    validate_file_list,
)

# Common coordinate name aliases for standardization
COMMON_COORDINATE_ALIASES: dict[str, list[str]] = {
    "time": ["time", "Time", "TSTEP", "t", "datetime", "date_time"],
    "z": [
        "z",
        "lev",
        "level",
        "levels",
        "z_level",
        "altitude",
        "height",
        "bottom_top",
        "LAY",
        "layer",
        "pfull",
        "phalf",
        "sigma",
    ],
    "lat": [
        "lat",
        "latitude",
        "LAT",
        "LATITUDE",
        "XLAT",
        "south_north",
        "nlat",
        "rlat",
        "grid_yt",
        "y",
    ],
    "lon": [
        "lon",
        "longitude",
        "LON",
        "LONGITUDE",
        "XLONG",
        "west_east",
        "nlon",
        "rlon",
        "grid_xt",
        "x",
    ],
}


def _cleanup_with_suppressed_errors() -> None:
    """Force garbage collection with stderr suppressed to hide cleanup errors.

    When xarray fails to open files, cleanup code in __del__ methods can
    produce ugly tracebacks like "Exception ignored in: CachingFileManager.__del__".
    These can't be caught with try/except, so we suppress stderr during gc.
    """
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        gc.collect()
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


@source_registry.register("generic")
class GenericReader:
    """Generic dataset reader for arbitrary NetCDF/grib files.

    This reader provides a fallback for dataset types that don't have
    a dedicated reader. It attempts to standardize dimensions and
    coordinates automatically.

    Examples
    --------
    >>> reader = GenericReader()
    >>> ds = reader.open(["dataset_output.nc"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "generic"

    @property
    def geometry(self) -> DataGeometry:
        """Dataset output is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open dataset files using xarray.

        Parameters
        ----------
        file_paths
            Paths to dataset output files.
        variables
            Variables to load. If None, loads all variables.
        progress_callback
            Optional callable ``(i, total, name) -> None`` invoked for each
            file as it is opened.  ``i`` is 1-based, ``total`` is the number
            of files, and ``name`` is the file basename.  When provided with
            multiple files the reader forces ``parallel=False`` so the
            preprocess hook runs sequentially and the counter is monotonically
            increasing.
        **kwargs
            Additional options:
            - engine: xarray engine to use ('netcdf4', 'h5netcdf', 'cfgrib', etc.)
            - standardize: bool, whether to standardize dimension names (default True)
            - concat_dim: Dimension to concatenate along (default 'time')
            - combine: How to combine files ('by_coords', 'nested')
            - parallel: Whether xarray should open multiple files in parallel
              (default False for NetCDF/HDF5 stability)

        Returns
        -------
        xr.Dataset
            Dataset data with optionally standardized dimensions.
        """
        file_list = validate_file_list(file_paths, source_label="")

        # Extract our custom kwargs
        standardize = kwargs.pop("standardize", True)
        concat_dim = kwargs.pop("concat_dim", "time")
        combine = kwargs.pop("combine", "by_coords")
        parallel = kwargs.pop("parallel", False)

        # Detect engine if not specified
        if "engine" not in kwargs:
            engine = self._detect_engine(file_list[0])
            if engine:
                kwargs["engine"] = engine

        # Open files with retry for transient NetCDF errors. On terminal
        # failure, suppress the noisy CachingFileManager.__del__ tracebacks
        # via _cleanup_with_suppressed_errors (passed as the on_failure hook).
        def _open() -> xr.Dataset:
            if len(file_list) > 1:
                concat_kwargs = {"concat_dim": concat_dim} if combine == "nested" else {}
                if progress_callback is not None:
                    # Sequential open so the preprocess counter is ordered.
                    total = len(file_list)
                    counter: list[int] = [0]

                    def _progress_preprocess(ds: xr.Dataset) -> xr.Dataset:
                        counter[0] += 1
                        source = ds.encoding.get("source", "")
                        name = Path(source).name if source else ""
                        progress_callback(counter[0], total, name)
                        return ds

                    return xr.open_mfdataset(
                        [str(f) for f in file_list],
                        combine=combine,
                        data_vars="all",
                        parallel=False,
                        preprocess=_progress_preprocess,
                        **concat_kwargs,
                        **kwargs,
                    )
                return xr.open_mfdataset(
                    [str(f) for f in file_list],
                    combine=combine,
                    data_vars="all",
                    parallel=parallel,
                    **concat_kwargs,
                    **kwargs,
                )
            return xr.open_dataset(str(file_list[0]), **kwargs)

        ds = retry_transient_open(
            _open,
            context="Opening files",
            on_failure=_cleanup_with_suppressed_errors,
        )

        # Select variables if specified
        ds = select_variables(ds, variables)

        # Standardize dimensions
        if standardize:
            ds = self._standardize_dataset(ds)

        return ds

    def _detect_engine(self, file_path: Path) -> str | None:
        """Detect appropriate xarray engine for file.

        Parameters
        ----------
        file_path
            Path to check.

        Returns
        -------
        str | None
            Engine name or None for default.
        """
        suffix = file_path.suffix.lower()

        if suffix in (".grib", ".grib2", ".grb", ".grb2"):
            return "cfgrib"
        elif suffix in (".zarr",):
            return "zarr"
        elif suffix in (".nc", ".nc4", ".netcdf"):
            return "netcdf4"

        return None

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize dataset dimensions and coordinates.

        Parameters
        ----------
        ds
            Raw dataset.

        Returns
        -------
        xr.Dataset
            Dataset with standardized dimension names.
        """
        dim_renames: dict[str, str] = {}
        coord_renames: dict[str, str] = {}

        # Check each standard name against aliases
        for standard_name, aliases in COMMON_COORDINATE_ALIASES.items():
            for alias in aliases:
                # Check dimensions
                if alias in ds.dims and alias != standard_name:
                    if standard_name not in ds.dims:
                        dim_renames[alias] = standard_name
                    break

                # Check coordinates
                if alias in ds.coords and alias != standard_name:
                    if standard_name not in ds.coords:
                        coord_renames[alias] = standard_name
                    break

        # Apply renames
        if dim_renames:
            ds = ds.rename(dim_renames)

        if coord_renames:
            ds = ds.rename(coord_renames)

        return ds
