"""UFS (Unified Forecast System) model reader.

This module provides the UFSReader class for reading UFS-AQM (Air Quality
Model) output, including RRFS (Rapid Refresh Forecast System) data.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    alias_coord,
    retry_transient_open,
    select_variables,
    standardize_dims,
    validate_file_list,
)


@source_registry.register("ufs")
class UFSReader:
    """Reader for UFS-AQM model output.

    Reads UFS Air Quality Model output, including RRFS-CMAQ data.
    Supports both grib2 and netCDF formats.

    Examples
    --------
    >>> reader = UFSReader()
    >>> ds = reader.open(["rrfs.t00z.natlevf024.tm00.grib2"])
    >>> print(ds.dims)
    Frozen({'time': 1, 'z': 65, 'lat': 1059, 'lon': 1799})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ufs"

    @property
    def geometry(self) -> DataGeometry:
        """Dataset output is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open UFS-AQM output files.

        Parameters
        ----------
        file_paths
            Paths to UFS output files.
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options:
            - fname_pm25: Path(s) to PM2.5 specific files
            - surf_only: bool, extract surface level only

        Returns
        -------
        xr.Dataset
            UFS data with standardized dimensions (time, z, lat, lon).
        """
        file_list = validate_file_list(file_paths, source_label="UFS")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "Some UFS-specific features may not work.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        # Standardize dimensions
        ds = self._standardize_dataset(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open UFS files using monetio.

        Parameters
        ----------
        file_paths
            UFS file paths.
        variables
            Variables to load.
        **kwargs
            Additional monetio options.

        Returns
        -------
        xr.Dataset
            Raw UFS dataset.
        """
        import monetio as mio

        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        if "fname_pm25" in kwargs:
            mio_kwargs["fname_pm25"] = kwargs["fname_pm25"]

        # Remove our custom kwargs before passing to monetio
        for key in ("fname_pm25", "surf_only"):
            kwargs.pop(key, None)

        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]

        # Try the ufs module first, then fall back to _rrfs_cmaq_mm when needed.
        ds: xr.Dataset
        if hasattr(mio.models, "ufs"):
            ds = mio.models.ufs.open_mfdataset(files, **mio_kwargs)
        else:
            warnings.warn(
                "Using _rrfs_cmaq_mm reader. Update monetio for better UFS support.",
                DeprecationWarning,
            )
            ds = mio.models._rrfs_cmaq_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open UFS files using xarray.

        Parameters
        ----------
        file_paths
            UFS file paths.
        variables
            Variables to load.
        **kwargs
            Additional xarray options.

        Returns
        -------
        xr.Dataset
            Raw UFS dataset.
        """
        # Filter out custom kwargs
        xr_kwargs = {k: v for k, v in kwargs.items() if k not in ("fname_pm25", "surf_only")}

        # Check if files are grib2
        is_grib = any(str(f).endswith((".grib2", ".grb2", ".grib")) for f in file_paths)

        if is_grib:
            xr_kwargs.setdefault("engine", "cfgrib")

        def _open() -> xr.Dataset:
            if len(file_paths) > 1:
                ds = xr.open_mfdataset(
                    [str(f) for f in file_paths],
                    combine="by_coords",
                    parallel=True,
                    **xr_kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_paths[0]), **xr_kwargs)

            return select_variables(ds, variables)

        return retry_transient_open(_open, context="Opening UFS files")

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize UFS dataset dimensions and coordinates.

        Parameters
        ----------
        ds
            Raw UFS dataset.

        Returns
        -------
        xr.Dataset
            Standardized dataset.
        """
        # UFS dimension renames (depends on file type): grib2 dims (step,
        # valid_time, level, heightAboveGround, isobaricInhPa) and NetCDF dims
        # (pfull, grid_yt, grid_xt).
        ds = standardize_dims(
            ds,
            {
                "step": "time",
                "valid_time": "time",
                "level": "z",
                "heightAboveGround": "z",
                "isobaricInhPa": "z",
                "pfull": "z",
                "grid_yt": "y",
                "grid_xt": "x",
            },
        )

        # Handle lat/lon coordinates
        ds = alias_coord(ds, "latitude", "lat")
        ds = alias_coord(ds, "longitude", "lon")

        return ds


@source_registry.register("rrfs")
class RRFSReader(UFSReader):
    """RRFS reader implemented through UFS loading."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "rrfs"
