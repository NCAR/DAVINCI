"""CMAQ model reader.

This module provides the CMAQReader class for reading Community Multiscale
Air Quality (CMAQ) model output files.
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
    promote_to_coords,
    retry_transient_open,
    select_variables,
    standardize_dims,
    validate_file_list,
)


@source_registry.register("cmaq")
class CMAQReader:
    """Reader for CMAQ model output.

    Reads CMAQ CONC (concentration), MET (meteorology), and other output files,
    standardizing them into a consistent xarray Dataset format.

    Examples
    --------
    >>> reader = CMAQReader()
    >>> ds = reader.open(["CCTM_CONC_20240101.nc", "CCTM_CONC_20240102.nc"])
    >>> print(ds.dims)
    Frozen({'time': 48, 'z': 35, 'lat': 100, 'lon': 120})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "cmaq"

    @property
    def geometry(self) -> DataGeometry:
        """CMAQ output is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CMAQ output files.

        Parameters
        ----------
        file_paths
            Paths to CMAQ output files.
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options:
            - fname_vert: Path(s) to vertical files (METCRO3D)
            - fname_surf: Path(s) to surface files (METCRO2D)
            - concatenate_forecasts: bool, concatenate forecast cycles
            - surf_only: bool, extract surface level only

        Returns
        -------
        xr.Dataset
            CMAQ data with standardized dimensions (time, z, lat, lon).
        """
        file_list = validate_file_list(file_paths, source_label="CMAQ")

        # Try to use monetio if available
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "Some CMAQ-specific features may not work.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        # Standardize dimensions and coordinates
        ds = self._standardize_dataset(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CMAQ files using monetio.

        Parameters
        ----------
        file_paths
            CMAQ file paths.
        variables
            Variables to load.
        **kwargs
            Additional monetio options.

        Returns
        -------
        xr.Dataset
            Raw CMAQ dataset.
        """
        import monetio as mio

        # Prepare kwargs for monetio
        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        if "fname_vert" in kwargs:
            mio_kwargs["fname_vert"] = kwargs["fname_vert"]

        if "fname_surf" in kwargs:
            mio_kwargs["fname_surf"] = kwargs["fname_surf"]

        if kwargs.get("concatenate_forecasts", False):
            mio_kwargs["concatenate_forecasts"] = True

        # Use monetio CMAQ reader
        files = [str(f) for f in file_paths]
        ds: xr.Dataset = mio.models._cmaq_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CMAQ files using basic xarray.

        Parameters
        ----------
        file_paths
            CMAQ file paths.
        variables
            Variables to load.
        **kwargs
            Additional xarray options.

        Returns
        -------
        xr.Dataset
            Raw CMAQ dataset.
        """
        # Filter out our custom kwargs
        xr_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("fname_vert", "fname_surf", "concatenate_forecasts", "surf_only")
        }

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

            # Select variables if specified
            return select_variables(ds, variables)

        return retry_transient_open(_open, context="Opening CMAQ files")

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize CMAQ dataset dimensions and coordinates.

        Parameters
        ----------
        ds
            Raw CMAQ dataset.

        Returns
        -------
        xr.Dataset
            Standardized dataset with dims (time, z, lat, lon).
        """
        # Common CMAQ dimension renames
        ds = standardize_dims(ds, {"TSTEP": "time", "LAY": "z", "ROW": "y", "COL": "x"})

        # Ensure latitude and longitude are proper coordinates
        # CMAQ often has these as data variables
        ds = promote_to_coords(ds, ("latitude", "longitude"))

        # Create lat/lon aliases if needed
        ds = alias_coord(ds, "latitude", "lat")
        ds = alias_coord(ds, "longitude", "lon")

        return ds
