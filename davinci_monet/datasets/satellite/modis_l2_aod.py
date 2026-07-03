"""MODIS L2 AOD dataset reader.

This module provides the MODISL2AODReader class for reading MODIS
(Moderate Resolution Imaging Spectroradiometer) L2 AOD products.

MODIS is on NASA's Terra and Aqua satellites, providing global
aerosol optical depth measurements since 2000.

Note
----
This reader requires monetio.sat._modis_l2_mm for full functionality.
Without monetio, it falls back to basic xarray reading which may not handle
all MODIS-specific features (HDF-EOS, swath geometry).
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
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

# Standard variable name mappings for MODIS AOD
MODIS_AOD_VARIABLE_MAPPING: dict[str, str] = {
    "aod": "Optical_Depth_Land_And_Ocean",
    "aod_550": "Optical_Depth_Land_And_Ocean",
    "aod_land": "Corrected_Optical_Depth_Land",
    "aod_ocean": "Effective_Optical_Depth_Best_Ocean",
    "angstrom": "Deep_Blue_Angstrom_Exponent_Land",
    "qa": "Land_Ocean_Quality_Flag",
}
_EPOCH_1993_SECONDS = int(datetime(1993, 1, 1, tzinfo=timezone.utc).timestamp())
DEFAULT_MODIS_AOD_VARIABLE_DICT: dict[str, dict[str, Any]] = {
    "AOD_550_Dark_Target_Deep_Blue_Combined": {
        "minimum": 0.0,
        "maximum": 10.0,
        "scale": 0.001,
    },
    "AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag": {},
    "AOD_550_Dark_Target_Deep_Blue_Combined_Algorithm_Flag": {},
}


@source_registry.register("modis_l2_aod")
class MODISL2AODReader:
    """Reader for MODIS L2 AOD satellite datasets.

    Reads MODIS L2 AOD (Aerosol Optical Depth) data from HDF-EOS files.
    Supports both Terra (MOD04) and Aqua (MYD04) products.

    Data is returned as swath geometry with appropriate dimensions.

    Note
    ----
    Full functionality requires monetio. Without monetio, the reader falls
    back to basic xarray which may not handle MODIS HDF-EOS swath geometry
    correctly.

    Examples
    --------
    >>> reader = MODISL2AODReader()
    >>> ds = reader.open(["MOD04_L2.A*.hdf"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "modis_l2_aod"

    @property
    def geometry(self) -> DataGeometry:
        """Return produced geometry."""
        return DataGeometry.SWATH

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        qa_threshold: int | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MODIS L2 AOD dataset files.

        Parameters
        ----------
        file_paths
            Paths to MODIS L2 AOD files (MOD04 or MYD04).
        variables
            Variables to load. If None, loads AOD at 550nm.
        qa_threshold
            Quality assurance threshold. Pixels with QA below threshold
            are masked. Common values: 3=best, 2=good, 1=marginal.
            Set to None to disable filtering.
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            MODIS datasets with swath dimensions.
        """
        file_list = validate_file_list(file_paths, source_label="MODIS")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "MODIS HDF-EOS swath handling may be incomplete.",
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
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MODIS files using monetio."""
        import monetio.sat._modis_l2_mm as modis_module

        files = [str(f) for f in file_paths]
        variable_dict = kwargs.pop("variable_dict", None)
        if variable_dict is None:
            variable_dict = self._variable_dict_for(variables)

        if len(files) == 1:
            ds: xr.Dataset = modis_module.read_dataset(files[0], variable_dict, **kwargs)
        else:
            granules = modis_module.read_mfdataset(files, variable_dict, **kwargs)
            if isinstance(granules, dict):
                ds = xr.concat(list(granules.values()), dim="granule")
            else:
                ds = granules

        ds = self._add_scan_start_time(ds)
        return select_variables(ds, variables)

    def _variable_dict_for(
        self,
        variables: Sequence[str] | None,
    ) -> dict[str, dict[str, Any]]:
        if variables is None:
            return dict(DEFAULT_MODIS_AOD_VARIABLE_DICT)
        variable_dict: dict[str, dict[str, Any]] = {}
        for name in variables:
            source_name = MODIS_AOD_VARIABLE_MAPPING.get(str(name), str(name))
            variable_dict[source_name] = dict(DEFAULT_MODIS_AOD_VARIABLE_DICT.get(source_name, {}))
        return variable_dict

    def _add_scan_start_time(self, ds: xr.Dataset) -> xr.Dataset:
        if "Scan_Start_Time" in ds:
            return ds
        if "time" not in ds.coords:
            return ds
        time = ds.coords["time"]
        if np.issubdtype(time.dtype, np.datetime64):
            seconds = time.values.astype("datetime64[s]").astype("int64").astype("float64")
            scan = xr.DataArray(seconds, dims=time.dims, coords=time.coords) - float(
                _EPOCH_1993_SECONDS
            )
        else:
            scan = time.astype("float64") - float(_EPOCH_1993_SECONDS)
        return ds.assign(Scan_Start_Time=scan)

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MODIS files using xarray."""
        ds_list = []
        for fpath in file_paths:
            try:
                # MODIS HDF-EOS files need special handling
                ds = xr.open_dataset(str(fpath), engine="netcdf4", **kwargs)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid MODIS data found")

        if len(ds_list) > 1:
            # Try to concatenate - may fail for swath data
            try:
                ds = xr.concat(ds_list, dim="time")
            except Exception:
                ds = ds_list[0]
                warnings.warn(
                    "Could not concatenate MODIS files, using first file only",
                    UserWarning,
                )
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _apply_qa_filter(self, ds: xr.Dataset, qa_threshold: int) -> xr.Dataset:
        """Apply quality assurance filtering to dataset."""
        qa_var = None
        for name in ["Land_Ocean_Quality_Flag", "qa", "Quality_Assurance_Land"]:
            if name in ds.data_vars:
                qa_var = name
                break

        if qa_var is not None:
            mask = ds[qa_var] >= qa_threshold
            for var in ds.data_vars:
                if var != qa_var:
                    ds[var] = ds[var].where(mask)

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize MODIS dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}

        if "Latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["Latitude"] = "lat"
        elif "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"

        if "Longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["Longitude"] = "lon"
        elif "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.SWATH)
