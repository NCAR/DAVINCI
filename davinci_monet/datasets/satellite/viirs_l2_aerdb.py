"""VIIRS Deep Blue AERDB_L2 aerosol reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    select_variables,
    set_geometry_attr,
    validate_file_list,
)

AERDB_DEFAULT_VARIABLES: tuple[str, ...] = (
    "Aerosol_Optical_Thickness_550_Land",
    "Aerosol_Optical_Thickness_550_Ocean",
    "Aerosol_Optical_Thickness_550_Expected_Uncertainty_Land",
    "Aerosol_Optical_Thickness_550_Expected_Uncertainty_Ocean",
    "Aerosol_Optical_Thickness_QA_Flag_Land",
    "Aerosol_Optical_Thickness_QA_Flag_Ocean",
    "Scan_Start_Time",
)


@source_registry.register("viirs_l2_aerdb")
class VIIRSL2AERDBReader:
    """Reader for VIIRS Deep Blue AERDB_L2 aerosol swath datasets."""

    @property
    def name(self) -> str:
        return "viirs_l2_aerdb"

    @property
    def geometry(self) -> DataGeometry:
        return DataGeometry.SWATH

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        qa_threshold: int | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open AERDB_L2 VIIRS granules."""

        file_list = validate_file_list(file_paths, source_label="VIIRS AERDB")
        if len(file_list) == 1:
            ds = xr.open_dataset(str(file_list[0]), **kwargs)
        else:
            ds = xr.open_mfdataset(
                [str(path) for path in file_list],
                combine="nested",
                concat_dim="Idx_Atrack",
                **kwargs,
            )

        if qa_threshold is not None:
            ds = self._apply_qa_filter(ds, qa_threshold)

        keep = list(variables) if variables is not None else list(AERDB_DEFAULT_VARIABLES)
        for essential in ("Latitude", "Longitude", "Scan_Start_Time"):
            if essential in ds.variables and essential not in keep:
                keep.append(essential)
        ds = select_variables(ds, keep)

        return self._standardize_dataset(ds)

    def _apply_qa_filter(self, ds: xr.Dataset, qa_threshold: int) -> xr.Dataset:
        for surface in ("Land", "Ocean"):
            aod_name = f"Aerosol_Optical_Thickness_550_{surface}"
            qa_name = f"Aerosol_Optical_Thickness_QA_Flag_{surface}"
            if aod_name in ds.data_vars and qa_name in ds.variables:
                ds[aod_name] = ds[aod_name].where(ds[qa_name] >= qa_threshold)
        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        renames: dict[str, str] = {}
        if "Latitude" in ds.variables and "lat" not in ds.coords:
            renames["Latitude"] = "lat"
        if "Longitude" in ds.variables and "lon" not in ds.coords:
            renames["Longitude"] = "lon"
        if renames:
            ds = ds.rename(renames)
        for coord in ("lat", "lon"):
            if coord in ds.variables and coord not in ds.coords:
                ds = ds.set_coords(coord)
        return set_geometry_attr(ds, DataGeometry.SWATH)
