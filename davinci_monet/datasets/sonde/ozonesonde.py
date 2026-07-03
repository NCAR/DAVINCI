"""Ozonesonde vertical profile dataset reader.

This module provides the OzonesondeReader class for reading ozonesonde
vertical profile data from balloon-borne instruments.
"""

from __future__ import annotations

import re
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
    select_variables,
    set_geometry_attr,
    validate_file_list,
)

_MISSING_KEYWORDS = ("missing", "fill", "sentinel", "bad")


def _numeric_tokens(text: str) -> list[float]:
    """Extract numeric tokens from metadata text."""
    values: list[float] = []
    for token in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _metadata_missing_values(metadata: dict[str, Any]) -> set[float]:
    """Return missing-value sentinels declared in metadata key/value pairs."""
    sentinels: set[float] = set()
    for key, value in metadata.items():
        combined = f"{key} {value}"
        if any(keyword in combined.lower() for keyword in _MISSING_KEYWORDS):
            sentinels.update(_numeric_tokens(str(value)))
    return sentinels


def _header_missing_values(lines: Sequence[str]) -> set[float]:
    """Return missing-value sentinels declared in free-form header lines."""
    sentinels: set[float] = set()
    for line in lines:
        if any(keyword in line.lower() for keyword in _MISSING_KEYWORDS):
            sentinels.update(_numeric_tokens(line))
    return sentinels


def _mask_missing_values(df: pd.DataFrame, sentinels: set[float]) -> pd.DataFrame:
    """Mask declared sentinel values in numeric profile data."""
    if not sentinels:
        return df
    sentinel_values = np.array(sorted(sentinels), dtype=float)
    result = df.copy()
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            values = result[column].to_numpy(dtype=float, copy=False)
            result[column] = result[column].mask(np.isin(values, sentinel_values))
    return result


_SHADOZ_OZONE_FLOOR = 9000.0


def _is_shadoz_ozone_column(name: str) -> bool:
    """Identify SHADOZ ozone mixing-ratio/partial-pressure columns by name."""
    lowered = name.lower()
    return "o3" in lowered or "ozone" in lowered


def _apply_shadoz_ozone_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Mask SHADOZ ozone-column values >= 9000, the SHADOZ missing-data floor.

    SHADOZ files additionally use values >= 9000 as a missing-data floor in
    ozone mixing-ratio/partial-pressure columns, independent of the header's
    declared missing-value code (e.g. 9005 or 9999 in an ``O3`` column are
    fills even when only ``-9999`` is declared). Altitude/geopotential-height
    columns legitimately exceed 9000 and are never floored; pressure columns
    are never floored either.
    """
    result = df.copy()
    for column in result.columns:
        if _is_shadoz_ozone_column(column) and pd.api.types.is_numeric_dtype(result[column]):
            values = result[column].to_numpy(dtype=float, copy=False)
            result[column] = result[column].mask(values >= _SHADOZ_OZONE_FLOOR)
    return result


@source_registry.register("ozonesonde")
class OzonesondeReader:
    """Reader for ozonesonde vertical profile datasets.

    Reads ozonesonde data from various formats including WOUDC,
    SHADOZ, and generic NetCDF profiles.
    Data is returned as profile geometry with (time, level) dimensions.

    Examples
    --------
    >>> reader = OzonesondeReader()
    >>> ds = reader.open(["sonde_profile.nc"])
    >>> print(ds.dims)
    Frozen({'time': 1, 'level': 100})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ozonesonde"

    @property
    def geometry(self) -> DataGeometry:
        """Return produced geometry."""
        return DataGeometry.PROFILE

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        format_type: str | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open ozonesonde dataset files.

        Parameters
        ----------
        file_paths
            Paths to ozonesonde files.
        variables
            Variables to load. If None, loads all available.
        format_type
            Data format ('woudc', 'shadoz', 'netcdf', or None for auto-detect).
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            Ozonesonde datasets with dimensions (time, level) and
            lat/lon coordinates.
        """
        file_list = validate_file_list(file_paths, source_label="Ozonesonde")

        # Auto-detect format if not specified
        if format_type is None:
            format_type = self._detect_format(file_list[0])

        if format_type == "netcdf":
            ds = self._open_netcdf(file_list, variables, **kwargs)
        elif format_type == "woudc":
            ds = self._open_woudc(file_list, variables, **kwargs)
        elif format_type == "shadoz":
            ds = self._open_shadoz(file_list, variables, **kwargs)
        else:
            # Default to NetCDF
            ds = self._open_netcdf(file_list, variables, **kwargs)

        return self._standardize_dataset(ds)

    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension and content."""
        suffix = file_path.suffix.lower()

        if suffix in (".nc", ".nc4", ".netcdf"):
            return "netcdf"
        elif suffix in (".csv", ".dat"):
            # Check first line for format hints
            try:
                with open(file_path, "r") as f:
                    first_line = f.readline()
                if "WOUDC" in first_line.upper():
                    return "woudc"
                elif "SHADOZ" in first_line.upper():
                    return "shadoz"
            except Exception:
                pass
            return "csv"
        else:
            return "netcdf"

    def _open_netcdf(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open ozonesonde NetCDF files."""
        ds_list = []
        for fpath in file_paths:
            try:
                ds = xr.open_dataset(str(fpath), **kwargs)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid ozonesonde data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _open_woudc(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open WOUDC format ozonesonde files."""
        ds_list = []
        for fpath in file_paths:
            try:
                ds = self._parse_woudc_file(fpath)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to parse WOUDC {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid WOUDC data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _parse_woudc_file(self, file_path: Path) -> xr.Dataset:
        """Parse a WOUDC format ozonesonde file."""
        # WOUDC files have a specific CSV-like format with metadata headers
        # This is a simplified parser
        data_section = False
        headers: list[str] = []
        data_rows: list[list[float]] = []
        metadata: dict[str, Any] = {}

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("#PROFILE"):
                    data_section = True
                    continue
                elif line.startswith("#"):
                    # Parse metadata sections
                    if "=" in line:
                        key, val = line[1:].split("=", 1)
                        metadata[key.strip()] = val.strip()
                    continue

                if data_section:
                    if not headers:
                        headers = [h.strip() for h in line.split(",")]
                    else:
                        values = []
                        for v in line.split(","):
                            try:
                                values.append(float(v.strip()))
                            except ValueError:
                                values.append(np.nan)
                        data_rows.append(values)

        if not data_rows:
            raise DataFormatError(f"No profile data found in {file_path}")

        df = pd.DataFrame(data_rows, columns=headers[: len(data_rows[0])])
        df = _mask_missing_values(df, _metadata_missing_values(metadata))

        # Add level coordinate
        df["level"] = range(len(df))
        df = df.set_index("level")

        ds: xr.Dataset = df.to_xarray()

        # Add metadata
        for key, val in metadata.items():
            ds.attrs[key] = val

        # Add time dimension
        ds = ds.expand_dims("time")

        return ds

    def _open_shadoz(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open SHADOZ format ozonesonde files."""
        ds_list = []
        for fpath in file_paths:
            try:
                ds = self._parse_shadoz_file(fpath)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to parse SHADOZ {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid SHADOZ data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _parse_shadoz_file(self, file_path: Path) -> xr.Dataset:
        """Parse a SHADOZ format ozonesonde file."""
        # SHADOZ files have header lines followed by data
        # Format: Press, Alt, Temp, RH, O3, ...
        header_lines = 0
        headers: list[str] = []

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Find data start - typically after lines starting with numbers
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit():
                header_lines = i
                break
            if "Press" in line or "Alt" in line:
                headers = [h.strip() for h in line.split()]

        if not headers:
            headers = ["Press", "Alt", "Temp", "RH", "O3"]

        missing_values = _header_missing_values(lines[:header_lines])

        data_rows = []
        for line in lines[header_lines:]:
            if line.strip():
                values = []
                for v in line.split():
                    try:
                        values.append(float(v))
                    except ValueError:
                        values.append(np.nan)
                if values:
                    data_rows.append(values)

        if not data_rows:
            raise DataFormatError(f"No profile data found in {file_path}")

        df = pd.DataFrame(data_rows, columns=headers[: len(data_rows[0])])
        df = _mask_missing_values(df, missing_values)
        df = _apply_shadoz_ozone_floor(df)
        df["level"] = range(len(df))
        df = df.set_index("level")

        ds: xr.Dataset = df.to_xarray()
        ds = ds.expand_dims("time")

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize ozonesonde dataset dimensions and coordinates."""
        dim_renames: dict[str, str] = {}
        coord_renames: dict[str, str] = {}

        # Standardize level dimension
        if "level" not in ds.dims:
            for alias in ["altitude", "pressure", "z"]:
                if alias in ds.dims:
                    dim_renames[alias] = "level"
                    break

        if dim_renames:
            ds = ds.rename(dim_renames)

        # Standardize coordinates
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.PROFILE)
