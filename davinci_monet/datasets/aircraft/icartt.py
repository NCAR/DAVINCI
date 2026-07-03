"""ICARTT (International Consortium for Atmospheric Research on Transport
and Transformation) format reader.

This module provides the ICARTTReader class for reading aircraft datasets
in ICARTT format, commonly used for field campaign data.
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
    select_variables,
    set_geometry_attr,
    validate_file_list,
)


@source_registry.register("icartt")
class ICARTTReader:
    """Reader for ICARTT format aircraft datasets.

    Reads aircraft data in ICARTT (.ict) format from field campaigns.
    Data is returned as track geometry with (time,) dimension.

    Examples
    --------
    >>> reader = ICARTTReader()
    >>> ds = reader.open(["flight_data.ict"])
    >>> print(ds.dims)
    Frozen({'time': 3600})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "icartt"

    @property
    def geometry(self) -> DataGeometry:
        """Return produced geometry."""
        return DataGeometry.TRACK

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open ICARTT dataset files.

        Parameters
        ----------
        file_paths
            Paths to ICARTT files.
        variables
            Variables to load. If None, loads all available.
        **kwargs
            Additional options passed to monetio or parser.

        Returns
        -------
        xr.Dataset
            Aircraft datasets with dimension (time,) and
            lat/lon/alt coordinates.
        """
        file_list = validate_file_list(file_paths, source_label="ICARTT")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic ICARTT parser.",
                UserWarning,
            )
            ds = self._open_with_parser(file_list, variables, **kwargs)

        # Apply header units before standardization so that _standardize_dataset
        # can overwrite header units for variables it converts (e.g. altitude ft→m).
        ds = self._apply_header_units(ds, file_list)

        return self._standardize_dataset(ds)

    def _parse_header_units(self, file_path: Path) -> dict[str, str]:
        """Parse variable units from an ICARTT FFI-1001 file header.

        Returns a mapping of ``{variable_name: units}`` derived from the
        independent-variable line (L9, 0-indexed line 8) and the dependent-
        variable definition lines (L13 onward, 0-indexed lines 12..12+NV-1).
        Returns an empty dict on any malformed or unreadable header.

        Parameters
        ----------
        file_path
            Path to a single ICARTT (.ict) file.

        Returns
        -------
        dict[str, str]
            Mapping of variable shortname to units string.
        """
        units: dict[str, str] = {}
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            nlhead = int(lines[0].split(",")[0])
            # Independent variable: 0-indexed line 8 (L9), format "name, units, ..."
            xparts = [p.strip() for p in lines[8].split(",")]
            if len(xparts) >= 2 and xparts[0]:
                units[xparts[0]] = xparts[1]
            # Number of dependent variables: 0-indexed line 9 (L10)
            nv = int(lines[9].split(",")[0])
            # Dependent variable defs: 0-indexed lines 12..12+NV-1
            # Each line: "shortname, units, longname ..."
            for i in range(12, 12 + nv):
                # Stop before the column-names line (last header line)
                if i >= nlhead - 1 or i >= len(lines):
                    break
                parts = [p.strip() for p in lines[i].split(",")]
                if len(parts) >= 2 and parts[0]:
                    units[parts[0]] = parts[1]
        except (ValueError, IndexError, OSError):
            return {}
        return units

    def _parse_base_date(self, lines: list[str]) -> datetime | None:
        """Parse the ICARTT base date from header line 7."""
        try:
            parts = [int(p.strip()) for p in lines[6].split(",")[:3]]
            return datetime(parts[0], parts[1], parts[2])
        except (ValueError, IndexError):
            return None

    def _parse_header_missing_values(self, lines: list[str]) -> dict[str, float]:
        """Parse per-variable missing-value sentinels from an FFI-1001 header."""
        try:
            n_header_lines = int(lines[0].split(",")[0])
            nv = int(lines[9].split(",")[0])
            missing_values = [float(p.strip()) for p in lines[11].split(",") if p.strip()]
        except (ValueError, IndexError):
            return {}

        sentinels: dict[str, float] = {}
        for i in range(nv):
            var_def_idx = 12 + i
            if var_def_idx >= n_header_lines - 1 or var_def_idx >= len(lines):
                break
            parts = [p.strip() for p in lines[var_def_idx].split(",")]
            if parts and parts[0] and i < len(missing_values):
                sentinels[parts[0]] = missing_values[i]
        return sentinels

    def _apply_header_units(self, ds: xr.Dataset, file_list: list[Path]) -> xr.Dataset:
        """Stamp units attrs onto dataset variables from the ICARTT file header.

        Only sets ``units`` when the variable is present in the header units map
        and does **not** already carry a ``units`` attribute.  Both data
        variables and coordinates are covered by iterating ``ds.variables``.

        Parameters
        ----------
        ds
            Dataset to annotate (modified in-place; also returned for chaining).
        file_list
            List of source files; the first file is used to parse the header
            (units are identical across concatenated files of one product).

        Returns
        -------
        xr.Dataset
            The same dataset with units attrs applied where missing.
        """
        if not file_list:
            return ds
        header_units = self._parse_header_units(file_list[0])
        if not header_units:
            return ds
        for raw_name in list(ds.variables):
            name = str(raw_name)
            if name in header_units and not ds[name].attrs.get("units"):
                ds[name].attrs["units"] = header_units[name]
        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open ICARTT files using monetio."""
        import monetio.profile.icartt as icartt_module

        ds_list = []
        for fpath in file_paths:
            try:
                result = icartt_module.add_data(str(fpath), **kwargs)
                # monetio returns xarray Dataset directly
                if isinstance(result, xr.Dataset):
                    if len(result.time) > 0:
                        ds_list.append(result)
                elif isinstance(result, pd.DataFrame):
                    # Fallback for older monetio versions
                    if not result.empty:
                        ds_list.append(self._dataframe_to_dataset(result))
            except Exception as e:
                warnings.warn(f"Failed to read {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid ICARTT data found")

        # Concatenate datasets along time dimension
        if len(ds_list) == 1:
            ds = ds_list[0]
        else:
            ds = xr.concat(ds_list, dim="time")

        return select_variables(ds, variables)

    def _open_with_parser(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open ICARTT files using basic parser."""
        ds_list = []
        for fpath in file_paths:
            try:
                ds = self._parse_icartt_file(fpath)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to parse {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid ICARTT data found")

        ds = xr.concat(ds_list, dim="time")

        return select_variables(ds, variables)

    def _parse_icartt_file(self, file_path: Path) -> xr.Dataset:
        """Parse a single ICARTT file.

        This is a simplified parser for basic ICARTT files.
        For full compliance, use monetio.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Parse header
        first_line = lines[0].strip().split(",")
        n_header_lines = int(first_line[0])
        base_date = self._parse_base_date(lines)
        missing_values = self._parse_header_missing_values(lines)

        # Get variable names from header
        var_line_idx = n_header_lines - 1
        if var_line_idx < len(lines):
            var_names = [v.strip() for v in lines[var_line_idx].strip().split(",")]
        else:
            raise DataFormatError(f"Invalid ICARTT header in {file_path}")

        # Read data
        data_lines = lines[n_header_lines:]
        data = []
        for line in data_lines:
            if line.strip():
                values = [
                    float(v) if v.strip() not in ("", "NaN") else np.nan
                    for v in line.strip().split(",")
                ]
                data.append(values)

        if not data:
            raise DataFormatError(f"No data found in {file_path}")

        # Create DataFrame
        df = pd.DataFrame(data, columns=var_names[: len(data[0])])
        for name, sentinel in missing_values.items():
            if name in df.columns:
                df[name] = df[name].mask(df[name] == sentinel)

        # Convert to xarray
        # Look for time variable (commonly first column or named time/Time_UTC)
        time_col = None
        for col in ["time", "Time_UTC", "UTC_Time", "Time"]:
            if col in df.columns:
                time_col = col
                break
        if time_col is None and len(df.columns) > 0:
            time_col = df.columns[0]

        if time_col:
            if base_date is not None and pd.api.types.is_numeric_dtype(df[time_col]):
                df[time_col] = pd.Timestamp(base_date) + pd.to_timedelta(df[time_col], unit="s")
            df = df.set_index(time_col)
            df.index.name = "time"

        ds: xr.Dataset = df.to_xarray()
        return ds

    def _dataframe_to_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert ICARTT DataFrame to xarray Dataset."""
        # Look for time column
        time_col = None
        for col in ["time", "Time_UTC", "UTC_Time", "Time"]:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df = df.set_index(time_col)
            df.index.name = "time"

        ds: xr.Dataset = df.to_xarray()
        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize ICARTT dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}

        # Standardize coordinate names (including campaign-specific suffixes)
        lat_aliases = [
            "Latitude",
            "latitude",
            "LAT",
            "lat",
            "Latitude_YANG",
            "Latitude_BENNETT",
            "G_LAT",
            "LATITUDE",
        ]
        for alias in lat_aliases:
            if alias in ds.data_vars and "latitude" not in ds.coords:
                ds = ds.set_coords(alias)
                if alias != "latitude":
                    coord_renames[alias] = "latitude"
                break

        lon_aliases = [
            "Longitude",
            "longitude",
            "LON",
            "lon",
            "Longitude_YANG",
            "Longitude_BENNETT",
            "G_LONG",
            "LONGITUDE",
        ]
        for alias in lon_aliases:
            if alias in ds.data_vars and "longitude" not in ds.coords:
                ds = ds.set_coords(alias)
                if alias != "longitude":
                    coord_renames[alias] = "longitude"
                break

        # For altitude, prefer geometric altitude (meters first, then km, feet)
        # Variables in non-meter units will be converted to meters
        FEET_TO_METERS = 0.3048
        KM_TO_METERS = 1000.0
        feet_vars = {"Pressure_Altitude_BENNETT", "GPS_Altitude_BENNETT"}
        km_vars = {"ALTP", "GPS_ALT", "GPS_ALT_MMS", "RadarAlt"}
        alt_aliases = [
            # Altitude in meters (preferred)
            "GPS_Altitude_m_DIGANGI",
            "Altitude_AGL_m_DIGANGI",
            "MSL_GPS_Altitude_YANG",
            "MSL_GPS_Altitude",
            "GPS_Altitude",
            "Altitude",
            "altitude",
            "ALT",
            "alt",
            # Altitude in km (DC3 merge files, etc.)
            "ALTP",
            "GPS_ALT",
            "GPS_ALT_MMS",
            "RadarAlt",
            # Altitude in feet (will be converted)
            "Pressure_Altitude_BENNETT",
            "GPS_Altitude_BENNETT",
            # Pressure as last resort (not recommended for plotting)
            "Static_Pressure_BENNETT",
            "Static_Pressure",
            "PRESSURE",
        ]
        for alias in alt_aliases:
            if alias in ds.data_vars and "altitude" not in ds.coords:
                # Convert feet to meters if needed
                if alias in feet_vars:
                    ds[alias] = ds[alias] * FEET_TO_METERS
                    ds[alias].attrs["units"] = "m"
                # Convert km to meters if needed
                elif alias in km_vars:
                    ds[alias] = ds[alias] * KM_TO_METERS
                    ds[alias].attrs["units"] = "m"
                ds = ds.set_coords(alias)
                if alias != "altitude":
                    coord_renames[alias] = "altitude"
                break

        if coord_renames:
            ds = ds.rename(coord_renames)

        # Add flight identifier based on date (for per-flight analysis)
        if "time" in ds.dims:
            times = pd.to_datetime(ds["time"].values)
            flight_dates = times.strftime("%Y-%m-%d")
            ds = ds.assign_coords(flight=("time", flight_dates))

        return set_geometry_attr(ds, DataGeometry.TRACK)
