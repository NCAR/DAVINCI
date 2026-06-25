"""CERES SSF (Single Scanner Footprint) L2 reader.

SSF files are flat 1-D footprint streams (~10^5 footprints/hour) in two
editions: HDF4 (Terra/Aqua Edition4A — SDS names with spaces, colatitude,
0-360 longitude, Julian-Date time) and netCDF-4 (NOAA-20 Edition1C —
grouped variables, true latitude, epoch time). This reader standardizes
both to dims ``(time,)`` with ``lat``/``lon`` coords, canonical variable
names via ``SSF_CATALOG`` (raw source names pass through as an escape
hatch; for netCDF use ``"Group/var"``), fill-masked values, and SWATH
geometry. Pairing flows through ``IntermediateGridStrategy``, which flattens and
bins footprints by lat/lon/time values — no 2-D swath dims required.

Footprints with invalid time/lat/lon are dropped (they cannot be paired);
invalid data values become NaN but keep their footprint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    apply_hdf4_scale,
    set_geometry_attr,
    validate_file_list,
)

_JD_EPOCH = 2440587.5  # Julian Date of 1970-01-01T00

# HDF4 coordinate SDS names (Edition4A).
_H4_TIME_ALIASES = ("Time of dataset", "Time of observation")
_H4_COLAT = "Colatitude of CERES FOV at surface"
_H4_LON = "Longitude of CERES FOV at surface"

# netCDF coordinate variables (Edition1C), all in Time_and_Position.
_NC_POSITION_GROUP = "Time_and_Position"
_NC_TIME = "time"
_NC_LAT = "instrument_fov_latitude"
_NC_LON = "instrument_fov_longitude"


@dataclass(frozen=True)
class SSFVariable:
    """Per-format source names for one canonical SSF variable."""

    hdf4_sds: str
    nc_group: str
    nc_var: str


# Canonical config name -> per-edition source. Surface fluxes standardize on
# Dataset B (the only parameterization with all-sky + clear-sky in both
# editions). Names outside this catalog are treated as raw source names.
SSF_CATALOG: dict[str, SSFVariable] = {
    "toa_sw_up": SSFVariable(
        "CERES SW TOA flux - upwards", "TOA_and_Surface_Fluxes", "toa_shortwave_flux"
    ),
    "toa_lw_up": SSFVariable(
        "CERES LW TOA flux - upwards", "TOA_and_Surface_Fluxes", "toa_longwave_flux"
    ),
    "toa_solar_in": SSFVariable(
        "TOA Incoming Solar Radiation",
        "TOA_and_Surface_Fluxes",
        "toa_incoming_solar_radiation",
    ),
    "sfc_sw_down": SSFVariable(
        "CERES downward SW surface flux - Dataset B",
        "TOA_and_Surface_Fluxes",
        "dataset_b_surface_shortwave_downward_flux",
    ),
    "sfc_sw_down_clr": SSFVariable(
        "CERES downward SW surface flux - Dataset B, clearsky",
        "TOA_and_Surface_Fluxes",
        "dataset_b_clearsky_surface_shortwave_downward_flux",
    ),
    "sfc_lw_down": SSFVariable(
        "CERES downward LW surface flux - Dataset B",
        "TOA_and_Surface_Fluxes",
        "dataset_b_surface_longwave_downward_flux",
    ),
    "sfc_lw_down_clr": SSFVariable(
        "CERES downward LW surface flux - Dataset B, clearsky",
        "TOA_and_Surface_Fluxes",
        "dataset_b_clearsky_surface_longwave_downward_flux",
    ),
    "sfc_sw_net": SSFVariable(
        "CERES net SW surface flux - Dataset B",
        "TOA_and_Surface_Fluxes",
        "dataset_b_surface_shortwave_net_flux",
    ),
    "sfc_lw_net": SSFVariable(
        "CERES net LW surface flux - Dataset B",
        "TOA_and_Surface_Fluxes",
        "dataset_b_surface_longwave_net_flux",
    ),
}

_HDF4_MAGIC = b"\x0e\x03\x13\x01"
_HDF5_MAGIC = b"\x89HDF"
_SSF_STAMP_RE = re.compile(r"\.(\d{10})(?:\.nc)?$")


def _sniff_format(path: Path) -> str:
    """Return ``"hdf4"`` or ``"netcdf"`` from the file magic bytes."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == _HDF4_MAGIC:
        return "hdf4"
    if magic == _HDF5_MAGIC or magic[:3] == b"CDF":
        return "netcdf"
    raise ValueError(f"Unrecognized SSF file format for {path.name!r} (magic {magic!r})")


def _jd_to_datetime64(jd: "np.ndarray[Any, np.dtype[Any]]") -> "np.ndarray[Any, np.dtype[Any]]":
    """Julian Date (float days) -> datetime64[ns] (millisecond precision).

    NaN JD values (fill-masked time entries) are propagated as NaT.
    """
    nan_mask = np.isnan(jd)
    ms = np.where(nan_mask, 0.0, np.round((jd - _JD_EPOCH) * 86400.0 * 1e3))
    result = np.datetime64("1970-01-01T00:00:00").astype("datetime64[ms]") + ms.astype(
        "timedelta64[ms]"
    )
    result = result.astype("datetime64[ns]")
    result[nan_mask] = np.datetime64("NaT")
    return result


def _wrap_lon(lon: "np.ndarray[Any, np.dtype[Any]]") -> "np.ndarray[Any, np.dtype[Any]]":
    """Wrap 0-360 longitudes to [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def _as_datetime64_ns(value: Any) -> np.datetime64:
    """Coerce config/string datetime values to ``datetime64[ns]``."""
    return np.datetime64(value).astype("datetime64[ns]")


def _ssf_hour_from_filename(path: Path) -> np.datetime64 | None:
    """Return the hourly granule start time encoded in an SSF filename."""
    match = _SSF_STAMP_RE.search(path.name)
    if match is None:
        return None
    stamp = match.group(1)
    try:
        return np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:00:00").astype(
            "datetime64[ns]"
        )
    except ValueError:
        return None


def _filter_files_to_time_range(
    file_list: Sequence[Path],
    time_range: tuple[Any, Any] | None,
) -> list[Path]:
    """Prune filename-stamped hourly SSF granules before expensive reads."""
    if time_range is None:
        return list(file_list)
    start = _as_datetime64_ns(time_range[0])
    end = _as_datetime64_ns(time_range[1])
    if end < start:
        raise ValueError(f"Invalid CERES SSF time_range: end {end} precedes start {start}")

    kept: list[Path] = []
    for path in file_list:
        granule_start = _ssf_hour_from_filename(path)
        if granule_start is None:
            kept.append(path)
            continue
        granule_end = granule_start + np.timedelta64(1, "h")
        if granule_start <= end and granule_end > start:
            kept.append(path)
    return kept


@source_registry.register("ceres_ssf")
class CERESSSFReader:
    """Reader for CERES SSF L2 footprints (HDF4 Ed4A and netCDF Ed1C)."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ceres_ssf"

    @property
    def geometry(self) -> DataGeometry:
        """SSF footprints are swath data (binned to a grid for pairing)."""
        return DataGeometry.SWATH

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open SSF granule(s) and standardize to a (time,) footprint stream.

        Parameters
        ----------
        file_paths
            SSF granule paths (resource-fork ``._*`` sidecars are ignored).
            All granules in one open must share a format (HDF4 or netCDF).
        variables
            Canonical names from ``SSF_CATALOG`` (e.g. ``toa_lw_up``) or raw
            source names (HDF4 SDS name; ``"Group/var"`` for netCDF). If
            None, loads every catalog variable present in the files.
        **kwargs
            Unused; accepted for protocol compatibility.

        Returns
        -------
        xr.Dataset
            Footprint stream with ``lat``/``lon`` coords on the ``time`` dim
            and SWATH geometry tagged.
        """
        real = [Path(f) for f in file_paths if not Path(f).name.startswith("._")]
        file_list = validate_file_list(real, source_label="CERES SSF")
        file_list = _filter_files_to_time_range(file_list, time_range)
        file_list = validate_file_list(file_list, source_label="CERES SSF")

        formats = {_sniff_format(p) for p in file_list}
        if len(formats) > 1:
            raise ValueError(
                "CERES SSF open received mixed file formats (HDF4 and netCDF); "
                "open each edition separately"
            )
        fmt = formats.pop()

        opener = self._open_one_hdf4 if fmt == "hdf4" else self._open_one_netcdf
        per_file = [opener(path, variables) for path in file_list]
        ds = per_file[0] if len(per_file) == 1 else xr.concat(per_file, dim="time")
        ds = ds.sortby("time")
        if time_range is not None and "time" in ds:
            ds = ds.sel(
                time=slice(_as_datetime64_ns(time_range[0]), _as_datetime64_ns(time_range[1]))
            )
        return set_geometry_attr(ds, DataGeometry.SWATH)

    # -- HDF4 (Edition4A) ---------------------------------------------------

    def _open_one_hdf4(self, path: Path, variables: Sequence[str] | None) -> xr.Dataset:
        """Read one Edition4A HDF4 granule into a footprint Dataset."""
        try:
            from pyhdf.SD import SD, SDC
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "pyhdf is required to read CERES SSF HDF4 files (conda install pyhdf)"
            ) from exc

        hdf = SD(str(path), SDC.READ)
        try:
            available = hdf.datasets()

            def _read(sds_name: str) -> "np.ndarray[Any, np.dtype[Any]]":
                sds = hdf.select(sds_name)
                values = apply_hdf4_scale(np.asarray(sds.get()), sds.attributes())
                sds.endaccess()
                return values

            time_name = next((name for name in _H4_TIME_ALIASES if name in available), None)
            if time_name is None:
                raise ValueError(
                    f"CERES SSF time SDS not found in {path.name!r}: "
                    f"expected one of {_H4_TIME_ALIASES!r}"
                )

            time = _jd_to_datetime64(_read(time_name))
            lat = 90.0 - _read(_H4_COLAT)
            lon = _wrap_lon(_read(_H4_LON))

            requested = self._resolve(variables, available, fmt="hdf4", where=path.name)
            data_vars = {name: ("time", _read(src)) for name, src in requested.items()}
        finally:
            hdf.end()

        return self._assemble(time, lat, lon, data_vars)

    # -- netCDF (Edition1C) ---------------------------------------------------

    def _open_one_netcdf(self, path: Path, variables: Sequence[str] | None) -> xr.Dataset:
        """Read one Edition1C netCDF granule into a footprint Dataset.

        The ``Time_and_Position`` group contains a ``julian_geometry_time``
        variable whose units (``"days since -4712-01-01 12:00:00"``) use a
        non-standard epoch. Decoding it emits a ``CFWarning`` (a
        ``UserWarning`` subclass); this repo's pytest config escalates
        ``UserWarning`` to errors, so the open fails under tests even with
        cftime installed — and in cftime-less environments it fails
        unconditionally. The variable duplicates the ``time`` coordinate and is
        not needed; drop it so xarray decodes ``time`` normally.
        """
        pos = xr.open_dataset(
            str(path),
            group=_NC_POSITION_GROUP,
            decode_times=True,
            drop_variables=["julian_geometry_time"],
        )
        try:
            time = pos[_NC_TIME].values
            lat = np.asarray(pos[_NC_LAT].values, dtype=np.float64)
            lon = _wrap_lon(np.asarray(pos[_NC_LON].values, dtype=np.float64))
        finally:
            pos.close()

        # Group requested variables by their netCDF group to minimize opens.
        requested = self._resolve(variables, None, fmt="netcdf", where=path.name)
        by_group: dict[str, dict[str, str]] = {}
        for name, src in requested.items():
            group, _, var = src.partition("/")
            by_group.setdefault(group, {})[name] = var

        scan = variables is None  # catalog scan skips absentees; explicit requests raise
        data_vars: dict[str, Any] = {}
        for group, mapping in by_group.items():
            grp = xr.open_dataset(str(path), group=group, decode_times=False)
            try:
                missing = [v for v in mapping.values() if v not in grp.data_vars]
                if missing:
                    mapping = {
                        name: (
                            model_b_var
                            if var not in grp.data_vars
                            and (model_b_var := var.replace("dataset_b_", "model_b_"))
                            in grp.data_vars
                            else var
                        )
                        for name, var in mapping.items()
                    }
                    missing = [v for v in mapping.values() if v not in grp.data_vars]
                if missing and not scan:
                    raise ValueError(
                        f"SSF variable(s) not found in {path.name!r} group "
                        f"{group!r}: {missing!r}"
                    )
                for name, var in mapping.items():
                    if var in grp.data_vars:
                        data_vars[name] = (
                            "time",
                            np.asarray(grp[var].values, dtype=np.float64),
                        )
            finally:
                grp.close()
        if scan and not data_vars:
            raise ValueError(
                f"No catalog SSF variables found in {path.name!r}; " "request explicit source names"
            )

        return self._assemble(time, lat, lon, data_vars)

    # -- shared ----------------------------------------------------------------

    def _resolve(
        self,
        variables: Sequence[str] | None,
        available: dict[str, Any] | None,
        *,
        fmt: str,
        where: str,
    ) -> dict[str, str]:
        """Map requested names to per-format source names.

        For HDF4, ``available`` is the SDS inventory and resolution is
        validated against it here. For netCDF, ``available`` is None and
        membership is validated group-by-group by the caller.
        """
        if variables is None:
            names = list(SSF_CATALOG)
        else:
            names = list(variables)

        resolved: dict[str, str] = {}
        for name in names:
            entry = SSF_CATALOG.get(name)
            if fmt == "hdf4":
                src = entry.hdf4_sds if entry is not None else name
                if available is not None and src not in available and "Dataset B" in src:
                    model_b_src = src.replace("Dataset B", "Model B")
                    if model_b_src in available:
                        src = model_b_src
            else:
                src = f"{entry.nc_group}/{entry.nc_var}" if entry is not None else name
            resolved[name] = src

        if available is not None:
            missing = [n for n, s in resolved.items() if s not in available]
            if variables is None:
                resolved = {n: s for n, s in resolved.items() if s in available}
                if not resolved:
                    raise ValueError(
                        f"No catalog SSF variables found in {where!r}; "
                        "request explicit source names"
                    )
            elif missing:
                raise ValueError(f"SSF variable(s) not found in {where!r}: {missing!r}")
        elif variables is None:
            # netCDF catalog-scan: all catalog entries share one group; the
            # caller validates membership and raises if a name is absent.
            pass
        return resolved

    @staticmethod
    def _assemble(
        time: "np.ndarray[Any, np.dtype[Any]]",
        lat: "np.ndarray[Any, np.dtype[Any]]",
        lon: "np.ndarray[Any, np.dtype[Any]]",
        data_vars: dict[str, Any],
    ) -> xr.Dataset:
        """Build the footprint Dataset, dropping footprints without position."""
        time_ns = np.asarray(time, dtype="datetime64[ns]")
        valid = (
            ~np.isnat(time_ns)
            & ~np.isnan(np.asarray(lat, dtype=np.float64))
            & ~np.isnan(np.asarray(lon, dtype=np.float64))
        )
        ds = xr.Dataset(
            {name: (dims, vals[valid]) for name, (dims, vals) in data_vars.items()},
            coords={
                "time": time_ns[valid],
                "lat": ("time", np.asarray(lat, dtype=np.float64)[valid]),
                "lon": ("time", np.asarray(lon, dtype=np.float64)[valid]),
            },
        )
        return ds
