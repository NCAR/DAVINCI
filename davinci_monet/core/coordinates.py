"""Shared coordinate and vertical-convention helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr

LATITUDE_NAMES = ("lat", "latitude", "LAT", "LATITUDE", "Latitude")
LONGITUDE_NAMES = ("lon", "longitude", "LON", "LONGITUDE", "Longitude")
VERTICAL_DIM_NAMES = ("lev", "z", "level", "altitude", "height", "vertical")

_PRESSURE_UNITS = {
    "pa",
    "pascal",
    "pascals",
    "hpa",
    "hectopascal",
    "hectopascals",
    "mb",
    "mbar",
    "millibar",
    "millibars",
    "bar",
}
_HEIGHT_UNITS = {
    "m",
    "meter",
    "meters",
    "metre",
    "metres",
    "km",
    "kilometer",
    "kilometers",
    "kilometre",
    "kilometres",
    "ft",
    "feet",
}
_HEIGHT_NAMES = {"altitude", "alt", "height", "geometric_height", "geopotential_height"}
_PRESSURE_NAMES = {"lev", "level", "plev", "pressure", "p"}


def wrap_longitudes(values: Any) -> np.ndarray:
    """Wrap longitude values to the canonical ``[-180, 180)`` convention."""
    arr = np.asarray(values, dtype=np.float64)
    return ((arr + 180.0) % 360.0) - 180.0


def match_longitude_convention(values: Any, reference: Any) -> np.ndarray:
    """Return longitudes in the convention implied by ``reference`` values."""
    arr = np.asarray(values, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size == 0:
        return arr
    if np.nanmin(finite_ref) >= 0.0 and np.nanmax(finite_ref) > 180.0:
        return np.where(arr < 0.0, arr + 360.0, arr)
    if np.nanmin(finite_ref) < 0.0:
        return wrap_longitudes(arr)
    return arr


def normalize_post_load_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Apply DAVINCI's post-load horizontal coordinate contract.

    The contract is deliberately conservative:

    - promote latitude/longitude variables to coordinates;
    - add ``lat``/``lon`` aliases for ``latitude``/``longitude`` when possible;
    - sort only true 1-D grid latitude axes ascending;
    - wrap true 1-D longitudes to ``[-180, 180)`` and sort only grid axes;
    - leave point/track/swath ordering and curvilinear coordinates intact.
    """
    out = _promote_lat_lon_vars(ds)
    out = _alias_standard_lat_lon(out)
    out = _sort_latitude_axis(out)
    out = _wrap_longitude_coord(out)
    out = _sort_longitude_axis(out)
    return _alias_standard_lat_lon(out)


def vertical_dim_name(
    data: xr.Dataset | xr.DataArray, candidates: Iterable[str] = VERTICAL_DIM_NAMES
) -> str | None:
    """Return the best vertical dimension name for a dataset or data array."""
    dims = tuple(str(dim) for dim in data.dims)
    for name in candidates:
        if name in dims:
            return name

    coords = getattr(data, "coords", {})
    for coord_name in coords:
        coord = coords[coord_name]
        if not coord.dims:
            continue
        dim = str(coord.dims[0])
        if dim not in dims:
            continue
        if _coord_looks_vertical(str(coord_name), coord):
            return dim
    return None


def surface_index(data: xr.Dataset | xr.DataArray, level_dim: str | None = None) -> int:
    """Return the surface index for a vertical dimension.

    CF metadata wins over coordinate ordering: ``positive=up`` and length units
    select the lowest coordinate; ``positive=down`` and pressure units select
    the highest coordinate. Ordering is only a fallback for legacy unitless
    pressure-like levels.
    """
    dim = level_dim or vertical_dim_name(data)
    if dim is None or dim not in data.dims:
        return 0
    coord = _vertical_coord(data, dim)
    if coord is None:
        return 0

    vals = np.asarray(coord.values, dtype=np.float64).reshape(-1)
    if vals.size < 2 or not np.isfinite(vals).any():
        return 0

    attrs = {str(k).lower(): str(v).lower() for k, v in coord.attrs.items()}
    positive = attrs.get("positive", "").strip().lower()
    units = _normalized_units(attrs.get("units", ""))
    name = str(getattr(coord, "name", dim)).lower()
    standard_name = attrs.get("standard_name", "").strip().lower()

    if positive == "up":
        return _nan_arg(vals, "min")
    if positive == "down":
        return _nan_arg(vals, "max")
    if units in _PRESSURE_UNITS or "pressure" in standard_name:
        return _nan_arg(vals, "max")
    if units in _HEIGHT_UNITS or "altitude" in standard_name or "height" in standard_name:
        return _nan_arg(vals, "min")
    if name in _PRESSURE_NAMES:
        return _nan_arg(vals, "max")
    if name in _HEIGHT_NAMES:
        return _nan_arg(vals, "min")

    return -1 if vals[-1] > vals[0] else 0


def extract_surface(data: xr.Dataset, level_dim: str | None = None) -> xr.Dataset:
    """Return ``data`` sliced to its CF-aware surface level when vertical."""
    dim = level_dim or vertical_dim_name(data)
    if dim is None or dim not in data.dims:
        return data
    return data.isel({dim: surface_index(data, dim)})


def _promote_lat_lon_vars(ds: xr.Dataset) -> xr.Dataset:
    names = [name for name in (*LATITUDE_NAMES, *LONGITUDE_NAMES) if name in ds.data_vars]
    return ds.set_coords(names) if names else ds


def _alias_standard_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    out = ds
    if "latitude" in out.coords and "lat" not in out.coords:
        out = out.assign_coords(lat=out["latitude"])
    if "longitude" in out.coords and "lon" not in out.coords:
        out = out.assign_coords(lon=out["longitude"])
    return out


def _sort_latitude_axis(ds: xr.Dataset) -> xr.Dataset:
    axis_name = _dimension_axis_name(ds, LATITUDE_NAMES)
    if axis_name is None:
        return ds
    values = np.asarray(ds[axis_name].values, dtype=np.float64)
    if values.size > 1 and np.any(np.diff(values) < 0):
        return ds.sortby(axis_name)
    return ds


def _wrap_longitude_coord(ds: xr.Dataset) -> xr.Dataset:
    coord_name = _dimension_axis_name(ds, LONGITUDE_NAMES) or _first_1d_coord_name(
        ds, LONGITUDE_NAMES
    )
    if coord_name is None:
        return ds
    coord = ds[coord_name]
    wrapped = wrap_longitudes(coord.values)
    if np.array_equal(np.asarray(coord.values, dtype=np.float64), wrapped, equal_nan=True):
        return ds

    out = ds.assign_coords({coord_name: (coord.dims, wrapped, coord.attrs)})
    for alias in ("longitude", "lon"):
        if alias != coord_name and alias in out.coords and out[alias].dims == coord.dims:
            out = out.assign_coords({alias: (out[alias].dims, wrapped, out[alias].attrs)})
    return out


def _sort_longitude_axis(ds: xr.Dataset) -> xr.Dataset:
    axis_name = _dimension_axis_name(ds, LONGITUDE_NAMES)
    if axis_name is None:
        return ds
    values = np.asarray(ds[axis_name].values, dtype=np.float64)
    if values.size > 1 and np.any(np.diff(values) < 0):
        return ds.sortby(axis_name)
    return ds


def _dimension_axis_name(ds: xr.Dataset, candidates: Iterable[str]) -> str | None:
    other_candidates = (
        LONGITUDE_NAMES if any(name in LATITUDE_NAMES for name in candidates) else LATITUDE_NAMES
    )
    other_name = _first_1d_coord_name(ds, other_candidates)
    other_dim = (
        str(ds[other_name].dims[0]) if other_name is not None and ds[other_name].dims else None
    )
    for name in candidates:
        if name not in ds.coords:
            continue
        coord = ds[name]
        if coord.ndim != 1 or not coord.dims:
            continue
        dim = str(coord.dims[0])
        if dim == str(name) or (other_dim is not None and other_dim != dim):
            return str(name)
    return None


def _first_1d_coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in ds.coords and ds[name].ndim == 1:
            return str(name)
    return None


def _vertical_coord(data: xr.Dataset | xr.DataArray, dim: str) -> xr.DataArray | None:
    coords = getattr(data, "coords", {})
    if dim in coords:
        return coords[dim]
    for coord_name in coords:
        coord = coords[coord_name]
        if coord.ndim == 1 and coord.dims and str(coord.dims[0]) == dim:
            return coord
    return None


def _coord_looks_vertical(name: str, coord: xr.DataArray) -> bool:
    attrs = {str(k).lower(): str(v).lower() for k, v in coord.attrs.items()}
    units = _normalized_units(attrs.get("units", ""))
    standard_name = attrs.get("standard_name", "")
    axis = attrs.get("axis", "")
    lname = name.lower()
    return (
        axis == "z"
        or units in _PRESSURE_UNITS
        or units in _HEIGHT_UNITS
        or lname in _PRESSURE_NAMES
        or lname in _HEIGHT_NAMES
        or "pressure" in standard_name
        or "altitude" in standard_name
        or "height" in standard_name
    )


def _normalized_units(units: str) -> str:
    return units.strip().lower().replace(" ", "")


def _nan_arg(values: np.ndarray, mode: str) -> int:
    if mode == "min":
        return int(np.nanargmin(values))
    return int(np.nanargmax(values))
