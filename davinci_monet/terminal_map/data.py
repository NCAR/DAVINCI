"""Lazy NetCDF selection and rectilinear geographic interpolation for ASCII maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from davinci_monet.core.coordinates import (
    LATITUDE_NAMES,
    LONGITUDE_NAMES,
    surface_index,
    vertical_dim_name,
    wrap_longitudes,
)


@dataclass(frozen=True)
class DisplayGrid:
    """Global cell centers, north first, with one character per target cell."""

    rows: int = 180
    columns: int = 360

    def __post_init__(self) -> None:
        if self.rows < 1 or self.columns < 1:
            raise ValueError("Grid rows and columns must be positive.")
        if self.rows * self.columns > 10_000_000:
            raise ValueError("The interactive display grid is limited to 10 million cells.")

    @property
    def latitudes(self) -> np.ndarray:
        return 90.0 - (np.arange(self.rows) + 0.5) * 180.0 / self.rows

    @property
    def longitudes(self) -> np.ndarray:
        return -180.0 + (np.arange(self.columns) + 0.5) * 360.0 / self.columns


def _coordinate(ds: xr.Dataset, names: tuple[str, ...], standard_name: str) -> str:
    candidates = [name for name in names if name in ds.variables]
    candidates.extend(
        str(name)
        for name, var in ds.variables.items()
        if var.attrs.get("standard_name") == standard_name and str(name) not in candidates
    )
    if not candidates:
        raise ValueError(f"No {standard_name} coordinate found in the file.")
    name = candidates[0]
    if ds[name].ndim != 1:
        raise ValueError(
            f"{name} is not one-dimensional; this viewer currently requires "
            "rectilinear latitude/longitude coordinates."
        )
    return name


def is_land_water_ice(field: xr.DataArray) -> bool:
    """Recognize MERRA-2's categorical LWI flag, including its native metadata."""
    description = str(field.attrs.get("long_name", "")).lower()
    return str(field.name).upper() == "LWI" or "land(1)_water(0)_ice(2)" in description


def time_dimension(field: xr.DataArray) -> str | None:
    for dim in field.dims:
        name = str(dim)
        coord = field.coords.get(name)
        if name.lower() in {"time", "times", "date"} or (
            coord is not None
            and (
                coord.attrs.get("axis", "").upper() == "T"
                or coord.attrs.get("standard_name") == "time"
            )
        ):
            return name
    return None


class MapDataset:
    """Select just one 2-D slice before loading data from an open xarray dataset.

    ``open`` owns and closes the file; wrapping an existing Dataset leaves its
    lifetime with the caller. The most recently selected slice is cached, so
    panning and grid-size changes do not reload the original field.
    """

    def __init__(self, dataset: xr.Dataset, *, filename: str = "xarray dataset") -> None:
        self.dataset = dataset
        self.filename = filename
        self._owns_dataset = False
        self.latitude = _coordinate(dataset, LATITUDE_NAMES, "latitude")
        self.longitude = _coordinate(dataset, LONGITUDE_NAMES, "longitude")
        self.latitude_dim = str(dataset[self.latitude].dims[0])
        self.longitude_dim = str(dataset[self.longitude].dims[0])
        if self.latitude_dim == self.longitude_dim:
            raise ValueError("Latitude and longitude must use separate grid dimensions.")
        self.variables = [
            str(name)
            for name, var in dataset.data_vars.items()
            if self.latitude_dim in var.dims
            and self.longitude_dim in var.dims
            and var.dtype.kind in "iuf"
            and name not in (self.latitude, self.longitude)
        ]
        if not self.variables:
            raise ValueError("No numeric variables with both horizontal dimensions found.")
        lat = np.asarray(dataset[self.latitude].values, dtype=float)
        lon = np.asarray(dataset[self.longitude].values, dtype=float)
        if len(lat) < 2 or len(lon) < 2:
            raise ValueError("At least two source latitudes and longitudes are required.")
        if not np.isfinite(lat).all() or not np.isfinite(lon).all():
            raise ValueError("Latitude and longitude coordinates must be finite.")
        if (
            np.any(np.abs(lat) > 90)
            or "radian" in str(dataset[self.latitude].attrs.get("units", "")).lower()
            or "radian" in str(dataset[self.longitude].attrs.get("units", "")).lower()
        ):
            raise ValueError("Coordinates must be geographic latitude/longitude in degrees.")
        if len(np.unique(lat)) != len(lat):
            raise ValueError("Source latitude coordinates must be unique.")
        self._cache_key: tuple[Any, ...] | None = None
        self._cache: xr.DataArray | None = None

    @classmethod
    def open(cls, filename: str | Path) -> MapDataset:
        path = Path(filename).expanduser()
        if path.name.startswith("._"):
            raise ValueError("macOS ._ resource-fork files are not NetCDF datasets.")
        dataset = xr.open_dataset(path)
        try:
            result = cls(dataset, filename=str(path))
        except Exception:
            dataset.close()
            raise
        result._owns_dataset = True
        return result

    def __enter__(self) -> MapDataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_dataset:
            self.dataset.close()

    def dimensions(self, variable: str) -> list[str]:
        if variable not in self.variables:
            raise ValueError(f"{variable!r} is not a supported map variable.")
        return [
            str(dim)
            for dim in self.dataset[variable].dims
            if dim not in (self.latitude_dim, self.longitude_dim)
        ]

    def default_indices(self, variable: str) -> dict[str, int]:
        result = {dim: 0 for dim in self.dimensions(variable)}
        field = self.dataset[variable]
        vertical = vertical_dim_name(field)
        if vertical in result:
            result[vertical] = surface_index(field, vertical) % field.sizes[vertical]
        return result

    def select(self, variable: str, indices: dict[str, int] | None = None) -> xr.DataArray:
        """Load a single native horizontal slice, preserving raw units and metadata."""
        selected = self.default_indices(variable)
        for dim, index in (indices or {}).items():
            if dim not in selected:
                raise ValueError(f"{variable} has no selectable dimension {dim!r}.")
            size = self.dataset[variable].sizes[dim]
            if not -size <= index < size:
                raise ValueError(f"{dim} index {index} is outside [-{size}, {size - 1}].")
            selected[dim] = index % size
        key = (variable, *sorted(selected.items()))
        if key == self._cache_key and self._cache is not None:
            return self._cache
        field = (
            self.dataset[variable].isel(selected).transpose(self.latitude_dim, self.longitude_dim)
        )
        field = field.load()
        result = (
            xr.DataArray(
                np.asarray(field.values, dtype=float),
                dims=("lat", "lon"),
                coords={
                    "lat": np.asarray(self.dataset[self.latitude].values, dtype=float),
                    "lon": wrap_longitudes(self.dataset[self.longitude].values),
                },
                name=variable,
                attrs=dict(field.attrs),
            )
            .sortby("lat")
            .sortby("lon")
        )
        _, unique = np.unique(result.lon.values, return_index=True)
        result = result.isel(lon=unique)
        if result.sizes["lon"] < 2:
            raise ValueError("At least two distinct longitudes are required.")
        self._cache_key, self._cache = key, result
        return result


def interpolate(field: xr.DataArray, grid: DisplayGrid, method: str = "linear") -> np.ndarray:
    """Remap a selected slice; wrap global seams, and leave regional gaps missing.

    Latitude is never extrapolated. A longitude gap more than twice the median
    spacing identifies a regional grid; rotate that gap outside the interpolation
    interval, including for regional domains crossing the dateline.
    """
    if method not in {"linear", "nearest"}:
        raise ValueError("Interpolation must be 'linear' or 'nearest'.")
    lon = np.asarray(field.lon.values)
    gaps = np.diff(np.r_[lon, lon[0] + 360.0])
    typical_gap = float(np.median(np.sort(gaps)[:-1]))
    cyclic = float(gaps.max()) <= 2.0 * typical_gap
    if cyclic:
        left = field.isel(lon=[-1]).assign_coords(lon=[lon[-1] - 360.0])
        right = field.isel(lon=[0]).assign_coords(lon=[lon[0] + 360.0])
        field = xr.concat([left, field, right], dim="lon")
        target_lon = grid.longitudes
    else:
        start = (int(np.argmax(gaps)) + 1) % len(lon)
        order = np.r_[np.arange(start, len(lon)), np.arange(start)]
        unwrapped = np.r_[lon[start:], lon[:start] + 360.0]
        field = field.isel(lon=order).assign_coords(lon=unwrapped)
        target_lon = (grid.longitudes - unwrapped[0]) % 360.0 + unwrapped[0]
    interpolation: Literal["linear", "nearest"] = (
        "nearest" if method == "nearest" or is_land_water_ice(field) else "linear"
    )
    result = field.interp(lat=grid.latitudes, lon=target_lon, method=interpolation)
    return np.asarray(result.values, dtype=float)
