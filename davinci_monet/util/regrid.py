"""Rectilinear spherical-grid regridding helpers.

Horizontal coordinates are interpreted as cell centers. Latitude cell edges
are the midpoints between adjacent centers, with the exterior half-spacing
extrapolated and clipped to the poles. Longitude edges use cyclic midpoints for
full 360-degree coverage after centers are normalized to ``[-180, 180)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from davinci_monet.core.coordinates import wrap_longitudes

CoordinateValues = xr.DataArray | NDArray[np.floating] | Sequence[float]

_TARGET_LAT_DIM = "__target_lat"
_TARGET_LON_DIM = "__target_lon"


@dataclass(frozen=True)
class _CoarseningGeometry:
    target_lat: NDArray[np.float64]
    target_lon: NDArray[np.float64]
    latitude_weights: NDArray[np.float64]
    longitude_weights: NDArray[np.float64]


def area_weighted_coarsen(
    data: xr.DataArray,
    *,
    resolution: float | None = None,
    target_lat: CoordinateValues | None = None,
    target_lon: CoordinateValues | None = None,
    latitude_name: str = "lat",
    longitude_name: str = "lon",
) -> xr.DataArray:
    """Conservatively average rectilinear cell means onto a target grid.

    The remapping weight is the exact spherical overlap area between each
    source and target cell (apart from the common radius-squared factor).
    Missing source values are excluded and the remaining overlap weights are
    renormalized independently for every non-spatial index.

    Specify either a global, regular ``resolution`` in degrees or both target
    center arrays. Explicit target centers may be non-commensurate with the
    source grid. Returned latitude and longitude coordinates are increasing,
    and longitude is canonicalized to ``[-180, 180)``.
    """
    source = _prepare_source(data, latitude_name, longitude_name)
    geometry = _coarsening_geometry(
        source,
        resolution=resolution,
        target_lat=target_lat,
        target_lon=target_lon,
        latitude_name=latitude_name,
        longitude_name=longitude_name,
    )
    valid = np.isfinite(source)
    numerator = _weighted_sum(
        xr.where(valid, source, 0.0),
        geometry,
        latitude_name,
        longitude_name,
    )
    denominator = _weighted_sum(
        xr.where(valid, 1.0, 0.0),
        geometry,
        latitude_name,
        longitude_name,
    )
    result = numerator / denominator.where(denominator > 0.0)
    result = _finish_coarsening_result(
        result,
        source,
        geometry,
        latitude_name,
        longitude_name,
    )
    result.name = data.name
    result.attrs = dict(data.attrs)
    result.attrs.update(
        {
            "regrid_method": "spherical_area_weighted_cell_overlap",
            "grid_cell_convention": "coordinates_are_cell_centers",
        }
    )
    return result


def area_weighted_coarsen_variance(
    standard_deviation: xr.DataArray,
    valid: xr.DataArray,
    *,
    resolution: float | None = None,
    target_lat: CoordinateValues | None = None,
    target_lon: CoordinateValues | None = None,
    latitude_name: str = "lat",
    longitude_name: str = "lon",
) -> xr.DataArray:
    """Propagate independent errors through an area-weighted mean.

    For normalized finite-cell weights ``alpha``, the returned standard
    deviation is ``sqrt(sum(alpha**2 * standard_deviation**2))``. ``valid``
    must identify the value cells used by the corresponding coarsening; a
    missing, non-finite, or negative standard deviation is excluded as well.
    """
    source_std = _prepare_source(standard_deviation, latitude_name, longitude_name)
    source_valid = _prepare_source(valid, latitude_name, longitude_name)
    source_std, source_valid = xr.align(source_std, source_valid, join="exact", copy=False)
    source_valid = source_valid.broadcast_like(source_std)

    geometry = _coarsening_geometry(
        source_std,
        resolution=resolution,
        target_lat=target_lat,
        target_lon=target_lon,
        latitude_name=latitude_name,
        longitude_name=longitude_name,
    )
    mask = source_valid.fillna(False).astype(bool)
    mask = mask & np.isfinite(source_std) & (source_std >= 0.0)
    variance_numerator = _weighted_sum(
        xr.where(mask, source_std**2, 0.0),
        geometry,
        latitude_name,
        longitude_name,
        square_weights=True,
    )
    denominator = _weighted_sum(
        xr.where(mask, 1.0, 0.0),
        geometry,
        latitude_name,
        longitude_name,
    )
    result = variance_numerator**0.5 / denominator.where(denominator > 0.0)
    result = _finish_coarsening_result(
        result,
        source_std,
        geometry,
        latitude_name,
        longitude_name,
    )
    result.name = standard_deviation.name
    result.attrs = dict(standard_deviation.attrs)
    result.attrs.update(
        {
            "regrid_method": "independent_error_area_weighted_mean",
            "grid_cell_convention": "coordinates_are_cell_centers",
        }
    )
    return result


def periodic_bilinear(
    data: xr.DataArray,
    target_lat: CoordinateValues,
    target_lon: CoordinateValues,
    *,
    latitude_name: str = "lat",
    longitude_name: str = "lon",
) -> xr.DataArray:
    """Bilinearly interpolate a rectilinear field across the longitude seam.

    Longitude interpolation is periodic. Latitude queries outside the source
    center range use the nearest polar row, consistent with center-defined
    cells extending from the first/last midpoint edge to the pole. Target
    coordinates are normalized and sorted before interpolation.
    """
    source = _prepare_source(data, latitude_name, longitude_name)
    destination_lat = _sorted_latitudes(target_lat, minimum_size=1)
    destination_lon = _sorted_longitudes(target_lon, minimum_size=1)
    source_lat = np.asarray(source[latitude_name].values, dtype=np.float64)
    source_lon = np.asarray(source[longitude_name].values, dtype=np.float64)

    lat_hi = np.searchsorted(source_lat, destination_lat, side="right")
    lat_hi = np.clip(lat_hi, 1, source_lat.size - 1)
    lat_lo = lat_hi - 1
    lat_fraction = np.clip(
        (destination_lat - source_lat[lat_lo]) / (source_lat[lat_hi] - source_lat[lat_lo]),
        0.0,
        1.0,
    )

    first_longitude = source_lon[0]
    query_lon = np.mod(destination_lon - first_longitude, 360.0) + first_longitude
    extended_lon = np.concatenate((source_lon, [first_longitude + 360.0]))
    extended_source = xr.concat(
        [
            source,
            source.isel({longitude_name: slice(0, 1)}).assign_coords(
                {longitude_name: [first_longitude + 360.0]}
            ),
        ],
        dim=longitude_name,
    )
    lon_hi = np.searchsorted(extended_lon, query_lon, side="right")
    lon_hi = np.clip(lon_hi, 1, extended_lon.size - 1)
    lon_lo = lon_hi - 1
    lon_fraction = (query_lon - extended_lon[lon_lo]) / (
        extended_lon[lon_hi] - extended_lon[lon_lo]
    )

    lat_lo_index = xr.DataArray(lat_lo, dims=(latitude_name,))
    lat_hi_index = xr.DataArray(lat_hi, dims=(latitude_name,))
    lon_lo_index = xr.DataArray(lon_lo, dims=(longitude_name,))
    lon_hi_index = xr.DataArray(lon_hi, dims=(longitude_name,))
    target_coords = {
        latitude_name: destination_lat,
        longitude_name: destination_lon,
    }

    def sample(lat_index: xr.DataArray, lon_index: xr.DataArray) -> xr.DataArray:
        sampled = extended_source.isel({latitude_name: lat_index, longitude_name: lon_index})
        return sampled.assign_coords(target_coords)

    lower_left = sample(lat_lo_index, lon_lo_index)
    lower_right = sample(lat_lo_index, lon_hi_index)
    upper_left = sample(lat_hi_index, lon_lo_index)
    upper_right = sample(lat_hi_index, lon_hi_index)
    lon_weight = xr.DataArray(
        lon_fraction,
        dims=(longitude_name,),
        coords={longitude_name: destination_lon},
    )
    lat_weight = xr.DataArray(
        lat_fraction,
        dims=(latitude_name,),
        coords={latitude_name: destination_lat},
    )
    lower = _linear_interpolate(lower_left, lower_right, lon_weight)
    upper = _linear_interpolate(upper_left, upper_right, lon_weight)
    result = _linear_interpolate(lower, upper, lat_weight)
    result = result.transpose(*source.dims)
    result = _assign_spatial_coordinate_attrs(result, source, latitude_name, longitude_name)
    result.name = data.name
    result.attrs = dict(data.attrs)
    result.attrs.update(
        {
            "regrid_method": "periodic_bilinear",
            "longitude_convention": "[-180, 180)",
        }
    )
    return result


def _prepare_source(data: xr.DataArray, latitude_name: str, longitude_name: str) -> xr.DataArray:
    if not isinstance(data, xr.DataArray):
        raise TypeError("regridding input must be an xarray.DataArray")
    if latitude_name not in data.dims or longitude_name not in data.dims:
        raise ValueError(f"input must have {latitude_name!r} and {longitude_name!r} dimensions")
    if latitude_name not in data.coords or longitude_name not in data.coords:
        raise ValueError("latitude and longitude dimensions must have coordinate values")
    if data[latitude_name].ndim != 1 or data[longitude_name].ndim != 1:
        raise ValueError("only rectilinear one-dimensional coordinates are supported")
    if not (np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)):
        raise TypeError("regridding input must contain numeric or boolean values")

    source_lat = _sorted_latitudes(data[latitude_name], minimum_size=2)
    original_lon = np.asarray(data[longitude_name].values, dtype=np.float64)
    if original_lon.ndim != 1 or original_lon.size < 2 or not np.isfinite(original_lon).all():
        raise ValueError("longitude coordinates must contain at least two finite values")
    normalized_lon = wrap_longitudes(original_lon)
    lon_order = np.argsort(normalized_lon)
    source_lon = normalized_lon[lon_order]
    _require_unique(source_lon, "longitude")

    original_lat = np.asarray(data[latitude_name].values, dtype=np.float64)
    lat_order = np.argsort(original_lat)
    source = data.isel({latitude_name: lat_order, longitude_name: lon_order})
    source = source.assign_coords(
        {
            latitude_name: source_lat,
            longitude_name: source_lon,
        }
    )
    return _assign_spatial_coordinate_attrs(source, data, latitude_name, longitude_name)


def _coarsening_geometry(
    source: xr.DataArray,
    *,
    resolution: float | None,
    target_lat: CoordinateValues | None,
    target_lon: CoordinateValues | None,
    latitude_name: str,
    longitude_name: str,
) -> _CoarseningGeometry:
    destination_lat, destination_lon, regular_resolution = _target_grid(
        resolution, target_lat, target_lon
    )
    source_lat = np.asarray(source[latitude_name].values, dtype=np.float64)
    source_lon = np.asarray(source[longitude_name].values, dtype=np.float64)
    source_lat_edges = _latitude_edges(source_lat)
    source_lon_edges = _longitude_edges(source_lon)
    target_lat_edges = _latitude_edges(destination_lat, spacing=regular_resolution)
    target_lon_edges = _longitude_edges(destination_lon)
    return _CoarseningGeometry(
        target_lat=destination_lat,
        target_lon=destination_lon,
        latitude_weights=_latitude_overlap(source_lat_edges, target_lat_edges),
        longitude_weights=_longitude_overlap(source_lon_edges, target_lon_edges),
    )


def _target_grid(
    resolution: float | None,
    target_lat: CoordinateValues | None,
    target_lon: CoordinateValues | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float | None]:
    if resolution is not None:
        if target_lat is not None or target_lon is not None:
            raise ValueError("specify resolution or explicit target coordinates, not both")
        grid_resolution = float(resolution)
        if not np.isfinite(grid_resolution) or grid_resolution <= 0.0:
            raise ValueError("resolution must be a positive finite number")
        latitude_count = int(round(180.0 / grid_resolution))
        longitude_count = int(round(360.0 / grid_resolution))
        if (
            latitude_count < 1
            or longitude_count < 2
            or not np.isclose(latitude_count * grid_resolution, 180.0)
            or not np.isclose(longitude_count * grid_resolution, 360.0)
        ):
            raise ValueError("resolution must divide both 180 and 360 degrees exactly")
        latitudes = -90.0 + grid_resolution * (np.arange(latitude_count) + 0.5)
        longitudes = -180.0 + grid_resolution * (np.arange(longitude_count) + 0.5)
        return (
            np.asarray(latitudes, dtype=np.float64),
            np.asarray(longitudes, dtype=np.float64),
            grid_resolution,
        )

    if target_lat is None or target_lon is None:
        raise ValueError("both target_lat and target_lon are required without resolution")
    return (
        _sorted_latitudes(target_lat, minimum_size=2),
        _sorted_longitudes(target_lon, minimum_size=2),
        None,
    )


def _sorted_latitudes(values: CoordinateValues, *, minimum_size: int) -> NDArray[np.float64]:
    latitudes = np.asarray(values, dtype=np.float64)
    if latitudes.ndim != 1 or latitudes.size < minimum_size or not np.isfinite(latitudes).all():
        raise ValueError(
            f"latitude coordinates must contain at least {minimum_size} finite value(s)"
        )
    if np.any((latitudes < -90.0) | (latitudes > 90.0)):
        raise ValueError("latitude coordinates must lie within [-90, 90]")
    latitudes = np.sort(latitudes)
    _require_unique(latitudes, "latitude")
    return latitudes


def _sorted_longitudes(values: CoordinateValues, *, minimum_size: int) -> NDArray[np.float64]:
    longitudes = np.asarray(values, dtype=np.float64)
    if longitudes.ndim != 1 or longitudes.size < minimum_size or not np.isfinite(longitudes).all():
        raise ValueError(
            f"longitude coordinates must contain at least {minimum_size} finite value(s)"
        )
    longitudes = np.sort(wrap_longitudes(longitudes))
    _require_unique(longitudes, "longitude")
    return longitudes


def _require_unique(values: NDArray[np.float64], coordinate: str) -> None:
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{coordinate} coordinates must be unique after normalization")


def _latitude_edges(
    centers: NDArray[np.float64], *, spacing: float | None = None
) -> NDArray[np.float64]:
    if centers.size == 1:
        if spacing is None:
            raise ValueError("a single target latitude requires a regular resolution")
        edges = np.array([centers[0] - spacing / 2.0, centers[0] + spacing / 2.0])
    else:
        edges = np.empty(centers.size + 1, dtype=np.float64)
        edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
        edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
        edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    edges = np.clip(edges, -90.0, 90.0)
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("latitude centers imply empty or overlapping cells")
    return edges


def _longitude_edges(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    previous_last = centers[-1] - 360.0
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[0] = (previous_last + centers[0]) / 2.0
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[-1] = edges[0] + 360.0
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("longitude centers imply empty or overlapping cyclic cells")
    return edges


def _latitude_overlap(
    source_edges: NDArray[np.float64], target_edges: NDArray[np.float64]
) -> NDArray[np.float64]:
    lower = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    upper = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    overlap = np.maximum(
        np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower)),
        0.0,
    )
    return np.asarray(overlap, dtype=np.float64)


def _longitude_overlap(
    source_edges: NDArray[np.float64], target_edges: NDArray[np.float64]
) -> NDArray[np.float64]:
    overlap = np.zeros((target_edges.size - 1, source_edges.size - 1), dtype=np.float64)
    for shift in (-360.0, 0.0, 360.0):
        lower = np.maximum(target_edges[:-1, None], source_edges[None, :-1] + shift)
        upper = np.minimum(target_edges[1:, None], source_edges[None, 1:] + shift)
        overlap += np.maximum(upper - lower, 0.0)
    return np.deg2rad(overlap)


def _weighted_sum(
    values: xr.DataArray,
    geometry: _CoarseningGeometry,
    latitude_name: str,
    longitude_name: str,
    *,
    square_weights: bool = False,
) -> xr.DataArray:
    latitude_weights = geometry.latitude_weights
    longitude_weights = geometry.longitude_weights
    if square_weights:
        latitude_weights = latitude_weights**2
        longitude_weights = longitude_weights**2
    lon_weights = xr.DataArray(
        longitude_weights,
        dims=(_TARGET_LON_DIM, longitude_name),
        coords={
            _TARGET_LON_DIM: geometry.target_lon,
            longitude_name: values[longitude_name],
        },
    )
    lat_weights = xr.DataArray(
        latitude_weights,
        dims=(_TARGET_LAT_DIM, latitude_name),
        coords={
            _TARGET_LAT_DIM: geometry.target_lat,
            latitude_name: values[latitude_name],
        },
    )
    longitude_sum = xr.dot(values.astype(np.float64), lon_weights, dim=longitude_name)
    return xr.dot(longitude_sum, lat_weights, dim=latitude_name)


def _finish_coarsening_result(
    result: xr.DataArray,
    source: xr.DataArray,
    geometry: _CoarseningGeometry,
    latitude_name: str,
    longitude_name: str,
) -> xr.DataArray:
    result = result.rename({_TARGET_LAT_DIM: latitude_name, _TARGET_LON_DIM: longitude_name})
    result = result.assign_coords(
        {
            latitude_name: geometry.target_lat,
            longitude_name: geometry.target_lon,
        }
    )
    result = result.transpose(*source.dims)
    return _assign_spatial_coordinate_attrs(result, source, latitude_name, longitude_name)


def _assign_spatial_coordinate_attrs(
    result: xr.DataArray,
    source: xr.DataArray,
    latitude_name: str,
    longitude_name: str,
) -> xr.DataArray:
    result[latitude_name].attrs = dict(source[latitude_name].attrs)
    result[longitude_name].attrs = dict(source[longitude_name].attrs)
    return result


def _linear_interpolate(
    left: xr.DataArray, right: xr.DataArray, weight: xr.DataArray
) -> xr.DataArray:
    interpolated = left * (1.0 - weight) + right * weight
    interpolated = xr.where(weight == 0.0, left, interpolated)
    return xr.where(weight == 1.0, right, interpolated)


__all__ = ["area_weighted_coarsen", "area_weighted_coarsen_variance", "periodic_bilinear"]
