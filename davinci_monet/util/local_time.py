"""Deterministic calendar-day local-solar-time sampling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.coordinates import wrap_longitudes

_NS_PER_HOUR = 3_600_000_000_000


@dataclass(frozen=True)
class LocalSolarTimeIndex:
    """Reusable nearest-time index for every field in one source dataset."""

    indices: xr.DataArray
    target_time: xr.DataArray
    sample_time: xr.DataArray
    offset_hours: xr.DataArray
    valid: xr.DataArray
    source_time: xr.DataArray
    source_longitude: xr.DataArray


def build_local_solar_time_index(
    time: xr.DataArray,
    longitude: xr.DataArray,
    *,
    start_time: str | datetime | np.datetime64 | pd.Timestamp,
    end_time: str | datetime | np.datetime64 | pd.Timestamp,
    local_hour: float,
    day_anchor_hour: float = 12.0,
    tolerance: str | pd.Timedelta | np.timedelta64 | None = None,
    output_time_name: str = "time",
) -> LocalSolarTimeIndex:
    """Build nearest source-time indices for each calendar day and longitude.

    Longitude is interpreted periodically. When two source times are equally
    close, the earlier source time is selected.
    """
    _validate_hour(local_hour, "local_hour")
    _validate_hour(day_anchor_hour, "day_anchor_hour")
    if time.ndim != 1 or longitude.ndim != 1:
        raise ValueError("time and longitude coordinates must be one-dimensional")
    if time.size == 0 or longitude.size == 0:
        raise ValueError("time and longitude coordinates must be nonempty")

    source_ns = np.asarray(time.values, dtype="datetime64[ns]").astype(np.int64)
    if np.any(source_ns == np.datetime64("NaT", "ns").astype(np.int64)):
        raise ValueError("source time contains NaT")
    if np.any(np.diff(source_ns) <= 0):
        raise ValueError("source time must be strictly increasing")

    start = _timestamp(start_time, "start_time")
    end = _timestamp(end_time, "end_time")
    if end < start:
        raise ValueError("end_time must not precede start_time")
    days = pd.date_range(start.normalize(), end.normalize(), freq="1D")
    anchor_offset = pd.to_timedelta(day_anchor_hour, unit="h")
    anchors = (days + anchor_offset).to_numpy(dtype="datetime64[ns]")
    midnight_ns = days.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    lon_values = np.asarray(longitude.values, dtype=np.float64)
    if not np.isfinite(lon_values).all():
        raise ValueError("longitude contains non-finite values")
    wrapped_lon = wrap_longitudes(lon_values)
    local_offset_ns = np.rint((local_hour - wrapped_lon / 15.0) * _NS_PER_HOUR).astype(np.int64)
    target_ns = midnight_ns[:, None] + local_offset_ns[None, :]

    right = np.searchsorted(source_ns, target_ns, side="left")
    right_clipped = np.clip(right, 0, source_ns.size - 1)
    left_clipped = np.clip(right - 1, 0, source_ns.size - 1)
    left_distance = np.abs(target_ns - source_ns[left_clipped])
    right_distance = np.abs(source_ns[right_clipped] - target_ns)
    # Strict comparison makes an exact tie select the earlier (left) sample.
    indices = np.where(right_distance < left_distance, right_clipped, left_clipped)

    sampled_ns = source_ns[indices]
    valid = (target_ns >= source_ns[0]) & (target_ns <= source_ns[-1])
    if tolerance is not None:
        tolerance_ns = _timedelta_ns(tolerance)
        valid &= np.abs(sampled_ns - target_ns) <= tolerance_ns

    lon_dim = str(longitude.dims[0])
    coords = {
        output_time_name: anchors,
        lon_dim: (lon_dim, longitude.values, longitude.attrs),
    }
    dims = (output_time_name, lon_dim)
    target = xr.DataArray(
        target_ns.astype("datetime64[ns]"), dims=dims, coords=coords, name="target_time"
    )
    sampled = xr.DataArray(
        sampled_ns.astype("datetime64[ns]"), dims=dims, coords=coords, name="sample_time"
    )
    offsets = xr.DataArray(
        (sampled_ns - target_ns) / _NS_PER_HOUR,
        dims=dims,
        coords=coords,
        name="sample_offset_hours",
        attrs={"units": "h"},
    )
    valid_array = xr.DataArray(valid, dims=dims, coords=coords, name="sample_valid")
    index_array = xr.DataArray(indices, dims=dims, coords=coords, name="source_time_index")
    return LocalSolarTimeIndex(
        indices=index_array,
        target_time=target,
        sample_time=sampled,
        offset_hours=offsets,
        valid=valid_array,
        source_time=time.copy(deep=False),
        source_longitude=longitude.copy(deep=False),
    )


def apply_local_solar_time_index(
    data: xr.DataArray,
    index: LocalSolarTimeIndex,
    *,
    time_name: str = "time",
) -> xr.DataArray:
    """Apply a reusable local-solar-time index to one source field."""
    if time_name not in data.dims:
        raise ValueError(f"data is missing time dimension {time_name!r}")
    lon_dim = str(index.source_longitude.dims[0])
    if lon_dim not in data.dims:
        raise ValueError(f"data is missing longitude dimension {lon_dim!r}")
    if not np.array_equal(data[time_name].values, index.source_time.values):
        raise ValueError("data time coordinate does not match the sampling index")
    if not np.array_equal(data[lon_dim].values, index.source_longitude.values):
        raise ValueError("data longitude coordinate does not match the sampling index")

    output_time_name = str(index.indices.dims[0])
    temporary_time = "__local_solar_day"
    if temporary_time in data.dims:
        raise ValueError(f"reserved dimension {temporary_time!r} is already present")
    indexer = index.indices.rename({output_time_name: temporary_time})
    selected = data.isel({time_name: indexer}).drop_vars(time_name)
    selected = selected.rename({temporary_time: output_time_name})
    selected = selected.where(index.valid)
    selected = selected.assign_coords(
        sample_time=index.sample_time,
        sample_offset_hours=index.offset_hours,
        sample_valid=index.valid,
    )
    selected.attrs = dict(data.attrs)
    selected.attrs["local_solar_time_sampled"] = True
    return selected


def sample_local_solar_time(
    data: xr.DataArray,
    *,
    start_time: str | datetime | np.datetime64 | pd.Timestamp,
    end_time: str | datetime | np.datetime64 | pd.Timestamp,
    local_hour: float,
    day_anchor_hour: float = 12.0,
    tolerance: str | pd.Timedelta | np.timedelta64 | None = None,
    time_name: str = "time",
    longitude_name: str = "lon",
) -> tuple[xr.DataArray, LocalSolarTimeIndex]:
    """Sample a field at one local solar time and return its reusable index."""
    if longitude_name not in data.coords:
        raise ValueError(f"data is missing longitude coordinate {longitude_name!r}")
    index = build_local_solar_time_index(
        data[time_name],
        data[longitude_name],
        start_time=start_time,
        end_time=end_time,
        local_hour=local_hour,
        day_anchor_hour=day_anchor_hour,
        tolerance=tolerance,
        output_time_name=time_name,
    )
    return apply_local_solar_time_index(data, index, time_name=time_name), index


def _timestamp(value: str | datetime | np.datetime64 | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _timedelta_ns(value: str | pd.Timedelta | np.timedelta64) -> int:
    duration = pd.Timedelta(value)
    if pd.isna(duration) or duration < pd.Timedelta(0):
        raise ValueError("tolerance must be non-negative")
    return int(duration.value)


def _validate_hour(value: float, name: str) -> None:
    if not np.isfinite(value) or not 0.0 <= value < 24.0:
        raise ValueError(f"{name} must be finite and in [0, 24)")
