"""Bounded observation-row loading for the joint projection-bias fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from davinci_monet.analysis.projection_batches import LoadedObservationChunk, _load_chunk
from davinci_monet.analysis.projection_core import EffectiveCovariance, build_effective_covariance
from davinci_monet.analysis.projection_inputs import ProjectionObservation

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class JointFitDayRows:
    """One fit day's stacked, covariance-aligned observation rows."""

    values: FloatArray
    design: FloatArray
    covariance: EffectiveCovariance
    cell: IntArray
    sensor: IntArray


def _rows_for_day(
    model: FloatArray,
    chunks: Sequence[LoadedObservationChunk],
    local_day: int,
    patterns: FloatArray,
    support: BoolArray,
    latitude_rows: FloatArray,
) -> JointFitDayRows:
    ncell = model.shape[-2] * model.shape[-1]
    common_count = chunks[0].factors.shape[1]
    values: list[FloatArray] = []
    designs: list[FloatArray] = []
    errors: list[FloatArray] = []
    latitudes: list[FloatArray] = []
    factors: list[FloatArray] = []
    cells: list[IntArray] = []
    sensors: list[IntArray] = []
    for sensor, chunk in enumerate(chunks):
        usable = (
            chunk.valid[local_day]
            & np.isfinite(chunk.values[local_day])
            & np.isfinite(model[local_day])
            & support
        )
        flat = np.flatnonzero(usable.reshape(-1)).astype(np.int64)
        if not flat.size:
            continue
        cells.append(flat)
        sensors.append(np.full(flat.size, sensor, dtype=np.int64))
        values.append((chunk.values[local_day] - model[local_day]).reshape(-1)[flat])
        designs.append(patterns[:, flat].T)
        errors.append(chunk.errors[local_day].reshape(-1)[flat])
        latitudes.append(latitude_rows[flat])
        factor = chunk.factors[local_day].reshape(common_count, ncell)
        factors.append(factor[:, flat].T)
    if not values:
        return JointFitDayRows(
            np.empty(0),
            np.empty((0, patterns.shape[0])),
            EffectiveCovariance(np.empty(0), np.empty((0, common_count))),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    error = np.concatenate(errors)
    latitude = np.concatenate(latitudes)
    common = np.concatenate(factors)
    return JointFitDayRows(
        np.concatenate(values),
        np.concatenate(designs),
        build_effective_covariance(error, latitude, common),
        np.concatenate(cells),
        np.concatenate(sensors),
    )


def iter_joint_fit_days(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    patterns: FloatArray,
    support: FloatArray,
    months: IntArray,
    fit_mask: BoolArray,
    time_chunk_size: int,
) -> Iterator[tuple[int, int, JointFitDayRows]]:
    """Yield one covariance-aligned fit day while loading only one time chunk."""
    nlat, nlon = model.sizes["lat"], model.sizes["lon"]
    latitude_rows = np.broadcast_to(np.asarray(model["lat"].values)[:, None], (nlat, nlon)).reshape(
        -1
    )
    for start in range(0, model.sizes["time"], time_chunk_size):
        stop = min(start + time_chunk_size, model.sizes["time"])
        if not np.any(fit_mask[start:stop]):
            continue
        model_values, chunks = _load_chunk(model, observations, start, stop)
        for local_day, day in enumerate(range(start, stop)):
            if not fit_mask[day]:
                continue
            month = int(months[day]) - 1
            yield day, month, _rows_for_day(
                model_values,
                chunks,
                local_day,
                patterns,
                support[month] > 0.0,
                latitude_rows,
            )


def sensor_overlap_counts(
    model: xr.DataArray,
    observations: Sequence[ProjectionObservation],
    support: FloatArray,
    months: IntArray,
    fit_mask: BoolArray,
    time_chunk_size: int,
) -> IntArray:
    """Count same-cell/day overlap independently of innovation values."""
    count = np.zeros((len(observations), len(observations)), dtype=np.int64)
    for start in range(0, model.sizes["time"], time_chunk_size):
        stop = min(start + time_chunk_size, model.sizes["time"])
        if not np.any(fit_mask[start:stop]):
            continue
        model_values, chunks = _load_chunk(model, observations, start, stop)
        for local_day, day in enumerate(range(start, stop)):
            if not fit_mask[day]:
                continue
            supported = support[int(months[day]) - 1] > 0.0
            masks = [
                chunk.valid[local_day]
                & np.isfinite(chunk.values[local_day])
                & np.isfinite(model_values[local_day])
                & supported
                for chunk in chunks
            ]
            for left, left_mask in enumerate(masks):
                for right in range(left, len(masks)):
                    shared = int(np.any(left_mask & masks[right]))
                    count[left, right] += shared
                    if left != right:
                        count[right, left] += shared
    return count


def require_connected_sensor_overlap(count: IntArray) -> None:
    """Require the relative-offset overlap graph to have one component."""
    if count.shape[0] < 2:
        return
    reached = {0}
    frontier = [0]
    while frontier:
        left = frontier.pop()
        for right in range(count.shape[0]):
            if right not in reached and left != right and count[left, right] > 0:
                reached.add(right)
                frontier.append(right)
    if len(reached) != count.shape[0]:
        raise ValueError("joint sensor offsets require a connected pairwise-overlap graph")


__all__ = [
    "JointFitDayRows",
    "iter_joint_fit_days",
    "require_connected_sensor_overlap",
    "sensor_overlap_counts",
]
