"""Data contracts and reductions for corrected-MERRA-2 AOD figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from davinci_monet.core.base import PlotSeries


SEASONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("DJF", (12, 1, 2)),
    ("MAM", (3, 4, 5)),
    ("JJA", (6, 7, 8)),
    ("SON", (9, 10, 11)),
)


@dataclass(frozen=True)
class AODCorrectionInputs:
    """Aligned physical AOD fields and the applied correction support."""

    model: xr.DataArray
    corrected: xr.DataArray
    observation: xr.DataArray
    support: xr.DataArray
    lower_clip: xr.DataArray
    upper_clip: xr.DataArray
    ratio: xr.DataArray

    @property
    def valid(self) -> xr.DataArray:
        return cast(
            xr.DataArray,
            np.isfinite(self.model) & np.isfinite(self.corrected) & np.isfinite(self.observation),
        )


def correction_inputs(
    series: list[PlotSeries],
    *,
    model_source: str,
    corrected_source: str,
    observation_source: str,
) -> AODCorrectionInputs:
    """Resolve explicitly labeled sources and enforce their shared grid/time contract."""
    by_source = {str(item.source_label): item for item in series if item.source_label}
    missing = [
        label
        for label in (model_source, corrected_source, observation_source)
        if label not in by_source
    ]
    if missing:
        raise ValueError("aod_correction_science is missing labeled sources: " + ", ".join(missing))

    model_item = by_source[model_source]
    corrected_item = by_source[corrected_source]
    observation_item = by_source[observation_source]
    model = _physical_field(model_item.dataset[model_item.var_name], "model AOD")
    corrected = _physical_field(corrected_item.dataset[corrected_item.var_name], "corrected AOD")
    observation = _physical_field(
        observation_item.dataset[observation_item.var_name], "observed AOD"
    )

    scaling = corrected_item.dataset
    required = {"r", "spatial_support", "lower_clip_mask", "upper_clip_mask"}
    absent = sorted(required - set(scaling.data_vars))
    if absent:
        raise ValueError(
            "corrected AOD source is missing scaling diagnostics: " + ", ".join(absent)
        )
    ratio = _physical_field(scaling["r"], "correction ratio")
    support = _physical_field(scaling["spatial_support"], "correction support")
    lower_clip = _physical_field(scaling["lower_clip_mask"], "lower clip mask")
    upper_clip = _physical_field(scaling["upper_clip_mask"], "upper clip mask")

    try:
        model, corrected, observation, ratio, support, lower_clip, upper_clip = xr.align(
            model,
            corrected,
            observation,
            ratio,
            support,
            lower_clip,
            upper_clip,
            join="exact",
        )
    except ValueError as exc:
        raise ValueError(
            "model, corrected, and observed AOD do not share an exact time/analysis grid"
        ) from exc
    if not np.issubdtype(model["time"].dtype, np.datetime64):
        raise ValueError("AOD correction science requires a datetime time coordinate")
    return AODCorrectionInputs(
        model=model,
        corrected=corrected,
        observation=observation,
        support=support,
        lower_clip=lower_clip.astype(bool),
        upper_clip=upper_clip.astype(bool),
        ratio=ratio,
    )


def _physical_field(field: xr.DataArray, description: str) -> xr.DataArray:
    expected = {"time", "lat", "lon"}
    if set(field.dims) != expected or len(field.dims) != 3:
        raise ValueError(f"{description} must have dimensions time, lat, lon; got {field.dims}")
    return field.transpose("time", "lat", "lon")


def spatial_weights(field: xr.DataArray) -> xr.DataArray:
    """Cosine-latitude weights broadcastable to one time/lat/lon field."""
    weights = cast(
        xr.DataArray,
        np.cos(np.deg2rad(field["lat"].astype(float))),
    ).clip(min=0.0)
    return weights.rename("area_weight")


def weighted_metrics(
    estimate: xr.DataArray,
    observation: xr.DataArray,
    valid: xr.DataArray,
) -> dict[str, float]:
    """Area-weighted agreement metrics over all selected times and cells."""
    weights = spatial_weights(estimate).broadcast_like(estimate).where(valid)
    weight_sum = weights.sum(skipna=True)
    if _scalar(weight_sum) <= 0.0:
        return {"N": 0.0, "MB": np.nan, "MAE": np.nan, "RMSE": np.nan, "R": np.nan}

    x = observation.where(valid)
    y = estimate.where(valid)
    difference = y - x
    mean_x = (x * weights).sum(skipna=True) / weight_sum
    mean_y = (y * weights).sum(skipna=True) / weight_sum
    centered_x = x - mean_x
    centered_y = y - mean_y
    covariance = (centered_x * centered_y * weights).sum(skipna=True) / weight_sum
    variance_x = ((centered_x**2) * weights).sum(skipna=True) / weight_sum
    variance_y = ((centered_y**2) * weights).sum(skipna=True) / weight_sum
    denominator = cast(xr.DataArray, np.sqrt(variance_x * variance_y))
    correlation = covariance / denominator.where(denominator > 0.0)
    return {
        "N": _scalar(valid.sum()),
        "MB": _scalar((difference * weights).sum(skipna=True) / weight_sum),
        "MAE": _scalar((abs(difference) * weights).sum(skipna=True) / weight_sum),
        "RMSE": _scalar(
            cast(
                xr.DataArray,
                np.sqrt(((difference**2) * weights).sum(skipna=True) / weight_sum),
            )
        ),
        "R": _scalar(correlation),
    }


def area_weighted_timeseries(field: xr.DataArray, valid: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean for each time using the common observation mask."""
    weights = spatial_weights(field).broadcast_like(field).where(valid)
    denominator = weights.sum(("lat", "lon"), skipna=True)
    return ((field.where(valid) * weights).sum(("lat", "lon"), skipna=True) / denominator).where(
        denominator > 0.0
    )


def area_weighted_rmse_timeseries(
    estimate: xr.DataArray,
    observation: xr.DataArray,
    valid: xr.DataArray,
) -> xr.DataArray:
    """Area-weighted RMSE for each time on the common observation mask."""
    weights = spatial_weights(estimate).broadcast_like(estimate).where(valid)
    denominator = weights.sum(("lat", "lon"), skipna=True)
    squared = ((estimate - observation) ** 2).where(valid)
    rmse = cast(
        xr.DataArray,
        np.sqrt((squared * weights).sum(("lat", "lon"), skipna=True) / denominator),
    )
    return rmse.where(denominator > 0.0)


def seasonal_mean(field: xr.DataArray, months: tuple[int, ...]) -> xr.DataArray:
    """Mean one field over the requested calendar months."""
    selected = field.where(field["time"].dt.month.isin(months), drop=True)
    if selected.sizes.get("time", 0) == 0:
        raise ValueError(f"no samples are available for calendar months {months}")
    return selected.mean("time", skipna=True)


def scatter_sample(
    inputs: AODCorrectionInputs,
    *,
    max_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically sample matched cells without materializing the full cube."""
    shape = tuple(int(inputs.model.sizes[name]) for name in ("time", "lat", "lon"))
    total = int(np.prod(shape, dtype=np.int64))
    size = min(max_points, total)
    rng = np.random.default_rng(seed)
    flat = np.sort(rng.choice(total, size=size, replace=False))
    time_index, remainder = np.divmod(flat, shape[1] * shape[2])
    lat_index, lon_index = np.divmod(remainder, shape[2])
    indexers = {
        "time": xr.DataArray(time_index, dims="sample"),
        "lat": xr.DataArray(lat_index, dims="sample"),
        "lon": xr.DataArray(lon_index, dims="sample"),
    }
    model = np.asarray(inputs.model.isel(indexers).values, dtype=float)
    corrected = np.asarray(inputs.corrected.isel(indexers).values, dtype=float)
    observation = np.asarray(inputs.observation.isel(indexers).values, dtype=float)
    valid = np.isfinite(model) & np.isfinite(corrected) & np.isfinite(observation)
    return model[valid], corrected[valid], observation[valid]


def finite_percentile(fields: list[xr.DataArray], percentile: float) -> float:
    """Percentile over finite map values after spatial/time reduction."""
    values = np.concatenate([np.asarray(field.values, dtype=float).ravel() for field in fields])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("AOD correction maps contain no finite values")
    return float(np.percentile(finite, percentile))


def symmetric_limit(fields: list[xr.DataArray], percentile: float = 99.0) -> float:
    """Robust nonzero symmetric color limit for signed AOD fields."""
    limit = finite_percentile([abs(field) for field in fields], percentile)
    return limit if limit > 0.0 else 1.0


def _scalar(value: xr.DataArray) -> float:
    if value.chunks is not None:
        value = value.compute()
    return float(value.item())


__all__ = [
    "AODCorrectionInputs",
    "SEASONS",
    "area_weighted_rmse_timeseries",
    "area_weighted_timeseries",
    "correction_inputs",
    "finite_percentile",
    "scatter_sample",
    "seasonal_mean",
    "symmetric_limit",
    "weighted_metrics",
]
