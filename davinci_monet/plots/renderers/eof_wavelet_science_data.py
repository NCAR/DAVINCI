"""Data preparation for the aerosol EOF/wavelet science figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from davinci_monet.core.base import PlotSeries


SEASONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("DJF", (12, 1, 2)),
    ("MAM", (3, 4, 5)),
    ("JJA", (6, 7, 8)),
    ("SON", (9, 10, 11)),
)


def dataset_roles(series: list[PlotSeries]) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Identify the basis, projection, and filtered datasets by contract variables."""
    datasets = [item.dataset for item in series]
    basis = next(
        (ds for ds in datasets if {"eofs", "climatology", "time_mean"} <= set(ds.data_vars)),
        None,
    )
    projection = next(
        (
            ds
            for ds in datasets
            if {"pc", "resolution", "posterior_variance", "spatial_support"} <= set(ds.data_vars)
        ),
        None,
    )
    filtered = next(
        (
            ds
            for ds in datasets
            if {"pc", "power", "retained_variance", "recon_error"} <= set(ds.data_vars)
        ),
        None,
    )
    if basis is None or projection is None or filtered is None:
        raise ValueError(
            "eof_wavelet_science requires basis, projection, and wavelet-filter datasets"
        )
    return basis, projection, filtered


def validate_identity(
    basis: xr.Dataset,
    projection: xr.Dataset,
    filtered: xr.Dataset,
) -> None:
    """Require all three artifacts to describe the same de-seasonalized EOF basis."""
    projection_signature = str(projection.attrs.get("projection_basis_signature", ""))
    filtered_signature = str(filtered.attrs.get("projection_basis_signature", ""))
    if not projection_signature or projection_signature != filtered_signature:
        raise ValueError("projection and wavelet artifacts use incompatible EOF basis identities")
    if str(basis.attrs.get("eof_remove_seasonal_cycle", "")).lower() not in {"true", "1"}:
        raise ValueError("seasonal science figures require a de-seasonalized EOF basis")
    if str(basis.attrs.get("eof_quantity", "")) != "log_aod":
        raise ValueError("seasonal science figures require an EOF basis of log_aod")

    basis_modes = np.asarray(basis["mode"].values)
    basis_time = np.asarray(basis["time"].values)
    for role, dataset in (("projection", projection), ("wavelet", filtered)):
        if not np.array_equal(basis_modes, np.asarray(dataset["mode"].values)):
            raise ValueError(f"{role} mode coordinates do not match the EOF basis")
        if not np.array_equal(basis_time, np.asarray(dataset["time"].values)):
            raise ValueError(f"{role} time coordinates do not match the EOF basis")


def select_modes(dataset: xr.Dataset, requested: list[int] | None, count: int) -> list[int]:
    """Resolve and validate a requested subset of EOF modes."""
    available = [int(value) for value in dataset["mode"].values]
    selected = available[:count] if requested is None else [int(value) for value in requested]
    missing = [mode for mode in selected if mode not in available]
    if missing:
        raise ValueError(f"requested EOF modes are unavailable: {missing}")
    if not selected:
        raise ValueError("at least one EOF mode must be selected")
    return selected


def month_weights(basis: xr.Dataset) -> xr.DataArray:
    """Count represented days by calendar month, including leap days."""
    months = pd.DatetimeIndex(np.asarray(basis["time"].values)).month
    counts = np.bincount(months, minlength=13)[1:].astype(float)
    if np.any(counts == 0.0):
        raise ValueError("EOF time coordinate must cover every calendar month")
    return xr.DataArray(counts, dims=("month",), coords={"month": np.arange(1, 13)})


def _weighted_months(
    field: xr.DataArray,
    weights: xr.DataArray,
    months: tuple[int, ...],
) -> xr.DataArray:
    selected_weights = weights.sel(month=list(months))
    return (field.sel(month=list(months)) * selected_weights).sum("month") / float(
        selected_weights.sum().item()
    )


def seasonal_fields(
    basis: xr.Dataset,
    projection: xr.Dataset,
) -> tuple[list[xr.DataArray], list[xr.DataArray], list[xr.DataArray]]:
    """Recover absolute seasonal AOD, departures, and projected climatology bias."""
    weights = month_weights(basis)
    epsilon = float(basis.attrs.get("eof_input_log_epsilon", np.nan))
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("EOF basis is missing a positive shifted-log epsilon")

    monthly_log = basis["time_mean"] + basis["climatology"]
    monthly_aod: xr.DataArray = (
        xr.apply_ufunc(np.exp, monthly_log, dask="allowed") - epsilon
    ).clip(min=0.0)
    annual_aod = _weighted_months(monthly_aod, weights, tuple(range(1, 13)))

    bias = projection["clim_bias_applied"]
    support = projection["spatial_support"]
    climatologies: list[xr.DataArray] = []
    departures: list[xr.DataArray] = []
    biases: list[xr.DataArray] = []
    for _label, months in SEASONS:
        seasonal_aod = _weighted_months(monthly_aod, weights, months)
        seasonal_support = _weighted_months(support, weights, months)
        climatologies.append(seasonal_aod)
        departures.append(seasonal_aod - annual_aod)
        biases.append(_weighted_months(bias, weights, months).where(seasonal_support > 0.0))
    return climatologies, departures, biases


def finite_percentile(fields: list[xr.DataArray], percentile: float) -> float:
    """Return a percentile across all finite values in a list of fields."""
    values = np.concatenate([np.asarray(field.values, dtype=float).ravel() for field in fields])
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("map fields contain no finite values")
    return float(np.nanpercentile(finite, percentile))


def symmetric_limit(fields: list[xr.DataArray], percentile: float = 99.0) -> float:
    """Return a robust, nonzero symmetric color limit."""
    limit = finite_percentile([abs(field) for field in fields], percentile)
    return limit if limit > 0.0 else 1.0


def spatial_rms(
    basis: xr.Dataset,
    projection: xr.Dataset,
    filtered: xr.Dataset,
    *,
    chunk_size: int = 31,
) -> list[xr.DataArray]:
    """Reconstruct support-weighted projected, filtered, and removed RMS fields."""
    patterns = basis["eofs"].transpose("mode", "lat", "lon")
    projected_pc = projection["pc"].transpose("time", "mode")
    filtered_pc = filtered["pc"].transpose("time", "mode")
    support = projection["spatial_support"].transpose("month", "lat", "lon")

    pattern_values = np.asarray(patterns.values, dtype=float)
    projected_values = np.asarray(projected_pc.values, dtype=float)
    filtered_values = np.asarray(filtered_pc.values, dtype=float)
    support_values = np.asarray(support.values, dtype=float)
    month_lookup = {int(month): index for index, month in enumerate(support["month"].values)}
    time_months = pd.DatetimeIndex(np.asarray(projected_pc["time"].values)).month

    n_mode, n_lat, n_lon = pattern_values.shape
    flat_patterns = pattern_values.reshape(n_mode, n_lat * n_lon)
    sums = np.zeros((3, n_lat * n_lon), dtype=float)
    sample_count = 0
    for start in range(0, projected_values.shape[0], chunk_size):
        stop = min(start + chunk_size, projected_values.shape[0])
        pcoeff = projected_values[start:stop]
        fcoeff = filtered_values[start:stop]
        valid = np.isfinite(pcoeff).all(axis=1) & np.isfinite(fcoeff).all(axis=1)
        if not np.any(valid):
            continue
        pfield = pcoeff[valid] @ flat_patterns
        ffield = fcoeff[valid] @ flat_patterns
        block_months = time_months[start:stop][valid]
        block_support = np.stack(
            [support_values[month_lookup[int(month)]].reshape(-1) for month in block_months]
        )
        block_support = np.nan_to_num(block_support, nan=0.0, posinf=0.0, neginf=0.0)
        pfield *= block_support
        ffield *= block_support
        sums[0] += np.square(pfield).sum(axis=0)
        sums[1] += np.square(ffield).sum(axis=0)
        sums[2] += np.square(pfield - ffield).sum(axis=0)
        sample_count += int(valid.sum())
    if sample_count == 0:
        raise ValueError("projection and wavelet coefficients have no common finite samples")

    names = ("Projected", "Filtered 4–180 day", "Removed outside-band")
    outputs: list[xr.DataArray] = []
    for name, total in zip(names, sums, strict=True):
        outputs.append(
            xr.DataArray(
                np.sqrt(total / sample_count).reshape(n_lat, n_lon),
                dims=("lat", "lon"),
                coords={"lat": patterns["lat"], "lon": patterns["lon"]},
                attrs={"long_name": name, "units": "1"},
            )
        )
    return outputs


__all__ = [
    "SEASONS",
    "dataset_roles",
    "finite_percentile",
    "month_weights",
    "seasonal_fields",
    "select_modes",
    "spatial_rms",
    "symmetric_limit",
    "validate_identity",
]
