from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.stats.calculator import MONTH_TO_METEOROLOGICAL_SEASON


class _WindowSpec(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def start(self) -> Any: ...

    @property
    def end(self) -> Any: ...


def _label_time(value: Any, freq: str) -> str:
    try:
        stamp = pd.Timestamp(value)
        year = int(stamp.year)
        month = int(stamp.month)
        day = int(stamp.day)
    except (TypeError, ValueError):
        if not all(hasattr(value, attr) for attr in ("year", "month", "day")):
            raise
        year = int(value.year)
        month = int(value.month)
        day = int(value.day)
    if freq == "day":
        return f"{year:04d}-{month:02d}-{day:02d}"
    if freq == "month":
        return f"{year:04d}-{month:02d}"
    if freq == "season":
        season = MONTH_TO_METEOROLOGICAL_SEASON[month]
        # December belongs to the following year's DJF (e.g. Dec 2008 -> 2009-DJF),
        # matching the meteorological convention in stats/calculator.py.
        season_year = year + 1 if month == 12 else year
        return f"{season_year}-{season}"
    return "all"


def group_dataset(
    ds: xr.Dataset,
    groupby: str | Sequence[Mapping[str, Any] | _WindowSpec],
) -> "OrderedDict[str, xr.Dataset]":
    """Split a dataset into deterministic named groups."""

    groups: "OrderedDict[str, xr.Dataset]" = OrderedDict()
    if groupby == "all":
        groups["all"] = ds
        return groups
    if isinstance(groupby, str) and groupby in {"day", "month", "season"}:
        labels = [_label_time(value, str(groupby)) for value in ds["time"].values]
        for label in dict.fromkeys(labels):
            mask = np.asarray(labels) == label
            groups[label] = ds.isel(time=mask)
        return groups
    if isinstance(groupby, Sequence) and not isinstance(groupby, str):
        for window in groupby:
            if isinstance(window, Mapping):
                if not {"name", "start", "end"} <= set(window):
                    raise ValueError(f"unsupported custom window value: {window!r}")
                name = str(window["name"])
                start = window["start"]
                end = window["end"]
            else:
                if not all(hasattr(window, attr) for attr in ("name", "start", "end")):
                    raise ValueError(f"unsupported custom window value: {window!r}")
                name = str(window.name)
                start = window.start
                end = window.end
            groups[name] = ds.sel(time=slice(start, end))
        return groups
    raise ValueError(f"unsupported groupby value: {groupby!r}")


def _finite(values: xr.DataArray) -> np.ndarray:
    arr = np.asarray(values.values, dtype=float)
    return arr[np.isfinite(arr)]


def robust_limits(
    values: xr.DataArray, percentiles: tuple[float, float] = (2.0, 98.0)
) -> tuple[float, float]:
    finite = _finite(values)
    if finite.size == 0:
        return 0.0, 1.0
    bounds = np.asarray(np.nanpercentile(finite, percentiles), dtype=float)
    lo = float(bounds[0])
    hi = float(bounds[1])
    if lo == hi:
        pad = max(abs(lo) * 0.05, 0.05)
        return lo - pad, hi + pad
    return lo, hi


def symmetric_limits(values: xr.DataArray, fallback: float = 0.05) -> tuple[float, float]:
    finite = _finite(values)
    if finite.size == 0:
        return -fallback, fallback
    abs_values = np.abs(finite)
    quartiles = np.asarray(np.nanpercentile(abs_values, [25.0, 75.0]), dtype=float)
    q1 = float(quartiles[0])
    q3 = float(quartiles[1])
    iqr = q3 - q1
    if iqr > 0.0:
        robust = abs_values[abs_values <= q3 + 1.5 * iqr]
        if robust.size:
            abs_values = robust
    limit = float(np.nanmax(abs_values))
    if limit == 0.0:
        limit = fallback
    return -limit, limit


def product_summary(fields: dict[str, xr.DataArray]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for name, field in fields.items():
        finite = _finite(field)
        if finite.size == 0:
            summary[name] = {
                "finite_count": 0,
                "min": None,
                "mean": None,
                "max": None,
                "p02": None,
                "p50": None,
                "p98": None,
            }
            continue
        summary[name] = {
            "finite_count": int(finite.size),
            "min": float(np.nanmin(finite)),
            "mean": float(np.nanmean(finite)),
            "max": float(np.nanmax(finite)),
            "p02": float(np.nanpercentile(finite, 2.0)),
            "p50": float(np.nanpercentile(finite, 50.0)),
            "p98": float(np.nanpercentile(finite, 98.0)),
        }
    return summary
