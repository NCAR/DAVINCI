from __future__ import annotations

import cftime
import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis.gridded_reductions import (
    group_dataset,
    product_summary,
    robust_limits,
    symmetric_limits,
)
from davinci_monet.config.schema import CustomWindowSpec


def _ds() -> xr.Dataset:
    time = np.array(
        [
            "2008-07-01T00:00",
            "2008-07-01T03:00",
            "2008-07-02T00:00",
            "2008-08-01T00:00",
        ],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        {
            "aod": (("time", "lat", "lon"), np.arange(16, dtype=float).reshape(4, 2, 2)),
            "mask": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[1, 0], [1, 0]],
                        [[1, 1], [0, 0]],
                        [[0, 1], [1, 1]],
                        [[1, 1], [1, 1]],
                    ],
                    dtype=float,
                ),
            ),
        },
        coords={"time": time, "lat": [-1.0, 1.0], "lon": [0.0, 90.0]},
    )


def test_group_dataset_day_and_month() -> None:
    daily = group_dataset(_ds(), "day")
    monthly = group_dataset(_ds(), "month")
    assert list(daily) == ["2008-07-01", "2008-07-02", "2008-08-01"]
    assert list(monthly) == ["2008-07", "2008-08"]
    assert daily["2008-07-01"].sizes["time"] == 2
    assert monthly["2008-07"].sizes["time"] == 3


def test_group_dataset_labels_cftime_calendar_values() -> None:
    ds = xr.Dataset(
        {"aod": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={
            "time": [
                cftime.DatetimeNoLeap(2008, 7, 1, 3),
                cftime.DatetimeNoLeap(2008, 7, 1, 6),
                cftime.DatetimeNoLeap(2008, 10, 1, 0),
            ],
            "lat": [0.0],
            "lon": [0.0],
        },
    )

    assert list(group_dataset(ds, "day")) == ["2008-07-01", "2008-10-01"]
    assert list(group_dataset(ds, "month")) == ["2008-07", "2008-10"]
    assert list(group_dataset(ds, "season")) == ["2008-JJA", "2008-SON"]


def test_group_dataset_season_groups_meteorological_season_dates_together() -> None:
    """July/August dates (JJA) fall in a single meteorological-season group."""
    seasonal = group_dataset(_ds(), "season")
    assert list(seasonal) == ["2008-JJA"]
    assert seasonal["2008-JJA"].sizes["time"] == 4


def test_group_dataset_season_rolls_december_into_next_years_djf() -> None:
    """Dec 2008 groups with Jan/Feb 2009 as a single '2009-DJF' winter."""
    ds = xr.Dataset(
        {"aod": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={
            "time": np.array(
                ["2008-12-15T00:00", "2009-01-15T00:00", "2009-02-15T00:00"],
                dtype="datetime64[ns]",
            ),
            "lat": [0.0],
            "lon": [0.0],
        },
    )

    seasonal = group_dataset(ds, "season")

    assert list(seasonal) == ["2009-DJF"]
    assert seasonal["2009-DJF"].sizes["time"] == 3


def test_group_dataset_custom_windows() -> None:
    groups = group_dataset(
        _ds(),
        [
            {"name": "first_two", "start": "2008-07-01T00:00", "end": "2008-07-01T23:59"},
            {"name": "all_july", "start": "2008-07-01", "end": "2008-07-31T23:59"},
        ],
    )
    assert list(groups) == ["first_two", "all_july"]
    assert groups["first_two"].sizes["time"] == 2
    assert groups["all_july"].sizes["time"] == 3


def test_group_dataset_accepts_schema_custom_windows() -> None:
    groups = group_dataset(
        _ds(),
        [
            CustomWindowSpec(
                name="all_july",
                start="2008-07-01",
                end="2008-07-31T23:59",
            )
        ],
    )
    assert list(groups) == ["all_july"]
    assert groups["all_july"].sizes["time"] == 3


def test_group_dataset_invalid_custom_window_raises_value_error() -> None:
    with pytest.raises(ValueError, match="custom window"):
        group_dataset(_ds(), [object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="custom window"):
        group_dataset(_ds(), [{"name": "missing_bounds"}])


def test_product_summary_and_limits_ignore_nan_and_outlier() -> None:
    values = xr.DataArray(np.array([0.0, 0.1, 0.2, np.nan, 10.0]), dims=["x"])
    summary = product_summary({"field": values})
    assert summary["field"]["finite_count"] == 4
    assert summary["field"]["max"] == 10.0
    lo, hi = robust_limits(values)
    assert lo <= 0.1
    assert hi < 10.0
    slo, shi = symmetric_limits(xr.DataArray(np.array([-0.02, 0.0, 0.03, 5.0]), dims=["x"]))
    assert slo < 0.0 and shi > 0.0
    assert shi < 5.0
