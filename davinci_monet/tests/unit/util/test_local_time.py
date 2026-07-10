"""Tests for deterministic local-solar-time sampling."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.util.local_time import (
    apply_local_solar_time_index,
    build_local_solar_time_index,
    sample_local_solar_time,
)


def _hourly_field(start: str, periods: int, longitudes: list[float]) -> xr.DataArray:
    time = np.arange(
        np.datetime64(start, "h"),
        np.datetime64(start, "h") + np.timedelta64(periods, "h"),
        np.timedelta64(1, "h"),
    )
    values = np.broadcast_to(np.arange(periods)[:, None], (periods, len(longitudes)))
    return xr.DataArray(
        values,
        dims=("time", "lon"),
        coords={"time": time, "lon": longitudes},
        name="aod",
    )


def test_sampling_uses_adjacent_utc_days_at_dateline() -> None:
    field = _hourly_field("2001-01-01T00", 72, [179.0, -179.0])

    sampled, index = sample_local_solar_time(
        field,
        start_time="2001-01-02",
        end_time="2001-01-02 23:59:59",
        local_hour=13.5,
        tolerance="31min",
    )

    np.testing.assert_array_equal(
        index.sample_time.values,
        np.array([["2001-01-02T02", "2001-01-03T01"]], dtype="datetime64[h]"),
    )
    np.testing.assert_array_equal(sampled.values, [[26, 49]])
    np.testing.assert_array_equal(sampled["time"], [np.datetime64("2001-01-02T12")])
    assert bool(index.valid.all())


def test_nearest_time_tie_selects_earlier_sample() -> None:
    field = _hourly_field("2001-01-01T12", 4, [0.0])

    sampled, index = sample_local_solar_time(
        field,
        start_time="2001-01-01",
        end_time="2001-01-01",
        local_hour=13.5,
    )

    assert index.sample_time.values[0, 0] == np.datetime64("2001-01-01T13")
    assert sampled.item() == 1


def test_missing_adjacent_day_is_invalid_and_masked() -> None:
    field = _hourly_field("2001-01-01T00", 24, [-179.0])

    sampled, index = sample_local_solar_time(
        field,
        start_time="2001-01-01",
        end_time="2001-01-01",
        local_hour=13.5,
    )

    assert not index.valid.item()
    assert np.isnan(sampled.item())


def test_tolerance_rejects_nearest_sample_across_gap() -> None:
    time = xr.DataArray(
        np.array(["2001-01-01T00", "2001-01-01T12"], dtype="datetime64[h]"),
        dims="time",
    )
    longitude = xr.DataArray([0.0], dims="lon", coords={"lon": [0.0]})
    index = build_local_solar_time_index(
        time,
        longitude,
        start_time="2001-01-01",
        end_time="2001-01-01",
        local_hour=6.0,
        tolerance="2h",
    )

    assert index.sample_time.values[0, 0] == np.datetime64("2001-01-01T00")
    assert not index.valid.item()


def test_index_reuses_identical_samples_for_qa_field() -> None:
    field = _hourly_field("2001-01-01T00", 48, [0.0, 90.0])
    index = build_local_solar_time_index(
        field["time"],
        field["lon"],
        start_time="2001-01-01",
        end_time="2001-01-01",
        local_hour=13.5,
    )
    qa = xr.ones_like(field, dtype=np.int8) * field

    sampled_field = apply_local_solar_time_index(field, index)
    sampled_qa = apply_local_solar_time_index(qa, index)

    np.testing.assert_array_equal(sampled_qa, sampled_field)
    np.testing.assert_array_equal(sampled_qa["sample_time"], sampled_field["sample_time"])


def test_invalid_timestamp_and_tolerance_are_rejected() -> None:
    field = _hourly_field("2001-01-01T00", 24, [0.0])
    with pytest.raises(ValueError, match="start_time"):
        sample_local_solar_time(
            field,
            start_time="NaT",
            end_time="2001-01-01",
            local_hour=12.0,
        )
    with pytest.raises(ValueError, match="tolerance"):
        sample_local_solar_time(
            field,
            start_time="2001-01-01",
            end_time="2001-01-01",
            local_hour=12.0,
            tolerance="NaT",
        )
