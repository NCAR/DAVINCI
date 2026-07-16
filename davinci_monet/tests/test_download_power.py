"""Offline unit tests for the NASA POWER staging helper.

Golden URLs and behaviours here were verified against the live POWER API
(v2.9.4/v2.9.5) on 2026-07-15; see POWER.md "API facts". No test in this
module touches the network.
"""

from __future__ import annotations

import pytest

from davinci_monet.io.download import power


def test_daily_point_url_matches_golden() -> None:
    """A daily point request formats dates YYYYMMDD and asks for UTC NetCDF."""
    url = power.build_power_url(
        temporal="daily",
        mode="point",
        params=["T2M", "ALLSKY_SFC_SW_DWN"],
        start="2024-02-01",
        end="2024-02-03",
        latitude=40.02,
        longitude=-105.27,
        community="RE",
    )
    assert url == (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=T2M%2CALLSKY_SFC_SW_DWN"
        "&community=RE"
        "&latitude=40.02"
        "&longitude=-105.27"
        "&start=20240201"
        "&end=20240203"
        "&format=NETCDF"
        "&time-standard=UTC"
    )


def test_monthly_url_uses_year_only_dates() -> None:
    """Monthly rejects YYYYMMDD with a 422 upstream, so we must send YYYY."""
    url = power.build_power_url(
        temporal="monthly",
        mode="point",
        params=["T2M"],
        start="2020-01-01",
        end="2021-12-31",
        latitude=40.02,
        longitude=-105.27,
    )
    assert "&start=2020&end=2021&" in url


def test_regional_url_uses_bbox_bounds() -> None:
    url = power.build_power_url(
        temporal="daily",
        mode="regional",
        params=["ALLSKY_SFC_SW_DWN"],
        start="2024-02-01",
        end="2024-02-02",
        bbox={"lat_min": 40, "lat_max": 42, "lon_min": -106, "lon_max": -104},
    )
    assert "latitude-min=40&latitude-max=42" in url
    assert "longitude-min=-106&longitude-max=-104" in url


def test_regional_rejects_more_than_one_parameter() -> None:
    """The API hard-caps regional at 1 parameter (2 -> HTTP 422)."""
    with pytest.raises(ValueError) as exc:
        power.build_power_url(
            temporal="daily",
            mode="regional",
            params=["T2M", "ALLSKY_SFC_SW_DWN"],
            start="2024-02-01",
            end="2024-02-02",
            bbox={"lat_min": 40, "lat_max": 42, "lon_min": -106, "lon_max": -104},
        )
    assert "one parameter" in str(exc.value)


def test_point_rejects_more_than_twenty_parameters() -> None:
    with pytest.raises(ValueError) as exc:
        power.build_power_url(
            temporal="daily",
            mode="point",
            params=[f"P{i}" for i in range(21)],
            start="2024-02-01",
            end="2024-02-02",
            latitude=40.0,
            longitude=-105.0,
        )
    assert "20 parameters" in str(exc.value)


def test_time_standard_defaults_to_utc_not_the_api_default_lst() -> None:
    """POWER defaults to LST; pairing LST against a UTC model phase-shifts it."""
    url = power.build_power_url(
        temporal="hourly",
        mode="point",
        params=["T2M"],
        start="2024-02-01",
        end="2024-02-01",
        latitude=40.02,
        longitude=-105.27,
    )
    assert "time-standard=UTC" in url
    assert "LST" not in url
