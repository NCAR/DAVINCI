"""Fetch NASA POWER data to local disk via the POWER REST API.

POWER (https://power.larc.nasa.gov/) serves analysis-ready solar and
meteorological parameters with no login and no granule staging. Network
access is isolated in ``_fetch`` so the rest of the module (and its tests)
run offline; only the stdlib is used, so there is no new dependency.

Behaviours encoded here were verified against the live API (v2.9.4/v2.9.5)
on 2026-07-15; see POWER.md "API facts". The load-bearing ones:

* **Time standard.** POWER defaults to Local Solar Time. Pairing LST data
  against a UTC model silently phase-shifts the diurnal cycle (~7 h at
  Boulder), so every request here asks for ``time-standard=UTC``.
* **Monthly dates are year-only.** ``start=20200101`` on the monthly
  endpoint is an HTTP 422; ``start=2020`` is correct.
* **Regional serves one parameter per request**; point serves up to 20.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

BASE_URL = "https://power.larc.nasa.gov/api/temporal"

#: Maximum parameters the API accepts per request, by mode (verified: 21
#: params -> 422 on point; 2 params -> 422 on regional).
POINT_MAX_PARAMS = 20
REGIONAL_MAX_PARAMS = 1

TEMPORAL_LEVELS = ("hourly", "daily", "monthly")
MODES = ("point", "regional")


def _coerce_date(value: str | date | datetime) -> date:
    """Coerce an ISO date string or date/datetime into a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def format_power_date(value: str | date | datetime, temporal: str) -> str:
    """Format a date the way ``temporal``'s endpoint requires.

    Monthly takes a bare year; hourly and daily take ``YYYYMMDD``. Sending
    ``YYYYMMDD`` to the monthly endpoint is a 422, which is why this is not
    one shared format.
    """
    parsed = _coerce_date(value)
    if temporal == "monthly":
        return f"{parsed.year:04d}"
    return parsed.strftime("%Y%m%d")


def build_power_url(
    temporal: str,
    mode: str,
    params: Sequence[str],
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    latitude: float | None = None,
    longitude: float | None = None,
    bbox: Mapping[str, float] | None = None,
    community: str = "RE",
    fmt: str = "NETCDF",
    time_standard: str = "UTC",
) -> str:
    """Build a POWER API URL.

    Parameters
    ----------
    temporal
        One of ``hourly``, ``daily``, ``monthly``.
    mode
        ``point`` (needs ``latitude``/``longitude``) or ``regional``
        (needs ``bbox`` with ``lat_min``/``lat_max``/``lon_min``/``lon_max``).
    params
        POWER parameter names, e.g. ``["T2M"]``.
    time_standard
        Defaults to ``UTC``. The API's own default is ``LST``; do not pass
        ``LST`` unless the consumer genuinely wants local solar time.

    Raises
    ------
    ValueError
        If the request would exceed the API's per-mode parameter cap, or if
        the coordinates for the mode are missing.
    """
    if temporal not in TEMPORAL_LEVELS:
        raise ValueError(f"Unknown temporal level {temporal!r}. Known: {', '.join(TEMPORAL_LEVELS)}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Known: {', '.join(MODES)}")
    if not params:
        raise ValueError("At least one parameter is required.")

    query: list[tuple[str, Any]] = [
        ("parameters", ",".join(params)),
        ("community", community),
    ]

    if mode == "point":
        if len(params) > POINT_MAX_PARAMS:
            raise ValueError(
                f"POWER point requests accept at most {POINT_MAX_PARAMS} parameters; "
                f"got {len(params)}. Split the request."
            )
        if latitude is None or longitude is None:
            raise ValueError("point mode requires latitude and longitude.")
        query += [("latitude", latitude), ("longitude", longitude)]
    else:
        if len(params) > REGIONAL_MAX_PARAMS:
            raise ValueError(
                f"POWER regional requests accept exactly one parameter; got {len(params)} "
                f"({', '.join(params)}). Issue one request per parameter."
            )
        if bbox is None:
            raise ValueError("regional mode requires a bbox.")
        missing = {"lat_min", "lat_max", "lon_min", "lon_max"} - set(bbox)
        if missing:
            raise ValueError(f"bbox is missing {', '.join(sorted(missing))}.")
        query += [
            ("latitude-min", bbox["lat_min"]),
            ("latitude-max", bbox["lat_max"]),
            ("longitude-min", bbox["lon_min"]),
            ("longitude-max", bbox["lon_max"]),
        ]

    query += [
        ("start", format_power_date(start, temporal)),
        ("end", format_power_date(end, temporal)),
        ("format", fmt),
        ("time-standard", time_standard),
    ]

    return f"{BASE_URL}/{temporal}/{mode}?{urlencode(query)}"
