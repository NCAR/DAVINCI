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

import hashlib
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

from davinci_monet.core.exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class PowerRequest:
    """One API-legal POWER request, plus the identity needed to reassemble it.

    ``site`` is None for regional requests. For point requests it carries the
    configured site name, because the API response has no site dimension --
    it returns ``(time, lat=1, lon=1)`` -- so the caller must label the
    response itself before concatenating.
    """

    url: str
    temporal: str
    mode: str
    params: tuple[str, ...]
    community: str
    start: str
    end: str
    site: str | None = None
    latitude: float | None = None
    longitude: float | None = None


def _chunk(items: Sequence[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]


def plan_requests(
    temporal: str,
    mode: str,
    params: Sequence[str],
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    sites: Sequence[Mapping[str, Any]] | None = None,
    bbox: Mapping[str, float] | None = None,
    community: str = "RE",
    fmt: str = "NETCDF",
    time_standard: str = "UTC",
) -> list[PowerRequest]:
    """Split an ask into API-legal requests.

    Point requests fan out over sites (the API serves one coordinate each) and
    chunk parameters at 20. Regional requests fan out over parameters, because
    the API serves exactly one parameter per regional request.
    """
    start_str = format_power_date(start, temporal)
    end_str = format_power_date(end, temporal)
    common = dict(
        temporal=temporal,
        mode=mode,
        community=community,
        start=start_str,
        end=end_str,
    )

    requests: list[PowerRequest] = []
    if mode == "point":
        if not sites:
            raise ValueError("point mode requires at least one site.")
        for site in sites:
            for chunk in _chunk(list(params), POINT_MAX_PARAMS):
                url = build_power_url(
                    temporal,
                    mode,
                    chunk,
                    start=start,
                    end=end,
                    latitude=site["latitude"],
                    longitude=site["longitude"],
                    community=community,
                    fmt=fmt,
                    time_standard=time_standard,
                )
                requests.append(
                    PowerRequest(
                        url=url,
                        params=chunk,
                        site=site.get("name"),
                        latitude=site["latitude"],
                        longitude=site["longitude"],
                        **common,
                    )
                )
    else:
        if bbox is None:
            raise ValueError("regional mode requires a bbox.")
        for param in params:
            url = build_power_url(
                temporal,
                mode,
                [param],
                start=start,
                end=end,
                bbox=bbox,
                community=community,
                fmt=fmt,
                time_standard=time_standard,
            )
            requests.append(PowerRequest(url=url, params=(param,), **common))
    return requests


def _slug(text: str) -> str:
    """Reduce text to a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def cache_path(cache_dir: str | Path, request: PowerRequest) -> Path:
    """Return the deterministic cache location for ``request``.

    Layout is ``<cache_dir>/<temporal>/<community>/<mode>/<slug>-<hash>.nc``.
    The slug keeps the directory browsable; the hash is taken over the full
    URL, so any difference that changes the response -- parameters, window,
    coordinates, format, time standard -- changes the path. Hashing the URL
    rather than a hand-picked field list means a new query field can never
    silently alias onto an existing cache entry.
    """
    digest = hashlib.sha256(request.url.encode()).hexdigest()[:12]
    if request.site is not None:
        label = _slug(request.site)
    elif request.mode == "regional":
        label = _slug("-".join(request.params))
    else:
        label = _slug(f"{request.latitude}-{request.longitude}")
    name = f"{label}-{request.start}-{request.end}-{digest}.nc"
    return Path(cache_dir) / request.temporal / request.community / request.mode / name


class PowerHTTPError(RuntimeError):
    """An error response from the POWER API, carrying the offending URL.

    The URL is part of the message on purpose: a POWER 422 names the field it
    rejected, and without the request beside it the message is unactionable.
    """

    def __init__(self, status: int, body: str, url: str) -> None:
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"POWER API returned HTTP {status} for {url}\n{body}")


#: Statuses worth retrying: rate limiting and transient server faults. A 422
#: is a validation failure -- the same request will fail identically forever,
#: so retrying it only burns the rate limit.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
DEFAULT_MAX_TRIES = 3
DEFAULT_TIMEOUT = 60.0


def _sleep(seconds: float) -> None:
    """Sleep between retries (its own function so tests can stub it)."""
    time.sleep(seconds)


def _fetch(url: str, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """Fetch ``url``, raising ``PowerHTTPError`` on an error status.

    This is the module's only network call, kept in one small function so the
    rest of the module -- and every test -- runs offline by stubbing it.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data: bytes = response.read()
            return data
    except urllib.error.HTTPError as exc:  # noqa: PERF203 - translate, don't swallow
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - body is best-effort context
            pass
        raise PowerHTTPError(exc.code, body, url) from exc
    except urllib.error.URLError as exc:
        raise PowerHTTPError(0, f"Network error: {exc.reason}", url) from exc


def fetch_to_cache(
    request: PowerRequest,
    cache_dir: str | Path,
    *,
    force: bool = False,
    offline: bool = False,
    max_tries: int = DEFAULT_MAX_TRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path:
    """Return the cached NetCDF for ``request``, fetching it if needed.

    Parameters
    ----------
    force
        Refetch even on a cache hit.
    offline
        Never fetch. A miss raises ``DataNotFoundError`` naming the command
        that would populate the cache.
    max_tries
        Attempts for retryable statuses (429/5xx), with exponential backoff.
    """
    path = cache_path(cache_dir, request)
    if path.exists() and not force:
        logger.debug("POWER cache hit: %s", path)
        return path

    if offline:
        raise DataNotFoundError(
            f"POWER cache miss for {request.temporal}/{request.mode} "
            f"[{request.start}..{request.end}] and offline=True.\n"
            f"Populate it with:\n  {suggested_stage_command(request, cache_dir)}"
        )

    body = _fetch_with_retries(request.url, max_tries=max_tries, timeout=timeout)

    # Write via a temp file in the same directory, then atomically replace, so
    # an interrupted write can never leave a truncated file that a later run
    # would mistake for a valid cache hit.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(body)
    tmp.replace(path)
    logger.debug("POWER cached %d bytes to %s", len(body), path)
    return path


def _fetch_with_retries(url: str, *, max_tries: int, timeout: float) -> bytes:
    last: PowerHTTPError | None = None
    for attempt in range(1, max_tries + 1):
        try:
            return _fetch(url, timeout=timeout)
        except PowerHTTPError as exc:
            if exc.status not in RETRY_STATUSES:
                raise
            last = exc
            if attempt < max_tries:
                backoff = 2.0 ** (attempt - 1)
                logger.warning(
                    "POWER HTTP %s (attempt %d/%d); retrying in %.0fs",
                    exc.status,
                    attempt,
                    max_tries,
                    backoff,
                )
                _sleep(backoff)
    assert last is not None  # only reachable after a retryable failure
    raise last


def suggested_stage_command(request: PowerRequest, cache_dir: str | Path) -> str:
    """Render the ``davinci-stage-power`` command that would cache ``request``."""
    parts = [
        "davinci-stage-power",
        f"--temporal {request.temporal}",
        f"--params {','.join(request.params)}",
        f"--start {request.start}",
        f"--end {request.end}",
        f"--community {request.community}",
        f"--cache-dir {cache_dir}",
    ]
    if request.mode == "point":
        site = f"{request.latitude},{request.longitude}"
        if request.site:
            site += f",{request.site}"
        parts.append(f"--site {site}")
    return " ".join(parts)


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "davinci" / "power"


def stage_power(
    temporal: str,
    params: Sequence[str],
    *,
    start: str | date | datetime,
    end: str | date | datetime,
    sites: Sequence[Mapping[str, Any]] | None = None,
    bbox: Mapping[str, float] | None = None,
    community: str = "RE",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    dry_run: bool = False,
    time_standard: str = "UTC",
) -> list[PowerRequest] | list[Path]:
    """Stage POWER data into the cache.

    Returns the planned requests when ``dry_run`` is set, otherwise the cached
    file paths. This is the same fetch path the reader uses, so anything staged
    here is a cache hit at pipeline runtime.
    """
    mode = "regional" if bbox is not None else "point"
    requests = plan_requests(
        temporal,
        mode,
        params,
        start=start,
        end=end,
        sites=sites,
        bbox=bbox,
        community=community,
        time_standard=time_standard,
    )
    if dry_run:
        return requests
    return [fetch_to_cache(req, cache_dir, force=force) for req in requests]


def _parse_site(value: str) -> dict[str, Any]:
    """Parse a ``LAT,LON[,NAME]`` CLI site argument."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) not in (2, 3):
        raise ValueError(f"--site expects LAT,LON[,NAME]; got {value!r}")
    site: dict[str, Any] = {"latitude": float(parts[0]), "longitude": float(parts[1])}
    site["name"] = parts[2] if len(parts) == 3 else f"{parts[0]}_{parts[1]}"
    return site


def _parse_bbox(value: str) -> dict[str, float]:
    """Parse a ``LATMIN,LATMAX,LONMIN,LONMAX`` CLI bbox argument."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox expects LATMIN,LATMAX,LONMIN,LONMAX; got {value!r}")
    lat_min, lat_max, lon_min, lon_max = (float(p) for p in parts)
    return {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: stage POWER data to the local cache. Returns a process exit code."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="davinci-stage-power",
        description="Stage NASA POWER data to a local cache via the POWER REST API.",
    )
    parser.add_argument("--temporal", required=True, choices=sorted(TEMPORAL_LEVELS))
    parser.add_argument("--params", required=True, help="Comma-separated POWER parameter names")
    parser.add_argument("--site", action="append", help="LAT,LON[,NAME]; repeatable")
    parser.add_argument("--bbox", help="LATMIN,LATMAX,LONMIN,LONMAX (regional mode)")
    parser.add_argument("--start", required=True, help="ISO start, e.g. 2024-02-01")
    parser.add_argument("--end", required=True, help="ISO end, e.g. 2024-02-28")
    parser.add_argument("--community", default="RE", help="POWER community (default: RE)")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force", action="store_true", help="Refetch over cache hits")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not fetch")
    ns = parser.parse_args(argv)

    if bool(ns.site) == bool(ns.bbox):
        parser.error("Pass exactly one of --site (point) or --bbox (regional).")

    params = [p.strip() for p in ns.params.split(",") if p.strip()]
    sites = [_parse_site(s) for s in ns.site] if ns.site else None
    bbox = _parse_bbox(ns.bbox) if ns.bbox else None

    result = stage_power(
        ns.temporal,
        params,
        start=ns.start,
        end=ns.end,
        sites=sites,
        bbox=bbox,
        community=ns.community,
        cache_dir=ns.cache_dir,
        force=ns.force,
        dry_run=ns.dry_run,
    )

    if ns.dry_run:
        requests = [r for r in result if isinstance(r, PowerRequest)]
        print(f"{len(requests)} request(s) planned:")
        for req in requests:
            print(f"  {req.url}")
    else:
        print(f"Staged {len(result)} file(s) to {ns.cache_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
