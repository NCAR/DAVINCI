"""Offline unit tests for the NASA POWER staging helper.

Golden URLs and behaviours here were verified against the live POWER API
(v2.9.4/v2.9.5) on 2026-07-15; see POWER.md "API facts". No test in this
module touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from davinci_monet.core.exceptions import DataNotFoundError
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


def test_hourly_regional_is_rejected_because_the_endpoint_does_not_exist() -> None:
    """Verified live: hourly/regional 404s (an HTML page, not an API error).

    Every other temporal x mode combination returns 200, so this one cell of
    the matrix has to be refused locally -- otherwise the user gets a wall of
    404 HTML with no hint that the combination is simply unsupported.
    """
    with pytest.raises(ValueError) as exc:
        power.build_power_url(
            temporal="hourly",
            mode="regional",
            params=["T2M"],
            start="2024-02-01",
            end="2024-02-02",
            bbox={"lat_min": 40, "lat_max": 42, "lon_min": -106, "lon_max": -104},
        )
    message = str(exc.value)
    assert "hourly" in message and "regional" in message
    assert "daily" in message  # point the user at the combination that works


def test_hourly_point_and_daily_regional_are_both_allowed() -> None:
    """Guard the fix against over-reach: only hourly+regional is unsupported."""
    assert power.build_power_url(
        temporal="hourly",
        mode="point",
        params=["T2M"],
        start="2024-02-01",
        end="2024-02-02",
        latitude=40.02,
        longitude=-105.27,
    )
    assert power.build_power_url(
        temporal="daily",
        mode="regional",
        params=["T2M"],
        start="2024-02-01",
        end="2024-02-02",
        bbox={"lat_min": 40, "lat_max": 42, "lon_min": -106, "lon_max": -104},
    )


def test_regional_rejects_a_bbox_narrower_than_two_degrees() -> None:
    """Verified live: <2 deg in either axis is a 422 telling you to use point."""
    with pytest.raises(ValueError) as exc:
        power.build_power_url(
            temporal="daily",
            mode="regional",
            params=["T2M"],
            start="2024-02-01",
            end="2024-02-02",
            bbox={"lat_min": 40, "lat_max": 41, "lon_min": -106, "lon_max": -104},
        )
    message = str(exc.value)
    assert "2 degree" in message
    assert "latitude" in message
    assert "point" in message  # the API's own advice: use the point endpoint


def test_regional_accepts_a_bbox_of_exactly_two_degrees() -> None:
    """Exactly 2 deg is accepted upstream, so we must not reject it."""
    url = power.build_power_url(
        temporal="daily",
        mode="regional",
        params=["T2M"],
        start="2024-02-01",
        end="2024-02-02",
        bbox={"lat_min": 40, "lat_max": 42, "lon_min": -106, "lon_max": -104},
    )
    assert "latitude-min=40" in url


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


SITES = [
    {"name": "boulder", "latitude": 40.02, "longitude": -105.27},
    {"name": "table_mtn", "latitude": 40.125, "longitude": -105.24},
]
BBOX = {"lat_min": 40, "lat_max": 42, "lon_min": -106, "lon_max": -104}


def test_plan_point_emits_one_request_per_site() -> None:
    reqs = power.plan_requests(
        temporal="daily",
        mode="point",
        params=["T2M", "ALLSKY_SFC_SW_DWN"],
        sites=SITES,
        start="2024-02-01",
        end="2024-02-03",
    )
    assert [r.site for r in reqs] == ["boulder", "table_mtn"]
    assert all(r.params == ("T2M", "ALLSKY_SFC_SW_DWN") for r in reqs)
    assert "latitude=40.02" in reqs[0].url


def test_plan_point_chunks_params_over_the_twenty_cap() -> None:
    """25 params must become 2 legal requests per site, not one 422."""
    params = [f"P{i}" for i in range(25)]
    reqs = power.plan_requests(
        temporal="daily",
        mode="point",
        params=params,
        sites=SITES[:1],
        start="2024-02-01",
        end="2024-02-03",
    )
    assert len(reqs) == 2
    assert len(reqs[0].params) == 20
    assert len(reqs[1].params) == 5
    # Every param survives the split exactly once.
    assert [p for r in reqs for p in r.params] == params


def test_plan_regional_emits_one_request_per_parameter() -> None:
    """Regional caps at 1 param, so N params must fan out into N requests."""
    reqs = power.plan_requests(
        temporal="daily",
        mode="regional",
        params=["T2M", "ALLSKY_SFC_SW_DWN"],
        bbox=BBOX,
        start="2024-02-01",
        end="2024-02-02",
    )
    assert len(reqs) == 2
    assert [r.params for r in reqs] == [("T2M",), ("ALLSKY_SFC_SW_DWN",)]
    assert all(r.site is None for r in reqs)


def test_regional_bbox_larger_than_ten_degrees_is_tiled() -> None:
    """Verified live: >10 deg on an axis is a 422. CONUS needs tiling."""
    reqs = power.plan_requests(
        temporal="daily",
        mode="regional",
        params=["ALLSKY_SFC_SW_DWN"],
        bbox={"lat_min": 30, "lat_max": 50, "lon_min": -125, "lon_max": -100},
        start="2024-02-01",
        end="2024-02-02",
    )
    # 20 deg lat -> 2 tiles; 25 deg lon -> 3 tiles.
    assert len(reqs) == 6, [r.bbox for r in reqs]
    for r in reqs:
        assert r.bbox is not None
        assert r.bbox["lat_max"] - r.bbox["lat_min"] <= 10.0
        assert r.bbox["lon_max"] - r.bbox["lon_min"] <= 10.0


def test_tiles_cover_the_whole_domain_without_gaps() -> None:
    reqs = power.plan_requests(
        temporal="daily",
        mode="regional",
        params=["T2M"],
        bbox={"lat_min": 30, "lat_max": 50, "lon_min": -125, "lon_max": -100},
        start="2024-02-01",
        end="2024-02-02",
    )
    lat_lo = min(r.bbox["lat_min"] for r in reqs)
    lat_hi = max(r.bbox["lat_max"] for r in reqs)
    lon_lo = min(r.bbox["lon_min"] for r in reqs)
    lon_hi = max(r.bbox["lon_max"] for r in reqs)
    assert (lat_lo, lat_hi) == (30, 50)
    assert (lon_lo, lon_hi) == (-125, -100)

    # Tile edges must be contiguous along each axis -- no gap between tiles.
    lat_edges = sorted({(r.bbox["lat_min"], r.bbox["lat_max"]) for r in reqs})
    for (_, prev_hi), (next_lo, _) in zip(lat_edges, lat_edges[1:]):
        assert next_lo == prev_hi, f"gap or overlap between lat tiles: {lat_edges}"


def test_tiling_never_emits_a_sliver_below_the_two_degree_minimum() -> None:
    """A 21-deg span must not tile into 10 + 10 + 1 -- that last tile is a 422."""
    reqs = power.plan_requests(
        temporal="daily",
        mode="regional",
        params=["T2M"],
        bbox={"lat_min": 30, "lat_max": 51, "lon_min": -110, "lon_max": -105},
        start="2024-02-01",
        end="2024-02-02",
    )
    for r in reqs:
        span = r.bbox["lat_max"] - r.bbox["lat_min"]
        assert span >= 2.0, f"tile {r.bbox} is below the API's 2 deg minimum"
        assert span <= 10.0


def test_a_bbox_within_limits_is_not_tiled() -> None:
    reqs = power.plan_requests(
        temporal="daily",
        mode="regional",
        params=["T2M"],
        bbox={"lat_min": 40, "lat_max": 45, "lon_min": -110, "lon_max": -105},
        start="2024-02-01",
        end="2024-02-02",
    )
    assert len(reqs) == 1


def test_cache_path_is_deterministic_for_the_same_request() -> None:
    kw = dict(
        temporal="daily",
        mode="point",
        params=["T2M"],
        sites=SITES[:1],
        start="2024-02-01",
        end="2024-02-03",
    )
    first = power.plan_requests(**kw)[0]
    second = power.plan_requests(**kw)[0]
    assert power.cache_path("/tmp/cache", first) == power.cache_path("/tmp/cache", second)


def test_cache_path_differs_when_the_request_differs() -> None:
    base = power.plan_requests(
        temporal="daily",
        mode="point",
        params=["T2M"],
        sites=SITES[:1],
        start="2024-02-01",
        end="2024-02-03",
    )[0]
    other_window = power.plan_requests(
        temporal="daily",
        mode="point",
        params=["T2M"],
        sites=SITES[:1],
        start="2024-02-01",
        end="2024-02-04",
    )[0]
    other_param = power.plan_requests(
        temporal="daily",
        mode="point",
        params=["PS"],
        sites=SITES[:1],
        start="2024-02-01",
        end="2024-02-03",
    )[0]
    paths = {
        power.cache_path("/tmp/cache", base),
        power.cache_path("/tmp/cache", other_window),
        power.cache_path("/tmp/cache", other_param),
    }
    assert len(paths) == 3


def test_cache_path_layout_is_readable() -> None:
    req = power.plan_requests(
        temporal="daily",
        mode="point",
        params=["T2M"],
        sites=SITES[:1],
        start="2024-02-01",
        end="2024-02-03",
        community="AG",
    )[0]
    path = power.cache_path("/tmp/cache", req)
    assert path.suffix == ".nc"
    assert path.parent == Path("/tmp/cache/daily/AG/point")


def _one_request() -> power.PowerRequest:
    return power.plan_requests(
        temporal="daily",
        mode="point",
        params=["T2M"],
        sites=SITES[:1],
        start="2024-02-01",
        end="2024-02-03",
    )[0]


def test_fetch_to_cache_writes_response_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(power, "_fetch", lambda url, timeout=60.0: b"NETCDF-BYTES")
    req = _one_request()
    path = power.fetch_to_cache(req, tmp_path)
    assert path.read_bytes() == b"NETCDF-BYTES"
    assert path == power.cache_path(tmp_path, req)


def test_fetch_to_cache_hit_does_not_touch_the_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    req = _one_request()
    cached = power.cache_path(tmp_path, req)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"CACHED")

    def _boom(url: str, timeout: float = 60.0) -> bytes:
        raise AssertionError("cache hit must not fetch")

    monkeypatch.setattr(power, "_fetch", _boom)
    assert power.fetch_to_cache(req, tmp_path).read_bytes() == b"CACHED"


def test_fetch_to_cache_force_refetches_over_a_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    req = _one_request()
    cached = power.cache_path(tmp_path, req)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"STALE")
    monkeypatch.setattr(power, "_fetch", lambda url, timeout=60.0: b"FRESH")
    assert power.fetch_to_cache(req, tmp_path, force=True).read_bytes() == b"FRESH"


def test_offline_cache_miss_names_the_command_that_would_fix_it(
    tmp_path: Path,
) -> None:
    """An offline miss must be actionable, not just 'not found'."""
    with pytest.raises(DataNotFoundError) as exc:
        power.fetch_to_cache(_one_request(), tmp_path, offline=True)
    assert "davinci-stage-power" in str(exc.value)


def test_http_422_raises_immediately_with_the_offending_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """422 is a validation error -- retrying it just burns the rate limit."""
    calls = {"n": 0}

    def _fake(url: str, timeout: float = 60.0) -> bytes:
        calls["n"] += 1
        raise power.PowerHTTPError(422, "Please provide a correct start date formatting.", url)

    monkeypatch.setattr(power, "_fetch", _fake)
    with pytest.raises(power.PowerHTTPError) as exc:
        power.fetch_to_cache(_one_request(), tmp_path)
    assert calls["n"] == 1, "422 must not be retried"
    assert "start date formatting" in str(exc.value)
    assert "power.larc.nasa.gov" in str(exc.value)


def test_http_429_is_retried_then_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def _fake(url: str, timeout: float = 60.0) -> bytes:
        calls["n"] += 1
        raise power.PowerHTTPError(429, "rate limited", url)

    monkeypatch.setattr(power, "_fetch", _fake)
    monkeypatch.setattr(power, "_sleep", lambda seconds: None)
    with pytest.raises(power.PowerHTTPError):
        power.fetch_to_cache(_one_request(), tmp_path, max_tries=3)
    assert calls["n"] == 3


def test_http_429_that_recovers_returns_the_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"n": 0}

    def _fake(url: str, timeout: float = 60.0) -> bytes:
        calls["n"] += 1
        if calls["n"] < 3:
            raise power.PowerHTTPError(503, "unavailable", url)
        return b"OK"

    monkeypatch.setattr(power, "_fetch", _fake)
    monkeypatch.setattr(power, "_sleep", lambda seconds: None)
    assert power.fetch_to_cache(_one_request(), tmp_path, max_tries=3).read_bytes() == b"OK"


def test_partial_write_is_not_left_in_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed fetch must not leave a truncated file that later reads as a hit."""
    req = _one_request()

    def _fake(url: str, timeout: float = 60.0) -> bytes:
        raise power.PowerHTTPError(500, "boom", url)

    monkeypatch.setattr(power, "_fetch", _fake)
    monkeypatch.setattr(power, "_sleep", lambda seconds: None)
    with pytest.raises(power.PowerHTTPError):
        power.fetch_to_cache(req, tmp_path, max_tries=1)
    assert not power.cache_path(tmp_path, req).exists()


def test_stage_power_dry_run_plans_but_does_not_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(url: str, timeout: float = 60.0) -> bytes:
        raise AssertionError("dry run must not fetch")

    monkeypatch.setattr(power, "_fetch", _boom)
    planned = power.stage_power(
        temporal="daily",
        params=["T2M", "PS"],
        sites=SITES,
        start="2024-02-01",
        end="2024-02-03",
        cache_dir=tmp_path,
        dry_run=True,
    )
    assert len(planned) == 2  # one request per site
    assert list(tmp_path.rglob("*.nc")) == []


def test_stage_power_fetches_every_planned_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(power, "_fetch", lambda url, timeout=60.0: b"NC")
    paths = power.stage_power(
        temporal="daily",
        params=["T2M"],
        sites=SITES,
        start="2024-02-01",
        end="2024-02-03",
        cache_dir=tmp_path,
    )
    assert len(paths) == 2
    assert all(Path(p).read_bytes() == b"NC" for p in paths)


def test_cli_site_argument_parses_lat_lon_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(power, "_fetch", lambda url, timeout=60.0: b"NC")
    rc = power.main(
        [
            "--temporal",
            "daily",
            "--params",
            "T2M",
            "--site",
            "40.02,-105.27,boulder",
            "--start",
            "2024-02-01",
            "--end",
            "2024-02-03",
            "--cache-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    cached = list(tmp_path.rglob("*.nc"))
    assert len(cached) == 1
    assert "boulder" in cached[0].name


def test_cli_dry_run_reports_plan_without_fetching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(url: str, timeout: float = 60.0) -> bytes:
        raise AssertionError("dry run must not fetch")

    monkeypatch.setattr(power, "_fetch", _boom)
    rc = power.main(
        [
            "--temporal",
            "monthly",
            "--params",
            "T2M,ALLSKY_SFC_SW_DWN",
            "--site",
            "40.02,-105.27,boulder",
            "--start",
            "2020-01-01",
            "--end",
            "2021-12-31",
            "--cache-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 request" in out
    assert "start=2020" in out  # monthly year-only formatting is visible in the plan


def test_cli_bbox_argument_plans_regional_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(power, "_fetch", lambda url, timeout=60.0: b"NC")
    rc = power.main(
        [
            "--temporal",
            "daily",
            "--params",
            "T2M,ALLSKY_SFC_SW_DWN",
            "--bbox",
            "40,42,-106,-104",
            "--start",
            "2024-02-01",
            "--end",
            "2024-02-02",
            "--cache-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    # Regional caps at 1 param, so 2 params must have become 2 cached files.
    assert len(list(tmp_path.rglob("*.nc"))) == 2


def test_cli_rejects_both_site_and_bbox(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        power.main(
            [
                "--temporal",
                "daily",
                "--params",
                "T2M",
                "--site",
                "40.02,-105.27",
                "--bbox",
                "40,42,-106,-104",
                "--start",
                "2024-02-01",
                "--end",
                "2024-02-02",
                "--cache-dir",
                str(tmp_path),
            ]
        )
