"""Unit tests for the NASA POWER reader.

Fixtures are synthetic POWER-shaped NetCDF built programmatically, matching
the real response shape measured on 2026-07-15: ``(time, lat, lon)`` with
``lat=lon=1`` for point, and CF-ish variable attrs carrying ``units``. No
test here touches the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.datasets import power as power_reader
from davinci_monet.io.download import power as power_client


def _power_response(
    values: dict[str, tuple[list[float], str]],
    *,
    n_time: int = 3,
    lats: list[float] | None = None,
    lons: list[float] | None = None,
) -> xr.Dataset:
    """Build a POWER-shaped response: (time, lat, lon) with units attrs."""
    lats = lats if lats is not None else [40.02]
    lons = lons if lons is not None else [-105.27]
    time = np.array(
        [np.datetime64("2024-02-01") + np.timedelta64(i, "D") for i in range(n_time)]
    )
    data_vars = {}
    for name, (vals, units) in values.items():
        arr = np.asarray(vals, dtype="float32").reshape(n_time, len(lats), len(lons))
        data_vars[name] = (
            ("time", "lat", "lon"),
            arr,
            {"units": units, "long_name": name, "valid_min": -999.0, "valid_max": 9999.0},
        )
    return xr.Dataset(data_vars, coords={"time": time, "lat": lats, "lon": lons})


@pytest.fixture
def stub_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Serve synthetic POWER NetCDF from the client's cache seam."""

    def _install(response_for: dict[str, xr.Dataset] | xr.Dataset) -> Path:
        def _fake_fetch_to_cache(request, cache_dir, **kwargs):  # type: ignore[no-untyped-def]
            ds = (
                response_for
                if isinstance(response_for, xr.Dataset)
                else response_for[request.site or "regional"]
            )
            path = power_client.cache_path(cache_dir, request)
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(path)
            return path

        monkeypatch.setattr(power_reader, "fetch_to_cache", _fake_fetch_to_cache)
        return tmp_path

    return _install


def test_sites_mode_returns_point_geometry_with_site_dim(stub_fetch, tmp_path: Path) -> None:
    """The API has no site dim -- the reader must build one by concatenating."""
    stub_fetch(
        {
            "boulder": _power_response({"T2M": ([0.0, 1.0, 2.0], "C")}),
            "table_mtn": _power_response({"T2M": ([3.0, 4.0, 5.0], "C")}, lats=[40.125], lons=[-105.24]),
        }
    )
    reader = power_reader.POWERReader()
    ds = reader.open(
        [],
        variables=["T2M"],
        temporal="daily",
        sites=[
            {"name": "boulder", "latitude": 40.02, "longitude": -105.27},
            {"name": "table_mtn", "latitude": 40.125, "longitude": -105.24},
        ],
        cache_dir=tmp_path,
        time_range=("2024-02-01", "2024-02-03"),
    )
    assert reader.geometry is DataGeometry.POINT
    assert ds.attrs["geometry"] == "point"
    assert set(ds.sizes) == {"time", "site"}
    assert ds.sizes["site"] == 2
    assert list(ds["site"].values) == ["boulder", "table_mtn"]
    assert ds["latitude"].dims == ("site",)
    np.testing.assert_allclose(ds["latitude"].values, [40.02, 40.125])


def test_bbox_mode_returns_grid_geometry(stub_fetch, tmp_path: Path) -> None:
    stub_fetch(
        _power_response(
            {"ALLSKY_SFC_SW_DWN": ([1.0] * 12, "kW-hr/m^2/day")},
            lats=[40.0, 40.5],
            lons=[-106.0, -105.5],
        )
    )
    reader = power_reader.POWERReader()
    ds = reader.open(
        [],
        variables=["ALLSKY_SFC_SW_DWN"],
        temporal="daily",
        bbox={"lat_min": 40, "lat_max": 42, "lon_min": -106, "lon_max": -104},
        cache_dir=tmp_path,
        time_range=("2024-02-01", "2024-02-03"),
    )
    assert reader.geometry is DataGeometry.GRID
    assert ds.attrs["geometry"] == "grid"
    assert set(ds.sizes) == {"time", "lat", "lon"}


def test_daily_solar_normalizes_kwh_to_watts(stub_fetch, tmp_path: Path) -> None:
    """1 kWh/m^2/day = 1000 Wh / 24 h = 41.667 W/m^2."""
    stub_fetch({"boulder": _power_response({"ALLSKY_SFC_SW_DWN": ([1.0, 2.0, 0.0], "kW-hr/m^2/day")})})
    reader = power_reader.POWERReader()
    ds = reader.open(
        [], variables=["ALLSKY_SFC_SW_DWN"], temporal="daily",
        sites=[{"name": "boulder", "latitude": 40.02, "longitude": -105.27}],
        cache_dir=tmp_path, time_range=("2024-02-01", "2024-02-03"),
    )
    np.testing.assert_allclose(
        ds["ALLSKY_SFC_SW_DWN"].values.ravel(), [41.6667, 83.3333, 0.0], rtol=1e-4
    )
    assert ds["ALLSKY_SFC_SW_DWN"].attrs["units"] == "W m-2"


def test_hourly_solar_normalizes_watt_hours_to_watts(stub_fetch, tmp_path: Path) -> None:
    """Hourly arrives as Wh/m^2, NOT W/m^2 -- a Wh over one hour is a W."""
    stub_fetch({"boulder": _power_response({"ALLSKY_SFC_SW_DWN": ([100.0, 200.0, 0.0], "Wh/m^2")})})
    reader = power_reader.POWERReader()
    ds = reader.open(
        [], variables=["ALLSKY_SFC_SW_DWN"], temporal="hourly",
        sites=[{"name": "boulder", "latitude": 40.02, "longitude": -105.27}],
        cache_dir=tmp_path, time_range=("2024-02-01", "2024-02-01"),
    )
    np.testing.assert_allclose(ds["ALLSKY_SFC_SW_DWN"].values.ravel(), [100.0, 200.0, 0.0])
    assert ds["ALLSKY_SFC_SW_DWN"].attrs["units"] == "W m-2"


def test_t2m_celsius_normalizes_to_kelvin(stub_fetch, tmp_path: Path) -> None:
    stub_fetch({"boulder": _power_response({"T2M": ([0.0, 15.0, -40.0], "C")})})
    reader = power_reader.POWERReader()
    ds = reader.open(
        [], variables=["T2M"], temporal="daily",
        sites=[{"name": "boulder", "latitude": 40.02, "longitude": -105.27}],
        cache_dir=tmp_path, time_range=("2024-02-01", "2024-02-03"),
    )
    np.testing.assert_allclose(ds["T2M"].values.ravel(), [273.15, 288.15, 233.15], rtol=1e-5)
    assert ds["T2M"].attrs["units"] == "K"


def test_unit_drift_fails_loudly_rather_than_corrupting_stats(stub_fetch, tmp_path: Path) -> None:
    """If upstream units change, scaling by a stale factor is silent corruption."""
    stub_fetch({"boulder": _power_response({"T2M": ([0.0, 1.0, 2.0], "K")})})
    reader = power_reader.POWERReader()
    with pytest.raises(ValueError) as exc:
        reader.open(
            [], variables=["T2M"], temporal="daily",
            sites=[{"name": "boulder", "latitude": 40.02, "longitude": -105.27}],
            cache_dir=tmp_path, time_range=("2024-02-01", "2024-02-03"),
        )
    assert "T2M" in str(exc.value)
    assert "expected 'C'" in str(exc.value)


def test_fill_value_becomes_nan(stub_fetch, tmp_path: Path) -> None:
    stub_fetch({"boulder": _power_response({"T2M": ([10.0, -999.0, 12.0], "C")})})
    reader = power_reader.POWERReader()
    ds = reader.open(
        [], variables=["T2M"], temporal="daily",
        sites=[{"name": "boulder", "latitude": 40.02, "longitude": -105.27}],
        cache_dir=tmp_path, time_range=("2024-02-01", "2024-02-03"),
    )
    values = ds["T2M"].values.ravel()
    assert np.isnan(values[1])
    np.testing.assert_allclose(values[[0, 2]], [283.15, 285.15], rtol=1e-5)


def test_uncatalogued_parameter_passes_through_with_a_warning(
    stub_fetch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stub_fetch({"boulder": _power_response({"WEIRD_PARAM": ([1.0, 2.0, 3.0], "furlongs")})})
    reader = power_reader.POWERReader()
    with caplog.at_level("WARNING"):
        ds = reader.open(
            [], variables=["WEIRD_PARAM"], temporal="daily",
            sites=[{"name": "boulder", "latitude": 40.02, "longitude": -105.27}],
            cache_dir=tmp_path, time_range=("2024-02-01", "2024-02-03"),
        )
    np.testing.assert_allclose(ds["WEIRD_PARAM"].values.ravel(), [1.0, 2.0, 3.0])
    assert ds["WEIRD_PARAM"].attrs["units"] == "furlongs"
    assert "WEIRD_PARAM" in caplog.text


def test_files_mode_reads_staged_netcdf_without_network(tmp_path: Path) -> None:
    """Offline reruns and CI must work from staged files alone."""
    staged = tmp_path / "staged.nc"
    _power_response({"T2M": ([0.0, 1.0, 2.0], "C")}).to_netcdf(staged)
    reader = power_reader.POWERReader()
    ds = reader.open([str(staged)], variables=["T2M"], temporal="daily")
    np.testing.assert_allclose(ds["T2M"].values.ravel(), [273.15, 274.15, 275.15], rtol=1e-5)


def test_sites_and_bbox_together_is_rejected(tmp_path: Path) -> None:
    reader = power_reader.POWERReader()
    with pytest.raises(ValueError) as exc:
        reader.open(
            [], variables=["T2M"], temporal="daily",
            sites=[{"name": "b", "latitude": 40.0, "longitude": -105.0}],
            bbox={"lat_min": 40, "lat_max": 42, "lon_min": -106, "lon_max": -104},
            cache_dir=tmp_path, time_range=("2024-02-01", "2024-02-03"),
        )
    assert "exactly one" in str(exc.value)


def test_registered_under_the_power_source_type() -> None:
    from davinci_monet.core.registry import source_registry
    from davinci_monet.io.source_registration import ensure_builtin_source_readers_registered

    ensure_builtin_source_readers_registered()
    assert source_registry.get("power") is power_reader.POWERReader
