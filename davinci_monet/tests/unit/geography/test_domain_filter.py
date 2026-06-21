"""Tests for domain filtering of paired datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.util.domain import filter_paired_by_domain


def test_domain_extent_catalog_is_not_plot_owned() -> None:
    """Named domains live in a neutral geography module, not spatial plotting."""
    from davinci_monet.geography.domains import get_domain_extent

    assert get_domain_extent("conus") == (-130, -60, 20, 55)


def test_asia_aq_domain_registered() -> None:
    """ASIA-AQ campaign domain is a named domain: lon 90-140E, lat 0-45N."""
    from davinci_monet.geography.domains import get_domain_extent

    assert get_domain_extent("asia_aq") == (90.0, 140.0, 0.0, 45.0)


def _paired_point_dataset() -> xr.Dataset:
    """Build a synthetic paired Dataset with sites spanning multiple regions."""
    times = pd.date_range("2024-01-01", periods=6, freq="h")
    # Sites: 3 in CONUS, 1 in EPA R5 (Great Lakes), 2 in Asia.
    site_lats = np.array([35.0, 40.0, 45.0, 43.0, 28.6, 13.1])
    site_lons = np.array([-100.0, -105.0, -95.0, -85.0, 77.2, 80.3])
    geometry = np.full((6, 6), 10.0)
    dataset = np.full((6, 6), 11.0)
    geometry[:, 4:] = 9999.0  # sentinel to detect leakage
    return xr.Dataset(
        {
            "x_pm25": (["time", "site"], geometry),
            "y_pm25": (["time", "site"], dataset),
        },
        coords={
            "time": times,
            "site": np.arange(6),
            "latitude": ("site", site_lats),
            "longitude": ("site", site_lons),
        },
    )


class TestFilterPairedByDomain:
    def test_all_returns_unchanged(self) -> None:
        ds = _paired_point_dataset()
        out = filter_paired_by_domain(ds, "all")
        assert out.sizes == ds.sizes
        # Same underlying data: identity check is fine since no isel happens
        xr.testing.assert_identical(out, ds)

    def test_none_returns_unchanged(self) -> None:
        ds = _paired_point_dataset()
        out = filter_paired_by_domain(ds, None)
        assert out.sizes == ds.sizes

    def test_conus_drops_asian_sites(self) -> None:
        ds = _paired_point_dataset()
        out = filter_paired_by_domain(ds, "conus")
        # 4 CONUS sites kept, 2 Asian sites dropped
        assert out.sizes["site"] == 4
        # Sentinel geometry values from dropped sites must not survive
        assert (out["x_pm25"].values < 9000.0).all()
        np.testing.assert_array_equal(
            np.sort(out["latitude"].values), np.array([35.0, 40.0, 43.0, 45.0])
        )

    def test_epa_region_r5_keeps_only_great_lakes(self) -> None:
        ds = _paired_point_dataset()
        out = filter_paired_by_domain(ds, "epa_region", "R5")
        # R5 extent: lon -97.5..-80.5, lat 36..49.5. Sites in R5 from our set:
        # (40, -105) → no, lon < -97.5. (45, -95) → yes. (43, -85) → yes.
        # (35, -100) → no, lat < 36.
        # Expect 2 sites kept.
        assert out.sizes["site"] == 2
        np.testing.assert_array_equal(np.sort(out["latitude"].values), np.array([43.0, 45.0]))

    def test_list_form_of_domain_type(self) -> None:
        """YAML schema declares domain_type as list[str]; helper must accept that."""
        ds = _paired_point_dataset()
        out = filter_paired_by_domain(ds, ["conus"])
        assert out.sizes["site"] == 4

    def test_list_form_of_domain_name(self) -> None:
        ds = _paired_point_dataset()
        out = filter_paired_by_domain(ds, ["epa_region"], ["R5"])
        assert out.sizes["site"] == 2

    def test_unknown_domain_returns_unchanged(self) -> None:
        ds = _paired_point_dataset()
        out = filter_paired_by_domain(ds, "mars")
        assert out.sizes == ds.sizes

    def test_dataset_without_latlon_returns_unchanged(self) -> None:
        # Synthetic gridded paired data with lat/lon as dims, not 1-D coords on
        # a single spatial dim — helper should bail and return unchanged.
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        lats = np.linspace(20, 60, 5)
        lons = np.linspace(-130, -60, 8)
        ds = xr.Dataset(
            {
                "x_pm25": (["time", "lat", "lon"], np.zeros((3, 5, 8))),
                "y_pm25": (["time", "lat", "lon"], np.zeros((3, 5, 8))),
            },
            coords={"time": times, "lat": lats, "lon": lons},
        )
        out = filter_paired_by_domain(ds, "conus")
        # lats/lons here are 1-D but with *different* dim names, so the helper
        # currently bails (different dims for lat vs lon). Behavior documented.
        assert out.sizes == ds.sizes
