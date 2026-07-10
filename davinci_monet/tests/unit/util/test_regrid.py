"""Tests for spherical rectilinear regridding helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.util.regrid import (
    area_weighted_coarsen,
    area_weighted_coarsen_variance,
    periodic_bilinear,
)


def _field(values: np.ndarray, lat: list[float], lon: list[float]) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=np.float64),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="aod",
        attrs={"units": "1"},
    )


def test_area_weighted_coarsen_uses_exact_spherical_latitude_area() -> None:
    source = _field(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0, 10.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        lat=[-60.0, 0.0, 60.0],
        lon=[-135.0, -45.0, 45.0, 135.0],
    )

    result = area_weighted_coarsen(source, resolution=90.0)

    # The southern target cell covers [-90, 0]. The source overlaps have
    # spherical areas sin(-30)-sin(-90) = 0.5 and sin(0)-sin(-30) = 0.5.
    np.testing.assert_allclose(result.sel(lat=-45.0), 5.0)
    np.testing.assert_allclose(result.sel(lat=45.0), 5.0)
    assert result.attrs["grid_cell_convention"] == "coordinates_are_cell_centers"


def test_area_weighted_coarsen_preserves_global_integral() -> None:
    source = _field(
        np.arange(9.0).reshape(3, 3),
        lat=[-60.0, 0.0, 60.0],
        lon=[-120.0, 0.0, 120.0],
    )

    result = area_weighted_coarsen(source, resolution=90.0)

    source_lat_area = np.array([0.5, 1.0, 0.5])
    target_lat_area = np.array([1.0, 1.0])
    source_integral = np.sum(source.values * source_lat_area[:, None] * np.deg2rad(120.0))
    target_integral = np.sum(result.values * target_lat_area[:, None] * np.deg2rad(90.0))
    assert target_integral == pytest.approx(source_integral)


def test_area_weighted_coarsen_handles_noncommensurate_cyclic_longitudes() -> None:
    source = _field(
        np.array([[0.0, 6.0, 0.0], [0.0, 6.0, 0.0]]),
        lat=[-45.0, 45.0],
        lon=[240.0, 0.0, 120.0],
    )

    result = area_weighted_coarsen(
        source,
        target_lat=[-45.0, 45.0],
        target_lon=[90.0, 270.0],
    )

    # Each 180-degree target cell contains 120 degrees of a zero-valued
    # source cell and 60 degrees of the six-valued center cell.
    np.testing.assert_allclose(result, 2.0)
    np.testing.assert_array_equal(result.lon, [-90.0, 90.0])


def test_area_weighted_coarsen_renormalizes_around_missing_values() -> None:
    source = _field(
        np.array([[2.0, np.nan, 4.0], [2.0, np.nan, 4.0]]),
        lat=[-45.0, 45.0],
        lon=[-120.0, 0.0, 120.0],
    )

    result = area_weighted_coarsen(
        source,
        target_lat=[-45.0, 45.0],
        target_lon=[-90.0, 90.0],
    )

    np.testing.assert_allclose(result.sel(lon=-90.0), 2.0)
    np.testing.assert_allclose(result.sel(lon=90.0), 4.0)


def test_area_weighted_coarsen_variance_uses_squared_normalized_weights() -> None:
    standard_deviation = _field(
        np.array([[3.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
        lat=[-45.0, 45.0],
        lon=[-120.0, 0.0, 120.0],
    )
    valid = xr.ones_like(standard_deviation, dtype=bool)

    result = area_weighted_coarsen_variance(
        standard_deviation,
        valid,
        target_lat=[-45.0, 45.0],
        target_lon=[-90.0, 90.0],
    )

    # The western target cell has normalized longitude weights 2/3 and 1/3.
    # sqrt((2/3)^2 * 3^2 + (1/3)^2 * 6^2) = sqrt(8).
    np.testing.assert_allclose(result.sel(lon=-90.0), np.sqrt(8.0))

    masked = valid.copy()
    masked.loc[{"lon": -120.0}] = False
    masked_result = area_weighted_coarsen_variance(
        standard_deviation,
        masked,
        target_lat=[-45.0, 45.0],
        target_lon=[-90.0, 90.0],
    )
    np.testing.assert_allclose(masked_result.sel(lon=-90.0), 6.0)


def test_periodic_bilinear_interpolates_continuously_across_seam() -> None:
    source = _field(
        np.array([[4.0, 0.0, 0.0, 2.0], [4.0, 0.0, 0.0, 2.0]]),
        lat=[-45.0, 45.0],
        lon=[-135.0, -45.0, 45.0, 135.0],
    )

    result = periodic_bilinear(source, [0.0], [181.0, 179.0])

    np.testing.assert_array_equal(result.lon, [-179.0, 179.0])
    assert result.sel(lon=-179.0).item() == pytest.approx(3.0 + 2.0 / 90.0)
    assert result.sel(lon=179.0).item() == pytest.approx(3.0 - 2.0 / 90.0)
    assert np.isfinite(result).all()


def test_periodic_bilinear_is_exact_for_linear_interior_field() -> None:
    lat = [-30.0, 30.0]
    lon = [-135.0, -45.0, 45.0, 135.0]
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    source = _field(lat_2d + 2.0 * lon_2d, lat=lat, lon=lon)

    result = periodic_bilinear(source, [-15.0, 15.0], [-90.0, 0.0, 90.0])

    target_lat_2d, target_lon_2d = np.meshgrid([-15.0, 15.0], [-90.0, 0.0, 90.0], indexing="ij")
    np.testing.assert_allclose(result, target_lat_2d + 2.0 * target_lon_2d)


def test_periodic_bilinear_clamps_latitude_and_preserves_extra_dimensions() -> None:
    base = _field(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        lat=[-45.0, 45.0],
        lon=[-90.0, 90.0],
    )
    source = xr.concat([base, base + 10.0], dim=xr.IndexVariable("time", [0, 1]))

    result = periodic_bilinear(source, [-89.0, 89.0], [-90.0, 90.0])

    assert result.dims == ("time", "lat", "lon")
    np.testing.assert_allclose(result.isel(time=0), base)
    np.testing.assert_allclose(result.isel(time=1), base + 10.0)


def test_regrid_validation_rejects_ambiguous_or_duplicate_coordinates() -> None:
    source = _field(np.ones((2, 2)), lat=[-45.0, 45.0], lon=[-90.0, 90.0])

    with pytest.raises(ValueError, match="resolution or explicit"):
        area_weighted_coarsen(
            source,
            resolution=90.0,
            target_lat=[-45.0, 45.0],
            target_lon=[-90.0, 90.0],
        )
    with pytest.raises(ValueError, match="divide both"):
        area_weighted_coarsen(source, resolution=7.0)
    with pytest.raises(ValueError, match="unique after normalization"):
        periodic_bilinear(source, [0.0], [0.0, 360.0])
