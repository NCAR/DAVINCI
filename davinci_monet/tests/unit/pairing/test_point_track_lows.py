"""Regression tests for FABLE_REVIEW rows L9 and L17.

L9: ``track.altitude_to_pressure`` uses the barometric formula, whose base
    (``1 - L*h/T0``) goes negative above ~44.3 km; raising a negative base to
    the formula's fractional exponent produces NaN. The fix clamps the base
    at 0.0 before exponentiation.

L17: ``PointStrategy._create_paired_output`` assigns the y source's raw
    ``.values`` into an ``xr.DataArray`` built with ``dims=x_ref.dims``. This
    is only correct when y's dims are already in x's order; when y's dims
    match x's dims as a *set* but differ in *order*, the values are silently
    mislabeled. The fix transposes y to x's dim order first.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.pairing.strategies.point import PointStrategy
from davinci_monet.pairing.strategies.track import altitude_to_pressure


class TestAltitudeToPressure:
    """L9: altitude_to_pressure must not produce NaN above ~44 km."""

    def test_finite_nonnegative_and_nonincreasing(self) -> None:
        altitudes_m = np.array([0.0, 10_000.0, 44_000.0, 50_000.0, 80_000.0])

        pressure = altitude_to_pressure(altitudes_m)

        assert np.all(np.isfinite(pressure)), pressure
        assert np.all(pressure >= 0.0), pressure
        assert np.all(np.diff(pressure) <= 0.0), pressure

    def test_mid_altitude_value_matches_pre_fix_result(self) -> None:
        # Snapshot of altitude_to_pressure(np.array([10_000.0])) computed
        # from the barometric formula before the >44 km clamp was added.
        # Values below ~44 km never touch the clamp, so this must remain
        # bit-identical after the fix.
        expected = 264.3690840619885

        pressure = altitude_to_pressure(np.array([10_000.0]))

        assert pressure[0] == expected


class TestCreatePairedOutputDimOrder:
    """L17: y's dims must be transposed to x's order, not assigned positionally."""

    def test_y_values_placed_by_dim_order_not_position(self) -> None:
        # x arrives with dims (time, site); y arrives with the same dims as
        # a set, but ordered (site, time). A purely positional assignment of
        # y's raw .values into dims=x_ref.dims silently transposes the data.
        time_coord = np.array([0, 1])
        site_coord = np.array(["A", "B"])

        x_data = xr.Dataset(
            {"o3": (["time", "site"], np.zeros((2, 2)))},
            coords={"time": time_coord, "site": site_coord},
        )

        # y_values[site_idx, time_idx] = site_idx * 10 + time_idx
        y_values = np.array([[100.0, 101.0], [110.0, 111.0]])
        y_at_sites = xr.Dataset(
            {"o3": (["site", "time"], y_values)},
            coords={"site": site_coord, "time": time_coord},
        )

        paired = PointStrategy()._create_paired_output(x_data, y_at_sites, "site")

        result = paired["y_o3"]
        assert result.dims == ("time", "site")

        for t_idx, t in enumerate(time_coord):
            for s_idx, s in enumerate(site_coord):
                expected = y_values[s_idx, t_idx]
                actual = result.sel(time=t, site=s).item()
                assert actual == expected, f"time={t}, site={s}: expected {expected}, got {actual}"
