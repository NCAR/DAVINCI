"""Tests for stable shifted-log AOD correction helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import xarray as xr

from davinci_monet.util.logspace import (
    apply_shifted_log_correction,
    ratio_delta_bounds,
    shifted_exp,
    shifted_log,
)


def _array(values: list[float]) -> xr.DataArray:
    return xr.DataArray(values, dims="cell", coords={"cell": np.arange(len(values))})


def test_shifted_log_round_trip_and_negative_mask() -> None:
    aod = _array([0.0, 0.01, 0.2, -1.0])
    restored = shifted_exp(shifted_log(aod, 0.01), 0.01)

    np.testing.assert_allclose(restored.values[:3], aod.values[:3], atol=1.0e-14)
    assert np.isnan(restored.values[3])


def test_exact_shifted_log_ratio_is_not_exp_delta() -> None:
    model = _array([0.01, 0.5])
    desired_ratio = _array([2.0, 2.0])
    delta = cast(
        xr.DataArray,
        np.log((model * desired_ratio + 0.01) / (model + 0.01)),
    )

    out = apply_shifted_log_correction(
        model,
        delta,
        epsilon=0.01,
        r_bounds=(0.2, 5.0),
        aod_floor=0.0,
    )

    np.testing.assert_allclose(out["ratio"], desired_ratio, rtol=1.0e-12)
    assert not np.isclose(float(np.exp(delta.isel(cell=0))), 2.0)


def test_ratio_bounds_are_applied_before_extreme_exponentiation() -> None:
    model = _array([0.1, 0.1])
    out = apply_shifted_log_correction(
        model,
        _array([-1.0e300, 1.0e300]),
        epsilon=0.01,
        r_bounds=(0.25, 4.0),
        aod_floor=0.0,
    )

    np.testing.assert_allclose(out["ratio"], [0.25, 4.0], rtol=1.0e-12)
    np.testing.assert_array_equal(out["clip_reason"], [-1, 1])
    assert np.isfinite(out["delta_safe"]).all()


def test_low_aod_and_zero_support_receive_identity() -> None:
    model = _array([0.0, 0.0005, 0.2])
    support = _array([1.0, 1.0, 0.0])
    out = apply_shifted_log_correction(
        model,
        _array([10.0, 10.0, 10.0]),
        epsilon=0.01,
        aod_floor=0.001,
        support=support,
    )

    np.testing.assert_allclose(out["ratio"], 1.0)
    np.testing.assert_array_equal(out["clip_reason"], [2, 2, 2])


def test_chunked_correction_validation_and_output_remain_lazy() -> None:
    model = _array([0.1, 0.2, 0.3]).chunk({"cell": 1})
    delta = _array([0.1, 0.2, 0.3]).chunk({"cell": 1})
    support = _array([1.0, 0.5, 0.0]).chunk({"cell": 1})

    out = apply_shifted_log_correction(
        model,
        delta,
        epsilon=0.01,
        support=support,
    )

    assert out["ratio"].chunks is not None
    np.testing.assert_allclose(out["ratio"].isel(cell=2).compute(), 1.0)

    invalid_support = support.where(support["cell"] != 1, 2.0)
    with pytest.raises(ValueError, match="support must be between 0 and 1"):
        apply_shifted_log_correction(
            model,
            delta,
            epsilon=0.01,
            support=invalid_support,
        )


def test_exact_zero_aod_is_identity_when_floor_is_zero() -> None:
    out = apply_shifted_log_correction(
        _array([0.0, 0.1]),
        _array([10.0, 0.2]),
        epsilon=0.01,
        aod_floor=0.0,
    )

    assert out["ratio"].isel(cell=0).item() == 1.0
    assert out["aod_target"].isel(cell=0).item() == 0.0
    assert out["clip_reason"].isel(cell=0).item() == 2
    assert np.isfinite(out["ratio"]).all()


def test_ratio_delta_bounds_match_exact_physical_ratios() -> None:
    model = _array([0.01, 0.2, 1.0])
    lower, upper = ratio_delta_bounds(model, (0.3, 2.5), 0.01)

    low_target = (model + 0.01) * np.exp(lower) - 0.01
    high_target = (model + 0.01) * np.exp(upper) - 0.01
    np.testing.assert_allclose(low_target / model, 0.3, rtol=1.0e-12)
    np.testing.assert_allclose(high_target / model, 2.5, rtol=1.0e-12)


@pytest.mark.parametrize("epsilon", [0.0, -0.1, np.inf])
def test_invalid_epsilon_rejected(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        shifted_log(_array([0.1]), epsilon)
