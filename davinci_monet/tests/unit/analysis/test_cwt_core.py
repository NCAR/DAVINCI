"""Tests for the shared CWT transform and inverse contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from davinci_monet.analysis.cwt_core import (
    cwt_reconstruct,
    cwt_reconstruction_error,
    cwt_transform,
)


def _series(size: int = 512) -> np.ndarray:
    time = np.arange(size)
    values = np.sin(2.0 * np.pi * time / 16.0)
    values += 0.4 * np.cos(2.0 * np.pi * time / 37.0)
    return (values - values.mean()) / values.std()


def test_shared_transform_matches_direct_pycwt_contract() -> None:
    import pycwt

    values = _series(256)
    mother = pycwt.Morlet(6.0)
    wave, scales, frequencies, coi, _, _ = pycwt.cwt(values, 1.0, 0.25, 2.0, -1, mother)
    direct_significance, _ = pycwt.significance(
        1.0,
        1.0,
        scales,
        0,
        0.2,
        significance_level=0.95,
        wavelet=mother,
    )

    result = cwt_transform(values, dt=1.0, dj=0.25, alpha=0.2)

    np.testing.assert_allclose(result.coefficients, wave, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.scales, scales, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.periods, 1.0 / frequencies, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.coi, coi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.local_significance, direct_significance, rtol=0.0, atol=0.0)


def test_global_significance_is_invariant_to_time_unit() -> None:
    values = _series(256)
    in_days = cwt_transform(values, dt=1.0, s0=2.0, j=20, alpha=0.2)
    in_hours = cwt_transform(values, dt=24.0, s0=48.0, j=20, alpha=0.2)

    np.testing.assert_allclose(in_hours.scales / 24.0, in_days.scales)
    np.testing.assert_allclose(
        in_hours.global_significance,
        in_days.global_significance,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_unfiltered_inverse_round_trip_is_within_five_percent() -> None:
    values = _series()
    result = cwt_transform(values, dt=1.0, alpha=0.0)

    reconstructed = cwt_reconstruct(result)

    assert reconstructed.shape == values.shape
    assert cwt_reconstruction_error(values, result) < 0.05
    assert np.corrcoef(values, reconstructed)[0, 1] > 0.999


def test_selected_coefficients_can_be_reconstructed() -> None:
    values = _series()
    result = cwt_transform(values, dt=1.0, alpha=0.0)
    selected = result.coefficients.copy()
    selected[(result.periods < 12.0) | (result.periods > 20.0), :] = 0.0

    reconstructed = cwt_reconstruct(result, selected)

    assert reconstructed.std() > 0.5
    with pytest.raises(ValueError, match="coefficient shape"):
        cwt_reconstruct(result, selected[:-1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0}, "dt"),
        ({"dt": 1.0, "dj": -0.1}, "dj"),
        ({"dt": 1.0, "alpha": 1.0}, "alpha"),
        ({"dt": 1.0, "significance_level": 1.0}, "significance_level"),
    ],
)
def test_transform_rejects_invalid_controls(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cwt_transform(_series(32), **kwargs)
