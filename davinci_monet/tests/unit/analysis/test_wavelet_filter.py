"""Tests for bounded-gap segment-aware projected coefficient filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis.wavelet_filter import (
    bridge_short_gaps,
    cosine_edge_taper,
    filter_projected_coefficients,
    filter_projected_mode,
)
from davinci_monet.config.schema import PeriodBandSpec, WaveletFilterSpec


def test_bridge_short_gaps_only_fills_bounded_gaps_within_limit() -> None:
    values = np.arange(12.0)
    valid = np.ones(12, dtype=bool)
    valid[:1] = False
    valid[3:5] = False
    valid[7:11] = False

    filled, bridged = bridge_short_gaps(values, valid, max_gap_samples=2)

    assert np.isnan(filled[0])
    np.testing.assert_allclose(filled[3:5], [3.0, 4.0])
    np.testing.assert_array_equal(np.flatnonzero(bridged), [3, 4])
    assert np.isnan(filled[7:11]).all()


def test_cosine_taper_is_exact_zero_at_edges_and_one_interior() -> None:
    taper = cosine_edge_taper(20, 4)

    assert taper[0] == 0.0
    assert taper[-1] == 0.0
    assert taper[4] == 1.0
    assert taper[-5] == 1.0
    np.testing.assert_allclose(taper, taper[::-1])


def test_filter_retains_configured_period_and_rejects_slow_component() -> None:
    sample_count = 512
    time = np.arange(sample_count)
    fast = np.sin(2.0 * np.pi * time / 16.0)
    slow = 0.8 * np.sin(2.0 * np.pi * time / 64.0)
    result, periods = filter_projected_mode(
        fast + slow,
        np.ones(sample_count, dtype=bool),
        dt_days=1.0,
        band=(12.0, 24.0),
        max_bridge_days=3.0,
        min_segment_days=128.0,
        keep_significant=False,
        significance_level=0.95,
        omega0=6.0,
        dj=0.25,
    )

    interior = slice(64, -64)
    fast_correlation = np.corrcoef(result.values[interior], fast[interior])[0, 1]
    slow_correlation = abs(np.corrcoef(result.values[interior], slow[interior])[0, 1])
    assert fast_correlation > 0.95
    assert slow_correlation < 0.2
    assert periods.max() >= 24.0
    assert result.reconstruction_error < 0.05


def test_filter_transfer_distinguishes_cutoff_transition_from_interior_band() -> None:
    sample_count = 2048
    time = np.arange(sample_count, dtype=np.float64)

    def gain(period: float) -> float:
        signal = np.sin(2.0 * np.pi * time / period + 0.37)
        result, _ = filter_projected_mode(
            signal,
            np.ones(sample_count, dtype=bool),
            dt_days=1.0,
            band=(4.0, 180.0),
            max_bridge_days=7.0,
            min_segment_days=360.0,
            keep_significant=False,
            significance_level=0.95,
            omega0=6.0,
            dj=0.25,
        )
        interior = slice(360, -360)
        expected = signal[interior]
        actual = result.values[interior]
        return float(np.dot(actual, expected) / np.dot(expected, expected))

    assert gain(2.0) < 0.05
    assert 0.45 < gain(4.0) < 0.65
    assert 0.95 < gain(20.0) < 1.05


def test_filter_bridges_short_gap_and_leaves_long_gap_at_identity() -> None:
    sample_count = 512
    time = np.arange(sample_count)
    values = np.sin(2.0 * np.pi * time / 16.0)
    valid = np.ones(sample_count, dtype=bool)
    valid[80:83] = False
    valid[240:300] = False

    result, _ = filter_projected_mode(
        values,
        valid,
        dt_days=1.0,
        band=(8.0, 24.0),
        max_bridge_days=3.0,
        min_segment_days=96.0,
        keep_significant=False,
        significance_level=0.95,
        omega0=6.0,
        dj=0.25,
    )

    np.testing.assert_array_equal(np.flatnonzero(result.bridged), [80, 81, 82])
    assert result.valid_segment[80:83].all()
    assert not result.valid_segment[240:300].any()
    np.testing.assert_array_equal(result.values[240:300], 0.0)
    assert result.synthesized_fraction == 3.0 / sample_count


def test_short_segments_emit_zero_correction_and_invalid_diagnostics() -> None:
    time = np.arange(100)
    values = np.sin(2.0 * np.pi * time / 12.0)
    result, _ = filter_projected_mode(
        values,
        np.ones(100, dtype=bool),
        dt_days=1.0,
        band=(8.0, 64.0),
        max_bridge_days=0.0,
        min_segment_days=128.0,
        keep_significant=False,
        significance_level=0.95,
        omega0=6.0,
        dj=0.25,
    )

    np.testing.assert_array_equal(result.values, 0.0)
    assert not result.valid_segment.any()
    assert np.isnan(result.power).all()
    assert np.isnan(result.global_power).all()


def test_dataset_filter_keeps_complete_axis_and_zeroes_unobservable_mode() -> None:
    sample_count = 256
    time = pd.date_range("2001-01-01", periods=sample_count, freq="D")
    signal = np.sin(2.0 * np.pi * np.arange(sample_count) / 16.0)
    data = xr.Dataset(
        {
            "pc": (("time", "mode"), np.stack((signal, signal), axis=1)),
            "resolution": (
                ("time", "mode"),
                np.stack((np.ones(sample_count), np.zeros(sample_count)), axis=1),
            ),
        },
        coords={"time": time, "mode": [1, 2]},
        attrs={
            "projection_basis_signature": "basis-signature-a",
            "projection_log_epsilon": 0.01,
        },
    )
    spec = WaveletFilterSpec(
        type="wavelet_filter",
        source="projection",
        band=PeriodBandSpec(min=8.0, max=32.0),
        min_segment_days=64.0,
        keep_significant=False,
    )

    output = filter_projected_coefficients(data, spec)

    assert output["pc"].dims == ("time", "mode")
    assert output["power"].dims == ("time", "mode", "period")
    np.testing.assert_array_equal(output["time"], time)
    assert output["valid_segment"].sel(mode=1).all()
    assert not output["valid_segment"].sel(mode=2).any()
    np.testing.assert_array_equal(output["pc"].sel(mode=2), 0.0)
    assert output.attrs["projection_basis_signature"] == "basis-signature-a"
    assert output.attrs["projection_log_epsilon"] == 0.01


def test_dataset_filter_rejects_missing_projection_identity() -> None:
    time = pd.date_range("2001-01-01", periods=8, freq="D")
    data = xr.Dataset(
        {"pc": (("time", "mode"), np.zeros((8, 1)))},
        coords={"time": time, "mode": [1]},
    )
    spec = WaveletFilterSpec(
        type="wavelet_filter",
        source="projection",
        band=PeriodBandSpec(min=2.0, max=3.0),
        min_segment_days=6.0,
    )

    with pytest.raises(ValueError, match="projection basis signature"):
        filter_projected_coefficients(data, spec)
