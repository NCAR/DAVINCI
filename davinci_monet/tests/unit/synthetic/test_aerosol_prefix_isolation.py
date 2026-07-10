"""Split-isolation tests for stochastic FABLE synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.tests.synthetic._aerosol_bundle import _common_error_realization
from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic._aerosol_inputs import (
    make_observations,
    mask_components,
)
from davinci_monet.tests.synthetic._aerosol_stochastic import correlated_standard_normal
from davinci_monet.tests.synthetic.generators import TimeConfig

ObservationResult = tuple[
    dict[str, xr.Dataset],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _stress_spec(*, extended: bool = False) -> SyntheticTuningSpec:
    end = "2001-02-27" if extended else "2001-02-17"
    development_end = end
    return SyntheticTuningSpec(
        scenario="synthetic_osse",
        master_seed=314159,
        time_config=TimeConfig("2001-01-01", end, "1h"),
        split_windows=(
            ("basis_train", "2001-01-01", "2001-01-12"),
            ("bias_fit", "2001-01-13", "2001-01-24"),
            ("calibration", "2001-01-25", "2001-02-05"),
            ("development_test", "2001-02-06", development_end),
        ),
        heteroscedastic_strength=0.7,
        error_temporal_correlation=0.55,
        error_spatial_correlation=0.6,
        cloud_fraction=0.35,
        mnar_cloud_strength=0.8,
        qa_failure_fraction=0.12,
    )


def _time(spec: SyntheticTuningSpec) -> pd.DatetimeIndex:
    return pd.date_range(spec.time_config.start, spec.time_config.end, freq="1D") + pd.Timedelta(
        hours=12
    )


def _signal(time: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    day = np.arange(time.size, dtype=np.float64)[:, None, None]
    lat_pattern = np.sin(np.deg2rad(lat))[None, :, None]
    lon_pattern = np.cos(np.deg2rad(lon))[None, None, :]
    return -1.5 + 0.15 * np.sin(day / 3.0) + 0.2 * lat_pattern + 0.1 * lon_pattern


def _mask_result(
    spec: SyntheticTuningSpec,
    time: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    signal: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return mask_components(spec, time, lat, lon, mnar_signal=signal)


def _assert_mask_prefix_equal(
    left: tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    prefix: int,
) -> None:
    for name in left[0]:
        np.testing.assert_array_equal(left[0][name][:, :prefix], right[0][name][:, :prefix])
    np.testing.assert_array_equal(left[1][:, :prefix], right[1][:, :prefix])
    np.testing.assert_array_equal(left[2][:, :prefix], right[2][:, :prefix])
    np.testing.assert_array_equal(left[3][:prefix], right[3][:prefix])
    np.testing.assert_array_equal(left[4][:prefix], right[4][:prefix])


def test_masks_ignore_changed_and_appended_development_suffix() -> None:
    spec = _stress_spec()
    extended_spec = _stress_spec(extended=True)
    time = _time(spec)
    extended_time = _time(extended_spec)
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([-135.0, -45.0, 45.0, 135.0])
    signal = _signal(time, lat, lon)
    prefix = 36

    changed = signal.copy()
    changed[prefix:] = -5.0 * signal[prefix:, ::-1, ::-1]
    baseline = _mask_result(spec, time, lat, lon, signal)
    changed_result = _mask_result(spec, time, lat, lon, changed)
    _assert_mask_prefix_equal(baseline, changed_result, prefix)
    assert np.any(baseline[0]["cloud"][:, prefix:] != changed_result[0]["cloud"][:, prefix:])

    extended_signal = _signal(extended_time, lat, lon)
    extended_signal[prefix:] = -7.0 * extended_signal[prefix:, ::-1, ::-1]
    extended = _mask_result(extended_spec, extended_time, lat, lon, extended_signal)
    _assert_mask_prefix_equal(baseline, extended, prefix)


def _observation_result(
    spec: SyntheticTuningSpec,
    time: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    signal: np.ndarray,
    valid: np.ndarray,
) -> ObservationResult:
    common = _common_error_realization(spec, signal.shape)
    return make_observations(spec, "prefix-isolation", time, lat, lon, signal, valid, common)


def _assert_observation_prefix_equal(
    left: ObservationResult, right: ObservationResult, prefix: int
) -> None:
    left_observations, left_realized, left_reported, left_common, left_holdout = left
    right_observations, right_realized, right_reported, right_common, right_holdout = right
    for sensor in left_observations:
        for name in left_observations[sensor].data_vars:
            np.testing.assert_array_equal(
                left_observations[sensor][name].values[:prefix],
                right_observations[sensor][name].values[:prefix],
            )
    np.testing.assert_array_equal(left_realized[:, :prefix], right_realized[:, :prefix])
    np.testing.assert_array_equal(left_reported[:, :prefix], right_reported[:, :prefix])
    np.testing.assert_array_equal(left_common[:prefix], right_common[:prefix])
    np.testing.assert_array_equal(left_holdout[:, :prefix], right_holdout[:, :prefix])


def test_observations_ignore_changed_and_appended_development_suffix() -> None:
    spec = _stress_spec()
    extended_spec = _stress_spec(extended=True)
    time = _time(spec)
    extended_time = _time(extended_spec)
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([-135.0, -45.0, 45.0, 135.0])
    signal = _signal(time, lat, lon)
    prefix = 36
    baseline_masks = _mask_result(spec, time, lat, lon, signal)
    baseline = _observation_result(spec, time, lat, lon, signal, baseline_masks[1])

    changed_signal = signal.copy()
    changed_signal[prefix:] = -5.0 * signal[prefix:, ::-1, ::-1]
    changed_masks = _mask_result(spec, time, lat, lon, changed_signal)
    changed = _observation_result(spec, time, lat, lon, changed_signal, changed_masks[1])
    _assert_observation_prefix_equal(baseline, changed, prefix)
    assert np.any(
        baseline[0]["sensor_a"]["aod_550nm"].values[prefix:]
        != changed[0]["sensor_a"]["aod_550nm"].values[prefix:]
    )

    extended_signal = _signal(extended_time, lat, lon)
    extended_signal[prefix:] = -7.0 * extended_signal[prefix:, ::-1, ::-1]
    extended_masks = _mask_result(extended_spec, extended_time, lat, lon, extended_signal)
    extended = _observation_result(
        extended_spec,
        extended_time,
        lat,
        lon,
        extended_signal,
        extended_masks[1],
    )
    _assert_observation_prefix_equal(baseline, extended, prefix)


def test_correlated_and_common_noise_have_exact_stable_prefixes() -> None:
    spec = _stress_spec()
    prefix_shape = (36, 3, 4)
    extended_shape = (58, 3, 4)

    correlated = correlated_standard_normal(spec, "sensor_a_noise", prefix_shape)
    extended_correlated = correlated_standard_normal(spec, "sensor_a_noise", extended_shape)
    np.testing.assert_array_equal(correlated, extended_correlated[: prefix_shape[0]])

    common = _common_error_realization(spec, prefix_shape)
    extended_common = _common_error_realization(spec, extended_shape)
    np.testing.assert_array_equal(common, extended_common[: prefix_shape[0]])
