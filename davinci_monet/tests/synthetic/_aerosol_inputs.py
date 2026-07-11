"""Synthetic hourly model, observing masks, and daily sensor products."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.tests.synthetic._aerosol_contracts import (
    SCHEMA_VERSION,
    SENSORS,
    SyntheticTuningSpec,
    named_rng,
)
from davinci_monet.tests.synthetic._aerosol_stochastic import (
    cloud_validity,
    correlated_standard_normal,
    heteroscedastic_sigma,
)
from davinci_monet.tests.synthetic.generators import Domain


def grid(domain: Domain) -> tuple[np.ndarray, np.ndarray]:
    lon_width = (domain.lon_max - domain.lon_min) / domain.n_lon
    lat_width = (domain.lat_max - domain.lat_min) / domain.n_lat
    lon = domain.lon_min + lon_width * (np.arange(domain.n_lon) + 0.5)
    lat = domain.lat_min + lat_width * (np.arange(domain.n_lat) + 0.5)
    return lat, lon


def weighted_orthonormal_patterns(
    lat: np.ndarray, lon: np.ndarray, count: int, *, isolate_last: bool = False
) -> np.ndarray:
    """Construct analytic weighted-orthonormal patterns without production EOF code."""
    lat2d, lon2d = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    unsupported = (lat[:, None] >= 25.0) & (lon[None, :] >= 60.0)
    candidates: list[np.ndarray] = []
    for mode in range(count):
        harmonic = mode // 2 + 1
        if mode % 3 == 0:
            field = np.cos(lat2d) ** harmonic * np.cos(harmonic * lon2d)
        elif mode % 3 == 1:
            field = np.sin(harmonic * lat2d) + 0.25 * np.sin(lon2d)
        else:
            field = np.cos((harmonic + 1) * lon2d) * np.cos(lat2d) + 0.2 * np.sin(lat2d)
        if isolate_last:
            if mode == count - 1:
                field = unsupported.astype(np.float64) * (
                    1.0 + 0.1 * np.cos(lon2d) + 0.1 * np.sin(lat2d)
                )
            else:
                field = np.where(unsupported, 0.0, field)
        candidates.append(field)
    weights = np.cos(np.deg2rad(lat))[:, None]
    weight_sum = float(np.broadcast_to(weights, (lat.size, lon.size)).sum())
    basis: list[np.ndarray] = []
    for candidate in candidates:
        vector = candidate.copy()
        for pattern in basis:
            vector -= float(np.sum(weights * vector * pattern) / weight_sum) * pattern
        norm = float(np.sqrt(np.sum(weights * vector**2) / weight_sum))
        if norm < 1.0e-12:
            raise ValueError("analytic pattern set became linearly dependent")
        basis.append(vector / norm)
    return np.stack(basis)


def attrs(spec: SyntheticTuningSpec, hash_value: str, role: str) -> dict[str, str | int]:
    return {
        "synthetic": "true",
        "scenario": spec.scenario,
        "schema_version": SCHEMA_VERSION,
        "root_seed": spec.master_seed,
        "spec_hash": hash_value,
        "role": role,
    }


def split_labels(spec: SyntheticTuningSpec, days: pd.DatetimeIndex) -> np.ndarray:
    labels = np.full(days.size, "", dtype="U20")
    normalized = days.normalize()
    for name, raw_start, raw_end in spec.split_windows:
        inside = (normalized >= pd.Timestamp(raw_start)) & (normalized <= pd.Timestamp(raw_end))
        labels[inside] = name
    return labels


def make_model(
    spec: SyntheticTuningSpec,
    hash_value: str,
    daily_time: pd.DatetimeIndex,
    native_lat: np.ndarray,
    native_lon: np.ndarray,
) -> xr.Dataset:
    start = daily_time[0].normalize() - pd.Timedelta(days=1)
    end = daily_time[-1].normalize() + pd.Timedelta(days=1, hours=23)
    hourly_time = pd.date_range(start, end, freq="1h")
    elapsed_days = (hourly_time - daily_time[0].normalize()) / pd.Timedelta(days=1)
    patterns = weighted_orthonormal_patterns(
        native_lat,
        native_lon,
        spec.n_modes,
        isolate_last=spec.scenario != "exact_micro",
    )
    if spec.scenario == "synthetic_osse":
        aod = _make_osse_model_values(
            spec, hourly_time, daily_time, native_lat, native_lon, patterns
        )
        return _model_dataset(spec, hash_value, hourly_time, native_lat, native_lon, aod)
    phases = np.linspace(0.1, 1.1, spec.n_modes)
    local_elapsed_days = np.asarray(elapsed_days)[:, None] + native_lon[None, :] / 360.0
    pcs = np.stack(
        [
            np.sin(2.0 * np.pi * local_elapsed_days / period + phase)
            for period, phase in zip(spec.model_periods_days, phases)
        ],
        axis=1,
    )
    lat2d, lon2d = np.meshgrid(np.deg2rad(native_lat), np.deg2rad(native_lon), indexing="ij")
    baseline = 0.14 + 0.07 * np.cos(lat2d) ** 2 + 0.015 * (1.0 + np.cos(lon2d))
    month = hourly_time.month.to_numpy()
    seasonal = 0.025 * np.cos(2.0 * np.pi * (month - 1) / 12.0)[:, None, None]
    seasonal = seasonal * np.cos(lat2d)[None, ...]
    variability = np.einsum(
        "tkj,k,kij->tij", pcs, np.linspace(0.045, 0.025, spec.n_modes), patterns
    )
    utc_hour = hourly_time.hour.to_numpy()[:, None, None]
    local_hour = np.mod(utc_hour + native_lon[None, None, :] / 15.0, 24.0)
    diurnal = 0.035 * np.sin(2.0 * np.pi * (local_hour - 13.5) / 24.0)
    diurnal = diurnal * (0.65 + 0.35 * np.cos(lat2d)[None, ...])
    residual = np.zeros((hourly_time.size, native_lat.size, native_lon.size))
    if spec.scenario != "exact_micro":
        residual = np.asarray(
            named_rng(spec.master_seed, "model_residual").normal(0.0, 0.0015, size=residual.shape),
            dtype=np.float64,
        )
    aod = (
        np.exp(
            np.log(baseline + spec.log_epsilon)[None, ...]
            + seasonal
            + variability
            + diurnal
            + residual
        )
        - spec.log_epsilon
    )
    if spec.scenario == "low_aod_ci":
        lat_block = max(1, spec.native_domain.n_lat // spec.mode_domain.n_lat)
        lon_block = max(1, spec.native_domain.n_lon // spec.mode_domain.n_lon)
        aod[:, :lat_block, :lon_block] = 0.0
        aod[:, :lat_block, lon_block : 2 * lon_block] = 0.5 * spec.aod_floor
    return _model_dataset(spec, hash_value, hourly_time, native_lat, native_lon, aod)


def _model_dataset(
    spec: SyntheticTuningSpec,
    hash_value: str,
    hourly_time: pd.DatetimeIndex,
    native_lat: np.ndarray,
    native_lon: np.ndarray,
    aod: np.ndarray,
) -> xr.Dataset:
    dataset = xr.Dataset(
        {
            "TOTEXTTAU": (
                ("time", "lat", "lon"),
                aod.astype(np.float32),
                {"long_name": "synthetic total aerosol optical depth", "units": "1"},
            )
        },
        coords={"time": hourly_time.values, "lat": native_lat, "lon": native_lon},
        attrs=attrs(spec, hash_value, "fitting_input:model_hourly"),
    )
    dataset["lat"].attrs = {"units": "degrees_north"}
    dataset["lon"].attrs = {"units": "degrees_east"}
    return dataset


def _make_osse_model_values(
    spec: SyntheticTuningSpec,
    hourly_time: pd.DatetimeIndex,
    daily_time: pd.DatetimeIndex,
    native_lat: np.ndarray,
    native_lon: np.ndarray,
    patterns: np.ndarray,
) -> np.ndarray:
    """Generate the large OSSE model field in bounded time chunks."""
    lat2d, lon2d = np.meshgrid(np.deg2rad(native_lat), np.deg2rad(native_lon), indexing="ij")
    baseline_log = np.log(
        0.14 + 0.07 * np.cos(lat2d) ** 2 + 0.015 * (1.0 + np.cos(lon2d)) + spec.log_epsilon
    )
    drift_pattern = np.sin(2.0 * lon2d) * np.cos(lat2d)
    drift_pattern /= np.sqrt(np.mean(np.square(drift_pattern)))
    phases = np.linspace(0.1, 1.1, spec.n_modes)
    amplitudes = np.linspace(0.045, 0.025, spec.n_modes)
    output = np.empty((hourly_time.size, native_lat.size, native_lon.size), dtype=np.float32)
    random = named_rng(spec.master_seed, "model_residual")
    elapsed_all = np.asarray(
        (hourly_time - daily_time[0].normalize()) / pd.Timedelta(days=1),
        dtype=np.float64,
    )
    drift_scale_all = np.linspace(-0.5, 0.5, hourly_time.size, dtype=np.float64)
    chunk_hours = 31 * 24
    for start in range(0, hourly_time.size, chunk_hours):
        stop = min(start + chunk_hours, hourly_time.size)
        selected_time = hourly_time[start:stop]
        elapsed = elapsed_all[start:stop]
        local_elapsed = elapsed[:, None] + native_lon[None, :] / 360.0
        pcs = np.stack(
            [
                np.sin(2.0 * np.pi * local_elapsed / period + phase)
                for period, phase in zip(spec.model_periods_days, phases)
            ],
            axis=1,
        )
        variability = np.einsum("tkj,k,kij->tij", pcs, amplitudes, patterns)
        drift_pc = np.sin(2.0 * np.pi * elapsed / spec.model_periods_days[0] + phases[0])
        variability += (
            spec.basis_drift_amplitude
            * amplitudes[0]
            * drift_scale_all[start:stop, None, None]
            * drift_pc[:, None, None]
            * drift_pattern[None, ...]
        )
        month = selected_time.month.to_numpy()
        seasonal = 0.025 * np.cos(2.0 * np.pi * (month - 1) / 12.0)[:, None, None]
        seasonal = seasonal * np.cos(lat2d)[None, ...]
        local_hour = np.mod(
            selected_time.hour.to_numpy()[:, None, None] + native_lon[None, None, :] / 15.0,
            24.0,
        )
        diurnal = 0.035 * np.sin(2.0 * np.pi * (local_hour - 13.5) / 24.0)
        diurnal = diurnal * (0.65 + 0.35 * np.cos(lat2d)[None, ...])
        residual = random.normal(
            0.0,
            0.0015,
            size=(stop - start, native_lat.size, native_lon.size),
        )
        log_aod = baseline_log[None, ...] + seasonal + variability + diurnal + residual
        output[start:stop] = (np.exp(log_aod) - spec.log_epsilon).astype(np.float32)
    return output


def mask_components(
    spec: SyntheticTuningSpec,
    daily_time: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    mnar_signal: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(SENSORS), daily_time.size, lat.size, lon.size)
    short_gap = np.zeros(daily_time.size, dtype=bool)
    long_gap = np.zeros(daily_time.size, dtype=bool)
    if spec.scenario == "exact_micro":
        components = {
            name: np.ones(shape, dtype=bool)
            for name in ("footprint", "seasonal_visibility", "cloud", "day_available", "qa_pass")
        }
    else:
        footprint = np.ones(shape, dtype=bool)
        footprint[0] &= lon[None, None, :] <= 60.0
        footprint[1] &= lon[None, None, :] >= -60.0
        permanent = (lat[:, None] >= 25.0) & (lon[None, :] >= 60.0)
        footprint[:, :, permanent] = False

        seasonal = np.ones(shape, dtype=bool)
        cloud = np.ones(shape, dtype=bool)
        qa_pass = np.ones(shape, dtype=bool)
        day_available = np.ones(shape, dtype=bool)
        if spec.scenario not in {"multi_sensor_ci", "low_aod_ci"}:
            month = daily_time.month.to_numpy()
            north_winter = np.isin(month, [11, 12, 1, 2, 3])[:, None, None]
            south_winter = np.isin(month, [5, 6, 7, 8, 9])[:, None, None]
            seasonal &= ~(north_winter[None, ...] & (lat[None, None, :, None] >= 45.0))
            seasonal &= ~(south_winter[None, ...] & (lat[None, None, :, None] <= -45.0))
            fitting_days = _fitting_prefix_indices(spec, daily_time)
            for sensor_index, sensor in enumerate(SENSORS):
                cloud[sensor_index] = cloud_validity(
                    spec,
                    sensor,
                    (daily_time.size, lat.size, lon.size),
                    mnar_signal,
                )
                qa_pass[sensor_index] = (
                    named_rng(spec.master_seed, f"{sensor}_qa").random(
                        (daily_time.size, lat.size, lon.size)
                    )
                    >= spec.qa_failure_fraction
                )
                outage_day = int(
                    fitting_days[
                        named_rng(spec.master_seed, f"{sensor}_outages").integers(
                            1, fitting_days.size - 1
                        )
                    ]
                )
                day_available[sensor_index, outage_day] = False
            day_available[:, 1] = False
            all_invalid_day = int(fitting_days[fitting_days.size // 2])
            cloud[:, all_invalid_day] = False
            if fitting_days.size >= spec.long_gap_days + spec.short_gap_days + 8:
                short_position = max(3, fitting_days.size // 4)
                long_position = max(
                    short_position + spec.short_gap_days + 2,
                    fitting_days.size // 2 + 2,
                )
                short_indices = fitting_days[short_position : short_position + spec.short_gap_days]
                long_indices = fitting_days[long_position : long_position + spec.long_gap_days]
                short_gap[short_indices] = True
                long_gap[long_indices] = True
                day_available[:, short_gap | long_gap] = False
        components = {
            "footprint": footprint,
            "seasonal_visibility": seasonal,
            "cloud": cloud,
            "day_available": day_available,
            "qa_pass": qa_pass,
        }
    valid = np.logical_and.reduce(tuple(components.values()))
    reason = np.zeros(shape, dtype=np.uint8)
    for bit, component in enumerate(components.values()):
        reason |= (~component).astype(np.uint8) << bit
    return components, valid, reason, short_gap, long_gap


def _fitting_prefix_indices(spec: SyntheticTuningSpec, daily_time: pd.DatetimeIndex) -> np.ndarray:
    labels = split_labels(spec, daily_time)
    indices = np.flatnonzero(np.isin(labels, ("basis_train", "bias_fit", "calibration")))
    if indices.size < 3:
        raise ValueError("synthetic masks require at least three non-development days")
    return indices


def monthly_support(
    spec: SyntheticTuningSpec, daily_time: pd.DatetimeIndex, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    shape = (12, valid.shape[-2], valid.shape[-1])
    support = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)
    any_sensor = np.asarray(valid.any(axis=0), dtype=bool)
    fit_days = split_labels(spec, daily_time) == "bias_fit"
    for month in range(1, 13):
        selected = np.asarray((daily_time.month == month) & fit_days, dtype=bool)
        if not np.any(selected):
            continue
        counts[month - 1] = any_sensor[selected].sum(axis=0)
        fraction = np.asarray(counts[month - 1] / int(np.sum(selected)))
        observed = counts[month - 1] > 0.0
        supported = (fraction >= spec.support_min_fraction) & observed
        raw_support = np.clip(
            (fraction - spec.support_min_fraction)
            / (spec.support_full_fraction - spec.support_min_fraction),
            0.0,
            1.0,
        )
        smoothed = _masked_support_smooth(raw_support, supported, passes=2)
        support[month - 1] = np.where(supported, np.clip(smoothed, 0.0, 1.0), 0.0)
    return support, counts


def _masked_support_smooth(values: np.ndarray, valid: np.ndarray, *, passes: int) -> np.ndarray:
    current = np.where(valid, np.asarray(values, dtype=np.float64), 0.0)
    mask = np.asarray(valid, dtype=bool)
    for _ in range(passes):
        lon_data = np.concatenate((current[:, -1:], current, current[:, :1]), axis=1)
        lon_mask = np.concatenate((mask[:, -1:], mask, mask[:, :1]), axis=1)
        padded_data = np.pad(lon_data, ((1, 1), (0, 0)), mode="constant")
        padded_mask = np.pad(lon_mask, ((1, 1), (0, 0)), mode="constant")
        total = np.zeros_like(current)
        count = np.zeros_like(current)
        nlat, nlon = current.shape
        for lat_offset in range(3):
            for lon_offset in range(3):
                neighbor = padded_data[
                    lat_offset : lat_offset + nlat,
                    lon_offset : lon_offset + nlon,
                ]
                neighbor_valid = padded_mask[
                    lat_offset : lat_offset + nlat,
                    lon_offset : lon_offset + nlon,
                ]
                total += np.where(neighbor_valid, neighbor, 0.0)
                count += neighbor_valid
        current = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
        current = np.where(mask, current, 0.0)
    return current


def make_observations(
    spec: SyntheticTuningSpec,
    hash_value: str,
    daily_time: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    nature_log: np.ndarray,
    valid: np.ndarray,
    common: np.ndarray,
) -> tuple[dict[str, xr.Dataset], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common = np.asarray(common, dtype=np.float64)
    if common.shape != nature_log.shape or not np.all(np.isfinite(common)):
        raise ValueError("common observation error must be finite and aligned with nature_log")
    realized = np.empty((len(SENSORS),) + nature_log.shape, dtype=np.float64)
    reported = np.empty_like(realized)
    observations: dict[str, xr.Dataset] = {}
    for sensor_index, sensor in enumerate(SENSORS):
        independent = np.zeros_like(nature_log)
        if spec.scenario != "exact_micro":
            independent = correlated_standard_normal(
                spec,
                f"{sensor}_noise",
                nature_log.shape,
            )
        sigma = heteroscedastic_sigma(
            spec.sensor_error_sigma[sensor_index],
            nature_log,
            spec.heteroscedastic_strength,
        )
        independent *= sigma
        realized[sensor_index] = common + independent + spec.sensor_bias_log[sensor_index]
        reported[sensor_index] = sigma
        observed = np.exp(nature_log + realized[sensor_index]) - spec.log_epsilon
        observed = np.where(valid[sensor_index], observed, 9.0 + sensor_index)
        qa = np.where(valid[sensor_index], 3, 0).astype(np.int8)
        sigma_values = reported[sensor_index].astype(np.float32)
        dataset = xr.Dataset(
            {
                "aod_550nm": (("time", "lat", "lon"), observed, {"units": "1"}),
                "log_aod": (
                    ("time", "lat", "lon"),
                    np.log(observed + spec.log_epsilon),
                    {"units": "1"},
                ),
                "reported_sigma_log": (
                    ("time", "lat", "lon"),
                    sigma_values,
                    {"units": "1", "space": "shifted_log"},
                ),
                "obs_error_std": (
                    ("time", "lat", "lon"),
                    sigma_values,
                    {"units": "1", "space": "shifted_log"},
                ),
                "common_error_factor": (
                    ("time", "common_mode", "lat", "lon"),
                    np.full(
                        (daily_time.size, 1, lat.size, lon.size),
                        spec.common_error_sigma,
                        dtype=np.float32,
                    ),
                    {"units": "1", "space": "shifted_log"},
                ),
                "QA": (("time", "lat", "lon"), qa, {"valid_min": 0, "valid_max": 3}),
                "qa_flag": (("time", "lat", "lon"), qa, {"valid_min": 0, "valid_max": 3}),
                "valid_mask": (("time", "lat", "lon"), valid[sensor_index].astype(np.uint8)),
            },
            coords={
                "time": daily_time.values,
                "common_mode": ["shared_sensor_error"],
                "lat": lat,
                "lon": lon,
            },
            attrs=attrs(spec, hash_value, f"fitting_input:observation:{sensor}"),
        )
        observations[sensor] = dataset
    holdout_error = named_rng(spec.master_seed, "holdout_noise").normal(
        0.0, max(spec.sensor_error_sigma), size=nature_log.shape
    )
    if spec.scenario == "exact_micro":
        holdout_error.fill(0.0)
    holdout = np.exp(nature_log + holdout_error) - spec.log_epsilon
    return observations, realized, reported, common, np.stack((holdout, holdout_error))
