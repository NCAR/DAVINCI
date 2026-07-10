"""Assembly of coupled synthetic model, observations, MMR inputs, and hidden truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.tests.synthetic._aerosol_contracts import (
    SENSORS,
    SyntheticTuningBundle,
    SyntheticTuningSpec,
    canonical_json,
    named_rng,
    provenance,
    spec_hash,
)
from davinci_monet.tests.synthetic._aerosol_inputs import (
    attrs,
    grid,
    make_model,
    make_observations,
    mask_components,
    monthly_support,
    split_labels,
    weighted_orthonormal_patterns,
)
from davinci_monet.tests.synthetic._aerosol_mmr import make_mmr
from davinci_monet.tests.synthetic._aerosol_oracles import (
    area_weighted_regrid_oracle,
    local_overpass_oracle,
    periodic_bilinear_oracle,
    shifted_log_ratio_oracle,
)
from davinci_monet.tests.synthetic._aerosol_temporal import (
    analytic_temporal_filter_target,
)


def generate_bundle(spec: SyntheticTuningSpec) -> SyntheticTuningBundle:
    hash_value = spec_hash(spec)
    daily_time = pd.date_range(spec.time_config.start, spec.time_config.end, freq="1D")
    daily_time += pd.Timedelta(hours=12)
    native_lat, native_lon = grid(spec.native_domain)
    mode_lat, mode_lon = grid(spec.mode_domain)
    isolate_last = spec.scenario != "exact_micro"
    patterns = weighted_orthonormal_patterns(
        mode_lat, mode_lon, spec.n_modes, isolate_last=isolate_last
    )
    model = make_model(spec, hash_value, daily_time, native_lat, native_lon)
    native_overpass = local_overpass_oracle(
        model["TOTEXTTAU"], daily_time, spec.local_overpass_hour
    )
    model_overpass = area_weighted_regrid_oracle(
        native_overpass.values, native_lat, native_lon, mode_lat, mode_lon
    )

    day_number = np.arange(daily_time.size, dtype=np.float64)
    model_pc = np.stack(
        [
            np.sin(2.0 * np.pi * (day_number + 0.5625) / period + phase)
            for period, phase in zip(spec.model_periods_days, np.linspace(0.1, 1.1, spec.n_modes))
        ],
        axis=1,
    )
    correction_pc_in_band = np.stack(
        [
            amplitude * np.sin(2.0 * np.pi * day_number / period + 0.4 * mode)
            for mode, (period, amplitude) in enumerate(
                zip(
                    spec.correction_periods_days,
                    np.linspace(0.065, 0.03, spec.n_modes),
                )
            )
        ],
        axis=1,
    )
    correction_pc_out_of_band = spec.out_of_band_amplitude * np.stack(
        [
            (1.0 - 0.2 * mode)
            * np.sin(2.0 * np.pi * day_number / spec.out_of_band_period_days + 0.2 * mode)
            for mode in range(spec.n_modes)
        ],
        axis=1,
    )
    centered_years = (day_number - day_number.mean()) / 365.25
    correction_pc_trend = (
        spec.correction_trend_per_year
        * centered_years[:, None]
        * np.linspace(1.0, -0.5, spec.n_modes)[None, :]
    )
    correction_pc = correction_pc_in_band + correction_pc_out_of_band + correction_pc_trend
    lat2d, lon2d = np.meshgrid(np.deg2rad(mode_lat), np.deg2rad(mode_lon), indexing="ij")
    month_number = np.arange(1, 13)[:, None, None]
    bias = 0.025 * np.sin(lat2d)[None, ...]
    bias = bias + 0.012 * np.cos(lon2d[None, ...] + 2.0 * np.pi * month_number / 12.0)
    bias = np.broadcast_to(bias, (12, mode_lat.size, mode_lon.size)).copy()
    if spec.scenario in {"null_ci", "calibration_null"}:
        correction_pc_in_band.fill(0.0)
        correction_pc_out_of_band.fill(0.0)
        correction_pc_trend.fill(0.0)
        correction_pc.fill(0.0)
        bias.fill(0.0)
    bias_for_day = bias[daily_time.month.to_numpy() - 1]
    delta_in_span = bias_for_day + np.einsum("tk,kij->tij", correction_pc, patterns)
    delta_perp = _perpendicular_correction(
        spec, daily_time.size, day_number, mode_lat, mode_lon, patterns
    )
    delta_requested = delta_in_span + delta_perp
    if spec.scenario == "low_aod_ci":
        delta_requested[:, 1, 0] = 10.0
        delta_requested[:, 1, 1] = -10.0

    nature_log = np.log(model_overpass + spec.log_epsilon) + delta_requested
    components, valid, mask_reason, short_gap, long_gap = mask_components(
        spec,
        daily_time,
        mode_lat,
        mode_lon,
        mnar_signal=nature_log,
    )
    support, support_counts = monthly_support(spec, daily_time, valid)
    support_for_day = support[daily_time.month.to_numpy() - 1]
    requested = shifted_log_ratio_oracle(
        model_overpass,
        delta_requested,
        epsilon=spec.log_epsilon,
        r_bounds=spec.r_bounds,
        aod_floor=spec.aod_floor,
    )
    scaling = shifted_log_ratio_oracle(
        model_overpass,
        delta_requested * support_for_day,
        epsilon=spec.log_epsilon,
        r_bounds=spec.r_bounds,
        aod_floor=spec.aod_floor,
        support=support_for_day,
    )
    representable = shifted_log_ratio_oracle(
        model_overpass,
        delta_in_span * support_for_day,
        epsilon=spec.log_epsilon,
        r_bounds=spec.r_bounds,
        aod_floor=spec.aod_floor,
        support=support_for_day,
    )
    observable = np.ones(spec.n_modes, dtype=np.uint8)
    if isolate_last:
        observable[-1] = 0
    temporal = analytic_temporal_filter_target(
        spec,
        correction_pc_in_band,
        correction_pc_out_of_band,
        correction_pc_trend,
        valid,
        observable,
    )
    delta_filter_requested = support_for_day * (
        bias_for_day + np.einsum("tk,kij->tij", temporal.coefficients, patterns)
    )
    filter_target = shifted_log_ratio_oracle(
        model_overpass,
        delta_filter_requested,
        epsilon=spec.log_epsilon,
        r_bounds=spec.r_bounds,
        aod_floor=spec.aod_floor,
        support=support_for_day,
    )
    common_error = _common_error_realization(spec, nature_log.shape)
    observations, realized_error, reported_sigma, common_error, holdout = make_observations(
        spec,
        hash_value,
        daily_time,
        mode_lat,
        mode_lon,
        nature_log,
        valid,
        common_error,
    )
    precision = valid / np.maximum(reported_sigma, 1.0e-12) ** 2
    precision_sum = precision.sum(axis=0)
    innovation_noise = np.divide(
        np.sum(precision * realized_error, axis=0),
        precision_sum,
        out=np.full_like(precision_sum, np.nan),
        where=precision_sum > 0.0,
    )
    log_r_native = periodic_bilinear_oracle(
        np.log(scaling.applied_ratio), mode_lat, mode_lon, native_lat, native_lon
    )
    support_native = periodic_bilinear_oracle(
        support_for_day, mode_lat, mode_lon, native_lat, native_lon
    )
    log_r_native[support_native <= 0.0] = 0.0
    r_native = np.exp(log_r_native)
    mmr, optical_truth = make_mmr(
        spec,
        hash_value,
        daily_time,
        native_lat,
        native_lon,
        native_overpass.values,
        r_native,
    )

    permanent = (mode_lat[:, None] >= 25.0) & (mode_lon[None, :] >= 60.0)
    truth = xr.Dataset(
        {
            "pattern_true": (
                ("truth_mode", "mode_lat", "mode_lon"),
                patterns,
                {"normalization": "cos(latitude)-weighted unit RMS"},
            ),
            "mode_observable_true": (("truth_mode",), observable),
            "unobservable_region_true": (
                ("mode_lat", "mode_lon"),
                permanent.astype(np.uint8),
            ),
            "model_pc_true": (("time", "truth_mode"), model_pc),
            "correction_pc_true": (("time", "truth_mode"), correction_pc),
            "correction_pc_in_band_true": (
                ("time", "truth_mode"),
                correction_pc_in_band,
            ),
            "correction_pc_out_of_band_true": (
                ("time", "truth_mode"),
                correction_pc_out_of_band,
            ),
            "correction_pc_trend_true": (
                ("time", "truth_mode"),
                correction_pc_trend,
            ),
            "correction_pc_filter_target_true": (
                ("time", "truth_mode"),
                temporal.coefficients,
            ),
            "filter_bridged_true": (
                ("time", "truth_mode"),
                temporal.bridged.astype(np.uint8),
            ),
            "filter_valid_segment_true": (
                ("time", "truth_mode"),
                temporal.valid_segment.astype(np.uint8),
            ),
            "filter_edge_weight_true": (
                ("time", "truth_mode"),
                temporal.edge_weight,
            ),
            "clim_bias_raw_true": (("month", "mode_lat", "mode_lon"), bias),
            "spatial_support_true": (("month", "mode_lat", "mode_lon"), support),
            "support_count_true": (("month", "mode_lat", "mode_lon"), support_counts),
            "delta_in_span_true": (("time", "mode_lat", "mode_lon"), delta_in_span),
            "delta_perp_true": (("time", "mode_lat", "mode_lon"), delta_perp),
            "delta_requested_true": (("time", "mode_lat", "mode_lon"), delta_requested),
            "delta_supported_true": (
                ("time", "mode_lat", "mode_lon"),
                delta_requested * support_for_day,
            ),
            "delta_applied_true": (
                ("time", "mode_lat", "mode_lon"),
                scaling.applied_delta,
            ),
            "delta_best_representable_true": (
                ("time", "mode_lat", "mode_lon"),
                representable.applied_delta,
            ),
            "delta_filter_target_true": (
                ("time", "mode_lat", "mode_lon"),
                filter_target.applied_delta,
            ),
            "model_aod_overpass_true": (
                ("time", "mode_lat", "mode_lon"),
                model_overpass,
                {"units": "1"},
            ),
            "r_requested_true": (
                ("time", "mode_lat", "mode_lon"),
                requested.requested_ratio,
            ),
            "r_applied_true": (
                ("time", "mode_lat", "mode_lon"),
                scaling.applied_ratio,
            ),
            "clip_mask_true": (
                ("time", "mode_lat", "mode_lon"),
                scaling.clip_mask,
                {"flag_masks": [1, 2, 4, 8], "flag_meanings": "lower upper low_aod unsupported"},
            ),
            "aod_target_requested_true": (
                ("time", "mode_lat", "mode_lon"),
                requested.requested_aod,
                {"units": "1"},
            ),
            "aod_target_applied_true": (
                ("time", "mode_lat", "mode_lon"),
                scaling.applied_aod,
                {"units": "1"},
            ),
            "aod_filter_target_true": (
                ("time", "mode_lat", "mode_lon"),
                filter_target.applied_aod,
                {"units": "1"},
            ),
            "r_native_true": (
                ("time", "native_lat", "native_lon"),
                r_native,
                {"units": "1"},
            ),
            "valid_mask": (
                ("sensor", "time", "mode_lat", "mode_lon"),
                valid.astype(np.uint8),
            ),
            "mask_reason": (
                ("sensor", "time", "mode_lat", "mode_lon"),
                mask_reason,
                {"flag_masks": [1, 2, 4, 8, 16]},
            ),
            "qa_flag": (
                ("sensor", "time", "mode_lat", "mode_lon"),
                np.where(valid, 3, 0).astype(np.int8),
            ),
            "obs_error_log": (
                ("sensor", "time", "mode_lat", "mode_lon"),
                realized_error,
            ),
            "common_error_log_true": (
                ("time", "mode_lat", "mode_lon"),
                common_error,
            ),
            "reported_sigma_log": (
                ("sensor", "time", "mode_lat", "mode_lon"),
                reported_sigma,
            ),
            "sensor_bias_log_true": (("sensor",), np.asarray(spec.sensor_bias_log)),
            "innovation_noise_true": (
                ("time", "mode_lat", "mode_lon"),
                innovation_noise,
            ),
            "obs_holdout_aod": (
                ("time", "mode_lat", "mode_lon"),
                holdout[0],
                {"units": "1"},
            ),
            "obs_holdout_error_log": (
                ("time", "mode_lat", "mode_lon"),
                holdout[1],
            ),
            "short_gap_day": (("time",), short_gap.astype(np.uint8)),
            "long_gap_day": (("time",), long_gap.astype(np.uint8)),
            "split": (("time",), split_labels(spec, daily_time)),
            **{
                name: (
                    ("sensor", "time", "mode_lat", "mode_lon"),
                    values.astype(np.uint8),
                )
                for name, values in components.items()
            },
        },
        coords={
            "sensor": list(SENSORS),
            "time": daily_time.values,
            "truth_mode": np.arange(1, spec.n_modes + 1),
            "month": np.arange(1, 13),
            "mode_lat": mode_lat,
            "mode_lon": mode_lon,
            "native_lat": native_lat,
            "native_lon": native_lon,
        },
        attrs=attrs(spec, hash_value, "evaluation_only:oracle"),
    )
    truth = xr.merge((truth, optical_truth), compat="no_conflicts")
    truth.attrs.update(attrs(spec, hash_value, "evaluation_only:oracle"))
    metadata = provenance(spec, hash_value)
    truth.attrs["stream_map"] = canonical_json(metadata["stream_map"])
    return SyntheticTuningBundle(spec, model, observations, mmr, truth, metadata)


def _perpendicular_correction(
    spec: SyntheticTuningSpec,
    n_days: int,
    day_number: np.ndarray,
    mode_lat: np.ndarray,
    mode_lon: np.ndarray,
    patterns: np.ndarray,
) -> np.ndarray:
    result = np.zeros((n_days, mode_lat.size, mode_lon.size), dtype=np.float64)
    if spec.scenario in {"exact_micro", "null_ci", "calibration_null"}:
        return result
    lat2d, lon2d = np.meshgrid(np.deg2rad(mode_lat), np.deg2rad(mode_lon), indexing="ij")
    candidate = np.sin(3.0 * lon2d) * np.cos(2.0 * lat2d)
    weights = np.cos(np.deg2rad(mode_lat))[:, None]
    weight_sum = float(np.broadcast_to(weights, candidate.shape).sum())
    for pattern in patterns:
        candidate -= float(np.sum(weights * candidate * pattern) / weight_sum) * pattern
    candidate /= np.sqrt(np.sum(weights * candidate**2) / weight_sum)
    residual = named_rng(spec.master_seed, "correction_residual").normal(0.0, 0.002, n_days)
    perpendicular_pc = spec.off_basis_amplitude * np.sin(2.0 * np.pi * day_number / 3.5) + residual
    return perpendicular_pc[:, None, None] * candidate[None, ...]


def _common_error_realization(spec: SyntheticTuningSpec, shape: tuple[int, ...]) -> np.ndarray:
    """Realize the serialized rank-one common covariance factor exactly."""
    if len(shape) != 3:
        raise ValueError("common observation error requires a time/lat/lon shape")
    if spec.scenario == "exact_micro" or spec.common_error_sigma == 0.0:
        return np.zeros(shape, dtype=np.float64)
    coefficient = named_rng(spec.master_seed, "common_error").normal(size=shape[0])
    temporal = spec.error_temporal_correlation
    if temporal > 0.0:
        innovation_scale = np.sqrt(1.0 - temporal**2)
        for index in range(1, coefficient.size):
            coefficient[index] = (
                temporal * coefficient[index - 1] + innovation_scale * coefficient[index]
            )
    return np.broadcast_to(
        spec.common_error_sigma * coefficient[:, None, None],
        shape,
    ).copy()
