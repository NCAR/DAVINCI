"""Segment-aware wavelet filtering for projected EOF coefficients."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from davinci_monet.analysis.base import DerivedAnalysis
from davinci_monet.analysis.cwt_core import (
    cwt_reconstruct,
    cwt_reconstruction_error,
    cwt_transform,
)
from davinci_monet.analysis.provenance import consistent_spec_hash
from davinci_monet.analysis.reductions import ar1_alpha, normalize_series
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry

if TYPE_CHECKING:
    from davinci_monet.config.schema import WaveletFilterSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilteredMode:
    """Filtered values and diagnostics for one coefficient mode."""

    values: NDArray[np.float64]
    power: NDArray[np.float64]
    power_significance: NDArray[np.float64]
    coi: NDArray[np.float64]
    global_power: NDArray[np.float64]
    global_significance: NDArray[np.float64]
    bridged: NDArray[np.bool_]
    valid_segment: NDArray[np.bool_]
    retained_variance: float
    reconstruction_error: float
    synthesized_fraction: float
    coi_valid_fraction: float


def bridge_short_gaps(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    max_gap_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Linearly bridge only bounded interior gaps at or below a size limit."""
    series = np.asarray(values, dtype=np.float64)
    accepted = np.asarray(valid, dtype=bool) & np.isfinite(series)
    if series.ndim != 1 or accepted.shape != series.shape:
        raise ValueError("values and valid must be aligned one-dimensional arrays")
    if max_gap_samples < 0:
        raise ValueError("max_gap_samples must be non-negative")

    filled = series.copy()
    filled[~accepted] = np.nan
    bridged = np.zeros(series.shape, dtype=bool)
    index = 0
    while index < series.size:
        if accepted[index]:
            index += 1
            continue
        gap_start = index
        while index < series.size and not accepted[index]:
            index += 1
        gap_end = index
        gap_size = gap_end - gap_start
        bounded = gap_start > 0 and gap_end < series.size
        if bounded and gap_size <= max_gap_samples:
            left = filled[gap_start - 1]
            right = filled[gap_end]
            fraction = np.arange(1, gap_size + 1, dtype=float) / (gap_size + 1)
            filled[gap_start:gap_end] = left + fraction * (right - left)
            bridged[gap_start:gap_end] = True
    return filled, bridged


def contiguous_segments(valid: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return half-open slices for contiguous true regions."""
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("valid must be one-dimensional")
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def cosine_edge_taper(length: int, edge_samples: int) -> NDArray[np.float64]:
    """Return a symmetric taper that reaches exact zero at both edges."""
    if length < 0 or edge_samples < 0:
        raise ValueError("length and edge_samples must be non-negative")
    weights = np.ones(length, dtype=np.float64)
    edge = min(edge_samples, length // 2)
    if edge == 0:
        return weights
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge, dtype=float) / edge))
    weights[:edge] = ramp
    weights[-edge:] = ramp[::-1]
    return weights


def filter_projected_mode(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    *,
    dt_days: float,
    band: tuple[float, float],
    max_bridge_days: float,
    min_segment_days: float,
    keep_significant: bool,
    significance_level: float,
    omega0: float,
    dj: float,
    s0: float | None = None,
) -> tuple[FilteredMode, NDArray[np.float64]]:
    """Filter one projected mode while preserving explicit long gaps."""
    if not np.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError("dt_days must be positive and finite")
    band_min, band_max = band
    if not 0.0 < band_min < band_max:
        raise ValueError("band limits must be positive and increasing")
    if min_segment_days < 2.0 * band_max:
        raise ValueError("min_segment_days must be at least twice the maximum period")
    max_gap_samples = int(np.floor(max_bridge_days / dt_days + 1.0e-12))
    minimum_samples = int(np.ceil(min_segment_days / dt_days - 1.0e-12))
    filled, bridged = bridge_short_gaps(values, valid, max_gap_samples)
    usable = np.isfinite(filled)

    first_scale = s0 if s0 is not None else 2.0 * dt_days
    periods, scale_count = _common_period_grid(dt_days, dj, first_scale, band_max, omega0)
    sample_count = filled.size
    period_count = periods.size
    filtered = np.zeros(sample_count, dtype=np.float64)
    power = np.full((sample_count, period_count), np.nan, dtype=np.float64)
    power_significance = np.full_like(power, np.nan)
    coi = np.full(sample_count, np.nan, dtype=np.float64)
    valid_segment = np.zeros(sample_count, dtype=bool)
    global_power_sum = np.zeros(period_count, dtype=np.float64)
    global_significance_sum = np.zeros(period_count, dtype=np.float64)
    total_segment_samples = 0
    source_variance_sum = 0.0
    filtered_variance_sum = 0.0
    reconstruction_error_sum = 0.0
    retained_coi = 0
    retained_total = 0

    for start, stop in contiguous_segments(usable):
        segment_size = stop - start
        if segment_size < minimum_samples:
            continue
        segment = filled[start:stop]
        x = np.arange(segment_size, dtype=np.float64) * dt_days
        centered_x = x - x.mean()
        slope, intercept = np.polyfit(centered_x, segment, 1)
        trend = intercept + slope * centered_x
        anomaly = segment - trend
        normalized, standard_deviation, anomaly_mean = normalize_series(anomaly)
        alpha = 0.0 if np.std(anomaly) <= np.finfo(float).eps else ar1_alpha(anomaly)
        alpha = float(np.clip(np.nan_to_num(alpha), -0.99, 0.99))
        transform = cwt_transform(
            normalized,
            dt=dt_days,
            dj=dj,
            s0=first_scale,
            j=scale_count,
            omega0=omega0,
            alpha=alpha,
            significance_level=significance_level,
        )
        roundtrip_transform = cwt_transform(
            normalized,
            dt=dt_days,
            dj=dj,
            s0=first_scale,
            j=None,
            omega0=omega0,
            alpha=alpha,
            significance_level=significance_level,
        )
        if not np.allclose(transform.periods, periods, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError("segment CWT period grid drifted from the configured common grid")
        selected = transform.coefficients.copy()
        retained = (periods >= band_min) & (periods <= band_max)
        selected[~retained, :] = 0.0
        significance_ratio = transform.power / transform.local_significance[:, None]
        if keep_significant:
            selected[significance_ratio < 1.0] = 0.0
        reconstructed_anomaly = (
            cwt_reconstruct(transform, selected) * standard_deviation + anomaly_mean
        )
        reconstructed = reconstructed_anomaly + trend
        edge_samples = int(np.floor(min(band_max / dt_days, segment_size / 4.0) + 1.0e-12))
        tapered = reconstructed * cosine_edge_taper(segment_size, edge_samples)

        filtered[start:stop] = tapered
        power[start:stop] = transform.power.T
        power_significance[start:stop] = significance_ratio.T
        coi[start:stop] = transform.coi
        valid_segment[start:stop] = True
        global_power_sum += transform.global_power * segment_size
        global_significance_sum += transform.global_significance * segment_size
        total_segment_samples += segment_size
        source_variance_sum += float(np.sum((segment - segment.mean()) ** 2))
        filtered_variance_sum += float(np.sum((tapered - tapered.mean()) ** 2))
        reconstruction_error_sum += (
            cwt_reconstruction_error(normalized, roundtrip_transform) * segment_size
        )
        selected_mask = np.abs(selected) > 0.0
        inside_coi = periods[:, None] <= transform.coi[None, :]
        retained_coi += int(np.count_nonzero(selected_mask & inside_coi))
        retained_total += int(np.count_nonzero(selected_mask))

    if total_segment_samples:
        global_power = global_power_sum / total_segment_samples
        global_significance = global_significance_sum / total_segment_samples
        reconstruction_error = reconstruction_error_sum / total_segment_samples
    else:
        global_power = np.full(period_count, np.nan, dtype=np.float64)
        global_significance = np.full(period_count, np.nan, dtype=np.float64)
        reconstruction_error = np.nan
    result = FilteredMode(
        values=filtered,
        power=power,
        power_significance=power_significance,
        coi=coi,
        global_power=global_power,
        global_significance=global_significance,
        bridged=bridged,
        valid_segment=valid_segment,
        retained_variance=(
            filtered_variance_sum / source_variance_sum if source_variance_sum > 0.0 else 0.0
        ),
        reconstruction_error=reconstruction_error,
        synthesized_fraction=float(np.count_nonzero(bridged) / max(sample_count, 1)),
        coi_valid_fraction=(retained_coi / retained_total if retained_total else 0.0),
    )
    return result, periods


def _common_period_grid(
    dt: float, dj: float, s0: float, max_period: float, omega0: float
) -> tuple[NDArray[np.float64], int]:
    import pycwt

    if not np.isfinite(dj) or dj <= 0.0:
        raise ValueError("dj must be positive and finite")
    mother = pycwt.Morlet(omega0)
    minimum_period = s0 * float(mother.flambda())
    scale_count = max(0, int(np.ceil(np.log2(max_period / minimum_period) / dj)))
    scales = s0 * 2.0 ** (np.arange(scale_count + 1) * dj)
    return np.asarray(scales * mother.flambda(), dtype=np.float64), scale_count


def filter_projected_coefficients(data: xr.Dataset, spec: WaveletFilterSpec) -> xr.Dataset:
    """Filter every mode in a projection dataset on one shared period grid."""
    if spec.variable not in data:
        raise ValueError(f"wavelet_filter source is missing variable {spec.variable!r}")
    basis_signature, log_epsilon = _projection_identity(data)
    coefficients = data[spec.variable]
    if coefficients.dims != ("time", "mode"):
        coefficients = coefficients.transpose("time", "mode")
    time = np.asarray(coefficients["time"].values)
    dt_days = _regular_daily_step(time)
    valid = coefficients.notnull()
    if spec.resolution_variable in data:
        resolution = data[spec.resolution_variable].transpose("time", "mode")
        coefficients, resolution = xr.align(coefficients, resolution, join="exact")
        valid = valid & (resolution >= spec.min_resolution)

    mode_results: list[FilteredMode] = []
    periods: NDArray[np.float64] | None = None
    for mode_index in range(coefficients.sizes["mode"]):
        result, result_periods = filter_projected_mode(
            np.asarray(coefficients.isel(mode=mode_index).values, dtype=np.float64),
            np.asarray(valid.isel(mode=mode_index).values, dtype=bool),
            dt_days=dt_days,
            band=(spec.band.min, spec.band.max),
            max_bridge_days=spec.max_bridge_days,
            min_segment_days=spec.min_segment_days,
            keep_significant=spec.keep_significant,
            significance_level=spec.significance_level,
            omega0=spec.omega0,
            dj=spec.dj,
            s0=spec.s0,
        )
        if periods is None:
            periods = result_periods
        if np.isfinite(result.reconstruction_error) and result.reconstruction_error > 0.05:
            logger.warning(
                "wavelet_filter mode %s unfiltered reconstruction error is %.1f%%",
                coefficients["mode"].values[mode_index],
                100.0 * result.reconstruction_error,
            )
        mode_results.append(result)
    if periods is None:
        raise ValueError("wavelet_filter requires at least one mode")

    modes = coefficients["mode"].values
    output = xr.Dataset(
        {
            "pc": (("time", "mode"), np.stack([result.values for result in mode_results], axis=1)),
            "power": (
                ("time", "mode", "period"),
                np.stack([result.power for result in mode_results], axis=1),
                {"kind": "power", "long_name": "Wavelet power"},
            ),
            "power_significance": (
                ("time", "mode", "period"),
                np.stack([result.power_significance for result in mode_results], axis=1),
                {"kind": "power"},
            ),
            "coi": (
                ("time", "mode"),
                np.stack([result.coi for result in mode_results], axis=1),
                {"kind": "coi", "units": "days"},
            ),
            "global_power": (
                ("mode", "period"),
                np.stack([result.global_power for result in mode_results]),
                {"kind": "global"},
            ),
            "global_significance": (
                ("mode", "period"),
                np.stack([result.global_significance for result in mode_results]),
                {"kind": "global"},
            ),
            "bridged": (
                ("time", "mode"),
                np.stack([result.bridged for result in mode_results], axis=1),
            ),
            "valid_segment": (
                ("time", "mode"),
                np.stack([result.valid_segment for result in mode_results], axis=1),
            ),
            "retained_variance": (
                ("mode",),
                [result.retained_variance for result in mode_results],
            ),
            "recon_error": (
                ("mode",),
                [result.reconstruction_error for result in mode_results],
                {"long_name": "Full-scale unfiltered CWT inverse relative error"},
            ),
            "synth_fraction": (
                ("mode",),
                [result.synthesized_fraction for result in mode_results],
            ),
            "coi_valid_fraction": (
                ("mode",),
                [result.coi_valid_fraction for result in mode_results],
            ),
        },
        coords={
            "time": coefficients["time"].values,
            "mode": modes,
            "period": ("period", periods, {"units": "days", "long_name": "Period"}),
        },
        attrs={
            "analysis_type": "wavelet_filter",
            "wavelet_quantity": spec.variable,
            "dt": float(dt_days),
            "dt_units": "days",
            "band_min": float(spec.band.min),
            "band_max": float(spec.band.max),
            "projection_basis_signature": basis_signature,
            "projection_log_epsilon": log_epsilon,
        },
    )
    spec_hash = consistent_spec_hash([data])
    if spec_hash is not None:
        output.attrs["source_spec_hash"] = spec_hash
    return output


def _projection_identity(data: xr.Dataset) -> tuple[str, float]:
    signature = str(data.attrs.get("projection_basis_signature", "")).strip()
    if not signature:
        raise ValueError("wavelet_filter source is missing projection basis signature metadata")
    raw_epsilon = data.attrs.get("projection_log_epsilon")
    if raw_epsilon is None:
        raise ValueError("wavelet_filter source is missing projection log epsilon metadata")
    epsilon = float(raw_epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("wavelet_filter projection log epsilon must be positive and finite")
    return signature, epsilon


def _regular_daily_step(time: NDArray[np.generic]) -> float:
    if time.ndim != 1 or time.size < 2 or not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError("wavelet_filter requires at least two datetime samples")
    differences = np.diff(time.astype("datetime64[ns]")).astype("timedelta64[ns]").astype(float)
    dt_days = float(differences[0] / 86_400_000_000_000)
    if dt_days <= 0.0 or not np.allclose(
        differences / 86_400_000_000_000, dt_days, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("wavelet_filter requires a strictly regular time axis")
    return dt_days


@analysis_registry.register("wavelet_filter")
class WaveletFilterAnalysis(DerivedAnalysis):
    """Pipeline adapter for segment-aware projected-coefficient filtering."""

    name = "wavelet_filter"
    long_name = "Segment-aware Wavelet Filter"
    output_geometry = DataGeometry.SPECTRUM

    def analyze(self, data: xr.Dataset, spec: WaveletFilterSpec) -> xr.Dataset:
        return filter_projected_coefficients(data, spec)


__all__ = [
    "FilteredMode",
    "WaveletFilterAnalysis",
    "bridge_short_gaps",
    "contiguous_segments",
    "cosine_edge_taper",
    "filter_projected_coefficients",
    "filter_projected_mode",
]
