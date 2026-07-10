"""Independent analytic temporal target for FABLE synthetic corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec


@dataclass(frozen=True)
class AnalyticTemporalTarget:
    """Known-component filter target and segment-policy diagnostics."""

    coefficients: np.ndarray
    bridged: np.ndarray
    valid_segment: np.ndarray
    edge_weight: np.ndarray


def analytic_temporal_filter_target(
    spec: SyntheticTuningSpec,
    in_band_components: np.ndarray,
    out_of_band_components: np.ndarray,
    trend_components: np.ndarray,
    valid_mask: np.ndarray,
    observable_modes: np.ndarray,
) -> AnalyticTemporalTarget:
    """Apply the declared frequency, gap, segment, and taper policy analytically.

    The input arrays are generator-known components, so this oracle never estimates a
    spectrum and never calls PyCWT or the production wavelet filter.
    """
    in_band = _coefficient_array(in_band_components, spec.n_modes, "in-band components")
    out_of_band = _coefficient_array(out_of_band_components, spec.n_modes, "out-of-band components")
    trend = _coefficient_array(trend_components, spec.n_modes, "trend components")
    if in_band.shape != out_of_band.shape or in_band.shape != trend.shape:
        raise ValueError("analytic temporal components must have identical shapes")
    if valid_mask.ndim != 4 or valid_mask.shape[1] != in_band.shape[0]:
        raise ValueError("valid_mask must have dimensions sensor/time/lat/lon")
    observable: np.ndarray = np.asarray(observable_modes, dtype=bool)
    if observable.shape != (spec.n_modes,):
        raise ValueError("observable_modes must contain one flag per mode")

    band_min, band_max = spec.filter_band_days
    retained = trend.copy()
    for mode, period in enumerate(spec.correction_periods_days):
        if band_min <= period <= band_max:
            retained[:, mode] += in_band[:, mode]
    if band_min <= spec.out_of_band_period_days <= band_max:
        retained += out_of_band

    day_observed = cast(np.ndarray, np.asarray(valid_mask, dtype=bool).any(axis=(0, 2, 3)))
    mode_valid = day_observed[:, None] & observable[None, :]
    if spec.scenario == "exact_micro":
        return AnalyticTemporalTarget(
            coefficients=retained,
            bridged=np.zeros_like(mode_valid),
            valid_segment=np.broadcast_to(observable, retained.shape).copy(),
            edge_weight=np.ones_like(retained),
        )

    filtered = np.zeros_like(retained)
    bridged = np.zeros_like(mode_valid)
    valid_segment = np.zeros_like(mode_valid)
    edge_weight = np.zeros_like(retained)
    for mode in range(spec.n_modes):
        filled, mode_bridged = _bridge_short_gaps(
            retained[:, mode], mode_valid[:, mode], spec.filter_max_bridge_days
        )
        usable = np.isfinite(filled)
        bridged[:, mode] = mode_bridged
        for start, stop in _contiguous_segments(usable):
            length = stop - start
            if length < spec.filter_min_segment_days:
                continue
            weights = _cosine_edge_taper(length, min(int(band_max), length // 4))
            filtered[start:stop, mode] = filled[start:stop] * weights
            valid_segment[start:stop, mode] = True
            edge_weight[start:stop, mode] = weights
    return AnalyticTemporalTarget(filtered, bridged, valid_segment, edge_weight)


def _coefficient_array(values: np.ndarray, modes: int, description: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != modes or not np.all(np.isfinite(result)):
        raise ValueError(f"{description} must be a finite time-by-mode array")
    return result


def _bridge_short_gaps(
    values: np.ndarray, valid: np.ndarray, maximum_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    filled = np.where(valid, values, np.nan).astype(np.float64)
    bridged = np.zeros(valid.shape, dtype=bool)
    index = 0
    while index < valid.size:
        if valid[index]:
            index += 1
            continue
        start = index
        while index < valid.size and not valid[index]:
            index += 1
        stop = index
        if start > 0 and stop < valid.size and stop - start <= maximum_gap:
            fraction = np.arange(1, stop - start + 1, dtype=np.float64) / (stop - start + 1)
            filled[start:stop] = filled[start - 1] + fraction * (filled[stop] - filled[start - 1])
            bridged[start:stop] = True
    return filled, bridged


def _contiguous_segments(valid: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(np.asarray(valid, dtype=np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _cosine_edge_taper(length: int, edge: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float64)
    edge = min(edge, length // 2)
    if edge == 0:
        return weights
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge, dtype=np.float64) / edge))
    weights[:edge] = ramp
    weights[-edge:] = ramp[::-1]
    return weights


__all__ = ["AnalyticTemporalTarget", "analytic_temporal_filter_target"]
