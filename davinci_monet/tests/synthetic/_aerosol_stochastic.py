"""Prefix-stable stochastic controls for synthetic FABLE observations."""

from __future__ import annotations

import numpy as np

from davinci_monet.tests.synthetic._aerosol_contracts import (
    SyntheticTuningSpec,
    named_rng,
)


def cloud_validity(
    spec: SyntheticTuningSpec,
    sensor: str,
    shape: tuple[int, int, int],
    mnar_signal: np.ndarray | None,
) -> np.ndarray:
    """Return a spatially correlated daily cloud-valid mask without suffix coupling."""
    cloud_score = named_rng(spec.master_seed, f"{sensor}_cloud").normal(size=shape)
    cloud_score = (
        cloud_score
        + np.roll(cloud_score, 1, axis=1)
        + np.roll(cloud_score, -1, axis=1)
        + np.roll(cloud_score, 1, axis=2)
        + np.roll(cloud_score, -1, axis=2)
    ) / 5.0
    if spec.mnar_cloud_strength > 0.0:
        if mnar_signal is None or mnar_signal.shape != cloud_score.shape:
            raise ValueError("MNAR cloud masking requires an aligned daily signal")
        normalized = _daily_spatial_normalize(mnar_signal, "MNAR cloud signal")
        cloud_score -= spec.mnar_cloud_strength * np.clip(normalized, -3.0, 3.0)
    if spec.cloud_fraction <= 0.0:
        return np.ones(shape, dtype=bool)
    if spec.cloud_fraction >= 1.0:
        return np.zeros(shape, dtype=bool)
    threshold = np.quantile(
        cloud_score,
        spec.cloud_fraction,
        axis=(1, 2),
        keepdims=True,
    )
    return cloud_score > threshold


def heteroscedastic_sigma(
    base_sigma: float,
    signal: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Scale errors from contemporaneous spatial contrast, never later days."""
    if strength == 0.0:
        return np.full(signal.shape, base_sigma, dtype=np.float64)
    normalized = _daily_spatial_normalize(signal, "heteroscedastic signal")
    multiplier = 1.0 + strength * (0.5 + 0.5 * np.tanh(normalized))
    return base_sigma * multiplier


def correlated_standard_normal(
    spec: SyntheticTuningSpec,
    stream: str,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a causal correlated unit-variance realization with stable prefixes."""
    if len(shape) != 3:
        raise ValueError("correlated observation noise requires time/lat/lon dimensions")
    values = named_rng(spec.master_seed, stream).normal(size=shape)
    spatial = spec.error_spatial_correlation
    if spatial > 0.0:
        smoothed = (
            values
            + np.roll(values, 1, axis=1)
            + np.roll(values, -1, axis=1)
            + np.roll(values, 1, axis=2)
            + np.roll(values, -1, axis=2)
        ) / np.sqrt(5.0)
        independent_weight = np.sqrt(1.0 - spatial**2)
        values = independent_weight * values + spatial * smoothed
        values /= _spatial_mixture_standard_deviation(shape[1:], spatial, independent_weight)
    temporal = spec.error_temporal_correlation
    if temporal > 0.0:
        innovation_scale = np.sqrt(1.0 - temporal**2)
        for index in range(1, values.shape[0]):
            values[index] = temporal * values[index - 1] + innovation_scale * values[index]
    return values


def _daily_spatial_normalize(values: np.ndarray, description: str) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 3 or not np.all(np.isfinite(data)):
        raise ValueError(f"{description} must be a finite time/lat/lon array")
    centered = data - np.median(data, axis=(1, 2), keepdims=True)
    scale = np.std(centered, axis=(1, 2), keepdims=True)
    return np.divide(
        centered,
        scale,
        out=np.zeros_like(centered),
        where=scale > 0.0,
    )


def _spatial_mixture_standard_deviation(
    shape: tuple[int, int], spatial: float, independent_weight: float
) -> float:
    kernel = np.zeros(shape, dtype=np.float64)
    scale = np.sqrt(5.0)
    for lat_index, lon_index in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        kernel[lat_index % shape[0], lon_index % shape[1]] += 1.0 / scale
    smoothed_variance = float(np.sum(np.square(kernel)))
    covariance = float(kernel[0, 0])
    variance = (
        independent_weight**2
        + spatial**2 * smoothed_variance
        + 2.0 * independent_weight * spatial * covariance
    )
    return float(np.sqrt(variance))


__all__ = ["cloud_validity", "correlated_standard_normal", "heteroscedastic_sigma"]
