"""Typed shared continuous-wavelet transform and reconstruction primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CWTResult:
    """Complete Morlet CWT state needed by diagnostics and reconstruction."""

    coefficients: NDArray[np.complex128]
    scales: NDArray[np.float64]
    periods: NDArray[np.float64]
    coi: NDArray[np.float64]
    alpha: float
    local_significance: NDArray[np.float64]
    global_power: NDArray[np.float64]
    global_significance: NDArray[np.float64]
    dt: float
    dj: float
    omega0: float

    @property
    def power(self) -> NDArray[np.float64]:
        """Pointwise wavelet power with shape ``(scale, time)``."""
        return np.asarray(np.abs(self.coefficients) ** 2, dtype=np.float64)


def cwt_transform(
    values: NDArray[np.floating] | list[float],
    *,
    dt: float,
    dj: float = 0.25,
    s0: float | None = None,
    j: int | None = None,
    omega0: float = 6.0,
    alpha: float = 0.0,
    significance_level: float = 0.95,
) -> CWTResult:
    """Compute one Morlet CWT using a fixed, reconstruction-ready contract."""
    import pycwt

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 2:
        raise ValueError("CWT input must be a one-dimensional series with at least two samples")
    if not np.isfinite(series).all():
        raise ValueError("CWT input must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(dj) or dj <= 0.0:
        raise ValueError("dj must be positive and finite")
    if s0 is not None and (not np.isfinite(s0) or s0 <= 0.0):
        raise ValueError("s0 must be positive and finite")
    if j is not None and j < 0:
        raise ValueError("j must be non-negative when specified")
    if not np.isfinite(omega0) or omega0 <= 0.0:
        raise ValueError("omega0 must be positive and finite")
    if not np.isfinite(alpha) or not -1.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and lie in (-1, 1)")
    if not np.isfinite(significance_level) or not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must lie in (0, 1)")

    mother = pycwt.Morlet(omega0)
    first_scale = s0 if s0 is not None else 2.0 * dt
    scale_count = j if j is not None else -1
    coefficients, scales, frequencies, coi, _, _ = pycwt.cwt(
        series, dt, dj, first_scale, scale_count, mother
    )
    scales_array = np.asarray(scales, dtype=np.float64)
    local_significance, _ = pycwt.significance(
        1.0,
        dt,
        scales_array,
        0,
        alpha,
        significance_level=significance_level,
        wavelet=mother,
    )
    power = np.asarray(np.abs(coefficients) ** 2, dtype=np.float64)
    global_power = power.mean(axis=1)
    degrees_of_freedom = series.size - scales_array
    global_significance, _ = pycwt.significance(
        1.0,
        dt,
        scales_array,
        1,
        alpha,
        significance_level=significance_level,
        dof=degrees_of_freedom,
        wavelet=mother,
    )
    return CWTResult(
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        scales=scales_array,
        periods=np.asarray(1.0 / frequencies, dtype=np.float64),
        coi=np.asarray(coi, dtype=np.float64),
        alpha=float(alpha),
        local_significance=np.asarray(local_significance, dtype=np.float64),
        global_power=global_power,
        global_significance=np.asarray(global_significance, dtype=np.float64),
        dt=float(dt),
        dj=float(dj),
        omega0=float(omega0),
    )


def cwt_reconstruct(
    result: CWTResult,
    coefficients: NDArray[np.complexfloating] | None = None,
) -> NDArray[np.float64]:
    """Reconstruct a real series from all or selected CWT coefficients."""
    import pycwt

    selected = result.coefficients if coefficients is None else np.asarray(coefficients)
    if selected.shape != result.coefficients.shape:
        raise ValueError(
            "reconstruction coefficients must match the original CWT coefficient shape"
        )
    mother = pycwt.Morlet(result.omega0)
    reconstructed = pycwt.icwt(
        selected,
        result.scales,
        result.dt,
        result.dj,
        mother,
    )
    return np.asarray(np.real(reconstructed), dtype=np.float64)


def cwt_reconstruction_error(values: NDArray[np.floating], result: CWTResult) -> float:
    """Return relative L2 error of an unfiltered inverse transform."""
    expected = np.asarray(values, dtype=np.float64)
    if expected.shape != (result.coefficients.shape[1],):
        raise ValueError("values length must match the CWT time dimension")
    denominator = float(np.linalg.norm(expected))
    error = float(np.linalg.norm(cwt_reconstruct(result) - expected))
    return error / denominator if denominator > 0.0 else error


__all__ = ["CWTResult", "cwt_reconstruct", "cwt_reconstruction_error", "cwt_transform"]
