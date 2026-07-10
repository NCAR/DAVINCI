"""Analytic tests for FABLE projection kernels."""

from __future__ import annotations

import numpy as np
import pytest

from davinci_monet.analysis.projection_core import (
    EffectiveCovariance,
    apply_inverse_covariance,
    build_effective_covariance,
    fit_monthly_bias,
    masked_boxcar_smooth,
    solve_one_day,
)


def test_woodbury_application_matches_dense_covariance() -> None:
    rng = np.random.default_rng(41)
    sigma = np.linspace(0.2, 0.5, 6)
    latitude = np.linspace(-60.0, 60.0, 6)
    factors = rng.normal(scale=0.08, size=(6, 2))
    rhs = rng.normal(size=(6, 3))

    covariance = build_effective_covariance(sigma, latitude, factors)
    dense = np.diag(covariance.diagonal) + covariance.factors @ covariance.factors.T

    np.testing.assert_allclose(
        apply_inverse_covariance(covariance, rhs),
        np.linalg.solve(dense, rhs),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_one_day_solver_uses_full_information_matrix() -> None:
    design = np.array([[1.0, 1.0], [1.0, 0.0], [0.5, -1.0]])
    residual = np.array([0.7, -0.1, 0.3])
    covariance = EffectiveCovariance(
        diagonal=np.array([0.4, 0.7, 0.3]),
        factors=np.array([[0.2], [0.1], [0.25]]),
    )
    ridge = 0.8
    dense_covariance = np.diag(covariance.diagonal) + covariance.factors @ covariance.factors.T
    inverse = np.linalg.inv(dense_covariance)
    information = design.T @ inverse @ design
    normal = information + ridge * np.eye(2)
    expected_pc = np.linalg.solve(normal, design.T @ inverse @ residual)
    expected_posterior = np.linalg.inv(normal)
    expected_resolution = np.diag(expected_posterior @ information)

    result = solve_one_day(design, residual, covariance, ridge)

    assert abs(result.information[0, 1]) > 0.1
    np.testing.assert_allclose(result.coefficients, expected_pc, rtol=1.0e-12)
    np.testing.assert_allclose(result.resolution, expected_resolution, rtol=1.0e-12)
    np.testing.assert_allclose(result.posterior_variance, np.diag(expected_posterior), rtol=1.0e-12)
    assert result.effective_rank == 2
    np.testing.assert_allclose(result.condition_number, np.linalg.cond(normal), rtol=1.0e-14)


def test_one_day_solver_returns_prior_uncertainty_without_observations() -> None:
    covariance = EffectiveCovariance(np.empty(0), np.empty((0, 1)))

    result = solve_one_day(np.empty((0, 3)), np.empty(0), covariance, ridge=2.0)

    np.testing.assert_array_equal(result.coefficients, np.zeros(3))
    np.testing.assert_array_equal(result.resolution, np.zeros(3))
    np.testing.assert_array_equal(result.posterior_variance, np.full(3, 0.5))
    np.testing.assert_array_equal(result.resolution_eigenvalues, np.zeros(3))
    np.testing.assert_array_equal(result.posterior_eigenvalues, np.full(3, 0.5))
    assert result.effective_rank == 0
    assert result.condition_number == 1.0


def test_unregularized_full_rank_projection_is_an_exact_algebra_oracle() -> None:
    design = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
            [1.0, 1.0, 0.0],
            [-0.5, 0.25, 1.0],
        ]
    )
    expected = np.array([0.7, -0.3, 0.2])
    covariance = EffectiveCovariance(np.ones(4), np.empty((4, 0)))

    result = solve_one_day(design, design @ expected, covariance, ridge=0.0)

    np.testing.assert_allclose(result.coefficients, expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(result.resolution, np.ones(3), rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        result.resolution_eigenvalues,
        np.ones(3),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert result.effective_rank == 3


def test_unregularized_projection_rejects_missing_or_rank_deficient_information() -> None:
    empty_covariance = EffectiveCovariance(np.empty(0), np.empty((0, 0)))
    with pytest.raises(ValueError, match="full-rank observation information"):
        solve_one_day(np.empty((0, 2)), np.empty(0), empty_covariance, ridge=0.0)

    design = np.array([[1.0, 1.0], [2.0, 2.0]])
    covariance = EffectiveCovariance(np.ones(2), np.empty((2, 0)))
    with pytest.raises(ValueError, match="full-rank observation information"):
        solve_one_day(design, np.array([1.0, 2.0]), covariance, ridge=0.0)


def test_monthly_fit_is_precision_weighted_and_counts_unique_days() -> None:
    innovations = np.full((2, 4, 1, 1), 99.0)
    errors = np.ones_like(innovations)
    valid = np.zeros_like(innovations, dtype=bool)
    innovations[0, 0, 0, 0] = 1.0
    innovations[0, 1, 0, 0] = 1.0
    innovations[1, 0, 0, 0] = 3.0
    errors[1, 0, 0, 0] = 2.0
    valid[0, :2, 0, 0] = True
    valid[1, 0, 0, 0] = True

    fit = fit_monthly_bias(
        innovations,
        errors,
        valid,
        np.ones(4, dtype=np.int64),
        np.array([True, True, False, False]),
        support_min_fraction=0.2,
        support_full_fraction=0.5,
        smoothing_passes=0,
        delta_bounds=(-5.0, 5.0),
    )

    expected_precision = 1.0 + 1.0 + 0.25
    assert fit.raw_mean[0, 0, 0] == (1.0 + 1.0 + 0.75) / expected_precision
    assert fit.standard_error[0, 0, 0] == np.sqrt(1.0 / expected_precision)
    np.testing.assert_array_equal(fit.sensor_count[0, :, 0, 0], [2, 1])
    assert fit.support_count[0, 0, 0] == 2
    assert fit.support_day_total[0] == 2
    assert fit.support_fraction[0, 0, 0] == 1.0
    assert fit.support[0, 0, 0] == 1.0


def test_masked_smoothing_wraps_longitude_but_not_latitude() -> None:
    values = np.array([[1.0, 0.0, 0.0, 0.0]])
    valid = np.ones_like(values, dtype=bool)

    smoothed = masked_boxcar_smooth(values, valid, passes=1)

    np.testing.assert_allclose(smoothed, [[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0]])


def test_monthly_support_uses_piecewise_taper() -> None:
    innovations = np.zeros((1, 4, 1, 1))
    errors = np.ones_like(innovations)
    valid = np.zeros_like(innovations, dtype=bool)
    valid[0, 0, 0, 0] = True

    fit = fit_monthly_bias(
        innovations,
        errors,
        valid,
        np.ones(4, dtype=np.int64),
        np.ones(4, dtype=bool),
        support_min_fraction=0.2,
        support_full_fraction=0.6,
        smoothing_passes=0,
        delta_bounds=(-1.0, 1.0),
    )

    assert fit.support_fraction[0, 0, 0] == 0.25
    assert fit.support[0, 0, 0] == (0.25 - 0.2) / (0.6 - 0.2)
