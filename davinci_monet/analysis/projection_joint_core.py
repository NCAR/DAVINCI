"""Pure constrained spatial kernels for the joint projection-bias fit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass
class MonthNormal:
    """Diagonal-minus-low-rank spatial normal equation for one month."""

    diagonal: FloatArray
    factors: list[FloatArray]
    rhs: FloatArray


def seasonal_design() -> FloatArray:
    """Return the fixed constant/annual monthly design matrix."""
    month = np.arange(1.0, 13.0)
    angle = 2.0 * np.pi * (month - 0.5) / 12.0
    return np.column_stack((np.ones(12), np.sin(angle), np.cos(angle)))


def laplacian_edges(active: BoolArray, nlat: int, nlon: int) -> tuple[IntArray, IntArray]:
    """Build unique clipped-latitude/cyclic-longitude active-cell edges."""
    index = np.full(active.size, -1, dtype=np.int64)
    index[np.flatnonzero(active)] = np.arange(np.count_nonzero(active))
    pairs: set[tuple[int, int]] = set()
    for lat in range(nlat):
        for lon in range(nlon):
            cell = lat * nlon + lon
            if not active[cell]:
                continue
            candidates = []
            if lat + 1 < nlat:
                candidates.append((lat + 1) * nlon + lon)
            if nlon > 1:
                candidates.append(lat * nlon + ((lon + 1) % nlon))
            for neighbor in candidates:
                if active[neighbor]:
                    left, right = sorted((int(index[cell]), int(index[neighbor])))
                    if left != right:
                        pairs.add((left, right))
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ordered = np.asarray(sorted(pairs), dtype=np.int64)
    return ordered[:, 0], ordered[:, 1]


def _pcg(
    matvec: Callable[[FloatArray], FloatArray],
    rhs: FloatArray,
    diagonal: FloatArray,
) -> FloatArray:
    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    rhs_norm = float(np.linalg.norm(rhs))
    if float(np.linalg.norm(residual)) <= 1.0e-12 * max(rhs_norm, 1.0):
        return solution
    preconditioned = residual / np.maximum(diagonal, np.finfo(np.float64).tiny)
    direction = preconditioned.copy()
    product = float(residual @ preconditioned)
    for _ in range(max(200, 10 * rhs.size)):
        image = np.asarray(matvec(direction), dtype=np.float64)
        denominator = float(direction @ image)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("joint spatial normal equation is not positive definite")
        step = product / denominator
        solution += step * direction
        residual -= step * image
        if float(np.linalg.norm(residual)) <= 1.0e-11 * max(rhs_norm, 1.0):
            return solution
        preconditioned = residual / np.maximum(diagonal, np.finfo(np.float64).tiny)
        updated = float(residual @ preconditioned)
        direction = preconditioned + (updated / product) * direction
        product = updated
    raise ValueError("joint spatial normal equation did not converge")


def solve_month_bias(
    normal: MonthNormal,
    patterns: FloatArray,
    active: BoolArray,
    nlat: int,
    nlon: int,
    laplacian_strength: float,
) -> tuple[FloatArray, int, FloatArray, float]:
    """Solve one constrained, support-aware monthly spatial bias field."""
    result = np.zeros(active.size, dtype=np.float64)
    selected = np.flatnonzero(active)
    eigenvalue = np.zeros(patterns.shape[0], dtype=np.float64)
    if not selected.size:
        return result, 0, eigenvalue, 0.0
    factor = (
        np.concatenate([item[selected] for item in normal.factors], axis=1)
        if normal.factors
        else np.empty((selected.size, 0))
    )
    diagonal = normal.diagonal[selected]

    def information(vector: FloatArray) -> FloatArray:
        value = diagonal * vector
        if factor.shape[1]:
            value -= factor @ (factor.T @ vector)
        return value

    information_diagonal = diagonal - np.square(factor).sum(axis=1)
    positive = information_diagonal[information_diagonal > 0.0]
    precision_scale = float(np.median(positive)) if positive.size else 0.0
    edge_left, edge_right = laplacian_edges(active, nlat, nlon)
    penalty = laplacian_strength * precision_scale

    def system(vector: FloatArray) -> FloatArray:
        value = information(vector)
        if edge_left.size and penalty:
            difference = vector[edge_left] - vector[edge_right]
            np.add.at(value, edge_left, penalty * difference)
            np.add.at(value, edge_right, -penalty * difference)
        return value

    system_diagonal = np.maximum(information_diagonal, 0.0)
    if edge_left.size and penalty:
        np.add.at(system_diagonal, edge_left, penalty)
        np.add.at(system_diagonal, edge_right, penalty)
    spatial_patterns = patterns[:, selected].T
    mode_information = spatial_patterns.T @ np.column_stack(
        [information(spatial_patterns[:, mode]) for mode in range(patterns.shape[0])]
    )
    mode_information = 0.5 * (mode_information + mode_information.T)
    eigenvalue, eigenvector = np.linalg.eigh(mode_information)
    tolerance = (
        np.finfo(np.float64).eps
        * max(mode_information.shape)
        * max(float(eigenvalue.max(initial=0.0)), 1.0)
    )
    observable = eigenvalue > tolerance
    rank = int(np.count_nonzero(observable))
    unconstrained = _pcg(system, normal.rhs[selected], system_diagonal)
    if rank:
        direction = spatial_patterns @ eigenvector[:, observable]
        constraint = np.column_stack([information(direction[:, column]) for column in range(rank)])
        response = np.column_stack(
            [_pcg(system, constraint[:, column], system_diagonal) for column in range(rank)]
        )
        gram = constraint.T @ response
        unconstrained -= response @ np.linalg.solve(gram, constraint.T @ unconstrained)
    result[selected] = unconstrained
    return result, rank, np.clip(eigenvalue, 0.0, None), precision_scale


__all__ = ["MonthNormal", "laplacian_edges", "seasonal_design", "solve_month_bias"]
