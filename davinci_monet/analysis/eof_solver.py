"""Small-reference and bounded-rank matrix solvers for EOF analysis."""

from __future__ import annotations

from dataclasses import dataclass

import dask.array as da
import numpy as np
import xarray as xr
from dask import compute
from dask.array.linalg import svd_compressed


@dataclass(frozen=True)
class TruncatedSVD:
    """The matrix pieces needed to construct EOF scores and diagnostics."""

    scores: np.ndarray
    singular_values: np.ndarray
    total_variance: float
    loadings: np.ndarray | None
    sample_count: int
    feature_count: int
    matrix_dtype: str

    @property
    def rank(self) -> int:
        """Return the number of retained singular triplets."""
        return int(self.singular_values.size)

    @property
    def retained_variance(self) -> float:
        """Return the fraction of total matrix variance retained by the solve."""
        if self.total_variance <= 0.0:
            return 0.0
        fraction = (
            float(np.square(self.singular_values, dtype=np.float64).sum()) / self.total_variance
        )
        return float(np.clip(fraction, 0.0, 1.0))


def _stack_features(weighted: xr.DataArray) -> xr.DataArray:
    spatial = [dim for dim in weighted.dims if dim != "time"]
    if not spatial:
        raise ValueError("EOF requires at least one non-time feature dimension")
    return weighted.transpose("time", *spatial).stack(_feature=spatial)


def _target_rank(shape: tuple[int, int], n_modes: int) -> int:
    if n_modes < 1:
        raise ValueError("EOF n_modes must be positive")
    rank = min(n_modes, *shape)
    if rank < 1:
        raise ValueError("EOF fit selection must contain at least one time and feature")
    return int(rank)


def _full_svd(stacked: xr.DataArray, n_modes: int, need_loadings: bool) -> TruncatedSVD:
    """Compute the exact small-array reference decomposition."""
    raw = np.asarray(stacked.values, dtype=np.float64)
    matrix = np.where(np.isfinite(raw), raw, 0.0)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    rank = _target_rank(matrix.shape, n_modes)

    u_matrix, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    retained = singular[:rank]
    scores = u_matrix[:, :rank] * retained
    loadings = vt[:rank].T * retained if need_loadings else None
    total_variance = float(np.square(matrix, dtype=np.float64).sum())
    return TruncatedSVD(
        scores=scores,
        singular_values=retained,
        total_variance=total_variance,
        loadings=loadings,
        sample_count=int(matrix.shape[0]),
        feature_count=int(matrix.shape[1]),
        matrix_dtype=str(matrix.dtype),
    )


def _as_chunked_float32(stacked: xr.DataArray, rank: int, oversampling: int) -> da.Array:
    raw = stacked.data
    if isinstance(raw, da.Array):
        matrix = raw.astype(np.float32)
    else:
        array = np.asarray(raw, dtype=np.float32)
        row_chunk = min(array.shape[0], max(32, 4 * (rank + oversampling)))
        feature_chunk = min(array.shape[1], 8192)
        matrix = da.from_array(array, chunks=(row_chunk, feature_chunk))
    finite = da.isfinite(matrix)
    matrix = da.where(finite, matrix, np.float32(0.0))
    return matrix - matrix.mean(axis=0, keepdims=True)


def _randomized_svd(
    stacked: xr.DataArray,
    n_modes: int,
    *,
    seed: int,
    oversampling: int,
    iterations: int,
    need_loadings: bool,
) -> TruncatedSVD:
    """Compute a deterministic rank-k SVD without a full right-singular matrix."""
    shape = (int(stacked.sizes["time"]), int(stacked.sizes["_feature"]))
    rank = _target_rank(shape, n_modes)
    matrix = _as_chunked_float32(stacked, rank, oversampling)
    u_matrix, singular, vt = svd_compressed(
        matrix,
        rank,
        n_power_iter=iterations,
        n_oversamples=oversampling,
        seed=seed,
        compute=False,
        coerce_signs=True,
    )
    scores_task = u_matrix * singular[None, :]
    total_task = da.square(matrix.astype(np.float64)).sum(dtype=np.float64)

    if need_loadings:
        scores, singular_values, right_vectors, total_variance = compute(
            scores_task, singular, vt, total_task
        )
        loadings = np.asarray(right_vectors).T * np.asarray(singular_values)
    else:
        scores, singular_values, total_variance = compute(scores_task, singular, total_task)
        loadings = None

    return TruncatedSVD(
        scores=np.asarray(scores),
        singular_values=np.asarray(singular_values),
        total_variance=float(total_variance),
        loadings=loadings,
        sample_count=shape[0],
        feature_count=shape[1],
        matrix_dtype=str(matrix.dtype),
    )


def decompose_weighted(
    weighted: xr.DataArray,
    n_modes: int,
    *,
    solver: str,
    seed: int,
    oversampling: int,
    iterations: int,
    need_loadings: bool,
) -> TruncatedSVD:
    """Decompose a weighted EOF matrix with the configured solver."""
    stacked = _stack_features(weighted)
    if solver == "full":
        return _full_svd(stacked, n_modes, need_loadings)
    if solver == "randomized":
        return _randomized_svd(
            stacked,
            n_modes,
            seed=seed,
            oversampling=oversampling,
            iterations=iterations,
            need_loadings=need_loadings,
        )
    raise ValueError(f"unknown EOF solver {solver!r}")


__all__ = ["TruncatedSVD", "decompose_weighted"]
