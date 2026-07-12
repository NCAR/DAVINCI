"""EOF (Empirical Orthogonal Function) decomposition of a gridded field."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from davinci_monet.analysis.base import AnalysisResult, ArtifactDeclaration, DerivedAnalysis
from davinci_monet.analysis.eof_fit import (
    copy_fit_artifact_metadata,
    copy_input_metadata,
    fit_time_text,
    select_fit_data,
)
from davinci_monet.analysis.eof_solver import TruncatedSVD, decompose_weighted
from davinci_monet.core.coordinates import VERTICAL_DIM_NAMES, vertical_dim_name
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry

if TYPE_CHECKING:
    from davinci_monet.analysis.base import AnalysisRuntime
    from davinci_monet.config.schema import EOFSpec

logger = logging.getLogger(__name__)

_LAT_NAMES = ("latitude", "lat", "LAT", "Latitude")
_LON_NAMES = ("longitude", "lon", "LON", "Longitude")
_MIN_NORTH_EFFECTIVE_N = 2.0


def _named_coord(da: xr.DataArray, names: tuple[str, ...], kind: str) -> xr.DataArray:
    for name in names:
        if name in da.coords:
            return da.coords[name]
    raise ValueError(f"EOF requires a {kind} coordinate (one of {names})")


def _lat_coord(da: xr.DataArray) -> xr.DataArray:
    return _named_coord(da, _LAT_NAMES, "latitude")


def _lon_coord(da: xr.DataArray) -> xr.DataArray:
    return _named_coord(da, _LON_NAMES, "longitude")


def _vertical_dim(da: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray) -> str | None:
    horiz = set(lat.dims) | set(lon.dims)
    extra_dims = [str(dim) for dim in da.dims if dim != "time" and dim not in horiz]
    if not extra_dims:
        return None
    if len(extra_dims) > 1:
        raise ValueError(
            f"EOF variable '{da.name or '<unnamed>'}' has unsupported non-spatial "
            f"dimensions {extra_dims}; select or reduce extra dimensions so only one "
            "recognized vertical dimension remains"
        )
    candidate = extra_dims[0]
    if vertical_dim_name(da) == candidate:
        return candidate
    recognized = ", ".join(VERTICAL_DIM_NAMES)
    raise ValueError(
        f"EOF variable '{da.name or '<unnamed>'}' has unsupported non-spatial dimension "
        f"'{candidate}'; recognized vertical dimensions include {recognized}. Select or "
        "reduce the extra dimension before EOF analysis"
    )


def _area_weight(da: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """sqrt(cos(lat)) broadcast over the latitude dimension."""
    coslat = xr.DataArray(np.cos(np.deg2rad(lat)).clip(min=0.0), dims=lat.dims, coords=lat.coords)
    return coslat**0.5


def _layer_mass_weight(data: xr.Dataset, vdim: str) -> xr.DataArray | None:
    """sqrt(normalized layer pressure thickness) over the vertical dim, or None.

    Uses ``ilev`` pressure edges if present, else CESM hybrid coefficients
    (hyai/hybi + PS or P0). Returns None when no vertical thickness info exists;
    the caller then falls back to equal layer weight (logged, not warned).
    """
    nlev = int(data.sizes[vdim])
    dp: np.ndarray | None = None
    if "ilev" in data.coords and int(data.sizes.get("ilev", 0)) == nlev + 1:
        dp = np.abs(np.diff(np.asarray(data["ilev"].values, dtype=float)))
    elif {"hyai", "hybi"} <= set(data.variables):
        p0 = float(data["P0"]) if "P0" in data.variables else 1.0e5
        ps = float(np.asarray(data["PS"].values).mean()) if "PS" in data.variables else p0
        edges = (
            np.asarray(data["hyai"].values, float) * p0
            + np.asarray(data["hybi"].values, float) * ps
        )
        if edges.size == nlev + 1:
            dp = np.abs(np.diff(edges))
    if dp is None:
        return None
    dpn = dp / dp.sum()
    return xr.DataArray(np.sqrt(dpn), dims=[vdim])


def _fix_sign(mode: xr.DataArray, pc: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Flip each mode so its largest-|loading| spatial point is positive.

    Deterministic and robust for dipole modes (a domain-mean rule is not).
    """
    spatial = [d for d in mode.dims if d != "mode"]
    flat = mode.stack(_pt=spatial)
    idx = abs(flat).argmax("_pt")
    if idx.chunks is not None:
        idx = idx.compute()
    peak = flat.isel(_pt=idx)
    raw_signs = xr.where(peak >= 0, 1.0, -1.0)
    if raw_signs.chunks is not None:
        raw_signs = raw_signs.compute()
    signs = xr.DataArray(
        raw_signs.data,
        dims=("mode",),
        coords={"mode": mode["mode"]},
    )
    return mode * signs, pc * signs


def _effective_n(anom: xr.DataArray, lat: xr.DataArray) -> float:
    """Effective independent sample count from the area-mean series lag-1 autocorr."""
    coslat = xr.DataArray(np.cos(np.deg2rad(lat)).clip(min=0.0), dims=lat.dims, coords=lat.coords)
    spatial = [d for d in anom.dims if d != "time"]
    am = anom.weighted(coslat).mean(dim=spatial)
    x = np.asarray(am.values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n < 3:
        return _MIN_NORTH_EFFECTIVE_N
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r1 = float(np.clip(r1, -0.99, 0.99))
    return max(_MIN_NORTH_EFFECTIVE_N, n * (1.0 - r1) / (1.0 + r1))


def _varimax_rotation(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Kaiser varimax: return the (k, k) orthogonal rotation maximizing loading variance."""
    _p, k = loadings.shape
    rot = np.eye(k)
    last = 0.0
    for _ in range(max_iter):
        lam = loadings @ rot
        diag = np.diag((lam**2).sum(axis=0))
        u, s, vt = np.linalg.svd(loadings.T @ (lam**3 - lam @ diag / _p))
        rot = u @ vt
        cur = float(s.sum())
        if last != 0.0 and cur / last < 1.0 + tol:
            break
        last = cur
    return rot


def _solve_eof_scores(
    weighted: xr.DataArray,
    n_modes: int,
    rotation: str,
    *,
    solver: str,
    seed: int,
    oversampling: int,
    iterations: int,
) -> tuple[xr.DataArray, xr.DataArray, TruncatedSVD]:
    """EOF decomposition via SVD of the weighted-anomaly matrix.

    Returns unit-variance principal components ``pc(time, mode)`` and the
    explained-variance ratio ``ev_ratio(mode)``. Replaces a third-party EOF
    library to stay within the project's ``pandas<2`` dependency pin.
    """
    decomposition = decompose_weighted(
        weighted,
        n_modes,
        solver=solver,
        seed=seed,
        oversampling=oversampling,
        iterations=iterations,
        need_loadings=rotation == "varimax",
    )
    scores = decomposition.scores
    singular = decomposition.singular_values
    total = decomposition.total_variance
    k = decomposition.rank
    ev_vals = np.square(singular) / total if total > 0 else np.zeros(k)

    if rotation == "varimax" and k > 1:
        loadings = decomposition.loadings
        if loadings is None:
            raise RuntimeError("EOF solver did not return loadings required for varimax")
        rot = _varimax_rotation(loadings)
        scores = scores @ rot
        var = (scores**2).sum(axis=0)
        ev_vals = var / total if total > 0 else np.zeros(k)
        order = np.argsort(ev_vals)[::-1]  # rotation does not preserve ordering
        scores = scores[:, order]
        ev_vals = ev_vals[order]

    std = scores.std(axis=0)
    std[(std == 0.0) | ~np.isfinite(std)] = 1.0
    modes = np.arange(1, k + 1)
    pc = xr.DataArray(
        scores / std,
        dims=("time", "mode"),
        coords={"time": weighted["time"].values, "mode": modes},
    )
    ev_ratio = xr.DataArray(ev_vals, dims=("mode",), coords={"mode": modes})
    return pc, ev_ratio, decomposition


def _svd_decompose(
    weighted: xr.DataArray,
    n_modes: int,
    rotation: str,
    *,
    solver: str = "full",
    seed: int = 0,
    oversampling: int = 10,
    iterations: int = 2,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compatibility wrapper returning PCs and variance without diagnostics."""
    pc, ev_ratio, _ = _solve_eof_scores(
        weighted,
        n_modes,
        rotation,
        solver=solver,
        seed=seed,
        oversampling=oversampling,
        iterations=iterations,
    )
    return pc, ev_ratio


def _patterns_from_pc(anom: xr.DataArray, pc: xr.DataArray) -> xr.DataArray:
    """Regress anomalies onto PCs without materializing time x space x mode."""
    spatial_dims = [d for d in anom.dims if d != "time"]
    stacked = anom.transpose("time", *spatial_dims).stack(_feat=spatial_dims)
    pc_t = pc.transpose("time", "mode")

    valid_matrix = np.isfinite(stacked)
    valid_pc = np.isfinite(pc_t)
    numerator = xr.dot(stacked.where(valid_matrix, 0.0), pc_t.where(valid_pc, 0.0), dim="time")
    counts = xr.dot(valid_matrix.astype(float), valid_pc.astype(float), dim="time")
    out = numerator / counts.where(counts > 0)
    return out.unstack("_feat").transpose("mode", *spatial_dims)


@analysis_registry.register("eof")
class EOFAnalysis(DerivedAnalysis):
    """Empirical Orthogonal Function decomposition of a gridded field."""

    name = "eof"
    long_name = "Empirical Orthogonal Function Decomposition"
    output_geometry = DataGeometry.GRID

    def analyze_inputs(
        self,
        inputs: Mapping[str, xr.Dataset],
        spec: "EOFSpec",
        runtime: "AnalysisRuntime",
    ) -> AnalysisResult:
        """Fit from the source plus an optional immutable split artifact."""
        del runtime
        try:
            data = inputs["source"]
        except KeyError as exc:
            raise ValueError("EOF analysis requires a named 'source' input") from exc
        dataset = self._analyze(data, spec, inputs.get("fit_artifact"))
        return AnalysisResult(
            dataset=dataset,
            artifacts=(
                ArtifactDeclaration(
                    kind="netcdf_collection",
                    role="basis_fit",
                    reload=False,
                    options={"time_chunk_size": 366},
                ),
            ),
        )

    def analyze(self, data: xr.Dataset, spec: "EOFSpec") -> xr.Dataset:
        return self._analyze(data, spec, None)

    def _analyze(
        self, data: xr.Dataset, spec: "EOFSpec", fit_artifact: xr.Dataset | None
    ) -> xr.Dataset:
        da = data[spec.variable]
        lat = _lat_coord(da)
        lon = _lon_coord(da)
        vdim = _vertical_dim(da, lat, lon)
        if spec.level is not None:
            if vdim is None:
                logger.warning(
                    "EOF level=%s ignored for surface-only variable '%s'; no recognized "
                    "vertical dimension is present",
                    spec.level,
                    spec.variable,
                )
            else:
                da = da.isel({vdim: spec.level})
                vdim = None

        fit_da, fit_selection = select_fit_data(da, spec, fit_artifact)
        time_mean = fit_da.mean("time")
        anom = fit_da - time_mean
        climatology: xr.DataArray | None = None
        if spec.remove_seasonal_cycle:
            climatology = anom.groupby("time.month").mean("time")
            anom = anom.groupby("time.month") - climatology
        standard_deviation: xr.DataArray | None = None
        if spec.standardize:
            standard_deviation = anom.std("time")
            anom = anom / standard_deviation.where(standard_deviation > 0)

        weight = _area_weight(anom, lat)
        if vdim is not None and not spec.standardize:
            mw = _layer_mass_weight(data, vdim)
            if mw is None:
                logger.warning(
                    "EOF 3-D mass weighting unavailable for '%s'; using equal layer weight",
                    spec.variable,
                )
            else:
                weight = weight * mw
        elif vdim is not None and spec.standardize:
            logger.warning(
                "EOF standardize=True with a 3-D field: vertical mass weighting disabled "
                "(per-cell standardization already equalizes variance)"
            )
        weight = weight.fillna(0.0)

        weighted = (anom * weight).fillna(0.0)
        pc, ev_ratio, decomposition = _solve_eof_scores(
            weighted,
            spec.n_modes,
            spec.rotation,
            solver=spec.solver,
            seed=spec.solver_seed,
            oversampling=spec.solver_oversampling,
            iterations=spec.solver_iterations,
        )
        # Regression onto unit-variance PCs preserves lazy spatial chunks.
        mode_raw = _patterns_from_pc(anom, pc)
        spatial_dims = [d for d in mode_raw.dims if d != "mode"]
        mode_raw, pc = _fix_sign(mode_raw, pc)

        n_modes = int(ev_ratio.sizes["mode"])
        mode_idx = np.arange(1, n_modes + 1)

        auxiliary = [name for name in mode_raw.coords if name not in mode_raw.dims]
        if auxiliary:
            mode_raw = mode_raw.drop_vars(auxiliary)
        mode_raw = mode_raw.assign_coords(mode=mode_idx).rename("eofs")
        mode_raw.attrs = {
            "units": "1" if spec.standardize else str(da.attrs.get("units", "")),
            "long_name": f"EOF spatial pattern of {spec.variable}",
            "kind": "eofs",
        }
        pc = pc.assign_coords(mode=mode_idx).transpose("time", "mode").rename("pc")
        pc.attrs = {
            "units": "1",
            "long_name": f"Principal component of {spec.variable}",
            "kind": "pc",
        }
        ev_ratio = ev_ratio.assign_coords(mode=mode_idx).rename("explained_variance")
        ev_ratio.attrs = {"kind": "scalar", "display_as_percent": True}
        singular_value = xr.DataArray(
            decomposition.singular_values,
            dims=("mode",),
            coords={"mode": mode_idx},
            name="singular_value",
            attrs={"kind": "scalar", "long_name": "retained weighted singular value"},
        )
        time_mean = time_mean.rename("time_mean")
        time_mean.attrs = {
            "units": str(da.attrs.get("units", "")),
            "long_name": f"EOF fit-window time mean of {spec.variable}",
            "kind": "preprocessing",
        }
        ds = xr.Dataset(
            {
                "eofs": mode_raw,
                "pc": pc,
                "explained_variance": ev_ratio,
                "singular_value": singular_value,
                "time_mean": time_mean,
            }
        )
        if climatology is not None:
            climatology = climatology.rename("climatology")
            climatology.attrs = {
                "units": str(da.attrs.get("units", "")),
                "long_name": f"EOF fit-window monthly climatology anomaly of {spec.variable}",
                "kind": "preprocessing",
            }
            ds["climatology"] = climatology
        if standard_deviation is not None:
            standard_deviation = standard_deviation.rename("standard_deviation")
            standard_deviation.attrs = {
                "units": str(da.attrs.get("units", "")),
                "long_name": f"EOF fit-window anomaly standard deviation of {spec.variable}",
                "kind": "preprocessing",
            }
            ds["standard_deviation"] = standard_deviation
        if spec.rotation == "none":
            n_eff = _effective_n(anom, lat)
            ds["explained_variance_error"] = xr.DataArray(
                np.minimum(ev_ratio.values * np.sqrt(2.0 / n_eff), ev_ratio.values),
                dims=("mode",),
                coords={"mode": mode_idx},
                attrs={"kind": "scalar"},
            )

        ds.attrs["eof_quantity"] = spec.variable
        ds.attrs.update(
            {
                "eof_solver": spec.solver,
                "eof_solver_seed": spec.solver_seed,
                "eof_solver_oversampling": spec.solver_oversampling,
                "eof_solver_iterations": spec.solver_iterations,
                "eof_solver_rank": decomposition.rank,
                "eof_solver_matrix_dtype": decomposition.matrix_dtype,
                "eof_rotation": spec.rotation,
                "eof_standardize": str(spec.standardize).lower(),
                "eof_remove_seasonal_cycle": str(spec.remove_seasonal_cycle).lower(),
                "eof_sample_count": decomposition.sample_count,
                "eof_feature_count": decomposition.feature_count,
                "eof_total_weighted_variance": decomposition.total_variance,
                "eof_retained_variance": decomposition.retained_variance,
                "eof_right_vectors": (
                    "truncated_for_rotation" if spec.rotation == "varimax" else "not_retained"
                ),
                "eof_fit_selection": fit_selection,
                "eof_fit_start": fit_time_text(fit_da["time"].values[0]),
                "eof_fit_end": fit_time_text(fit_da["time"].values[-1]),
                "eof_fit_count": int(fit_da.sizes["time"]),
                "eof_fit_split": spec.fit_split if spec.fit_artifact is not None else "",
                "eof_latitude_coordinate": str(lat.name or ""),
                "eof_longitude_coordinate": str(lon.name or ""),
                "eof_grid_shape": ",".join(
                    f"{dim}:{int(fit_da.sizes[dim])}" for dim in spatial_dims
                ),
            }
        )
        if spec.fit_artifact is not None:
            ds.attrs["eof_fit_artifact"] = spec.fit_artifact
        if spec.fit_window is not None:
            ds.attrs["eof_fit_requested_start"] = str(spec.fit_window.start)
            ds.attrs["eof_fit_requested_end"] = str(spec.fit_window.end)
        copy_input_metadata(ds, data, da)
        if fit_artifact is not None:
            copy_fit_artifact_metadata(ds, fit_artifact)
        return ds
