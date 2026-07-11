"""Support-aware reconstruction and exact shifted-log AOD scaling."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from davinci_monet.analysis.base import (
    AnalysisResult,
    AnalysisRuntime,
    ArtifactDeclaration,
    DerivedAnalysis,
)
from davinci_monet.analysis.provenance import consistent_spec_hash
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry
from davinci_monet.util.logspace import (
    apply_shifted_log_correction,
    ratio_delta_bounds,
)

if TYPE_CHECKING:
    from davinci_monet.config.schema import AODScalingSpec


def reconstruct_log_correction(
    basis: xr.DataArray,
    coefficients: xr.DataArray,
    bias_applied: xr.DataArray,
    support: xr.DataArray,
    model_time: xr.DataArray,
) -> xr.Dataset:
    """Reconstruct the requested correction on the complete model time grid."""
    patterns = _transpose_exact(basis, ("mode", "lat", "lon"), "basis")
    values = _transpose_exact(coefficients, ("time", "mode"), "coefficients")
    monthly_bias = _transpose_exact(bias_applied, ("month", "lat", "lon"), "climatological bias")
    monthly_support = _transpose_exact(support, ("month", "lat", "lon"), "spatial support")
    _validate_model_time(model_time)
    _validate_months(monthly_bias, "climatological bias")
    _validate_months(monthly_support, "spatial support")
    _require_finite(patterns, "basis")
    _require_finite(monthly_bias, "climatological bias")
    _require_finite(monthly_support, "spatial support")
    if _any((monthly_support < 0.0) | (monthly_support > 1.0)):
        raise ValueError("spatial support must be between zero and one")

    try:
        patterns, values = xr.align(patterns, values, join="exact")
        patterns, monthly_bias, monthly_support = xr.align(
            patterns, monthly_bias, monthly_support, join="exact"
        )
    except ValueError as exc:
        raise ValueError(
            "scaling inputs have incompatible mode or analysis-grid coordinates"
        ) from exc

    if values.get_index("time").has_duplicates:
        raise ValueError("filtered coefficients contain duplicate times")
    values = values.sortby("time").reindex(time=model_time)
    finite_values = cast(xr.DataArray, np.isfinite(values))
    coefficient_available = finite_values.all("mode")
    safe_values = values.where(coefficient_available, 0.0).fillna(0.0)
    anomaly = xr.dot(safe_values, patterns, dim="mode").transpose("time", "lat", "lon")

    month_index = xr.DataArray(
        model_time.dt.month.data,
        dims=("time",),
        coords={"time": model_time},
        name="month",
    )
    try:
        daily_bias = monthly_bias.sel(month=month_index).transpose("time", "lat", "lon")
        daily_support = monthly_support.sel(month=month_index).transpose("time", "lat", "lon")
    except KeyError as exc:
        raise ValueError("monthly bias/support do not cover every model calendar month") from exc

    requested = (daily_bias + daily_support * anomaly).where(daily_support > 0.0, 0.0)
    requested = requested.rename("delta_log_requested")
    requested.attrs.update(
        units="1",
        long_name="Requested support-tapered shifted-log AOD correction",
    )
    anomaly = anomaly.rename("delta_log_anomaly")
    anomaly.attrs.update(units="1", long_name="Filtered EOF anomaly reconstruction")
    daily_bias = daily_bias.rename("clim_bias_applied")
    daily_bias.attrs.update(units="1", long_name="Applied climatological log-AOD bias")
    daily_support = daily_support.rename("spatial_support")
    daily_support.attrs.update(units="1", long_name="Monthly spatial correction support")
    coefficient_available = coefficient_available.rename("coefficient_available")
    coefficient_available.attrs.update(
        units="1", long_name="All filtered mode coefficients are available"
    )
    return xr.Dataset(
        {
            "delta_log_requested": requested,
            "delta_log_anomaly": anomaly,
            "clim_bias_applied": daily_bias,
            "spatial_support": daily_support,
            "coefficient_available": coefficient_available,
        }
    )


def scale_reconstructed_aod(
    model_aod: xr.DataArray,
    delta_requested: xr.DataArray,
    support: xr.DataArray,
    *,
    epsilon: float,
    r_bounds: tuple[float, float],
    aod_floor: float,
) -> xr.Dataset:
    """Apply exact bounded shifted-log conversion and build diagnostics."""
    model = _transpose_exact(model_aod, ("time", "lat", "lon"), "model AOD")
    requested = _transpose_exact(delta_requested, ("time", "lat", "lon"), "requested correction")
    spatial_support = _transpose_exact(support, ("time", "lat", "lon"), "daily spatial support")
    try:
        model, requested, spatial_support = xr.align(
            model, requested, spatial_support, join="exact"
        )
    except ValueError as exc:
        raise ValueError("model AOD and reconstructed corrections are not aligned") from exc

    converted = apply_shifted_log_correction(
        model,
        requested,
        epsilon=epsilon,
        r_bounds=r_bounds,
        aod_floor=aod_floor,
        support=spatial_support,
    )
    lower_bound, upper_bound = ratio_delta_bounds(model.fillna(0.0), r_bounds, epsilon)
    valid = cast(
        xr.DataArray,
        np.isfinite(model) & np.isfinite(requested) & np.isfinite(spatial_support),
    )
    active = valid & (model > 0.0) & (model >= aod_floor) & (spatial_support > 0.0)
    masks: dict[str, xr.DataArray] = {
        "support_identity_mask": valid & (spatial_support <= 0.0),
        "low_aod_identity_mask": valid
        & (spatial_support > 0.0)
        & ((model <= 0.0) | (model < aod_floor)),
        "lower_clip_mask": active & (requested < lower_bound),
        "upper_clip_mask": active & (requested > upper_bound),
        "correction_active_mask": active,
    }

    output = xr.Dataset(
        {
            "r": converted["ratio"].rename("r"),
            "delta_log_requested": converted["delta_requested"].rename("delta_log_requested"),
            "delta_log_safe": converted["delta_safe"].rename("delta_log_safe"),
            "delta_log_applied": converted["delta_applied"].rename("delta_log_applied"),
            "aod_target": converted["aod_target"],
            "clip_reason": converted["clip_reason"],
            "spatial_support": spatial_support,
            **{name: mask.rename(name) for name, mask in masks.items()},
        }
    )
    output["r"].attrs.update(units="1", long_name="Applied physical AOD ratio")
    output["aod_target"].attrs = dict(model.attrs)
    output["aod_target"].attrs["long_name"] = "Target aerosol optical depth"
    output["spatial_support"].attrs.update(
        units="1", long_name="Monthly spatial correction support"
    )
    for name in masks:
        output[name].attrs.update(units="1", long_name=name.replace("_", " ").title())

    spatial_dims = ("lat", "lon")
    valid_count = valid.sum(dim=list(spatial_dims)).rename("valid_cell_count")
    valid_count.attrs.update(units="1", long_name="Finite scaling cells")
    output["valid_cell_count"] = valid_count
    for name, mask in masks.items():
        stem = name.removesuffix("_mask")
        count = mask.sum(dim=list(spatial_dims)).rename(f"{stem}_count")
        fraction = (count / valid_count.where(valid_count > 0)).fillna(0.0)
        fraction = fraction.rename(f"{stem}_fraction")
        count.attrs.update(units="1", long_name=f"{stem.replace('_', ' ').title()} cells")
        fraction.attrs.update(
            units="1", long_name=f"Fraction of finite cells with {stem.replace('_', ' ')}"
        )
        output[count.name] = count
        output[fraction.name] = fraction
    output.attrs.update(
        {
            "analysis_type": "aod_scaling",
            "log_epsilon": float(epsilon),
            "r_min": float(r_bounds[0]),
            "r_max": float(r_bounds[1]),
            "aod_floor": float(aod_floor),
            "scaling_space": "shifted_log_aod",
        }
    )
    return output


def build_aod_scaling(
    basis: xr.Dataset,
    projection: xr.Dataset,
    coefficients: xr.Dataset,
    model: xr.Dataset,
    spec: AODScalingSpec,
) -> xr.Dataset:
    """Build one analysis-grid scaling product from its four named inputs."""
    spec_hash = consistent_spec_hash([basis, projection, coefficients, model])
    patterns = _require_variable(basis, spec.basis_variable, "basis")
    bias = _require_variable(projection, spec.bias_variable, "projection")
    support = _require_variable(projection, spec.support_variable, "projection")
    filtered = _require_variable(coefficients, spec.coefficients_variable, "filtered coefficients")
    model_aod = _require_variable(model, spec.model_variable, "model")
    _validate_basis_metadata(basis)
    basis_signature = _validate_projection_identity(patterns, projection, coefficients)
    epsilon = _log_epsilon(model, model_aod, basis, projection, coefficients)

    reconstruction = reconstruct_log_correction(
        patterns, filtered, bias, support, model_aod["time"]
    )
    converted = scale_reconstructed_aod(
        model_aod,
        reconstruction["delta_log_requested"],
        reconstruction["spatial_support"],
        epsilon=epsilon,
        r_bounds=spec.r_bounds,
        aod_floor=spec.aod_floor,
    )
    converted["delta_log_anomaly"] = reconstruction["delta_log_anomaly"]
    converted["clim_bias_applied"] = reconstruction["clim_bias_applied"]
    converted["coefficient_available"] = reconstruction["coefficient_available"]
    converted["eofs"] = patterns
    converted["pc"] = filtered
    for variable in ("resolution",):
        if variable in projection:
            converted[variable] = projection[variable]
    for variable in ("valid_segment", "coi", "power_significance"):
        if variable in coefficients:
            converted[variable] = coefficients[variable]
    converted.attrs.update(
        {
            "basis_variable": spec.basis_variable,
            "bias_variable": spec.bias_variable,
            "support_variable": spec.support_variable,
            "coefficients_variable": spec.coefficients_variable,
            "model_variable": spec.model_variable,
            "artifact_policy": "time_chunked_lazy",
            "time_chunk_days": int(spec.time_chunk_days),
            "projection_basis_signature": basis_signature,
            "projection_log_epsilon": epsilon,
        }
    )
    if "band_max" in coefficients.attrs:
        converted.attrs["band_max"] = float(coefficients.attrs["band_max"])
    if spec_hash is not None:
        converted.attrs["spec_hash"] = spec_hash
    time_size = int(converted.sizes.get("time", 0))
    if time_size:
        converted = converted.chunk({"time": min(spec.time_chunk_days, time_size)})
    return converted


def _require_variable(dataset: xr.Dataset, variable: str, role: str) -> xr.DataArray:
    if variable not in dataset:
        raise ValueError(f"{role} input is missing variable {variable!r}")
    return dataset[variable]


def _transpose_exact(data: xr.DataArray, dims: tuple[str, ...], description: str) -> xr.DataArray:
    if set(data.dims) != set(dims) or len(data.dims) != len(dims):
        raise ValueError(f"{description} must have dimensions {dims}, got {data.dims}")
    for dim in dims:
        if dim not in data.coords:
            raise ValueError(f"{description} is missing its {dim!r} coordinate")
    return data.transpose(*dims)


def _validate_model_time(time: xr.DataArray) -> None:
    if time.dims != ("time",) or not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError("model time must be a one-dimensional datetime coordinate")
    values = np.asarray(time.values).astype("datetime64[ns]")
    if values.size == 0 or np.isnat(values).any():
        raise ValueError("model time must contain finite daily samples")
    if values.size > 1 and not np.all(np.diff(values) == np.timedelta64(1, "D")):
        raise ValueError("model time must be a complete, strictly increasing daily axis")


def _validate_months(data: xr.DataArray, description: str) -> None:
    months = np.asarray(data["month"].values, dtype=np.int64)
    if len(np.unique(months)) != months.size or np.any((months < 1) | (months > 12)):
        raise ValueError(f"{description} month coordinate must be unique integers in [1, 12]")


def _validate_basis_metadata(basis: xr.Dataset) -> None:
    missing = {"eof_rotation", "eof_standardize"}.difference(basis.attrs)
    if missing:
        raise ValueError(f"AOD scaling basis is missing metadata: {sorted(missing)}")
    rotation = str(basis.attrs["eof_rotation"]).lower()
    standardized = str(basis.attrs["eof_standardize"]).lower()
    if rotation != "none" or standardized not in {"false", "0"}:
        raise ValueError("AOD scaling requires unrotated, unstandardized EOF patterns")


def _validate_projection_identity(
    patterns: xr.DataArray,
    projection: xr.Dataset,
    coefficients: xr.Dataset,
) -> str:
    signature = _basis_signature(patterns)
    for role, dataset in (("projection", projection), ("filtered coefficients", coefficients)):
        candidate = str(dataset.attrs.get("projection_basis_signature", "")).strip()
        if not candidate:
            raise ValueError(f"{role} input is missing projection basis signature metadata")
        if candidate != signature:
            raise ValueError(f"{role} input uses an incompatible projection basis signature")
    return signature


def _basis_signature(patterns: xr.DataArray) -> str:
    ordered = _transpose_exact(patterns, ("mode", "lat", "lon"), "basis")
    digest = hashlib.sha256()
    for values in (
        np.asarray(ordered["mode"].values),
        np.asarray(ordered["lat"].values),
        np.asarray(ordered["lon"].values),
        np.asarray(ordered.values, dtype=np.float32),
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _log_epsilon(
    model: xr.Dataset,
    model_aod: xr.DataArray,
    basis: xr.Dataset,
    projection: xr.Dataset,
    coefficients: xr.Dataset,
) -> float:
    raw = model.attrs.get("log_epsilon", model_aod.attrs.get("log_epsilon"))
    if raw is None:
        raise ValueError("model AOD input is missing log_epsilon metadata")
    epsilon = float(raw)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("model log_epsilon must be positive and finite")
    candidates = (
        basis.attrs.get("eof_input_log_epsilon"),
        basis.attrs.get("log_epsilon"),
        projection.attrs.get("projection_log_epsilon"),
        projection.attrs.get("log_epsilon"),
        coefficients.attrs.get("projection_log_epsilon"),
    )
    if projection.attrs.get("projection_log_epsilon") is None:
        raise ValueError("projection input is missing projection log epsilon metadata")
    if coefficients.attrs.get("projection_log_epsilon") is None:
        raise ValueError("filtered coefficients input is missing projection log epsilon metadata")
    for candidate in candidates:
        if candidate is not None and not np.isclose(
            float(candidate), epsilon, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError("scaling inputs use inconsistent shifted-log epsilon values")
    return epsilon


def _require_finite(data: xr.DataArray, description: str) -> None:
    if _any(cast(xr.DataArray, ~np.isfinite(data))):
        raise ValueError(f"{description} must be finite")


def _any(condition: xr.DataArray) -> bool:
    value = condition.any()
    if value.chunks is not None:
        value = value.compute()
    return bool(value.item())


@analysis_registry.register("aod_scaling")
class AODScalingAnalysis(DerivedAnalysis):
    """Pipeline adapter for support-aware shifted-log AOD scaling."""

    name = "aod_scaling"
    long_name = "AOD Scaling"
    output_geometry = DataGeometry.GRID

    def analyze_inputs(
        self,
        inputs: Mapping[str, xr.Dataset],
        spec: AODScalingSpec,
        runtime: AnalysisRuntime,
    ) -> AnalysisResult:
        del runtime
        missing = [
            role for role in ("basis", "projection", "coefficients", "model") if role not in inputs
        ]
        if missing:
            raise ValueError(f"aod_scaling is missing named inputs: {', '.join(missing)}")
        output = build_aod_scaling(
            inputs["basis"],
            inputs["projection"],
            inputs["coefficients"],
            inputs["model"],
            spec,
        )
        return AnalysisResult(
            dataset=output,
            artifacts=(
                ArtifactDeclaration(
                    kind="netcdf_collection",
                    role="scaling",
                    reload=True,
                    options={"time_chunk_size": spec.time_chunk_days},
                ),
            ),
        )


__all__ = [
    "AODScalingAnalysis",
    "build_aod_scaling",
    "reconstruct_log_correction",
    "scale_reconstructed_aod",
]
