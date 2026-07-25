"""Shared AOD screening, daily sampling, regridding, and log transform."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.analysis.base import AnalysisRuntime, DerivedAnalysis
from davinci_monet.core.coordinates import LATITUDE_NAMES, LONGITUDE_NAMES, wrap_longitudes
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry
from davinci_monet.util.local_time import (
    apply_local_solar_time_index,
    build_local_solar_time_index,
)
from davinci_monet.util.logspace import shifted_log
from davinci_monet.util.regrid import (
    area_weighted_coarsen,
    area_weighted_coarsen_variance,
)

if TYPE_CHECKING:
    from davinci_monet.config.schema import AODPreprocessSpec


def preprocess_aod(
    source: xr.Dataset,
    spec: AODPreprocessSpec,
    runtime: AnalysisRuntime,
    *,
    target_grid: xr.Dataset | None = None,
) -> xr.Dataset:
    """Apply the FABLE preprocessing contract to one raw AOD dataset."""
    if spec.variable not in source:
        raise ValueError(f"AOD source is missing variable {spec.variable!r}")

    aod = _canonical_horizontal(source[spec.variable])
    _validate_aod_dimensions(aod)
    raw_valid = cast(xr.DataArray, np.isfinite(aod)) & (aod >= 0.0)
    fields: dict[str, xr.DataArray] = {"aod": aod.where(raw_valid)}

    uncertainty_model = spec.uncertainty_model
    uncertainty_variable = (
        uncertainty_model.source_variable
        if uncertainty_model is not None
        else spec.uncertainty_variable
    )
    if uncertainty_variable is not None:
        fields["obs_error_std"] = _source_field(source, uncertainty_variable, raw_valid)
    for factor_name in spec.common_factor_variables:
        fields[f"factor:{factor_name}"] = _source_field(source, factor_name, raw_valid)

    if spec.sample_local_time is not None:
        start_time, end_time = _runtime_or_source_window(runtime, aod)
        index = build_local_solar_time_index(
            aod["time"],
            aod["lon"],
            start_time=start_time,
            end_time=end_time,
            local_hour=spec.sample_local_time,
            day_anchor_hour=spec.day_anchor_hour,
            tolerance=spec.sample_tolerance,
        )
        fields = {
            name: apply_local_solar_time_index(field, index) for name, field in fields.items()
        }
    else:
        fields = {
            name: _anchor_calendar_days(field, spec.day_anchor_hour)
            for name, field in fields.items()
        }

    fields = {
        name: _clip_requested_days(field, runtime, spec.day_anchor_hour)
        for name, field in fields.items()
    }
    if fields["aod"].sizes.get("time", 0) == 0:
        raise ValueError("AOD preprocessing has no samples in the requested window")

    source_valid = cast(xr.DataArray, np.isfinite(fields["aod"]))
    if "obs_error_std" in fields:
        invalid_error = source_valid & ~cast(xr.DataArray, np.isfinite(fields["obs_error_std"]))
        invalid_error |= source_valid & (
            fields["obs_error_std"] < 0.0
            if uncertainty_model is not None
            else fields["obs_error_std"] <= 0.0
        )
        if _any(invalid_error):
            if uncertainty_model is not None:
                raise ValueError(
                    "source uncertainty must be finite and non-negative " "for every valid AOD cell"
                )
            raise ValueError("obs_error_std must be finite and positive at every valid AOD cell")
    for name, field in fields.items():
        if name.startswith("factor:") and _any(
            source_valid & ~cast(xr.DataArray, np.isfinite(field))
        ):
            factor_name = name.removeprefix("factor:")
            raise ValueError(
                f"common factor {factor_name!r} must be finite at every valid AOD cell"
            )

    target_lat, target_lon = _target_coordinates(target_grid)
    needs_regrid = spec.target_grid is not None or (
        target_lat is not None
        and target_lon is not None
        and not _same_grid(fields["aod"], target_lat, target_lon)
    )
    if needs_regrid:
        fields["aod"] = area_weighted_coarsen(
            fields["aod"],
            resolution=spec.target_grid,
            target_lat=target_lat,
            target_lon=target_lon,
        )
        if "obs_error_std" in fields:
            covariance = (
                uncertainty_model.covariance
                if uncertainty_model is not None
                else spec.uncertainty_covariance
            )
            if covariance != "independent":
                raise ValueError(
                    "coarsening uncertainty requires uncertainty_covariance='independent'"
                )
            fields["obs_error_std"] = area_weighted_coarsen_variance(
                fields["obs_error_std"],
                source_valid,
                resolution=spec.target_grid,
                target_lat=target_lat,
                target_lon=target_lon,
            )
        for name in tuple(fields):
            if name.startswith("factor:"):
                fields[name] = area_weighted_coarsen(
                    fields[name].where(source_valid),
                    resolution=spec.target_grid,
                    target_lat=target_lat,
                    target_lon=target_lon,
                )

    processed_aod = fields.pop("aod").transpose("time", "lat", "lon")
    valid = cast(xr.DataArray, np.isfinite(processed_aod)) & (processed_aod >= 0.0)
    processed_aod = processed_aod.where(valid).rename("aod")
    processed_aod.attrs = dict(aod.attrs)
    processed_aod.attrs.setdefault("units", "1")
    processed_aod.attrs["long_name"] = "Screened and daily sampled aerosol optical depth"
    log_aod = shifted_log(processed_aod, spec.log_epsilon).rename("log_aod")
    log_aod.attrs["units"] = "1"
    valid = valid.rename("valid")
    valid.attrs.update(long_name="Valid preprocessed AOD", units="1")

    output_fields: dict[str, xr.DataArray] = {
        "aod": processed_aod,
        "log_aod": log_aod,
        "valid": valid,
    }
    if "obs_error_std" in fields:
        source_error = fields.pop("obs_error_std").transpose("time", "lat", "lon").where(valid)
        if uncertainty_model is not None:
            linear_variance = (
                source_error**2
                + float(uncertainty_model.absolute_floor) ** 2
                + (float(uncertainty_model.relative_fraction) * processed_aod) ** 2
            )
            transformed_error = cast(xr.DataArray, np.sqrt(linear_variance)) / (
                processed_aod + float(spec.log_epsilon)
            )
            error = transformed_error.rename("obs_error_std")
        else:
            error = source_error.rename("obs_error_std")
        missing_error = valid & (~cast(xr.DataArray, np.isfinite(error)) | (error <= 0.0))
        if _any(missing_error):
            raise ValueError("obs_error_std must be finite and positive at every valid AOD cell")
        error.attrs.update(
            units="1",
            long_name="Observation error standard deviation in shifted-log space",
            space="shifted_log",
        )
        if uncertainty_model is not None:
            error.attrs.update(
                uncertainty_contract=uncertainty_model.name,
                uncertainty_model=uncertainty_model.type,
                uncertainty_source_variable=uncertainty_model.source_variable,
                uncertainty_combination=uncertainty_model.combination,
                uncertainty_transform=uncertainty_model.transform,
                uncertainty_covariance=uncertainty_model.covariance,
                uncertainty_absolute_floor=float(uncertainty_model.absolute_floor),
                uncertainty_relative_fraction=float(uncertainty_model.relative_fraction),
            )
        output_fields["obs_error_std"] = error

    common_factors = _combine_common_factors(fields, valid)
    if common_factors is not None:
        output_fields["common_error_factor"] = common_factors

    output = xr.Dataset(output_fields)
    output.attrs.update(
        {
            "analysis_type": "aod_preprocess",
            "source": spec.source,
            "source_variable": spec.variable,
            "preprocessing_order": "screen,sample_local_time,regrid,shifted_log",
            "log_epsilon": float(spec.log_epsilon),
            "day_anchor_hour": float(spec.day_anchor_hour),
        }
    )
    for attr_name in ("scenario", "schema_version", "root_seed", "spec_hash"):
        if attr_name in source.attrs:
            output.attrs[f"source_{attr_name}"] = source.attrs[attr_name]
    if spec.sample_local_time is not None:
        output.attrs["sample_local_time"] = float(spec.sample_local_time)
    if spec.target_grid is not None:
        output.attrs["target_grid_degrees"] = float(spec.target_grid)
    elif spec.target_grid_from is not None:
        output.attrs["target_grid_from"] = spec.target_grid_from
    if uncertainty_model is not None:
        output.attrs.update(
            uncertainty_contract=uncertainty_model.name,
            uncertainty_model=uncertainty_model.type,
            uncertainty_source_variable=uncertainty_model.source_variable,
            uncertainty_combination=uncertainty_model.combination,
            uncertainty_transform=uncertainty_model.transform,
            uncertainty_covariance=uncertainty_model.covariance,
            uncertainty_absolute_floor=float(uncertainty_model.absolute_floor),
            uncertainty_relative_fraction=float(uncertainty_model.relative_fraction),
        )
    return output


def _source_field(source: xr.Dataset, variable: str, valid: xr.DataArray) -> xr.DataArray:
    if variable not in source:
        raise ValueError(f"AOD source is missing variable {variable!r}")
    field = _canonical_horizontal(source[variable])
    try:
        field, aligned_valid = xr.align(field, valid, join="exact")
    except ValueError as exc:
        raise ValueError(f"field {variable!r} is not aligned with AOD") from exc
    return field.where(aligned_valid)


def _canonical_horizontal(data: xr.DataArray) -> xr.DataArray:
    output = data
    for desired, candidates in (("lat", LATITUDE_NAMES), ("lon", LONGITUDE_NAMES)):
        coord = _named_coordinate(output, candidates, desired)
        if coord.ndim != 1 or not coord.dims:
            raise ValueError(f"AOD preprocessing requires a one-dimensional {desired} coordinate")
        source_dim = str(coord.dims[0])
        if source_dim not in output.dims:
            raise ValueError(f"{desired} coordinate does not index the AOD field")
        if source_dim != desired:
            if desired in output.coords and output[desired].dims == (source_dim,):
                output = output.swap_dims({source_dim: desired})
            else:
                values = coord.values
                attrs = dict(coord.attrs)
                output = output.rename({source_dim: desired})
                output = output.assign_coords({desired: (desired, values, attrs)})
        elif str(coord.name) != desired:
            output = output.assign_coords({desired: (desired, coord.values, dict(coord.attrs))})
    if "time" not in output.dims or "time" not in output.coords:
        raise ValueError("AOD preprocessing requires a time coordinate")
    return output


def _named_coordinate(
    data: xr.DataArray | xr.Dataset, candidates: Sequence[str], kind: str
) -> xr.DataArray:
    for name in candidates:
        if name in data.coords:
            return data.coords[name]
    raise ValueError(f"AOD preprocessing requires a {kind} coordinate")


def _validate_aod_dimensions(aod: xr.DataArray) -> None:
    extras = set(aod.dims) - {"time", "lat", "lon"}
    if extras:
        raise ValueError(f"AOD preprocessing requires a 3-D time/lat/lon field, got {aod.dims}")


def _runtime_or_source_window(
    runtime: AnalysisRuntime, aod: xr.DataArray
) -> tuple[pd.Timestamp, pd.Timestamp]:
    time = pd.DatetimeIndex(aod["time"].values)
    if time.empty:
        raise ValueError("AOD source time coordinate is empty")
    start = pd.Timestamp(runtime.start_time) if runtime.start_time is not None else time.min()
    end = pd.Timestamp(runtime.end_time) if runtime.end_time is not None else time.max()
    return start, end


def _anchor_calendar_days(data: xr.DataArray, anchor_hour: float) -> xr.DataArray:
    time = pd.DatetimeIndex(data["time"].values)
    if time.hasnans:
        raise ValueError("AOD source time contains NaT")
    anchored = time.normalize() + pd.to_timedelta(anchor_hour, unit="h")
    if anchored.duplicated().any():
        raise ValueError("AOD source contains multiple samples for one calendar day")
    return data.assign_coords(time=anchored.values).sortby("time")


def _clip_requested_days(
    data: xr.DataArray, runtime: AnalysisRuntime, anchor_hour: float
) -> xr.DataArray:
    if runtime.start_time is None and runtime.end_time is None:
        return data
    time = pd.DatetimeIndex(data["time"].values)
    if time.empty:
        return data
    anchor = pd.to_timedelta(anchor_hour, unit="h")
    start = (
        pd.Timestamp(runtime.start_time).normalize() + anchor
        if runtime.start_time is not None
        else time.min()
    )
    end = (
        pd.Timestamp(runtime.end_time).normalize() + anchor
        if runtime.end_time is not None
        else time.max()
    )
    return data.sel(time=slice(start, end))


def _target_coordinates(
    target: xr.Dataset | None,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    if target is None:
        return None, None
    return (
        _named_coordinate(target, LATITUDE_NAMES, "latitude"),
        _named_coordinate(target, LONGITUDE_NAMES, "longitude"),
    )


def _same_grid(source: xr.DataArray, target_lat: xr.DataArray, target_lon: xr.DataArray) -> bool:
    source_lat = np.asarray(source["lat"].values, dtype=np.float64)
    source_lon = wrap_longitudes(source["lon"].values)
    wanted_lat = np.asarray(target_lat.values, dtype=np.float64)
    wanted_lon = wrap_longitudes(target_lon.values)
    return np.array_equal(source_lat, wanted_lat) and np.array_equal(source_lon, wanted_lon)


def _combine_common_factors(
    fields: Mapping[str, xr.DataArray], valid: xr.DataArray
) -> xr.DataArray | None:
    factors: list[xr.DataArray] = []
    labels: set[str] = set()
    for key, field in fields.items():
        if not key.startswith("factor:"):
            continue
        configured_name = key.removeprefix("factor:")
        factor = field.where(valid)
        missing = valid & ~cast(xr.DataArray, np.isfinite(factor))
        if _any(missing):
            raise ValueError(
                f"common factor {configured_name!r} must be finite at every valid AOD cell"
            )
        if "common_mode" in factor.dims:
            mode_values = [str(value) for value in factor["common_mode"].values]
            factor = factor.assign_coords(common_mode=mode_values)
        else:
            mode_values = [configured_name]
            factor = factor.expand_dims(common_mode=mode_values)
        overlap = labels.intersection(mode_values)
        if overlap:
            raise ValueError(f"duplicate common_mode labels: {sorted(overlap)}")
        labels.update(mode_values)
        factors.append(factor)
    if not factors:
        return None
    combined = xr.concat(factors, dim="common_mode").rename("common_error_factor")
    combined = combined.transpose("time", "common_mode", "lat", "lon")
    combined.attrs.update(
        units="1",
        long_name="Low-rank common observation-error factor in shifted-log space",
        space="shifted_log",
    )
    return combined


def _any(condition: xr.DataArray) -> bool:
    value = condition.any()
    if value.chunks is not None:
        value = value.compute()
    return bool(value.item())


@analysis_registry.register("aod_preprocess")
class AODPreprocessAnalysis(DerivedAnalysis):
    """Pipeline adapter for the shared AOD preprocessing contract."""

    name = "aod_preprocess"
    long_name = "AOD Preprocessing"
    output_geometry = DataGeometry.GRID

    def analyze_inputs(
        self,
        inputs: Mapping[str, xr.Dataset],
        spec: AODPreprocessSpec,
        runtime: AnalysisRuntime,
    ) -> xr.Dataset:
        try:
            source = inputs["source"]
        except KeyError as exc:
            raise ValueError("aod_preprocess requires a named 'source' input") from exc
        target = inputs.get("target_grid_from")
        if spec.target_grid_from is not None and target is None:
            raise ValueError("aod_preprocess target_grid_from input was not resolved")
        return preprocess_aod(source, spec, runtime, target_grid=target)
