"""Input normalization and metadata validation for EOF projection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class ProjectionObservation:
    """One validated sensor aligned to the complete model time axis."""

    name: str
    values: xr.DataArray
    errors: xr.DataArray
    valid: xr.DataArray
    factors: xr.DataArray
    factor_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedProjectionInputs:
    """Validated arrays and identities consumed by projection orchestration."""

    patterns: xr.DataArray
    model: xr.DataArray
    time: pd.DatetimeIndex
    epsilon: float
    basis_signature: str
    grid_signature: str
    observations: tuple[ProjectionObservation, ...]


def field_value(entry: Any, name: str, default: Any = None) -> Any:
    """Read one field from a Pydantic model or mapping."""
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def same_coordinate(left: xr.DataArray, right: xr.DataArray, name: str) -> None:
    """Require exact dimensions and decoded coordinate values."""
    if left.dims != right.dims or not np.array_equal(left.values, right.values):
        raise ValueError(f"projection {name} coordinates do not match exactly")


def _grid_field(dataset: xr.Dataset, variable: str, *, time: bool) -> xr.DataArray:
    if variable not in dataset:
        raise ValueError(f"projection input is missing variable {variable!r}")
    required = ("time", "lat", "lon") if time else ("mode", "lat", "lon")
    field = dataset[variable]
    if set(field.dims) != set(required):
        raise ValueError(f"{variable!r} must have dimensions {required}, got {field.dims}")
    return field.transpose(*required)


def _validate_daily_time(time: xr.DataArray) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(time.values)
    except (TypeError, ValueError) as exc:
        raise ValueError("projection model time must be datetime-like") from exc
    if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("projection model time must be nonempty, unique, and increasing")
    if len(index) > 1 and not np.all((index[1:] - index[:-1]) == pd.Timedelta(days=1)):
        raise ValueError("projection model time must be a complete regular daily axis")
    return index


def _false_metadata(value: Any) -> bool:
    return value is False or str(value).strip().lower() in {"false", "0", "no"}


def _log_epsilon(dataset: xr.Dataset, key: str, label: str) -> float:
    if key not in dataset.attrs:
        raise ValueError(f"{label} is missing required {key!r} metadata")
    value = float(dataset.attrs[key])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} has invalid {key!r} metadata")
    return value


def _require_unitless(field: xr.DataArray, label: str) -> None:
    units = str(field.attrs.get("units", "")).strip().lower()
    if units not in {"1", "dimensionless"}:
        raise ValueError(f"{label} must declare dimensionless units")


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_basis_and_model(
    basis: xr.Dataset, model: xr.Dataset, model_variable: str
) -> tuple[xr.DataArray, xr.DataArray, pd.DatetimeIndex, float, str, str]:
    patterns = _grid_field(basis, "eofs", time=False)
    model_field = _grid_field(model, model_variable, time=True)
    same_coordinate(patterns["lat"], model_field["lat"], "latitude")
    same_coordinate(patterns["lon"], model_field["lon"], "longitude")
    if np.any(~np.isfinite(patterns.values)):
        raise ValueError("projection basis patterns must be finite")
    if basis.attrs.get("eof_rotation") != "none":
        raise ValueError("projection basis requires eof_rotation='none'")
    if not _false_metadata(basis.attrs.get("eof_standardize")):
        raise ValueError("projection basis requires eof_standardize='false'")
    epsilon = _log_epsilon(basis, "eof_input_log_epsilon", "projection basis")
    model_epsilon = _log_epsilon(model, "log_epsilon", "projection model")
    if not np.isclose(epsilon, model_epsilon, rtol=0.0, atol=1.0e-15):
        raise ValueError("projection basis and model log_epsilon metadata differ")
    time = _validate_daily_time(model_field["time"])
    basis_signature = _digest_arrays(
        np.asarray(patterns["mode"].values),
        np.asarray(patterns["lat"].values),
        np.asarray(patterns["lon"].values),
        # Product artifacts persist floating fields as float32; normalizing here
        # keeps the identity stable before and after an artifact round trip.
        np.asarray(patterns.values, dtype=np.float32),
    )
    grid_signature = _digest_arrays(
        np.asarray(model_field["lat"].values), np.asarray(model_field["lon"].values)
    )
    return patterns, model_field, time, epsilon, basis_signature, grid_signature


def _factor_variables(entry: Any, dataset: xr.Dataset) -> list[str]:
    configured = field_value(entry, "common_factor_variables", None)
    if configured:
        return [str(name) for name in configured]
    single = field_value(entry, "common_factor_variable", None)
    if single:
        return [str(single)]
    return ["common_error_factor"] if "common_error_factor" in dataset else []


def _common_factors(
    dataset: xr.Dataset, entry: Any, model_field: xr.DataArray
) -> tuple[xr.DataArray, tuple[str, ...]]:
    fields: list[xr.DataArray] = []
    names: list[str] = []
    for variable in _factor_variables(entry, dataset):
        if variable not in dataset:
            raise ValueError(f"projection input is missing common factor {variable!r}")
        factor = dataset[variable]
        if set(factor.dims) == {"time", "lat", "lon"}:
            factor = factor.expand_dims(common_mode=[variable])
        if set(factor.dims) != {"time", "common_mode", "lat", "lon"}:
            raise ValueError(
                f"common factor {variable!r} must have time/common_mode/lat/lon dimensions"
            )
        if factor.attrs.get("space") != "shifted_log":
            raise ValueError(f"common factor {variable!r} must declare space='shifted_log'")
        _require_unitless(factor, f"common factor {variable!r}")
        factor = factor.transpose("time", "common_mode", "lat", "lon")
        same_coordinate(factor["lat"], model_field["lat"], "factor latitude")
        same_coordinate(factor["lon"], model_field["lon"], "factor longitude")
        factor = factor.reindex(time=model_field["time"])
        labels = [str(value) for value in factor["common_mode"].values]
        overlap = set(names).intersection(labels)
        if overlap:
            raise ValueError(f"duplicate common covariance modes: {sorted(overlap)}")
        names.extend(labels)
        fields.append(factor)
    if not fields:
        return (
            xr.DataArray(
                np.empty(
                    (
                        model_field.sizes["time"],
                        0,
                        model_field.sizes["lat"],
                        model_field.sizes["lon"],
                    ),
                    dtype=np.float64,
                ),
                dims=("time", "common_mode", "lat", "lon"),
                coords={
                    "time": model_field["time"],
                    "common_mode": [],
                    "lat": model_field["lat"],
                    "lon": model_field["lon"],
                },
            ),
            (),
        )
    combined = xr.concat(fields, dim="common_mode")
    return combined, tuple(names)


def _load_observation(
    dataset: xr.Dataset,
    entry: Any,
    model_field: xr.DataArray,
    model_time: pd.DatetimeIndex,
    epsilon: float,
) -> ProjectionObservation:
    name = str(field_value(entry, "source"))
    variable = str(field_value(entry, "variable"))
    error_variable = str(field_value(entry, "error_variable"))
    values = _grid_field(dataset, variable, time=True)
    errors = _grid_field(dataset, error_variable, time=True)
    same_coordinate(values["lat"], model_field["lat"], f"{name} latitude")
    same_coordinate(values["lon"], model_field["lon"], f"{name} longitude")
    same_coordinate(errors["lat"], model_field["lat"], f"{name} error latitude")
    same_coordinate(errors["lon"], model_field["lon"], f"{name} error longitude")
    obs_epsilon = _log_epsilon(dataset, "log_epsilon", f"projection observation {name!r}")
    if not np.isclose(epsilon, obs_epsilon, rtol=0.0, atol=1.0e-15):
        raise ValueError(f"projection observation {name!r} has a different log_epsilon")
    if errors.attrs.get("space") != "shifted_log":
        raise ValueError(f"projection error {error_variable!r} must declare space='shifted_log'")
    _require_unitless(errors, f"projection error {error_variable!r}")
    obs_time = pd.DatetimeIndex(values["time"].values)
    if obs_time.has_duplicates:
        raise ValueError(f"projection observation {name!r} has duplicate times")
    if len(model_time.intersection(obs_time)) == 0:
        raise ValueError(f"projection observation {name!r} has no model-time overlap")

    values = values.reindex(time=model_field["time"])
    errors = errors.reindex(time=model_field["time"])
    valid = xr.apply_ufunc(np.isfinite, values, dask="allowed")
    if "valid" in dataset:
        declared = _grid_field(dataset, "valid", time=True)
        same_coordinate(declared["lat"], model_field["lat"], f"{name} validity latitude")
        same_coordinate(declared["lon"], model_field["lon"], f"{name} validity longitude")
        valid &= declared.reindex(time=model_field["time"]).fillna(False).astype(bool)
    factors, factor_names = _common_factors(dataset, entry, model_field)
    return ProjectionObservation(name, values, errors, valid, factors, factor_names)


def _align_factor_modes(
    observations: Sequence[ProjectionObservation],
) -> tuple[ProjectionObservation, ...]:
    reference = observations[0].factor_names
    aligned: list[ProjectionObservation] = []
    for observation in observations:
        if set(observation.factor_names) != set(reference):
            raise ValueError("all projection sensors must provide the same common covariance modes")
        if observation.factor_names == reference:
            aligned.append(observation)
            continue
        order = [observation.factor_names.index(name) for name in reference]
        aligned.append(
            ProjectionObservation(
                observation.name,
                observation.values,
                observation.errors,
                observation.valid,
                observation.factors.isel(common_mode=order),
                reference,
            )
        )
    return tuple(aligned)


def prepare_projection_inputs(
    basis: xr.Dataset,
    model: xr.Dataset,
    observation_inputs: Sequence[tuple[Any, xr.Dataset]],
    model_variable: str,
) -> PreparedProjectionInputs:
    """Validate and align every projection input without fitting or solving."""
    patterns, model_field, time, epsilon, basis_signature, grid_signature = (
        _validate_basis_and_model(basis, model, model_variable)
    )
    if not observation_inputs:
        raise ValueError("eof_projection requires at least one observation input")
    observations = _align_factor_modes(
        tuple(
            _load_observation(dataset, entry, model_field, time, epsilon)
            for entry, dataset in observation_inputs
        )
    )
    names = [observation.name for observation in observations]
    if len(names) != len(set(names)):
        raise ValueError("eof_projection sensor source names must be unique")
    return PreparedProjectionInputs(
        patterns,
        model_field,
        time,
        epsilon,
        basis_signature,
        grid_signature,
        observations,
    )


__all__ = [
    "PreparedProjectionInputs",
    "ProjectionObservation",
    "field_value",
    "prepare_projection_inputs",
    "same_coordinate",
]
