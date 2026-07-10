"""Low-level, metadata-preserving NetCDF mutation for the FABLE MMR writer."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VariableContract:
    """Storage metadata that must survive correction unchanged."""

    dimensions: tuple[str, ...]
    dtype: str
    attributes: Mapping[str, Any]
    chunking: str | tuple[int, ...]
    filters: Mapping[str, Any] | None


@dataclass(frozen=True)
class FileContract:
    """Preflight metadata for one native MMR input."""

    global_attributes: Mapping[str, Any]
    variables: Mapping[str, VariableContract]


def sha256_file(path: str | Path) -> str:
    """Return the byte-level SHA-256 checksum of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_mmr_file(path: str | Path, species: Sequence[str]) -> FileContract:
    """Validate all configured aerosol variables and snapshot storage metadata."""
    source = Path(path)
    with netCDF4.Dataset(source, mode="r") as dataset:
        missing = [name for name in species if name not in dataset.variables]
        if missing:
            raise ValueError(
                f"{source} is missing configured aerosol species: {', '.join(missing)}"
            )
        contracts: dict[str, VariableContract] = {}
        for name, variable in dataset.variables.items():
            chunking = variable.chunking()
            normalized_chunking: str | tuple[int, ...]
            if isinstance(chunking, str):
                normalized_chunking = chunking
            else:
                normalized_chunking = tuple(int(value) for value in chunking)
            filters = variable.filters()
            contracts[str(name)] = VariableContract(
                dimensions=tuple(str(dim) for dim in variable.dimensions),
                dtype=np.dtype(variable.dtype).str,
                attributes=_attributes(variable),
                chunking=normalized_chunking,
                filters=dict(filters) if filters is not None else None,
            )
        for name in species:
            variable = dataset.variables[name]
            dimensions = set(variable.dimensions)
            missing_dimensions = {"time", "lat", "lon"} - dimensions
            if missing_dimensions:
                raise ValueError(
                    f"{source}:{name} is missing dimensions "
                    f"{', '.join(sorted(missing_dimensions))}"
                )
            if not np.issubdtype(variable.dtype, np.floating):
                raise TypeError(f"{source}:{name} must use a floating-point dtype")
        return FileContract(
            global_attributes=_attributes(dataset),
            variables=contracts,
        )


def read_writer_hashes(path: str | Path) -> dict[str, str]:
    """Read embedded writer provenance hashes from an existing result."""
    names = {
        "input": "davinci_input_sha256",
        "config": "davinci_config_sha256",
        "scaling": "davinci_scaling_sha256",
        "code": "davinci_code_sha256",
    }
    with netCDF4.Dataset(path, mode="r") as dataset:
        return {
            key: str(dataset.getncattr(attribute))
            for key, attribute in names.items()
            if attribute in dataset.ncattrs()
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _update_payload(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "little", signed=False))
    digest.update(payload)


def _normalized_payload(values: NDArray[Any]) -> bytes:
    array = np.asarray(values)
    if array.dtype.kind in "UOS":
        return json.dumps(
            array.astype(str).reshape(-1).tolist(), ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    normalized = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    if normalized.dtype.kind in "fc" and np.isnan(normalized).any():
        normalized = normalized.copy()
        normalized[np.isnan(normalized)] = np.nan
    return normalized.tobytes()


def mmr_payload_sha256(path: str | Path) -> str:
    """Hash all native dimensions, metadata, and raw variable values in bounded slices."""
    digest = hashlib.sha256()
    with netCDF4.Dataset(path, mode="r") as dataset:
        attrs = _attributes(dataset)
        attrs.pop("davinci_payload_sha256", None)
        metadata = {
            "dimensions": {
                str(name): {"size": len(dim), "unlimited": bool(dim.isunlimited())}
                for name, dim in dataset.dimensions.items()
            },
            "attrs": _json_value(attrs),
        }
        _update_payload(
            digest,
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        for name in sorted(dataset.variables):
            variable = dataset.variables[name]
            variable.set_auto_maskandscale(False)
            _update_payload(
                digest,
                json.dumps(
                    {
                        "name": name,
                        "dimensions": list(variable.dimensions),
                        "dtype": np.dtype(variable.dtype).str,
                        "shape": list(variable.shape),
                        "attrs": _json_value(_attributes(variable)),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            if variable.ndim == 0:
                _update_payload(digest, _normalized_payload(np.asarray(variable[...])))
                continue
            remaining = int(np.prod(variable.shape[1:], dtype=np.int64))
            batch = max(1, 1_000_000 // max(1, remaining))
            for start in range(0, variable.shape[0], batch):
                _update_payload(
                    digest,
                    _normalized_payload(np.asarray(variable[start : start + batch])),
                )
    return digest.hexdigest()


def validate_resumable_mmr_file(
    path: str | Path,
    species: Sequence[str],
    contract: FileContract,
    provenance: Mapping[str, str],
) -> str:
    """Validate storage/provenance and return a verified embedded payload digest."""
    source = Path(path)
    try:
        _validate_temporary(source, species, contract, provenance)
        with netCDF4.Dataset(source, mode="r") as dataset:
            stored = str(dataset.getncattr("davinci_payload_sha256"))
    except (OSError, RuntimeError, AttributeError) as exc:
        raise ValueError(f"existing MMR output is not resumable: {source}") from exc
    actual = mmr_payload_sha256(source)
    if not hmac.compare_digest(stored, actual):
        raise ValueError(f"existing MMR output payload checksum does not match: {source}")
    return actual


def atomic_scale_mmr_file(
    input_path: str | Path,
    output_path: str | Path,
    species: Sequence[str],
    ratio: NDArray[np.float64],
    *,
    contract: FileContract,
    provenance: Mapping[str, str],
) -> tuple[str, str]:
    """Copy, mutate, fsync, reopen-validate, and atomically replace one file."""
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        shutil.copy2(source, temporary)
        expected_input_hash = provenance.get("input_sha256")
        if expected_input_hash is not None and sha256_file(temporary) != expected_input_hash:
            raise RuntimeError(f"input changed after MMR writer preflight: {source}")
        _mutate_aerosols(temporary, species, ratio, provenance)
        _fsync_file(temporary)
        _validate_temporary(temporary, species, contract, provenance)
        payload_sha256 = mmr_payload_sha256(temporary)
        with netCDF4.Dataset(temporary, mode="r+") as dataset:
            dataset.setncattr("davinci_payload_sha256", payload_sha256)
            dataset.sync()
        _fsync_file(temporary)
        _validate_temporary(temporary, species, contract, provenance)
        if mmr_payload_sha256(temporary) != payload_sha256:
            raise RuntimeError("atomic validation failed: MMR payload checksum changed")
        output_sha256 = sha256_file(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return output_sha256, payload_sha256
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _mutate_aerosols(
    path: Path,
    species: Sequence[str],
    ratio: NDArray[np.float64],
    provenance: Mapping[str, str],
) -> None:
    with netCDF4.Dataset(path, mode="r+") as dataset:
        for key, value in provenance.items():
            dataset.setncattr(f"davinci_{key}", str(value))
        for name in species:
            variable = dataset.variables[name]
            variable.set_auto_maskandscale(False)
            time_axis = variable.dimensions.index("time")
            latitude_axis = variable.dimensions.index("lat")
            longitude_axis = variable.dimensions.index("lon")
            for time_index in range(variable.shape[time_axis]):
                index: list[int | slice] = [slice(None)] * variable.ndim
                index[time_axis] = time_index
                raw = np.asarray(variable[tuple(index)])
                scale = _reshape_ratio(
                    ratio[time_index],
                    variable.dimensions,
                    time_axis,
                    latitude_axis,
                    longitude_axis,
                )
                valid = _valid_data_mask(raw, variable)
                corrected = raw.copy()
                np.multiply(raw, scale, out=corrected, where=valid, casting="unsafe")
                variable[tuple(index)] = corrected
        dataset.sync()


def _reshape_ratio(
    ratio: NDArray[np.float64],
    dimensions: tuple[str, ...],
    time_axis: int,
    latitude_axis: int,
    longitude_axis: int,
) -> NDArray[np.float64]:
    remaining = [dim for axis, dim in enumerate(dimensions) if axis != time_axis]
    shape = [1] * len(remaining)
    shape[remaining.index(dimensions[latitude_axis])] = ratio.shape[0]
    shape[remaining.index(dimensions[longitude_axis])] = ratio.shape[1]
    return ratio.reshape(shape)


def _valid_data_mask(raw: NDArray[Any], variable: netCDF4.Variable) -> NDArray[np.bool_]:
    valid = np.isfinite(raw)
    for attribute in ("_FillValue", "missing_value"):
        if attribute not in variable.ncattrs():
            continue
        for missing in np.asarray(variable.getncattr(attribute)).reshape(-1):
            if np.issubdtype(np.asarray(missing).dtype, np.floating) and np.isnan(missing):
                valid &= ~np.isnan(raw)
            else:
                valid &= raw != missing
    return np.asarray(valid, dtype=np.bool_)


def _validate_temporary(
    path: Path,
    species: Sequence[str],
    contract: FileContract,
    provenance: Mapping[str, str],
) -> None:
    with netCDF4.Dataset(path, mode="r") as dataset:
        for name in species:
            if name not in dataset.variables:
                raise RuntimeError(f"atomic validation failed: missing {name!r}")
        for name, expected in contract.variables.items():
            if name not in dataset.variables:
                raise RuntimeError(f"atomic validation failed: missing variable {name!r}")
            actual = dataset.variables[name]
            chunking = actual.chunking()
            normalized_chunking: str | tuple[int, ...]
            if isinstance(chunking, str):
                normalized_chunking = chunking
            else:
                normalized_chunking = tuple(int(value) for value in chunking)
            filters = actual.filters()
            normalized_filters = dict(filters) if filters is not None else None
            if (
                tuple(actual.dimensions) != expected.dimensions
                or np.dtype(actual.dtype).str != expected.dtype
                or normalized_chunking != expected.chunking
                or normalized_filters != expected.filters
                or not _attributes_equal(_attributes(actual), expected.attributes)
            ):
                raise RuntimeError(
                    f"atomic validation failed: storage metadata changed for {name!r}"
                )
        actual_global = _attributes(dataset)
        if not all(
            key in actual_global and _values_equal(actual_global[key], value)
            for key, value in contract.global_attributes.items()
        ):
            raise RuntimeError("atomic validation failed: original global attributes changed")
        for key, value in provenance.items():
            attribute = f"davinci_{key}"
            if attribute not in actual_global or str(actual_global[attribute]) != str(value):
                raise RuntimeError(f"atomic validation failed: missing provenance {attribute!r}")


def _attributes(owner: Any) -> dict[str, Any]:
    return {str(name): owner.getncattr(name) for name in owner.ncattrs()}


def _attributes_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return actual.keys() == expected.keys() and all(
        _values_equal(actual[key], value) for key, value in expected.items()
    )


def _values_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind in "fc" and right_array.dtype.kind in "fc":
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FileContract",
    "atomic_scale_mmr_file",
    "inspect_mmr_file",
    "mmr_payload_sha256",
    "read_writer_hashes",
    "sha256_file",
    "validate_resumable_mmr_file",
]
