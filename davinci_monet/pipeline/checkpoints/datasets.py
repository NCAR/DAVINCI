"""Scientific hashing and durable NetCDF collections for checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

_HASH_BATCH_ELEMENTS = 1_000_000
_NETCDF_PACKING_ATTRS = frozenset(
    {
        "_FillValue",
        "add_offset",
        "missing_value",
        "scale_factor",
    }
)


@dataclass(frozen=True)
class DatasetCollection:
    """One finalized collection of checkpoint NetCDF chunks."""

    paths: tuple[Path, ...]
    summary_path: Path
    checksums: dict[str, str]
    summary_checksum: str


def _identity_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _update_digest(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "little", signed=False))
    digest.update(payload)


def _normalized_array_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values)
    if array.dtype.kind in "UOS":
        payload = [_identity_value(value) for value in array.astype(str).reshape(-1)]
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if array.dtype.kind == "M":
        array = array.astype("datetime64[ns]").view("<i8")
    elif array.dtype.kind == "m":
        array = array.astype("timedelta64[ns]").view("<i8")
    else:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    normalized = np.ascontiguousarray(array)
    if normalized.dtype.kind in "fc" and np.isnan(normalized).any():
        normalized = normalized.copy()
        normalized[np.isnan(normalized)] = np.nan
    return normalized.tobytes()


def _update_variable_values(digest: Any, variable: xr.DataArray) -> None:
    if variable.ndim == 0:
        _update_digest(digest, _normalized_array_bytes(np.asarray(variable.values)))
        return
    remaining = int(np.prod(variable.shape[1:], dtype=np.int64))
    batch = max(1, _HASH_BATCH_ELEMENTS // max(1, remaining))
    leading = variable.dims[0]
    for start in range(0, variable.shape[0], batch):
        values = variable.isel({leading: slice(start, start + batch)}).values
        _update_digest(digest, _normalized_array_bytes(np.asarray(values)))


def scientific_dataset_sha256(dataset: xr.Dataset) -> str:
    """Hash decoded coordinates, values, dimensions, and attributes."""
    digest = hashlib.sha256()
    _update_digest(
        digest,
        json.dumps(
            {
                "sizes": {str(name): int(size) for name, size in sorted(dataset.sizes.items())},
                "attrs": {str(key): _identity_value(value) for key, value in dataset.attrs.items()},
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
    )
    for raw_name in sorted(dataset.variables, key=str):
        name = str(raw_name)
        variable = dataset[raw_name]
        _update_digest(
            digest,
            json.dumps(
                {
                    "name": name,
                    "coordinate": name in dataset.coords,
                    "dims": list(variable.dims),
                    "shape": list(variable.shape),
                    "dtype": str(variable.dtype),
                    "attrs": {
                        str(key): _identity_value(value) for key, value in variable.attrs.items()
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8"),
        )
        _update_variable_values(digest, variable)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _netcdf_safe_attrs(
    attrs: Mapping[Any, Any],
    *,
    discard_packing: bool = False,
) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attrs.items():
        if discard_packing and str(key) in _NETCDF_PACKING_ATTRS:
            continue
        safe[str(key)] = str(value) if isinstance(value, bool) else value
    return safe


def _netcdf_safe_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Remove decoded HDF/NetCDF packing state before writing checkpoints."""
    writable = dataset.copy(deep=False)
    writable.attrs = _netcdf_safe_attrs(dataset.attrs)
    for name in writable.data_vars:
        writable[name].attrs = _netcdf_safe_attrs(
            writable[name].attrs,
            discard_packing=True,
        )
        writable[name].encoding = {}
    for name in writable.coords:
        writable.coords[name].attrs = _netcdf_safe_attrs(
            writable.coords[name].attrs,
            discard_packing=True,
        )
        writable.coords[name].encoding = {}
    return writable


def _collection_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for raw_name, variable in dataset.data_vars.items():
        if variable.ndim == 0 or variable.dtype.kind not in "fiu":
            continue
        encoding[str(raw_name)] = {
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
            "chunksizes": tuple(max(1, min(int(size), 64)) for size in variable.shape),
        }
    return encoding


def _reuse_collection(
    destination: Path,
    identity: Mapping[str, Any],
    time_chunk_size: int,
) -> DatasetCollection:
    summary_path = destination / "summary.json"
    try:
        summary = json.loads(summary_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(f"checkpoint collection receipt is invalid: {destination}") from exc
    if (
        not isinstance(summary, dict)
        or summary.get("identity") != dict(identity)
        or summary.get("time_chunk_size") != time_chunk_size
    ):
        raise FileExistsError(f"checkpoint collection identity changed: {destination}")
    receipt = summary.get("checksums")
    file_hashes = receipt.get("files") if isinstance(receipt, dict) else None
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise FileExistsError(f"checkpoint collection receipt is incomplete: {destination}")
    paths = tuple(destination / str(name) for name in sorted(file_hashes))
    actual_names = {path.name for path in destination.glob("chunk-*.nc")}
    checksums = {path.name: _file_sha256(path) for path in paths if path.is_file()}
    if actual_names != set(file_hashes) or checksums != file_hashes:
        raise FileExistsError(f"checkpoint collection checksum changed: {destination}")
    return DatasetCollection(
        paths=paths,
        summary_path=summary_path,
        checksums=checksums,
        summary_checksum=_file_sha256(summary_path),
    )


def write_dataset_collection(
    parent: Path,
    storage_name: str,
    dataset: xr.Dataset,
    *,
    time_chunk_size: int,
    identity: Mapping[str, Any],
) -> DatasetCollection:
    """Atomically publish a bounded NetCDF collection or reuse valid bytes."""
    if time_chunk_size < 1:
        raise ValueError("time_chunk_size must be positive")
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / storage_name
    if destination.exists():
        return _reuse_collection(destination, identity, time_chunk_size)

    staging = Path(tempfile.mkdtemp(prefix=f".{storage_name}-", dir=parent))
    filenames: list[str] = []
    try:
        timed_variables = [
            str(name) for name, data in dataset.data_vars.items() if "time" in data.dims
        ]
        static_variables = [
            str(name) for name in dataset.data_vars if str(name) not in timed_variables
        ]
        if "time" in dataset.dims and timed_variables:
            selections: list[xr.Dataset] = []
            timed = dataset[timed_variables]
            for start in range(0, dataset.sizes["time"], time_chunk_size):
                selection = timed.isel(
                    time=slice(start, min(start + time_chunk_size, dataset.sizes["time"]))
                )
                if start == 0 and static_variables:
                    selection = xr.merge(
                        (selection, dataset[static_variables]),
                        compat="no_conflicts",
                    )
                selections.append(selection)
        else:
            selections = [dataset]

        for index, selection in enumerate(selections):
            filename = f"chunk-{index:05d}.nc"
            path = staging / filename
            writable = _netcdf_safe_dataset(selection)
            writable.to_netcdf(
                path,
                engine="netcdf4",
                encoding=_collection_encoding(writable),
            )
            _fsync_path(path)
            filenames.append(filename)

        staged_paths = tuple(staging / filename for filename in filenames)
        checksums = {path.name: _file_sha256(path) for path in staged_paths}
        summary = {
            "identity": dict(identity),
            "dimensions": {str(name): int(size) for name, size in dataset.sizes.items()},
            "files": filenames,
            "time_chunk_size": time_chunk_size,
            "checksums": {"files": checksums},
        }
        summary_path = staging / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _fsync_path(summary_path)
        summary_checksum = _file_sha256(summary_path)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return DatasetCollection(
        paths=tuple(destination / filename for filename in filenames),
        summary_path=destination / "summary.json",
        checksums=checksums,
        summary_checksum=summary_checksum,
    )


def load_dataset_collection(paths: Iterable[str | Path]) -> xr.Dataset:
    """Lazily reopen an ordered checkpoint collection."""
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("checkpoint collection contains no files")
    if len(resolved) == 1:
        return xr.open_dataset(resolved[0], chunks="auto")
    return xr.open_mfdataset(
        resolved,
        combine="nested",
        concat_dim="time",
        chunks="auto",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    )
