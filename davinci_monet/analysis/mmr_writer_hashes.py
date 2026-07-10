"""Deterministic scientific/configuration hashes for native MMR outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def implementation_hash() -> str:
    """Hash writer, storage, hashing, and interpolation implementation sources."""
    digest = hashlib.sha256()
    root = Path(__file__).parents[1]
    for path in (
        Path(__file__).with_name("mmr_writer.py"),
        Path(__file__).with_name("mmr_writer_scaling.py"),
        Path(__file__).with_name("mmr_writer_io.py"),
        Path(__file__),
        root / "util" / "regrid.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_scaling_dataset(
    ratio: xr.DataArray, support: xr.DataArray, attrs: Mapping[Any, Any]
) -> str:
    """Hash a scaling artifact in bounded time batches."""
    digest = hashlib.sha256()
    digest.update(_canonical_json({"attrs": attrs}).encode("utf-8"))
    for name, data in (("r", ratio), ("spatial_support", support)):
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json({"dims": data.dims, "attrs": data.attrs}).encode("utf-8"))
        for start in range(0, data.sizes["time"], 31):
            values = np.asarray(data.isel(time=slice(start, start + 31)).values)
            digest.update(_normalized_bytes(values))
    for name in ("time", "lat", "lon"):
        digest.update(name.encode("ascii"))
        digest.update(_normalized_bytes(np.asarray(ratio[name].values)))
    return digest.hexdigest()


def hash_application_config(species: Sequence[str], time_interp: str, outside_coverage: str) -> str:
    """Hash only controls that affect corrected scientific values."""
    payload = {
        "species": list(species),
        "time_interp": time_interp,
        "outside_coverage": outside_coverage,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalized_bytes(values: NDArray[Any]) -> bytes:
    array = np.asarray(values)
    if array.dtype.kind in "mM":
        return np.ascontiguousarray(array.astype("datetime64[ns]").view("<i8")).tobytes()
    normalized = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    if normalized.dtype.kind in "fc" and np.isnan(normalized).any():
        normalized = normalized.copy()
        normalized[np.isnan(normalized)] = np.nan
    return normalized.tobytes()


__all__ = ["hash_application_config", "hash_scaling_dataset", "implementation_hash"]
