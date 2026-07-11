"""Serialization adapter and reproducibility hashes for aerosol tuning bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Mapping
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.tests.synthetic._aerosol_bundle import generate_bundle
from davinci_monet.tests.synthetic._aerosol_contracts import (
    SyntheticTuningBundle,
    SyntheticTuningSpec,
    canonical_json,
)

NETCDF_CHUNK_TARGET_BYTES = 4 * 1024**2


def scientific_dataset_hash(dataset: xr.Dataset) -> str:
    """Hash decoded coordinates, arrays, dimensions, and attrs independent of NetCDF bytes."""
    digest = hashlib.sha256()
    digest.update(canonical_json({"sizes": dict(sorted(dataset.sizes.items()))}).encode("utf-8"))
    digest.update(canonical_json(_json_attrs(dataset.attrs)).encode("utf-8"))
    for raw_name in sorted(dataset.variables, key=str):
        variable = dataset[raw_name]
        digest.update(str(raw_name).encode("utf-8"))
        digest.update(canonical_json(list(variable.dims)).encode("utf-8"))
        digest.update(canonical_json(_json_attrs(variable.attrs)).encode("utf-8"))
        array = np.asarray(variable.values)
        if array.dtype.kind in "UOS":
            dtype_name = "string"
        elif array.dtype.kind in "mM":
            dtype_name = "datetime64[ns]"
        else:
            dtype_name = array.dtype.newbyteorder("<").str
        digest.update(canonical_json({"dtype": dtype_name, "shape": array.shape}).encode("utf-8"))
        if array.dtype.kind in "UOS":
            digest.update(canonical_json(array.astype(str).tolist()).encode("utf-8"))
        elif array.dtype.kind in "mM":
            digest.update(
                np.ascontiguousarray(array.astype("datetime64[ns]").view("<i8")).tobytes()
            )
        else:
            normalized = np.ascontiguousarray(
                array.astype(array.dtype.newbyteorder("<"), copy=False)
            )
            if normalized.dtype.kind in "fc":
                normalized = normalized.copy()
                normalized[np.isnan(normalized)] = np.nan
            digest.update(normalized.tobytes())
    return digest.hexdigest()


def _json_attrs(attrs: Mapping[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, np.ndarray):
            normalized[str(key)] = value.tolist()
        elif isinstance(value, np.generic):
            normalized[str(key)] = value.item()
        elif isinstance(value, tuple):
            normalized[str(key)] = list(value)
        else:
            normalized[str(key)] = value
    return normalized


def _bounded_netcdf_chunks(
    dimensions: tuple[Hashable, ...],
    shape: tuple[int, ...],
    itemsize: int,
    *,
    target_bytes: int = NETCDF_CHUNK_TARGET_BYTES,
) -> tuple[int, ...]:
    """Choose bounded chunks, preserving complete non-time planes when practical."""
    if len(dimensions) != len(shape):
        raise ValueError("chunk dimensions and shape must have equal lengths")
    if itemsize <= 0 or target_bytes <= 0:
        raise ValueError("chunk item size and byte target must be positive")

    chunks = [max(1, int(size)) for size in shape]
    temporal = [
        index
        for index, dimension in enumerate(dimensions)
        if dimension == "time" or (isinstance(dimension, str) and dimension.endswith("_time"))
    ]
    remaining = sorted(
        (index for index in range(len(chunks)) if index not in temporal),
        key=lambda index: chunks[index],
        reverse=True,
    )
    for index in (*temporal, *remaining):
        if itemsize * prod(chunks) <= target_bytes:
            break
        other_elements = prod(chunks[:index] + chunks[index + 1 :])
        chunks[index] = min(
            chunks[index],
            max(1, target_bytes // (itemsize * other_elements)),
        )
    return tuple(chunks)


def _write_dataset(path: Path, dataset: xr.Dataset) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding: dict[str, dict[str, Any]] = {}
    for raw_name, variable in dataset.data_vars.items():
        if variable.ndim and variable.dtype.kind in "fiu":
            settings: dict[str, Any] = {"zlib": True, "complevel": 1, "shuffle": True}
            settings["chunksizes"] = _bounded_netcdf_chunks(
                variable.dims,
                variable.shape,
                variable.dtype.itemsize,
            )
            if "_FillValue" in variable.encoding:
                settings["_FillValue"] = variable.encoding["_FillValue"]
            encoding[str(raw_name)] = settings
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return {
        "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scientific_sha256": scientific_dataset_hash(dataset),
    }


def write_aerosol_tuning_bundle(
    root: str | Path, bundle: SyntheticTuningBundle, *, overwrite: bool = False
) -> Path:
    """Serialize a bundle and deterministic scenario manifest below ``root``."""
    destination = Path(root)
    files: dict[str, dict[str, str]] = {}
    targets = {
        "inputs/model/MERRA2_SYNTH.tavg1_2d_aer_Nx.nc4": bundle.model,
        "inputs/obs/sensor_a.nc": bundle.observations["sensor_a"],
        "inputs/obs/sensor_b.nc": bundle.observations["sensor_b"],
        "oracle/truth.nc": bundle.truth,
    }
    for day, dataset in bundle.mmr.items():
        targets[f"inputs/mmr/MERRA2_SYNTH.inst3_3d_aer_Nv.{day.replace('-', '')}.nc4"] = dataset
    scenario_path = destination / "scenario.json"
    existing = [destination / relative for relative in targets if (destination / relative).exists()]
    if scenario_path.exists():
        existing.append(scenario_path)
    if existing and not overwrite:
        raise FileExistsError(f"synthetic bundle destination already exists: {existing[0]}")
    for relative_path, dataset in targets.items():
        hashes = _write_dataset(destination / relative_path, dataset)
        files[relative_path] = {**hashes, "role": str(dataset.attrs["role"])}
    manifest = dict(bundle.provenance)
    manifest["files"] = dict(sorted(files.items()))
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return scenario_path


def generate_aerosol_tuning_bundle(
    root: str | Path | SyntheticTuningSpec | None = None,
    spec: SyntheticTuningSpec | None = None,
) -> SyntheticTuningBundle:
    """Generate a coupled bundle, optionally serializing it below ``root``."""
    if isinstance(root, SyntheticTuningSpec):
        if spec is not None:
            raise TypeError("spec was provided twice")
        spec = root
        root = None
    resolved = spec or SyntheticTuningSpec()
    bundle = generate_bundle(resolved)
    if root is not None:
        write_aerosol_tuning_bundle(root, bundle)
    return bundle
