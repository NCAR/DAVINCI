from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import xarray as xr

from davinci_monet.analysis.artifact_manifest import artifact_identity as _artifact_identity
from davinci_monet.analysis.artifact_manifest import (
    build_analysis_artifact_identity,
    validate_finalized_artifact_manifest,
)
from davinci_monet.analysis.base import ArtifactDeclaration
from davinci_monet.analysis.gridded_reductions import product_summary


@dataclass(frozen=True)
class ProductArtifactResult:
    product: str
    analysis_path: Path
    summary_path: Path
    analysis_checksum: str
    summary_checksum: str
    reused: bool = False


@dataclass(frozen=True)
class CollectionArtifactResult:
    """Atomic time-chunked NetCDF collection and lightweight summary."""

    root: Path
    paths: tuple[Path, ...]
    summary_path: Path
    checksums: Mapping[str, str]
    summary_checksum: str
    collection_checksum: str
    reused: bool = False


@dataclass(frozen=True)
class ArtifactMaterialization:
    """Persisted analysis payload returned to the pipeline stage."""

    dataset: xr.Dataset
    source_config: Mapping[str, Any]
    product_metadata: Mapping[str, Any] | None
    manifest_entries: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ArtifactService:
    """Filesystem artifact service exposed through :class:`AnalysisRuntime`."""

    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def materialize(
        self,
        analysis_name: str,
        dataset: xr.Dataset,
        declarations: Iterable[ArtifactDeclaration],
    ) -> ArtifactMaterialization:
        """Persist declared artifacts without consulting the analysis type."""
        current = dataset
        source_config: dict[str, Any] = {}
        product_metadata: dict[str, Any] | None = None
        manifest_entries: list[Mapping[str, Any]] = []

        for declaration in declarations:
            identity = _artifact_identity(dataset, declaration)
            if declaration.kind == "product":
                artifact = write_product_artifacts(
                    self.output_dir, analysis_name, current, identity=identity
                )
                reused = artifact.reused
                metadata = {
                    "artifact_path": str(artifact.analysis_path),
                    "summary_path": str(artifact.summary_path),
                }
                source_config.update(metadata)
                product_metadata = metadata
                checksums: Mapping[str, Any] = {
                    "analysis_sha256": artifact.analysis_checksum,
                    "summary_sha256": artifact.summary_checksum,
                }
                if declaration.reload:
                    current = load_product_dataset(artifact.analysis_path)
            elif declaration.kind == "netcdf_collection":
                collection = write_dataset_collection(
                    self.output_dir,
                    analysis_name,
                    current,
                    time_chunk_size=int(declaration.options.get("time_chunk_size", 31)),
                    identity=identity,
                )
                reused = collection.reused
                metadata = {
                    "artifact_dir": str(collection.root),
                    "artifact_glob": str(collection.root / "chunk-*.nc"),
                    "summary_path": str(collection.summary_path),
                }
                source_config.update(metadata)
                checksums = {
                    "files": dict(collection.checksums),
                    "summary_sha256": collection.summary_checksum,
                    "collection_sha256": collection.collection_checksum,
                }
                if declaration.reload:
                    current = load_dataset_collection(collection.paths)
            else:
                raise ValueError(f"unknown analysis artifact kind '{declaration.kind}'")

            manifest_entries.append(
                {
                    "analysis": analysis_name,
                    "role": declaration.role,
                    "kind": declaration.kind,
                    "status": "finalized",
                    "publication": "reused" if reused else "written",
                    **metadata,
                    "checksums": checksums,
                    "identity": identity,
                    "dimensions": {str(name): int(size) for name, size in dataset.sizes.items()},
                    "chunks": _dataset_chunks(dataset),
                }
            )

        return ArtifactMaterialization(
            dataset=current,
            source_config=source_config,
            product_metadata=product_metadata,
            manifest_entries=tuple(manifest_entries),
        )


def _field_arrays(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    return {str(name): ds[name] for name in ds.data_vars}


def _dataset_chunks(ds: xr.Dataset) -> dict[str, list[list[int]] | None]:
    return {
        str(name): (
            [list(map(int, chunk)) for chunk in data.chunks] if data.chunks is not None else None
        )
        for name, data in ds.data_vars.items()
    }


def _netcdf_safe_attrs(attrs: dict[Any, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, bool):
            safe[str(key)] = str(value)
        else:
            safe[str(key)] = value
    return safe


def _netcdf_safe_dataset(ds: xr.Dataset) -> xr.Dataset:
    writable = ds.copy(deep=False)
    writable.attrs = _netcdf_safe_attrs(dict(ds.attrs))
    for name in writable.data_vars:
        writable[name].attrs = _netcdf_safe_attrs(dict(writable[name].attrs))
    for name in writable.coords:
        writable.coords[name].attrs = _netcdf_safe_attrs(dict(writable.coords[name].attrs))
    return writable


def _sha256(path: Path) -> str:
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


def _read_summary(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(f"existing {description} has no valid receipt: {path}") from exc
    if not isinstance(value, dict):
        raise FileExistsError(f"existing {description} has no valid receipt: {path}")
    return value


def _collection_digest(checksums: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for filename, checksum in sorted(checksums.items()):
        digest.update(filename.encode("utf-8"))
        digest.update(checksum.encode("ascii"))
    return digest.hexdigest()


def _reuse_collection(
    destination: Path,
    analysis_name: str,
    identity: Mapping[str, Any],
    time_chunk_size: int,
) -> CollectionArtifactResult:
    summary_path = destination / "summary.json"
    summary = _read_summary(summary_path, "analysis artifact")
    if (
        summary.get("analysis") != analysis_name
        or summary.get("identity") != identity
        or summary.get("time_chunk_size") != time_chunk_size
    ):
        raise FileExistsError(f"existing analysis artifact identity does not match: {destination}")
    receipt = summary.get("checksums")
    if not isinstance(receipt, dict):
        raise FileExistsError(f"existing analysis artifact has no complete receipt: {destination}")
    file_hashes = receipt.get("files")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise FileExistsError(f"existing analysis artifact has no complete receipt: {destination}")
    paths = tuple(destination / str(filename) for filename in sorted(file_hashes))
    actual_names = {path.name for path in destination.glob("chunk-*.nc")}
    if actual_names != set(file_hashes):
        raise FileExistsError(f"existing analysis artifact file set does not match: {destination}")
    checksums = {path.name: _sha256(path) for path in paths if path.is_file()}
    collection_checksum = _collection_digest(checksums)
    if checksums != file_hashes or collection_checksum != receipt.get("collection_sha256"):
        raise FileExistsError(f"existing analysis artifact checksum does not match: {destination}")
    return CollectionArtifactResult(
        root=destination,
        paths=paths,
        summary_path=summary_path,
        checksums=checksums,
        summary_checksum=_sha256(summary_path),
        collection_checksum=collection_checksum,
        reused=True,
    )


def _reuse_product(
    destination: Path, product: str, identity: Mapping[str, Any]
) -> ProductArtifactResult:
    analysis_path = destination / "analysis.nc"
    summary_path = destination / "summary.json"
    summary = _read_summary(summary_path, "analysis product")
    receipt = summary.get("checksums")
    expected = receipt.get("analysis_sha256") if isinstance(receipt, dict) else None
    if summary.get("product") != product or summary.get("identity") != identity:
        raise FileExistsError(f"existing analysis product identity does not match: {destination}")
    if not analysis_path.is_file() or _sha256(analysis_path) != expected:
        raise FileExistsError(f"existing analysis product checksum does not match: {destination}")
    return ProductArtifactResult(
        product=product,
        analysis_path=analysis_path,
        summary_path=summary_path,
        analysis_checksum=str(expected),
        summary_checksum=_sha256(summary_path),
        reused=True,
    )


def _collection_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for raw_name, variable in ds.data_vars.items():
        if variable.ndim == 0 or variable.dtype.kind not in "fiu":
            continue
        encoding[str(raw_name)] = {
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
            "chunksizes": tuple(max(1, min(int(size), 64)) for size in variable.shape),
        }
    return encoding


def write_dataset_collection(
    output_dir: str | Path,
    analysis_name: str,
    ds: xr.Dataset,
    *,
    time_chunk_size: int = 31,
    identity: Mapping[str, Any] | None = None,
) -> CollectionArtifactResult:
    """Atomically persist a dataset as bounded NetCDF time chunks."""
    if not analysis_name or Path(analysis_name).name != analysis_name:
        raise ValueError("analysis_name must be one safe path component")
    if time_chunk_size < 1:
        raise ValueError("time_chunk_size must be positive")
    parent = Path(output_dir) / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / analysis_name
    resolved_identity = dict(identity or _artifact_identity(ds))
    if destination.exists():
        return _reuse_collection(destination, analysis_name, resolved_identity, time_chunk_size)

    staging = Path(tempfile.mkdtemp(prefix=f".{analysis_name}-", dir=parent))
    filenames: list[str] = []
    try:
        timed_variables = [name for name, data in ds.data_vars.items() if "time" in data.dims]
        static_variables = [name for name in ds.data_vars if name not in timed_variables]
        if "time" in ds.dims and timed_variables:
            timed = ds[timed_variables]
            selections = []
            for start in range(0, ds.sizes["time"], time_chunk_size):
                selection = timed.isel(
                    time=slice(start, min(start + time_chunk_size, ds.sizes["time"]))
                )
                if start == 0 and static_variables:
                    selection = xr.merge((selection, ds[static_variables]), compat="no_conflicts")
                selections.append(selection)
        else:
            selections = [ds]
        if not selections:
            raise ValueError("cannot persist an empty time collection")

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
        checksums = {path.name: _sha256(path) for path in staged_paths}
        collection_checksum = _collection_digest(checksums)
        summary = {
            "analysis": analysis_name,
            "attrs": {str(key): str(value) for key, value in ds.attrs.items()},
            "identity": resolved_identity,
            "dimensions": {str(name): int(size) for name, size in ds.sizes.items()},
            "chunks": _dataset_chunks(ds),
            "files": filenames,
            "time_chunk_size": time_chunk_size,
            "checksums": {
                "files": checksums,
                "collection_sha256": collection_checksum,
            },
        }
        summary_path = staging / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _fsync_path(summary_path)
        summary_checksum = _sha256(summary_path)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    paths = tuple(destination / filename for filename in filenames)
    summary_path = destination / "summary.json"
    return CollectionArtifactResult(
        root=destination,
        paths=paths,
        summary_path=summary_path,
        checksums=checksums,
        summary_checksum=summary_checksum,
        collection_checksum=collection_checksum,
    )


def load_dataset_collection(paths: Iterable[str | Path]) -> xr.Dataset:
    """Lazily reopen an ordered NetCDF artifact collection."""
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("artifact collection contains no files")
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


def write_product_artifacts(
    output_dir: str | Path,
    product: str,
    ds: xr.Dataset,
    *,
    identity: Mapping[str, Any] | None = None,
) -> ProductArtifactResult:
    if not product or Path(product).name != product:
        raise ValueError("product must be one safe path component")
    parent = Path(output_dir) / "products"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / product
    resolved_identity = dict(identity or _artifact_identity(ds))
    if destination.exists():
        return _reuse_product(destination, product, resolved_identity)
    encoding: dict[str, dict[str, Any]] = {}
    for raw_name, da in ds.data_vars.items():
        name = str(raw_name)
        if np.issubdtype(da.dtype, np.floating):
            encoding[name] = {"dtype": "float32", "_FillValue": np.float32(np.nan)}
    staging = Path(tempfile.mkdtemp(prefix=f".{product}-", dir=parent))
    analysis_tmp = staging / "analysis.nc"
    summary_tmp = staging / "summary.json"
    try:
        _netcdf_safe_dataset(ds).to_netcdf(analysis_tmp, engine="netcdf4", encoding=encoding)
        _fsync_path(analysis_tmp)
        analysis_checksum = _sha256(analysis_tmp)
        summary = {
            "product": product,
            "attrs": {str(k): str(v) for k, v in ds.attrs.items()},
            "identity": resolved_identity,
            "groups": [str(v) for v in ds["group"].values] if "group" in ds.coords else [],
            "fields": product_summary(_field_arrays(ds)),
            "checksums": {"analysis_sha256": analysis_checksum},
        }
        summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _fsync_path(summary_tmp)
        summary_checksum = _sha256(summary_tmp)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    analysis_path = destination / "analysis.nc"
    summary_path = destination / "summary.json"
    return ProductArtifactResult(
        product=product,
        analysis_path=analysis_path,
        summary_path=summary_path,
        analysis_checksum=analysis_checksum,
        summary_checksum=summary_checksum,
    )


def load_product_dataset(path: str | Path) -> xr.Dataset:
    with xr.open_dataset(path) as ds:
        return ds.load()
