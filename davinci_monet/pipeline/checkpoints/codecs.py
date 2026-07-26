"""Attempt-local codecs for datasets, JSON values, and file products."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

from davinci_monet.core.base import PairedData
from davinci_monet.core.identity import canonical_sha256, canonicalize
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.checkpoints.datasets import (
    load_dataset_collection,
    scientific_dataset_sha256,
    write_dataset_collection,
)
from davinci_monet.pipeline.checkpoints.models import (
    OBJECT_SCHEMA_VERSION,
    CheckpointObject,
)
from davinci_monet.pipeline.checkpoints.store import _atomic_create

if TYPE_CHECKING:
    from davinci_monet.pipeline.stages.base import SourceData

_NETCDF_PACKING_ATTRS = frozenset(
    {
        "_FillValue",
        "add_offset",
        "missing_value",
        "scale_factor",
    }
)


class CheckpointCodecError(ValueError):
    """Raised when checkpoint object bytes cannot be validated or restored."""


def _storage_names(parent: Path, object_id: str) -> tuple[str, ...]:
    """Return the canonical name followed by append-only repair generations."""
    repairs: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(object_id)}-r(\d{{3,}})$")
    if parent.is_dir():
        for path in parent.iterdir():
            match = pattern.fullmatch(path.name)
            if path.is_dir() and match is not None:
                repairs.append((int(match.group(1)), path.name))
    return (object_id, *(name for _, name in sorted(repairs)))


def _next_storage_name(parent: Path, object_id: str) -> str:
    """Allocate the first unused append-only repair generation name."""
    used = set(_storage_names(parent, object_id))
    generation = 1
    while True:
        name = f"{object_id}-r{generation:03d}"
        if name not in used and not (parent / name).exists():
            return name
        generation += 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_metadata(dataset: xr.Dataset) -> dict[str, Any]:
    def variable_metadata(data: xr.DataArray) -> dict[str, Any]:
        return {
            "attrs": canonicalize(
                {
                    str(name): value
                    for name, value in data.attrs.items()
                    if str(name) not in _NETCDF_PACKING_ATTRS
                }
            ),
            "dtype": str(data.dtype),
            "chunks": (
                [list(map(int, axis)) for axis in data.chunks] if data.chunks is not None else None
            ),
        }

    return {
        "attrs": canonicalize(dict(dataset.attrs)),
        "data_vars": {
            str(name): variable_metadata(data) for name, data in dataset.data_vars.items()
        },
        "coords": {str(name): variable_metadata(data) for name, data in dataset.coords.items()},
    }


def _restore_variable(
    data: xr.DataArray,
    metadata: Mapping[str, Any],
) -> xr.DataArray:
    dtype = metadata.get("dtype")
    if isinstance(dtype, str) and str(data.dtype) != dtype:
        data = data.astype(dtype)
    chunks = metadata.get("chunks", ...)
    if chunks is None:
        data.load()
    elif isinstance(chunks, list) and len(chunks) == data.ndim:
        data = data.chunk(
            {
                dimension: tuple(int(size) for size in axis)
                for dimension, axis in zip(data.dims, chunks, strict=True)
            }
        )
    variable_attrs = metadata.get("attrs")
    if isinstance(variable_attrs, Mapping):
        data.attrs = dict(variable_attrs)
    return data


def _restore_dataset_metadata(dataset: xr.Dataset, metadata: Mapping[str, Any]) -> None:
    attrs = metadata.get("attrs")
    if isinstance(attrs, Mapping):
        dataset.attrs = dict(attrs)
    variables = metadata.get("data_vars")
    if isinstance(variables, Mapping):
        for name, value in variables.items():
            if str(name) not in dataset.data_vars or not isinstance(value, Mapping):
                continue
            dataset[str(name)] = _restore_variable(dataset[str(name)], value)
    coordinates = metadata.get("coords")
    if isinstance(coordinates, Mapping):
        for name, value in coordinates.items():
            if str(name) not in dataset.coords or not isinstance(value, Mapping):
                continue
            dataset.coords[str(name)] = _restore_variable(
                dataset.coords[str(name)],
                value,
            )


class CheckpointCodecs:
    """Serialize and verify checkpoint objects below one attempt root."""

    def __init__(self, attempt_root: str | Path) -> None:
        self.attempt_root = Path(attempt_root).expanduser().resolve()
        self.objects_root = self.attempt_root / "objects" / "sha256"
        self.objects_root.mkdir(parents=True, exist_ok=True)

    def write_dataset(
        self,
        dataset: xr.Dataset,
        *,
        time_chunk_size: int = 31,
    ) -> CheckpointObject:
        metadata = _dataset_metadata(dataset)
        object_id = canonical_sha256(
            {
                "kind": "dataset",
                "scientific_sha256": scientific_dataset_sha256(dataset),
                "metadata": metadata,
                "time_chunk_size": time_chunk_size,
            }
        )
        parent = self.objects_root / object_id[:2]
        identity = {
            "checkpoint_object_id": object_id,
            "checkpoint_metadata": metadata,
        }
        result = None
        for storage_name in _storage_names(parent, object_id):
            try:
                result = write_dataset_collection(
                    parent,
                    storage_name,
                    dataset,
                    time_chunk_size=time_chunk_size,
                    identity=identity,
                )
            except FileExistsError:
                continue
            break
        if result is None:
            result = write_dataset_collection(
                parent,
                _next_storage_name(parent, object_id),
                dataset,
                time_chunk_size=time_chunk_size,
                identity=identity,
            )
        paths = (*result.paths, result.summary_path)
        checksums = {
            **{str(path): str(result.checksums[path.name]) for path in result.paths},
            str(result.summary_path): result.summary_checksum,
        }
        return CheckpointObject(
            schema_version=OBJECT_SCHEMA_VERSION,
            object_id=object_id,
            kind="dataset",
            paths=tuple(str(path) for path in paths),
            checksums=checksums,
            size_bytes=sum(path.stat().st_size for path in paths),
        )

    @staticmethod
    def dataset_metadata(dataset: xr.Dataset) -> dict[str, Any]:
        """Return the exact metadata and chunk contract needed for restoration."""
        return _dataset_metadata(dataset)

    def reference_finalized_dataset(
        self,
        dataset: xr.Dataset,
        source_config: Mapping[str, Any],
    ) -> CheckpointObject | None:
        """Return no external reference; main checkpoints own their bytes."""
        return None

    def read_dataset(
        self,
        obj: CheckpointObject,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> xr.Dataset:
        if obj.kind != "dataset":
            raise CheckpointCodecError(f"expected dataset object, got {obj.kind}")
        self.require_valid_object(obj)
        paths = sorted(Path(path) for path in obj.paths if Path(path).name.startswith("chunk-"))
        summary_paths = [Path(path) for path in obj.paths if Path(path).name == "summary.json"]
        if len(summary_paths) != 1:
            raise CheckpointCodecError("dataset object must contain one summary")
        resolved_metadata = metadata
        if resolved_metadata is None:
            try:
                summary = json.loads(summary_paths[0].read_text("utf-8"))
                resolved_metadata = summary["identity"]["checkpoint_metadata"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise CheckpointCodecError("dataset checkpoint metadata is unreadable") from exc
        if paths:
            dataset = load_dataset_collection(paths)
        else:
            raise CheckpointCodecError("dataset object contains no NetCDF payload")
        _restore_dataset_metadata(dataset, resolved_metadata)
        return dataset

    def write_json(self, value: Any) -> CheckpointObject:
        normalized = canonicalize(value)
        object_id = canonical_sha256({"kind": "json", "value": normalized})
        parent = self.objects_root / object_id[:2]
        parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(normalized, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
        ).encode("utf-8")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        path = None
        for storage_name in _storage_names(parent, object_id):
            candidate = parent / storage_name / "data.json"
            if candidate.is_file() and _file_sha256(candidate) == payload_sha256:
                path = candidate
                break
            if not candidate.parent.exists():
                candidate.parent.mkdir()
                _atomic_create(candidate, payload)
                path = candidate
                break
        if path is None:
            root = parent / _next_storage_name(parent, object_id)
            root.mkdir()
            path = root / "data.json"
            _atomic_create(path, payload)
        checksum = _file_sha256(path)
        return CheckpointObject(
            schema_version=OBJECT_SCHEMA_VERSION,
            object_id=object_id,
            kind="json",
            paths=(str(path),),
            checksums={str(path): checksum},
            size_bytes=path.stat().st_size,
        )

    def read_json(self, obj: CheckpointObject) -> Any:
        if obj.kind != "json" or len(obj.paths) != 1:
            raise CheckpointCodecError("expected one-file JSON object")
        self.require_valid_object(obj)
        try:
            return json.loads(Path(obj.paths[0]).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCodecError("JSON checkpoint is unreadable") from exc

    def capture_files(self, paths: Iterable[str | Path]) -> CheckpointObject:
        resolved = tuple(
            sorted(
                (Path(path).expanduser().resolve() for path in paths),
                key=str,
            )
        )
        if not resolved or any(not path.is_file() or path.stat().st_size < 1 for path in resolved):
            raise CheckpointCodecError("file checkpoint requires nonempty existing files")
        checksums = {str(path): _file_sha256(path) for path in resolved}
        entries = [
            {
                "path": str(path),
                "sha256": checksums[str(path)],
                "size_bytes": path.stat().st_size,
            }
            for path in resolved
        ]
        object_id = canonical_sha256({"kind": "files", "files": entries})
        return CheckpointObject(
            schema_version=OBJECT_SCHEMA_VERSION,
            object_id=object_id,
            kind="files",
            paths=tuple(str(path) for path in resolved),
            checksums=checksums,
            size_bytes=sum(path.stat().st_size for path in resolved),
        )

    def validate_object(self, obj: CheckpointObject) -> bool:
        if not obj.paths:
            return False
        total_size = 0
        for raw_path in obj.paths:
            path = Path(raw_path)
            expected = obj.checksums.get(str(path))
            if expected is None:
                expected = obj.checksums.get(path.name)
            if not path.is_file() or expected is None:
                return False
            if _file_sha256(path) != expected:
                return False
            total_size += path.stat().st_size
        return total_size == obj.size_bytes

    def require_valid_object(self, obj: CheckpointObject) -> None:
        if not self.validate_object(obj):
            raise CheckpointCodecError(
                f"checkpoint object checksum or file set is invalid: {obj.object_id}"
            )

    @staticmethod
    def source_metadata(source: SourceData) -> dict[str, Any]:
        return {
            "label": source.label,
            "source_type": source.source_type,
            "geometry": source.geometry.name,
            "variables": canonicalize(source.variables),
            "config": canonicalize(source.config),
        }

    @staticmethod
    def restore_source(dataset: xr.Dataset, metadata: Mapping[str, Any]) -> SourceData:
        from davinci_monet.pipeline.stages.base import SourceData

        return SourceData(
            data=dataset,
            label=str(metadata["label"]),
            source_type=str(metadata["source_type"]),
            geometry=DataGeometry[str(metadata["geometry"])],
            variables=dict(metadata.get("variables", {})),
            config=dict(metadata.get("config", {})),
        )

    @staticmethod
    def paired_metadata(paired: PairedData) -> dict[str, Any]:
        return {
            "x_source": paired.x_source,
            "y_source": paired.y_source,
            "geometry": paired.geometry.name,
            "pairing_info": canonicalize(paired.pairing_info),
        }

    @staticmethod
    def restore_paired(dataset: xr.Dataset, metadata: Mapping[str, Any]) -> PairedData:
        return PairedData.from_sources(
            data=dataset,
            x_source=str(metadata["x_source"]),
            y_source=str(metadata["y_source"]),
            geometry=DataGeometry[str(metadata["geometry"])],
            pairing_info=dict(metadata.get("pairing_info", {})),
        )
