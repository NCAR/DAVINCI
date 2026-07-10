"""Artifact scientific identity and finalized-manifest verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.analysis.base import ArtifactDeclaration

_HASH_BATCH_ELEMENTS = 1_000_000


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    """Hash decoded coordinates, values, dimensions, and attributes in bounded batches."""
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


@lru_cache(maxsize=1)
def _source_tree_sha256(root: Path) -> str:
    """Hash the production Python tree, including numerical companion modules."""
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and "tests" not in path.relative_to(root).parts
    )
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def artifact_identity(
    dataset: xr.Dataset, declaration: ArtifactDeclaration | None = None
) -> dict[str, Any]:
    """Collect scientific content, configuration, code, fit, and split identity."""
    attrs = {str(key): _identity_value(value) for key, value in dataset.attrs.items()}
    declared: dict[str, Any] = {}
    if declaration is not None and isinstance(declaration.options.get("identity"), Mapping):
        declared = {
            str(key): _identity_value(value)
            for key, value in declaration.options["identity"].items()
        }
    source_metadata = {
        "dimensions": {str(name): int(size) for name, size in dataset.sizes.items()},
        "variables": {
            str(name): {"dims": list(data.dims), "dtype": str(data.dtype)}
            for name, data in dataset.variables.items()
        },
        "provenance_attrs": {
            key: value
            for key, value in attrs.items()
            if "hash" in key
            or "sha256" in key
            or key in {"scenario", "schema_version", "root_seed"}
        },
    }
    fallback: dict[str, Any] = {
        "source_metadata_sha256": _canonical_sha256(source_metadata),
        "config_declaration_sha256": _canonical_sha256(
            {
                "kind": declaration.kind if declaration is not None else "product",
                "role": declaration.role if declaration is not None else "product",
                "options": (
                    {
                        str(key): _identity_value(value)
                        for key, value in declaration.options.items()
                        if key != "identity"
                    }
                    if declaration is not None
                    else {}
                ),
            }
        ),
        "code_source_tree_sha256": _source_tree_sha256(Path(__file__).parents[1]),
    }
    if not declared.get("source_inputs_sha256"):
        fallback["source_content_sha256"] = scientific_dataset_sha256(dataset)
    combined = {**attrs, **fallback, **declared}
    hashes = {key: value for key, value in combined.items() if "hash" in key or "sha256" in key}
    source_hashes = {key: value for key, value in hashes.items() if key.startswith("source_")}
    config_hashes = {key: value for key, value in hashes.items() if key.startswith("config_")}
    code_hashes = {key: value for key, value in hashes.items() if key.startswith("code_")}
    scientific_hashes = {
        key: value
        for key, value in hashes.items()
        if key not in source_hashes and key not in config_hashes and key not in code_hashes
    }
    identity: dict[str, Any] = {
        "source_hashes": source_hashes,
        "config_hashes": config_hashes,
        "code_hashes": code_hashes,
        "scientific_hashes": scientific_hashes,
        "fit_identity": {key: value for key, value in combined.items() if "fit" in key},
        "split_identity": {key: value for key, value in combined.items() if "split" in key},
        "provenance": {
            key: attrs[key]
            for key in ("scenario", "schema_version", "root_seed", "analysis_type")
            if key in attrs
        },
    }
    if declared:
        identity["declared"] = declared
    return identity


def build_analysis_artifact_identity(
    spec: Any,
    inputs: Mapping[str, xr.Dataset],
    analysis: Any,
) -> Mapping[str, Any]:
    """Build nonempty input/config/code hashes for an analysis declaration."""
    if hasattr(spec, "model_dump"):
        try:
            normalized = spec.model_dump(mode="json", exclude_none=True)
        except TypeError:
            normalized = spec.model_dump()
    else:
        normalized = dict(vars(spec))
    source_identity = {
        role: {
            "content_sha256": scientific_dataset_sha256(dataset),
            "dimensions": {str(name): int(size) for name, size in dataset.sizes.items()},
            "variables": {
                str(name): {"dims": list(data.dims), "dtype": str(data.dtype)}
                for name, data in dataset.variables.items()
            },
            "hash_attrs": {
                str(key): _identity_value(value)
                for key, value in dataset.attrs.items()
                if "hash" in str(key) or "sha256" in str(key)
            },
            "fit_attrs": {
                str(key): _identity_value(value)
                for key, value in dataset.attrs.items()
                if "fit" in str(key)
            },
            "split_attrs": {
                str(key): _identity_value(value)
                for key, value in dataset.attrs.items()
                if "split" in str(key)
            },
        }
        for role, dataset in sorted(inputs.items())
    }
    del analysis
    code_hash = _source_tree_sha256(Path(__file__).parents[1])
    return {
        "source_inputs_sha256": _canonical_sha256(source_identity),
        "config_sha256": _canonical_sha256(normalized),
        "code_sha256": code_hash,
        "config_normalized": normalized,
        "fit_identity": {
            "config": {
                str(key): _identity_value(value)
                for key, value in normalized.items()
                if "fit" in str(key)
            },
            "inputs": {
                role: value["fit_attrs"]
                for role, value in source_identity.items()
                if value["fit_attrs"]
            },
        },
        "split_identity": {
            "config": {
                str(key): _identity_value(value)
                for key, value in normalized.items()
                if "split" in str(key)
            },
            "inputs": {
                role: value["split_attrs"]
                for role, value in source_identity.items()
                if value["split_attrs"]
            },
        },
    }


def validate_finalized_artifact_manifest(
    manifest_path: str | Path,
    artifact_paths: Iterable[str | Path],
    *,
    role: str,
    analysis: str | None = None,
) -> Mapping[str, Any]:
    """Verify a finalized manifest entry and every configured artifact byte."""
    manifest_file = Path(manifest_path).expanduser()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"artifact manifest does not exist: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact manifest is unreadable: {manifest_file}") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError("artifact manifest is not a completed pipeline run")
    entries = manifest.get("analysis_artifacts")
    if not isinstance(entries, list):
        raise ValueError("artifact manifest has no analysis_artifacts list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("role") == role
        and (analysis is None or entry.get("analysis") == analysis)
    ]
    if len(matches) != 1:
        qualifier = f" and analysis {analysis!r}" if analysis is not None else ""
        raise ValueError(
            f"artifact manifest must contain exactly one role {role!r}{qualifier}; "
            f"found {len(matches)}"
        )
    entry = matches[0]
    if entry.get("status") != "finalized":
        raise ValueError(f"artifact manifest role {role!r} is not finalized")
    identity = entry.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"artifact manifest role {role!r} has no identity metadata")
    for bucket in ("source_hashes", "config_hashes", "code_hashes"):
        hashes = identity.get(bucket)
        if not isinstance(hashes, dict) or not hashes or not all(hashes.values()):
            raise ValueError(
                f"artifact manifest role {role!r} has no nonempty {bucket.replace('_', ' ')}"
            )

    configured = {Path(path).expanduser().resolve() for path in artifact_paths}
    checksums = entry.get("checksums")
    if not configured or not isinstance(checksums, dict):
        raise ValueError(f"artifact manifest role {role!r} has no configured files or checksums")
    kind = entry.get("kind")
    if kind == "netcdf_collection":
        _validate_collection(entry, checksums, configured)
    elif kind == "product":
        _validate_product(entry, checksums, configured)
    else:
        raise ValueError(f"unsupported finalized artifact kind: {kind!r}")
    return entry


def _validate_collection(
    entry: Mapping[str, Any], checksums: Mapping[str, Any], configured: set[Path]
) -> None:
    root = Path(str(entry.get("artifact_dir", ""))).expanduser().resolve()
    file_hashes = checksums.get("files")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("artifact collection manifest has no file checksums")
    if any(Path(str(filename)).name != str(filename) for filename in file_hashes):
        raise ValueError("artifact collection manifest contains an unsafe filename")
    expected = {(root / str(filename)).resolve() for filename in file_hashes}
    if not configured.issubset(expected):
        raise ValueError("configured artifact files do not match the finalized collection")
    for filename, checksum in file_hashes.items():
        path = (root / str(filename)).resolve()
        if not path.is_file() or _file_sha256(path) != str(checksum):
            raise ValueError(f"artifact checksum mismatch: {path}")
    _validate_summary(entry, checksums)
    digest = hashlib.sha256()
    for filename, checksum in sorted(file_hashes.items()):
        digest.update(str(filename).encode("utf-8"))
        digest.update(str(checksum).encode("ascii"))
    if digest.hexdigest() != str(checksums.get("collection_sha256", "")):
        raise ValueError("artifact collection checksum mismatch")


def _validate_product(
    entry: Mapping[str, Any], checksums: Mapping[str, Any], configured: set[Path]
) -> None:
    analysis_path = Path(str(entry.get("artifact_path", ""))).expanduser().resolve()
    if configured != {analysis_path}:
        raise ValueError("configured artifact file does not match the finalized product")
    if not analysis_path.is_file() or _file_sha256(analysis_path) != str(
        checksums.get("analysis_sha256", "")
    ):
        raise ValueError(f"artifact checksum mismatch: {analysis_path}")
    _validate_summary(entry, checksums)


def _validate_summary(entry: Mapping[str, Any], checksums: Mapping[str, Any]) -> None:
    summary_path = Path(str(entry.get("summary_path", ""))).expanduser().resolve()
    if not summary_path.is_file() or _file_sha256(summary_path) != str(
        checksums.get("summary_sha256", "")
    ):
        raise ValueError(f"artifact summary checksum mismatch: {summary_path}")


__all__ = [
    "artifact_identity",
    "build_analysis_artifact_identity",
    "scientific_dataset_sha256",
    "validate_finalized_artifact_manifest",
]
