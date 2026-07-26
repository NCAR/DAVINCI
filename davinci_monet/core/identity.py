"""Canonical scientific and execution identity construction."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

_SHA256_LENGTH = 64


def canonicalize(value: Any) -> Any:
    """Return a JSON-compatible value with deterministic scientific types."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [canonicalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist())
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"__float__": "nan"}
        return {"__float__": "infinity" if value > 0 else "-infinity"}
    return value


def canonical_sha256(value: Any) -> str:
    """Hash one canonical JSON representation."""
    payload = json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configuration_sha256(config: Any) -> str:
    """Hash a complete normalized configuration or config fragment."""
    return canonical_sha256(config)


def runtime_versions() -> dict[str, str]:
    """Return the numerical runtime versions that participate in resume identity."""
    versions = {"python": platform.python_version()}
    for distribution in (
        "numpy",
        "scipy",
        "xarray",
        "dask",
        "pandas",
        "netCDF4",
    ):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    try:
        versions["davinci"] = version("davinci")
    except PackageNotFoundError:
        try:
            versions["davinci"] = version("davinci-monet")
        except PackageNotFoundError:
            versions["davinci"] = "not-installed"
    return versions


def git_commit(root: str | Path) -> str | None:
    """Return the repository commit containing *root*, when available."""
    resolved = Path(root).expanduser().resolve()
    if resolved.is_file():
        resolved = resolved.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip().lower()
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        return None
    return commit


def _code_tree_sha256(root: Path) -> str:
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


@lru_cache(maxsize=8)
def _cached_code_tree_sha256(root: Path) -> str:
    return _code_tree_sha256(root)


def code_tree_sha256(root: str | Path, *, use_cache: bool = True) -> str:
    """Hash production Python files below *root*, excluding tests and caches."""
    resolved = Path(root).expanduser().resolve()
    if use_cache:
        return _cached_code_tree_sha256(resolved)
    return _code_tree_sha256(resolved)


def _validate_authoritative_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("authoritative checksum must be a SHA-256 digest")
    return normalized


def inventory_sources(
    paths: Iterable[str | Path],
    *,
    authoritative_checksums: Mapping[str | Path, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Inventory source bytes without hashing entire raw collections."""
    checksum_by_path = {
        str(Path(path).expanduser().resolve()): _validate_authoritative_sha256(checksum)
        for path, checksum in (authoritative_checksums or {}).items()
    }
    entries: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        stat = path.stat()
        entry: dict[str, Any] = {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        checksum = checksum_by_path.get(str(path))
        if checksum is not None:
            entry["authoritative_sha256"] = checksum
        entries.append(entry)
    return tuple(sorted(entries, key=lambda entry: str(entry["path"])))


def source_inventory_sha256(inventory: Iterable[Mapping[str, Any]]) -> str:
    """Hash one normalized source inventory."""
    return canonical_sha256(list(inventory))


def compose_checkpoint_identity(
    *,
    stage: str,
    item: str | None,
    config: Any,
    dependencies: Iterable[Any],
    source_inventory: Iterable[Mapping[str, Any]],
    code_sha256: str,
) -> dict[str, str]:
    """Compose one stage/item identity from all approved identity dimensions."""
    normalized_dependencies = sorted(
        (dependency.model_dump(mode="json") for dependency in dependencies),
        key=lambda dependency: (
            str(dependency["stage"]),
            str(dependency["item"]),
        ),
    )
    normalized_inventory = list(source_inventory)
    config_digest = configuration_sha256(config)
    dependency_digest = canonical_sha256(normalized_dependencies)
    inventory_digest = source_inventory_sha256(normalized_inventory)
    payload = {
        "stage": stage,
        "item": item,
        "config_sha256": config_digest,
        "dependencies_sha256": dependency_digest,
        "source_inventory_sha256": inventory_digest,
        "code_sha256": code_sha256,
    }
    return {
        **{key: str(value) for key, value in payload.items() if key.endswith("_sha256")},
        "identity_sha256": canonical_sha256(payload),
    }
