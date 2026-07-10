"""Durable immutable-seed and mutable-progress acceptance records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def file_identity(path: Path) -> dict[str, str]:
    """Return the path and byte identity used by acceptance evidence."""
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def create_seed_lock(root: Path, path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """Durably create a read-only seed lock with exclusive filesystem ownership."""
    payload = (_canonical_json(document) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise FileExistsError(
            f"acceptance seed record is already locked: {path}; use a new root"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(root)
    return {"status": "immutable", "file": file_identity(path)}


def verify_seed_lock(
    path: Path,
    document: Mapping[str, Any],
    seeds: tuple[int, int, int],
    record: Mapping[str, Any],
) -> None:
    """Reject any lock-byte, lock-identity, or recorded-seed mutation."""
    expected_payload = (_canonical_json(document) + "\n").encode("utf-8")
    if not path.is_file() or path.read_bytes() != expected_payload:
        raise ValueError("immutable acceptance seed lock changed during the run")
    expected_identity = {"status": "immutable", "file": file_identity(path)}
    if record.get("seed_lock") != expected_identity or record.get("seeds") != list(seeds):
        raise ValueError("acceptance record does not match its immutable seed lock")


def write_record(root: Path, path: Path, record: Mapping[str, Any]) -> None:
    """Atomically replace and fsync the mutable acceptance progress record."""
    payload = (_canonical_json(_json_record_value(record)) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(root)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_record_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_record_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_record_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["create_seed_lock", "file_identity", "verify_seed_lock", "write_record"]
