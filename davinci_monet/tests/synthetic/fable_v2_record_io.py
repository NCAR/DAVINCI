"""Canonical immutable JSON-envelope helpers for FABLE v2 records."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_record_once(path: str | Path, value: Mapping[str, Any], label: str) -> Path:
    """Atomically link one read-only canonical record envelope into place."""
    destination = Path(path).expanduser().resolve()
    envelope = {"record": value, "record_sha256": json_sha256(value)}
    payload = canonical_json(envelope) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"frozen v2 {label} already exists: {destination}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"frozen v2 {label} already exists: {destination}") from exc
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_record_value(path: str | Path, label: str) -> Mapping[str, Any]:
    """Load canonical envelope bytes and verify their scientific hash."""
    source = Path(path).expanduser().resolve()
    raw = source.read_text(encoding="ascii")
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid v2 {label} JSON") from exc
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"record", "record_sha256"}
        or raw != canonical_json(envelope) + "\n"
        or not isinstance(envelope["record"], Mapping)
        or not hmac.compare_digest(str(envelope["record_sha256"]), json_sha256(envelope["record"]))
    ):
        raise ValueError(f"v2 {label} record envelope is invalid")
    return envelope["record"]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["canonical_json", "json_sha256", "load_record_value", "write_record_once"]
