"""Durable single-attempt claims for irreversible FABLE v2 phases."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import CYCLE_ID, protocol_sha256

ATTEMPT_SCHEMA = "fable-v2-phase-attempt-v1"
_PHASES = {"calibration", "preflight", "acceptance"}


def claim_v2_phase_attempt(
    frozen: FrozenPreregistration,
    phase: str,
    roots: Mapping[str, str | Path],
) -> FrozenFileIdentity:
    """Exclusively claim one irreversible phase before any scientific generation."""
    if phase not in _PHASES:
        raise ValueError(f"unsupported irreversible v2 phase: {phase!r}")
    if not roots or any(not name or name != name.strip() for name in roots):
        raise ValueError("v2 phase attempt roots must have nonempty names")
    frozen.file.verify()
    ledger = _ledger_root(frozen)
    ledger.mkdir(parents=True, exist_ok=True)
    path = ledger / f"{phase}.json"
    document = {
        "cycle_id": CYCLE_ID,
        "phase": phase,
        "preregistration": frozen.normalized(),
        "protocol_sha256": protocol_sha256(),
        "roots": {
            name: str(Path(root).expanduser().resolve()) for name, root in sorted(roots.items())
        },
        "schema_version": ATTEMPT_SCHEMA,
        "status": "claimed_before_generation",
    }
    payload = (_canonical_json(document) + "\n").encode("ascii")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise FileExistsError(
            f"v2 {phase} was already attempted for this preregistration: {path}"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(ledger)
    return FrozenFileIdentity.capture(path)


def validate_v2_phase_attempt(
    identity: FrozenFileIdentity,
    frozen: FrozenPreregistration,
    phase: str,
) -> Mapping[str, Any]:
    """Reverify immutable bytes and the expected cycle/preregistration/phase identity."""
    identity.verify()
    expected_path = _ledger_root(frozen) / f"{phase}.json"
    if Path(identity.path) != expected_path:
        raise ValueError("v2 phase attempt is outside its canonical durable ledger")
    try:
        document = json.loads(expected_path.read_text(encoding="ascii"))
    except json.JSONDecodeError as exc:
        raise ValueError("v2 phase attempt ledger is invalid JSON") from exc
    if (
        not isinstance(document, Mapping)
        or document.get("cycle_id") != CYCLE_ID
        or document.get("phase") != phase
        or document.get("preregistration") != frozen.normalized()
        or document.get("protocol_sha256") != protocol_sha256()
        or document.get("schema_version") != ATTEMPT_SCHEMA
        or document.get("status") != "claimed_before_generation"
    ):
        raise ValueError("v2 phase attempt identity does not match its frozen cycle")
    return document


def _ledger_root(frozen: FrozenPreregistration) -> Path:
    repository = Path(__file__).resolve().parents[3]
    return (
        repository
        / "analyses"
        / "aerosol-tuning"
        / "synthetic"
        / f".{CYCLE_ID}-attempts"
        / frozen.preregistration_sha256
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ATTEMPT_SCHEMA",
    "claim_v2_phase_attempt",
    "validate_v2_phase_attempt",
]
