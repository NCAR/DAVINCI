"""Phase guards and generation-root locks for the FABLE v2 cycle."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
    V2Preregistration,
    verify_v2_preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import (
    CYCLE_ID,
    GENERATION_LOCK_SCHEMA,
    protocol_sha256,
    validate_role_seeds,
)


@dataclass(frozen=True)
class V2GenerationPrerequisites:
    """Frozen evidence available before opening a role-specific generation root."""

    current_preregistration: V2Preregistration | None = None
    frozen_preregistration: FrozenPreregistration | None = None
    calibration_record: FrozenFileIdentity | None = None
    preflight_record: FrozenFileIdentity | None = None
    preflight_status: str | None = None

    def __post_init__(self) -> None:
        if (self.current_preregistration is None) != (self.frozen_preregistration is None):
            raise ValueError("current and frozen preregistration must be supplied together")
        if self.calibration_record is not None and self.frozen_preregistration is None:
            raise ValueError("a calibration record requires a frozen preregistration")
        if (self.preflight_record is None) != (self.preflight_status is None):
            raise ValueError("preflight record and status must be supplied together")
        if self.preflight_record is not None and self.calibration_record is None:
            raise ValueError("a preflight record requires a frozen calibration record")
        if self.preflight_status not in (None, "passed", "failed"):
            raise ValueError("preflight status must be 'passed' or 'failed'")

    def verify(self) -> None:
        if self.frozen_preregistration is not None:
            assert self.current_preregistration is not None
            verify_v2_preregistration(self.frozen_preregistration, self.current_preregistration)
        if self.calibration_record is not None:
            self.calibration_record.verify()
        if self.preflight_record is not None:
            self.preflight_record.verify()

    def normalized(self) -> dict[str, Any]:
        return {
            "calibration_record": (
                None if self.calibration_record is None else self.calibration_record.normalized()
            ),
            "preflight_record": (
                None if self.preflight_record is None else self.preflight_record.normalized()
            ),
            "preflight_status": self.preflight_status,
            "preregistration": (
                None
                if self.frozen_preregistration is None
                else self.frozen_preregistration.normalized()
            ),
        }


def validate_v2_generation_request(
    role: str,
    seeds: Sequence[int],
    prerequisites: V2GenerationPrerequisites | None = None,
) -> tuple[int, ...]:
    """Fail before generation when role, phase, or frozen evidence is invalid."""
    values = validate_role_seeds(role, seeds)
    evidence = prerequisites or V2GenerationPrerequisites()
    evidence.verify()
    has_preregistration = evidence.frozen_preregistration is not None
    if role == "development":
        if has_preregistration or evidence.calibration_record or evidence.preflight_record:
            raise ValueError("development generation is closed once preregistration is frozen")
    elif role in ("calibration_recovery", "calibration_null"):
        if not has_preregistration:
            raise ValueError("calibration generation requires a valid frozen preregistration")
        if evidence.calibration_record or evidence.preflight_record:
            raise ValueError("calibration generation is closed after selection is frozen")
    elif role == "preflight":
        if not has_preregistration or evidence.calibration_record is None:
            raise ValueError("preflight generation requires frozen preregistration and calibration")
        if evidence.preflight_record is not None:
            raise ValueError("preflight generation cannot be repeated after its result is frozen")
    else:
        if (
            not has_preregistration
            or evidence.calibration_record is None
            or evidence.preflight_record is None
        ):
            raise ValueError("acceptance is locked until all prerequisite identities are frozen")
        if evidence.preflight_status != "passed":
            raise ValueError("acceptance is locked unless the frozen preflight passed")
    return values


@dataclass(frozen=True)
class V2GenerationLock:
    """Immutable proof that a runner owned and locked its root before generation."""

    root: Path
    path: Path
    role: str
    seeds: tuple[int, ...]
    prerequisites: V2GenerationPrerequisites
    file: FrozenFileIdentity


def prepare_v2_generation(
    root: str | Path,
    role: str,
    seeds: Sequence[int],
    prerequisites: V2GenerationPrerequisites | None = None,
) -> V2GenerationLock:
    """Validate a request, exclusively create its root, and write its lock first."""
    evidence = prerequisites or V2GenerationPrerequisites()
    values = validate_v2_generation_request(role, seeds, evidence)
    _validate_v2_generation_evidence(role, evidence)
    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    lock_path = destination / "generation-lock.json"
    document = _generation_lock_document(role, values, evidence)
    _write_read_only(lock_path, (_canonical_json(document) + "\n").encode("ascii"))
    return V2GenerationLock(
        destination,
        lock_path,
        role,
        values,
        evidence,
        FrozenFileIdentity.capture(lock_path),
    )


def verify_v2_generation_lock(lock: V2GenerationLock) -> None:
    """Reject role, prerequisite, byte, or identity drift after root creation."""
    values = validate_v2_generation_request(lock.role, lock.seeds, lock.prerequisites)
    _validate_v2_generation_evidence(lock.role, lock.prerequisites)
    document = _generation_lock_document(lock.role, values, lock.prerequisites)
    expected = (_canonical_json(document) + "\n").encode("ascii")
    if lock.path != lock.root / "generation-lock.json" or lock.path.read_bytes() != expected:
        raise ValueError("immutable v2 generation lock changed")
    lock.file.verify()


@lru_cache(maxsize=32)
def _validate_v2_generation_evidence(role: str, evidence: V2GenerationPrerequisites) -> None:
    """Recompute phase records at the last boundary before scientific generation."""
    if role == "development":
        return
    assert evidence.current_preregistration is not None
    assert evidence.frozen_preregistration is not None
    if role in {"calibration_recovery", "calibration_null"}:
        from davinci_monet.tests.synthetic.fable_v2_development_approval import (
            eligible_v2_calibration_policies,
        )

        eligible_v2_calibration_policies(evidence.current_preregistration)
        return
    assert evidence.calibration_record is not None
    from davinci_monet.tests.synthetic.fable_v2_calibration_record import (
        load_v2_calibration_record,
        validate_v2_calibration_record,
    )

    calibration = validate_v2_calibration_record(
        load_v2_calibration_record(evidence.calibration_record.path),
        evidence.current_preregistration,
        evidence.frozen_preregistration,
    )
    if role == "preflight":
        return
    assert evidence.preflight_record is not None
    from davinci_monet.tests.synthetic.fable_v2_preflight import (
        load_v2_preflight_record,
        validate_v2_preflight_record,
    )

    preflight = validate_v2_preflight_record(
        load_v2_preflight_record(evidence.preflight_record.path),
        evidence.current_preregistration,
        evidence.frozen_preregistration,
        evidence.calibration_record,
    )
    if preflight.status != "passed" or evidence.preflight_status != preflight.status:
        raise ValueError("acceptance generation requires the exact frozen passing preflight")
    if preflight.selected_policy != calibration.selected_policy:
        raise ValueError("acceptance prerequisite policies do not match")


def _generation_lock_document(
    role: str, seeds: tuple[int, ...], prerequisites: V2GenerationPrerequisites
) -> dict[str, Any]:
    return {
        "cycle_id": CYCLE_ID,
        "prerequisites": prerequisites.normalized(),
        "protocol_sha256": protocol_sha256(),
        "role": role,
        "schema_version": GENERATION_LOCK_SCHEMA,
        "seeds": list(seeds),
        "status": "locked_before_generation",
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_read_only(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable v2 generation lock already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "V2GenerationLock",
    "V2GenerationPrerequisites",
    "prepare_v2_generation",
    "validate_v2_generation_request",
    "verify_v2_generation_lock",
]
