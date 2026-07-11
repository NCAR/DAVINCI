"""Write-once terminal rejection evidence for irreversible FABLE v2 phases."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from davinci_monet.tests.synthetic.fable_v2_attempts import validate_v2_phase_attempt
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
    V2Preregistration,
    load_v2_preregistration,
    verify_v2_preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import V2GenerationPrerequisites
from davinci_monet.tests.synthetic.fable_v2_lock_evidence import (
    validate_v2_generation_lock_identity,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import (
    CYCLE_ID,
    protocol_sha256,
    seed_roles,
)
from davinci_monet.tests.synthetic.fable_v2_record_io import (
    canonical_json,
    load_record_value,
    write_record_once,
)

V2_PHASE_REJECTION_SCHEMA = "fable-v2-phase-rejection-v1"
V2Phase = Literal["calibration", "preflight", "acceptance"]
V2FailureClassification = Literal[
    "generation_exception",
    "pipeline_exception",
    "no_eligible_calibration_candidate",
    "selection_exception",
    "record_validation_exception",
]

_FAILURE_MESSAGES: dict[V2FailureClassification, str] = {
    "generation_exception": "scenario generation raised an exception",
    "pipeline_exception": "scenario pipeline execution raised an exception",
    "no_eligible_calibration_candidate": (
        "no development-approved calibration candidate passed every frozen gate"
    ),
    "selection_exception": "calibration policy selection raised an exception",
    "record_validation_exception": "phase record validation raised an exception",
}
_PHASE_FAILURES: dict[V2Phase, frozenset[V2FailureClassification]] = {
    "calibration": frozenset(_FAILURE_MESSAGES),
    "preflight": frozenset(
        {"generation_exception", "pipeline_exception", "record_validation_exception"}
    ),
    "acceptance": frozenset(
        {"generation_exception", "pipeline_exception", "record_validation_exception"}
    ),
}
_LOCK_COUNTS: dict[V2Phase, int] = {"calibration": 2, "preflight": 1, "acceptance": 1}
_LOCK_ROLES: dict[V2Phase, tuple[tuple[str, str], ...]] = {
    "calibration": (("recovery", "calibration_recovery"), ("null", "calibration_null")),
    "preflight": (("root", "preflight"),),
    "acceptance": (("root", "acceptance"),),
}


@dataclass(frozen=True)
class V2PhaseRejection:
    """Self-contained terminal evidence when a typed phase record cannot be produced."""

    phase: V2Phase
    preregistration: FrozenPreregistration
    attempt: FrozenFileIdentity
    generation_locks: tuple[FrozenFileIdentity, ...]
    classification: V2FailureClassification
    progress_json: str

    def __post_init__(self) -> None:
        if self.phase not in _PHASE_FAILURES:
            raise ValueError(f"unsupported v2 rejection phase: {self.phase!r}")
        if self.classification not in _PHASE_FAILURES[self.phase]:
            raise ValueError(
                f"unsupported {self.phase} rejection classification: {self.classification!r}"
            )
        if len(self.generation_locks) != _LOCK_COUNTS[self.phase]:
            raise ValueError(f"{self.phase} rejection has the wrong generation-lock count")
        progress = _progress_from_json(self.progress_json)
        expected_failure = {
            "classification": self.classification,
            "message": self.failure_message,
        }
        if (
            progress.get("attempt") != self.attempt.normalized()
            or progress.get("failure") != expected_failure
            or progress.get("status") != "rejected"
        ):
            raise ValueError("v2 rejection progress does not match its terminal identity")

    @property
    def failure_message(self) -> str:
        return _FAILURE_MESSAGES[self.classification]

    @property
    def progress(self) -> Mapping[str, Any]:
        return _progress_from_json(self.progress_json)

    def normalized(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt.normalized(),
            "cycle_id": CYCLE_ID,
            "failure": {
                "classification": self.classification,
                "message": self.failure_message,
            },
            "generation_locks": [item.normalized() for item in self.generation_locks],
            "phase": self.phase,
            "preregistration": self.preregistration.normalized(),
            "progress": dict(self.progress),
            "protocol_sha256": protocol_sha256(),
            "schema_version": V2_PHASE_REJECTION_SCHEMA,
            "status": "rejected",
        }


def rejection_failure(
    classification: V2FailureClassification,
) -> dict[str, str]:
    """Return the fixed failure document shared by progress and rejection evidence."""
    return {"classification": classification, "message": _FAILURE_MESSAGES[classification]}


def write_v2_phase_rejection(
    path: str | Path,
    phase: V2Phase,
    frozen: FrozenPreregistration,
    attempt: FrozenFileIdentity,
    generation_locks: Sequence[FrozenFileIdentity],
    classification: V2FailureClassification,
    progress: Mapping[str, Any],
) -> Path:
    """Verify frozen inputs and atomically publish one canonical rejection record."""
    frozen.file.verify()
    locks = tuple(generation_locks)
    record = V2PhaseRejection(
        phase,
        frozen,
        attempt,
        locks,
        classification,
        canonical_json(progress),
    )
    _validate_terminal_identities(record)
    return write_record_once(path, record.normalized(), f"{phase} rejection")


def load_v2_phase_rejection(path: str | Path) -> V2PhaseRejection:
    """Load a canonical rejection and rederive all fixed terminal fields."""
    value = load_record_value(path, "phase rejection")
    expected = {
        "attempt",
        "cycle_id",
        "failure",
        "generation_locks",
        "phase",
        "preregistration",
        "progress",
        "protocol_sha256",
        "schema_version",
        "status",
    }
    if (
        set(value) != expected
        or value.get("cycle_id") != CYCLE_ID
        or value.get("protocol_sha256") != protocol_sha256()
        or value.get("schema_version") != V2_PHASE_REJECTION_SCHEMA
        or value.get("status") != "rejected"
    ):
        raise ValueError("v2 phase rejection protocol identity mismatch")
    phase = _phase(value["phase"])
    failure = _mapping(value["failure"], "phase rejection failure")
    classification = _classification(failure.get("classification"))
    preregistration = _mapping(value["preregistration"], "phase rejection preregistration")
    locks = value["generation_locks"]
    if not isinstance(locks, list):
        raise ValueError("phase rejection generation locks must be a list")
    record = V2PhaseRejection(
        phase,
        FrozenPreregistration(
            _file_identity(_mapping(preregistration["file"], "preregistration file")),
            str(preregistration["preregistration_sha256"]),
        ),
        _file_identity(_mapping(value["attempt"], "phase rejection attempt")),
        tuple(_file_identity(_mapping(item, "phase rejection generation lock")) for item in locks),
        classification,
        canonical_json(_mapping(value["progress"], "phase rejection progress")),
    )
    if record.normalized() != value:
        raise ValueError("v2 phase rejection content mismatch")
    _validate_terminal_identities(record)
    return record


def _validate_terminal_identities(record: V2PhaseRejection) -> None:
    record.preregistration.file.verify()
    current = load_v2_preregistration(record.preregistration.file.path)
    verify_v2_preregistration(record.preregistration, current)
    attempt = validate_v2_phase_attempt(
        record.attempt,
        record.preregistration,
        record.phase,
    )
    roots = _mapping(attempt.get("roots"), "phase rejection attempt roots")
    roles = _LOCK_ROLES[record.phase]
    if set(roots) != {name for name, _ in roles}:
        raise ValueError("v2 rejection attempt roots do not match its phase")
    for identity, (root_name, role) in zip(record.generation_locks, roles, strict=True):
        prerequisites = _lock_prerequisites(
            identity,
            record.phase,
            current,
            record.preregistration,
        )
        root = validate_v2_generation_lock_identity(
            identity,
            role,
            seed_roles()[role],
            prerequisites,
        )
        if roots[root_name] != str(root):
            raise ValueError("v2 rejection generation lock root differs from attempt ledger")


def _lock_prerequisites(
    identity: FrozenFileIdentity,
    phase: V2Phase,
    current: V2Preregistration,
    frozen: FrozenPreregistration,
) -> V2GenerationPrerequisites:
    identity.verify()
    try:
        document = json.loads(Path(identity.path).read_text(encoding="ascii"))
    except json.JSONDecodeError as exc:
        raise ValueError("v2 rejection generation lock is invalid JSON") from exc
    lock = _mapping(document, "phase rejection generation lock")
    prerequisites = _mapping(lock.get("prerequisites"), "phase rejection generation prerequisites")
    expected = {"calibration_record", "preflight_record", "preflight_status", "preregistration"}
    if set(prerequisites) != expected or prerequisites["preregistration"] != frozen.normalized():
        raise ValueError("v2 rejection generation prerequisites do not match preregistration")
    calibration = _optional_file_identity(prerequisites["calibration_record"], "calibration")
    preflight = _optional_file_identity(prerequisites["preflight_record"], "preflight")
    status = prerequisites["preflight_status"]
    if phase == "calibration":
        if calibration is not None or preflight is not None or status is not None:
            raise ValueError("calibration rejection has downstream generation prerequisites")
    elif phase == "preflight":
        if calibration is None or preflight is not None or status is not None:
            raise ValueError("preflight rejection generation prerequisites are invalid")
    elif calibration is None or preflight is None or status != "passed":
        raise ValueError("acceptance rejection generation prerequisites are invalid")
    evidence = V2GenerationPrerequisites(
        current,
        frozen,
        calibration_record=calibration,
        preflight_record=preflight,
        preflight_status=cast(str | None, status),
    )
    _validate_phase_dependencies(phase, evidence)
    return evidence


def _validate_phase_dependencies(
    phase: V2Phase,
    evidence: V2GenerationPrerequisites,
) -> None:
    assert evidence.current_preregistration is not None
    assert evidence.frozen_preregistration is not None
    if phase == "calibration":
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
    if phase == "preflight":
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
    if preflight.status != "passed" or preflight.selected_policy != calibration.selected_policy:
        raise ValueError("acceptance rejection requires the exact frozen passing preflight")


def _optional_file_identity(value: Any, name: str) -> FrozenFileIdentity | None:
    if value is None:
        return None
    identity = _file_identity(_mapping(value, f"{name} prerequisite identity"))
    identity.verify()
    return identity


def _phase(value: Any) -> V2Phase:
    if not isinstance(value, str) or value not in _PHASE_FAILURES:
        raise ValueError(f"unsupported v2 rejection phase: {value!r}")
    return value


def _classification(value: Any) -> V2FailureClassification:
    if not isinstance(value, str) or value not in _FAILURE_MESSAGES:
        raise ValueError(f"unsupported v2 rejection classification: {value!r}")
    return value


def _file_identity(value: Mapping[str, Any]) -> FrozenFileIdentity:
    return FrozenFileIdentity(str(value["path"]), str(value["sha256"]))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _progress_from_json(value: str) -> Mapping[str, Any]:
    try:
        progress = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("v2 rejection progress is invalid JSON") from exc
    if not isinstance(progress, Mapping) or canonical_json(progress) != value:
        raise ValueError("v2 rejection progress is not a canonical mapping")
    return progress


__all__ = [
    "V2FailureClassification",
    "V2Phase",
    "V2PhaseRejection",
    "V2_PHASE_REJECTION_SCHEMA",
    "load_v2_phase_rejection",
    "rejection_failure",
    "write_v2_phase_rejection",
]
