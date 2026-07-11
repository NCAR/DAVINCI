"""Immutable selected-policy record for the FABLE v2 calibration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic.fable_v2_attempts import validate_v2_phase_attempt
from davinci_monet.tests.synthetic.fable_v2_calibration import (
    V2CalibrationCandidate,
    V2NullSeedResult,
    V2RecoverySeedResult,
)
from davinci_monet.tests.synthetic.fable_v2_development_approval import (
    eligible_v2_calibration_policies,
)
from davinci_monet.tests.synthetic.fable_v2_evidence import validate_v2_scenario_evidence
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
    V2Preregistration,
    verify_v2_preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import (
    V2GenerationPrerequisites,
)
from davinci_monet.tests.synthetic.fable_v2_lock_evidence import (
    validate_v2_generation_lock_identity,
)
from davinci_monet.tests.synthetic.fable_v2_policy import (
    FableV2Policy,
    v2_calibration_policies,
    v2_policy_from_normalized,
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

V2_CALIBRATION_SCHEMA = "fable-v2-calibration-v1"


@dataclass(frozen=True)
class V2CalibrationRecord:
    """Selected v2 policy bound to preregistration, locks, and full outcomes."""

    preregistration: FrozenPreregistration
    attempt: FrozenFileIdentity
    generation_locks: tuple[FrozenFileIdentity, FrozenFileIdentity]
    candidates: tuple[V2CalibrationCandidate, ...]
    selected_policy_id: str

    def __post_init__(self) -> None:
        selected = [
            item for item in self.candidates if item.policy.policy_id == self.selected_policy_id
        ]
        if len(selected) != 1 or not selected[0].eligible:
            raise ValueError("selected v2 calibration policy must be uniquely eligible")
        eligible = [item for item in self.candidates if item.eligible]
        if min(eligible, key=_ranking_key).policy.policy_id != self.selected_policy_id:
            raise ValueError("selected v2 calibration policy does not match frozen ranking")

    @property
    def selected_policy(self) -> FableV2Policy:
        return next(
            item.policy
            for item in self.candidates
            if item.policy.policy_id == self.selected_policy_id
        )

    def normalized(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt.normalized(),
            "candidates": [item.normalized() for item in self.candidates],
            "cycle_id": CYCLE_ID,
            "generation_locks": [item.normalized() for item in self.generation_locks],
            "preregistration": self.preregistration.normalized(),
            "protocol_sha256": protocol_sha256(),
            "schema_version": V2_CALIBRATION_SCHEMA,
            "selected_policy_id": self.selected_policy_id,
        }


def select_v2_calibration_policy(
    candidates: Sequence[V2CalibrationCandidate],
    preregistration: FrozenPreregistration,
    attempt: FrozenFileIdentity,
    generation_locks: tuple[FrozenFileIdentity, FrozenFileIdentity],
    *,
    eligible_policies: Sequence[FableV2Policy] | None = None,
) -> V2CalibrationRecord:
    """Hard-reject failed seeds/means, then apply the complete frozen menu/ranking."""
    values = tuple(sorted(candidates, key=lambda item: item.policy.policy_id))
    menu = tuple(v2_calibration_policies() if eligible_policies is None else eligible_policies)
    if not menu:
        raise ValueError("v2 calibration has no development-approved policy")
    expected = {item.policy_id: item.normalized() for item in menu}
    actual = {item.policy.policy_id: item.policy.normalized() for item in values}
    if actual != expected or len(values) != len(expected):
        raise ValueError("v2 calibration requires its exact development-approved policies")
    eligible = [item for item in values if item.eligible]
    if not eligible:
        raise ValueError("no v2 calibration candidate passes every seed and equal-seed gate")
    selected = min(eligible, key=_ranking_key)
    return V2CalibrationRecord(
        preregistration, attempt, generation_locks, values, selected.policy.policy_id
    )


def write_v2_calibration_record(path: str | Path, record: V2CalibrationRecord) -> Path:
    return write_record_once(path, record.normalized(), "calibration")


def load_v2_calibration_record(path: str | Path) -> V2CalibrationRecord:
    value = load_record_value(path, "calibration")
    record = _record_from_normalized(value)
    if record.normalized() != value:
        raise ValueError("v2 calibration record scientific or selection mismatch")
    return record


def validate_v2_calibration_record(
    record: V2CalibrationRecord,
    current: V2Preregistration,
    frozen: FrozenPreregistration,
) -> V2CalibrationRecord:
    """Reverify identities, locks, every artifact, menu, and deterministic selection."""
    verify_v2_preregistration(frozen, current)
    if record.preregistration.normalized() != frozen.normalized():
        raise ValueError("v2 calibration record uses a different preregistration")
    approved = eligible_v2_calibration_policies(current)
    expected = {item.policy_id: item.normalized() for item in approved}
    actual = {item.policy.policy_id: item.policy.normalized() for item in record.candidates}
    if actual != expected:
        raise ValueError("v2 calibration candidate menu drifted")
    attempt = validate_v2_phase_attempt(record.attempt, frozen, "calibration")
    roots = attempt.get("roots")
    if not isinstance(roots, Mapping):
        raise ValueError("calibration attempt has no roots")
    prerequisites = V2GenerationPrerequisites(current, frozen)
    roles = ("calibration_recovery", "calibration_null")
    for identity, role in zip(record.generation_locks, roles, strict=True):
        root = validate_v2_generation_lock_identity(
            identity, role, seed_roles()[role], prerequisites
        )
        name = "recovery" if role == "calibration_recovery" else "null"
        if roots.get(name) != str(root):
            raise ValueError("calibration generation lock root differs from attempt ledger")
    for candidate in record.candidates:
        _validate_candidate_evidence(candidate, verify_files=True)
    expected_record = select_v2_calibration_policy(
        record.candidates,
        frozen,
        record.attempt,
        record.generation_locks,
        eligible_policies=approved,
    )
    if expected_record != record:
        raise ValueError("v2 calibration selection drifted")
    return record


def _validate_candidate_evidence(candidate: V2CalibrationCandidate, *, verify_files: bool) -> None:
    for recovery_item in candidate.recovery:
        validate_v2_scenario_evidence(
            recovery_item.evidence,
            SyntheticTuningSpec.synthetic_osse(recovery_item.seed),
            candidate.policy,
            score_kind="recovery",
            evaluation_splits=("calibration",),
            verify_files=verify_files,
        )
    for null_item in candidate.null:
        validate_v2_scenario_evidence(
            null_item.evidence,
            SyntheticTuningSpec.synthetic_osse_null(null_item.seed),
            candidate.policy,
            score_kind="null",
            evaluation_splits=(),
            verify_files=verify_files,
        )


def _record_from_normalized(value: Mapping[str, Any]) -> V2CalibrationRecord:
    if (
        value.get("cycle_id") != CYCLE_ID
        or value.get("protocol_sha256") != protocol_sha256()
        or value.get("schema_version") != V2_CALIBRATION_SCHEMA
    ):
        raise ValueError("v2 calibration protocol identity mismatch")
    prereg = _mapping(value.get("preregistration"), "frozen preregistration")
    frozen = FrozenPreregistration(
        _file_identity(_mapping(prereg.get("file"), "preregistration file")),
        str(prereg["preregistration_sha256"]),
    )
    attempt = _file_identity(_mapping(value.get("attempt"), "calibration attempt"))
    raw_locks = value.get("generation_locks")
    if not isinstance(raw_locks, list) or len(raw_locks) != 2:
        raise ValueError("calibration record must bind two generation locks")
    locks = tuple(_file_identity(_mapping(item, "generation lock")) for item in raw_locks)
    candidates = tuple(
        _candidate_from_normalized(item) for item in _sequence(value.get("candidates"))
    )
    return select_v2_calibration_policy(
        candidates,
        frozen,
        attempt,
        cast(tuple[FrozenFileIdentity, FrozenFileIdentity], locks),
        eligible_policies=tuple(candidate.policy for candidate in candidates),
    )


def _candidate_from_normalized(value: Any) -> V2CalibrationCandidate:
    raw = _mapping(value, "calibration candidate")
    policy = v2_policy_from_normalized(_mapping(raw.get("policy"), "v2 policy"))
    recovery = tuple(_recovery_from_normalized(item) for item in _sequence(raw.get("recovery")))
    null = tuple(_null_from_normalized(item) for item in _sequence(raw.get("null")))
    candidate = V2CalibrationCandidate(policy, recovery, null)
    if candidate.normalized() != raw:
        raise ValueError("v2 calibration candidate derived fields changed")
    return candidate


def _recovery_from_normalized(value: Any) -> V2RecoverySeedResult:
    raw = _mapping(value, "recovery seed result")
    result = V2RecoverySeedResult(
        canonical_json(raw.get("evidence")), str(raw.get("evidence_sha256"))
    )
    if result.normalized() != raw:
        raise ValueError("recovery result fields do not match scenario evidence")
    return result


def _null_from_normalized(value: Any) -> V2NullSeedResult:
    raw = _mapping(value, "null seed result")
    result = V2NullSeedResult(canonical_json(raw.get("evidence")), str(raw.get("evidence_sha256")))
    if result.normalized() != raw:
        raise ValueError("null result fields do not match scenario evidence")
    return result


def _ranking_key(candidate: V2CalibrationCandidate) -> tuple[float, float, float, int, str]:
    values = candidate.aggregate
    return (
        values["field_nrmse"],
        values["aod_rmse_ratio"],
        abs(values["field_origin_slope"] - 1.0),
        candidate.policy.simplicity_rank,
        candidate.policy.policy_id,
    )


def _file_identity(value: Mapping[str, Any]) -> FrozenFileIdentity:
    return FrozenFileIdentity(str(value.get("path")), str(value.get("sha256")))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError("v2 calibration record field must be a list")
    return value


__all__ = [
    "V2_CALIBRATION_SCHEMA",
    "V2CalibrationRecord",
    "load_v2_calibration_record",
    "select_v2_calibration_policy",
    "validate_v2_calibration_record",
    "write_v2_calibration_record",
]
