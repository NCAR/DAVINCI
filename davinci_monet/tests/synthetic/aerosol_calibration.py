"""Deterministic, synthetic-only scientific policy calibration records."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from davinci_monet.tests.synthetic._aerosol_contracts import canonical_json
from davinci_monet.tests.synthetic._aerosol_policy import (
    ScientificPolicy,
    policy_from_normalized,
)
from davinci_monet.tests.synthetic.fable_thresholds import RECOVERY_THRESHOLDS

CALIBRATION_SCHEMA = "fable-synthetic-calibration-v2"
NULL_FRACTION_MAX = 0.10
RANKING_ORDER = (
    "field_nrmse",
    "aod_rmse_ratio",
    "abs_field_origin_slope_error",
    "simplicity_rank",
    "policy_id",
)


EVIDENCE_HASH_KEYS = (
    "calibration_config_sha256",
    "calibration_manifest_sha256",
    "calibration_report_sha256",
    "calibration_spec_sha256",
    "code_sha256",
    "null_config_sha256",
    "null_manifest_sha256",
    "null_report_sha256",
    "null_spec_sha256",
)


@dataclass(frozen=True)
class CalibrationEvidence:
    """Immutable provenance for metrics derived from calibration-only pipeline runs."""

    calibration_seed: int
    null_seed: int
    calibration_scenario: str
    null_scenario: str
    calibration_split: str
    hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("calibration_seed", "null_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("calibration_scenario", "null_scenario", "calibration_split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a nonempty trimmed string")
        hashes = tuple(sorted((str(key), str(value)) for key, value in self.hashes))
        if tuple(key for key, _ in hashes) != EVIDENCE_HASH_KEYS:
            raise ValueError("calibration evidence must contain every required hash identity")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for _, value in hashes
        ):
            raise ValueError("calibration evidence hashes must be lowercase SHA-256 values")
        object.__setattr__(self, "hashes", hashes)

    def normalized(self) -> dict[str, Any]:
        """Return the canonical evidence document."""
        return {
            "calibration_scenario": self.calibration_scenario,
            "calibration_seed": self.calibration_seed,
            "calibration_split": self.calibration_split,
            "hashes": dict(self.hashes),
            "null_scenario": self.null_scenario,
            "null_seed": self.null_seed,
        }


@dataclass(frozen=True)
class CandidateMetrics:
    """Immutable calibration and null-control metrics for one policy."""

    field_correlation: float
    field_origin_slope: float
    field_nrmse: float
    aod_rmse_ratio: float
    full_target_aod_rmse_ratio: float
    null_retained_energy_fraction: float
    null_significant_fraction: float

    def __post_init__(self) -> None:
        correlation = _finite_float(self.field_correlation, "field_correlation")
        if not -1.0 <= correlation <= 1.0:
            raise ValueError("field_correlation must be in [-1, 1]")
        object.__setattr__(self, "field_correlation", correlation)
        slope = _finite_float(self.field_origin_slope, "field_origin_slope")
        object.__setattr__(self, "field_origin_slope", slope)
        for name in (
            "field_nrmse",
            "aod_rmse_ratio",
            "full_target_aod_rmse_ratio",
            "null_retained_energy_fraction",
        ):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        significant_fraction = _unit_fraction(
            self.null_significant_fraction, "null_significant_fraction"
        )
        object.__setattr__(self, "null_significant_fraction", significant_fraction)

    def normalized(self) -> dict[str, float]:
        """Return the canonical JSON-safe metrics."""
        return {
            "aod_rmse_ratio": self.aod_rmse_ratio,
            "field_correlation": self.field_correlation,
            "field_nrmse": self.field_nrmse,
            "field_origin_slope": self.field_origin_slope,
            "full_target_aod_rmse_ratio": self.full_target_aod_rmse_ratio,
            "null_retained_energy_fraction": self.null_retained_energy_fraction,
            "null_significant_fraction": self.null_significant_fraction,
        }


@dataclass(frozen=True)
class CalibrationCandidate:
    """One immutable policy and its calibration-only evidence."""

    policy: ScientificPolicy
    metrics: CandidateMetrics
    evidence: CalibrationEvidence


@dataclass(frozen=True)
class SyntheticCalibrationRecord:
    """Canonical selection result suitable for freezing before development scoring."""

    candidates: tuple[CalibrationCandidate, ...]
    selected_policy_id: str

    def __post_init__(self) -> None:
        candidates = tuple(sorted(self.candidates, key=lambda item: item.policy.policy_id))
        object.__setattr__(self, "candidates", candidates)
        expected = _selected_policy_id(candidates)
        if self.selected_policy_id != expected:
            raise ValueError(
                f"selected_policy_id does not match deterministic selection: expected {expected!r}"
            )

    @property
    def selected_policy(self) -> ScientificPolicy:
        """Return the selected frozen scientific controls."""
        return next(
            candidate.policy
            for candidate in self.candidates
            if candidate.policy.policy_id == self.selected_policy_id
        )

    def normalized(self) -> dict[str, Any]:
        """Return the complete canonical decision record."""
        candidates: list[dict[str, Any]] = []
        for candidate in self.candidates:
            reasons = candidate_rejection_reasons(candidate.metrics)
            rank = None if reasons else list(_ranking_key(candidate))
            candidates.append(
                {
                    "eligible": not reasons,
                    "evidence": candidate.evidence.normalized(),
                    "metrics": candidate.metrics.normalized(),
                    "policy": candidate.policy.normalized(),
                    "ranking_key": rank,
                    "rejection_reasons": list(reasons),
                }
            )
        return {
            "candidates": candidates,
            "null_thresholds": {
                "retained_energy_fraction_max": NULL_FRACTION_MAX,
                "significant_fraction_max": NULL_FRACTION_MAX,
            },
            "ranking_order": list(RANKING_ORDER),
            "recovery_thresholds": dict(RECOVERY_THRESHOLDS),
            "schema_version": CALIBRATION_SCHEMA,
            "selected_policy_id": self.selected_policy_id,
        }


def candidate_rejection_reasons(metrics: CandidateMetrics) -> tuple[str, ...]:
    """Return stable hard-rejection reasons for a candidate."""
    reasons: list[str] = []
    if metrics.field_correlation < RECOVERY_THRESHOLDS["field_correlation_min"]:
        reasons.append("field_correlation_below_minimum")
    if not (
        RECOVERY_THRESHOLDS["field_origin_slope_min"]
        <= metrics.field_origin_slope
        <= RECOVERY_THRESHOLDS["field_origin_slope_max"]
    ):
        reasons.append("field_origin_slope_outside_bounds")
    if metrics.field_nrmse > RECOVERY_THRESHOLDS["field_nrmse_max"]:
        reasons.append("field_nrmse_above_maximum")
    if metrics.aod_rmse_ratio > RECOVERY_THRESHOLDS["aod_rmse_ratio_max"]:
        reasons.append("aod_rmse_ratio_above_maximum")
    if metrics.full_target_aod_rmse_ratio >= RECOVERY_THRESHOLDS["full_target_aod_rmse_ratio_max"]:
        reasons.append("full_target_aod_does_not_improve_model")
    if metrics.null_retained_energy_fraction > NULL_FRACTION_MAX:
        reasons.append("null_retained_energy_fraction_above_maximum")
    if metrics.null_significant_fraction > NULL_FRACTION_MAX:
        reasons.append("null_significant_fraction_above_maximum")
    return tuple(reasons)


def select_calibration_policy(
    candidates: Sequence[CalibrationCandidate],
) -> SyntheticCalibrationRecord:
    """Hard-filter candidates and deterministically choose the best eligible policy."""
    canonical_candidates = tuple(sorted(candidates, key=lambda item: item.policy.policy_id))
    selected = _selected_policy_id(canonical_candidates)
    return SyntheticCalibrationRecord(canonical_candidates, selected)


def canonical_calibration_json(record: SyntheticCalibrationRecord) -> str:
    """Serialize a calibration record with stable key ordering and separators."""
    return canonical_json(record.normalized())


def calibration_record_sha256(record: SyntheticCalibrationRecord) -> str:
    """Return the SHA-256 of the canonical scientific decision record."""
    return hashlib.sha256(canonical_calibration_json(record).encode("ascii")).hexdigest()


def write_calibration_record(path: str | Path, record: SyntheticCalibrationRecord) -> Path:
    """Atomically publish an immutable, self-verifying calibration record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"calibration record already exists: {destination}")
    document = {
        "record": record.normalized(),
        "record_sha256": calibration_record_sha256(record),
    }
    payload = (canonical_json(document) + "\n").encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"calibration record already exists: {destination}") from exc
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_calibration_record(path: str | Path) -> SyntheticCalibrationRecord:
    """Load and verify canonical encoding, SHA-256, and selection semantics."""
    source = Path(path)
    raw = source.read_text(encoding="ascii")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid calibration record JSON: {source}") from exc
    if not isinstance(document, Mapping) or set(document) != {"record", "record_sha256"}:
        raise ValueError("calibration record envelope has unexpected fields")
    if raw != canonical_json(document) + "\n":
        raise ValueError("calibration record is not canonically encoded")
    normalized = document["record"]
    stored_hash = document["record_sha256"]
    if not isinstance(normalized, Mapping) or not isinstance(stored_hash, str):
        raise ValueError("calibration record envelope has invalid value types")
    actual_hash = hashlib.sha256(canonical_json(normalized).encode("ascii")).hexdigest()
    if not hmac.compare_digest(stored_hash, actual_hash):
        raise ValueError("calibration record SHA-256 mismatch")
    record = _record_from_normalized(normalized)
    if record.normalized() != normalized:
        raise ValueError("calibration record scientific policy or selection mismatch")
    return record


def verify_calibration_record(path: str | Path, expected_sha256: str) -> SyntheticCalibrationRecord:
    """Load a valid record and compare it with an independently held identity."""
    record = load_calibration_record(path)
    actual = calibration_record_sha256(record)
    if not hmac.compare_digest(expected_sha256, actual):
        raise ValueError("calibration record does not match expected SHA-256")
    return record


def _selected_policy_id(candidates: tuple[CalibrationCandidate, ...]) -> str:
    if not candidates:
        raise ValueError("at least one calibration candidate is required")
    policy_ids = [candidate.policy.policy_id for candidate in candidates]
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("calibration candidate policy_id values must be unique")
    eligible = [
        candidate for candidate in candidates if not candidate_rejection_reasons(candidate.metrics)
    ]
    if not eligible:
        raise ValueError("no calibration candidate passes all recovery and null thresholds")
    return min(eligible, key=_ranking_key).policy.policy_id


def _ranking_key(candidate: CalibrationCandidate) -> tuple[float, float, float, int, str]:
    metrics = candidate.metrics
    return (
        metrics.field_nrmse,
        metrics.aod_rmse_ratio,
        abs(metrics.field_origin_slope - 1.0),
        candidate.policy.simplicity_rank,
        candidate.policy.policy_id,
    )


def _record_from_normalized(value: Mapping[str, Any]) -> SyntheticCalibrationRecord:
    try:
        raw_candidates = value["candidates"]
        selected_policy_id = value["selected_policy_id"]
        if not isinstance(raw_candidates, list) or not isinstance(selected_policy_id, str):
            raise TypeError
        candidates = tuple(_candidate_from_normalized(item) for item in raw_candidates)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration record has invalid scientific candidate data") from exc
    return select_calibration_policy(candidates)


def _candidate_from_normalized(value: Any) -> CalibrationCandidate:
    if not isinstance(value, Mapping):
        raise TypeError
    raw_policy = value["policy"]
    raw_metrics = value["metrics"]
    raw_evidence = value["evidence"]
    if not all(isinstance(item, Mapping) for item in (raw_policy, raw_metrics, raw_evidence)):
        raise TypeError
    policy = policy_from_normalized(raw_policy)
    metrics = CandidateMetrics(
        field_correlation=raw_metrics["field_correlation"],
        field_origin_slope=raw_metrics["field_origin_slope"],
        field_nrmse=raw_metrics["field_nrmse"],
        aod_rmse_ratio=raw_metrics["aod_rmse_ratio"],
        full_target_aod_rmse_ratio=raw_metrics["full_target_aod_rmse_ratio"],
        null_retained_energy_fraction=raw_metrics["null_retained_energy_fraction"],
        null_significant_fraction=raw_metrics["null_significant_fraction"],
    )
    raw_hashes = raw_evidence["hashes"]
    if not isinstance(raw_hashes, Mapping):
        raise TypeError
    evidence = CalibrationEvidence(
        calibration_seed=raw_evidence["calibration_seed"],
        null_seed=raw_evidence["null_seed"],
        calibration_scenario=raw_evidence["calibration_scenario"],
        null_scenario=raw_evidence["null_scenario"],
        calibration_split=raw_evidence["calibration_split"],
        hashes=tuple((str(key), str(value)) for key, value in raw_hashes.items()),
    )
    return CalibrationCandidate(policy=policy, metrics=metrics, evidence=evidence)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not (float("-inf") < converted < float("inf")):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _positive_finite(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _unit_fraction(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return converted


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CALIBRATION_SCHEMA",
    "EVIDENCE_HASH_KEYS",
    "NULL_FRACTION_MAX",
    "CalibrationEvidence",
    "RANKING_ORDER",
    "CalibrationCandidate",
    "CandidateMetrics",
    "ScientificPolicy",
    "SyntheticCalibrationRecord",
    "calibration_record_sha256",
    "candidate_rejection_reasons",
    "canonical_calibration_json",
    "load_calibration_record",
    "select_calibration_policy",
    "verify_calibration_record",
    "write_calibration_record",
]
