"""Multi-seed calibration records and frozen selection for FABLE v2."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    ACCEPTANCE_EXCLUDED_FRACTION_MAX,
)
from davinci_monet.tests.synthetic.fable_thresholds import RECOVERY_THRESHOLDS
from davinci_monet.tests.synthetic.fable_v2_evidence import validate_v2_scenario_evidence
from davinci_monet.tests.synthetic.fable_v2_identity import NULL_FRACTION_MAX
from davinci_monet.tests.synthetic.fable_v2_policy import FableV2Policy
from davinci_monet.tests.synthetic.fable_v2_protocol import validate_role_seeds
from davinci_monet.tests.synthetic.fable_v2_runner import V2ScenarioOutcome


@dataclass(frozen=True)
class V2RecoverySeedResult:
    """Recovery values derived only from one canonical scenario outcome."""

    evidence_json: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _validate_evidence_json(self.evidence_json, self.evidence_sha256)
        _seed(self.seed)
        for name in _RECOVERY_FIELDS:
            _finite(getattr(self, name), name)
        if not -1.0 <= self.field_correlation <= 1.0:
            raise ValueError("field_correlation must be in [-1, 1]")
        for name in (
            "field_nrmse",
            "aod_rmse_ratio",
            "full_target_aod_rmse_ratio",
            "learned_basis_oracle_nrmse",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.excluded_fraction <= 1.0:
            raise ValueError("excluded_fraction must be in [0, 1]")
        if not isinstance(self.evidence_complete, bool) or not isinstance(
            self.resources_passed, bool
        ):
            raise ValueError("recovery evidence/resource states must be boolean")
        if not isinstance(self.score_passed, bool):
            raise ValueError("recovery scientific score state must be boolean")
        _score_failures(self.evidence)
        _sha256(self.diagnostic_report_sha256, "diagnostic report sha256")
        _finite(self.learned_basis_oracle_nrmse, "learned basis oracle nrmse")

    @classmethod
    def from_outcome(cls, outcome: V2ScenarioOutcome) -> V2RecoverySeedResult:
        evidence = outcome.normalized()
        return cls(_canonical_json(evidence), _json_sha256(evidence))

    @property
    def evidence(self) -> Mapping[str, Any]:
        return _evidence_mapping(self.evidence_json)

    @property
    def seed(self) -> int:
        return _integer(self.evidence.get("seed"), "recovery seed")

    @property
    def field_correlation(self) -> float:
        return _score_metric(self.evidence, "field_correlation")

    @property
    def field_origin_slope(self) -> float:
        return _score_metric(self.evidence, "field_origin_slope")

    @property
    def field_nrmse(self) -> float:
        return _score_metric(self.evidence, "field_nrmse")

    @property
    def aod_rmse_ratio(self) -> float:
        return _score_metric(self.evidence, "aod_rmse_ratio")

    @property
    def full_target_aod_rmse_ratio(self) -> float:
        return _score_metric(self.evidence, "full_target_aod_rmse_ratio")

    @property
    def excluded_fraction(self) -> float:
        score = _mapping(self.evidence.get("score"), "recovery score")
        diagnostics = _mapping(score.get("diagnostics"), "recovery diagnostics")
        return _finite(diagnostics.get("excluded_fraction"), "excluded_fraction")

    @property
    def evidence_complete(self) -> bool:
        return _boolean(self.evidence.get("evidence_passed"), "evidence_passed")

    @property
    def score_passed(self) -> bool:
        score = _mapping(self.evidence.get("score"), "recovery score")
        return _boolean(score.get("passed"), "recovery score passed")

    @property
    def score_failures(self) -> tuple[str, ...]:
        return _score_failures(self.evidence)

    @property
    def resources_passed(self) -> bool:
        resources = _mapping(self.evidence.get("resources"), "recovery resources")
        return _boolean(resources.get("passed"), "resource passed")

    @property
    def diagnostic_report_sha256(self) -> str:
        return _sha256(self.evidence.get("diagnostic_report_sha256"), "diagnostic report sha256")

    @property
    def learned_basis_oracle_nrmse(self) -> float:
        return _finite(
            self.evidence.get("learned_basis_oracle_nrmse"),
            "learned basis oracle nrmse",
        )

    def normalized(self) -> dict[str, Any]:
        return {
            "aod_rmse_ratio": self.aod_rmse_ratio,
            "evidence_complete": self.evidence_complete,
            "evidence": self.evidence,
            "evidence_sha256": self.evidence_sha256,
            "diagnostic_report_sha256": self.diagnostic_report_sha256,
            "excluded_fraction": self.excluded_fraction,
            "field_correlation": self.field_correlation,
            "field_nrmse": self.field_nrmse,
            "field_origin_slope": self.field_origin_slope,
            "full_target_aod_rmse_ratio": self.full_target_aod_rmse_ratio,
            "learned_basis_oracle_nrmse": self.learned_basis_oracle_nrmse,
            "resources_passed": self.resources_passed,
            "score_failures": list(self.score_failures),
            "score_passed": self.score_passed,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class V2NullSeedResult:
    """False-positive values derived only from one canonical scenario outcome."""

    evidence_json: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _validate_evidence_json(self.evidence_json, self.evidence_sha256)
        _seed(self.seed)
        _finite(self.null_retained_energy_fraction, "null retained energy fraction")
        _finite(self.null_significant_fraction, "null significant fraction")
        if self.null_retained_energy_fraction < 0.0:
            raise ValueError("null retained energy fraction must be nonnegative")
        if not 0.0 <= self.null_significant_fraction <= 1.0:
            raise ValueError("null significant fraction must be in [0, 1]")
        if not isinstance(self.evidence_complete, bool) or not isinstance(
            self.resources_passed, bool
        ):
            raise ValueError("null evidence/resource states must be boolean")
        if not isinstance(self.score_passed, bool):
            raise ValueError("null scientific score state must be boolean")

    @classmethod
    def from_outcome(cls, outcome: V2ScenarioOutcome) -> V2NullSeedResult:
        evidence = outcome.normalized()
        return cls(_canonical_json(evidence), _json_sha256(evidence))

    @property
    def evidence(self) -> Mapping[str, Any]:
        return _evidence_mapping(self.evidence_json)

    @property
    def seed(self) -> int:
        return _integer(self.evidence.get("seed"), "null seed")

    @property
    def null_retained_energy_fraction(self) -> float:
        return _score_metric(self.evidence, "null_retained_energy_fraction")

    @property
    def null_significant_fraction(self) -> float:
        return _score_metric(self.evidence, "null_significant_fraction")

    @property
    def evidence_complete(self) -> bool:
        return _boolean(self.evidence.get("evidence_passed"), "evidence_passed")

    @property
    def score_passed(self) -> bool:
        score = _mapping(self.evidence.get("score"), "null score")
        return _boolean(score.get("passed"), "null score passed")

    @property
    def resources_passed(self) -> bool:
        resources = _mapping(self.evidence.get("resources"), "null resources")
        return _boolean(resources.get("passed"), "resource passed")

    def normalized(self) -> dict[str, Any]:
        return {
            "evidence_complete": self.evidence_complete,
            "evidence": self.evidence,
            "evidence_sha256": self.evidence_sha256,
            "null_retained_energy_fraction": self.null_retained_energy_fraction,
            "null_significant_fraction": self.null_significant_fraction,
            "resources_passed": self.resources_passed,
            "score_passed": self.score_passed,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class V2CalibrationCandidate:
    """One eligible-menu policy scored on every frozen calibration seed."""

    policy: FableV2Policy
    recovery: tuple[V2RecoverySeedResult, ...]
    null: tuple[V2NullSeedResult, ...]

    def __post_init__(self) -> None:
        validate_role_seeds("calibration_recovery", [item.seed for item in self.recovery])
        validate_role_seeds("calibration_null", [item.seed for item in self.null])
        for recovery_item in self.recovery:
            validate_v2_scenario_evidence(
                _evidence_mapping(recovery_item.evidence_json),
                SyntheticTuningSpec.synthetic_osse(recovery_item.seed),
                self.policy,
                score_kind="recovery",
                evaluation_splits=("calibration",),
                verify_files=False,
            )
        for null_item in self.null:
            validate_v2_scenario_evidence(
                _evidence_mapping(null_item.evidence_json),
                SyntheticTuningSpec.synthetic_osse_null(null_item.seed),
                self.policy,
                score_kind="null",
                evaluation_splits=(),
                verify_files=False,
            )

    @property
    def aggregate(self) -> dict[str, float]:
        values = {
            name: float(np.mean([getattr(item, name) for item in self.recovery]))
            for name in _RECOVERY_FIELDS
        }
        values.update(
            {
                name: float(np.mean([getattr(item, name) for item in self.null]))
                for name in _NULL_FIELDS
            }
        )
        return values

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        failures: list[str] = []
        for recovery_item in self.recovery:
            failures.extend(
                f"recovery_seed_{recovery_item.seed}:{value}"
                for value in _recovery_failures(recovery_item)
            )
        for null_item in self.null:
            failures.extend(
                f"null_seed_{null_item.seed}:{value}" for value in _null_failures(null_item)
            )
        failures.extend(f"equal_seed_mean:{value}" for value in _recovery_failures(self.aggregate))
        failures.extend(f"equal_seed_mean:{value}" for value in _null_failures(self.aggregate))
        return tuple(failures)

    @property
    def eligible(self) -> bool:
        return not self.rejection_reasons

    def normalized(self) -> dict[str, Any]:
        return {
            "aggregate": self.aggregate,
            "eligible": self.eligible,
            "null": [item.normalized() for item in self.null],
            "policy": self.policy.normalized(),
            "recovery": [item.normalized() for item in self.recovery],
            "rejection_reasons": list(self.rejection_reasons),
        }


_RECOVERY_FIELDS = (
    "field_correlation",
    "field_origin_slope",
    "field_nrmse",
    "aod_rmse_ratio",
    "full_target_aod_rmse_ratio",
    "excluded_fraction",
)
_NULL_FIELDS = ("null_retained_energy_fraction", "null_significant_fraction")


def _recovery_failures(value: Any) -> list[str]:
    failures: list[str] = []
    if hasattr(value, "score_passed") and not value.score_passed:
        reasons = value.score_failures
        failures.extend(
            f"scientific_score_failed:{reason}" for reason in (reasons or ("unspecified",))
        )
    if hasattr(value, "evidence_complete") and not value.evidence_complete:
        failures.append("incomplete_evidence")
    if hasattr(value, "resources_passed") and not value.resources_passed:
        failures.append("resource_gate_failed")
    field_correlation = _metric(value, "field_correlation")
    field_origin_slope = _metric(value, "field_origin_slope")
    field_nrmse = _metric(value, "field_nrmse")
    aod_rmse_ratio = _metric(value, "aod_rmse_ratio")
    full_target_ratio = _metric(value, "full_target_aod_rmse_ratio")
    excluded_fraction = _metric(value, "excluded_fraction")
    if field_correlation < RECOVERY_THRESHOLDS["field_correlation_min"]:
        failures.append("field_correlation_below_minimum")
    if not (
        RECOVERY_THRESHOLDS["field_origin_slope_min"]
        <= field_origin_slope
        <= RECOVERY_THRESHOLDS["field_origin_slope_max"]
    ):
        failures.append("field_origin_slope_outside_range")
    if field_nrmse > RECOVERY_THRESHOLDS["field_nrmse_max"]:
        failures.append("field_nrmse_above_maximum")
    if aod_rmse_ratio > RECOVERY_THRESHOLDS["aod_rmse_ratio_max"]:
        failures.append("aod_rmse_ratio_above_maximum")
    if full_target_ratio >= RECOVERY_THRESHOLDS["full_target_aod_rmse_ratio_max"]:
        failures.append("full_target_aod_rmse_ratio_not_improved")
    if excluded_fraction > ACCEPTANCE_EXCLUDED_FRACTION_MAX:
        failures.append("excluded_fraction_above_maximum")
    return failures


def _null_failures(value: Any) -> list[str]:
    failures: list[str] = []
    if hasattr(value, "score_passed") and not value.score_passed:
        failures.append("scientific_score_failed")
    if hasattr(value, "evidence_complete") and not value.evidence_complete:
        failures.append("incomplete_evidence")
    if hasattr(value, "resources_passed") and not value.resources_passed:
        failures.append("resource_gate_failed")
    retained = _metric(value, "null_retained_energy_fraction")
    significant = _metric(value, "null_significant_fraction")
    if retained > NULL_FRACTION_MAX:
        failures.append("null_retained_energy_fraction_above_maximum")
    if significant > NULL_FRACTION_MAX:
        failures.append("null_significant_fraction_above_maximum")
    return failures


def _metric(value: Any, name: str) -> float:
    raw = value[name] if isinstance(value, Mapping) else getattr(value, name)
    return float(raw)


def recovery_summary(
    values: Sequence[V2RecoverySeedResult],
) -> dict[str, Any]:
    """Return equal-seed means and Student-t intervals for final reporting."""
    result: dict[str, Any] = {}
    for name in _RECOVERY_FIELDS:
        array = np.asarray([getattr(item, name) for item in values], dtype=np.float64)
        mean = float(np.mean(array))
        if array.size == 3:
            half_width = 4.302652729911275 * float(np.std(array, ddof=1)) / math.sqrt(3)
            interval: list[float] | None = [mean - half_width, mean + half_width]
        else:
            interval = None
        result[name] = {"confidence_interval_95": interval, "mean": mean}
    return result


def recovery_result_failures(value: V2RecoverySeedResult) -> tuple[str, ...]:
    """Expose the unchanged hard gates for preflight and acceptance reuse."""
    return tuple(_recovery_failures(value))


def aggregate_recovery_failures(
    values: Sequence[V2RecoverySeedResult],
) -> tuple[str, ...]:
    """Apply every per-seed gate and the equal-seed mean gate."""
    if not values:
        return ("no_recovery_seed_results",)
    failures: list[str] = []
    for item in values:
        failures.extend(f"seed_{item.seed}:{reason}" for reason in _recovery_failures(item))
    means = {
        name: float(np.mean([getattr(item, name) for item in values])) for name in _RECOVERY_FIELDS
    }
    failures.extend(f"equal_seed_mean:{reason}" for reason in _recovery_failures(means))
    return tuple(failures)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _evidence_mapping(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("v2 scenario evidence is invalid JSON") from exc
    return _mapping(parsed, "v2 scenario evidence")


def _validate_evidence_json(value: Any, expected_sha256: Any) -> None:
    if not isinstance(value, str):
        raise ValueError("scenario evidence must be canonical JSON")
    evidence = _evidence_mapping(value)
    if value != _canonical_json(evidence):
        raise ValueError("scenario evidence must be canonically encoded")
    _sha256(expected_sha256, "scenario evidence sha256")
    if not hmac.compare_digest(_json_sha256(evidence), expected_sha256):
        raise ValueError("scenario evidence SHA-256 mismatch")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _score_metric(evidence: Mapping[str, Any], name: str) -> float:
    score = _mapping(evidence.get("score"), "scenario score")
    metrics = _mapping(score.get("metrics"), "scenario metrics")
    return _finite(metrics.get(name), name)


def _score_failures(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    score = _mapping(evidence.get("score"), "recovery score")
    failures = score.get("failures")
    if not isinstance(failures, list) or any(
        not isinstance(item, str) or not item.strip() for item in failures
    ):
        raise ValueError("recovery score failures must be a list of nonempty strings")
    return tuple(failures)


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


__all__ = [
    "V2CalibrationCandidate",
    "V2NullSeedResult",
    "V2RecoverySeedResult",
    "aggregate_recovery_failures",
    "recovery_result_failures",
    "recovery_summary",
]
