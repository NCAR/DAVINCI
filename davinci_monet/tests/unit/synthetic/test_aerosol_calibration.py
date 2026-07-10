"""Tests for deterministic synthetic-only calibration policy selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from davinci_monet.tests.synthetic._aerosol_contracts import canonical_json
from davinci_monet.tests.synthetic.aerosol_calibration import (
    EVIDENCE_HASH_KEYS,
    CalibrationCandidate,
    CalibrationEvidence,
    CandidateMetrics,
    ScientificPolicy,
    calibration_record_sha256,
    canonical_calibration_json,
    load_calibration_record,
    select_calibration_policy,
    verify_calibration_record,
    write_calibration_record,
)


def _metrics(**overrides: float) -> CandidateMetrics:
    values = {
        "field_correlation": 0.96,
        "field_origin_slope": 1.02,
        "field_nrmse": 0.20,
        "aod_rmse_ratio": 0.55,
        "full_target_aod_rmse_ratio": 0.70,
        "null_retained_energy_fraction": 0.04,
        "null_significant_fraction": 0.03,
    }
    values.update(overrides)
    return CandidateMetrics(**values)


def _candidate(
    policy_id: str,
    *,
    simplicity_rank: int = 0,
    metrics: CandidateMetrics | None = None,
) -> CalibrationCandidate:
    return CalibrationCandidate(
        policy=ScientificPolicy(policy_id=policy_id, simplicity_rank=simplicity_rank),
        metrics=metrics or _metrics(),
        evidence=CalibrationEvidence(
            calibration_seed=101,
            null_seed=202,
            calibration_scenario="writer_ci",
            null_scenario="calibration_null",
            calibration_split="calibration",
            hashes=tuple((name, "a" * 64) for name in EVIDENCE_HASH_KEYS),
        ),
    )


def test_null_controls_hard_reject_better_recovery_candidates() -> None:
    energy_failure = _candidate(
        "best-recovery-energy-failure",
        metrics=_metrics(
            field_nrmse=0.05,
            aod_rmse_ratio=0.20,
            null_retained_energy_fraction=0.100001,
        ),
    )
    significance_failure = _candidate(
        "best-recovery-significance-failure",
        metrics=_metrics(
            field_nrmse=0.06,
            aod_rmse_ratio=0.21,
            null_significant_fraction=0.100001,
        ),
    )
    eligible = _candidate(
        "eligible",
        metrics=_metrics(
            field_nrmse=0.25,
            aod_rmse_ratio=0.60,
            null_retained_energy_fraction=0.10,
            null_significant_fraction=0.10,
        ),
    )

    record = select_calibration_policy([energy_failure, eligible, significance_failure])

    assert record.selected_policy_id == "eligible"
    assessments = {item["policy"]["policy_id"]: item for item in record.normalized()["candidates"]}
    assert assessments["best-recovery-energy-failure"]["rejection_reasons"] == [
        "null_retained_energy_fraction_above_maximum"
    ]
    assert assessments["best-recovery-significance-failure"]["rejection_reasons"] == [
        "null_significant_fraction_above_maximum"
    ]
    assert assessments["eligible"]["eligible"] is True


def test_recovery_thresholds_are_hard_rejections() -> None:
    failing = _candidate("fails-recovery", metrics=_metrics(field_correlation=0.899))
    passing = _candidate("passes", metrics=_metrics(field_nrmse=0.30))

    record = select_calibration_policy([failing, passing])

    assert record.selected_policy_id == "passes"
    with pytest.raises(ValueError, match="no calibration candidate"):
        select_calibration_policy([failing])


def test_selection_ties_and_hashes_are_deterministic() -> None:
    complex_policy = _candidate("complex", simplicity_rank=3)
    simple_policy = _candidate("simple", simplicity_rank=1)

    forward = select_calibration_policy([complex_policy, simple_policy])
    reverse = select_calibration_policy([simple_policy, complex_policy])

    assert forward == reverse
    assert forward.selected_policy_id == "simple"
    assert canonical_calibration_json(forward) == canonical_calibration_json(reverse)
    assert calibration_record_sha256(forward) == calibration_record_sha256(reverse)
    assert len(calibration_record_sha256(forward)) == 64

    lexical = select_calibration_policy(
        [_candidate("zeta", simplicity_rank=1), _candidate("alpha", simplicity_rank=1)]
    )
    assert lexical.selected_policy_id == "alpha"
    with pytest.raises(FrozenInstanceError):
        lexical.selected_policy.simplicity_rank = 5  # type: ignore[misc]


def test_calibration_evidence_requires_complete_sha256_identity() -> None:
    with pytest.raises(ValueError, match="every required hash"):
        CalibrationEvidence(
            calibration_seed=1,
            null_seed=2,
            calibration_scenario="writer_ci",
            null_scenario="calibration_null",
            calibration_split="calibration",
            hashes=((EVIDENCE_HASH_KEYS[0], "a" * 64),),
        )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CalibrationEvidence(
            calibration_seed=1,
            null_seed=2,
            calibration_scenario="writer_ci",
            null_scenario="calibration_null",
            calibration_split="calibration",
            hashes=tuple((name, "not-a-hash") for name in EVIDENCE_HASH_KEYS),
        )


def test_recovery_ranking_is_lexicographic_before_simplicity() -> None:
    best_nrmse = _candidate(
        "best-nrmse",
        simplicity_rank=9,
        metrics=_metrics(field_nrmse=0.10, aod_rmse_ratio=0.69, field_origin_slope=1.19),
    )
    assert (
        select_calibration_policy(
            [
                best_nrmse,
                _candidate(
                    "better-later-metrics",
                    metrics=_metrics(
                        field_nrmse=0.11,
                        aod_rmse_ratio=0.20,
                        field_origin_slope=1.0,
                    ),
                ),
            ]
        ).selected_policy_id
        == "best-nrmse"
    )

    best_aod_ratio = _candidate(
        "best-aod-ratio",
        simplicity_rank=9,
        metrics=_metrics(field_nrmse=0.10, aod_rmse_ratio=0.30, field_origin_slope=1.19),
    )
    assert (
        select_calibration_policy(
            [
                best_aod_ratio,
                _candidate(
                    "better-slope",
                    metrics=_metrics(
                        field_nrmse=0.10,
                        aod_rmse_ratio=0.31,
                        field_origin_slope=1.0,
                    ),
                ),
            ]
        ).selected_policy_id
        == "best-aod-ratio"
    )

    best_slope = _candidate(
        "best-slope",
        simplicity_rank=9,
        metrics=_metrics(field_nrmse=0.10, aod_rmse_ratio=0.30, field_origin_slope=1.01),
    )
    assert (
        select_calibration_policy(
            [
                best_slope,
                _candidate(
                    "simpler",
                    metrics=_metrics(
                        field_nrmse=0.10,
                        aod_rmse_ratio=0.30,
                        field_origin_slope=1.02,
                    ),
                ),
            ]
        ).selected_policy_id
        == "best-slope"
    )


def test_atomic_writer_refuses_overwrite_and_verifies_hashes(tmp_path: Path) -> None:
    record = select_calibration_policy([_candidate("selected")])
    path = write_calibration_record(tmp_path / "calibration.json", record)

    assert load_calibration_record(path) == record
    assert verify_calibration_record(path, calibration_record_sha256(record)) == record
    with pytest.raises(FileExistsError, match="already exists"):
        write_calibration_record(path, record)
    with pytest.raises(ValueError, match="expected SHA-256"):
        verify_calibration_record(path, "0" * 64)

    document = json.loads(path.read_text(encoding="ascii"))
    document["record"]["candidates"][0]["metrics"]["field_nrmse"] = 0.21
    path.write_text(canonical_json(document) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_calibration_record(path)


def test_loader_rejects_rehashed_selection_mismatch(tmp_path: Path) -> None:
    preferred = _candidate("preferred", metrics=_metrics(field_nrmse=0.10))
    other = _candidate("other", metrics=_metrics(field_nrmse=0.20))
    record = select_calibration_policy([preferred, other])
    path = write_calibration_record(tmp_path / "calibration.json", record)
    document = json.loads(path.read_text(encoding="ascii"))
    document["record"]["selected_policy_id"] = "other"
    record_json = canonical_json(document["record"])
    document["record_sha256"] = hashlib.sha256(record_json.encode("ascii")).hexdigest()
    path.write_text(canonical_json(document) + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="scientific policy or selection mismatch"):
        load_calibration_record(path)


def test_policy_and_metrics_validate_scientific_domains() -> None:
    with pytest.raises(ValueError, match="twice the maximum"):
        ScientificPolicy(policy_id="short", min_segment_days=359)
    with pytest.raises(ValueError, match="field_nrmse must be nonnegative"):
        replace(_metrics(), field_nrmse=-0.01)
    with pytest.raises(ValueError, match="unique"):
        select_calibration_policy([_candidate("same"), _candidate("same")])
