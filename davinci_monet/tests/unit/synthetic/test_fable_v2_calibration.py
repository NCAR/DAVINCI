"""Frozen multi-seed calibration contracts for FABLE v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec, spec_hash
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_gate
from davinci_monet.tests.synthetic.fable_v2_calibration import (
    V2CalibrationCandidate,
    V2NullSeedResult,
    V2RecoverySeedResult,
)
from davinci_monet.tests.synthetic.fable_v2_calibration_record import (
    load_v2_calibration_record,
    select_v2_calibration_policy,
    write_v2_calibration_record,
)
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
)
from davinci_monet.tests.synthetic.fable_v2_policy import (
    FableV2Policy,
    v2_calibration_policies,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import seed_roles
from davinci_monet.tests.synthetic.fable_v2_record_io import json_sha256
from davinci_monet.tests.synthetic.fable_v2_runner import V2ScenarioOutcome


def _identity(label: str) -> FrozenFileIdentity:
    return FrozenFileIdentity(
        f"/unit-only/{label}", hashlib.sha256(label.encode("ascii")).hexdigest()
    )


def _frozen_preregistration() -> FrozenPreregistration:
    return FrozenPreregistration(_identity("preregistration.json"), "f" * 64)


def _outcome(
    policy: FableV2Policy,
    seed: int,
    *,
    score_kind: Literal["recovery", "null"],
    field_correlation: float = 0.95,
    field_origin_slope: float = 1.0,
    field_nrmse: float = 0.20,
    aod_rmse_ratio: float = 0.50,
    full_target_aod_rmse_ratio: float = 0.70,
    excluded_fraction: float = 0.20,
    retained_energy: float = 0.03,
    significant_fraction: float = 0.02,
) -> V2ScenarioOutcome:
    splits: tuple[str, ...]
    spec = (
        SyntheticTuningSpec.synthetic_osse(seed)
        if score_kind == "recovery"
        else SyntheticTuningSpec.synthetic_osse_null(seed)
    )
    if score_kind == "recovery":
        score: dict[str, Any] = {
            "diagnostics": {
                "estimate_vs_unfiltered_in_span_nrmse": 0.10,
                "excluded_fraction": excluded_fraction,
            },
            "metrics": {
                "aod_rmse_ratio": aod_rmse_ratio,
                "field_correlation": field_correlation,
                "field_nrmse": field_nrmse,
                "field_origin_slope": field_origin_slope,
                "full_target_aod_rmse_ratio": full_target_aod_rmse_ratio,
            },
            "failures": [],
            "passed": True,
        }
        splits = ("calibration",)
        evaluation_identity: FrozenFileIdentity | None = _identity(
            f"evaluation-{policy.policy_id}-{seed}.json"
        )
        diagnostic_hash: str | None = "d" * 64
        oracle_nrmse: float | None = 0.03
        report_hash = "e" * 64
    else:
        score = {
            "metrics": {
                "null_retained_energy_fraction": retained_energy,
                "null_significant_fraction": significant_fraction,
            },
            "passed": retained_energy <= 0.10 and significant_fraction <= 0.10,
        }
        splits = ()
        evaluation_identity = None
        diagnostic_hash = None
        oracle_nrmse = None
        report_hash = json_sha256(score)
    common_identity = _identity(f"{policy.policy_id}-{seed}-{score_kind}")
    return V2ScenarioOutcome(
        seed=seed,
        scenario=spec.scenario,
        policy=policy,
        score_kind=score_kind,
        evaluation_splits=splits,
        spec_sha256=spec_hash(spec),
        scenario_manifest=common_identity,
        fitting_config=common_identity,
        fitting_manifest=common_identity,
        evaluation_config=evaluation_identity,
        evaluation_manifest=evaluation_identity,
        report_sha256=report_hash,
        diagnostic_report_sha256=diagnostic_hash,
        learned_basis_oracle_nrmse=oracle_nrmse,
        score=score,
        evidence_passed=True,
        elapsed_seconds=1.0,
        process_peak_rss_bytes=1024,
        resources=resource_gate(1.0, 1024),
    )


def _candidate(
    policy: FableV2Policy,
    *,
    nrmse: float = 0.20,
    aod_ratio: float = 0.50,
    correlations: tuple[float, float, float] = (0.95, 0.95, 0.95),
    retained: tuple[float, float, float] = (0.03, 0.03, 0.03),
) -> V2CalibrationCandidate:
    roles = seed_roles()
    recovery = tuple(
        V2RecoverySeedResult.from_outcome(
            _outcome(
                policy,
                seed,
                score_kind="recovery",
                field_correlation=correlations[index],
                field_nrmse=nrmse,
                aod_rmse_ratio=aod_ratio,
            )
        )
        for index, seed in enumerate(roles["calibration_recovery"])
    )
    null = tuple(
        V2NullSeedResult.from_outcome(
            _outcome(
                policy,
                seed,
                score_kind="null",
                retained_energy=retained[index],
            )
        )
        for index, seed in enumerate(roles["calibration_null"])
    )
    return V2CalibrationCandidate(policy, recovery, null)


def _select(
    candidates: tuple[V2CalibrationCandidate, ...],
) -> Any:
    return select_v2_calibration_policy(
        candidates,
        _frozen_preregistration(),
        _identity("calibration-attempt.json"),
        (_identity("recovery-lock.json"), _identity("null-lock.json")),
    )


def test_three_by_three_aggregation_and_above_one_null_energy_hard_failure() -> None:
    policy = v2_calibration_policies()[0]
    passing = _candidate(policy, retained=(0.02, 0.05, 0.08))

    assert len(passing.recovery) == 3
    assert len(passing.null) == 3
    assert passing.aggregate["field_nrmse"] == pytest.approx(0.20)
    assert passing.aggregate["null_retained_energy_fraction"] == pytest.approx(0.05)
    assert passing.eligible

    rejected = _candidate(policy, retained=(0.02, 1.25, 0.03))
    failed_seed = seed_roles()["calibration_null"][1]
    assert rejected.null[1].null_retained_energy_fraction == pytest.approx(1.25)
    assert not rejected.eligible
    assert (
        f"null_seed_{failed_seed}:null_retained_energy_fraction_above_maximum"
        in rejected.rejection_reasons
    )


def test_any_failed_seed_rejects_even_when_equal_seed_mean_passes() -> None:
    policy = v2_calibration_policies()[0]
    candidate = _candidate(policy, correlations=(0.89, 0.99, 0.99))
    failed_seed = seed_roles()["calibration_recovery"][0]

    assert candidate.aggregate["field_correlation"] > 0.90
    assert not candidate.eligible
    assert (
        f"recovery_seed_{failed_seed}:field_correlation_below_minimum"
        in candidate.rejection_reasons
    )
    assert not any(
        reason == "equal_seed_mean:field_correlation_below_minimum"
        for reason in candidate.rejection_reasons
    )


def test_false_embedded_recovery_score_rejects_otherwise_passing_seed() -> None:
    policy = v2_calibration_policies()[0]
    roles = seed_roles()
    recovery = []
    for index, seed in enumerate(roles["calibration_recovery"]):
        outcome = _outcome(policy, seed, score_kind="recovery")
        if index == 1:
            score = dict(outcome.score)
            score.update(passed=False, failures=["required diagnostic is nonfinite"])
            outcome = replace(outcome, score=score)
        recovery.append(V2RecoverySeedResult.from_outcome(outcome))
    candidate = V2CalibrationCandidate(
        policy,
        tuple(recovery),
        _candidate(policy).null,
    )

    assert not candidate.eligible
    assert any(
        "scientific_score_failed:required diagnostic is nonfinite" in reason
        for reason in candidate.rejection_reasons
    )


def test_exact_menu_deterministic_ranking_and_no_eligible_outcome() -> None:
    simple, offset = v2_calibration_policies()
    simple_candidate = _candidate(simple, nrmse=0.20, aod_ratio=0.40)
    offset_candidate = _candidate(offset, nrmse=0.19, aod_ratio=0.69)

    ranked = _select((offset_candidate, simple_candidate))
    assert ranked.selected_policy_id == offset.policy_id
    tied = _select((_candidate(offset), _candidate(simple)))
    assert tied.selected_policy_id == simple.policy_id

    with pytest.raises(ValueError, match="exact .*policies"):
        _select((simple_candidate,))
    with pytest.raises(ValueError, match="no v2 calibration candidate"):
        _select(
            (
                _candidate(simple, retained=(0.11, 0.11, 0.11)),
                _candidate(offset, retained=(0.11, 0.11, 0.11)),
            )
        )


def test_calibration_record_is_canonical_hash_bound_and_write_once(tmp_path: Path) -> None:
    simple, offset = v2_calibration_policies()
    record = _select((_candidate(simple), _candidate(offset, nrmse=0.25)))
    path = write_v2_calibration_record(tmp_path / "calibration.json", record)
    raw = path.read_text(encoding="ascii")

    assert (
        raw
        == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    )
    assert path.stat().st_mode & 0o777 == 0o444
    assert load_v2_calibration_record(path) == record
    with pytest.raises(FileExistsError, match="already exists"):
        write_v2_calibration_record(path, record)

    envelope = json.loads(raw)
    envelope["record"]["selected_policy_id"] = offset.policy_id
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="envelope is invalid"):
        load_v2_calibration_record(tampered)
