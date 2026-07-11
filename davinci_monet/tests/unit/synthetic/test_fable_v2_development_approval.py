"""Frozen development-report and approval contracts for FABLE v2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from davinci_monet.tests.synthetic import fable_v2_development_approval as approval
from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_policy import v2_development_policies
from davinci_monet.tests.synthetic.fable_v2_protocol import seed_roles
from davinci_monet.tests.synthetic.fable_v2_record_io import canonical_json


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _assessments(*, eligible: bool = True) -> list[dict[str, Any]]:
    policies = v2_development_policies()
    return [
        {
            "aggregate": {"field_nrmse": 0.5 if index == 0 else 0.2},
            "eligible": eligible and index == 1,
            "eligible_for_calibration": eligible and index == 1,
            "policy_id": policy.policy_id,
            "rejection_reasons": [] if eligible and index == 1 else ["diagnostic_or_failed"],
        }
        for index, policy in enumerate(policies)
    ]


def _report(tmp_path: Path, *, status: str = "completed") -> Path:
    root = tmp_path / "development"
    root.mkdir()
    lock = root / "generation-lock.json"
    lock.write_text("{}\n", encoding="ascii")
    seeds = seed_roles()["development"]
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        scenario = root / "bundles" / f"seed-{seed}" / "scenario.json"
        scenario.parent.mkdir(parents=True)
        scenario.write_text("{}\n", encoding="ascii")
        runs.append(
            {
                "manifest": _identity(scenario),
                "phase": "generation",
                "seed": seed,
                "status": "completed",
            }
        )
    for index, policy in enumerate(v2_development_policies()):
        for seed in seeds:
            run_root = root / "runs" / policy.policy_id / f"seed-{seed}"
            passed = index == 1
            runs.append(
                {
                    "outcome": {
                        "evaluation_manifest": {
                            "path": str((run_root / "evaluation/manifest.json").resolve()),
                            "sha256": "1" * 64,
                        },
                        "fitting_manifest": {
                            "path": str((run_root / "output/manifest.json").resolve()),
                            "sha256": "2" * 64,
                        },
                        "passed": passed,
                        "scenario_manifest": _identity(
                            root / "bundles" / f"seed-{seed}" / "scenario.json"
                        ),
                    },
                    "passed": passed,
                    "policy_id": policy.policy_id,
                    "seed": seed,
                    "status": "completed" if passed else "failed",
                }
            )
    value = {
        "candidate_assessments": _assessments(eligible=status == "completed"),
        "generation_lock": _identity(lock),
        "mode": "full",
        "policies": [policy.normalized() for policy in v2_development_policies()],
        "runs": runs,
        "schema_version": approval.DEVELOPMENT_REPORT_SCHEMA,
        "seeds": list(seeds),
        "status": status,
    }
    path = root / "development.json"
    path.write_text(canonical_json(value) + "\n", encoding="ascii")
    return path


def _stub_deep_validation(
    monkeypatch: pytest.MonkeyPatch, report: Path, *, eligible: bool = True
) -> None:
    monkeypatch.setattr(
        approval,
        "validate_v2_generation_lock_identity",
        lambda *args, **kwargs: report.parent,
    )
    monkeypatch.setattr(approval, "validate_v2_scenario_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        approval,
        "candidate_assessments",
        lambda runs: _assessments(eligible=eligible),
    )


def test_completed_report_writes_one_canonical_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path)
    _stub_deep_validation(monkeypatch, report)

    assert approval.validate_v2_development_report(report, verify_files=False) == (
        "v2-joint-seasonal",
    )
    identity = approval.write_v2_development_approval(tmp_path / "approval.json", report)
    loaded = approval.validate_v2_development_approval(identity, verify_files=False)

    assert loaded.eligible_policy_ids == ("v2-joint-seasonal",)
    assert loaded.development_report == FrozenFileIdentity.capture(report)
    with pytest.raises(FileExistsError, match="already exists"):
        approval.write_v2_development_approval(identity.path, report)


def test_report_failure_or_missing_eligibility_blocks_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path, status="failed")
    _stub_deep_validation(monkeypatch, report, eligible=False)

    with pytest.raises(ValueError, match="completed fixed campaign"):
        approval.validate_v2_development_report(report, verify_files=False)


def test_approval_detects_development_report_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path)
    _stub_deep_validation(monkeypatch, report)
    identity = approval.write_v2_development_approval(tmp_path / "approval.json", report)

    report.chmod(0o644)
    report.write_text("{}\n", encoding="ascii")
    with pytest.raises(ValueError, match="frozen file identity changed"):
        approval.validate_v2_development_approval(identity, verify_files=False)


def test_preregistration_requires_both_bound_development_files() -> None:
    from davinci_monet.tests.synthetic.fable_v2_freeze import (
        V2FileBinding,
        V2Preregistration,
        V2TestEvidence,
    )

    preregistration = V2Preregistration(
        generator_schema_version="schema",
        generator_spec_sha256="1" * 64,
        thresholds_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        configs=(V2FileBinding("fit", "fit.yaml", "5" * 64),),
        tests=(V2TestEvidence("unit", "pytest", "6" * 64),),
    )
    with pytest.raises(ValueError, match="development approval bindings"):
        approval.eligible_v2_calibration_policies(preregistration)


def test_preregistration_resolves_only_approved_policy_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from davinci_monet.tests.synthetic.fable_v2_freeze import (
        V2FileBinding,
        V2Preregistration,
        V2TestEvidence,
    )

    report_path = tmp_path / "development.json"
    approval_path = tmp_path / "approval.json"
    report_path.write_text("report\n", encoding="ascii")
    approval_path.write_text("approval\n", encoding="ascii")
    report_identity = FrozenFileIdentity.capture(report_path)
    expected = approval.V2DevelopmentApproval(
        report_identity,
        ("v2-joint-seasonal-offset",),
    )
    monkeypatch.setattr(
        approval,
        "validate_v2_development_approval",
        lambda identity, verify_files: expected,
    )
    preregistration = V2Preregistration(
        generator_schema_version="schema",
        generator_spec_sha256="1" * 64,
        thresholds_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        configs=(
            V2FileBinding.capture(approval.DEVELOPMENT_REPORT_BINDING, report_path),
            V2FileBinding.capture(approval.DEVELOPMENT_APPROVAL_BINDING, approval_path),
        ),
        tests=(V2TestEvidence("unit", "pytest", "6" * 64),),
    )

    policies = approval.eligible_v2_calibration_policies(preregistration)

    assert [policy.policy_id for policy in policies] == ["v2-joint-seasonal-offset"]
