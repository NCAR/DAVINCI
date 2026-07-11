"""Frozen development evidence and explicit approval for FABLE v2 calibration."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic.fable_v2_development import (
    DEVELOPMENT_REPORT_SCHEMA,
    candidate_assessments,
)
from davinci_monet.tests.synthetic.fable_v2_evidence import validate_v2_scenario_evidence
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    V2Preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import V2GenerationPrerequisites
from davinci_monet.tests.synthetic.fable_v2_lock_evidence import (
    validate_v2_generation_lock_identity,
)
from davinci_monet.tests.synthetic.fable_v2_policy import (
    FableV2Policy,
    v2_calibration_policies,
    v2_development_policies,
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

DEVELOPMENT_APPROVAL_SCHEMA = "fable-v2-development-approval-v1"
DEVELOPMENT_REPORT_BINDING = "development_report"
DEVELOPMENT_APPROVAL_BINDING = "development_approval"


@dataclass(frozen=True)
class V2DevelopmentApproval:
    """Write-once authorization bound to one completed development report."""

    development_report: FrozenFileIdentity
    eligible_policy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        allowed = tuple(policy.policy_id for policy in v2_calibration_policies())
        if (
            not self.eligible_policy_ids
            or tuple(item for item in allowed if item in self.eligible_policy_ids)
            != self.eligible_policy_ids
            or len(set(self.eligible_policy_ids)) != len(self.eligible_policy_ids)
        ):
            raise ValueError("development approval has an invalid eligible policy subset")

    def normalized(self) -> dict[str, Any]:
        return {
            "approval_basis": "explicit_user_instruction_to_complete_fable_v2",
            "authority": "user",
            "cycle_id": CYCLE_ID,
            "development_report": self.development_report.normalized(),
            "disposition": "approved_for_calibration",
            "eligible_policy_ids": list(self.eligible_policy_ids),
            "protocol_sha256": protocol_sha256(),
            "schema_version": DEVELOPMENT_APPROVAL_SCHEMA,
        }


def validate_v2_development_report(path: str | Path, *, verify_files: bool) -> tuple[str, ...]:
    """Recompute the completed report and return its exact calibration subset."""
    source = Path(path).expanduser().resolve()
    raw = source.read_text(encoding="ascii")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid v2 development report JSON") from exc
    if not isinstance(value, Mapping) or raw != canonical_json(value) + "\n":
        raise ValueError("v2 development report is not canonically encoded")
    expected_keys = {
        "candidate_assessments",
        "generation_lock",
        "mode",
        "policies",
        "runs",
        "schema_version",
        "seeds",
        "status",
    }
    seeds = seed_roles()["development"]
    if (
        set(value) != expected_keys
        or value.get("schema_version") != DEVELOPMENT_REPORT_SCHEMA
        or value.get("mode") != "full"
        or value.get("status") != "completed"
        or tuple(value.get("seeds", ())) != seeds
        or value.get("policies") != [policy.normalized() for policy in v2_development_policies()]
    ):
        raise ValueError("v2 development report is not a completed fixed campaign")
    lock = _identity(value.get("generation_lock"), "development generation lock")
    root = validate_v2_generation_lock_identity(
        lock,
        "development",
        seeds,
        V2GenerationPrerequisites(),
    )
    if source != root / "development.json":
        raise ValueError("v2 development report is outside its generation root")
    runs = value.get("runs")
    if not isinstance(runs, list) or not all(isinstance(item, Mapping) for item in runs):
        raise ValueError("v2 development report runs must be mappings")
    _validate_generation_runs(runs, root, seeds)
    _validate_policy_runs(runs, root, seeds, verify_files=verify_files)
    normalized_runs = [dict(item) for item in runs]
    expected_assessments = candidate_assessments(normalized_runs)
    if value.get("candidate_assessments") != expected_assessments:
        raise ValueError("v2 development candidate assessments changed")
    eligible = tuple(
        str(item["policy_id"])
        for item in expected_assessments
        if item.get("eligible_for_calibration") is True
    )
    allowed = tuple(policy.policy_id for policy in v2_calibration_policies())
    if not eligible or tuple(item for item in allowed if item in eligible) != eligible:
        raise ValueError("v2 development produced no valid calibration subset")
    return eligible


def write_v2_development_approval(
    destination: str | Path, development_report: str | Path
) -> FrozenFileIdentity:
    """Publish the explicit user-authorized transition to frozen calibration."""
    eligible = validate_v2_development_report(development_report, verify_files=True)
    report = FrozenFileIdentity.capture(development_report)
    os.chmod(report.path, 0o444)
    report = FrozenFileIdentity.capture(report.path)
    approval = V2DevelopmentApproval(report, eligible)
    path = write_record_once(destination, approval.normalized(), "development approval")
    return FrozenFileIdentity.capture(path)


def load_v2_development_approval(path: str | Path) -> V2DevelopmentApproval:
    """Load canonical approval bytes and recompute all derived fields."""
    value = load_record_value(path, "development approval")
    if (
        value.get("cycle_id") != CYCLE_ID
        or value.get("protocol_sha256") != protocol_sha256()
        or value.get("schema_version") != DEVELOPMENT_APPROVAL_SCHEMA
        or value.get("authority") != "user"
        or value.get("approval_basis") != "explicit_user_instruction_to_complete_fable_v2"
        or value.get("disposition") != "approved_for_calibration"
    ):
        raise ValueError("v2 development approval protocol identity changed")
    report = _identity(value.get("development_report"), "approved development report")
    raw_ids = value.get("eligible_policy_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
        raise ValueError("v2 development approval policy subset is invalid")
    approval = V2DevelopmentApproval(report, tuple(raw_ids))
    if approval.normalized() != value:
        raise ValueError("v2 development approval derived fields changed")
    return approval


def validate_v2_development_approval(
    identity: FrozenFileIdentity, *, verify_files: bool
) -> V2DevelopmentApproval:
    """Verify approval bytes, report bytes, evidence, and eligible subset."""
    identity.verify()
    approval = load_v2_development_approval(identity.path)
    approval.development_report.verify()
    eligible = validate_v2_development_report(
        approval.development_report.path,
        verify_files=verify_files,
    )
    if approval.eligible_policy_ids != eligible:
        raise ValueError("v2 approval eligible subset differs from development evidence")
    return approval


def eligible_v2_calibration_policies(
    preregistration: V2Preregistration, *, verify_files: bool = True
) -> tuple[FableV2Policy, ...]:
    """Resolve the exact development-approved subset bound into preregistration."""
    bindings = {item.name: item for item in preregistration.configs}
    try:
        report_binding = bindings[DEVELOPMENT_REPORT_BINDING]
        approval_binding = bindings[DEVELOPMENT_APPROVAL_BINDING]
    except KeyError as exc:
        raise ValueError("v2 preregistration lacks development approval bindings") from exc
    report = FrozenFileIdentity(report_binding.path, report_binding.sha256)
    approval_identity = FrozenFileIdentity(approval_binding.path, approval_binding.sha256)
    approval = validate_v2_development_approval(
        approval_identity,
        verify_files=verify_files,
    )
    if approval.development_report.normalized() != report.normalized():
        raise ValueError("v2 preregistration development bindings disagree")
    policies = {policy.policy_id: policy for policy in v2_calibration_policies()}
    return tuple(policies[policy_id] for policy_id in approval.eligible_policy_ids)


def _validate_generation_runs(
    runs: Sequence[Mapping[str, Any]], root: Path, seeds: tuple[int, ...]
) -> None:
    entries = [item for item in runs if item.get("phase") == "generation"]
    if len(entries) != len(seeds) or tuple(item.get("seed") for item in entries) != seeds:
        raise ValueError("v2 development generation entries changed")
    for item, seed in zip(entries, seeds, strict=True):
        manifest = _identity(item.get("manifest"), "development scenario manifest")
        expected = root / "bundles" / f"seed-{seed}" / "scenario.json"
        if item.get("status") != "completed" or Path(manifest.path) != expected:
            raise ValueError("v2 development scenario generation is incomplete")
        manifest.verify()


def _validate_policy_runs(
    runs: Sequence[Mapping[str, Any]],
    root: Path,
    seeds: tuple[int, ...],
    *,
    verify_files: bool,
) -> None:
    entries = [item for item in runs if item.get("phase") != "generation"]
    policies = v2_development_policies()
    expected_pairs = [(policy.policy_id, seed) for policy in policies for seed in seeds]
    actual_pairs = [(item.get("policy_id"), item.get("seed")) for item in entries]
    if actual_pairs != expected_pairs:
        raise ValueError("v2 development policy/seed run matrix changed")
    by_id = {policy.policy_id: policy for policy in policies}
    for item in entries:
        policy_id = str(item["policy_id"])
        seed = int(item["seed"])
        outcome = item.get("outcome")
        if not isinstance(outcome, Mapping):
            raise ValueError("v2 development run lacks canonical scenario evidence")
        passed = outcome.get("passed")
        if (
            not isinstance(passed, bool)
            or item.get("passed") is not passed
            or item.get("status") != ("completed" if passed else "failed")
        ):
            raise ValueError("v2 development run disposition differs from its outcome")
        validate_v2_scenario_evidence(
            outcome,
            SyntheticTuningSpec.synthetic_osse(seed),
            by_id[policy_id],
            score_kind="recovery",
            evaluation_splits=("development_test",),
            verify_files=verify_files,
        )
        run_root = root / "runs" / policy_id / f"seed-{seed}"
        expected_paths = {
            "scenario_manifest": root / "bundles" / f"seed-{seed}" / "scenario.json",
            "fitting_manifest": run_root / "output" / "manifest.json",
            "evaluation_manifest": run_root / "evaluation" / "manifest.json",
        }
        for name, expected in expected_paths.items():
            identity = _identity(outcome.get(name), f"development {name}")
            if Path(identity.path) != expected:
                raise ValueError("v2 development outcome is outside its locked run root")


def _identity(value: Any, name: str) -> FrozenFileIdentity:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a file identity")
    try:
        return FrozenFileIdentity(str(value["path"]), str(value["sha256"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid") from exc


def file_sha256(path: str | Path) -> str:
    """Return a file hash for preregistration evidence commands."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "DEVELOPMENT_APPROVAL_BINDING",
    "DEVELOPMENT_APPROVAL_SCHEMA",
    "DEVELOPMENT_REPORT_BINDING",
    "V2DevelopmentApproval",
    "eligible_v2_calibration_policies",
    "file_sha256",
    "load_v2_development_approval",
    "validate_v2_development_approval",
    "validate_v2_development_report",
    "write_v2_development_approval",
]
