"""Repeatable, diagnostic-only development campaign for FABLE v2."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_gate
from davinci_monet.tests.synthetic.fable_v2_bundle import generate_v2_scenario_bundle
from davinci_monet.tests.synthetic.fable_v2_calibration import (
    V2RecoverySeedResult,
    aggregate_recovery_failures,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import prepare_v2_generation
from davinci_monet.tests.synthetic.fable_v2_policy import (
    v2_calibration_policies,
    v2_development_policies,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import seed_roles
from davinci_monet.tests.synthetic.fable_v2_runner import run_v2_scenario

DEVELOPMENT_REPORT_SCHEMA = "fable-v2-development-report-v1"


def run_v2_development(root: str | Path, *, dry_run: bool = False) -> Path:
    """Lock and optionally execute the fixed repeatable development campaign."""
    seeds = seed_roles()["development"]
    lock = prepare_v2_generation(root, "development", seeds)
    report_path = lock.root / "development.json"
    report: dict[str, Any] = {
        "generation_lock": lock.file.normalized(),
        "mode": "dry_run" if dry_run else "full",
        "policies": [item.normalized() for item in v2_development_policies()],
        "runs": [],
        "schema_version": DEVELOPMENT_REPORT_SCHEMA,
        "seeds": list(seeds),
        "status": "planned" if dry_run else "running",
    }
    _write_json(report_path, report)
    if dry_run:
        return report_path
    bundle_roots: dict[int, Path] = {}
    generation_elapsed: dict[int, float] = {}
    for seed in seeds:
        generation_entry: dict[str, Any] = {
            "phase": "generation",
            "seed": seed,
            "status": "running",
        }
        report["runs"].append(generation_entry)
        _write_json(report_path, report)
        bundle_root = lock.root / "bundles" / f"seed-{seed}"
        started = time.perf_counter()
        try:
            identity = generate_v2_scenario_bundle(lock, SyntheticTuningSpec.synthetic_osse(seed))
            bundle_roots[seed] = bundle_root
            generation_elapsed[seed] = time.perf_counter() - started
            generation_entry.update(manifest=identity.normalized(), status="completed")
        except Exception as exc:
            generation_entry.update(error=f"{type(exc).__name__}: {exc}", status="failed")
        _write_json(report_path, report)
    for policy in v2_development_policies():
        for seed in seeds:
            run_entry: dict[str, Any] = {
                "policy_id": policy.policy_id,
                "seed": seed,
                "status": "running",
            }
            report["runs"].append(run_entry)
            _write_json(report_path, report)
            if seed not in bundle_roots:
                run_entry.update(
                    error="blocked_by_generation_failure", passed=False, status="failed"
                )
                _write_json(report_path, report)
                continue
            try:
                outcome = run_v2_scenario(
                    lock.root / "runs" / policy.policy_id / f"seed-{seed}",
                    SyntheticTuningSpec.synthetic_osse(seed),
                    policy,
                    score_kind="recovery",
                    lock=lock,
                )
                total_elapsed = generation_elapsed[seed] + outcome.elapsed_seconds
                outcome = replace(
                    outcome,
                    elapsed_seconds=total_elapsed,
                    resources=resource_gate(total_elapsed, outcome.process_peak_rss_bytes),
                )
                run_entry["outcome"] = outcome.normalized()
                run_entry["passed"] = outcome.passed
                run_entry["status"] = "completed" if outcome.passed else "failed"
            except Exception as exc:
                run_entry.update(
                    error=f"{type(exc).__name__}: {exc}",
                    passed=False,
                    status="failed",
                )
            _write_json(report_path, report)
    assessments = candidate_assessments(report["runs"])
    report["candidate_assessments"] = assessments
    report["status"] = "completed" if any(item["eligible"] for item in assessments) else "failed"
    _write_json(report_path, report)
    return report_path


def candidate_assessments(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    calibration_ids = {policy.policy_id for policy in v2_calibration_policies()}
    for policy in v2_development_policies():
        selected = [
            item
            for item in runs
            if item.get("phase") != "generation" and item.get("policy_id") == policy.policy_id
        ]
        failures: list[str] = []
        results: list[V2RecoverySeedResult] = []
        selectable = policy.policy_id in calibration_ids
        if not selectable:
            failures.append("diagnostic_only_policy")
        if len(selected) != len(seed_roles()["development"]):
            failures.append("incomplete_development_seed_set")
        for item in selected:
            try:
                results.append(_result_from_run(item))
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"seed_{item.get('seed')}:invalid_evidence:{exc}")
            if item.get("status") != "completed" or item.get("passed") is not True:
                failures.append(f"seed_{item.get('seed')}:per_seed_gate_failed")
        aggregate: dict[str, Any] = {}
        if len(results) == len(seed_roles()["development"]):
            aggregate = {
                name: sum(getattr(item, name) for item in results) / len(results)
                for name in (
                    "field_correlation",
                    "field_origin_slope",
                    "field_nrmse",
                    "aod_rmse_ratio",
                    "full_target_aod_rmse_ratio",
                    "excluded_fraction",
                )
            }
            failures.extend(aggregate_recovery_failures(results))
        assessments.append(
            {
                "aggregate": aggregate,
                "eligible": selectable and not failures,
                "eligible_for_calibration": selectable and not failures,
                "policy_id": policy.policy_id,
                "rejection_reasons": list(dict.fromkeys(failures)),
            }
        )
    return assessments


_candidate_assessments = candidate_assessments


def _result_from_run(value: Mapping[str, Any]) -> V2RecoverySeedResult:
    evidence = _mapping(value["outcome"], "development outcome")
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return V2RecoverySeedResult(payload, _sha256_json(evidence))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_json(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


__all__ = [
    "DEVELOPMENT_REPORT_SCHEMA",
    "candidate_assessments",
    "run_v2_development",
]
