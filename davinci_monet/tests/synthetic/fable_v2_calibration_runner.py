"""Frozen role orchestration for the FABLE v2 multi-seed calibration."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_gate
from davinci_monet.tests.synthetic.fable_v2_attempts import claim_v2_phase_attempt
from davinci_monet.tests.synthetic.fable_v2_bundle import generate_v2_scenario_bundle
from davinci_monet.tests.synthetic.fable_v2_calibration import (
    V2CalibrationCandidate,
    V2NullSeedResult,
    V2RecoverySeedResult,
)
from davinci_monet.tests.synthetic.fable_v2_calibration_record import (
    select_v2_calibration_policy,
    validate_v2_calibration_record,
    write_v2_calibration_record,
)
from davinci_monet.tests.synthetic.fable_v2_development_approval import (
    eligible_v2_calibration_policies,
)
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
    V2Preregistration,
    verify_v2_preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import (
    V2GenerationLock,
    V2GenerationPrerequisites,
    prepare_v2_generation,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import seed_roles
from davinci_monet.tests.synthetic.fable_v2_rejection import (
    V2FailureClassification,
    rejection_failure,
    write_v2_phase_rejection,
)
from davinci_monet.tests.synthetic.fable_v2_runner import (
    V2ScenarioOutcome,
    run_v2_scenario,
)

ScenarioRunner = Callable[..., V2ScenarioOutcome]
BundleGenerator = Callable[[V2GenerationLock, SyntheticTuningSpec], FrozenFileIdentity]


def run_v2_calibration(
    work_root: str | Path,
    destination: str | Path,
    current: V2Preregistration,
    frozen: FrozenPreregistration,
    *,
    dry_run: bool = False,
    scenario_runner: ScenarioRunner = run_v2_scenario,
    bundle_generator: BundleGenerator = generate_v2_scenario_bundle,
) -> Path:
    """Lock both roles, then run every candidate on all recovery and null seeds."""
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(f"frozen v2 calibration already exists: {destination_path}")
    verify_v2_preregistration(frozen, current)
    policies = eligible_v2_calibration_policies(current)
    root = Path(work_root).expanduser().resolve()
    prerequisites = V2GenerationPrerequisites(current, frozen)
    roles = seed_roles()
    recovery_lock = prepare_v2_generation(
        root / "recovery", "calibration_recovery", roles["calibration_recovery"], prerequisites
    )
    null_lock = prepare_v2_generation(
        root / "null", "calibration_null", roles["calibration_null"], prerequisites
    )
    if dry_run:
        path = root / "calibration-plan.json"
        _write_plan(
            path,
            {
                "candidates": [item.normalized() for item in policies],
                "generation_locks": [
                    recovery_lock.file.normalized(),
                    null_lock.file.normalized(),
                ],
                "mode": "dry_run",
                "seed_roles": {
                    "calibration_null": list(roles["calibration_null"]),
                    "calibration_recovery": list(roles["calibration_recovery"]),
                },
                "status": "planned",
            },
        )
        return path
    attempt = claim_v2_phase_attempt(
        frozen,
        "calibration",
        {"null": null_lock.root, "recovery": recovery_lock.root},
    )
    progress_path = root / "calibration-progress.json"
    progress: dict[str, object] = {
        "attempt": attempt.normalized(),
        "candidates": [item.policy_id for item in policies],
        "runs": [],
        "status": "running",
    }
    _write_plan(progress_path, progress)
    generation_elapsed: dict[tuple[str, int], float] = {}
    generation_failed = False
    for role, seeds, spec_factory, destination_root in (
        (
            "calibration_recovery",
            roles["calibration_recovery"],
            SyntheticTuningSpec.synthetic_osse,
            recovery_lock.root,
        ),
        (
            "calibration_null",
            roles["calibration_null"],
            SyntheticTuningSpec.synthetic_osse_null,
            null_lock.root,
        ),
    ):
        for seed in seeds:
            generation_entry: dict[str, object] = {
                "phase": "generation",
                "role": role,
                "seed": seed,
                "status": "running",
            }
            runs = progress["runs"]
            assert isinstance(runs, list)
            runs.append(generation_entry)
            _write_plan(progress_path, progress)
            bundle_root = destination_root / "bundles" / f"seed-{seed}"
            started = time.perf_counter()
            try:
                generation_lock = recovery_lock if role == "calibration_recovery" else null_lock
                identity = bundle_generator(generation_lock, spec_factory(seed))
                if Path(identity.path) != bundle_root / "scenario.json":
                    raise ValueError("v2 bundle generator returned a noncanonical manifest path")
                generation_elapsed[(role, seed)] = time.perf_counter() - started
                generation_entry.update(manifest=identity.normalized(), status="completed")
            except Exception as exc:
                generation_failed = True
                generation_entry.update(error=f"{type(exc).__name__}: {exc}", status="failed")
            _write_plan(progress_path, progress)
    if generation_failed:
        _reject_calibration(
            destination_path,
            progress_path,
            progress,
            frozen,
            attempt,
            (recovery_lock.file, null_lock.file),
            "generation_exception",
        )
        raise RuntimeError(f"v2 calibration generation failed; evidence: {progress_path}")
    candidates: list[V2CalibrationCandidate] = []
    execution_failed = False
    for policy in policies:
        recovery: list[V2RecoverySeedResult] = []
        null: list[V2NullSeedResult] = []
        for role, seeds, spec_factory, score_kind, destination_root in (
            (
                "calibration_recovery",
                roles["calibration_recovery"],
                SyntheticTuningSpec.synthetic_osse,
                "recovery",
                recovery_lock.root,
            ),
            (
                "calibration_null",
                roles["calibration_null"],
                SyntheticTuningSpec.synthetic_osse_null,
                "null",
                null_lock.root,
            ),
        ):
            for seed in seeds:
                run_entry: dict[str, object] = {
                    "policy_id": policy.policy_id,
                    "role": role,
                    "seed": seed,
                    "status": "running",
                }
                runs = progress["runs"]
                assert isinstance(runs, list)
                runs.append(run_entry)
                _write_plan(progress_path, progress)
                try:
                    outcome = scenario_runner(
                        destination_root / "runs" / policy.policy_id / f"seed-{seed}",
                        spec_factory(seed),
                        policy,
                        score_kind=score_kind,
                        lock=(recovery_lock if role == "calibration_recovery" else null_lock),
                        evaluation_splits=("calibration",) if score_kind == "recovery" else (),
                    )
                    total_elapsed = generation_elapsed[(role, seed)] + outcome.elapsed_seconds
                    outcome = replace(
                        outcome,
                        elapsed_seconds=total_elapsed,
                        resources=resource_gate(total_elapsed, outcome.process_peak_rss_bytes),
                    )
                    run_entry["outcome"] = outcome.normalized()
                    run_entry["status"] = "completed"
                    if score_kind == "recovery":
                        recovery.append(V2RecoverySeedResult.from_outcome(outcome))
                    else:
                        null.append(V2NullSeedResult.from_outcome(outcome))
                except Exception as exc:
                    execution_failed = True
                    run_entry.update(
                        error=f"{type(exc).__name__}: {exc}",
                        status="failed",
                    )
                _write_plan(progress_path, progress)
        if len(recovery) == 3 and len(null) == 3:
            candidates.append(V2CalibrationCandidate(policy, tuple(recovery), tuple(null)))
    if execution_failed:
        _reject_calibration(
            destination_path,
            progress_path,
            progress,
            frozen,
            attempt,
            (recovery_lock.file, null_lock.file),
            "pipeline_exception",
        )
        raise RuntimeError(f"v2 calibration execution failed; evidence: {progress_path}")
    try:
        record = select_v2_calibration_policy(
            candidates,
            frozen,
            attempt,
            (recovery_lock.file, null_lock.file),
            eligible_policies=policies,
        )
    except Exception as exc:
        progress["selection_error"] = f"{type(exc).__name__}: {exc}"
        classification: V2FailureClassification = (
            "no_eligible_calibration_candidate"
            if candidates and not any(item.eligible for item in candidates)
            else "selection_exception"
        )
        _reject_calibration(
            destination_path,
            progress_path,
            progress,
            frozen,
            attempt,
            (recovery_lock.file, null_lock.file),
            classification,
        )
        raise
    try:
        validate_v2_calibration_record(record, current, frozen)
    except Exception as exc:
        progress["record_error"] = f"{type(exc).__name__}: {exc}"
        _reject_calibration(
            destination_path,
            progress_path,
            progress,
            frozen,
            attempt,
            (recovery_lock.file, null_lock.file),
            "record_validation_exception",
        )
        raise
    record_path = write_v2_calibration_record(destination_path, record)
    progress["record"] = str(record_path)
    progress["status"] = "completed"
    _write_plan(progress_path, progress)
    return record_path


def _reject_calibration(
    destination: Path,
    progress_path: Path,
    progress: dict[str, object],
    frozen: FrozenPreregistration,
    attempt: FrozenFileIdentity,
    generation_locks: tuple[FrozenFileIdentity, FrozenFileIdentity],
    classification: V2FailureClassification,
) -> None:
    progress.update(
        failure=rejection_failure(classification),
        record=str(destination),
        status="rejected",
    )
    write_v2_phase_rejection(
        destination,
        "calibration",
        frozen,
        attempt,
        generation_locks,
        classification,
        progress,
    )
    _write_plan(progress_path, progress)


def _write_plan(path: Path, value: object) -> None:
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


__all__ = ["BundleGenerator", "ScenarioRunner", "run_v2_calibration"]
