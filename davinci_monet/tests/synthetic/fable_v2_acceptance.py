"""Exact-tuple, single-use synthetic acceptance runner for FABLE v2."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_gate
from davinci_monet.tests.synthetic.fable_v2_attempts import (
    claim_v2_phase_attempt,
    validate_v2_phase_attempt,
)
from davinci_monet.tests.synthetic.fable_v2_bundle import generate_v2_scenario_bundle
from davinci_monet.tests.synthetic.fable_v2_calibration import (
    V2RecoverySeedResult,
    aggregate_recovery_failures,
    recovery_summary,
)
from davinci_monet.tests.synthetic.fable_v2_calibration_record import (
    load_v2_calibration_record,
    validate_v2_calibration_record,
)
from davinci_monet.tests.synthetic.fable_v2_evidence import validate_v2_scenario_evidence
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
    V2Preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import (
    V2GenerationPrerequisites,
    prepare_v2_generation,
)
from davinci_monet.tests.synthetic.fable_v2_lock_evidence import (
    validate_v2_generation_lock_identity,
)
from davinci_monet.tests.synthetic.fable_v2_policy import (
    FableV2Policy,
    v2_policy_from_normalized,
)
from davinci_monet.tests.synthetic.fable_v2_preflight import (
    load_v2_preflight_record,
    validate_v2_preflight_record,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import (
    ACCEPTANCE_SEEDS,
    CYCLE_ID,
    protocol_sha256,
    validate_role_seeds,
)
from davinci_monet.tests.synthetic.fable_v2_rejection import (
    V2FailureClassification,
    rejection_failure,
    write_v2_phase_rejection,
)
from davinci_monet.tests.synthetic.fable_v2_runner import (
    V2ScenarioOutcome,
    run_v2_scenario,
)

V2_ACCEPTANCE_SCHEMA = "fable-v2-acceptance-v1"
ScenarioRunner = Callable[..., V2ScenarioOutcome]


@dataclass(frozen=True)
class V2AcceptanceRecord:
    """Immutable final program result pending explicit user review when passing."""

    preregistration: FrozenPreregistration
    attempt: FrozenFileIdentity
    generation_lock: FrozenFileIdentity
    calibration_record: FrozenFileIdentity
    preflight_record: FrozenFileIdentity
    selected_policy: FableV2Policy
    results: tuple[V2RecoverySeedResult, ...]

    def __post_init__(self) -> None:
        validate_role_seeds("acceptance", [item.seed for item in self.results])
        for result in self.results:
            validate_v2_scenario_evidence(
                _evidence_mapping(result.evidence_json),
                SyntheticTuningSpec.synthetic_osse(result.seed),
                self.selected_policy,
                score_kind="recovery",
                evaluation_splits=("development_test",),
                verify_files=False,
            )

    @property
    def failures(self) -> tuple[str, ...]:
        return aggregate_recovery_failures(self.results)

    @property
    def status(self) -> str:
        return "passed_pending_user_review" if not self.failures else "failed"

    def normalized(self) -> dict[str, Any]:
        return {
            "aggregate": recovery_summary(self.results),
            "attempt": self.attempt.normalized(),
            "generation_lock": self.generation_lock.normalized(),
            "calibration_record": self.calibration_record.normalized(),
            "cycle_id": CYCLE_ID,
            "failures": list(self.failures),
            "preflight_record": self.preflight_record.normalized(),
            "preregistration": self.preregistration.normalized(),
            "protocol_sha256": protocol_sha256(),
            "results": [item.normalized() for item in self.results],
            "schema_version": V2_ACCEPTANCE_SCHEMA,
            "seeds": list(ACCEPTANCE_SEEDS),
            "selected_policy": self.selected_policy.normalized(),
            "status": self.status,
        }


def write_v2_acceptance_record(path: str | Path, record: V2AcceptanceRecord) -> Path:
    """Atomically publish the immutable final program result exactly once."""
    destination = Path(path).expanduser().resolve()
    value = record.normalized()
    _write_once(
        destination,
        _canonical_json({"record": value, "record_sha256": _json_sha256(value)}) + "\n",
    )
    return destination


def load_v2_acceptance_record(path: str | Path) -> V2AcceptanceRecord:
    """Load canonical bytes and recompute every per-seed and aggregate gate."""
    source = Path(path).expanduser().resolve()
    raw = source.read_text(encoding="ascii")
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid v2 acceptance JSON") from exc
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"record", "record_sha256"}
        or raw != _canonical_json(envelope) + "\n"
    ):
        raise ValueError("v2 acceptance record is not canonically encoded")
    value = _mapping(envelope["record"], "acceptance record")
    if not hmac.compare_digest(str(envelope["record_sha256"]), _json_sha256(value)):
        raise ValueError("v2 acceptance record SHA-256 mismatch")
    record = _record_from_normalized(value)
    if record.normalized() != value:
        raise ValueError("v2 acceptance record scientific disposition mismatch")
    return record


def validate_v2_acceptance_record(
    record: V2AcceptanceRecord,
    current: V2Preregistration,
    frozen: FrozenPreregistration,
    calibration_identity: FrozenFileIdentity,
    preflight_identity: FrozenFileIdentity,
) -> V2AcceptanceRecord:
    """Reverify every frozen prerequisite and per-seed scenario artifact."""
    calibration = validate_v2_calibration_record(
        load_v2_calibration_record(calibration_identity.path), current, frozen
    )
    preflight = validate_v2_preflight_record(
        load_v2_preflight_record(preflight_identity.path),
        current,
        frozen,
        calibration_identity,
    )
    if (
        record.preregistration.normalized() != frozen.normalized()
        or record.calibration_record.normalized() != calibration_identity.normalized()
        or record.preflight_record.normalized() != preflight_identity.normalized()
        or record.selected_policy != calibration.selected_policy
        or preflight.status != "passed"
    ):
        raise ValueError("v2 acceptance prerequisite identity mismatch")
    attempt_document = validate_v2_phase_attempt(record.attempt, frozen, "acceptance")
    prerequisites = V2GenerationPrerequisites(
        current,
        frozen,
        calibration_record=calibration_identity,
        preflight_record=preflight_identity,
        preflight_status="passed",
    )
    root = validate_v2_generation_lock_identity(
        record.generation_lock,
        "acceptance",
        ACCEPTANCE_SEEDS,
        prerequisites,
    )
    roots = attempt_document.get("roots")
    if not isinstance(roots, Mapping) or roots.get("root") != str(root):
        raise ValueError("acceptance generation lock root differs from attempt ledger")
    for result in record.results:
        validate_v2_scenario_evidence(
            _evidence_mapping(result.evidence_json),
            SyntheticTuningSpec.synthetic_osse(result.seed),
            record.selected_policy,
            score_kind="recovery",
            evaluation_splits=("development_test",),
            verify_files=True,
        )
    return record


def run_v2_acceptance(
    root: str | Path,
    current: V2Preregistration,
    frozen: FrozenPreregistration,
    calibration_path: str | Path,
    preflight_path: str | Path,
    *,
    dry_run: bool = False,
    scenario_runner: ScenarioRunner = run_v2_scenario,
) -> Path:
    """Run only the protocol-owned ordered tuple after a frozen passing preflight."""
    calibration_identity = FrozenFileIdentity.capture(calibration_path)
    preflight_identity = FrozenFileIdentity.capture(preflight_path)
    calibration = validate_v2_calibration_record(
        load_v2_calibration_record(calibration_identity.path), current, frozen
    )
    preflight = validate_v2_preflight_record(
        load_v2_preflight_record(preflight_identity.path),
        current,
        frozen,
        calibration_identity,
    )
    prerequisites = V2GenerationPrerequisites(
        current,
        frozen,
        calibration_record=calibration_identity,
        preflight_record=preflight_identity,
        preflight_status=preflight.status,
    )
    lock = prepare_v2_generation(root, "acceptance", ACCEPTANCE_SEEDS, prerequisites)
    record_path = lock.root / "acceptance.json"
    if dry_run:
        record_path.write_text(
            _canonical_json(
                {
                    "calibration_record": calibration_identity.normalized(),
                    "generation_lock": lock.file.normalized(),
                    "mode": "dry_run",
                    "preflight_record": preflight_identity.normalized(),
                    "seeds": list(ACCEPTANCE_SEEDS),
                    "selected_policy": calibration.selected_policy.normalized(),
                    "status": "planned",
                }
            )
            + "\n",
            encoding="ascii",
        )
        return record_path
    attempt = claim_v2_phase_attempt(frozen, "acceptance", {"root": lock.root})
    progress_path = lock.root / "acceptance-progress.json"
    progress: dict[str, Any] = {
        "attempt": attempt.normalized(),
        "runs": [],
        "seeds": list(ACCEPTANCE_SEEDS),
        "selected_policy_id": calibration.selected_policy_id,
        "status": "running",
    }
    _write_progress(progress_path, progress)
    outcomes: list[V2ScenarioOutcome] = []
    for seed in ACCEPTANCE_SEEDS:
        entry: dict[str, Any] = {"seed": seed, "status": "running"}
        progress["runs"].append(entry)
        _write_progress(progress_path, progress)
        try:
            spec = SyntheticTuningSpec.synthetic_osse(seed)
            generation_started = time.perf_counter()
            manifest = generate_v2_scenario_bundle(lock, spec)
            expected_manifest = lock.root / "bundles" / f"seed-{seed}" / "scenario.json"
            if Path(manifest.path) != expected_manifest:
                raise ValueError("v2 bundle generator returned a noncanonical manifest path")
            generation_elapsed = time.perf_counter() - generation_started
            manifest_value = manifest.normalized()
        except Exception as exc:
            entry.update(error=f"{type(exc).__name__}: {exc}", status="failed")
            _reject_acceptance(
                record_path,
                progress_path,
                progress,
                frozen,
                attempt,
                lock.file,
                "generation_exception",
            )
            raise
        entry["scenario_manifest"] = manifest_value
        _write_progress(progress_path, progress)
        try:
            outcome = scenario_runner(
                lock.root / "runs" / calibration.selected_policy_id / f"seed-{seed}",
                spec,
                calibration.selected_policy,
                score_kind="recovery",
                lock=lock,
                evaluation_splits=("development_test",),
            )
            total_elapsed = generation_elapsed + outcome.elapsed_seconds
            outcome = replace(
                outcome,
                elapsed_seconds=total_elapsed,
                resources=resource_gate(total_elapsed, outcome.process_peak_rss_bytes),
            )
            outcomes.append(outcome)
            entry["outcome"] = outcome.normalized()
            entry["passed"] = outcome.passed
            entry["status"] = "completed" if outcome.passed else "failed"
        except Exception as exc:
            entry.update(error=f"{type(exc).__name__}: {exc}", status="failed")
            _reject_acceptance(
                record_path,
                progress_path,
                progress,
                frozen,
                attempt,
                lock.file,
                "pipeline_exception",
            )
            raise
        _write_progress(progress_path, progress)
    try:
        record = V2AcceptanceRecord(
            frozen,
            attempt,
            lock.file,
            calibration_identity,
            preflight_identity,
            calibration.selected_policy,
            tuple(V2RecoverySeedResult.from_outcome(item) for item in outcomes),
        )
        validate_v2_acceptance_record(
            record,
            current,
            frozen,
            calibration_identity,
            preflight_identity,
        )
    except Exception as exc:
        progress["record_error"] = f"{type(exc).__name__}: {exc}"
        _reject_acceptance(
            record_path,
            progress_path,
            progress,
            frozen,
            attempt,
            lock.file,
            "record_validation_exception",
        )
        raise
    final_path = write_v2_acceptance_record(record_path, record)
    progress.update(record=str(final_path), status=record.status)
    _write_progress(progress_path, progress)
    return final_path


def _reject_acceptance(
    destination: Path,
    progress_path: Path,
    progress: dict[str, Any],
    frozen: FrozenPreregistration,
    attempt: FrozenFileIdentity,
    generation_lock: FrozenFileIdentity,
    classification: V2FailureClassification,
) -> None:
    progress.update(
        failure=rejection_failure(classification),
        record=str(destination),
        status="rejected",
    )
    write_v2_phase_rejection(
        destination,
        "acceptance",
        frozen,
        attempt,
        (generation_lock,),
        classification,
        progress,
    )
    _write_progress(progress_path, progress)


def _record_from_normalized(value: Mapping[str, Any]) -> V2AcceptanceRecord:
    if (
        value.get("cycle_id") != CYCLE_ID
        or value.get("protocol_sha256") != protocol_sha256()
        or value.get("schema_version") != V2_ACCEPTANCE_SCHEMA
        or tuple(value.get("seeds", ())) != ACCEPTANCE_SEEDS
    ):
        raise ValueError("v2 acceptance protocol identity mismatch")
    prereg = _mapping(value["preregistration"], "acceptance preregistration")
    frozen = FrozenPreregistration(
        _file_identity(_mapping(prereg["file"], "preregistration file")),
        str(prereg["preregistration_sha256"]),
    )
    results = tuple(_result_from_normalized(item) for item in value["results"])
    return V2AcceptanceRecord(
        frozen,
        _file_identity(_mapping(value["attempt"], "acceptance attempt")),
        _file_identity(_mapping(value["generation_lock"], "acceptance generation lock")),
        _file_identity(_mapping(value["calibration_record"], "calibration file")),
        _file_identity(_mapping(value["preflight_record"], "preflight file")),
        v2_policy_from_normalized(_mapping(value["selected_policy"], "selected policy")),
        results,
    )


def _file_identity(value: Mapping[str, Any]) -> FrozenFileIdentity:
    return FrozenFileIdentity(str(value["path"]), str(value["sha256"]))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _write_once(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"frozen v2 acceptance already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"frozen v2 acceptance already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_progress(path: Path, value: Any) -> None:
    payload = _canonical_json(value) + "\n"
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _result_from_normalized(value: Any) -> V2RecoverySeedResult:
    fields = dict(_mapping(value, "acceptance seed result"))
    result = V2RecoverySeedResult(
        _canonical_json(fields["evidence"]), str(fields["evidence_sha256"])
    )
    if result.normalized() != fields:
        raise ValueError("acceptance result fields do not match scenario evidence")
    return result


def _evidence_mapping(value: str) -> Mapping[str, Any]:
    parsed = json.loads(value)
    return _mapping(parsed, "acceptance scenario evidence")


__all__ = [
    "V2_ACCEPTANCE_SCHEMA",
    "V2AcceptanceRecord",
    "load_v2_acceptance_record",
    "run_v2_acceptance",
    "validate_v2_acceptance_record",
    "write_v2_acceptance_record",
]
