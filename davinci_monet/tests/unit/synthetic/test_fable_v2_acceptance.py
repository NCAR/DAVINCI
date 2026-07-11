"""Single-use ordered acceptance contracts for FABLE v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import davinci_monet.tests.synthetic.fable_v2_acceptance as acceptance
import davinci_monet.tests.synthetic.fable_v2_lifecycle as lifecycle
from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec, spec_hash
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_gate
from davinci_monet.tests.synthetic.fable_v2_acceptance import (
    V2AcceptanceRecord,
    load_v2_acceptance_record,
    run_v2_acceptance,
    write_v2_acceptance_record,
)
from davinci_monet.tests.synthetic.fable_v2_attempts import claim_v2_phase_attempt
from davinci_monet.tests.synthetic.fable_v2_calibration import V2RecoverySeedResult
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
    V2FileBinding,
    V2Preregistration,
    V2TestEvidence,
    write_v2_preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import (
    V2GenerationPrerequisites,
    prepare_v2_generation,
    validate_v2_generation_request,
    verify_v2_generation_lock,
)
from davinci_monet.tests.synthetic.fable_v2_policy import (
    FableV2Policy,
    v2_calibration_policies,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import ACCEPTANCE_SEEDS
from davinci_monet.tests.synthetic.fable_v2_record_io import canonical_json, json_sha256
from davinci_monet.tests.synthetic.fable_v2_runner import V2ScenarioOutcome

_LOCAL_SUBSTITUTION_SEED = 424245


def _identity(label: str) -> FrozenFileIdentity:
    return FrozenFileIdentity(
        f"/unit-only/{label}", hashlib.sha256(label.encode("ascii")).hexdigest()
    )


def _dummy_frozen() -> FrozenPreregistration:
    return FrozenPreregistration(_identity("preregistration.json"), "f" * 64)


def _outcome(
    policy: FableV2Policy,
    seed: int,
    *,
    nrmse: float = 0.20,
    evaluation_splits: tuple[str, ...] = ("development_test",),
) -> V2ScenarioOutcome:
    spec = SyntheticTuningSpec.synthetic_osse(seed)
    score: dict[str, Any] = {
        "diagnostics": {
            "estimate_vs_unfiltered_in_span_nrmse": 0.10,
            "excluded_fraction": 0.20,
        },
        "metrics": {
            "aod_rmse_ratio": 0.50,
            "field_correlation": 0.95,
            "field_nrmse": nrmse,
            "field_origin_slope": 1.0,
            "full_target_aod_rmse_ratio": 0.70,
        },
        "failures": [],
        "passed": nrmse <= 0.35,
    }
    identity = _identity(f"{policy.policy_id}-{seed}")
    return V2ScenarioOutcome(
        seed=seed,
        scenario=spec.scenario,
        policy=policy,
        score_kind="recovery",
        evaluation_splits=evaluation_splits,
        spec_sha256=spec_hash(spec),
        scenario_manifest=identity,
        fitting_config=identity,
        fitting_manifest=identity,
        evaluation_config=identity,
        evaluation_manifest=identity,
        report_sha256="e" * 64,
        diagnostic_report_sha256="d" * 64,
        learned_basis_oracle_nrmse=0.03,
        score=score,
        evidence_passed=True,
        elapsed_seconds=1.0,
        process_peak_rss_bytes=1024,
        resources=resource_gate(1.0, 1024),
    )


def _results(
    policy: FableV2Policy, *, failed_seed: int | None = None
) -> tuple[V2RecoverySeedResult, ...]:
    return tuple(
        V2RecoverySeedResult.from_outcome(
            _outcome(policy, seed, nrmse=0.36 if seed == failed_seed else 0.20)
        )
        for seed in ACCEPTANCE_SEEDS
    )


def _record(
    policy: FableV2Policy,
    *,
    failed_seed: int | None = None,
    results: tuple[V2RecoverySeedResult, ...] | None = None,
) -> V2AcceptanceRecord:
    return V2AcceptanceRecord(
        preregistration=_dummy_frozen(),
        attempt=_identity("acceptance-attempt.json"),
        generation_lock=_identity("acceptance-lock.json"),
        calibration_record=_identity("calibration.json"),
        preflight_record=_identity("preflight.json"),
        selected_policy=policy,
        results=results if results is not None else _results(policy, failed_seed=failed_seed),
    )


def _preregistration(
    tmp_path: Path,
) -> tuple[V2Preregistration, FrozenPreregistration]:
    config = tmp_path / "fit.yaml"
    config.write_text("analysis: {}\n", encoding="ascii")
    current = V2Preregistration(
        generator_schema_version="unit-test-schema",
        generator_spec_sha256="1" * 64,
        thresholds_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        configs=(V2FileBinding.capture("fit", config),),
        tests=(V2TestEvidence("unit", "pytest", "5" * 64),),
    )
    return current, write_v2_preregistration(tmp_path / "preregistration.json", current)


def _prerequisite_files(tmp_path: Path) -> tuple[FrozenFileIdentity, FrozenFileIdentity]:
    calibration = tmp_path / "calibration.json"
    preflight = tmp_path / "preflight.json"
    calibration.write_text("calibration\n", encoding="ascii")
    preflight.write_text("preflight\n", encoding="ascii")
    return FrozenFileIdentity.capture(calibration), FrozenFileIdentity.capture(preflight)


def test_acceptance_requires_exact_order_and_phase_semantics() -> None:
    policy = v2_calibration_policies()[0]
    exact = _results(policy)
    assert _record(policy, results=exact).status == "passed_pending_user_review"

    with pytest.raises(ValueError, match="exact ordered seed tuple"):
        _record(policy, results=tuple(reversed(exact)))
    substituted = (
        *exact[:-1],
        V2RecoverySeedResult.from_outcome(_outcome(policy, _LOCAL_SUBSTITUTION_SEED)),
    )
    with pytest.raises(ValueError, match="exact ordered seed tuple"):
        _record(policy, results=substituted)

    wrong_phase = V2RecoverySeedResult.from_outcome(
        _outcome(policy, ACCEPTANCE_SEEDS[0], evaluation_splits=("calibration",))
    )
    with pytest.raises(ValueError, match="phase spec"):
        _record(policy, results=(wrong_phase, *exact[1:]))


def test_acceptance_record_rejects_derived_metric_and_policy_tampering(
    tmp_path: Path,
) -> None:
    policy = v2_calibration_policies()[0]
    record = _record(policy)
    path = write_v2_acceptance_record(tmp_path / "acceptance.json", record)
    raw = path.read_text(encoding="ascii")

    assert raw == canonical_json(json.loads(raw)) + "\n"
    assert path.stat().st_mode & 0o777 == 0o444
    assert load_v2_acceptance_record(path) == record

    envelope = json.loads(raw)
    envelope["record"]["results"][0]["field_nrmse"] = 0.34
    envelope["record_sha256"] = json_sha256(envelope["record"])
    derived = tmp_path / "derived-metric-tamper.json"
    derived.write_text(canonical_json(envelope) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="fields do not match scenario evidence"):
        load_v2_acceptance_record(derived)

    envelope = json.loads(raw)
    envelope["record"]["selected_policy"]["ridge"] = 0.1
    envelope["record_sha256"] = json_sha256(envelope["record"])
    policy_path = tmp_path / "policy-tamper.json"
    policy_path.write_text(canonical_json(envelope) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="frozen v1 all-band control"):
        load_v2_acceptance_record(policy_path)


def test_acceptance_generation_denies_early_failed_reused_and_changed_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current, frozen = _preregistration(tmp_path)
    calibration, preflight = _prerequisite_files(tmp_path)

    with pytest.raises(ValueError, match="all prerequisite identities"):
        validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS)
    registered = V2GenerationPrerequisites(current, frozen)
    with pytest.raises(ValueError, match="all prerequisite identities"):
        validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS, registered)
    calibrated = V2GenerationPrerequisites(current, frozen, calibration_record=calibration)
    with pytest.raises(ValueError, match="all prerequisite identities"):
        validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS, calibrated)
    failed = V2GenerationPrerequisites(
        current,
        frozen,
        calibration_record=calibration,
        preflight_record=preflight,
        preflight_status="failed",
    )
    with pytest.raises(ValueError, match="frozen preflight passed"):
        validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS, failed)

    passed = V2GenerationPrerequisites(
        current,
        frozen,
        calibration_record=calibration,
        preflight_record=preflight,
        preflight_status="passed",
    )
    assert (
        validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS, passed) == ACCEPTANCE_SEEDS
    )
    with pytest.raises(ValueError, match="exact ordered seed tuple"):
        validate_v2_generation_request("acceptance", tuple(reversed(ACCEPTANCE_SEEDS)), passed)
    with pytest.raises(ValueError, match="exact ordered seed tuple"):
        validate_v2_generation_request(
            "acceptance", (*ACCEPTANCE_SEEDS[:-1], _LOCAL_SUBSTITUTION_SEED), passed
        )

    with pytest.raises(ValueError, match="invalid v2 calibration JSON"):
        prepare_v2_generation(
            tmp_path / "invalid-semantic-records",
            "acceptance",
            ACCEPTANCE_SEEDS,
            passed,
        )
    monkeypatch.setattr(lifecycle, "_validate_v2_generation_evidence", lambda *args: None)
    lock = prepare_v2_generation(
        tmp_path / "acceptance-run", "acceptance", ACCEPTANCE_SEEDS, passed
    )
    verify_v2_generation_lock(lock)
    with pytest.raises(FileExistsError):
        prepare_v2_generation(tmp_path / "acceptance-run", "acceptance", ACCEPTANCE_SEEDS, passed)
    claim_v2_phase_attempt(frozen, "acceptance", {"root": lock.root})
    with pytest.raises(FileExistsError, match="already attempted"):
        claim_v2_phase_attempt(frozen, "acceptance", {"root": lock.root})

    copied_preregistration = tmp_path / "copied" / "preregistration.json"
    copied_preregistration.parent.mkdir()
    copied_preregistration.write_bytes(Path(frozen.file.path).read_bytes())
    copied_frozen = FrozenPreregistration(
        FrozenFileIdentity.capture(copied_preregistration),
        frozen.preregistration_sha256,
    )
    with pytest.raises(FileExistsError, match="already attempted"):
        claim_v2_phase_attempt(copied_frozen, "acceptance", {"root": lock.root})

    Path(preflight.path).write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="frozen file identity changed"):
        validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS, passed)


def test_acceptance_dry_run_requires_passing_preflight_and_never_generates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current, frozen = _preregistration(tmp_path)
    calibration_identity, preflight_identity = _prerequisite_files(tmp_path)
    policy = v2_calibration_policies()[0]
    calibration = SimpleNamespace(selected_policy=policy)
    preflight = SimpleNamespace(status="failed")
    monkeypatch.setattr(acceptance, "load_v2_calibration_record", lambda path: calibration)
    monkeypatch.setattr(
        acceptance,
        "validate_v2_calibration_record",
        lambda record, current, frozen: record,
    )
    monkeypatch.setattr(acceptance, "load_v2_preflight_record", lambda path: preflight)
    monkeypatch.setattr(
        acceptance,
        "validate_v2_preflight_record",
        lambda record, current, frozen, calibration_identity: record,
    )

    def forbidden(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("acceptance dry-run attempted scientific generation")

    monkeypatch.setattr(acceptance, "generate_v2_scenario_bundle", forbidden)
    failed_root = tmp_path / "failed-preflight"
    with pytest.raises(ValueError, match="frozen preflight passed"):
        run_v2_acceptance(
            failed_root,
            current,
            frozen,
            calibration_identity.path,
            preflight_identity.path,
            dry_run=True,
            scenario_runner=forbidden,
        )
    assert not failed_root.exists()

    preflight.status = "passed"
    monkeypatch.setattr(lifecycle, "_validate_v2_generation_evidence", lambda *args: None)
    plan = run_v2_acceptance(
        tmp_path / "dry-run",
        current,
        frozen,
        calibration_identity.path,
        preflight_identity.path,
        dry_run=True,
        scenario_runner=forbidden,
    )
    document = json.loads(plan.read_text(encoding="ascii"))
    assert document["seeds"] == list(ACCEPTANCE_SEEDS)
    assert document["status"] == "planned"
    assert not (plan.parent / "bundles").exists()


def test_failed_acceptance_disposition_is_permanent(tmp_path: Path) -> None:
    policy = v2_calibration_policies()[0]
    failed = _record(policy, failed_seed=ACCEPTANCE_SEEDS[1])
    path = write_v2_acceptance_record(tmp_path / "acceptance.json", failed)

    assert load_v2_acceptance_record(path).status == "failed"
    assert any(
        f"seed_{ACCEPTANCE_SEEDS[1]}:field_nrmse_above_maximum" in failure
        for failure in failed.failures
    )
    with pytest.raises(FileExistsError, match="already exists"):
        write_v2_acceptance_record(path, _record(policy))
    assert load_v2_acceptance_record(path).status == "failed"
