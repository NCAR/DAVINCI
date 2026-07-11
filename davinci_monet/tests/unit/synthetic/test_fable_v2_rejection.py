"""Terminal rejection evidence for irreversible FABLE v2 runners."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

import davinci_monet.tests.synthetic.fable_v2_acceptance as acceptance_module
import davinci_monet.tests.synthetic.fable_v2_attempts as attempts_module
import davinci_monet.tests.synthetic.fable_v2_calibration_runner as calibration_module
import davinci_monet.tests.synthetic.fable_v2_preflight as preflight_module
import davinci_monet.tests.synthetic.fable_v2_rejection as rejection_module
from davinci_monet.tests.synthetic.fable_v2_acceptance import (
    load_v2_acceptance_record,
    run_v2_acceptance,
)
from davinci_monet.tests.synthetic.fable_v2_calibration_record import (
    load_v2_calibration_record,
)
from davinci_monet.tests.synthetic.fable_v2_calibration_runner import run_v2_calibration
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
    V2FileBinding,
    V2Preregistration,
    V2TestEvidence,
    load_v2_preregistration,
    write_v2_preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import V2GenerationPrerequisites
from davinci_monet.tests.synthetic.fable_v2_policy import v2_calibration_policies
from davinci_monet.tests.synthetic.fable_v2_preflight import (
    load_v2_preflight_record,
    run_v2_preflight,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import (
    CYCLE_ID,
    GENERATION_LOCK_SCHEMA,
    protocol_sha256,
)
from davinci_monet.tests.synthetic.fable_v2_record_io import canonical_json
from davinci_monet.tests.synthetic.fable_v2_rejection import (
    V2FailureClassification,
    V2Phase,
    load_v2_phase_rejection,
)

TypedLoader = Callable[[str | Path], object]


def _identity(path: Path, content: str = "identity") -> FrozenFileIdentity:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="ascii")
    return FrozenFileIdentity.capture(path)


def _frozen(tmp_path: Path) -> FrozenPreregistration:
    config = tmp_path / "config.yaml"
    config.write_text("analysis: {}\n", encoding="ascii")
    current = V2Preregistration(
        generator_schema_version="test-schema",
        generator_spec_sha256="1" * 64,
        thresholds_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        configs=(V2FileBinding.capture("test", config),),
        tests=(V2TestEvidence("suite", "pytest", "5" * 64),),
    )
    return write_v2_preregistration(tmp_path / "preregistration.json", current)


def _lock(
    root: str | Path,
    role: str,
    seeds: tuple[int, ...],
    prerequisites: V2GenerationPrerequisites,
) -> Any:
    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    document = {
        "cycle_id": CYCLE_ID,
        "prerequisites": prerequisites.normalized(),
        "protocol_sha256": protocol_sha256(),
        "role": role,
        "schema_version": GENERATION_LOCK_SCHEMA,
        "seeds": list(seeds),
        "status": "locked_before_generation",
    }
    identity = _identity(destination / "generation-lock.json", canonical_json(document) + "\n")
    return SimpleNamespace(root=destination, path=Path(identity.path), file=identity)


def _prepare_lock(root: str | Path, role: str, seeds: Any, prerequisites: Any) -> Any:
    return _lock(root, role, tuple(seeds), prerequisites)


def _install_semantic_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attempts_module,
        "_ledger_root",
        lambda frozen: tmp_path / "attempt-ledger" / frozen.preregistration_sha256,
    )
    monkeypatch.setattr(rejection_module, "_validate_phase_dependencies", lambda *args: None)


def _manifest(lock: Any, spec: Any) -> FrozenFileIdentity:
    return _identity(
        Path(lock.root) / "bundles" / f"seed-{spec.master_seed}" / "scenario.json",
        str(spec.master_seed),
    )


def _wrong_manifest(lock: Any, spec: Any) -> FrozenFileIdentity:
    del spec
    return _identity(Path(lock.root) / "wrong-scenario.json")


def _raising(message: str) -> Callable[..., Any]:
    def raise_error(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise RuntimeError(message)

    return raise_error


def _assert_rejection(
    path: Path,
    progress_path: Path,
    frozen: FrozenPreregistration,
    phase: V2Phase,
    classification: V2FailureClassification,
    typed_loader: TypedLoader,
) -> Any:
    rejection = load_v2_phase_rejection(path)
    progress = json.loads(progress_path.read_text(encoding="ascii"))
    assert rejection.phase == phase
    assert rejection.classification == classification
    assert rejection.preregistration == frozen
    assert dict(rejection.progress) == progress
    assert progress["status"] == "rejected"
    assert progress["failure"] == rejection.normalized()["failure"]
    assert rejection.attempt.normalized() == progress["attempt"]
    with pytest.raises(ValueError, match="protocol identity mismatch"):
        typed_loader(path)
    return rejection


def _install_calibration_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen: FrozenPreregistration,
) -> None:
    del frozen
    _install_semantic_boundaries(tmp_path, monkeypatch)
    monkeypatch.setattr(calibration_module, "verify_v2_preregistration", lambda *args: None)
    monkeypatch.setattr(
        calibration_module,
        "eligible_v2_calibration_policies",
        lambda current: v2_calibration_policies(),
    )
    monkeypatch.setattr(calibration_module, "prepare_v2_generation", _prepare_lock)


def test_calibration_generation_exception_writes_once_and_preserves_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    _install_calibration_boundaries(tmp_path, monkeypatch, frozen)
    destination = tmp_path / "calibration.json"
    work_root = tmp_path / "calibration"

    with pytest.raises(RuntimeError, match="calibration generation failed"):
        run_v2_calibration(
            work_root,
            destination,
            load_v2_preregistration(frozen.file.path),
            frozen,
            bundle_generator=_raising("generation stopped"),
        )

    _assert_rejection(
        destination,
        work_root / "calibration-progress.json",
        frozen,
        "calibration",
        "generation_exception",
        load_v2_calibration_record,
    )


def test_calibration_pipeline_exception_writes_terminal_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    _install_calibration_boundaries(tmp_path, monkeypatch, frozen)
    destination = tmp_path / "calibration.json"
    work_root = tmp_path / "calibration"

    with pytest.raises(RuntimeError, match="calibration execution failed"):
        run_v2_calibration(
            work_root,
            destination,
            load_v2_preregistration(frozen.file.path),
            frozen,
            scenario_runner=_raising("pipeline stopped"),
            bundle_generator=_manifest,
        )

    _assert_rejection(
        destination,
        work_root / "calibration-progress.json",
        frozen,
        "calibration",
        "pipeline_exception",
        load_v2_calibration_record,
    )


def _install_calibration_selection_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    eligible: bool,
    message: str,
) -> Any:
    outcome = SimpleNamespace(
        elapsed_seconds=1.0,
        process_peak_rss_bytes=1024,
        normalized=lambda: {"status": "completed"},
    )
    monkeypatch.setattr(calibration_module, "replace", lambda value, **changes: value)
    monkeypatch.setattr(
        calibration_module.V2RecoverySeedResult,
        "from_outcome",
        staticmethod(lambda value: object()),
    )
    monkeypatch.setattr(
        calibration_module.V2NullSeedResult,
        "from_outcome",
        staticmethod(lambda value: object()),
    )
    monkeypatch.setattr(
        calibration_module,
        "V2CalibrationCandidate",
        lambda policy, recovery, null: SimpleNamespace(policy=policy, eligible=eligible),
    )
    monkeypatch.setattr(
        calibration_module,
        "select_v2_calibration_policy",
        _raising(message),
    )
    return outcome


@pytest.mark.parametrize(
    ("eligible", "message", "classification"),
    (
        (
            False,
            "no v2 calibration candidate passes every seed and equal-seed gate",
            "no_eligible_calibration_candidate",
        ),
        (True, "selection stopped", "selection_exception"),
    ),
)
def test_calibration_selection_failures_write_terminal_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eligible: bool,
    message: str,
    classification: V2FailureClassification,
) -> None:
    frozen = _frozen(tmp_path)
    _install_calibration_boundaries(tmp_path, monkeypatch, frozen)
    destination = tmp_path / "calibration.json"
    work_root = tmp_path / "calibration"
    outcome = _install_calibration_selection_path(
        monkeypatch,
        eligible=eligible,
        message=message,
    )

    with pytest.raises(RuntimeError, match=message):
        run_v2_calibration(
            work_root,
            destination,
            load_v2_preregistration(frozen.file.path),
            frozen,
            scenario_runner=lambda *args, **kwargs: outcome,
            bundle_generator=_manifest,
        )

    _assert_rejection(
        destination,
        work_root / "calibration-progress.json",
        frozen,
        "calibration",
        classification,
        load_v2_calibration_record,
    )


def _install_preflight_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, FrozenFileIdentity]:
    _install_semantic_boundaries(tmp_path, monkeypatch)
    calibration_path = Path(_identity(tmp_path / "calibration-source.json").path)
    policy = v2_calibration_policies()[0]
    calibration = SimpleNamespace(
        selected_policy=policy,
        selected_policy_id=policy.policy_id,
    )
    monkeypatch.setattr(preflight_module, "load_v2_calibration_record", lambda path: calibration)
    monkeypatch.setattr(
        preflight_module,
        "validate_v2_calibration_record",
        lambda record, current, frozen: record,
    )
    monkeypatch.setattr(preflight_module, "prepare_v2_generation", _prepare_lock)
    return calibration_path, FrozenFileIdentity.capture(calibration_path)


@pytest.mark.parametrize(
    ("stage", "classification", "error"),
    (
        ("spec", "generation_exception", "spec stopped"),
        ("generation", "generation_exception", "generation stopped"),
        ("manifest", "generation_exception", "noncanonical manifest path"),
        ("pipeline", "pipeline_exception", "pipeline stopped"),
    ),
)
def test_preflight_exceptions_write_terminal_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    classification: V2FailureClassification,
    error: str,
) -> None:
    frozen = _frozen(tmp_path)
    calibration_path, _ = _install_preflight_boundaries(tmp_path, monkeypatch)
    destination = tmp_path / "preflight.json"
    root = tmp_path / "preflight"
    generator = {
        "generation": _raising("generation stopped"),
        "manifest": _wrong_manifest,
    }.get(stage, _manifest)
    scenario_runner = _raising("pipeline stopped")
    monkeypatch.setattr(preflight_module, "generate_v2_scenario_bundle", generator)
    if stage == "spec":
        monkeypatch.setattr(
            preflight_module.SyntheticTuningSpec,
            "synthetic_osse",
            staticmethod(_raising("spec stopped")),
        )

    with pytest.raises((RuntimeError, ValueError), match=error):
        run_v2_preflight(
            root,
            destination,
            load_v2_preregistration(frozen.file.path),
            frozen,
            calibration_path,
            scenario_runner=scenario_runner,
        )

    _assert_rejection(
        destination,
        root / "preflight-progress.json",
        frozen,
        "preflight",
        classification,
        load_v2_preflight_record,
    )


def test_preflight_record_validation_rejection_embeds_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    calibration_path, _ = _install_preflight_boundaries(tmp_path, monkeypatch)
    destination = tmp_path / "preflight.json"
    root = tmp_path / "preflight"
    outcome = SimpleNamespace(
        elapsed_seconds=1.0,
        process_peak_rss_bytes=1024,
        normalized=lambda: {"marker": "complete"},
    )
    monkeypatch.setattr(preflight_module, "generate_v2_scenario_bundle", _manifest)
    monkeypatch.setattr(preflight_module, "replace", lambda value, **changes: value)

    with pytest.raises((KeyError, ValueError)):
        run_v2_preflight(
            root,
            destination,
            load_v2_preregistration(frozen.file.path),
            frozen,
            calibration_path,
            scenario_runner=cast(Any, lambda *args, **kwargs: outcome),
        )

    rejection = _assert_rejection(
        destination,
        root / "preflight-progress.json",
        frozen,
        "preflight",
        "record_validation_exception",
        load_v2_preflight_record,
    )
    assert rejection.progress["outcome"] == {"marker": "complete"}


def _install_acceptance_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    _install_semantic_boundaries(tmp_path, monkeypatch)
    calibration_path = Path(_identity(tmp_path / "calibration-source.json").path)
    preflight_path = Path(_identity(tmp_path / "preflight-source.json").path)
    policy = v2_calibration_policies()[0]
    calibration = SimpleNamespace(
        selected_policy=policy,
        selected_policy_id=policy.policy_id,
    )
    preflight = SimpleNamespace(status="passed")
    monkeypatch.setattr(acceptance_module, "load_v2_calibration_record", lambda path: calibration)
    monkeypatch.setattr(acceptance_module, "load_v2_preflight_record", lambda path: preflight)
    monkeypatch.setattr(
        acceptance_module,
        "validate_v2_calibration_record",
        lambda record, current, frozen: record,
    )
    monkeypatch.setattr(
        acceptance_module,
        "validate_v2_preflight_record",
        lambda record, current, frozen, calibration_identity: record,
    )
    monkeypatch.setattr(acceptance_module, "prepare_v2_generation", _prepare_lock)
    return calibration_path, preflight_path


@pytest.mark.parametrize(
    ("stage", "classification", "error"),
    (
        ("spec", "generation_exception", "spec stopped"),
        ("generation", "generation_exception", "generation stopped"),
        ("manifest", "generation_exception", "noncanonical manifest path"),
        ("pipeline", "pipeline_exception", "pipeline stopped"),
    ),
)
def test_acceptance_exceptions_write_terminal_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    classification: V2FailureClassification,
    error: str,
) -> None:
    frozen = _frozen(tmp_path)
    calibration_path, preflight_path = _install_acceptance_boundaries(tmp_path, monkeypatch)
    root = tmp_path / "acceptance"
    generator = {
        "generation": _raising("generation stopped"),
        "manifest": _wrong_manifest,
    }.get(stage, _manifest)
    monkeypatch.setattr(acceptance_module, "generate_v2_scenario_bundle", generator)
    if stage == "spec":
        monkeypatch.setattr(
            acceptance_module.SyntheticTuningSpec,
            "synthetic_osse",
            staticmethod(_raising("spec stopped")),
        )

    with pytest.raises((RuntimeError, ValueError), match=error):
        run_v2_acceptance(
            root,
            load_v2_preregistration(frozen.file.path),
            frozen,
            calibration_path,
            preflight_path,
            scenario_runner=_raising("pipeline stopped"),
        )

    _assert_rejection(
        root / "acceptance.json",
        root / "acceptance-progress.json",
        frozen,
        "acceptance",
        classification,
        load_v2_acceptance_record,
    )
