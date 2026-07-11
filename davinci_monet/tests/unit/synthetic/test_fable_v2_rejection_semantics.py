"""Semantic binding and publication durability for FABLE v2 rejections."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import davinci_monet.tests.synthetic.fable_v2_acceptance as acceptance_module
import davinci_monet.tests.synthetic.fable_v2_attempts as attempts_module
import davinci_monet.tests.synthetic.fable_v2_calibration_runner as calibration_module
import davinci_monet.tests.synthetic.fable_v2_preflight as preflight_module
import davinci_monet.tests.synthetic.fable_v2_record_io as record_io_module
import davinci_monet.tests.synthetic.fable_v2_rejection as rejection_module
from davinci_monet.tests.synthetic.fable_v2_attempts import claim_v2_phase_attempt
from davinci_monet.tests.synthetic.fable_v2_calibration_record import (
    load_v2_calibration_record,
)
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
from davinci_monet.tests.synthetic.fable_v2_protocol import (
    CYCLE_ID,
    GENERATION_LOCK_SCHEMA,
    protocol_sha256,
    seed_roles,
)
from davinci_monet.tests.synthetic.fable_v2_record_io import (
    canonical_json,
    write_record_once,
)
from davinci_monet.tests.synthetic.fable_v2_rejection import (
    load_v2_phase_rejection,
    rejection_failure,
    write_v2_phase_rejection,
)


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
    root: Path,
    role: str,
    prerequisites: V2GenerationPrerequisites,
) -> Any:
    root.mkdir(parents=True, exist_ok=False)
    document = {
        "cycle_id": CYCLE_ID,
        "prerequisites": prerequisites.normalized(),
        "protocol_sha256": protocol_sha256(),
        "role": role,
        "schema_version": GENERATION_LOCK_SCHEMA,
        "seeds": list(seed_roles()[role]),
        "status": "locked_before_generation",
    }
    identity = _identity(root / "generation-lock.json", canonical_json(document) + "\n")
    return SimpleNamespace(root=root.resolve(), file=identity)


def _ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        attempts_module,
        "_ledger_root",
        lambda frozen: tmp_path / "attempt-ledger" / frozen.preregistration_sha256,
    )


def test_rejection_round_trip_rejects_cross_wired_attempt_lock_and_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    current = load_v2_preregistration(frozen.file.path)
    _ledger(tmp_path, monkeypatch)
    monkeypatch.setattr(rejection_module, "_validate_phase_dependencies", lambda *args: None)
    prerequisites = V2GenerationPrerequisites(current, frozen)
    recovery = _lock(tmp_path / "recovery", "calibration_recovery", prerequisites)
    null = _lock(tmp_path / "null", "calibration_null", prerequisites)
    attempt = claim_v2_phase_attempt(
        frozen,
        "calibration",
        {"recovery": recovery.root, "null": null.root},
    )
    progress = {
        "attempt": attempt.normalized(),
        "failure": rejection_failure("generation_exception"),
        "status": "rejected",
    }
    destination = tmp_path / "calibration.json"

    write_v2_phase_rejection(
        destination,
        "calibration",
        frozen,
        attempt,
        (recovery.file, null.file),
        "generation_exception",
        progress,
    )
    rejection = load_v2_phase_rejection(destination)
    assert rejection.progress == progress
    with pytest.raises(ValueError, match="protocol identity mismatch"):
        load_v2_calibration_record(destination)
    with pytest.raises(FileExistsError, match="already exists"):
        write_v2_phase_rejection(
            destination,
            "calibration",
            frozen,
            attempt,
            (recovery.file, null.file),
            "generation_exception",
            progress,
        )

    wrong_attempt = claim_v2_phase_attempt(
        frozen,
        "preflight",
        {"root": recovery.root},
    )
    wrong_progress = dict(progress)
    wrong_progress["attempt"] = wrong_attempt.normalized()
    with pytest.raises(ValueError, match="attempt"):
        write_v2_phase_rejection(
            tmp_path / "wrong-attempt.json",
            "calibration",
            frozen,
            wrong_attempt,
            (recovery.file, null.file),
            "generation_exception",
            wrong_progress,
        )

    upstream = _identity(tmp_path / "upstream.json")
    preflight_prerequisites = V2GenerationPrerequisites(
        current,
        frozen,
        calibration_record=upstream,
    )
    wrong_lock = _lock(tmp_path / "wrong-lock", "preflight", preflight_prerequisites)
    with pytest.raises(ValueError, match="downstream generation prerequisites"):
        write_v2_phase_rejection(
            tmp_path / "wrong-lock.json",
            "calibration",
            frozen,
            attempt,
            (wrong_lock.file, null.file),
            "generation_exception",
            progress,
        )

    preregistration_path = Path(frozen.file.path)
    os.chmod(preregistration_path, 0o644)
    preregistration_path.write_text("tampered\n", encoding="ascii")
    with pytest.raises(ValueError, match="frozen file identity changed"):
        load_v2_phase_rejection(destination)


def test_rejection_rejects_untyped_upstream_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    current = load_v2_preregistration(frozen.file.path)
    _ledger(tmp_path, monkeypatch)
    calibration = _identity(tmp_path / "not-a-calibration-record.json")
    prerequisites = V2GenerationPrerequisites(
        current,
        frozen,
        calibration_record=calibration,
    )
    lock = _lock(tmp_path / "preflight", "preflight", prerequisites)
    attempt = claim_v2_phase_attempt(frozen, "preflight", {"root": lock.root})
    progress = {
        "attempt": attempt.normalized(),
        "failure": rejection_failure("generation_exception"),
        "status": "rejected",
    }

    with pytest.raises(ValueError, match="invalid v2 calibration JSON"):
        write_v2_phase_rejection(
            tmp_path / "preflight.json",
            "preflight",
            frozen,
            attempt,
            (lock.file,),
            "generation_exception",
            progress,
        )


def test_rejection_is_published_before_mutable_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen(tmp_path)
    attempt = _identity(tmp_path / "attempt.json")
    first_lock = _identity(tmp_path / "first-lock.json")
    second_lock = _identity(tmp_path / "second-lock.json")
    cases: tuple[tuple[Any, str, str, Any], ...] = (
        (calibration_module, "_reject_calibration", "_write_plan", (first_lock, second_lock)),
        (preflight_module, "_reject_preflight", "_write_progress", first_lock),
        (acceptance_module, "_reject_acceptance", "_write_progress", first_lock),
    )
    for index, (module, helper_name, progress_writer, locks) in enumerate(cases):
        order: list[str] = []
        with monkeypatch.context() as patch:
            patch.setattr(
                module,
                "write_v2_phase_rejection",
                lambda *args, **kwargs: order.append("immutable"),
            )
            patch.setattr(
                module,
                progress_writer,
                lambda *args, **kwargs: order.append("mutable"),
            )
            getattr(module, helper_name)(
                tmp_path / f"rejection-{index}.json",
                tmp_path / f"progress-{index}.json",
                {},
                frozen,
                attempt,
                locks,
                "generation_exception",
            )
        assert order == ["immutable", "mutable"]


def test_write_once_fsyncs_file_and_destination_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(record_io_module.os, "fsync", fsync_calls.append)

    write_record_once(tmp_path / "record.json", {"status": "test"}, "test")

    assert len(fsync_calls) == 2
