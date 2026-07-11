"""Read-only source tests for FABLE v2 acceptance diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.tests.synthetic import fable_v2_acceptance_diagnostics as diagnostics
from davinci_monet.tests.synthetic.fable_acceptance_spatial_plots import _disposition

_SHA = "0" * 64


class _LockIdentity:
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.verified = False

    def verify(self) -> None:
        self.verified = True


def _evidence(fitting: Path, evaluation: Path) -> dict[str, Any]:
    return {
        "fitting_manifest": {"path": str(fitting), "sha256": _SHA},
        "evaluation_manifest": {"path": str(evaluation), "sha256": _SHA},
    }


def _record(root: Path, *, seed: int = 101, policy_id: str = "policy-a") -> Any:
    lock_path = root / "generation-lock.json"
    lock_path.write_text("{}\n", encoding="ascii")
    run_root = root / "runs" / policy_id / f"seed-{seed}"
    (run_root / "output").mkdir(parents=True)
    (run_root / "evaluation").mkdir()
    bundle_oracle = root / "bundles" / f"seed-{seed}" / "oracle"
    bundle_oracle.mkdir(parents=True)
    (run_root / "oracle").symlink_to(bundle_oracle, target_is_directory=True)
    result = SimpleNamespace(
        seed=seed,
        evidence=_evidence(
            run_root / "output/manifest.json", run_root / "evaluation/manifest.json"
        ),
    )
    return SimpleNamespace(
        preregistration=SimpleNamespace(file=SimpleNamespace(path=str(root / "prereg.json"))),
        calibration_record=object(),
        preflight_record=object(),
        generation_lock=_LockIdentity(lock_path),
        selected_policy=SimpleNamespace(policy_id=policy_id),
        results=(result,),
        status="passed_pending_user_review",
    )


def test_build_source_invokes_lifecycle_and_deep_evidence_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "acceptance"
    root.mkdir()
    record_path = root / "acceptance.json"
    record_path.write_text("{}\n", encoding="ascii")
    record = _record(root)
    lifecycle_calls: list[tuple[Any, ...]] = []
    evidence_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(diagnostics, "load_v2_acceptance_record", lambda path: record)
    monkeypatch.setattr(
        diagnostics, "load_current_v2_preregistration", lambda path: ("current", "frozen")
    )

    def validate_lifecycle(*args: Any) -> Any:
        lifecycle_calls.append(args)
        return record

    def validate_evidence(value: Any, *args: Any, **kwargs: Any) -> None:
        evidence_calls.append({"value": value, **kwargs})

    monkeypatch.setattr(diagnostics, "validate_v2_acceptance_record", validate_lifecycle)
    monkeypatch.setattr(diagnostics, "validate_v2_scenario_evidence", validate_evidence)
    monkeypatch.setattr(diagnostics, "_load_seed_inputs", lambda *args: object())
    monkeypatch.setattr(diagnostics, "_close_seed_inputs", lambda value: None)

    prepared = xr.Dataset(
        {"score_day": ("time", [True])},
        coords={"time": np.array(["2001-10-01"], dtype="datetime64[ns]")},
        attrs={"root_seed": 101},
    )
    monkeypatch.setattr(diagnostics, "_prepare_seed", lambda value: prepared)
    monkeypatch.setattr(diagnostics, "_finalize_seed", lambda value, snapshot: value)

    source = diagnostics.build_v2_acceptance_diagnostic_source(root)

    assert lifecycle_calls == [
        (record, "current", "frozen", record.calibration_record, record.preflight_record)
    ]
    assert len(evidence_calls) == 1
    assert evidence_calls[0]["verify_files"] is True
    assert source.run_root(101) == root / "runs/policy-a/seed-101"
    assert source.dataset.attrs["diagnostic_disposition"] == (
        "diagnostic only; acceptance passed pending user review"
    )
    assert record.generation_lock.verified


def test_run_root_must_match_locked_policy_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "acceptance"
    root.mkdir()
    record = _record(root)
    wrong = root / "runs/other-policy/seed-101/output/manifest.json"
    record.results[0].evidence["fitting_manifest"]["path"] = str(wrong)
    monkeypatch.setattr(diagnostics, "validate_v2_scenario_evidence", lambda *a, **k: None)

    with pytest.raises(ValueError, match="outside its locked policy run root"):
        diagnostics._validated_run_roots(root, record)


def test_collection_config_uses_only_manifest_receipt_files(tmp_path: Path) -> None:
    root = tmp_path / "acceptance"
    run_root = root / "runs/policy-a/seed-101"
    artifact_root = run_root / "output/artifacts/obs_pcs"
    artifact_root.mkdir(parents=True)
    expected = artifact_root / "chunk-00000.nc"
    expected.write_bytes(b"expected")
    (artifact_root / "chunk-99999.nc").write_bytes(b"unreceipted")
    manifest = run_root / "output/manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "analysis_artifacts": [
                    {
                        "analysis": "obs_pcs",
                        "role": "projection_fit",
                        "kind": "netcdf_collection",
                        "artifact_dir": str(artifact_root),
                        "checksums": {"files": {expected.name: "ignored"}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source = diagnostics.V2AcceptanceDiagnosticSource(
        xr.Dataset(),
        root,
        root / "acceptance.json",
        (101,),
        np.datetime64("2001-10-01", "ns"),
        ((101, run_root),),
    )

    config = diagnostics.v2_acceptance_collection_config(
        source, 101, "projection", {"pc": {"units": "1"}}
    )

    assert config["files"] == [str(expected)]
    assert config["artifact_manifest"] == str(manifest)
    assert config["artifact_role"] == "projection_fit"


def test_derived_source_writer_rejects_acceptance_root(tmp_path: Path) -> None:
    root = tmp_path / "acceptance"
    source = diagnostics.V2AcceptanceDiagnosticSource(
        xr.Dataset(),
        root,
        root / "acceptance.json",
        (101,),
        np.datetime64("2001-10-01", "ns"),
        ((101, root / "runs/policy-a/seed-101"),),
    )

    with pytest.raises(ValueError, match="must not alter"):
        diagnostics.write_v2_acceptance_diagnostic_source(source, root / "plots/source.nc")


def test_v2_envelope_detection_and_disposition_are_source_driven(tmp_path: Path) -> None:
    record_path = tmp_path / "acceptance.json"
    record_path.write_text(
        json.dumps(
            {
                "record": {"schema_version": diagnostics.V2_ACCEPTANCE_SCHEMA},
                "record_sha256": _SHA,
            }
        ),
        encoding="ascii",
    )

    assert diagnostics.is_v2_acceptance_record(record_path)
    assert _disposition(SimpleNamespace(attrs={})) == (
        "fable-v1-all-band | diagnostic only; frozen acceptance remains rejected"
    )
    assert (
        _disposition(
            SimpleNamespace(
                attrs={
                    "selected_policy_id": "policy-a",
                    "diagnostic_disposition": "diagnostic only; acceptance failed",
                }
            )
        )
        == "policy-a | diagnostic only; acceptance failed"
    )
