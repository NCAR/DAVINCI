"""Deep artifact, MMR, and report-hash verification for FABLE v2 evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.analysis.artifact_manifest import validate_finalized_artifact_manifest
from davinci_monet.analysis.artifacts import load_dataset_collection
from davinci_monet.analysis.mmr_writer_io import mmr_payload_sha256, sha256_file
from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic._aerosol_io import scientific_dataset_hash
from davinci_monet.tests.synthetic.fable_v2_bundle import validate_v2_scenario_bundle
from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_policy import FableV2Policy
from davinci_monet.tests.synthetic.fable_v2_scoring import (
    v2_null_score,
    v2_recovery_score,
)


def validate_v2_artifact_evidence(
    value: Mapping[str, Any],
    spec: SyntheticTuningSpec,
    policy: FableV2Policy,
    *,
    recovery: bool,
) -> Mapping[str, Any]:
    """Rehash artifacts and rederive every gate-driving scientific value."""
    scenario = _identity(value, "scenario_manifest")
    scenario.verify()
    bundle_root = Path(scenario.path).resolve().parent
    validate_v2_scenario_bundle(bundle_root, spec, verify_scientific=True)
    fitting_manifest = _identity(value, "fitting_manifest")
    fitting_entries = _manifest_entries(fitting_manifest)
    for role, analysis in (
        ("basis_fit", "aod_basis"),
        ("projection_fit", "obs_pcs"),
    ):
        _validate_artifact_entry(fitting_manifest, fitting_entries, role, analysis)
    scaling_entry = _validate_artifact_entry(
        fitting_manifest, fitting_entries, "scaling", "scaling"
    )
    _validate_mmr_entries(fitting_entries, bundle_root, Path(fitting_manifest.path).parents[1])
    if not recovery:
        scaling = _load_artifact(scaling_entry)
        truth: xr.Dataset | None = None
        try:
            truth = xr.open_dataset(bundle_root / "oracle" / "truth.nc")
            return {"score": v2_null_score(scaling, scaling, truth, policy)}
        finally:
            scaling.close()
            if truth is not None:
                truth.close()
    evaluation_manifest = _identity(value, "evaluation_manifest")
    evaluation_entries = _manifest_entries(evaluation_manifest)
    recovery_entry = _validate_artifact_entry(
        evaluation_manifest, evaluation_entries, "recovery_report", "recovery"
    )
    diagnostic_entry = _validate_artifact_entry(
        evaluation_manifest,
        evaluation_entries,
        "v2_diagnostic_report",
        "v2_diagnostics",
    )
    recovery_report = _load_artifact(recovery_entry)
    diagnostic_report = _load_artifact(diagnostic_entry)
    try:
        if scientific_dataset_hash(recovery_report) != value.get("report_sha256"):
            raise ValueError("v2 recovery artifact scientific hash changed")
        if scientific_dataset_hash(diagnostic_report) != value.get("diagnostic_report_sha256"):
            raise ValueError("v2 diagnostic artifact scientific hash changed")
        if "learned_basis_oracle_nrmse" not in diagnostic_report:
            raise ValueError("v2 diagnostic artifact lacks learned_basis_oracle_nrmse")
        learned = float(diagnostic_report["learned_basis_oracle_nrmse"].load().item())
        if not np.isfinite(learned):
            raise ValueError("v2 diagnostic learned-basis oracle NRMSE is not finite")
        return {
            "learned_basis_oracle_nrmse": learned,
            "score": v2_recovery_score(recovery_report),
        }
    finally:
        recovery_report.close()
        diagnostic_report.close()


def _manifest_entries(identity: FrozenFileIdentity) -> tuple[Mapping[str, Any], ...]:
    identity.verify()
    try:
        document = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("v2 pipeline manifest is invalid JSON") from exc
    entries = document.get("analysis_artifacts") if isinstance(document, Mapping) else None
    if (
        not isinstance(document, Mapping)
        or document.get("status") != "completed"
        or not isinstance(entries, list)
    ):
        raise ValueError("v2 pipeline manifest is incomplete")
    if not all(isinstance(item, Mapping) for item in entries):
        raise ValueError("v2 pipeline manifest has malformed artifact entries")
    return tuple(entries)


def _validate_artifact_entry(
    manifest: FrozenFileIdentity,
    entries: tuple[Mapping[str, Any], ...],
    role: str,
    analysis: str,
) -> Mapping[str, Any]:
    matches = [
        item for item in entries if item.get("role") == role and item.get("analysis") == analysis
    ]
    if len(matches) != 1:
        raise ValueError(f"v2 manifest must contain one {analysis}/{role} artifact")
    entry = matches[0]
    paths = _artifact_paths(entry)
    validated = validate_finalized_artifact_manifest(
        manifest.path, paths, role=role, analysis=analysis
    )
    if dict(validated) != dict(entry):
        raise ValueError(f"v2 {analysis}/{role} artifact receipt changed")
    return entry


def _artifact_paths(entry: Mapping[str, Any]) -> list[Path]:
    checksums = entry.get("checksums")
    if not isinstance(checksums, Mapping):
        raise ValueError("v2 artifact has no checksum table")
    if entry.get("kind") == "netcdf_collection":
        files = checksums.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ValueError("v2 artifact collection has no files")
        root = Path(str(entry.get("artifact_dir", "")))
        return [root / str(name) for name in sorted(files)]
    if entry.get("kind") == "product":
        return [Path(str(entry.get("artifact_path", "")))]
    raise ValueError("v2 artifact kind is unsupported")


def _load_artifact(entry: Mapping[str, Any]) -> xr.Dataset:
    return load_dataset_collection(_artifact_paths(entry))


def _validate_mmr_entries(
    entries: tuple[Mapping[str, Any], ...], bundle_root: Path, run_root: Path
) -> None:
    mmr_entries = [item for item in entries if item.get("role") == "corrected_mmr"]
    expected_inputs = {path.resolve() for path in (bundle_root / "inputs/mmr").glob("*.nc4")}
    actual_inputs: set[Path] = set()
    for entry in mmr_entries:
        input_path = Path(str(entry.get("input_path", "")))
        output_path = Path(str(entry.get("path", "")))
        checksums = entry.get("checksums")
        if (
            entry.get("kind") != "mmr_file"
            or entry.get("status") not in {"written", "resumed"}
            or not isinstance(checksums, Mapping)
            or input_path.resolve() not in expected_inputs
            or output_path.resolve().parent != (run_root / "corrected").resolve()
            or sha256_file(input_path) != checksums.get("input_sha256")
            or sha256_file(output_path) != checksums.get("output_sha256")
            or mmr_payload_sha256(output_path) != checksums.get("payload_sha256")
            or not all(
                checksums.get(name) for name in ("scaling_sha256", "config_sha256", "code_sha256")
            )
        ):
            raise ValueError("v2 corrected MMR artifact evidence is invalid")
        actual_inputs.add(input_path.resolve())
    if actual_inputs != expected_inputs:
        raise ValueError("v2 corrected MMR receipts do not cover every synthetic input")


def _identity(value: Mapping[str, Any], name: str) -> FrozenFileIdentity:
    raw = value.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"v2 evidence is missing {name}")
    return FrozenFileIdentity(str(raw.get("path")), str(raw.get("sha256")))


__all__ = ["validate_v2_artifact_evidence"]
