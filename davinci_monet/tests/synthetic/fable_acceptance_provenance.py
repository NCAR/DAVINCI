"""Receipt validation for frozen FABLE acceptance diagnostic inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import xarray as xr

from davinci_monet.analysis.artifact_manifest import validate_finalized_artifact_manifest
from davinci_monet.tests.synthetic._aerosol_io import scientific_dataset_hash
from davinci_monet.tests.synthetic.fable_acceptance_record import file_identity


def validate_seed_receipts(root: Path, seed: int, run: Mapping[str, Any]) -> None:
    """Bind every plotted fitting and oracle input to its frozen receipt."""
    seed_root = root / f"seed-{seed}"
    fitting = run.get("fitting")
    evaluation = run.get("evaluation")
    if not isinstance(fitting, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError(f"seed {seed} acceptance results are incomplete")
    fitting_manifest = seed_root / "output/manifest.json"
    evaluation_manifest = seed_root / "evaluation/manifest.json"
    scenario_path = seed_root / "scenario.json"
    _validate_file_identity(fitting.get("manifest"), fitting_manifest, "fitting manifest")
    _validate_file_identity(evaluation.get("manifest"), evaluation_manifest, "evaluation manifest")
    _validate_file_identity(run.get("scenario_manifest"), scenario_path, "scenario manifest")

    for analysis, role in (
        ("aod_basis", "basis_fit"),
        ("obs_pcs", "projection_fit"),
        ("scaling", "scaling"),
    ):
        paths = sorted((seed_root / f"output/artifacts/{analysis}").glob("chunk-*.nc"))
        validate_finalized_artifact_manifest(
            fitting_manifest,
            paths,
            role=role,
            analysis=analysis,
        )

    recovery_artifact_path(seed_root, run)
    try:
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        truth_receipt = scenario["files"]["oracle/truth.nc"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"seed {seed} scenario has no oracle receipt") from exc
    if (
        scenario.get("root_seed") != seed
        or scenario.get("scenario") != run.get("scenario")
        or not isinstance(truth_receipt, Mapping)
        or truth_receipt.get("role") != "evaluation_only:oracle"
    ):
        raise ValueError(f"seed {seed} scenario oracle receipt does not match acceptance")
    truth_path = seed_root / "oracle/truth.nc"
    if not truth_path.is_file() or _sha256(truth_path) != truth_receipt.get("byte_sha256"):
        raise ValueError(f"seed {seed} oracle checksum does not match its scenario receipt")
    with xr.open_dataset(truth_path) as truth_dataset:
        scientific_hash = scientific_dataset_hash(truth_dataset)
    if scientific_hash != truth_receipt.get("scientific_sha256"):
        raise ValueError(f"seed {seed} oracle scientific hash does not match its receipt")


def recovery_artifact_path(seed_root: Path, run: Mapping[str, Any]) -> Path:
    """Resolve the sole recovery file from its validated acceptance receipt."""
    evaluation = run.get("evaluation")
    recovery = evaluation.get("recovery_artifact") if isinstance(evaluation, Mapping) else None
    expected_root = (seed_root / "evaluation/artifacts/recovery").resolve()
    if (
        not isinstance(recovery, Mapping)
        or Path(str(recovery.get("artifact_dir", ""))).expanduser().resolve() != expected_root
    ):
        raise ValueError("recovery artifact is outside its acceptance run")
    checksums = recovery.get("checksums")
    files = checksums.get("files") if isinstance(checksums, Mapping) else None
    if not isinstance(files, Mapping) or len(files) != 1:
        raise ValueError("recovery artifact must contain exactly one recorded file")
    filename = str(next(iter(files)))
    if Path(filename).name != filename:
        raise ValueError("recovery artifact receipt has an unsafe filename")
    path = expected_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"recovery artifact does not exist: {path}")
    return path


def _validate_file_identity(value: Any, expected: Path, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"acceptance record has no {label} identity")
    try:
        actual = file_identity(expected)
    except OSError as exc:
        raise ValueError(f"acceptance {label} identity changed") from exc
    if dict(value) != actual:
        raise ValueError(f"acceptance {label} identity changed")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["recovery_artifact_path", "validate_seed_receipts"]
