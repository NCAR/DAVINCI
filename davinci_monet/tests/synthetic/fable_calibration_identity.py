"""Immutable identities for reproducible FABLE synthetic calibration."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from davinci_monet.tests.synthetic._aerosol_policy import ScientificPolicy
from davinci_monet.tests.synthetic.aerosol_calibration import SyntheticCalibrationRecord

CALIBRATION_SEED = 20260710
NULL_SEED = 20260711
CALIBRATION_SPLIT = "calibration"
CALIBRATION_CONFIG_TEMPLATES = (
    "fable-synthetic.example.yaml",
    "fable-synthetic-eval.example.yaml",
)
CALIBRATION_SYNTHETIC_CODE = (
    "_aerosol_bundle.py",
    "_aerosol_contracts.py",
    "_aerosol_inputs.py",
    "_aerosol_io.py",
    "_aerosol_mmr.py",
    "_aerosol_oracles.py",
    "_aerosol_policy.py",
    "_aerosol_stochastic.py",
    "_aerosol_temporal.py",
    "aerosol_calibration.py",
    "fable_calibration_identity.py",
    "fable_calibration_runner.py",
    "fable_thresholds.py",
    "generators.py",
)


def calibration_policy_candidates() -> tuple[ScientificPolicy, ...]:
    """Return the candidate set fixed before calibration scoring."""
    return (
        ScientificPolicy(
            policy_id="fable-v1-diagonal",
            covariance_model="diagonal",
            simplicity_rank=0,
        ),
        ScientificPolicy(
            policy_id="fable-v1-significant",
            covariance_model="diagonal_plus_low_rank_common",
            simplicity_rank=1,
        ),
        ScientificPolicy(
            policy_id="fable-v1-all-band",
            covariance_model="diagonal_plus_low_rank_common",
            keep_significant=False,
            simplicity_rank=2,
        ),
    )


def calibration_code_sha256() -> str:
    """Hash production code, synthetic calibration code, and the environment contract."""
    package_root = Path(__file__).resolve().parents[2]
    repository = package_root.parent
    synthetic_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or "tests" in relative.parts:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    for name in CALIBRATION_SYNTHETIC_CODE:
        digest.update(f"tests/synthetic/{name}".encode("utf-8"))
        digest.update((synthetic_root / name).read_bytes())
    config_root = repository / "analyses" / "aerosol-tuning" / "configs"
    for name in CALIBRATION_CONFIG_TEMPLATES:
        digest.update(f"analyses/aerosol-tuning/configs/{name}".encode("utf-8"))
        digest.update((config_root / name).read_bytes())
    for name in ("environment.yml", "pyproject.toml"):
        path = repository / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_frozen_calibration_record(
    record: SyntheticCalibrationRecord,
) -> SyntheticCalibrationRecord:
    """Require the exact predeclared design and current calibration code identity."""
    expected_policies = {
        policy.policy_id: policy.normalized() for policy in calibration_policy_candidates()
    }
    actual_policies = {
        candidate.policy.policy_id: candidate.policy.normalized() for candidate in record.candidates
    }
    if actual_policies != expected_policies:
        raise ValueError("frozen calibration candidate set does not match the predeclared design")

    expected_code_hash = calibration_code_sha256()
    for candidate in record.candidates:
        evidence = candidate.evidence
        if (
            evidence.calibration_seed != CALIBRATION_SEED
            or evidence.null_seed != NULL_SEED
            or evidence.calibration_scenario != "writer_ci"
            or evidence.null_scenario != "calibration_null"
            or evidence.calibration_split != CALIBRATION_SPLIT
        ):
            raise ValueError("frozen calibration evidence does not use the fixed design")
        code_hash = dict(evidence.hashes)["code_sha256"]
        if not hmac.compare_digest(code_hash, expected_code_hash):
            raise ValueError("frozen calibration evidence does not match the current code")
    return record


__all__ = [
    "CALIBRATION_SEED",
    "CALIBRATION_SPLIT",
    "CALIBRATION_CONFIG_TEMPLATES",
    "CALIBRATION_SYNTHETIC_CODE",
    "NULL_SEED",
    "calibration_code_sha256",
    "calibration_policy_candidates",
    "validate_frozen_calibration_record",
]
