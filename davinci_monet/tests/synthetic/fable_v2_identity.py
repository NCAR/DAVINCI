"""Current code, config, generator, and environment identities for FABLE v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from davinci_monet.tests.synthetic._aerosol_contracts import (
    SCHEMA_VERSION,
    SyntheticTuningSpec,
)
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    ACCEPTANCE_EXCLUDED_FRACTION_MAX,
    resource_limits,
)
from davinci_monet.tests.synthetic.fable_thresholds import RECOVERY_THRESHOLDS
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    FrozenPreregistration,
    V2FileBinding,
    V2Preregistration,
    V2TestEvidence,
    load_v2_preregistration,
    preregistration_sha256,
)

V2_CONFIG_TEMPLATES = (
    "fable-synthetic-v2.example.yaml",
    "fable-synthetic-v2-eval.example.yaml",
)
V2_ENTRY_SCRIPTS = (
    "run_v2_diagnostics.py",
    "calibrate_v2_synthetic.py",
    "run_v2_preflight.py",
    "run_v2_acceptance.py",
)
V2_SYNTHETIC_CODE = (
    "_aerosol_bundle.py",
    "_aerosol_contracts.py",
    "_aerosol_inputs.py",
    "_aerosol_io.py",
    "_aerosol_mmr.py",
    "_aerosol_oracles.py",
    "_aerosol_policy.py",
    "_aerosol_stochastic.py",
    "_aerosol_temporal.py",
    "fable_acceptance_gate.py",
    "fable_thresholds.py",
    "fable_v2_acceptance.py",
    "fable_v2_attempts.py",
    "fable_v2_artifact_evidence.py",
    "fable_v2_bundle.py",
    "fable_v2_calibration.py",
    "fable_v2_calibration_record.py",
    "fable_v2_calibration_runner.py",
    "fable_v2_development.py",
    "fable_v2_development_approval.py",
    "fable_v2_evidence.py",
    "fable_v2_freeze.py",
    "fable_v2_identity.py",
    "fable_v2_lifecycle.py",
    "fable_v2_lock_evidence.py",
    "fable_v2_policy.py",
    "fable_v2_preflight.py",
    "fable_v2_protocol.py",
    "fable_v2_record_io.py",
    "fable_v2_rejection.py",
    "fable_v2_runner.py",
    "fable_v2_scoring.py",
    "generators.py",
)
NULL_FRACTION_MAX = 0.10


def v2_code_sha256() -> str:
    """Hash every executable scientific surface used by the v2 lifecycle."""
    repository = _repository()
    package_root = repository / "davinci_monet"
    paths: list[tuple[str, Path]] = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or "tests" in relative.parts:
            continue
        paths.append((f"davinci_monet/{relative.as_posix()}", path))
    synthetic = package_root / "tests" / "synthetic"
    paths.extend(
        (f"davinci_monet/tests/synthetic/{path.name}", path)
        for path in sorted(synthetic.glob("*.py"))
    )
    scripts = repository / "analyses" / "aerosol-tuning" / "scripts"
    paths.extend(
        (f"analyses/aerosol-tuning/scripts/{name}", scripts / name) for name in V2_ENTRY_SCRIPTS
    )
    return _named_files_sha256(paths)


def v2_environment_sha256() -> str:
    """Hash the declared environment without depending on an active prefix path."""
    repository = _repository()
    return _named_files_sha256(
        [(name, repository / name) for name in ("environment.yml", "pyproject.toml")]
    )


def v2_generator_contract_sha256() -> str:
    """Hash the seed-neutral recovery and null OSSE challenge documents."""
    recovery = SyntheticTuningSpec.synthetic_osse(0).normalized()
    null = SyntheticTuningSpec.synthetic_osse_null(0).normalized()
    recovery["master_seed"] = "${ROLE_SEED}"
    null["master_seed"] = "${ROLE_SEED}"
    document = {
        "recovery": recovery,
        "null": null,
        "schema_version": SCHEMA_VERSION,
    }
    return _json_sha256(document)


def v2_thresholds_sha256() -> str:
    """Hash the unchanged recovery, null, exclusion, and resource gates."""
    return _json_sha256(
        {
            "excluded_fraction_max": ACCEPTANCE_EXCLUDED_FRACTION_MAX,
            "null_fraction_max": NULL_FRACTION_MAX,
            "recovery": RECOVERY_THRESHOLDS,
            "resources": resource_limits(),
        }
    )


def current_v2_preregistration(
    tests: Sequence[V2TestEvidence],
    development_report: str | Path,
    development_approval: str | Path,
) -> V2Preregistration:
    """Build identities only after completed development is explicitly approved."""
    from davinci_monet.tests.synthetic.fable_v2_development_approval import (
        DEVELOPMENT_APPROVAL_BINDING,
        DEVELOPMENT_REPORT_BINDING,
        validate_v2_development_approval,
    )

    repository = _repository()
    config_root = repository / "analyses" / "aerosol-tuning" / "configs"
    report_identity = FrozenFileIdentity.capture(development_report)
    approval_identity = FrozenFileIdentity.capture(development_approval)
    approval = validate_v2_development_approval(approval_identity, verify_files=True)
    if approval.development_report.normalized() != report_identity.normalized():
        raise ValueError("v2 development approval does not bind the requested report")
    return V2Preregistration(
        generator_schema_version=SCHEMA_VERSION,
        generator_spec_sha256=v2_generator_contract_sha256(),
        thresholds_sha256=v2_thresholds_sha256(),
        code_sha256=v2_code_sha256(),
        environment_sha256=v2_environment_sha256(),
        configs=tuple(
            V2FileBinding.capture(name, config_root / name, relative_to=repository)
            for name in V2_CONFIG_TEMPLATES
        )
        + (
            V2FileBinding.capture(DEVELOPMENT_REPORT_BINDING, report_identity.path),
            V2FileBinding.capture(DEVELOPMENT_APPROVAL_BINDING, approval_identity.path),
        ),
        tests=tuple(tests),
    )


def load_current_v2_preregistration(
    path: str | Path,
) -> tuple[V2Preregistration, FrozenPreregistration]:
    """Load frozen test evidence and reconstruct every current scientific identity."""
    source = Path(path).expanduser().resolve()
    frozen_value = load_v2_preregistration(source)
    bindings = {item.name: item for item in frozen_value.configs}
    try:
        development_report = bindings["development_report"].path
        development_approval = bindings["development_approval"].path
    except KeyError as exc:
        raise ValueError("frozen v2 preregistration lacks development approval") from exc
    current = current_v2_preregistration(
        frozen_value.tests,
        development_report,
        development_approval,
    )
    frozen = FrozenPreregistration(
        FrozenFileIdentity.capture(source), preregistration_sha256(frozen_value)
    )
    return current, frozen


def _named_files_sha256(paths: Sequence[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for name, path in paths:
        if not path.is_file():
            raise ValueError(f"v2 identity source is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _repository() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = [
    "NULL_FRACTION_MAX",
    "V2_CONFIG_TEMPLATES",
    "V2_ENTRY_SCRIPTS",
    "V2_SYNTHETIC_CODE",
    "current_v2_preregistration",
    "load_current_v2_preregistration",
    "v2_code_sha256",
    "v2_environment_sha256",
    "v2_generator_contract_sha256",
    "v2_thresholds_sha256",
]
