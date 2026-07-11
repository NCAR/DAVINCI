"""Contract tests for the FABLE v2 seed, freeze, and lifecycle protocol."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    V2FileBinding,
    V2Preregistration,
    V2TestEvidence,
    load_v2_preregistration,
    preregistration_sha256,
    verify_v2_preregistration,
    write_v2_preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import (
    V2GenerationPrerequisites,
    prepare_v2_generation,
    validate_v2_generation_request,
    verify_v2_generation_lock,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import (
    ACCEPTANCE_SEEDS,
    CANDIDATE_MENU,
    ENTRY_POINTS,
    RANKING_ORDER,
    V1_EXPOSED_SEEDS,
    derive_role_seed,
    protocol_document,
    protocol_sha256,
    seed_roles,
    validate_role_seeds,
    validate_seed_roles,
)

EXPECTED_ROLES = {
    "development": (
        8958027244578499926,
        7058240817492126009,
        6541432702848222996,
    ),
    "calibration_recovery": (
        4720161833425845668,
        7615923448626770708,
        7027338798249911494,
    ),
    "calibration_null": (
        6922119454902611484,
        8687442551985640685,
        1663300583890477700,
    ),
    "preflight": (736479105464814019,),
    "acceptance": (1969, 2010, 2013),
}


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _preregistration(*, code_sha256: str | None = None) -> V2Preregistration:
    return V2Preregistration(
        generator_schema_version="fable-synthetic-v1",
        generator_spec_sha256=_hash("generator-spec"),
        thresholds_sha256=_hash("thresholds"),
        code_sha256=code_sha256 or _hash("code"),
        environment_sha256=_hash("environment"),
        configs=(
            V2FileBinding("fitting", "configs/fitting.yaml", _hash("fitting")),
            V2FileBinding("evaluation", "configs/evaluation.yaml", _hash("evaluation")),
        ),
        tests=(
            V2TestEvidence(
                "v2-unit-and-integration",
                "pytest v2 tests",
                _hash("test-report"),
            ),
        ),
    )


def _frozen_preregistration(tmp_path: Path):
    preregistration = _preregistration()
    frozen = write_v2_preregistration(tmp_path / "preregistration.json", preregistration)
    return preregistration, frozen


def _identity(tmp_path: Path, name: str, content: str) -> FrozenFileIdentity:
    path = tmp_path / name
    path.write_text(content, encoding="ascii")
    return FrozenFileIdentity.capture(path)


def test_role_seed_derivation_and_fixed_assignments_are_exact() -> None:
    roles = seed_roles()

    assert roles == EXPECTED_ROLES
    assert ACCEPTANCE_SEEDS == (1969, 2010, 2013)
    for role, values in EXPECTED_ROLES.items():
        if role == "acceptance":
            continue
        for index, value in enumerate(values):
            payload = f"fable-v2\0{role}\0{index}".encode("ascii")
            independent = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
            assert derive_role_seed(role, index) == independent & (2**63 - 1) == value

    roles["development"] = (1,)
    assert seed_roles() == EXPECTED_ROLES
    with pytest.raises(ValueError, match="does not use deterministic derivation"):
        derive_role_seed("acceptance", 0)
    with pytest.raises(ValueError, match="outside the predeclared"):
        derive_role_seed("development", 3)
    with pytest.raises(ValueError, match="must be an integer"):
        derive_role_seed("development", True)


def test_role_validation_enforces_order_denylists_and_exact_completeness() -> None:
    canonical = validate_seed_roles(EXPECTED_ROLES)
    assert dict(canonical) == EXPECTED_ROLES
    assert len({seed for values in EXPECTED_ROLES.values() for seed in values}) == 13
    assert not set(V1_EXPOSED_SEEDS) & {
        seed for values in EXPECTED_ROLES.values() for seed in values
    }

    missing = dict(EXPECTED_ROLES)
    missing.pop("preflight")
    with pytest.raises(ValueError, match="exact v2 role set"):
        validate_seed_roles(missing)
    changed = dict(EXPECTED_ROLES)
    changed["preflight"] = (1,)
    with pytest.raises(ValueError, match="predeclared v2 assignment"):
        validate_seed_roles(changed)
    with pytest.raises(ValueError, match="exact ordered seed tuple"):
        validate_role_seeds("acceptance", (2010, 1969, 2013))
    with pytest.raises(ValueError, match="acceptance seeds are denied"):
        validate_role_seeds("development", (1969, 2, 3))
    with pytest.raises(ValueError, match="exposed v1 seeds are denied"):
        validate_role_seeds("development", (1179, 2, 3))
    with pytest.raises(ValueError, match="complete predeclared seed tuple"):
        validate_role_seeds("preflight", (123,))
    with pytest.raises(ValueError, match="integers"):
        validate_role_seeds("preflight", (True,))
    with pytest.raises(ValueError, match="unknown v2 seed role"):
        validate_role_seeds("other", (1,))


def test_protocol_identity_binds_candidates_roles_ranking_and_entry_points() -> None:
    document = protocol_document()

    assert document["acceptance_seeds"] == [1969, 2010, 2013]
    assert document["seed_roles"] == {role: list(values) for role, values in EXPECTED_ROLES.items()}
    assert document["v1_exposed_seed_denylist"] == list(V1_EXPOSED_SEEDS)
    assert document["candidate_menu"] == [item.normalized() for item in CANDIDATE_MENU]
    assert document["ranking_order"] == list(RANKING_ORDER)
    assert document["entry_points"] == list(ENTRY_POINTS)
    expected = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    assert protocol_sha256() == expected


def test_preregistration_validates_complete_unique_passing_bindings() -> None:
    preregistration = _preregistration()

    assert [item.name for item in preregistration.configs] == ["evaluation", "fitting"]
    assert preregistration.normalized()["protocol_sha256"] == protocol_sha256()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(preregistration, code_sha256="bad")
    with pytest.raises(ValueError, match="uniquely named"):
        replace(preregistration, configs=(preregistration.configs[0],) * 2)
    with pytest.raises(ValueError, match="passing test evidence"):
        V2TestEvidence("suite", "pytest", _hash("evidence"), passed=False)


def test_file_binding_capture_supports_stable_relative_paths(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "fit.yaml"
    config.parent.mkdir()
    config.write_text("synthetic: true\n", encoding="ascii")

    binding = V2FileBinding.capture("fit", config, relative_to=tmp_path)

    assert binding.path == "configs/fit.yaml"
    assert binding.sha256 == hashlib.sha256(config.read_bytes()).hexdigest()


def test_preregistration_is_canonical_write_once_and_read_only(tmp_path: Path) -> None:
    preregistration, frozen = _frozen_preregistration(tmp_path)
    path = Path(frozen.file.path)

    assert load_v2_preregistration(path) == preregistration
    assert verify_v2_preregistration(frozen, preregistration) == preregistration
    assert frozen.preregistration_sha256 == preregistration_sha256(preregistration)
    assert path.stat().st_mode & 0o222 == 0
    assert path.read_bytes().endswith(b"\n")
    assert list(tmp_path.glob(".*.tmp")) == []
    with pytest.raises(FileExistsError, match="already exists"):
        write_v2_preregistration(path, preregistration)


def test_preregistration_detects_current_file_and_protocol_mutation(tmp_path: Path) -> None:
    preregistration, frozen = _frozen_preregistration(tmp_path)
    changed = replace(preregistration, code_sha256=_hash("changed-code"))
    with pytest.raises(ValueError, match="scientific identity drifted"):
        verify_v2_preregistration(frozen, changed)

    path = Path(frozen.file.path)
    os.chmod(path, 0o644)
    document = json.loads(path.read_text(encoding="ascii"))
    document["preregistration"]["protocol"]["acceptance_seeds"] = [1, 2, 3]
    document["preregistration_sha256"] = hashlib.sha256(
        json.dumps(document["preregistration"], sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="frozen file identity changed"):
        verify_v2_preregistration(frozen, preregistration)
    with pytest.raises(ValueError, match="protocol identity mismatch"):
        load_v2_preregistration(path)


def test_phase_guards_require_strictly_ordered_frozen_evidence(tmp_path: Path) -> None:
    roles = seed_roles()
    preregistration, frozen = _frozen_preregistration(tmp_path)
    calibration = _identity(tmp_path, "calibration.json", "selected\n")
    preflight = _identity(tmp_path, "preflight.json", "passed\n")
    registered = V2GenerationPrerequisites(preregistration, frozen)
    selected = V2GenerationPrerequisites(preregistration, frozen, calibration)
    passed = V2GenerationPrerequisites(preregistration, frozen, calibration, preflight, "passed")

    assert (
        validate_v2_generation_request("development", roles["development"]) == roles["development"]
    )
    with pytest.raises(ValueError, match="closed once preregistration"):
        validate_v2_generation_request("development", roles["development"], registered)
    with pytest.raises(ValueError, match="requires a valid frozen preregistration"):
        validate_v2_generation_request("calibration_recovery", roles["calibration_recovery"])
    assert (
        validate_v2_generation_request(
            "calibration_recovery", roles["calibration_recovery"], registered
        )
        == roles["calibration_recovery"]
    )
    assert (
        validate_v2_generation_request("calibration_null", roles["calibration_null"], registered)
        == roles["calibration_null"]
    )
    with pytest.raises(ValueError, match="requires frozen preregistration and calibration"):
        validate_v2_generation_request("preflight", roles["preflight"], registered)
    assert (
        validate_v2_generation_request("preflight", roles["preflight"], selected)
        == roles["preflight"]
    )
    with pytest.raises(ValueError, match="until all prerequisite identities"):
        validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS, selected)
    failed = replace(passed, preflight_status="failed")
    with pytest.raises(ValueError, match="unless the frozen preflight passed"):
        validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS, failed)
    assert validate_v2_generation_request("acceptance", ACCEPTANCE_SEEDS, passed) == (
        1969,
        2010,
        2013,
    )


def test_generation_guard_rejects_drift_before_creating_root(tmp_path: Path) -> None:
    roles = seed_roles()
    preregistration, frozen = _frozen_preregistration(tmp_path)
    calibration = _identity(tmp_path, "calibration.json", "selected\n")
    selected = V2GenerationPrerequisites(preregistration, frozen, calibration)
    Path(calibration.path).write_text("mutated\n", encoding="ascii")
    root = tmp_path / "preflight-run"

    with pytest.raises(ValueError, match="frozen file identity changed"):
        prepare_v2_generation(root, "preflight", roles["preflight"], selected)
    assert not root.exists()


def test_generation_root_is_exclusive_locked_first_and_mutation_detected(
    tmp_path: Path,
) -> None:
    roles = seed_roles()
    root = tmp_path / "development-run"
    lock = prepare_v2_generation(root, "development", roles["development"])
    document = json.loads(lock.path.read_text(encoding="ascii"))

    assert document["role"] == "development"
    assert document["seeds"] == list(roles["development"])
    assert document["protocol_sha256"] == protocol_sha256()
    assert document["status"] == "locked_before_generation"
    assert lock.path.stat().st_mode & 0o222 == 0
    assert list(root.iterdir()) == [lock.path]
    verify_v2_generation_lock(lock)
    with pytest.raises(FileExistsError):
        prepare_v2_generation(root, "development", roles["development"])

    os.chmod(lock.path, 0o644)
    lock.path.write_text("{}\n", encoding="ascii")
    with pytest.raises(ValueError, match="generation lock changed"):
        verify_v2_generation_lock(lock)
