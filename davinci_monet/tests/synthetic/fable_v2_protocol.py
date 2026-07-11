"""Fixed seed roles and scientific protocol identity for FABLE v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

CYCLE_ID = "fable-recovery-v2"
PROTOCOL_SCHEMA = "fable-v2-protocol-v1"
PREREGISTRATION_SCHEMA = "fable-v2-preregistration-v1"
GENERATION_LOCK_SCHEMA = "fable-v2-generation-lock-v1"

ACCEPTANCE_SEEDS = (1969, 2010, 2013)
V1_EXPOSED_SEEDS = (1179, 2358, 11, 20260710, 20260711, 20260712)
ROLE_ORDER = (
    "development",
    "calibration_recovery",
    "calibration_null",
    "preflight",
    "acceptance",
)
DERIVED_ROLE_COUNTS = MappingProxyType(
    {
        "development": 3,
        "calibration_recovery": 3,
        "calibration_null": 3,
        "preflight": 1,
    }
)
ENTRY_POINTS = (
    "run_v2_diagnostics.py",
    "calibrate_v2_synthetic.py",
    "run_v2_preflight.py",
    "run_v2_acceptance.py",
)
RANKING_ORDER = (
    "equal_seed_mean_field_nrmse",
    "equal_seed_mean_aod_rmse_ratio",
    "equal_seed_mean_abs_field_origin_slope_error",
    "simplicity_rank",
    "policy_id",
)
_MAX_SEED = 2**63 - 1


@dataclass(frozen=True)
class V2CandidateDefinition:
    """One method fixed before v2 development scoring."""

    policy_id: str
    bias_anomaly_fit: str
    relative_sensor_offsets: str
    eligible_for_calibration: bool
    simplicity_rank: int

    def normalized(self) -> dict[str, Any]:
        return {
            "bias_anomaly_fit": self.bias_anomaly_fit,
            "eligible_for_calibration": self.eligible_for_calibration,
            "policy_id": self.policy_id,
            "relative_sensor_offsets": self.relative_sensor_offsets,
            "simplicity_rank": self.simplicity_rank,
        }


CANDIDATE_MENU = (
    V2CandidateDefinition(
        "v2-sequential-control",
        "frozen_v1_sequential_monthly_mean",
        "none",
        False,
        0,
    ),
    V2CandidateDefinition("v2-joint-seasonal", "joint_seasonal", "none", True, 1),
    V2CandidateDefinition(
        "v2-joint-seasonal-offset",
        "joint_seasonal",
        "overlap_zero_sum",
        True,
        2,
    ),
)


def derive_role_seed(role: str, index: int) -> int:
    """Derive one predeclared non-acceptance seed from its role and index."""
    if role not in DERIVED_ROLE_COUNTS:
        raise ValueError(f"role does not use deterministic derivation: {role!r}")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("seed index must be an integer")
    if not 0 <= index < DERIVED_ROLE_COUNTS[role]:
        raise ValueError(f"seed index is outside the predeclared {role!r} role")
    payload = b"fable-v2\0" + role.encode("ascii") + b"\0" + str(index).encode("ascii")
    raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
    return raw & _MAX_SEED


def seed_roles() -> dict[str, tuple[int, ...]]:
    """Return a fresh copy of the complete ordered v2 seed-role assignment."""
    roles = {
        role: tuple(derive_role_seed(role, index) for index in range(count))
        for role, count in DERIVED_ROLE_COUNTS.items()
    }
    roles["acceptance"] = ACCEPTANCE_SEEDS
    return {role: roles[role] for role in ROLE_ORDER}


def validate_seed_roles(
    roles: Mapping[str, Sequence[int]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Validate the exact role table, disjointness, and historical denylists."""
    if set(roles) != set(ROLE_ORDER):
        raise ValueError("seed roles must contain the exact v2 role set")
    canonical: list[tuple[str, tuple[int, ...]]] = []
    all_values: list[int] = []
    expected = seed_roles()
    for role in ROLE_ORDER:
        values = _seed_tuple(roles[role])
        if values != expected[role]:
            raise ValueError(f"{role!r} seeds do not match the predeclared v2 assignment")
        canonical.append((role, values))
        all_values.extend(values)
    if len(all_values) != len(set(all_values)):
        raise ValueError("v2 seed roles must be mutually exclusive")
    if set(all_values) & set(V1_EXPOSED_SEEDS):
        raise ValueError("v2 seed roles overlap the exposed v1 seed denylist")
    non_acceptance = all_values[: -len(ACCEPTANCE_SEEDS)]
    if set(non_acceptance) & set(ACCEPTANCE_SEEDS):
        raise ValueError("acceptance seeds cannot appear in a non-acceptance role")
    return tuple(canonical)


def validate_role_seeds(role: str, seeds: Sequence[int]) -> tuple[int, ...]:
    """Require a runner to use one complete role tuple in its frozen order."""
    if role not in ROLE_ORDER:
        raise ValueError(f"unknown v2 seed role: {role!r}")
    values = _seed_tuple(seeds)
    expected = seed_roles()[role]
    if values != expected:
        if role == "acceptance":
            raise ValueError("acceptance requires the exact ordered seed tuple")
        if set(values) & set(ACCEPTANCE_SEEDS):
            raise ValueError("acceptance seeds are denied outside the acceptance role")
        if set(values) & set(V1_EXPOSED_SEEDS):
            raise ValueError("exposed v1 seeds are denied in every v2 role")
        raise ValueError(f"{role!r} requires its complete predeclared seed tuple")
    return values


def protocol_document() -> dict[str, Any]:
    """Return the canonical scientific choices fixed before implementation."""
    roles = validate_seed_roles(seed_roles())
    return {
        "acceptance_seeds": list(ACCEPTANCE_SEEDS),
        "candidate_menu": [candidate.normalized() for candidate in CANDIDATE_MENU],
        "cycle_id": CYCLE_ID,
        "entry_points": list(ENTRY_POINTS),
        "ranking_order": list(RANKING_ORDER),
        "schema_version": PROTOCOL_SCHEMA,
        "seed_derivation": {
            "algorithm": "sha256_first_8_bytes_little_endian_low_63_bits",
            "domain_separator": "fable-v2\u0000{role}\u0000{index}",
        },
        "seed_roles": {role: list(values) for role, values in roles},
        "v1_exposed_seed_denylist": list(V1_EXPOSED_SEEDS),
    }


def protocol_sha256() -> str:
    """Return the identity of the fixed role, candidate, and entry-point protocol."""
    payload = json.dumps(
        protocol_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _seed_tuple(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("seed values must be an integer sequence")
    result = tuple(values)
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED
        for seed in result
    ):
        raise ValueError("seed values must be integers in [0, 2**63 - 1]")
    return result


__all__ = [
    "ACCEPTANCE_SEEDS",
    "CANDIDATE_MENU",
    "CYCLE_ID",
    "DERIVED_ROLE_COUNTS",
    "ENTRY_POINTS",
    "RANKING_ORDER",
    "ROLE_ORDER",
    "V1_EXPOSED_SEEDS",
    "V2CandidateDefinition",
    "derive_role_seed",
    "protocol_document",
    "protocol_sha256",
    "seed_roles",
    "validate_role_seeds",
    "validate_seed_roles",
]
