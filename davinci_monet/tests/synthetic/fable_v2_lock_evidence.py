"""Independent verification of frozen FABLE v2 generation-lock identities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_lifecycle import V2GenerationPrerequisites
from davinci_monet.tests.synthetic.fable_v2_protocol import (
    CYCLE_ID,
    GENERATION_LOCK_SCHEMA,
    protocol_sha256,
    validate_role_seeds,
)


def validate_v2_generation_lock_identity(
    identity: FrozenFileIdentity,
    role: str,
    seeds: Sequence[int],
    prerequisites: V2GenerationPrerequisites,
) -> Path:
    """Rebuild and verify canonical lock bytes without trusting a live lock object."""
    values = validate_role_seeds(role, seeds)
    prerequisites.verify()
    identity.verify()
    path = Path(identity.path).expanduser().resolve()
    if path.name != "generation-lock.json":
        raise ValueError("v2 generation lock has a noncanonical filename")
    expected = {
        "cycle_id": CYCLE_ID,
        "prerequisites": prerequisites.normalized(),
        "protocol_sha256": protocol_sha256(),
        "role": role,
        "schema_version": GENERATION_LOCK_SCHEMA,
        "seeds": list(values),
        "status": "locked_before_generation",
    }
    raw = path.read_text(encoding="ascii")
    if raw != json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n":
        raise ValueError("v2 generation lock canonical identity changed")
    return path.parent


__all__ = ["validate_v2_generation_lock_identity"]
