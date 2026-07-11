"""Semantic and byte-level validation of frozen FABLE v2 scenario evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec, spec_hash
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_gate
from davinci_monet.tests.synthetic.fable_v2_artifact_evidence import (
    validate_v2_artifact_evidence,
)
from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_policy import FableV2Policy
from davinci_monet.tests.synthetic.fable_v2_record_io import canonical_json, json_sha256

ScoreKind = Literal["recovery", "null"]


def validate_v2_scenario_evidence(
    value: Mapping[str, Any],
    spec: SyntheticTuningSpec,
    policy: FableV2Policy,
    *,
    score_kind: ScoreKind,
    evaluation_splits: Sequence[str],
    verify_files: bool,
) -> None:
    """Validate phase semantics and optionally every referenced file identity."""
    expected = {
        "seed": spec.master_seed,
        "scenario": spec.scenario,
        "score_kind": score_kind,
        "spec_sha256": spec_hash(spec),
        "evaluation_splits": list(evaluation_splits),
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError("v2 scenario evidence does not match its phase spec")
    if value.get("policy") != policy.normalized():
        raise ValueError("v2 scenario evidence does not match its selected policy")
    score = value.get("score")
    if not isinstance(score, Mapping):
        raise ValueError("v2 scenario evidence is missing its score")
    score_passed = score.get("passed")
    if not isinstance(score_passed, bool):
        raise ValueError("v2 scenario score passed state must be boolean")
    evidence_passed = value.get("evidence_passed")
    if not isinstance(evidence_passed, bool):
        raise ValueError("v2 scenario evidence completeness must be boolean")
    elapsed = _finite_nonnegative(value.get("elapsed_seconds"), "elapsed_seconds")
    peak_rss = value.get("process_peak_rss_bytes")
    if isinstance(peak_rss, bool) or not isinstance(peak_rss, int) or peak_rss < 0:
        raise ValueError("v2 scenario process_peak_rss_bytes must be a nonnegative integer")
    resources = value.get("resources")
    expected_resources = resource_gate(elapsed, peak_rss)
    if not isinstance(resources, Mapping) or canonical_json(resources) != canonical_json(
        expected_resources
    ):
        raise ValueError("v2 scenario resources do not match the frozen resource gate")
    expected_passed = score_passed and evidence_passed and bool(expected_resources["passed"])
    if value.get("passed") is not expected_passed:
        raise ValueError("v2 scenario passed state is not derived from its frozen gates")
    diagnostics = score.get("diagnostics")
    if score_kind == "recovery":
        if (
            not isinstance(value.get("diagnostic_report_sha256"), str)
            or not isinstance(value.get("learned_basis_oracle_nrmse"), (int, float))
            or not math.isfinite(float(value["learned_basis_oracle_nrmse"]))
            or not isinstance(diagnostics, Mapping)
            or "best_representable_nrmse" in diagnostics
            or "estimate_vs_unfiltered_in_span_nrmse" not in diagnostics
        ):
            raise ValueError("v2 recovery evidence lacks corrected diagnostic semantics")
    else:
        if value.get("diagnostic_report_sha256") is not None:
            raise ValueError("v2 null evidence unexpectedly contains oracle diagnostics")
        if value.get("report_sha256") != json_sha256(score):
            raise ValueError("v2 null score hash does not match its canonical evidence")
    if not verify_files:
        return
    for name in ("scenario_manifest", "fitting_config", "fitting_manifest"):
        _identity_from_value(value.get(name), name).verify()
    if not completed_manifest_identity(
        _identity_from_value(value.get("fitting_manifest"), "fitting_manifest")
    ):
        raise ValueError("v2 fitting manifest is not completed")
    if score_kind == "recovery":
        _identity_from_value(value.get("evaluation_config"), "evaluation_config").verify()
        evaluation_manifest = _identity_from_value(
            value.get("evaluation_manifest"), "evaluation_manifest"
        )
        if not completed_manifest_identity(evaluation_manifest):
            raise ValueError("v2 evaluation manifest is not completed")
    derived = validate_v2_artifact_evidence(
        value,
        spec,
        policy,
        recovery=score_kind == "recovery",
    )
    if canonical_json(score) != canonical_json(derived.get("score")):
        raise ValueError("v2 scenario score differs from its immutable scientific artifacts")
    if score_kind == "recovery" and value.get("learned_basis_oracle_nrmse") != derived.get(
        "learned_basis_oracle_nrmse"
    ):
        raise ValueError(
            "v2 learned-basis oracle NRMSE differs from its immutable diagnostic artifact"
        )


def completed_manifest_identity(identity: FrozenFileIdentity) -> bool:
    """Require both frozen bytes and a completed pipeline disposition."""
    try:
        identity.verify()
        document = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(document, Mapping) and document.get("status") == "completed"


def _identity_from_value(value: Any, name: str) -> FrozenFileIdentity:
    if not isinstance(value, Mapping):
        raise ValueError(f"v2 scenario evidence is missing {name}")
    try:
        return FrozenFileIdentity(str(value["path"]), str(value["sha256"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"v2 scenario evidence has an invalid {name}") from exc


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"v2 scenario {name} must be finite and nonnegative")
    return float(value)


__all__ = ["completed_manifest_identity", "validate_v2_scenario_evidence"]
