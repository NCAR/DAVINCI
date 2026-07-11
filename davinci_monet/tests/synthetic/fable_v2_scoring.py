"""Single-source scientific score construction for FABLE v2."""

from __future__ import annotations

from typing import Any

import xarray as xr

from davinci_monet.tests.synthetic._aerosol_policy import ScientificPolicy
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    evaluate_synthetic_recovery_gate,
)
from davinci_monet.tests.synthetic.fable_calibration_runner import evaluate_null_control
from davinci_monet.tests.synthetic.fable_v2_policy import FableV2Policy


def v2_scientific_policy(policy: FableV2Policy) -> ScientificPolicy:
    """Return the frozen v1 controls shared by every v2 candidate."""
    return ScientificPolicy(
        policy_id=policy.policy_id,
        covariance_model="diagonal_plus_low_rank_common",
        keep_significant=False,
    )


def v2_recovery_score(report: xr.Dataset) -> dict[str, Any]:
    """Apply the production recovery gate with the corrected v2 diagnostic name."""
    gate = evaluate_synthetic_recovery_gate(report)
    diagnostics = gate.get("diagnostics")
    if isinstance(diagnostics, dict) and "best_representable_nrmse" in diagnostics:
        diagnostics["estimate_vs_unfiltered_in_span_nrmse"] = diagnostics.pop(
            "best_representable_nrmse"
        )
    requirements = gate.get("diagnostic_requirements")
    if isinstance(requirements, dict):
        variables = requirements.get("required_variables")
        if isinstance(variables, list):
            requirements["required_variables"] = [
                (
                    "estimate_vs_unfiltered_in_span_nrmse"
                    if name == "best_representable_nrmse"
                    else name
                )
                for name in variables
            ]
    return gate


def v2_null_score(
    scaling: xr.Dataset,
    filtered: xr.Dataset,
    truth: xr.Dataset,
    policy: FableV2Policy,
) -> dict[str, Any]:
    """Apply the exact frozen null evaluator and hard thresholds."""
    metrics = evaluate_null_control(
        scaling,
        filtered,
        truth,
        v2_scientific_policy(policy),
        split="calibration",
    )
    return {
        "metrics": metrics,
        "passed": bool(
            metrics["null_retained_energy_fraction"] <= 0.10
            and metrics["null_significant_fraction"] <= 0.10
        ),
    }


__all__ = ["v2_null_score", "v2_recovery_score", "v2_scientific_policy"]
