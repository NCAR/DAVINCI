"""Development-report aggregation contracts for FABLE v2."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import davinci_monet.tests.synthetic.fable_v2_development as development
from davinci_monet.tests.synthetic.fable_v2_policy import v2_development_policies
from davinci_monet.tests.synthetic.fable_v2_protocol import seed_roles


def test_development_reports_control_and_failed_seed_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = seed_roles()["development"]
    policies = v2_development_policies()
    runs: list[dict[str, Any]] = []
    for policy in policies:
        for seed in seeds:
            passed = not (policy == policies[1] and seed == seeds[1])
            runs.append(
                {
                    "outcome": {"seed": seed},
                    "passed": passed,
                    "policy_id": policy.policy_id,
                    "seed": seed,
                    "status": "completed" if passed else "failed",
                }
            )

    def result_from_run(run: dict[str, Any]) -> SimpleNamespace:
        seed = int(run["seed"])
        return SimpleNamespace(
            field_correlation=0.9,
            field_origin_slope=1.0,
            field_nrmse=0.2 + (0.2 if not run["passed"] else 0.0),
            aod_rmse_ratio=0.5,
            full_target_aod_rmse_ratio=0.7,
            excluded_fraction=0.2,
            seed=seed,
        )

    monkeypatch.setattr(development, "_result_from_run", result_from_run)
    monkeypatch.setattr(development, "aggregate_recovery_failures", lambda values: ())

    assessments = development._candidate_assessments(runs)

    assert [item["policy_id"] for item in assessments] == [policy.policy_id for policy in policies]
    assert assessments[0]["aggregate"]["field_nrmse"] == pytest.approx(0.2)
    assert assessments[0]["eligible_for_calibration"] is False
    assert "diagnostic_only_policy" in assessments[0]["rejection_reasons"]
    assert assessments[1]["aggregate"]["field_nrmse"] == pytest.approx(0.2 + 0.2 / 3.0)
    assert assessments[1]["eligible_for_calibration"] is False
    assert f"seed_{seeds[1]}:per_seed_gate_failed" in assessments[1]["rejection_reasons"]
    assert assessments[2]["eligible_for_calibration"] is True
