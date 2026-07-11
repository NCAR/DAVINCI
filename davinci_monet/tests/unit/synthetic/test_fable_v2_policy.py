"""Tests for the versioned FABLE v2 scientific-policy renderer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from davinci_monet.tests.synthetic.fable_v2_policy import (
    FableV2Policy,
    apply_v2_fitting_policy,
    v2_calibration_policies,
    v2_development_policies,
    v2_fitting_policy_values,
    v2_policy_from_normalized,
)


def _template() -> dict[str, object]:
    path = (
        Path(__file__).parents[4]
        / "analyses"
        / "aerosol-tuning"
        / "configs"
        / "fable-synthetic-v2.example.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v2_menu_keeps_control_nonselectable_and_candidates_fixed() -> None:
    development = v2_development_policies()

    assert tuple(policy.policy_id for policy in development) == (
        "v2-sequential-control",
        "v2-joint-seasonal",
        "v2-joint-seasonal-offset",
    )
    assert v2_calibration_policies() == development[1:]


@pytest.mark.parametrize("policy", v2_development_policies())
def test_v2_policy_round_trips_through_the_fitting_template(policy: FableV2Policy) -> None:
    fitting = deepcopy(_template())

    apply_v2_fitting_policy(fitting, policy)

    expected = policy.normalized()
    expected.pop("policy_id")
    expected.pop("simplicity_rank")
    assert v2_fitting_policy_values(fitting) == expected
    assert v2_policy_from_normalized(policy.normalized()) == policy


def test_relative_offsets_require_joint_seasonal_fit() -> None:
    with pytest.raises(ValueError, match="require the joint"):
        FableV2Policy(
            policy_id="invalid",
            bias_fit_method="monthly_mean",
            sensor_offset_method="overlap_zero_sum",
        )


def test_v2_policy_parser_rejects_frozen_base_drift() -> None:
    normalized = v2_calibration_policies()[0].normalized()
    normalized["ridge"] = 0.3

    with pytest.raises(ValueError, match="frozen v1 all-band"):
        v2_policy_from_normalized(normalized)
