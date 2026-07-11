"""Versioned scientific policies for the FABLE v2 synthetic recovery cycle."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from davinci_monet.tests.synthetic._aerosol_policy import (
    ScientificPolicy,
    apply_fitting_policy,
    fitting_policy_values,
)

BiasFitMethod = Literal["monthly_mean", "joint_seasonal"]
SensorOffsetMethod = Literal["none", "overlap_zero_sum"]


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _v1_all_band_base() -> ScientificPolicy:
    return ScientificPolicy(
        policy_id="fable-v1-all-band-base",
        covariance_model="diagonal_plus_low_rank_common",
        keep_significant=False,
    )


@dataclass(frozen=True)
class FableV2Policy:
    """V2-only controls layered over the frozen v1 all-band policy."""

    policy_id: str
    bias_fit_method: BiasFitMethod
    sensor_offset_method: SensorOffsetMethod = "none"
    joint_bias_laplacian_strength: float = 1.0
    joint_bias_tolerance: float = 1.0e-6
    joint_bias_max_iterations: int = 20
    simplicity_rank: int = 0

    def __post_init__(self) -> None:
        if not self.policy_id or self.policy_id != self.policy_id.strip():
            raise ValueError("policy_id must be a nonempty trimmed string")
        if self.bias_fit_method not in {"monthly_mean", "joint_seasonal"}:
            raise ValueError("unsupported v2 bias_fit_method")
        if self.sensor_offset_method not in {"none", "overlap_zero_sum"}:
            raise ValueError("unsupported v2 sensor_offset_method")
        if self.bias_fit_method != "joint_seasonal" and self.sensor_offset_method != "none":
            raise ValueError("relative sensor offsets require the joint seasonal bias fit")
        if not _finite_number(self.joint_bias_laplacian_strength) or (
            self.joint_bias_laplacian_strength <= 0.0
        ):
            raise ValueError("joint_bias_laplacian_strength must be positive")
        if not _finite_number(self.joint_bias_tolerance) or not (
            0.0 < self.joint_bias_tolerance < 1.0
        ):
            raise ValueError("joint_bias_tolerance must be in (0, 1)")
        if (
            isinstance(self.joint_bias_max_iterations, bool)
            or not isinstance(self.joint_bias_max_iterations, int)
            or self.joint_bias_max_iterations < 1
        ):
            raise ValueError("joint_bias_max_iterations must be positive")
        if (
            isinstance(self.simplicity_rank, bool)
            or not isinstance(self.simplicity_rank, int)
            or self.simplicity_rank < 0
        ):
            raise ValueError("simplicity_rank must be nonnegative")

    def normalized(self) -> dict[str, Any]:
        """Return all frozen v1 controls plus the explicit v2 extension."""
        result = _v1_all_band_base().normalized()
        result.update(
            policy_id=self.policy_id,
            simplicity_rank=self.simplicity_rank,
            bias_fit_method=self.bias_fit_method,
            sensor_offset_method=self.sensor_offset_method,
            joint_bias_laplacian_strength=self.joint_bias_laplacian_strength,
            joint_bias_tolerance=self.joint_bias_tolerance,
            joint_bias_max_iterations=self.joint_bias_max_iterations,
        )
        return result


def v2_development_policies() -> tuple[FableV2Policy, ...]:
    """Return the fixed control and eligible development menu in declared order."""
    return (
        FableV2Policy(
            policy_id="v2-sequential-control",
            bias_fit_method="monthly_mean",
            simplicity_rank=0,
        ),
        FableV2Policy(
            policy_id="v2-joint-seasonal",
            bias_fit_method="joint_seasonal",
            simplicity_rank=1,
        ),
        FableV2Policy(
            policy_id="v2-joint-seasonal-offset",
            bias_fit_method="joint_seasonal",
            sensor_offset_method="overlap_zero_sum",
            simplicity_rank=2,
        ),
    )


def v2_calibration_policies() -> tuple[FableV2Policy, ...]:
    """Return only the two preregistered selectable v2 candidates."""
    return v2_development_policies()[1:]


def apply_v2_fitting_policy(fitting: dict[str, Any], policy: FableV2Policy) -> None:
    """Render one v2 policy without mutating the frozen v1 policy contract."""
    apply_fitting_policy(fitting, _v1_all_band_base())
    projection = fitting["analyses"]["obs_pcs"]
    projection.update(
        bias_fit_method=policy.bias_fit_method,
        sensor_offset_method=policy.sensor_offset_method,
        joint_bias_laplacian_strength=policy.joint_bias_laplacian_strength,
        joint_bias_tolerance=policy.joint_bias_tolerance,
        joint_bias_max_iterations=policy.joint_bias_max_iterations,
    )
    actual = v2_fitting_policy_values(fitting)
    expected = policy.normalized()
    expected.pop("policy_id")
    expected.pop("simplicity_rank")
    if actual != expected:
        raise RuntimeError("rendered fitting config does not match the requested v2 policy")


def v2_fitting_policy_values(fitting: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the complete v2 controls from a rendered fitting configuration."""
    result = fitting_policy_values(fitting)
    projection = fitting["analyses"]["obs_pcs"]
    result.update(
        bias_fit_method=str(projection.get("bias_fit_method", "monthly_mean")),
        sensor_offset_method=str(projection.get("sensor_offset_method", "none")),
        joint_bias_laplacian_strength=float(projection.get("joint_bias_laplacian_strength", 1.0)),
        joint_bias_tolerance=float(projection.get("joint_bias_tolerance", 1.0e-6)),
        joint_bias_max_iterations=int(projection.get("joint_bias_max_iterations", 20)),
    )
    return result


def v2_policy_from_normalized(value: Mapping[str, Any]) -> FableV2Policy:
    """Rebuild a v2 policy from its canonical document and verify the frozen base."""
    fields = dict(value)
    policy = FableV2Policy(
        policy_id=fields.pop("policy_id"),
        bias_fit_method=fields.pop("bias_fit_method"),
        sensor_offset_method=fields.pop("sensor_offset_method"),
        joint_bias_laplacian_strength=fields.pop("joint_bias_laplacian_strength"),
        joint_bias_tolerance=fields.pop("joint_bias_tolerance"),
        joint_bias_max_iterations=fields.pop("joint_bias_max_iterations"),
        simplicity_rank=fields.pop("simplicity_rank"),
    )
    expected_base = _v1_all_band_base().normalized()
    expected_base.pop("policy_id")
    expected_base.pop("simplicity_rank")
    if fields != expected_base:
        raise ValueError("normalized v2 policy changed a frozen v1 all-band control")
    return policy


__all__ = [
    "BiasFitMethod",
    "FableV2Policy",
    "SensorOffsetMethod",
    "apply_v2_fitting_policy",
    "v2_calibration_policies",
    "v2_development_policies",
    "v2_fitting_policy_values",
    "v2_policy_from_normalized",
]
