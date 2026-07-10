"""Canonical scientific controls frozen before synthetic acceptance."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScientificPolicy:
    """Complete tunable FABLE policy; scenario geometry and dates are separate evidence."""

    policy_id: str
    n_modes: int = 3
    eof_remove_seasonal_cycle: bool = True
    eof_standardize: bool = False
    eof_rotation: str = "none"
    eof_solver: str = "full"
    eof_solver_seed: int = 0
    eof_solver_oversampling: int = 10
    eof_solver_iterations: int = 2
    covariance_model: str = "diagonal"
    ridge: float = 1.0
    clim_bias: bool = True
    spatial_support: str = "monthly_taper"
    support_min_fraction: float = 0.2
    support_full_fraction: float = 0.5
    support_smoothing_passes: int = 2
    delta_bounds: tuple[float, float] = (-1.6094379, 1.6094379)
    band_days: tuple[float, float] = (4.0, 180.0)
    min_resolution: float = 0.3
    max_bridge_days: int = 7
    min_segment_days: int = 360
    keep_significant: bool = True
    significance_level: float = 0.95
    omega0: float = 6.0
    dj: float = 0.25
    s0: float | None = None
    log_epsilon: float = 0.01
    local_overpass_hour: float = 13.5
    day_anchor_hour: float = 12.0
    r_bounds: tuple[float, float] = (0.2, 5.0)
    aod_floor: float = 0.001
    mmr_time_interp: str = "log_linear"
    mmr_outside_coverage: str = "identity"
    simplicity_rank: int = 0

    def __post_init__(self) -> None:
        _label(self.policy_id, "policy_id")
        _positive_integer(self.n_modes, "n_modes")
        _label(self.eof_rotation, "eof_rotation")
        _label(self.eof_solver, "eof_solver")
        _nonnegative_integer(self.eof_solver_seed, "eof_solver_seed")
        _nonnegative_integer(self.eof_solver_oversampling, "eof_solver_oversampling")
        _nonnegative_integer(self.eof_solver_iterations, "eof_solver_iterations")
        _label(self.covariance_model, "covariance_model")
        _label(self.spatial_support, "spatial_support")
        _label(self.mmr_time_interp, "mmr_time_interp")
        _label(self.mmr_outside_coverage, "mmr_outside_coverage")
        for name in (
            "eof_remove_seasonal_cycle",
            "eof_standardize",
            "clim_bias",
            "keep_significant",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        object.__setattr__(self, "ridge", _positive_finite(self.ridge, "ridge"))
        support_min = _unit_fraction(self.support_min_fraction, "support_min_fraction")
        support_full = _unit_fraction(self.support_full_fraction, "support_full_fraction")
        if support_min >= support_full:
            raise ValueError("support_min_fraction must be below support_full_fraction")
        object.__setattr__(self, "support_min_fraction", support_min)
        object.__setattr__(self, "support_full_fraction", support_full)
        _nonnegative_integer(self.support_smoothing_passes, "support_smoothing_passes")
        object.__setattr__(self, "delta_bounds", _ordered_pair(self.delta_bounds, "delta_bounds"))
        band = _ordered_positive_pair(self.band_days, "band_days")
        object.__setattr__(self, "band_days", band)
        object.__setattr__(
            self, "min_resolution", _unit_fraction(self.min_resolution, "min_resolution")
        )
        _nonnegative_integer(self.max_bridge_days, "max_bridge_days")
        _positive_integer(self.min_segment_days, "min_segment_days")
        if self.min_segment_days < 2.0 * band[1]:
            raise ValueError("min_segment_days must be at least twice the maximum period")
        significance = _finite_float(self.significance_level, "significance_level")
        if not 0.0 < significance < 1.0:
            raise ValueError("significance_level must be strictly between zero and one")
        object.__setattr__(self, "significance_level", significance)
        object.__setattr__(self, "omega0", _positive_finite(self.omega0, "omega0"))
        object.__setattr__(self, "dj", _positive_finite(self.dj, "dj"))
        if self.s0 is not None:
            object.__setattr__(self, "s0", _positive_finite(self.s0, "s0"))
        object.__setattr__(self, "log_epsilon", _positive_finite(self.log_epsilon, "log_epsilon"))
        for name in ("local_overpass_hour", "day_anchor_hour"):
            hour = _finite_float(getattr(self, name), name)
            if not 0.0 <= hour < 24.0:
                raise ValueError(f"{name} must be in [0, 24)")
            object.__setattr__(self, name, hour)
        ratio = _ordered_positive_pair(self.r_bounds, "r_bounds")
        if not ratio[0] <= 1.0 <= ratio[1]:
            raise ValueError("r_bounds must contain identity")
        object.__setattr__(self, "r_bounds", ratio)
        floor = _finite_float(self.aod_floor, "aod_floor")
        if floor < 0.0:
            raise ValueError("aod_floor must be nonnegative")
        object.__setattr__(self, "aod_floor", floor)
        _nonnegative_integer(self.simplicity_rank, "simplicity_rank")

    def normalized(self) -> dict[str, Any]:
        """Return one canonical JSON-safe control document."""
        return {
            "aod_floor": self.aod_floor,
            "band_days": list(self.band_days),
            "clim_bias": self.clim_bias,
            "covariance_model": self.covariance_model,
            "day_anchor_hour": self.day_anchor_hour,
            "delta_bounds": list(self.delta_bounds),
            "dj": self.dj,
            "eof_remove_seasonal_cycle": self.eof_remove_seasonal_cycle,
            "eof_rotation": self.eof_rotation,
            "eof_solver": self.eof_solver,
            "eof_solver_iterations": self.eof_solver_iterations,
            "eof_solver_oversampling": self.eof_solver_oversampling,
            "eof_solver_seed": self.eof_solver_seed,
            "eof_standardize": self.eof_standardize,
            "keep_significant": self.keep_significant,
            "local_overpass_hour": self.local_overpass_hour,
            "log_epsilon": self.log_epsilon,
            "max_bridge_days": self.max_bridge_days,
            "min_resolution": self.min_resolution,
            "min_segment_days": self.min_segment_days,
            "mmr_outside_coverage": self.mmr_outside_coverage,
            "mmr_time_interp": self.mmr_time_interp,
            "n_modes": self.n_modes,
            "omega0": self.omega0,
            "policy_id": self.policy_id,
            "r_bounds": list(self.r_bounds),
            "ridge": self.ridge,
            "s0": self.s0,
            "significance_level": self.significance_level,
            "simplicity_rank": self.simplicity_rank,
            "spatial_support": self.spatial_support,
            "support_full_fraction": self.support_full_fraction,
            "support_min_fraction": self.support_min_fraction,
            "support_smoothing_passes": self.support_smoothing_passes,
        }


def fitting_policy_values(fitting: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the canonical controls from a rendered fitting configuration."""
    analyses = fitting["analyses"]
    model = analyses["model_daily"]
    basis = analyses["aod_basis"]
    projection = analyses["obs_pcs"]
    filtered = analyses["filtered_pcs"]
    scaling = analyses["scaling"]
    writer = analyses["corrected"]
    projection_resolution = float(projection["min_resolution"])
    filter_resolution = float(filtered["min_resolution"])
    if projection_resolution != filter_resolution:
        raise ValueError("projection and filter min_resolution controls must match")
    preprocess = [
        value
        for key, value in analyses.items()
        if key.endswith("_daily") and value["type"] == "aod_preprocess"
    ]
    if any(float(value["log_epsilon"]) != float(model["log_epsilon"]) for value in preprocess):
        raise ValueError("all preprocessing log_epsilon controls must match")
    if any(
        float(value["day_anchor_hour"]) != float(model["day_anchor_hour"]) for value in preprocess
    ):
        raise ValueError("all preprocessing day_anchor_hour controls must match")
    covariance = "diagonal_plus_low_rank_common"
    if not all(entry.get("common_factor_variables") for entry in projection["obs"]):
        covariance = "diagonal"
    return {
        "aod_floor": float(scaling["aod_floor"]),
        "band_days": [float(filtered["band"]["min"]), float(filtered["band"]["max"])],
        "clim_bias": bool(projection["clim_bias"]),
        "covariance_model": covariance,
        "day_anchor_hour": float(model["day_anchor_hour"]),
        "delta_bounds": [float(value) for value in projection["delta_bounds"]],
        "dj": float(filtered.get("dj", 0.25)),
        "eof_remove_seasonal_cycle": bool(basis["remove_seasonal_cycle"]),
        "eof_rotation": str(basis["rotation"]),
        "eof_solver": str(basis["solver"]),
        "eof_solver_iterations": int(basis.get("solver_iterations", 2)),
        "eof_solver_oversampling": int(basis.get("solver_oversampling", 10)),
        "eof_solver_seed": int(basis.get("solver_seed", 0)),
        "eof_standardize": bool(basis["standardize"]),
        "keep_significant": bool(filtered["keep_significant"]),
        "local_overpass_hour": float(model["sample_local_time"]),
        "log_epsilon": float(model["log_epsilon"]),
        "max_bridge_days": int(filtered["max_bridge_days"]),
        "min_resolution": projection_resolution,
        "min_segment_days": int(filtered["min_segment_days"]),
        "mmr_outside_coverage": str(writer["outside_coverage"]),
        "mmr_time_interp": str(writer["time_interp"]),
        "n_modes": int(basis["n_modes"]),
        "omega0": float(filtered.get("omega0", 6.0)),
        "r_bounds": [float(value) for value in scaling["r_bounds"]],
        "ridge": float(projection["ridge"]),
        "s0": None if filtered.get("s0") is None else float(filtered["s0"]),
        "significance_level": float(filtered["significance_level"]),
        "spatial_support": str(projection["spatial_support"]),
        "support_full_fraction": float(projection["support_full_fraction"]),
        "support_min_fraction": float(projection["support_min_fraction"]),
        "support_smoothing_passes": int(projection["support_smoothing_passes"]),
    }


def apply_fitting_policy(fitting: dict[str, Any], policy: ScientificPolicy) -> None:
    """Apply every frozen scientific control without changing scenario geometry or dates."""
    analyses = fitting["analyses"]
    for name, preprocess in analyses.items():
        if name.endswith("_daily") and preprocess["type"] == "aod_preprocess":
            preprocess["log_epsilon"] = policy.log_epsilon
            preprocess["day_anchor_hour"] = policy.day_anchor_hour
    analyses["model_daily"]["sample_local_time"] = policy.local_overpass_hour
    basis = analyses["aod_basis"]
    basis.update(
        n_modes=policy.n_modes,
        remove_seasonal_cycle=policy.eof_remove_seasonal_cycle,
        standardize=policy.eof_standardize,
        rotation=policy.eof_rotation,
        solver=policy.eof_solver,
        solver_seed=policy.eof_solver_seed,
        solver_oversampling=policy.eof_solver_oversampling,
        solver_iterations=policy.eof_solver_iterations,
    )
    projection = analyses["obs_pcs"]
    if policy.covariance_model == "diagonal":
        for observation in projection["obs"]:
            observation["common_factor_variables"] = []
    elif policy.covariance_model == "diagonal_plus_low_rank_common":
        for observation in projection["obs"]:
            observation["common_factor_variables"] = ["common_error_factor"]
    else:
        raise ValueError(f"unsupported covariance_model {policy.covariance_model!r}")
    projection.update(
        ridge=policy.ridge,
        clim_bias=policy.clim_bias,
        spatial_support=policy.spatial_support,
        support_min_fraction=policy.support_min_fraction,
        support_full_fraction=policy.support_full_fraction,
        support_smoothing_passes=policy.support_smoothing_passes,
        delta_bounds=list(policy.delta_bounds),
        min_resolution=policy.min_resolution,
    )
    filtered = analyses["filtered_pcs"]
    filtered.update(
        min_resolution=policy.min_resolution,
        keep_significant=policy.keep_significant,
        significance_level=policy.significance_level,
        band={"min": policy.band_days[0], "max": policy.band_days[1], "units": "days"},
        max_bridge_days=policy.max_bridge_days,
        min_segment_days=policy.min_segment_days,
        omega0=policy.omega0,
        dj=policy.dj,
    )
    if policy.s0 is None:
        filtered.pop("s0", None)
    else:
        filtered["s0"] = policy.s0
    analyses["scaling"].update(r_bounds=list(policy.r_bounds), aod_floor=policy.aod_floor)
    analyses["corrected"].update(
        time_interp=policy.mmr_time_interp,
        outside_coverage=policy.mmr_outside_coverage,
    )


def policy_from_normalized(value: Mapping[str, Any]) -> ScientificPolicy:
    """Rebuild a policy only from the canonical document."""
    fields = dict(value)
    for name in ("band_days", "delta_bounds", "r_bounds"):
        fields[name] = tuple(fields[name])
    return ScientificPolicy(**fields)


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _positive_finite(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _unit_fraction(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return converted


def _ordered_pair(values: Any, name: str) -> tuple[float, float]:
    pair = tuple(values)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two bounds")
    lower, upper = (_finite_float(value, name) for value in pair)
    if lower >= upper:
        raise ValueError(f"{name} must be strictly increasing")
    return lower, upper


def _ordered_positive_pair(values: Any, name: str) -> tuple[float, float]:
    lower, upper = _ordered_pair(values, name)
    if lower <= 0.0:
        raise ValueError(f"{name} must be positive")
    return lower, upper


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


__all__ = [
    "ScientificPolicy",
    "apply_fitting_policy",
    "fitting_policy_values",
    "policy_from_normalized",
]
