"""Frozen scientific, evidence, aggregate, and resource acceptance gates."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.tests.synthetic.fable_artifact_validation import (
    validate_recovery_artifact,
)
from davinci_monet.tests.synthetic.fable_thresholds import RECOVERY_THRESHOLDS

ACCEPTANCE_SEED_COUNT = 3
# Frozen before user seeds: the development report excluded 0.765 of candidate cells.
ACCEPTANCE_EXCLUDED_FRACTION_MAX = 0.80
ACCEPTANCE_MAX_ELAPSED_SECONDS = 30 * 60
ACCEPTANCE_MAX_PEAK_RSS_BYTES = 8 * 1024**3
RECOVERY_METRICS = (
    "field_correlation",
    "field_origin_slope",
    "field_nrmse",
    "aod_rmse_ratio",
    "full_target_aod_rmse_ratio",
)
REQUIRED_RECOVERY_DIAGNOSTICS = (
    "valid_count",
    "candidate_count",
    "excluded_fraction",
    "best_representable_nrmse",
    "off_basis_floor_nrmse",
    "full_delta_nrmse",
    "in_span_truth_rms",
    "perpendicular_truth_rms",
    "perpendicular_to_full_rms_ratio",
    "clip_fraction",
    "holdout_aod_rmse_ratio",
    "subspace_projector_error",
    "basis_mode_similarity",
    "matched_mode_observable",
    "coefficient_correlation",
    "coefficient_origin_slope",
    "coefficient_bias",
    "coefficient_nrmse",
)
REQUIRED_RECOVERY_STRATA = (
    "primary",
    "full_domain",
    "support_zero",
    "support_partial",
    "support_full",
    "resolution_low",
    "resolution_medium",
    "resolution_high",
    "season_DJF",
    "season_MAM",
    "season_JJA",
    "season_SON",
    "latitude_south_high",
    "latitude_south_mid",
    "latitude_tropical",
    "latitude_north_mid",
    "latitude_north_high",
)


def evaluate_synthetic_recovery_gate(report: xr.Dataset) -> dict[str, Any]:
    """Evaluate frozen per-seed recovery and diagnostic requirements."""
    requirements = diagnostic_requirements()
    strata = (
        set(np.asarray(report["stratum"].values).astype(str).tolist())
        if "stratum" in report.coords
        else set()
    )
    missing_strata = [name for name in REQUIRED_RECOVERY_STRATA if name not in strata]
    missing = [
        name for name in (*RECOVERY_METRICS, *REQUIRED_RECOVERY_DIAGNOSTICS) if name not in report
    ]
    failures: list[str] = []
    if missing_strata:
        failures.append(f"recovery report is missing strata: {', '.join(missing_strata)}")
    if missing:
        failures.append(f"recovery report is missing diagnostics: {', '.join(missing)}")
    if "primary" not in strata or missing:
        return {
            "passed": False,
            "thresholds": dict(RECOVERY_THRESHOLDS),
            "diagnostic_requirements": requirements,
            "metrics": {},
            "diagnostics": {},
            "failures": failures,
        }
    primary = report.sel(stratum="primary")
    metrics = {name: _scalar(primary[name]) for name in RECOVERY_METRICS}
    diagnostics = {
        name: _scalar(primary[name])
        for name in (
            "valid_count",
            "candidate_count",
            "excluded_fraction",
            "best_representable_nrmse",
            "off_basis_floor_nrmse",
            "full_delta_nrmse",
            "in_span_truth_rms",
            "perpendicular_truth_rms",
            "perpendicular_to_full_rms_ratio",
            "clip_fraction",
            "holdout_aod_rmse_ratio",
            "subspace_projector_error",
        )
    }
    failures.extend(_metric_failures(metrics))
    finite = (
        "off_basis_floor_nrmse",
        "clip_fraction",
        "holdout_aod_rmse_ratio",
        "best_representable_nrmse",
        "subspace_projector_error",
        "full_delta_nrmse",
        "in_span_truth_rms",
        "perpendicular_truth_rms",
        "perpendicular_to_full_rms_ratio",
    )
    if not all(np.isfinite(diagnostics[name]) for name in finite):
        failures.append("primary recovery diagnostics must all be finite")
    if diagnostics["valid_count"] <= 0 or diagnostics["candidate_count"] <= 0:
        failures.append("primary recovery counts must be positive")
    excluded = diagnostics["excluded_fraction"]
    if not np.isfinite(excluded) or excluded > ACCEPTANCE_EXCLUDED_FRACTION_MAX:
        failures.append("excluded_fraction exceeds the frozen 0.80 ceiling")
    return {
        "passed": not failures,
        "thresholds": dict(RECOVERY_THRESHOLDS),
        "diagnostic_requirements": requirements,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "failures": failures,
    }


def _scalar(value: xr.DataArray) -> float:
    if value.chunks is not None:
        value = value.compute()
    return float(value.item())


def _metric_failures(metrics: Mapping[str, float]) -> list[str]:
    failures: list[str] = []
    if not all(np.isfinite(value) for value in metrics.values()):
        failures.append("primary recovery metrics must all be finite")
    if metrics["field_correlation"] < RECOVERY_THRESHOLDS["field_correlation_min"]:
        failures.append("field_correlation is below 0.90")
    if not (
        RECOVERY_THRESHOLDS["field_origin_slope_min"]
        <= metrics["field_origin_slope"]
        <= RECOVERY_THRESHOLDS["field_origin_slope_max"]
    ):
        failures.append("field_origin_slope is outside [0.80, 1.20]")
    if metrics["field_nrmse"] > RECOVERY_THRESHOLDS["field_nrmse_max"]:
        failures.append("field_nrmse exceeds 0.35")
    if metrics["aod_rmse_ratio"] > RECOVERY_THRESHOLDS["aod_rmse_ratio_max"]:
        failures.append("aod_rmse_ratio exceeds 0.70")
    if (
        metrics["full_target_aod_rmse_ratio"]
        >= RECOVERY_THRESHOLDS["full_target_aod_rmse_ratio_max"]
    ):
        failures.append("full_target_aod_rmse_ratio does not improve on the model")
    return failures


def diagnostic_requirements() -> dict[str, Any]:
    """Return the predeclared per-seed diagnostic contract."""
    return {
        "required_strata": list(REQUIRED_RECOVERY_STRATA),
        "required_variables": list(REQUIRED_RECOVERY_DIAGNOSTICS),
        "finite_primary": [
            "off_basis_floor_nrmse",
            "full_delta_nrmse",
            "in_span_truth_rms",
            "perpendicular_truth_rms",
            "perpendicular_to_full_rms_ratio",
            "clip_fraction",
            "holdout_aod_rmse_ratio",
        ],
        "excluded_fraction_max": ACCEPTANCE_EXCLUDED_FRACTION_MAX,
    }


def resource_limits() -> dict[str, Any]:
    """Return the predeclared per-seed OSSE resource contract."""
    return {
        "scope": "per_seed_generation_fit_and_evaluation",
        "elapsed_seconds_max": ACCEPTANCE_MAX_ELAPSED_SECONDS,
        "process_peak_rss_bytes_max": ACCEPTANCE_MAX_PEAK_RSS_BYTES,
    }


def resource_gate(elapsed: float, peak_rss: int) -> dict[str, Any]:
    """Apply the frozen wall-time and peak-memory limits."""
    failures: list[str] = []
    if elapsed >= ACCEPTANCE_MAX_ELAPSED_SECONDS:
        failures.append("elapsed time does not satisfy the 30-minute per-seed limit")
    if peak_rss >= ACCEPTANCE_MAX_PEAK_RSS_BYTES:
        failures.append("peak RSS does not satisfy the 8-GiB per-seed limit")
    return {
        "passed": not failures,
        "limits": resource_limits(),
        "elapsed_seconds": elapsed,
        "process_peak_rss_bytes": peak_rss,
        "failures": failures,
    }


def _valid_file_identity(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    path_value = value.get("path")
    checksum = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(checksum, str):
        return False
    path = Path(path_value)
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def _valid_completed_manifest_identity(value: Any) -> bool:
    if not _valid_file_identity(value):
        return False
    assert isinstance(value, Mapping)
    try:
        document = json.loads(Path(str(value["path"])).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False
    return isinstance(document, Mapping) and document.get("status") == "completed"


def _valid_artifact_identity(value: Any, manifest: Any) -> bool:
    if not isinstance(value, Mapping) or not _valid_file_identity(manifest):
        return False
    identity = value.get("identity")
    checksums = value.get("checksums")
    structurally_valid = (
        value.get("status") == "finalized"
        and isinstance(identity, Mapping)
        and all(
            isinstance(identity.get(name), Mapping) and bool(identity[name])
            for name in ("source_hashes", "config_hashes", "code_hashes")
        )
        and isinstance(checksums, Mapping)
        and bool(checksums)
    )
    if not structurally_valid:
        return False
    try:
        validate_recovery_artifact(value, str(manifest["path"]))
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def evidence_gate(fitting: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Require byte-valid pipeline manifests, recovery artifact, and full report."""
    failures: list[str] = []
    if not _valid_completed_manifest_identity(fitting.get("manifest")):
        failures.append("fitting result is missing its completed manifest identity")
    if not _valid_completed_manifest_identity(evaluation.get("manifest")):
        failures.append("evaluation result is missing its completed manifest identity")
    if not _valid_artifact_identity(
        evaluation.get("recovery_artifact"), evaluation.get("manifest")
    ):
        failures.append("evaluation result is missing the finalized recovery-artifact identity")
    report = evaluation.get("recovery_report")
    if not (
        isinstance(report, Mapping)
        and isinstance(report.get("coords"), Mapping)
        and isinstance(report.get("data_vars"), Mapping)
        and isinstance(report.get("attrs"), Mapping)
    ):
        failures.append("evaluation result is missing the full recovery report")
    return {"passed": not failures, "failures": failures}


def _summary(values: Sequence[float], seeds: Sequence[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1))
    half_width = 4.302652729911275 * standard_deviation / math.sqrt(array.size)
    return {
        "per_seed": [
            {"seed": int(seed), "value": float(value)}
            for seed, value in zip(seeds, array, strict=True)
        ],
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "confidence_interval_95": [mean - half_width, mean + half_width],
    }


def aggregate_recovery(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the equal-seed decision and confidence-interval report."""
    failures: list[str] = []
    seeds = [int(run["seed"]) for run in runs]
    gates: list[Mapping[str, Any]] = []
    aggregate_diagnostics = (
        "excluded_fraction",
        "best_representable_nrmse",
        "off_basis_floor_nrmse",
        "full_delta_nrmse",
        "in_span_truth_rms",
        "perpendicular_truth_rms",
        "perpendicular_to_full_rms_ratio",
        "clip_fraction",
        "holdout_aod_rmse_ratio",
        "subspace_projector_error",
    )
    for run in runs:
        evaluation = run.get("evaluation")
        gate = evaluation.get("recovery_gate") if isinstance(evaluation, Mapping) else None
        complete = (
            isinstance(gate, Mapping)
            and gate.get("passed")
            and isinstance(gate.get("metrics"), Mapping)
            and all(name in gate["metrics"] for name in RECOVERY_METRICS)
            and isinstance(gate.get("diagnostics"), Mapping)
            and all(name in gate["diagnostics"] for name in aggregate_diagnostics)
        )
        if not complete:
            failures.append(f"seed {run['seed']} did not pass its recovery gate")
        else:
            assert isinstance(gate, Mapping)
            gates.append(gate)
        if run.get("status") != "completed":
            failures.append(f"seed {run['seed']} did not complete all acceptance gates")
    if len(gates) != ACCEPTANCE_SEED_COUNT:
        return {
            "status": "failed",
            "passed": False,
            "weighting": "equal_seed",
            "seed_weights": {str(seed): 1.0 / ACCEPTANCE_SEED_COUNT for seed in seeds},
            "failures": failures,
        }
    metric_report = {
        name: _summary([float(gate["metrics"][name]) for gate in gates], seeds)
        for name in RECOVERY_METRICS
    }
    diagnostic_report = {
        name: _summary([float(gate["diagnostics"][name]) for gate in gates], seeds)
        for name in aggregate_diagnostics
    }
    means = {name: value["mean"] for name, value in metric_report.items()}
    failures.extend(f"equal-seed aggregate: {value}" for value in _metric_failures(means))
    if diagnostic_report["excluded_fraction"]["mean"] > ACCEPTANCE_EXCLUDED_FRACTION_MAX:
        failures.append("equal-seed aggregate excluded_fraction exceeds 0.80")
    return {
        "status": "completed" if not failures else "failed",
        "passed": not failures,
        "weighting": "equal_seed",
        "seed_weights": {str(seed): 1.0 / ACCEPTANCE_SEED_COUNT for seed in seeds},
        "confidence_interval": "two_sided_student_t_95_percent_df_2",
        "metrics": metric_report,
        "diagnostics": diagnostic_report,
        "failures": failures,
    }


def peak_rss_bytes() -> int:
    """Return the process peak resident set size in bytes."""
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


__all__ = [
    "ACCEPTANCE_EXCLUDED_FRACTION_MAX",
    "ACCEPTANCE_MAX_ELAPSED_SECONDS",
    "ACCEPTANCE_MAX_PEAK_RSS_BYTES",
    "ACCEPTANCE_SEED_COUNT",
    "RECOVERY_METRICS",
    "REQUIRED_RECOVERY_DIAGNOSTICS",
    "REQUIRED_RECOVERY_STRATA",
    "aggregate_recovery",
    "diagnostic_requirements",
    "evaluate_synthetic_recovery_gate",
    "evidence_gate",
    "peak_rss_bytes",
    "resource_gate",
    "resource_limits",
]
