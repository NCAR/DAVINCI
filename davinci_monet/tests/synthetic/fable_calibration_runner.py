"""Reproducible pipeline runner for the frozen FABLE calibration policy."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import xarray as xr
import yaml

from davinci_monet.analysis.known_truth_metrics import match_weighted_modes
from davinci_monet.pipeline.runner import PipelineResult, PipelineRunner
from davinci_monet.tests.synthetic._aerosol_contracts import (
    SyntheticTuningSpec,
    canonical_json,
)
from davinci_monet.tests.synthetic._aerosol_io import (
    generate_aerosol_tuning_bundle,
    scientific_dataset_hash,
    write_aerosol_tuning_bundle,
)
from davinci_monet.tests.synthetic._aerosol_policy import (
    ScientificPolicy,
    apply_fitting_policy,
    fitting_policy_values,
)
from davinci_monet.tests.synthetic.aerosol_calibration import (
    EVIDENCE_HASH_KEYS,
    CalibrationCandidate,
    CalibrationEvidence,
    CandidateMetrics,
    select_calibration_policy,
    write_calibration_record,
)
from davinci_monet.tests.synthetic.fable_calibration_identity import (
    CALIBRATION_SEED,
    CALIBRATION_SPLIT,
    NULL_SEED,
    calibration_code_sha256,
    calibration_policy_candidates,
    validate_frozen_calibration_record,
)


@dataclass(frozen=True)
class _ScenarioRun:
    root: Path
    spec_sha256: str
    config_sha256: str
    manifest_sha256: str
    result: PipelineResult
    truth: xr.Dataset


def run_frozen_calibration(
    work_root: str | Path,
    destination: str | Path,
    *,
    policies: Sequence[ScientificPolicy] | None = None,
    calibration_seed: int = CALIBRATION_SEED,
    null_seed: int = NULL_SEED,
) -> Path:
    """Run the predeclared candidate set and atomically freeze its selected policy."""
    candidate_policies = tuple(policies or calibration_policy_candidates())
    if len(candidate_policies) < 2:
        raise ValueError("calibration requires at least two predeclared candidate policies")
    if len({policy.policy_id for policy in candidate_policies}) != len(candidate_policies):
        raise ValueError("calibration candidate policy IDs must be unique")
    root = Path(work_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    candidates = [
        _evaluate_candidate(
            root / policy.policy_id,
            policy,
            calibration_seed=calibration_seed,
            null_seed=null_seed,
        )
        for policy in candidate_policies
    ]
    return write_calibration_record(destination, select_calibration_policy(candidates))


def _evaluate_candidate(
    root: Path,
    policy: ScientificPolicy,
    *,
    calibration_seed: int,
    null_seed: int,
) -> CalibrationCandidate:
    calibration: _ScenarioRun | None = None
    null: _ScenarioRun | None = None
    try:
        calibration = _run_scenario(
            root / "calibration",
            SyntheticTuningSpec.writer_ci(calibration_seed),
            policy,
            include_evaluation=True,
        )
        null = _run_scenario(
            root / "null",
            SyntheticTuningSpec.calibration_null(null_seed),
            policy,
            include_evaluation=False,
        )
        calibration_report = _recovery_report(calibration.result)
        recovery_metrics = _primary_recovery_metrics(calibration_report)
        null_metrics = evaluate_null_control(
            _source(null.result, "scaling"),
            _source(null.result, "filtered_pcs"),
            null.truth,
            policy,
            split=CALIBRATION_SPLIT,
        )
        null_report = {
            "metrics": null_metrics,
            "policy": policy.normalized(),
            "scenario": "calibration_null",
            "seed": null_seed,
            "spec_sha256": null.spec_sha256,
            "split": CALIBRATION_SPLIT,
        }
        null_report_sha256 = hashlib.sha256(canonical_json(null_report).encode("ascii")).hexdigest()
        _write_atomic_json(root / "null-report.json", null_report)
        hashes = {
            "calibration_config_sha256": calibration.config_sha256,
            "calibration_manifest_sha256": calibration.manifest_sha256,
            "calibration_report_sha256": scientific_dataset_hash(calibration_report),
            "calibration_spec_sha256": calibration.spec_sha256,
            "code_sha256": calibration_code_sha256(),
            "null_config_sha256": null.config_sha256,
            "null_manifest_sha256": null.manifest_sha256,
            "null_report_sha256": null_report_sha256,
            "null_spec_sha256": null.spec_sha256,
        }
        if tuple(sorted(hashes)) != EVIDENCE_HASH_KEYS:
            raise RuntimeError("calibration runner evidence schema drifted")
        evidence = CalibrationEvidence(
            calibration_seed=calibration_seed,
            null_seed=null_seed,
            calibration_scenario="writer_ci",
            null_scenario="calibration_null",
            calibration_split=CALIBRATION_SPLIT,
            hashes=tuple(hashes.items()),
        )
        candidate = CalibrationCandidate(
            policy=policy,
            metrics=CandidateMetrics(
                **recovery_metrics,
                null_retained_energy_fraction=null_metrics["null_retained_energy_fraction"],
                null_significant_fraction=null_metrics["null_significant_fraction"],
            ),
            evidence=evidence,
        )
        return candidate
    finally:
        for scenario in (calibration, null):
            if scenario is not None:
                _close_result(scenario.result)
                scenario.truth.close()


def evaluate_null_control(
    scaling: xr.Dataset,
    filtered: xr.Dataset,
    truth: xr.Dataset,
    policy: ScientificPolicy,
    *,
    split: str,
) -> dict[str, float]:
    """Score the exact frozen wavelet policy on a zero-correction scenario."""
    truth = truth.sel(time=np.asarray(scaling["time"].values))
    split_days = np.asarray(truth["split"].values).astype(str) == split
    observable = _estimate_observable_modes(scaling, truth)
    valid_segment = np.asarray(filtered["valid_segment"].values, dtype=bool)
    coi = np.asarray(filtered["coi"].values, dtype=np.float64)
    periods = np.asarray(filtered["period"].values, dtype=np.float64)
    in_band = (periods >= policy.band_days[0]) & (periods <= policy.band_days[1])
    if not np.any(in_band):
        raise ValueError("null report has no wavelet periods inside the frozen band")
    score_days = split_days & np.all(
        valid_segment[:, observable] & (coi[:, observable] >= policy.band_days[1]), axis=1
    )
    if "coefficient_available" in scaling:
        score_days &= np.asarray(scaling["coefficient_available"].values, dtype=bool)
    if not np.any(score_days):
        raise ValueError("null report has no non-COI calibration days")
    support = np.asarray(scaling["spatial_support"].values, dtype=np.float64)
    applied = np.asarray(scaling["delta_log_applied"].values, dtype=np.float64)
    noise = np.asarray(truth["innovation_noise_true"].values, dtype=np.float64)
    scoring = score_days[:, None, None] & (support > 0.0) & np.isfinite(noise)
    latitude_weights = np.cos(np.deg2rad(np.asarray(scaling["lat"].values)))[:, None]
    raw_weights = np.where(scoring, latitude_weights[None], 0.0)
    daily_weight = raw_weights.sum(axis=(1, 2))
    valid_days = daily_weight > 0.0
    normalized = np.divide(
        raw_weights,
        daily_weight[:, None, None],
        out=np.zeros_like(raw_weights),
        where=valid_days[:, None, None],
    )
    if np.any(valid_days):
        normalized /= int(np.count_nonzero(valid_days))
    numerator = float(np.sum(normalized * np.square(np.where(scoring, applied, 0.0))))
    denominator = float(np.sum(normalized * np.square(np.where(scoring, noise, 0.0))))
    if denominator <= 0.0:
        raise ValueError("null report innovation-noise energy is zero")
    significance = np.asarray(filtered["power_significance"].values, dtype=np.float64)
    candidates = (
        split_days[:, None, None]
        & observable[None, :, None]
        & valid_segment[:, :, None]
        & in_band[None, None, :]
        & (periods[None, None, :] <= coi[:, :, None])
        & np.isfinite(significance)
    )
    count = int(np.count_nonzero(candidates))
    if count == 0:
        raise ValueError("null report has no significant-coefficient candidates")
    return {
        "null_retained_energy_fraction": numerator / denominator,
        "null_significant_fraction": float(
            np.count_nonzero(significance[candidates] >= 1.0) / count
        ),
        "scored_day_count": float(np.count_nonzero(score_days)),
        "significance_candidate_count": float(count),
    }


def _estimate_observable_modes(scaling: xr.Dataset, truth: xr.Dataset) -> np.ndarray:
    matches = match_weighted_modes(scaling["eofs"], truth["pattern_true"])
    truth_observable = np.asarray(truth["mode_observable_true"].values, dtype=bool)
    estimate_observable = np.zeros(scaling.sizes["mode"], dtype=bool)
    for match in matches:
        estimate_observable[match.estimate_index] = truth_observable[match.truth_index]
    if not np.any(estimate_observable):
        raise ValueError("null report has no observable matched estimate modes")
    return estimate_observable


def _run_scenario(
    root: Path,
    spec: SyntheticTuningSpec,
    policy: ScientificPolicy,
    *,
    include_evaluation: bool,
) -> _ScenarioRun:
    root.mkdir(parents=True, exist_ok=False)
    truth: xr.Dataset | None = None
    fitting: PipelineResult | None = None
    result: PipelineResult | None = None
    try:
        bundle = generate_aerosol_tuning_bundle(spec)
        truth = bundle.truth
        spec_sha256 = str(bundle.provenance["spec_hash"])
        write_aerosol_tuning_bundle(root, bundle)
        fitting_path, evaluation_path = _render_configs(root, policy, include_evaluation)
        paths = [fitting_path]
        with _synthetic_root(root):
            fitting = PipelineRunner(
                show_progress=False, close_datasets_after_run=False
            ).run_from_config(str(fitting_path))
        if not fitting.success:
            raise RuntimeError(f"calibration fitting pipeline failed: {fitting.stage_errors}")
        result = fitting
        manifests = [root / "output" / "manifest.json"]
        if include_evaluation:
            assert evaluation_path is not None
            paths.append(evaluation_path)
            with _synthetic_root(root):
                result = PipelineRunner(
                    show_progress=False, close_datasets_after_run=False
                ).run_from_config(str(evaluation_path))
            if not result.success:
                raise RuntimeError(f"calibration evaluation pipeline failed: {result.stage_errors}")
            manifests.append(root / "evaluation" / "manifest.json")
            _close_result(fitting)
            fitting = None
        return _ScenarioRun(
            root=root,
            spec_sha256=spec_sha256,
            config_sha256=_combined_file_sha256(paths),
            manifest_sha256=_combined_manifest_sha256(manifests, root),
            result=result,
            truth=truth,
        )
    except BaseException:
        if result is not None and result is not fitting:
            _close_result(result)
        if fitting is not None:
            _close_result(fitting)
        if truth is not None:
            truth.close()
        raise


def _render_configs(
    root: Path, policy: ScientificPolicy, include_evaluation: bool
) -> tuple[Path, Path | None]:
    repository = Path(__file__).resolve().parents[3]
    config_root = repository / "analyses" / "aerosol-tuning" / "configs"
    fitting = yaml.safe_load(
        (config_root / "fable-synthetic.example.yaml").read_text(encoding="utf-8")
    )
    apply_fitting_policy(fitting, policy)
    expected = dict(policy.normalized())
    expected.pop("policy_id")
    expected.pop("simplicity_rank")
    if fitting_policy_values(fitting) != expected:
        raise RuntimeError("rendered calibration config does not match the frozen policy")
    fitting["analysis"]["end_time"] = "2005-12-31 23:59:59"
    fitting_path = root / "fable-calibration-fit.yaml"
    fitting_path.write_text(yaml.safe_dump(fitting, sort_keys=False), encoding="utf-8")
    if not include_evaluation:
        return fitting_path, None
    evaluation = yaml.safe_load(
        (config_root / "fable-synthetic-eval.example.yaml").read_text(encoding="utf-8")
    )
    evaluation["analysis"]["start_time"] = "2005-01-01"
    evaluation["analysis"]["end_time"] = "2005-12-31 23:59:59"
    evaluation["analyses"]["recovery"]["evaluation_splits"] = [CALIBRATION_SPLIT]
    evaluation_path = root / "fable-calibration-eval.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    return fitting_path, evaluation_path


def _primary_recovery_metrics(report: xr.Dataset) -> dict[str, float]:
    primary = report.sel(stratum="primary")
    names = (
        "field_correlation",
        "field_origin_slope",
        "field_nrmse",
        "aod_rmse_ratio",
        "full_target_aod_rmse_ratio",
    )
    values = {name: float(primary[name].item()) for name in names}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("calibration recovery metrics must all be finite")
    return values


def _source(result: PipelineResult, name: str) -> xr.Dataset:
    if result.context is None or name not in result.context.sources:
        raise RuntimeError(f"calibration pipeline did not produce {name!r}")
    return result.context.sources[name].data


def _recovery_report(result: PipelineResult) -> xr.Dataset:
    return _source(result, "recovery").load()


def _combined_file_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _combined_manifest_sha256(paths: list[Path], root: Path) -> str:
    normalized: dict[str, Any] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        normalized[path.parent.name + "/" + path.name] = _normalize_root(document, root)
    return hashlib.sha256(canonical_json(normalized).encode("ascii")).hexdigest()


def _normalize_root(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_root(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_root(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "${FABLE_SYNTH}")
    return value


def _write_atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _close_result(result: PipelineResult) -> None:
    if result.context is None:
        return
    from davinci_monet.pipeline.lifecycle import PipelineResourcePolicy

    PipelineResourcePolicy().cleanup_context_datasets(result.context)


@contextmanager
def _synthetic_root(root: Path) -> Iterator[None]:
    previous = os.environ.get("FABLE_SYNTH")
    os.environ["FABLE_SYNTH"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FABLE_SYNTH", None)
        else:
            os.environ["FABLE_SYNTH"] = previous


__all__ = [
    "CALIBRATION_SEED",
    "CALIBRATION_SPLIT",
    "NULL_SEED",
    "calibration_code_sha256",
    "calibration_policy_candidates",
    "evaluate_null_control",
    "run_frozen_calibration",
    "validate_frozen_calibration_record",
]
