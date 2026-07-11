"""Versioned config rendering and one-scenario pipeline execution for FABLE v2."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import xarray as xr
import yaml

from davinci_monet.analysis.artifacts import load_dataset_collection
from davinci_monet.pipeline.lifecycle import PipelineResourcePolicy
from davinci_monet.pipeline.runner import PipelineResult, PipelineRunner
from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec, spec_hash
from davinci_monet.tests.synthetic._aerosol_io import scientific_dataset_hash
from davinci_monet.tests.synthetic._aerosol_policy import ScientificPolicy
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    evidence_gate,
    peak_rss_bytes,
    resource_gate,
)
from davinci_monet.tests.synthetic.fable_v2_bundle import link_v2_scenario_bundle
from davinci_monet.tests.synthetic.fable_v2_evidence import completed_manifest_identity
from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_lifecycle import V2GenerationLock
from davinci_monet.tests.synthetic.fable_v2_policy import (
    FableV2Policy,
    apply_v2_fitting_policy,
    v2_fitting_policy_values,
)
from davinci_monet.tests.synthetic.fable_v2_scoring import (
    v2_null_score,
    v2_recovery_score,
    v2_scientific_policy,
)

ScoreKind = Literal["recovery", "null"]
FITTING_TEMPLATE = "fable-synthetic-v2.example.yaml"
EVALUATION_TEMPLATE = "fable-synthetic-v2-eval.example.yaml"


@dataclass(frozen=True)
class V2ScenarioOutcome:
    """Serializable evidence from one generated scenario and one policy."""

    seed: int
    scenario: str
    policy: FableV2Policy
    score_kind: ScoreKind
    evaluation_splits: tuple[str, ...]
    spec_sha256: str
    scenario_manifest: FrozenFileIdentity
    fitting_config: FrozenFileIdentity
    fitting_manifest: FrozenFileIdentity
    evaluation_config: FrozenFileIdentity | None
    evaluation_manifest: FrozenFileIdentity | None
    report_sha256: str
    diagnostic_report_sha256: str | None
    learned_basis_oracle_nrmse: float | None
    score: Mapping[str, Any]
    evidence_passed: bool
    elapsed_seconds: float
    process_peak_rss_bytes: int
    resources: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return (
            bool(self.score.get("passed", False))
            and self.evidence_passed
            and bool(self.resources.get("passed", False))
        )

    def normalized(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "evaluation_splits": list(self.evaluation_splits),
            "diagnostic_report_sha256": self.diagnostic_report_sha256,
            "evaluation_config": _optional_identity(self.evaluation_config),
            "evaluation_manifest": _optional_identity(self.evaluation_manifest),
            "evidence_passed": self.evidence_passed,
            "fitting_config": self.fitting_config.normalized(),
            "fitting_manifest": self.fitting_manifest.normalized(),
            "learned_basis_oracle_nrmse": self.learned_basis_oracle_nrmse,
            "passed": self.passed,
            "policy": self.policy.normalized(),
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "report_sha256": self.report_sha256,
            "resources": _json_value(self.resources),
            "scenario": self.scenario,
            "scenario_manifest": self.scenario_manifest.normalized(),
            "score": _json_value(self.score),
            "score_kind": self.score_kind,
            "seed": self.seed,
            "spec_sha256": self.spec_sha256,
        }


def render_v2_scenario_configs(
    root: str | Path,
    spec: SyntheticTuningSpec,
    policy: FableV2Policy,
    *,
    evaluation_splits: Sequence[str],
    include_evaluation: bool = True,
) -> tuple[Path, Path | None]:
    """Render only the v2 templates for the exact scenario grid and windows."""
    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fitting_template, evaluation_template = _template_paths()
    _validate_templates(fitting_template, evaluation_template)
    fitting = yaml.safe_load(fitting_template.read_text(encoding="utf-8"))
    apply_v2_fitting_policy(fitting, policy)
    _require_rendered_policy(fitting, policy)
    first = str(spec.time_config.start).split()[0]
    last = str(spec.time_config.end).split()[0]
    fitting["analysis"]["start_time"] = first
    fitting["analysis"]["end_time"] = f"{last} 23:59:59"
    fitting["analyses"]["model_daily"]["target_grid"] = (
        spec.mode_domain.lon_range / spec.mode_domain.n_lon
    )
    fitting["analyses"]["aod_basis"]["fit_window"] = _window(spec, "basis_train")
    fitting["analyses"]["obs_pcs"]["bias_fit_window"] = _window(spec, "bias_fit")
    fitting_path = destination / "fable-v2-fit.yaml"
    fitting_path.write_text(yaml.safe_dump(fitting, sort_keys=False), encoding="utf-8")
    if not include_evaluation:
        return fitting_path, None
    split_values = tuple(evaluation_splits)
    if not split_values:
        raise ValueError("v2 evaluation requires at least one declared split")
    split_windows = [_window(spec, name) for name in split_values]
    evaluation = yaml.safe_load(evaluation_template.read_text(encoding="utf-8"))
    evaluation["analysis"]["start_time"] = min(item["start"] for item in split_windows)
    evaluation["analysis"]["end_time"] = max(item["end"] for item in split_windows)
    evaluation["analyses"]["recovery"]["evaluation_splits"] = list(split_values)
    evaluation["analyses"]["v2_diagnostics"]["evaluation_splits"] = list(split_values)
    evaluation["analyses"]["v2_diagnostics"][
        "reported_common_factor_amplitude"
    ] = spec.common_error_sigma
    evaluation_path = destination / "fable-v2-eval.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    return fitting_path, evaluation_path


def run_v2_scenario(
    root: str | Path,
    spec: SyntheticTuningSpec,
    policy: FableV2Policy,
    *,
    score_kind: ScoreKind,
    lock: V2GenerationLock,
    evaluation_splits: Sequence[str] = ("development_test",),
    show_progress: bool = False,
) -> V2ScenarioOutcome:
    """Execute one locked, pre-generated scenario through PipelineRunner."""
    if score_kind not in {"recovery", "null"}:
        raise ValueError("unsupported v2 score kind")
    split_values = tuple(evaluation_splits)
    expected_kind = "null" if lock.role == "calibration_null" else "recovery"
    if score_kind != expected_kind:
        raise ValueError(f"v2 {lock.role} lock requires score_kind={expected_kind!r}")
    expected_splits = (
        ()
        if score_kind == "null"
        else (("calibration",) if lock.role == "calibration_recovery" else ("development_test",))
    )
    if split_values != expected_splits:
        raise ValueError(f"v2 {lock.role} lock requires evaluation_splits={expected_splits!r}")
    destination = Path(root).expanduser().resolve()
    scenario_manifest_path = link_v2_scenario_bundle(lock, destination, spec, policy.policy_id)
    spec_sha256 = spec_hash(spec)
    started = time.perf_counter()
    fitting_result: PipelineResult | None = None
    evaluation_result: PipelineResult | None = None
    truth: xr.Dataset | None = None
    try:
        if score_kind == "null":
            truth = xr.open_dataset(destination / "oracle" / "truth.nc")
        fitting_path, evaluation_path = render_v2_scenario_configs(
            destination,
            spec,
            policy,
            evaluation_splits=evaluation_splits,
            include_evaluation=score_kind == "recovery",
        )
        with _synthetic_root(destination):
            fitting_result = PipelineRunner(
                show_progress=show_progress, close_datasets_after_run=False
            ).run_from_config(str(fitting_path))
        _require_pipeline_success(fitting_result, "fitting")
        fitting_manifest = FrozenFileIdentity.capture(destination / "output" / "manifest.json")
        fitting_evidence_passed = completed_manifest_identity(fitting_manifest)
        diagnostic_report_sha256: str | None
        learned_basis_oracle_nrmse: float | None
        if score_kind == "null":
            scaling = _source(fitting_result, "scaling")
            filtered = _source(fitting_result, "filtered_pcs")
            assert truth is not None
            score = v2_null_score(scaling, filtered, truth, policy)
            report_sha256 = _json_sha256(score)
            evaluation_config = None
            evaluation_manifest = None
            evidence_passed = fitting_evidence_passed
        else:
            assert evaluation_path is not None
            with _synthetic_root(destination):
                evaluation_result = PipelineRunner(
                    show_progress=show_progress, close_datasets_after_run=False
                ).run_from_config(str(evaluation_path))
            _require_pipeline_success(evaluation_result, "evaluation")
            report = _loaded_finalized_artifact(evaluation_result, "recovery", "recovery_report")
            score = _v2_recovery_gate(report)
            report_sha256 = scientific_dataset_hash(report)
            diagnostic = _loaded_finalized_artifact(
                evaluation_result, "v2_diagnostics", "v2_diagnostic_report"
            )
            _validate_diagnostic_report(diagnostic)
            diagnostic_report_sha256 = scientific_dataset_hash(diagnostic)
            learned_basis_oracle_nrmse = float(diagnostic["learned_basis_oracle_nrmse"].item())
            evaluation_config = FrozenFileIdentity.capture(evaluation_path)
            evaluation_manifest = FrozenFileIdentity.capture(
                destination / "evaluation" / "manifest.json"
            )
            evidence_passed = fitting_evidence_passed and bool(
                evidence_gate(
                    {"manifest": fitting_manifest.normalized()},
                    {
                        "manifest": evaluation_manifest.normalized(),
                        "recovery_artifact": _recovery_artifact(evaluation_result),
                        "recovery_report": report.to_dict(data="list"),
                    },
                )["passed"]
            )
        if score_kind == "null":
            diagnostic_report_sha256 = None
            learned_basis_oracle_nrmse = None
        elapsed = time.perf_counter() - started
        rss = peak_rss_bytes()
        resources = resource_gate(elapsed, rss)
        return V2ScenarioOutcome(
            seed=spec.master_seed,
            scenario=spec.scenario,
            policy=policy,
            score_kind=score_kind,
            evaluation_splits=split_values,
            spec_sha256=spec_sha256,
            scenario_manifest=FrozenFileIdentity.capture(scenario_manifest_path),
            fitting_config=FrozenFileIdentity.capture(fitting_path),
            fitting_manifest=fitting_manifest,
            evaluation_config=evaluation_config,
            evaluation_manifest=evaluation_manifest,
            report_sha256=report_sha256,
            diagnostic_report_sha256=diagnostic_report_sha256,
            learned_basis_oracle_nrmse=learned_basis_oracle_nrmse,
            score=score,
            evidence_passed=evidence_passed,
            elapsed_seconds=elapsed,
            process_peak_rss_bytes=rss,
            resources=resources,
        )
    finally:
        if evaluation_result is not None:
            _close_result(evaluation_result)
        if fitting_result is not None:
            _close_result(fitting_result)
        if truth is not None:
            truth.close()


def _template_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[3] / "analyses" / "aerosol-tuning" / "configs"
    return root / FITTING_TEMPLATE, root / EVALUATION_TEMPLATE


def _validate_templates(fitting: Path, evaluation: Path) -> None:
    fitting_text = fitting.read_text(encoding="utf-8")
    evaluation_text = evaluation.read_text(encoding="utf-8")
    if "${FABLE_SYNTH}/inputs/" not in fitting_text:
        raise ValueError("v2 fitting template is not rooted in synthetic inputs")
    if "/oracle/" in fitting_text or "_true" in fitting_text:
        raise ValueError("v2 fitting template leaks synthetic oracle data")
    required = ("${FABLE_SYNTH}/oracle/truth.nc", "type: known_truth")
    if not all(marker in evaluation_text for marker in required):
        raise ValueError("v2 evaluation template is not a known-truth pipeline")


def _require_rendered_policy(fitting: Mapping[str, Any], policy: FableV2Policy) -> None:
    expected = policy.normalized()
    expected.pop("policy_id")
    expected.pop("simplicity_rank")
    if v2_fitting_policy_values(fitting) != expected:
        raise RuntimeError("rendered v2 config does not match its complete policy")


def _window(spec: SyntheticTuningSpec, name: str) -> dict[str, str]:
    matches = [item for item in spec.split_windows if item[0] == name]
    if len(matches) != 1:
        raise ValueError(f"scenario does not declare exactly one {name!r} split")
    _, start, end = matches[0]
    return {"start": start, "end": f"{end} 23:59:59"}


def _base_policy(policy: FableV2Policy) -> ScientificPolicy:
    return v2_scientific_policy(policy)


def _source(result: PipelineResult, name: str) -> xr.Dataset:
    if result.context is None or name not in result.context.sources:
        raise RuntimeError(f"v2 pipeline did not produce {name!r}")
    return result.context.sources[name].data


def _require_pipeline_success(result: PipelineResult, label: str) -> None:
    if not result.success:
        raise RuntimeError(f"v2 {label} pipeline failed: {result.stage_errors}")


def _artifact_entry(result: PipelineResult, analysis: str, role: str) -> Mapping[str, Any]:
    if result.context is None:
        raise RuntimeError("v2 pipeline has no artifact context")
    matches = [
        item
        for item in result.context.metadata.get("analysis_artifacts", [])
        if isinstance(item, Mapping)
        and item.get("analysis") == analysis
        and item.get("role") == role
    ]
    if len(matches) != 1:
        raise RuntimeError(f"v2 pipeline did not finalize exactly one {analysis}/{role} artifact")
    return matches[0]


def _loaded_finalized_artifact(result: PipelineResult, analysis: str, role: str) -> xr.Dataset:
    entry = _artifact_entry(result, analysis, role)
    checksums = entry.get("checksums")
    files = checksums.get("files") if isinstance(checksums, Mapping) else None
    if entry.get("kind") != "netcdf_collection" or not isinstance(files, Mapping) or not files:
        raise RuntimeError(f"v2 {analysis}/{role} artifact receipt is invalid")
    root = Path(str(entry.get("artifact_dir", "")))
    dataset = load_dataset_collection(root / str(name) for name in sorted(files))
    try:
        return dataset.load()
    finally:
        dataset.close()


def _recovery_artifact(result: PipelineResult) -> dict[str, Any]:
    return _json_value(_artifact_entry(result, "recovery", "recovery_report"))


def _validate_diagnostic_report(report: xr.Dataset) -> None:
    if "learned_basis_oracle_nrmse" not in report:
        raise ValueError("v2 diagnostic report is missing learned_basis_oracle_nrmse")
    if (
        str(report.attrs.get("diagnostic_only", "")).lower() != "true"
        or str(report.attrs.get("eligible_for_calibration", "")).lower() != "false"
    ):
        raise ValueError("v2 diagnostic report lacks non-ranking oracle provenance")


def _v2_recovery_gate(report: xr.Dataset) -> dict[str, Any]:
    return v2_recovery_score(report)


def _close_result(result: PipelineResult) -> None:
    if result.context is not None:
        PipelineResourcePolicy(close_datasets_after_run=False).cleanup_context_datasets(
            result.context
        )


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


def _json_sha256(value: Any) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _optional_identity(value: FrozenFileIdentity | None) -> dict[str, str] | None:
    return None if value is None else value.normalized()


__all__ = [
    "EVALUATION_TEMPLATE",
    "FITTING_TEMPLATE",
    "ScoreKind",
    "V2ScenarioOutcome",
    "render_v2_scenario_configs",
    "run_v2_scenario",
]
