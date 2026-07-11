"""User-seeded, synthetic-only acceptance-run orchestration."""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic._aerosol_io import (
    generate_aerosol_tuning_bundle,
    write_aerosol_tuning_bundle,
)
from davinci_monet.tests.synthetic._aerosol_policy import (
    apply_fitting_policy,
    fitting_policy_values,
)
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    ACCEPTANCE_EXCLUDED_FRACTION_MAX,
    ACCEPTANCE_MAX_ELAPSED_SECONDS,
    ACCEPTANCE_MAX_PEAK_RSS_BYTES,
    ACCEPTANCE_SEED_COUNT,
    RECOVERY_METRICS,
    REQUIRED_RECOVERY_DIAGNOSTICS,
    REQUIRED_RECOVERY_STRATA,
)
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    aggregate_recovery as _aggregate_recovery,
)
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    diagnostic_requirements as _diagnostic_requirements,
)
from davinci_monet.tests.synthetic.fable_acceptance_gate import (
    evaluate_synthetic_recovery_gate,
)
from davinci_monet.tests.synthetic.fable_acceptance_gate import evidence_gate as _evidence_gate
from davinci_monet.tests.synthetic.fable_acceptance_gate import peak_rss_bytes as _peak_rss_bytes
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_gate as _resource_gate
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_limits as _resource_limits
from davinci_monet.tests.synthetic.fable_acceptance_record import (
    create_seed_lock as _create_seed_lock_file,
)
from davinci_monet.tests.synthetic.fable_acceptance_record import file_identity as _file_identity
from davinci_monet.tests.synthetic.fable_acceptance_record import (
    verify_seed_lock as _verify_seed_lock_file,
)
from davinci_monet.tests.synthetic.fable_acceptance_record import write_record as _write_record_file
from davinci_monet.tests.synthetic.fable_thresholds import RECOVERY_THRESHOLDS

ACCEPTANCE_SCHEMA = "fable-synthetic-acceptance-v2"
FROZEN_CALIBRATION_FILENAME = "fable-synthetic-calibration.json"
FROZEN_CALIBRATION_SHA256 = "de4b0074259ea4bca3819495ae8b2c6a0b1a994d2f0c6e5ebeef9d62b358fc09"
PipelineExecutor = Callable[[Path], Mapping[str, Any]]


@dataclass(frozen=True)
class SyntheticAcceptancePlan:
    """Validated immutable locations and user-supplied seeds for one gate run."""

    root: Path
    seeds: tuple[int, int, int]
    fitting_template: Path
    evaluation_template: Path
    calibration_record: Path
    calibration_record_sha256: str
    calibration_policy_id: str
    seed_lock_path: Path
    record_path: Path


def build_synthetic_acceptance_plan(
    root: str | Path, seeds: Sequence[int]
) -> SyntheticAcceptancePlan:
    """Validate three distinct supplied seeds without generating OSSE arrays."""
    seed_values = tuple(seeds)
    if len(seed_values) != ACCEPTANCE_SEED_COUNT:
        raise ValueError("acceptance requires exactly three supplied seeds")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("acceptance seeds must be distinct")
    for seed in seed_values:
        SyntheticTuningSpec.synthetic_osse(seed)
    repository = Path(__file__).resolve().parents[3]
    config_root = repository / "analyses" / "aerosol-tuning" / "configs"
    fitting = config_root / "fable-synthetic.example.yaml"
    evaluation = config_root / "fable-synthetic-eval.example.yaml"
    _validate_synthetic_templates(fitting, evaluation)
    calibration_path = config_root / FROZEN_CALIBRATION_FILENAME
    calibration = _load_calibration(calibration_path)
    calibration_sha256 = _calibration_sha256(calibration)
    if calibration_sha256 != FROZEN_CALIBRATION_SHA256:
        raise ValueError("frozen synthetic calibration identity does not match acceptance code")
    fitting_config = yaml.safe_load(fitting.read_text(encoding="utf-8"))
    _validate_fitting_policy(fitting_config, calibration.selected_policy)
    resolved_root = Path(root).expanduser().resolve()
    locked_seeds = (seed_values[0], seed_values[1], seed_values[2])
    return SyntheticAcceptancePlan(
        root=resolved_root,
        seeds=locked_seeds,
        fitting_template=fitting,
        evaluation_template=evaluation,
        calibration_record=calibration_path,
        calibration_record_sha256=calibration_sha256,
        calibration_policy_id=calibration.selected_policy_id,
        seed_lock_path=resolved_root / "seed-lock.json",
        record_path=resolved_root / "acceptance.json",
    )


def run_synthetic_acceptance(
    root: str | Path,
    seeds: Sequence[int],
    *,
    generate_only: bool = False,
    dry_run: bool = False,
    pipeline_executor: PipelineExecutor | None = None,
) -> Path:
    """Record supplied seeds, then generate and optionally run only synthetic configs."""
    if generate_only and dry_run:
        raise ValueError("generate_only and dry_run are mutually exclusive")
    plan = build_synthetic_acceptance_plan(root, seeds)
    if not dry_run and not generate_only and pipeline_executor is None:
        raise ValueError("a pipeline_executor is required for a full acceptance run")
    plan.root.mkdir(parents=True, exist_ok=True)
    if plan.record_path.exists():
        raise FileExistsError(
            f"acceptance seed record is already locked: {plan.record_path}; use a new root"
        )
    seed_lock = _create_seed_lock(plan)
    record: dict[str, Any] = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "scenario": "synthetic_osse",
        "seeds": list(plan.seeds),
        "seed_lock": seed_lock,
        "recovery_thresholds": dict(RECOVERY_THRESHOLDS),
        "diagnostic_requirements": _diagnostic_requirements(),
        "resource_limits": _resource_limits(),
        "aggregate_weighting": "equal_seed",
        "mode": "dry_run" if dry_run else ("generate_only" if generate_only else "full"),
        "status": "planned" if dry_run else "running",
        "templates": {
            "fitting": _file_identity(plan.fitting_template),
            "evaluation": _file_identity(plan.evaluation_template),
        },
        "calibration": {
            "record": _file_identity(plan.calibration_record),
            "record_sha256": plan.calibration_record_sha256,
            "selected_policy_id": plan.calibration_policy_id,
        },
        "runs": [],
    }
    _write_record(plan, record)
    if dry_run:
        return plan.record_path

    for seed in plan.seeds:
        entry: dict[str, Any] = {
            "seed": seed,
            "scenario": "synthetic_osse",
            "status": "running",
        }
        record["runs"].append(entry)
        _write_record(plan, record)
        started = time.perf_counter()
        seed_root = plan.root / f"seed-{seed}"
        try:
            fitting_config, evaluation_config = render_synthetic_acceptance_configs(plan, seed_root)
            spec = SyntheticTuningSpec.synthetic_osse(seed)
            bundle = generate_aerosol_tuning_bundle(spec)
            manifest = write_aerosol_tuning_bundle(seed_root, bundle)
            del bundle
            gc.collect()
            entry["scenario_manifest"] = _file_identity(manifest)
            entry["configs"] = {
                "fitting": _file_identity(fitting_config),
                "evaluation": _file_identity(evaluation_config),
            }
            if generate_only:
                entry["status"] = "generated"
            else:
                assert pipeline_executor is not None
                with _synthetic_root(seed_root):
                    fitting_result = dict(pipeline_executor(fitting_config))
                entry["fitting"] = fitting_result
                if bool(fitting_result.get("success")):
                    with _synthetic_root(seed_root):
                        evaluation_result = dict(pipeline_executor(evaluation_config))
                    entry["evaluation"] = evaluation_result
                    recovery_gate = evaluation_result.get("recovery_gate")
                    gate_passed = isinstance(recovery_gate, Mapping) and bool(
                        recovery_gate.get("passed")
                    )
                    evidence = _evidence_gate(fitting_result, evaluation_result)
                    entry["evidence_gate"] = evidence
                    entry["status"] = (
                        "completed"
                        if bool(evaluation_result.get("success"))
                        and gate_passed
                        and evidence["passed"]
                        else "failed"
                    )
                else:
                    entry["status"] = "failed"
                    entry["evaluation"] = {"status": "blocked_by_fitting_failure"}
        except Exception as exc:  # The record must survive a failed acceptance seed.
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["elapsed_seconds"] = time.perf_counter() - started
        entry["process_peak_rss_bytes"] = _peak_rss_bytes()
        entry["resource_gate"] = _resource_gate(
            entry["elapsed_seconds"], entry["process_peak_rss_bytes"]
        )
        if not entry["resource_gate"]["passed"]:
            entry["status"] = "failed"
        _write_record(plan, record)

    expected = "generated" if generate_only else "completed"
    record["aggregate"] = (
        _aggregate_recovery(record["runs"])
        if not generate_only
        else {"status": "not_evaluated", "reason": "generate_only"}
    )
    all_expected = all(run["status"] == expected for run in record["runs"])
    aggregate_passed = generate_only or bool(record["aggregate"].get("passed"))
    record["status"] = expected if all_expected and aggregate_passed else "failed"
    _write_record(plan, record)
    return plan.record_path


def render_synthetic_acceptance_configs(
    plan: SyntheticAcceptancePlan, seed_root: Path
) -> tuple[Path, Path]:
    """Render synthetic templates for the OSSE period and 10-degree grid."""
    fitting = yaml.safe_load(plan.fitting_template.read_text(encoding="utf-8"))
    evaluation = yaml.safe_load(plan.evaluation_template.read_text(encoding="utf-8"))
    calibration = _verify_calibration(plan)
    _validate_fitting_policy(fitting, calibration.selected_policy)
    _apply_fitting_policy(fitting, calibration.selected_policy)
    fitting["analysis"]["end_time"] = "2008-12-31 23:59:59"
    fitting["analyses"]["model_daily"]["target_grid"] = 10.0
    fitting["analyses"]["aod_basis"]["fit_window"] = {
        "start": "2001-01-01",
        "end": "2003-12-31 23:59:59",
    }
    fitting["analyses"]["obs_pcs"]["bias_fit_window"] = {
        "start": "2004-01-01",
        "end": "2005-12-31 23:59:59",
    }
    evaluation["analysis"]["start_time"] = "2007-01-01"
    evaluation["analysis"]["end_time"] = "2008-12-31 23:59:59"
    seed_root.mkdir(parents=True, exist_ok=True)
    fitting_path = seed_root / "fable-synthetic-osse.yaml"
    evaluation_path = seed_root / "fable-synthetic-osse-eval.yaml"
    fitting_path.write_text(yaml.safe_dump(fitting, sort_keys=False), encoding="utf-8")
    evaluation_path.write_text(yaml.safe_dump(evaluation, sort_keys=False), encoding="utf-8")
    return fitting_path, evaluation_path


def _validate_synthetic_templates(fitting: Path, evaluation: Path) -> None:
    fitting_text = fitting.read_text(encoding="utf-8")
    evaluation_text = evaluation.read_text(encoding="utf-8")
    if "${FABLE_SYNTH}/inputs/" not in fitting_text:
        raise ValueError("fitting template is not rooted in generated synthetic inputs")
    if "/oracle/" in fitting_text or "_true" in fitting_text:
        raise ValueError("fitting template leaks synthetic evaluation truth")
    required_evaluation_markers = (
        "${FABLE_SYNTH}/oracle/truth.nc",
        "delta_filter_target_true",
        "type: known_truth",
    )
    if not all(marker in evaluation_text for marker in required_evaluation_markers):
        raise ValueError("evaluation template is not the synthetic known-truth pipeline")


def _load_calibration(path: Path) -> Any:
    from davinci_monet.tests.synthetic.aerosol_calibration import load_calibration_record
    from davinci_monet.tests.synthetic.fable_calibration_identity import (
        validate_frozen_calibration_record,
    )

    if not path.is_file():
        raise ValueError(f"frozen synthetic calibration record is missing: {path}")
    return validate_frozen_calibration_record(load_calibration_record(path))


def _calibration_sha256(record: Any) -> str:
    from davinci_monet.tests.synthetic.aerosol_calibration import calibration_record_sha256

    return str(calibration_record_sha256(record))


def _verify_calibration(plan: SyntheticAcceptancePlan) -> Any:
    from davinci_monet.tests.synthetic.aerosol_calibration import verify_calibration_record
    from davinci_monet.tests.synthetic.fable_calibration_identity import (
        validate_frozen_calibration_record,
    )

    record = validate_frozen_calibration_record(
        verify_calibration_record(
            plan.calibration_record,
            plan.calibration_record_sha256,
        )
    )
    if record.selected_policy_id != plan.calibration_policy_id:
        raise ValueError("frozen synthetic calibration policy changed after acceptance planning")
    return record


def _validate_fitting_policy(fitting: Mapping[str, Any], policy: Any) -> None:
    expected = dict(policy.normalized())
    expected.pop("policy_id")
    expected.pop("simplicity_rank")
    try:
        actual = fitting_policy_values(fitting)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "fitting template scientific controls do not match the frozen calibration policy"
        ) from exc
    if actual != expected:
        raise ValueError(
            "fitting template scientific controls do not match the frozen calibration policy"
        )


def _apply_fitting_policy(fitting: dict[str, Any], policy: Any) -> None:
    apply_fitting_policy(fitting, policy)
    _validate_fitting_policy(fitting, policy)


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


def _seed_lock_document(plan: SyntheticAcceptancePlan) -> dict[str, Any]:
    calibration = _verify_calibration(plan)
    return {
        "schema_version": "fable-synthetic-acceptance-seed-lock-v1",
        "scenario": "synthetic_osse",
        "seeds": list(plan.seeds),
        "templates": {
            "fitting": _file_identity(plan.fitting_template),
            "evaluation": _file_identity(plan.evaluation_template),
        },
        "calibration_record_sha256": plan.calibration_record_sha256,
        "scientific_policy": calibration.selected_policy.normalized(),
    }


def _create_seed_lock(plan: SyntheticAcceptancePlan) -> dict[str, Any]:
    return _create_seed_lock_file(plan.root, plan.seed_lock_path, _seed_lock_document(plan))


def _verify_seed_lock(plan: SyntheticAcceptancePlan, record: Mapping[str, Any]) -> None:
    _verify_seed_lock_file(
        plan.seed_lock_path,
        _seed_lock_document(plan),
        plan.seeds,
        record,
    )


def _write_record(plan: SyntheticAcceptancePlan, record: dict[str, Any]) -> None:
    _verify_seed_lock(plan, record)
    _write_record_file(plan.root, plan.record_path, record)


__all__ = [
    "ACCEPTANCE_EXCLUDED_FRACTION_MAX",
    "ACCEPTANCE_MAX_ELAPSED_SECONDS",
    "ACCEPTANCE_MAX_PEAK_RSS_BYTES",
    "ACCEPTANCE_SCHEMA",
    "ACCEPTANCE_SEED_COUNT",
    "FROZEN_CALIBRATION_FILENAME",
    "FROZEN_CALIBRATION_SHA256",
    "PipelineExecutor",
    "RECOVERY_THRESHOLDS",
    "SyntheticAcceptancePlan",
    "build_synthetic_acceptance_plan",
    "evaluate_synthetic_recovery_gate",
    "render_synthetic_acceptance_configs",
    "run_synthetic_acceptance",
]
