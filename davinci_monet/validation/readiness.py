"""Read-only readiness checks for scheduled DAVINCI run controls."""

from __future__ import annotations

import glob
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from davinci_monet.config.schema import MonetConfig
from davinci_monet.inspection.presets import BUILTIN_INSPECTION_PRESETS

_ARTIFACT_ROLES_BY_ANALYSIS_TYPE: dict[str, frozenset[str]] = {
    "eof": frozenset({"basis_fit"}),
    "eof_projection": frozenset({"projection_fit"}),
    "wavelet_filter": frozenset({"wavelet_filter"}),
    "aod_scaling": frozenset({"scaling"}),
    "mmr_writer": frozenset({"corrected_mmr"}),
    "gridded_analysis": frozenset({"product"}),
    "known_truth": frozenset({"recovery_report"}),
    "fable_v2_diagnostics": frozenset({"v2_diagnostic_report"}),
}


@dataclass(frozen=True)
class ReadinessCheck:
    """One deterministic readiness decision."""

    name: str
    status: Literal["passed", "failed", "skipped"]
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    """Structured readiness result emitted by the CLI and configuration skill."""

    ready: bool
    run_id: str | None
    run_kind: str | None
    mode: Literal["fresh", "resume"]
    config_path: str
    checks: tuple[ReadinessCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "mode": self.mode,
            "config_path": self.config_path,
            "checks": [asdict(check) for check in self.checks],
        }


def _check(name: str, passed: bool, success: str, failure: str) -> ReadinessCheck:
    return ReadinessCheck(
        name=name,
        status="passed" if passed else "failed",
        detail=success if passed else failure,
    )


def _directory_is_empty(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and next(path.iterdir(), None) is None)


def _filename_check(config: MonetConfig, config_path: Path) -> ReadinessCheck:
    run = config.run
    if run is None:
        return ReadinessCheck("config_filename", "failed", "run section is required")
    if run.kind == "production":
        expected = f"{run.id}.yaml"
        return _check(
            "config_filename",
            config_path.name == expected,
            f"production config filename is {expected}",
            f"production config filename must be {expected}, got {config_path.name}",
        )
    if run.kind == "example":
        valid = config_path.name.endswith(".example.yaml")
        return _check(
            "config_filename",
            valid,
            "example config uses .example.yaml",
            "example config filename must end in .example.yaml",
        )
    valid = not config_path.name.endswith(".example.yaml")
    return _check(
        "config_filename",
        valid,
        f"{run.kind} config is not labeled as an example",
        f"{run.kind} config filename must not end in .example.yaml",
    )


def _identity_check(config: MonetConfig) -> ReadinessCheck:
    run = config.run
    if run is None:
        return ReadinessCheck("run_identity", "failed", "run section is required")
    reserved = run.kind == "production" and "fable" in run.id.split("-")
    return _check(
        "run_identity",
        not reserved,
        f"{run.kind} run identity is scientific and versioned",
        "production run identity uses reserved unrelated model name 'fable'",
    )


def _artifact_contract_check(config: MonetConfig) -> ReadinessCheck:
    run = config.run
    completion = run.completion if run is not None else None
    if completion is None:
        return ReadinessCheck(
            "artifact_contracts",
            "skipped",
            "run has no production completion contract",
        )
    errors: list[str] = []
    artifacts_by_analysis: dict[str, set[str]] = {}
    for requirement in completion.required_artifacts:
        analysis = config.analyses.get(requirement.analysis)
        if analysis is None:
            continue
        declared_roles = _ARTIFACT_ROLES_BY_ANALYSIS_TYPE.get(analysis.type, frozenset())
        if requirement.role not in declared_roles:
            errors.append(
                f"{requirement.analysis}:{requirement.role} is not published by {analysis.type}"
            )
        artifacts_by_analysis.setdefault(requirement.analysis, set()).add(requirement.role)

    required_names = set(completion.required_analyses)
    dependency_names = {
        dependency
        for name, analysis in config.analyses.items()
        if name in required_names
        for dependency in analysis.input_refs().values()
        if dependency in required_names
    }
    terminal_names = required_names - dependency_names
    for name in sorted(terminal_names):
        if name not in artifacts_by_analysis:
            errors.append(f"terminal analysis {name!r} has no required durable artifact")

    return _check(
        "artifact_contracts",
        not errors,
        "required and terminal analysis artifacts have durable publishers",
        "; ".join(errors),
    )


def _inspection_check(config: MonetConfig) -> ReadinessCheck:
    run = config.run
    completion = run.completion if run is not None else None
    if completion is None:
        return ReadinessCheck(
            "inspection_contract",
            "skipped",
            "run has no production completion contract",
        )
    unknown = sorted(set(completion.inspection.presets) - BUILTIN_INSPECTION_PRESETS)
    return _check(
        "inspection_contract",
        not unknown,
        "inspection presets are registered",
        "unknown inspection presets: " + ", ".join(unknown),
    )


def _mmr_input_inventory_check(config: MonetConfig) -> ReadinessCheck:
    """Require every configured MMR-writer glob to resolve before production."""
    writers = [
        (name, analysis)
        for name, analysis in config.analyses.items()
        if analysis.type == "mmr_writer"
    ]
    if not writers:
        return ReadinessCheck(
            "mmr_input_inventory",
            "skipped",
            "run has no native MMR writer",
        )

    problems: list[str] = []
    counts: list[str] = []
    for name, writer in writers:
        pattern = os.path.expandvars(os.path.expanduser(str(writer.files)))
        paths = sorted(Path(value) for value in glob.glob(pattern, recursive=False))
        files = [path for path in paths if path.is_file()]
        if not files:
            problems.append(f"analyses.{name}.files matched no files: {pattern}")
            continue
        nonfiles = len(paths) - len(files)
        if nonfiles:
            problems.append(f"analyses.{name}.files matched {nonfiles} non-file paths")
        counts.append(f"{name}={len(files)}")

    return _check(
        "mmr_input_inventory",
        not problems,
        "native MMR inputs resolve: " + ", ".join(counts),
        "; ".join(problems),
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _attempt_paths_check(
    config: MonetConfig,
    mode: Literal["fresh", "resume"],
) -> ReadinessCheck:
    execution = config.execution
    if execution is None:
        return ReadinessCheck(
            "attempt_paths",
            "skipped",
            "run does not enable durable attempts",
        )
    attempt_root = execution.attempt_root.expanduser().resolve()
    output_dir = Path(str(config.analysis.output_dir))
    log_dir = Path(str(config.analysis.log_dir))
    problems: list[str] = []
    if output_dir.name != "output" or log_dir.name != "logs":
        problems.append("analysis output_dir/log_dir must end in output and logs")
    if output_dir.parent != log_dir.parent:
        problems.append("analysis output_dir and log_dir must share one attempt root")
    writable_parent = _nearest_existing_parent(attempt_root)
    if not writable_parent.is_dir() or not os.access(writable_parent, os.W_OK | os.X_OK):
        problems.append(f"attempt root is not writable: {attempt_root}")
    if mode == "fresh":
        if not _directory_is_empty(output_dir):
            problems.append(f"output directory is not empty: {output_dir}")
        if not _directory_is_empty(log_dir):
            problems.append(f"log directory is not empty: {log_dir}")
        if not _directory_is_empty(attempt_root):
            problems.append(f"fresh attempt root is not empty: {attempt_root}")
    else:
        try:
            from davinci_monet.pipeline.checkpoints.manager import CheckpointManager

            manager = CheckpointManager.create(
                config,
                resume=True,
                read_only=True,
            )
            if manager is not None and manager.blocked_reasons:
                raise ValueError(", ".join(manager.blocked_reasons))
        except Exception as exc:
            problems.append(f"attempt is not resumable: {exc}")
    return _check(
        "attempt_paths",
        not problems,
        (
            f"attempt root is unused: {attempt_root}"
            if mode == "fresh"
            else f"attempt is initialized and incomplete: {attempt_root}"
        ),
        "; ".join(problems),
    )


def _checkpoint_policy_check(config: MonetConfig) -> ReadinessCheck:
    execution = config.execution
    if execution is None:
        return ReadinessCheck(
            "checkpoint_policy",
            "skipped",
            "run does not enable checkpoints",
        )
    checkpoints = execution.checkpoints
    supported = (
        checkpoints.mode in {"required", "best_effort"}
        and checkpoints.granularity in {"item", "stage"}
        and checkpoints.retain in {"all", "failed", "none"}
    )
    return _check(
        "checkpoint_policy",
        supported,
        (
            f"{checkpoints.mode} checkpointing uses {checkpoints.granularity} "
            "granularity and supported dataset/JSON/file codecs"
        ),
        "checkpoint policy disables or requests unsupported durable codecs",
    )


def _checkpoint_restore_check(config: MonetConfig) -> ReadinessCheck:
    execution = config.execution
    restore = None if execution is None else execution.checkpoints.restore_from
    if restore is None:
        return ReadinessCheck(
            "checkpoint_restore",
            "skipped",
            "run does not restore a prior-attempt stage boundary",
        )
    assert execution is not None
    problems: list[str] = []
    try:
        from davinci_monet.core.identity import canonical_sha256
        from davinci_monet.pipeline.checkpoints.codecs import CheckpointCodecs
        from davinci_monet.pipeline.checkpoints.manager import CheckpointManager
        from davinci_monet.pipeline.checkpoints.models import AttemptStatus
        from davinci_monet.pipeline.checkpoints.store import AttemptStore

        source_root = restore.source_attempt_root.expanduser().resolve()
        if source_root == execution.attempt_root.expanduser().resolve():
            raise ValueError("restore source and target attempts are identical")
        store = AttemptStore(source_root)
        attempt = store.read_attempt()
        if attempt.status is AttemptStatus.IN_PROGRESS:
            raise ValueError("restore source attempt is not terminal")
        receipt = store.read_receipt(restore.through_stage, None)
        if receipt is None:
            raise ValueError(f"missing stage receipt: {restore.through_stage}")
        if canonical_sha256(receipt) != restore.receipt_sha256:
            raise ValueError("stage receipt SHA-256 mismatch")
        CheckpointManager._validate_restore_receipt_chain(
            store,
            CheckpointCodecs(source_root),
            receipt,
        )
    except Exception as exc:
        problems.append(str(exc))
    return _check(
        "checkpoint_restore",
        not problems,
        (
            f"pinned {restore.through_stage} boundary is reusable from "
            f"{restore.source_attempt_root}"
        ),
        "; ".join(problems),
    )


def _noninteractive_execution_check() -> ReadinessCheck:
    from davinci_monet.pipeline.stages import create_standard_pipeline

    stage_names = [stage.name for stage in create_standard_pipeline()]
    interactive_tokens = ("approval", "confirm", "interactive", "prompt")
    blocked = [
        name for name in stage_names if any(token in name.lower() for token in interactive_tokens)
    ]
    return _check(
        "noninteractive_execution",
        not blocked,
        "DAVINCI pipeline execution contains no interactive approval stage",
        "interactive pipeline stages found: " + ", ".join(blocked),
    )


def evaluate_run_readiness(
    config: MonetConfig,
    config_path: str | Path,
    *,
    mode: Literal["fresh", "resume"] = "fresh",
) -> ReadinessReport:
    """Evaluate a parsed config without creating files or changing scheduler state."""
    path = Path(config_path).resolve()
    checks = (
        _filename_check(config, path),
        _identity_check(config),
        _artifact_contract_check(config),
        _mmr_input_inventory_check(config),
        _inspection_check(config),
        _attempt_paths_check(config, mode),
        _checkpoint_policy_check(config),
        _checkpoint_restore_check(config),
        ReadinessCheck(
            "source_time_coverage",
            "skipped",
            "configured readers do not expose a common pre-open coverage interface",
        ),
        _noninteractive_execution_check(),
    )
    ready = all(check.status != "failed" for check in checks)
    run = config.run
    return ReadinessReport(
        ready=ready,
        run_id=run.id if run is not None else None,
        run_kind=run.kind if run is not None else None,
        mode=mode,
        config_path=str(path),
        checks=checks,
    )
