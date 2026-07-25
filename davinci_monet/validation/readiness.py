"""Read-only readiness checks for scheduled DAVINCI run controls."""

from __future__ import annotations

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
    config_path: str
    checks: tuple[ReadinessCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
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


def _attempt_paths_check(config: MonetConfig) -> ReadinessCheck:
    run = config.run
    if run is None or run.kind != "production":
        return ReadinessCheck(
            "attempt_paths",
            "skipped",
            "immutable attempt paths are required only for production",
        )
    output_dir = Path(str(config.analysis.output_dir))
    log_dir = Path(str(config.analysis.log_dir))
    problems: list[str] = []
    if output_dir.name != "output" or log_dir.name != "logs":
        problems.append("analysis output_dir/log_dir must end in output and logs")
    if output_dir.parent != log_dir.parent:
        problems.append("analysis output_dir and log_dir must share one attempt root")
    if not _directory_is_empty(output_dir):
        problems.append(f"output directory is not empty: {output_dir}")
    if not _directory_is_empty(log_dir):
        problems.append(f"log directory is not empty: {log_dir}")
    return _check(
        "attempt_paths",
        not problems,
        f"attempt root is unused: {output_dir.parent}",
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


def evaluate_run_readiness(config: MonetConfig, config_path: str | Path) -> ReadinessReport:
    """Evaluate a parsed config without creating files or changing scheduler state."""
    path = Path(config_path).resolve()
    checks = (
        _filename_check(config, path),
        _identity_check(config),
        _artifact_contract_check(config),
        _inspection_check(config),
        _attempt_paths_check(config),
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
        config_path=str(path),
        checks=checks,
    )
