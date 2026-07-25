"""Verify the declared production outputs before publishing the run manifest."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from davinci_monet.config.schema import MonetConfig, RequiredArtifactSpec
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)

_ITEM_ERROR_KEYS = (
    "pairing_errors",
    "stats_errors",
    "plot_errors",
    "analysis_errors",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, description: str, errors: list[str]) -> bool:
    if not path.is_file():
        errors.append(f"{description} does not exist: {path}")
        return False
    if path.stat().st_size <= 0:
        errors.append(f"{description} is empty: {path}")
        return False
    return True


def _verify_artifact_entry(entry: Mapping[str, Any], errors: list[str]) -> None:
    analysis = str(entry.get("analysis", ""))
    role = str(entry.get("role", ""))
    description = f"required artifact {analysis}:{role}"
    if entry.get("status") != "finalized":
        errors.append(f"{description} is not finalized")
        return

    checksums = entry.get("checksums")
    if not isinstance(checksums, Mapping):
        errors.append(f"{description} has no checksum receipt")
        return

    kind = entry.get("kind")
    if kind == "netcdf_collection":
        artifact_dir = Path(str(entry.get("artifact_dir", "")))
        files = checksums.get("files")
        if not isinstance(files, Mapping) or not files:
            errors.append(f"{description} has no collection file receipt")
        else:
            for filename, expected in files.items():
                path = artifact_dir / str(filename)
                if _require_file(path, description, errors) and _sha256(path) != expected:
                    errors.append(f"{description} checksum mismatch: {path}")
        summary_path = Path(str(entry.get("summary_path", "")))
        if _require_file(summary_path, f"{description} summary", errors):
            expected_summary = checksums.get("summary_sha256")
            if expected_summary is not None and _sha256(summary_path) != expected_summary:
                errors.append(f"{description} summary checksum mismatch: {summary_path}")
        return

    if kind == "product":
        artifact_path = Path(str(entry.get("artifact_path", "")))
        if _require_file(artifact_path, description, errors):
            expected = checksums.get("analysis_sha256")
            if expected is not None and _sha256(artifact_path) != expected:
                errors.append(f"{description} checksum mismatch: {artifact_path}")
        summary_path = Path(str(entry.get("summary_path", "")))
        if _require_file(summary_path, f"{description} summary", errors):
            expected_summary = checksums.get("summary_sha256")
            if expected_summary is not None and _sha256(summary_path) != expected_summary:
                errors.append(f"{description} summary checksum mismatch: {summary_path}")
        return

    errors.append(f"{description} uses unsupported artifact kind {kind!r}")


def _required_artifact_entry(
    requirement: RequiredArtifactSpec,
    entries: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for entry in entries:
        if entry.get("analysis") == requirement.analysis and entry.get("role") == requirement.role:
            return entry
    return None


def _plot_formats(plot_spec: Mapping[str, Any]) -> set[str]:
    raw = plot_spec.get("formats", plot_spec.get("output_formats"))
    if raw is None:
        return {"png", "pdf"}
    if isinstance(raw, str):
        return {raw.lower().lstrip(".")}
    return {str(value).lower().lstrip(".") for value in raw}


def _verify_plot(
    plot_name: str,
    raw_paths: Any,
    plot_spec: Mapping[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(raw_paths, list) or not raw_paths:
        errors.append(f"required plot '{plot_name}' produced no outputs")
        return
    requested_formats = _plot_formats(plot_spec)
    paths = [Path(str(raw_path)) for raw_path in raw_paths]
    for path in paths:
        _require_file(path, f"required plot '{plot_name}' output", errors)
    stems: dict[Path, set[str]] = {}
    for path in paths:
        stems.setdefault(path.with_suffix(""), set()).add(path.suffix.lower().lstrip("."))
    for stem, formats in stems.items():
        missing = sorted(requested_formats - formats)
        if missing:
            errors.append(
                f"required plot '{plot_name}' is missing formats "
                f"{', '.join(missing)} for {stem.name}"
            )


class CompletionStage(BaseStage):
    """Evaluate a production completion contract against runtime evidence."""

    def __init__(self) -> None:
        super().__init__("completion")

    def execute(self, context: PipelineContext) -> StageResult:
        start = time.time()
        if not isinstance(context.config, MonetConfig):
            return self._create_result(
                StageStatus.SKIPPED,
                data={"skipped": "no production completion contract"},
                duration=time.time() - start,
            )
        run = context.config.run
        completion = run.completion if run is not None and run.kind == "production" else None
        if completion is None:
            return self._create_result(
                StageStatus.SKIPPED,
                data={"skipped": "no production completion contract"},
                duration=time.time() - start,
            )

        errors: list[str] = []
        checks: list[dict[str, Any]] = []

        analysis_status = context.metadata.get("analysis_status")
        statuses = analysis_status if isinstance(analysis_status, Mapping) else {}
        for name in completion.required_analyses:
            status = statuses.get(name)
            passed = status == "completed"
            checks.append(
                {
                    "name": f"analysis:{name}",
                    "passed": passed,
                    "detail": str(status or "missing"),
                }
            )
            if not passed:
                errors.append(f"required analysis '{name}' status is {status or 'missing'}")

        raw_entries = context.metadata.get("analysis_artifacts")
        artifact_entries = (
            [entry for entry in raw_entries if isinstance(entry, Mapping)]
            if isinstance(raw_entries, list)
            else []
        )
        for requirement in completion.required_artifacts:
            entry = _required_artifact_entry(requirement, artifact_entries)
            if entry is None:
                errors.append(
                    "required artifact "
                    f"{requirement.analysis}:{requirement.role} was not published"
                )
                continue
            _verify_artifact_entry(entry, errors)

        save_result = context.results.get("save_results")
        save_data = save_result.data if save_result and isinstance(save_result.data, dict) else {}
        saved_products = save_data.get("saved_products")
        products = saved_products if isinstance(saved_products, Mapping) else {}
        for product_name in completion.required_saved_files:
            raw_path = products.get(product_name)
            if raw_path is None:
                errors.append(f"required saved file '{product_name}' was not published")
                continue
            _require_file(
                Path(str(raw_path)),
                f"required saved file '{product_name}'",
                errors,
            )

        plotting = context.results.get("plotting")
        plot_data = plotting.data if plotting and isinstance(plotting.data, dict) else {}
        raw_plot_products = plot_data.get("plot_products")
        plot_products = raw_plot_products if isinstance(raw_plot_products, Mapping) else {}
        config_plots = context.config_dict().get("plots", {})
        for plot_name in completion.required_plots:
            _verify_plot(
                plot_name,
                plot_products.get(plot_name),
                config_plots.get(plot_name, {}),
                errors,
            )

        inspection = context.results.get("inspection")
        inspection_data = (
            inspection.data if inspection and isinstance(inspection.data, dict) else {}
        )
        if inspection is None or inspection.status is not StageStatus.COMPLETED:
            errors.append("required inspection did not complete")
        elif inspection_data.get("passed") is not True:
            errors.append("required inspection did not pass")
        else:
            for key in ("inspection_json", "inspection_markdown"):
                raw_path = inspection_data.get(key)
                if raw_path is None:
                    errors.append(f"required inspection did not publish {key}")
                else:
                    _require_file(Path(str(raw_path)), f"required inspection {key}", errors)
            previews = inspection_data.get("inspection_previews")
            if not isinstance(previews, list) or not previews:
                errors.append("required inspection produced no previews")
            else:
                for raw_path in previews:
                    _require_file(
                        Path(str(raw_path)),
                        "required inspection preview",
                        errors,
                    )

        if not completion.allow_item_errors:
            for key in _ITEM_ERROR_KEYS:
                values = context.metadata.get(key)
                if values:
                    errors.append(f"{key} is not empty: {'; '.join(map(str, values))}")

        data = {
            "passed": not errors,
            "checks": checks,
            "errors": errors,
        }
        if errors:
            return self._create_result(
                StageStatus.FAILED,
                data=data,
                error="; ".join(errors),
                duration=time.time() - start,
            )
        return self._create_result(
            StageStatus.COMPLETED,
            data=data,
            duration=time.time() - start,
        )
