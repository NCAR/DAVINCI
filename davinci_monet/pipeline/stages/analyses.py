"""Run derived analyses and register their outputs as pipeline sources."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import xarray as xr

from davinci_monet.analysis.artifacts import ArtifactService, build_analysis_artifact_identity
from davinci_monet.analysis.base import (
    AnalysisExecutionError,
    AnalysisResult,
    AnalysisRuntime,
    ArtifactDeclaration,
)
from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.checkpoints.manager import (
    CheckpointRequest,
    item_checkpoint_manager,
)
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourceData,
    StageResult,
    StageStatus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AnalysisInputResolver:
    """One named-input view used for both dependency ordering and execution."""

    specs: Mapping[str, Any]

    def input_refs(self, analysis_name: str) -> dict[str, str]:
        spec = self.specs[analysis_name]
        refs = spec.input_refs()
        if not isinstance(refs, Mapping):
            raise TypeError(f"analysis '{analysis_name}' input_refs() must return a mapping")
        normalized: dict[str, str] = {}
        for role, ref in refs.items():
            if not isinstance(role, str) or not role:
                raise ValueError(f"analysis '{analysis_name}' has an invalid input role {role!r}")
            if not isinstance(ref, str) or not ref:
                raise ValueError(
                    f"analysis '{analysis_name}' input '{role}' has an invalid reference {ref!r}"
                )
            normalized[role] = ref
        return normalized

    def dependencies(self, analysis_name: str) -> tuple[str, ...]:
        keys = self.specs.keys()
        return tuple(
            dict.fromkeys(ref for ref in self.input_refs(analysis_name).values() if ref in keys)
        )

    def topological_order(self) -> list[str]:
        state: dict[str, int] = {}
        order: list[str] = []

        def visit(node: str) -> None:
            if state.get(node, 0) == 2:
                return
            if state.get(node, 0) == 1:
                raise ValueError(f"analyses dependency cycle detected at '{node}'")
            state[node] = 1
            for dependency in self.dependencies(node):
                visit(dependency)
            state[node] = 2
            order.append(node)

        for key in self.specs:
            visit(key)
        return order

    def resolve(self, analysis_name: str, sources: Mapping[str, Any]) -> dict[str, xr.Dataset]:
        inputs: dict[str, xr.Dataset] = {}
        for role, ref in self.input_refs(analysis_name).items():
            source_obj = sources.get(ref)
            if source_obj is None:
                raise ValueError(
                    f"analysis '{analysis_name}' input '{role}' references unknown source '{ref}'"
                )
            dataset = source_obj.data if hasattr(source_obj, "data") else source_obj
            if not isinstance(dataset, xr.Dataset):
                raise TypeError(
                    f"analysis '{analysis_name}' input '{role}' source '{ref}' is not an "
                    "xarray.Dataset"
                )
            inputs[role] = dataset
        return inputs


def _topological_order(specs: dict[str, Any]) -> list[str]:
    """Compatibility wrapper around the named-input resolver."""
    return _AnalysisInputResolver(specs).topological_order()


class AnalysesStage(BaseStage):
    """Run derived analyses (EOF, wavelet, ...) and register pseudo-sources."""

    def __init__(self) -> None:
        super().__init__("analyses")

    def validate(self, context: PipelineContext) -> bool:
        return bool(context.analyses_config())

    def execute(self, context: PipelineContext) -> StageResult:
        import time

        import davinci_monet.analysis  # noqa: F401  (registers concrete analyses)

        start = time.time()
        specs = context.analyses_config()
        resolver = _AnalysisInputResolver(specs)
        summary: dict[str, Any] = {}
        states: dict[str, str] = {}
        stage_errors: list[str] = []
        fatal_errors: list[str] = []
        success_count = 0

        try:
            order = resolver.topological_order()
        except (TypeError, ValueError) as exc:
            message = str(exc)
            context.metadata.setdefault("analysis_errors", []).append(message)
            return self._create_result(
                StageStatus.FAILED,
                data=summary,
                error=message,
                duration=time.time() - start,
            )

        analysis_config = context.analysis_config()
        artifact_service = ArtifactService(Path(analysis_config.output_dir or "."))
        runtime = AnalysisRuntime(
            start_time=cast(datetime | None, analysis_config.start_time),
            end_time=cast(datetime | None, analysis_config.end_time),
            artifact_service=artifact_service,
        )

        def record_error(message: str) -> None:
            stage_errors.append(message)
            context.metadata.setdefault("analysis_errors", []).append(message)

        for key in order:
            spec = specs[key]
            dependencies = resolver.dependencies(key)
            blockers = [
                dependency for dependency in dependencies if states.get(dependency) != "completed"
            ]
            if blockers:
                message = f"{key}: dependency blocked by {', '.join(blockers)}"
                states[key] = "dependency_blocked"
                record_error(message)
                blocked_entry = {
                    "analysis": key,
                    "dependencies": blockers,
                    "required": bool(spec.required),
                }
                context.metadata.setdefault("analysis_dependency_blocked", []).append(blocked_entry)
                summary[key] = {
                    "type": spec.type,
                    "status": "dependency_blocked",
                    "dependencies": blockers,
                }
                if spec.required:
                    fatal_errors.append(message)
                continue

            manager = item_checkpoint_manager(context)
            input_refs = tuple(dict.fromkeys(resolver.input_refs(key).values()))
            request = CheckpointRequest(
                stage=self.name,
                item=key,
                config=spec,
                dependencies=tuple(
                    ((self.name, ref) if ref in specs else ("load_sources", ref))
                    for ref in input_refs
                ),
            )
            lookup = manager.lookup(request) if manager is not None else None
            if manager is not None and lookup is not None and lookup.receipt is not None:
                restored = manager.restore_source(lookup.receipt)
                context.sources[key] = restored
                delta = lookup.receipt.context_delta
                summary[key] = dict(delta.get("summary", {}))
                artifacts = delta.get("analysis_artifacts")
                if isinstance(artifacts, list) and artifacts:
                    context.metadata.setdefault("analysis_artifacts", []).extend(artifacts)
                product = delta.get("product_artifact")
                if isinstance(product, dict):
                    context.metadata.setdefault("product_artifacts", {})[key] = product
                states[key] = "completed"
                success_count += 1
                context.log_progress(f"    Restored analysis checkpoint: {key}")
                continue

            try:
                context.log_progress(f"    Analysis: {key} ({spec.type})")
                inputs = resolver.resolve(key, context.sources)
                analysis = analysis_registry.get(spec.type)()
                adopted = None
                if (
                    manager is not None
                    and manager.resume
                    and artifact_service.has_finalized_candidate(key)
                ):
                    adopted = artifact_service.restore_finalized(
                        key,
                        build_analysis_artifact_identity(spec, inputs, analysis),
                    )
                if adopted is not None:
                    out_ds = adopted.dataset
                    geometry = analysis.output_geometry
                    out_ds.attrs["geometry"] = geometry.name.lower()
                    out_ds.attrs["derived"] = True
                    out_ds.attrs.setdefault("source_label", key)
                    context.metadata.setdefault("analysis_artifacts", []).extend(
                        adopted.manifest_entries
                    )
                    if adopted.product_metadata is not None:
                        context.metadata.setdefault("product_artifacts", {})[key] = dict(
                            adopted.product_metadata
                        )
                    context.sources[key] = SourceData(
                        data=out_ds,
                        label=key,
                        source_type=spec.type,
                        geometry=geometry,
                        variables={},
                        config={
                            **spec.model_dump(),
                            **dict(adopted.source_config),
                        },
                    )
                    summary[key] = {
                        "type": spec.type,
                        "geometry": geometry.name.lower(),
                        "variables": list(out_ds.data_vars),
                    }
                    states[key] = "completed"
                    success_count += 1
                    if manager is not None:
                        manager.capture_source(
                            request,
                            context.sources[key],
                            context_delta={
                                "summary": summary[key],
                                "analysis_artifacts": [
                                    dict(entry) for entry in adopted.manifest_entries
                                ],
                                "product_artifact": adopted.product_metadata,
                            },
                        )
                    context.log_progress(
                        f"    Adopted finalized analysis artifact checkpoint: {key}"
                    )
                    continue
                result = AnalysisResult.adapt(analysis.analyze_inputs(inputs, spec, runtime))
                out_ds = result.dataset

                geometry = analysis.output_geometry
                out_ds.attrs["geometry"] = geometry.name.lower()
                out_ds.attrs["derived"] = True
                out_ds.attrs.setdefault("source_label", key)
            except Exception as exc:  # noqa: BLE001 - optional analyses are soft failures
                logger.exception("Analysis '%s' failed", key)
                if isinstance(exc, AnalysisExecutionError) and exc.manifest_entries:
                    partial_entries = [
                        {**dict(entry), "analysis": key} for entry in exc.manifest_entries
                    ]
                    context.metadata.setdefault("analysis_artifacts", []).extend(partial_entries)
                    context.metadata.setdefault("analysis_partial_failure", []).append(
                        {"analysis": key, "finalized_artifacts": len(partial_entries)}
                    )
                message = f"{key}: {exc}"
                states[key] = "failed"
                record_error(message)
                context.log_progress(f"warning: analysis failed for {key}: {exc}")
                if spec.required:
                    fatal_errors.append(message)
                continue

            try:
                declarations: tuple[ArtifactDeclaration, ...] = ()
                if result.artifacts:
                    execution_identity = build_analysis_artifact_identity(spec, inputs, analysis)
                    declarations = tuple(
                        ArtifactDeclaration(
                            kind=declaration.kind,
                            role=declaration.role,
                            reload=declaration.reload,
                            options={
                                **dict(declaration.options),
                                "identity": {
                                    **(
                                        dict(declaration.options["identity"])
                                        if isinstance(declaration.options.get("identity"), Mapping)
                                        else {}
                                    ),
                                    **dict(execution_identity),
                                },
                            },
                        )
                        for declaration in result.artifacts
                    )
                materialized = artifact_service.materialize(key, out_ds, declarations)
            except Exception as exc:  # noqa: BLE001 - persistence failures are always fatal
                logger.exception("Artifact write failed for analysis '%s'", key)
                message = f"{key}: artifact write failed: {exc}"
                states[key] = "failed"
                record_error(message)
                fatal_errors.append(message)
                context.log_progress(f"error: {message}")
                continue

            out_ds = materialized.dataset
            out_ds.attrs["geometry"] = geometry.name.lower()
            out_ds.attrs["derived"] = True
            out_ds.attrs.setdefault("source_label", key)

            manifest_entries: list[Mapping[str, Any]] = [
                {**dict(entry), "analysis": key} for entry in result.manifest_entries
            ]
            manifest_entries.extend(materialized.manifest_entries)
            if manifest_entries:
                context.metadata.setdefault("analysis_artifacts", []).extend(manifest_entries)
            if materialized.product_metadata is not None:
                context.metadata.setdefault("product_artifacts", {})[key] = dict(
                    materialized.product_metadata
                )

            context.sources[key] = SourceData(
                data=out_ds,
                label=key,
                source_type=spec.type,
                geometry=geometry,
                variables={},
                config={**spec.model_dump(), **dict(materialized.source_config)},
            )
            summary[key] = {
                "type": spec.type,
                "geometry": geometry.name.lower(),
                "variables": list(out_ds.data_vars),
            }
            states[key] = "completed"
            success_count += 1
            if manager is not None:
                analysis_artifacts = [
                    dict(entry)
                    for entry in context.metadata.get("analysis_artifacts", [])
                    if isinstance(entry, Mapping) and entry.get("analysis") == key
                ]
                product_artifact = context.metadata.get("product_artifacts", {}).get(key)
                manager.capture_source(
                    request,
                    context.sources[key],
                    context_delta={
                        "summary": summary[key],
                        "analysis_artifacts": analysis_artifacts,
                        "product_artifact": product_artifact,
                    },
                )

        context.metadata["analysis_status"] = states
        if fatal_errors:
            return self._create_result(
                StageStatus.FAILED,
                data=summary,
                error=fatal_errors[0],
                duration=time.time() - start,
                warnings=list(stage_errors),
            )
        if stage_errors:
            if success_count == 0:
                return self._create_result(
                    StageStatus.FAILED,
                    data=summary,
                    error=f"all {len(specs)} analyses failed or were dependency blocked",
                    duration=time.time() - start,
                    warnings=list(stage_errors),
                )
            return self._create_result(
                StageStatus.COMPLETED,
                data=summary,
                duration=time.time() - start,
                warnings=list(stage_errors),
            )
        return self._create_result(
            StageStatus.COMPLETED, data=summary, duration=time.time() - start
        )
