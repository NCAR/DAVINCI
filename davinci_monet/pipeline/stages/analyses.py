"""Pipeline stage that runs derived analyses and registers their outputs.

Each analysis consumes one source dataset (a raw source or a prior analysis
output) and emits a derived dataset, which is wrapped in a ``SourceData`` and
inserted back into ``context.sources`` so the rest of the pipeline treats it
like any other source.
"""

from __future__ import annotations

from typing import Any

from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourceData,
    StageResult,
    StageStatus,
)


def _topological_order(specs: dict[str, Any]) -> list[str]:
    """Order analysis keys so each runs after the analysis it depends on."""
    keys = set(specs)
    state: dict[str, int] = {}
    order: list[str] = []

    def visit(node: str) -> None:
        if state.get(node, 0) == 2:
            return
        if state.get(node, 0) == 1:
            raise ValueError(f"analyses dependency cycle detected at '{node}'")
        state[node] = 1
        dep = specs[node].source
        if dep in keys:
            visit(dep)
        state[node] = 2
        order.append(node)

    for key in specs:
        visit(key)
    return order


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
        summary: dict[str, Any] = {}

        try:
            order = _topological_order(specs)
        except ValueError as exc:
            context.metadata.setdefault("analysis_errors", []).append(str(exc))
            return self._create_result(
                StageStatus.FAILED, data=summary, error=str(exc), duration=time.time() - start
            )

        for key in order:
            spec = specs[key]
            try:
                context.log_progress(f"    Analysis: {key} ({spec.type})")
                src_obj = context.sources.get(spec.source)
                if src_obj is None:
                    raise ValueError(f"analysis '{key}' references unknown source '{spec.source}'")
                in_ds = src_obj.data if hasattr(src_obj, "data") else src_obj

                analysis = analysis_registry.get(spec.type)()
                out_ds = analysis.analyze(in_ds, spec)

                geometry = analysis.output_geometry
                out_ds.attrs["geometry"] = geometry.name.lower()
                out_ds.attrs["derived"] = True
                out_ds.attrs.setdefault("source_label", key)

                artifact_config: dict[str, Any] = {}
                if getattr(spec, "type", None) == "gridded_analysis":
                    from davinci_monet.analysis.artifacts import (
                        load_product_dataset,
                        write_product_artifacts,
                    )

                    artifact = write_product_artifacts(
                        context.analysis_config().output_dir or ".", key, out_ds
                    )
                    out_ds = load_product_dataset(artifact.analysis_path)
                    artifact_config = {
                        "artifact_path": str(artifact.analysis_path),
                        "summary_path": str(artifact.summary_path),
                    }
                    context.metadata.setdefault("product_artifacts", {})[key] = artifact_config

                out_ds.attrs["geometry"] = geometry.name.lower()
                out_ds.attrs["derived"] = True
                out_ds.attrs.setdefault("source_label", key)

                context.sources[key] = SourceData(
                    data=out_ds,
                    label=key,
                    source_type=spec.type,
                    geometry=geometry,
                    variables={},
                    config={**spec.model_dump(), **artifact_config},
                )
                summary[key] = {
                    "type": spec.type,
                    "geometry": geometry.name.lower(),
                    "variables": list(out_ds.data_vars),
                }
            except Exception as exc:  # noqa: BLE001 - soft per-analysis failure
                context.metadata.setdefault("analysis_errors", []).append(f"{key}: {exc}")
                context.log_progress(f"warning: analysis failed for {key}: {exc}")

        errors = context.metadata.get("analysis_errors") or []
        if errors:
            return self._create_result(
                StageStatus.FAILED,
                data=summary,
                error="Analyses failed: " + "; ".join(str(e) for e in errors),
                duration=time.time() - start,
            )
        return self._create_result(
            StageStatus.COMPLETED, data=summary, duration=time.time() - start
        )
