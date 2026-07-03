"""Collect the data the summary prompt needs out of a PipelineContext."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from davinci_monet.core.schema_utils import dump_schema, is_schema_object

if TYPE_CHECKING:
    from davinci_monet.config.schema import SummaryConfig
    from davinci_monet.pipeline.stages import PipelineContext


@dataclass
class ImageRef:
    """A plot image selected for the prompt."""

    caption: str
    path: str


@dataclass
class SummaryPayload:
    """Everything the summary prompt is built from."""

    period: dict[str, Any]
    sources_summary: list[str]
    pairs_summary: list[str]
    stats_rows: list[dict[str, Any]]
    images: list[ImageRef]
    instructions: str | None
    item_errors: dict[str, list[str]] = field(default_factory=dict)


# StatisticsStage/PlottingStage emit under these single keys for both
# single-source and paired-source run types.
_STATS_STAGES = ("statistics",)
_PLOT_STAGES = ("plotting",)


def _context_config_dict(context: "PipelineContext") -> dict[str, Any]:
    """Return the context config as a plain dict."""
    config_dict = getattr(context, "config_dict", None)
    if callable(config_dict):
        return config_dict()

    config = context.config
    if is_schema_object(config):
        return dump_schema(config, exclude_none=True)
    return cast(dict[str, Any], config)


def collect_payload(context: "PipelineContext", cfg: "SummaryConfig") -> SummaryPayload:
    """Build a :class:`SummaryPayload` from the run's config and stage results."""
    config = _context_config_dict(context)
    analysis = config.get("analysis", {}) or {}
    period = {"start": analysis.get("start_time"), "end": analysis.get("end_time")}

    sources_summary: list[str] = []
    sources = config.get("sources") or {}
    if isinstance(sources, dict):
        for label, spec in sources.items():
            if isinstance(spec, dict):
                stype = spec.get("type") or "?"
                sources_summary.append(f"{label} ({stype})")

    pairs = config.get("pairs") or {}
    pairs_summary = list(pairs.keys()) if isinstance(pairs, dict) else []

    stats_rows: list[dict[str, Any]] = []
    for stage_key in _STATS_STAGES:
        result = context.results.get(stage_key)
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            continue
        for pair_key, pair_stats in data.items():
            if not isinstance(pair_stats, dict):
                continue
            for var_name, var_stats in pair_stats.items():
                if (
                    not isinstance(var_name, str)
                    or var_name.startswith("_")
                    or not isinstance(var_stats, dict)
                ):
                    continue
                metrics = {k: v for k, v in var_stats.items() if not k.startswith("_")}
                stats_rows.append({"pair": pair_key, "variable": var_name, "metrics": metrics})

    from davinci_monet.ai.templates import resolve_template_for

    overrides = cfg.template_overrides or {}
    for row in stats_rows:
        row["template"] = resolve_template_for(
            row["variable"],
            override=overrides.get(row["pair"]),
            inline=cfg.templates,
        )

    all_plots: list[str] = []
    for stage_key in _PLOT_STAGES:
        result = context.results.get(stage_key)
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            continue
        for path in data.get("plots_generated", []) or []:
            if str(path).lower().endswith(".png"):
                all_plots.append(str(path))

    if cfg.plots:
        selected = [p for p in all_plots if any(k in Path(p).stem for k in cfg.plots)]
    else:
        selected = all_plots[: cfg.max_images]

    images = [ImageRef(caption=Path(p).stem, path=p) for p in selected]
    metadata_obj = getattr(context, "metadata", {})
    metadata: Mapping[str, Any] = metadata_obj if isinstance(metadata_obj, Mapping) else {}
    item_errors = {
        key: [str(message) for message in value]
        for key in (
            "pairing_errors",
            "stats_errors",
            "plot_errors",
            "analysis_errors",
        )
        if (value := metadata.get(key))
    }

    return SummaryPayload(
        period=period,
        sources_summary=sources_summary,
        pairs_summary=pairs_summary,
        stats_rows=stats_rows,
        images=images,
        instructions=cfg.instructions,
        item_errors=item_errors,
    )
