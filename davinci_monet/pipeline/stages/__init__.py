"""Pipeline stage definitions.

This package provides the Stage protocol and concrete stage implementations
for the analysis pipeline. Each stage is a composable unit of work that
transforms data through the analysis workflow.

The implementation is split across submodules for maintainability; this
``__init__`` re-exports the full public surface so existing
``from davinci_monet.pipeline.stages import X`` imports keep working:

- module `~davinci_monet.pipeline.stages.base` — framework primitives
  (:class:`StageStatus`, :class:`StageResult`, :class:`Stage`,
  :class:`PipelineContext`, :class:`SourceData`, :class:`SourcePairJob`,
  :class:`BaseStage`).
- module `~davinci_monet.pipeline.stages.helpers` — module-level helpers
  (dataset labels, variable resolution, formatters).
- module `~davinci_monet.pipeline.stages.load` — :class:`LoadSourcesStage`.
- module `~davinci_monet.pipeline.stages.plot_suites` —
  :class:`PlotSuiteStage`.
- module `~davinci_monet.pipeline.stages.pair` — :class:`PairingStage`.
- module `~davinci_monet.pipeline.stages.stats` — :class:`StatisticsStage`.
- module `~davinci_monet.pipeline.stages.plot` — :class:`PlottingStage`.
- module `~davinci_monet.pipeline.stages.io` — :class:`SaveResultsStage`.
- module `~davinci_monet.pipeline.stages.summary` — :class:`SummaryStage`.
- module `~davinci_monet.pipeline.stages.inspection` —
  :class:`InspectionStage`.
- module `~davinci_monet.pipeline.stages.manifest` —
  :class:`ManifestStage`.
- module `~davinci_monet.pipeline.stages.factory` — pipeline constructors.
"""

from __future__ import annotations

from davinci_monet.pipeline.stages.analyses import AnalysesStage
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourceData,
    SourcePairJob,
    Stage,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.factory import (
    create_geometry_pipeline,
    create_standard_pipeline,
)
from davinci_monet.pipeline.stages.helpers import (
    _format_duration,
    _format_size,
    iter_single_source_datasets,
    resolve_paired_var_names,
    tag_source_label,
)
from davinci_monet.pipeline.stages.inspection import InspectionStage
from davinci_monet.pipeline.stages.io import SaveResultsStage
from davinci_monet.pipeline.stages.load import LoadSourcesStage
from davinci_monet.pipeline.stages.manifest import ManifestStage
from davinci_monet.pipeline.stages.pair import PairingStage
from davinci_monet.pipeline.stages.plot import PlottingStage
from davinci_monet.pipeline.stages.plot_suites import PlotSuiteStage
from davinci_monet.pipeline.stages.stats import StatisticsStage
from davinci_monet.pipeline.stages.summary import SummaryStage

__all__ = [
    # Framework primitives
    "Stage",
    "BaseStage",
    "StageStatus",
    "StageResult",
    "PipelineContext",
    "SourceData",
    "SourcePairJob",
    # Module-level helpers
    "tag_source_label",
    "iter_single_source_datasets",
    "resolve_paired_var_names",
    "_format_size",
    "_format_duration",
    # Stage classes
    "AnalysesStage",
    "LoadSourcesStage",
    "PlotSuiteStage",
    "PairingStage",
    "StatisticsStage",
    "PlottingStage",
    "SaveResultsStage",
    "SummaryStage",
    "InspectionStage",
    "ManifestStage",
    # Pipeline factories
    "create_standard_pipeline",
    "create_geometry_pipeline",
]
