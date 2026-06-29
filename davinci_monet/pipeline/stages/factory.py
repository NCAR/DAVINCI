"""Pipeline factory helpers.

Convenience constructors that assemble the standard ordered list of stages.
"""

from __future__ import annotations

from davinci_monet.pipeline.stages.analyses import AnalysesStage
from davinci_monet.pipeline.stages.base import BaseStage
from davinci_monet.pipeline.stages.inspection import InspectionStage
from davinci_monet.pipeline.stages.io import SaveResultsStage
from davinci_monet.pipeline.stages.load import LoadSourcesStage
from davinci_monet.pipeline.stages.manifest import ManifestStage
from davinci_monet.pipeline.stages.pair import PairingStage
from davinci_monet.pipeline.stages.plot import PlottingStage
from davinci_monet.pipeline.stages.plot_suites import PlotSuiteStage
from davinci_monet.pipeline.stages.stats import StatisticsStage
from davinci_monet.pipeline.stages.summary import SummaryStage


# Convenience function to create a standard analysis pipeline
def create_standard_pipeline() -> list[BaseStage]:
    """Create a standard analysis pipeline with all stages.

    Returns
    -------
    list[BaseStage]
        List of stages for a complete analysis.
    """
    return [
        LoadSourcesStage(),
        AnalysesStage(),
        PlotSuiteStage(),
        PairingStage(),
        StatisticsStage(),
        PlottingStage(),
        SaveResultsStage(),
        SummaryStage(),
        InspectionStage(),
        ManifestStage(),
    ]


def create_geometry_pipeline() -> list[BaseStage]:
    """Create a single-source pipeline (no pairing stage).

    Returns
    -------
    list[BaseStage]
        List of stages for single-source analysis.
    """
    return [
        LoadSourcesStage(),
        AnalysesStage(),
        PlotSuiteStage(),
        StatisticsStage(),
        PlottingStage(),
        SaveResultsStage(),
        SummaryStage(),
        InspectionStage(),
        ManifestStage(),
    ]
