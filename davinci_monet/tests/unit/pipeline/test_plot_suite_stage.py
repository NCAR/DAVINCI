from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.config.schema import MonetConfig
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages import PipelineContext, PlotSuiteStage, SourceData, StageStatus


def test_plot_suite_stage_expands_into_context_plots(tmp_path) -> None:
    ds = xr.Dataset(
        {
            "analyzed_aod": (("group", "lat", "lon"), np.ones((1, 1, 2))),
            "nudge_fraction": (("group", "lat", "lon"), np.ones((1, 1, 2))),
        },
        coords={"group": ["2008-07-01"], "lat": [0.0], "lon": [0.0, 90.0]},
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "plot_suites": {
                "daily": {
                    "preset": "gridded_aod_diagnostics",
                    "source": "daily_aod",
                    "output_subdir": "plots/daily",
                }
            },
        },
        sources={"daily_aod": SourceData(ds, "daily_aod", "gridded_analysis", DataGeometry.GRID)},
    )
    result = PlotSuiteStage().execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert "daily_analyzed_aod" in ctx.config["plots"]
    assert ctx.config["plots"]["daily_analyzed_aod"]["source"] == "daily_aod"
    assert ctx.config["plots"]["daily_analyzed_aod"]["output_subdir"] == "plots/daily"


def test_plot_suite_stage_preserves_typed_config(tmp_path) -> None:
    ds = xr.Dataset(
        {"analyzed_aod": (("group", "lat", "lon"), np.ones((1, 1, 2)))},
        coords={"group": ["2008-07-01"], "lat": [0.0], "lon": [0.0, 90.0]},
    )
    cfg = MonetConfig(
        sources={"daily_aod": {"type": "generic", "files": "daily.nc"}},
        plots={
            "existing": {
                "type": "spatial",
                "source": "daily_aod",
                "variable": "analyzed_aod",
            }
        },
        plot_suites={
            "daily": {
                "preset": "gridded_aod_diagnostics",
                "source": "daily_aod",
                "output_subdir": "plots/daily",
            }
        },
    )
    ctx = PipelineContext(
        config=cfg,
        sources={"daily_aod": SourceData(ds, "daily_aod", "gridded_analysis", DataGeometry.GRID)},
    )
    result = PlotSuiteStage().execute(ctx)
    assert result.status == StageStatus.COMPLETED
    assert isinstance(ctx.config, MonetConfig)
    assert "existing" in ctx.config.plots
    assert ctx.config.plots["daily_analyzed_aod"].source == "daily_aod"
    assert getattr(ctx.config.plots["daily_analyzed_aod"], "output_subdir") == "plots/daily"


def test_plot_suite_stage_fails_for_missing_source(tmp_path) -> None:
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "plot_suites": {
                "daily": {
                    "preset": "gridded_aod_diagnostics",
                    "source": "missing",
                }
            },
        }
    )
    result = PlotSuiteStage().execute(ctx)
    assert result.status == StageStatus.FAILED
    assert "missing" in str(result.error)
