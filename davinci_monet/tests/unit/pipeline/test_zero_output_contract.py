"""Zero-output half of the stage error contract (FABLE A1/B4, Gap 1).

Per-item stage failures are non-fatal warnings *when at least one item
succeeds*. But a stage that attempts one or more items and produces **zero**
successes must return :class:`StageStatus.FAILED` with a stage-level error --
otherwise a run in which every pair (or every plot) fails would report
``success=True`` and print a success banner over an empty result set. These
tests pin the FAILED half without disturbing the partial-success half covered in
``test_pipeline_core.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline import (
    PairingStage,
    PipelineContext,
    PipelineRunner,
    PlottingStage,
    SaveResultsStage,
    StageStatus,
    StatisticsStage,
)
from davinci_monet.pipeline.stages.base import SourceData


def _point_sources() -> dict[str, SourceData]:
    source_ds = xr.Dataset(
        {"o3": ("time", [1.0, 2.0])},
        coords={"time": [0, 1]},
        attrs={"geometry": "point"},
    )
    return {
        "airnow": SourceData(source_ds, "airnow", "pt_sfc", DataGeometry.POINT),
        "cam": SourceData(source_ds, "cam", "generic", DataGeometry.POINT),
    }


def _two_pair_config() -> dict[str, Any]:
    axes = {"x": {"source": "airnow", "variable": "o3"}, "y": {"source": "cam", "variable": "o3"}}
    return {"pairs": {"bad1": axes, "bad2": axes}}


def test_all_pairs_failed_returns_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every pair job failing -> PairingStage FAILED with an 'all N pairs failed' error."""
    ctx = PipelineContext(config=_two_pair_config(), sources=_point_sources())
    stage = PairingStage()

    def _fail(context, job, pairing_config_dict, debug):
        return job, None, f"{job.pair_key}: forced pair failure"

    monkeypatch.setattr(stage, "_run_pair_job", _fail)

    result = stage.execute(ctx)

    assert result.status is StageStatus.FAILED
    assert result.error == "all 2 pairs failed"
    assert ctx.metadata["pairing_errors"] == [
        "bad1: forced pair failure",
        "bad2: forced pair failure",
    ]


def test_all_pairs_failed_fails_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The all-pairs-failed FAILED status propagates to PipelineResult.success."""
    ctx = PipelineContext(config=_two_pair_config(), sources=_point_sources())
    stage = PairingStage()

    def _fail(context, job, pairing_config_dict, debug):
        return job, None, f"{job.pair_key}: forced pair failure"

    monkeypatch.setattr(stage, "_run_pair_job", _fail)

    result = PipelineRunner(stages=[stage], show_progress=False).run(ctx)

    assert result.success is False
    assert [sr.stage_name for sr in result.failed_stages] == ["pairing"]


def test_all_plots_failed_keeps_stats_csv(tmp_path: Any) -> None:
    """All plots failing -> success False, but the earlier stats CSV survives on disk.

    ``SaveResultsStage`` runs before ``PlottingStage`` in the standard pipeline,
    so a run in which every plot fails must still leave ``statistics_summary.csv``
    written.
    """
    times = pd.date_range("2024-01-01", periods=3, freq="h")
    paired = xr.Dataset(
        {
            "airnow_o3": (
                "time",
                np.array([1.0, 2.0, 3.0]),
                {
                    "axis": "x",
                    "source_label": "airnow",
                    "dataset_variable": "o3",
                    "canonical_name": "o3",
                },
            ),
            "cam_o3": (
                "time",
                np.array([1.1, 2.2, 2.9]),
                {
                    "axis": "y",
                    "source_label": "cam",
                    "dataset_variable": "O3",
                    "canonical_name": "o3",
                },
            ),
        },
        coords={"time": times},
    )
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "stats": {"metrics": ["N", "MB", "RMSE"]},
            "plots": {"scatter": {"type": "scatter", "pairs": ["missing_pair"]}},
        },
        paired={"cam_airnow_o3": paired},
    )

    runner = PipelineRunner(
        stages=[StatisticsStage(), SaveResultsStage(), PlottingStage()],
        show_progress=False,
    )
    result = runner.run(ctx)

    assert result.success is False
    plotting = ctx.results["plotting"]
    assert plotting.status is StageStatus.FAILED
    assert plotting.error == "all 1 plots failed"
    assert (tmp_path / "statistics_summary.csv").exists()


def test_empty_pairs_config_stays_completed() -> None:
    """Zero configured pairs is not a failure -- the stage completes with no output."""
    ctx = PipelineContext(config={"pairs": {}}, sources=_point_sources())

    result = PairingStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    assert result.error is None
