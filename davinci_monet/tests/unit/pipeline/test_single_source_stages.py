"""Single-source statistics and plotting stage behavior.

The standard PlottingStage and StatisticsStage handle source-only runs when
there are loaded sources but no pairs, and descriptive statistics are written to
``statistics_descriptive.csv`` while paired summaries remain separate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages import (
    PipelineContext,
    PlottingStage,
    SaveResultsStage,
    SourceData,
    StageStatus,
    StatisticsStage,
    create_standard_pipeline,
)


def _geometry_ctx(tmp_path: Any) -> PipelineContext:
    n = 100
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "O3": ("time", rng.uniform(20, 120, n), {"units": "ppbv"}),
            "CO": ("time", rng.uniform(50, 300, n), {"units": "ppbv"}),
        },
        coords={"time": np.datetime64("2024-02-01") + np.arange(n) * np.timedelta64(1, "h")},
    )
    geometry = SourceData(
        data=ds,
        label="airnow",
        source_type="pt_sfc",
        geometry=DataGeometry.POINT,
    )
    return PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path / "out")},
            "plots": {
                "o3_hist": {
                    "type": "histogram",
                    "source": "airnow",
                    "variable": "O3",
                    "title": "O3",
                }
            },
            "stats": {"metrics": ["N", "mean", "median", "std", "min", "max", "p10", "p90"]},
        },
        sources={"airnow": geometry},
    )


class TestUnifiedStatisticsStage:
    def test_validate_true_for_geometry_only(self, tmp_path: Any) -> None:
        assert StatisticsStage().validate(_geometry_ctx(tmp_path)) is True

    def test_descriptive_stats_for_geometry_only(self, tmp_path: Any) -> None:
        ctx = _geometry_ctx(tmp_path)
        res = StatisticsStage().execute(ctx)
        assert res.status == StageStatus.COMPLETED
        assert "airnow" in res.data
        assert "O3" in res.data["airnow"]
        for m in ["N", "mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90"]:
            assert m in res.data["airnow"]["O3"]
        assert res.data["airnow"]["O3"]["N"] == 100
        assert ctx.metadata.get("statistics_kind") == "descriptive"


class TestUnifiedPlottingStage:
    def test_validate_true_for_geometry_only(self, tmp_path: Any) -> None:
        assert PlottingStage().validate(_geometry_ctx(tmp_path)) is True

    def test_execute_creates_geometry_plots(self, tmp_path: Any) -> None:
        ctx = _geometry_ctx(tmp_path)
        res = PlottingStage().execute(ctx)
        assert res.status == StageStatus.COMPLETED
        pngs = list((tmp_path / "out").glob("*.png"))
        assert any("o3_hist" in p.name for p in pngs)

    def test_execute_honors_single_source_output_subdir(self, tmp_path: Any) -> None:
        ctx = _geometry_ctx(tmp_path)
        ctx.config["plots"]["o3_hist"]["output_subdir"] = "plots/daily"

        res = PlottingStage().execute(ctx)

        assert res.status == StageStatus.COMPLETED
        generated = [str(path) for path in res.data["plots_generated"]]
        expected = tmp_path / "out" / "plots" / "daily" / "o3_hist.png"
        assert str(expected) in generated
        assert expected.exists()
        assert not (tmp_path / "out" / "o3_hist.png").exists()

    def test_execute_honors_single_source_plot_formats(self, tmp_path: Any) -> None:
        ctx = _geometry_ctx(tmp_path)
        plot = ctx.config["plots"].pop("o3_hist")
        plot["formats"] = ["pdf"]
        ctx.config["plots"]["o3.hist"] = plot
        stale_png = tmp_path / "out" / "o3.hist.png"
        stale_png.parent.mkdir(parents=True, exist_ok=True)
        stale_png.write_text("stale")

        res = PlottingStage().execute(ctx)

        assert res.status == StageStatus.COMPLETED
        generated = [str(path) for path in res.data["plots_generated"]]
        expected = tmp_path / "out" / "o3.hist.pdf"
        assert str(expected) in generated
        assert expected.exists()
        assert not stale_png.exists()
        assert not (tmp_path / "out" / "o3.pdf").exists()


class TestSaveResultsDescriptive:
    def test_descriptive_writes_separate_csv_not_summary(self, tmp_path: Any) -> None:
        ctx = _geometry_ctx(tmp_path)
        ctx.results["statistics"] = StatisticsStage().execute(ctx)
        SaveResultsStage().execute(ctx)
        out = tmp_path / "out"
        # Descriptive stats go to their own file; the comparison summary is NOT written.
        assert (out / "statistics_descriptive.csv").exists()
        assert not (out / "statistics_summary.csv").exists()
        df = pd.read_csv(out / "statistics_descriptive.csv")
        assert {"mean", "median", "p90"}.issubset(df.columns)
        assert {"O3", "CO"}.issubset(set(df["Variable"]))


class TestUnifiedPipelineComposition:
    def test_geometry_stages_dropped_from_standard_pipeline(self) -> None:
        names = [s.name for s in create_standard_pipeline()]
        assert "geometry_statistics" not in names
        assert "geometry_plotting" not in names
        assert "statistics" in names
        assert "plotting" in names
