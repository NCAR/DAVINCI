"""Single-source statistics and plotting stage behavior.

The standard PlottingStage and StatisticsStage handle source-only runs when
there are loaded sources but no pairs, and descriptive statistics are written to
``statistics_descriptive.csv`` while paired summaries remain separate.
"""

from __future__ import annotations

from typing import Any, cast

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed

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

    def test_standard_workflow_keeps_implicit_descriptive_stats(self) -> None:
        ctx = PipelineContext(sources={"raw": xr.Dataset({"value": ("time", [1.0, 3.0])})})

        assert StatisticsStage().validate(ctx) is True
        result = StatisticsStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        assert result.data["raw"]["value"]["mean"] == 2.0

    def test_descriptive_stats_handle_boolean_and_skip_text(self) -> None:
        ds = xr.Dataset(
            {
                "value": ("sample", [1.0, 3.0]),
                "valid": ("sample", [True, False]),
                "label": ("sample", ["a", "b"]),
            }
        )
        ctx = PipelineContext(config={"stats": {}}, sources={"synthetic": ds})

        res = StatisticsStage().execute(ctx)

        assert res.status == StageStatus.COMPLETED
        assert res.data["synthetic"]["valid"]["N"] == 2
        assert res.data["synthetic"]["valid"]["mean"] == 0.5
        assert "label" not in res.data["synthetic"]

    def test_descriptive_stats_never_compute_artifact_derived_sources(self) -> None:
        @delayed
        def fail_if_computed() -> np.ndarray:
            raise AssertionError("artifact payload was eagerly scanned")

        guarded = da.from_delayed(fail_if_computed(), shape=(2,), dtype=np.float64)
        artifact = SourceData(
            data=xr.Dataset({"large": ("time", guarded)}),
            label="scaling",
            source_type="aod_scaling",
            geometry=DataGeometry.GRID,
            config={"artifact_dir": "/ignored/finalized/scaling"},
        )
        raw = xr.Dataset({"value": ("time", [1.0, 3.0])})
        ctx = PipelineContext(
            config={"stats": {}},
            sources={"raw": raw, "scaling": artifact},
        )

        result = StatisticsStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        assert set(result.data) == {"raw"}
        assert ctx.metadata["statistics_skipped_artifact_sources"] == ["scaling"]


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
        config = cast(dict[str, Any], ctx.config)
        config["plots"]["o3_hist"]["output_subdir"] = "plots/daily"

        res = PlottingStage().execute(ctx)

        assert res.status == StageStatus.COMPLETED
        generated = [str(path) for path in res.data["plots_generated"]]
        expected = tmp_path / "out" / "plots" / "daily" / "o3_hist.png"
        assert str(expected) in generated
        assert expected.exists()
        assert not (tmp_path / "out" / "o3_hist.png").exists()

    def test_execute_honors_single_source_plot_formats(self, tmp_path: Any) -> None:
        ctx = _geometry_ctx(tmp_path)
        config = cast(dict[str, Any], ctx.config)
        plot = config["plots"].pop("o3_hist")
        plot["formats"] = ["pdf"]
        config["plots"]["o3.hist"] = plot
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

    def test_mode_selection_precedes_lazy_finite_probe(self, tmp_path: Any) -> None:
        @delayed
        def selected_mode() -> np.ndarray:
            return np.arange(4.0)[:, None]

        @delayed
        def unrelated_mode() -> np.ndarray:
            raise AssertionError("unselected mode was computed")

        selected = da.from_delayed(selected_mode(), shape=(4, 1), dtype=np.float64)
        unrelated = da.from_delayed(unrelated_mode(), shape=(4, 1), dtype=np.float64)
        values = da.concatenate((selected, unrelated), axis=1)
        dataset = xr.Dataset(
            {"pc": (("time", "mode"), values)},
            coords={"time": np.arange(4), "mode": [1, 2]},
        )
        source = SourceData(dataset, "basis", "eof", DataGeometry.GRID)
        ctx = PipelineContext(
            config={
                "analysis": {"output_dir": str(tmp_path / "out")},
                "plots": {
                    "pc_hist": {
                        "type": "histogram",
                        "source": "basis",
                        "variable": "pc",
                        "mode": 1,
                        "formats": ["png"],
                    }
                },
            },
            sources={"basis": source},
        )

        result = PlottingStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        assert result.data["plot_count"] == 1

    def test_artifact_geometry_is_rejected_before_rendering(self, tmp_path: Any) -> None:
        artifact = SourceData(
            xr.Dataset({"count": ("file", [1, 2])}),
            "corrected",
            "mmr_writer",
            DataGeometry.ARTIFACT,
        )
        ctx = PipelineContext(
            config={
                "analysis": {"output_dir": str(tmp_path / "out")},
                "plots": {
                    "invalid": {
                        "type": "histogram",
                        "source": "corrected",
                        "variable": "count",
                    }
                },
            },
            sources={"corrected": artifact},
        )

        result = PlottingStage().execute(ctx)

        assert result.status is StageStatus.FAILED
        assert "ARTIFACT geometry" in ctx.metadata["plot_errors"][0]


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
