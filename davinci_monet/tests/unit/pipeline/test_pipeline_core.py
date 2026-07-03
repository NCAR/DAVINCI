"""Tests for pipeline module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.base import PairedData
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline import (
    BaseStage,
    LoadSourcesStage,
    PairingStage,
    PipelineBuilder,
    PipelineContext,
    PipelineResult,
    PipelineRunner,
    PlottingStage,
    SaveResultsStage,
    Stage,
    StageResult,
    StageStatus,
    StatisticsStage,
    create_standard_pipeline,
)
from davinci_monet.pipeline.stages.base import SourceData

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> PipelineContext:
    """Create a sample pipeline context (unified sources schema)."""
    return PipelineContext(
        config={
            "sources": {
                "test_dataset": {
                    "type": "generic",
                    "files": "/path/to/dataset.nc",
                },
                "test_geometry": {
                    "type": "pt_sfc",
                    "filename": "/path/to/geometry.nc",
                },
            },
        }
    )


@pytest.fixture
def sample_paired_dataset() -> xr.Dataset:
    """Create a sample paired dataset."""
    n_times = 100
    times = np.datetime64("2020-01-01") + np.arange(n_times) * np.timedelta64(1, "h")

    # Simulated dataset and geometry data with some correlation
    np.random.seed(42)
    x_vals = 50 + 20 * np.random.randn(n_times)
    y_vals = x_vals + 5 + 3 * np.random.randn(n_times)  # Bias of ~5

    return xr.Dataset(
        {
            "y_o3": (["time"], y_vals),
            "x_o3": (["time"], x_vals),
            "latitude": (["time"], np.full(n_times, 40.0)),
            "longitude": (["time"], np.full(n_times, -105.0)),
        },
        coords={"time": times},
    )


@pytest.fixture
def context_with_paired(sample_paired_dataset: xr.Dataset) -> PipelineContext:
    """Create a context with paired data."""
    ctx = PipelineContext()
    ctx.paired["test_dataset_test_geometry"] = sample_paired_dataset
    return ctx


# =============================================================================
# StageStatus Tests
# =============================================================================


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert StageStatus.PENDING
        assert StageStatus.RUNNING
        assert StageStatus.COMPLETED
        assert StageStatus.FAILED
        assert StageStatus.SKIPPED


# =============================================================================
# StageResult Tests
# =============================================================================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_create_result(self):
        """Test creating a stage result."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            data={"key": "value"},
            duration_seconds=1.5,
        )

        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.COMPLETED
        assert result.data == {"key": "value"}
        assert result.duration_seconds == 1.5
        assert result.error is None

    def test_result_with_error(self):
        """Test creating a failed result with error."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.FAILED,
            error="Something went wrong",
        )

        assert result.status == StageStatus.FAILED
        assert result.error == "Something went wrong"


# =============================================================================
# PipelineContext Tests
# =============================================================================


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_empty_context(self):
        """Test creating empty context."""
        ctx = PipelineContext()

        assert ctx.config == {}
        assert ctx.sources == {}
        assert ctx.paired == {}
        assert ctx.results == {}

    def test_context_with_config(self, sample_context: PipelineContext):
        """Test context with configuration."""
        config = sample_context.config_dict()
        assert "sources" in config
        assert "test_dataset" in config["sources"]
        assert "test_geometry" in config["sources"]

    def test_get_source(self):
        """Test getting a source from context."""
        ctx = PipelineContext()
        ctx.sources["test"] = {"data": "source_data"}

        assert ctx.get_source("test") == {"data": "source_data"}

    def test_get_source_not_found(self):
        """Test getting non-existent source raises error."""
        ctx = PipelineContext()

        with pytest.raises(KeyError, match="Source 'missing' not found"):
            ctx.get_source("missing")

    def test_get_paired(self, context_with_paired: PipelineContext):
        """Test getting paired data from context."""
        paired = context_with_paired.get_paired("test_dataset_test_geometry")
        assert isinstance(paired, xr.Dataset)

    def test_get_paired_not_found(self):
        """Test getting non-existent paired data raises error."""
        ctx = PipelineContext()

        with pytest.raises(KeyError, match="Paired data 'missing' not found"):
            ctx.get_paired("missing")

    def test_config_dict_supports_model_and_dict(self):
        """Stages can use config_dict() for both typed and legacy dict contexts."""
        from davinci_monet.config.parser import validate_config

        model_context = PipelineContext(
            config=validate_config({"sources": {"cam": {"type": "generic"}}})
        )
        dict_context = PipelineContext(config={"sources": {"cam": {"type": "generic"}}})

        assert model_context.config_dict()["sources"]["cam"]["type"] == "generic"
        assert dict_context.config_dict()["sources"]["cam"]["type"] == "generic"

    def test_typed_config_accessors(self):
        """Typed accessors return sub-models for both MonetConfig and dict contexts."""
        from davinci_monet.config.parser import validate_config
        from davinci_monet.config.schema import AnalysisConfig, StatsConfig

        section = {"stats": {"metrics": ["N", "MB"], "min_samples": 7}}
        model_context = PipelineContext(config=validate_config(section))
        dict_context = PipelineContext(config=section)

        for ctx in (model_context, dict_context):
            stats = ctx.stats_config()
            assert isinstance(stats, StatsConfig)
            assert stats.min_samples == 7
            assert isinstance(ctx.analysis_config(), AnalysisConfig)

        # No stats section -> None, so the stage falls back to legacy defaults.
        empty_context = PipelineContext()
        assert empty_context.stats_config() is None
        assert isinstance(empty_context.analysis_config(), AnalysisConfig)


# =============================================================================
# BaseStage Tests
# =============================================================================


class TestBaseStage:
    """Tests for BaseStage abstract class."""

    def test_custom_stage(self):
        """Test creating a custom stage."""

        class CustomStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(
                    StageStatus.COMPLETED,
                    data={"executed": True},
                )

        stage = CustomStage(name="custom")
        assert stage.name == "custom"

        ctx = PipelineContext()
        result = stage.execute(ctx)

        assert result.status == StageStatus.COMPLETED
        assert result.data == {"executed": True}

    def test_default_name(self):
        """Test stage uses class name by default."""

        class MyCustomStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        stage = MyCustomStage()
        assert stage.name == "MyCustomStage"

    def test_default_validation(self):
        """Test default validation passes."""

        class TestStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        stage = TestStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is True


# =============================================================================
# Concrete Stage Tests
# =============================================================================


class TestLoadSourcesStage:
    """Tests for the unified LoadSourcesStage."""

    def test_name(self):
        """Test stage name."""
        stage = LoadSourcesStage()
        assert stage.name == "load_sources"

    def test_validation_rejects_top_level_dataset_geometry_config(self):
        """validate() fails for a top-level dataset:/geometry: config.

        Top-level dataset:/geometry: controls are rejected at config load; the stage
        only recognizes the unified ``sources:`` shape (or pre-populated containers).
        """
        stage = LoadSourcesStage()
        ctx = PipelineContext(
            config={
                "dataset": {"test_dataset": {"type": "generic", "files": "/path/to/m.nc"}},
                "geometry": {"test_geometry": {"type": "pt_sfc", "filename": "/path/to/o.nc"}},
            }
        )
        assert stage.validate(ctx) is False

    def test_validation_with_sources_config(self):
        """validate() passes for a native sources: config."""
        stage = LoadSourcesStage()
        ctx = PipelineContext(
            config={"sources": {"cam": {"type": "generic", "files": "/path/to/m.nc"}}}
        )
        assert stage.validate(ctx) is True

    def test_validation_without_config(self):
        """validate() fails when nothing is configured or pre-loaded."""
        stage = LoadSourcesStage()
        ctx = PipelineContext()
        assert stage.validate(ctx) is False

    def test_source_kwargs_flow_through_to_reader(self, monkeypatch):
        """A unified source's reader-specific kwargs reach the registered reader.

        Exercises the unified loader path: ``_load_unified_source`` passes
        through-config kwargs (e.g. ``mech``) to the registered reader's
        ``open()`` while filtering out the loader/schema control keys.
        """
        captured: dict[str, Any] = {}

        class _StubReader:
            geometry = DataGeometry.GRID

            def open(self, file_paths, variables=None, **kwargs):
                captured["file_paths"] = list(file_paths)
                captured["variables"] = variables
                captured.update(kwargs)
                return xr.Dataset({"O3": ("time", [1.0])}, coords={"time": [0]})

        from davinci_monet.core.registry import source_registry

        monkeypatch.setattr(source_registry, "get", lambda name: _StubReader)

        stage = LoadSourcesStage()
        ctx = PipelineContext(
            config={
                "sources": {
                    "WRF-Chem": {
                        "type": "wrfchem",
                        "files": "/path/to/wrfout.nc",
                        "mech": "racm_esrl_vcp",
                    }
                }
            }
        )

        result = stage.execute(ctx)

        assert result.status == StageStatus.COMPLETED, result.error
        assert captured.get("mech") == "racm_esrl_vcp"
        assert "WRF-Chem" in ctx.sources


class TestPairingStage:
    """Tests for PairingStage."""

    def test_name(self):
        """Test stage name."""
        stage = PairingStage()
        assert stage.name == "pairing"

    def test_validation_with_data(self):
        """Test validation passes with loaded dataset and geometry sources."""
        stage = PairingStage()
        ctx = PipelineContext(
            config={
                "pairs": {
                    "dataset_geometry": {
                        "x": {"source": "geometry", "variable": "v"},
                        "y": {"source": "dataset", "variable": "v"},
                    }
                }
            }
        )
        ds = xr.Dataset({"v": ("time", [1.0])}, coords={"time": [0]})
        ctx.sources["dataset"] = SourceData(
            data=ds,
            label="dataset",
            source_type="generic",
            geometry=DataGeometry.GRID,
        )
        ctx.sources["geometry"] = SourceData(
            data=ds,
            label="geometry",
            source_type="pt_sfc",
            geometry=DataGeometry.POINT,
        )

        assert stage.validate(ctx) is True

    def test_validation_without_data(self):
        """Test validation fails without data."""
        stage = PairingStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is False

    def test_strategy_options_drops_engine_named_kwargs(self):
        """vertical_method/horizontal_method are passed by name by the engine, so they
        must be stripped from strategy_options to avoid a duplicate-keyword TypeError;
        genuine strategy options pass through."""
        opts = PairingStage._strategy_options(
            pairing_config_dict={},
            pair_spec={
                "vertical_method": "linear",
                "horizontal_method": "bilinear",
                "time_method": "linear",
                "extract_surface": True,
            },
        )
        assert "vertical_method" not in opts
        assert "horizontal_method" not in opts
        assert "time_method" not in opts
        assert opts["extract_surface"] is True

    def test_execute_forwards_pair_strategy_options(self):
        """Pair-level strategy options flow into the unified pairing strategy."""
        stage = PairingStage()
        times = pd.date_range("2024-01-01T00:00:00", periods=3, freq="1h")
        lat = np.array([0.0, 1.0])
        lon = np.array([10.0, 11.0])
        dataset = xr.Dataset(
            {"flux": (("time", "lat", "lon"), np.ones((3, 2, 2), dtype=np.float32))},
            coords={"time": times, "lat": lat, "lon": lon},
            attrs={"geometry": "grid"},
        )
        geometry = xr.Dataset(
            {"flux": ("time", np.array([10.0, 20.0, 30.0], dtype=np.float32))},
            coords={
                "time": times,
                "lat": ("time", np.array([0.0, 0.0, 0.0])),
                "lon": ("time", np.array([10.0, 10.0, 10.0])),
            },
            attrs={"geometry": "swath"},
        )
        ctx = PipelineContext(
            config={
                "pairs": {
                    "met_vs_ceres": {
                        "x": {"source": "ceres", "variable": "flux"},
                        "y": {"source": "met", "variable": "flux"},
                        "time_resolution": "1h",
                    }
                },
            }
        )
        ctx.sources["met"] = SourceData(
            data=dataset,
            label="met",
            source_type="generic",
            geometry=DataGeometry.GRID,
        )
        ctx.sources["ceres"] = SourceData(
            data=geometry,
            label="ceres",
            source_type="ceres_ssf",
            geometry=DataGeometry.SWATH,
        )

        result = stage.execute(ctx)

        assert result.status is StageStatus.COMPLETED
        paired = ctx.paired["met_vs_ceres"].data
        assert paired.sizes["time"] == 3
        assert set(paired.data_vars) == {"ceres_flux", "met_flux"}
        assert all(not str(name).startswith(("geometry_", "dataset_")) for name in paired.data_vars)

    def test_pairing_item_error_completes_with_warning(
        self,
        sample_paired_dataset: xr.Dataset,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A failed pair job is a warning, not a stage failure."""
        source_ds = xr.Dataset(
            {"o3": ("time", [1.0, 2.0])},
            coords={"time": [0, 1]},
            attrs={"geometry": "point"},
        )
        ctx = PipelineContext(
            config={
                "pairs": {
                    "bad": {
                        "x": {"source": "airnow", "variable": "o3"},
                        "y": {"source": "cam", "variable": "o3"},
                    },
                    "keep": {
                        "x": {"source": "airnow", "variable": "o3"},
                        "y": {"source": "cam", "variable": "o3"},
                    },
                }
            }
        )
        ctx.sources["airnow"] = SourceData(
            source_ds,
            label="airnow",
            source_type="pt_sfc",
            geometry=DataGeometry.POINT,
        )
        ctx.sources["cam"] = SourceData(
            source_ds,
            label="cam",
            source_type="generic",
            geometry=DataGeometry.POINT,
        )
        stage = PairingStage()

        def _fake_run_pair_job(context, job, pairing_config_dict, debug):
            if job.pair_key == "bad":
                return job, None, "bad: forced pair failure"
            paired = PairedData(
                data=sample_paired_dataset,
                x_source=job.x_source,
                y_source=job.y_source,
                geometry=DataGeometry.POINT,
            )
            return job, paired, None

        monkeypatch.setattr(stage, "_run_pair_job", _fake_run_pair_job)

        result = stage.execute(ctx)

        assert result.status is StageStatus.COMPLETED
        assert result.error is None
        assert "keep" in ctx.paired
        assert ctx.metadata["pairing_errors"] == ["bad: forced pair failure"]


class TestStatisticsStage:
    """Tests for StatisticsStage."""

    def test_name(self):
        """Test stage name."""
        stage = StatisticsStage()
        assert stage.name == "statistics"

    def test_validation_with_paired(self, context_with_paired: PipelineContext):
        """Test validation passes with paired data."""
        stage = StatisticsStage()
        assert stage.validate(context_with_paired) is True

    def test_validation_without_paired(self):
        """Test validation fails without paired data."""
        stage = StatisticsStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is False

    def test_execute_calculates_stats(self, context_with_paired: PipelineContext):
        """Test statistics are calculated correctly."""
        stage = StatisticsStage()
        result = stage.execute(context_with_paired)

        assert result.status == StageStatus.COMPLETED
        assert result.data is not None

        # Check stats for o3 variable
        pair_stats = result.data.get("test_dataset_test_geometry", {})
        o3_stats = pair_stats.get("o3", {})

        assert "n" in o3_stats
        assert "mean_bias" in o3_stats
        assert "rmse" in o3_stats
        assert "correlation" in o3_stats
        assert o3_stats["n"] == 100

    def test_metrics_takes_precedence_over_stat_list(self, context_with_paired: PipelineContext):
        """Test that 'metrics' key takes precedence over 'stat_list'.

        Regression test for review finding #2: when both keys are present
        (as happens after dataset_dump() with defaults), the user-specified
        'metrics' should win over the default 'stat_list'.
        """
        config = context_with_paired.config_dict()
        config["stats"] = {
            "stat_list": ["MB", "NMB", "R2", "RMSE"],  # defaults
            "metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"],  # user
        }
        context_with_paired.config = config
        stage = StatisticsStage()
        result = stage.execute(context_with_paired)

        assert result.status == StageStatus.COMPLETED
        pair_stats = result.data.get("test_dataset_test_geometry", {})
        o3_stats = pair_stats.get("o3", {})
        # IOA is in 'metrics' but not in 'stat_list' - must be present
        assert "IOA" in o3_stats

    def test_stats_data_filters_pairs(self, sample_paired_dataset: xr.Dataset):
        """stats.data selects the requested pair subset."""
        ctx = PipelineContext(
            config={"stats": {"data": ["keep"], "metrics": ["N", "MB"]}},
            paired={
                "keep": sample_paired_dataset,
                "skip": sample_paired_dataset.copy(),
            },
        )

        result = StatisticsStage().execute(ctx)

        assert result.status == StageStatus.COMPLETED
        assert set(result.data) == {"keep"}

    def test_stats_data_missing_pair_fails(self, sample_paired_dataset: xr.Dataset):
        """A missing stats.data reference fails the stage."""
        ctx = PipelineContext(
            config={"stats": {"data": ["missing"], "metrics": ["N", "MB"]}},
            paired={"keep": sample_paired_dataset},
        )

        result = StatisticsStage().execute(ctx)

        assert result.status == StageStatus.FAILED
        assert "missing" in (result.error or "")

    def test_per_flight_knob_via_validated_config(self):
        """per_flight is no longer a dead-pinned knob: a validated config that sets it
        (previously rejected by StrictSchema) now reaches the stage and emits per-flight
        stats."""
        from davinci_monet.config.parser import validate_config

        times = pd.date_range("2024-01-01", periods=6, freq="h")
        ds = xr.Dataset(
            {
                "obs_o3": (
                    "time",
                    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                    {"axis": "x", "source_label": "obs", "canonical_name": "o3"},
                ),
                "mod_o3": (
                    "time",
                    np.array([1.1, 2.2, 2.9, 4.2, 5.1, 5.8]),
                    {"axis": "y", "source_label": "mod", "canonical_name": "o3"},
                ),
            },
            coords={"time": times, "flight": ("time", ["F1", "F1", "F1", "F2", "F2", "F2"])},
        )
        cfg = validate_config({"stats": {"per_flight": True, "metrics": ["N", "MB", "R"]}})
        ctx = PipelineContext(config=cfg, paired={"p": ds})

        result = StatisticsStage().execute(ctx)

        assert result.status == StageStatus.COMPLETED
        assert "_per_flight" in result.data["p"]
        flights = {row["flight"] for row in result.data["p"]["_per_flight"]}
        assert flights == {"F1", "F2"}

    def test_stats_item_error_completes_with_warning(
        self,
        context_with_paired: PipelineContext,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Per-pair stats errors should complete the stage with warning metadata."""
        stage = StatisticsStage()

        def _raise(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("forced stats failure")

        monkeypatch.setattr(stage, "_calculate_stats", _raise)

        result = stage.execute(context_with_paired)

        assert result.status == StageStatus.COMPLETED
        assert result.error is None
        assert context_with_paired.metadata["stats_errors"]

    def test_domain_filter_restricts_stats_to_conus(self):
        """Stats config domain_type / domain_name must filter paired_data before metrics.

        Regression test for the silent no-op: pre-fix, writing
        ``stats: {domain_type: [conus]}`` had no effect — stats were
        computed over the full paired dataset including non-CONUS sites.
        """
        # 6 sites across regions: 4 CONUS, 2 in Asia. The Asian sites have
        # large geometry values that would skew an unfiltered mean.
        n_times = 12
        times = np.datetime64("2024-01-01") + np.arange(n_times) * np.timedelta64(1, "h")
        site_lats = np.array([35.0, 40.0, 45.0, 43.0, 28.6, 13.1])
        site_lons = np.array([-100.0, -105.0, -95.0, -85.0, 77.2, 80.3])
        geometry = np.full((n_times, 6), 10.0)
        geometry[:, 4:] = 200.0  # Asian sites: dramatically different
        dataset = np.full((n_times, 6), 11.0)

        paired = xr.Dataset(
            {
                "x_pm25": (["time", "site"], geometry),
                "y_pm25": (["time", "site"], dataset),
            },
            coords={
                "time": times,
                "site": np.arange(6),
                "latitude": ("site", site_lats),
                "longitude": ("site", site_lons),
            },
        )

        ctx = PipelineContext()
        ctx.paired["dataset_geometry"] = paired
        ctx.config = {"stats": {"domain_type": ["conus"], "metrics": ["N", "MB"]}}

        stage = StatisticsStage()
        result = stage.execute(ctx)

        assert result.status == StageStatus.COMPLETED
        pm25_stats = result.data["dataset_geometry"]["pm25"]
        # 4 CONUS sites x 12 timesteps = 48 paired points
        assert pm25_stats["N"] == 48, (
            f"Expected N=48 after CONUS filter, got {pm25_stats['N']}. "
            "domain_type filter is not being applied in StatisticsStage."
        )
        # MB should reflect only CONUS sites: dataset=11, geometry=10 → MB=+1
        assert abs(pm25_stats["MB"] - 1.0) < 0.01, (
            f"Expected MB≈1.0 (CONUS only), got {pm25_stats['MB']}. "
            "Asian sites leaked through the filter."
        )


class TestPlottingStage:
    """Tests for PlottingStage."""

    def test_name(self):
        """Test stage name."""
        stage = PlottingStage()
        assert stage.name == "plotting"

    def test_skips_without_config(self, context_with_paired: PipelineContext):
        """Test stage skips when no plot config provided."""
        stage = PlottingStage()
        result = stage.execute(context_with_paired)

        assert result.status == StageStatus.SKIPPED

    def test_plot_specs_use_pairs_key_only(self):
        """PlottingStage uses the explicit pairs key; data is no longer a fallback."""
        plot_spec = {"type": "scatter", "pairs": ["dataset_geometry_pm25"]}
        assert plot_spec.get("pairs", []) == ["dataset_geometry_pm25"]
        assert "data" not in plot_spec

    def test_plot_options_use_separate_subtitle_not_caption(
        self,
        context_with_paired: PipelineContext,
    ):
        """Plotting stage must keep subtitles separate from titles and captions."""
        title = "AOD: MERRA2 vs MODIS Terra"
        stage = PlottingStage()
        plotter_config, _ = stage._resolve_plot_options(
            context=context_with_paired,
            plot_type="scatter",
            plot_spec={},
            analysis_config={"start_time": "2003-01-01", "end_time": "2003-12-31"},
            title=title,
            paired_data=context_with_paired.paired["test_dataset_test_geometry"],
            var_spec={"x_var": "o3", "y_var": "o3"},
            x_source="test_geometry",
            y_source="test_dataset",
        )

        assert plotter_config["title"] == title
        assert plotter_config["subtitle"] == "2003-01-01 – 2003-12-31"
        assert "caption" not in plotter_config

    def test_single_source_plot_title_config_is_forwarded(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Configured single-source titles must reach the renderer config."""
        import matplotlib.pyplot as plt

        from davinci_monet.plots.base import PlotConfig

        ds = xr.Dataset(
            {
                "analyzed_aod": (
                    ("lat", "lon"),
                    np.ones((1, 2)),
                    {"display_name": "CAM7 Analyzed AOD"},
                )
            },
            coords={"lat": [0.0], "lon": [0.0, 90.0]},
            attrs={"geometry": "grid", "source_label": "five_day_cam_aod"},
        )
        seen: dict[str, Any] = {}

        class FakePlotter:
            def __init__(self) -> None:
                self.config = PlotConfig()

            def render(self, series, **kwargs):
                seen["config_title"] = self.config.title
                seen["render_title"] = kwargs.get("title")
                fig, _ax = plt.subplots()
                return fig

            def save(self, fig, path) -> None:
                path.write_text("plot")

        fake = FakePlotter()

        monkeypatch.setattr(
            "davinci_monet.plots.registry.get_plotter",
            lambda _plot_type: fake,
        )

        stage = PlottingStage()
        stage._render_single_source_plot(
            context=PipelineContext(),
            plot_name="five_day_analyzed",
            plot_type="spatial",
            plot_spec={
                "type": "spatial",
                "source": "five_day_aod",
                "variable": "analyzed_aod",
                "title": "Mean CAM7 Analyzed AOD",
                "formats": ["pdf"],
            },
            analysis_config={},
            output_dir=tmp_path,
            source_map={
                "five_day_aod": (
                    SourceData(ds, "five_day_aod", "gridded_analysis", DataGeometry.GRID),
                    ds,
                )
            },
            plots_generated=[],
            file_index=0,
        )

        assert seen["config_title"] == "Mean CAM7 Analyzed AOD"
        assert seen["render_title"] is None

    def test_missing_plot_pair_completes_with_warning(
        self,
        context_with_paired: PipelineContext,
    ):
        """A missing plot data reference is collected as warning metadata."""
        context_with_paired.config = {
            "analysis": {"output_dir": "."},
            "plots": {"scatter": {"type": "scatter", "pairs": ["missing_pair"]}},
        }

        result = PlottingStage().execute(context_with_paired)

        assert result.status == StageStatus.COMPLETED
        assert result.error is None
        assert "missing_pair" in context_with_paired.metadata["plot_errors"][0]

    def test_labeled_render_results_are_saved_without_legacy_split_helpers(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The plotting stage consumes render() lists directly."""
        from davinci_monet.plots.base import BasePlotter
        from davinci_monet.plots.registry import plotter_registry

        class LabeledMultiPlotter(BasePlotter):
            name = "scatter"
            plot_arity = "pairwise"
            plot_category = "statistical"

            def render(self, series, ax=None, **kwargs):
                fig_a, _ = self.create_figure()
                fig_b, _ = self.create_figure()
                return [("flight_a", fig_a), ("flight_b", fig_b)]

        monkeypatch.setitem(plotter_registry._components, "scatter", LabeledMultiPlotter)

        paired = xr.Dataset(
            {
                "airnow_o3": ("time", np.array([1.0, 2.0])),
                "cam_O3": ("time", np.array([1.1, 2.2])),
            },
            coords={"time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")},
        )
        paired["airnow_o3"].attrs.update(
            {
                "axis": "x",
                "source_label": "airnow",
                "dataset_variable": "o3",
                "canonical_name": "o3",
            }
        )
        paired["cam_O3"].attrs.update(
            {
                "axis": "y",
                "source_label": "cam",
                "dataset_variable": "O3",
                "canonical_name": "o3",
            }
        )
        ctx = PipelineContext(
            config={
                "analysis": {"output_dir": str(tmp_path)},
                "pairs": {
                    "cam_airnow_o3": {
                        "x": {"source": "airnow", "variable": "o3"},
                        "y": {"source": "cam", "variable": "O3"},
                    }
                },
                "plots": {"multi": {"type": "scatter", "pairs": ["cam_airnow_o3"]}},
            },
            paired={"cam_airnow_o3": paired},
        )

        result = PlottingStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        pngs = [path for path in result.data["plots_generated"] if path.endswith(".png")]
        assert len(pngs) == 2
        assert any("flight_a_" in path for path in pngs)
        assert any("flight_b_" in path for path in pngs)


class TestSaveResultsStage:
    """Tests for SaveResultsStage."""

    def test_name(self):
        """Test stage name."""
        stage = SaveResultsStage()
        assert stage.name == "save_results"

    def test_summary_csv_preserves_requested_metric_columns(
        self,
        tmp_path: Any,
        context_with_paired: PipelineContext,
    ):
        """Comparison CSV includes all computed metrics, not only fixed summary columns."""
        context_with_paired.config = {
            "analysis": {"output_dir": str(tmp_path)},
            "stats": {"metrics": ["N", "MB", "RMSE", "R2", "MdnB"]},
        }
        context_with_paired.results["statistics"] = StatisticsStage().execute(context_with_paired)

        result = SaveResultsStage().execute(context_with_paired)

        assert result.status == StageStatus.COMPLETED
        df = pd.read_csv(tmp_path / "statistics_summary.csv")
        assert {"R2", "MdnB"}.issubset(df.columns)


# =============================================================================
# create_standard_pipeline Tests
# =============================================================================


class TestCreateStandardPipeline:
    """Tests for create_standard_pipeline function."""

    def test_creates_all_stages(self):
        """Test all standard stages are created."""
        stages = create_standard_pipeline()

        assert len(stages) == 10

        stage_names = [s.name for s in stages]
        assert stage_names == [
            "load_sources",
            "analyses",
            "plot_suites",
            "pairing",
            "statistics",
            "save_results",
            "plotting",
            "summary",
            "inspection",
            "manifest",
        ]

    def test_stages_are_in_order(self):
        """Test stages are in correct execution order."""
        stages = create_standard_pipeline()

        assert stages[0].name == "load_sources"
        assert stages[1].name == "analyses"
        assert stages[2].name == "plot_suites"
        assert stages[3].name == "pairing"
        assert stages[4].name == "statistics"
        assert stages[5].name == "save_results"
        assert stages[6].name == "plotting"
        assert stages[-2].name == "inspection"
        assert stages[-1].name == "manifest"


# =============================================================================
# PipelineRunner Tests
# =============================================================================


class TestPipelineRunner:
    """Tests for PipelineRunner."""

    def test_default_stages(self):
        """Test runner uses standard pipeline by default."""
        runner = PipelineRunner()

        # Unified pipeline (geometry/paired stage fork collapsed): load_sources,
        # analyses, plot_suites, pairing, statistics, save_results, plotting,
        # summary, inspection, manifest.
        assert len(runner.stages) == 10

    def test_custom_stages(self):
        """Test runner accepts custom stages."""

        class SimpleStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(stages=[SimpleStage(name="simple")])

        assert len(runner.stages) == 1
        assert runner.stages[0].name == "simple"

    def test_add_stage(self):
        """Test adding a stage to runner."""

        class NewStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        # Pass empty list to avoid default stages
        runner = PipelineRunner(stages=[])
        runner.add_stage(NewStage(name="new"))

        assert len(runner.stages) == 1

    def test_add_stage_at_position(self):
        """Test adding a stage at specific position."""

        class StageA(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        class StageB(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(stages=[StageA(name="a")])
        runner.add_stage(StageB(name="b"), position=0)

        assert runner.stages[0].name == "b"
        assert runner.stages[1].name == "a"

    def test_remove_stage(self):
        """Test removing a stage by name."""
        runner = PipelineRunner()
        initial_count = len(runner.stages)

        assert runner.remove_stage("plotting") is True
        assert len(runner.stages) == initial_count - 1

    def test_remove_nonexistent_stage(self):
        """Test removing non-existent stage returns False."""
        runner = PipelineRunner()

        assert runner.remove_stage("nonexistent") is False

    def test_run_empty_pipeline(self):
        """Test running pipeline with no stages."""
        # Explicitly pass empty list to avoid default stages
        runner = PipelineRunner(stages=[])
        result = runner.run()

        assert result.success is True
        assert len(result.stage_results) == 0

    def test_run_simple_pipeline(self):
        """Test running a simple pipeline."""

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                context.metadata["executed"] = True
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(stages=[SuccessStage(name="success")])
        result = runner.run()

        assert result.success is True
        assert len(result.stage_results) == 1
        assert result.context is not None
        assert result.context.metadata.get("executed") is True

    def test_run_from_config_preserves_monet_config_in_context(self, tmp_path):
        """run_from_config keeps the validated MonetConfig object in PipelineContext."""
        from davinci_monet.config.schema import MonetConfig

        config = {
            "analysis": {"output_dir": str(tmp_path)},
            "sources": {"cam": {"type": "generic"}},
        }

        runner = PipelineRunner(stages=[], show_progress=False)
        result = runner.run_from_config(config)

        assert result.context is not None
        assert isinstance(result.context.config, MonetConfig)
        assert result.context.config.analysis.output_dir == tmp_path

    def test_runner_can_leave_context_datasets_open(self):
        """Callers can opt out of result-context dataset cleanup after run()."""

        class CloseableData:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class SourceStage(BaseStage):
            def __init__(self, data: CloseableData) -> None:
                super().__init__("source")
                self.data = data

            def execute(self, context: PipelineContext) -> StageResult:
                context.sources["synthetic"] = self.data
                return self._create_result(StageStatus.COMPLETED)

        data = CloseableData()
        runner = PipelineRunner(
            stages=[SourceStage(data)],
            show_progress=False,
            close_datasets_after_run=False,
        )

        result = runner.run(PipelineContext())

        assert result.success is True
        assert data.closed is False

    def test_run_with_failing_stage(self):
        """Test pipeline handles failing stage."""

        class FailStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.FAILED, error="Test failure")

        runner = PipelineRunner(stages=[FailStage(name="fail")])
        result = runner.run()

        assert result.success is False
        assert len(result.failed_stages) == 1
        assert result.failed_stages[0].error == "Test failure"

    def test_fail_fast(self):
        """Test fail_fast stops on first failure."""

        class FailStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.FAILED, error="Failed")

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(
            stages=[FailStage(name="fail"), SuccessStage(name="success")],
            fail_fast=True,
        )
        result = runner.run()

        assert result.success is False
        assert len(result.stage_results) == 1  # Only first stage ran

    def test_fail_fast_continues_after_plot_item_warning(
        self,
        context_with_paired: PipelineContext,
        tmp_path: Any,
    ):
        """Per-item plotting warnings do not stop later stages."""

        class MarkerStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                context.metadata["marker_ran"] = True
                return self._create_result(StageStatus.COMPLETED)

        context_with_paired.config = {
            "analysis": {"output_dir": str(tmp_path)},
            "plots": {"scatter": {"type": "scatter", "pairs": ["missing_pair"]}},
        }
        runner = PipelineRunner(
            stages=[PlottingStage(), MarkerStage(name="marker")],
            fail_fast=True,
            show_progress=False,
        )

        result = runner.run(context_with_paired)

        assert result.success is True
        assert [stage.stage_name for stage in result.stage_results] == ["plotting", "marker"]
        assert result.context is not None
        assert result.context.metadata["marker_ran"] is True

    def test_fail_fast_still_runs_manifest_stage(self, tmp_path):
        """The final manifest is written even when fail_fast stops normal stages."""
        import json

        from davinci_monet.pipeline.stages import ManifestStage

        class FailStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.FAILED, error="Failed")

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        ctx = PipelineContext(config={"analysis": {"output_dir": str(tmp_path)}})
        runner = PipelineRunner(
            stages=[FailStage(name="fail"), SuccessStage(name="success"), ManifestStage()],
            fail_fast=True,
            show_progress=False,
        )

        result = runner.run(ctx)

        assert result.success is False
        assert [stage.stage_name for stage in result.stage_results] == ["fail", "manifest"]
        assert "success" not in ctx.results
        manifest = tmp_path / "manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["status"] == "failed"
        assert data["failed_stages"] == ["fail"]
        assert data["stages"]["fail"] == "failed"

    def test_continue_on_failure(self):
        """Test pipeline continues when fail_fast is False."""

        class FailStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.FAILED, error="Failed")

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(
            stages=[FailStage(name="fail"), SuccessStage(name="success")],
            fail_fast=False,
        )
        result = runner.run()

        assert result.success is False
        assert len(result.stage_results) == 2  # Both stages ran

    def test_hooks(self):
        """Test pipeline hooks are called."""
        hook_calls: list[str] = []

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(
            stages=[SuccessStage(name="success")],
            hooks={
                "on_start": lambda ctx: hook_calls.append("on_start"),
                "on_stage_start": lambda s, ctx: hook_calls.append("on_stage_start"),
                "on_stage_end": lambda s, r, ctx: hook_calls.append("on_stage_end"),
                "on_end": lambda result: hook_calls.append("on_end"),
            },
        )
        runner.run()

        assert "on_start" in hook_calls
        assert "on_stage_start" in hook_calls
        assert "on_stage_end" in hook_calls
        assert "on_end" in hook_calls

    def test_stage_validation_failure(self):
        """Test stage is skipped when validation fails."""

        class ValidatingStage(BaseStage):
            def validate(self, context: PipelineContext) -> bool:
                return False

            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(stages=[ValidatingStage(name="validating")])
        result = runner.run()

        assert result.success is True
        assert result.stage_results[0].status == StageStatus.SKIPPED

    def test_stage_exception_handling(self):
        """Test exceptions in stages are handled."""

        class ExceptionStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                raise RuntimeError("Unexpected error")

        runner = PipelineRunner(stages=[ExceptionStage(name="exception")])
        result = runner.run()

        assert result.success is False
        assert "Unexpected error" in (result.stage_results[0].error or "")


# =============================================================================
# PipelineResult Tests
# =============================================================================


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_failed_stages(self):
        """Test getting failed stages."""
        result = PipelineResult(
            success=False,
            stage_results=[
                StageResult("a", StageStatus.COMPLETED),
                StageResult("b", StageStatus.FAILED, error="Error"),
                StageResult("c", StageStatus.COMPLETED),
            ],
        )

        assert len(result.failed_stages) == 1
        assert result.failed_stages[0].stage_name == "b"

    def test_completed_stages(self):
        """Test getting completed stage names."""
        result = PipelineResult(
            success=True,
            stage_results=[
                StageResult("a", StageStatus.COMPLETED),
                StageResult("b", StageStatus.SKIPPED),
                StageResult("c", StageStatus.COMPLETED),
            ],
        )

        assert result.completed_stages == ["a", "c"]

    def test_get_stage_result(self):
        """Test getting specific stage result."""
        result = PipelineResult(
            success=True,
            stage_results=[
                StageResult("a", StageStatus.COMPLETED),
                StageResult("b", StageStatus.COMPLETED),
            ],
        )

        stage_a = result.get_stage_result("a")
        assert stage_a is not None
        assert stage_a.stage_name == "a"

    def test_get_stage_result_not_found(self):
        """Test getting non-existent stage result."""
        result = PipelineResult(success=True, stage_results=[])

        assert result.get_stage_result("missing") is None


# =============================================================================
# PipelineBuilder Tests
# =============================================================================


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_empty_builder(self):
        """Test building pipeline with no stages added.

        Note: PipelineBuilder starts empty, but PipelineRunner defaults
        to standard stages if given an empty list. So an empty builder
        produces a runner with no stages only if we don't add any.
        """
        builder = PipelineBuilder()
        runner = builder.build()

        # Builder starts with empty list, runner gets that empty list
        assert len(runner.stages) == 0

    def test_add_standard_stages(self):
        """Test adding standard stages via the unified builder."""
        runner = PipelineBuilder().add_sources().add_pairing().add_statistics().build()

        assert len(runner.stages) == 3
        assert runner.stages[0].name == "load_sources"
        assert runner.stages[1].name == "pairing"
        assert runner.stages[2].name == "statistics"

    def test_add_custom_stage(self):
        """Test adding custom stage."""

        class CustomStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineBuilder().add_stage(CustomStage(name="custom")).build()

        assert len(runner.stages) == 1
        assert runner.stages[0].name == "custom"

    def test_fail_fast_setting(self):
        """Test setting fail_fast option."""
        runner = PipelineBuilder().fail_fast(False).build()

        assert runner._fail_fast is False

    def test_with_hook(self):
        """Test adding hook."""
        called = []

        runner = PipelineBuilder().with_hook("on_start", lambda ctx: called.append("start")).build()

        assert "on_start" in runner._hooks
