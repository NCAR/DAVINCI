"""Tests for configuration schema (Pydantic datasets)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from davinci_monet.config.schema import (
    AnalysisConfig,
    DataProcConfig,
    MonetConfig,
    PlotGroupConfig,
    PlotStyleConfig,
    SourceConfig,
    StatsConfig,
    VariableConfig,
)
from davinci_monet.core.schema_utils import validate_schema


class TestAnalysisConfig:
    """Tests for AnalysisConfig."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = AnalysisConfig()
        assert config.start_time is None
        assert config.end_time is None
        assert config.debug is False
        assert config.workflow == "standard"

    def test_datetime_parsing_standard(self) -> None:
        """Test standard datetime format parsing."""
        config = AnalysisConfig(start_time="2024-01-01", end_time="2024-01-02")
        assert config.start_time == datetime(2024, 1, 1)
        assert config.end_time == datetime(2024, 1, 2)

    def test_datetime_parsing_legacy_hyphenated_format(self) -> None:
        """Test the legacy hyphenated datetime format."""
        config = AnalysisConfig(start_time="2019-08-02-12:00:00")
        assert config.start_time == datetime(2019, 8, 2, 12, 0, 0)

    def test_datetime_parsing_iso_format(self) -> None:
        """Test ISO datetime format."""
        config = AnalysisConfig(start_time="2024-01-01T14:30:00")
        assert config.start_time == datetime(2024, 1, 1, 14, 30, 0)

    def test_datetime_parsing_minute_precision(self) -> None:
        """Test space-separated datetimes without seconds."""
        config = AnalysisConfig(start_time="2024-01-15 00:00")
        assert config.start_time == datetime(2024, 1, 15, 0, 0)

    def test_output_dir_path_conversion(self) -> None:
        """Test output_dir is converted to Path."""
        config = AnalysisConfig(output_dir="/path/to/output")
        assert config.output_dir == Path("/path/to/output")

    def test_invalid_datetime_raises(self) -> None:
        """Test invalid datetime raises ValueError."""
        with pytest.raises(ValueError):
            AnalysisConfig(start_time="not-a-date")

    def test_style_config_none(self) -> None:
        """Test style config defaults to None."""
        config = AnalysisConfig()
        assert config.style is None

    def test_style_config_from_dict(self) -> None:
        """Test style config parsed from dict."""
        config = AnalysisConfig(style={"theme": "ncar", "context": "presentation"})
        assert config.style is not None
        assert config.style.theme == "ncar"  # type: ignore[union-attr]
        assert config.style.context == "presentation"  # type: ignore[union-attr]

    def test_style_config_object(self) -> None:
        """Test style config as PlotStyleConfig object."""
        style = PlotStyleConfig(theme="ncar", context="publication")
        config = AnalysisConfig(style=style)
        assert config.style.theme == "ncar"  # type: ignore[union-attr]
        assert config.style.context == "publication"  # type: ignore[union-attr]

    def test_production_run_requires_complete_cross_referenced_contract(self) -> None:
        raw: dict[str, Any] = {
            "run": {
                "id": "aod-model-sensor-2008-eof-wavelet-r01",
                "kind": "production",
                "completion": {
                    "required_analyses": {
                        "basis": "eof",
                        "filtered": "wavelet_filter",
                    },
                    "required_artifacts": [
                        {"analysis": "basis", "role": "basis_fit"},
                        {"analysis": "filtered", "role": "wavelet_filter"},
                    ],
                    "required_plots": ["filtered_pc1"],
                    "inspection": {
                        "required": True,
                        "presets": ["eof_wavelet"],
                    },
                },
            },
            "analysis": {
                "output_dir": "/run/a001/output",
                "log_dir": "/run/a001/logs",
            },
            "execution": {
                "attempt_root": "/run/a001",
                "checkpoints": {
                    "mode": "required",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "sources": {"model": {"type": "generic"}},
            "analyses": {
                "basis": {
                    "type": "eof",
                    "source": "model",
                    "variable": "aod",
                    "required": True,
                }
            },
            "plots": {
                "filtered_pc1": {
                    "type": "timeseries",
                    "source": "filtered",
                    "variable": "pc",
                    "mode": 1,
                }
            },
            "inspection": {
                "enabled": True,
                "required": True,
                "presets": ["eof_wavelet"],
            },
        }

        with pytest.raises(ValueError, match="requires analyses.filtered"):
            validate_schema(MonetConfig, raw)

        raw["analyses"]["filtered"] = {
            "type": "wavelet_filter",
            "source": "basis",
            "variable": "pc",
            "band": {"min": 4.0, "max": 10.0, "units": "days"},
            "min_segment_days": 20.0,
            "required": True,
        }
        config = validate_schema(MonetConfig, raw)
        assert config.run is not None
        assert config.run.kind == "production"
        assert config.run.completion is not None
        assert config.run.completion.required_plots == ["filtered_pc1"]

    def test_production_run_rejects_optional_analysis_and_inspection(self) -> None:
        raw: dict[str, Any] = {
            "run": {
                "id": "aod-model-sensor-2008-eof-r01",
                "kind": "production",
                "completion": {
                    "required_analyses": {"basis": "eof"},
                    "required_artifacts": [{"analysis": "basis", "role": "basis_fit"}],
                    "required_plots": ["basis_scree"],
                    "inspection": {
                        "required": True,
                        "presets": ["eof_wavelet"],
                    },
                },
            },
            "analysis": {
                "output_dir": "/run/a001/output",
                "log_dir": "/run/a001/logs",
            },
            "execution": {
                "attempt_root": "/run/a001",
                "checkpoints": {
                    "mode": "required",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "sources": {"model": {"type": "generic"}},
            "analyses": {
                "basis": {
                    "type": "eof",
                    "source": "model",
                    "variable": "aod",
                    "required": False,
                }
            },
            "plots": {
                "basis_scree": {
                    "type": "eof_scree",
                    "source": "basis",
                    "variable": "explained_variance",
                }
            },
            "inspection": {
                "enabled": False,
                "required": False,
                "presets": ["eof_wavelet"],
            },
        }

        with pytest.raises(ValueError, match="analyses.basis.required=true"):
            validate_schema(MonetConfig, raw)

        raw["analyses"]["basis"]["required"] = True
        with pytest.raises(ValueError, match="inspection.enabled=true"):
            validate_schema(MonetConfig, raw)

    def test_nonproduction_run_must_not_declare_completion_contract(self) -> None:
        raw: dict[str, Any] = {
            "run": {
                "id": "aod-model-sensor-2008-eof-preflight",
                "kind": "preflight",
                "completion": {
                    "required_analyses": {"basis": "eof"},
                    "required_artifacts": [{"analysis": "basis", "role": "basis_fit"}],
                    "required_plots": ["basis_scree"],
                    "inspection": {
                        "required": True,
                        "presets": ["eof_wavelet"],
                    },
                },
            },
            "sources": {"model": {"type": "generic"}},
        }

        with pytest.raises(ValueError, match="only production runs may declare completion"):
            validate_schema(MonetConfig, raw)

    @pytest.mark.parametrize(
        ("kind", "run_id"),
        [
            ("example", "aod-model-sensor-template"),
            ("smoke", "aod-model-sensor-smoke"),
            ("preflight", "aod-model-sensor-preflight"),
        ],
    )
    def test_nonproduction_run_kinds_parse_without_completion(
        self,
        kind: str,
        run_id: str,
    ) -> None:
        config = validate_schema(
            MonetConfig,
            {
                "run": {"id": run_id, "kind": kind},
                "sources": {"model": {"type": "generic"}},
            },
        )

        assert config.run is not None
        assert config.run.kind == kind
        assert config.run.completion is None

    @pytest.mark.parametrize("run_id", ["AOD-production-r01", "aod_production_r01"])
    def test_run_id_requires_lowercase_kebab_case(self, run_id: str) -> None:
        with pytest.raises(ValueError, match="lowercase kebab-case"):
            validate_schema(
                MonetConfig,
                {
                    "run": {"id": run_id, "kind": "smoke"},
                    "sources": {"model": {"type": "generic"}},
                },
            )

    def test_production_run_id_requires_revision_suffix(self) -> None:
        with pytest.raises(ValueError, match="must end in -rNN"):
            validate_schema(
                MonetConfig,
                {
                    "run": {
                        "id": "aod-model-sensor-production",
                        "kind": "production",
                        "completion": {
                            "required_analyses": {"basis": "eof"},
                            "required_artifacts": [{"analysis": "basis", "role": "basis_fit"}],
                            "required_plots": ["basis_scree"],
                            "inspection": {
                                "required": True,
                                "presets": ["eof_wavelet"],
                            },
                        },
                    },
                    "sources": {"model": {"type": "generic"}},
                },
            )

    def test_legacy_execution_contract_is_rejected(self) -> None:
        raw: dict[str, Any] = {
            "analysis": {
                "execution_contract": {
                    "name": "legacy",
                    "required_analyses": {"basis": "eof"},
                }
            },
            "sources": {"model": {"type": "generic"}},
        }

        with pytest.raises(ValueError, match="analysis.execution_contract"):
            validate_schema(MonetConfig, raw)

    def test_production_run_requires_checkpoint_execution_policy(self) -> None:
        raw = self._minimal_production_config()
        raw.pop("execution")

        with pytest.raises(ValueError, match="production runs require execution"):
            validate_schema(MonetConfig, raw)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("mode", "best_effort", "checkpoint mode"),
            ("granularity", "stage", "checkpoint granularity"),
            ("loaded_sources", False, "loaded-source checkpoints"),
            ("retain", "failed", "checkpoint retention"),
        ],
    )
    def test_production_run_rejects_incomplete_checkpoint_policy(
        self,
        field: str,
        value: Any,
        message: str,
    ) -> None:
        raw = self._minimal_production_config()
        raw["execution"]["checkpoints"][field] = value

        with pytest.raises(ValueError, match=message):
            validate_schema(MonetConfig, raw)

    def test_execution_attempt_root_must_own_output_and_logs(self) -> None:
        raw = self._minimal_production_config()
        raw["execution"]["attempt_root"] = "/different/a001"

        with pytest.raises(ValueError, match="attempt_root must be the common parent"):
            validate_schema(MonetConfig, raw)

    def test_execution_attempt_root_requires_attempt_notation(self) -> None:
        raw = self._minimal_production_config()
        raw["analysis"]["output_dir"] = "/run/current/output"
        raw["analysis"]["log_dir"] = "/run/current/logs"
        raw["execution"]["attempt_root"] = "/run/current"

        with pytest.raises(ValueError, match="aNNN notation"):
            validate_schema(MonetConfig, raw)

    def test_preflight_accepts_best_effort_item_checkpoints(self) -> None:
        config = validate_schema(
            MonetConfig,
            {
                "run": {
                    "id": "aod-model-sensor-preflight",
                    "kind": "preflight",
                },
                "analysis": {
                    "output_dir": "/run/a001/output",
                    "log_dir": "/run/a001/logs",
                },
                "execution": {
                    "attempt_root": "/run/a001",
                    "checkpoints": {
                        "mode": "best_effort",
                        "granularity": "item",
                        "loaded_sources": True,
                        "retain": "all",
                    },
                },
                "sources": {"model": {"type": "generic"}},
            },
        )

        assert config.execution is not None
        assert config.execution.checkpoints.mode == "best_effort"

    @pytest.mark.parametrize("legacy_key", ["checkpoint_dir", "resume_from"])
    def test_execution_rejects_unknown_or_legacy_keys(self, legacy_key: str) -> None:
        raw = self._minimal_production_config()
        raw["execution"][legacy_key] = "/legacy"

        with pytest.raises(ValueError, match=legacy_key):
            validate_schema(MonetConfig, raw)

    @staticmethod
    def _minimal_production_config() -> dict[str, Any]:
        return {
            "run": {
                "id": "aod-model-sensor-2008-eof-r01",
                "kind": "production",
                "completion": {
                    "required_analyses": {"basis": "eof"},
                    "required_artifacts": [{"analysis": "basis", "role": "basis_fit"}],
                    "required_plots": ["basis_scree"],
                    "inspection": {
                        "required": True,
                        "presets": ["eof_wavelet"],
                    },
                },
            },
            "analysis": {
                "output_dir": "/run/a001/output",
                "log_dir": "/run/a001/logs",
            },
            "execution": {
                "attempt_root": "/run/a001",
                "checkpoints": {
                    "mode": "required",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "sources": {"model": {"type": "generic"}},
            "analyses": {
                "basis": {
                    "type": "eof",
                    "source": "model",
                    "variable": "aod",
                    "required": True,
                }
            },
            "plots": {
                "basis_scree": {
                    "type": "eof_scree",
                    "source": "basis",
                    "variable": "explained_variance",
                }
            },
            "inspection": {
                "enabled": True,
                "required": True,
                "presets": ["eof_wavelet"],
            },
        }


class TestPlotStyleConfig:
    """Tests for PlotStyleConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = PlotStyleConfig()
        assert config.theme is None
        assert config.context == "default"
        assert config.use_seaborn is True
        assert config.seaborn_style == "whitegrid"

    def test_ncar_theme(self) -> None:
        """Test NCAR theme configuration."""
        config = PlotStyleConfig(theme="ncar", context="presentation")
        assert config.theme == "ncar"
        assert config.context == "presentation"

    def test_default_theme(self) -> None:
        """Test explicit default theme."""
        config = PlotStyleConfig(theme="default")
        assert config.theme == "default"

    def test_seaborn_options(self) -> None:
        """Test seaborn configuration options."""
        config = PlotStyleConfig(
            use_seaborn=False,
            seaborn_style="darkgrid",
        )
        assert config.use_seaborn is False
        assert config.seaborn_style == "darkgrid"


class TestVariableConfig:
    """Tests for VariableConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = VariableConfig()
        assert config.unit_scale == 1.0
        assert config.unit_scale_method == "*"

    def test_all_fields(self) -> None:
        """Test all fields can be set."""
        config = validate_schema(
            VariableConfig,
            {
                "unit_scale": 1000.0,
                "unit_scale_method": "+",
                "valid_min": 0.0,
                "valid_max": 100.0,
                "nan_value": -1.0,
                "rename": "new_name",
                "ylabel_plot": "Label",
                "vmin_plot": 0.0,
                "vmax_plot": 50.0,
                "vdiff_plot": 10.0,
                "nlevels_plot": 20,
                "style_preset": "geosit_aod",
                "levels_plot": [0.0, 0.1, 0.5, 1.0],
                "cmap_plot": "turbo",
                "extend_plot": "max",
            },
        )
        assert config.unit_scale == 1000.0
        assert config.unit_scale_method == "+"
        assert config.nan_value == -1.0
        assert config.style_preset == "geosit_aod"
        assert config.levels_plot == [0.0, 0.1, 0.5, 1.0]
        assert config.cmap_plot == "turbo"
        assert config.extend_plot == "max"


class TestSourceConfig:
    """Tests for the unified SourceConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = SourceConfig()
        assert config.radius_of_influence == 12000.0
        assert config.variables == {}

    def test_type(self) -> None:
        """Test source type field."""
        config = SourceConfig(type="cmaq")
        assert config.type == "cmaq"

    def test_files_kept_as_string(self) -> None:
        """Test files are kept as strings for glob patterns."""
        config = SourceConfig(files="/path/to/*.nc")
        assert config.files == "/path/to/*.nc"

    def test_time_padding_duration_is_validated(self) -> None:
        config = SourceConfig(time_padding="1D")
        assert config.time_padding == "1D"

        with pytest.raises(ValueError, match="time_padding must be non-negative"):
            SourceConfig(time_padding="-1h")

        with pytest.raises(ValueError, match="valid duration"):
            SourceConfig(time_padding="tomorrowish")

    def test_finalized_artifact_descriptor_is_all_or_nothing(self) -> None:
        config = SourceConfig(
            artifact_manifest="/run/manifest.json",
            artifact_role="scaling",
            artifact_analysis="scaling",
        )
        assert config.artifact_role == "scaling"

        with pytest.raises(ValueError, match="configured together"):
            SourceConfig(artifact_manifest="/run/manifest.json")
        with pytest.raises(ValueError, match="artifact_analysis requires"):
            SourceConfig(artifact_analysis="scaling")

    def test_source_mapping_is_rejected(self) -> None:
        """Pair variables must be declared in pairs."""
        with pytest.raises(ValueError, match="source-level mapping"):
            validate_schema(
                SourceConfig,
                {"mapping": {"airnow": {"O3": "OZONE", "PM25": "PM2.5"}}},
            )

    def test_dataset_variables_parsing(self) -> None:
        """Test dataset-flavored source variables are parsed as VariableConfig."""
        config = validate_schema(
            SourceConfig,
            {"type": "cmaq", "variables": {"co": {"unit_scale": 1000.0, "rename": "CO"}}},
        )
        assert config.variables["co"].unit_scale == 1000.0
        assert config.variables["co"].rename == "CO"

    def test_geometry_type_via_type(self) -> None:
        """Test geometry-flavored source uses ``type`` and ``filename``."""
        config = SourceConfig(type="pt_sfc", filename="/data/airnow.nc")
        assert config.type == "pt_sfc"
        assert config.filename == "/data/airnow.nc"

        config = SourceConfig(type="aircraft")
        assert config.type == "aircraft"

    def test_geometry_variables_parsing(self) -> None:
        """Test geometry-flavored source variables are parsed correctly."""
        config = validate_schema(
            SourceConfig,
            {
                "type": "pt_sfc",
                "variables": {
                    "O3": {"unit_scale": 1.0, "nan_value": -1.0},
                    "PM25": {"ylabel_plot": "PM2.5 (ug/m3)"},
                },
            },
        )
        assert config.variables["O3"].nan_value == -1.0
        assert config.variables["PM25"].ylabel_plot == "PM2.5 (ug/m3)"


class TestDataProcConfig:
    """Tests for DataProcConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = DataProcConfig()
        assert config.rem_nan is True
        assert config.ts_select_time == "time"
        assert config.set_axis is False

    def test_all_fields(self) -> None:
        """Test all fields."""
        config = DataProcConfig(
            rem_nan=False,
            ts_select_time="time_local",
            ts_avg_window="h",
            set_axis=True,
        )
        assert config.rem_nan is False
        assert config.ts_select_time == "time_local"


class TestPlotGroupConfig:
    """Tests for PlotGroupConfig."""

    def test_required_type(self) -> None:
        """Test type is required."""
        config = PlotGroupConfig(type="timeseries")
        assert config.type == "timeseries"

    def test_default_domain(self) -> None:
        """Test default domain settings."""
        config = PlotGroupConfig(type="taylor")
        assert config.domain_type == ["all"]
        assert config.domain_name == ["CONUS"]

    def test_data_list(self) -> None:
        """Test pairs list."""
        config = PlotGroupConfig(
            type="spatial_bias",
            pairs=["airnow_cmaq", "airnow_wrfchem"],
        )
        assert len(config.pairs) == 2

    def test_data_proc_parsing(self) -> None:
        """Test data_proc is parsed correctly."""
        config = validate_schema(
            PlotGroupConfig,
            {
                "type": "boxplot",
                "data_proc": {"rem_nan": False, "set_axis": True},
            },
        )
        assert isinstance(config.data_proc, DataProcConfig)
        assert config.data_proc.rem_nan is False


class TestStatsConfig:
    """Tests for StatsConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = StatsConfig()
        assert "MB" in config.stat_list
        assert config.round_output == 3
        assert config.output_table is False

    def test_stat_list(self) -> None:
        """Test custom stat list."""
        config = StatsConfig(stat_list=["MB", "RMSE", "R2"])
        assert config.stat_list == ["MB", "RMSE", "R2"]

    def test_runtime_knobs_have_defaults(self) -> None:
        """The previously dead-pinned runtime knobs default to legacy values."""
        config = StatsConfig()
        assert config.include_counts is True
        assert config.remove_nan is True
        assert config.min_samples == 3
        assert config.per_flight is False

    def test_runtime_knobs_are_accepted(self) -> None:
        """StrictSchema (extra=forbid) now accepts these knobs instead of rejecting them."""
        config = StatsConfig(
            include_counts=False,
            remove_nan=False,
            min_samples=10,
            per_flight=True,
        )
        assert config.include_counts is False
        assert config.remove_nan is False
        assert config.min_samples == 10
        assert config.per_flight is True


class TestMonetConfig:
    """Tests for root MonetConfig."""

    def test_empty_config(self) -> None:
        """Test empty config is valid."""
        config = MonetConfig()
        assert config.sources == {}
        assert config.pairs == {}
        assert config.plots == {}

    def test_analysis_section(self) -> None:
        """Test analysis section parsing."""
        config = validate_schema(
            MonetConfig,
            {
                "analysis": {
                    "start_time": "2024-01-01",
                    "end_time": "2024-01-02",
                    "debug": True,
                }
            },
        )
        assert config.analysis.debug is True
        assert config.analysis.start_time == datetime(2024, 1, 1)

    def test_gridded_sources_section(self) -> None:
        """Test gridded sources parse into typed SourceConfig."""
        config = validate_schema(
            MonetConfig,
            {
                "sources": {
                    "cmaq_test": {"type": "cmaq", "files": "/data/*.nc"},
                    "wrf_test": {"type": "wrfchem"},
                }
            },
        )
        assert "cmaq_test" in config.sources
        assert config.sources["cmaq_test"].type == "cmaq"
        assert config.sources["wrf_test"].type == "wrfchem"

    def test_point_sources_section(self) -> None:
        """Test point sources parse into typed SourceConfig."""
        config = validate_schema(
            MonetConfig,
            {
                "sources": {
                    "airnow": {"type": "pt_sfc", "filename": "/data/airnow.nc"},
                }
            },
        )
        assert "airnow" in config.sources
        assert config.sources["airnow"].type == "pt_sfc"

    def test_plots_section(self) -> None:
        """Test plots section parsing."""
        config = validate_schema(
            MonetConfig,
            {
                "sources": {
                    "airnow": {"type": "pt_sfc", "filename": "/data/airnow.nc"},
                    "cmaq": {"type": "cmaq", "files": "/data/cmaq.nc"},
                },
                "pairs": {
                    "airnow_cmaq": {
                        "x": {"source": "airnow", "variable": "o3"},
                        "y": {"source": "cmaq", "variable": "O3"},
                    }
                },
                "plots": {
                    "plot_grp1": {
                        "type": "timeseries",
                        "pairs": ["airnow_cmaq"],
                    },
                },
            },
        )
        assert "plot_grp1" in config.plots
        assert config.plots["plot_grp1"].type == "timeseries"

    def test_stats_section(self) -> None:
        """Test stats section parsing."""
        config = validate_schema(
            MonetConfig, {"stats": {"stat_list": ["MB", "R2"], "round_output": 2}}
        )
        assert config.stats is not None
        assert config.stats.round_output == 2

    @staticmethod
    def _synthetic_evaluation_config() -> dict[str, Any]:
        return {
            "analysis": {"workflow": "synthetic_evaluation"},
            "sources": {
                "estimate": {
                    "type": "generic",
                    "files": "/run/artifacts/scaling/chunk-*.nc",
                    "artifact_manifest": "/run/manifest.json",
                    "artifact_role": "scaling",
                    "artifact_analysis": "scaling",
                },
                "truth": {
                    "type": "generic",
                    "files": "/run/oracle/truth.nc",
                    "evaluation_only": True,
                },
            },
            "analyses": {
                "recovery": {
                    "type": "known_truth",
                    "estimate": "estimate",
                    "truth": "truth",
                }
            },
        }

    def test_synthetic_evaluation_requires_finalized_inputs_and_explicit_truth(self) -> None:
        config = validate_schema(MonetConfig, self._synthetic_evaluation_config())
        assert config.analysis.workflow == "synthetic_evaluation"
        assert config.sources["truth"].evaluation_only is True

        raw_estimate = self._synthetic_evaluation_config()
        raw_estimate["sources"]["estimate"].pop("artifact_manifest")
        raw_estimate["sources"]["estimate"].pop("artifact_role")
        raw_estimate["sources"]["estimate"].pop("artifact_analysis")
        with pytest.raises(ValueError, match="finalized manifest-validated artifact"):
            validate_schema(MonetConfig, raw_estimate)

        unmarked_truth = self._synthetic_evaluation_config()
        unmarked_truth["sources"]["truth"].pop("evaluation_only")
        with pytest.raises(ValueError, match="evaluation-only truth source"):
            validate_schema(MonetConfig, unmarked_truth)

        oracle_estimate = self._synthetic_evaluation_config()
        oracle_estimate["sources"]["estimate"]["files"] = "/run/oracle/truth.nc"
        with pytest.raises(ValueError, match="estimate must reference a finalized fit artifact"):
            validate_schema(MonetConfig, oracle_estimate)

    def test_known_truth_is_rejected_outside_synthetic_evaluation(self) -> None:
        config = self._synthetic_evaluation_config()
        config["analysis"]["workflow"] = "standard"
        with pytest.raises(ValueError, match="known_truth requires"):
            validate_schema(MonetConfig, config)

    def test_synthetic_fit_rejects_oracle_sources(self) -> None:
        with pytest.raises(ValueError, match="cannot load oracle truth"):
            validate_schema(
                MonetConfig,
                {
                    "analysis": {"workflow": "synthetic_fit"},
                    "sources": {
                        "model": {"type": "generic", "files": "/run/inputs/model.nc"},
                        "truth": {
                            "type": "generic",
                            "files": "/run/oracle/truth.nc",
                            "variables": {"delta_filter_target_true": {}},
                        },
                    },
                    "analyses": {
                        "basis": {
                            "type": "eof",
                            "source": "model",
                            "variable": "aod",
                        }
                    },
                },
            )

    def test_synthetic_saved_fits_require_finalized_manifest_descriptors(self) -> None:
        config: dict[str, Any] = {
            "analysis": {"workflow": "synthetic_fit"},
            "sources": {
                "basis": {"type": "generic", "files": "/run/basis.nc"},
                "projection_fit": {"type": "generic", "files": "/run/projection.nc"},
                "model": {"type": "generic", "files": "/run/model.nc"},
                "obs": {"type": "generic", "files": "/run/obs.nc"},
            },
            "analyses": {
                "projection": {
                    "type": "eof_projection",
                    "basis": "basis",
                    "model": "model",
                    "model_variable": "log_aod",
                    "obs": [
                        {
                            "source": "obs",
                            "variable": "log_aod",
                            "error_variable": "sigma",
                        }
                    ],
                    "bias_fit_artifact": "projection_fit",
                }
            },
        }

        with pytest.raises(ValueError, match="saved fit source 'basis'.*basis_fit"):
            validate_schema(MonetConfig, config)

        manifest = "/run/manifest.json"
        config["sources"]["basis"].update(
            artifact_manifest=manifest,
            artifact_role="basis_fit",
            artifact_analysis="aod_basis",
        )
        with pytest.raises(ValueError, match="saved fit source 'projection_fit'.*projection_fit"):
            validate_schema(MonetConfig, config)

        config["sources"]["projection_fit"].update(
            artifact_manifest=manifest,
            artifact_role="projection_fit",
            artifact_analysis="obs_pcs",
        )
        parsed = validate_schema(MonetConfig, config)
        assert parsed.sources["basis"].artifact_role == "basis_fit"

    def test_wavelet_resolution_threshold_matches_projection_source(self) -> None:
        config: dict[str, Any] = {
            "sources": {
                "basis": {"type": "generic"},
                "model": {"type": "generic"},
                "obs": {"type": "generic"},
            },
            "analyses": {
                "projection": {
                    "type": "eof_projection",
                    "basis": "basis",
                    "model": "model",
                    "model_variable": "log_aod",
                    "obs": [
                        {
                            "source": "obs",
                            "variable": "log_aod",
                            "error_variable": "sigma",
                        }
                    ],
                    "bias_fit_window": {"start": "2001-01-01", "end": "2001-12-31"},
                    "min_resolution": 0.4,
                },
                "filtered": {
                    "type": "wavelet_filter",
                    "source": "projection",
                    "min_resolution": 0.3,
                    "band": {"min": 4.0, "max": 32.0},
                    "min_segment_days": 64.0,
                },
            },
        }

        with pytest.raises(ValueError, match="min_resolution must match"):
            validate_schema(MonetConfig, config)

        config["analyses"]["filtered"]["min_resolution"] = 0.4
        parsed = validate_schema(MonetConfig, config)
        assert parsed.analyses["filtered"].min_resolution == 0.4

    def test_projection_rejects_incompatible_derived_eof_basis(self) -> None:
        def config_with_basis(*, standardize: bool, rotation: str) -> dict[str, Any]:
            return {
                "sources": {
                    "model": {"type": "generic"},
                    "obs": {"type": "generic"},
                },
                "analyses": {
                    "basis": {
                        "type": "eof",
                        "source": "model",
                        "variable": "log_aod",
                        "standardize": standardize,
                        "rotation": rotation,
                    },
                    "projection": {
                        "type": "eof_projection",
                        "basis": "basis",
                        "model": "model",
                        "model_variable": "log_aod",
                        "obs": [
                            {
                                "source": "obs",
                                "variable": "log_aod",
                                "error_variable": "sigma",
                            }
                        ],
                        "bias_fit_window": {
                            "start": "2001-01-01",
                            "end": "2001-12-31",
                        },
                    },
                },
            }

        with pytest.raises(ValueError, match="standardize=false"):
            validate_schema(
                MonetConfig,
                config_with_basis(standardize=True, rotation="none"),
            )
        with pytest.raises(ValueError, match="rotation=none"):
            validate_schema(
                MonetConfig,
                config_with_basis(standardize=False, rotation="varimax"),
            )

    def test_plot_rejects_artifact_analysis_source(self) -> None:
        with pytest.raises(ValueError, match="ARTIFACT analysis output"):
            validate_schema(
                MonetConfig,
                {
                    "sources": {"scaling": {"type": "generic"}},
                    "analyses": {
                        "corrected": {
                            "type": "mmr_writer",
                            "scaling": "scaling",
                            "files": "/run/mmr/*.nc4",
                            "output_dir": "/run/corrected",
                        }
                    },
                    "plots": {
                        "invalid": {
                            "type": "histogram",
                            "source": "corrected",
                            "variable": "outside_coverage_identity_count",
                        }
                    },
                },
            )

    def test_synthetic_evaluation_cannot_refit(self) -> None:
        config = self._synthetic_evaluation_config()
        config["analyses"]["refit"] = {
            "type": "eof",
            "source": "estimate",
            "variable": "aod_target",
        }
        with pytest.raises(ValueError, match="only known_truth"):
            validate_schema(MonetConfig, config)

    def test_normal_configs_do_not_apply_synthetic_oracle_path_rules(self) -> None:
        config = validate_schema(
            MonetConfig,
            {
                "sources": {
                    "reference": {
                        "type": "generic",
                        "files": "/ordinary/oracle/reference.nc",
                        "variables": {"temperature_true": {}},
                    }
                },
                "analyses": {
                    "reference_eof": {
                        "type": "eof",
                        "source": "reference",
                        "variable": "temperature_true",
                    }
                },
            },
        )
        assert "reference" in config.sources
        assert "reference_eof" in config.analyses

    def test_full_config(self) -> None:
        """Test full configuration using the unified sources/pairs schema."""
        config = validate_schema(
            MonetConfig,
            {
                "analysis": {
                    "start_time": "2024-01-01",
                    "end_time": "2024-01-02",
                    "output_dir": "/output",
                    "debug": True,
                },
                "sources": {
                    "cmaq": {
                        "files": "/data/cmaq/*.nc",
                        "type": "cmaq",
                        "radius_of_influence": 15000,
                    },
                    "airnow": {
                        "filename": "/data/airnow.nc",
                        "type": "pt_sfc",
                        "variables": {"OZONE": {"nan_value": -1.0}},
                    },
                },
                "pairs": {
                    "cmaq_airnow_o3": {
                        "x": {"source": "airnow", "variable": "OZONE"},
                        "y": {"source": "cmaq", "variable": "OZONE"},
                    }
                },
                "plots": {
                    "timeseries": {
                        "type": "timeseries",
                        "pairs": ["cmaq_airnow_o3"],
                        "domain_type": ["all"],
                    }
                },
                "stats": {"stat_list": ["MB", "RMSE"]},
            },
        )

        assert config.analysis.debug is True
        assert config.sources["cmaq"].radius_of_influence == 15000
        assert config.sources["airnow"].variables["OZONE"].nan_value == -1.0

    def test_monet_config_parses_unified_pairs(self) -> None:
        """Test root config parses unified source pairs as typed configs."""
        config = validate_schema(
            MonetConfig,
            {
                "sources": {
                    "a": {"type": "generic", "files": "/tmp/a.nc"},
                    "b": {"type": "generic", "files": "/tmp/b.nc"},
                },
                "pairs": {
                    "a_b": {
                        "x": {"source": "a", "variable": "O3"},
                        "y": {"source": "b", "variable": "O3"},
                    }
                },
            },
        )

        assert "a_b" in config.pairs
        assert config.pairs["a_b"].sources == ["a", "b"]
        assert config.pairs["a_b"].x.source == "a"
        assert config.pairs["a_b"].x.variable == "O3"
        assert config.pairs["a_b"].y.source == "b"
        assert config.pairs["a_b"].y.variable == "O3"

    def test_plot_data_is_rejected(self) -> None:
        """Plot configs use pairs; legacy data is a hard validation error."""
        with pytest.raises(ValueError, match="plots.*data.*no longer supported"):
            validate_schema(
                MonetConfig,
                {
                    "sources": {
                        "a": {"type": "generic", "files": "/tmp/a.nc"},
                        "b": {"type": "generic", "files": "/tmp/b.nc"},
                    },
                    "pairs": {
                        "a_b": {
                            "x": {"source": "a", "variable": "O3"},
                            "y": {"source": "b", "variable": "O3"},
                        }
                    },
                    "plots": {"scatter": {"type": "scatter", "data": ["a_b"]}},
                },
            )

    def test_plot_pairs_names_must_resolve_to_pairs(self) -> None:
        """The newer plots.*.pairs spelling gets the same validation as data."""
        with pytest.raises(ValueError, match="plots.scatter.pairs.*missing_pair"):
            validate_schema(
                MonetConfig,
                {
                    "sources": {
                        "a": {"type": "generic", "files": "/tmp/a.nc"},
                        "b": {"type": "generic", "files": "/tmp/b.nc"},
                    },
                    "pairs": {
                        "a_b": {
                            "x": {"source": "a", "variable": "O3"},
                            "y": {"source": "b", "variable": "O3"},
                        }
                    },
                    "plots": {"scatter": {"type": "scatter", "pairs": ["missing_pair"]}},
                },
            )

    def test_single_source_plot_source_must_resolve_to_source(self) -> None:
        """Single-source plot source references are validated."""
        with pytest.raises(ValueError, match="plots.hist.source.*missing_source"):
            validate_schema(
                MonetConfig,
                {
                    "sources": {"airnow": {"type": "pt_sfc", "filename": "/tmp/a.nc"}},
                    "plots": {
                        "hist": {
                            "type": "histogram",
                            "source": "missing_source",
                            "variable": "O3",
                        }
                    },
                },
            )

    def test_stats_data_names_must_resolve_to_pairs(self) -> None:
        """Stats data references are validated instead of ignored."""
        with pytest.raises(ValueError, match="stats.data.*missing_pair"):
            validate_schema(
                MonetConfig,
                {
                    "sources": {
                        "a": {"type": "generic", "files": "/tmp/a.nc"},
                        "b": {"type": "generic", "files": "/tmp/b.nc"},
                    },
                    "pairs": {
                        "a_b": {
                            "x": {"source": "a", "variable": "O3"},
                            "y": {"source": "b", "variable": "O3"},
                        }
                    },
                    "stats": {"data": ["missing_pair"]},
                },
            )


class TestExtraFieldsHandling:
    """Tests for handling extra/unknown fields."""

    def test_core_extra_fields_rejected(self) -> None:
        """Modeled core sections reject typos by default."""
        with pytest.raises(ValueError, match="analysis.*unknown_field"):
            validate_schema(
                MonetConfig,
                {
                    "analysis": {"unknown_field": "value"},
                    "sources": {"cmaq": {"type": "cmaq"}},
                },
            )

    def test_source_reader_extra_fields_allowed(self) -> None:
        """Source reader kwargs remain the extension point."""
        config = validate_schema(
            MonetConfig,
            {"sources": {"cmaq": {"type": "cmaq", "custom_option": True}}},
        )
        extra = config.sources["cmaq"].__pydantic_extra__
        assert extra is not None
        assert extra.get("custom_option") is True

    def test_axis_extra_fields_rejected(self) -> None:
        """Pair axes are modeled, not arbitrary extension points."""
        with pytest.raises(ValueError, match="role"):
            validate_schema(
                MonetConfig,
                {
                    "sources": {
                        "a": {"type": "generic", "files": "/tmp/a.nc"},
                        "b": {"type": "generic", "files": "/tmp/b.nc"},
                    },
                    "pairs": {
                        "a_b": {
                            "x": {"source": "a", "variable": "O3", "role": "old"},
                            "y": {"source": "b", "variable": "O3"},
                        }
                    },
                },
            )
