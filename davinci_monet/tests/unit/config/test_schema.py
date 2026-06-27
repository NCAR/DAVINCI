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

    def test_datetime_parsing_standard(self) -> None:
        """Test standard datetime format parsing."""
        config = AnalysisConfig(start_time="2024-01-01", end_time="2024-01-02")
        assert config.start_time == datetime(2024, 1, 1)
        assert config.end_time == datetime(2024, 1, 2)

    def test_datetime_parsing_melodies_format(self) -> None:
        """Test MELODIES-MONET datetime format."""
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
