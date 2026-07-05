"""Tests for configuration parser."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from davinci_monet.config.parser import (
    ConfigBuilder,
    config_to_yaml,
    dump_config,
    expand_env_vars,
    load_config,
    load_yaml,
    merge_configs,
    preprocess_config,
    validate_config,
)
from davinci_monet.config.schema import MonetConfig
from davinci_monet.core.exceptions import ConfigurationError
from davinci_monet.core.schema_utils import validate_schema


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_from_string(self) -> None:
        """Test loading YAML from string."""
        yaml_str = """
analysis:
  debug: true
sources: {}
"""
        data = load_yaml(yaml_str)
        assert data["analysis"]["debug"] is True
        assert data["sources"] == {}

    def test_load_from_file(self) -> None:
        """Test loading YAML from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("analysis:\n  debug: true\n")
            f.flush()
            try:
                data = load_yaml(f.name)
                assert data["analysis"]["debug"] is True
            finally:
                os.unlink(f.name)

    def test_load_from_path(self) -> None:
        """Test loading YAML from Path object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("sources:\n  cmaq: {}\n")
            f.flush()
            try:
                data = load_yaml(Path(f.name))
                assert "cmaq" in data["sources"]
            finally:
                os.unlink(f.name)

    def test_empty_yaml(self) -> None:
        """Test empty/null YAML returns empty dict."""
        # Use explicit YAML null/empty content
        data = load_yaml("---\n")
        assert data == {}
        # Also test with just whitespace
        data = load_yaml("   \n\n")
        assert data == {}

    def test_invalid_yaml_raises(self) -> None:
        """Test invalid YAML raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            load_yaml("{ invalid yaml: [")

    def test_non_dict_yaml_raises(self) -> None:
        """Test non-dict YAML raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            load_yaml("- item1\n- item2")

    def test_file_not_found_raises(self) -> None:
        """Test missing file raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_yaml("/nonexistent/path/config.yaml")

    def test_missing_relative_yaml_file_is_not_parsed_as_yaml(self) -> None:
        """A typo'd config path should report a missing file, not a scalar root."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_yaml("missing_config.yaml")

    def test_inline_one_line_mapping_string_still_loads(self) -> None:
        """Inline YAML strings that look like mappings remain supported."""
        data = load_yaml("analysis: {debug: true}")
        assert data["analysis"]["debug"] is True

    def test_cmaq_airnow_example_yaml_is_parseable(self) -> None:
        """Tracked example config is well-formed YAML."""
        data = load_yaml(Path("examples/configs/cmaq_airnow.yaml"))
        assert data["sources"]["cmaq_v54"]["variables"]["O3"]["vmin_plot"] == 0


class TestExpandEnvVars:
    """Tests for expand_env_vars function."""

    def test_expand_basic(self) -> None:
        """Test basic environment variable expansion."""
        os.environ["TEST_VAR"] = "/test/path"
        data = expand_env_vars({"path": "${TEST_VAR}/file"})
        assert data["path"] == "/test/path/file"

    def test_expand_nested(self) -> None:
        """Test nested dictionary expansion."""
        os.environ["TEST_DIR"] = "/data"
        data = expand_env_vars({"analysis": {"output_dir": "${TEST_DIR}/output"}})
        assert data["analysis"]["output_dir"] == "/data/output"

    def test_expand_in_list(self) -> None:
        """Test expansion in list values."""
        os.environ["TEST_BASE"] = "/base"
        data = expand_env_vars({"files": ["${TEST_BASE}/a.nc", "${TEST_BASE}/b.nc"]})
        assert data["files"][0] == "/base/a.nc"

    def test_no_expansion_needed(self) -> None:
        """Test values without env vars are unchanged."""
        data = expand_env_vars({"path": "/static/path"})
        assert data["path"] == "/static/path"

    def test_missing_environment_variable_raises(self) -> None:
        """Unset environment variables are rejected instead of left literal."""
        with pytest.raises(ConfigurationError, match="UNSET_DAVINCI_TEST_VAR"):
            expand_env_vars({"analysis": {"output_dir": "${UNSET_DAVINCI_TEST_VAR}/output"}})

    def test_missing_bare_environment_variable_raises(self) -> None:
        """Unset bare $VAR references are rejected in path-like values."""
        with pytest.raises(ConfigurationError, match="UNSET_DAVINCI_TEST_VAR"):
            expand_env_vars({"files": "$UNSET_DAVINCI_TEST_VAR/input.nc"})

    def test_mathtext_like_dollar_strings_are_left_alone(self) -> None:
        """Plot labels with mathtext-style dollars are normal scalar strings."""
        data = expand_env_vars({"label": "$O_3$ concentration"})
        assert data["label"] == "$O_3$ concentration"

    def test_expand_user_home_in_strings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Leading ~/ is expanded for path-like strings wherever they appear."""
        monkeypatch.setenv("HOME", str(tmp_path))

        data = expand_env_vars(
            {
                "analysis": {"output_dir": "~/davinci-output", "log_dir": "~/davinci-logs"},
                "sources": {"cam": {"files": "~/data/*.nc"}},
            }
        )

        assert data["analysis"]["output_dir"] == str(tmp_path / "davinci-output")
        assert data["analysis"]["log_dir"] == str(tmp_path / "davinci-logs")
        assert data["sources"]["cam"]["files"] == str(tmp_path / "data/*.nc")


class TestPreprocessConfig:
    """Tests for preprocess_config function."""

    def test_null_sections_become_empty_dicts(self) -> None:
        """Test null sections are converted to empty dicts."""
        data = preprocess_config({"analysis": {}, "sources": None, "pairs": None})
        assert data["sources"] == {}
        assert data["pairs"] == {}

    def test_preprocess_expands_output_and_log_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Preprocessing expands home directories before schema validation."""
        monkeypatch.setenv("HOME", str(tmp_path))

        data = preprocess_config({"analysis": {"output_dir": "~/out", "log_dir": "~/logs"}})

        assert data["analysis"]["output_dir"] == str(tmp_path / "out")
        assert data["analysis"]["log_dir"] == str(tmp_path / "logs")


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_minimal_config(self) -> None:
        """Test loading minimal configuration."""
        yaml_str = """
analysis:
  debug: true
"""
        config = load_config(yaml_str)
        assert isinstance(config, MonetConfig)
        assert config.analysis.debug is True

    def test_load_full_config(self) -> None:
        """Test loading full configuration."""
        yaml_str = """
analysis:
  start_time: '2024-01-01'
  end_time: '2024-01-02'
  debug: true
sources:
  cmaq:
    type: cmaq
    files: /data/*.nc
  airnow:
    type: pt_sfc
"""
        config = load_config(yaml_str)
        assert config.analysis.start_time == datetime(2024, 1, 1)
        assert "cmaq" in config.sources
        assert "airnow" in config.sources

    def test_unknown_top_level_sections_rejected(self) -> None:
        """Unknown top-level config sections are rejected."""
        yaml_str = """
analysis:
  start_time: '2024-01-01'
  end_time: '2024-01-02'
unsupported_group:
  cmaq:
    type: cmaq
    files: /data/*.nc
"""
        with pytest.raises(ConfigurationError, match="Extra inputs"):
            load_config(yaml_str)

    def test_load_from_file(self) -> None:
        """Test loading from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("analysis:\n  debug: true\n")
            f.flush()
            try:
                config = load_config(f.name)
                assert config.analysis.debug is True
            finally:
                os.unlink(f.name)


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_valid_config(self) -> None:
        """Test validating a valid config."""
        data: dict[str, Any] = {
            "analysis": {"start_time": "2024-01-01"},
            "sources": {"test": {"type": "cmaq"}},
        }
        config = validate_config(data)
        assert isinstance(config, MonetConfig)

    def test_validate_unknown_top_level_section_rejected(self) -> None:
        """validate_config rejects unknown top-level sections."""
        with pytest.raises(ConfigurationError, match="Extra inputs"):
            validate_config({"unsupported_group": {"test": {"type": "cmaq"}}})

    def test_invalid_plot_type_rejected_at_validation(self) -> None:
        """Unknown plot types fail during config validation, not plotting."""
        with pytest.raises(ConfigurationError, match="Unknown plot type 'fake_plot_type_xyz'"):
            validate_config(
                {
                    "sources": {"cam": {"type": "generic"}},
                    "plots": {"bad": {"type": "fake_plot_type_xyz", "pairs": []}},
                }
            )

    def test_strict_rejects_extra_analysis_field(self) -> None:
        """Strict mode rejects unknown core schema fields."""
        with pytest.raises(ConfigurationError, match="analysis.extra_field"):
            validate_config(
                {
                    "analysis": {"start_time": "2024-01-01", "extra_field": "nope"},
                    "sources": {"cam": {"type": "generic"}},
                },
                strict=True,
            )

    def test_strict_rejects_extra_pairing_field(self) -> None:
        """Strict mode rejects unknown fields in modeled runtime sections."""
        with pytest.raises(ConfigurationError, match="pairing.extra_field"):
            validate_config(
                {
                    "sources": {"cam": {"type": "generic"}},
                    "pairing": {"max_pair_workers": 2, "extra_field": "nope"},
                },
                strict=True,
            )

    def test_strict_allows_source_reader_extra_fields(self) -> None:
        """Reader-specific source kwargs remain an extension point in strict mode."""
        cfg = validate_config(
            {
                "sources": {
                    "cam": {
                        "type": "generic",
                        "reader_specific_kwarg": "passed-through",
                    }
                }
            },
            strict=True,
        )

        assert getattr(cfg.sources["cam"], "reader_specific_kwarg") == "passed-through"

    def test_strict_allows_renderer_specific_plot_kwargs(self) -> None:
        """Renderer-specific plot kwargs remain an extension point in strict mode."""
        cfg = validate_config(
            {
                "sources": {"cam": {"type": "generic"}},
                "plots": {
                    "plot1": {
                        "type": "timeseries",
                        "source": "cam",
                        "variable": "O3",
                        "renderer_specific_kwarg": "passed-through",
                    }
                },
            },
            strict=True,
        )

        assert getattr(cfg.plots["plot1"], "renderer_specific_kwarg") == "passed-through"

    def test_load_config_rejects_extra_field_by_default(self) -> None:
        """load_config rejects core extra fields without requiring strict mode."""
        with pytest.raises(
            ConfigurationError,
            match=r"analysis\.extra_field",
        ):
            load_config("""
analysis:
  extra_field: nope
sources:
  cam:
    type: generic
""")

    def test_default_rejects_extra_analysis_field(self) -> None:
        """Default validation rejects unknown core fields."""
        with pytest.raises(ConfigurationError, match="Extra inputs"):
            validate_config(
                {
                    "analysis": {"start_time": "2024-01-01", "extra_field": "nope"},
                    "sources": {"cam": {"type": "generic"}},
                }
            )

    def test_validate_empty_config(self) -> None:
        """Test validating empty config."""
        config = validate_config({})
        assert isinstance(config, MonetConfig)


class TestDumpConfig:
    """Tests for dump_config function."""

    def test_dump_and_reload(self) -> None:
        """Test dumping and reloading config."""
        config = validate_schema(
            MonetConfig,
            {
                "analysis": {"debug": True},
                "sources": {"cmaq": {"type": "cmaq"}},
            },
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            try:
                dump_config(config, f.name)
                reloaded = load_config(f.name)
                assert reloaded.analysis.debug is True
                assert "cmaq" in reloaded.sources
            finally:
                os.unlink(f.name)

    def test_dump_and_reload_paths_and_datetimes_are_yaml_safe(self, tmp_path: Path) -> None:
        """Dumped configs use plain YAML scalars for Path and datetime values."""
        config = validate_schema(
            MonetConfig,
            {
                "analysis": {
                    "start_time": "2024-01-01",
                    "end_time": "2024-01-02T03:04:05",
                    "output_dir": tmp_path / "output",
                    "log_dir": tmp_path / "logs",
                },
                "sources": {"cmaq": {"type": "cmaq"}},
            },
        )
        path = tmp_path / "config.yaml"

        dump_config(config, path)
        dumped = path.read_text()
        reloaded = load_config(path)

        assert "!!python/object" not in dumped
        assert reloaded.analysis.start_time == datetime(2024, 1, 1)
        assert reloaded.analysis.end_time == datetime(2024, 1, 2, 3, 4, 5)
        assert reloaded.analysis.output_dir == tmp_path / "output"
        assert reloaded.analysis.log_dir == tmp_path / "logs"


class TestConfigToYaml:
    """Tests for config_to_yaml function."""

    def test_basic_conversion(self) -> None:
        """Test basic config to YAML conversion."""
        config = validate_schema(MonetConfig, {"analysis": {"debug": True}})
        yaml_str = config_to_yaml(config)
        assert "debug: true" in yaml_str

    def test_paths_and_datetimes_are_yaml_safe(self, tmp_path: Path) -> None:
        """Config YAML strings are reloadable by safe_load."""
        config = validate_schema(
            MonetConfig,
            {
                "analysis": {
                    "start_time": "2024-01-01",
                    "output_dir": tmp_path / "output",
                }
            },
        )

        yaml_str = config_to_yaml(config)
        reloaded = load_config(yaml_str)

        assert "!!python/object" not in yaml_str
        assert reloaded.analysis.start_time == datetime(2024, 1, 1)
        assert reloaded.analysis.output_dir == tmp_path / "output"


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_merge_two_configs(self) -> None:
        """Test merging two configurations."""
        base = validate_schema(MonetConfig, {"analysis": {"debug": False}})
        override: dict[str, Any] = {"analysis": {"debug": True}}
        merged = merge_configs(base, override)
        assert merged.analysis.debug is True

    def test_merge_adds_new_sections(self) -> None:
        """Test merging adds new sections."""
        base = validate_schema(MonetConfig, {"analysis": {}})
        override: dict[str, Any] = {"sources": {"cmaq": {"type": "cmaq"}}}
        merged = merge_configs(base, override)
        assert "cmaq" in merged.sources

    def test_merge_deep_dict(self) -> None:
        """Test deep merging of nested dicts."""
        base = validate_schema(
            MonetConfig, {"sources": {"cmaq": {"type": "cmaq", "radius_of_influence": 10000}}}
        )
        override: dict[str, Any] = {"sources": {"cmaq": {"radius_of_influence": 15000}}}
        merged = merge_configs(base, override)
        assert merged.sources["cmaq"].type == "cmaq"
        assert merged.sources["cmaq"].radius_of_influence == 15000


class TestConfigBuilder:
    """Tests for ConfigBuilder class."""

    def test_build_minimal(self) -> None:
        """Test building minimal config."""
        config = ConfigBuilder().build()
        assert isinstance(config, MonetConfig)

    def test_set_analysis(self) -> None:
        """Test setting analysis options."""
        config = ConfigBuilder().set_analysis(start_time="2024-01-01", debug=True).build()
        assert config.analysis.debug is True

    def test_add_source_dataset(self) -> None:
        """Test adding a gridded source."""
        config = ConfigBuilder().add_source("cmaq", type="cmaq", files="/data/*.nc").build()
        assert "cmaq" in config.sources
        assert config.sources["cmaq"].type == "cmaq"

    def test_add_source_geometry(self) -> None:
        """Test adding a point source."""
        config = ConfigBuilder().add_source("airnow", type="pt_sfc").build()
        assert "airnow" in config.sources

    def test_add_source_and_pair(self) -> None:
        """Test adding unified sources and source pairs."""
        config = (
            ConfigBuilder()
            .add_source("cam", type="generic", files="/data/cam.nc")
            .add_source("airnow", type="pt_sfc", filename="/data/airnow.nc")
            .add_pair(
                "cam_airnow_o3",
                x={"source": "airnow", "variable": "o3"},
                y={"source": "cam", "variable": "O3"},
            )
            .build()
        )

        assert set(config.sources) == {"cam", "airnow"}
        assert config.pairs["cam_airnow_o3"].x.source == "airnow"
        assert config.pairs["cam_airnow_o3"].sources == ["airnow", "cam"]

    def test_add_plot(self) -> None:
        """Test adding plot."""
        config = (
            ConfigBuilder()
            .add_source("airnow", type="pt_sfc")
            .add_source("cmaq", type="cmaq")
            .add_pair(
                "airnow_cmaq",
                x={"source": "airnow", "variable": "o3"},
                y={"source": "cmaq", "variable": "O3"},
            )
            .add_plot("ts1", "timeseries", pairs=["airnow_cmaq"])
            .build()
        )
        assert "ts1" in config.plots
        assert config.plots["ts1"].type == "timeseries"

    def test_set_stats(self) -> None:
        """Test setting stats."""
        config = ConfigBuilder().set_stats(stat_list=["MB", "R2"], round_output=2).build()
        assert config.stats is not None
        assert config.stats.round_output == 2

    def test_chaining(self) -> None:
        """Test method chaining."""
        config = (
            ConfigBuilder()
            .set_analysis(start_time="2024-01-01", end_time="2024-01-02")
            .add_source("cmaq", type="cmaq")
            .add_source("airnow", type="pt_sfc")
            .add_pair(
                "airnow_cmaq",
                x={"source": "airnow", "variable": "o3"},
                y={"source": "cmaq", "variable": "O3"},
            )
            .add_plot("ts", "timeseries", pairs=["airnow_cmaq"])
            .build()
        )
        assert config.analysis.start_time is not None
        assert "cmaq" in config.sources
        assert "airnow" in config.sources
        assert "ts" in config.plots

    def test_to_dict(self) -> None:
        """Test getting raw dictionary."""
        builder = ConfigBuilder().set_analysis(debug=True)
        data = builder.to_dict()
        assert data["analysis"]["debug"] is True
