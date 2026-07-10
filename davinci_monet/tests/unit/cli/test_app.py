"""Tests for the CLI module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from davinci_monet.cli.app import app, cli, timer

# =============================================================================
# Test CLI Application Setup
# =============================================================================


class TestCLIApp:
    """Tests for the main CLI application."""

    def test_app_exists(self) -> None:
        """Test that the app is created."""
        assert app is not None

    def test_cli_alias(self) -> None:
        """Test that cli is an alias for app."""
        assert cli is app

    def test_app_has_commands(self) -> None:
        """Test that the app has registered commands."""
        from typer import Typer

        assert isinstance(app, Typer)

    def test_version_callback(self) -> None:
        """Test the version callback."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["--version"])

        # Should exit successfully and show version
        assert result.exit_code == 0
        assert "davinci-monet" in result.stdout

    def test_help_command(self) -> None:
        """Test the help output."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "DAVINCI" in result.stdout or "davinci-monet" in result.stdout


# =============================================================================
# Test Timer Context Manager
# =============================================================================


class TestTimer:
    """Tests for the timer context manager."""

    def test_timer_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test timer with successful operation."""
        with timer("Test operation"):
            pass

        captured = capsys.readouterr()
        assert "Test operation" in captured.out
        assert "succeeded" in captured.out

    def test_timer_shows_elapsed_time(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test timer shows elapsed time."""
        import time

        with timer("Timed operation"):
            time.sleep(0.01)

        captured = capsys.readouterr()
        assert "seconds" in captured.out

    def test_timer_failure_without_debug(self) -> None:
        """Test timer with failed operation (no debug)."""
        import sys

        from typer import Exit

        # Access the module using sys.modules to avoid import shadowing
        cli_module = sys.modules["davinci_monet.cli.app"]
        original_debug = cli_module.DEBUG
        cli_module.DEBUG = False  # type: ignore[attr-defined]

        try:
            with pytest.raises(Exit):
                with timer("Failing operation"):
                    raise ValueError("Test error")
        finally:
            cli_module.DEBUG = original_debug  # type: ignore[attr-defined]

    def test_timer_failure_with_debug(self) -> None:
        """Test timer with failed operation (debug mode)."""
        import sys

        # Access the module using sys.modules to avoid import shadowing
        cli_module = sys.modules["davinci_monet.cli.app"]
        original_debug = cli_module.DEBUG
        cli_module.DEBUG = True  # type: ignore[attr-defined]

        try:
            with pytest.raises(ValueError, match="Test error"):
                with timer("Failing operation"):
                    raise ValueError("Test error")
        finally:
            cli_module.DEBUG = original_debug  # type: ignore[attr-defined]


# =============================================================================
# Test Run Command
# =============================================================================


class TestRunCommand:
    """Tests for the run command."""

    def test_run_missing_file(self) -> None:
        """Test run with non-existent file."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["run", "nonexistent.yaml"])

        assert result.exit_code != 0
        assert "does not exist" in result.stdout

    def test_run_with_debug_flag(self) -> None:
        """Test run command parses debug flag."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--debug", "nonexistent.yaml"])

        assert result.exit_code != 0
        # Debug flag parsed, but file doesn't exist

    @pytest.fixture
    def sample_config(self, tmp_path: Path) -> Path:
        """Create a sample configuration file."""
        config_content = f"""
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02
  output_dir: "{tmp_path / 'output'}"

sources:
  test_dataset:
    type: cmaq
    files: test.nc
  test_geometry:
    type: airnow
    filename: geometry.nc

pairs:
  test_dataset_test_geometry:
    x: {{ source: test_geometry, variable: o3 }}
    y: {{ source: test_dataset, variable: O3 }}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_run_valid_config_parses(self, sample_config: Path) -> None:
        """Test run with valid config file parses correctly."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # This will fail when trying to open dataset files, but should parse config
        result = runner.invoke(app, ["run", str(sample_config)])

        # Config parses but pipeline fails on missing data files
        assert result.exit_code != 0

    def test_run_invokes_pipeline_runner_with_plot_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test run command forwards CLI options into the pipeline runner."""
        from typer.testing import CliRunner

        import davinci_monet.pipeline.runner as runner_module

        config_file = tmp_path / "config.yaml"
        config_file.write_text("analysis: {}\nsources: {}\n")
        captured: dict[str, object] = {}

        def fake_pipeline_run(config_path: str, **kwargs: object) -> SimpleNamespace:
            captured["config_path"] = config_path
            captured.update(kwargs)
            return SimpleNamespace(
                success=True,
                total_duration_seconds=1.25,
                completed_stages=["load_sources", "plotting"],
            )

        monkeypatch.setattr(runner_module, "run_analysis", fake_pipeline_run)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run",
                "--show-plots",
                "--preview-format",
                "png",
                str(config_file),
            ],
        )

        assert result.exit_code == 0
        assert captured == {
            "config_path": str(config_file),
            "show_progress": True,
            "show_plots": True,
            "preview_format": "png",
        }
        assert "Analysis complete" in result.stdout


# =============================================================================
# Test Validate Command
# =============================================================================


class TestCLIValidate:
    """Tests for the validate command."""

    def test_validate_missing_file(self) -> None:
        """Test validate with non-existent file."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["validate", "nonexistent.yaml"])

        assert result.exit_code != 0
        assert "does not exist" in result.stdout

    @pytest.fixture
    def valid_config(self, tmp_path: Path) -> Path:
        """Create a valid minimal configuration file (unified sources schema)."""
        (tmp_path / "test.nc").touch()
        (tmp_path / "geometry.nc").touch()
        config_content = """
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02

sources:
  test_dataset:
    type: cmaq
    files: test.nc
  test_geometry:
    type: airnow
    filename: geometry.nc

pairs:
  test_dataset_test_geometry:
    x: { source: test_geometry, variable: o3 }
    y: { source: test_dataset, variable: O3 }
"""
        config_file = tmp_path / "valid_config.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_validate_valid_config(self, valid_config: Path) -> None:
        """Test validate with valid config file."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(valid_config)])

        assert "Validation passed" in result.stdout

    def test_validate_show_config(self, valid_config: Path) -> None:
        """Test validate with --show flag."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["validate", "--show", str(valid_config)])

        assert "Parsed configuration" in result.stdout

    def test_validate_rejects_core_extra_field_by_default(self, tmp_path: Path) -> None:
        """Core extra fields are rejected by default."""
        # Config with extra field
        config_content = """
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02
  extra_field: should_be_ignored

sources:
  test_dataset:
    type: cmaq
    files: test.nc
"""
        config_file = tmp_path / "config_extra.yaml"
        config_file.write_text(config_content)

        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(app, ["validate", str(config_file)])
        assert result.exit_code != 0
        assert "analysis.extra_field" in result.stdout

    def test_validate_strict_mode_rejects_core_extra_field(self, tmp_path: Path) -> None:
        """The --strict flag changes validation behavior."""
        config_file = tmp_path / "config_extra.yaml"
        config_file.write_text("""
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02
  extra_field: should_fail_in_strict_mode

sources:
  test_dataset:
    type: generic
""")

        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["validate", "--strict", str(config_file)])

        assert result.exit_code != 0
        assert "analysis.extra_field" in result.stdout

    @pytest.fixture
    def invalid_config(self, tmp_path: Path) -> Path:
        """Create an invalid configuration file."""
        config_content = """
analysis:
  start_time: not-a-date
"""
        config_file = tmp_path / "invalid_config.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_validate_invalid_config(self, invalid_config: Path) -> None:
        """Test validate with invalid config file."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(invalid_config)])

        assert result.exit_code != 0

    def test_validate_show_config_uses_strict_loader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test validate command forwards strict mode and prints parsed config."""
        from typer.testing import CliRunner

        import davinci_monet.config as config_module
        from davinci_monet.config.parser import validate_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("analysis: {}\nsources: {}\n")
        captured: dict[str, object] = {}
        parsed_config = validate_config(
            {
                "analysis": {"start_time": "2024-01-01", "end_time": "2024-01-02"},
                "sources": {"mock_source": {"type": "generic"}},
            }
        )

        def fake_load_config(path: Path, strict: bool = False) -> object:
            captured["path"] = path
            captured["strict"] = strict
            return parsed_config

        monkeypatch.setattr(config_module, "load_config", fake_load_config)

        runner = CliRunner()
        result = runner.invoke(app, ["validate", "--strict", "--show", str(config_file)])

        assert result.exit_code == 0
        assert captured == {"path": config_file, "strict": True}
        assert "Mode: strict" in result.stdout
        assert "Parsed configuration" in result.stdout
        assert '"mock_source"' in result.stdout


# =============================================================================
# Test Get Data Commands
# =============================================================================


class TestGetDataCommands:
    """Tests for the data download commands."""

    def test_get_help(self) -> None:
        """Test get subcommand help."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["get", "--help"])

        assert result.exit_code == 0
        assert "aeronet" in result.stdout.lower() or "Download" in result.stdout

    def test_get_aeronet_help(self) -> None:
        """Test get aeronet help."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["get", "aeronet", "--help"])

        assert result.exit_code == 0
        assert "start-date" in result.stdout or "AERONET" in result.stdout

    def test_get_airnow_help(self) -> None:
        """Test get airnow help."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["get", "airnow", "--help"])

        assert result.exit_code == 0
        assert "start-date" in result.stdout or "AirNow" in result.stdout

    def test_airnow_dataframe_to_xarray_produces_current_shape(self) -> None:
        """`_dataframe_to_xarray` must produce the current (time, y=1, x) layout
        with units attrs and `time_local`, so AirNow files written by
        `davinci-monet get airnow` remain readable by existing pipelines."""
        import numpy as np
        import pandas as pd

        from davinci_monet.datasets.surface.airnow import _dataframe_to_xarray

        times = pd.date_range("2025-08-01 00:00", periods=3, freq="h")
        rows = []
        for t in times:
            for siteid, site, lat, lon, uo in (
                ("A1", "Site A", 40.0, -105.0, -7),
                ("B2", "Site B", 39.0, -104.0, -7),
            ):
                rows.append(
                    {
                        "time": t,
                        "siteid": siteid,
                        "site": site,
                        "utcoffset": uo,
                        "latitude": lat,
                        "longitude": lon,
                        "cmsa_name": "",
                        "msa_code": "",
                        "msa_name": "",
                        "state_name": "CO",
                        "epa_region": "R8",
                        "OZONE": 30.0 + lat / 10,
                        "OZONE_unit": "ppb",
                        "PM2.5": 5.0,
                        "PM2.5_unit": "ug/m3",
                    }
                )
        df = pd.DataFrame(rows)

        ds = _dataframe_to_xarray(df, daily=False)

        assert ds.sizes["time"] == 3
        assert ds.sizes["y"] == 1
        assert ds.sizes["x"] == 2
        assert ds["OZONE"].dims == ("time", "y", "x")
        assert "OZONE" in ds.data_vars
        assert "PM2.5" in ds.data_vars
        assert ds["OZONE"].attrs.get("units") == "ppb"
        assert ds["PM2.5"].attrs.get("units") == "ug/m3"
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "time_local" in ds.data_vars
        assert ds["time_local"].dtype == np.dtype("datetime64[ns]")

    def test_get_airnow_invokes_download_conversion_and_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get airnow orchestrates download, conversion, and output writing."""
        import sys
        from types import ModuleType

        from typer.testing import CliRunner

        import davinci_monet.cli.commands.get_data as get_data_module
        import davinci_monet.datasets.surface.airnow as airnow_module

        fake_monetio = ModuleType("monetio")
        calls: dict[str, object] = {}

        def fake_add_data(
            dates: object,
            *,
            download: bool,
            wide_fmt: bool,
            n_procs: int,
            daily: bool,
        ) -> object:
            calls["dates"] = dates
            calls["download"] = download
            calls["wide_fmt"] = wide_fmt
            calls["n_procs"] = n_procs
            calls["daily_download"] = daily
            return object()

        fake_monetio.airnow = SimpleNamespace(add_data=fake_add_data)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "monetio", fake_monetio)

        converted_dataset = object()

        def fake_dataframe_to_xarray(dataframe: object, *, daily: bool) -> object:
            calls["dataframe"] = dataframe
            calls["daily_convert"] = daily
            return converted_dataset

        def fake_write_dataset(ds: object, output_path: Path, *, compress: bool) -> bool:
            calls["written_dataset"] = ds
            calls["output_path"] = output_path
            calls["compress"] = compress
            return True

        monkeypatch.setattr(airnow_module, "_dataframe_to_xarray", fake_dataframe_to_xarray)
        monkeypatch.setattr(get_data_module, "_write_dataset_safe", fake_write_dataset)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "get",
                "airnow",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-01",
                "--dst",
                str(tmp_path),
                "-o",
                "airnow.nc",
                "--num-workers",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert calls["download"] is False
        assert calls["wide_fmt"] is True
        assert calls["n_procs"] == 2
        assert calls["daily_download"] is False
        assert calls["daily_convert"] is False
        assert calls["written_dataset"] is converted_dataset
        assert calls["output_path"] == tmp_path / "airnow.nc"
        assert calls["compress"] is True
        assert "Output written to" in result.stdout

    def test_get_aqs_help(self) -> None:
        """Test get aqs help."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["get", "aqs", "--help"])

        assert result.exit_code == 0
        assert "start-date" in result.stdout or "AQS" in result.stdout

    def test_get_openaq_help(self) -> None:
        """Test get openaq help."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["get", "openaq", "--help"])

        assert result.exit_code == 0
        assert "start-date" in result.stdout or "OpenAQ" in result.stdout

    def test_get_aeronet_missing_dates(self) -> None:
        """Test get aeronet without required dates."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["get", "aeronet"])

        # Should fail due to missing required options
        assert result.exit_code != 0

    def test_get_openaq_no_sensors(self) -> None:
        """Test get openaq with no sensor types."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "get",
                "openaq",
                "-s",
                "2024-01-01",
                "-e",
                "2024-01-02",
                "--no-geometry-grade",
                "--no-low-cost",
            ],
        )

        # Should fail due to no sensor types
        assert result.exit_code != 0 or "no sensor types" in result.stdout.lower()


# =============================================================================
# Test Parse Output Path
# =============================================================================


class TestParseOutputPath:
    """Tests for output path parsing utility."""

    def test_default_output_path(self) -> None:
        """Test default output path generation."""
        from davinci_monet.cli.commands.get_data import _parse_output_path

        dst, out_name = _parse_output_path(
            out_name=None,
            dst=Path("."),
            default_prefix="AERONET",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        assert dst == Path(".")
        assert "AERONET" in out_name
        assert "20240101" in out_name
        assert "20240131" in out_name
        assert out_name.endswith(".nc")

    def test_custom_output_name(self) -> None:
        """Test custom output name."""
        from davinci_monet.cli.commands.get_data import _parse_output_path

        dst, out_name = _parse_output_path(
            out_name="custom.nc",
            dst=Path("/tmp"),
            default_prefix="TEST",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        assert dst == Path("/tmp")
        assert out_name == "custom.nc"

    def test_output_path_with_directory(self) -> None:
        """Test output name with directory path."""
        from davinci_monet.cli.commands.get_data import _parse_output_path

        dst, out_name = _parse_output_path(
            out_name="/data/output/custom.nc",
            dst=Path("."),
            default_prefix="TEST",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        assert dst == Path("/data/output")
        assert out_name == "custom.nc"


# =============================================================================
# Test CLI Module Imports
# =============================================================================


class TestCLIImports:
    """Tests for CLI module imports."""

    def test_import_cli_module(self) -> None:
        """Test importing the CLI module."""
        from davinci_monet import cli

        assert hasattr(cli, "app")
        assert hasattr(cli, "cli")

    def test_import_app_from_cli(self) -> None:
        """Test importing app from CLI module."""
        from davinci_monet.cli import app

        assert app is not None

    def test_import_timer_from_cli(self) -> None:
        """Test importing timer from CLI module."""
        from davinci_monet.cli import timer

        assert callable(timer)

    def test_import_colors_from_cli(self) -> None:
        """Test importing colors from CLI module."""
        from davinci_monet.cli import ERROR_COLOR, INFO_COLOR, SUCCESS_COLOR, WARNING_COLOR

        assert INFO_COLOR is not None
        assert ERROR_COLOR is not None
        assert SUCCESS_COLOR is not None
        assert WARNING_COLOR is not None


# =============================================================================
# CLI Command Tests
# =============================================================================


class TestCLICommands:
    """CLI command tests (calls internal APIs directly, does not use PipelineRunner.run_from_config)."""

    @pytest.fixture
    def complete_config(self, tmp_path: Path) -> Path:
        """Create a complete configuration file."""
        (tmp_path / "dataset.nc").touch()
        (tmp_path / "geometry.nc").touch()
        config_content = """
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02
  output_dir: output

sources:
  dataset1:
    type: cmaq
    files: dataset.nc
  geometry1:
    type: airnow
    filename: geometry.nc
    variables:
      o3: {}
      pm25: {}

pairs:
  dataset1_geometry1:
    x: { source: geometry1, variable: o3 }
    y: { source: dataset1, variable: O3 }

plots:
  timeseries_plot:
    type: timeseries
    domain_type:
      - all
    domain_name:
      - CONUS
    pairs:
      - dataset1_geometry1

stats:
  output_table: true
  metrics:
    - MB
    - RMSE
    - R
"""
        config_file = tmp_path / "complete.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_validate_complete_config(self, complete_config: Path) -> None:
        """Test validating a complete config."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(complete_config)])

        # Should pass validation
        assert "Validation passed" in result.stdout

    def test_cli_multiple_commands(self) -> None:
        """Test that multiple commands are available."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Check main help lists commands
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Commands should be mentioned
        output_lower = result.stdout.lower()
        assert "run" in output_lower or "validate" in output_lower or "get" in output_lower
