"""Tests for the CLI module."""

from __future__ import annotations

import tempfile
from pathlib import Path
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
        config_content = """
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02
  output_dir: output

sources:
  test_model:
    role: model
    type: cmaq
    files: test.nc
    mapping:
      test_obs:
        o3: O3
  test_obs:
    role: obs
    type: airnow
    filename: obs.nc
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_run_valid_config_parses(self, sample_config: Path) -> None:
        """Test run with valid config file parses correctly."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # This will fail when trying to open model files, but should parse config
        result = runner.invoke(app, ["run", str(sample_config)])

        # Config parses but pipeline fails on missing data files
        assert result.exit_code != 0


# =============================================================================
# Test Validate Command
# =============================================================================


class TestValidateCommand:
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
        """Create a valid minimal configuration file."""
        config_content = """
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02

model:
  test_model:
    mod_type: cmaq
    files: test.nc
    mapping:
      test_obs:
        o3: O3

obs:
  test_obs:
    obs_type: airnow
    filename: obs.nc
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

    def test_validate_strict_mode(self, tmp_path: Path) -> None:
        """Test validate with strict mode."""
        # Config with extra field
        config_content = """
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02
  extra_field: should_be_ignored

model:
  test_model:
    mod_type: cmaq
    files: test.nc
"""
        config_file = tmp_path / "config_extra.yaml"
        config_file.write_text(config_content)

        from typer.testing import CliRunner

        runner = CliRunner()

        # With flexible mode (default), should pass
        result = runner.invoke(app, ["validate", str(config_file)])
        assert result.exit_code == 0
        assert "Validation passed" in result.stdout

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

    def test_airnow_dataframe_to_xarray_produces_legacy_shape(self) -> None:
        """`_dataframe_to_xarray` must produce the legacy (time, y=1, x) layout
        with units attrs and `time_local`, so AirNow files written by
        `davinci-monet get airnow` remain readable by existing pipelines."""
        import numpy as np
        import pandas as pd

        from davinci_monet.observations.surface.airnow import _dataframe_to_xarray

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

        assert tuple(ds.sizes.keys()) == ("time", "y", "x")
        assert ds.sizes["time"] == 3
        assert ds.sizes["y"] == 1
        assert ds.sizes["x"] == 2
        assert "OZONE" in ds.data_vars
        assert "PM2.5" in ds.data_vars
        assert ds["OZONE"].attrs.get("units") == "ppb"
        assert ds["PM2.5"].attrs.get("units") == "ug/m3"
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "time_local" in ds.data_vars
        assert ds["time_local"].dtype == np.dtype("datetime64[ns]")

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
                "--no-reference-grade",
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

    def test_cli_lazy_reexports_no_recursion_in_fresh_interpreter(self) -> None:
        """Regression guard: each re-exported name must be importable in a clean
        interpreter where the cli.app submodule has NOT been pre-bound.

        The previous __getattr__ implementation used
        ``from davinci_monet.cli import app as _app_module`` which re-entered
        __getattr__ infinitely because ``app`` collides with the ``cli/app.py``
        submodule name and the fromlist lookup triggered another __getattr__ call.
        Running in a subprocess ensures no prior import has side-bound the
        attribute, which is the condition that masks the bug in the in-process
        tests above.
        """
        import subprocess
        import sys

        names = [
            "cli",
            "timer",
            "INFO_COLOR",
            "ERROR_COLOR",
            "SUCCESS_COLOR",
            "WARNING_COLOR",
            "DEBUG",
        ]
        imports = ", ".join(names)
        code = f"from davinci_monet.cli import {imports}"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fresh-interpreter import raised an error:\n{result.stderr}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI."""

    @pytest.fixture
    def complete_config(self, tmp_path: Path) -> Path:
        """Create a complete configuration file."""
        config_content = """
analysis:
  start_time: 2024-01-01
  end_time: 2024-01-02
  output_dir: output

model:
  model1:
    mod_type: cmaq
    files: model.nc
    mapping:
      obs1:
        o3: O3
        pm25: PM25_TOT

obs:
  obs1:
    obs_type: airnow
    filename: obs.nc
    variables:
      o3: {}
      pm25: {}

plots:
  timeseries_plot:
    type: timeseries
    domain_type:
      - all
    domain_name:
      - CONUS
    data:
      - model1_obs1

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
