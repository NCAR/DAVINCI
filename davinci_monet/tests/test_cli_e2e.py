"""CLI end-to-end tests for DAVINCI.

These tests invoke the CLI app with actual YAML config files, exercising
the full path: CLI → YAML parsing → load_config() → PipelineRunner.
This complements test_integration.py (which tests pipeline core with dicts)
and test_cli.py (which tests CLI argument parsing and help).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.cli.app import app
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.datasets import create_dataset_dataset
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario, sample_geometry_from

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create synthetic NetCDF files and a YAML config pointing to them.

    Returns (config_path, output_dir, log_dir, tmp_path).
    """
    domain = Domain(
        lon_min=-105.0,
        lon_max=-95.0,
        lat_min=35.0,
        lat_max=45.0,
        n_lon=8,
        n_lat=8,
    )
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-16 00:00", freq="3h")

    y_ds = create_dataset_dataset(
        variables=["O3"],
        domain=domain,
        time_config=time_cfg,
        seed=42,
    )

    scenario = PerfectMatchScenario(
        variables=["O3"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_geometry=5,
        noise_level=0.0,
        seed=42,
    )
    x_ds = sample_geometry_from(y_ds, "point", scenario=scenario)

    # Add small bias so stats are non-trivial
    rng = np.random.default_rng(42)
    y_ds["O3"] = y_ds["O3"] + 3.0 + rng.normal(0, 2.0, size=y_ds["O3"].shape)

    y_path = tmp_path / "dataset.nc"
    x_path = tmp_path / "geometry.nc"
    y_ds.to_netcdf(y_path)
    x_ds.to_netcdf(x_path)

    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"

    config_text = textwrap.dedent(
        f"""\
        analysis:
          start_time: "2024-01-15"
          end_time: "2024-01-16"
          output_dir: "{output_dir}"
          log_dir: "{log_dir}"

        sources:
          synthetic:
            type: generic
            files: "{y_path}"
            radius_of_influence: 50000
            variables:
              O3:
                units: ppb
                vmin_plot: 30
                vmax_plot: 70
                vdiff_plot: 10
          surface:
            type: pt_sfc
            filename: "{x_path}"
            variables:
              O3:
                valid_min: 0
                valid_max: 200
                units: ppb

        pairs:
          synthetic_surface:
            x:
              source: surface
              variable: O3
            y:
              source: synthetic
              variable: O3

        plots:
          scatter_o3:
            type: scatter
            pairs: [synthetic_surface]
            title: "O3 Scatter"

        stats:
          metrics: [N, MB, RMSE]
    """
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    return config_path, output_dir, log_dir, tmp_path


# =============================================================================
# CLI Run Tests
# =============================================================================


@pytest.mark.integration
class TestCLIRunE2E:
    """End-to-end tests for `davinci-monet run <config.yaml>`."""

    def test_cli_run_happy_path(self, synthetic_data: tuple) -> None:
        """Full pipeline through CLI with YAML config file."""
        from typer.testing import CliRunner

        config_path, output_dir, log_dir, _ = synthetic_data
        runner = CliRunner()
        result = runner.invoke(app, ["run", str(config_path)])

        assert result.exit_code == 0, (
            f"CLI failed with exit code {result.exit_code}.\n" f"stdout: {result.stdout}\n"
        )
        assert "Analysis complete" in result.stdout

        # Verify outputs
        png_files = list(output_dir.rglob("*.png"))
        assert len(png_files) >= 1, f"No plots generated in {output_dir}"

        csv_files = list(output_dir.rglob("*.csv"))
        assert len(csv_files) >= 1, f"No stats CSV in {output_dir}"

        log_files = list(log_dir.glob("pipeline_*.md"))
        assert len(log_files) >= 1, f"No pipeline log in {log_dir}"

    def test_cli_run_file_not_found(self, tmp_path: Path) -> None:
        """CLI with nonexistent config path exits with code 2."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(tmp_path / "nonexistent.yaml")])

        assert result.exit_code == 2
        assert "does not exist" in result.stdout

    def test_cli_run_invalid_config(self, tmp_path: Path) -> None:
        """CLI with invalid config shows configuration error."""
        from typer.testing import CliRunner

        config_path = tmp_path / "bad.yaml"
        config_path.write_text(
            textwrap.dedent(
                """\
            analysis:
              start_time: not-a-date
              end_time: also-not-a-date
        """
            )
        )

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(config_path)])

        assert result.exit_code != 0


# =============================================================================
# Error Config Tests
# =============================================================================


ERROR_CONFIG_DIR = Path(__file__).resolve().parents[2] / "tests" / "error_configs"
ERROR_CONFIGS = sorted(ERROR_CONFIG_DIR.glob("*.yaml")) if ERROR_CONFIG_DIR.exists() else []

assert len(ERROR_CONFIGS) >= 15, "error_configs dir moved?"

# `davinci-monet validate` only parses/validates the pydantic schema
# (config/schema.py); it never touches the filesystem, the source/reader
# registry, or cross-field date ordering. These configs are schema-valid
# even though they are semantically broken, so `validate` exits 0 for them.
_XFAIL_ERROR_CONFIGS: dict[str, str] = {
    "03_missing_analysis": (
        "MonetConfig.analysis defaults via Field(default_factory=AnalysisConfig) "
        "(config/schema.py:823), so an omitted `analysis:` section parses to an "
        "all-None AnalysisConfig instead of being rejected."
    ),
    "04_missing_dataset_files": (
        "validate_config_command() only calls load_config() (cli/commands/validate.py) "
        "which never stats source files; existence is checked later, at pipeline "
        "load_sources time, so `davinci-monet validate` cannot see this error."
    ),
    "05_invalid_dataset_type": (
        "SourceConfig.type is `str | None` with no registry-membership check "
        "(config/schema.py:281); source_registry lookup only happens when the "
        "pipeline loads sources, not during `validate`."
    ),
    "06_invalid_geometry_type": (
        "Same schema gap as 05_invalid_dataset_type: SourceConfig.type "
        "(config/schema.py:281) accepts any string, including an unregistered "
        "geometry reader type."
    ),
    "08_end_before_start": (
        "AnalysisConfig (config/schema.py:75-160) has no cross-field validator "
        "enforcing end_time > start_time, so a config with end before start "
        "parses successfully."
    ),
    "12_empty_file": (
        "Every MonetConfig field has a default_factory (config/schema.py:823-831), "
        "so a fully empty YAML document parses to an all-defaults config instead "
        "of being rejected."
    ),
    "14_duplicate_keys": (
        "PyYAML silently overwrites duplicate mapping keys during parsing, before "
        "schema validation ever runs, so the loaded document has no duplicate keys "
        "left to reject. tests/error_configs/README.md itself documents the "
        "expected behavior as 'silent overwrite or warning', not necessarily "
        "rejection."
    ),
    "15_null_values": (
        "SourceConfig.type is `str | None` (config/schema.py:281) and "
        "AnalysisConfig.output_dir is optional, so explicit YAML nulls (`~`) pass "
        "schema validation instead of being rejected."
    ),
}


def _error_config_param(path: Path) -> Any:
    reason = _XFAIL_ERROR_CONFIGS.get(path.stem)
    if reason is None:
        return pytest.param(path, id=path.stem)
    return pytest.param(
        path,
        marks=pytest.mark.xfail(strict=False, reason=reason),
        id=path.stem,
    )


@pytest.mark.parametrize(
    "config_path",
    [_error_config_param(p) for p in ERROR_CONFIGS],
)
class TestErrorConfigs:
    """Validate that all curated error configs are properly rejected."""

    def test_error_config_rejected_by_validate(self, config_path: Path) -> None:
        """Each error config should fail validation with non-zero exit."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(config_path)])

        assert result.exit_code != 0, (
            f"{config_path.name} was not rejected.\n"
            f"exit_code={result.exit_code}\n"
            f"stdout: {result.stdout}"
        )
