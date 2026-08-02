"""Integration: the corrected-AOD suite is enforced across the full pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from davinci_monet.core.exceptions import PlottingError
from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.plots.renderers.aod_correction_science import (
    AODCorrectionSciencePlotter,
)
from davinci_monet.tests.unit.plots.test_aod_correction_science import correction_series


def _install_fake_pdftoppm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_pdftoppm = bin_dir / "pdftoppm"
    fake_pdftoppm.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[-1] + '.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')\n"
    )
    fake_pdftoppm.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}")


def _pipeline_config(tmp_path: Path, *, formats: list[str]) -> dict[str, Any]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    series = correction_series()
    datasets = {
        "merra2": series[0].dataset,
        "corrected_merra2": series[1].dataset,
        "modis_aqua": series[2].dataset,
    }
    for label, dataset in datasets.items():
        dataset.to_netcdf(inputs / f"{label}.nc")

    output = tmp_path / "output"
    return {
        "analysis": {
            "start_time": "2008-01-01",
            "end_time": "2008-12-31 23:59:59",
            "output_dir": str(output),
            "style": {"theme": "ncar", "context": "publication"},
        },
        "sources": {
            label: {"type": "generic", "files": str(inputs / f"{label}.nc")} for label in datasets
        },
        "plots": {
            "aod_correction_science": {
                "type": "aod_correction_science",
                "sources": [
                    {"source": "merra2", "variable": "aod"},
                    {"source": "corrected_merra2", "variable": "aod_target"},
                    {"source": "modis_aqua", "variable": "aod"},
                ],
                "model_source": "merra2",
                "corrected_source": "corrected_merra2",
                "observation_source": "modis_aqua",
                "max_scatter_points": 500,
                "formats": formats,
            }
        },
    }


@pytest.mark.integration
def test_aod_correction_science_suite_runs_through_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pdftoppm(tmp_path, monkeypatch)
    config = _pipeline_config(tmp_path, formats=["png", "pdf"])
    config["inspection"] = {
        "enabled": True,
        "required": True,
        "presets": ["aod_correction"],
        "preview_format": "png",
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, result.stage_errors
    assert result.context is not None
    plotting = result.context.results["plotting"].data
    assert len(plotting["plots_generated"]) == 16
    report = plotting["plot_protocol_reports"]["aod_correction_science"]
    assert report["passed"] is True
    assert report["protocol"] == "davinci-aod-correction-v2"
    inspection = result.context.results["inspection"].data
    assert inspection["passed"] is True
    protocol_check = next(
        check for check in inspection["checks"] if check["name"] == "aod_correction_protocol"
    )
    assert protocol_check["passed"] is True
    for label in report["figures"]:
        assert any(
            f"aod_correction_science_{label}.png" in path for path in plotting["plots_generated"]
        )


@pytest.mark.integration
def test_aod_protocol_violation_fails_plotting_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _pipeline_config(tmp_path, formats=["png"])

    def reject_protocol(self: Any, figures: Any) -> dict[str, Any]:
        del self, figures
        raise PlottingError("AOD correction plot protocol failed: injected violation")

    monkeypatch.setattr(
        AODCorrectionSciencePlotter,
        "validate_rendered_figures",
        reject_protocol,
    )

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success is False
    assert result.context is not None
    plotting = result.context.results["plotting"]
    assert plotting.status.name == "FAILED"
    assert any("injected violation" in warning for warning in plotting.metadata["warnings"])
