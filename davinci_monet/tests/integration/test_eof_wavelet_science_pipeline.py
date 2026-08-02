"""Integration: finalized EOF/wavelet artifacts drive the scientific plot suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.tests.unit.plots.test_eof_wavelet_science import science_inputs


@pytest.mark.integration
def test_eof_wavelet_science_suite_runs_through_pipeline(tmp_path: Path) -> None:
    basis, projection, filtered = science_inputs()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    split = basis.sizes["time"] // 2
    for label, dataset in (
        ("basis", basis),
        ("projection", projection),
        ("filtered", filtered),
    ):
        dataset.isel(time=slice(0, split)).to_netcdf(inputs / f"{label}-00000.nc")
        dataset.isel(time=slice(split, None)).to_netcdf(inputs / f"{label}-00001.nc")

    output = tmp_path / "output"
    config = {
        "analysis": {
            "start_time": "2008-01-01",
            "end_time": "2008-12-31 23:59:59",
            "output_dir": str(output),
            "style": {"theme": "ncar", "context": "publication"},
        },
        "sources": {
            label: {
                "type": "generic",
                "files": str(inputs / f"{label}-*.nc"),
                "combine": "nested",
                "concat_dim": "time",
                "data_vars": "minimal",
                "coords": "minimal",
                "compat": "override",
                "join": "exact",
            }
            for label in ("basis", "projection", "filtered")
        },
        "plots": {
            "science": {
                "type": "eof_wavelet_science",
                "sources": [
                    {"source": "basis", "variable": "eofs"},
                    {"source": "projection", "variable": "pc"},
                    {"source": "filtered", "variable": "pc"},
                ],
                "modes": [1, 2, 3],
                "pc_modes": [1, 2],
                "wavelet_modes": [1, 2],
                "formats": ["png"],
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, getattr(result, "error", None)
    assert result.context is not None
    generated = result.context.results["plotting"].data["plots_generated"]
    assert len(generated) == 10
    for label in (
        "seasonal_aod",
        "seasonal_aod_departure",
        "seasonal_projection_bias",
        "eof_patterns",
        "variance_summary",
        "pc_comparison",
        "mode_quality",
        "spatial_wavelet_rms",
        "wavelet_mode1",
        "wavelet_mode2",
    ):
        assert any(f"science_{label}.png" in path for path in generated)
