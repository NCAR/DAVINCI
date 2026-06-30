from __future__ import annotations

import json

import numpy as np
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner


def test_gridded_aod_product_pipeline_creates_artifacts_plots_inspection_and_manifest(
    tmp_path,
) -> None:
    source = tmp_path / "cam.nc"
    ds = xr.Dataset(
        {
            "OBS": (("time", "lat", "lon"), np.ones((4, 2, 2)) * 0.2),
            "ANALYSIS": (("time", "lat", "lon"), np.ones((4, 2, 2)) * 0.3),
            "MASK": (("time", "lat", "lon"), np.ones((4, 2, 2))),
        },
        coords={
            "time": np.array(
                [
                    "2008-07-01T00:00",
                    "2008-07-01T03:00",
                    "2008-07-02T00:00",
                    "2008-07-02T03:00",
                ],
                dtype="datetime64[ns]",
            ),
            "lat": [-1.0, 1.0],
            "lon": [0.0, 90.0],
        },
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(source)
    config = {
        "analysis": {"output_dir": str(tmp_path / "run"), "domain": "global"},
        "sources": {
            "cam": {
                "type": "generic",
                "files": str(source),
                "variables": {"OBS": {}, "ANALYSIS": {}, "MASK": {}},
            }
        },
        "analyses": {
            "daily_aod": {
                "type": "gridded_analysis",
                "source": "cam",
                "groupby": "day",
                "roles": {
                    "observation": "OBS",
                    "analysis": "ANALYSIS",
                    "mask": "MASK",
                },
                "fields": {
                    "observation_aod": {"formula": 'mean(observation, dim="time")'},
                    "analyzed_aod": {"formula": 'mean(analysis, dim="time")'},
                    "analysis_minus_observation_aod": {
                        "formula": 'mean(analysis - observation, dim="time")'
                    },
                    "nudge_fraction": {"formula": 'mean(mask, dim="time")'},
                },
            }
        },
        "plot_suites": {
            "daily": {
                "preset": "gridded_aod_diagnostics",
                "source": "daily_aod",
                "output_subdir": "plots/daily",
            }
        },
        "inspection": {
            "enabled": True,
            "required": True,
            "presets": ["gridded_aod_diagnostics"],
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, getattr(result, "error", None)
    run = tmp_path / "run"
    assert (run / "products" / "daily_aod" / "analysis.nc").exists()
    assert (run / "products" / "daily_aod" / "summary.json").exists()
    assert (run / "manifest.json").exists()
    assert (run / "inspection" / "inspection.json").exists()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["plots"]
    assert len(manifest["plots"]) == 8
    assert all(path.endswith(".pdf") for path in manifest["plots"])
    assert any("2008-07-01" in path for path in manifest["plots"])
    assert any("2008-07-02" in path for path in manifest["plots"])
    previews = sorted((run / "inspection" / "previews").rglob("*.png"))
    assert len(previews) == len(manifest["plots"])
    assert len(manifest["inspection"]["inspection_previews"]) == len(manifest["plots"])
    product_pngs = [p for p in run.rglob("*.png") if "inspection/previews" not in str(p)]
    assert product_pngs == []
