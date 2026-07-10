from __future__ import annotations

import json

import numpy as np
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner


def test_gridded_analysis_pipeline_writes_and_uses_product_artifact(tmp_path) -> None:
    source = tmp_path / "cam.nc"
    ds = xr.Dataset(
        {
            "AOD": (("time", "lat", "lon"), np.ones((2, 1, 2))),
            "MASK": (("time", "lat", "lon"), np.ones((2, 1, 2))),
        },
        coords={
            "time": np.array(["2008-07-01T00:00", "2008-07-01T03:00"], dtype="datetime64[ns]"),
            "lat": [0.0],
            "lon": [0.0, 90.0],
        },
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(source)
    config = {
        "analysis": {"output_dir": str(tmp_path / "run")},
        "sources": {
            "cam": {"type": "generic", "files": str(source), "variables": {"AOD": {}, "MASK": {}}}
        },
        "analyses": {
            "daily_aod": {
                "type": "gridded_analysis",
                "source": "cam",
                "groupby": "day",
                "roles": {"analysis": "AOD", "mask": "MASK"},
                "fields": {"analyzed_aod": {"formula": 'mean(analysis, dim="time")'}},
            }
        },
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success
    analysis_nc = tmp_path / "run" / "products" / "daily_aod" / "analysis.nc"
    summary_json = tmp_path / "run" / "products" / "daily_aod" / "summary.json"
    assert analysis_nc.exists()
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text())
    assert summary["fields"]["analyzed_aod"]["finite_count"] == 2
    assert summary["attrs"]["source_label"] == "daily_aod"
    assert result.context is not None
    source = result.context.sources["daily_aod"]
    assert source.config["artifact_path"] == str(analysis_nc)
    assert source.config["summary_path"] == str(summary_json)
    assert result.context.metadata["product_artifacts"]["daily_aod"] == {
        "artifact_path": str(analysis_nc),
        "summary_path": str(summary_json),
    }
    artifact_entry = result.context.metadata["analysis_artifacts"][0]
    assert artifact_entry["analysis"] == "daily_aod"
    assert artifact_entry["role"] == "product"
    assert len(artifact_entry["checksums"]["analysis_sha256"]) == 64
    assert source.data.attrs["source_label"] == "daily_aod"
    assert source.data.attrs["derived"] is True
    assert source.data["analyzed_aod"].dtype == np.dtype("float32")
    with xr.open_dataset(analysis_nc) as artifact_ds:
        assert artifact_ds.attrs["source_label"] == "daily_aod"
        assert artifact_ds.attrs["derived"] == "True"
        np.testing.assert_allclose(source.data["analyzed_aod"], artifact_ds["analyzed_aod"])
