from __future__ import annotations

import json

import numpy as np
import xarray as xr

from davinci_monet.analysis.artifacts import write_product_artifacts


def test_write_product_artifacts_writes_netcdf_and_summary(tmp_path) -> None:
    ds = xr.Dataset(
        {"aod": (("group", "lat", "lon"), np.array([[[0.1, np.nan], [0.2, 0.3]]]))},
        coords={"group": ["2008-07-01"], "lat": [-1.0, 1.0], "lon": [0.0, 90.0]},
        attrs={"analysis_type": "gridded_analysis"},
    )
    result = write_product_artifacts(tmp_path, "daily_aod", ds)
    assert result.analysis_path == tmp_path / "products" / "daily_aod" / "analysis.nc"
    assert result.summary_path == tmp_path / "products" / "daily_aod" / "summary.json"
    assert result.analysis_path.exists()
    summary = json.loads(result.summary_path.read_text())
    assert summary["product"] == "daily_aod"
    assert summary["fields"]["aod"]["finite_count"] == 3


def test_write_product_artifacts_serializes_boolean_variable_and_coord_attrs(tmp_path) -> None:
    ds = xr.Dataset(
        {"aod": (("group", "lat"), np.array([[0.1, 0.2]]))},
        coords={"group": ["2008-07-01"], "lat": [-1.0, 1.0]},
        attrs={"derived": True},
    )
    ds["aod"].attrs["screened"] = True
    ds["lat"].attrs["edge"] = False

    result = write_product_artifacts(tmp_path, "daily_aod", ds)

    with xr.open_dataset(result.analysis_path) as artifact:
        assert artifact.attrs["derived"] == "True"
        assert artifact["aod"].attrs["screened"] == "True"
        assert artifact["lat"].attrs["edge"] == "False"
