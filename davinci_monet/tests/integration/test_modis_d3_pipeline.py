"""Pipeline contract for staged daily MODIS D3 AOD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner

_D3_AOD = "AOD_550_Dark_Target_Deep_Blue_Combined_Mean"


def _write_d3_day(path: Path, day: str, token: str, value: float) -> None:
    ds = xr.Dataset(
        {
            _D3_AOD: (
                ("time", "lat", "lon"),
                np.full((1, 2, 3), value, dtype=np.float32),
            )
        },
        coords={
            "time": np.array([f"{day}T12:00:00"], dtype="datetime64[ns]"),
            "lat": [-0.5, 0.5],
            "lon": [-1.5, -0.5, 0.5],
        },
        attrs={"source_product": "MYD08_D3"},
    )
    ds[_D3_AOD].attrs.update(units="1", valid_min=0.0, valid_max=5.0)
    ds.to_netcdf(path / f"MYD08_D3.A{token}.061.synthetic.AOD550.nc4")


@pytest.mark.integration
def test_staged_modis_d3_flows_through_required_aod_preprocess(tmp_path: Path) -> None:
    _write_d3_day(tmp_path, "2008-07-01", "2008183", 0.2)
    _write_d3_day(tmp_path, "2008-07-02", "2008184", 0.3)
    config = {
        "analysis": {
            "start_time": "2008-07-01 00:00:00",
            "end_time": "2008-07-02 23:59:59",
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "modis_aqua_raw": {
                "type": "modis_viirs",
                "product": "MYD08_D3",
                "files": str(tmp_path / "MYD08_D3*.nc4"),
                "variables": {"aod_550nm": {"units": "1"}},
            }
        },
        "analyses": {
            "modis_aqua_daily": {
                "type": "aod_preprocess",
                "source": "modis_aqua_raw",
                "variable": "aod_550nm",
                "day_anchor_hour": 12.0,
                "log_epsilon": 0.01,
                "required": True,
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, result.stage_errors
    assert result.context is not None
    assert result.context.metadata["analysis_status"] == {"modis_aqua_daily": "completed"}
    daily = result.context.sources["modis_aqua_daily"].data
    np.testing.assert_array_equal(
        daily["time"].values,
        np.array(["2008-07-01T12:00:00", "2008-07-02T12:00:00"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(daily["aod"].mean(("lat", "lon")), [0.2, 0.3])
    assert bool(daily["valid"].all())
