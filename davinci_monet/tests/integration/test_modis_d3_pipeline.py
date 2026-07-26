"""Pipeline contract for staged daily MODIS D3 AOD."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner

_D3_AOD = "AOD_550_Dark_Target_Deep_Blue_Combined_Mean"
_D3_AOD_STDDEV = "AOD_550_Dark_Target_Deep_Blue_Combined_Standard_Deviation"


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


def _write_d3_hdf4_day(path: Path, token: str, value: float, stddev: float) -> None:
    hdf_api = pytest.importorskip("pyhdf.SD", reason="pyhdf required for MODIS D3 HDF4 tests")
    SD, SDC = hdf_api.SD, hdf_api.SDC
    hdf = SD(str(path / f"MYD08_D3.A{token}.061.synthetic.hdf"), SDC.WRITE | SDC.CREATE)
    try:
        lon = hdf.create("XDim", SDC.FLOAT32, 3)
        lon.dim(0).setname("XDim:mod08")
        lon[:] = np.array([-1.5, -0.5, 0.5], dtype=np.float32)
        lon.endaccess()
        lat = hdf.create("YDim", SDC.FLOAT32, 2)
        lat.dim(0).setname("YDim:mod08")
        lat[:] = np.array([0.5, -0.5], dtype=np.float32)
        lat.endaccess()

        for name, physical in ((_D3_AOD, value), (_D3_AOD_STDDEV, stddev)):
            field = hdf.create(name, SDC.INT16, (2, 3))
            field.dim(0).setname("YDim:mod08")
            field.dim(1).setname("XDim:mod08")
            raw = np.full((2, 3), round(physical * 1000), dtype=np.int16)
            raw[0, 2] = -9999
            field[:] = raw
            field.attr("_FillValue").set(SDC.INT16, -9999)
            field.attr("valid_range").set(SDC.INT16, [0, 5000])
            field.attr("scale_factor").set(SDC.FLOAT32, 0.001)
            field.attr("add_offset").set(SDC.FLOAT32, 0.0)
            field.attr("Masked_With_QA_Usefulness_Flag").set(SDC.CHAR, "True")
            field.endaccess()
    finally:
        hdf.end()


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


@pytest.mark.integration
def test_hdf4_modis_d3_round_trips_through_required_checkpoints(tmp_path: Path) -> None:
    _write_d3_hdf4_day(tmp_path, "2008183", 0.2, 0.04)
    _write_d3_hdf4_day(tmp_path, "2008184", 0.3, 0.05)
    attempt_root = tmp_path / "a001"
    config = {
        "run": {"id": "modis-d3-checkpoint-regression", "kind": "smoke"},
        "execution": {
            "attempt_root": str(attempt_root),
            "checkpoints": {
                "mode": "required",
                "granularity": "item",
                "loaded_sources": True,
                "retain": "all",
            },
        },
        "analysis": {
            "start_time": "2008-07-01 00:00:00",
            "end_time": "2008-07-02 23:59:59",
            "output_dir": str(attempt_root / "output"),
            "log_dir": str(attempt_root / "logs"),
        },
        "sources": {
            "modis_aqua_raw": {
                "type": "modis_viirs",
                "product": "MYD08_D3",
                "files": str(tmp_path / "MYD08_D3*.hdf"),
                "variables": {
                    "aod_550nm": {"units": "1"},
                    "aod_550nm_stddev": {"units": "1"},
                },
            }
        },
        "analyses": {
            "modis_aqua_daily": {
                "type": "aod_preprocess",
                "source": "modis_aqua_raw",
                "variable": "aod_550nm",
                "day_anchor_hour": 12.0,
                "log_epsilon": 0.01,
                "uncertainty_model": {
                    "type": "linear_aod_rss",
                    "name": "test-myd08-d3-error",
                    "source_variable": "aod_550nm_stddev",
                    "absolute_floor": 0.05,
                    "relative_fraction": 0.15,
                    "combination": "root_sum_square",
                    "transform": "delta_method",
                    "covariance": "independent",
                },
                "required": True,
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, result.stage_errors
    assert result.context is not None
    daily = result.context.sources["modis_aqua_daily"].data
    np.testing.assert_allclose(daily["aod"].mean(("lat", "lon")), [0.2, 0.3])
    assert int(daily["valid"].sum()) == 10
    assert int(daily["obs_error_std"].notnull().sum()) == 10

    expected = {
        "load_sources/items/modis_aqua_raw": ("aod_550nm", [0.2, 0.3]),
        "analyses/items/modis_aqua_daily": ("aod", [0.2, 0.3]),
    }
    for receipt_name, (variable, values) in expected.items():
        receipt_path = attempt_root / "checkpoints" / receipt_name / "r001.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "finalized"
        netcdf_paths = [
            Path(path) for path in receipt["objects"][0]["paths"] if str(path).endswith(".nc")
        ]
        assert len(netcdf_paths) == 1
        with xr.open_dataset(netcdf_paths[0]) as restored:
            np.testing.assert_allclose(restored[variable].mean(("lat", "lon")), values)
            for name in restored.data_vars:
                assert "scale_factor" not in restored[name].attrs
                assert "add_offset" not in restored[name].attrs
