import sys
import types
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import monetio.sat as monetio_sat

from davinci_monet.datasets.satellite.modis_l2_aod import MODISL2AODReader


def test_open_with_monetio_supplies_default_variable_dict(monkeypatch, tmp_path):
    captured = {}
    epoch_1993 = int(datetime(1993, 1, 1, tzinfo=timezone.utc).timestamp())

    def read_dataset(path, variable_dict, **kwargs):
        captured["path"] = path
        captured["variable_dict"] = variable_dict
        return xr.Dataset(
            {
                "AOD_550_Dark_Target_Deep_Blue_Combined": (
                    ("Cell_Along_Swath", "Cell_Across_Swath"),
                    np.array([[0.5]], np.float32),
                ),
                "AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag": (
                    ("Cell_Along_Swath", "Cell_Across_Swath"),
                    np.array([[3]], np.int16),
                ),
                "AOD_550_Dark_Target_Deep_Blue_Combined_Algorithm_Flag": (
                    ("Cell_Along_Swath", "Cell_Across_Swath"),
                    np.array([[1]], np.int16),
                ),
            },
            coords={
                "lat": (("Cell_Along_Swath", "Cell_Across_Swath"), [[10.0]]),
                "lon": (("Cell_Along_Swath", "Cell_Across_Swath"), [[20.0]]),
                "time": (
                    ("Cell_Along_Swath", "Cell_Across_Swath"),
                    [[float(epoch_1993 + 123.0)]],
                ),
            },
        )

    fake_module = types.SimpleNamespace(read_dataset=read_dataset)
    monkeypatch.setitem(sys.modules, "monetio.sat._modis_l2_mm", fake_module)
    monkeypatch.setattr(monetio_sat, "_modis_l2_mm", fake_module, raising=False)
    path = tmp_path / "MOD04_L2.A2020167.0100.061.nc"
    path.write_text("placeholder")

    ds = MODISL2AODReader()._open_with_monetio([path], variables=None)

    assert captured["path"] == str(path)
    assert set(captured["variable_dict"]) == {
        "AOD_550_Dark_Target_Deep_Blue_Combined",
        "AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag",
        "AOD_550_Dark_Target_Deep_Blue_Combined_Algorithm_Flag",
    }
    assert captured["variable_dict"]["AOD_550_Dark_Target_Deep_Blue_Combined"]["scale"] == 0.001
    assert "Scan_Start_Time" in ds
    assert float(ds["Scan_Start_Time"].values[0, 0]) == 123.0


def test_open_with_monetio_does_not_nan_mask_integer_flag_fields(monkeypatch, tmp_path):
    epoch_1993 = int(datetime(1993, 1, 1, tzinfo=timezone.utc).timestamp())

    def read_dataset(path, variable_dict, **kwargs):
        raw_values = {
            "AOD_550_Dark_Target_Deep_Blue_Combined": np.array([[-9999, 500]], np.int16),
            "AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag": np.array([[0, 3]], np.int16),
            "AOD_550_Dark_Target_Deep_Blue_Combined_Algorithm_Flag": np.array([[0, 1]], np.int16),
        }
        data_vars = {}
        for name, options in variable_dict.items():
            values = raw_values[name]
            if "scale" in options:
                values = options["scale"] * values
            if np.issubdtype(values.dtype, np.integer) and (
                "minimum" in options or "maximum" in options
            ):
                raise ValueError("cannot convert float NaN to integer")
            data_vars[name] = (("Cell_Along_Swath", "Cell_Across_Swath"), values)

        return xr.Dataset(
            data_vars,
            coords={
                "lat": (("Cell_Along_Swath", "Cell_Across_Swath"), [[10.0, 11.0]]),
                "lon": (("Cell_Along_Swath", "Cell_Across_Swath"), [[20.0, 21.0]]),
                "time": (
                    ("Cell_Along_Swath", "Cell_Across_Swath"),
                    [[float(epoch_1993 + 123.0), float(epoch_1993 + 124.0)]],
                ),
            },
        )

    fake_module = types.SimpleNamespace(read_dataset=read_dataset)
    monkeypatch.setitem(sys.modules, "monetio.sat._modis_l2_mm", fake_module)
    monkeypatch.setattr(monetio_sat, "_modis_l2_mm", fake_module, raising=False)
    path = tmp_path / "MOD04_L2.A2020167.0100.061.nc"
    path.write_text("placeholder")

    ds = MODISL2AODReader()._open_with_monetio([path], variables=None)

    assert ds["AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag"].dtype == np.int16
    assert ds["AOD_550_Dark_Target_Deep_Blue_Combined_Algorithm_Flag"].dtype == np.int16
