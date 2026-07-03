from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError
from davinci_monet.datasets.aircraft.icartt import ICARTTReader
from davinci_monet.datasets.generic import GenericReader
from davinci_monet.datasets.satellite.modis_l2_aod import MODISL2AODReader
from davinci_monet.datasets.satellite.tropomi import TROPOMIReader
from davinci_monet.datasets.sonde.ozonesonde import OzonesondeReader


def test_icartt_fallback_anchors_time_to_header_date_and_masks_missing(tmp_path: Path) -> None:
    ict = tmp_path / "flight.ict"
    ict.write_text(
        "\n".join(
            [
                "18, 1001",
                "Test Organization",
                "ICARTT Regression Test",
                "Test Source",
                "TEST_CAMPAIGN",
                "1, 1",
                "2024, 01, 15, 2024, 1, 15",
                "-1",
                "Time_Start, seconds, start time",
                "3",
                "1, 1, 1",
                "-999999, -999999, -999999",
                "O3_CL, ppbv, ozone",
                "CO_DACOM, ppbv, carbon monoxide",
                "MSL_GPS_Altitude, m, altitude",
                "0",
                "0",
                "Time_Start, O3_CL, CO_DACOM, MSL_GPS_Altitude",
                "0, -999999, 120.5, 2500.0",
                "60, 46.1, -999999, 2510.0",
            ]
        )
        + "\n"
    )

    ds = ICARTTReader()._open_with_parser([ict], variables=None)

    np.testing.assert_array_equal(
        ds["time"].values,
        np.array(["2024-01-15T00:00:00", "2024-01-15T00:01:00"], dtype="datetime64[ns]"),
    )
    assert np.isnan(ds["O3_CL"].values[0])
    assert np.isnan(ds["CO_DACOM"].values[1])


def test_ozonesonde_woudc_masks_header_declared_missing_values(tmp_path: Path) -> None:
    woudc = tmp_path / "profile.csv"
    woudc.write_text(
        "\n".join(
            [
                "#WOUDC",
                "#MISSING_VALUE=-9999, 9000",
                "#PROFILE",
                "Pressure,O3,Temp",
                "1000,-9999,280",
                "900,50,9000",
            ]
        )
        + "\n"
    )

    ds = OzonesondeReader()._parse_woudc_file(woudc)

    assert np.isnan(ds["O3"].values[0, 0])
    assert np.isnan(ds["Temp"].values[0, 1])


def test_ozonesonde_shadoz_masks_header_declared_missing_values(tmp_path: Path) -> None:
    shadoz = tmp_path / "profile.dat"
    shadoz.write_text(
        "\n".join(
            [
                "SHADOZ regression file",
                "Missing values: -9999 9000",
                "Press Alt Temp RH O3",
                "1000 100 290 50 9000",
                "900 -9999 285 60 40",
            ]
        )
        + "\n"
    )

    ds = OzonesondeReader()._parse_shadoz_file(shadoz)

    assert np.isnan(ds["O3"].values[0, 0])
    assert np.isnan(ds["Alt"].values[0, 1])


def test_generic_nested_concat_sorts_time_after_concatenation(tmp_path: Path) -> None:
    late = tmp_path / "out_10.nc"
    early = tmp_path / "out_2.nc"
    xr.Dataset(
        {"aod": ("time", [2.0])},
        coords={"time": np.array(["2024-01-02"], dtype="datetime64[ns]")},
    ).to_netcdf(late)
    xr.Dataset(
        {"aod": ("time", [1.0])},
        coords={"time": np.array(["2024-01-01"], dtype="datetime64[ns]")},
    ).to_netcdf(early)

    ds = GenericReader().open([late, early], combine="nested", concat_dim="time")

    np.testing.assert_array_equal(
        ds["time"].values,
        np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(ds["aod"].values, np.array([1.0, 2.0]))


def test_generic_nested_concat_rejects_duplicate_time_after_concatenation(
    tmp_path: Path,
) -> None:
    first = tmp_path / "out_1.nc"
    second = tmp_path / "out_2.nc"
    for path, value in [(first, 1.0), (second, 2.0)]:
        xr.Dataset(
            {"aod": ("time", [value])},
            coords={"time": np.array(["2024-01-01"], dtype="datetime64[ns]")},
        ).to_netcdf(path)

    with pytest.raises(DataFormatError, match="Duplicate time values"):
        GenericReader().open([first, second], combine="nested", concat_dim="time")


def test_tropomi_ground_pixel_standardizes_to_pixel_dimension() -> None:
    ds = xr.Dataset(
        {"no2": (("nscans", "ground_pixel"), np.ones((2, 3)))},
        coords={
            "latitude": (("nscans", "ground_pixel"), np.ones((2, 3))),
            "longitude": (("nscans", "ground_pixel"), np.ones((2, 3))),
        },
    )

    out = TROPOMIReader()._standardize_dataset(ds)

    assert "scanline" in out.dims
    assert "pixel" in out.dims
    assert "ground_pixel" not in out.dims
    assert out["no2"].dims == ("scanline", "pixel")


def test_modis_l2_aod_scan_start_time_handles_datetime64_time_coord() -> None:
    ds = xr.Dataset(
        coords={
            "time": (
                "scan",
                pd.to_datetime(["1993-01-01T00:02:03", "1993-01-01T00:02:04"]),
            )
        }
    )

    out = MODISL2AODReader()._add_scan_start_time(ds)

    np.testing.assert_array_equal(out["Scan_Start_Time"].values, np.array([123.0, 124.0]))
