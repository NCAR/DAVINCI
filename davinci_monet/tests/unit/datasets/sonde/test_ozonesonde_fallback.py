"""Tests for the ozonesonde SHADOZ fallback parser's missing-data handling.

The SHADOZ convention masks ozone mixing-ratio/partial-pressure fills via
values >= 9000, independent of (and in addition to) any exact missing code
declared in the file header (B11). Non-ozone columns (altitude, geopotential
height, pressure) legitimately exceed 9000 and must not be floored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from davinci_monet.datasets.sonde.ozonesonde import OzonesondeReader


def test_shadoz_ozone_floor_masks_undeclared_fill_above_9000(tmp_path: Path) -> None:
    shadoz = tmp_path / "profile.dat"
    shadoz.write_text(
        "\n".join(
            [
                "SHADOZ regression file",
                "Missing values: -9999",
                "Press Alt Temp RH O3",
                "1000 12000 290 50 9005",
                "900 100 285 60 40",
            ]
        )
        + "\n"
    )

    ds = OzonesondeReader()._parse_shadoz_file(shadoz)

    assert np.isnan(ds["O3"].values[0, 0])
    assert ds["O3"].values[0, 1] == 40.0


def test_shadoz_ozone_floor_does_not_mask_high_altitude_values(tmp_path: Path) -> None:
    shadoz = tmp_path / "profile.dat"
    shadoz.write_text(
        "\n".join(
            [
                "SHADOZ regression file",
                "Missing values: -9999",
                "Press Alt Temp RH O3",
                "1000 12000 290 50 30",
                "900 9500 285 60 40",
            ]
        )
        + "\n"
    )

    ds = OzonesondeReader()._parse_shadoz_file(shadoz)

    assert ds["Alt"].values[0, 0] == 12000.0
    assert ds["Alt"].values[0, 1] == 9500.0


def test_shadoz_ozone_floor_masks_9999_sentinel_in_ozone_column(tmp_path: Path) -> None:
    shadoz = tmp_path / "profile.dat"
    shadoz.write_text(
        "\n".join(
            [
                "SHADOZ regression file",
                "Missing values: -9999",
                "Press Alt Temp RH O3",
                "1000 100 290 50 9999",
                "900 100 285 60 40",
            ]
        )
        + "\n"
    )

    ds = OzonesondeReader()._parse_shadoz_file(shadoz)

    assert np.isnan(ds["O3"].values[0, 0])
