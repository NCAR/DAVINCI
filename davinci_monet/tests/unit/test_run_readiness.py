"""Production readiness includes the direct native-MMR input inventory."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from davinci_monet.validation.readiness import _mmr_input_inventory_check


def test_mmr_readiness_rejects_an_empty_glob(tmp_path) -> None:  # noqa: ANN001
    config: Any = SimpleNamespace(
        analyses={
            "corrected_mmr": SimpleNamespace(
                type="mmr_writer",
                files=str(tmp_path / "MERRA2_*.nc4"),
            )
        }
    )

    check = _mmr_input_inventory_check(config)

    assert check.status == "failed"
    assert "matched no files" in check.detail


def test_mmr_readiness_reports_every_matching_file(tmp_path) -> None:  # noqa: ANN001
    for index in range(2):
        (tmp_path / f"MERRA2_{index}.nc4").write_bytes(b"mmr")
    config: Any = SimpleNamespace(
        analyses={
            "corrected_mmr": SimpleNamespace(
                type="mmr_writer",
                files=str(tmp_path / "MERRA2_*.nc4"),
            )
        }
    )

    check = _mmr_input_inventory_check(config)

    assert check.status == "passed"
    assert "corrected_mmr=2" in check.detail
