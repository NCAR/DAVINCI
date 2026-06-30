"""Tests for per-file progress callback in GenericReader.open().

Two tests:
- test_generic_progress_callback_called_per_file: verifies that when a
  progress_callback is supplied to GenericReader.open() for a 3-file list,
  the callback is called exactly 3 times with the correct (i, total, name)
  arguments in file order.
- test_generic_no_callback_uses_default_path: regression guard confirming that
  the default path (no callback) still loads the data correctly without error.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import davinci_monet.datasets  # noqa: F401  (registers "generic")
from davinci_monet.datasets.generic import GenericReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_nc(path: Path, value: float, time_offset_days: int = 0) -> None:
    """Write a tiny (time, lat, lon) NetCDF file with non-overlapping times."""
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=time_offset_days)
    t = pd.date_range(start, periods=2, freq="D")
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    ds = xr.Dataset(
        {"temp": (("time", "lat", "lon"), np.full((2, 2, 2), value))},
        coords={"time": t, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


def _make_scalar_time_grid_nc(path: Path, value: float, timestamp: str) -> None:
    """Write a tiny gridded NetCDF file with scalar time metadata."""
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    ds = xr.Dataset(
        {"aod": (("lat", "lon"), np.full((2, 2), value))},
        coords={"time": pd.Timestamp(timestamp), "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generic_progress_callback_called_per_file(tmp_path: Path) -> None:
    """progress_callback(i, total, name) is called once per file in order."""
    # Create 3 synthetic NetCDF files with names that sort deterministically.
    # Each file has non-overlapping time values so combine="by_coords" works.
    file_a = tmp_path / "file_a.nc"
    file_b = tmp_path / "file_b.nc"
    file_c = tmp_path / "file_c.nc"
    _make_tiny_nc(file_a, 1.0, time_offset_days=0)
    _make_tiny_nc(file_b, 2.0, time_offset_days=2)
    _make_tiny_nc(file_c, 3.0, time_offset_days=4)

    file_paths = [file_a, file_b, file_c]

    calls: List[Tuple[int, int, str]] = []

    def _cb(i: int, total: int, name: str) -> None:
        calls.append((i, total, name))

    reader = GenericReader()
    ds = reader.open(file_paths, variables=["temp"], progress_callback=_cb)

    # Dataset should still load correctly.
    assert "temp" in ds.data_vars

    # Callback must have been called exactly once per file.
    assert len(calls) == 3, f"Expected 3 callback calls, got {len(calls)}: {calls}"

    # i runs 1-based; total is always 3.
    indices = [c[0] for c in calls]
    totals = [c[1] for c in calls]
    names = [c[2] for c in calls]

    assert indices == [1, 2, 3], f"Expected i=[1,2,3], got {indices}"
    assert totals == [3, 3, 3], f"Expected total=[3,3,3], got {totals}"

    # Names must be the file basenames in the order the files were opened.
    expected_names = [p.name for p in file_paths]
    assert names == expected_names, f"Expected names {expected_names}, got {names}"


def test_generic_no_callback_uses_default_path(tmp_path: Path) -> None:
    """Without a progress_callback, open() still loads all files correctly."""
    file_a = tmp_path / "file_a.nc"
    file_b = tmp_path / "file_b.nc"
    file_c = tmp_path / "file_c.nc"
    _make_tiny_nc(file_a, 1.0, time_offset_days=0)
    _make_tiny_nc(file_b, 2.0, time_offset_days=2)
    _make_tiny_nc(file_c, 3.0, time_offset_days=4)

    reader = GenericReader()
    ds = reader.open([file_a, file_b, file_c], variables=["temp"])

    assert "temp" in ds.data_vars
    # 3 files × 2 time steps each = 6 combined time steps along time dim.
    assert ds.sizes["time"] == 6


def test_generic_nested_concat_uses_requested_concat_dim_for_scalar_time_files(
    tmp_path: Path,
) -> None:
    """Scalar-time grids can be stacked along the requested concat dimension."""
    file_a = tmp_path / "aod_a.nc"
    file_b = tmp_path / "aod_b.nc"
    _make_scalar_time_grid_nc(file_a, 1.0, "2024-01-01T00:00:00")
    _make_scalar_time_grid_nc(file_b, 2.0, "2024-01-01T03:00:00")

    ds = GenericReader().open(
        [file_a, file_b],
        variables=["aod"],
        combine="nested",
        concat_dim="time",
    )

    assert ds["aod"].dims == ("time", "lat", "lon")
    assert ds.sizes["time"] == 2
    np.testing.assert_allclose(ds["aod"].isel(time=0), 1.0)
    np.testing.assert_allclose(ds["aod"].isel(time=1), 2.0)


def test_generic_nested_concat_with_progress_callback_stacks_scalar_time_files(
    tmp_path: Path,
) -> None:
    """Nested scalar-time concat also works on the callback-enabled path."""
    file_a = tmp_path / "aod_a.nc"
    file_b = tmp_path / "aod_b.nc"
    _make_scalar_time_grid_nc(file_a, 1.0, "2024-01-01T00:00:00")
    _make_scalar_time_grid_nc(file_b, 2.0, "2024-01-01T03:00:00")
    calls: list[tuple[int, int, str]] = []

    def _cb(i: int, total: int, name: str) -> None:
        calls.append((i, total, name))

    ds = GenericReader().open(
        [file_a, file_b],
        variables=["aod"],
        combine="nested",
        concat_dim="time",
        progress_callback=_cb,
    )

    assert ds["aod"].dims == ("time", "lat", "lon")
    assert ds.sizes["time"] == 2
    assert calls == [(1, 2, file_a.name), (2, 2, file_b.name)]
    np.testing.assert_allclose(ds["aod"].isel(time=0), 1.0)
    np.testing.assert_allclose(ds["aod"].isel(time=1), 2.0)


@pytest.mark.parametrize(
    ("parallel_kwarg", "expected_parallel"),
    [(None, False), (True, True)],
)
def test_generic_no_callback_parallel_default_and_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parallel_kwarg: bool | None,
    expected_parallel: bool,
) -> None:
    """Multi-file opens default to sequential access unless explicitly overridden."""
    file_paths = [tmp_path / "file_a.nc", tmp_path / "file_b.nc"]
    for path in file_paths:
        path.touch()

    captured_kwargs: dict[str, object] = {}

    def _fake_open_mfdataset(paths: list[str], **kwargs: object) -> xr.Dataset:
        captured_kwargs.update(kwargs)
        return xr.Dataset(
            {"temp": ("time", np.array([1.0, 2.0]))},
            coords={"time": pd.date_range("2024-01-01", periods=2, freq="D")},
        )

    monkeypatch.setattr(xr, "open_mfdataset", _fake_open_mfdataset)

    if parallel_kwarg is None:
        ds = GenericReader().open(file_paths, variables=["temp"], standardize=False)
    else:
        ds = GenericReader().open(
            file_paths,
            variables=["temp"],
            standardize=False,
            parallel=parallel_kwarg,
        )

    assert "temp" in ds.data_vars
    assert captured_kwargs["parallel"] is expected_parallel
