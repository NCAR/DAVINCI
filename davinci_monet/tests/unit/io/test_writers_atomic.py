"""Tests for atomic-write behavior of io.writers (FABLE_REVIEW L15)."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError
from davinci_monet.io import writers
from davinci_monet.io.writers import write_dataset, write_pickle


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Create a small sample xarray Dataset."""
    times = np.datetime64("2020-01-01") + np.arange(4) * np.timedelta64(1, "h")
    lats = np.linspace(30, 40, 3)
    lons = np.linspace(-110, -100, 3)

    data = np.arange(4 * 3 * 3, dtype=float).reshape(4, 3, 3)

    return xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _tmp_siblings(path: Path) -> list[Path]:
    return list(path.parent.glob(f"{path.name}*.tmp"))


class TestWriteDatasetNetCDFAtomic:
    def test_success_leaves_no_tmp_sibling(self, sample_dataset: xr.Dataset, temp_dir: Path):
        filepath = temp_dir / "output.nc"
        write_dataset(sample_dataset, filepath)

        assert filepath.exists()
        assert _tmp_siblings(filepath) == []

        result = xr.open_dataset(filepath)
        xr.testing.assert_equal(result, sample_dataset)
        result.close()

    def test_failure_leaves_target_absent(
        self, sample_dataset: xr.Dataset, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        filepath = temp_dir / "output.nc"

        def fake_to_netcdf(self: xr.Dataset, path: str, *args: Any, **kwargs: Any) -> None:
            Path(path).write_bytes(b"PARTIAL-NETCDF-BYTES")
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

        with pytest.raises(DataFormatError):
            write_dataset(sample_dataset, filepath)

        assert not filepath.exists()
        assert _tmp_siblings(filepath) == []

    def test_failure_preserves_existing_content(
        self, sample_dataset: xr.Dataset, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        filepath = temp_dir / "output.nc"
        filepath.write_bytes(b"ORIGINAL-CONTENT")

        def fake_to_netcdf(self: xr.Dataset, path: str, *args: Any, **kwargs: Any) -> None:
            Path(path).write_bytes(b"PARTIAL-NETCDF-BYTES")
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

        with pytest.raises(DataFormatError):
            write_dataset(sample_dataset, filepath)

        assert filepath.read_bytes() == b"ORIGINAL-CONTENT"
        assert _tmp_siblings(filepath) == []


class TestWritePickleAtomic:
    def test_success_leaves_no_tmp_sibling(self, sample_dataset: xr.Dataset, temp_dir: Path):
        filepath = temp_dir / "output.pkl"
        write_pickle(sample_dataset, filepath)

        assert filepath.exists()
        assert _tmp_siblings(filepath) == []

        with open(filepath, "rb") as f:
            result = pickle.load(f)
        xr.testing.assert_equal(result, sample_dataset)

    def test_failure_leaves_target_absent(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
        filepath = temp_dir / "output.pkl"

        def fake_dump(obj: Any, f: Any, protocol: int | None = None) -> None:
            f.write(b"PARTIAL-PICKLE-BYTES")
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(writers.pickle, "dump", fake_dump)

        with pytest.raises(DataFormatError):
            write_pickle({"key": "value"}, filepath)

        assert not filepath.exists()
        assert _tmp_siblings(filepath) == []

    def test_failure_preserves_existing_content(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        filepath = temp_dir / "output.pkl"
        filepath.write_bytes(b"ORIGINAL-CONTENT")

        def fake_dump(obj: Any, f: Any, protocol: int | None = None) -> None:
            f.write(b"PARTIAL-PICKLE-BYTES")
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(writers.pickle, "dump", fake_dump)

        with pytest.raises(DataFormatError):
            write_pickle({"key": "value"}, filepath)

        assert filepath.read_bytes() == b"ORIGINAL-CONTENT"
        assert _tmp_siblings(filepath) == []


class TestWriteDatasetZarrAtomic:
    def test_success_leaves_no_tmp_sibling(self, sample_dataset: xr.Dataset, temp_dir: Path):
        pytest.importorskip("zarr")

        filepath = temp_dir / "output.zarr"
        write_dataset(sample_dataset, filepath)

        assert filepath.exists()
        assert filepath.is_dir()
        assert _tmp_siblings(filepath) == []

        result = xr.open_zarr(filepath)
        xr.testing.assert_equal(result.load(), sample_dataset)

    def test_failure_leaves_target_absent(
        self, sample_dataset: xr.Dataset, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pytest.importorskip("zarr")

        filepath = temp_dir / "output.zarr"

        def fake_to_zarr(self: xr.Dataset, path: str, *args: Any, **kwargs: Any) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "partial.marker").write_bytes(b"PARTIAL-ZARR")
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)

        with pytest.raises(DataFormatError):
            write_dataset(sample_dataset, filepath)

        assert not filepath.exists()
        assert _tmp_siblings(filepath) == []

    def test_failure_preserves_existing_store(
        self, sample_dataset: xr.Dataset, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pytest.importorskip("zarr")

        filepath = temp_dir / "output.zarr"
        write_dataset(sample_dataset, filepath)
        original_children = sorted(p.name for p in filepath.iterdir())

        def fake_to_zarr(self: xr.Dataset, path: str, *args: Any, **kwargs: Any) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "partial.marker").write_bytes(b"PARTIAL-ZARR")
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)

        with pytest.raises(DataFormatError):
            write_dataset(sample_dataset, filepath)

        assert filepath.exists()
        assert sorted(p.name for p in filepath.iterdir()) == original_children
        assert _tmp_siblings(filepath) == []
