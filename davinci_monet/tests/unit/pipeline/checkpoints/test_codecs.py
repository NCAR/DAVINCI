"""Checkpoint object codec round trips and integrity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.base import PairedData
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.checkpoints.codecs import (
    CheckpointCodecError,
    CheckpointCodecs,
)
from davinci_monet.pipeline.stages.base import SourceData


def _timed_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "aod": (
                ("time", "lat", "lon"),
                np.arange(24, dtype=np.float32).reshape(4, 2, 3),
                {"units": "1", "screened": True},
            ),
            "static": (("lat", "lon"), np.ones((2, 3)), {"role": "mask"}),
        },
        coords={
            "time": np.arange("2008-01-01", "2008-01-05", dtype="datetime64[D]"),
            "lat": ("lat", [-0.5, 0.5], {"units": "degrees_north"}),
            "lon": ("lon", [0.5, 1.5, 2.5], {"units": "degrees_east"}),
        },
        attrs={"geometry": "grid", "validated": True},
    )


def test_dataset_codec_round_trips_chunked_dataset_and_attrs(tmp_path: Path) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    source = _timed_dataset()

    obj = codecs.write_dataset(source, time_chunk_size=2)
    restored = codecs.read_dataset(obj)
    try:
        xr.testing.assert_allclose(restored.compute(), source)
        assert restored.attrs["validated"] is True
        assert restored["aod"].attrs["screened"] is True
        assert restored["lat"].attrs["units"] == "degrees_north"
        assert len([path for path in obj.paths if path.endswith(".nc")]) == 2
        assert all(data.chunks is None for data in restored.variables.values())
    finally:
        restored.close()

    repeated = codecs.write_dataset(source, time_chunk_size=2)
    assert repeated.object_id == obj.object_id
    assert repeated.paths == obj.paths


def test_dataset_codec_restores_dask_chunk_topology(tmp_path: Path) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    source = _timed_dataset().chunk({"time": (1, 3), "lat": 1})

    obj = codecs.write_dataset(source, time_chunk_size=2)
    restored = codecs.read_dataset(obj)
    try:
        xr.testing.assert_allclose(restored.compute(), source.compute())
        assert {name: data.chunks for name, data in restored.variables.items()} == {
            name: data.chunks for name, data in source.variables.items()
        }
    finally:
        restored.close()


def test_dataset_codec_round_trips_static_dataset(tmp_path: Path) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    source = xr.Dataset({"value": (("x",), np.asarray([1.0, 2.0]))})

    obj = codecs.write_dataset(source)
    restored = codecs.read_dataset(obj)
    try:
        xr.testing.assert_identical(restored.compute(), source)
    finally:
        restored.close()


def test_dataset_codec_appends_repair_when_existing_object_is_corrupt(
    tmp_path: Path,
) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    source = _timed_dataset()
    damaged = codecs.write_dataset(source, time_chunk_size=2)
    damaged_path = Path(damaged.paths[0])
    damaged_path.write_bytes(b"corrupt")

    repaired = codecs.write_dataset(source, time_chunk_size=2)
    restored = codecs.read_dataset(repaired)
    try:
        assert repaired.object_id == damaged.object_id
        assert repaired.paths != damaged.paths
        assert damaged_path.read_bytes() == b"corrupt"
        assert codecs.validate_object(repaired)
        xr.testing.assert_allclose(restored.compute(), source)
    finally:
        restored.close()

    repeated = codecs.write_dataset(source, time_chunk_size=2)
    assert repeated.paths == repaired.paths


def test_json_codec_is_canonical_and_reusable(tmp_path: Path) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")

    first = codecs.write_json({"b": 2, "a": [1, 3]})
    second = codecs.write_json({"a": (1, 3), "b": 2})

    assert first.object_id == second.object_id
    assert codecs.read_json(first) == {"a": [1, 3], "b": 2}


def test_json_codec_appends_repair_when_existing_object_is_corrupt(
    tmp_path: Path,
) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    damaged = codecs.write_json({"answer": 42})
    damaged_path = Path(damaged.paths[0])
    damaged_path.write_bytes(b"corrupt")

    repaired = codecs.write_json({"answer": 42})

    assert repaired.object_id == damaged.object_id
    assert repaired.paths != damaged.paths
    assert damaged_path.read_bytes() == b"corrupt"
    assert codecs.read_json(repaired) == {"answer": 42}
    assert codecs.write_json({"answer": 42}).paths == repaired.paths


def test_file_collection_detects_changed_or_missing_bytes(tmp_path: Path) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    pdf = tmp_path / "a001" / "output" / "plot.pdf"
    png = tmp_path / "a001" / "output" / "plot.png"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-content")
    png.write_bytes(b"PNG-content")
    obj = codecs.capture_files([pdf, png])

    assert codecs.validate_object(obj)
    png.write_bytes(b"changed")
    assert not codecs.validate_object(obj)
    with pytest.raises(CheckpointCodecError, match="checksum"):
        codecs.require_valid_object(obj)


def test_file_collection_rejects_an_empty_logical_product(tmp_path: Path) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    complete = tmp_path / "a001" / "output" / "plot.pdf"
    empty = tmp_path / "a001" / "output" / "preview.png"
    complete.parent.mkdir(parents=True)
    complete.write_bytes(b"%PDF-content")
    empty.touch()

    with pytest.raises(CheckpointCodecError, match="nonempty"):
        codecs.capture_files([complete, empty])


def test_source_metadata_reconstructs_source_container(tmp_path: Path) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    source = SourceData(
        data=_timed_dataset(),
        label="model",
        source_type="merra2",
        geometry=DataGeometry.GRID,
        variables={"aod": {"units": "1"}},
        config={"files": "/data/*.nc"},
    )
    metadata = codecs.source_metadata(source)
    obj = codecs.write_dataset(source.data)
    restored_dataset = codecs.read_dataset(obj)
    restored = codecs.restore_source(restored_dataset, metadata)

    assert restored.label == "model"
    assert restored.source_type == "merra2"
    assert restored.geometry is DataGeometry.GRID
    assert restored.variables == source.variables
    assert restored.config == source.config
    restored.data.close()


def test_paired_metadata_reconstructs_paired_container(tmp_path: Path) -> None:
    codecs = CheckpointCodecs(tmp_path / "a001")
    data = xr.Dataset(
        {
            "obs_aod": (("time",), [0.1, 0.2], {"axis": "x", "source_label": "obs"}),
            "cam_aod": (("time",), [0.11, 0.19], {"axis": "y", "source_label": "cam"}),
        },
        coords={"time": np.arange("2008-01-01", "2008-01-03", dtype="datetime64[D]")},
    )
    paired = PairedData.from_sources(
        data=data,
        x_source="obs",
        y_source="cam",
        geometry=DataGeometry.POINT,
        pairing_info={"method": "nearest"},
    )
    metadata = codecs.paired_metadata(paired)
    obj = codecs.write_dataset(data)
    restored_dataset = codecs.read_dataset(obj)
    restored = codecs.restore_paired(restored_dataset, metadata)

    assert restored.x_source == "obs"
    assert restored.y_source == "cam"
    assert restored.geometry is DataGeometry.POINT
    assert restored.pairing_info["method"] == "nearest"
    restored.data.close()
