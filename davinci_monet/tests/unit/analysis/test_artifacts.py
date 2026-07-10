from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import davinci_monet.analysis.artifacts as artifacts_module
from davinci_monet.analysis.artifact_manifest import build_analysis_artifact_identity
from davinci_monet.analysis.artifacts import (
    ArtifactService,
    validate_finalized_artifact_manifest,
    write_dataset_collection,
    write_product_artifacts,
)
from davinci_monet.analysis.base import ArtifactDeclaration


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
    assert len(result.analysis_checksum) == 64
    assert len(result.summary_checksum) == 64
    assert not list(result.analysis_path.parent.glob(".davinci-*"))
    summary = json.loads(result.summary_path.read_text())
    assert summary["product"] == "daily_aod"
    assert summary["fields"]["aod"]["finite_count"] == 3
    for bucket in ("source_hashes", "config_hashes", "code_hashes"):
        assert summary["identity"][bucket]
        assert all(len(value) == 64 for value in summary["identity"][bucket].values())


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


def test_write_product_artifacts_cleans_temporary_files_on_failure(tmp_path, monkeypatch) -> None:
    ds = xr.Dataset({"aod": ("time", np.array([0.1, 0.2]))})

    def fail_write(*args, **kwargs):  # noqa: ANN002, ANN003
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_write)

    with pytest.raises(OSError, match="disk full"):
        write_product_artifacts(tmp_path, "daily_aod", ds)

    root = tmp_path / "products" / "daily_aod"
    assert not list((tmp_path / "products").glob(".daily_aod-*"))
    assert not (root / "analysis.nc").exists()


def test_write_product_artifacts_publish_failure_exposes_no_partial_pair(
    tmp_path, monkeypatch
) -> None:
    ds = xr.Dataset({"aod": ("time", np.array([0.1, 0.2]))})

    def fail_publish(source, destination):  # noqa: ANN001
        raise OSError("injected atomic publish failure")

    monkeypatch.setattr(artifacts_module.os, "replace", fail_publish)

    with pytest.raises(OSError, match="injected atomic publish failure"):
        write_product_artifacts(tmp_path, "daily_aod", ds)

    destination = tmp_path / "products" / "daily_aod"
    assert not destination.exists()
    assert not list((tmp_path / "products").glob(".daily_aod-*"))


def test_artifact_service_writes_and_lazily_reopens_time_collection(tmp_path) -> None:
    time = np.arange(
        np.datetime64("2001-01-01"),
        np.datetime64("2001-03-12"),
        np.timedelta64(1, "D"),
    )
    source = xr.Dataset(
        {
            "r": (("time", "lat", "lon"), np.ones((70, 2, 3), dtype=np.float32)),
            "active": (("time", "lat", "lon"), np.ones((70, 2, 3), dtype=bool)),
            "monthly_support": (("month", "lat", "lon"), np.ones((12, 2, 3))),
        },
        coords={
            "time": time,
            "month": np.arange(1, 13),
            "lat": [-30.0, 30.0],
            "lon": [-120.0, 0.0, 120.0],
        },
        attrs={"synthetic": True},
    ).chunk({"time": 10})
    declaration = ArtifactDeclaration(
        kind="netcdf_collection",
        role="scaling",
        reload=True,
        options={"time_chunk_size": 31},
    )

    materialized = ArtifactService(tmp_path).materialize("scaling", source, (declaration,))

    paths = sorted((tmp_path / "artifacts" / "scaling").glob("chunk-*.nc"))
    assert len(paths) == 3
    assert materialized.dataset["r"].chunks is not None
    assert materialized.dataset["monthly_support"].dims == ("month", "lat", "lon")
    xr.testing.assert_allclose(materialized.dataset.compute(), source.compute())
    entry = materialized.manifest_entries[0]
    assert entry["kind"] == "netcdf_collection"
    assert len(entry["checksums"]["collection_sha256"]) == 64
    assert len(entry["checksums"]["files"]) == 3
    assert entry["status"] == "finalized"
    for bucket in ("source_hashes", "config_hashes", "code_hashes"):
        assert entry["identity"][bucket]
        assert all(len(value) == 64 for value in entry["identity"][bucket].values())
    assert not list((tmp_path / "artifacts").glob(".scaling-*"))


def test_finalized_manifest_validation_hashes_every_collection_file(tmp_path) -> None:
    source = xr.Dataset(
        {"r": ("time", np.ones(3))},
        coords={"time": np.arange(3)},
        attrs={"spec_hash": "synthetic-spec"},
    )
    declaration = ArtifactDeclaration("netcdf_collection", role="scaling", reload=False)
    materialized = ArtifactService(tmp_path).materialize("scaling", source, (declaration,))
    entry = dict(materialized.manifest_entries[0])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"status": "completed", "analysis_artifacts": [entry]}) + "\n"
    )
    paths = sorted((tmp_path / "artifacts" / "scaling").glob("chunk-*.nc"))

    validated = validate_finalized_artifact_manifest(
        manifest_path, paths, role="scaling", analysis="scaling"
    )

    assert validated["checksums"]["collection_sha256"] == entry["checksums"]["collection_sha256"]
    selected = validate_finalized_artifact_manifest(
        manifest_path, paths[:1], role="scaling", analysis="scaling"
    )
    assert selected == validated
    unlisted = tmp_path / "unlisted.nc"
    unlisted.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="configured artifact files do not match"):
        validate_finalized_artifact_manifest(
            manifest_path, [unlisted], role="scaling", analysis="scaling"
        )
    paths[0].write_bytes(paths[0].read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_finalized_artifact_manifest(
            manifest_path, paths, role="scaling", analysis="scaling"
        )


def test_source_identity_hashes_coordinate_and_array_content() -> None:
    first = xr.Dataset(
        {"aod": ("time", [1.0, 2.0])},
        coords={"time": np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[D]")},
        attrs={"spec_hash": "same-scenario"},
    )
    changed_values = first.copy(deep=True)
    changed_values["aod"][1] = 9.0
    changed_time = first.assign_coords(
        time=np.array(["2002-01-01", "2002-01-02"], dtype="datetime64[D]")
    )
    spec = SimpleNamespace(type="content-test")

    identities = [
        build_analysis_artifact_identity(spec, {"source": dataset}, None)["source_inputs_sha256"]
        for dataset in (first, changed_values, changed_time)
    ]

    assert len(set(identities)) == 3


def test_dataset_collection_failure_removes_staging_directory(tmp_path, monkeypatch) -> None:
    source = xr.Dataset(
        {"r": ("time", np.ones(3))},
        coords={"time": np.arange(3)},
    )

    def fail_write(*args, **kwargs):  # noqa: ANN002, ANN003
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_write)
    with pytest.raises(OSError, match="disk full"):
        write_dataset_collection(tmp_path, "scaling", source)

    parent = tmp_path / "artifacts"
    assert not (parent / "scaling").exists()
    assert not list(parent.glob(".scaling-*"))


def test_dataset_collection_checksum_failure_occurs_before_publish(tmp_path, monkeypatch) -> None:
    source = xr.Dataset({"r": ("time", np.ones(3))}, coords={"time": np.arange(3)})
    original = artifacts_module._sha256

    def fail_staged_checksum(path):  # noqa: ANN001
        if path.name.startswith("chunk-"):
            raise OSError("injected checksum failure")
        return original(path)

    monkeypatch.setattr(artifacts_module, "_sha256", fail_staged_checksum)

    with pytest.raises(OSError, match="injected checksum failure"):
        write_dataset_collection(tmp_path, "scaling", source)

    assert not (tmp_path / "artifacts" / "scaling").exists()
    assert not list((tmp_path / "artifacts").glob(".scaling-*"))


def test_dataset_collection_idempotently_reuses_only_verified_identity(tmp_path) -> None:
    source = xr.Dataset({"r": ("time", np.ones(3))}, coords={"time": np.arange(3)})
    first = write_dataset_collection(tmp_path, "scaling", source)
    second = write_dataset_collection(tmp_path, "scaling", source)

    assert first.reused is False
    assert second.reused is True
    assert second.checksums == first.checksums

    changed = source.copy(deep=True)
    changed["r"][0] = 2.0
    with pytest.raises(FileExistsError, match="identity does not match"):
        write_dataset_collection(tmp_path, "scaling", changed)

    first.paths[0].write_bytes(first.paths[0].read_bytes() + b"tampered")
    with pytest.raises(FileExistsError, match="checksum does not match"):
        write_dataset_collection(tmp_path, "scaling", source)


def test_product_idempotently_reuses_only_verified_identity(tmp_path) -> None:
    source = xr.Dataset({"aod": ("time", np.array([0.1, 0.2]))})
    first = write_product_artifacts(tmp_path, "daily", source)
    second = write_product_artifacts(tmp_path, "daily", source)

    assert first.reused is False
    assert second.reused is True
    assert second.analysis_checksum == first.analysis_checksum

    source["aod"][0] = 0.3
    with pytest.raises(FileExistsError, match="identity does not match"):
        write_product_artifacts(tmp_path, "daily", source)
