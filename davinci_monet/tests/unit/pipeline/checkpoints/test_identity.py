"""Canonical checkpoint identity construction."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from davinci_monet.config.schema import MonetConfig
from davinci_monet.pipeline.checkpoints.identity import (
    canonical_sha256,
    canonicalize,
    code_tree_sha256,
    compose_checkpoint_identity,
    configuration_sha256,
    git_commit,
    inventory_sources,
    runtime_versions,
    source_inventory_sha256,
)
from davinci_monet.pipeline.checkpoints.models import CheckpointDependency


def test_canonical_hash_is_stable_across_order_and_supported_types(tmp_path: Path) -> None:
    first = {
        "path": tmp_path / "data.nc",
        "when": datetime(2026, 7, 25, 12, tzinfo=UTC),
        "values": np.asarray([np.int64(2), np.int64(3)]),
        "pair": ("x", np.float32(1.5)),
        "mapping": {"b": 2, "a": 1},
    }
    second = {
        "mapping": {"a": 1, "b": 2},
        "pair": ["x", 1.5],
        "values": [2, 3],
        "when": "2026-07-25T12:00:00+00:00",
        "path": str(tmp_path / "data.nc"),
    }

    assert canonicalize(first) == canonicalize(second)
    assert canonical_sha256(first) == canonical_sha256(second)


def test_configuration_hash_uses_normalized_pydantic_values(tmp_path: Path) -> None:
    attempt = tmp_path / "a001"
    config = MonetConfig.model_validate(
        {
            "run": {"id": "resume-smoke", "kind": "smoke"},
            "analysis": {
                "output_dir": attempt / "output",
                "log_dir": attempt / "logs",
            },
            "execution": {
                "attempt_root": attempt,
                "checkpoints": {
                    "mode": "best_effort",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
        }
    )

    assert configuration_sha256(config) == canonical_sha256(config.model_dump(mode="json"))


def test_runtime_and_repository_provenance_are_explicit() -> None:
    versions = runtime_versions()

    assert set(versions) == {
        "python",
        "davinci",
        "numpy",
        "scipy",
        "xarray",
        "dask",
        "pandas",
        "netCDF4",
    }
    assert versions["python"]
    assert versions["davinci"] != "not-installed"
    assert versions["numpy"]
    assert git_commit(Path(__file__).resolve()) is not None


def test_source_inventory_changes_with_stat_or_authoritative_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    source.write_bytes(b"first")
    first = inventory_sources([source])
    first_hash = source_inventory_sha256(first)

    source.write_bytes(b"second-value")
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1))
    second = inventory_sources([source])

    assert first_hash != source_inventory_sha256(second)
    with_checksum = inventory_sources(
        [source],
        authoritative_checksums={source: "a" * 64},
    )
    assert with_checksum[0]["authoritative_sha256"] == "a" * 64
    assert source_inventory_sha256(with_checksum) != source_inventory_sha256(second)


def test_code_tree_hash_ignores_tests_and_python_cache(tmp_path: Path) -> None:
    package = tmp_path / "package"
    tests = package / "tests"
    cache = package / "__pycache__"
    tests.mkdir(parents=True)
    cache.mkdir()
    (package / "module.py").write_text("VALUE = 1\n")
    (tests / "test_module.py").write_text("assert True\n")
    (cache / "module.py").write_text("ignored = True\n")
    before = code_tree_sha256(package)

    (tests / "test_module.py").write_text("assert False\n")
    (cache / "module.py").write_text("ignored = False\n")

    assert code_tree_sha256(package, use_cache=False) == before
    (package / "module.py").write_text("VALUE = 2\n")
    assert code_tree_sha256(package, use_cache=False) != before


def test_composed_identity_changes_one_dimension_at_a_time(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    source.write_bytes(b"source")
    inventory = inventory_sources([source])
    dependency = CheckpointDependency(
        stage="load_sources",
        item="model",
        receipt_sha256="a" * 64,
    )
    baseline = compose_checkpoint_identity(
        stage="analyses",
        item="basis",
        config={"n_modes": 10},
        dependencies=[dependency],
        source_inventory=inventory,
        code_sha256="b" * 64,
    )

    variants = (
        compose_checkpoint_identity(
            stage="analyses",
            item="basis",
            config={"n_modes": 20},
            dependencies=[dependency],
            source_inventory=inventory,
            code_sha256="b" * 64,
        ),
        compose_checkpoint_identity(
            stage="analyses",
            item="basis",
            config={"n_modes": 10},
            dependencies=[dependency.model_copy(update={"receipt_sha256": "c" * 64})],
            source_inventory=inventory,
            code_sha256="b" * 64,
        ),
        compose_checkpoint_identity(
            stage="analyses",
            item="basis",
            config={"n_modes": 10},
            dependencies=[dependency],
            source_inventory=inventory,
            code_sha256="d" * 64,
        ),
    )

    assert all(variant["identity_sha256"] != baseline["identity_sha256"] for variant in variants)
