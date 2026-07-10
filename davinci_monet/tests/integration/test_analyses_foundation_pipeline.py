"""Integration: a derived analysis runs through the full pipeline and its output
is registered as a pseudo-source (proves the foundation end-to-end)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis import DerivedAnalysis
from davinci_monet.analysis.artifact_manifest import validate_finalized_artifact_manifest
from davinci_monet.analysis.base import AnalysisExecutionError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry
from davinci_monet.pipeline.lifecycle import PipelineResourcePolicy
from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.pipeline.stages.base import StageStatus


@pytest.fixture
def _passthrough_eof():
    """Register a trivial 'eof' that emits a (time, mode) pc + (mode) variance."""

    _prev = analysis_registry.get_or_none("eof")

    class _PassEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.GRID

        def analyze(self, data, spec):  # noqa: ANN001
            nt = data.sizes["time"]
            return xr.Dataset(
                {
                    "pc": (("time", "mode"), np.zeros((nt, 2)), {"kind": "pc", "units": "1"}),
                    "explained_variance": ("mode", np.array([0.7, 0.3]), {"kind": "scalar"}),
                },
                coords={"time": data["time"].values, "mode": [1, 2]},
            )

    analysis_registry.register("eof", _PassEOF, replace=True)
    yield
    if _prev is not None:
        analysis_registry.register("eof", _prev, replace=True)
    else:
        analysis_registry.unregister("eof")


@pytest.fixture
def _required_failure_chain():
    previous = {name: analysis_registry.get_or_none(name) for name in ("eof", "wavelet")}
    calls = {"wavelet": 0}

    class _FailingEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.GRID

        def analyze(self, data, spec):  # noqa: ANN001
            if spec.variable == "BAD":
                raise RuntimeError("synthetic required failure")
            return xr.Dataset({"pc": ("time", np.zeros(data.sizes["time"]))})

    class _CountingWavelet(DerivedAnalysis):
        name = "wavelet"
        output_geometry = DataGeometry.SPECTRUM

        def analyze(self, data, spec):  # noqa: ANN001
            calls["wavelet"] += 1
            return xr.Dataset({"power": ("time", np.ones(data.sizes["time"]))})

    analysis_registry.register("eof", _FailingEOF, replace=True)
    analysis_registry.register("wavelet", _CountingWavelet, replace=True)
    yield calls
    for name, old in previous.items():
        if old is not None:
            analysis_registry.register(name, old, replace=True)
        else:
            analysis_registry.unregister(name)


@pytest.fixture
def _partial_failure_eof(tmp_path: Path):
    previous = analysis_registry.get_or_none("eof")
    finalized = tmp_path / "first-finalized.nc4"

    class _PartialFailureEOF(DerivedAnalysis):
        name = "eof"
        output_geometry = DataGeometry.ARTIFACT

        def analyze(self, data, spec):  # noqa: ANN001
            finalized.write_bytes(b"finalized-before-failure")
            raise AnalysisExecutionError(
                "second file failed",
                manifest_entries=(
                    {
                        "role": "corrected_mmr",
                        "kind": "mmr_file",
                        "status": "written",
                        "path": str(finalized),
                        "checksums": {
                            "output_sha256": hashlib.sha256(finalized.read_bytes()).hexdigest()
                        },
                    },
                ),
            )

    analysis_registry.register("eof", _PartialFailureEOF, replace=True)
    yield finalized
    if previous is not None:
        analysis_registry.register("eof", previous, replace=True)
    else:
        analysis_registry.unregister("eof")


def _grid_nc(path: Path) -> None:
    times = pd.date_range("2024-01-01", periods=6, freq="D")
    lat = np.linspace(20, 50, 4)
    lon = np.linspace(-120, -90, 5)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(times), len(lat), len(lon)))
    xr.Dataset(
        {"O3": (("time", "lat", "lon"), data, {"units": "ppb"})},
        coords={
            "time": times,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    ).to_netcdf(path)


@pytest.mark.integration
def test_analysis_runs_through_pipeline(tmp_path: Path, _passthrough_eof) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}
        },
        "analyses": {
            "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 2}
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    assert "cam_O3_eof" in ctx.sources
    derived = ctx.sources["cam_O3_eof"]
    assert derived.source_type == "eof"
    assert derived.geometry is DataGeometry.GRID
    assert derived.data.attrs["derived"] is True
    assert set(derived.data.data_vars) == {"pc", "explained_variance"}
    assert "cam_O3_eof" in ctx.results["analyses"].data


@pytest.mark.integration
def test_aerosol_tuning_required_failure_is_fatal(tmp_path: Path, _required_failure_chain) -> None:
    src = tmp_path / "grid.nc"
    output_dir = tmp_path / "out"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(output_dir)},
        "sources": {
            "cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}
        },
        "analyses": {
            "blocked_wavelet": {
                "type": "wavelet",
                "source": "required_eof",
                "variable": "pc",
                "required": True,
            },
            "required_eof": {
                "type": "eof",
                "source": "cam",
                "variable": "BAD",
                "required": True,
            },
            "independent_eof": {"type": "eof", "source": "cam", "variable": "O3"},
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success is False
    assert _required_failure_chain["wavelet"] == 0
    assert [stage.stage_name for stage in result.stage_results] == [
        "load_sources",
        "analyses",
        "manifest",
    ]
    assert result.context is not None
    analyses = result.context.results["analyses"]
    assert analyses.status is StageStatus.FAILED
    assert result.context.metadata["analysis_status"]["blocked_wavelet"] == "dependency_blocked"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["analysis_status"]["required_eof"] == "failed"
    assert manifest["analysis_dependency_blocked"][0]["analysis"] == "blocked_wavelet"


@pytest.mark.integration
def test_partial_analysis_receipt_reaches_failed_run_manifest(
    tmp_path: Path, _partial_failure_eof: Path
) -> None:
    src = tmp_path / "grid.nc"
    output_dir = tmp_path / "out"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(output_dir)},
        "sources": {"cam": {"type": "generic", "files": str(src), "variables": {"O3": {}}}},
        "analyses": {
            "writer": {
                "type": "eof",
                "source": "cam",
                "variable": "O3",
                "required": True,
            }
        },
    }

    result = PipelineRunner(show_progress=False).run_from_config(config)

    assert result.success is False
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["analysis_partial_failure"] == [
        {"analysis": "writer", "finalized_artifacts": 1}
    ]
    assert manifest["analysis_artifacts"][0]["analysis"] == "writer"
    assert manifest["analysis_artifacts"][0]["path"] == str(_partial_failure_eof)


@pytest.mark.integration
def test_identical_pipeline_rerun_reuses_verified_artifact(tmp_path: Path) -> None:
    src = tmp_path / "grid.nc"
    output_dir = tmp_path / "out"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(output_dir)},
        "sources": {"cam": {"type": "generic", "files": str(src), "variables": {"O3": {}}}},
        "analyses": {
            "basis": {
                "type": "eof",
                "source": "cam",
                "variable": "O3",
                "n_modes": 2,
                "required": True,
            }
        },
    }

    first = PipelineRunner(show_progress=False).run_from_config(config)
    second = PipelineRunner(show_progress=False).run_from_config(config)

    assert first.success and second.success
    manifest = json.loads((output_dir / "manifest.json").read_text())
    entry = next(item for item in manifest["analysis_artifacts"] if item["role"] == "basis_fit")
    assert entry["publication"] == "reused"
    assert entry["status"] == "finalized"


@pytest.mark.integration
def test_changed_rerun_preserves_completed_artifact_receipt(tmp_path: Path) -> None:
    src = tmp_path / "grid.nc"
    output_dir = tmp_path / "out"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(output_dir)},
        "sources": {"cam": {"type": "generic", "files": str(src), "variables": {"O3": {}}}},
        "analyses": {
            "basis": {
                "type": "eof",
                "source": "cam",
                "variable": "O3",
                "n_modes": 2,
                "required": True,
            }
        },
    }

    first = PipelineRunner(show_progress=False).run_from_config(config)
    assert first.success
    completed_path = output_dir / "manifest.json"
    completed_bytes = completed_path.read_bytes()
    completed = json.loads(completed_bytes)
    entry = next(item for item in completed["analysis_artifacts"] if item["role"] == "basis_fit")

    assert first.context is not None
    PipelineResourcePolicy().cleanup_context_datasets(first.context)
    with netCDF4.Dataset(src, "r+") as dataset:
        dataset.variables["O3"][0, 0, 0] += 1.0
    second = PipelineRunner(show_progress=False).run_from_config(config)

    assert not second.success
    assert completed_path.read_bytes() == completed_bytes
    failed = json.loads((output_dir / "manifest.failed.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    paths = [Path(entry["artifact_dir"]) / name for name in entry["checksums"]["files"]]
    validated = validate_finalized_artifact_manifest(
        completed_path,
        paths,
        role="basis_fit",
        analysis="basis",
    )
    assert validated["checksums"] == entry["checksums"]
