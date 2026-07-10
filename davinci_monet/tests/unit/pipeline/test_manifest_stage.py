from __future__ import annotations

import json

import pytest

import davinci_monet.pipeline.stages.manifest as manifest_module
from davinci_monet.pipeline.stages import (
    ManifestStage,
    PipelineContext,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.factory import create_standard_pipeline


def test_manifest_stage_writes_run_manifest(tmp_path) -> None:
    ctx = PipelineContext(config={"analysis": {"output_dir": str(tmp_path)}})
    ctx.metadata["product_artifacts"] = {
        "daily_aod": {
            "artifact_path": str(tmp_path / "products" / "daily_aod" / "analysis.nc"),
            "summary_path": str(tmp_path / "products" / "daily_aod" / "summary.json"),
        }
    }
    ctx.metadata["analysis_errors"] = ["pc1_wavelet: irregular data"]
    ctx.metadata["analysis_partial_failure"] = [{"analysis": "writer", "finalized_artifacts": 1}]
    ctx.results["plotting"] = StageResult(
        "plotting",
        StageStatus.COMPLETED,
        data={"plots_generated": [str(tmp_path / "plots" / "daily" / "aod.pdf")]},
    )

    result = ManifestStage().execute(ctx)

    assert result.status == StageStatus.COMPLETED
    manifest = tmp_path / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["status"] == "completed"
    assert "daily_aod" in data["products"]
    assert str(tmp_path / "plots" / "daily" / "aod.pdf") in data["plots"]
    assert data["errors"] == {"analysis_errors": ["pc1_wavelet: irregular data"]}
    assert data["analysis_partial_failure"] == [{"analysis": "writer", "finalized_artifacts": 1}]


def test_manifest_stage_serializes_path_like_payloads(tmp_path) -> None:
    ctx = PipelineContext(config={"analysis": {"output_dir": str(tmp_path)}})
    ctx.metadata["product_artifacts"] = {
        "daily_aod": {"artifact_path": tmp_path / "products" / "daily_aod" / "analysis.nc"}
    }
    ctx.results["inspection"] = StageResult(
        "inspection",
        StageStatus.COMPLETED,
        data={"preview": tmp_path / "inspection" / "preview.png"},
    )

    ManifestStage().execute(ctx)

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["products"]["daily_aod"]["artifact_path"].endswith("analysis.nc")
    assert data["inspection"]["preview"].endswith("preview.png")


def test_manifest_is_last_standard_stage() -> None:
    names = [stage.name for stage in create_standard_pipeline()]
    assert names[-1] == "manifest"


def test_manifest_publish_failure_preserves_previous_valid_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "manifest.json"
    previous = b'{"status":"completed","sentinel":"prior"}\n'
    path.write_bytes(previous)
    ctx = PipelineContext(config={"analysis": {"output_dir": str(tmp_path)}})

    def fail_replace(source, destination):  # noqa: ANN001
        raise OSError("injected manifest publish failure")

    monkeypatch.setattr(manifest_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="manifest publish failure"):
        ManifestStage().execute(ctx)

    assert path.read_bytes() == previous
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_failed_rerun_preserves_completed_manifest_and_publishes_failure(tmp_path) -> None:
    completed_path = tmp_path / "manifest.json"
    completed = {"status": "completed", "analysis_artifacts": [{"role": "basis_fit"}]}
    completed_path.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    ctx = PipelineContext(config={"analysis": {"output_dir": str(tmp_path)}})
    ctx.results["analyses"] = StageResult("analyses", StageStatus.FAILED, error="changed identity")

    result = ManifestStage().execute(ctx)

    assert json.loads(completed_path.read_text(encoding="utf-8")) == completed
    failed_path = tmp_path / "manifest.failed.json"
    assert result.data == {"manifest": str(failed_path)}
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["preserved_completed_manifest"] == str(completed_path)
