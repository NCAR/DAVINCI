from __future__ import annotations

import json

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
