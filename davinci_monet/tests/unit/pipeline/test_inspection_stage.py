from __future__ import annotations

from davinci_monet.config.schema import MonetConfig
from davinci_monet.pipeline.stages import InspectionStage, PipelineContext, StageStatus


def test_inspection_stage_skips_when_disabled(tmp_path) -> None:
    ctx = PipelineContext(config={"analysis": {"output_dir": str(tmp_path)}})
    result = InspectionStage().execute(ctx)
    assert result.status == StageStatus.SKIPPED


def test_required_inspection_failure_fails_stage(tmp_path) -> None:
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "inspection": {
                "enabled": True,
                "required": True,
                "presets": ["gridded_aod_diagnostics"],
            },
        }
    )
    result = InspectionStage().execute(ctx)
    assert result.status == StageStatus.FAILED
    assert result.error == "required inspection failed"


def test_optional_inspection_failure_completes_stage(tmp_path) -> None:
    ctx = PipelineContext(
        config={
            "analysis": {"output_dir": str(tmp_path)},
            "inspection": {
                "enabled": True,
                "required": False,
                "presets": ["gridded_aod_diagnostics"],
            },
        }
    )

    result = InspectionStage().execute(ctx)

    assert result.status == StageStatus.COMPLETED
    assert result.data["passed"] is False
    assert (tmp_path / "inspection" / "inspection.json").exists()


def test_enabled_inspection_passes_for_typed_config(tmp_path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "aod.pdf").write_bytes(b"%PDF-1.4\n")
    ctx = PipelineContext(
        config=MonetConfig(
            analysis={"output_dir": str(tmp_path)},
            inspection={
                "enabled": True,
                "required": True,
                "presets": ["gridded_aod_diagnostics"],
            },
        )
    )

    result = InspectionStage().execute(ctx)

    assert result.status == StageStatus.COMPLETED
    assert result.data["passed"] is True
    assert result.data["inspection_json"].endswith("inspection.json")
