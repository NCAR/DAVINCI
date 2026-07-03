from __future__ import annotations

import sys
from pathlib import Path

from davinci_monet.config.schema import MonetConfig
from davinci_monet.pipeline.stages import (
    InspectionStage,
    PipelineContext,
    StageResult,
    StageStatus,
)


def _install_fake_pdftoppm(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_pdftoppm = bin_dir / "pdftoppm"
    fake_pdftoppm.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[-1] + '.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')\n"
    )
    fake_pdftoppm.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}")


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


def test_enabled_inspection_passes_for_typed_config(tmp_path, monkeypatch) -> None:
    _install_fake_pdftoppm(tmp_path, monkeypatch)
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "aod.pdf").write_bytes(b"%PDF-1.4\n")
    ctx = PipelineContext(
        config=MonetConfig.model_validate(
            {
                "analysis": {"output_dir": str(tmp_path)},
                "inspection": {
                    "enabled": True,
                    "required": True,
                    "presets": ["gridded_aod_diagnostics"],
                },
            }
        )
    )

    result = InspectionStage().execute(ctx)

    assert result.status == StageStatus.COMPLETED
    assert result.data["passed"] is True
    assert result.data["inspection_json"].endswith("inspection.json")
    assert result.data["inspection_previews"] == [
        str(tmp_path / "inspection" / "previews" / "aod.png")
    ]


def test_inspection_stage_uses_plotting_stage_products(tmp_path, monkeypatch) -> None:
    _install_fake_pdftoppm(tmp_path, monkeypatch)
    plot_dir = tmp_path / "cam"
    plot_dir.mkdir()
    pdf = plot_dir / "aod.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
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
    ctx.results["plotting"] = StageResult(
        "plotting",
        StageStatus.COMPLETED,
        data={"plots_generated": [str(pdf)]},
    )

    result = InspectionStage().execute(ctx)

    assert result.status == StageStatus.COMPLETED
    assert result.data["passed"] is True
