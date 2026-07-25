from __future__ import annotations

from pathlib import Path

import xarray as xr

from davinci_monet.analysis.artifacts import ArtifactService
from davinci_monet.analysis.base import ArtifactDeclaration
from davinci_monet.config.schema import MonetConfig
from davinci_monet.pipeline.stages.base import (
    PipelineContext,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.completion import CompletionStage


def _production_config(output_dir: Path) -> MonetConfig:
    return MonetConfig.model_validate(
        {
            "run": {
                "id": "aod-model-sensor-2008-eof-r01",
                "kind": "production",
                "completion": {
                    "required_analyses": {"basis": "eof"},
                    "required_artifacts": [
                        {"analysis": "basis", "role": "basis_fit"},
                    ],
                    "required_saved_files": ["statistics_summary"],
                    "required_plots": ["basis_scree"],
                    "inspection": {
                        "required": True,
                        "presets": ["eof_wavelet"],
                    },
                    "allow_item_errors": False,
                },
            },
            "analysis": {
                "output_dir": str(output_dir),
                "log_dir": str(output_dir.parent / "logs"),
            },
            "execution": {
                "attempt_root": str(output_dir.parent),
                "checkpoints": {
                    "mode": "required",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "sources": {"model": {"type": "generic"}},
            "analyses": {
                "basis": {
                    "type": "eof",
                    "source": "model",
                    "variable": "aod",
                    "required": True,
                }
            },
            "plots": {
                "basis_scree": {
                    "type": "eof_scree",
                    "source": "basis",
                    "variable": "explained_variance",
                }
            },
            "inspection": {
                "enabled": True,
                "required": True,
                "presets": ["eof_wavelet"],
                "preview_format": "png",
            },
        }
    )


def _completed_context(tmp_path: Path) -> PipelineContext:
    output_dir = tmp_path / "a001" / "output"
    context = PipelineContext(config=_production_config(output_dir))
    dataset = xr.Dataset(
        {"explained_variance": ("mode", [0.75, 0.25])},
        coords={"mode": [1, 2]},
    )
    materialized = ArtifactService(output_dir).materialize(
        "basis",
        dataset,
        [
            ArtifactDeclaration(
                kind="netcdf_collection",
                role="basis_fit",
                options={"time_chunk_size": 31},
            )
        ],
    )
    context.metadata["analysis_status"] = {"basis": "completed"}
    context.metadata["analysis_artifacts"] = list(materialized.manifest_entries)

    statistics = output_dir / "statistics_summary.csv"
    statistics.write_text("metric,value\nrmse,0.1\n", encoding="utf-8")
    context.results["save_results"] = StageResult(
        "save_results",
        StageStatus.COMPLETED,
        data={
            "saved_files": [str(statistics)],
            "saved_products": {"statistics_summary": str(statistics)},
        },
    )

    plot_paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"basis_scree.{suffix}"
        path.write_bytes(b"plot")
        plot_paths.append(str(path))
    context.results["plotting"] = StageResult(
        "plotting",
        StageStatus.COMPLETED,
        data={
            "plots_generated": plot_paths,
            "plot_products": {"basis_scree": plot_paths},
        },
    )

    inspection_dir = output_dir / "inspection"
    inspection_dir.mkdir()
    inspection_json = inspection_dir / "inspection.json"
    inspection_markdown = inspection_dir / "inspection.md"
    preview = inspection_dir / "previews" / "basis_scree.png"
    preview.parent.mkdir()
    inspection_json.write_text('{"passed": true}\n', encoding="utf-8")
    inspection_markdown.write_text("# passed\n", encoding="utf-8")
    preview.write_bytes(b"preview")
    context.results["inspection"] = StageResult(
        "inspection",
        StageStatus.COMPLETED,
        data={
            "passed": True,
            "inspection_json": str(inspection_json),
            "inspection_markdown": str(inspection_markdown),
            "inspection_previews": [str(preview)],
        },
    )
    return context


def test_completion_stage_accepts_exact_durable_products(tmp_path: Path) -> None:
    result = CompletionStage().execute(_completed_context(tmp_path))

    assert result.status is StageStatus.COMPLETED
    assert result.data["passed"] is True
    assert result.data["errors"] == []


def test_completion_stage_rejects_missing_output_and_item_error(tmp_path: Path) -> None:
    context = _completed_context(tmp_path)
    plot_path = Path(context.results["plotting"].data["plot_products"]["basis_scree"][0])
    plot_path.unlink()
    context.metadata["plot_errors"] = ["basis_scree: renderer failed"]

    result = CompletionStage().execute(context)

    assert result.status is StageStatus.FAILED
    assert result.data["passed"] is False
    assert result.error is not None
    assert "required plot 'basis_scree' output does not exist" in result.error
    assert "plot_errors" in result.error


def test_completion_stage_skips_nonproduction_run(tmp_path: Path) -> None:
    context = PipelineContext(
        config=MonetConfig.model_validate(
            {
                "run": {
                    "id": "aod-model-sensor-2008-eof-preflight",
                    "kind": "preflight",
                },
                "sources": {"model": {"type": "generic"}},
            }
        )
    )

    result = CompletionStage().execute(context)

    assert result.status is StageStatus.SKIPPED
    assert result.data == {"skipped": "no production completion contract"}
