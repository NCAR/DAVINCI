from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)


class ManifestStage(BaseStage):
    """Write the final self-contained run manifest."""

    def __init__(self) -> None:
        super().__init__("manifest")

    def execute(self, context: PipelineContext) -> StageResult:
        import time

        start = time.time()
        output_dir = Path(context.analysis_config().output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        plots: list[str] = []
        plotting = context.results.get("plotting")
        if plotting and isinstance(plotting.data, dict):
            plots = list(plotting.data.get("plots_generated", []))

        inspection = context.results.get("inspection")
        failed = [
            name for name, result in context.results.items() if result.status == StageStatus.FAILED
        ]
        status = "failed" if failed else "completed"
        manifest: dict[str, Any] = {
            "status": status,
            "failed_stages": failed,
            "products": context.metadata.get("product_artifacts", {}),
            "plots": plots,
            "inspection": (
                inspection.data if inspection and isinstance(inspection.data, dict) else {}
            ),
            "stages": {
                name: result.status.name.lower() for name, result in context.results.items()
            },
        }

        path = output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        return self._create_result(
            StageStatus.COMPLETED,
            data={"manifest": str(path)},
            duration=time.time() - start,
        )
