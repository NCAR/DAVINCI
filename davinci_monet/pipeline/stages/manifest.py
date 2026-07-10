from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
        errors = {
            key: list(value)
            for key in (
                "pairing_errors",
                "stats_errors",
                "plot_errors",
                "analysis_errors",
            )
            if (value := context.metadata.get(key))
        }
        status = "failed" if failed else "completed"
        manifest: dict[str, Any] = {
            "status": status,
            "failed_stages": failed,
            "errors": errors,
            "products": context.metadata.get("product_artifacts", {}),
            "analysis_artifacts": context.metadata.get("analysis_artifacts", []),
            "analysis_status": context.metadata.get("analysis_status", {}),
            "analysis_dependency_blocked": context.metadata.get("analysis_dependency_blocked", []),
            "analysis_partial_failure": context.metadata.get("analysis_partial_failure", []),
            "plots": plots,
            "inspection": (
                inspection.data if inspection and isinstance(inspection.data, dict) else {}
            ),
            "stages": {
                name: result.status.name.lower() for name, result in context.results.items()
            },
        }

        path = output_dir / "manifest.json"
        if status == "failed" and _is_completed_manifest(path):
            manifest["preserved_completed_manifest"] = str(path)
            path = output_dir / "manifest.failed.json"
        payload = (json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        )
        _atomic_write(path, payload)
        return self._create_result(
            StageStatus.COMPLETED,
            data={"manifest": str(path)},
            duration=time.time() - start,
        )


def _is_completed_manifest(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("status") == "completed"
