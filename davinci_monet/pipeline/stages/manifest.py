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
        manager = context.checkpoint_manager
        checkpoint_receipts = (
            [receipt.model_dump(mode="json") for receipt in manager.store.iter_receipts()]
            if manager is not None
            else []
        )
        receipt_dispositions: dict[str, int] = {}
        for receipt in checkpoint_receipts:
            disposition = str(receipt["disposition"])
            receipt_dispositions[disposition] = receipt_dispositions.get(disposition, 0) + 1
        checkpoint_events = manager.store.read_events() if manager is not None else []
        execution_id = manager.execution_id if manager is not None else None
        execution_records = manager.store.list_executions() if manager is not None else []
        manifest_attempt = manager.attempt if manager is not None else None
        current_execution = next(
            (record for record in execution_records if record.execution_id == execution_id),
            None,
        )
        if manifest_attempt is not None and current_execution is not None:
            from davinci_monet.pipeline.checkpoints.models import (
                AttemptStatus,
                ExecutionStatus,
            )

            if current_execution.status is not ExecutionStatus.RUNNING:
                attempt_status = (
                    AttemptStatus.COMPLETED
                    if current_execution.status is ExecutionStatus.COMPLETED
                    else AttemptStatus.FAILED
                )
                manifest_attempt = manifest_attempt.model_copy(
                    update={
                        "status": attempt_status,
                        "completed_at": (
                            current_execution.ended_at
                            if attempt_status is AttemptStatus.COMPLETED
                            else None
                        ),
                    }
                )
        decision_events: dict[str, dict[str, Any]] = {}
        for event in checkpoint_events:
            if event.get("execution_id") != execution_id or event.get("event") not in {
                "checkpoint_decision",
                "checkpoint_finalized",
                "checkpoint_restored",
            }:
                continue
            key = (
                str(event["stage"])
                if event.get("item") is None
                else f"{event['stage']}:{event['item']}"
            )
            decision_events[key] = event
        dispositions: dict[str, int] = {}
        for event in decision_events.values():
            disposition = str(event["disposition"])
            dispositions[disposition] = dispositions.get(disposition, 0) + 1
        checkpointing: dict[str, Any] = {}
        if manager is not None:
            assert manifest_attempt is not None
            checkpointing = {
                "attempt": manifest_attempt.model_dump(mode="json"),
                "executions": [record.model_dump(mode="json") for record in execution_records],
                "receipts": checkpoint_receipts,
                "current_execution_id": execution_id,
                "current_execution_decisions": list(decision_events.values()),
                "disposition_totals": dispositions,
                "receipt_disposition_totals": receipt_dispositions,
                "events": checkpoint_events,
            }
        manifest: dict[str, Any] = {
            "status": status,
            "failed_stages": failed,
            "errors": errors,
            "products": context.metadata.get("product_artifacts", {}),
            "plots": plots,
            "inspection": (
                inspection.data if inspection and isinstance(inspection.data, dict) else {}
            ),
            "stages": {
                name: result.status.name.lower() for name, result in context.results.items()
            },
            "checkpointing": checkpointing,
        }

        path = output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        return self._create_result(
            StageStatus.COMPLETED,
            data={"manifest": str(path)},
            duration=time.time() - start,
        )
