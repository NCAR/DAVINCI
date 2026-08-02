"""Cross-attempt restoration of a pinned pipeline stage boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from davinci_monet.config.schema import MonetConfig
from davinci_monet.pipeline.checkpoints.manager import CheckpointManager, CheckpointRequest
from davinci_monet.pipeline.checkpoints.models import ExecutionStatus, ResumeDisposition
from davinci_monet.pipeline.checkpoints.store import AttemptStore, AttemptStoreError
from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.pipeline.stages.base import PipelineContext, StageResult, StageStatus
from davinci_monet.pipeline.stages.manifest import ManifestStage


def _config(
    root: Path,
    *,
    restore_from: dict[str, Any] | None = None,
) -> MonetConfig:
    checkpoints: dict[str, Any] = {
        "mode": "required",
        "granularity": "item",
        "loaded_sources": True,
        "retain": "all",
    }
    if restore_from is not None:
        checkpoints["restore_from"] = restore_from
    return MonetConfig.model_validate(
        {
            "run": {"id": "checkpoint-restore-smoke", "kind": "smoke"},
            "execution": {
                "attempt_root": root,
                "checkpoints": checkpoints,
            },
            "analysis": {
                "output_dir": root / "output",
                "log_dir": root / "logs",
            },
            "sources": {"unused": {"type": "generic"}},
        }
    )


class _CountingStage:
    def __init__(self, name: str, callback: Callable[[], None]) -> None:
        self.name = name
        self._callback = callback

    def validate(self, context: PipelineContext) -> bool:
        return True

    def execute(self, context: PipelineContext) -> StageResult:
        self._callback()
        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            data={"executed": self.name},
        )


def _seed_statistics_boundary(root: Path) -> tuple[MonetConfig, str]:
    config = _config(root)
    manager = CheckpointManager.create(config)
    assert manager is not None
    manager.begin_execution()
    pairing_request = CheckpointRequest(
        stage="pairing",
        item="model_vs_obs",
        config={"stage": "pairing", "item": "model_vs_obs"},
    )
    manager.publish(
        pairing_request,
        objects=(manager.codecs.write_json({"paired": True}),),
    )
    receipt = manager.publish(
        CheckpointRequest(
            stage="statistics",
            item=None,
            config={"stage": "statistics", "config": config},
            dependencies=(("pairing", "model_vs_obs"),),
        ),
        context_delta={
            "sources": [],
            "paired": [],
            "metadata": {"statistics_kind": "comparison"},
            "result": {
                "stage_name": "statistics",
                "status": "COMPLETED",
                "data": {"restored": True},
                "metadata": {},
                "error": None,
                "error_type": None,
                "traceback_str": None,
                "duration_seconds": 1.0,
            },
        },
    )
    manager.finish_execution(ExecutionStatus.COMPLETED)
    return config, manager.receipt_sha256(receipt)


def test_plot_only_revision_skips_upstream_execution(tmp_path: Path) -> None:
    source_root = tmp_path / "source" / "a001"
    _, receipt_sha256 = _seed_statistics_boundary(source_root)
    target_root = tmp_path / "target" / "a001"
    target = _config(
        target_root,
        restore_from={
            "source_attempt_root": source_root,
            "through_stage": "statistics",
            "receipt_sha256": receipt_sha256,
        },
    )
    calls = {name: 0 for name in ("load_sources", "pairing", "statistics", "plotting")}

    def count(name: str) -> Callable[[], None]:
        def increment() -> None:
            calls[name] += 1

        return increment

    runner = PipelineRunner(
        stages=[
            _CountingStage("load_sources", count("load_sources")),
            _CountingStage("pairing", count("pairing")),
            _CountingStage("statistics", count("statistics")),
            _CountingStage("plotting", count("plotting")),
            ManifestStage(),
        ],
        show_progress=False,
    )

    result = runner.run_from_config(target)

    assert result.success
    assert calls == {
        "load_sources": 0,
        "pairing": 0,
        "statistics": 0,
        "plotting": 1,
    }
    assert [stage.status for stage in result.stage_results] == [
        StageStatus.SKIPPED,
        StageStatus.SKIPPED,
        StageStatus.COMPLETED,
        StageStatus.COMPLETED,
        StageStatus.COMPLETED,
    ]
    restored = result.context.results["statistics"]
    assert restored.data == {"restored": True}
    assert restored.metadata["resume_disposition"] == ResumeDisposition.RESTORED.value
    assert restored.metadata["upstream_checkpoint"]["receipt_sha256"] == receipt_sha256

    target_store = AttemptStore(target_root)
    adopted = target_store.read_receipt("statistics", None)
    assert adopted is not None
    assert adopted.disposition is ResumeDisposition.RESTORED
    assert adopted.context_delta["upstream_checkpoint"]["source_attempt_root"] == str(
        source_root.resolve()
    )
    assert any(event.get("event") == "checkpoint_adopted" for event in target_store.read_events())
    manifest = json.loads((target_root / "output" / "manifest.json").read_text("utf-8"))
    assert any(
        event.get("event") == "checkpoint_adopted"
        and event.get("upstream_checkpoint", {}).get("receipt_sha256") == receipt_sha256
        for event in manifest["checkpointing"]["events"]
    )
    statistics_receipt = next(
        receipt
        for receipt in manifest["checkpointing"]["receipts"]
        if receipt["stage"] == "statistics" and receipt["item"] is None
    )
    assert statistics_receipt["receipt_sha256"] == CheckpointManager.receipt_sha256(adopted)


def test_bad_restore_pin_does_not_initialize_target_attempt(tmp_path: Path) -> None:
    source_root = tmp_path / "source" / "a001"
    _seed_statistics_boundary(source_root)
    target_root = tmp_path / "target" / "a001"
    target = _config(
        target_root,
        restore_from={
            "source_attempt_root": source_root,
            "through_stage": "statistics",
            "receipt_sha256": "0" * 64,
        },
    )

    with pytest.raises(AttemptStoreError, match="SHA-256 mismatch"):
        CheckpointManager.create(target)

    assert not target_root.exists()


def test_corrupt_restore_dependency_does_not_initialize_target_attempt(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "a001"
    _, receipt_sha256 = _seed_statistics_boundary(source_root)
    source_store = AttemptStore(source_root)
    pairing = source_store.read_receipt("pairing", "model_vs_obs")
    assert pairing is not None
    Path(pairing.objects[0].paths[0]).write_text("corrupt", encoding="utf-8")
    target_root = tmp_path / "target" / "a001"
    target = _config(
        target_root,
        restore_from={
            "source_attempt_root": source_root,
            "through_stage": "statistics",
            "receipt_sha256": receipt_sha256,
        },
    )

    with pytest.raises(AttemptStoreError, match="invalid object"):
        CheckpointManager.create(target)

    assert not target_root.exists()
