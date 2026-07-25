"""Pipeline-entry tests for stage-boundary interruption and restoration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from davinci_monet.config.schema import MonetConfig
from davinci_monet.core.exceptions import ConfigurationError
from davinci_monet.pipeline.checkpoints.manager import CheckpointManager
from davinci_monet.pipeline.checkpoints.models import (
    AttemptStatus,
    ExecutionStatus,
    ResumeDisposition,
    ResumePlan,
)
from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.pipeline.stages import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.manifest import ManifestStage


class RecordingStage(BaseStage):
    def __init__(
        self,
        name: str,
        calls: dict[str, int],
        *,
        interrupt_once: bool = False,
    ) -> None:
        super().__init__(name)
        self.calls = calls
        self.interrupt_once = interrupt_once

    def execute(self, context: PipelineContext) -> StageResult:
        self.calls[self.name] = self.calls.get(self.name, 0) + 1
        if self.interrupt_once and self.calls[self.name] == 1:
            raise KeyboardInterrupt
        context.metadata[f"value_{self.name}"] = self.calls[self.name]
        return StageResult(
            stage_name=self.name,
            status=StageStatus.COMPLETED,
            data={"value": self.name},
        )


def _config(root: Path) -> MonetConfig:
    return MonetConfig.model_validate(
        {
            "run": {"id": "runner-smoke", "kind": "smoke"},
            "execution": {
                "attempt_root": root,
                "checkpoints": {
                    "mode": "required",
                    "granularity": "stage",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "analysis": {
                "output_dir": root / "output",
                "log_dir": root / "logs",
            },
            "sources": {"fixture": {"type": "generic"}},
        }
    )


def test_runner_resumes_after_stage_boundary_interruption(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    calls: dict[str, int] = {}
    stages = [
        RecordingStage("first", calls),
        RecordingStage("second", calls, interrupt_once=True),
        RecordingStage("third", calls),
    ]
    runner = PipelineRunner(stages=stages, show_progress=False)

    with pytest.raises(KeyboardInterrupt):
        runner.run_from_config(_config(root))

    plan = PipelineRunner(stages=stages, show_progress=False).run_from_config(
        _config(root),
        resume_plan=True,
    )
    assert isinstance(plan, ResumePlan)
    assert [item.disposition for item in plan.items] == [
        ResumeDisposition.RESTORED,
        ResumeDisposition.COMPUTED,
        ResumeDisposition.RECOMPUTED,
    ]
    assert calls == {"first": 1, "second": 1}

    with pytest.raises(ConfigurationError, match="not in this pipeline"):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(
            _config(root),
            resume=True,
            restart_from="missing",
        )
    with pytest.raises(ConfigurationError, match="has no checkpoint receipt"):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(
            _config(root),
            resume=True,
            restart_from="first:missing",
        )

    resumed = PipelineRunner(stages=stages, show_progress=False).run_from_config(
        _config(root),
        resume=True,
    )

    assert resumed.success
    assert calls == {"first": 1, "second": 2, "third": 1}
    first = resumed.get_stage_result("first")
    assert first is not None
    assert first.metadata["resume_disposition"] == "restored"
    assert resumed.context is not None
    manager = resumed.context.checkpoint_manager
    assert manager is not None
    executions = manager.store.list_executions()
    assert [record.status for record in executions] == [
        ExecutionStatus.INTERRUPTED,
        ExecutionStatus.COMPLETED,
    ]
    log_names = [path.name for path in (root / "logs").glob("pipeline_*.md")]
    assert any(name.endswith("_e001.md") for name in log_names)
    assert any(name.endswith("_e002.md") for name in log_names)
    assert any(
        event.get("event") == "checkpoint_restored"
        and event.get("execution_id") == "e002"
        and event.get("stage") == "first"
        for event in manager.store.read_events()
    )


def test_terminal_manifest_refresh_failure_leaves_attempt_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "a002"
    calls: dict[str, int] = {}
    stages = [RecordingStage("first", calls), ManifestStage()]
    original_execute = ManifestStage.execute
    manifest_calls = 0

    def fail_first_terminal_refresh(self, context):  # noqa: ANN001
        nonlocal manifest_calls
        manifest_calls += 1
        if manifest_calls == 2:
            raise RuntimeError("injected terminal refresh failure")
        return original_execute(self, context)

    monkeypatch.setattr(ManifestStage, "execute", fail_first_terminal_refresh)
    with pytest.raises(RuntimeError, match="terminal refresh"):
        PipelineRunner(stages=stages, show_progress=False).run_from_config(_config(root))

    manager = CheckpointManager.create(
        _config(root),
        resume=True,
        read_only=True,
    )
    assert manager is not None
    store = manager.store
    assert store.read_attempt().status is AttemptStatus.IN_PROGRESS
    assert store.list_executions()[0].status is ExecutionStatus.COMPLETED

    resumed = PipelineRunner(stages=stages, show_progress=False).run_from_config(
        _config(root),
        resume=True,
    )

    assert resumed.success
    assert calls == {"first": 1}
    assert store.read_attempt().status is AttemptStatus.COMPLETED
    manifest = json.loads((root / "output" / "manifest.json").read_text("utf-8"))
    assert manifest["checkpointing"]["attempt"]["status"] == "completed"
    assert manifest["checkpointing"]["executions"][-1]["execution_id"] == "e002"
    assert manifest["checkpointing"]["executions"][-1]["status"] == "completed"
