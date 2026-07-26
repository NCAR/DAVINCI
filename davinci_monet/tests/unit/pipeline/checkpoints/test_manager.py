"""Tests for attempt management and deterministic resume planning."""

from __future__ import annotations

from pathlib import Path

import pytest

from davinci_monet.config.schema import MonetConfig
from davinci_monet.pipeline.checkpoints.manager import (
    REASON_CORRUPT,
    REASON_DEPENDENCY,
    REASON_RESTART,
    CheckpointManager,
    CheckpointRequest,
)
from davinci_monet.pipeline.checkpoints.models import (
    ExecutionStatus,
    ResumeDisposition,
)
from davinci_monet.pipeline.checkpoints.store import AttemptStoreError


def _config(root: Path, *, debug: bool = False) -> MonetConfig:
    return MonetConfig.model_validate(
        {
            "run": {"id": "checkpoint-smoke", "kind": "smoke"},
            "execution": {
                "attempt_root": root,
                "checkpoints": {
                    "mode": "required",
                    "granularity": "item",
                    "loaded_sources": True,
                    "retain": "all",
                },
            },
            "analysis": {
                "output_dir": root / "output",
                "log_dir": root / "logs",
                "debug": debug,
            },
        }
    )


def _request(
    stage: str,
    item: str,
    *,
    dependencies: tuple[tuple[str, str | None], ...] = (),
) -> CheckpointRequest:
    return CheckpointRequest(
        stage=stage,
        item=item,
        config={"stage": stage, "item": item},
        dependencies=dependencies,
    )


def _seed(root: Path) -> tuple[CheckpointRequest, CheckpointRequest, CheckpointRequest]:
    manager = CheckpointManager.create(_config(root))
    assert manager is not None
    manager.begin_execution()
    first = _request("source", "a")
    second = _request("analysis", "b", dependencies=(("source", "a"),))
    branch = _request("analysis", "independent")
    manager.publish(first, context_delta={"value": 1})
    manager.publish(second, context_delta={"value": 2})
    manager.publish(branch, context_delta={"value": 3})
    manager.finish_execution(ExecutionStatus.FAILED, error="synthetic interruption")
    return first, second, branch


def test_fully_reusable_linear_pipeline_and_independent_branch(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    first, second, branch = _seed(root)

    manager = CheckpointManager.create(_config(root), resume=True, read_only=True)
    assert manager is not None
    plan = manager.plan((first, second, branch))

    assert [item.disposition for item in plan.items] == [
        ResumeDisposition.RESTORED,
        ResumeDisposition.RESTORED,
        ResumeDisposition.RESTORED,
    ]


def test_invalid_middle_item_recomputes_only_dependents(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    first, second, branch = _seed(root)
    resumed = CheckpointManager.create(_config(root), resume=True, read_only=True)
    assert resumed is not None
    receipt = resumed.store.read_receipt("source", "a")
    assert receipt is not None
    data_path = Path(receipt.context_delta.get("unused", root / "missing"))
    assert not data_path.exists()

    changed_first = CheckpointRequest(
        stage=first.stage,
        item=first.item,
        config={"changed": True},
    )
    manager = CheckpointManager.create(_config(root), resume=True, read_only=True)
    assert manager is not None
    plan = manager.plan((changed_first, second, branch))

    assert [item.disposition for item in plan.items] == [
        ResumeDisposition.RECOMPUTED,
        ResumeDisposition.RECOMPUTED,
        ResumeDisposition.RESTORED,
    ]
    assert plan.items[1].reason == REASON_DEPENDENCY


def test_corrupt_object_is_not_reused(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    manager = CheckpointManager.create(_config(root))
    assert manager is not None
    manager.begin_execution()
    request = _request("stats", "one")
    obj = manager.codecs.write_json({"value": 1})
    manager.publish(request, objects=(obj,))
    manager.finish_execution(ExecutionStatus.FAILED)
    Path(obj.paths[0]).write_text("changed", encoding="utf-8")

    resumed = CheckpointManager.create(_config(root), resume=True, read_only=True)
    assert resumed is not None
    item = resumed.plan((request,)).items[0]

    assert item.disposition is ResumeDisposition.RECOMPUTED
    assert item.reason == REASON_CORRUPT


def test_restart_from_invalidates_target_and_downstream(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    first, second, branch = _seed(root)
    downstream = _request(
        "plotting",
        "dependent",
        dependencies=(("analysis", "b"),),
    )
    writer = CheckpointManager.create(_config(root), resume=True)
    assert writer is not None
    writer.begin_execution()
    writer.publish(downstream, context_delta={"value": 4})
    writer.finish_execution(ExecutionStatus.FAILED, error="seed downstream")
    manager = CheckpointManager.create(
        _config(root),
        resume=True,
        read_only=True,
        restart_from="analysis:b",
    )
    assert manager is not None

    plan = manager.plan((first, second, branch, downstream))

    assert plan.items[0].disposition is ResumeDisposition.RESTORED
    assert plan.items[1].reason == REASON_RESTART
    assert plan.items[2].disposition is ResumeDisposition.RESTORED
    assert plan.items[3].reason == REASON_DEPENDENCY
    assert plan.items[3].invalidated_by == ("analysis:b",)


def test_attempt_identity_mismatch_blocks_resume(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    _seed(root)

    with pytest.raises(AttemptStoreError, match="identity mismatch"):
        CheckpointManager.create(_config(root, debug=True), resume=True)


def test_runtime_environment_mismatch_blocks_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "a001"
    _seed(root)
    monkeypatch.setattr(
        "davinci_monet.pipeline.checkpoints.manager.runtime_versions",
        lambda: {"python": "changed", "numpy": "changed"},
    )

    with pytest.raises(AttemptStoreError, match="identity mismatch"):
        CheckpointManager.create(_config(root), resume=True)

    planner = CheckpointManager.create(_config(root), resume=True, read_only=True)
    assert planner is not None
    plan = planner.plan(())
    assert plan.blocked_reasons == ("attempt_identity_mismatch:environment_sha256",)


def test_read_only_plan_reports_identity_mismatch_without_writing(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    requests = _seed(root)
    before = {
        path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }

    manager = CheckpointManager.create(
        _config(root, debug=True),
        resume=True,
        read_only=True,
    )
    assert manager is not None
    plan = manager.plan(requests)

    assert plan.blocked
    assert plan.items == ()
    assert plan.blocked_reasons == ("attempt_identity_mismatch:config_sha256",)
    after = {
        path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_resume_plan_does_not_write_attempt_tree(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    requests = _seed(root)
    before = {
        path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }

    manager = CheckpointManager.create(_config(root), resume=True, read_only=True)
    assert manager is not None
    manager.plan(requests)
    after = {
        path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_attempt_plan_inventories_item_and_stage_receipts(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    manager = CheckpointManager.create(_config(root))
    assert manager is not None
    manager.begin_execution()
    item = _request("load_sources", "model")
    manager.publish(item, context_delta={"value": 1})
    stage = CheckpointRequest(
        stage="load_sources",
        item=None,
        config={"stage": "load_sources"},
        dependencies=(("load_sources", "model"),),
    )
    manager.publish(stage, context_delta={"value": 2})
    manager.finish_execution(ExecutionStatus.FAILED, error="keep attempt open")

    resumed = CheckpointManager.create(_config(root), resume=True, read_only=True)
    assert resumed is not None
    plan = resumed.plan_attempt((stage,))

    assert [(entry.item, entry.disposition) for entry in plan.items] == [
        ("model", ResumeDisposition.RESTORED),
        (None, ResumeDisposition.RESTORED),
    ]
