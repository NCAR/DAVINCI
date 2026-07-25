"""Attempt-local atomic checkpoint store."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from davinci_monet.pipeline.checkpoints.models import (
    ATTEMPT_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    EXECUTION_SCHEMA_VERSION,
    AttemptRecord,
    AttemptStatus,
    CheckpointReceipt,
    CheckpointStatus,
    ExecutionRecord,
    ExecutionStatus,
    ResumeDisposition,
)
from davinci_monet.pipeline.checkpoints.store import (
    AttemptCompletedError,
    AttemptLockError,
    AttemptStore,
    AttemptStoreError,
)

NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


def _attempt(root: Path, *, status: AttemptStatus = AttemptStatus.IN_PROGRESS) -> AttemptRecord:
    return AttemptRecord(
        schema_version=ATTEMPT_SCHEMA_VERSION,
        run_id="aod-eof-wavelet-r01",
        run_kind="production",
        attempt_id="a001",
        attempt_root=str(root),
        status=status,
        config_path="/repo/control.yaml",
        identities={
            "config_sha256": "a" * 64,
            "code_sha256": "b" * 64,
            "environment_sha256": "d" * 64,
            "source_inventory_sha256": "c" * 64,
        },
        runtime_versions={"python": "3.13.14", "numpy": "2.3.0"},
        git_commit="e" * 40,
        created_at=NOW,
        completed_at=NOW if status is AttemptStatus.COMPLETED else None,
        host="derecho1",
    )


def _execution(status: ExecutionStatus, *, execution_id: str = "e001") -> ExecutionRecord:
    return ExecutionRecord(
        schema_version=EXECUTION_SCHEMA_VERSION,
        execution_id=execution_id,
        attempt_id="a001",
        status=status,
        started_at=NOW,
        ended_at=None if status is ExecutionStatus.RUNNING else NOW,
        host="derecho1",
        pid=123,
    )


def test_initialize_and_open_incomplete_attempt(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    store = AttemptStore.initialize(root, _attempt(root))

    assert store.read_attempt().attempt_id == "a001"
    assert (root / "executions").is_dir()
    assert (root / "state").is_dir()
    assert (root / "checkpoints").is_dir()
    assert (root / "objects" / "sha256").is_dir()
    assert AttemptStore.open_for_resume(root).read_attempt().status is AttemptStatus.IN_PROGRESS


def test_initialize_rejects_unrelated_existing_content(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    root.mkdir()
    (root / "unrelated.txt").write_text("do not overwrite\n")

    with pytest.raises(AttemptStoreError, match="empty"):
        AttemptStore.initialize(root, _attempt(root))

    assert (root / "unrelated.txt").read_text() == "do not overwrite\n"


def test_resume_requires_initialized_incomplete_attempt(tmp_path: Path) -> None:
    with pytest.raises(AttemptStoreError, match="not initialized"):
        AttemptStore.open_for_resume(tmp_path / "a001")

    root = tmp_path / "a001"
    store = AttemptStore.initialize(root, _attempt(root))
    store.update_attempt(_attempt(root, status=AttemptStatus.COMPLETED))
    with pytest.raises(AttemptCompletedError, match="completed"):
        AttemptStore.open_for_resume(root)


def test_resume_rejects_expected_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    AttemptStore.initialize(root, _attempt(root))

    with pytest.raises(AttemptStoreError, match="identity mismatch"):
        AttemptStore.open_for_resume(
            root,
            expected_identities={
                "config_sha256": "f" * 64,
                "code_sha256": "b" * 64,
                "environment_sha256": "d" * 64,
                "source_inventory_sha256": "c" * 64,
            },
        )


def test_attempt_lock_allows_only_one_writer(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    first = AttemptStore.initialize(root, _attempt(root))
    second = AttemptStore.open_for_resume(root)

    with first.lock():
        with pytest.raises(AttemptLockError, match="already locked"):
            with second.lock():
                pass


def test_atomic_snapshot_failure_exposes_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "a001"
    store = AttemptStore.initialize(root, _attempt(root))

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("davinci_monet.pipeline.checkpoints.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.write_snapshot({"sequence": 1})

    assert not (root / "state" / "snapshot.json").exists()
    assert not list((root / "state").glob(".*.tmp"))


def test_journal_ignores_orphan_temp_and_truncated_tail(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    store = AttemptStore.initialize(root, _attempt(root))
    store.append_event({"sequence": 1, "event": "started"})
    store.append_event({"sequence": 2, "event": "checkpoint"})
    journal = root / "state" / "events.jsonl"
    with journal.open("ab") as stream:
        stream.write(b'{"sequence":3')
    (root / "state" / ".snapshot.json.orphan.tmp").write_text("partial")

    assert store.read_events() == [
        {"event": "started", "sequence": 1},
        {"event": "checkpoint", "sequence": 2},
    ]


def test_execution_ids_are_monotonic_and_transitions_append(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    store = AttemptStore.initialize(root, _attempt(root))

    assert store.next_execution_id() == "e001"
    store.publish_execution(_execution(ExecutionStatus.RUNNING))
    assert store.next_execution_id() == "e002"
    store.publish_execution(_execution(ExecutionStatus.COMPLETED))
    store.publish_execution(_execution(ExecutionStatus.RUNNING, execution_id="e002"))

    records = store.list_executions()
    assert [record.execution_id for record in records] == ["e001", "e002"]
    assert records[0].status is ExecutionStatus.COMPLETED
    assert (root / "executions" / "e001" / "started.json").is_file()
    assert (root / "executions" / "e001" / "finished.json").is_file()


def test_checkpoint_receipts_append_generations_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "a001"
    store = AttemptStore.initialize(root, _attempt(root))
    receipt = CheckpointReceipt(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        stage="load_sources",
        item="model",
        status=CheckpointStatus.FINALIZED,
        identity_sha256="a" * 64,
        context_delta={"label": "model"},
        disposition=ResumeDisposition.COMPUTED,
        execution_id="e001",
        finalized_at=NOW,
    )

    first_path = store.publish_receipt(receipt)
    second_path = store.publish_receipt(receipt.model_copy(update={"identity_sha256": "b" * 64}))

    latest = store.read_receipt("load_sources", "model")
    assert latest is not None
    assert latest.generation == 2
    assert latest.identity_sha256 == "b" * 64
    assert first_path == root / "checkpoints" / "load_sources" / "items" / "model" / "r001.json"
    assert second_path == root / "checkpoints" / "load_sources" / "items" / "model" / "r002.json"
    assert first_path.is_file()
    assert second_path.is_file()
    assert [item.generation for item in store.iter_receipts()] == [2]
