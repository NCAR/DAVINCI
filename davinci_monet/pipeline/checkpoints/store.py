"""Atomic attempt-local storage for executions and checkpoint receipts."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from davinci_monet.pipeline.checkpoints.models import (
    AttemptRecord,
    AttemptStatus,
    CheckpointReceipt,
    ExecutionRecord,
    ExecutionStatus,
)

_EXECUTION_ID_PATTERN = re.compile(r"^e(\d{3,})$")


class AttemptStoreError(RuntimeError):
    """Base error for invalid or unsafe attempt-store operations."""


class AttemptCompletedError(AttemptStoreError):
    """Raised when an operator tries to resume a completed attempt."""


class AttemptLockError(AttemptStoreError):
    """Raised when another execution owns the attempt writer lock."""


def _json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode(
        "utf-8"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temporary(destination: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _atomic_create(destination: Path, payload: bytes) -> None:
    temporary = _write_temporary(destination, payload)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise AttemptStoreError(f"durable record already exists: {destination}") from exc
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace(destination: Path, payload: bytes) -> None:
    temporary = _write_temporary(destination, payload)
    try:
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class AttemptStore:
    """Filesystem owner for one attempt's append-only durable state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    def attempt_path(self) -> Path:
        return self.root / "attempt.json"

    @classmethod
    def initialize(cls, root: str | Path, record: AttemptRecord) -> "AttemptStore":
        """Create one fresh attempt in a missing or empty directory."""
        store = cls(root)
        if Path(record.attempt_root).expanduser().resolve() != store.root:
            raise AttemptStoreError("attempt record root does not match target root")
        if store.root.exists():
            if not store.root.is_dir() or next(store.root.iterdir(), None) is not None:
                raise AttemptStoreError(f"fresh attempt root must be empty: {store.root}")
        else:
            store.root.mkdir(parents=True)
        for directory in (
            store.root / "executions",
            store.root / "state",
            store.root / "checkpoints",
            store.root / "objects" / "sha256",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        _atomic_create(store.attempt_path, _json_bytes(record))
        return store

    @classmethod
    def open_for_resume(
        cls,
        root: str | Path,
        *,
        expected_identities: Mapping[str, str] | None = None,
    ) -> "AttemptStore":
        """Open one initialized, incomplete attempt after identity validation."""
        store = cls(root)
        if not store.attempt_path.is_file():
            raise AttemptStoreError(f"attempt is not initialized: {store.root}")
        record = store.read_attempt()
        if record.status is AttemptStatus.COMPLETED:
            raise AttemptCompletedError(f"attempt is already completed: {store.root}")
        if expected_identities is not None and dict(expected_identities) != record.identities:
            raise AttemptStoreError("attempt identity mismatch")
        return store

    def read_attempt(self) -> AttemptRecord:
        try:
            return AttemptRecord.model_validate_json(self.attempt_path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise AttemptStoreError(f"attempt record is unreadable: {self.attempt_path}") from exc

    def update_attempt(self, record: AttemptRecord) -> None:
        """Atomically update lifecycle state while preserving attempt identity."""
        existing = self.read_attempt()
        if (
            record.attempt_id != existing.attempt_id
            or record.run_id != existing.run_id
            or record.identities != existing.identities
            or record.runtime_versions != existing.runtime_versions
            or record.git_commit != existing.git_commit
            or Path(record.attempt_root).expanduser().resolve() != self.root
        ):
            raise AttemptStoreError("attempt lifecycle update changes immutable identity")
        _atomic_replace(self.attempt_path, _json_bytes(record))

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Acquire this attempt's nonblocking advisory writer lock."""
        lock_path = self.root / "state" / "attempt.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AttemptLockError(f"attempt is already locked: {self.root}") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def write_snapshot(self, value: Mapping[str, Any]) -> Path:
        path = self.root / "state" / "snapshot.json"
        _atomic_replace(path, _json_bytes(dict(value)))
        return path

    def append_event(self, value: Mapping[str, Any]) -> None:
        payload = (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        path = self.root / "state" / "events.jsonl"
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read_events(self) -> list[dict[str, Any]]:
        path = self.root / "state" / "events.jsonl"
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        with path.open("rb") as stream:
            for raw_line in stream:
                if not raw_line.endswith(b"\n"):
                    break
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    break
                if not isinstance(value, dict):
                    break
                events.append(value)
        return events

    def next_execution_id(self) -> str:
        numbers = [
            int(match.group(1))
            for path in (self.root / "executions").iterdir()
            if path.is_dir() and (match := _EXECUTION_ID_PATTERN.fullmatch(path.name))
        ]
        return f"e{max(numbers, default=0) + 1:03d}"

    def publish_execution(self, record: ExecutionRecord) -> Path:
        attempt = self.read_attempt()
        if record.attempt_id != attempt.attempt_id:
            raise AttemptStoreError("execution attempt_id does not match attempt")
        execution_root = self.root / "executions" / record.execution_id
        execution_root.mkdir(exist_ok=True)
        filename = "started.json" if record.status is ExecutionStatus.RUNNING else "finished.json"
        path = execution_root / filename
        _atomic_create(path, _json_bytes(record))
        return path

    def list_executions(self) -> list[ExecutionRecord]:
        records: list[ExecutionRecord] = []
        roots = sorted(
            path
            for path in (self.root / "executions").iterdir()
            if path.is_dir() and _EXECUTION_ID_PATTERN.fullmatch(path.name)
        )
        for root in roots:
            path = root / "finished.json"
            if not path.is_file():
                path = root / "started.json"
            if not path.is_file():
                continue
            try:
                records.append(ExecutionRecord.model_validate_json(path.read_text("utf-8")))
            except (OSError, ValueError) as exc:
                raise AttemptStoreError(f"execution record is unreadable: {path}") from exc
        return records

    def _receipt_root(self, stage: str, item: str | None) -> Path:
        stage_root = self.root / "checkpoints" / stage
        if item is None:
            return stage_root / "stage"
        return stage_root / "items" / item

    def _receipt_paths(self, stage: str, item: str | None) -> list[Path]:
        root = self._receipt_root(stage, item)
        if not root.is_dir():
            return []
        return sorted(
            path
            for path in root.iterdir()
            if path.is_file()
            and path.suffix == ".json"
            and path.stem.startswith("r")
            and path.stem[1:].isdigit()
        )

    def publish_receipt(self, receipt: CheckpointReceipt) -> Path:
        root = self._receipt_root(receipt.stage, receipt.item)
        root.mkdir(parents=True, exist_ok=True)
        generation = len(self._receipt_paths(receipt.stage, receipt.item)) + 1
        finalized = receipt.model_copy(update={"generation": generation})
        path = root / f"r{generation:03d}.json"
        _atomic_create(path, _json_bytes(finalized))
        return path

    def read_receipt(self, stage: str, item: str | None) -> CheckpointReceipt | None:
        paths = self._receipt_paths(stage, item)
        if not paths:
            return None
        path = paths[-1]
        try:
            return CheckpointReceipt.model_validate_json(path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise AttemptStoreError(f"checkpoint receipt is unreadable: {path}") from exc

    def iter_receipts(self) -> Iterator[CheckpointReceipt]:
        checkpoints = self.root / "checkpoints"
        if not checkpoints.is_dir():
            return
        for stage_root in sorted(path for path in checkpoints.iterdir() if path.is_dir()):
            stage_receipt = self.read_receipt(stage_root.name, None)
            if stage_receipt is not None:
                yield stage_receipt
            items_root = stage_root / "items"
            if not items_root.is_dir():
                continue
            for item_root in sorted(path for path in items_root.iterdir() if path.is_dir()):
                item_receipt = self.read_receipt(stage_root.name, item_root.name)
                if item_receipt is not None:
                    yield item_receipt
