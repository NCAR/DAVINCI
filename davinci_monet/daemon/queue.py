"""Serial FIFO run queue with watch_name coalescing for the daemon supervisor.

The supervisor's watcher emits ``TriggerEvent``s; this queue holds the pending
runs and collapses repeat triggers for the same ``watch_name`` into a single
pending entry (spec decision #5; "Data flow" step 2). It is an in-memory,
single-thread structure accessed only from the supervisor loop. Concurrency
(``max_concurrent``) is enforced by the supervisor/dispatcher, NOT here:
``next_job`` simply returns the oldest pending entry (FIFO) or ``None``.

Isolation invariant: this module imports stdlib + the daemon contracts only.
It never imports matplotlib/xarray/monet/the pipeline.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

from davinci_monet.daemon.contracts import SettleMode, TriggerEvent

__all__ = ["PendingJob", "RunQueue"]


@dataclass
class PendingJob:
    """A coalesced pending run for one watch (queue-local, not a contract type).

    ``new_files`` is the de-duplicated, sorted union of every coalesced
    ``TriggerEvent.new_files``. ``detected_at`` / ``settle_mode`` track the most
    recent triggering event. The dispatcher consumes this to build a JobSpec.
    """

    watch_name: str
    new_files: list[str] = field(default_factory=list)
    detected_at: datetime | None = None
    settle_mode: SettleMode = "quiescence"
    _file_set: set[str] = field(default_factory=set, repr=False)

    def merge(self, event: TriggerEvent) -> None:
        """Fold another TriggerEvent for the same watch into this entry."""
        for path in event.new_files:
            self._file_set.add(path)
        self.new_files = sorted(self._file_set)
        # Latest event wins for detected_at / settle_mode.
        if self.detected_at is None or event.detected_at >= self.detected_at:
            self.detected_at = event.detected_at
            self.settle_mode = event.settle_mode


class RunQueue:
    """In-memory serial FIFO of pending runs, coalescing by ``watch_name``.

    Lifecycle per watch: a ``submit`` either creates a new pending entry (FIFO
    tail) or coalesces into the existing pending entry for that watch. ``next_job``
    pops the oldest pending entry and marks the watch RUNNING. ``mark_running`` /
    ``mark_done`` let the supervisor drive state when it pops/spawns/finishes a
    job out of band.
    """

    def __init__(self) -> None:
        # Insertion-ordered: watch_name -> PendingJob. OrderedDict gives O(1)
        # membership test for coalescing while preserving FIFO order.
        self._pending: "OrderedDict[str, PendingJob]" = OrderedDict()
        self._running: set[str] = set()

    # ---- producer side ----------------------------------------------------
    def submit(self, event: TriggerEvent) -> bool:
        """Enqueue a TriggerEvent, coalescing into an existing pending entry.

        Returns ``True`` if the event was coalesced into an existing pending
        entry, ``False`` if it created a new pending entry. A watch that is
        currently RUNNING (but has no pending entry) creates a fresh pending
        entry — the in-flight run is never mutated.
        """
        existing = self._pending.get(event.watch_name)
        if existing is not None:
            existing.merge(event)
            return True
        job = PendingJob(watch_name=event.watch_name)
        job.merge(event)
        self._pending[event.watch_name] = job
        return False

    # ---- consumer side ----------------------------------------------------
    def next_job(self) -> PendingJob | None:
        """Pop the oldest pending entry (FIFO) and mark its watch RUNNING.

        Returns ``None`` if no pending entries. Does NOT consider running count;
        the supervisor decides whether it is allowed to start another worker.
        """
        if not self._pending:
            return None
        _, job = self._pending.popitem(last=False)
        self._running.add(job.watch_name)
        return job

    def mark_running(self, watch_name: str) -> None:
        """Record that ``watch_name`` has an in-flight run (idempotent)."""
        self._running.add(watch_name)

    def mark_done(self, watch_name: str) -> None:
        """Clear the RUNNING marker for ``watch_name`` (idempotent)."""
        self._running.discard(watch_name)

    # ---- introspection (backs status/top) ---------------------------------
    def pending_count(self) -> int:
        return len(self._pending)

    def pending_names(self) -> list[str]:
        """Pending watch names in FIFO order."""
        return list(self._pending.keys())

    def running_names(self) -> set[str]:
        return set(self._running)

    def is_running(self, watch_name: str) -> bool:
        return watch_name in self._running

    # ---- QueueLike protocol compatibility -----------------------------------
    def pop(self) -> tuple[str, list[str]] | None:
        """Pop the oldest pending entry and return (watch_name, new_files).

        Alias for ``next_job()`` that conforms to the ``QueueLike`` protocol
        used by the supervisor. Returns ``None`` if no entries are pending.
        """
        job = self.next_job()
        if job is None:
            return None
        return job.watch_name, job.new_files

    def __len__(self) -> int:
        """Return the number of pending (not yet started) entries."""
        return self.pending_count()
