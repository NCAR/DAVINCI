"""Unit tests for the daemon RunQueue: serial FIFO with watch_name coalescing.

Exercises the in-memory queue (davinci_monet.daemon.queue.RunQueue) directly —
no filesystem, no subprocess, no scientific stack. TriggerEvents are built from
the shared contract (davinci_monet.daemon.contracts.TriggerEvent).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from davinci_monet.daemon.contracts import TriggerEvent
from davinci_monet.daemon.queue import PendingJob, RunQueue


def _event(
    watch_name: str,
    files: list[str],
    *,
    when: datetime | None = None,
    settle_mode: str = "quiescence",
) -> TriggerEvent:
    """Build a TriggerEvent with sorted absolute-ish paths and a fixed time."""
    return TriggerEvent(
        watch_name=watch_name,
        new_files=sorted(files),
        detected_at=when or datetime(2026, 5, 31, 12, 0, 0),
        settle_mode=settle_mode,  # type: ignore[arg-type]
    )


class TestCoalescing:
    def test_four_rapid_triggers_collapse_to_one_pending(self) -> None:
        q = RunQueue()
        base = datetime(2026, 5, 31, 12, 0, 0)
        # Four rapid triggers for the same watch, none popped in between.
        q.submit(_event("cam", ["/d/a.nc"], when=base))
        q.submit(_event("cam", ["/d/b.nc"], when=base + timedelta(seconds=1)))
        q.submit(_event("cam", ["/d/a.nc", "/d/c.nc"], when=base + timedelta(seconds=2)))
        q.submit(_event("cam", ["/d/d.nc"], when=base + timedelta(seconds=3)))

        assert q.pending_count() == 1
        assert q.pending_names() == ["cam"]

        job = q.next_job()
        assert isinstance(job, PendingJob)
        assert job.watch_name == "cam"
        # Files are the de-duplicated, sorted union of all four triggers.
        assert job.new_files == ["/d/a.nc", "/d/b.nc", "/d/c.nc", "/d/d.nc"]
        # detected_at reflects the most-recent triggering event.
        assert job.detected_at == base + timedelta(seconds=3)
        # Nothing left pending after the single coalesced pop.
        assert q.next_job() is None
        assert q.pending_count() == 0

    def test_coalesce_preserves_settle_mode_and_fifo_slot(self) -> None:
        q = RunQueue()
        # Distinct watch enqueued first, then two triggers for a second watch.
        q.submit(_event("first", ["/x/1.nc"]))
        q.submit(_event("second", ["/y/1.nc"], settle_mode="sentinel"))
        q.submit(_event("second", ["/y/2.nc"], settle_mode="sentinel"))

        # Coalescing does NOT move 'second' ahead of 'first' in the FIFO.
        assert q.pending_names() == ["first", "second"]
        assert q.pending_count() == 2

        first = q.next_job()
        assert first is not None and first.watch_name == "first"
        second = q.next_job()
        assert second is not None and second.watch_name == "second"
        # The coalesced 'second' keeps its (latest) settle_mode.
        assert second.settle_mode == "sentinel"
        assert second.new_files == ["/y/1.nc", "/y/2.nc"]

    def test_empty_queue_next_job_is_none(self) -> None:
        q = RunQueue()
        assert q.pending_count() == 0
        assert q.next_job() is None


class TestFifoOrder:
    def test_distinct_watches_pop_in_submission_order(self) -> None:
        q = RunQueue()
        base = datetime(2026, 5, 31, 12, 0, 0)
        q.submit(_event("alpha", ["/a/1.nc"], when=base))
        q.submit(_event("bravo", ["/b/1.nc"], when=base + timedelta(seconds=1)))
        q.submit(_event("charlie", ["/c/1.nc"], when=base + timedelta(seconds=2)))

        assert q.pending_names() == ["alpha", "bravo", "charlie"]
        popped = [q.next_job(), q.next_job(), q.next_job()]
        assert [j.watch_name for j in popped if j is not None] == [
            "alpha",
            "bravo",
            "charlie",
        ]
        assert q.next_job() is None

    def test_coalesce_does_not_reorder_fifo(self) -> None:
        q = RunQueue()
        q.submit(_event("alpha", ["/a/1.nc"]))
        q.submit(_event("bravo", ["/b/1.nc"]))
        # A late repeat for 'alpha' coalesces in place; 'alpha' must NOT jump
        # behind 'bravo' nor ahead of its original slot.
        q.submit(_event("alpha", ["/a/2.nc"]))
        assert q.pending_names() == ["alpha", "bravo"]
        first = q.next_job()
        assert first is not None and first.watch_name == "alpha"
        assert first.new_files == ["/a/1.nc", "/a/2.nc"]


class TestRunningLifecycle:
    def test_trigger_while_running_requeues_exactly_one_pending(self) -> None:
        q = RunQueue()
        # Initial trigger, then pop it -> the watch is now RUNNING.
        q.submit(_event("cam", ["/d/a.nc"]))
        running_job = q.next_job()
        assert running_job is not None and running_job.watch_name == "cam"
        assert q.is_running("cam") is True
        assert q.pending_count() == 0

        # A new trigger arrives while 'cam' is still running -> re-queue one.
        q.submit(_event("cam", ["/d/b.nc"]))
        assert q.pending_count() == 1
        assert q.pending_names() == ["cam"]

        # Two more rapid triggers during the same running window coalesce into
        # that single pending re-queue (still exactly one pending).
        q.submit(_event("cam", ["/d/c.nc"]))
        q.submit(_event("cam", ["/d/b.nc", "/d/d.nc"]))
        assert q.pending_count() == 1

        # Finish the in-flight run; the re-queued pending is independent of it.
        q.mark_done("cam")
        assert q.is_running("cam") is False

        requeued = q.next_job()
        assert requeued is not None and requeued.watch_name == "cam"
        # The re-queued entry only contains files that arrived AFTER the pop;
        # the originally-running run's file ("/d/a.nc") is not folded back in.
        assert requeued.new_files == ["/d/b.nc", "/d/c.nc", "/d/d.nc"]
        assert q.next_job() is None

    def test_mark_running_and_done_are_idempotent(self) -> None:
        q = RunQueue()
        q.mark_running("cam")
        q.mark_running("cam")
        assert q.running_names() == {"cam"}
        q.mark_done("cam")
        q.mark_done("cam")  # second call must not raise
        assert q.running_names() == set()
        assert q.is_running("cam") is False
