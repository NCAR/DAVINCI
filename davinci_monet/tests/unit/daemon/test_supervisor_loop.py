"""Unit tests for the supervisor coalesce + dispatch loop (no real pipeline)."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, Optional

from davinci_monet.daemon.config import DaemonConfig, WatchesFile, WatchRule
from davinci_monet.daemon.contracts import JobSpec, JobStatus, TriggerEvent


class FakeClock:
    """Deterministic monotonic clock implementing contracts.Clock."""

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds


class FakeWatcher:
    """Emits a pre-seeded list of TriggerEvents once, then nothing."""

    def __init__(self, events: list[TriggerEvent]) -> None:
        self._pending = list(events)

    def poll(self) -> list[TriggerEvent]:
        out, self._pending = self._pending, []
        return out

    def set_rules(self, rules) -> None:
        # Satisfies the WatcherLike Protocol (set_rules added for `reload`).
        self.rules = list(rules)


class FakeQueue:
    """Coalescing FIFO: repeat submits for the same watch_name union new_files."""

    def __init__(self) -> None:
        self._items: dict[str, list[str]] = {}
        self._order: list[str] = []
        self.mark_done_calls: list[str] = []  # records watch_names passed to mark_done

    def submit(self, event: TriggerEvent) -> bool:
        coalesced = event.watch_name in self._items
        if not coalesced:
            self._order.append(event.watch_name)
            self._items[event.watch_name] = []
        merged = sorted(set(self._items[event.watch_name]) | set(event.new_files))
        self._items[event.watch_name] = merged
        return coalesced

    def pop(self) -> Optional[tuple[str, list[str]]]:
        if not self._order:
            return None
        name = self._order.pop(0)
        return name, self._items.pop(name)

    def mark_done(self, watch_name: str) -> None:
        """Track that the supervisor cleared the RUNNING marker for watch_name."""
        self.mark_done_calls.append(watch_name)

    def __len__(self) -> int:
        return len(self._order)


class FakeStateStore:
    """In-memory StateStore stand-in covering the methods the loop calls."""

    def __init__(self) -> None:
        self.jobs: dict[int, dict[str, Any]] = {}
        self._next = 1

    def create_job(self, watch_name, config_path, on_fire, files) -> int:
        jid = self._next
        self._next += 1
        self.jobs[jid] = {
            "watch_name": watch_name,
            "config_path": config_path,
            "on_fire": on_fire,
            "files": list(files),
            "status": JobStatus.QUEUED,
        }
        return jid

    def mark_running(self, job_id, started_at=None) -> None:
        self.jobs[job_id]["status"] = JobStatus.RUNNING

    def mark_completed(self, job_id, exit_code, log_path, result_summary, ended_at=None) -> None:
        self.jobs[job_id]["status"] = JobStatus.COMPLETED
        self.jobs[job_id]["exit_code"] = exit_code
        self.jobs[job_id]["result_summary"] = result_summary

    def mark_failed(self, job_id, exit_code, error, log_path=None, ended_at=None) -> None:
        self.jobs[job_id]["status"] = JobStatus.FAILED
        self.jobs[job_id]["error"] = error

    def get_job(self, job_id: int) -> Optional[Any]:
        return self.jobs.get(job_id)


def _rule(name: str, run: str = "/tmp/cfg.yaml") -> WatchRule:
    return WatchRule(name=name, watch="/tmp/incoming/*.nc", run=run)


def _watches_file() -> WatchesFile:
    return WatchesFile(
        daemon=DaemonConfig(max_concurrent=1),
        watches={"cam": _rule("cam")},
    )


class DispatchResult:
    """Minimal stand-in for what spawn_worker returns to the supervisor."""

    def __init__(self, success: bool = True) -> None:
        self.success = success
        self.exit_code = 0 if success else 1
        self.log_path = "/tmp/run.md"
        self.result_summary = {"stages": 7}
        self.output_dir = "/tmp/out"
        self.plots: list[str] = []
        self.error = None if success else "boom"


def test_isolation_invariant_no_sci_stack_imported() -> None:
    """Importing the supervisor must NOT pull in matplotlib/xarray/the pipeline.

    Runs in a subprocess so the check is not polluted by other test modules
    that have already imported the sci stack in this process.
    """
    import subprocess

    check = (
        "import sys, davinci_monet.daemon.supervisor; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib leaked'; "
        "assert 'xarray' not in sys.modules, 'xarray leaked'; "
        "assert 'davinci_monet.pipeline.runner' not in sys.modules, 'pipeline leaked'"
    )
    result = subprocess.run(
        [sys.executable, "-c", check],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Isolation check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_one_coalesced_trigger_dispatches_exactly_one_job() -> None:
    """Two TriggerEvents for the same watch in one tick -> one coalesced job."""
    from davinci_monet.daemon.supervisor import Supervisor

    dispatched: list[JobSpec] = []

    def fake_dispatch(spec: JobSpec, daemon_cfg, on_event=None) -> DispatchResult:
        dispatched.append(spec)
        return DispatchResult(success=True)

    state = FakeStateStore()
    queue = FakeQueue()
    wf = _watches_file()
    events = [
        TriggerEvent(
            watch_name="cam",
            new_files=["/tmp/incoming/a.nc"],
            detected_at=datetime(2026, 5, 31, 12, 0, 0),
            settle_mode="quiescence",
        ),
        TriggerEvent(
            watch_name="cam",
            new_files=["/tmp/incoming/b.nc"],
            detected_at=datetime(2026, 5, 31, 12, 0, 1),
            settle_mode="quiescence",
        ),
    ]
    sup = Supervisor(
        watches_file=wf,
        watcher=FakeWatcher(events),
        queue=queue,
        dispatcher=fake_dispatch,
        state=state,
        notifier=None,
        clock=FakeClock(),
    )

    sup.run_once()

    assert len(dispatched) == 1
    spec = dispatched[0]
    assert spec.watch_name == "cam"
    assert spec.config_path.endswith("cfg.yaml")
    assert spec.new_files == ["/tmp/incoming/a.nc", "/tmp/incoming/b.nc"]
    assert spec.job_id == 1
    assert state.jobs[1]["status"] == JobStatus.COMPLETED


def test_failed_dispatch_marks_job_failed_and_loop_continues() -> None:
    """A failing worker -> job FAILED; the supervisor does not raise."""
    from davinci_monet.daemon.supervisor import Supervisor

    def fake_dispatch(spec: JobSpec, daemon_cfg, on_event=None) -> DispatchResult:
        return DispatchResult(success=False)

    state = FakeStateStore()
    sup = Supervisor(
        watches_file=_watches_file(),
        watcher=FakeWatcher(
            [
                TriggerEvent(
                    watch_name="cam",
                    new_files=["/tmp/incoming/a.nc"],
                    detected_at=datetime(2026, 5, 31, 12, 0, 0),
                    settle_mode="quiescence",
                )
            ]
        ),
        queue=FakeQueue(),
        dispatcher=fake_dispatch,
        state=state,
        notifier=None,
        clock=FakeClock(),
    )

    sup.run_once()

    assert state.jobs[1]["status"] == JobStatus.FAILED
    assert state.jobs[1]["error"] == "boom"


def test_max_concurrent_one_dispatches_serially() -> None:
    """With max_concurrent=1, two distinct watches need two ticks (one job/tick)."""
    from davinci_monet.daemon.supervisor import Supervisor

    dispatched: list[str] = []

    def fake_dispatch(spec: JobSpec, daemon_cfg, on_event=None) -> DispatchResult:
        dispatched.append(spec.watch_name)
        return DispatchResult(success=True)

    wf = WatchesFile(
        daemon=DaemonConfig(max_concurrent=1),
        watches={"cam": _rule("cam"), "modis": _rule("modis", run="/tmp/m.yaml")},
    )
    queue = FakeQueue()
    sup = Supervisor(
        watches_file=wf,
        watcher=FakeWatcher(
            [
                TriggerEvent(
                    watch_name="cam",
                    new_files=["/tmp/incoming/a.nc"],
                    detected_at=datetime(2026, 5, 31, 12, 0, 0),
                    settle_mode="quiescence",
                ),
                TriggerEvent(
                    watch_name="modis",
                    new_files=["/tmp/incoming/m.hdf"],
                    detected_at=datetime(2026, 5, 31, 12, 0, 0),
                    settle_mode="quiescence",
                ),
            ]
        ),
        queue=queue,
        dispatcher=fake_dispatch,
        state=FakeStateStore(),
        notifier=None,
        clock=FakeClock(),
    )

    sup.run_once()  # tick 1: drains both into queue, dispatches one
    assert dispatched == ["cam"]
    assert len(queue) == 1

    sup.run_once()  # tick 2: dispatches the second
    assert dispatched == ["cam", "modis"]
    assert len(queue) == 0


def test_result_summary_persisted_with_output_dir_and_plots() -> None:
    """_record_result folds output_dir and plots into the persisted result_summary.

    This is the Phase-7 wiring contract: notify_outcome reads
    result_summary['output_dir'] and result_summary['plots'] for the iCloud
    copy; if they are absent the copy silently does nothing.
    """
    from davinci_monet.daemon.supervisor import Supervisor

    def fake_dispatch(spec: JobSpec, daemon_cfg, on_event=None) -> DispatchResult:
        r = DispatchResult(success=True)
        r.result_summary = {"stages": 4, "duration_s": 12.3}
        r.output_dir = "/tmp/analysis/output"
        r.plots = ["/tmp/analysis/output/scatter.png", "/tmp/analysis/output/bias.png"]
        return r

    state = FakeStateStore()
    sup = Supervisor(
        watches_file=_watches_file(),
        watcher=FakeWatcher(
            [
                TriggerEvent(
                    watch_name="cam",
                    new_files=["/tmp/incoming/a.nc"],
                    detected_at=datetime(2026, 5, 31, 12, 0, 0),
                    settle_mode="quiescence",
                )
            ]
        ),
        queue=FakeQueue(),
        dispatcher=fake_dispatch,
        state=state,
        notifier=None,
        clock=FakeClock(),
    )

    sup.run_once()

    assert state.jobs[1]["status"] == JobStatus.COMPLETED
    rs = state.jobs[1]["result_summary"]
    assert rs is not None, "result_summary must be persisted (not None)"
    assert rs["output_dir"] == "/tmp/analysis/output", "output_dir must be in result_summary"
    assert rs["plots"] == [
        "/tmp/analysis/output/scatter.png",
        "/tmp/analysis/output/bias.png",
    ], "plots must be in result_summary"
    assert rs["stages"] == 4, "original result_summary keys must be preserved"


def test_mark_done_called_after_successful_dispatch() -> None:
    """Supervisor calls queue.mark_done(watch_name) after a job finishes.

    Without mark_done the RunQueue._running set retains the watch name forever,
    causing is_running/running_names to report the watch as still running.
    """
    from davinci_monet.daemon.supervisor import Supervisor

    def fake_dispatch(spec: JobSpec, daemon_cfg, on_event=None) -> DispatchResult:
        return DispatchResult(success=True)

    state = FakeStateStore()
    queue = FakeQueue()
    sup = Supervisor(
        watches_file=_watches_file(),
        watcher=FakeWatcher(
            [
                TriggerEvent(
                    watch_name="cam",
                    new_files=["/tmp/incoming/a.nc"],
                    detected_at=datetime(2026, 5, 31, 12, 0, 0),
                    settle_mode="quiescence",
                )
            ]
        ),
        queue=queue,
        dispatcher=fake_dispatch,
        state=state,
        notifier=None,
        clock=FakeClock(),
    )

    sup.run_once()

    assert queue.mark_done_calls == [
        "cam"
    ], "mark_done must be called with the watch_name after dispatch completes"


def test_mark_done_called_after_failed_dispatch() -> None:
    """Supervisor calls queue.mark_done even when the worker fails.

    A failed worker should not leave the watch stuck in the RUNNING state.
    """
    from davinci_monet.daemon.supervisor import Supervisor

    def fake_dispatch(spec: JobSpec, daemon_cfg, on_event=None) -> DispatchResult:
        return DispatchResult(success=False)

    state = FakeStateStore()
    queue = FakeQueue()
    sup = Supervisor(
        watches_file=_watches_file(),
        watcher=FakeWatcher(
            [
                TriggerEvent(
                    watch_name="cam",
                    new_files=["/tmp/incoming/a.nc"],
                    detected_at=datetime(2026, 5, 31, 12, 0, 0),
                    settle_mode="quiescence",
                )
            ]
        ),
        queue=queue,
        dispatcher=fake_dispatch,
        state=state,
        notifier=None,
        clock=FakeClock(),
    )

    sup.run_once()

    assert state.jobs[1]["status"] == JobStatus.FAILED
    assert queue.mark_done_calls == ["cam"], "mark_done must be called even when the worker fails"


class FakeWorkerRunResult:
    """Minimal WorkerRunResult stand-in with a terminal result ProgressEvent.

    Mirrors the shape of dispatcher.WorkerRunResult: exit_code int, stderr str,
    a `success` property (exit_code == 0), and a `result_event` property that
    returns the terminal ProgressEvent (or None).
    """

    def __init__(
        self,
        *,
        success: bool = True,
        output_dir: str = "/tmp/out",
        plots: Optional[list[str]] = None,
        summary: Optional[dict] = None,
        log_path: str = "/tmp/run.md",
        error: Optional[str] = None,
        stderr: str = "",
    ) -> None:
        from davinci_monet.daemon.contracts import ProgressEvent

        self.job_id = 1
        self.exit_code = 0 if success else 1
        self.stderr = stderr
        # Build a minimal ProgressEvent that _WorkerResultAdapter reads.
        self._result_event = ProgressEvent(
            kind="result",
            job_id=1,
            ts=datetime(2026, 5, 31, 12, 0, 5),
            success=success,
            output_dir=output_dir,
            plots=plots or [],
            summary=summary or {"stages": 3},
            log_path=log_path,
            error=error,
            total_duration_seconds=5.0,
        )

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def result_event(self):  # type: ignore[override]
        return self._result_event


def test_worker_result_adapter_extracts_output_dir_and_plots() -> None:
    """_WorkerResultAdapter translates WorkerRunResult -> shape _record_result reads.

    This is the load-bearing adapter in build_supervisor (supervisor.py).
    WorkerRunResult has NO result_summary/output_dir/plots/log_path attributes;
    they live on the terminal 'result' ProgressEvent. The adapter must extract
    them so _record_result (and thus notify_outcome) can see them.
    """
    # Import and exercise the adapter logic directly via build_supervisor's
    # internal _dispatch closure by injecting a fake spawn_worker that returns
    # a FakeWorkerRunResult.  We call the adapter through the full
    # Supervisor._dispatch_one path so the integration is real.
    from unittest.mock import patch

    from davinci_monet.daemon.supervisor import build_supervisor

    fake_run_result = FakeWorkerRunResult(
        success=True,
        output_dir="/icloud/Analysis/output",
        plots=["/icloud/Analysis/output/scatter.png"],
        summary={"stages": 5, "N": 42},
        log_path="/logs/job1.md",
    )

    state = FakeStateStore()

    with patch(
        "davinci_monet.daemon.dispatcher.spawn_worker",
        return_value=fake_run_result,
    ):
        # We build a real supervisor (which wires _WorkerResultAdapter) but
        # inject our FakeStateStore so we can inspect the persisted row.
        from davinci_monet.daemon.config import DaemonConfig, WatchesFile, WatchRule

        wf = WatchesFile(
            daemon=DaemonConfig(max_concurrent=1),
            watches={"cam": WatchRule(name="cam", watch="/tmp/*.nc", run="/tmp/cfg.yaml")},
        )
        sup = build_supervisor(wf, state=state)
        # Inject a FakeWatcher so we don't need a real filesystem.
        sup.watcher = FakeWatcher(
            [
                TriggerEvent(
                    watch_name="cam",
                    new_files=["/tmp/a.nc"],
                    detected_at=datetime(2026, 5, 31, 12, 0, 0),
                    settle_mode="quiescence",
                )
            ]
        )
        # Replace the real RunQueue with a FakeQueue so we can inject events.
        sup.queue = FakeQueue()
        sup.run_once()

    assert state.jobs[1]["status"] == JobStatus.COMPLETED
    rs = state.jobs[1]["result_summary"]
    assert rs is not None, "_WorkerResultAdapter must populate result_summary"
    assert rs["output_dir"] == "/icloud/Analysis/output"
    assert rs["plots"] == ["/icloud/Analysis/output/scatter.png"]
    assert rs["stages"] == 5
    assert rs["N"] == 42
