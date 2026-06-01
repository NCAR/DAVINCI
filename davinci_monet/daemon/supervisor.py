"""Long-lived supervisor event loop: watcher -> queue -> dispatcher -> state/notify.

ISOLATION INVARIANT (spec lines 90-95): this module and everything it imports at
module scope must never import matplotlib, xarray, monet/monetio, or the
pipeline. Pipelines run only inside the worker subprocess that dispatcher
launches. Keep imports stdlib + daemon-pure.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional, Protocol

from davinci_monet.daemon.config import DaemonConfig, WatchesFile, WatchRule
from davinci_monet.daemon.contracts import Clock, JobSpec, JobStatus, TriggerEvent

logger = logging.getLogger("davinci.daemon.supervisor")


class _RealClock:
    """Production monotonic clock (Clock protocol)."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class WatcherLike(Protocol):
    def poll(self) -> list[TriggerEvent]: ...


class QueueLike(Protocol):
    def submit(self, event: TriggerEvent) -> bool: ...

    def pop(self) -> Optional[tuple[str, list[str]]]: ...

    def __len__(self) -> int: ...


# A dispatcher callable mirrors dispatcher.spawn_worker's call shape. It returns
# an object exposing: success, exit_code, log_path, result_summary, output_dir,
# plots, error.
DispatchFn = Callable[..., Any]


class Supervisor:
    """Assembles watcher -> queue -> dispatcher -> state + notify.

    Collaborators are injected for testability; build_supervisor() constructs the
    production ones. The supervisor NEVER runs a pipeline in-process.
    """

    def __init__(
        self,
        *,
        watches_file: WatchesFile,
        watcher: WatcherLike,
        queue: QueueLike,
        dispatcher: DispatchFn,
        state: Any,  # StateStore (duck-typed for tests)
        notifier: Any = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.watches_file = watches_file
        self.daemon_cfg: DaemonConfig = watches_file.daemon
        self.rules: dict[str, WatchRule] = dict(watches_file.watches)
        self.watcher = watcher
        self.queue = queue
        self.dispatcher = dispatcher
        self.state = state
        self.notifier = notifier
        self.clock = clock or _RealClock()
        self._draining = False
        self._stopped = False
        self._running_count = 0
        self._started_at = self.clock.now()

    # ---- one loop tick ----------------------------------------------------
    def run_once(self) -> None:
        """Drain settled triggers into the queue, then dispatch up to capacity."""
        for event in self.watcher.poll():
            if self._draining:
                continue
            coalesced = self.queue.submit(event)
            logger.debug(
                "queued %s (coalesced=%s, %d files)",
                event.watch_name,
                coalesced,
                len(event.new_files),
            )
        self.dispatch_pending()

    def dispatch_pending(self) -> None:
        """Pop and dispatch one pending job if a slot is available.

        Dispatches at most one job per call: the real dispatcher is synchronous
        (blocks while the worker subprocess runs), so ``_running_count`` returns
        to zero immediately after each call. Limiting to one dispatch per
        ``run_once()`` tick ensures max_concurrent=1 means one job per poll cycle,
        giving the event loop a chance to drain new watcher events between runs.
        """
        if self._running_count >= self.daemon_cfg.max_concurrent:
            return
        popped = self.queue.pop()
        if popped is None:
            return
        watch_name, new_files = popped
        self._dispatch_one(watch_name, new_files)

    def _dispatch_one(self, watch_name: str, new_files: list[str]) -> None:
        rule = self.rules.get(watch_name)
        if rule is None:
            logger.warning("dropping job for unknown watch %s", watch_name)
            return
        files = sorted(new_files)
        job_id = self.state.create_job(
            watch_name=watch_name,
            config_path=rule.run,
            on_fire=rule.on_fire,
            files=files,
        )
        env = dict(rule.env)
        spec = JobSpec(
            job_id=job_id,
            watch_name=watch_name,
            config_path=rule.run,
            on_fire=rule.on_fire,
            inject_into=rule.inject_into,
            new_files=files,
            env=env,
            hdf5_file_locking=self.daemon_cfg.hdf5_file_locking,
            worker_timeout=self.daemon_cfg.worker_timeout,
            log_dir=None,
        )
        self._running_count += 1
        self.state.mark_running(job_id, started_at=datetime.now())
        try:
            result = self.dispatcher(spec, self.daemon_cfg)
        except Exception as exc:  # never wedge the daemon on a bad run
            logger.exception("dispatch raised for job %d", job_id)
            self.state.mark_failed(job_id, exit_code=None, error=str(exc))
            self._running_count -= 1
            self._notify(job_id, rule)
            return
        self._running_count -= 1
        self._record_result(job_id, rule, result)

    def _record_result(self, job_id: int, rule: WatchRule, result: Any) -> None:
        success = bool(getattr(result, "success", False))
        log_path = getattr(result, "log_path", None)
        if success:
            self.state.mark_completed(
                job_id,
                exit_code=getattr(result, "exit_code", 0) or 0,
                log_path=log_path,
                result_summary=getattr(result, "result_summary", None),
            )
        else:
            self.state.mark_failed(
                job_id,
                exit_code=getattr(result, "exit_code", None),
                error=getattr(result, "error", None) or "worker failed",
                log_path=log_path,
            )
        self._notify(job_id, rule)

    def _notify(self, job_id: int, rule: WatchRule) -> None:
        """Hand the finished job + its rule to the Notifier (no-op if unset).

        The Notifier (``davinci_monet.daemon.notify.Notifier``) takes the
        persisted ``JobRecord`` and the rule and routes the outcome through
        ``notify_outcome``; the supervisor reads the record back from state so the
        notify payload (status / result_summary / plots / output_dir) is the one
        that was just committed.
        """
        if self.notifier is None:
            return
        try:
            job = self.state.get_job(job_id)
            if job is None:
                return
            self.notifier.notify_result(job, rule)
        except Exception:
            logger.exception("notify failed for watch %s", rule.name)

    # ---- lifecycle --------------------------------------------------------
    def request_shutdown(self, reason: str = "signal") -> None:
        """Stop accepting new triggers; let in-flight work finish (graceful drain).

        Accepts an optional ``reason`` so it conforms to ``Callable[[str], None]``
        and can be passed straight to ``install_signal_handlers`` as the on-drain
        callback (the handler calls it with the signal name, e.g. ``"SIGTERM"``).
        """
        if not self._draining:
            logger.info("Shutdown requested (reason=%s)", reason)
        self._draining = True

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def uptime_s(self) -> float:
        return max(0.0, self.clock.now() - self._started_at)

    def serve(self, control_server: Any = None) -> None:
        """Run the poll loop until shutdown. control_server (if given) is started
        once and stopped on exit. Sleeps poll_interval between ticks."""
        if control_server is not None:
            control_server.start()
        try:
            while not self._stopped:
                self.run_once()
                if self._draining and self._running_count == 0 and len(self.queue) == 0:
                    break
                self.clock.sleep(self.daemon_cfg.poll_interval)
        finally:
            if control_server is not None:
                control_server.stop()


def build_supervisor(
    watches_file: WatchesFile,
    *,
    state: Any = None,
    clock: Optional[Clock] = None,
    notifier: Any = None,
) -> Supervisor:
    """Production wiring: construct the real watcher/queue/dispatcher/state/notify
    from a parsed WatchesFile. Imported lazily so the module-level import graph
    stays sci-stack-free.

    ``clock`` (a contracts.Clock) is threaded into BOTH the PollingWatcher and the
    Supervisor so settle/quiescence and uptime share one time source; tests pass a
    FakeClock here. ``state``/``notifier`` may be injected (integration tests pass
    a StateStore); otherwise the production StateStore + Notifier are built.
    """
    from davinci_monet.daemon.dispatcher import spawn_worker
    from davinci_monet.daemon.notify import Notifier
    from davinci_monet.daemon.queue import RunQueue
    from davinci_monet.daemon.state import StateStore
    from davinci_monet.daemon.watcher import PollingWatcher

    cfg = watches_file.daemon
    the_clock: Clock = clock or _RealClock()
    if state is None:
        state = StateStore(cfg.db_path)
        state.init_schema()
    watcher = PollingWatcher(list(watches_file.watches.values()), cfg, the_clock)
    queue = RunQueue()
    if notifier is None:
        notifier = Notifier(cfg)

    # spawn_worker is spawn_worker(spec, *, on_event=None); the Supervisor calls
    # its dispatcher as dispatcher(spec, daemon_cfg, on_event=...). Adapt the
    # arity with a thin wrapper so the call site and the worker launcher agree.
    def _dispatch(spec: JobSpec, daemon_cfg: DaemonConfig, *, on_event: Any = None) -> Any:
        return spawn_worker(spec, on_event=on_event)

    return Supervisor(
        watches_file=watches_file,
        watcher=watcher,
        queue=queue,
        dispatcher=_dispatch,
        state=state,
        notifier=notifier,
        clock=the_clock,
    )
