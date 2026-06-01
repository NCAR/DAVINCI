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
from davinci_monet.daemon.contracts import (
    PROTOCOL_VERSION,
    Clock,
    ControlResponse,
    JobRecord,
    JobSpec,
    JobStatus,
    TriggerEvent,
)

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
            # Clear the RunQueue RUNNING marker so introspection (is_running /
            # running_names / WatchSummary state) does not report this watch as
            # forever-running after the job finishes.
            if hasattr(self.queue, "mark_done"):
                self.queue.mark_done(watch_name)
            self._notify(job_id, rule)
            return
        self._running_count -= 1
        # Clear the RunQueue RUNNING marker (see above).
        if hasattr(self.queue, "mark_done"):
            self.queue.mark_done(watch_name)
        self._record_result(job_id, rule, result)

    def _record_result(self, job_id: int, rule: WatchRule, result: Any) -> None:
        success = bool(getattr(result, "success", False))
        log_path = getattr(result, "log_path", None)
        if success:
            # Build result_summary with output_dir and plots folded in so that
            # notify_outcome can read result_summary['output_dir'] and
            # result_summary['plots'] for the iCloud copy step.
            raw_summary: dict[str, Any] = dict(getattr(result, "result_summary", None) or {})
            output_dir = getattr(result, "output_dir", None)
            plots = getattr(result, "plots", None)
            if output_dir is not None:
                raw_summary["output_dir"] = output_dir
            if plots is not None:
                raw_summary["plots"] = list(plots)
            self.state.mark_completed(
                job_id,
                exit_code=getattr(result, "exit_code", 0) or 0,
                log_path=log_path,
                result_summary=raw_summary or None,
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

    # ---- control command dispatch ----------------------------------------
    def handle_command(self, cmd: str, args: Optional[dict[str, Any]] = None) -> ControlResponse:
        """Route one control request to a handler and return a ControlResponse.

        Mirrors the COMMAND CATALOG in contracts.py. Streaming commands
        (subscribe/logs_tail) are handled by the control server's push loop, not
        here; this returns the ack data for them.
        """
        args = args or {}
        handler = self._handler_table().get(cmd)
        if handler is None:
            return ControlResponse(ok=False, error=f"unknown command: {cmd}", code="unsupported")
        try:
            return handler(args)
        except KeyError as exc:
            return ControlResponse(ok=False, error=f"missing arg: {exc}", code="invalid_args")

    def _handler_table(self) -> dict[str, Callable[[dict[str, Any]], ControlResponse]]:
        return {
            "ping": self._cmd_ping,
            "status": self._cmd_status,
            "watch_list": self._cmd_watch_list,
            "watch_pause": self._cmd_watch_pause,
            "watch_resume": self._cmd_watch_resume,
            "watch_trigger": self._cmd_watch_trigger,
            "history": self._cmd_history,
            "job_get": self._cmd_job_get,
            "shutdown": self._cmd_shutdown,
            "subscribe": self._cmd_subscribe_ack,
            "logs_tail": self._cmd_logs_tail_ack,
        }

    def _cmd_ping(self, args: dict[str, Any]) -> ControlResponse:
        import os

        return ControlResponse(
            ok=True,
            data={
                "pong": True,
                "version": PROTOCOL_VERSION,
                "pid": os.getpid(),
                "uptime_s": self.uptime_s,
            },
        )

    def watch_summaries(self) -> list[dict[str, Any]]:
        """Compact per-watch rows (WatchSummary shape) for list/status/top."""
        summaries: list[dict[str, Any]] = []
        for name, rule in self.rules.items():
            state = "paused" if not rule.enabled else "armed"
            summaries.append(
                {
                    "name": name,
                    "enabled": rule.enabled,
                    "source": "file",
                    "on_fire": rule.on_fire,
                    "settle_mode": rule.settle_mode,
                    "watch": rule.watch,
                    "run": rule.run,
                    "state": state,
                    "last_job_id": None,
                    "last_status": None,
                    "last_fired_at": None,
                }
            )
        return summaries

    def _cmd_watch_list(self, args: dict[str, Any]) -> ControlResponse:
        return ControlResponse(ok=True, data={"watches": self.watch_summaries()})

    def _set_enabled(self, name: str, enabled: bool) -> ControlResponse:
        rule = self.rules.get(name)
        if rule is None:
            return ControlResponse(ok=False, error=f"no such watch: {name}", code="not_found")
        self.rules[name] = rule.model_copy(update={"enabled": enabled})
        self.state.set_enabled(name, enabled)
        return ControlResponse(ok=True, data={"name": name, "enabled": enabled})

    def _cmd_watch_pause(self, args: dict[str, Any]) -> ControlResponse:
        return self._set_enabled(args["name"], False)

    def _cmd_watch_resume(self, args: dict[str, Any]) -> ControlResponse:
        return self._set_enabled(args["name"], True)

    def _cmd_watch_trigger(self, args: dict[str, Any]) -> ControlResponse:
        name = args["name"]
        rule = self.rules.get(name)
        if rule is None:
            return ControlResponse(ok=False, error=f"no such watch: {name}", code="not_found")
        files = sorted(args.get("files", []) or [])
        event = TriggerEvent(
            watch_name=name,
            new_files=files,
            detected_at=datetime.now(),
            settle_mode=rule.settle_mode,
        )
        coalesced = self.queue.submit(event)
        return ControlResponse(
            ok=True,
            data={"queued_job_id": None, "coalesced": bool(coalesced)},
        )

    def build_status(self) -> dict[str, Any]:
        """Assemble the StatusData payload (running/queued/watches/recent)."""
        import os

        active = list(self.state.active_jobs())
        running = [self._job_dump(j) for j in active if self._job_status(j) == JobStatus.RUNNING]
        queued = [self._job_dump(j) for j in active if self._job_status(j) == JobStatus.QUEUED]
        recent = [self._job_dump(j) for j in self.state.list_jobs(limit=10)]
        return {
            "version": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "uptime_s": self.uptime_s,
            "draining": self._draining,
            "max_concurrent": self.daemon_cfg.max_concurrent,
            "running": running,
            "queued": queued,
            "watches": self.watch_summaries(),
            "recent": recent,
        }

    @staticmethod
    def _job_status(job: Any) -> Any:
        return getattr(job, "status", None)

    @staticmethod
    def _job_dump(job: Any) -> Any:
        if hasattr(job, "model_dump"):
            return job.model_dump(mode="json")
        return job

    def _cmd_status(self, args: dict[str, Any]) -> ControlResponse:
        return ControlResponse(ok=True, data=self.build_status())

    def _cmd_history(self, args: dict[str, Any]) -> ControlResponse:
        jobs = self.state.list_jobs(
            watch_name=args.get("watch"),
            status=JobStatus.FAILED if args.get("failed") else None,
            limit=int(args.get("limit", 50)),
        )
        return ControlResponse(ok=True, data={"jobs": [self._job_dump(j) for j in jobs]})

    def _cmd_job_get(self, args: dict[str, Any]) -> ControlResponse:
        job = self.state.get_job(int(args["job_id"]))
        return ControlResponse(ok=True, data={"job": self._job_dump(job) if job else None})

    def _cmd_shutdown(self, args: dict[str, Any]) -> ControlResponse:
        drain = bool(args.get("drain", True))
        self.request_shutdown()
        if not drain:
            self._stopped = True
        return ControlResponse(ok=True, data={"shutting_down": True, "draining": drain})

    def _cmd_subscribe_ack(self, args: dict[str, Any]) -> ControlResponse:
        topics = args.get("topics") or ["jobs", "watches", "stats"]
        return ControlResponse(ok=True, data={"subscribed": list(topics)})

    def _cmd_logs_tail_ack(self, args: dict[str, Any]) -> ControlResponse:
        return ControlResponse(ok=True, data={"streaming": True})

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

    # spawn_worker returns a WorkerRunResult that carries only success/exit_code/
    # events/stderr.  The supervisor's _record_result expects an object with
    # result_summary (dict containing output_dir + plots), log_path, and error.
    # _WorkerResultAdapter translates the worker's terminal "result" ProgressEvent
    # (kind=="result", fields: output_dir/plots/summary/log_path/error per
    # contracts.py) into that shape so _record_result and notify_outcome both see
    # the full payload.
    class _WorkerResultAdapter:
        """Wraps WorkerRunResult + its terminal ProgressEvent for _record_result."""

        def __init__(self, run_result: Any) -> None:
            result_event = run_result.result_event  # ProgressEvent | None
            self.success: bool = run_result.success
            self.exit_code: int = run_result.exit_code
            # Terminal result event fields — fall back gracefully when absent.
            if result_event is not None:
                self.log_path: Any = getattr(result_event, "log_path", None)
                self.output_dir: Any = getattr(result_event, "output_dir", None)
                self.plots: Any = getattr(result_event, "plots", None) or []
                self.error: Any = getattr(result_event, "error", None)
                # "summary" is the per-run stats/stage digest that goes into
                # jobs.result_summary; output_dir and plots are folded in by
                # _record_result so notify_outcome can read them from the dict.
                self.result_summary: Any = dict(getattr(result_event, "summary", None) or {})
            else:
                # Worker exited without emitting a result line (crash / timeout).
                self.log_path = None
                self.output_dir = None
                self.plots = []
                self.result_summary = {}
                self.error = run_result.stderr.strip() or "worker exited without result"

    def _dispatch(spec: JobSpec, daemon_cfg: DaemonConfig, *, on_event: Any = None) -> Any:
        run_result = spawn_worker(spec, on_event=on_event)
        return _WorkerResultAdapter(run_result)

    return Supervisor(
        watches_file=watches_file,
        watcher=watcher,
        queue=queue,
        dispatcher=_dispatch,
        state=state,
        notifier=notifier,
        clock=the_clock,
    )
