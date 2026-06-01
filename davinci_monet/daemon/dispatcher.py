"""Daemon dispatcher: build a JobSpec and spawn the isolated worker subprocess.

This module runs INSIDE the supervisor process and therefore MUST NOT import
the pipeline, matplotlib, or xarray. It only assembles per-job state and
launches `python -m davinci_monet.daemon.worker` as a fresh subprocess.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import JobSpec, ProgressEvent, TriggerEvent

WORKER_MODULE = "davinci_monet.daemon.worker"


def build_job_spec(
    rule: WatchRule,
    trigger: TriggerEvent,
    daemon_cfg: DaemonConfig,
    *,
    job_id: int,
    base_env: dict[str, str] | None = None,
) -> JobSpec:
    """Build the JobSpec the dispatcher hands to the worker.

    The per-job env is the daemon/base env overlaid with the rule's ``env:``
    block (rule values win). For ``on_fire == "new_files_only"`` the trigger's
    new files become the injection list and ``inject_into`` names the source
    whose ``files:`` the worker overrides; for ``whole_config`` no injection is
    performed (``inject_into`` is None) though ``new_files`` is still recorded
    for history.
    """
    if base_env is None:
        base_env = dict(os.environ)
    job_env: dict[str, str] = dict(base_env)
    job_env.update(rule.env)

    inject_into = rule.inject_into if rule.on_fire == "new_files_only" else None

    return JobSpec(
        job_id=job_id,
        watch_name=rule.name,
        config_path=str(rule.run),
        on_fire=rule.on_fire,
        inject_into=inject_into,
        new_files=list(trigger.new_files),
        env=job_env,
        hdf5_file_locking=daemon_cfg.hdf5_file_locking,
        worker_timeout=daemon_cfg.worker_timeout,
        log_dir=None,
    )


@dataclass
class WorkerRunResult:
    """Outcome of one spawned worker subprocess."""

    job_id: int
    exit_code: int
    events: list[ProgressEvent] = field(default_factory=list)
    stderr: str = ""

    @property
    def success(self) -> bool:
        """True iff the worker exited 0 (PipelineResult.success was True)."""
        return self.exit_code == 0

    @property
    def result_event(self) -> ProgressEvent | None:
        """The single terminal 'result' event, if one was emitted."""
        for event in reversed(self.events):
            if event.kind == "result":
                return event
        return None


def spawn_worker(
    spec: JobSpec,
    *,
    on_event: Optional[Callable[[ProgressEvent], None]] = None,
    attached: bool = False,
) -> WorkerRunResult:
    """Launch the worker subprocess for ``spec`` and collect its progress.

    Two modes, selected by ``attached``:

    * **Headless** (``attached=False``, the default and original behavior): the
      worker's stdout is PIPEd back; it streams one JSON object per line, each
      parsed into a ProgressEvent and (optionally) forwarded to ``on_event``.
      Non-JSON lines are ignored. stderr is drained on a background thread.

    * **Attached** (``attached=True``): the worker INHERITS the serve terminal's
      stdout/stderr so the pipeline's native animated progress renders directly
      to the TTY. Progress events are instead written by the worker to a temp
      file named by ``DAEMON_EVENTS_PATH``; after the worker exits we read and
      parse that file's lines (same parsing as headless), forward them to
      ``on_event``, and build the identical WorkerRunResult so downstream
      consumers (output_dir/plots/log_path) still work.

    The JobSpec is serialized to JSON and written to the worker's stdin in both
    modes. The worker's exit code is captured and surfaced via WorkerRunResult.
    """
    env = dict(spec.env)
    env["HDF5_USE_FILE_LOCKING"] = "TRUE" if spec.hdf5_file_locking else "FALSE"

    if attached:
        return _spawn_worker_attached(spec, env, on_event)
    return _spawn_worker_headless(spec, env, on_event)


def _spawn_worker_headless(
    spec: JobSpec,
    env: dict[str, str],
    on_event: Optional[Callable[[ProgressEvent], None]],
) -> WorkerRunResult:
    """Original behavior: PIPE stdout, parse JSON lines, drain stderr thread."""
    proc = subprocess.Popen(
        [sys.executable, "-m", WORKER_MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None

    proc.stdin.write(spec.to_json())
    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.stdin.close()

    # Drain stderr on a background thread so a full stderr pipe buffer (>64 KB
    # on macOS/Linux) never blocks the worker from writing to stdout and never
    # deadlocks the stdout-drain loop below.
    assert proc.stderr is not None
    proc_stderr = proc.stderr  # capture for closure — narrows type to IO[str]
    stderr_buf = io.StringIO()

    def _drain_stderr() -> None:
        for chunk in iter(lambda: proc_stderr.read(4096), ""):
            stderr_buf.write(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    events: list[ProgressEvent] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        event = ProgressEvent.parse_line(line)
        if event is None:
            continue
        events.append(event)
        if on_event is not None:
            on_event(event)

    # proc.stdout is exhausted (EOF); now wait for the process to exit.
    timeout = spec.worker_timeout
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    stderr_thread.join()
    stderr = stderr_buf.getvalue()

    return WorkerRunResult(
        job_id=spec.job_id,
        exit_code=proc.returncode if proc.returncode is not None else 1,
        events=events,
        stderr=stderr or "",
    )


def _spawn_worker_attached(
    spec: JobSpec,
    env: dict[str, str],
    on_event: Optional[Callable[[ProgressEvent], None]],
) -> WorkerRunResult:
    """Attached behavior: INHERIT serve TTY for native progress; events via temp file.

    A temp file path is passed to the worker via ``DAEMON_EVENTS_PATH``; the
    worker writes its JSON progress/started/result lines there instead of stdout.
    stdout/stderr are inherited (``stdout=None, stderr=None``) so the pipeline's
    animated NCAR progress display renders directly to the serve terminal. After
    the worker exits we parse the events file exactly like the headless stdout.
    """
    fd, events_path = tempfile.mkstemp(prefix="davinci-daemon-events-", suffix=".jsonl")
    os.close(fd)
    env["DAEMON_EVENTS_PATH"] = events_path

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=None,  # inherit serve terminal — animation shows live
            stderr=None,  # inherit serve terminal
            env=env,
            text=True,
        )
        assert proc.stdin is not None
        proc.stdin.write(spec.to_json())
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.stdin.close()

        timeout = spec.worker_timeout
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        events: list[ProgressEvent] = []
        try:
            with open(events_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    event = ProgressEvent.parse_line(line)
                    if event is None:
                        continue
                    events.append(event)
                    if on_event is not None:
                        on_event(event)
        except OSError:
            pass

        return WorkerRunResult(
            job_id=spec.job_id,
            exit_code=proc.returncode if proc.returncode is not None else 1,
            events=events,
            stderr="",
        )
    finally:
        try:
            os.unlink(events_path)
        except OSError:
            pass
