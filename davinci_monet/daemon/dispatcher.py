"""Daemon dispatcher: build a JobSpec and spawn the isolated worker subprocess.

This module runs INSIDE the supervisor process and therefore MUST NOT import
the pipeline, matplotlib, or xarray. It only assembles per-job state and
launches `python -m davinci_monet.daemon.worker` as a fresh subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
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
) -> WorkerRunResult:
    """Launch the worker subprocess for ``spec`` and collect its progress.

    The JobSpec is serialized to JSON and written to the worker's stdin. The
    worker streams one JSON object per stdout line; each is parsed into a
    ProgressEvent and (optionally) forwarded to ``on_event``. Non-JSON stdout
    lines are ignored by the parser. The worker's exit code is captured and
    surfaced via WorkerRunResult.
    """
    env = dict(spec.env)
    env["HDF5_USE_FILE_LOCKING"] = "TRUE" if spec.hdf5_file_locking else "FALSE"

    proc = subprocess.Popen(
        [sys.executable, "-m", WORKER_MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None

    proc.stdin.write(spec.to_json())
    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.stdin.close()

    events: list[ProgressEvent] = []
    for line in proc.stdout:
        event = ProgressEvent.parse_line(line)
        if event is None:
            continue
        events.append(event)
        if on_event is not None:
            on_event(event)

    timeout = spec.worker_timeout
    assert proc.stderr is not None
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    stderr = proc.stderr.read()

    return WorkerRunResult(
        job_id=spec.job_id,
        exit_code=proc.returncode if proc.returncode is not None else 1,
        events=events,
        stderr=stderr or "",
    )
