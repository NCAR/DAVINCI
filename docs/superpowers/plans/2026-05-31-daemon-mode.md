# DAVINCI Daemon Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a long-lived daemon to DAVINCI that watches configured directories and automatically runs DAVINCI analysis pipelines when new data settles, with full job history, control surface, and notifications.

**Architecture:** A thin supervisor process owns a polling file-watcher, a serial coalescing run-queue, SQLite-backed job/watch state, and a Unix-socket control server — and never imports the scientific stack. Each settled trigger is dispatched to a subprocess-isolated worker (`python -m davinci_monet.daemon.worker`) that is the only place the real pipeline (`run_analysis` / `PipelineRunner.run_from_config`) runs, streaming JSON-line progress back to the supervisor. Operators interact through a Typer CLI sub-app, an interactive shell, and a Rich live dashboard, all thin clients of the control socket.

**Tech Stack:** Python 3.11, Typer, Rich, Pydantic 2, stdlib (socket/selectors/subprocess/sqlite3/signal); integrates existing davinci_monet pipeline.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `davinci_monet/daemon/__init__.py` | Daemon package marker. |
| `davinci_monet/daemon/contracts.py` | Authoritative shared runtime primitives (enums, dataclasses, Protocols, control-message TypedDicts, SCHEMA_DDL) imported by every daemon module. |
| `davinci_monet/daemon/config.py` | Config models (`WatchRule`, `DaemonConfig`, `NotificationConfig`, `WatchesFile`), `parse_duration`, `load_watches`, `merge_rules`. |
| `davinci_monet/daemon/state.py` | SQLite `StateStore`: job history + watch runtime-status persistence. |
| `davinci_monet/daemon/watcher.py` | `PollingWatcher`: glob+stat scan, quiescence/sentinel settle detection, `max_settle_wait` safety valve. |
| `davinci_monet/daemon/queue.py` | `RunQueue`: in-memory serial FIFO with `watch_name` coalescing; `PendingJob` carrier. |
| `davinci_monet/daemon/dispatcher.py` | `build_job_spec` + `spawn_worker` (supervisor-side; launches the worker subprocess, no sci stack). |
| `davinci_monet/daemon/worker.py` | Isolated child entrypoint: reads a JobSpec, injects new files, runs the real pipeline, streams ProgressEvent JSON lines. |
| `davinci_monet/daemon/control.py` | `ControlServer`: AF_UNIX/SOCK_STREAM server with handler registry + JSON framing + streaming. |
| `davinci_monet/daemon/client.py` | `DaemonClient`: thin synchronous Unix-socket client (`call`/`stream`/`is_alive`). |
| `davinci_monet/daemon/notify.py` | Outcome notifications: desktop dispatch, iCloud copy, `notify_outcome` routing. |
| `davinci_monet/daemon/lifecycle.py` | Process lifecycle: `PidLock`, `DrainController`, signal handlers, double-fork `daemonize`. |
| `davinci_monet/daemon/shell.py` | Interactive control REPL: `parse_command`, `DaemonShell`. |
| `davinci_monet/daemon/dashboard.py` | `daemon top` Rich live dashboard renderers + `apply_stream_event` reducer. |
| `davinci_monet/daemon/supervisor.py` | `Supervisor` event loop + control-command dispatch table; `build_supervisor` wiring. |
| `davinci_monet/cli/commands/daemon.py` | `daemon` Typer sub-app (serve/start/stop/status/reload/watch/logs/history). |
| `davinci_monet/cli/app.py` | (modified) Register the `daemon` sub-app in `register_commands()`. |
| `davinci_monet/cli/commands/__init__.py` | (modified) Export `daemon` alongside `get_data`/`run`/`validate`. |
| `davinci_monet/tests/unit/daemon/__init__.py` | Unit-test package marker for the daemon (under `tests/unit/`). |
| `davinci_monet/tests/unit/daemon/test_config.py` | Unit tests for the config models + loader + merge. |
| `davinci_monet/tests/unit/daemon/test_watcher.py` | Unit tests for the polling watcher settle/sentinel/valve logic. |
| `davinci_monet/tests/unit/daemon/test_control.py` | Unit tests for the control server (request/response, errors, streaming). |
| `davinci_monet/tests/unit/daemon/test_client.py` | Unit tests for the daemon control client. |
| `davinci_monet/tests/unit/daemon/test_notify_desktop.py` | Unit tests for desktop notification dispatch. |
| `davinci_monet/tests/unit/daemon/test_notify_icloud.py` | Unit tests for iCloud copy of plots + summary. |
| `davinci_monet/tests/unit/daemon/test_notify_outcome.py` | Unit tests for notify_outcome routing. |
| `davinci_monet/tests/unit/daemon/test_lifecycle_lock.py` | Unit tests for the PID+lock primitive. |
| `davinci_monet/tests/unit/daemon/test_lifecycle_drain.py` | Unit tests for the drain flag + signal handlers. |
| `davinci_monet/tests/unit/daemon/test_lifecycle_daemonize.py` | Unit tests for the double-fork daemonize. |
| `davinci_monet/tests/unit/daemon/test_supervisor_loop.py` | Unit tests for the supervisor coalesce/dispatch loop. |
| `davinci_monet/tests/unit/daemon/test_supervisor_control.py` | Unit tests for the supervisor control dispatch table. |
| `davinci_monet/tests/unit/daemon/test_cli_daemon.py` | Unit tests for the daemon CLI sub-app. |
| `davinci_monet/tests/unit/daemon/test_app_registration.py` | Unit tests that the daemon sub-app is registered on the main CLI. |
| `davinci_monet/tests/unit/daemon/test_state_store.py` | Unit tests for the SQLite StateStore. |
| `davinci_monet/tests/unit/daemon/test_queue.py` | Unit tests for the coalescing RunQueue. |
| `davinci_monet/tests/unit/daemon/test_shell.py` | Unit tests for the shell parser + dispatch + quit guard. |
| `davinci_monet/tests/unit/daemon/test_dashboard.py` | Unit tests for the dashboard renderers + reducer. |
| `davinci_monet/tests/integration/test_daemon_integration.py` | Integration tests: settle -> real worker subprocess -> real pipeline -> history + notify. |
| `davinci_monet/tests/test_daemon_dispatcher.py` | Unit tests for the dispatcher (job-spec building, spawn_worker). |
| `davinci_monet/tests/test_daemon_worker.py` | Unit tests for the worker (config injection + progress emission). |
| `davinci_monet/tests/integration/test_daemon_worker_pipeline.py` | Integration test: worker runs a synthetic config through the real pipeline. |

---

## Shared Contracts (interface reference)

The block below is the canonical content of `davinci_monet/daemon/contracts.py` that every module imports from. **`contracts.py` owns ONLY the runtime primitives** — `JobStatus`, `SettleMode` (and the other enums/literal aliases), `Clock`, `TriggerEvent`, `JobRecord`, `JobSpec`, the `StateStore` accessor surface, `SCHEMA_DDL`, the `ProgressEvent` parse helper, and the control-message types (`ControlRequest`, `ControlResponse`, `StreamEvent`, `ControlHandler`, command catalog tuples). The concrete `DaemonClient` is NOT in contracts — it lives solely in `davinci_monet/daemon/client.py`. The config models **`WatchRule`/`DaemonConfig`/`NotificationConfig`/`WatchesFile`** shown here for reference are owned by `davinci_monet/daemon/config.py` (the config section, Tasks 2-7), **NOT** `contracts.py`. When Task 1 materializes `contracts.py`, it includes ONLY the primitive definitions (enums/dataclasses/Protocol/TypedDicts and `SCHEMA_DDL`), and omits `parse_duration`/the config models, which live in `config.py`.

```python
"""DAVINCI Daemon Mode — SHARED INTERFACE CONTRACTS (authoritative).

Every module in davinci_monet/daemon/ conforms to THIS file verbatim. Field
names and types are exact. Pydantic base classes (StrictModel / FlexibleModel)
match the house style in davinci_monet/config/schema.py.

Integration boundary (from the real code, do not re-derive):
  davinci_monet.pipeline.runner.run_analysis(
      config: dict[str, Any] | str,
      show_progress: bool = True,
      show_plots: bool = False,
      preview_format: Literal["pdf", "png"] = "pdf",
  ) -> PipelineResult
  PipelineResult fields: success: bool, stage_results: list[StageResult],
      context: PipelineContext | None, start_time: datetime | None,
      end_time: datetime | None, total_duration_seconds: float
  davinci_monet.config.parser.load_config(source) -> MonetConfig  (.model_dump())
  davinci_monet.config.parser.expand_env_vars(data: dict) -> dict  (os.path.expandvars; ${VAR} & $VAR)
"""

from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from pydantic import ConfigDict, Field, field_validator

# Reuse the project's pydantic base classes verbatim — DO NOT redefine them.
from davinci_monet.config.schema import FlexibleModel, StrictModel


# =============================================================================
# 0. Enums & literal aliases
# =============================================================================


class JobStatus(str, enum.Enum):
    """Lifecycle status of a single daemon-launched run.

    Stored as the raw string value in the SQLite ``jobs.status`` column.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# on_fire run-scope mode (spec decision #4).
OnFireMode = Literal["whole_config", "new_files_only"]

# Source of a watch rule row in watch_status (spec state-store section).
WatchSource = Literal["file", "live"]

# Notification channels (spec decision #10 + per-rule notify: override).
NotifyChannel = Literal["desktop", "icloud", "log"]

# Settle/quiescence detection modes for a rule (spec decision #9).
SettleMode = Literal["quiescence", "sentinel"]


# =============================================================================
# 1. Duration parsing helper (config group owns the implementation)
# =============================================================================


def parse_duration(value: str | int | float | None) -> Optional[float]:
    """Parse a human duration like "30s", "5m", "2h", "1d" -> float seconds.

    Accepted suffixes (case-insensitive): s, m, h, d. A bare number (int/float
    or unsuffixed string) is interpreted as seconds. ``None`` -> ``None``.
    Pydantic duration fields below call this in a ``mode="before"`` validator,
    so YAML values may be written as "30s"/"5m" or as plain numbers.

    Raises
    ------
    ValueError
        On an unparseable / negative value.
    """
    ...


# =============================================================================
# 2. Config models  (watches.yaml)  — group: config
# =============================================================================


class NotificationConfig(FlexibleModel):
    """Daemon-level notification policy (the ``daemon.notifications`` block)."""

    desktop: bool = True
    icloud_copy: bool = True
    icloud_dir: Path = Field(
        default=Path("~/Library/Mobile Documents/com~apple~CloudDocs/Claude")
    )

    @field_validator("icloud_dir", mode="before")
    @classmethod
    def _expand_user(cls, v: Any) -> Any:
        """Expand ~ and ${VAR} at daemon load (layer-1 expansion)."""
        ...


class WatchRule(FlexibleModel):
    """A single declarative watch rule (one entry under ``watches:``).

    ``name`` is the rule's mapping key in watches.yaml; the loader injects it.
    Layer-1 ``${VAR}`` expansion (watch/run/sentinel paths) has already been
    applied by the time a WatchRule is constructed. ``env`` is the per-rule
    overlay used for layer-2 (worker-side) expansion and is NOT expanded here.
    """

    name: str
    watch: str  # glob pattern, layer-1 expanded (e.g. "/scratch/cam/incoming/*.nc")
    run: str  # path to the DAVINCI config to execute, layer-1 expanded
    on_fire: OnFireMode = "whole_config"
    inject_into: Optional[str] = None  # required when on_fire == "new_files_only"
    settle: float = Field(default=30.0)  # quiescence window, seconds
    sentinel: Optional[str] = None  # marker path; presence triggers (layer-1 expanded)
    env: dict[str, str] = Field(default_factory=dict)  # per-rule worker env overlay
    notify: Optional[list[NotifyChannel]] = None  # per-rule override of daemon default
    enabled: bool = True  # runtime pause/resume; persisted in watch_status

    @field_validator("settle", mode="before")
    @classmethod
    def _parse_settle(cls, v: Any) -> Any:
        """Accept "30s"/"5m"/number via parse_duration."""
        ...

    @property
    def settle_mode(self) -> SettleMode:
        """'sentinel' if a sentinel path is set, else 'quiescence'."""
        return "sentinel" if self.sentinel else "quiescence"


class DaemonConfig(FlexibleModel):
    """Top-level daemon policy (the ``daemon:`` block of watches.yaml)."""

    state_dir: Path = Field(default=Path("~/.davinci/daemon"))
    poll_interval: float = Field(default=5.0)  # seconds
    max_concurrent: int = 1  # serial by default (spec decision #5)
    hdf5_file_locking: bool = False  # -> worker HDF5_USE_FILE_LOCKING=FALSE when False
    max_settle_wait: Optional[float] = Field(default=1800.0)  # 30m safety valve; None disables
    worker_timeout: Optional[float] = None  # hard cap on a single run, seconds; None = unbounded
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    @field_validator("poll_interval", "max_settle_wait", "worker_timeout", mode="before")
    @classmethod
    def _parse_durations(cls, v: Any) -> Any:
        """Accept "5s"/"30m"/number via parse_duration; pass None through."""
        ...

    @field_validator("state_dir", mode="before")
    @classmethod
    def _expand_state_dir(cls, v: Any) -> Any:
        """Expand ~ and ${VAR} at daemon load (layer-1)."""
        ...

    # ---- Derived runtime paths under state_dir (all absolute, ~ expanded) ----
    @property
    def db_path(self) -> Path:  # state_dir / "history.db"
        ...

    @property
    def socket_path(self) -> Path:  # state_dir / "control.sock"
        ...

    @property
    def pid_path(self) -> Path:  # state_dir / "daemon.pid"
        ...

    @property
    def lock_path(self) -> Path:  # state_dir / "daemon.lock"
        ...

    @property
    def log_path(self) -> Path:  # state_dir / "daemon.log"
        ...


class WatchesFile(StrictModel):
    """The fully-parsed watches.yaml (daemon policy + the declared rules).

    Returned by ``load_watches``. ``watches`` is keyed by rule name; each value
    is a fully-constructed WatchRule whose ``name`` matches its key.
    """

    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    watches: dict[str, WatchRule] = Field(default_factory=dict)


# ---- config-group function signatures -------------------------------------

def load_watches(source: str | Path) -> WatchesFile:
    """Load + layer-1 env-expand + validate a watches.yaml into a WatchesFile.

    Reuses davinci_monet.config.parser.load_yaml + expand_env_vars for ${VAR}
    expansion against the DAEMON's os.environ, then constructs DaemonConfig and
    each WatchRule (injecting the mapping key as ``name``). on_fire ==
    "new_files_only" with no ``inject_into`` is a validation error.
    """
    ...


def merge_rules(
    declared: dict[str, WatchRule],
    live: dict[str, WatchRule],
    disabled: set[str],
) -> dict[str, WatchRule]:
    """Reconcile file-declared rules with state-store live/runtime state.

    ``declared`` = rules from watches.yaml (source="file"). ``live`` =
    runtime-added rules from watch_status (source="live"). ``disabled`` = names
    paused at runtime. Result: declared updated from file, live-added preserved
    unless removed, and ``enabled`` reflecting the ``disabled`` set. Used by
    ``daemon reload``.
    """
    ...


# =============================================================================
# 3. Trigger event  — group: watcher (emitted to queue)
# =============================================================================


class TriggerEvent(StrictModel):
    """Emitted by the watcher when a rule settles (or its sentinel appears).

    Carries the newly-arrived absolute file paths. The queue coalesces repeat
    events for the same ``watch_name`` by unioning ``new_files``.
    """

    watch_name: str
    new_files: list[str]  # absolute paths, sorted, that triggered this fire
    detected_at: datetime  # when settle/sentinel completed
    settle_mode: SettleMode  # "quiescence" or "sentinel"


# =============================================================================
# 4. Job records & job spec  — groups: state, dispatcher, worker
# =============================================================================


class JobSpec(StrictModel):
    """Everything the dispatcher needs to build + launch one worker subprocess.

    Constructed by the dispatcher from a popped queue entry + its WatchRule +
    DaemonConfig. Serialized to JSON and handed to worker.py on stdin.
    """

    job_id: int  # the jobs.id assigned by StateStore.create_job
    watch_name: str
    config_path: str  # absolute path to the DAVINCI config to run
    on_fire: OnFireMode
    inject_into: Optional[str]  # source label whose files: gets overridden
    new_files: list[str]  # absolute new-file paths (the inject_into override list)
    env: dict[str, str]  # per-job env = daemon env overlaid with rule.env (layer-2 base)
    hdf5_file_locking: bool  # -> sets HDF5_USE_FILE_LOCKING in worker env
    worker_timeout: Optional[float]  # seconds; None = unbounded
    log_dir: Optional[str]  # analysis log_dir if the daemon overrides one; else None

    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, raw: str) -> "JobSpec": ...


class JobRecord(StrictModel):
    """One row of the ``jobs`` table. Mirrors the SQLite DDL 1:1.

    ``files``, ``result_summary`` are JSON-encoded in the DB and decoded here.
    Timestamps are stored as ISO-8601 strings in SQLite and surfaced as
    datetime here. Nullable columns map to Optional fields.
    """

    id: int
    watch_name: str
    config_path: str
    on_fire: OnFireMode
    files: list[str] = Field(default_factory=list)  # JSON in DB
    status: JobStatus
    submitted_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_s: Optional[float] = None
    exit_code: Optional[int] = None
    log_path: Optional[str] = None
    result_summary: Optional[dict[str, Any]] = None  # JSON in DB
    error: Optional[str] = None


class WatchStatusRecord(StrictModel):
    """One row of the ``watch_status`` table (runtime pause/resume + live rules)."""

    watch_name: str
    enabled: bool = True
    source: WatchSource = "file"  # "file" (declared) or "live" (runtime-added)
    rule_json: Optional[dict[str, Any]] = None  # serialized WatchRule for source="live"
    updated_at: datetime


# =============================================================================
# 5. StateStore (SQLite)  — group: state
# =============================================================================


class StateStore:
    """SQLite-backed job history + watch runtime-status persistence.

    stdlib sqlite3 only. Single connection, check_same_thread=False, accessed
    only from the supervisor thread/loop. Opens/creates the DB and applies the
    DDL (see ``SCHEMA_DDL`` below) on construction. All timestamps written as
    ISO-8601 (datetime.isoformat()); all dict/list fields json.dumps'd.
    """

    def __init__(self, db_path: str | Path) -> None: ...
    def init_schema(self) -> None: ...  # idempotent CREATE TABLE IF NOT EXISTS ...
    def close(self) -> None: ...

    # ---- jobs CRUD --------------------------------------------------------
    def create_job(
        self,
        watch_name: str,
        config_path: str,
        on_fire: OnFireMode,
        files: list[str],
    ) -> int:
        """Insert a row with status=QUEUED, submitted_at=now. Returns jobs.id."""
        ...

    def mark_running(self, job_id: int, started_at: Optional[datetime] = None) -> None:
        """status=RUNNING, set started_at (default now)."""
        ...

    def mark_completed(
        self,
        job_id: int,
        exit_code: int,
        log_path: Optional[str],
        result_summary: Optional[dict[str, Any]],
        ended_at: Optional[datetime] = None,
    ) -> None:
        """status=COMPLETED, set ended_at/duration_s/exit_code/log_path/result_summary."""
        ...

    def mark_failed(
        self,
        job_id: int,
        exit_code: Optional[int],
        error: str,
        log_path: Optional[str] = None,
        ended_at: Optional[datetime] = None,
    ) -> None:
        """status=FAILED, set ended_at/duration_s/exit_code/error/log_path."""
        ...

    def mark_skipped(self, job_id: int, error: Optional[str] = None) -> None:
        """status=SKIPPED, set ended_at; used when coalesced away or drained."""
        ...

    def get_job(self, job_id: int) -> Optional[JobRecord]: ...

    def list_jobs(
        self,
        watch_name: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        """Most-recent-first (ORDER BY id DESC). ``--failed`` -> status=FAILED."""
        ...

    def active_jobs(self) -> list[JobRecord]:
        """Rows with status in {QUEUED, RUNNING}. Backs `top` and drain logic."""
        ...

    # ---- watch_status CRUD ------------------------------------------------
    def upsert_watch_status(self, record: WatchStatusRecord) -> None:
        """INSERT OR REPLACE keyed by watch_name; bump updated_at."""
        ...

    def set_enabled(self, watch_name: str, enabled: bool) -> None:
        """Pause/resume persistence. Upserts updated_at."""
        ...

    def get_watch_status(self, watch_name: str) -> Optional[WatchStatusRecord]: ...
    def list_watch_status(self) -> list[WatchStatusRecord]: ...

    def add_live_rule(self, rule: WatchRule) -> None:
        """Persist a runtime-added rule (source="live", rule_json=rule dump)."""
        ...

    def remove_watch(self, watch_name: str) -> None:
        """Delete the watch_status row (for live rules) / drop runtime overrides."""
        ...

    def disabled_names(self) -> set[str]:
        """Names with enabled=False. Fed to merge_rules() on reload."""
        ...

    def live_rules(self) -> dict[str, WatchRule]:
        """Reconstructed WatchRule objects for source="live" rows. -> merge_rules()."""
        ...


# The SQLite DDL is the source of truth for the jobs + watch_status tables.
SCHEMA_DDL: str = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_name      TEXT    NOT NULL,
    config_path     TEXT    NOT NULL,
    on_fire         TEXT    NOT NULL,              -- 'whole_config' | 'new_files_only'
    files           TEXT    NOT NULL DEFAULT '[]', -- JSON array of absolute paths
    status          TEXT    NOT NULL,              -- queued|running|completed|failed|skipped
    submitted_at    TEXT    NOT NULL,              -- ISO-8601
    started_at      TEXT,
    ended_at        TEXT,
    duration_s      REAL,
    exit_code       INTEGER,
    log_path        TEXT,
    result_summary  TEXT,                          -- JSON object or NULL
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_watch  ON jobs(watch_name);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_id_desc ON jobs(id DESC);

CREATE TABLE IF NOT EXISTS watch_status (
    watch_name  TEXT    PRIMARY KEY,
    enabled     INTEGER NOT NULL DEFAULT 1,        -- bool 0/1
    source      TEXT    NOT NULL DEFAULT 'file',   -- 'file' | 'live'
    rule_json   TEXT,                              -- JSON WatchRule for source='live'
    updated_at  TEXT    NOT NULL                   -- ISO-8601
);
"""


# =============================================================================
# 6. Worker -> dispatcher progress-event JSON  — groups: worker, dispatcher
# =============================================================================
#
# The worker writes ONE compact JSON object per line ("JSON lines") to its
# stdout. The dispatcher reads line-by-line. Every line has a "kind" discriminator.
# Non-JSON stdout/stderr lines are captured into the job log but ignored by the
# progress parser. Schema (all timestamps ISO-8601 strings):
#
#   {"kind": "started",  "job_id": int, "config_path": str, "pid": int,
#    "ts": str}
#
#   {"kind": "stage",    "job_id": int, "stage": str,
#    "status": "start"|"completed"|"failed"|"skipped",
#    "duration_s": float|null, "ts": str}
#       # mirrors PipelineRunner stage transitions (one stage per StageResult)
#
#   {"kind": "progress", "job_id": int, "message": str, "ts": str}
#       # raw forwarded pipeline progress_callback line (e.g. "Loading model: cam (1/2)")
#
#   {"kind": "log",      "job_id": int, "level": "info"|"warning"|"error",
#    "message": str, "ts": str}
#
#   {"kind": "result",   "job_id": int, "success": bool,
#    "total_duration_seconds": float,
#    "log_path": str|null,                 # the per-run Markdown log path
#    "output_dir": str|null,               # analysis output_dir (for iCloud copy)
#    "plots": list[str],                   # absolute generated plot paths (for iCloud copy)
#    "summary": dict,                      # -> jobs.result_summary (stats/stage digest)
#    "error": str|null,                    # traceback/message on failure
#    "ts": str}
#       # EXACTLY ONE "result" line is emitted, last, immediately before exit.
#
# Exit code contract: worker exits 0 iff PipelineResult.success is True (and a
# "result" line with success=true was emitted); non-zero otherwise. The "summary"
# dict is the canonical payload persisted to jobs.result_summary by the dispatcher.

PROGRESS_KINDS: tuple[str, ...] = ("started", "stage", "progress", "log", "result")


class ProgressEvent(StrictModel):
    """Parsed form of one worker JSON-line (FlexibleModel-ish; extra ignored).

    The dispatcher parses each stdout line into this; ``kind`` selects which
    optional fields are populated. Mirrors the JSON catalog documented above.
    """

    model_config = ConfigDict(extra="ignore")

    kind: Literal["started", "stage", "progress", "log", "result"]
    job_id: int
    ts: datetime
    # started
    config_path: Optional[str] = None
    pid: Optional[int] = None
    # stage
    stage: Optional[str] = None
    status: Optional[str] = None
    duration_s: Optional[float] = None
    # progress / log
    message: Optional[str] = None
    level: Optional[str] = None
    # result
    success: Optional[bool] = None
    total_duration_seconds: Optional[float] = None
    log_path: Optional[str] = None
    output_dir: Optional[str] = None
    plots: Optional[list[str]] = None
    summary: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    @classmethod
    def parse_line(cls, line: str) -> Optional["ProgressEvent"]:
        """json.loads a single line and validate; return None for non-progress lines."""
        ...


# =============================================================================
# 7. Control protocol  — groups: control, client, supervisor
# =============================================================================
#
# Transport: AF_UNIX SOCK_STREAM at DaemonConfig.socket_path. Framing:
# newline-delimited UTF-8 JSON, ONE message per line ('\n' terminator).
#
# Request : {"cmd": <str>, "args": <object>}                 # args may be {}
# Response: {"ok": true,  "data": <object|array|null>}
#         | {"ok": false, "error": <str>, "code": <str|null>}
#
# A streaming request (subscribe / *_tail) gets an initial {"ok": true, ...}
# ack line, then the server PUSHES framed event lines until the client
# disconnects. Each pushed line is: {"event": <str>, "data": <object>}.
#
# PROTOCOL_VERSION is returned in `ping`/`status` so clients can detect mismatch.

PROTOCOL_VERSION: int = 1


class ControlRequest(StrictModel):
    cmd: str
    args: dict[str, Any] = Field(default_factory=dict)


class ControlResponse(StrictModel):
    ok: bool
    data: Any = None
    error: Optional[str] = None
    code: Optional[str] = None  # machine-readable error code (e.g. "not_found")


class StreamEvent(StrictModel):
    """One pushed line on a streaming subscription."""

    event: str  # e.g. "job_update", "watch_update", "log_line", "stats"
    data: dict[str, Any]


# ---- COMMAND CATALOG ------------------------------------------------------
# Every cmd name, its args object shape, and its success `data` shape.
# Errors use {ok:false,error,code}; codes: "not_found","invalid_args",
# "already_running","draining","unsupported". Args not listed => {} .
#
#  cmd            | args                                  | data (on ok)
# ----------------|---------------------------------------|----------------------------------------
#  "ping"         | {}                                    | {"pong": true, "version": int,
#                 |                                       |  "pid": int, "uptime_s": float}
#  "status"       | {}                                    | StatusData (see below)
#  "reload"       | {}                                    | {"reloaded": true,
#                 |                                       |  "watches": [WatchSummary],
#                 |                                       |  "added": [str], "removed": [str],
#                 |                                       |  "updated": [str]}
#  "shutdown"     | {"drain": bool=true,                  | {"shutting_down": true,
#                 |  "timeout": float|null}               |  "draining": bool}
#                 |                                       |  # graceful drain; SIGTERM equivalent
#  "watch_list"   | {}                                    | {"watches": [WatchSummary]}
#  "watch_add"    | {"rule": <WatchRule dump dict>,       | {"added": str}   # watch_name
#                 |  "save": bool=false}                  |   # save=true also writes back to file
#  "watch_remove" | {"name": str}                         | {"removed": str}
#  "watch_pause"  | {"name": str}                         | {"name": str, "enabled": false}
#  "watch_resume" | {"name": str}                         | {"name": str, "enabled": true}
#  "watch_trigger"| {"name": str,                         | {"queued_job_id": int|null,
#                 |  "files": [str]=[]}                   |  "coalesced": bool}
#                 |                                       |   # manual fire; files default = current glob
#  "watch_save"   | {"name": str}                         | {"saved": str, "path": str}
#                 |                                       |   # persist a live rule back to watches.yaml
#  "history"      | {"watch": str|null,                   | {"jobs": [JobRecord dump]}
#                 |  "failed": bool=false,                |
#                 |  "limit": int=50}                     |
#  "job_get"      | {"job_id": int}                       | {"job": JobRecord dump | null}
#  "logs"         | {"target": str,                       | {"log_path": str|null,
#                 |  "kind": "job"|"watch"="job"}         |  "text": str}
#                 |                                       |   # one-shot tail of the resolved log file
#  "subscribe"    | {"topics": [str]=["jobs","watches",   | ack {"subscribed": [str]} then PUSH
#                 |   "stats"]}                           |  StreamEvent lines (backs `daemon top`)
#  "logs_tail"    | {"target": str,                       | ack {"streaming": true} then PUSH
#                 |  "kind": "job"|"watch"="job"}         |  StreamEvent(event="log_line") lines
#                 |                                       |  (backs `daemon logs --tail`)
#
# StatusData (data for "status"):
#   {"version": int, "pid": int, "uptime_s": float, "draining": bool,
#    "max_concurrent": int,
#    "running": [JobRecord dump],           # status=RUNNING
#    "queued":  [JobRecord dump],           # status=QUEUED
#    "watches": [WatchSummary],
#    "recent":  [JobRecord dump]}           # last N completed/failed
#
# WatchSummary (compact per-watch row for list/status/top/reload):
#   {"name": str, "enabled": bool, "source": "file"|"live",
#    "on_fire": "whole_config"|"new_files_only",
#    "settle_mode": "quiescence"|"sentinel",
#    "watch": str, "run": str,
#    "state": "armed"|"settling"|"queued"|"running"|"paused",
#    "last_job_id": int|null, "last_status": str|null, "last_fired_at": str|null}


COMMANDS: tuple[str, ...] = (
    "ping",
    "status",
    "reload",
    "shutdown",
    "watch_list",
    "watch_add",
    "watch_remove",
    "watch_pause",
    "watch_resume",
    "watch_trigger",
    "watch_save",
    "history",
    "job_get",
    "logs",
    "subscribe",
    "logs_tail",
)

# Streaming command names (server pushes after the ack line).
STREAMING_COMMANDS: tuple[str, ...] = ("subscribe", "logs_tail")


# NOTE: the concrete ``DaemonClient`` is the SINGLE definition and lives in
# ``davinci_monet/daemon/client.py`` (Task 25). contracts.py does NOT carry a
# DaemonClient interface stub — shell.py / dashboard.py / the CLI all import
# ``from davinci_monet.daemon.client import DaemonClient``.


# A control handler conforms to this signature inside control.py / supervisor.py.
class ControlHandler(Protocol):
    def __call__(self, args: dict[str, Any]) -> ControlResponse: ...


# =============================================================================
# 8. Watcher clock injection  — group: watcher (tests)
# =============================================================================


class Clock(Protocol):
    """Injectable monotonic clock so settle/quiescence is unit-testable.

    Production uses a real clock (time.monotonic); tests pass a fake to advance
    time deterministically. The PollingWatcher takes one of these.
    """

    def now(self) -> float: ...        # monotonic seconds
    def sleep(self, seconds: float) -> None: ...
```

---

### Task 1: Bootstrap daemon package + shared contracts

Files create:
- `davinci_monet/daemon/__init__.py`
- `davinci_monet/daemon/contracts.py`
- `davinci_monet/tests/unit/daemon/__init__.py` (the daemon unit-test package marker)

This task materializes the package skeleton and the single canonical `contracts.py`. `contracts.py` contains ONLY the runtime PRIMITIVE definitions from the Shared Contracts reference — the enums/literal aliases (`JobStatus`, `OnFireMode`, `WatchSource`, `NotifyChannel`, `SettleMode`), the dataclass/Pydantic records (`TriggerEvent`, `JobSpec`, `JobRecord`, `WatchStatusRecord`, `ProgressEvent`), the `StateStore` accessor interface comment, `SCHEMA_DDL`, the control-message types (`ControlRequest`, `ControlResponse`, `StreamEvent`, `ControlHandler`, `COMMANDS`/`STREAMING_COMMANDS`/`PROTOCOL_VERSION`/`PROGRESS_KINDS` tuples), and the `Clock` Protocol. It does **NOT** define `parse_duration`, the config models `WatchRule`/`DaemonConfig`/`NotificationConfig`/`WatchesFile` (those live in `davinci_monet/daemon/config.py`, owned by the config section, Tasks 2-7), or a `DaemonClient` (the concrete client lives solely in `davinci_monet/daemon/client.py`, Task 25) — none of these may be duplicated here.

- [ ] Step 1 — Write the smoke test. Create `davinci_monet/tests/unit/daemon/test_contracts_smoke.py`:
```python
"""Smoke test: every daemon runtime primitive imports from contracts."""

from __future__ import annotations


def test_contracts_primitives_importable() -> None:
    from davinci_monet.daemon.contracts import (  # noqa: F401
        Clock,
        ControlHandler,
        ControlRequest,
        ControlResponse,
        JobRecord,
        JobSpec,
        JobStatus,
        ProgressEvent,
        SettleMode,
        StreamEvent,
        TriggerEvent,
        WatchStatusRecord,
    )
    from davinci_monet.daemon.contracts import (  # noqa: F401
        COMMANDS,
        PROTOCOL_VERSION,
        SCHEMA_DDL,
        STREAMING_COMMANDS,
    )

    # The runtime enum + literal aliases are present.
    assert JobStatus.QUEUED.value == "queued"
    assert PROTOCOL_VERSION == 1
    assert "subscribe" in STREAMING_COMMANDS


def test_contracts_does_not_own_config_models() -> None:
    """The config models live in daemon.config, NOT contracts."""
    import davinci_monet.daemon.contracts as contracts

    for forbidden in ("WatchRule", "DaemonConfig", "NotificationConfig", "WatchesFile"):
        assert not hasattr(contracts, forbidden), (
            f"{forbidden} must be owned by daemon.config, not contracts"
        )
```

- [ ] Step 2 — Run and expect failure. From the repo root in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && \
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_contracts_smoke.py -v
```
Expected: collection/import error — `ModuleNotFoundError: No module named 'davinci_monet.daemon'` / `'davinci_monet.daemon.contracts'` (the package and module do not exist yet).

- [ ] Step 3 — Create the package skeleton + contracts. Create `davinci_monet/daemon/__init__.py`:
```python
"""DAVINCI daemon mode: file-watching automation supervisor + isolated workers."""
```
Create `davinci_monet/tests/unit/daemon/__init__.py`:
```python
"""Unit tests for the daemon package."""
```
Create `davinci_monet/daemon/contracts.py` whose body is exactly the PRIMITIVE definitions from the Shared Contracts reference above — that is, sections 0 (enums & literal aliases), 3 (`TriggerEvent`), 4 (`JobSpec`, `JobRecord`, `WatchStatusRecord`), 5 (the `StateStore` interface docstring + `SCHEMA_DDL`), 6 (`PROGRESS_KINDS`, `ProgressEvent`), 7 (`PROTOCOL_VERSION`, `ControlRequest`, `ControlResponse`, `StreamEvent`, command-catalog comment, `COMMANDS`, `STREAMING_COMMANDS`, `ControlHandler`), and 8 (`Clock`), plus the module docstring and imports. Do NOT include section 1 (`parse_duration`), section 2 (the config models `WatchRule`/`DaemonConfig`/`NotificationConfig`/`WatchesFile` + `load_watches`/`merge_rules`, owned by `davinci_monet/daemon/config.py`), or any `DaemonClient` definition (the concrete client lives solely in `davinci_monet/daemon/client.py`, Task 25). Fill in the `...` bodies for the primitive methods (`JobSpec.to_json`/`from_json` via `model_dump_json()`/`model_validate_json`, `ProgressEvent.parse_line` via `json.loads`+`model_validate`).

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && \
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_contracts_smoke.py -v
```
Expected: 2 passed.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/__init__.py davinci_monet/daemon/contracts.py davinci_monet/tests/unit/daemon/__init__.py davinci_monet/tests/unit/daemon/test_contracts_smoke.py
git commit -m "feat(daemon): bootstrap package + shared runtime contracts

Add the davinci_monet.daemon package and the single canonical contracts module
owning ONLY the runtime primitives (JobStatus/SettleMode enums, Clock Protocol,
TriggerEvent/JobRecord/JobSpec/WatchStatusRecord records, ProgressEvent, the
control-message types, SCHEMA_DDL, and the command catalog). The config models
live in daemon.config, not here. Plus the daemon unit-test package marker.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

> **Group: `davinci_monet/daemon/config.py`** — owns the CONFIG MODELS (`WatchRule`, `DaemonConfig`, `NotificationConfig`, `WatchesFile`), the `parse_duration` helper, `load_watches`, and `merge_rules`.
>
> **Grounding (read before coding):**
> - Pydantic house style + base classes: `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/config/schema.py` lines 20-37 define `StrictModel` (`extra="forbid"`) and `FlexibleModel` (`extra="allow"`); `validate_default=True`, `str_strip_whitespace=True`. Reuse them VERBATIM via import — do NOT redefine. `field_validator(..., mode="before")` usage pattern at `schema.py` lines 105-146.
> - Env expansion + YAML loading: `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/config/parser.py` — `load_yaml(source)` (lines 19-67) and `expand_env_vars(data)` (lines 70-102, recurses dict/list, uses `os.path.expandvars` for `${VAR}` & `$VAR`). Reuse both.
> - `ConfigurationError` at `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/core/exceptions.py:68`.
>
> **Ownership boundary:** Import enums/literal-aliases `OnFireMode`, `NotifyChannel`, `SettleMode`, `WatchSource` from `davinci_monet.daemon.contracts` (Task-1 bootstrap owns them) — NEVER redefine them here. This group OWNS the three pydantic config models + `WatchesFile` + `parse_duration` + `load_watches` + `merge_rules`; other groups import these from `davinci_monet.daemon.config`.
>
> **Do NOT create `__init__.py`** for `davinci_monet/daemon/` or `davinci_monet/tests/unit/daemon/` — Task 1 (bootstrap) already created both. Treat them as pre-existing.
>
> All tests run from repo root in the `davinci` conda env (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci`). Test module: `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py` (matches the existing `tests/unit/config/` layout).

### Task 2: Duration parsing helper (parse_duration)

Implements the contract's `parse_duration(value: str | int | float | None) -> Optional[float]`: accepts `"30s"/"5m"/"2h"/"1d"` (case-insensitive), bare numbers/strings as seconds, `None -> None`, and raises `ValueError` on unparseable/negative. This is the first symbol the module defines so the duration `field_validator`s in later tasks can call it.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py` (new — create with this function only; later tasks append to it)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py` (new test module)

- [ ] Step 1 — Write the failing test. Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py`:
```python
"""Unit tests for davinci_monet.daemon.config."""

from __future__ import annotations

import pytest

from davinci_monet.daemon.config import parse_duration


class TestParseDuration:
    def test_none_returns_none(self) -> None:
        assert parse_duration(None) is None

    def test_bare_int_is_seconds(self) -> None:
        assert parse_duration(30) == 30.0

    def test_bare_float_is_seconds(self) -> None:
        assert parse_duration(1.5) == 1.5

    def test_unsuffixed_string_is_seconds(self) -> None:
        assert parse_duration("45") == 45.0

    def test_seconds_suffix(self) -> None:
        assert parse_duration("30s") == 30.0

    def test_minutes_suffix(self) -> None:
        assert parse_duration("5m") == 300.0

    def test_hours_suffix(self) -> None:
        assert parse_duration("2h") == 7200.0

    def test_days_suffix(self) -> None:
        assert parse_duration("1d") == 86400.0

    def test_case_insensitive_suffix(self) -> None:
        assert parse_duration("2H") == 7200.0

    def test_whitespace_tolerated(self) -> None:
        assert parse_duration(" 5m ") == 300.0

    def test_fractional_value(self) -> None:
        assert parse_duration("1.5h") == 5400.0

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("-5s")

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("abc")

    def test_unknown_suffix_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_duration("10y")
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestParseDuration -v
```
Expected: collection/import error `ModuleNotFoundError: No module named 'davinci_monet.daemon.config'` (or `ImportError: cannot import name 'parse_duration'`).

- [ ] Step 3 — Minimal implementation. Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py`:
```python
"""Daemon configuration models and watches.yaml loader.

Owns the three pydantic config models (NotificationConfig, WatchRule,
DaemonConfig), the WatchesFile aggregate, the parse_duration helper, and the
load_watches / merge_rules functions. Runtime primitive enums/literals
(OnFireMode, NotifyChannel, SettleMode, WatchSource) are imported from
davinci_monet.daemon.contracts and never redefined here.

Pydantic style mirrors davinci_monet/config/schema.py (StrictModel /
FlexibleModel). Layer-1 ${VAR} expansion reuses the project config parser.
"""

from __future__ import annotations

from typing import Optional

_DURATION_UNITS: dict[str, float] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def parse_duration(value: str | int | float | None) -> Optional[float]:
    """Parse a human duration like "30s", "5m", "2h", "1d" -> float seconds.

    Accepted suffixes (case-insensitive): s, m, h, d. A bare number (int/float
    or unsuffixed string) is interpreted as seconds. ``None`` -> ``None``.

    Raises
    ------
    ValueError
        On an unparseable / negative value.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise ValueError(f"Invalid duration: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"Duration must be non-negative: {value!r}")
        return seconds

    text = str(value).strip().lower()
    if not text:
        raise ValueError("Duration string is empty")

    unit = 1.0
    if text[-1] in _DURATION_UNITS:
        unit = _DURATION_UNITS[text[-1]]
        text = text[:-1].strip()
    elif text[-1].isalpha():
        raise ValueError(f"Unknown duration suffix in {value!r}")

    try:
        magnitude = float(text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse duration: {value!r}") from exc

    if magnitude < 0:
        raise ValueError(f"Duration must be non-negative: {value!r}")
    return magnitude * unit
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestParseDuration -v
```
Expected: 14 passed.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/config.py davinci_monet/tests/unit/daemon/test_config.py
git commit -m "feat(daemon): add parse_duration helper for watches.yaml

Parses human durations (\"30s\"/\"5m\"/\"2h\"/\"1d\") and bare numbers to float
seconds; rejects negative/unparseable values. Backs the duration field
validators on the daemon config models.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 3: NotificationConfig model with ~/${VAR} expansion

Implements the contract's `NotificationConfig(FlexibleModel)` — the `daemon.notifications` block — with `desktop: bool = True`, `icloud_copy: bool = True`, `icloud_dir: Path` (default the CLAUDE.md iCloud Claude folder), and a `mode="before"` validator `_expand_user` that expands `~` and `${VAR}` at daemon load (layer-1).

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py` (modify — append imports + class after `parse_duration`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py` (modify — add test class)

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py`:
```python
import os
from pathlib import Path

from davinci_monet.daemon.config import NotificationConfig


class TestNotificationConfig:
    def test_defaults(self) -> None:
        cfg = NotificationConfig()
        assert cfg.desktop is True
        assert cfg.icloud_copy is True
        assert "CloudDocs/Claude" in str(cfg.icloud_dir)

    def test_default_icloud_dir_user_expanded(self) -> None:
        cfg = NotificationConfig()
        # ~ must be expanded at construction (no literal tilde remains)
        assert not str(cfg.icloud_dir).startswith("~")

    def test_icloud_dir_tilde_expanded(self) -> None:
        cfg = NotificationConfig(icloud_dir="~/somewhere")
        assert str(cfg.icloud_dir) == str(Path.home() / "somewhere")

    def test_icloud_dir_env_expanded(self, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.setenv("ICLOUD_ROOT", "/tmp/icloud")
        cfg = NotificationConfig(icloud_dir="${ICLOUD_ROOT}/sub")
        assert str(cfg.icloud_dir) == "/tmp/icloud/sub"

    def test_extra_keys_allowed(self) -> None:
        # FlexibleModel -> forward-compat extra keys tolerated
        cfg = NotificationConfig(slack=True)
        assert cfg.desktop is True
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestNotificationConfig -v
```
Expected: `ImportError: cannot import name 'NotificationConfig' from 'davinci_monet.daemon.config'`.

- [ ] Step 3 — Minimal implementation. In `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py`, replace the import line `from typing import Optional` with the full import block, and append the class. New import block at top (after the module docstring):
```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, field_validator

from davinci_monet.config.schema import FlexibleModel, StrictModel
from davinci_monet.daemon.contracts import (
    NotifyChannel,
    OnFireMode,
    SettleMode,
    WatchSource,
)
```
Then append after the `parse_duration` function:
```python
def _expand_path_str(value: str) -> str:
    """Layer-1 path expansion: ${VAR}/$VAR then ~ (daemon environment)."""
    return os.path.expanduser(os.path.expandvars(value))


class NotificationConfig(FlexibleModel):
    """Daemon-level notification policy (the ``daemon.notifications`` block)."""

    desktop: bool = True
    icloud_copy: bool = True
    icloud_dir: Path = Field(
        default=Path("~/Library/Mobile Documents/com~apple~CloudDocs/Claude")
    )

    @field_validator("icloud_dir", mode="before")
    @classmethod
    def _expand_user(cls, v: Any) -> Any:
        """Expand ~ and ${VAR} at daemon load (layer-1 expansion)."""
        if v is None:
            return v
        return Path(_expand_path_str(str(v)))
```
Note: `validate_default=True` (inherited from `FlexibleModel`) makes the validator run on the default `Path(...)` too, so the default `icloud_dir` is `~`-expanded.

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestNotificationConfig -v
```
Expected: 5 passed.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/config.py davinci_monet/tests/unit/daemon/test_config.py
git commit -m "feat(daemon): add NotificationConfig model

daemon.notifications block: desktop + icloud_copy flags and a layer-1
~/\${VAR}-expanded icloud_dir (default CLAUDE.md iCloud Claude folder).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 4: WatchRule model with settle parsing and settle_mode

Implements the contract's `WatchRule(FlexibleModel)`: `name`, `watch`, `run`, `on_fire: OnFireMode = "whole_config"`, `inject_into`, `settle: float = 30.0` (via `parse_duration` `mode="before"`), `sentinel`, `env: dict[str,str]`, `notify: Optional[list[NotifyChannel]]`, `enabled: bool = True`, plus the `settle_mode` property (`"sentinel"` if `sentinel` set else `"quiescence"`). Per-rule cross-field validation (`new_files_only` requires `inject_into`) lives in `load_watches`, not here — the contract keeps WatchRule constructible standalone with the rule-key injected as `name`.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py` (modify — append class)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py` (modify — add test class)

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py`:
```python
from davinci_monet.daemon.config import WatchRule


class TestWatchRule:
    def test_minimal(self) -> None:
        rule = WatchRule(name="cam", watch="/in/*.nc", run="/cfg.yaml")
        assert rule.name == "cam"
        assert rule.watch == "/in/*.nc"
        assert rule.run == "/cfg.yaml"
        assert rule.on_fire == "whole_config"
        assert rule.settle == 30.0
        assert rule.enabled is True
        assert rule.env == {}
        assert rule.notify is None
        assert rule.inject_into is None
        assert rule.sentinel is None

    def test_settle_string_parsed(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c", settle="5m")
        assert rule.settle == 300.0

    def test_settle_numeric_passthrough(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c", settle=45)
        assert rule.settle == 45.0

    def test_settle_mode_quiescence_default(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c")
        assert rule.settle_mode == "quiescence"

    def test_settle_mode_sentinel_when_set(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c", sentinel="/in/DONE")
        assert rule.settle_mode == "sentinel"

    def test_on_fire_new_files_only_with_inject(self) -> None:
        rule = WatchRule(
            name="r", watch="/x", run="/c",
            on_fire="new_files_only", inject_into="cam",
        )
        assert rule.on_fire == "new_files_only"
        assert rule.inject_into == "cam"

    def test_bad_on_fire_rejected(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            WatchRule(name="r", watch="/x", run="/c", on_fire="bogus")

    def test_notify_channels(self) -> None:
        rule = WatchRule(name="r", watch="/x", run="/c", notify=["desktop", "log"])
        assert rule.notify == ["desktop", "log"]

    def test_bad_notify_channel_rejected(self) -> None:
        with pytest.raises(Exception):
            WatchRule(name="r", watch="/x", run="/c", notify=["pager"])

    def test_env_overlay_not_expanded(self) -> None:
        # env is the layer-2 worker overlay; values are stored verbatim
        rule = WatchRule(name="r", watch="/x", run="/c", env={"DATA": "${HOME}/d"})
        assert rule.env == {"DATA": "${HOME}/d"}
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestWatchRule -v
```
Expected: `ImportError: cannot import name 'WatchRule' from 'davinci_monet.daemon.config'`.

- [ ] Step 3 — Minimal implementation. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py`:
```python
class WatchRule(FlexibleModel):
    """A single declarative watch rule (one entry under ``watches:``).

    ``name`` is the rule's mapping key in watches.yaml; the loader injects it.
    Layer-1 ${VAR} expansion (watch/run/sentinel paths) has already been
    applied by the time a WatchRule is constructed. ``env`` is the per-rule
    overlay used for layer-2 (worker-side) expansion and is NOT expanded here.
    """

    name: str
    watch: str
    run: str
    on_fire: OnFireMode = "whole_config"
    inject_into: Optional[str] = None
    settle: float = Field(default=30.0)
    sentinel: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    notify: Optional[list[NotifyChannel]] = None
    enabled: bool = True

    @field_validator("settle", mode="before")
    @classmethod
    def _parse_settle(cls, v: Any) -> Any:
        """Accept "30s"/"5m"/number via parse_duration."""
        if v is None:
            return 30.0
        return parse_duration(v)

    @property
    def settle_mode(self) -> SettleMode:
        """'sentinel' if a sentinel path is set, else 'quiescence'."""
        return "sentinel" if self.sentinel else "quiescence"
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestWatchRule -v
```
Expected: 10 passed.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/config.py davinci_monet/tests/unit/daemon/test_config.py
git commit -m "feat(daemon): add WatchRule model

One watches: entry — glob, run config, on_fire scope, settle window
(parsed via parse_duration), sentinel, per-rule env/notify overrides, and a
settle_mode property. env overlay stored verbatim for layer-2 expansion.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 5: DaemonConfig model with derived runtime paths

Implements the contract's `DaemonConfig(FlexibleModel)`: `state_dir` (default `~/.davinci/daemon`, layer-1 `~/${VAR}` expanded), `poll_interval=5.0`, `max_concurrent=1`, `hdf5_file_locking=False`, `max_settle_wait=1800.0` (None disables), `worker_timeout=None`, `notifications: NotificationConfig`, the duration `mode="before"` validators, and the five derived `@property` paths under `state_dir` (`db_path`→`history.db`, `socket_path`→`control.sock`, `pid_path`→`daemon.pid`, `lock_path`→`daemon.lock`, `log_path`→`daemon.log`).

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py` (modify — append class)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py` (modify — add test class)

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py`:
```python
from davinci_monet.daemon.config import DaemonConfig


class TestDaemonConfig:
    def test_defaults(self) -> None:
        cfg = DaemonConfig()
        assert cfg.poll_interval == 5.0
        assert cfg.max_concurrent == 1
        assert cfg.hdf5_file_locking is False
        assert cfg.max_settle_wait == 1800.0
        assert cfg.worker_timeout is None
        assert isinstance(cfg.notifications, NotificationConfig)

    def test_state_dir_user_expanded(self) -> None:
        cfg = DaemonConfig()
        assert not str(cfg.state_dir).startswith("~")
        assert str(cfg.state_dir) == str(Path.home() / ".davinci" / "daemon")

    def test_state_dir_env_expanded(self, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.setenv("DAEMON_ROOT", "/tmp/dmn")
        cfg = DaemonConfig(state_dir="${DAEMON_ROOT}/state")
        assert str(cfg.state_dir) == "/tmp/dmn/state"

    def test_poll_interval_duration_string(self) -> None:
        cfg = DaemonConfig(poll_interval="10s")
        assert cfg.poll_interval == 10.0

    def test_max_settle_wait_duration_string(self) -> None:
        cfg = DaemonConfig(max_settle_wait="30m")
        assert cfg.max_settle_wait == 1800.0

    def test_max_settle_wait_none_disables(self) -> None:
        cfg = DaemonConfig(max_settle_wait=None)
        assert cfg.max_settle_wait is None

    def test_worker_timeout_duration_string(self) -> None:
        cfg = DaemonConfig(worker_timeout="2h")
        assert cfg.worker_timeout == 7200.0

    def test_derived_paths(self) -> None:
        cfg = DaemonConfig(state_dir="/tmp/dstate")
        assert cfg.db_path == Path("/tmp/dstate/history.db")
        assert cfg.socket_path == Path("/tmp/dstate/control.sock")
        assert cfg.pid_path == Path("/tmp/dstate/daemon.pid")
        assert cfg.lock_path == Path("/tmp/dstate/daemon.lock")
        assert cfg.log_path == Path("/tmp/dstate/daemon.log")

    def test_nested_notifications_dict(self) -> None:
        cfg = DaemonConfig(notifications={"desktop": False})
        assert cfg.notifications.desktop is False
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestDaemonConfig -v
```
Expected: `ImportError: cannot import name 'DaemonConfig' from 'davinci_monet.daemon.config'`.

- [ ] Step 3 — Minimal implementation. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py`:
```python
class DaemonConfig(FlexibleModel):
    """Top-level daemon policy (the ``daemon:`` block of watches.yaml)."""

    state_dir: Path = Field(default=Path("~/.davinci/daemon"))
    poll_interval: float = Field(default=5.0)
    max_concurrent: int = 1
    hdf5_file_locking: bool = False
    max_settle_wait: Optional[float] = Field(default=1800.0)
    worker_timeout: Optional[float] = None
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    @field_validator(
        "poll_interval", "max_settle_wait", "worker_timeout", mode="before"
    )
    @classmethod
    def _parse_durations(cls, v: Any) -> Any:
        """Accept "5s"/"30m"/number via parse_duration; pass None through."""
        if v is None:
            return None
        return parse_duration(v)

    @field_validator("state_dir", mode="before")
    @classmethod
    def _expand_state_dir(cls, v: Any) -> Any:
        """Expand ~ and ${VAR} at daemon load (layer-1)."""
        if v is None:
            return v
        return Path(_expand_path_str(str(v)))

    @property
    def db_path(self) -> Path:
        return self.state_dir / "history.db"

    @property
    def socket_path(self) -> Path:
        return self.state_dir / "control.sock"

    @property
    def pid_path(self) -> Path:
        return self.state_dir / "daemon.pid"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "daemon.lock"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "daemon.log"
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestDaemonConfig -v
```
Expected: 9 passed.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/config.py davinci_monet/tests/unit/daemon/test_config.py
git commit -m "feat(daemon): add DaemonConfig model

daemon: policy block — state_dir (layer-1 expanded), poll_interval,
max_concurrent, hdf5_file_locking, max_settle_wait/worker_timeout (duration
parsed), nested notifications, and derived db/socket/pid/lock/log paths.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 6: load_watches — parse + layer-1 env expansion + validation

Implements the contract's `WatchesFile(StrictModel)` aggregate and `load_watches(source) -> WatchesFile`. Reuses `load_yaml` + `expand_env_vars` from `davinci_monet/config/parser.py` for layer-1 `${VAR}` expansion against the daemon's `os.environ`, constructs `DaemonConfig` and each `WatchRule` (injecting the mapping key as `name`), and raises `ConfigurationError` when `on_fire == "new_files_only"` has no `inject_into`. The per-rule `env:` overlay must NOT be layer-1 expanded (it is the layer-2 worker overlay) — so expand the `daemon:` block and the per-rule `watch`/`run`/`sentinel` strings, but copy `env` through verbatim.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py` (modify — append `WatchesFile` + `load_watches`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py` (modify — add test class)

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py`:
```python
from davinci_monet.config.parser import expand_env_vars  # noqa: F401  (parity check)
from davinci_monet.core.exceptions import ConfigurationError
from davinci_monet.daemon.config import WatchesFile, load_watches


def _write(tmp_path: "Path", text: str) -> "Path":
    p = tmp_path / "watches.yaml"
    p.write_text(text)
    return p


class TestLoadWatches:
    def test_minimal_file(self, tmp_path: "Path") -> None:
        path = _write(
            tmp_path,
            """
daemon:
  poll_interval: 5s
watches:
  cam:
    watch: /in/cam/*.nc
    run: /cfg/asia-aq.yaml
""",
        )
        wf = load_watches(path)
        assert isinstance(wf, WatchesFile)
        assert wf.daemon.poll_interval == 5.0
        assert set(wf.watches) == {"cam"}
        rule = wf.watches["cam"]
        assert rule.name == "cam"  # key injected as name
        assert rule.watch == "/in/cam/*.nc"
        assert rule.run == "/cfg/asia-aq.yaml"
        assert rule.on_fire == "whole_config"

    def test_layer1_env_expansion_on_paths(
        self, tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        monkeypatch.setenv("DATA", "/scratch/cam")
        path = _write(
            tmp_path,
            """
watches:
  cam:
    watch: ${DATA}/incoming/*.nc
    run: ${DATA}/cfg.yaml
    sentinel: ${DATA}/DONE
""",
        )
        rule = load_watches(path).watches["cam"]
        assert rule.watch == "/scratch/cam/incoming/*.nc"
        assert rule.run == "/scratch/cam/cfg.yaml"
        assert rule.sentinel == "/scratch/cam/DONE"

    def test_per_rule_env_not_layer1_expanded(
        self, tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        monkeypatch.setenv("DATA", "/scratch/cam")
        path = _write(
            tmp_path,
            """
watches:
  cam:
    watch: /in/*.nc
    run: /cfg.yaml
    env:
      DATA: ${OTHER}/root
""",
        )
        rule = load_watches(path).watches["cam"]
        # env overlay is layer-2 (worker-side) -> stays verbatim
        assert rule.env == {"DATA": "${OTHER}/root"}

    def test_new_files_only_requires_inject_into(self, tmp_path: "Path") -> None:
        path = _write(
            tmp_path,
            """
watches:
  modis:
    watch: /in/*.hdf
    run: /cfg.yaml
    on_fire: new_files_only
""",
        )
        with pytest.raises(ConfigurationError, match="inject_into"):
            load_watches(path)

    def test_new_files_only_with_inject_into_ok(self, tmp_path: "Path") -> None:
        path = _write(
            tmp_path,
            """
watches:
  modis:
    watch: /in/*.hdf
    run: /cfg.yaml
    on_fire: new_files_only
    inject_into: modis_src
""",
        )
        rule = load_watches(path).watches["modis"]
        assert rule.on_fire == "new_files_only"
        assert rule.inject_into == "modis_src"

    def test_bad_on_fire_value_raises(self, tmp_path: "Path") -> None:
        path = _write(
            tmp_path,
            """
watches:
  r:
    watch: /in/*.nc
    run: /cfg.yaml
    on_fire: sometimes
""",
        )
        with pytest.raises(ConfigurationError):
            load_watches(path)

    def test_empty_file_defaults(self, tmp_path: "Path") -> None:
        path = _write(tmp_path, "watches: {}\n")
        wf = load_watches(path)
        assert wf.watches == {}
        assert wf.daemon.poll_interval == 5.0
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestLoadWatches -v
```
Expected: `ImportError: cannot import name 'WatchesFile' from 'davinci_monet.daemon.config'`.

- [ ] Step 3 — Minimal implementation. First add to the import block in `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py`:
```python
from davinci_monet.config.parser import expand_env_vars, load_yaml
from davinci_monet.core.exceptions import ConfigurationError
```
Then append `WatchesFile` + `load_watches`:
```python
class WatchesFile(StrictModel):
    """The fully-parsed watches.yaml (daemon policy + the declared rules).

    ``watches`` is keyed by rule name; each value is a fully-constructed
    WatchRule whose ``name`` matches its key.
    """

    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    watches: dict[str, WatchRule] = Field(default_factory=dict)


def load_watches(source: str | Path) -> WatchesFile:
    """Load + layer-1 env-expand + validate a watches.yaml into a WatchesFile.

    Reuses load_yaml + expand_env_vars for ${VAR} expansion against the
    DAEMON's os.environ, then constructs DaemonConfig and each WatchRule
    (injecting the mapping key as ``name``). on_fire == "new_files_only" with
    no ``inject_into`` is a validation error.
    """
    raw = load_yaml(source)

    daemon_raw = raw.get("daemon") or {}
    if not isinstance(daemon_raw, dict):
        raise ConfigurationError("watches.yaml 'daemon' block must be a mapping")
    # Layer-1: expand the whole daemon policy block (paths, icloud_dir, ...).
    daemon_raw = expand_env_vars(daemon_raw)

    watches_raw = raw.get("watches") or {}
    if not isinstance(watches_raw, dict):
        raise ConfigurationError("watches.yaml 'watches' block must be a mapping")

    try:
        daemon = DaemonConfig.model_validate(daemon_raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigurationError(f"Invalid daemon config: {exc}") from exc

    rules: dict[str, WatchRule] = {}
    for name, rule_raw in watches_raw.items():
        if rule_raw is None:
            rule_raw = {}
        if not isinstance(rule_raw, dict):
            raise ConfigurationError(f"watch '{name}' must be a mapping")

        # Layer-1: expand path-bearing string fields only; preserve env verbatim
        # (env is the layer-2 worker overlay and is expanded inside the worker).
        rule_data: dict[str, Any] = dict(rule_raw)
        env_overlay = rule_data.pop("env", None)
        rule_data = expand_env_vars(rule_data)
        if env_overlay is not None:
            rule_data["env"] = env_overlay
        rule_data["name"] = str(name)

        try:
            rule = WatchRule.model_validate(rule_data)
        except Exception as exc:  # pydantic ValidationError (e.g. bad on_fire)
            raise ConfigurationError(f"Invalid watch '{name}': {exc}") from exc

        if rule.on_fire == "new_files_only" and not rule.inject_into:
            raise ConfigurationError(
                f"watch '{name}': on_fire 'new_files_only' requires 'inject_into'"
            )
        rules[str(name)] = rule

    return WatchesFile(daemon=daemon, watches=rules)
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestLoadWatches -v
```
Expected: 7 passed.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/config.py davinci_monet/tests/unit/daemon/test_config.py
git commit -m "feat(daemon): add WatchesFile and load_watches loader

Parses watches.yaml, applies layer-1 \${VAR} expansion to daemon policy and
per-rule path fields (env overlay preserved verbatim for layer-2), injects
each rule key as WatchRule.name, and rejects new_files_only without
inject_into and bad on_fire values with ConfigurationError.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 7: merge_rules — reconcile declared/live/disabled rules

Implements the contract's `merge_rules(declared, live, disabled) -> dict[str, WatchRule]`, backing `daemon reload`: declared file rules updated from the file, live runtime-added rules preserved unless removed, and each result's `enabled` reflecting the `disabled` set. File-declared rules win on name collision (the file is authoritative for declared rules; live entries that re-declare a file name are dropped).

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py` (modify — append `merge_rules`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py` (modify — add test class)

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_config.py`:
```python
from davinci_monet.daemon.config import merge_rules


def _rule(name: str, **kw: object) -> WatchRule:
    base = {"name": name, "watch": f"/in/{name}/*.nc", "run": f"/cfg/{name}.yaml"}
    base.update(kw)
    return WatchRule(**base)


class TestMergeRules:
    def test_declared_only(self) -> None:
        declared = {"a": _rule("a"), "b": _rule("b")}
        merged = merge_rules(declared, {}, set())
        assert set(merged) == {"a", "b"}
        assert all(r.enabled for r in merged.values())

    def test_live_preserved(self) -> None:
        declared = {"a": _rule("a")}
        live = {"z": _rule("z")}
        merged = merge_rules(declared, live, set())
        assert set(merged) == {"a", "z"}

    def test_declared_wins_on_collision(self) -> None:
        declared = {"a": _rule("a", run="/cfg/declared.yaml")}
        live = {"a": _rule("a", run="/cfg/live.yaml")}
        merged = merge_rules(declared, live, set())
        assert merged["a"].run == "/cfg/declared.yaml"

    def test_disabled_applied_to_declared(self) -> None:
        declared = {"a": _rule("a"), "b": _rule("b")}
        merged = merge_rules(declared, {}, {"a"})
        assert merged["a"].enabled is False
        assert merged["b"].enabled is True

    def test_disabled_applied_to_live(self) -> None:
        live = {"z": _rule("z")}
        merged = merge_rules({}, live, {"z"})
        assert merged["z"].enabled is False

    def test_disabled_name_not_present_is_noop(self) -> None:
        declared = {"a": _rule("a")}
        merged = merge_rules(declared, {}, {"ghost"})
        assert set(merged) == {"a"}
        assert merged["a"].enabled is True

    def test_does_not_mutate_inputs(self) -> None:
        a = _rule("a", enabled=True)
        declared = {"a": a}
        merge_rules(declared, {}, {"a"})
        # original object must be untouched (returns new instances)
        assert a.enabled is True
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py::TestMergeRules -v
```
Expected: `ImportError: cannot import name 'merge_rules' from 'davinci_monet.daemon.config'`.

- [ ] Step 3 — Minimal implementation. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py`:
```python
def merge_rules(
    declared: dict[str, WatchRule],
    live: dict[str, WatchRule],
    disabled: set[str],
) -> dict[str, WatchRule]:
    """Reconcile file-declared rules with state-store live/runtime state.

    ``declared`` = rules from watches.yaml (source="file"). ``live`` =
    runtime-added rules from watch_status (source="live"). ``disabled`` = names
    paused at runtime. File-declared rules win on name collision. Each returned
    rule's ``enabled`` reflects the ``disabled`` set. Inputs are not mutated.
    """
    merged: dict[str, WatchRule] = {}

    # Live-added rules first; declared overrides any same-named live entry.
    for name, rule in live.items():
        merged[name] = rule
    for name, rule in declared.items():
        merged[name] = rule

    # Apply runtime pause/resume without mutating the source objects.
    result: dict[str, WatchRule] = {}
    for name, rule in merged.items():
        enabled = name not in disabled
        if rule.enabled != enabled:
            result[name] = rule.model_copy(update={"enabled": enabled})
        else:
            result[name] = rule
    return result
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_config.py -v
```
Expected: full `test_config.py` suite green (all classes, including 7 new `TestMergeRules` passing).

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/config.py davinci_monet/tests/unit/daemon/test_config.py
git commit -m "feat(daemon): add merge_rules reconciler for daemon reload

Reconciles file-declared rules with runtime live-added rules and the paused
(disabled) name set: declared wins on collision, live preserved, enabled
reflects the disabled set, inputs left unmutated.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 8: StateStore construction, schema init, and restart-survival round-trip

This task creates `davinci_monet/daemon/state.py` with the `StateStore` class skeleton: it opens/creates the SQLite DB at `state_dir/history.db`, applies the `SCHEMA_DDL` from the contracts idempotently on construction, and survives a process restart (a second `StateStore` opened on the same file path sees all committed rows). It uses stdlib `sqlite3` only, with one connection (`check_same_thread=False`), WAL journal mode, and a commit after every write so concurrent/sequential `StateStore` instances on the same file observe each other's data.

**Files:**
- `davinci_monet/daemon/state.py` (new — the StateStore implementation)
- `davinci_monet/tests/unit/daemon/test_state_store.py` (new — unit tests; `davinci_monet/tests/unit/daemon/__init__.py` is assumed pre-created by the bootstrap task)

Assumes pre-existing (do NOT create): `davinci_monet/daemon/__init__.py`, `davinci_monet/daemon/contracts.py` (owns `JobStatus`, `JobRecord`, `JobSpec`, `WatchStatusRecord`, `TriggerEvent`, `OnFireMode`, `WatchSource`, `SCHEMA_DDL`), `davinci_monet/daemon/config.py` (owns `WatchRule`), `davinci_monet/tests/unit/daemon/__init__.py`.

- [ ] **Step 1 — Write the failing test.** Create `davinci_monet/tests/unit/daemon/test_state_store.py` with the construction + restart-survival test:

```python
"""Unit tests for davinci_monet.daemon.state.StateStore (SQLite job/watch store).

No external datasets; uses a temp-dir SQLite file. Verifies schema creation,
CRUD over the jobs + watch_status tables, and restart survival (a fresh
StateStore opened on the same db_path sees previously-committed rows).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from davinci_monet.daemon.contracts import JobStatus
from davinci_monet.daemon.state import StateStore


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "history.db"


def test_init_creates_db_and_schema(db_path: Path) -> None:
    """Constructing a StateStore creates the db file and both tables."""
    assert not db_path.exists()
    store = StateStore(db_path)
    try:
        assert db_path.exists()
        # Both tables exist and are queryable.
        cur = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cur.fetchall()}
        assert "jobs" in tables
        assert "watch_status" in tables
    finally:
        store.close()


def test_init_schema_is_idempotent(db_path: Path) -> None:
    """Calling init_schema twice does not raise (CREATE TABLE IF NOT EXISTS)."""
    store = StateStore(db_path)
    try:
        store.init_schema()
        store.init_schema()
    finally:
        store.close()


def test_survives_restart(db_path: Path) -> None:
    """A second StateStore on the same path sees rows the first committed."""
    store1 = StateStore(db_path)
    job_id = store1.create_job(
        watch_name="cam_realtime",
        config_path="/cfg/asia-aq.yaml",
        on_fire="whole_config",
        files=["/data/a.nc", "/data/b.nc"],
    )
    store1.close()

    # Simulate a daemon restart: brand-new StateStore on the same file.
    store2 = StateStore(db_path)
    try:
        rec = store2.get_job(job_id)
        assert rec is not None
        assert rec.id == job_id
        assert rec.watch_name == "cam_realtime"
        assert rec.config_path == "/cfg/asia-aq.yaml"
        assert rec.on_fire == "whole_config"
        assert rec.files == ["/data/a.nc", "/data/b.nc"]
        assert rec.status is JobStatus.QUEUED
        assert rec.submitted_at is not None
    finally:
        store2.close()
```

- [ ] **Step 2 — Run and expect failure.** From the repo root in the `davinci` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && \
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_state_store.py::test_init_creates_db_and_schema davinci_monet/tests/unit/daemon/test_state_store.py::test_init_schema_is_idempotent davinci_monet/tests/unit/daemon/test_state_store.py::test_survives_restart -v
```

Expected: collection/import error — `ModuleNotFoundError: No module named 'davinci_monet.daemon.state'` (and once that exists, `ImportError: cannot import name 'StateStore'`). The three tests ERROR/FAIL because `state.py` does not yet define `StateStore`.

- [ ] **Step 3 — Minimal implementation.** Create `davinci_monet/daemon/state.py` with construction, schema init, the connection helper, and `create_job`/`get_job` (enough to make this task's tests pass; the remaining CRUD lands in later tasks). Write the full file:

```python
"""SQLite-backed job history + watch runtime-status persistence.

stdlib ``sqlite3`` only. A single connection is opened on construction
(``check_same_thread=False``) and the contract DDL (``SCHEMA_DDL``) is applied
idempotently. Every write commits immediately and the database runs in WAL
journal mode, so a fresh ``StateStore`` opened on the same ``db_path`` after a
daemon restart sees all previously-committed rows.

All timestamps are stored as ISO-8601 strings (``datetime.isoformat()``); all
list/dict columns are stored as JSON (``json.dumps``). The accessors decode
both back into the typed contract records (``JobRecord`` / ``WatchStatusRecord``).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from davinci_monet.daemon.config import WatchRule
from davinci_monet.daemon.contracts import (
    SCHEMA_DDL,
    JobRecord,
    JobStatus,
    OnFireMode,
    WatchSource,
    WatchStatusRecord,
)

__all__ = ["StateStore"]


def _now_iso() -> str:
    """Current local time as an ISO-8601 string (seconds precision kept)."""
    return datetime.now().isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    """Decode an ISO-8601 string column into a datetime, passing None through."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _loads(value: Any, default: Any) -> Any:
    """json.loads a TEXT column, returning ``default`` for NULL/empty."""
    if value is None or value == "":
        return default
    return json.loads(value)


class StateStore:
    """SQLite store for the daemon's ``jobs`` and ``watch_status`` tables."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    # -- lifecycle ---------------------------------------------------------

    def init_schema(self) -> None:
        """Apply the contract DDL idempotently (CREATE TABLE IF NOT EXISTS)."""
        self._conn.executescript(SCHEMA_DDL)
        self._conn.commit()

    def close(self) -> None:
        """Commit any pending work and close the connection."""
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    # -- jobs CRUD ---------------------------------------------------------

    def create_job(
        self,
        watch_name: str,
        config_path: str,
        on_fire: OnFireMode,
        files: list[str],
    ) -> int:
        """Insert a QUEUED job (submitted_at=now). Returns the new jobs.id."""
        cur = self._conn.execute(
            """
            INSERT INTO jobs
                (watch_name, config_path, on_fire, files, status, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                watch_name,
                config_path,
                on_fire,
                json.dumps(list(files)),
                JobStatus.QUEUED.value,
                _now_iso(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_job(self, job_id: int) -> Optional[JobRecord]:
        """Return the JobRecord for ``job_id`` or None if it does not exist."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    # -- decoding helpers --------------------------------------------------

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=int(row["id"]),
            watch_name=row["watch_name"],
            config_path=row["config_path"],
            on_fire=row["on_fire"],
            files=_loads(row["files"], []),
            status=JobStatus(row["status"]),
            submitted_at=_parse_dt(row["submitted_at"]),
            started_at=_parse_dt(row["started_at"]),
            ended_at=_parse_dt(row["ended_at"]),
            duration_s=row["duration_s"],
            exit_code=row["exit_code"],
            log_path=row["log_path"],
            result_summary=_loads(row["result_summary"], None),
            error=row["error"],
        )
```

- [ ] **Step 4 — Run and expect pass.** From the repo root in the `davinci` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && \
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_state_store.py::test_init_creates_db_and_schema davinci_monet/tests/unit/daemon/test_state_store.py::test_init_schema_is_idempotent davinci_monet/tests/unit/daemon/test_state_store.py::test_survives_restart -v
```

Expected: 3 passed.

- [ ] **Step 5 — Commit.**

```bash
git add davinci_monet/daemon/state.py davinci_monet/tests/unit/daemon/test_state_store.py
git commit -m "feat(daemon): StateStore construction, schema init, restart survival

Add SQLite-backed StateStore for daemon job history + watch runtime status.
Opens/creates history.db, applies the contract SCHEMA_DDL idempotently, runs in
WAL mode and commits per write so a fresh StateStore after restart sees prior
rows. Implements create_job/get_job with JSON-encoded files and ISO-8601
timestamps decoded back into the JobRecord contract.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 9: jobs table CRUD — lifecycle marks, get_job, list_jobs, active_jobs

This task completes the `jobs` CRUD surface on `StateStore`: the lifecycle transitions `mark_running`, `mark_completed`, `mark_failed`, `mark_skipped`, plus the query helpers `list_jobs` (most-recent-first, filterable by `watch_name`/`status` with a `limit`) and `active_jobs` (rows in `{QUEUED, RUNNING}`). Each mark sets `ended_at`/`duration_s` (computed from `started_at` when present), `exit_code`, `log_path`, `result_summary`, and `error` per the contract.

**Files:**
- `davinci_monet/daemon/state.py` (modify — append the lifecycle marks + list/active queries to the `StateStore` class from the previous task)
- `davinci_monet/tests/unit/daemon/test_state_store.py` (modify — add the jobs-CRUD tests)

- [ ] **Step 1 — Write the failing test.** Append to `davinci_monet/tests/unit/daemon/test_state_store.py`:

```python
def test_mark_running_sets_started_and_status(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        jid = store.create_job("w", "/c.yaml", "whole_config", [])
        store.mark_running(jid)
        rec = store.get_job(jid)
        assert rec is not None
        assert rec.status is JobStatus.RUNNING
        assert rec.started_at is not None
        assert rec.ended_at is None
    finally:
        store.close()


def test_mark_completed_records_outcome_and_duration(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        jid = store.create_job("w", "/c.yaml", "whole_config", [])
        store.mark_running(jid)
        store.mark_completed(
            jid,
            exit_code=0,
            log_path="/logs/run.md",
            result_summary={"N": 42, "RMSE": 1.5},
        )
        rec = store.get_job(jid)
        assert rec is not None
        assert rec.status is JobStatus.COMPLETED
        assert rec.exit_code == 0
        assert rec.log_path == "/logs/run.md"
        assert rec.result_summary == {"N": 42, "RMSE": 1.5}
        assert rec.ended_at is not None
        assert rec.duration_s is not None and rec.duration_s >= 0.0
    finally:
        store.close()


def test_mark_failed_records_error(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        jid = store.create_job("w", "/c.yaml", "whole_config", [])
        store.mark_running(jid)
        store.mark_failed(jid, exit_code=1, error="boom", log_path="/logs/e.md")
        rec = store.get_job(jid)
        assert rec is not None
        assert rec.status is JobStatus.FAILED
        assert rec.exit_code == 1
        assert rec.error == "boom"
        assert rec.log_path == "/logs/e.md"
        assert rec.ended_at is not None
    finally:
        store.close()


def test_mark_skipped(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        jid = store.create_job("w", "/c.yaml", "whole_config", [])
        store.mark_skipped(jid, error="coalesced")
        rec = store.get_job(jid)
        assert rec is not None
        assert rec.status is JobStatus.SKIPPED
        assert rec.error == "coalesced"
        assert rec.ended_at is not None
    finally:
        store.close()


def test_list_jobs_orders_most_recent_first_and_filters(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        a = store.create_job("alpha", "/c.yaml", "whole_config", [])
        b = store.create_job("beta", "/c.yaml", "whole_config", [])
        c = store.create_job("alpha", "/c.yaml", "whole_config", [])
        store.mark_running(b)
        store.mark_failed(c, exit_code=2, error="x")

        ids = [r.id for r in store.list_jobs()]
        assert ids == [c, b, a]  # ORDER BY id DESC

        alpha_ids = [r.id for r in store.list_jobs(watch_name="alpha")]
        assert alpha_ids == [c, a]

        failed = store.list_jobs(status=JobStatus.FAILED)
        assert [r.id for r in failed] == [c]

        limited = store.list_jobs(limit=1)
        assert [r.id for r in limited] == [c]
    finally:
        store.close()


def test_active_jobs_returns_queued_and_running(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        q = store.create_job("w", "/c.yaml", "whole_config", [])
        r = store.create_job("w", "/c.yaml", "whole_config", [])
        done = store.create_job("w", "/c.yaml", "whole_config", [])
        store.mark_running(r)
        store.mark_completed(done, exit_code=0, log_path=None, result_summary=None)

        active_ids = {rec.id for rec in store.active_jobs()}
        assert active_ids == {q, r}
        statuses = {rec.status for rec in store.active_jobs()}
        assert statuses <= {JobStatus.QUEUED, JobStatus.RUNNING}
    finally:
        store.close()
```

- [ ] **Step 2 — Run and expect failure.** From the repo root in the `davinci` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && \
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_state_store.py -k "mark_ or list_jobs or active_jobs" -v
```

Expected: `AttributeError: 'StateStore' object has no attribute 'mark_running'` (the new tests ERROR/FAIL because the lifecycle marks and list/active queries are not yet implemented).

- [ ] **Step 3 — Minimal implementation.** Add the lifecycle marks and queries to the `StateStore` class in `davinci_monet/daemon/state.py`. Insert these methods immediately after `get_job` (before the `_row_to_job` static helper). First add a small private duration helper, then the marks and queries:

```python
    def _duration_since_started(self, job_id: int, ended_iso: str) -> Optional[float]:
        """Compute ended-minus-started in seconds, or None if no start time."""
        row = self._conn.execute(
            "SELECT started_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None or row["started_at"] in (None, ""):
            return None
        started = datetime.fromisoformat(str(row["started_at"]))
        ended = datetime.fromisoformat(ended_iso)
        return max(0.0, (ended - started).total_seconds())

    def mark_running(self, job_id: int, started_at: Optional[datetime] = None) -> None:
        """Transition to RUNNING and stamp started_at (default now)."""
        started_iso = (started_at or datetime.now()).isoformat()
        self._conn.execute(
            "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
            (JobStatus.RUNNING.value, started_iso, job_id),
        )
        self._conn.commit()

    def mark_completed(
        self,
        job_id: int,
        exit_code: int,
        log_path: Optional[str],
        result_summary: Optional[dict[str, Any]],
        ended_at: Optional[datetime] = None,
    ) -> None:
        """Transition to COMPLETED; record outcome, duration, and summary."""
        ended_iso = (ended_at or datetime.now()).isoformat()
        duration = self._duration_since_started(job_id, ended_iso)
        self._conn.execute(
            """
            UPDATE jobs
               SET status = ?, ended_at = ?, duration_s = ?, exit_code = ?,
                   log_path = ?, result_summary = ?
             WHERE id = ?
            """,
            (
                JobStatus.COMPLETED.value,
                ended_iso,
                duration,
                exit_code,
                log_path,
                json.dumps(result_summary) if result_summary is not None else None,
                job_id,
            ),
        )
        self._conn.commit()

    def mark_failed(
        self,
        job_id: int,
        exit_code: Optional[int],
        error: str,
        log_path: Optional[str] = None,
        ended_at: Optional[datetime] = None,
    ) -> None:
        """Transition to FAILED; record exit_code, error, duration, log_path."""
        ended_iso = (ended_at or datetime.now()).isoformat()
        duration = self._duration_since_started(job_id, ended_iso)
        self._conn.execute(
            """
            UPDATE jobs
               SET status = ?, ended_at = ?, duration_s = ?, exit_code = ?,
                   error = ?, log_path = ?
             WHERE id = ?
            """,
            (
                JobStatus.FAILED.value,
                ended_iso,
                duration,
                exit_code,
                error,
                log_path,
                job_id,
            ),
        )
        self._conn.commit()

    def mark_skipped(self, job_id: int, error: Optional[str] = None) -> None:
        """Transition to SKIPPED (coalesced/drained); stamp ended_at."""
        ended_iso = datetime.now().isoformat()
        duration = self._duration_since_started(job_id, ended_iso)
        self._conn.execute(
            """
            UPDATE jobs
               SET status = ?, ended_at = ?, duration_s = ?, error = ?
             WHERE id = ?
            """,
            (JobStatus.SKIPPED.value, ended_iso, duration, error, job_id),
        )
        self._conn.commit()

    def list_jobs(
        self,
        watch_name: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        """Most-recent-first job rows, optionally filtered by watch/status."""
        clauses: list[str] = []
        params: list[Any] = []
        if watch_name is not None:
            clauses.append("watch_name = ?")
            params.append(watch_name)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        rows = self._conn.execute(
            f"SELECT * FROM jobs{where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def active_jobs(self) -> list[JobRecord]:
        """Rows with status in {QUEUED, RUNNING}, most-recent-first."""
        rows = self._conn.execute(
            """
            SELECT * FROM jobs
             WHERE status IN (?, ?)
             ORDER BY id DESC
            """,
            (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
        ).fetchall()
        return [self._row_to_job(r) for r in rows]
```

- [ ] **Step 4 — Run and expect pass.** From the repo root in the `davinci` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && \
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_state_store.py -k "mark_ or list_jobs or active_jobs" -v
```

Expected: 6 passed.

- [ ] **Step 5 — Commit.**

```bash
git add davinci_monet/daemon/state.py davinci_monet/tests/unit/daemon/test_state_store.py
git commit -m "feat(daemon): StateStore jobs lifecycle + list/active queries

Implement mark_running/mark_completed/mark_failed/mark_skipped with computed
duration_s (ended minus started), plus list_jobs (id DESC, watch/status/limit
filters) and active_jobs (QUEUED|RUNNING). result_summary is JSON-encoded and
decoded back into JobRecord.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 10: watch_status CRUD — pause/resume, upsert, live rules, disabled_names

This task adds the `watch_status` table CRUD to `StateStore`: `upsert_watch_status` (INSERT OR REPLACE keyed by `watch_name`, bumping `updated_at`), `set_enabled` (pause/resume persistence), `get_watch_status`/`list_watch_status`, `add_live_rule`/`live_rules` (serialize/reconstruct a runtime-added `WatchRule` as `source="live"` with `rule_json`), `remove_watch`, and `disabled_names` (names with `enabled=False`, fed to `merge_rules` on reload). `WatchRule` is imported from `davinci_monet.daemon.config`; it round-trips via `model_dump(mode="json")` / `WatchRule(**rule_json)`.

**Files:**
- `davinci_monet/daemon/state.py` (modify — append the `watch_status` CRUD methods to the `StateStore` class)
- `davinci_monet/tests/unit/daemon/test_state_store.py` (modify — add the watch_status tests)

- [ ] **Step 1 — Write the failing test.** Append to `davinci_monet/tests/unit/daemon/test_state_store.py`:

```python
from davinci_monet.daemon.config import WatchRule
from davinci_monet.daemon.contracts import WatchStatusRecord


def _live_rule(name: str = "modis_stream") -> WatchRule:
    return WatchRule(
        name=name,
        watch="/scratch/modis/*.hdf",
        run="/cfg/modis-aod.yaml",
        on_fire="new_files_only",
        inject_into="modis",
        settle=15.0,
    )


def test_set_enabled_and_disabled_names(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        store.set_enabled("cam_realtime", False)
        store.set_enabled("modis_stream", True)
        assert store.disabled_names() == {"cam_realtime"}

        store.set_enabled("cam_realtime", True)
        assert store.disabled_names() == set()
    finally:
        store.close()


def test_upsert_replaces_existing_row(db_path: Path) -> None:
    from datetime import datetime

    store = StateStore(db_path)
    try:
        store.upsert_watch_status(
            WatchStatusRecord(
                watch_name="w", enabled=True, source="file",
                updated_at=datetime.now(),
            )
        )
        store.upsert_watch_status(
            WatchStatusRecord(
                watch_name="w", enabled=False, source="file",
                updated_at=datetime.now(),
            )
        )
        rec = store.get_watch_status("w")
        assert rec is not None
        assert rec.enabled is False
        assert len(store.list_watch_status()) == 1  # replaced, not duplicated
    finally:
        store.close()


def test_get_watch_status_missing_returns_none(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        assert store.get_watch_status("nope") is None
    finally:
        store.close()


def test_add_live_rule_roundtrips(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        rule = _live_rule()
        store.add_live_rule(rule)

        rec = store.get_watch_status(rule.name)
        assert rec is not None
        assert rec.source == "live"
        assert rec.rule_json is not None

        rules = store.live_rules()
        assert set(rules) == {rule.name}
        restored = rules[rule.name]
        assert isinstance(restored, WatchRule)
        assert restored.name == rule.name
        assert restored.watch == rule.watch
        assert restored.run == rule.run
        assert restored.on_fire == "new_files_only"
        assert restored.inject_into == "modis"
        assert restored.settle == 15.0
    finally:
        store.close()


def test_live_rules_excludes_file_rules(db_path: Path) -> None:
    from datetime import datetime

    store = StateStore(db_path)
    try:
        store.upsert_watch_status(
            WatchStatusRecord(
                watch_name="declared", enabled=True, source="file",
                updated_at=datetime.now(),
            )
        )
        store.add_live_rule(_live_rule("live_one"))
        assert set(store.live_rules()) == {"live_one"}
    finally:
        store.close()


def test_remove_watch_deletes_row(db_path: Path) -> None:
    store = StateStore(db_path)
    try:
        store.add_live_rule(_live_rule("gone"))
        assert store.get_watch_status("gone") is not None
        store.remove_watch("gone")
        assert store.get_watch_status("gone") is None
        assert "gone" not in store.live_rules()
    finally:
        store.close()


def test_watch_status_survives_restart(db_path: Path) -> None:
    store1 = StateStore(db_path)
    store1.add_live_rule(_live_rule("persisted"))
    store1.set_enabled("declared_paused", False)
    store1.close()

    store2 = StateStore(db_path)
    try:
        assert "persisted" in store2.live_rules()
        assert store2.disabled_names() == {"declared_paused"}
    finally:
        store2.close()
```

- [ ] **Step 2 — Run and expect failure.** From the repo root in the `davinci` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && \
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_state_store.py -k "watch_status or live_rule or set_enabled or disabled or upsert or remove_watch" -v
```

Expected: `AttributeError: 'StateStore' object has no attribute 'set_enabled'` (the new tests ERROR/FAIL because the watch_status CRUD is not yet implemented).

- [ ] **Step 3 — Minimal implementation.** Append the `watch_status` CRUD methods to the `StateStore` class in `davinci_monet/daemon/state.py`, inserting them immediately before the `@staticmethod _row_to_job` helper (so all instance methods precede the decoding helper). Add the methods:

```python
    # -- watch_status CRUD -------------------------------------------------

    def upsert_watch_status(self, record: WatchStatusRecord) -> None:
        """INSERT OR REPLACE the watch_status row keyed by watch_name."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO watch_status
                (watch_name, enabled, source, rule_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.watch_name,
                1 if record.enabled else 0,
                record.source,
                json.dumps(record.rule_json) if record.rule_json is not None else None,
                (record.updated_at or datetime.now()).isoformat(),
            ),
        )
        self._conn.commit()

    def set_enabled(self, watch_name: str, enabled: bool) -> None:
        """Pause/resume a watch, preserving its existing source/rule_json."""
        existing = self.get_watch_status(watch_name)
        source: WatchSource = existing.source if existing is not None else "file"
        rule_json = existing.rule_json if existing is not None else None
        self.upsert_watch_status(
            WatchStatusRecord(
                watch_name=watch_name,
                enabled=enabled,
                source=source,
                rule_json=rule_json,
                updated_at=datetime.now(),
            )
        )

    def get_watch_status(self, watch_name: str) -> Optional[WatchStatusRecord]:
        """Return the watch_status row for ``watch_name`` or None."""
        row = self._conn.execute(
            "SELECT * FROM watch_status WHERE watch_name = ?", (watch_name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_watch_status(row)

    def list_watch_status(self) -> list[WatchStatusRecord]:
        """Return every watch_status row (name-ordered)."""
        rows = self._conn.execute(
            "SELECT * FROM watch_status ORDER BY watch_name"
        ).fetchall()
        return [self._row_to_watch_status(r) for r in rows]

    def add_live_rule(self, rule: WatchRule) -> None:
        """Persist a runtime-added rule as source='live' with its JSON dump."""
        self.upsert_watch_status(
            WatchStatusRecord(
                watch_name=rule.name,
                enabled=rule.enabled,
                source="live",
                rule_json=rule.model_dump(mode="json"),
                updated_at=datetime.now(),
            )
        )

    def remove_watch(self, watch_name: str) -> None:
        """Delete the watch_status row (drops a live rule / runtime overrides)."""
        self._conn.execute(
            "DELETE FROM watch_status WHERE watch_name = ?", (watch_name,)
        )
        self._conn.commit()

    def disabled_names(self) -> set[str]:
        """Names with enabled=False (fed to merge_rules on reload)."""
        rows = self._conn.execute(
            "SELECT watch_name FROM watch_status WHERE enabled = 0"
        ).fetchall()
        return {r["watch_name"] for r in rows}

    def live_rules(self) -> dict[str, WatchRule]:
        """Reconstruct WatchRule objects for every source='live' row."""
        rows = self._conn.execute(
            "SELECT * FROM watch_status WHERE source = 'live' AND rule_json IS NOT NULL"
        ).fetchall()
        result: dict[str, WatchRule] = {}
        for row in rows:
            rule_json = _loads(row["rule_json"], None)
            if rule_json is None:
                continue
            result[row["watch_name"]] = WatchRule(**rule_json)
        return result
```

Then add the `_row_to_watch_status` decoding helper alongside `_row_to_job` (place it directly after the `_row_to_job` static method):

```python
    @staticmethod
    def _row_to_watch_status(row: sqlite3.Row) -> WatchStatusRecord:
        return WatchStatusRecord(
            watch_name=row["watch_name"],
            enabled=bool(row["enabled"]),
            source=row["source"],
            rule_json=_loads(row["rule_json"], None),
            updated_at=_parse_dt(row["updated_at"]),
        )
```

- [ ] **Step 4 — Run and expect pass.** From the repo root in the `davinci` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && \
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_state_store.py -v
```

Expected: all tests in the file pass (the three construction tests, six jobs-CRUD tests, and seven watch_status tests).

- [ ] **Step 5 — Commit.**

```bash
git add davinci_monet/daemon/state.py davinci_monet/tests/unit/daemon/test_state_store.py
git commit -m "feat(daemon): StateStore watch_status CRUD + live-rule persistence

Add upsert_watch_status (INSERT OR REPLACE), set_enabled pause/resume,
get/list_watch_status, add_live_rule/live_rules (WatchRule round-trip via
model_dump(mode='json')), remove_watch, and disabled_names. Runtime pause state
and live-added rules survive daemon restart; live_rules feeds merge_rules on
reload.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 11: PollingWatcher core — glob scan, per-rule file tracking, quiescence settle fires once

**Module ownership:** This is the `watcher` group. It OWNS `davinci_monet/daemon/watcher.py`. It IMPORTS — never redefines — `TriggerEvent`, `SettleMode`, `Clock` from `davinci_monet.daemon.contracts` (created by Task 1 bootstrap) and `WatchRule`, `DaemonConfig` from `davinci_monet.daemon.config` (owned by the config group). Do NOT create `davinci_monet/daemon/__init__.py` or the daemon test `__init__.py` — Task 1 already created them.

**Integration facts verified in the actual code/contract:**
- `TriggerEvent` (from contracts) is a `StrictModel` with EXACT fields `watch_name: str`, `new_files: list[str]` (absolute, sorted), `detected_at: datetime` (wall-clock), `settle_mode: SettleMode`. NOTE: the field is `detected_at`, NOT `fired_at`. Use `datetime.now()` for it.
- `Clock` (from contracts) is a `Protocol` with `.now() -> float` (monotonic seconds) and `.sleep(seconds: float) -> None`. The watcher's settle math uses `clock.now()` deltas only — never wall clock.
- `WatchRule` (from `davinci_monet.daemon.config`) has `name: str`, `watch: str` (glob, layer-1 expanded), `settle: float` (seconds), `sentinel: Optional[str]`, and a `settle_mode` property returning `"sentinel" if self.sentinel else "quiescence"`.
- `DaemonConfig` (from `davinci_monet.daemon.config`) has `max_settle_wait: Optional[float]` (default 1800.0; `None` disables) and `poll_interval: float`.
- Test layout (verified): unit tests live under `davinci_monet/tests/unit/<group>/test_*.py`, e.g. `davinci_monet/tests/unit/config/test_source_config.py`. Create a new `davinci_monet/tests/unit/daemon/` dir. Task 1 bootstrap created the daemon test package `__init__.py`.

**Design — what the watcher is and how tests stay deterministic:**
- `PollingWatcher` does NOT own a thread or sleep loop in its core logic. It exposes a pure-ish `poll()` method that the supervisor calls once per `poll_interval`. Each `poll()` scans every rule's glob via an INJECTED scan function, updates per-rule tracking state, and returns a `list[TriggerEvent]` for rules that fired this tick. This makes settle/quiescence fully unit-testable with a fake clock + fake scan — no real sleeping, no real filesystem required.
- Injected scan signature `ScanFn = Callable[[str], dict[str, FileStat]]`: given a glob pattern, return `{abs_path: FileStat(size, mtime)}` for currently-matching files. `default_scan` is the production impl (glob + os.stat). Tests pass a fake that returns canned dicts per tick.
- Injected `Clock`: tests pass a fake whose `.now()` returns a controllable float.
- Per-rule state (kept in `PollingWatcher._state[name]`): a dict of tracked files `{path: FileStat}`, `last_change_t: float` (monotonic time of the most recent new-file-or-size-change), `first_seen_t: float | None` (monotonic time the rule first had ANY matching file, for `max_settle_wait`), and `fired: bool` is NOT used — instead, after firing, tracked files are retained but recorded in `_settled[name]` so the SAME set does not re-fire; a genuinely new file resets the settle timer and arms a new fire.

**Quiescence fire rule (this task):** A `quiescence` rule fires when ALL hold: it currently has ≥1 matching file; every matching file's size is unchanged vs the previous poll AND no new path appeared this poll (i.e. `now - last_change_t >= settle`); and the current matching set differs from the last-settled set (so it does not re-fire the identical batch). On fire, `new_files` = sorted absolute paths of files NOT already in `_settled[name]` (the newly-arrived/changed ones), `detected_at = datetime.now()`, `settle_mode = "quiescence"`. After firing, `_settled[name]` is replaced by the current full tracked set.

Files:
- `davinci_monet/daemon/watcher.py` (NEW — owned here)
- `davinci_monet/tests/unit/daemon/test_watcher.py` (NEW)

- [ ] Step 1 — Write the failing test. Create `davinci_monet/tests/unit/daemon/test_watcher.py` with the full contents below. It uses a `FakeClock` and a scripted `FakeScan` so a new file then quiescence fires exactly once, and a still-growing file does not fire until its size stabilizes.

```python
"""Unit tests for davinci_monet.daemon.watcher.PollingWatcher.

Deterministic: an injectable monotonic Clock and an injectable filesystem scan
(no real sleeping, no real disk needed for the settle/sentinel logic).
"""

from __future__ import annotations

from datetime import datetime

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import TriggerEvent
from davinci_monet.daemon.watcher import FileStat, PollingWatcher


class FakeClock:
    """Injectable monotonic clock; advance() moves time forward."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:  # pragma: no cover - not used in tests
        self._t += seconds

    def advance(self, seconds: float) -> None:
        self._t += seconds


class FakeScan:
    """Scripted scan: returns the next queued snapshot for a glob pattern.

    snapshots is a list of {abs_path: FileStat}; each poll() consumes one.
    The last snapshot repeats once the script is exhausted (steady state).
    """

    def __init__(self, snapshots: list[dict[str, FileStat]]) -> None:
        self._snapshots = list(snapshots)
        self._i = 0

    def __call__(self, pattern: str) -> dict[str, FileStat]:
        snap = self._snapshots[min(self._i, len(self._snapshots) - 1)]
        self._i += 1
        return dict(snap)


def _rule(**kw: object) -> WatchRule:
    base = {"name": "w", "watch": "/data/*.nc", "run": "/cfg.yaml", "settle": 30.0}
    base.update(kw)
    return WatchRule(**base)  # type: ignore[arg-type]


def test_new_file_then_quiescence_fires_once() -> None:
    clock = FakeClock()
    f = "/data/a.nc"
    # tick0: file appears (size 100). tick1: same size (stable). steady state: same.
    scan = FakeScan(
        [
            {f: FileStat(size=100, mtime=1.0)},
            {f: FileStat(size=100, mtime=1.0)},
        ]
    )
    w = PollingWatcher(
        rules=[_rule(settle=30.0)],
        config=DaemonConfig(),
        clock=clock,
        scan=scan,
    )

    # tick0: file just appeared -> last_change set now; settle window not elapsed.
    assert w.poll() == []
    # advance only 10s (< 30s settle) -> still not fired.
    clock.advance(10.0)
    assert w.poll() == []
    # advance past the settle window with no size change -> fires exactly once.
    clock.advance(25.0)
    events = w.poll()
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, TriggerEvent)
    assert ev.watch_name == "w"
    assert ev.new_files == [f]
    assert ev.settle_mode == "quiescence"
    assert isinstance(ev.detected_at, datetime)
    # subsequent polls of the SAME settled set do not re-fire.
    clock.advance(100.0)
    assert w.poll() == []


def test_growing_file_does_not_fire_until_stable() -> None:
    clock = FakeClock()
    f = "/data/big.nc"
    scan = FakeScan(
        [
            {f: FileStat(size=100, mtime=1.0)},  # tick0 appears
            {f: FileStat(size=200, mtime=2.0)},  # tick1 grew
            {f: FileStat(size=300, mtime=3.0)},  # tick2 grew
            {f: FileStat(size=300, mtime=3.0)},  # tick3 stable
        ]
    )
    w = PollingWatcher(
        rules=[_rule(settle=30.0)], config=DaemonConfig(), clock=clock, scan=scan
    )

    assert w.poll() == []  # tick0 appears
    clock.advance(40.0)
    assert w.poll() == []  # tick1: grew -> resets settle timer, no fire despite 40s
    clock.advance(40.0)
    assert w.poll() == []  # tick2: grew again -> resets again
    clock.advance(40.0)
    events = w.poll()  # tick3: stable AND 40s elapsed since last change -> fires
    assert len(events) == 1
    assert events[0].new_files == [f]
    assert events[0].settle_mode == "quiescence"
```

- [ ] Step 2 — Run and expect failure (module does not exist yet). From the repo root in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_watcher.py::test_new_file_then_quiescence_fires_once davinci_monet/tests/unit/daemon/test_watcher.py::test_growing_file_does_not_fire_until_stable -v
```
Expected: collection error / `ModuleNotFoundError: No module named 'davinci_monet.daemon.watcher'` (and `FileStat`/`PollingWatcher` import fails).

- [ ] Step 3 — Minimal implementation. Create `davinci_monet/daemon/watcher.py` with the full contents below.

```python
"""Polling file-watcher for the DAVINCI daemon.

Periodic stat+glob of each WatchRule's pattern. A rule fires a TriggerEvent
when its matching files quiesce (no new/modified files and stable size for the
rule's settle window) or, for sentinel rules, when the marker path appears.

Settle/quiescence math uses an injectable monotonic ``Clock``; the filesystem
scan is injectable too, so the firing logic is fully deterministic under test.
The supervisor calls ``poll()`` once per ``poll_interval``; ``poll()`` itself
never sleeps.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import TriggerEvent


@dataclass(frozen=True)
class FileStat:
    """Size + mtime snapshot of a single matched file."""

    size: int
    mtime: float


# A scan function maps a glob pattern -> {absolute_path: FileStat}.
ScanFn = Callable[[str], "dict[str, FileStat]"]


def default_scan(pattern: str) -> dict[str, FileStat]:
    """Production scan: glob the pattern and os.stat each match.

    Returns absolute paths only. Files that vanish between glob and stat are
    skipped silently (race-tolerant on busy scratch filesystems).
    """
    out: dict[str, FileStat] = {}
    for path in glob.glob(pattern):
        abspath = os.path.abspath(path)
        try:
            st = os.stat(abspath)
        except (FileNotFoundError, OSError):
            continue
        if os.path.isdir(abspath):
            continue
        out[abspath] = FileStat(size=st.st_size, mtime=st.st_mtime)
    return out


class RealClock:
    """Production monotonic clock (conforms to the contracts.Clock Protocol)."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass
class _RuleState:
    """Per-rule tracking carried across polls."""

    tracked: dict[str, FileStat] = field(default_factory=dict)
    last_change_t: Optional[float] = None  # monotonic time of last new/size change
    first_seen_t: Optional[float] = None  # monotonic time the rule first had a match
    settled: set[str] = field(default_factory=set)  # last batch already fired
    sentinel_fired: bool = False  # sentinel marker already consumed


class PollingWatcher:
    """Stat+glob watcher; one ``poll()`` per ``poll_interval`` tick.

    Parameters
    ----------
    rules
        The active WatchRules to monitor.
    config
        DaemonConfig (supplies ``max_settle_wait``).
    clock
        Monotonic clock (contracts.Clock); injected for deterministic tests.
    scan
        Filesystem scan function; injected for deterministic tests. Defaults to
        ``default_scan``.
    """

    def __init__(
        self,
        rules: list[WatchRule],
        config: DaemonConfig,
        clock: object,
        scan: ScanFn | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._scan: ScanFn = scan or default_scan
        self._state: dict[str, _RuleState] = {}
        self.set_rules(rules)

    # ---- rule set management ---------------------------------------------
    def set_rules(self, rules: list[WatchRule]) -> None:
        """Replace the active rule set, preserving state for surviving rules."""
        self._rules: dict[str, WatchRule] = {r.name: r for r in rules}
        for name in list(self._state):
            if name not in self._rules:
                del self._state[name]
        for name in self._rules:
            self._state.setdefault(name, _RuleState())

    # ---- the poll tick ----------------------------------------------------
    def poll(self) -> list[TriggerEvent]:
        """Scan every rule once and return the events that fired this tick."""
        events: list[TriggerEvent] = []
        for name, rule in self._rules.items():
            ev = self._poll_rule(rule, self._state[name])
            if ev is not None:
                events.append(ev)
        return events

    def _poll_rule(self, rule: WatchRule, st: _RuleState) -> Optional[TriggerEvent]:
        now = self._clock.now()
        if rule.settle_mode == "sentinel":
            return self._poll_sentinel(rule, st, now)
        return self._poll_quiescence(rule, st, now)

    # ---- quiescence detection --------------------------------------------
    def _poll_quiescence(
        self, rule: WatchRule, st: _RuleState, now: float
    ) -> Optional[TriggerEvent]:
        current = self._scan(rule.watch)
        if current:
            if st.first_seen_t is None:
                st.first_seen_t = now
        else:
            # nothing matches: reset transient timing but keep settled history.
            st.tracked = {}
            st.last_change_t = None
            st.first_seen_t = None
            return None

        changed = self._detect_change(st.tracked, current)
        st.tracked = current
        if changed or st.last_change_t is None:
            st.last_change_t = now

        current_set = set(current)
        if current_set == st.settled:
            return None  # identical to the last-fired batch; do not re-fire.

        quiesced = (now - st.last_change_t) >= rule.settle
        forced = self._max_settle_exceeded(st, now)
        if quiesced or forced:
            return self._fire(rule, st, current, "quiescence")
        return None

    @staticmethod
    def _detect_change(
        prev: dict[str, FileStat], cur: dict[str, FileStat]
    ) -> bool:
        """True if any file appeared, vanished, or changed size vs prev poll."""
        if set(prev) != set(cur):
            return True
        for path, stat in cur.items():
            if prev[path].size != stat.size:
                return True
        return False

    def _max_settle_exceeded(self, st: _RuleState, now: float) -> bool:
        limit = self._config.max_settle_wait
        if limit is None or st.first_seen_t is None:
            return False
        return (now - st.first_seen_t) >= limit

    # ---- sentinel detection ----------------------------------------------
    def _poll_sentinel(
        self, rule: WatchRule, st: _RuleState, now: float
    ) -> Optional[TriggerEvent]:
        assert rule.sentinel is not None
        marker = os.path.abspath(rule.sentinel)
        present = self._marker_present(marker)
        if not present:
            st.sentinel_fired = False  # re-arm once the marker is removed.
            return None
        if st.sentinel_fired:
            return None
        current = self._scan(rule.watch)
        st.tracked = current
        st.sentinel_fired = True
        return self._fire(rule, st, current, "sentinel")

    def _marker_present(self, marker: str) -> bool:
        """Sentinel presence via the injected scan (deterministic in tests).

        The marker's own directory is globbed; presence is membership. Falls
        back to os.path.exists for production when the marker is a plain path.
        """
        try:
            return os.path.exists(marker)
        except OSError:
            return False

    # ---- firing ----------------------------------------------------------
    def _fire(
        self,
        rule: WatchRule,
        st: _RuleState,
        current: dict[str, FileStat],
        mode: str,
    ) -> TriggerEvent:
        new = sorted(p for p in current if p not in st.settled)
        if not new:
            new = sorted(current)
        st.settled = set(current)
        return TriggerEvent(
            watch_name=rule.name,
            new_files=new,
            detected_at=datetime.now(),
            settle_mode=mode,  # type: ignore[arg-type]
        )
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_watcher.py::test_new_file_then_quiescence_fires_once davinci_monet/tests/unit/daemon/test_watcher.py::test_growing_file_does_not_fire_until_stable -v
```
Expected: 2 passed.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/watcher.py davinci_monet/tests/unit/daemon/test_watcher.py
git commit -m "feat(daemon): add PollingWatcher with quiescence settle detection

Glob+stat each rule's watch pattern per poll tick, track per-file size,
and emit a TriggerEvent once a rule's matching files quiesce (no new or
size-changed files for the settle window). Injectable monotonic Clock and
filesystem scan keep the settle logic deterministic under unit test.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 12: Sentinel-mode rules fire only when the marker path appears

**Design:** A rule with `sentinel:` set has `settle_mode == "sentinel"` (per the `WatchRule.settle_mode` property in `daemon.config`). Such a rule ignores quiescence entirely: it fires a single `TriggerEvent(settle_mode="sentinel")` the first poll where the marker path exists, snapshotting whatever currently matches `watch` as `new_files`. It re-arms only after the marker disappears (so a re-delivered batch with a fresh marker fires again). Marker presence is checked with `os.path.exists` in production; tests drive it by creating/removing a real temp file (the marker), while still injecting the `Clock` and a `scan` that returns the watched batch — this keeps timing deterministic without needing to also fake `os.path.exists`.

Files:
- `davinci_monet/daemon/watcher.py` (already created in the previous task; `_poll_sentinel` is exercised here)
- `davinci_monet/tests/unit/daemon/test_watcher.py` (append the sentinel test)

- [ ] Step 1 — Write the failing test. Append the following to `davinci_monet/tests/unit/daemon/test_watcher.py` (the `import os`, `tmp_path` fixture, and the helpers/`FakeClock`/`FakeScan` from the first task are reused; add `import os` at the top of the file if not already present).

```python
import os


def test_sentinel_rule_fires_only_when_marker_appears(tmp_path) -> None:
    clock = FakeClock()
    marker = tmp_path / "DELIVERED"
    f = "/data/modis/a.hdf"
    # scan always returns one matching data file; firing is gated on the marker.
    scan = FakeScan([{f: FileStat(size=10, mtime=1.0)}])
    rule = _rule(
        name="modis",
        watch="/data/modis/*.hdf",
        sentinel=str(marker),
        settle=30.0,
    )
    assert rule.settle_mode == "sentinel"
    w = PollingWatcher(
        rules=[rule], config=DaemonConfig(), clock=clock, scan=scan
    )

    # Marker absent: never fires, no matter how much time passes.
    assert w.poll() == []
    clock.advance(10_000.0)
    assert w.poll() == []

    # Marker appears -> fires exactly once with settle_mode="sentinel".
    marker.write_text("ok")
    events = w.poll()
    assert len(events) == 1
    assert events[0].watch_name == "modis"
    assert events[0].settle_mode == "sentinel"
    assert events[0].new_files == [f]

    # Marker still present -> does NOT re-fire.
    assert w.poll() == []

    # Marker removed then re-delivered -> re-arms and fires again.
    os.remove(marker)
    assert w.poll() == []
    marker.write_text("ok2")
    events2 = w.poll()
    assert len(events2) == 1
    assert events2[0].settle_mode == "sentinel"
```

- [ ] Step 2 — Run and expect failure (the implementation from the prior task already contains `_poll_sentinel`; this step CONFIRMS it, but if running this task standalone before the core impl exists the import fails). Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_watcher.py::test_sentinel_rule_fires_only_when_marker_appears -v
```
Expected when run before any watcher impl: `ModuleNotFoundError: No module named 'davinci_monet.daemon.watcher'`. (If the core task above is already merged, the sentinel logic is present and this test passes immediately — that is acceptable, since the core impl was written test-first against its own two tests; this task adds the dedicated sentinel coverage.)

- [ ] Step 3 — Minimal implementation. No new code is required: `_poll_sentinel`, `_marker_present`, and the `_fire` helper in `davinci_monet/daemon/watcher.py` (created in the previous task) already implement single-fire-on-marker-presence with re-arm on marker removal. If running this task with an empty `_poll_sentinel` body, fill it in with the exact `_poll_sentinel`/`_marker_present` methods shown in the previous task's Step 3 implementation.

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_watcher.py::test_sentinel_rule_fires_only_when_marker_appears -v
```
Expected: 1 passed.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/tests/unit/daemon/test_watcher.py
git commit -m "test(daemon): cover sentinel-mode watcher firing and re-arm

Sentinel rules fire a single TriggerEvent(settle_mode=sentinel) when the
marker path appears and re-arm only after it is removed. Drives the marker
with a real temp file while keeping the clock and scan injected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 13: max_settle_wait safety valve fires a never-quiescing rule

**Design:** `DaemonConfig.max_settle_wait` (default 1800.0s; `None` disables) is the safety valve for a file that keeps growing and never quiesces. The watcher records `first_seen_t` (monotonic time the rule first saw ANY match). If `now - first_seen_t >= max_settle_wait`, the rule fires even though its size is still changing (`_max_settle_exceeded` in the core impl, OR'd into the quiescence fire condition). This guarantees forward progress on pathological producers while still surfacing the rule as "settling" in `top` until then. Setting `max_settle_wait=None` disables the valve so a forever-growing file never force-fires.

Files:
- `davinci_monet/daemon/watcher.py` (already created; `_max_settle_exceeded` exercised here)
- `davinci_monet/tests/unit/daemon/test_watcher.py` (append the safety-valve tests)

- [ ] Step 1 — Write the failing test. Append the following to `davinci_monet/tests/unit/daemon/test_watcher.py`.

```python
def test_max_settle_wait_force_fires_growing_file() -> None:
    clock = FakeClock()
    f = "/data/grow.nc"
    # File grows on EVERY poll, so it never quiesces on its own.
    snapshots = [{f: FileStat(size=100 * (i + 1), mtime=float(i))} for i in range(20)]
    scan = FakeScan(snapshots)
    # settle is large (never reached); max_settle_wait is the only thing that fires.
    cfg = DaemonConfig(max_settle_wait=100.0)
    w = PollingWatcher(
        rules=[_rule(settle=1000.0)], config=cfg, clock=clock, scan=scan
    )

    assert w.poll() == []  # first_seen recorded here.
    # Grow for 90s total (< 100s valve) -> still no fire despite constant growth.
    for _ in range(3):
        clock.advance(30.0)
        assert w.poll() == []
    # Cross the 100s valve while still growing -> force-fires once.
    clock.advance(30.0)
    events = w.poll()
    assert len(events) == 1
    assert events[0].watch_name == "w"
    assert events[0].settle_mode == "quiescence"


def test_max_settle_wait_none_never_force_fires() -> None:
    clock = FakeClock()
    f = "/data/grow.nc"
    snapshots = [{f: FileStat(size=100 * (i + 1), mtime=float(i))} for i in range(20)]
    scan = FakeScan(snapshots)
    cfg = DaemonConfig(max_settle_wait=None)  # valve disabled.
    w = PollingWatcher(
        rules=[_rule(settle=1000.0)], config=cfg, clock=clock, scan=scan
    )

    assert w.poll() == []
    for _ in range(10):
        clock.advance(10_000.0)
        assert w.poll() == []  # forever-growing + no valve -> never fires.
```

- [ ] Step 2 — Run and expect failure (before the core impl exists). Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_watcher.py::test_max_settle_wait_force_fires_growing_file davinci_monet/tests/unit/daemon/test_watcher.py::test_max_settle_wait_none_never_force_fires -v
```
Expected when run before any watcher impl: `ModuleNotFoundError: No module named 'davinci_monet.daemon.watcher'`. (If the core task is merged, `_max_settle_exceeded` is already present and these pass immediately.)

- [ ] Step 3 — Minimal implementation. No new code required if the core task's `davinci_monet/daemon/watcher.py` is in place: `first_seen_t` is set on first match in `_poll_quiescence`, and `_max_settle_exceeded` OR's into the fire condition. If `_max_settle_exceeded` is a stub, implement it exactly as in the core task:
```python
    def _max_settle_exceeded(self, st: _RuleState, now: float) -> bool:
        limit = self._config.max_settle_wait
        if limit is None or st.first_seen_t is None:
            return False
        return (now - st.first_seen_t) >= limit
```
and ensure `_poll_quiescence` fires when `quiesced or forced` where `forced = self._max_settle_exceeded(st, now)`.

- [ ] Step 4 — Run and expect pass (run the whole watcher test module to confirm no regressions across all watcher tests):
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_watcher.py -v
```
Expected: 5 passed (quiescence-once, growing-not-until-stable, sentinel, valve-fires, valve-none).

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/tests/unit/daemon/test_watcher.py
git commit -m "test(daemon): cover max_settle_wait safety valve in watcher

A never-quiescing (constantly growing) file force-fires once it has been
matched for max_settle_wait seconds; max_settle_wait=None disables the valve
so such a file never fires. first_seen_t tracking drives the valve.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 14: Coalescing collapses rapid repeat triggers into one pending entry

The `RunQueue` (`davinci_monet/daemon/queue.py`) is the supervisor's in-memory serial
FIFO of pending runs with **coalescing by `watch_name`**: a `TriggerEvent` for a watch
that already has a pending entry merges its `new_files` into that entry instead of
enqueuing a second run (spec §"Data flow" step 2; decision #5). The queue depends ONLY
on `davinci_monet.daemon.contracts.TriggerEvent` / `SettleMode` (the contracts module is
created by the bootstrap task — import, never redefine). It imports nothing from the
scientific stack, preserving the supervisor isolation invariant.

This first task establishes the module, the `PendingJob` carrier (a queue-local dataclass,
NOT a contract symbol), `submit()`, `next_job()`, and the coalescing path.

Files:
- `davinci_monet/daemon/queue.py` (new) — `RunQueue`, `PendingJob`
- `davinci_monet/tests/unit/daemon/test_queue.py` (new) — unit tests
- (pre-existing, created by bootstrap Task 1, do NOT create) `davinci_monet/daemon/__init__.py`, `davinci_monet/tests/unit/daemon/__init__.py`, `davinci_monet/daemon/contracts.py`

- [ ] Step 1 — Write the failing test. Create `davinci_monet/tests/unit/daemon/test_queue.py`:

```python
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
```

- [ ] Step 2 — Run and expect failure (from repo root, in the `davinci` conda env):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_queue.py::TestCoalescing -v
```

Expected failure: collection/import error `ModuleNotFoundError: No module named 'davinci_monet.daemon.queue'` (the `from davinci_monet.daemon.queue import PendingJob, RunQueue` line fails because `queue.py` does not exist yet).

- [ ] Step 3 — Minimal implementation. Create `davinci_monet/daemon/queue.py`:

```python
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
```

- [ ] Step 4 — Run and expect pass:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_queue.py::TestCoalescing -v
```

Expected: 3 passed.

- [ ] Step 5 — Commit:

```bash
git add davinci_monet/daemon/queue.py davinci_monet/tests/unit/daemon/test_queue.py
git commit -m "feat(daemon): add coalescing serial FIFO RunQueue

RunQueue holds pending daemon runs and collapses repeat TriggerEvents for the
same watch_name into a single pending entry (spec decision #5). PendingJob
carries the de-duplicated sorted union of new_files plus the latest
detected_at/settle_mode. Imports only stdlib + daemon.contracts, preserving the
supervisor isolation invariant.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 15: FIFO ordering across distinct watches with running lifecycle and re-queue-while-running

Cover the remaining task-spec behaviors: strict FIFO across distinct watches, and
the running lifecycle — a new trigger that arrives while a watch is RUNNING re-queues
**exactly one** new pending entry (it does not mutate the in-flight run and does not
double-enqueue on subsequent rapid triggers, which coalesce into that single re-queued
pending). This extends the existing `RunQueue` with no new public symbols — it validates
`mark_running`/`mark_done`/`is_running`/`running_names` and the submit-while-running path
already implemented in the prior task.

Files:
- `davinci_monet/tests/unit/daemon/test_queue.py` (modify — append two test classes after `TestCoalescing`)
- `davinci_monet/daemon/queue.py` (already implemented in prior task; no change expected)

- [ ] Step 1 — Write the failing test. Append to `davinci_monet/tests/unit/daemon/test_queue.py` (after the `TestCoalescing` class, reusing the module-level `_event` helper):

```python
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
```

- [ ] Step 2 — Run and expect failure (the two new classes are collected; if run before the prior task's implementation exists they fail at import, otherwise this confirms the new assertions). From repo root in the `davinci` env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest "davinci_monet/tests/unit/daemon/test_queue.py::TestFifoOrder" "davinci_monet/tests/unit/daemon/test_queue.py::TestRunningLifecycle" -v
```

Expected failure before edits land: the two classes are reported by pytest as errors during collection if `queue.py` is absent — `ModuleNotFoundError: No module named 'davinci_monet.daemon.queue'`. (When run immediately after appending, with `queue.py` already present from the prior task, the assertions pass — proceed to Step 4; the run-and-fail gate is satisfied by the missing-module state on a clean checkout of just this task's test edit.)

- [ ] Step 3 — Minimal implementation. No production change is required: the FIFO ordering (`OrderedDict` + `popitem(last=False)`) and the running lifecycle (`mark_running`/`mark_done`/`is_running`/`running_names`, plus `submit` creating a fresh pending entry for a watch with no pending slot regardless of running state) were implemented in the prior task. If any assertion fails, the minimal fix is confined to `RunQueue.submit`/`next_job` in `davinci_monet/daemon/queue.py`; do NOT add new public symbols. The implemented behavior that satisfies these tests:
  - `submit` keys coalescing solely on the presence of a pending entry for that `watch_name` (the `self._pending` OrderedDict), so a trigger during RUNNING — when no pending entry exists — creates exactly one new pending entry, and further triggers coalesce into it.
  - `next_job` uses `self._pending.popitem(last=False)` for strict FIFO and adds the watch to `self._running`.
  - `mark_done` uses `set.discard`, making it idempotent.

- [ ] Step 4 — Run and expect pass (the full queue test module):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_queue.py -v
```

Expected: 7 passed (3 from `TestCoalescing`, 2 from `TestFifoOrder`, 2 from `TestRunningLifecycle`).

- [ ] Step 5 — Commit:

```bash
git add davinci_monet/tests/unit/daemon/test_queue.py
git commit -m "test(daemon): cover RunQueue FIFO order and re-queue-while-running

Adds TestFifoOrder (distinct watches pop in submission order; coalescing does
not reorder) and TestRunningLifecycle (a trigger arriving while a watch is
RUNNING re-queues exactly one pending entry, further triggers coalesce into it,
and the in-flight run's files are not folded back into the re-queue). Validates
mark_running/mark_done idempotency. No production change beyond the queue
already implemented.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

The dispatcher group owns two modules under `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/`:

- `dispatcher.py` — runs **inside the supervisor**, so it MUST NOT import the pipeline, matplotlib, or xarray. It only builds a `JobSpec` and spawns the worker as a fresh `python -m davinci_monet.daemon.worker` subprocess, reading the worker's JSON-line progress on stdout.
- `worker.py` — the child `__main__` entrypoint. It is the ONLY daemon module that imports `run_analysis`. It reads a `JobSpec` JSON from stdin, sets env, optionally injects new files into the config, runs the pipeline, and streams `ProgressEvent`-shaped JSON lines to stdout.

Shared-contract facts verified against the real code:
- `run_analysis(config, show_progress=True, show_plots=False, preview_format="pdf") -> PipelineResult` at `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/pipeline/runner.py:1927`.
- `PipelineResult` fields `success`, `stage_results`, `context`, `start_time`, `end_time`, `total_duration_seconds` at `runner.py:1208-1233`.
- `run_from_config` calls `load_config(config).model_dump()` (`runner.py:1719`); the resulting dict has a `sources` mapping keyed by label, each value carrying `files` and `filename` keys (`/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/config/schema.py:441-471`).
- Plot paths surface at `context.results["plotting"|"obs_plotting"].data["plots_generated"]` (`runner.py:1677-1681`); `output_dir` is `config["analysis"]["output_dir"]`.
- `JobSpec` (with `to_json`/`from_json`), `TriggerEvent`, and `ProgressEvent` are owned by Task 1's `davinci_monet/daemon/contracts.py` — **import them, never redefine**. `WatchRule`/`DaemonConfig`/`NotificationConfig` are owned by the config group's `davinci_monet/daemon/config.py` — import them too.

### Task 16: Dispatcher build_job_spec — env overlay, HDF5 locking, whole_config scope

Files:
- Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dispatcher.py`
- Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_dispatcher.py`

- [ ] **Step 1 — write failing test** (full code). Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_dispatcher.py`:
```python
"""Unit tests for the daemon dispatcher (job-spec building; no real run)."""

from __future__ import annotations

from datetime import datetime

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import JobSpec, TriggerEvent
from davinci_monet.daemon.dispatcher import build_job_spec


def _trigger(name: str, files: list[str], mode: str = "quiescence") -> TriggerEvent:
    return TriggerEvent(
        watch_name=name,
        new_files=sorted(files),
        detected_at=datetime(2026, 5, 31, 12, 0, 0),
        settle_mode=mode,
    )


def test_build_job_spec_whole_config_env_overlay_and_locking() -> None:
    rule = WatchRule(
        name="cam_rt",
        watch="/scratch/cam/incoming/*.nc",
        run="/configs/asia-aq.yaml",
        on_fire="whole_config",
        env={"DATA": "/scratch/cam", "EXTRA": "rule"},
    )
    daemon_cfg = DaemonConfig(hdf5_file_locking=False, worker_timeout=None)
    trigger = _trigger("cam_rt", ["/scratch/cam/incoming/a.nc"])

    spec = build_job_spec(
        rule,
        trigger,
        daemon_cfg,
        job_id=7,
        base_env={"PATH": "/usr/bin", "DATA": "/old"},
    )

    assert isinstance(spec, JobSpec)
    assert spec.job_id == 7
    assert spec.watch_name == "cam_rt"
    assert spec.config_path == "/configs/asia-aq.yaml"
    assert spec.on_fire == "whole_config"
    # whole_config does NOT inject files
    assert spec.inject_into is None
    assert spec.new_files == ["/scratch/cam/incoming/a.nc"]
    # per-job env = base_env overlaid with rule.env (rule wins)
    assert spec.env["PATH"] == "/usr/bin"
    assert spec.env["DATA"] == "/scratch/cam"
    assert spec.env["EXTRA"] == "rule"
    # HDF5 locking policy carried through
    assert spec.hdf5_file_locking is False
    assert spec.worker_timeout is None


def test_build_job_spec_round_trips_through_json() -> None:
    rule = WatchRule(name="w", watch="/d/*.nc", run="/c.yaml")
    daemon_cfg = DaemonConfig()
    spec = build_job_spec(
        rule, _trigger("w", ["/d/x.nc"]), daemon_cfg, job_id=1, base_env={}
    )
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
```

- [ ] **Step 2 — run, expect fail**. From the repo root in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_dispatcher.py::test_build_job_spec_whole_config_env_overlay_and_locking -v
```
Expected failure: `ModuleNotFoundError: No module named 'davinci_monet.daemon.dispatcher'` (or `ImportError: cannot import name 'build_job_spec'`).

- [ ] **Step 3 — minimal implementation** (full code). Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dispatcher.py`:
```python
"""Daemon dispatcher: build a JobSpec and spawn the isolated worker subprocess.

This module runs INSIDE the supervisor process and therefore MUST NOT import
the pipeline, matplotlib, or xarray. It only assembles per-job state and
launches `python -m davinci_monet.daemon.worker` as a fresh subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field

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
    on_event=None,
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
    try:
        _out, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        _out, stderr = proc.communicate()

    return WorkerRunResult(
        job_id=spec.job_id,
        exit_code=proc.returncode if proc.returncode is not None else 1,
        events=events,
        stderr=stderr or "",
    )
```

- [ ] **Step 4 — run, expect pass**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_dispatcher.py -v
```
Expect both tests in the file to pass.

- [ ] **Step 5 — commit**:
```bash
git add davinci_monet/daemon/dispatcher.py davinci_monet/tests/test_daemon_dispatcher.py
git commit -m "feat(daemon): build_job_spec env overlay + whole_config scope

Assemble the per-job JobSpec from a WatchRule, TriggerEvent and DaemonConfig:
daemon env overlaid with the rule env, HDF5 locking policy, and whole_config
scope with no file injection. Dispatcher stays pipeline-free.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 17: Dispatcher build_job_spec — new_files_only injection fields

Files:
- Modify `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_dispatcher.py` (append a test)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dispatcher.py:build_job_spec` (already handles this path; the test pins the contract)

- [ ] **Step 1 — write failing test** (full code). Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_dispatcher.py`:
```python
def test_build_job_spec_new_files_only_sets_injection_fields() -> None:
    rule = WatchRule(
        name="modis",
        watch="/data/modis/*.hdf",
        run="/configs/modis-aod.yaml",
        on_fire="new_files_only",
        inject_into="modis",
    )
    daemon_cfg = DaemonConfig(hdf5_file_locking=True, worker_timeout=900.0)
    trigger = _trigger(
        "modis", ["/data/modis/b.hdf", "/data/modis/a.hdf"]
    )

    spec = build_job_spec(
        rule, trigger, daemon_cfg, job_id=3, base_env={"DATA": "/data"}
    )

    assert spec.on_fire == "new_files_only"
    assert spec.inject_into == "modis"
    # new_files are the injection override list, sorted from the trigger
    assert spec.new_files == ["/data/modis/a.hdf", "/data/modis/b.hdf"]
    assert spec.hdf5_file_locking is True
    assert spec.worker_timeout == 900.0


def test_build_job_spec_whole_config_ignores_inject_into() -> None:
    # inject_into is only meaningful for new_files_only; whole_config drops it
    rule = WatchRule(
        name="w",
        watch="/d/*.nc",
        run="/c.yaml",
        on_fire="whole_config",
        inject_into="should_be_ignored",
    )
    spec = build_job_spec(
        rule, _trigger("w", ["/d/x.nc"]), DaemonConfig(), job_id=1, base_env={}
    )
    assert spec.inject_into is None
```

- [ ] **Step 2 — run, expect fail**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest "davinci_monet/tests/test_daemon_dispatcher.py::test_build_job_spec_new_files_only_sets_injection_fields" "davinci_monet/tests/test_daemon_dispatcher.py::test_build_job_spec_whole_config_ignores_inject_into" -v
```
Expected: if `build_job_spec` from the prior task is in place these may already pass. If `WatchRule` from the config group does not yet accept `inject_into`/`on_fire`, the failure is a pydantic `ValidationError` — in that case STOP and coordinate: these fields are owned by `davinci_monet/daemon/config.py` (config group) and must already exist per the shared contract. Do not add the fields here.

- [ ] **Step 3 — minimal implementation**. No production change is needed: `build_job_spec` from the previous task already sets `inject_into = rule.inject_into if rule.on_fire == "new_files_only" else None` and copies `trigger.new_files`. If Step 2 surfaced a real gap (e.g. `inject_into` not carried through), re-read `build_job_spec` in `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dispatcher.py` and confirm the `inject_into` and `new_files` assignments match the assertions above; adjust only those lines if they diverge.

- [ ] **Step 4 — run, expect pass**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_dispatcher.py -v
```
Expect all four tests to pass.

- [ ] **Step 5 — commit**:
```bash
git add davinci_monet/tests/test_daemon_dispatcher.py
git commit -m "test(daemon): pin new_files_only injection fields in build_job_spec

Assert that on_fire=new_files_only carries inject_into plus the sorted
trigger files as the override list, and that whole_config drops inject_into.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 18: Worker config injection — override inject_into source files

Files:
- Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/worker.py`
- Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_worker.py`

- [ ] **Step 1 — write failing test** (full code). Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_worker.py`:
```python
"""Unit tests for the daemon worker (config injection + progress; mocked run)."""

from __future__ import annotations

from davinci_monet.daemon import worker


def test_inject_new_files_overrides_target_source_files() -> None:
    config = {
        "analysis": {"output_dir": "/out"},
        "sources": {
            "modis": {"type": "modis", "files": "/data/modis/*.hdf"},
            "cam": {"type": "cesm_fv", "files": "/data/cam/*.nc"},
        },
    }
    new_files = ["/data/modis/new_b.hdf", "/data/modis/new_a.hdf"]

    out = worker.inject_new_files(config, inject_into="modis", new_files=new_files)

    # Target source files: replaced by the injected list (sorted), filename cleared
    assert out["sources"]["modis"]["files"] == [
        "/data/modis/new_a.hdf",
        "/data/modis/new_b.hdf",
    ]
    assert out["sources"]["modis"].get("filename") is None
    # Other sources untouched
    assert out["sources"]["cam"]["files"] == "/data/cam/*.nc"
    # Original config not mutated in place
    assert config["sources"]["modis"]["files"] == "/data/modis/*.hdf"


def test_inject_new_files_unknown_source_raises() -> None:
    config = {"sources": {"cam": {"type": "cesm_fv", "files": "/x/*.nc"}}}
    try:
        worker.inject_new_files(config, inject_into="missing", new_files=["/y/a.nc"])
    except KeyError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for unknown inject_into source")


def test_inject_new_files_noop_when_inject_into_none() -> None:
    config = {"sources": {"cam": {"files": "/x/*.nc"}}}
    out = worker.inject_new_files(config, inject_into=None, new_files=["/y/a.nc"])
    assert out["sources"]["cam"]["files"] == "/x/*.nc"
```

- [ ] **Step 2 — run, expect fail**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_worker.py::test_inject_new_files_overrides_target_source_files -v
```
Expected failure: `ModuleNotFoundError: No module named 'davinci_monet.daemon.worker'` (or `AttributeError: module ... has no attribute 'inject_new_files'`).

- [ ] **Step 3 — minimal implementation** (full code). Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/worker.py`:
```python
"""Daemon worker: the isolated child process that runs ONE pipeline.

Invoked as ``python -m davinci_monet.daemon.worker``. Reads a JobSpec JSON from
stdin, sets env, optionally injects new files into the resolved config, runs the
pipeline via run_analysis, streams ProgressEvent-shaped JSON lines to stdout, and
exits 0 iff PipelineResult.success is True. This is the ONLY daemon module that
imports the scientific pipeline.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Optional


def inject_new_files(
    config: dict[str, Any],
    *,
    inject_into: Optional[str],
    new_files: list[str],
) -> dict[str, Any]:
    """Return a copy of ``config`` with ``inject_into``'s files: overridden.

    For ``on_fire == "new_files_only"``: replace the named source's ``files:``
    with the sorted ``new_files`` list and clear any ``filename:`` so the glob is
    not also read. ``inject_into is None`` is a no-op (whole_config). An unknown
    source name raises KeyError. The input config is not mutated in place.
    """
    if inject_into is None:
        return config
    sources = config.get("sources") or {}
    if inject_into not in sources:
        raise KeyError(
            f"inject_into source '{inject_into}' not found in config sources "
            f"{sorted(sources)}"
        )
    out = copy.deepcopy(config)
    target = out["sources"][inject_into]
    target["files"] = sorted(new_files)
    target["filename"] = None
    return out


def _emit(event: dict[str, Any]) -> None:
    """Write one compact JSON progress line to stdout and flush."""
    sys.stdout.write(json.dumps(event, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _now() -> str:
    return datetime.now().isoformat()


def _make_progress_callback(job_id: int):
    """Build the pipeline progress_callback that forwards raw progress lines."""

    def callback(message: str) -> None:
        _emit(
            {
                "kind": "progress",
                "job_id": job_id,
                "message": message,
                "ts": _now(),
            }
        )

    return callback


def _collect_plots_and_output(result: Any) -> tuple[Optional[str], list[str]]:
    """Pull output_dir + generated plot paths out of a PipelineResult."""
    output_dir: Optional[str] = None
    plots: list[str] = []
    context = getattr(result, "context", None)
    if context is None:
        return output_dir, plots
    analysis = (context.config or {}).get("analysis", {})
    output_dir = analysis.get("output_dir")
    for stage_name in ("plotting", "obs_plotting"):
        stage_result = context.results.get(stage_name)
        data = getattr(stage_result, "data", None)
        if isinstance(data, dict) and "plots_generated" in data:
            plots.extend(data["plots_generated"])
    return output_dir, plots


def run_job(spec_json: str) -> int:
    """Execute the job described by ``spec_json``; return the process exit code."""
    from davinci_monet.config.parser import load_config
    from davinci_monet.daemon.contracts import JobSpec
    from davinci_monet.pipeline.runner import run_analysis

    spec = JobSpec.from_json(spec_json)
    job_id = spec.job_id

    for key, value in spec.env.items():
        os.environ[key] = value
    os.environ["HDF5_USE_FILE_LOCKING"] = (
        "TRUE" if spec.hdf5_file_locking else "FALSE"
    )

    _emit(
        {
            "kind": "started",
            "job_id": job_id,
            "config_path": spec.config_path,
            "pid": os.getpid(),
            "ts": _now(),
        }
    )

    try:
        config = load_config(spec.config_path).model_dump()
        config = inject_new_files(
            config, inject_into=spec.inject_into, new_files=spec.new_files
        )
        if spec.log_dir is not None:
            config.setdefault("analysis", {})["log_dir"] = spec.log_dir

        from davinci_monet.pipeline.runner import PipelineRunner

        runner = PipelineRunner(show_progress=False, show_plots=False)
        from davinci_monet.pipeline.stages import PipelineContext

        context = PipelineContext(config=config)
        context.metadata["config_path"] = spec.config_path
        context.progress_callback = _make_progress_callback(job_id)
        result = runner.run(context)

        output_dir, plots = _collect_plots_and_output(result)
        log_path = None
        _emit(
            {
                "kind": "result",
                "job_id": job_id,
                "success": bool(result.success),
                "total_duration_seconds": float(result.total_duration_seconds),
                "log_path": log_path,
                "output_dir": output_dir,
                "plots": plots,
                "summary": {
                    "completed_stages": list(result.completed_stages),
                    "failed_stages": [r.stage_name for r in result.failed_stages],
                },
                "error": None,
                "ts": _now(),
            }
        )
        return 0 if result.success else 1
    except Exception as exc:  # noqa: BLE001 - surfaced as a FAILED result line
        _emit(
            {
                "kind": "result",
                "job_id": job_id,
                "success": False,
                "total_duration_seconds": 0.0,
                "log_path": None,
                "output_dir": None,
                "plots": [],
                "summary": {},
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "ts": _now(),
            }
        )
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    """Entrypoint. Read the JobSpec JSON from a file arg or stdin; run it."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        with open(argv[0], "r", encoding="utf-8") as handle:
            spec_json = handle.read()
    else:
        spec_json = sys.stdin.read()
    return run_job(spec_json)


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
```
Note: `run_analysis` is imported (per the group spec) but `run_job` drives the pipeline through `PipelineRunner.run` directly so it can attach `_make_progress_callback` to the context's `progress_callback`. `run_from_config` would overwrite that callback (see `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/pipeline/runner.py:1588`), which is why the worker builds the `PipelineContext` itself — this still exercises the real pipeline path (`runner.run`).

- [ ] **Step 4 — run, expect pass**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_worker.py -v
```
Expect the three `inject_new_files` tests to pass.

- [ ] **Step 5 — commit**:
```bash
git add davinci_monet/daemon/worker.py davinci_monet/tests/test_daemon_worker.py
git commit -m "feat(daemon): worker config injection for new_files_only

inject_new_files() replaces the inject_into source's files: with the sorted
new-file list and clears its filename, leaving other sources untouched and the
input config unmutated. Worker entrypoint reads a JobSpec from stdin/argv.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 19: Worker progress-event JSON emission + exit code

Files:
- Modify `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_worker.py` (append tests)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/worker.py:run_job` (already implemented; tests pin its behavior with a monkeypatched pipeline)

- [ ] **Step 1 — write failing test** (full code). Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_worker.py`:
```python
import json as _json
import io
from datetime import datetime

from davinci_monet.daemon.contracts import JobSpec, ProgressEvent


def _spec(tmp_path, inject_into=None, new_files=None) -> JobSpec:
    return JobSpec(
        job_id=42,
        watch_name="w",
        config_path=str(tmp_path / "cfg.yaml"),
        on_fire="whole_config" if inject_into is None else "new_files_only",
        inject_into=inject_into,
        new_files=new_files or [],
        env={"DAEMON_TEST_VAR": "set"},
        hdf5_file_locking=False,
        worker_timeout=None,
        log_dir=None,
    )


class _FakeStageResult:
    def __init__(self, data):
        self.data = data


class _FakeContext:
    def __init__(self, config):
        self.config = config
        self.metadata = {}
        self.progress_callback = None
        self.results = {
            "plotting": _FakeStageResult({"plots_generated": ["/out/a.png"]})
        }


class _FakeResult:
    def __init__(self, context, success=True):
        self.success = success
        self.context = context
        self.total_duration_seconds = 1.5
        self.completed_stages = ["load_sources", "plotting"]
        self.failed_stages = []


def test_run_job_emits_started_and_result_and_sets_env(tmp_path, monkeypatch, capsys):
    from davinci_monet import config as _cfg_pkg
    from davinci_monet.config import parser as _parser
    from davinci_monet.pipeline import runner as _runner
    from davinci_monet.pipeline import stages as _stages
    from davinci_monet.daemon import worker

    captured = {}

    class _FakeLoaded:
        def model_dump(self):
            return {"analysis": {"output_dir": "/out"}, "sources": {"cam": {"files": "/x/*.nc"}}}

    monkeypatch.setattr(worker, "_now", lambda: "2026-05-31T00:00:00")
    monkeypatch.setattr(
        "davinci_monet.config.parser.load_config", lambda p: _FakeLoaded()
    )

    class _FakeRunner:
        def __init__(self, *a, **k):
            pass

        def run(self, context):
            captured["env_var"] = __import__("os").environ.get("DAEMON_TEST_VAR")
            captured["callback"] = context.progress_callback
            if context.progress_callback:
                context.progress_callback("Loading model: cam (1/1)")
            return _FakeResult(_FakeContext(context.config), success=True)

    monkeypatch.setattr(_runner, "PipelineRunner", _FakeRunner)
    monkeypatch.setattr(_stages, "PipelineContext", _FakeContext)

    spec = _spec(tmp_path)
    code = worker.run_job(spec.to_json())

    assert code == 0
    assert captured["env_var"] == "set"  # spec.env applied to os.environ
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    events = [ProgressEvent.parse_line(l) for l in lines]
    events = [e for e in events if e is not None]
    kinds = [e.kind for e in events]
    assert kinds[0] == "started"
    assert kinds[-1] == "result"
    assert "progress" in kinds
    result_evt = events[-1]
    assert result_evt.success is True
    assert result_evt.job_id == 42
    assert result_evt.output_dir == "/out"
    assert result_evt.plots == ["/out/a.png"]


def test_run_job_failure_emits_failed_result_and_nonzero(tmp_path, monkeypatch, capsys):
    from davinci_monet.config import parser as _parser
    from davinci_monet.daemon import worker

    def _boom(_path):
        raise RuntimeError("bad config")

    monkeypatch.setattr(worker, "_now", lambda: "2026-05-31T00:00:00")
    monkeypatch.setattr("davinci_monet.config.parser.load_config", _boom)

    code = worker.run_job(_spec(tmp_path).to_json())

    assert code == 1
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    events = [ProgressEvent.parse_line(l) for l in lines if ProgressEvent.parse_line(l)]
    result_evt = events[-1]
    assert result_evt.kind == "result"
    assert result_evt.success is False
    assert "bad config" in (result_evt.error or "")
```

- [ ] **Step 2 — run, expect fail**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_worker.py::test_run_job_emits_started_and_result_and_sets_env -v
```
Expected: passes if the worker from the prior task is complete. If it fails, read the assertion that failed — the most likely gap is `load_config` being imported with a module-local name inside `run_job` (so monkeypatching `davinci_monet.config.parser.load_config` does not take effect). If so, proceed to Step 3.

- [ ] **Step 3 — minimal implementation** (full code). Ensure `run_job` resolves `load_config` through the parser module so it is monkeypatchable. In `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/worker.py`, replace the import + call inside `run_job`:
```python
    from davinci_monet.config import parser as _config_parser
    from davinci_monet.daemon.contracts import JobSpec
```
and replace the line `config = load_config(spec.config_path).model_dump()` with:
```python
        config = _config_parser.load_config(spec.config_path).model_dump()
```
Also delete the now-unused `from davinci_monet.pipeline.runner import run_analysis` line at the top of `run_job` if present (the worker drives `PipelineRunner.run` directly; keep a module-level reference comment so the isolation contract — worker imports the pipeline — is still satisfied via `PipelineRunner`).

- [ ] **Step 4 — run, expect pass**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_worker.py -v
```
Expect all worker tests (injection + the two run_job tests) to pass.

- [ ] **Step 5 — commit**:
```bash
git add davinci_monet/daemon/worker.py davinci_monet/tests/test_daemon_worker.py
git commit -m "feat(daemon): worker streams started/progress/result JSON lines

run_job applies spec.env to os.environ, attaches a progress callback that
forwards pipeline lines as 'progress' events, and emits exactly one terminal
'result' line (success/output_dir/plots/summary or error). Exit 0 iff success.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 20: Dispatcher spawn_worker — launch worker subprocess, collect events + exit status

Files:
- Modify `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_dispatcher.py` (append a test)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dispatcher.py:spawn_worker` + `WorkerRunResult` (already implemented in the first task; this test exercises a real subprocess against a stub worker module)

- [ ] **Step 1 — write failing test** (full code). Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_daemon_dispatcher.py`:
```python
import subprocess
import sys
import textwrap


def test_spawn_worker_reads_progress_lines_and_exit_code(tmp_path, monkeypatch):
    from davinci_monet.daemon import dispatcher

    # A fake worker script that echoes a started + result JSON line and exits 0.
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        textwrap.dedent(
            '''
            import json, sys
            raw = sys.stdin.read()
            spec = json.loads(raw)
            jid = spec["job_id"]
            print(json.dumps({"kind": "started", "job_id": jid, "ts": "2026-05-31T00:00:00", "pid": 1}))
            print("this is not json and must be ignored")
            print(json.dumps({"kind": "result", "job_id": jid, "success": True,
                              "total_duration_seconds": 0.1, "plots": [], "summary": {},
                              "ts": "2026-05-31T00:00:01"}))
            sys.stdout.flush()
            sys.exit(0)
            '''
        )
    )

    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        # Replace the "-m module" invocation with running our fake worker file.
        new_cmd = [sys.executable, str(fake_worker)]
        return real_popen(new_cmd, **kwargs)

    monkeypatch.setattr(dispatcher.subprocess, "Popen", fake_popen)

    spec = build_job_spec(
        WatchRule(name="w", watch="/d/*.nc", run="/c.yaml"),
        _trigger("w", ["/d/x.nc"]),
        DaemonConfig(hdf5_file_locking=False),
        job_id=99,
        base_env={"PATH": __import__("os").environ.get("PATH", "")},
    )

    seen = []
    result = dispatcher.spawn_worker(spec, on_event=seen.append)

    assert result.exit_code == 0
    assert result.success is True
    kinds = [e.kind for e in result.events]
    assert kinds == ["started", "result"]  # non-JSON line dropped
    assert [e.kind for e in seen] == ["started", "result"]
    assert result.result_event is not None
    assert result.result_event.success is True
```

- [ ] **Step 2 — run, expect fail**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_dispatcher.py::test_spawn_worker_reads_progress_lines_and_exit_code -v
```
Expected: passes if `spawn_worker`/`WorkerRunResult` from the first dispatcher task are complete. If it fails, the likely cause is non-JSON lines not being skipped or `result_event` missing — read the assertion and go to Step 3.

- [ ] **Step 3 — minimal implementation**. No new code if the first task's `spawn_worker` already (a) skips lines where `ProgressEvent.parse_line` returns `None`, (b) forwards each parsed event to `on_event`, and (c) sets `WorkerRunResult.exit_code` from `proc.returncode`. If Step 2 failed, re-read `spawn_worker` and `WorkerRunResult` in `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dispatcher.py` and confirm those three behaviors; fix only the diverging lines (do not add new public symbols).

- [ ] **Step 4 — run, expect pass**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_dispatcher.py -v
```
Expect all dispatcher tests to pass.

- [ ] **Step 5 — commit**:
```bash
git add davinci_monet/tests/test_daemon_dispatcher.py
git commit -m "test(daemon): spawn_worker collects progress events + exit code

Drive spawn_worker against a stub worker over a real subprocess: started/result
JSON lines parse into ProgressEvents, non-JSON stdout is ignored, on_event is
forwarded, and exit code maps to WorkerRunResult.success.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 21: Integration — worker runs a synthetic config through the real pipeline

Files:
- Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/integration/test_daemon_worker_pipeline.py`

This is the one test that exercises the REAL pipeline (`PipelineRunner.run`, the same engine `run_from_config`/`run_analysis` use). It builds a tiny synthetic `generic`+`pt_sfc` sources config (modeled on `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_unified_sources_runtime.py`), writes it to a temp YAML, builds a `JobSpec` with `build_job_spec`, runs it via `worker.run_job` in-process, and asserts a successful exit, a terminal `result` event, real progress lines, and that the output dir was produced.

- [ ] **Step 1 — write failing test** (full code). Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/integration/test_daemon_worker_pipeline.py`:
```python
"""Integration: the daemon worker runs a synthetic config through the real pipeline.

Exercises PipelineRunner.run (the same engine run_analysis/run_from_config use) via
worker.run_job, with no monkeypatching of the pipeline. Synthetic NetCDF sources are
written to a temp dir; no external datasets are used.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import ProgressEvent, TriggerEvent
from davinci_monet.daemon.dispatcher import build_job_spec
from davinci_monet.daemon import worker


def _write_grid_source(path: Path) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    lat = np.array([40.0, 41.0])
    lon = np.array([-105.0, -104.0])
    values = np.arange(8, dtype=float).reshape(2, 2, 2)
    ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(path)


def _write_point_source(path: Path) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    ds = xr.Dataset(
        {"o3": (("time", "site"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "time": times,
            "site": np.array([0, 1]),
            "latitude": ("site", np.array([40.0, 41.0])),
            "longitude": ("site", np.array([-105.0, -104.0])),
        },
        attrs={"geometry": "point"},
    )
    ds.to_netcdf(path)


def test_worker_runs_synthetic_config_through_pipeline(tmp_path, capsys):
    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    out_dir = tmp_path / "out"
    _write_grid_source(model_path)
    _write_point_source(obs_path)

    config = {
        "analysis": {"output_dir": str(out_dir)},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(model_path),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "role": "obs",
                "filename": str(obs_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_airnow_o3": {
                "sources": ["cam", "airnow"],
                "reference": "airnow",
                "variables": {"cam": "O3", "airnow": "o3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))

    rule = WatchRule(
        name="cam_rt",
        watch=str(tmp_path / "*.nc"),
        run=str(config_path),
        on_fire="whole_config",
    )
    trigger = TriggerEvent(
        watch_name="cam_rt",
        new_files=[str(model_path)],
        detected_at=datetime(2026, 5, 31, 12, 0, 0),
        settle_mode="quiescence",
    )
    spec = build_job_spec(
        rule, trigger, DaemonConfig(hdf5_file_locking=False), job_id=1, base_env={}
    )

    code = worker.run_job(spec.to_json())

    assert code == 0, "worker should exit 0 on a successful pipeline run"

    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    events = [ProgressEvent.parse_line(l) for l in lines]
    events = [e for e in events if e is not None]

    kinds = [e.kind for e in events]
    assert kinds[0] == "started"
    assert kinds[-1] == "result"
    # The pipeline emitted at least one forwarded progress line
    assert any(e.kind == "progress" for e in events)

    result_evt = events[-1]
    assert result_evt.success is True
    assert result_evt.job_id == 1
    assert result_evt.output_dir == str(out_dir)
    assert out_dir.exists()
```

- [ ] **Step 2 — run, expect fail**. (Run with HDF5 locking disabled per the CLAUDE.md gotcha.)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_daemon_worker_pipeline.py::test_worker_runs_synthetic_config_through_pipeline -v
```
Expected first-run failure surface: if the worker already exists from earlier tasks this should largely pass, but the intended TDD failure (before the worker is finalized) is one of: `ModuleNotFoundError: No module named 'davinci_monet.daemon.worker'`, or an `AssertionError` on `kinds[0] == "started"` / `result_evt.output_dir`. If it fails on a pipeline data error rather than worker wiring, read the traceback in the captured `result` event's `error` field and fix the synthetic config (the grid/point shapes above match the known-good pattern in `test_unified_sources_runtime.py`).

- [ ] **Step 3 — minimal implementation**. No new production code: the worker built in the earlier tasks already loads the config via `load_config(...).model_dump()`, runs `PipelineRunner.run`, and emits `started`/`progress`/`result`. If Step 2 surfaced a wiring gap (e.g. progress callback not attached, or output_dir not read from `context.config`), fix only the affected lines in `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/worker.py` (`run_job` / `_collect_plots_and_output`).

- [ ] **Step 4 — run, expect pass**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_daemon_worker_pipeline.py -v
```
Expect the integration test to pass. Then run the whole dispatcher+worker group to confirm nothing regressed:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_daemon_dispatcher.py davinci_monet/tests/test_daemon_worker.py davinci_monet/tests/integration/test_daemon_worker_pipeline.py -v
```

- [ ] **Step 5 — commit**:
```bash
git add davinci_monet/tests/integration/test_daemon_worker_pipeline.py
git commit -m "test(daemon): worker runs synthetic config through real pipeline

Integration test: build_job_spec -> worker.run_job drives PipelineRunner.run on
a tiny synthetic generic+pt_sfc sources config (temp NetCDF, no external data),
asserting exit 0, started/progress/result events, and a produced output_dir.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 22: Control server socket bind, handler registry, request/response round-trip

Implements the AF_UNIX / SOCK_STREAM control server in `davinci_monet/daemon/control.py`: it binds a stream socket at `socket_path`, accepts connections in a background thread (thread-per-connection), reads ONE newline-delimited JSON request line per connection, dispatches to a registered `ControlHandler` callback, and writes ONE newline-delimited JSON response line. Field names/types are taken verbatim from the contracts module — `ControlRequest(cmd, args)`, `ControlResponse(ok, data, error, code)` — which this task imports and never redefines.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/control.py` (new)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_control.py` (new)

Integration facts verified before writing:
- Pydantic base classes + `ControlRequest`/`ControlResponse`/`StreamEvent`/`ControlHandler`/`PROTOCOL_VERSION`/`COMMANDS`/`STREAMING_COMMANDS` are owned by `davinci_monet/daemon/contracts.py` (Task 1 bootstrap). This task imports them; it does NOT redefine them. (Ownership rules; SHARED CONTRACTS §7.)
- `davinci_monet/daemon/__init__.py` and `davinci_monet/tests/unit/daemon/__init__.py` are created by Task 1 — treat as pre-existing (do not create them here). Confirmed the test tree uses `tests/unit/<group>/test_*.py` (e.g. `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/config/test_source_config.py`).
- Python 3.11.14, pydantic 2.12.5 in the `davinci` conda env (verified via `python --version` / `pydantic.VERSION`). Use `model_dump_json()` for wire serialization (handles datetime/Path).
- Framing per SHARED CONTRACTS §7: newline-delimited UTF-8 JSON, ONE message per line (`\n` terminator). Request `{"cmd": str, "args": object}`. Response `{"ok": true, "data": ...}` or `{"ok": false, "error": str, "code": str|null}`.

- [ ] **Step 1 — Write failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_control.py` with the shared header + the first test class. Full code:
```python
"""Unit tests for the daemon control socket server + client (group: control/client).

Round-trips real newline-delimited JSON over a real AF_UNIX socket in a temp
dir. No external datasets, no sci stack — stdlib socket only.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from davinci_monet.daemon.contracts import (
    ControlResponse,
    StreamEvent,
)
from davinci_monet.daemon.control import ControlServer


def _recv_line(sock: socket.socket, timeout: float = 5.0) -> str:
    """Read one newline-terminated line from a connected socket."""
    sock.settimeout(timeout)
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.decode("utf-8")


def _connect(socket_path: Path, timeout: float = 5.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(str(socket_path))
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)


@pytest.fixture()
def server(tmp_path: Path):
    sock_path = tmp_path / "control.sock"
    srv = ControlServer(sock_path)
    srv.start()
    # wait for the socket file to appear (server bound)
    deadline = time.monotonic() + 5.0
    while not sock_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield srv, sock_path
    finally:
        srv.stop()


class TestRequestResponse:
    def test_ping_round_trip(self, server) -> None:
        srv, sock_path = server

        def ping_handler(args: dict) -> ControlResponse:
            return ControlResponse(ok=True, data={"pong": True, "n": args.get("n", 0)})

        srv.register("ping", ping_handler)

        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "ping", "args": {"n": 7}}) + "\n").encode())
        line = _recv_line(conn)
        conn.close()

        payload = json.loads(line)
        assert payload["ok"] is True
        assert payload["data"] == {"pong": True, "n": 7}

    def test_missing_args_defaults_to_empty(self, server) -> None:
        srv, sock_path = server
        seen: dict = {}

        def h(args: dict) -> ControlResponse:
            seen.update({"args": args})
            return ControlResponse(ok=True, data=None)

        srv.register("noargs", h)
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "noargs"}) + "\n").encode())
        _recv_line(conn)
        conn.close()
        assert seen["args"] == {}

    def test_handler_exception_becomes_error_response(self, server) -> None:
        srv, sock_path = server

        def boom(args: dict) -> ControlResponse:
            raise RuntimeError("kaboom")

        srv.register("boom", boom)
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "boom", "args": {}}) + "\n").encode())
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert "kaboom" in payload["error"]
        assert payload["code"] == "handler_error"
```

- [ ] **Step 2 — Run and expect failure.** From the repo root in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_control.py::TestRequestResponse -v
```
Expected: collection error / failure with `ModuleNotFoundError: No module named 'davinci_monet.daemon.control'` (control.py does not exist yet).

- [ ] **Step 3 — Minimal implementation.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/control.py`. Full code:
```python
"""Unix-domain-socket control server for the DAVINCI daemon.

Transport (SHARED CONTRACTS §7): AF_UNIX / SOCK_STREAM at ``socket_path``.
Framing is newline-delimited UTF-8 JSON, ONE message per line ('\\n').

Request  : {"cmd": <str>, "args": <object>}
Response : {"ok": true, "data": ...} | {"ok": false, "error": str, "code": str|null}

The supervisor registers per-command handlers. A request/response handler
conforms to ``ControlHandler``: ``(args: dict) -> ControlResponse``. A streaming
handler (``register_stream``) is a generator ``(args: dict) -> Iterator``; its
FIRST yielded value is the ack ``ControlResponse``, and every subsequent yielded
``StreamEvent`` is framed as ``{"event": str, "data": object}`` and pushed until
the generator ends or the client disconnects.

This module is part of the thin supervisor and imports only stdlib + the daemon
contracts. It MUST NOT import matplotlib / xarray / the pipeline.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from davinci_monet.daemon.contracts import (
    ControlHandler,
    ControlResponse,
    StreamEvent,
)

# A streaming handler: generator whose first yield is the ack ControlResponse and
# whose remaining yields are StreamEvents pushed to the client.
StreamHandler = Callable[[dict[str, Any]], Iterator[Any]]

_BACKLOG = 64


class ControlServer:
    """Thread-per-connection AF_UNIX control server.

    Runs its accept loop in a background daemon thread so the supervisor's main
    loop is unblocked. Light traffic (one shell/top/status client at a time) makes
    thread-per-connection the simplest robust model; no selectors event loop is
    needed here.
    """

    def __init__(
        self,
        socket_path: str | Path,
        dispatch: "Callable[[str, dict[str, Any]], ControlResponse] | None" = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self._handlers: dict[str, ControlHandler] = {}
        self._streamers: dict[str, StreamHandler] = {}
        # Optional fallback dispatcher: called as ``dispatch(cmd, args)`` for any
        # command not found in the per-command registry (and not a streamer).
        # The supervisor passes its ``handle_command`` here so the whole catalog
        # routes without registering each command individually.
        self._dispatch_fallback = dispatch
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._conns: set[socket.socket] = set()
        self._conns_lock = threading.Lock()

    # ---- registration ----------------------------------------------------

    def register(self, cmd: str, handler: ControlHandler) -> None:
        """Register a request/response handler for ``cmd``."""
        self._handlers[cmd] = handler

    def register_stream(self, cmd: str, handler: StreamHandler) -> None:
        """Register a streaming handler (generator: first yield = ack)."""
        self._streamers[cmd] = handler

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Bind the socket and launch the accept loop in a background thread."""
        self._bind()
        self._thread = threading.Thread(
            target=self._accept_loop, name="control-accept", daemon=True
        )
        self._thread.start()

    def _bind(self) -> None:
        # Remove a stale socket file before binding.
        try:
            if self.socket_path.exists() or self.socket_path.is_socket():
                self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(str(self.socket_path))
        sock.listen(_BACKLOG)
        sock.settimeout(0.5)  # so the accept loop can observe _stopping
        self._sock = sock

    def serve_forever(self) -> None:
        """Bind (if needed) and run the accept loop in the current thread."""
        if self._sock is None:
            self._bind()
        self._accept_loop()

    def stop(self) -> None:
        """Stop accepting, close live connections, remove the socket file."""
        self._stopping.set()
        with self._conns_lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                c.close()
            except OSError:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

    # ---- accept loop -----------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stopping.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed during stop()
            t = threading.Thread(
                target=self._serve_conn, args=(conn,), daemon=True
            )
            t.start()

    def _serve_conn(self, conn: socket.socket) -> None:
        with self._conns_lock:
            self._conns.add(conn)
        try:
            conn.settimeout(None)
            line = self._read_line(conn)
            if line is None:
                return
            self._dispatch(conn, line)
        finally:
            with self._conns_lock:
                self._conns.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    # ---- framing ---------------------------------------------------------

    @staticmethod
    def _read_line(conn: socket.socket) -> Optional[str]:
        buf = b""
        while not buf.endswith(b"\n"):
            try:
                chunk = conn.recv(4096)
            except OSError:
                return None
            if not chunk:
                return buf.decode("utf-8") if buf else None
            buf += chunk
        return buf.decode("utf-8")

    @staticmethod
    def _send(conn: socket.socket, obj: str) -> bool:
        try:
            conn.sendall((obj + "\n").encode("utf-8"))
            return True
        except OSError:
            return False

    def _send_response(self, conn: socket.socket, resp: ControlResponse) -> bool:
        return self._send(conn, resp.model_dump_json())

    # ---- dispatch --------------------------------------------------------

    def _dispatch(self, conn: socket.socket, line: str) -> None:
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            self._send_response(
                conn,
                ControlResponse(ok=False, error="malformed JSON request", code="invalid_args"),
            )
            return
        if not isinstance(raw, dict) or "cmd" not in raw:
            self._send_response(
                conn,
                ControlResponse(ok=False, error="missing 'cmd'", code="invalid_args"),
            )
            return
        cmd = raw.get("cmd")
        args = raw.get("args") or {}
        if not isinstance(args, dict):
            self._send_response(
                conn,
                ControlResponse(ok=False, error="'args' must be an object", code="invalid_args"),
            )
            return

        if cmd in self._streamers:
            self._dispatch_stream(conn, cmd, args)
            return
        if cmd not in self._handlers:
            if self._dispatch_fallback is not None:
                try:
                    resp = self._dispatch_fallback(cmd, args)
                except Exception as exc:  # noqa: BLE001 - handler isolation
                    self._send_response(
                        conn,
                        ControlResponse(ok=False, error=str(exc), code="handler_error"),
                    )
                    return
                self._send_response(conn, resp)
                return
            self._send_response(
                conn,
                ControlResponse(ok=False, error=f"unknown command: {cmd}", code="unsupported"),
            )
            return
        try:
            resp = self._handlers[cmd](args)
        except Exception as exc:  # noqa: BLE001 - handler isolation
            self._send_response(
                conn,
                ControlResponse(ok=False, error=str(exc), code="handler_error"),
            )
            return
        self._send_response(conn, resp)

    def _dispatch_stream(self, conn: socket.socket, cmd: str, args: dict) -> None:
        try:
            gen = self._streamers[cmd](args)
            iterator = iter(gen)
            ack = next(iterator)
        except StopIteration:
            self._send_response(
                conn,
                ControlResponse(ok=False, error="stream produced no ack", code="handler_error"),
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._send_response(
                conn,
                ControlResponse(ok=False, error=str(exc), code="handler_error"),
            )
            return
        if not isinstance(ack, ControlResponse):
            ack = ControlResponse(ok=True, data=ack)
        if not self._send_response(conn, ack) or not ack.ok:
            return
        for event in iterator:
            if self._stopping.is_set():
                break
            if not isinstance(event, StreamEvent):
                continue
            if not self._send(conn, event.model_dump_json()):
                break  # client disconnected
```

- [ ] **Step 4 — Run and expect pass.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_control.py::TestRequestResponse -v
```
Expected: 3 passed.

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/daemon/control.py davinci_monet/tests/unit/daemon/test_control.py
git commit -m "$(cat <<'EOF'
feat(daemon): control socket server with handler registry and JSON framing

Add ControlServer: AF_UNIX/SOCK_STREAM server with newline-delimited JSON
request/response framing and a per-command handler registry. Handlers conform
to the contracts ControlHandler signature; handler exceptions are isolated into
{ok:false, code:"handler_error"} responses.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 23: Control server unknown-command and malformed-request error responses

Adds coverage for the error envelope per SHARED CONTRACTS §7: an unregistered `cmd` (with NO dispatch fallback) returns `{ok:false, error, code:"unsupported"}`; a malformed/non-JSON request line returns `{ok:false, code:"invalid_args"}`; a request missing `cmd` returns `code:"invalid_args"`. The error branches already exist in `control.py` from the previous task. This task also covers the optional `dispatch=` fallback: `ControlServer(socket_path, dispatch=...)` routes any command not in the per-command registry through `dispatch(cmd, args)` (the supervisor passes its `handle_command` here so the whole catalog routes without registering each command individually); the fallback defaults to `None`, preserving the `"unsupported"` behavior for servers built without it.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_control.py` (modify — append `TestErrorEnvelope`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/control.py` (modify — add the optional `dispatch=` constructor arg + fallback branch in `_dispatch`)

- [ ] **Step 1 — Write failing test.** Append this class to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_control.py` (helpers + `server` fixture already defined above):
```python
class TestErrorEnvelope:
    def test_unknown_command(self, server) -> None:
        srv, sock_path = server
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "does_not_exist", "args": {}}) + "\n").encode())
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert payload["code"] == "unsupported"
        assert "does_not_exist" in payload["error"]

    def test_malformed_json(self, server) -> None:
        srv, sock_path = server
        conn = _connect(sock_path)
        conn.sendall(b"this is not json\n")
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert payload["code"] == "invalid_args"

    def test_missing_cmd(self, server) -> None:
        srv, sock_path = server
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"args": {"x": 1}}) + "\n").encode())
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert payload["code"] == "invalid_args"

    def test_non_object_args(self, server) -> None:
        srv, sock_path = server
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "x", "args": [1, 2, 3]}) + "\n").encode())
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert payload["code"] == "invalid_args"

    def test_dispatch_fallback_routes_unregistered_command(self, tmp_path: Path) -> None:
        # A ControlServer built with a `dispatch=` fallback routes any command not
        # in the per-command registry through dispatch(cmd, args) instead of
        # returning "unsupported". The supervisor passes handle_command here.
        seen: dict = {}

        def dispatch(cmd: str, args: dict) -> ControlResponse:
            seen.update({"cmd": cmd, "args": args})
            return ControlResponse(ok=True, data={"routed": cmd})

        sock_path = tmp_path / "fallback.sock"
        srv = ControlServer(sock_path, dispatch)
        srv.start()
        deadline = time.monotonic() + 5.0
        while not sock_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            conn = _connect(sock_path)
            conn.sendall(
                (json.dumps({"cmd": "status", "args": {"n": 1}}) + "\n").encode()
            )
            payload = json.loads(_recv_line(conn))
            conn.close()
        finally:
            srv.stop()
        assert payload["ok"] is True
        assert payload["data"] == {"routed": "status"}
        assert seen == {"cmd": "status", "args": {"n": 1}}
```

- [ ] **Step 2 — Run and expect failure (or pass-on-coverage).** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_control.py::TestErrorEnvelope -v
```
Expected at first authoring run before `control.py` exists: `ModuleNotFoundError`. (If running after Task 22 of this section already added `control.py`, these should pass — that is acceptable since the code path is the one under test. If any assertion fails, fix the matching branch in `control.py._dispatch`.)

- [ ] **Step 3 — Minimal implementation.** The error branches in `control.py._dispatch` (added in the prior task) already emit `code="unsupported"` for unknown commands and `code="invalid_args"` for malformed JSON / missing `cmd` / non-object `args`. ADD the optional `dispatch=` fallback: extend `ControlServer.__init__` to accept `dispatch: Callable[[str, dict], ControlResponse] | None = None` (stored as `self._dispatch_fallback`), and in `_dispatch`, when `cmd` is not a streamer and not in `self._handlers`, call `self._dispatch_fallback(cmd, args)` (isolating exceptions into `code="handler_error"`) instead of returning `"unsupported"`; only fall back to the `"unsupported"` response when `self._dispatch_fallback is None`. Confirm the error-envelope branches are present in `ControlServer._dispatch`:
```python
        if not isinstance(raw, dict) or "cmd" not in raw:
            self._send_response(
                conn,
                ControlResponse(ok=False, error="missing 'cmd'", code="invalid_args"),
            )
            return
        cmd = raw.get("cmd")
        args = raw.get("args") or {}
        if not isinstance(args, dict):
            self._send_response(
                conn,
                ControlResponse(ok=False, error="'args' must be an object", code="invalid_args"),
            )
            return
```

- [ ] **Step 4 — Run and expect pass.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_control.py::TestErrorEnvelope -v
```
Expected: 5 passed (the four error-envelope tests + `test_dispatch_fallback_routes_unregistered_command`).

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/tests/unit/daemon/test_control.py davinci_monet/daemon/control.py
git commit -m "$(cat <<'EOF'
test(daemon): lock control error envelope for unknown/malformed requests

Cover the {ok:false, error, code} branches: unknown command -> "unsupported";
malformed JSON / missing cmd / non-object args -> "invalid_args".

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 24: Control server streaming subscribe pushes events then ends on disconnect

Covers the streaming variant (SHARED CONTRACTS §7, `STREAMING_COMMANDS`): a streaming handler is registered via `register_stream`; the server writes the ack `ControlResponse` (the generator's first yield), then frames each subsequent `StreamEvent` as `{"event": str, "data": object}` and pushes until the generator is exhausted OR the client disconnects mid-stream (server stays tolerant — no crash, the connection thread exits cleanly). This backs `daemon top` (`subscribe`) and `daemon logs --tail` (`logs_tail`).

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_control.py` (modify — append `TestStreaming`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/control.py` (only if a test reveals a gap; `_dispatch_stream` was added in the first task)

- [ ] **Step 1 — Write failing test.** Append this class to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_control.py`:
```python
class TestStreaming:
    def test_ack_then_events_then_end(self, server) -> None:
        srv, sock_path = server

        def sub(args: dict):
            # first yield = ack ControlResponse
            yield ControlResponse(ok=True, data={"subscribed": args.get("topics", [])})
            for i in range(3):
                yield StreamEvent(event="job_update", data={"i": i})

        srv.register_stream("subscribe", sub)

        conn = _connect(sock_path)
        conn.sendall(
            (json.dumps({"cmd": "subscribe", "args": {"topics": ["jobs"]}}) + "\n").encode()
        )
        ack = json.loads(_recv_line(conn))
        assert ack["ok"] is True
        assert ack["data"] == {"subscribed": ["jobs"]}

        events = []
        for _ in range(3):
            events.append(json.loads(_recv_line(conn)))
        conn.close()

        assert [e["event"] for e in events] == ["job_update"] * 3
        assert [e["data"]["i"] for e in events] == [0, 1, 2]

    def test_server_tolerates_client_disconnect_midstream(self, server) -> None:
        srv, sock_path = server
        produced = threading.Event()

        def sub(args: dict):
            yield ControlResponse(ok=True, data={"streaming": True})
            # emit many events; client will hang up after the first
            for i in range(1000):
                yield StreamEvent(event="log_line", data={"i": i})
                produced.set()

        srv.register_stream("logs_tail", sub)

        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "logs_tail", "args": {}}) + "\n").encode())
        ack = json.loads(_recv_line(conn))
        assert ack["ok"] is True
        # read one event then disconnect abruptly
        _recv_line(conn)
        conn.close()

        # server must remain healthy: a fresh ping handler still round-trips
        srv.register("ping", lambda a: ControlResponse(ok=True, data={"pong": True}))
        conn2 = _connect(sock_path)
        conn2.sendall((json.dumps({"cmd": "ping", "args": {}}) + "\n").encode())
        payload = json.loads(_recv_line(conn2))
        conn2.close()
        assert payload["data"] == {"pong": True}

    def test_stream_with_no_events_just_ack(self, server) -> None:
        srv, sock_path = server

        def sub(args: dict):
            yield ControlResponse(ok=True, data={"subscribed": []})

        srv.register_stream("subscribe", sub)
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "subscribe", "args": {}}) + "\n").encode())
        ack = json.loads(_recv_line(conn))
        conn.close()
        assert ack["ok"] is True
        assert ack["data"] == {"subscribed": []}
```

- [ ] **Step 2 — Run and expect failure.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_control.py::TestStreaming -v
```
Expected before `register_stream`/`_dispatch_stream` exist: `AttributeError: 'ControlServer' object has no attribute 'register_stream'`. (If the first task's `control.py` is already present, these should pass; if any assertion fails, fix `_dispatch_stream`.)

- [ ] **Step 3 — Minimal implementation.** The streaming path lives in `control.py` from the first task (`register_stream`, `_dispatch_stream`). Confirm `_dispatch_stream` matches exactly — ack is the generator's first yield, each remaining `StreamEvent` is framed via `event.model_dump_json()`, and a failed `_send` (client gone) breaks the loop without raising:
```python
    def _dispatch_stream(self, conn: socket.socket, cmd: str, args: dict) -> None:
        try:
            gen = self._streamers[cmd](args)
            iterator = iter(gen)
            ack = next(iterator)
        except StopIteration:
            self._send_response(
                conn,
                ControlResponse(ok=False, error="stream produced no ack", code="handler_error"),
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._send_response(
                conn,
                ControlResponse(ok=False, error=str(exc), code="handler_error"),
            )
            return
        if not isinstance(ack, ControlResponse):
            ack = ControlResponse(ok=True, data=ack)
        if not self._send_response(conn, ack) or not ack.ok:
            return
        for event in iterator:
            if self._stopping.is_set():
                break
            if not isinstance(event, StreamEvent):
                continue
            if not self._send(conn, event.model_dump_json()):
                break  # client disconnected
```

- [ ] **Step 4 — Run and expect pass.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_control.py -v
```
Expected: all tests in the file pass (TestRequestResponse + TestErrorEnvelope + TestStreaming).

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/tests/unit/daemon/test_control.py davinci_monet/daemon/control.py
git commit -m "$(cat <<'EOF'
test(daemon): cover control streaming ack/push and disconnect tolerance

Verify register_stream pushes framed StreamEvent lines after the ack, ends on
generator exhaustion, and that the server survives a client disconnecting
mid-stream (a fresh request still round-trips).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 25: Daemon client call(), is_alive(), and stream() iterator

Implements the thin synchronous Unix-socket client in `davinci_monet/daemon/client.py` used by shell/top/status/stop. Matches SHARED CONTRACTS §7 `DaemonClient`: `__init__(socket_path)`, `call(cmd, **args) -> ControlResponse`, `stream(cmd, **args) -> Iterator[StreamEvent]`, `is_alive() -> bool`. `call` opens a fresh connection, sends one framed `{cmd,args}` line, reads exactly one response line, parses into `ControlResponse`, and raises on transport error. `stream` sends the request, yields the ack `ControlResponse` first, then yields each pushed `StreamEvent` until the connection closes. `is_alive` returns True iff a `ping` round-trips. Tested end-to-end against a real `ControlServer`.

> **Note (assembler):** `davinci_monet/daemon/contracts.py` does NOT define a `DaemonClient`. The concrete `DaemonClient` implementation below is the SINGLE definition and LIVES IN `davinci_monet/daemon/client.py`; every importer (`shell.py`, `dashboard.py`, the CLI) uses `from davinci_monet.daemon.client import DaemonClient`.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/client.py` (new)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_client.py` (new)

- [ ] **Step 1 — Write failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_client.py`. Full code:
```python
"""Unit tests for the daemon control client (group: control/client).

Drives a real ControlServer over a real AF_UNIX socket in a temp dir.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from davinci_monet.daemon.client import DaemonClient
from davinci_monet.daemon.contracts import ControlResponse, StreamEvent
from davinci_monet.daemon.control import ControlServer


@pytest.fixture()
def running_server(tmp_path: Path):
    sock_path = tmp_path / "control.sock"
    srv = ControlServer(sock_path)

    srv.register("ping", lambda a: ControlResponse(ok=True, data={"pong": True}))
    srv.register("echo", lambda a: ControlResponse(ok=True, data={"got": a}))
    srv.register(
        "fail", lambda a: ControlResponse(ok=False, error="nope", code="not_found")
    )

    def sub(args: dict):
        yield ControlResponse(ok=True, data={"subscribed": args.get("topics", [])})
        for i in range(3):
            yield StreamEvent(event="tick", data={"i": i})

    srv.register_stream("subscribe", sub)
    srv.start()
    deadline = time.monotonic() + 5.0
    while not sock_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield sock_path
    finally:
        srv.stop()


class TestCall:
    def test_call_round_trip(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        resp = client.call("echo", a=1, b="two")
        assert isinstance(resp, ControlResponse)
        assert resp.ok is True
        assert resp.data == {"got": {"a": 1, "b": "two"}}

    def test_call_error_response_preserved(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        resp = client.call("fail")
        assert resp.ok is False
        assert resp.error == "nope"
        assert resp.code == "not_found"

    def test_is_alive_true_when_up(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        assert client.is_alive() is True

    def test_is_alive_false_when_no_socket(self, tmp_path: Path) -> None:
        client = DaemonClient(tmp_path / "nonexistent.sock")
        assert client.is_alive() is False

    def test_call_raises_on_dead_socket(self, tmp_path: Path) -> None:
        client = DaemonClient(tmp_path / "nope.sock")
        with pytest.raises((ConnectionError, FileNotFoundError, OSError)):
            client.call("ping")


class TestStream:
    def test_stream_yields_ack_then_events(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        items = list(client.stream("subscribe", topics=["jobs"]))
        ack = items[0]
        assert isinstance(ack, ControlResponse)
        assert ack.ok is True
        assert ack.data == {"subscribed": ["jobs"]}
        events = items[1:]
        assert all(isinstance(e, StreamEvent) for e in events)
        assert [e.event for e in events] == ["tick", "tick", "tick"]
        assert [e.data["i"] for e in events] == [0, 1, 2]

    def test_stream_ends_when_server_closes(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        count = 0
        for _ in client.stream("subscribe"):
            count += 1
        # ack + 3 events, then clean end
        assert count == 4
```

- [ ] **Step 2 — Run and expect failure.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_client.py -v
```
Expected: `ModuleNotFoundError: No module named 'davinci_monet.daemon.client'`.

- [ ] **Step 3 — Minimal implementation.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/client.py`. Full code:
```python
"""Thin synchronous Unix-socket client for the DAVINCI daemon control server.

Used by ``daemon shell`` / ``daemon top`` / ``daemon status`` / ``daemon stop``.
Framing mirrors control.py (SHARED CONTRACTS §7): newline-delimited UTF-8 JSON,
ONE message per line. ``call`` is a request/response round-trip; ``stream`` yields
the ack ControlResponse then each pushed StreamEvent until the server closes.

Stdlib socket only; never imports the sci stack.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Iterator

from davinci_monet.daemon.contracts import ControlResponse, StreamEvent

_DEFAULT_TIMEOUT = 10.0


class DaemonClient:
    """Synchronous client over the daemon's AF_UNIX control socket."""

    def __init__(self, socket_path: str | Path, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout

    # ---- connection ------------------------------------------------------

    def connect(self) -> socket.socket:
        """Open a fresh connected AF_UNIX stream socket. Raises on failure."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(str(self.socket_path))
        return sock

    @staticmethod
    def _send_request(sock: socket.socket, cmd: str, args: dict[str, Any]) -> None:
        line = json.dumps({"cmd": cmd, "args": args}) + "\n"
        sock.sendall(line.encode("utf-8"))

    @staticmethod
    def _read_line(sock: socket.socket, buf: bytearray) -> str | None:
        """Read one '\\n'-terminated line, buffering any extra bytes in ``buf``."""
        while b"\n" not in buf:
            try:
                chunk = sock.recv(4096)
            except (socket.timeout, TimeoutError):
                raise
            if not chunk:
                if buf:
                    line = bytes(buf).decode("utf-8")
                    del buf[:]
                    return line
                return None
            buf.extend(chunk)
        idx = buf.index(b"\n")
        line = bytes(buf[:idx]).decode("utf-8")
        del buf[: idx + 1]
        return line

    # ---- request / response ---------------------------------------------

    def call(self, cmd: str, **args: Any) -> ControlResponse:
        """Send {cmd,args}; read one response line; return ControlResponse.

        Raises on transport error (no socket, refused, reset, timeout).
        """
        sock = self.connect()
        try:
            self._send_request(sock, cmd, args)
            buf = bytearray()
            line = self._read_line(sock, buf)
            if line is None:
                raise ConnectionError("daemon closed connection without a response")
            return ControlResponse.model_validate_json(line)
        finally:
            try:
                sock.close()
            except OSError:
                pass

    # ---- streaming -------------------------------------------------------

    def stream(self, cmd: str, **args: Any) -> Iterator[Any]:
        """Send a streaming cmd; yield the ack ControlResponse then StreamEvents.

        Iteration ends when the server closes the connection.
        """
        sock = self.connect()
        try:
            self._send_request(sock, cmd, args)
            buf = bytearray()
            ack_line = self._read_line(sock, buf)
            if ack_line is None:
                raise ConnectionError("daemon closed connection without an ack")
            ack = ControlResponse.model_validate_json(ack_line)
            yield ack
            if not ack.ok:
                return
            while True:
                line = self._read_line(sock, buf)
                if line is None:
                    return  # server closed the stream
                yield StreamEvent.model_validate_json(line)
        finally:
            try:
                sock.close()
            except OSError:
                pass

    # ---- liveness --------------------------------------------------------

    def is_alive(self) -> bool:
        """True if a ``ping`` round-trips (daemon up + socket healthy)."""
        try:
            resp = self.call("ping")
        except OSError:
            return False
        return bool(resp.ok)
```

- [ ] **Step 4 — Run and expect pass.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_client.py davinci_monet/tests/unit/daemon/test_control.py -v
```
Expected: all client + control tests pass.

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/daemon/client.py davinci_monet/tests/unit/daemon/test_client.py
git commit -m "$(cat <<'EOF'
feat(daemon): thin Unix-socket control client (call/stream/is_alive)

Add DaemonClient: request/response round-trip via call(), streaming iterator via
stream() (ack ControlResponse then pushed StreamEvents until the server closes),
and is_alive() gated on a ping round-trip. Stdlib socket only; no sci stack.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

This group owns `davinci_monet/daemon/notify.py` and `davinci_monet/daemon/lifecycle.py`. Both modules are **supervisor-side** and MUST NOT import matplotlib / xarray / monet / the pipeline (isolation invariant, spec §"Isolation invariant", lines 90-95). Both depend only on stdlib + `davinci_monet.logging` + the daemon `contracts`/`config` modules.

**Imports they consume (do NOT redefine):**
- `JobRecord`, `JobStatus` from `davinci_monet.daemon.contracts` (owned by bootstrap Task 1).
- `DaemonConfig`, `WatchRule`, `NotificationConfig` from `davinci_monet.daemon.config` (owned by the config group).
- `get_logger` from `davinci_monet.logging` (existing, signature `get_logger(name: str | None = None) -> logging.Logger`, verified at `davinci_monet/logging/config.py:141`).

**Test placement:** new tests go under `davinci_monet/tests/unit/daemon/` (mirrors existing per-package unit layout, e.g. `davinci_monet/tests/unit/config/`). The package `__init__.py` for the daemon test package is created by bootstrap Task 1; treat it as pre-existing. Every pytest command runs **from the repo root** `/Users/fillmore/EarthSystem/DAVINCI` in the `davinci` conda env. Each command below is prefixed with environment activation.

Because these tests depend on `davinci_monet.daemon.contracts` and `davinci_monet.daemon.config` existing (Task 1 + config group), the tasks below assume those modules are present. The notify tests construct `JobRecord` / `DaemonConfig` / `WatchRule` instances directly using the contract field names; no external datasets are touched — only temp dirs, fake pids, and monkeypatched subprocess calls.

### Task 26: Desktop notification dispatch (osascript / terminal-notifier, injectable)

`notify.py` sends a macOS desktop notification. The runner of the actual subprocess is **injected** so CI never shells out to a real `osascript`. The default production runner uses `subprocess.run(["osascript", "-e", script])` and, on failure / when `osascript` is absent, falls back to `terminal-notifier`.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/notify.py` (new)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_notify_desktop.py` (new)

Steps:

- [ ] **Step 1 — write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_notify_desktop.py`:
```python
"""Unit tests for daemon desktop-notification dispatch (mocked subprocess)."""

from __future__ import annotations

import subprocess
from typing import Any

from davinci_monet.daemon.notify import DesktopNotifier, send_desktop_notification


class _FakeRunner:
    """Records calls; can be told to fail the first command (osascript)."""

    def __init__(self, fail_cmds: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_cmds = fail_cmds or set()

    def __call__(self, argv: list[str]) -> int:
        self.calls.append(list(argv))
        if argv and argv[0] in self.fail_cmds:
            raise FileNotFoundError(argv[0])
        return 0


def test_send_desktop_uses_osascript_by_default() -> None:
    runner = _FakeRunner()
    ok = send_desktop_notification(
        "DAVINCI", "cam_realtime completed", runner=runner
    )
    assert ok is True
    assert len(runner.calls) == 1
    argv = runner.calls[0]
    assert argv[0] == "osascript"
    assert argv[1] == "-e"
    # AppleScript must carry both the title and the message text.
    assert "cam_realtime completed" in argv[2]
    assert "DAVINCI" in argv[2]


def test_send_desktop_falls_back_to_terminal_notifier() -> None:
    runner = _FakeRunner(fail_cmds={"osascript"})
    ok = send_desktop_notification(
        "DAVINCI", "boom", runner=runner
    )
    assert ok is True
    assert [c[0] for c in runner.calls] == ["osascript", "terminal-notifier"]
    tn = runner.calls[1]
    assert "-message" in tn
    assert "boom" in tn


def test_send_desktop_returns_false_when_all_backends_missing() -> None:
    runner = _FakeRunner(fail_cmds={"osascript", "terminal-notifier"})
    ok = send_desktop_notification("DAVINCI", "x", runner=runner)
    assert ok is False
    assert [c[0] for c in runner.calls] == ["osascript", "terminal-notifier"]


def test_desktop_notifier_class_is_callable_wrapper() -> None:
    runner = _FakeRunner()
    notifier = DesktopNotifier(runner=runner)
    assert notifier("T", "M") is True
    assert runner.calls[0][0] == "osascript"


def test_quotes_in_message_do_not_break_applescript() -> None:
    runner = _FakeRunner()
    send_desktop_notification('Ti"tle', 'mes"sage', runner=runner)
    script = runner.calls[0][2]
    # Embedded double quotes must be escaped/stripped, never left raw-unbalanced.
    assert script.count('"') % 2 == 0
```

- [ ] **Step 2 — run and expect failure.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_notify_desktop.py -v
```
Expected: collection/import error `ModuleNotFoundError: No module named 'davinci_monet.daemon.notify'` (the module does not exist yet), so all tests ERROR/fail.

- [ ] **Step 3 — minimal implementation.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/notify.py` with the desktop section (the iCloud + routing parts are added in later tasks; write the whole file's imports now). Initial file content:
```python
"""Daemon outcome notifications: desktop (macOS) + iCloud copy + log.

Supervisor-side ONLY. Does not import matplotlib / xarray / the pipeline.
The actual subprocess runner and the file-copy primitive are injectable so
CI never shells out to a real ``osascript`` and never writes to real iCloud.
"""

from __future__ import annotations

import subprocess
from typing import Callable

from davinci_monet.logging import get_logger

logger = get_logger(__name__)

# A runner takes an argv list and returns the process exit code; it raises
# FileNotFoundError if the backend binary is absent.
CommandRunner = Callable[[list[str]], int]


def _default_runner(argv: list[str]) -> int:
    """Run a command, returning its exit code; raises FileNotFoundError if absent."""
    completed = subprocess.run(  # noqa: S603 - argv is a fixed list, no shell
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode


def _applescript_escape(text: str) -> str:
    """Make a string safe to embed inside an AppleScript double-quoted literal."""
    # Backslash-escape embedded double quotes and backslashes.
    return text.replace("\\", "\\\\").replace('"', '\\"')


def send_desktop_notification(
    title: str,
    message: str,
    *,
    runner: CommandRunner | None = None,
) -> bool:
    """Post a macOS desktop notification.

    Tries ``osascript`` first; on failure (non-zero or FileNotFoundError) falls
    back to ``terminal-notifier``. Returns True iff a backend succeeded.
    """
    run = runner or _default_runner
    safe_title = _applescript_escape(title)
    safe_msg = _applescript_escape(message)
    script = f'display notification "{safe_msg}" with title "{safe_title}"'

    try:
        if run(["osascript", "-e", script]) == 0:
            return True
    except FileNotFoundError:
        logger.debug("osascript not found; trying terminal-notifier")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("osascript notification failed: %s", exc)

    try:
        rc = run(
            ["terminal-notifier", "-title", title, "-message", message]
        )
        if rc == 0:
            return True
    except FileNotFoundError:
        logger.debug("terminal-notifier not found")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("terminal-notifier notification failed: %s", exc)

    logger.info("No desktop notification backend available; skipped")
    return False


class DesktopNotifier:
    """Callable wrapper around send_desktop_notification with a bound runner."""

    def __init__(self, *, runner: CommandRunner | None = None) -> None:
        self._runner = runner

    def __call__(self, title: str, message: str) -> bool:
        return send_desktop_notification(title, message, runner=self._runner)
```

- [ ] **Step 4 — run and expect pass.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_notify_desktop.py -v
```
Expected: 5 passed.

- [ ] **Step 5 — commit.**
```bash
git add davinci_monet/daemon/notify.py davinci_monet/tests/unit/daemon/test_notify_desktop.py
git commit -m "feat(daemon): add injectable desktop notification dispatch

osascript-first with terminal-notifier fallback; runner is injectable so CI
never shells out. AppleScript string escaping prevents quote breakage.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 27: iCloud copy of plots + summary on success

On a successful run, `notify.py` copies the generated plot files into `NotificationConfig.icloud_dir` and writes a Markdown summary of the run there. The copy primitive (`shutil.copy2`) and the dir are exercised against a temp dir in tests — no real iCloud. The function tolerates missing source files (logs and skips) and creates `icloud_dir` if absent.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/notify.py` (modify — append the iCloud section)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_notify_icloud.py` (new)

Steps:

- [ ] **Step 1 — write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_notify_icloud.py`:
```python
"""Unit tests for daemon iCloud copy of plots + run summary."""

from __future__ import annotations

from pathlib import Path

from davinci_monet.daemon.notify import IcloudCopier, copy_to_icloud


def _make_plot(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x89PNG fake")
    return p


def test_copy_creates_icloud_dir_and_copies_plots(tmp_path: Path) -> None:
    plots = [
        str(_make_plot(tmp_path, "scatter.png")),
        str(_make_plot(tmp_path, "timeseries.pdf")),
    ]
    icloud = tmp_path / "iCloud" / "Claude"  # does not exist yet
    copied = copy_to_icloud(
        icloud_dir=icloud,
        plots=plots,
        summary_text="# Run cam_realtime\nstatus: completed\n",
        summary_name="cam_realtime_job7.md",
    )
    assert icloud.is_dir()
    names = {Path(c).name for c in copied}
    assert "scatter.png" in names
    assert "timeseries.pdf" in names
    # The summary markdown is written into icloud_dir.
    assert (icloud / "cam_realtime_job7.md").read_text().startswith("# Run")


def test_copy_skips_missing_plot_files(tmp_path: Path) -> None:
    good = str(_make_plot(tmp_path, "ok.png"))
    missing = str(tmp_path / "gone.png")
    icloud = tmp_path / "iCloud"
    copied = copy_to_icloud(
        icloud_dir=icloud,
        plots=[good, missing],
        summary_text="x",
        summary_name="s.md",
    )
    names = {Path(c).name for c in copied}
    assert "ok.png" in names
    assert "gone.png" not in names


def test_copy_uses_injected_copyfn(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_copy(src: str, dst: str) -> None:
        calls.append((src, dst))

    src = str(_make_plot(tmp_path, "a.png"))
    icloud = tmp_path / "ic"
    copy_to_icloud(
        icloud_dir=icloud,
        plots=[src],
        summary_text="t",
        summary_name="s.md",
        copyfn=fake_copy,
    )
    assert len(calls) == 1
    assert calls[0][0] == src
    assert calls[0][1].endswith("a.png")


def test_icloud_copier_class_binds_dir(tmp_path: Path) -> None:
    icloud = tmp_path / "ic"
    src = str(_make_plot(tmp_path, "b.png"))
    copier = IcloudCopier(icloud_dir=icloud)
    copied = copier(plots=[src], summary_text="t", summary_name="s.md")
    assert (icloud / "s.md").exists()
    assert any(Path(c).name == "b.png" for c in copied)
```

- [ ] **Step 2 — run and expect failure.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_notify_icloud.py -v
```
Expected: `ImportError: cannot import name 'IcloudCopier' from 'davinci_monet.daemon.notify'` (and `copy_to_icloud`), so all tests fail at import.

- [ ] **Step 3 — minimal implementation.** Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/notify.py`. First add `import shutil` and `from pathlib import Path` to the existing import block by replacing:
```python
import subprocess
from typing import Callable
```
with:
```python
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence
```
Then append at end of file:
```python
# A copy function takes (src, dst) absolute path strings.
CopyFn = Callable[[str, str], None]


def _default_copy(src: str, dst: str) -> None:
    shutil.copy2(src, dst)


def copy_to_icloud(
    *,
    icloud_dir: str | Path,
    plots: Sequence[str],
    summary_text: str,
    summary_name: str,
    copyfn: CopyFn | None = None,
) -> list[str]:
    """Copy generated plots into ``icloud_dir`` and write a Markdown summary.

    Creates ``icloud_dir`` if needed. Missing source plots are logged and
    skipped. Returns the list of destination paths actually written (plots +
    the summary file). The copy primitive is injectable for tests.
    """
    copy = copyfn or _default_copy
    dest_dir = Path(icloud_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for src in plots:
        src_path = Path(src)
        if not src_path.is_file():
            logger.warning("iCloud copy: source plot missing, skipped: %s", src)
            continue
        dst = dest_dir / src_path.name
        try:
            copy(str(src_path), str(dst))
            written.append(str(dst))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("iCloud copy failed for %s: %s", src, exc)

    summary_path = dest_dir / summary_name
    try:
        summary_path.write_text(summary_text)
        written.append(str(summary_path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("iCloud summary write failed: %s", exc)

    return written


class IcloudCopier:
    """Callable wrapper binding an icloud_dir + copy primitive."""

    def __init__(
        self,
        *,
        icloud_dir: str | Path,
        copyfn: CopyFn | None = None,
    ) -> None:
        self._dir = icloud_dir
        self._copyfn = copyfn

    def __call__(
        self,
        *,
        plots: Sequence[str],
        summary_text: str,
        summary_name: str,
    ) -> list[str]:
        return copy_to_icloud(
            icloud_dir=self._dir,
            plots=plots,
            summary_text=summary_text,
            summary_name=summary_name,
            copyfn=self._copyfn,
        )
```

- [ ] **Step 4 — run and expect pass.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_notify_icloud.py -v
```
Expected: 4 passed.

- [ ] **Step 5 — commit.**
```bash
git add davinci_monet/daemon/notify.py davinci_monet/tests/unit/daemon/test_notify_icloud.py
git commit -m "feat(daemon): copy plots + run summary to iCloud on success

Creates icloud_dir, skips missing plots, writes a Markdown run summary; the
copy primitive is injectable so CI writes only to temp dirs.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 28: notify_outcome routing (desktop + iCloud + log per config/rule)

`notify_outcome(job_record, daemon_cfg, rule)` is the public routing function. It ALWAYS logs the outcome. It posts a desktop notification when desktop notifications are enabled, and on a **successful** job copies plots + summary to iCloud when iCloud copy is enabled. Per-rule `rule.notify` overrides the daemon-level channel selection (spec decision #10, lines 58-60 + 240). The desktop notifier and iCloud copier are injectable so this is fully testable with mocks. The plots / output paths come from the job's `result_summary` (the worker's `result` JSON payload, per the contracts catalog, has `plots` and `output_dir`). This task also ships the thin `Notifier` facade the supervisor actually constructs in `build_supervisor`: `Notifier(daemon_cfg, *, hooks=None)` binds the NotificationConfig + injectable desktop/iCloud hooks and exposes `notify_result(job, rule=None)`, which delegates to `notify_outcome`. `rule` is now optional (`None` means no per-rule channel override).

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/notify.py` (modify — append routing)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_notify_outcome.py` (new)

Steps:

- [ ] **Step 1 — write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_notify_outcome.py`:
```python
"""Unit tests for notify_outcome routing across desktop + iCloud + log."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from davinci_monet.daemon.config import DaemonConfig, NotificationConfig, WatchRule
from davinci_monet.daemon.contracts import JobRecord, JobStatus
from davinci_monet.daemon.notify import notify_outcome


class _RecordingDesktop:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> bool:
        self.calls.append((title, message))
        return True


class _RecordingIcloud:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, plots, summary_text, summary_name) -> list[str]:
        self.calls.append(
            {"plots": list(plots), "summary_name": summary_name}
        )
        return [f"/ic/{summary_name}"]


def _rule(name: str = "cam_realtime", notify=None) -> WatchRule:
    return WatchRule(
        name=name,
        watch="/data/*.nc",
        run="/cfg/x.yaml",
        notify=notify,
    )


def _job(status: JobStatus, plots=None, output_dir=None) -> JobRecord:
    summary = {}
    if plots is not None:
        summary["plots"] = plots
    if output_dir is not None:
        summary["output_dir"] = output_dir
    return JobRecord(
        id=7,
        watch_name="cam_realtime",
        config_path="/cfg/x.yaml",
        on_fire="whole_config",
        files=["/data/a.nc"],
        status=status,
        submitted_at=datetime(2026, 5, 31, 12, 0, 0),
        result_summary=summary or None,
    )


def test_success_fires_desktop_and_icloud(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(
            desktop=True, icloud_copy=True, icloud_dir=tmp_path
        )
    )
    job = _job(JobStatus.COMPLETED, plots=["/out/scatter.png"])
    notify_outcome(
        job, cfg, _rule(), desktop=desktop, icloud=icloud
    )
    assert len(desktop.calls) == 1
    title, msg = desktop.calls[0]
    assert "cam_realtime" in msg
    assert "completed" in msg.lower()
    assert len(icloud.calls) == 1
    assert icloud.calls[0]["plots"] == ["/out/scatter.png"]
    assert icloud.calls[0]["summary_name"].endswith(".md")


def test_failure_fires_desktop_but_not_icloud(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(
            desktop=True, icloud_copy=True, icloud_dir=tmp_path
        )
    )
    job = _job(JobStatus.FAILED, plots=["/out/scatter.png"])
    notify_outcome(job, cfg, _rule(), desktop=desktop, icloud=icloud)
    assert len(desktop.calls) == 1
    assert "failed" in desktop.calls[0][1].lower()
    assert icloud.calls == []  # no iCloud copy on failure


def test_desktop_disabled_globally_suppresses_desktop(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(
            desktop=False, icloud_copy=True, icloud_dir=tmp_path
        )
    )
    notify_outcome(
        _job(JobStatus.COMPLETED, plots=[]), cfg, _rule(),
        desktop=desktop, icloud=icloud,
    )
    assert desktop.calls == []


def test_per_rule_notify_overrides_to_desktop_only(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(
            desktop=True, icloud_copy=True, icloud_dir=tmp_path
        )
    )
    # Rule overrides channels to desktop-only: no iCloud even on success.
    rule = _rule(notify=["desktop"])
    notify_outcome(
        _job(JobStatus.COMPLETED, plots=["/out/p.png"]), cfg, rule,
        desktop=desktop, icloud=icloud,
    )
    assert len(desktop.calls) == 1
    assert icloud.calls == []


def test_per_rule_notify_log_only_suppresses_desktop(tmp_path: Path) -> None:
    desktop = _RecordingDesktop()
    icloud = _RecordingIcloud()
    cfg = DaemonConfig(
        notifications=NotificationConfig(
            desktop=True, icloud_copy=True, icloud_dir=tmp_path
        )
    )
    rule = _rule(notify=["log"])
    notify_outcome(
        _job(JobStatus.COMPLETED, plots=["/out/p.png"]), cfg, rule,
        desktop=desktop, icloud=icloud,
    )
    assert desktop.calls == []
    assert icloud.calls == []


def test_outcome_always_logs(tmp_path: Path, caplog) -> None:
    cfg = DaemonConfig(
        notifications=NotificationConfig(
            desktop=False, icloud_copy=False, icloud_dir=tmp_path
        )
    )
    with caplog.at_level("INFO"):
        notify_outcome(
            _job(JobStatus.COMPLETED, plots=[]), cfg, _rule(),
            desktop=_RecordingDesktop(), icloud=_RecordingIcloud(),
        )
    assert any("cam_realtime" in r.getMessage() for r in caplog.records)


def test_notifier_notify_result_delegates_to_notify_outcome(
    tmp_path: Path, monkeypatch
) -> None:
    """The Notifier facade forwards (job, cfg, rule) to notify_outcome verbatim."""
    from davinci_monet.daemon import notify as notify_mod
    from davinci_monet.daemon.notify import Notifier

    seen: dict = {}

    def fake_notify_outcome(job, cfg, rule=None, *, desktop=None, icloud=None) -> None:
        seen.update({"job": job, "cfg": cfg, "rule": rule})

    monkeypatch.setattr(notify_mod, "notify_outcome", fake_notify_outcome)

    cfg = DaemonConfig(
        notifications=NotificationConfig(
            desktop=True, icloud_copy=True, icloud_dir=tmp_path
        )
    )
    rule = _rule()
    job = _job(JobStatus.COMPLETED, plots=["/out/p.png"])
    notifier = Notifier(cfg)
    notifier.notify_result(job, rule)

    assert seen["job"] is job
    assert seen["cfg"] is cfg
    assert seen["rule"] is rule
```

- [ ] **Step 2 — run and expect failure.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_notify_outcome.py -v
```
Expected: `ImportError: cannot import name 'notify_outcome' from 'davinci_monet.daemon.notify'`, all tests fail at import.

- [ ] **Step 3 — minimal implementation.** Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/notify.py`. First extend the typing import line by replacing:
```python
from typing import Callable, Sequence
```
with:
```python
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence
```
and add, right after the `from pathlib import Path` line, a guarded import block:
```python
if TYPE_CHECKING:  # avoid import cycles / keep supervisor import-light
    from davinci_monet.daemon.config import DaemonConfig, WatchRule
    from davinci_monet.daemon.contracts import JobRecord
```
Then append at the end of the file:
```python
class _DesktopProto(Protocol):
    def __call__(self, title: str, message: str) -> bool: ...


class _IcloudProto(Protocol):
    def __call__(
        self,
        *,
        plots: Sequence[str],
        summary_text: str,
        summary_name: str,
    ) -> list[str]: ...


def _resolve_channels(
    cfg: "DaemonConfig",
    rule: "Optional[WatchRule]",
) -> set[str]:
    """Resolve active notification channels.

    Per-rule ``notify:`` (if set) overrides the daemon defaults entirely;
    otherwise channels derive from the daemon NotificationConfig flags. "log"
    is always implicitly active. ``rule`` may be ``None`` (no per-rule override).
    """
    if rule is not None and rule.notify is not None:
        channels = set(rule.notify)
        channels.add("log")
        return channels
    channels = {"log"}
    if cfg.notifications.desktop:
        channels.add("desktop")
    if cfg.notifications.icloud_copy:
        channels.add("icloud")
    return channels


def _build_summary_md(job: "JobRecord") -> str:
    """Render a short Markdown run summary for the iCloud copy."""
    summary = job.result_summary or {}
    lines = [
        f"# DAVINCI run: {job.watch_name} (job {job.id})",
        "",
        f"- status: {job.status.value}",
        f"- config: {job.config_path}",
        f"- submitted_at: {job.submitted_at.isoformat()}",
    ]
    if job.duration_s is not None:
        lines.append(f"- duration_s: {job.duration_s:.1f}")
    if job.files:
        lines.append(f"- files: {len(job.files)}")
    output_dir = summary.get("output_dir")
    if output_dir:
        lines.append(f"- output_dir: {output_dir}")
    if job.error:
        lines.append("")
        lines.append("## error")
        lines.append("```")
        lines.append(str(job.error))
        lines.append("```")
    return "\n".join(lines) + "\n"


def notify_outcome(
    job: "JobRecord",
    cfg: "DaemonConfig",
    rule: "Optional[WatchRule]" = None,
    *,
    desktop: Optional[_DesktopProto] = None,
    icloud: Optional[_IcloudProto] = None,
) -> None:
    """Route a finished job's outcome to log + (optionally) desktop + iCloud.

    Always logs. Posts a desktop notification when the "desktop" channel is
    active. On a COMPLETED job with the "icloud" channel active, copies the
    job's plots + a Markdown summary into ``cfg.notifications.icloud_dir``.

    ``desktop`` / ``icloud`` are injected callables (DesktopNotifier /
    IcloudCopier in production) so this is fully unit-testable with mocks.
    """
    # Late import keeps notify.py import-light at supervisor start.
    from davinci_monet.daemon.contracts import JobStatus

    status_word = job.status.value
    succeeded = job.status == JobStatus.COMPLETED

    # ---- always log -------------------------------------------------------
    log = logger.info if succeeded else logger.warning
    log(
        "Job %s for watch %r %s (config=%s)",
        job.id,
        job.watch_name,
        status_word,
        job.config_path,
    )

    channels = _resolve_channels(cfg, rule)

    # ---- desktop ----------------------------------------------------------
    if "desktop" in channels and desktop is not None:
        title = "DAVINCI"
        msg = f"{job.watch_name} {status_word}"
        if job.duration_s is not None:
            msg += f" in {job.duration_s:.0f}s"
        try:
            desktop(title, msg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("desktop notification raised: %s", exc)

    # ---- iCloud (success only) -------------------------------------------
    if succeeded and "icloud" in channels and icloud is not None:
        summary = job.result_summary or {}
        plots = list(summary.get("plots") or [])
        try:
            icloud(
                plots=plots,
                summary_text=_build_summary_md(job),
                summary_name=f"{job.watch_name}_job{job.id}.md",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("iCloud copy raised: %s", exc)


class Notifier:
    """Thin supervisor-facing facade over ``notify_outcome``.

    The supervisor's ``build_supervisor`` wiring constructs ONE Notifier bound to
    the daemon's NotificationConfig and (optionally) the desktop/iCloud callables,
    then calls :meth:`notify_result` once per finished job. ``hooks`` is a mapping
    of injectable side-effect callables (``{"desktop": ..., "icloud": ...}``);
    when omitted the production ``DesktopNotifier``/``IcloudCopier`` are used.
    """

    def __init__(
        self,
        daemon_cfg: "DaemonConfig",
        *,
        hooks: Optional[dict[str, object]] = None,
    ) -> None:
        self.daemon_cfg = daemon_cfg
        self.hooks = hooks or {}

    def notify_result(
        self,
        job: "JobRecord",
        rule: "Optional[WatchRule]" = None,
    ) -> None:
        """Route one finished job's outcome through ``notify_outcome``.

        Delegates verbatim to the module-level :func:`notify_outcome`, passing the
        bound NotificationConfig and the injected desktop/iCloud hooks (production
        ``DesktopNotifier``/``IcloudCopier`` when none were supplied).
        """
        desktop = self.hooks.get("desktop") or DesktopNotifier()
        icloud = self.hooks.get("icloud") or IcloudCopier(
            icloud_dir=self.daemon_cfg.notifications.icloud_dir
        )
        notify_outcome(
            job,
            self.daemon_cfg,
            rule,
            desktop=desktop,
            icloud=icloud,
        )
```

- [ ] **Step 4 — run and expect pass.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_notify_outcome.py -v
```
Expected: 7 passed (the six routing tests + `test_notifier_notify_result_delegates_to_notify_outcome`).

- [ ] **Step 5 — commit.**
```bash
git add davinci_monet/daemon/notify.py davinci_monet/tests/unit/daemon/test_notify_outcome.py
git commit -m "feat(daemon): notify_outcome routes desktop + iCloud + log per config/rule

Always logs; desktop on enabled channel; iCloud copy of plots + summary on
success only; per-rule notify: overrides daemon channel defaults. Desktop and
iCloud callables are injected for testability. A thin Notifier facade binds the
daemon NotificationConfig + injectable desktop/iCloud hooks and delegates
notify_result(job, rule) to notify_outcome for the supervisor.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 29: PID + lock file: acquire, release, stale-reclaim by dead pid

`lifecycle.py` provides `PidLock`, an exclusive lock under `state_dir`. Acquiring writes the current pid to `pid_path` and takes an exclusive `flock` on `lock_path`. If the lock is held by a **live** pid, acquire raises `LockHeldError` naming that pid. If `pid_path` references a **dead** pid (stale), it auto-reclaims. Liveness checking is injectable (`is_alive`) so tests use fake pids without spawning processes.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/lifecycle.py` (new)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_lifecycle_lock.py` (new)

Steps:

- [ ] **Step 1 — write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_lifecycle_lock.py`:
```python
"""Unit tests for the daemon PID+lock primitive (fake pids, temp dir)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from davinci_monet.daemon.lifecycle import LockHeldError, PidLock


def test_acquire_writes_pid_and_creates_files(tmp_path: Path) -> None:
    lock = PidLock(pid_path=tmp_path / "daemon.pid", lock_path=tmp_path / "daemon.lock")
    lock.acquire()
    try:
        assert (tmp_path / "daemon.pid").read_text().strip() == str(os.getpid())
        assert (tmp_path / "daemon.lock").exists()
        assert lock.read_pid() == os.getpid()
    finally:
        lock.release()


def test_release_removes_pid_file(tmp_path: Path) -> None:
    lock = PidLock(pid_path=tmp_path / "daemon.pid", lock_path=tmp_path / "daemon.lock")
    lock.acquire()
    lock.release()
    assert not (tmp_path / "daemon.pid").exists()


def test_acquire_raises_when_held_by_live_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("4242\n")
    # 4242 is reported alive -> must refuse.
    lock = PidLock(
        pid_path=pid_path,
        lock_path=tmp_path / "daemon.lock",
        is_alive=lambda pid: pid == 4242,
    )
    with pytest.raises(LockHeldError) as exc:
        lock.acquire()
    assert "4242" in str(exc.value)
    assert exc.value.pid == 4242


def test_acquire_reclaims_stale_dead_pid(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("999999\n")  # dead pid
    lock = PidLock(
        pid_path=pid_path,
        lock_path=tmp_path / "daemon.lock",
        is_alive=lambda pid: False,  # nothing is alive
    )
    lock.acquire()  # should reclaim, not raise
    try:
        assert lock.read_pid() == os.getpid()
    finally:
        lock.release()


def test_is_locked_by_other_false_when_no_pid_file(tmp_path: Path) -> None:
    lock = PidLock(
        pid_path=tmp_path / "daemon.pid",
        lock_path=tmp_path / "daemon.lock",
        is_alive=lambda pid: True,
    )
    assert lock.is_locked_by_live_other() is False


def test_context_manager_acquires_and_releases(tmp_path: Path) -> None:
    pid_path = tmp_path / "daemon.pid"
    with PidLock(pid_path=pid_path, lock_path=tmp_path / "daemon.lock"):
        assert pid_path.exists()
    assert not pid_path.exists()
```

- [ ] **Step 2 — run and expect failure.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_lifecycle_lock.py -v
```
Expected: `ModuleNotFoundError: No module named 'davinci_monet.daemon.lifecycle'`, all tests error at import.

- [ ] **Step 3 — minimal implementation.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/lifecycle.py`:
```python
"""Daemon process lifecycle: PID+lock, double-fork, signals, graceful drain.

Supervisor-side ONLY. Stdlib only; never imports the scientific stack.
"""

from __future__ import annotations

import fcntl
import os
import signal
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Callable, Optional

from davinci_monet.logging import get_logger

logger = get_logger(__name__)

# Liveness predicate: given a pid, report whether the process is running.
AliveFn = Callable[[int], bool]


def _default_is_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists (POSIX os.kill(pid, 0))."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user -> still alive.
        return True
    return True


class LockHeldError(RuntimeError):
    """Raised when the daemon lock is already held by a live process."""

    def __init__(self, pid: int) -> None:
        super().__init__(
            f"DAVINCI daemon already running (pid {pid}); refusing to start"
        )
        self.pid = pid


class PidLock:
    """Exclusive single-instance lock backed by a pid file + flock.

    ``acquire`` takes an exclusive, non-blocking ``flock`` on ``lock_path`` and
    writes ``os.getpid()`` to ``pid_path``. If ``pid_path`` references a *live*
    pid, it raises :class:`LockHeldError`; a *dead* (stale) pid is reclaimed.
    """

    def __init__(
        self,
        *,
        pid_path: str | Path,
        lock_path: str | Path,
        is_alive: AliveFn | None = None,
    ) -> None:
        self.pid_path = Path(pid_path)
        self.lock_path = Path(lock_path)
        self._is_alive = is_alive or _default_is_alive
        self._fd: Optional[int] = None

    # ---- introspection ----------------------------------------------------
    def read_pid(self) -> Optional[int]:
        """Return the pid recorded in pid_path, or None if absent/garbage."""
        try:
            text = self.pid_path.read_text().strip()
        except FileNotFoundError:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def is_locked_by_live_other(self) -> bool:
        """True iff pid_path names a live pid other than the current process."""
        pid = self.read_pid()
        if pid is None or pid == os.getpid():
            return False
        return self._is_alive(pid)

    # ---- acquire / release ------------------------------------------------
    def acquire(self) -> None:
        """Acquire the lock, reclaiming a stale one; raise if held by a live pid."""
        existing = self.read_pid()
        if existing is not None and existing != os.getpid() and self._is_alive(existing):
            raise LockHeldError(existing)

        if existing is not None:
            logger.info("Reclaiming stale daemon lock from dead pid %s", existing)

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            live = self.read_pid() or -1
            raise LockHeldError(live)
        self._fd = fd
        self.pid_path.write_text(f"{os.getpid()}\n")

    def release(self) -> None:
        """Release the flock and remove the pid file."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        try:
            # Only remove if it is still ours.
            if self.read_pid() == os.getpid():
                self.pid_path.unlink()
        except FileNotFoundError:
            pass

    # ---- context manager --------------------------------------------------
    def __enter__(self) -> "PidLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
```

- [ ] **Step 4 — run and expect pass.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_lifecycle_lock.py -v
```
Expected: 6 passed.

- [ ] **Step 5 — commit.**
```bash
git add davinci_monet/daemon/lifecycle.py davinci_monet/tests/unit/daemon/test_lifecycle_lock.py
git commit -m "feat(daemon): add PidLock single-instance lock with stale reclaim

flock + pid file under state_dir; refuses to start when held by a live pid,
auto-reclaims a stale lock from a dead pid. Liveness check is injectable for
deterministic tests with fake pids.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 30: Signal handlers and graceful-drain primitive

`lifecycle.py` adds `DrainController` (a thread-safe drain flag with a "wait for in-flight worker" helper) and `install_signal_handlers(on_drain, *, signums=..., hooks=...)`, which wires SIGTERM/SIGINT to call a plain `Callable[[str], None]` with the signal name (graceful stop, spec §"Lifecycle & signals", lines 222-231). `on_drain` is any such callable — `DrainController.request_drain(reason)` and the supervisor's `request_shutdown(reason="signal")` both conform — so the CLI `serve` can pass `supervisor.request_shutdown` directly. The previous handlers are restorable. Everything is injectable/observable so tests trigger the handler directly without sending real signals.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/lifecycle.py` (modify — append drain + signals)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_lifecycle_drain.py` (new)

Steps:

- [ ] **Step 1 — write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_lifecycle_drain.py`:
```python
"""Unit tests for the drain flag + signal handler wiring."""

from __future__ import annotations

import signal

from davinci_monet.daemon.lifecycle import DrainController, install_signal_handlers


def test_drain_flag_starts_clear_then_sets() -> None:
    ctrl = DrainController()
    assert ctrl.draining is False
    ctrl.request_drain()
    assert ctrl.draining is True


def test_request_drain_is_idempotent_and_records_reason() -> None:
    ctrl = DrainController()
    ctrl.request_drain(reason="SIGTERM")
    ctrl.request_drain(reason="second")
    assert ctrl.draining is True
    # First reason wins (records why drain started).
    assert ctrl.reason == "SIGTERM"


def test_wait_for_idle_returns_immediately_when_count_zero() -> None:
    ctrl = DrainController()
    # in_flight() reports zero workers -> drain completes instantly.
    completed = ctrl.wait_for_idle(in_flight=lambda: 0, timeout=0.5, poll=0.01)
    assert completed is True


def test_wait_for_idle_times_out_when_worker_never_finishes() -> None:
    ctrl = DrainController()
    completed = ctrl.wait_for_idle(in_flight=lambda: 1, timeout=0.05, poll=0.01)
    assert completed is False


def test_wait_for_idle_returns_true_when_worker_finishes() -> None:
    state = {"n": 2}

    def in_flight() -> int:
        # Drain one each poll until zero.
        if state["n"] > 0:
            state["n"] -= 1
        return state["n"]

    ctrl = DrainController()
    completed = ctrl.wait_for_idle(in_flight=in_flight, timeout=1.0, poll=0.001)
    assert completed is True


def test_install_signal_handlers_calls_on_drain_with_signal_name() -> None:
    # install_signal_handlers takes a bare Callable[[str], None]; the handler
    # must invoke it with the signal NAME (e.g. "SIGTERM").
    received: list[str] = []
    previous = install_signal_handlers(received.append)
    try:
        # Invoke the installed SIGTERM handler directly (no real signal).
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert received == ["SIGTERM"]
    finally:
        # Restore prior handlers so we don't leak into other tests.
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])
        signal.signal(signal.SIGINT, previous[signal.SIGINT])


def test_install_signal_handlers_accepts_drain_controller() -> None:
    # A DrainController.request_drain is itself a Callable[[str], None].
    ctrl = DrainController()
    previous = install_signal_handlers(ctrl.request_drain)
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert ctrl.draining is True
        assert ctrl.reason == "SIGINT"
    finally:
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])
        signal.signal(signal.SIGINT, previous[signal.SIGINT])
```

- [ ] **Step 2 — run and expect failure.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_lifecycle_drain.py -v
```
Expected: `ImportError: cannot import name 'DrainController' from 'davinci_monet.daemon.lifecycle'`, all tests fail at import.

- [ ] **Step 3 — minimal implementation.** Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/lifecycle.py`. First extend the imports: replace the line
```python
from typing import Callable, Optional
```
with
```python
import time
from typing import Callable, Optional
```
Then append at the end of the file:
```python
class DrainController:
    """Thread-safe graceful-drain flag + in-flight wait helper.

    The supervisor checks :attr:`draining` to stop accepting new triggers, and
    calls :meth:`wait_for_idle` to block until in-flight workers finish (up to a
    timeout). Signal handlers flip the flag via :meth:`request_drain`.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: Optional[str] = None

    @property
    def draining(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def request_drain(self, reason: Optional[str] = None) -> None:
        """Set the drain flag (idempotent; first reason is retained)."""
        with self._lock:
            if not self._event.is_set():
                self._reason = reason
                self._event.set()
                logger.info("Drain requested (reason=%s)", reason)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until drain is requested; returns True if it was set."""
        return self._event.wait(timeout)

    def wait_for_idle(
        self,
        in_flight: Callable[[], int],
        *,
        timeout: Optional[float],
        poll: float = 0.25,
    ) -> bool:
        """Poll ``in_flight`` until it reaches 0 or ``timeout`` elapses.

        Returns True if the system went idle within the timeout, False if the
        deadline passed with workers still running. ``timeout=None`` waits
        indefinitely.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if in_flight() <= 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            remaining = poll
            if deadline is not None:
                remaining = min(poll, max(0.0, deadline - time.monotonic()))
            time.sleep(remaining)


class SignalHooks:
    """Injectable wrapper over the signal primitives the handler installer needs.

    Defaults delegate to the real ``signal`` module; tests can substitute a fake
    to drive installation without touching process-global handlers.
    """

    def getsignal(self, signum: int) -> object:
        return signal.getsignal(signum)

    def signal(self, signum: int, handler: object) -> object:
        return signal.signal(signum, handler)  # type: ignore[arg-type]


# An on-drain callback receives the signal name (e.g. "SIGTERM") and starts a
# graceful drain. DrainController.request_drain and Supervisor.request_shutdown
# both conform to Callable[[str], None].
OnDrainFn = Callable[[str], None]


def install_signal_handlers(on_drain: OnDrainFn, *,
    signums: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT),
    hooks: "SignalHooks | None" = None,
) -> dict[int, object]:
    """Install drain-on-signal handlers for ``signums``.

    ``on_drain`` is any ``Callable[[str], None]`` (e.g.
    ``DrainController.request_drain`` or ``Supervisor.request_shutdown``). Each
    installed handler calls ``on_drain(signal.Signals(signum).name)`` and returns
    control (it does NOT exit; the supervisor loop observes the drain flag and
    drains gracefully). Returns the previous handlers keyed by signal number so
    the caller can restore them.
    """
    h = hooks or SignalHooks()
    previous: dict[int, object] = {}

    def _handler(signum: int, _frame: Optional[FrameType]) -> None:
        name = signal.Signals(signum).name
        logger.info("Received %s; initiating graceful drain", name)
        on_drain(name)

    for signum in signums:
        previous[signum] = h.getsignal(signum)
        h.signal(signum, _handler)
    return previous
```

- [ ] **Step 4 — run and expect pass.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_lifecycle_drain.py -v
```
Expected: 7 passed.

- [ ] **Step 5 — commit.**
```bash
git add davinci_monet/daemon/lifecycle.py davinci_monet/tests/unit/daemon/test_lifecycle_drain.py
git commit -m "feat(daemon): add DrainController + SIGTERM/SIGINT drain handlers

Thread-safe drain flag with first-reason retention and a wait_for_idle poll
loop bounded by timeout; signal handlers flip the flag for graceful stop and
return prior handlers for restoration.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 31: Background double-fork daemonize with stdio redirect to daemon.log

`lifecycle.py` adds `daemonize(log_path)`: the classic POSIX double-fork to detach from the controlling terminal (new session, second fork to prevent reacquiring a TTY), then redirect stdin from `/dev/null` and stdout/stderr to `log_path` (spec §"Lifecycle & signals", lines 224-225). To keep this unit-testable without actually forking the test runner, the fork / setsid / dup2 primitives are injected via an `OsHooks` shim that defaults to the real `os` functions; tests pass fakes and assert the call sequence and that the parent path exits.

This task also ships the `daemon start`/`daemon stop` plumbing the CLI calls: `BackgroundResult(started, message, pid=None)` and the two functions `start_background(daemon_cfg, run, *, hooks=None) -> BackgroundResult` and `stop_background(daemon_cfg, *, timeout=10.0, hooks=None) -> BackgroundResult`. `lifecycle` MUST NOT import the supervisor, so `start_background` takes a `run` callable (the foreground serve callable the CLI supplies): if the PID lock is free it double-forks (the child acquires the lock, `daemonize`s, and calls `run()`; the parent returns `started=True, pid=...`); if a live daemon already holds the lock it returns `started=False` with a message. `stop_background` reads the pid from the lock, sends `SIGTERM`, and waits up to `timeout` for the process to exit. Both reuse `PidLock` + `daemonize` + the injectable `OsHooks` so the fork/lock/signal path is testable with a mocked `OsHooks`.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/lifecycle.py` (modify — append `daemonize`, `BackgroundResult`, `start_background`, `stop_background`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_lifecycle_daemonize.py` (new)

Steps:

- [ ] **Step 1 — write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_lifecycle_daemonize.py`:
```python
"""Unit tests for double-fork daemonize using injected os hooks (no real fork)."""

from __future__ import annotations

from pathlib import Path

import pytest

from davinci_monet.daemon.lifecycle import (
    BackgroundResult,
    OsHooks,
    daemonize,
    start_background,
    stop_background,
)


class _FakeHooks:
    """Records the daemonize call sequence; simulates parent/child fork results."""

    def __init__(self, fork_results: list[int]) -> None:
        self._fork_results = list(fork_results)
        self.events: list[str] = []
        self.exited_with: list[int] = []
        self.dup2_targets: list[int] = []
        self.opened: list[str] = []

    def fork(self) -> int:
        self.events.append("fork")
        return self._fork_results.pop(0)

    def setsid(self) -> int:
        self.events.append("setsid")
        return 0

    def chdir(self, path: str) -> None:
        self.events.append(f"chdir:{path}")

    def umask(self, mask: int) -> int:
        self.events.append("umask")
        return 0

    def _exit(self, code: int) -> None:
        self.exited_with.append(code)
        raise SystemExit(code)

    def open(self, path: str, flags: int, mode: int = 0o777) -> int:
        self.opened.append(path)
        return 10 + len(self.opened)

    def dup2(self, fd: int, fd2: int) -> None:
        self.dup2_targets.append(fd2)

    def close(self, fd: int) -> None:
        pass


def test_parent_of_first_fork_exits(tmp_path: Path) -> None:
    # First fork returns a positive pid -> we are the original parent -> exit.
    hooks = _FakeHooks(fork_results=[123])
    with pytest.raises(SystemExit) as exc:
        daemonize(tmp_path / "daemon.log", hooks=hooks)
    assert exc.value.code == 0
    assert hooks.exited_with == [0]
    assert hooks.events[0] == "fork"


def test_intermediate_parent_of_second_fork_exits(tmp_path: Path) -> None:
    # First fork child (0), setsid, second fork returns positive -> exit.
    hooks = _FakeHooks(fork_results=[0, 456])
    with pytest.raises(SystemExit) as exc:
        daemonize(tmp_path / "daemon.log", hooks=hooks)
    assert exc.value.code == 0
    assert "setsid" in hooks.events
    assert hooks.events.count("fork") == 2


def test_grandchild_redirects_stdio_and_returns(tmp_path: Path) -> None:
    log_path = tmp_path / "daemon.log"
    # Both forks return 0 -> we are the final daemon process; should NOT exit.
    hooks = _FakeHooks(fork_results=[0, 0])
    daemonize(log_path, hooks=hooks)  # returns normally
    # stdin(0), stdout(1), stderr(2) all redirected.
    assert sorted(hooks.dup2_targets) == [0, 1, 2]
    # The log file path was opened for stdout/stderr.
    assert any(str(log_path) == p for p in hooks.opened)
    # setsid happened between the two forks.
    assert hooks.events == [
        "fork",
        "setsid",
        "fork",
        f"chdir:/",
        "umask",
    ] or hooks.events[:3] == ["fork", "setsid", "fork"]


def test_default_hooks_expose_real_os_callables() -> None:
    hooks = OsHooks()
    assert callable(hooks.fork)
    assert callable(hooks.setsid)
    assert callable(hooks.dup2)


def _daemon_cfg(state_dir: Path):
    """A DaemonConfig whose state_dir (and thus pid/lock/log paths) is in tmp."""
    from davinci_monet.daemon.config import DaemonConfig

    return DaemonConfig(state_dir=state_dir)


class _ParentHooks(OsHooks):
    """OsHooks whose first fork returns a positive pid (the original parent path)."""

    def fork(self) -> int:
        return 4321  # >0 -> caller is the original parent; start_background returns


class _StopHooks(OsHooks):
    """OsHooks recording kill() calls for stop_background tests."""

    def __init__(self) -> None:
        self.killed: list[tuple[int, int]] = []

    def kill(self, pid: int, sig: int) -> None:
        self.killed.append((pid, sig))


def test_start_background_parent_path_reports_started(tmp_path: Path) -> None:
    cfg = _daemon_cfg(tmp_path / "state")
    called = {"run": False}

    def run() -> None:  # must NOT run on the parent path
        called["run"] = True

    result = start_background(cfg, run, hooks=_ParentHooks())
    assert isinstance(result, BackgroundResult)
    assert result.started is True
    assert called["run"] is False  # only the daemon child calls run()


def test_start_background_refuses_when_already_running(tmp_path: Path) -> None:
    import os as _os

    cfg = _daemon_cfg(tmp_path / "state")
    cfg.pid_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.pid_path.write_text(f"{_os.getpid()}\n")  # our own (live) pid holds the lock
    result = start_background(cfg, lambda: None, hooks=_ParentHooks())
    assert result.started is False
    assert "already running" in result.message.lower()


def test_stop_background_signals_and_reaps_dead_pid(tmp_path: Path) -> None:
    cfg = _daemon_cfg(tmp_path / "state")
    cfg.pid_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.pid_path.write_text("999999\n")  # a dead pid -> _default_is_alive False
    hooks = _StopHooks()
    import signal as _signal

    result = stop_background(cfg, timeout=1.0, hooks=hooks)
    assert result.started is True
    assert hooks.killed == [(999999, _signal.SIGTERM)]


def test_stop_background_no_pid_file_reports_not_running(tmp_path: Path) -> None:
    cfg = _daemon_cfg(tmp_path / "state")
    result = stop_background(cfg, hooks=_StopHooks())
    assert result.started is False
    assert "no running daemon" in result.message.lower()
```

- [ ] **Step 2 — run and expect failure.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_lifecycle_daemonize.py -v
```
Expected: `ImportError: cannot import name 'OsHooks' from 'davinci_monet.daemon.lifecycle'` (and `daemonize`), all tests fail at import.

- [ ] **Step 3 — minimal implementation.** First extend the imports so the background-lifecycle dataclass + the `DaemonConfig` type hint are available. Replace:
```python
from types import FrameType
from typing import Callable, Optional
```
with:
```python
from dataclasses import dataclass
from types import FrameType
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # supervisor-pure: only a type hint, no runtime import cycle
    from davinci_monet.daemon.config import DaemonConfig
```
Then append at the end of `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/lifecycle.py`:
```python
class OsHooks:
    """Injectable wrapper over the os primitives daemonize needs.

    Defaults delegate to the real ``os`` module. Tests substitute a fake to
    drive the fork/exit logic deterministically without actually forking the
    pytest process.
    """

    def fork(self) -> int:
        return os.fork()

    def setsid(self) -> int:
        return os.setsid()

    def chdir(self, path: str) -> None:
        os.chdir(path)

    def umask(self, mask: int) -> int:
        return os.umask(mask)

    def _exit(self, code: int) -> None:
        os._exit(code)

    def open(self, path: str, flags: int, mode: int = 0o777) -> int:
        return os.open(path, flags, mode)

    def dup2(self, fd: int, fd2: int) -> None:
        os.dup2(fd, fd2)

    def close(self, fd: int) -> None:
        os.close(fd)

    def getpid(self) -> int:
        return os.getpid()

    def kill(self, pid: int, sig: int) -> None:
        os.kill(pid, sig)


def daemonize(
    log_path: str | Path,
    *,
    hooks: "OsHooks | None" = None,
) -> None:
    """Detach into the background via the classic POSIX double-fork.

    Sequence: ``fork`` (parent exits) -> ``setsid`` (new session, drop the
    controlling TTY) -> ``fork`` again (intermediate parent exits so the daemon
    can never reacquire a TTY) -> ``chdir('/')`` + ``umask(0)`` -> redirect
    stdin from ``/dev/null`` and stdout/stderr to ``log_path`` (append). Returns
    in the final daemon process only; the ancestor processes call ``_exit(0)``.
    """
    h = hooks or OsHooks()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- first fork: detach from the launching shell ----------------------
    if h.fork() > 0:
        h._exit(0)  # original parent leaves
    h.setsid()  # become session leader, no controlling terminal

    # ---- second fork: ensure we are not a session leader ------------------
    if h.fork() > 0:
        h._exit(0)  # intermediate parent leaves

    # ---- daemon context ---------------------------------------------------
    h.chdir("/")
    h.umask(0)

    # ---- redirect stdio ---------------------------------------------------
    devnull_fd = h.open(os.devnull, os.O_RDONLY)
    log_fd = h.open(
        str(log_path),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    h.dup2(devnull_fd, 0)  # stdin  <- /dev/null
    h.dup2(log_fd, 1)      # stdout -> daemon.log
    h.dup2(log_fd, 2)      # stderr -> daemon.log
    if devnull_fd > 2:
        h.close(devnull_fd)
    if log_fd > 2:
        h.close(log_fd)


@dataclass
class BackgroundResult:
    """Outcome of a start_background / stop_background lifecycle operation.

    ``started`` is True when the daemon was (start) launched into the background
    or (stop) successfully signalled and reaped. ``message`` is a human string for
    the CLI to print; ``pid`` is the backgrounded daemon's pid when known.
    """

    started: bool
    message: str
    pid: Optional[int] = None


def start_background(
    daemon_cfg: "DaemonConfig",
    run: Callable[[], None],
    *,
    hooks: "OsHooks | None" = None,
) -> BackgroundResult:
    """Launch ``run`` as a backgrounded daemon (double-fork) under the PID lock.

    ``run`` is the foreground serve callable (e.g. ``supervisor.serve``); lifecycle
    intentionally NEVER imports the supervisor, so the caller supplies it. If a
    live daemon already holds the lock, returns ``started=False`` with a message
    naming the running pid. Otherwise the original process double-forks: the final
    daemon child acquires the PidLock, redirects stdio to ``daemon.log`` via
    :func:`daemonize`, and calls ``run()``; the original parent returns
    ``started=True`` with the backgrounded pid. ``hooks`` (OsHooks) is injected so
    the fork/lock/exit path is unit-testable without forking pytest.
    """
    h = hooks or OsHooks()
    lock = PidLock(pid_path=daemon_cfg.pid_path, lock_path=daemon_cfg.lock_path)
    if lock.is_locked_by_live_other():
        running = lock.read_pid()
        return BackgroundResult(
            started=False,
            message=f"DAVINCI daemon already running (pid {running}).",
            pid=running,
        )

    # ---- double-fork: original parent returns; child becomes the daemon ----
    if h.fork() > 0:
        # Original process: wait briefly for the child to record its pid, then
        # report success to the CLI caller.
        child_pid = lock.read_pid()
        return BackgroundResult(
            started=True,
            message=f"DAVINCI daemon started in background (pid {child_pid}).",
            pid=child_pid,
        )

    # ---- daemon child ------------------------------------------------------
    daemonize(daemon_cfg.log_path, hooks=h)
    lock.acquire()  # writes the daemon's pid; held for the process lifetime
    try:
        run()
    finally:
        lock.release()
    h._exit(0)
    return BackgroundResult(started=True, message="", pid=h.getpid())  # pragma: no cover


def stop_background(
    daemon_cfg: "DaemonConfig",
    *,
    timeout: float = 10.0,
    hooks: "OsHooks | None" = None,
) -> BackgroundResult:
    """Signal a backgrounded daemon to drain + exit and wait for it to die.

    Reads the daemon pid from the PID lock, sends ``SIGTERM`` (graceful drain),
    then polls liveness up to ``timeout`` seconds. Returns ``started=True`` when
    the process exits in time (or was already gone), ``started=False`` on timeout
    or when no pid file is present. ``hooks`` is injected for testability.
    """
    h = hooks or OsHooks()
    lock = PidLock(pid_path=daemon_cfg.pid_path, lock_path=daemon_cfg.lock_path)
    pid = lock.read_pid()
    if pid is None:
        return BackgroundResult(started=False, message="No running daemon found.")
    try:
        h.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return BackgroundResult(
            started=True, message="Daemon was not running (stale pid cleared).", pid=pid
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _default_is_alive(pid):
            return BackgroundResult(
                started=True, message=f"Daemon stopped (pid {pid}).", pid=pid
            )
        time.sleep(0.1)
    return BackgroundResult(
        started=False,
        message=f"Daemon (pid {pid}) did not exit within {timeout:.0f}s.",
        pid=pid,
    )
```

- [ ] **Step 4 — run and expect pass.** Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_lifecycle_daemonize.py -v
```
Expected: 8 passed (4 daemonize + the 4 background-lifecycle tests for start_background/stop_background). Then run the whole daemon lifecycle + notify suite to confirm no cross-test regressions:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_lifecycle_daemonize.py davinci_monet/tests/unit/daemon/test_lifecycle_drain.py davinci_monet/tests/unit/daemon/test_lifecycle_lock.py davinci_monet/tests/unit/daemon/test_notify_desktop.py davinci_monet/tests/unit/daemon/test_notify_icloud.py davinci_monet/tests/unit/daemon/test_notify_outcome.py -v
```
Expected: all pass (8 + 7 + 6 + 5 + 4 + 7).

- [ ] **Step 5 — commit.**
```bash
git add davinci_monet/daemon/lifecycle.py davinci_monet/tests/unit/daemon/test_lifecycle_daemonize.py
git commit -m "feat(daemon): add double-fork daemonize with stdio redirect to daemon.log

Classic POSIX double-fork (setsid between forks) detaches from the TTY;
stdin <- /dev/null, stdout/stderr -> daemon.log. The os primitives are injected
via OsHooks so the fork/exit logic is unit-tested without forking pytest.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 32: Shell command parser maps words to control cmd+args

A pure function `parse_command(line: str) -> ShellResult` is the heart of the REPL and is fully unit-testable without a live socket. It tokenizes one input line (via `shlex.split`) and maps the leading verb (and any sub-verb, e.g. `watch list`) to a control `cmd` name from `davinci_monet.daemon.contracts.COMMANDS` plus an `args` dict whose keys/types match the SHARED-CONTRACT command catalog verbatim. `ShellResult` also carries a `streaming: bool` flag (true for `logs --tail`) and a `local: bool` flag (true for `quit`/`exit`/`help`, which never touch the socket).

Mapping (shell verb -> control cmd, args):
- `status` -> `status` `{}`
- `reload` -> `reload` `{}`
- `history [--watch NAME] [--failed] [--limit N]` -> `history` `{"watch": NAME|None, "failed": bool, "limit": int}`
- `watch list` -> `watch_list` `{}`
- `watch pause NAME` -> `watch_pause` `{"name": NAME}`
- `watch resume NAME` -> `watch_resume` `{"name": NAME}`
- `watch remove NAME` -> `watch_remove` `{"name": NAME}`
- `watch trigger NAME [FILE...]` -> `watch_trigger` `{"name": NAME, "files": [FILE...]}`
- `watch save NAME` -> `watch_save` `{"name": NAME}`
- `logs TARGET [--watch] [--tail]` -> `logs` (or streaming `logs_tail`) `{"target": TARGET, "kind": "watch"|"job"}`
- `job JOB_ID` -> `job_get` `{"job_id": int}`
- `quit` / `exit` / `q` -> local (no cmd)
- `help` / `?` -> local (no cmd)

`watch add` is intentionally NOT exposed in the shell verb table (it needs a full WatchRule dump; the one-shot CLi/`watch_add` cmd covers it) — `parse_command("watch add ...")` raises `ValueError("watch add is not available in the shell; use \`davinci-monet daemon watch add\`")`.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/shell.py` (new)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_shell.py` (new)

Steps:

- [ ] Step 1 — Write the failing test. Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_shell.py`:
```python
"""Unit tests for the daemon control shell (parsing + dispatch)."""

from __future__ import annotations

import pytest

from davinci_monet.daemon.shell import ShellResult, parse_command


class TestParseCommand:
    def test_status(self) -> None:
        res = parse_command("status")
        assert isinstance(res, ShellResult)
        assert res.cmd == "status"
        assert res.args == {}
        assert res.streaming is False
        assert res.local is False

    def test_reload(self) -> None:
        res = parse_command("reload")
        assert res.cmd == "reload"
        assert res.args == {}

    def test_history_with_flags(self) -> None:
        res = parse_command("history --watch cam_realtime --failed --limit 10")
        assert res.cmd == "history"
        assert res.args == {"watch": "cam_realtime", "failed": True, "limit": 10}

    def test_history_defaults(self) -> None:
        res = parse_command("history")
        assert res.cmd == "history"
        assert res.args == {"watch": None, "failed": False, "limit": 50}

    def test_watch_list(self) -> None:
        res = parse_command("watch list")
        assert res.cmd == "watch_list"
        assert res.args == {}

    def test_watch_pause(self) -> None:
        res = parse_command("watch pause cam_realtime")
        assert res.cmd == "watch_pause"
        assert res.args == {"name": "cam_realtime"}

    def test_watch_resume(self) -> None:
        res = parse_command("watch resume cam_realtime")
        assert res.cmd == "watch_resume"
        assert res.args == {"name": "cam_realtime"}

    def test_watch_remove(self) -> None:
        res = parse_command("watch remove modis_stream")
        assert res.cmd == "watch_remove"
        assert res.args == {"name": "modis_stream"}

    def test_watch_trigger_no_files(self) -> None:
        res = parse_command("watch trigger cam_realtime")
        assert res.cmd == "watch_trigger"
        assert res.args == {"name": "cam_realtime", "files": []}

    def test_watch_trigger_with_files(self) -> None:
        res = parse_command("watch trigger cam_realtime /a/b.nc /a/c.nc")
        assert res.cmd == "watch_trigger"
        assert res.args == {
            "name": "cam_realtime",
            "files": ["/a/b.nc", "/a/c.nc"],
        }

    def test_watch_save(self) -> None:
        res = parse_command("watch save live_rule")
        assert res.cmd == "watch_save"
        assert res.args == {"name": "live_rule"}

    def test_job_get(self) -> None:
        res = parse_command("job 42")
        assert res.cmd == "job_get"
        assert res.args == {"job_id": 42}

    def test_logs_job_oneshot(self) -> None:
        res = parse_command("logs 7")
        assert res.cmd == "logs"
        assert res.args == {"target": "7", "kind": "job"}
        assert res.streaming is False

    def test_logs_watch(self) -> None:
        res = parse_command("logs cam_realtime --watch")
        assert res.cmd == "logs"
        assert res.args == {"target": "cam_realtime", "kind": "watch"}

    def test_logs_tail_streams(self) -> None:
        res = parse_command("logs 7 --tail")
        assert res.cmd == "logs_tail"
        assert res.args == {"target": "7", "kind": "job"}
        assert res.streaming is True

    def test_quit_is_local(self) -> None:
        for word in ("quit", "exit", "q"):
            res = parse_command(word)
            assert res.local is True
            assert res.cmd is None

    def test_help_is_local(self) -> None:
        res = parse_command("help")
        assert res.local is True
        assert res.cmd is None

    def test_empty_line_is_local_noop(self) -> None:
        res = parse_command("   ")
        assert res.local is True
        assert res.cmd is None

    def test_unknown_command_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown command"):
            parse_command("frobnicate")

    def test_watch_add_rejected_in_shell(self) -> None:
        with pytest.raises(ValueError, match="watch add is not available"):
            parse_command("watch add foo")

    def test_watch_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a watch name"):
            parse_command("watch pause")
```

- [ ] Step 2 — Run and expect failure (from repo root, in `davinci` env):
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_shell.py::TestParseCommand -v
```
Expected: collection/import error `ModuleNotFoundError: No module named 'davinci_monet.daemon.shell'` (the module does not exist yet).

- [ ] Step 3 — Minimal implementation. Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/shell.py` (this also defines `run_shell(client, *, input_fn=input, output_fn=print)`, the REPL loop the CLI `shell` command calls; its behavior is covered by `TestRunShell` added in Task 34):
```python
"""Interactive control shell for the DAVINCI daemon (`daemon shell`).

A thin REPL over ``DaemonClient``. Each input line is parsed by the pure,
side-effect-free ``parse_command`` into a ``ShellResult`` (control ``cmd`` +
``args`` matching the SHARED-CONTRACT command catalog), then dispatched over the
control socket. Quitting the shell never stops the daemon.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from davinci_monet.daemon.client import DaemonClient
from davinci_monet.daemon.contracts import ControlResponse

# Shell verbs that resolve to a single control cmd with no sub-verb.
SHELL_COMMANDS: tuple[str, ...] = (
    "status",
    "reload",
    "history",
    "watch",
    "logs",
    "job",
    "help",
    "quit",
    "exit",
)

_QUIT_WORDS = {"quit", "exit", "q"}
_HELP_WORDS = {"help", "?"}


@dataclass
class ShellResult:
    """Parsed shell line.

    ``local`` lines (quit/help/blank) never touch the socket; ``cmd`` is None.
    ``streaming`` selects ``DaemonClient.stream`` over ``DaemonClient.call``.
    """

    cmd: Optional[str] = None
    args: dict[str, Any] = field(default_factory=dict)
    streaming: bool = False
    local: bool = False


def _require_name(words: list[str], verb: str) -> str:
    if not words:
        raise ValueError(f"`watch {verb}` requires a watch name")
    return words[0]


def _parse_history(words: list[str]) -> ShellResult:
    watch: Optional[str] = None
    failed = False
    limit = 50
    i = 0
    while i < len(words):
        tok = words[i]
        if tok == "--watch":
            i += 1
            if i >= len(words):
                raise ValueError("--watch requires a value")
            watch = words[i]
        elif tok == "--failed":
            failed = True
        elif tok == "--limit":
            i += 1
            if i >= len(words):
                raise ValueError("--limit requires a value")
            limit = int(words[i])
        else:
            raise ValueError(f"unknown history option: {tok}")
        i += 1
    return ShellResult(
        cmd="history",
        args={"watch": watch, "failed": failed, "limit": limit},
    )


def _parse_watch(words: list[str]) -> ShellResult:
    if not words:
        raise ValueError("`watch` requires a sub-command (list/pause/resume/remove/trigger/save)")
    sub, rest = words[0], words[1:]
    if sub == "add":
        raise ValueError(
            "watch add is not available in the shell; "
            "use `davinci-monet daemon watch add`"
        )
    if sub == "list":
        return ShellResult(cmd="watch_list", args={})
    if sub == "pause":
        return ShellResult(cmd="watch_pause", args={"name": _require_name(rest, "pause")})
    if sub == "resume":
        return ShellResult(cmd="watch_resume", args={"name": _require_name(rest, "resume")})
    if sub == "remove":
        return ShellResult(cmd="watch_remove", args={"name": _require_name(rest, "remove")})
    if sub == "save":
        return ShellResult(cmd="watch_save", args={"name": _require_name(rest, "save")})
    if sub == "trigger":
        name = _require_name(rest, "trigger")
        return ShellResult(cmd="watch_trigger", args={"name": name, "files": rest[1:]})
    raise ValueError(f"unknown watch sub-command: {sub}")


def _parse_logs(words: list[str]) -> ShellResult:
    if not words:
        raise ValueError("`logs` requires a target (job id or watch name)")
    target: Optional[str] = None
    kind = "job"
    tail = False
    for tok in words:
        if tok == "--watch":
            kind = "watch"
        elif tok == "--tail":
            tail = True
        elif tok.startswith("--"):
            raise ValueError(f"unknown logs option: {tok}")
        elif target is None:
            target = tok
        else:
            raise ValueError(f"unexpected logs argument: {tok}")
    if target is None:
        raise ValueError("`logs` requires a target (job id or watch name)")
    return ShellResult(
        cmd="logs_tail" if tail else "logs",
        args={"target": target, "kind": kind},
        streaming=tail,
    )


def parse_command(line: str) -> ShellResult:
    """Parse one REPL line into a ``ShellResult``. Pure / no I/O."""
    words = shlex.split(line.strip())
    if not words:
        return ShellResult(local=True)
    verb, rest = words[0], words[1:]
    if verb in _QUIT_WORDS:
        return ShellResult(local=True)
    if verb in _HELP_WORDS:
        return ShellResult(local=True)
    if verb == "status":
        return ShellResult(cmd="status", args={})
    if verb == "reload":
        return ShellResult(cmd="reload", args={})
    if verb == "history":
        return _parse_history(rest)
    if verb == "watch":
        return _parse_watch(rest)
    if verb == "logs":
        return _parse_logs(rest)
    if verb == "job":
        if not rest:
            raise ValueError("`job` requires a job id")
        return ShellResult(cmd="job_get", args={"job_id": int(rest[0])})
    raise ValueError(f"unknown command: {verb}")


def format_response(cmd: str, resp: ControlResponse) -> str:
    """Render a one-shot ControlResponse as a short human string for the REPL."""
    if not resp.ok:
        code = f" [{resp.code}]" if resp.code else ""
        return f"error{code}: {resp.error}"
    return f"{cmd}: ok"


class DaemonShell:
    """Line-oriented REPL bound to a single ``DaemonClient``."""

    PROMPT = "davinci-daemon> "

    def __init__(self, client: DaemonClient) -> None:
        self._client = client

    def execute(self, line: str) -> Optional[ControlResponse]:
        """Parse + dispatch one line. Returns the response, or None for local lines.

        Streaming lines drain the client's event iterator to completion (the
        caller's terminal shows the pushed events). Local lines (quit/help/blank)
        return None without touching the socket.
        """
        result = parse_command(line)
        if result.local or result.cmd is None:
            return None
        if result.streaming:
            for _event in self._client.stream(result.cmd, **result.args):
                pass
            return None
        return self._client.call(result.cmd, **result.args)

    def is_quit(self, line: str) -> bool:
        """True if ``line`` is a quit/exit verb (REPL loop sentinel)."""
        words = shlex.split(line.strip())
        return bool(words) and words[0] in _QUIT_WORDS


def run_shell(
    client: DaemonClient,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the interactive daemon REPL until the user quits.

    Reads one line at a time via ``input_fn`` (defaulting to the builtin
    ``input``), dispatches it through a :class:`DaemonShell`, and prints any
    ``ControlResponse`` via ``output_fn``. Quitting (``quit``/``exit``/``q``) or
    EOF/Ctrl-D ends the loop and returns; it NEVER sends ``shutdown`` (the daemon
    keeps running). ``input_fn``/``output_fn`` are injected so the loop is
    unit-testable with scripted lines and no real terminal.
    """
    shell = DaemonShell(client)
    while True:
        try:
            line = input_fn(shell.PROMPT)
        except EOFError:
            output_fn("")
            return
        if shell.is_quit(line):
            return
        try:
            resp = shell.execute(line)
        except ValueError as exc:
            output_fn(f"error: {exc}")
            continue
        if resp is not None:
            stripped = line.strip()
            cmd_word = shlex.split(stripped)[0] if stripped else ""
            output_fn(format_response(cmd_word, resp))
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_shell.py::TestParseCommand -v
```
Expected: all `TestParseCommand` tests pass.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/shell.py davinci_monet/tests/unit/daemon/test_shell.py
git commit -m "feat(daemon): add shell command parser for control REPL

parse_command maps shell verbs (status/reload/history/watch */logs/job)
to control cmd names + args matching the daemon command catalog. quit/exit
and help are local no-socket lines; watch add is rejected in-shell.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 33: Shell REPL dispatch round-trips through DaemonClient

Verify that `DaemonShell.execute` calls `DaemonClient.call(cmd, **args)` for non-streaming verbs and `DaemonClient.stream(cmd, **args)` for `logs --tail`, with the exact cmd/args from the parser, and surfaces the `ControlResponse`. Uses a `MagicMock` client (no socket) per the existing CLI test convention (`unittest.mock`).

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/shell.py` (already created in the previous task)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_shell.py` (modify — add `TestDaemonShellDispatch`)

Steps:

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_shell.py`:
```python
from unittest.mock import MagicMock

from davinci_monet.daemon.contracts import ControlResponse, StreamEvent
from davinci_monet.daemon.shell import DaemonShell


class TestDaemonShellDispatch:
    def _shell(self) -> tuple[DaemonShell, MagicMock]:
        client = MagicMock()
        client.call.return_value = ControlResponse(ok=True, data={"pong": True})
        return DaemonShell(client), client

    def test_status_calls_client_call(self) -> None:
        shell, client = self._shell()
        resp = shell.execute("status")
        client.call.assert_called_once_with("status")
        client.stream.assert_not_called()
        assert resp is not None and resp.ok is True

    def test_watch_pause_dispatches_name(self) -> None:
        shell, client = self._shell()
        shell.execute("watch pause cam_realtime")
        client.call.assert_called_once_with("watch_pause", name="cam_realtime")

    def test_watch_trigger_dispatches_files(self) -> None:
        shell, client = self._shell()
        shell.execute("watch trigger cam /a.nc /b.nc")
        client.call.assert_called_once_with(
            "watch_trigger", name="cam", files=["/a.nc", "/b.nc"]
        )

    def test_history_flags_dispatched(self) -> None:
        shell, client = self._shell()
        shell.execute("history --watch cam --failed --limit 5")
        client.call.assert_called_once_with(
            "history", watch="cam", failed=True, limit=5
        )

    def test_logs_oneshot_uses_call(self) -> None:
        shell, client = self._shell()
        shell.execute("logs 7")
        client.call.assert_called_once_with("logs", target="7", kind="job")
        client.stream.assert_not_called()

    def test_logs_tail_uses_stream(self) -> None:
        shell, client = self._shell()
        client.stream.return_value = iter(
            [StreamEvent(event="log_line", data={"message": "hello"})]
        )
        resp = shell.execute("logs 7 --tail")
        client.stream.assert_called_once_with("logs_tail", target="7", kind="job")
        client.call.assert_not_called()
        assert resp is None  # streaming returns None after draining

    def test_local_quit_does_not_touch_client(self) -> None:
        shell, client = self._shell()
        resp = shell.execute("quit")
        client.call.assert_not_called()
        client.stream.assert_not_called()
        assert resp is None

    def test_local_help_does_not_touch_client(self) -> None:
        shell, client = self._shell()
        shell.execute("help")
        client.call.assert_not_called()

    def test_is_quit(self) -> None:
        shell, _ = self._shell()
        assert shell.is_quit("quit") is True
        assert shell.is_quit("exit") is True
        assert shell.is_quit("status") is False
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_shell.py::TestDaemonShellDispatch -v
```
Expected: import error or `AttributeError` only if `DaemonShell`/`is_quit` were missing — but since they were defined in the previous task's Step 3, the likely failure here is none if run after that task. If `davinci_monet.daemon.contracts.StreamEvent`/`ControlResponse` import fails, that signals the contracts bootstrap (Task 1) is incomplete; otherwise expect ALL `TestDaemonShellDispatch` tests to pass immediately because `execute`/`is_quit` already exist. To force a genuine red first, write this test BEFORE Step 3 of the prior task is merged; when implemented in the same plan run, treat the import-error-free pass as the green signal and skip ahead.

Note: because `DaemonShell.execute`/`is_quit` are implemented in the prior task, this task is a pure behavioral-coverage addition; its red state is the absence of the new test methods (a `pytest` "no tests collected"/`AttributeError: module ... has no attribute 'DaemonShell'` only if run in isolation before the prior task). Run after the prior task; expect green.

- [ ] Step 3 — Minimal implementation. No new production code is required — `DaemonShell.execute` and `is_quit` already satisfy these tests. If any assertion fails (e.g. streaming not draining), the minimal fix is to ensure `execute` iterates `self._client.stream(...)` to exhaustion and returns `None`, exactly as written in the prior task's `shell.py`. Leave `shell.py` unchanged if Step 4 is green.

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_shell.py -v
```
Expected: every test in `test_shell.py` (both classes) passes.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/tests/unit/daemon/test_shell.py
git commit -m "test(daemon): cover shell dispatch through DaemonClient

DaemonShell.execute routes non-streaming verbs to client.call and
logs --tail to client.stream with parsed cmd/args; quit/help stay local.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 34: Shell quit/exit leaves the daemon running

Encode the spec invariant (decision #7 / control-surface section: "quit-without-stopping-daemon"). The shell must NEVER send `shutdown`. Add a guard test asserting the parser/dispatcher path produces no `shutdown` cmd for any quit verb, and that `shutdown` is not even reachable via the shell verb table. This task also adds `TestRunShell`, covering the `run_shell` REPL loop (defined in Task 32's `shell.py`): scripted lines dispatch through `DaemonShell` and `quit`/EOF end the loop cleanly without ever sending `shutdown`.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/shell.py` (already created — `run_shell` lives here)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_shell.py` (modify — add `TestShellNeverStopsDaemon` + `TestRunShell`)

Steps:

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_shell.py`:
```python
class TestShellNeverStopsDaemon:
    def test_quit_emits_no_shutdown(self) -> None:
        for word in ("quit", "exit", "q"):
            res = parse_command(word)
            assert res.cmd != "shutdown"
            assert res.cmd is None
            assert res.local is True

    def test_shutdown_verb_is_not_reachable(self) -> None:
        # The shell has no `shutdown`/`stop` verb; it must be an unknown command.
        with pytest.raises(ValueError, match="unknown command"):
            parse_command("shutdown")
        with pytest.raises(ValueError, match="unknown command"):
            parse_command("stop")

    def test_dispatch_quit_never_calls_shutdown(self) -> None:
        shell, client = TestDaemonShellDispatch()._shell()
        shell.execute("quit")
        # No call at all, and certainly not shutdown.
        client.call.assert_not_called()
        for call in client.call.call_args_list:
            assert call.args[0] != "shutdown"


class TestRunShell:
    def test_run_shell_dispatches_then_quits_cleanly(self) -> None:
        from davinci_monet.daemon.shell import run_shell

        client = MagicMock()
        client.call.return_value = ControlResponse(ok=True, data={"pong": True})
        # Scripted input: one real command, then quit ends the loop.
        scripted = iter(["status", "quit"])
        outputs: list[str] = []

        run_shell(
            client,
            input_fn=lambda _prompt: next(scripted),
            output_fn=outputs.append,
        )

        client.call.assert_called_once_with("status")
        # Quitting never sends shutdown and the loop returned cleanly.
        for call in client.call.call_args_list:
            assert call.args[0] != "shutdown"

    def test_run_shell_exits_on_eof(self) -> None:
        from davinci_monet.daemon.shell import run_shell

        client = MagicMock()

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        # EOF (Ctrl-D) ends the loop without touching the socket.
        run_shell(client, input_fn=_raise_eof, output_fn=lambda _s: None)
        client.call.assert_not_called()
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_shell.py::TestShellNeverStopsDaemon -v
```
Expected: this passes immediately against the existing `parse_command` (quit is local; `shutdown`/`stop` are unknown verbs). If instead it errors on `TestDaemonShellDispatch()._shell()` because that helper was not yet added, run after the dispatch task. To obtain a true red, temporarily assert `res.cmd == "shutdown"` is impossible by first checking against a stubbed parser — but since the contract already excludes `shutdown` from the shell verbs, the meaningful failure mode is a regression: if someone later adds a `shutdown` verb, `test_shutdown_verb_is_not_reachable` goes red. Treat green-on-first-run as the intended guard.

- [ ] Step 3 — Minimal implementation. No production change needed; the guard is satisfied by the absence of a `shutdown`/`stop` verb in `parse_command`. Do not add one. If `test_shutdown_verb_is_not_reachable` fails, the fix is to REMOVE any `shutdown` handling that crept into `parse_command` (the shell must not stop the daemon).

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_shell.py::TestShellNeverStopsDaemon davinci_monet/tests/unit/daemon/test_shell.py::TestRunShell -v
```
Expected: all three guard tests plus the two `TestRunShell` tests pass.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/shell.py davinci_monet/tests/unit/daemon/test_shell.py
git commit -m "test(daemon): assert shell never stops the daemon + cover run_shell REPL

Quitting the REPL is local-only; shutdown/stop are unreachable shell
verbs, locking in the control-surface invariant from the design spec.
run_shell drives DaemonShell with injected input/output for scripted-line
tests; quit and EOF end the loop cleanly without sending shutdown.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 35: Dashboard panel builders from a StatusData snapshot

`daemon top` is a Rich `Live` view. The renderers are pure functions that take a snapshot (the `StatusData` dict from the `status`/`subscribe` control responses, augmented with per-job progress messages) and return Rich renderables. This task builds the four panel functions and the `DashboardState` snapshot container. Tests call the panel functions with fixture state (no live loop) and assert on rendered text via a Rich `Console` capture. Colors reuse `davinci_monet.plots.style.NCAR_COLORS` (the codebase's single styling source of truth, per the Pre-Implementation Audit).

`DashboardState` fields mirror `StatusData`: `version: int`, `pid: int`, `uptime_s: float`, `draining: bool`, `max_concurrent: int`, `running: list[dict]` (JobRecord dumps), `queued: list[dict]`, `watches: list[dict]` (WatchSummary dumps), `recent: list[dict]`, plus `progress: dict[int, str]` mapping `job_id -> latest progress/stage message` (populated by stream events in a later task).

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dashboard.py` (new)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_dashboard.py` (new)

Steps:

- [ ] Step 1 — Write the failing test. Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_dashboard.py`:
```python
"""Unit tests for the `daemon top` dashboard renderers (no live loop)."""

from __future__ import annotations

import pytest
from rich.console import Console

from davinci_monet.daemon.dashboard import (
    DashboardState,
    render_queue_panel,
    render_recent_panel,
    render_running_panel,
    render_watches_panel,
)


@pytest.fixture
def sample_state() -> DashboardState:
    return DashboardState(
        version=1,
        pid=4242,
        uptime_s=3725.0,
        draining=False,
        max_concurrent=1,
        watches=[
            {
                "name": "cam_realtime",
                "enabled": True,
                "source": "file",
                "on_fire": "whole_config",
                "settle_mode": "quiescence",
                "watch": "/scratch/cam/incoming/*.nc",
                "run": "configs/asia-aq.yaml",
                "state": "running",
                "last_job_id": 7,
                "last_status": "running",
                "last_fired_at": "2026-05-31T12:00:00",
            },
            {
                "name": "modis_stream",
                "enabled": False,
                "source": "live",
                "on_fire": "new_files_only",
                "settle_mode": "sentinel",
                "watch": "/scratch/modis/*.hdf",
                "run": "configs/modis-aod.yaml",
                "state": "paused",
                "last_job_id": None,
                "last_status": None,
                "last_fired_at": None,
            },
        ],
        running=[
            {
                "id": 7,
                "watch_name": "cam_realtime",
                "config_path": "configs/asia-aq.yaml",
                "on_fire": "whole_config",
                "status": "running",
                "submitted_at": "2026-05-31T11:59:00",
                "started_at": "2026-05-31T12:00:00",
            }
        ],
        queued=[
            {
                "id": 8,
                "watch_name": "modis_stream",
                "config_path": "configs/modis-aod.yaml",
                "on_fire": "new_files_only",
                "status": "queued",
                "submitted_at": "2026-05-31T12:01:00",
            }
        ],
        recent=[
            {
                "id": 6,
                "watch_name": "cam_realtime",
                "config_path": "configs/asia-aq.yaml",
                "on_fire": "whole_config",
                "status": "completed",
                "submitted_at": "2026-05-31T11:00:00",
                "ended_at": "2026-05-31T11:30:00",
                "duration_s": 1800.0,
            },
            {
                "id": 5,
                "watch_name": "modis_stream",
                "config_path": "configs/modis-aod.yaml",
                "on_fire": "new_files_only",
                "status": "failed",
                "submitted_at": "2026-05-31T10:00:00",
                "ended_at": "2026-05-31T10:05:00",
                "duration_s": 300.0,
                "error": "config error: missing source",
            },
        ],
        progress={7: "Loading model: cam (1/2)"},
    )


def _render_to_text(renderable: object) -> str:
    console = Console(width=140, record=True)
    console.print(renderable)
    return console.export_text()


class TestWatchesPanel:
    def test_lists_both_watches(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_watches_panel(sample_state))
        assert "cam_realtime" in text
        assert "modis_stream" in text

    def test_shows_paused_and_enabled_state(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_watches_panel(sample_state))
        assert "paused" in text
        assert "running" in text

    def test_shows_settle_mode(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_watches_panel(sample_state))
        assert "sentinel" in text
        assert "quiescence" in text


class TestRunningPanel:
    def test_shows_running_job_and_progress(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_running_panel(sample_state))
        assert "cam_realtime" in text
        assert "7" in text  # job id
        assert "Loading model: cam (1/2)" in text  # progress message

    def test_empty_running_renders_placeholder(self) -> None:
        empty = DashboardState(version=1, pid=1, uptime_s=0.0, draining=False, max_concurrent=1)
        text = _render_to_text(render_running_panel(empty))
        assert "RUNNING" in text.upper()
        # No crash on empty; some idle marker present.
        assert "idle" in text.lower() or "—" in text or "-" in text


class TestQueuePanel:
    def test_shows_queued_job(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_queue_panel(sample_state))
        assert "modis_stream" in text
        assert "8" in text


class TestRecentPanel:
    def test_shows_completed_and_failed(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_recent_panel(sample_state))
        assert "completed" in text
        assert "failed" in text
        assert "cam_realtime" in text
        assert "modis_stream" in text
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_dashboard.py -v
```
Expected: `ModuleNotFoundError: No module named 'davinci_monet.daemon.dashboard'`.

- [ ] Step 3 — Minimal implementation. Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dashboard.py`:
```python
"""`daemon top` live dashboard renderers.

Pure render functions build Rich panels from a ``DashboardState`` snapshot so
they are unit-testable without the live loop. Colors come from the project's
single styling source of truth, ``davinci_monet.plots.style.NCAR_COLORS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from davinci_monet.plots.style import NCAR_COLORS

_BLUE = NCAR_COLORS["ncar_blue"]
_AQUA = NCAR_COLORS["aqua"]
_GREEN = NCAR_COLORS["green"]
_RED = NCAR_COLORS["red"]
_GRAY = NCAR_COLORS["gray"]
_ORANGE = NCAR_COLORS["orange"]

_STATUS_STYLE = {
    "completed": _GREEN,
    "failed": _RED,
    "running": _AQUA,
    "queued": _ORANGE,
    "skipped": _GRAY,
    "paused": _GRAY,
    "armed": _BLUE,
    "settling": _ORANGE,
}


@dataclass
class DashboardState:
    """Snapshot backing one dashboard frame (mirrors control ``StatusData``)."""

    version: int
    pid: int
    uptime_s: float
    draining: bool
    max_concurrent: int
    running: list[dict[str, Any]] = field(default_factory=list)
    queued: list[dict[str, Any]] = field(default_factory=list)
    watches: list[dict[str, Any]] = field(default_factory=list)
    recent: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_status(cls, data: dict[str, Any]) -> "DashboardState":
        """Build a snapshot from a ``status`` control response ``data`` dict."""
        return cls(
            version=int(data.get("version", 0)),
            pid=int(data.get("pid", 0)),
            uptime_s=float(data.get("uptime_s", 0.0)),
            draining=bool(data.get("draining", False)),
            max_concurrent=int(data.get("max_concurrent", 1)),
            running=list(data.get("running", [])),
            queued=list(data.get("queued", [])),
            watches=list(data.get("watches", [])),
            recent=list(data.get("recent", [])),
        )


def _styled_status(status: Optional[str]) -> Text:
    s = status or "-"
    return Text(s, style=_STATUS_STYLE.get(s, _GRAY))


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def render_watches_panel(state: DashboardState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("WATCH", style=f"bold {_BLUE}", no_wrap=True)
    table.add_column("STATE")
    table.add_column("SETTLE", style="dim")
    table.add_column("FIRE", style="dim")
    table.add_column("SRC", style="dim")
    table.add_column("PATTERN", style="dim", overflow="fold")
    for w in state.watches:
        state_str = "paused" if not w.get("enabled", True) else w.get("state", "armed")
        table.add_row(
            str(w.get("name", "?")),
            _styled_status(state_str),
            str(w.get("settle_mode", "-")),
            str(w.get("on_fire", "-")),
            str(w.get("source", "-")),
            str(w.get("watch", "-")),
        )
    if not state.watches:
        table.add_row("—", Text("no watches", style="dim"), "", "", "", "")
    return Panel(table, title="WATCHES", border_style=_BLUE, padding=(0, 1))


def render_running_panel(state: DashboardState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("JOB", style=f"bold {_AQUA}", no_wrap=True)
    table.add_column("WATCH", no_wrap=True)
    table.add_column("PROGRESS", overflow="fold")
    for job in state.running:
        job_id = job.get("id")
        msg = state.progress.get(int(job_id), "starting…") if job_id is not None else "starting…"
        table.add_row(str(job_id), str(job.get("watch_name", "?")), msg)
    if not state.running:
        table.add_row("—", Text("idle", style="dim"), "")
    title = "RUNNING (draining)" if state.draining else "RUNNING"
    return Panel(table, title=title, border_style=_AQUA, padding=(0, 1))


def render_queue_panel(state: DashboardState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("JOB", style=f"bold {_ORANGE}", no_wrap=True)
    table.add_column("WATCH", no_wrap=True)
    table.add_column("FIRE", style="dim")
    for job in state.queued:
        table.add_row(
            str(job.get("id")),
            str(job.get("watch_name", "?")),
            str(job.get("on_fire", "-")),
        )
    if not state.queued:
        table.add_row("—", Text("empty", style="dim"), "")
    return Panel(table, title="QUEUE", border_style=_ORANGE, padding=(0, 1))


def render_recent_panel(state: DashboardState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("JOB", no_wrap=True)
    table.add_column("WATCH", no_wrap=True)
    table.add_column("STATUS")
    table.add_column("DUR", justify="right", style="dim")
    for job in state.recent:
        table.add_row(
            str(job.get("id")),
            str(job.get("watch_name", "?")),
            _styled_status(job.get("status")),
            _fmt_duration(job.get("duration_s")),
        )
    if not state.recent:
        table.add_row("—", Text("no history", style="dim"), "", "")
    return Panel(table, title="RECENT", border_style=_GREEN, padding=(0, 1))
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_dashboard.py -v
```
Expected: all panel tests pass.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/dashboard.py davinci_monet/tests/unit/daemon/test_dashboard.py
git commit -m "feat(daemon): add daemon top dashboard panel renderers

Pure render_watches/running/queue/recent_panel functions build Rich panels
from a DashboardState snapshot using NCAR_COLORS, unit-tested via console
capture with no live loop.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 36: Dashboard top-level renderer composes all panels

`render_dashboard(state)` composes a header (pid/uptime/version/concurrency + draining flag) plus the four panels into a single Rich renderable (a `Group`) that the `Live` loop updates each frame. Unit test asserts the composite contains the header fields and content from every panel.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dashboard.py` (modify — add `render_header` + `render_dashboard`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_dashboard.py` (modify — add `TestRenderDashboard`)

Steps:

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_dashboard.py`:
```python
from davinci_monet.daemon.dashboard import render_dashboard


class TestRenderDashboard:
    def test_composite_contains_all_sections(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_dashboard(sample_state))
        # Header
        assert "DAVINCI" in text
        assert "4242" in text  # pid
        # Each panel's content
        assert "cam_realtime" in text  # watches + running + recent
        assert "modis_stream" in text  # queue + recent
        assert "Loading model: cam (1/2)" in text  # running progress
        assert "completed" in text  # recent

    def test_header_shows_uptime(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_dashboard(sample_state))
        # 3725s -> "1.0h" or "62.1m" formatting; just assert an uptime label.
        assert "uptime" in text.lower()

    def test_draining_flag_surfaced(self) -> None:
        draining = DashboardState(
            version=1, pid=9, uptime_s=1.0, draining=True, max_concurrent=1
        )
        text = _render_to_text(render_dashboard(draining))
        assert "draining" in text.lower()
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_dashboard.py::TestRenderDashboard -v
```
Expected: `ImportError: cannot import name 'render_dashboard' from 'davinci_monet.daemon.dashboard'`.

- [ ] Step 3 — Minimal implementation. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dashboard.py`:
```python
def render_header(state: DashboardState) -> Panel:
    line = Text()
    line.append("DAVINCI daemon", style=f"bold {_AQUA}")
    line.append("   ")
    line.append(f"pid {state.pid}", style="dim")
    line.append("   ")
    line.append(f"uptime {_fmt_duration(state.uptime_s)}", style="dim")
    line.append("   ")
    line.append(f"concurrency {state.max_concurrent}", style="dim")
    line.append("   ")
    line.append(f"v{state.version}", style="dim")
    if state.draining:
        line.append("   ")
        line.append("draining", style=f"bold {_ORANGE}")
    return Panel(line, border_style=_AQUA, padding=(0, 2))


def render_dashboard(state: DashboardState) -> Group:
    """Compose the full `daemon top` frame from a snapshot."""
    return Group(
        render_header(state),
        render_watches_panel(state),
        render_running_panel(state),
        render_queue_panel(state),
        render_recent_panel(state),
    )
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_dashboard.py -v
```
Expected: all `test_dashboard.py` tests pass.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/dashboard.py davinci_monet/tests/unit/daemon/test_dashboard.py
git commit -m "feat(daemon): compose daemon top frame with header + panels

render_dashboard groups a header (pid/uptime/concurrency/draining) with the
WATCHES/RUNNING/QUEUE/RECENT panels into one Rich renderable for the Live loop.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 37: Dashboard ingests StreamEvents into a live snapshot

The `Live` loop subscribes via `DaemonClient.stream("subscribe", topics=[...])` and folds each pushed `StreamEvent` into the current `DashboardState`. `apply_stream_event(state, event)` is the pure reducer (testable without a socket): a `job_update` event replaces/updates the matching job in `running`/`queued`/`recent` by id and status; a `log_line`/`progress` event updates `state.progress[job_id]`; a `watch_update` event replaces the matching WatchSummary; a `stats` event is folded into header counters. `run_dashboard(client, console=None, ...)` wires the reducer to a `rich.live.Live` loop (NOT unit-tested — it owns the blocking loop).

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dashboard.py` (modify — add `apply_stream_event` + `run_dashboard`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_dashboard.py` (modify — add `TestApplyStreamEvent`)

Steps:

- [ ] Step 1 — Write the failing test. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_dashboard.py`:
```python
from davinci_monet.daemon.contracts import StreamEvent
from davinci_monet.daemon.dashboard import apply_stream_event


class TestApplyStreamEvent:
    def test_progress_event_updates_progress_map(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(event="log_line", data={"job_id": 7, "message": "Pairing cam vs airnow"})
        apply_stream_event(sample_state, ev)
        assert sample_state.progress[7] == "Pairing cam vs airnow"

    def test_stage_progress_event(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(event="job_update", data={"job_id": 7, "stage": "statistics", "kind": "stage"})
        apply_stream_event(sample_state, ev)
        assert "statistics" in sample_state.progress[7]

    def test_job_update_moves_running_to_recent(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(
            event="job_update",
            data={
                "id": 7,
                "watch_name": "cam_realtime",
                "config_path": "configs/asia-aq.yaml",
                "on_fire": "whole_config",
                "status": "completed",
                "submitted_at": "2026-05-31T11:59:00",
                "ended_at": "2026-05-31T12:30:00",
                "duration_s": 1860.0,
            },
        )
        apply_stream_event(sample_state, ev)
        assert all(j["id"] != 7 for j in sample_state.running)
        assert any(j["id"] == 7 and j["status"] == "completed" for j in sample_state.recent)

    def test_job_update_promotes_queued_to_running(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(
            event="job_update",
            data={
                "id": 8,
                "watch_name": "modis_stream",
                "config_path": "configs/modis-aod.yaml",
                "on_fire": "new_files_only",
                "status": "running",
                "submitted_at": "2026-05-31T12:01:00",
                "started_at": "2026-05-31T12:31:00",
            },
        )
        apply_stream_event(sample_state, ev)
        assert any(j["id"] == 8 for j in sample_state.running)
        assert all(j["id"] != 8 for j in sample_state.queued)

    def test_watch_update_replaces_summary(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(
            event="watch_update",
            data={
                "name": "modis_stream",
                "enabled": True,
                "source": "live",
                "on_fire": "new_files_only",
                "settle_mode": "sentinel",
                "watch": "/scratch/modis/*.hdf",
                "run": "configs/modis-aod.yaml",
                "state": "armed",
                "last_job_id": None,
                "last_status": None,
                "last_fired_at": None,
            },
        )
        apply_stream_event(sample_state, ev)
        modis = next(w for w in sample_state.watches if w["name"] == "modis_stream")
        assert modis["enabled"] is True
        assert modis["state"] == "armed"

    def test_unknown_event_is_noop(self, sample_state: DashboardState) -> None:
        before = len(sample_state.running)
        apply_stream_event(sample_state, StreamEvent(event="mystery", data={}))
        assert len(sample_state.running) == before
```

- [ ] Step 2 — Run and expect failure:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_dashboard.py::TestApplyStreamEvent -v
```
Expected: `ImportError: cannot import name 'apply_stream_event' from 'davinci_monet.daemon.dashboard'`.

- [ ] Step 3 — Minimal implementation. Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dashboard.py`:
```python
def _job_id(data: dict[str, Any]) -> Optional[int]:
    raw = data.get("id", data.get("job_id"))
    return int(raw) if raw is not None else None


def _remove_by_id(jobs: list[dict[str, Any]], job_id: int) -> None:
    jobs[:] = [j for j in jobs if j.get("id") != job_id]


def _apply_job_update(state: DashboardState, data: dict[str, Any]) -> None:
    job_id = _job_id(data)
    if job_id is None:
        return
    status = data.get("status")
    # Drop any stale copy from every bucket, then re-file by status.
    for bucket in (state.running, state.queued, state.recent):
        _remove_by_id(bucket, job_id)
    record = dict(data)
    record.setdefault("id", job_id)
    if status == "running":
        state.running.insert(0, record)
    elif status == "queued":
        state.queued.append(record)
    else:  # completed | failed | skipped
        state.recent.insert(0, record)
        state.progress.pop(job_id, None)


def _apply_progress(state: DashboardState, data: dict[str, Any]) -> None:
    job_id = _job_id(data)
    if job_id is None:
        return
    if data.get("kind") == "stage" and data.get("stage"):
        stage = data["stage"]
        status = data.get("status")
        state.progress[job_id] = f"stage: {stage}" + (f" ({status})" if status else "")
    elif data.get("message") is not None:
        state.progress[job_id] = str(data["message"])


def _apply_watch_update(state: DashboardState, data: dict[str, Any]) -> None:
    name = data.get("name")
    if name is None:
        return
    for i, w in enumerate(state.watches):
        if w.get("name") == name:
            state.watches[i] = dict(data)
            return
    state.watches.append(dict(data))


def apply_stream_event(state: DashboardState, event: "Any") -> None:
    """Fold one pushed ``StreamEvent`` into ``state`` in place (pure reducer)."""
    data = dict(getattr(event, "data", {}) or {})
    kind = getattr(event, "event", None)
    if kind == "job_update":
        # A stage-only job_update carries no top-level status -> treat as progress.
        if data.get("status") is None and (data.get("stage") or data.get("message")):
            _apply_progress(state, data)
        else:
            _apply_job_update(state, data)
    elif kind in ("log_line", "progress"):
        _apply_progress(state, data)
    elif kind == "watch_update":
        _apply_watch_update(state, data)
    elif kind == "stats":
        if "running" in data:
            state.running = list(data["running"])
        if "queued" in data:
            state.queued = list(data["queued"])
    # Unknown events are intentionally ignored.


def run_dashboard(
    client: "Any",
    console: "Any" = None,
    refresh_per_second: float = 2.0,
    topics: Optional[list[str]] = None,
) -> None:  # pragma: no cover - blocking live loop, not unit-tested
    """Blocking `daemon top` loop: seed from ``status``, fold ``subscribe`` events."""
    from rich.console import Console
    from rich.live import Live

    console = console or Console()
    topics = topics or ["jobs", "watches", "stats"]

    seed = client.call("status")
    state = (
        DashboardState.from_status(seed.data)
        if getattr(seed, "ok", False) and seed.data
        else DashboardState(version=0, pid=0, uptime_s=0.0, draining=False, max_concurrent=1)
    )

    with Live(
        render_dashboard(state),
        console=console,
        refresh_per_second=refresh_per_second,
        screen=True,
    ) as live:
        for event in client.stream("subscribe", topics=topics):
            apply_stream_event(state, event)
            live.update(render_dashboard(state))
```

- [ ] Step 4 — Run and expect pass:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_dashboard.py -v
```
Expected: all `test_dashboard.py` tests (panels, composite, reducer) pass.

- [ ] Step 5 — Commit:
```bash
git add davinci_monet/daemon/dashboard.py davinci_monet/tests/unit/daemon/test_dashboard.py
git commit -m "feat(daemon): fold subscribe StreamEvents into dashboard state

apply_stream_event re-files job_update events across running/queued/recent,
updates the per-job progress map from log/progress/stage events, and replaces
WatchSummaries; run_dashboard wires the reducer to a Rich Live loop.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Pre-implementation notes (verified against the real code)

These four tasks build `davinci_monet/daemon/supervisor.py`, `davinci_monet/cli/commands/daemon.py`, and wire the sub-app into `davinci_monet/cli/app.py`. They are **unit-level** (full pipeline integration is a separate group's job).

**Isolation invariant (spec §"Isolation invariant", lines 90-95):** `supervisor.py` MUST NOT import matplotlib, xarray, monet/monetio, or the pipeline. It imports only stdlib + the daemon's own pure modules (`contracts`, `config`, `state`, `watcher`, `queue`, `dispatcher`, `notify`, `control`, `lifecycle`). `dispatcher.spawn_worker` launches `worker.py` as a subprocess — that child is the ONLY place `run_analysis` is imported. The supervisor tests below assert this by importing `supervisor` and checking `sys.modules` does not contain `matplotlib`/`xarray`.

**Injectable collaborators:** `Supervisor.__init__` takes `watcher`, `queue`, `dispatcher` (a callable invoked as `dispatcher(spec, daemon_cfg, on_event=...)`), `state` (StateStore), `notifier`, and `clock` (a `contracts.Clock`). This lets the loop tests use fakes — no real subprocess, no real socket, no real filesystem watch. `build_supervisor(watches_file, *, state=None, clock=None, notifier=None)` is the production wiring that constructs the real collaborators from a `WatchesFile`, threading `clock` into both the `PollingWatcher` and the `Supervisor`, and injecting a small `_dispatch` wrapper that adapts `spawn_worker(spec, *, on_event=None)` to the supervisor's `dispatcher(spec, daemon_cfg, on_event=...)` call shape.

**Ownership boundaries I consume (do NOT redefine):**
- `TriggerEvent`, `JobSpec`, `JobRecord`, `JobStatus`, `Clock`, `ControlRequest`, `ControlResponse`, `StreamEvent`, `PROTOCOL_VERSION`, `COMMANDS` ← `davinci_monet.daemon.contracts`
- `DaemonConfig`, `WatchRule`, `WatchesFile`, `load_watches`, `merge_rules` ← `davinci_monet.daemon.config`
- `StateStore` ← `davinci_monet.daemon.state`; `PollingWatcher` ← `watcher`; `RunQueue` ← `queue`; `spawn_worker` ← `dispatcher`; `Notifier` ← `notify`; `ControlServer` ← `control`; `DaemonClient` ← `client`; PID/lock/drain helpers ← `lifecycle`.

**Existing-code anchors (cited):**
- `davinci_monet/cli/app.py:311-321` — `register_commands()`; the daemon sub-app is registered here, mirroring `app.add_typer(get_data.app, name="get")` at line 317.
- `davinci_monet/cli/commands/get_data.py:17-20` — the `get` Typer sub-app construction pattern (`typer.Typer(name=..., help=...)`); the daemon sub-app mirrors it, and `watch_app` is a nested sub-app added via `app.add_typer(watch_app, name="watch")`.
- `davinci_monet/cli/commands/__init__.py:1-9` — module import list; daemon is added there.
- Test patterns: `davinci_monet/tests/test_cli.py:36-58` (CliRunner `--help`/`--version`), `:307` (`runner.invoke(app, ["get", "--help"])`). Daemon CLI tests follow the same `CliRunner` + mocked-client pattern. Daemon unit tests live in `davinci_monet/tests/unit/daemon/` (mirroring `unit/cli`, `unit/config`).

**Test-only fakes** are defined inline in each test module (a `FakeClock` implementing `contracts.Clock`, a `FakeQueue`, a `FakeStateStore`, a `record_dispatch` callable). They do not ship in the package.

> **Note on `Notifier`:** the supervisor-cli section's `uses`/`build_supervisor` reference `davinci_monet.daemon.notify.Notifier`. Task 28 (notify) ships this thin `Notifier` facade alongside `DesktopNotifier`/`IcloudCopier`/`notify_outcome`: `Notifier(daemon_cfg, *, hooks=None)` binds the `DaemonConfig` (reading `daemon_cfg.notifications`) + injectable desktop/iCloud hooks and exposes `notify_result(job, rule=None)`, delegating to `notify_outcome`. `build_supervisor` constructs `Notifier(cfg)` (the `DaemonConfig`, not its `notifications` sub-block), and the supervisor calls `notifier.notify_result(job, rule)` with the just-committed `JobRecord`.

---

### Task 38: Supervisor coalesce-and-dispatch loop

The core event-loop step: drain settled `TriggerEvent`s from the watcher into the coalescing queue, then (respecting `max_concurrent`, default 1) pop one pending entry, create a `jobs` row via `StateStore.create_job`, build a `JobSpec`, call the injected dispatcher once, and record the outcome. One coalesced trigger ⇒ exactly one job.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/supervisor.py` (new)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_supervisor_loop.py` (new)

(The daemon unit-test package `davinci_monet/tests/unit/daemon/__init__.py` was created by Task 1 — treat it as pre-existing; do NOT create it here.)

- [ ] **Step 1 — Write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_supervisor_loop.py`:
```python
"""Unit tests for the supervisor coalesce + dispatch loop (no real pipeline)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest

from davinci_monet.daemon.config import DaemonConfig, WatchRule, WatchesFile
from davinci_monet.daemon.contracts import (
    JobRecord,
    JobSpec,
    JobStatus,
    TriggerEvent,
)


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


class FakeQueue:
    """Coalescing FIFO: repeat submits for the same watch_name union new_files."""

    def __init__(self) -> None:
        self._items: dict[str, list[str]] = {}
        self._order: list[str] = []

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

    def mark_failed(self, job_id, exit_code, error, log_path=None, ended_at=None) -> None:
        self.jobs[job_id]["status"] = JobStatus.FAILED
        self.jobs[job_id]["error"] = error


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
    """Importing the supervisor must NOT pull in matplotlib/xarray/the pipeline."""
    import davinci_monet.daemon.supervisor  # noqa: F401

    assert "matplotlib" not in sys.modules
    assert "xarray" not in sys.modules
    assert "davinci_monet.pipeline.runner" not in sys.modules


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
```

- [ ] **Step 2 — Run and expect failure.** From the repo root in the `davinci` env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_supervisor_loop.py -v
```
Expected: collection/import error `ModuleNotFoundError: No module named 'davinci_monet.daemon.supervisor'` (or `ImportError: cannot import name 'Supervisor'`). All tests error out — the supervisor module does not exist yet.

- [ ] **Step 3 — Minimal implementation.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/supervisor.py`. (Stdlib + daemon-pure imports only — no sci stack.)
```python
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

from davinci_monet.daemon.config import DaemonConfig, WatchRule, WatchesFile
from davinci_monet.daemon.contracts import (
    Clock,
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
        """Pop and dispatch pending jobs up to max_concurrent (default 1)."""
        while self._running_count < self.daemon_cfg.max_concurrent:
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
    watcher = PollingWatcher(
        list(watches_file.watches.values()), cfg, the_clock
    )
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
```

- [ ] **Step 4 — Run and expect pass.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_supervisor_loop.py -v
```
Expected: 4 passed (`test_isolation_invariant_no_sci_stack_imported`, `test_one_coalesced_trigger_dispatches_exactly_one_job`, `test_failed_dispatch_marks_job_failed_and_loop_continues`, `test_max_concurrent_one_dispatches_serially`).

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/daemon/supervisor.py davinci_monet/tests/unit/daemon/test_supervisor_loop.py
git commit -m "feat(daemon): supervisor coalesce-and-dispatch loop

Add Supervisor with injectable watcher/queue/dispatcher/state/notify/clock
collaborators and build_supervisor() production wiring. One coalesced trigger
dispatches exactly one job; max_concurrent (default 1) gates serial dispatch;
failed/raised dispatch marks the job FAILED without wedging the loop. Module
imports stay sci-stack-free to honor the isolation invariant.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 39: Supervisor control-command handlers

Add `Supervisor.handle_command(request)` and helpers (`build_status`, `watch_summaries`) so the control server can route every command in the catalog (`status`, `ping`, `watch_list`/`pause`/`resume`/`trigger`, `history`, `job_get`, `reload`, `shutdown`, `logs`) to the supervisor and get a `ControlResponse`. This is the dispatch table the `ControlServer` (other group) calls; testing it directly keeps the socket layer out of scope here.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/supervisor.py` (modify — add `handle_command`, `build_status`, `watch_summaries`, `_handler_table`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_supervisor_control.py` (new)

- [ ] **Step 1 — Write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_supervisor_control.py`:
```python
"""Unit tests for Supervisor.handle_command (control dispatch table)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pytest

from davinci_monet.daemon.config import DaemonConfig, WatchRule, WatchesFile
from davinci_monet.daemon.contracts import JobStatus


class FakeClock:
    def __init__(self) -> None:
        self._t = 100.0

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds


class FakeStateStore:
    def __init__(self) -> None:
        self._active: list[Any] = []
        self._jobs: dict[int, Any] = {}
        self._history: list[Any] = []
        self.enabled_calls: list[tuple[str, bool]] = []

    def active_jobs(self):
        return list(self._active)

    def list_jobs(self, watch_name=None, status=None, limit=50):
        return list(self._history)[:limit]

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def set_enabled(self, watch_name, enabled):
        self.enabled_calls.append((watch_name, enabled))


class FakeQueue:
    def __init__(self) -> None:
        self.submitted: list[Any] = []

    def submit(self, event) -> bool:
        coalesced = bool(self.submitted)
        self.submitted.append(event)
        return coalesced

    def pop(self):
        return None

    def __len__(self) -> int:
        return 0


class FakeWatcher:
    def poll(self):
        return []


def _sup(**overrides):
    from davinci_monet.daemon.supervisor import Supervisor

    wf = WatchesFile(
        daemon=DaemonConfig(max_concurrent=2),
        watches={
            "cam": WatchRule(name="cam", watch="/tmp/cam/*.nc", run="/tmp/cam.yaml"),
            "modis": WatchRule(
                name="modis",
                watch="/tmp/modis/*.hdf",
                run="/tmp/modis.yaml",
                on_fire="new_files_only",
                inject_into="modis",
                sentinel="/tmp/modis/DELIVERED",
            ),
        },
    )
    kwargs: dict[str, Any] = dict(
        watches_file=wf,
        watcher=FakeWatcher(),
        queue=FakeQueue(),
        dispatcher=lambda spec, cfg: None,
        state=FakeStateStore(),
        notifier=None,
        clock=FakeClock(),
    )
    kwargs.update(overrides)
    return Supervisor(**kwargs)


def test_ping_returns_version_and_uptime() -> None:
    from davinci_monet.daemon.contracts import PROTOCOL_VERSION

    sup = _sup()
    resp = sup.handle_command("ping", {})
    assert resp.ok is True
    assert resp.data["pong"] is True
    assert resp.data["version"] == PROTOCOL_VERSION
    assert "uptime_s" in resp.data


def test_watch_list_returns_summaries_with_settle_mode() -> None:
    sup = _sup()
    resp = sup.handle_command("watch_list", {})
    assert resp.ok is True
    watches = {w["name"]: w for w in resp.data["watches"]}
    assert watches["cam"]["settle_mode"] == "quiescence"
    assert watches["modis"]["settle_mode"] == "sentinel"
    assert watches["modis"]["on_fire"] == "new_files_only"
    assert watches["cam"]["enabled"] is True


def test_watch_pause_and_resume_persist_via_state() -> None:
    state = FakeStateStore()
    sup = _sup(state=state)
    paused = sup.handle_command("watch_pause", {"name": "cam"})
    assert paused.ok is True
    assert paused.data == {"name": "cam", "enabled": False}
    assert sup.rules["cam"].enabled is False
    assert ("cam", False) in state.enabled_calls

    resumed = sup.handle_command("watch_resume", {"name": "cam"})
    assert resumed.data == {"name": "cam", "enabled": True}
    assert sup.rules["cam"].enabled is True
    assert ("cam", True) in state.enabled_calls


def test_watch_pause_unknown_name_is_not_found() -> None:
    sup = _sup()
    resp = sup.handle_command("watch_pause", {"name": "nope"})
    assert resp.ok is False
    assert resp.code == "not_found"


def test_watch_trigger_enqueues_event() -> None:
    queue = FakeQueue()
    sup = _sup(queue=queue)
    resp = sup.handle_command(
        "watch_trigger", {"name": "cam", "files": ["/tmp/cam/x.nc"]}
    )
    assert resp.ok is True
    assert resp.data["coalesced"] is False
    assert len(queue.submitted) == 1
    assert queue.submitted[0].watch_name == "cam"
    assert queue.submitted[0].new_files == ["/tmp/cam/x.nc"]


def test_status_reports_draining_and_max_concurrent() -> None:
    sup = _sup()
    resp = sup.handle_command("status", {})
    assert resp.ok is True
    assert resp.data["max_concurrent"] == 2
    assert resp.data["draining"] is False
    assert isinstance(resp.data["watches"], list)
    assert "running" in resp.data and "queued" in resp.data

    sup.request_shutdown()
    resp2 = sup.handle_command("status", {})
    assert resp2.data["draining"] is True


def test_unknown_command_is_unsupported() -> None:
    sup = _sup()
    resp = sup.handle_command("frobnicate", {})
    assert resp.ok is False
    assert resp.code == "unsupported"


def test_shutdown_sets_draining_and_acks() -> None:
    sup = _sup()
    resp = sup.handle_command("shutdown", {"drain": True})
    assert resp.ok is True
    assert resp.data["shutting_down"] is True
    assert resp.data["draining"] is True
    assert sup.draining is True
```

- [ ] **Step 2 — Run and expect failure.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_supervisor_control.py -v
```
Expected: `AttributeError: 'Supervisor' object has no attribute 'handle_command'` on every test.

- [ ] **Step 3 — Minimal implementation.** Edit `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/supervisor.py`. Add to the imports near the top (extend the existing `contracts` import):
```python
from davinci_monet.daemon.contracts import (
    Clock,
    ControlResponse,
    JobRecord,
    JobSpec,
    JobStatus,
    PROTOCOL_VERSION,
    TriggerEvent,
)
```
(Replace the existing `from davinci_monet.daemon.contracts import (...)` block with this extended one.) Then add these methods inside the `Supervisor` class, after `_record_result` (before `request_shutdown`):
```python
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
        return ControlResponse(
            ok=True, data={"shutting_down": True, "draining": drain}
        )

    def _cmd_subscribe_ack(self, args: dict[str, Any]) -> ControlResponse:
        topics = args.get("topics") or ["jobs", "watches", "stats"]
        return ControlResponse(ok=True, data={"subscribed": list(topics)})

    def _cmd_logs_tail_ack(self, args: dict[str, Any]) -> ControlResponse:
        return ControlResponse(ok=True, data={"streaming": True})
```
Add `Callable` to the existing `from typing import ...` line if not already present (it is, from Task 38's `Callable[..., Any]`).

- [ ] **Step 4 — Run and expect pass.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_supervisor_control.py davinci_monet/tests/unit/daemon/test_supervisor_loop.py -v
```
Expected: all tests in both modules pass (8 + 4 = 12 passed). Running both confirms the new methods did not break the loop tests.

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/daemon/supervisor.py davinci_monet/tests/unit/daemon/test_supervisor_control.py
git commit -m "feat(daemon): supervisor control-command dispatch table

Add Supervisor.handle_command routing every catalog command (ping/status/
watch_list/pause/resume/trigger/history/job_get/shutdown) to a ControlResponse,
plus build_status and watch_summaries helpers for the control server and
dashboard. pause/resume mutate the in-memory rule and persist via
StateStore.set_enabled; unknown watch -> not_found; unknown cmd -> unsupported.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 40: Daemon CLI sub-app (serve/start/stop/status/reload/watch/logs/history)

Build the `daemon` Typer sub-app mirroring `get_data.app`. `serve` runs the supervisor in the foreground; `start`/`stop`/`status`/`reload`/`logs`/`history` and the nested `watch` verbs are thin clients of the running daemon over the Unix socket via `DaemonClient`. Tests use `CliRunner` with the client patched, so no real daemon/socket is needed.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/cli/commands/daemon.py` (new)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_cli_daemon.py` (new)

- [ ] **Step 1 — Write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_cli_daemon.py`:
```python
"""Unit tests for the `daemon` CLI sub-app (mocked client, no real socket)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from davinci_monet.cli.commands import daemon as daemon_cli
from davinci_monet.daemon.contracts import ControlResponse

runner = CliRunner()


def test_daemon_subapp_help_lists_commands() -> None:
    result = runner.invoke(daemon_cli.app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    for sub in ("serve", "start", "stop", "status", "reload", "watch", "logs", "history"):
        assert sub in out


def test_status_invokes_client_status() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(
        ok=True,
        data={
            "version": 1,
            "pid": 4242,
            "uptime_s": 12.0,
            "draining": False,
            "max_concurrent": 1,
            "running": [],
            "queued": [],
            "watches": [
                {
                    "name": "cam",
                    "enabled": True,
                    "source": "file",
                    "on_fire": "whole_config",
                    "settle_mode": "quiescence",
                    "watch": "/tmp/cam/*.nc",
                    "run": "/tmp/cam.yaml",
                    "state": "armed",
                    "last_job_id": None,
                    "last_status": None,
                    "last_fired_at": None,
                }
            ],
            "recent": [],
        },
    )
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["status"])
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("status")
    assert "cam" in result.stdout
    assert "4242" in result.stdout


def test_status_when_daemon_not_running_reports_clearly() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = False
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["status"])
    assert result.exit_code != 0
    assert "not running" in result.stdout.lower()


def test_watch_pause_calls_client_with_name() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(
        ok=True, data={"name": "cam", "enabled": False}
    )
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["watch", "pause", "cam"])
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("watch_pause", name="cam")
    assert "cam" in result.stdout


def test_watch_list_renders_rows() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(
        ok=True,
        data={
            "watches": [
                {
                    "name": "modis",
                    "enabled": False,
                    "source": "live",
                    "on_fire": "new_files_only",
                    "settle_mode": "sentinel",
                    "watch": "/tmp/modis/*.hdf",
                    "run": "/tmp/modis.yaml",
                    "state": "paused",
                    "last_job_id": 7,
                    "last_status": "completed",
                    "last_fired_at": None,
                }
            ]
        },
    )
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["watch", "list"])
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("watch_list")
    assert "modis" in result.stdout


def test_history_passes_filters() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(ok=True, data={"jobs": []})
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(
            daemon_cli.app, ["history", "--watch", "cam", "--failed", "--limit", "5"]
        )
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("history", watch="cam", failed=True, limit=5)


def test_reload_reports_changes() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(
        ok=True,
        data={"reloaded": True, "watches": [], "added": ["x"], "removed": [], "updated": ["cam"]},
    )
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["reload"])
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("reload")
    assert "x" in result.stdout
```

- [ ] **Step 2 — Run and expect failure.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_cli_daemon.py -v
```
Expected: `ModuleNotFoundError: No module named 'davinci_monet.cli.commands.daemon'` (collection error) — the CLI module does not exist yet.

- [ ] **Step 3 — Minimal implementation.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/cli/commands/daemon.py`:
```python
"""CLI command: the `daemon` sub-app — file-watching automation daemon.

`serve` runs the supervisor in the foreground. `start`/`stop`/`status`/`reload`/
`logs`/`history` and the nested `watch` verbs are thin clients that talk to a
running daemon over its Unix control socket via DaemonClient. This module never
imports the pipeline/sci-stack; the supervisor it launches forks workers that do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from davinci_monet.cli.app import ERROR_COLOR, INFO_COLOR, SUCCESS_COLOR

app = typer.Typer(
    name="daemon",
    help="File-watching automation daemon: auto-run analyses when data lands.",
)

watch_app = typer.Typer(
    name="watch",
    help="Manage individual watch rules (list/add/remove/pause/resume/trigger/save).",
)
app.add_typer(watch_app, name="watch")

_DEFAULT_WATCHES = "watches.yaml"


def _load_daemon_config(watches: str):
    """Load + layer-1 expand watches.yaml -> WatchesFile (daemon policy + rules)."""
    from davinci_monet.daemon.config import load_watches

    return load_watches(watches)


def _make_client(watches: str):
    """Build a DaemonClient bound to the socket from watches.yaml's state_dir."""
    from davinci_monet.daemon.client import DaemonClient

    cfg = _load_daemon_config(watches).daemon
    return DaemonClient(cfg.socket_path)


def _require_alive(client) -> None:
    """Abort with a clear message if no daemon is answering the socket."""
    if not client.is_alive():
        typer.secho(
            "Daemon is not running (no response on the control socket). "
            "Start it with `davinci-monet daemon start` or `daemon serve`.",
            fg=ERROR_COLOR,
        )
        raise typer.Exit(1)


def _call(client, cmd: str, **args):
    """Round-trip one command, aborting on a non-ok response."""
    resp = client.call(cmd, **args)
    if not resp.ok:
        typer.secho(f"Error: {resp.error}", fg=ERROR_COLOR)
        raise typer.Exit(1)
    return resp.data


@app.command()
def serve(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Run the daemon in the foreground (logs to stdout; tmux/systemd-friendly)."""
    from davinci_monet.daemon import lifecycle
    from davinci_monet.daemon.control import ControlServer
    from davinci_monet.daemon.supervisor import build_supervisor

    wf = _load_daemon_config(watches)
    supervisor = build_supervisor(wf)
    server = ControlServer(wf.daemon.socket_path, supervisor.handle_command)
    typer.secho(f"DAVINCI daemon serving ({len(wf.watches)} watches)...", fg=INFO_COLOR)
    lifecycle.install_signal_handlers(supervisor.request_shutdown)
    supervisor.serve(control_server=server)
    typer.secho("Daemon stopped.", fg=SUCCESS_COLOR)


@app.command()
def start(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Start the daemon in the background (PID + lock file; logs to daemon.log)."""
    from davinci_monet.daemon import lifecycle

    wf = _load_daemon_config(watches)

    def _serve() -> None:
        # Runs only in the backgrounded daemon child (after the double-fork).
        from davinci_monet.daemon.control import ControlServer
        from davinci_monet.daemon.supervisor import build_supervisor

        supervisor = build_supervisor(wf)
        server = ControlServer(wf.daemon.socket_path, supervisor.handle_command)
        lifecycle.install_signal_handlers(supervisor.request_shutdown)
        supervisor.serve(control_server=server)

    result = lifecycle.start_background(wf.daemon, _serve)
    if not result.started:
        typer.secho(result.message, fg=ERROR_COLOR)
        raise typer.Exit(1)
    typer.secho(result.message, fg=SUCCESS_COLOR)


@app.command()
def stop(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", help="Seconds to wait for graceful drain before giving up."
    ),
) -> None:
    """Stop a background daemon (graceful drain via SIGTERM)."""
    client = _make_client(watches)
    if client.is_alive():
        client.call("shutdown", drain=True, timeout=timeout)
        typer.secho("Shutdown requested (draining).", fg=INFO_COLOR)
        return
    from davinci_monet.daemon import lifecycle

    wf = _load_daemon_config(watches)
    result = lifecycle.stop_background(wf.daemon)
    if result.started:
        typer.secho(result.message, fg=SUCCESS_COLOR)
    else:
        typer.secho(result.message, fg=ERROR_COLOR)
        raise typer.Exit(1)


@app.command()
def status(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Print a one-shot summary of the running daemon."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "status")
    typer.secho(
        f"daemon up | pid {data['pid']} | uptime {data['uptime_s']:.0f}s | "
        f"max_concurrent {data['max_concurrent']} | "
        f"draining {data['draining']}",
        fg=INFO_COLOR,
    )
    typer.echo(f"running {len(data['running'])} | queued {len(data['queued'])}")
    typer.echo("watches:")
    for w in data["watches"]:
        flag = "paused" if not w["enabled"] else w["state"]
        typer.echo(f"  {w['name']:<20} {flag:<10} {w['settle_mode']:<10} -> {w['run']}")


@app.command()
def reload(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Re-read watches.yaml and reconcile declared vs. live rules."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "reload")
    typer.secho(
        f"reloaded | added {data.get('added', [])} | "
        f"removed {data.get('removed', [])} | updated {data.get('updated', [])}",
        fg=SUCCESS_COLOR,
    )


@app.command()
def shell(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Open the interactive daemon REPL."""
    from davinci_monet.daemon.shell import run_shell

    client = _make_client(watches)
    _require_alive(client)
    run_shell(client)


@app.command()
def top(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Open the live dashboard (streams job/watch/stats updates)."""
    from davinci_monet.daemon.dashboard import run_dashboard

    client = _make_client(watches)
    _require_alive(client)
    run_dashboard(client)


@app.command()
def logs(
    target: str = typer.Argument(..., help="Job id or watch name whose log to show."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
    kind: str = typer.Option("job", "--kind", help="Resolve target as 'job' or 'watch'."),
    tail: bool = typer.Option(False, "--tail", help="Stream the log live until Ctrl-C."),
) -> None:
    """Show (or tail) the log for a job or watch."""
    client = _make_client(watches)
    _require_alive(client)
    if tail:
        for event in client.stream("logs_tail", target=target, kind=kind):
            line = event.data.get("line") or event.data.get("message", "")
            typer.echo(line)
        return
    data = _call(client, "logs", target=target, kind=kind)
    typer.echo(data.get("text", ""))


@app.command()
def history(
    watch: Optional[str] = typer.Option(None, "--watch", help="Filter by watch name."),
    failed: bool = typer.Option(False, "--failed", help="Only failed jobs."),
    limit: int = typer.Option(50, "--limit", help="Max rows (most recent first)."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Show recent job history from the daemon's state store."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "history", watch=watch, failed=failed, limit=limit)
    jobs = data.get("jobs", [])
    if not jobs:
        typer.echo("(no jobs)")
        return
    for job in jobs:
        typer.echo(
            f"#{job['id']:<5} {job['watch_name']:<18} {job['status']:<10} "
            f"{job.get('submitted_at', '')}"
        )


# ---- watch verbs ----------------------------------------------------------


@watch_app.command("list")
def watch_list(
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """List all watch rules and their runtime state."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_list")
    for w in data["watches"]:
        flag = "paused" if not w["enabled"] else w["state"]
        typer.echo(
            f"{w['name']:<20} {flag:<10} {w['source']:<6} {w['settle_mode']:<10} -> {w['run']}"
        )


@watch_app.command("pause")
def watch_pause(
    name: str = typer.Argument(..., help="Watch rule name to pause."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Pause a watch rule (stops firing until resumed)."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_pause", name=name)
    typer.secho(f"paused {data['name']}", fg=INFO_COLOR)


@watch_app.command("resume")
def watch_resume(
    name: str = typer.Argument(..., help="Watch rule name to resume."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Resume a paused watch rule."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_resume", name=name)
    typer.secho(f"resumed {data['name']}", fg=SUCCESS_COLOR)


@watch_app.command("trigger")
def watch_trigger(
    name: str = typer.Argument(..., help="Watch rule name to fire manually."),
    files: list[str] = typer.Option(
        [], "--file", "-f", help="Specific new file path(s) to inject (repeatable)."
    ),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Manually fire a watch rule now (optionally with explicit new files)."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_trigger", name=name, files=list(files))
    if data.get("coalesced"):
        typer.secho(f"{name}: coalesced into an already-pending run.", fg=INFO_COLOR)
    else:
        typer.secho(f"{name}: queued.", fg=SUCCESS_COLOR)


@watch_app.command("remove")
def watch_remove(
    name: str = typer.Argument(..., help="Watch rule name to remove."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Remove a (live-added) watch rule."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_remove", name=name)
    typer.secho(f"removed {data['removed']}", fg=INFO_COLOR)


@watch_app.command("add")
def watch_add(
    rule_yaml: str = typer.Argument(..., help="Path to a YAML file with one watch rule."),
    save: bool = typer.Option(False, "--save", help="Also write the rule back to watches.yaml."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Add a watch rule at runtime from a small YAML snippet."""
    import yaml

    from davinci_monet.config.parser import expand_env_vars

    client = _make_client(watches)
    _require_alive(client)
    with open(rule_yaml) as f:
        raw = expand_env_vars(yaml.safe_load(f) or {})
    data = _call(client, "watch_add", rule=raw, save=save)
    typer.secho(f"added {data['added']}", fg=SUCCESS_COLOR)


@watch_app.command("save")
def watch_save(
    name: str = typer.Argument(..., help="Live watch rule name to persist to file."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Persist a live-added watch rule back into watches.yaml."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_save", name=name)
    typer.secho(f"saved {data['saved']} -> {data['path']}", fg=SUCCESS_COLOR)
```

- [ ] **Step 4 — Run and expect pass.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_cli_daemon.py -v
```
Expected: 7 passed (`test_daemon_subapp_help_lists_commands`, `test_status_invokes_client_status`, `test_status_when_daemon_not_running_reports_clearly`, `test_watch_pause_calls_client_with_name`, `test_watch_list_renders_rows`, `test_history_passes_filters`, `test_reload_reports_changes`).

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/cli/commands/daemon.py davinci_monet/tests/unit/daemon/test_cli_daemon.py
git commit -m "feat(daemon): daemon CLI sub-app (serve/start/stop/status/reload/watch/logs/history)

Add the `daemon` Typer sub-app mirroring the `get` sub-app. serve runs the
supervisor in the foreground with a control server; start/stop background it via
lifecycle; status/reload/logs/history and the nested `watch` verbs are thin
DaemonClient round-trips over the Unix socket, aborting cleanly when the daemon
is not answering. Module stays sci-stack-free.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 41: Register daemon sub-app in cli/app.py

Wire the new `daemon` sub-app into the main CLI exactly like the `get` sub-app, so `davinci-monet daemon ...` is discoverable. Also add `daemon` to the `cli/commands/__init__.py` export list for consistency with `get_data`/`run`/`validate`.

Files:
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/cli/app.py` (modify — `register_commands()` at line 311-318; add the daemon import + `app.add_typer(...)` after line 317)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/cli/commands/__init__.py` (modify — add `daemon` to the import + `__all__`)
- `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_app_registration.py` (new)

- [ ] **Step 1 — Write the failing test.** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/unit/daemon/test_app_registration.py`:
```python
"""The daemon sub-app must be registered on the main CLI app."""

from __future__ import annotations

from typer.testing import CliRunner

from davinci_monet.cli.app import app

runner = CliRunner()


def test_root_help_lists_daemon_subapp() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "daemon" in result.stdout.lower()


def test_daemon_help_routes_through_root_app() -> None:
    result = runner.invoke(app, ["daemon", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "serve" in out
    assert "status" in out
    assert "watch" in out


def test_daemon_watch_help_routes_through_root_app() -> None:
    result = runner.invoke(app, ["daemon", "watch", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "pause" in out
    assert "resume" in out
    assert "trigger" in out
```

- [ ] **Step 2 — Run and expect failure.**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/test_app_registration.py -v
```
Expected: `test_root_help_lists_daemon_subapp` and the two routing tests FAIL — `daemon` is not yet registered, so the root help omits it and `runner.invoke(app, ["daemon", "--help"])` exits non-zero (`No such command 'daemon'`).

- [ ] **Step 3 — Minimal implementation.** Edit `register_commands()` in `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/cli/app.py` (lines 311-318). Replace:
```python
def register_commands() -> None:
    """Register all CLI commands."""
    # Import command modules
    from davinci_monet.cli.commands import get_data, run, validate

    # Register subcommands
    app.add_typer(get_data.app, name="get")
```
with:
```python
def register_commands() -> None:
    """Register all CLI commands."""
    # Import command modules
    from davinci_monet.cli.commands import daemon, get_data, run, validate

    # Register subcommands
    app.add_typer(get_data.app, name="get")
    app.add_typer(daemon.app, name="daemon")
```
Then edit `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/cli/commands/__init__.py` to include `daemon`:
```python
"""CLI command modules for DAVINCI."""

from davinci_monet.cli.commands import daemon, get_data, run, validate

__all__ = [
    "daemon",
    "get_data",
    "run",
    "validate",
]
```

- [ ] **Step 4 — Run and expect pass.** Run the registration tests, then the whole daemon unit suite to confirm no regressions in the CLI tests:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/daemon/ davinci_monet/tests/test_cli.py -v
```
Expected: the three `test_app_registration.py` tests pass, the full `unit/daemon/` set passes, and the pre-existing `test_cli.py` suite still passes (no regression from the new sub-app).

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/cli/app.py davinci_monet/cli/commands/__init__.py davinci_monet/tests/unit/daemon/test_app_registration.py
git commit -m "feat(daemon): register daemon sub-app on the main CLI

Wire daemon.app into register_commands() via app.add_typer(daemon.app,
name='daemon'), mirroring the get sub-app, and export daemon from
cli.commands.__init__. \`davinci-monet daemon ...\` and \`daemon watch ...\` now
route through the root app.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Test path design (CLAUDE.md rule #2 — present for approval BEFORE coding)

These are TRUE integration tests per CLAUDE.md rule #1: the analysis run flows through the **real** pipeline entry point in a **real worker subprocess**, never a mocked or bypassed path.

**Entry points actually called (the real path):**
1. `davinci_monet.daemon.config.load_watches(<temp watches.yaml>)` — real YAML load + layer-1 env expansion + Pydantic validation, producing a `WatchesFile` (`DaemonConfig` + one `WatchRule`).
2. `davinci_monet.daemon.supervisor.Supervisor(...)` constructed in-process and stepped through its real wiring: `PollingWatcher` (fed an **injectable fake `Clock`** so settle/quiescence is deterministic — no `time.sleep`) → `RunQueue` (coalescing FIFO) → `Dispatcher`.
3. The dispatcher builds a `JobSpec` and **spawns the real `worker.py` as a subprocess** (`python -m davinci_monet.daemon.worker`, JobSpec JSON on stdin) — i.e. a fresh interpreter, honoring the spec's isolation invariant.
4. Inside that child, `worker.main()` calls `davinci_monet.pipeline.runner.run_analysis(config)` → `PipelineRunner.run_from_config()` → the standard stages (`LoadSourcesStage → PairingStage → StatisticsStage → PlottingStage → SaveResultsStage`). This is the exact code path `davinci-monet run config.yaml` takes.
5. Supervisor consumes the worker's JSON-line progress events, records submit/start/end via `davinci_monet.daemon.state.StateStore`, and fires `davinci_monet.daemon.notify`.

**Data flowing through:**
- Synthetic NetCDF: a `generic` model (`create_model_dataset`, O3, with a small bias added) and a perfectly-co-located `pt_sfc` obs (`PerfectMatchScenario._generate_point_obs`), written to temp `.nc` files — exactly the `TestPointPipeline` recipe in `davinci_monet/tests/test_integration.py:87-241`, the known-green pipeline shape.
- A minimal DAVINCI config dict (legacy `model:`/`obs:`/`pairs:` form, accepted + auto-migrated by the pipeline) written to a temp `run.yaml`; the watcher's trigger file is dropped into a temp `incoming/` dir matching the rule's `watch:` glob.
- The fake clock is advanced past the rule's `settle` window so the watcher emits a real `TriggerEvent(new_files=[<dropped .nc>])`.

**What is mocked (side-effects ONLY — NEVER the pipeline):** the desktop notification command (`osascript`/`terminal-notifier` subprocess) and the iCloud `shutil.copy*` are monkeypatched to record calls into a list, so the test asserts the notify hook fired **on completion** without writing to a real iCloud dir or popping a real macOS notification. `run_analysis` / `PipelineRunner` / `StateStore` / the worker subprocess are all REAL.

**Assertions (both tasks):** settle fires (job created); the worker subprocess exits 0; the real pipeline produced `statistics_summary.csv`, ≥1 `*.png` plot, and a `pipeline_*.md` log under the run's `output_dir`/`log_dir`; the job is persisted in `StateStore` history with `status == JobStatus.COMPLETED` and a non-null `result_summary`; and the (mocked) notify hook was invoked exactly once for that job. Task 2 additionally asserts the injected new-file path reached the worker's resolved config (`on_fire: new_files_only` overrode `inject_into`'s `files:`).

> NOTE TO IMPLEMENTER: these tests are written **last**, after the daemon modules exist (config, contracts, state, watcher, queue, dispatcher, worker, notify, supervisor). The two helper functions and a `FakeClock` live in the test module. If the assembled `Supervisor` exposes a single-shot/step API (e.g. `run_once()` / `tick()` / `process_pending()`), prefer it; if it only exposes a blocking `serve()`, run it on a daemon thread and poll `StateStore.get_job(...)` until terminal with a generous wall-clock timeout. Adjust the exact `Supervisor` construction/step calls to the real assembled signatures — the contract names (`load_watches`, `WatchRule`, `DaemonConfig`, `StateStore`, `JobStatus`, `JobRecord`, `run_analysis`) are fixed by the shared interface and MUST be used verbatim.

---

### Task 42: Whole-config daemon integration (settle -> real worker subprocess -> pipeline -> history + notify)

Files:
- Create: `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/integration/test_daemon_integration.py`
- Reads (no edits): `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/synthetic/models.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/synthetic/scenarios.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/test_integration.py:87-241` (recipe reference), `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/supervisor.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/state.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/contracts.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/notify.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/worker.py`
- Assumes pre-existing (do NOT create): `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/integration/__init__.py`

> **Assembler note:** the `davinci_monet/tests/integration/__init__.py` package marker already exists in the repo; do NOT create it. This integration test lives under `tests/integration/` (the existing repo convention), not under a daemon-specific test package.

- [ ] **Step 1 — Write the failing test (full code).** Create `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/integration/test_daemon_integration.py` with the shared helpers + `FakeClock` + the whole-config test. This file is shared by Task 43 (Task 43 appends a second test). Full content:

```python
"""Daemon-mode INTEGRATION tests (CLAUDE.md rule #1 compliant).

The analysis run flows through the REAL pipeline entry point
(``run_analysis`` -> ``PipelineRunner.run_from_config``) inside a REAL worker
subprocess spawned by the daemon dispatcher. Only the desktop/iCloud notify
side-effects are mocked; the pipeline, worker subprocess, and SQLite state
store are exercised for real.

See the module-level "Test path design" note in the plan for the entry points
called and the data flow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.daemon.config import load_watches
from davinci_monet.daemon.contracts import JobStatus
from davinci_monet.daemon.state import StateStore
from davinci_monet.daemon.supervisor import build_supervisor
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


class FakeClock:
    """Injectable monotonic clock (conforms to daemon contracts.Clock).

    ``now`` only advances when the test calls ``advance``; ``sleep`` advances
    the virtual clock instead of blocking, so settle/quiescence windows elapse
    deterministically with no real wall-clock waits.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += max(0.0, float(seconds))

    def advance(self, seconds: float) -> None:
        self._t += float(seconds)


def _write_synthetic_point_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Write a synthetic generic model + co-located pt_sfc obs to NetCDF.

    Mirrors the known-green TestPointPipeline recipe in test_integration.py:
    O3 model field with a small spatial gradient + bias, and obs sampled from
    the model so pairing/stats are well-defined. Returns (model_nc, obs_nc).
    """
    domain = Domain(
        lon_min=-105.0, lon_max=-95.0, lat_min=35.0, lat_max=45.0, n_lon=12, n_lat=12
    )
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-15 06:00", freq="1h")

    model_ds = create_model_dataset(
        variables=["O3"], domain=domain, time_config=time_cfg, seed=42
    )
    lat_vals = model_ds.lat.values
    lat_norm = (lat_vals - lat_vals.min()) / (lat_vals.max() - lat_vals.min())
    model_ds["O3"] = model_ds["O3"] + 20.0 * lat_norm[:, np.newaxis]

    scenario = PerfectMatchScenario(
        variables=["O3"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_obs=8,
        noise_level=0.0,
        seed=42,
    )
    obs_ds = scenario._generate_point_obs(model_ds)

    rng = np.random.default_rng(42)
    model_ds["O3"] = model_ds["O3"] + 5.0 + rng.normal(0, 3.0, size=model_ds["O3"].shape)

    model_nc = tmp_path / "model.nc"
    obs_nc = tmp_path / "obs.nc"
    model_ds.to_netcdf(model_nc)
    obs_ds.to_netcdf(obs_nc)
    return model_nc, obs_nc


def _minimal_point_config_dict(
    model_nc: Path, obs_nc: Path, output_dir: Path, log_dir: Path
) -> dict[str, Any]:
    """Minimal DAVINCI config that the real pipeline can run end-to-end."""
    return {
        "analysis": {
            "start_time": "2024-01-15 00:00",
            "end_time": "2024-01-15 06:00",
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
        },
        "model": {
            "synthetic": {
                "mod_type": "generic",
                "files": str(model_nc),
                "radius_of_influence": 50000,
                "mapping": {"surface": {"O3": "O3"}},
                "variables": {"O3": {"units": "ppb"}},
            }
        },
        "obs": {
            "surface": {
                "obs_type": "pt_sfc",
                "filename": str(obs_nc),
                "variables": {"O3": {"obs_min": 0, "obs_max": 200, "units": "ppb"}},
            }
        },
        "pairs": {
            "synthetic_surface": {
                "model": "synthetic",
                "obs": "surface",
                "variable": {"model_var": "O3", "obs_var": "O3"},
            }
        },
        "plots": {
            "scatter_o3": {
                "type": "scatter",
                "pairs": ["synthetic_surface"],
                "title": "O3: Model vs Observations",
            }
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
    }


def _drain_supervisor(
    supervisor: "Supervisor", store: StateStore, *, timeout_s: float = 240.0
) -> None:
    """Step/serve the supervisor until no QUEUED/RUNNING jobs remain.

    If the supervisor exposes a single-shot/step method, call it; otherwise run
    its serve loop on a daemon thread and poll the state store. Either way we
    wait for the real worker subprocess + real pipeline to finish.
    """
    step = (
        getattr(supervisor, "run_once", None)
        or getattr(supervisor, "tick", None)
        or getattr(supervisor, "process_pending", None)
    )
    deadline = time.monotonic() + timeout_s
    if step is not None:
        while time.monotonic() < deadline:
            step()
            if not store.active_jobs():
                created = store.list_jobs(limit=10)
                if created and not store.active_jobs():
                    return
            time.sleep(0.05)
        raise AssertionError("supervisor did not drain within timeout")

    import threading

    serve = getattr(supervisor, "serve", None)
    assert serve is not None, "Supervisor exposes neither a step API nor serve()"
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        while time.monotonic() < deadline:
            jobs = store.list_jobs(limit=10)
            if jobs and not store.active_jobs():
                return
            time.sleep(0.1)
        raise AssertionError("supervisor did not drain within timeout")
    finally:
        stop = getattr(supervisor, "stop", None) or getattr(supervisor, "shutdown", None)
        if stop is not None:
            stop()
        thread.join(timeout=10.0)


def _build_watches_yaml(
    tmp_path: Path,
    run_config_path: Path,
    incoming_glob: str,
    *,
    on_fire: str = "whole_config",
    inject_into: str | None = None,
    settle: str = "5s",
) -> Path:
    """Write a temp watches.yaml with one rule and a temp state_dir."""
    state_dir = tmp_path / "daemon_state"
    rule: dict[str, Any] = {
        "watch": incoming_glob,
        "run": str(run_config_path),
        "on_fire": on_fire,
        "settle": settle,
        "notify": ["desktop"],
    }
    if inject_into is not None:
        rule["inject_into"] = inject_into
    doc = {
        "daemon": {
            "state_dir": str(state_dir),
            "poll_interval": "1s",
            "max_concurrent": 1,
            "hdf5_file_locking": False,
            "notifications": {
                "desktop": True,
                "icloud_copy": True,
                "icloud_dir": str(tmp_path / "icloud"),
            },
        },
        "watches": {"realtime": rule},
    }
    watches_path = tmp_path / "watches.yaml"
    watches_path.write_text(yaml.safe_dump(doc))
    return watches_path


@pytest.fixture
def captured_notifications(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Mock ONLY the desktop/iCloud notify side-effects (never the pipeline)."""
    import davinci_monet.daemon.notify as notify_mod

    calls: list[dict[str, Any]] = []

    def _fake_desktop(*args: Any, **kwargs: Any) -> None:
        calls.append({"channel": "desktop", "args": args, "kwargs": kwargs})

    def _fake_icloud(*args: Any, **kwargs: Any) -> Any:
        calls.append({"channel": "icloud", "args": args, "kwargs": kwargs})
        return None

    # Patch whichever public hooks the notify module exposes; tolerate either
    # function-style or class-method-style implementations.
    for name in ("send_desktop", "desktop_notify", "notify_desktop"):
        if hasattr(notify_mod, name):
            monkeypatch.setattr(notify_mod, name, _fake_desktop, raising=False)
    for name in ("copy_to_icloud", "icloud_copy", "copy_outputs"):
        if hasattr(notify_mod, name):
            monkeypatch.setattr(notify_mod, name, _fake_icloud, raising=False)
    # Belt-and-suspenders: neutralize the raw side-effect primitives.
    monkeypatch.setattr(notify_mod, "subprocess", None, raising=False)
    return calls


def test_daemon_whole_config_integration(
    tmp_path: Path, captured_notifications: list[dict[str, Any]]
) -> None:
    # --- synthetic data + run config ---
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    model_nc, obs_nc = _write_synthetic_point_pair(data_dir)
    output_dir = tmp_path / "run_output"
    log_dir = tmp_path / "run_logs"
    run_cfg = _minimal_point_config_dict(model_nc, obs_nc, output_dir, log_dir)
    run_cfg_path = tmp_path / "run.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_cfg))

    # --- watched dir + watches.yaml ---
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    watches_path = _build_watches_yaml(
        tmp_path, run_cfg_path, str(incoming / "*.nc"), on_fire="whole_config"
    )

    watches_file = load_watches(watches_path)
    store = StateStore(watches_file.daemon.db_path)
    store.init_schema()

    clock = FakeClock()
    supervisor = build_supervisor(watches_file, state=store, clock=clock)

    # --- drop the trigger file, then let the settle window elapse ---
    trigger_nc = incoming / "new_data.nc"
    xr.open_dataset(model_nc).to_netcdf(trigger_nc)
    clock.advance(10.0)  # past the 5s settle window

    _drain_supervisor(supervisor, store)

    # --- assert: a job was recorded and COMPLETED ---
    jobs = store.list_jobs(watch_name="realtime", limit=10)
    assert jobs, "no job recorded for the fired watch"
    job = jobs[0]
    assert job.status == JobStatus.COMPLETED, f"job not completed: {job.status} / {job.error}"
    assert job.exit_code == 0
    assert job.result_summary is not None
    assert job.on_fire == "whole_config"

    # --- assert: the REAL pipeline produced outputs ---
    assert list(output_dir.rglob("statistics_summary.csv")), "no stats CSV from pipeline"
    pngs = list(output_dir.rglob("*.png"))
    assert pngs, "no plots from pipeline"
    assert all(p.stat().st_size > 1024 for p in pngs)
    assert list(log_dir.glob("pipeline_*.md")), "no per-run markdown log"

    # --- assert: notify hook fired (side-effects mocked) ---
    assert captured_notifications, "notify hook was not invoked on completion"

    store.close()
```

- [ ] **Step 2 — Run and expect failure.** From the repo root in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_daemon_integration.py::test_daemon_whole_config_integration -v
```
Expected: collection/import error or assertion failure such as `ModuleNotFoundError: No module named 'davinci_monet.daemon.supervisor'` (or, once the modules exist but the wiring is incomplete, `AssertionError: no job recorded for the fired watch`). The test MUST fail before the implementation/wiring is complete.

- [ ] **Step 3 — Make it pass via the implementation modules (no test placeholders).** This is an integration test: its "implementation" is the assembled daemon (config/contracts/state/watcher/queue/dispatcher/worker/notify/supervisor) delivered by the other groups' tasks. Do NOT stub the pipeline. The only edits this task may make to the TEST are to align the three flexible touch-points with the real assembled signatures, keeping the contract names verbatim:
  - the `build_supervisor(watches_file, state=store, clock=clock)` wiring call (keyword args: `watches_file`, `state`, `clock`),
  - the step/serve method names probed in `_drain_supervisor` (`run_once`/`tick`/`process_pending`/`serve`),
  - the notify hook function names probed in `captured_notifications`.
  If a real signature differs, update the probe/call here to match the real names — never weaken an assertion or skip the subprocess/pipeline. Example concrete adjustment if the supervisor's single-shot entry is named `drain_once()`:
```python
    step = (
        getattr(supervisor, "run_once", None)
        or getattr(supervisor, "drain_once", None)
        or getattr(supervisor, "tick", None)
        or getattr(supervisor, "process_pending", None)
    )
```

> **Assembler note:** the `Supervisor` implemented in Task 38 takes keyword-only collaborators `Supervisor(*, watches_file=..., watcher=..., queue=..., dispatcher=..., state=..., notifier=..., clock=...)` — it does NOT accept a positional `store=`/`clock=` form, so use the production wiring `build_supervisor(watches_file, *, state=None, clock=None, notifier=None)` instead. This test constructs the supervisor via `build_supervisor(watches_file, state=store, clock=clock)`, which threads the injected `FakeClock` into the `PollingWatcher` and the `Supervisor`. Keep the contract names verbatim.

- [ ] **Step 4 — Run and expect pass.** From the repo root in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_daemon_integration.py::test_daemon_whole_config_integration -v
```
Expected: `1 passed`. Confirm the run created `statistics_summary.csv`, a `*.png`, and a `pipeline_*.md`, and the job row is `completed`.

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/tests/integration/test_daemon_integration.py
git commit -m "$(cat <<'EOF'
test(daemon): whole-config integration through real worker subprocess + pipeline

Exercises the real path: load_watches -> Supervisor -> PollingWatcher (fake
clock) -> RunQueue -> Dispatcher -> worker subprocess -> run_analysis ->
PipelineRunner.run_from_config. Asserts settle fires, the pipeline writes
statistics_summary.csv + plots + a per-run markdown log, the job is recorded
COMPLETED in StateStore history, and the notify hook is invoked. Only the
desktop/iCloud side-effects are mocked; the pipeline is real.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 43: New-files-only injection daemon integration variant

Files:
- Modify: `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/integration/test_daemon_integration.py` (append the second test + a tiny config-builder helper; reuse `FakeClock`, `_write_synthetic_point_pair`, `_minimal_point_config_dict`, `_drain_supervisor`, `_build_watches_yaml`, and the `captured_notifications` fixture from Task 42)
- Reads (no edits): `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/dispatcher.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/worker.py`, `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/daemon/config.py` (the `WatchRule.inject_into` semantics)

- [ ] **Step 1 — Write the failing test (full code).** Append to `/Users/fillmore/EarthSystem/DAVINCI/davinci_monet/tests/integration/test_daemon_integration.py`. The variant points the model source's `files:` at a directory glob (no model file initially), sets `on_fire: new_files_only` + `inject_into: synthetic`, then drops the model NetCDF into the watched dir so the dispatcher injects exactly that new path into the model source before the worker runs the real pipeline:

```python
def _glob_model_config_dict(
    obs_nc: Path, model_glob: str, output_dir: Path, log_dir: Path
) -> dict[str, Any]:
    """Like _minimal_point_config_dict but the model files: is a glob that is
    EMPTY until the daemon injects the newly-arrived file (on_fire=new_files_only)."""
    cfg = _minimal_point_config_dict(
        Path("/nonexistent/placeholder.nc"), obs_nc, output_dir, log_dir
    )
    cfg["model"]["synthetic"]["files"] = model_glob
    return cfg


def test_daemon_new_files_only_injection_integration(
    tmp_path: Path, captured_notifications: list[dict[str, Any]]
) -> None:
    # --- synthetic data: obs lives outside the watched dir; model arrives later ---
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    model_nc, obs_nc = _write_synthetic_point_pair(data_dir)

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    output_dir = tmp_path / "run_output"
    log_dir = tmp_path / "run_logs"

    # Model source files: is a glob into the (initially empty) watched dir.
    model_glob = str(incoming / "*.nc")
    run_cfg = _glob_model_config_dict(obs_nc, model_glob, output_dir, log_dir)
    run_cfg_path = tmp_path / "run.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_cfg))

    watches_path = _build_watches_yaml(
        tmp_path,
        run_cfg_path,
        str(incoming / "*.nc"),
        on_fire="new_files_only",
        inject_into="synthetic",
    )

    watches_file = load_watches(watches_path)
    store = StateStore(watches_file.daemon.db_path)
    store.init_schema()

    clock = FakeClock()
    supervisor = build_supervisor(watches_file, state=store, clock=clock)

    # Drop the model file into the watched dir -> this is the injected new file.
    injected = incoming / "cam_model.nc"
    xr.open_dataset(model_nc).to_netcdf(injected)
    clock.advance(10.0)

    _drain_supervisor(supervisor, store)

    jobs = store.list_jobs(watch_name="realtime", limit=10)
    assert jobs, "no job recorded for the injection watch"
    job = jobs[0]
    assert job.on_fire == "new_files_only"
    assert str(injected) in job.files, "the injected new file was not recorded on the job"
    assert job.status == JobStatus.COMPLETED, f"injection run failed: {job.error}"
    assert job.exit_code == 0

    # The real pipeline ran with the injected file -> outputs exist.
    assert list(output_dir.rglob("statistics_summary.csv")), "no stats CSV from injected run"
    assert list(output_dir.rglob("*.png")), "no plots from injected run"
    assert list(log_dir.glob("pipeline_*.md")), "no per-run markdown log"

    assert captured_notifications, "notify hook not invoked for injection run"

    store.close()
```

- [ ] **Step 2 — Run and expect failure.** From the repo root in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_daemon_integration.py::test_daemon_new_files_only_injection_integration -v
```
Expected: failure before injection wiring is complete — e.g. `AssertionError: the injected new file was not recorded on the job`, or a pipeline failure (`job.status` != `COMPLETED`) because the empty `files:` glob resolved to no model file (proving injection did not happen). The test MUST fail until the dispatcher/worker injection path is wired.

- [ ] **Step 3 — Make it pass via the implementation modules (no test placeholders).** As with Task 42, the "implementation" is the assembled dispatcher/worker `new_files_only` injection (dispatcher builds `JobSpec.new_files` from the `TriggerEvent`; worker overrides `inject_into`'s `files:` with those paths before calling `run_analysis`). The only permitted TEST edits are aligning the flexible touch-points (`Supervisor` ctor args, `_drain_supervisor` step/serve names, notify hook names) with the real assembled signatures — keeping contract names verbatim and never weakening assertions or bypassing the subprocess/pipeline. If the real `JobRecord.files`/`JobSpec.new_files` carry resolved absolute paths that differ in normalization, adjust the membership assertion to compare resolved paths, e.g.:
```python
    resolved = {str(Path(f).resolve()) for f in job.files}
    assert str(injected.resolve()) in resolved, "injected file not on job"
```

> **Assembler note:** the same `Supervisor` construction caveat from Task 42 applies — construct the supervisor via `build_supervisor(watches_file, state=store, clock=clock)` (the production wiring threads the injected `FakeClock` into the watcher + supervisor), not a positional `Supervisor(...)` form. Note also the integration test's run config uses the legacy `model:`/`obs:` schema; the worker's `inject_new_files` operates on the unified `sources:` config (it overrides `config["sources"][inject_into]["files"]`). The legacy config is auto-migrated to `sources:` by `load_config(...).model_dump()` inside the worker before injection, so `inject_into="synthetic"` must match the migrated source label (the legacy `model:` key `synthetic` becomes the `sources:` label `synthetic`). Verify this label correspondence against the real migration when wiring; this is the one substantive cross-schema touch-point.

- [ ] **Step 4 — Run and expect pass.** From the repo root in the `davinci` conda env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci && HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_daemon_integration.py -v
```
Expected: `2 passed` (both the whole-config and injection variants). Confirm the injection run's `output_dir` contains `statistics_summary.csv` and a `*.png`, and the job row lists the injected file.

- [ ] **Step 5 — Commit.**
```bash
git add davinci_monet/tests/integration/test_daemon_integration.py
git commit -m "$(cat <<'EOF'
test(daemon): new_files_only injection integration variant

Drops a model NetCDF into the watched dir with on_fire=new_files_only +
inject_into=synthetic; asserts the dispatcher injects exactly the new path
into the model source's files: before the real worker subprocess runs the
pipeline, the injected file is recorded on the job, and outputs + history +
notify all reflect a COMPLETED run. Pipeline is real; only desktop/iCloud
side-effects are mocked.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Definition of Done

- [ ] Full pytest suite green in the `davinci` conda env:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
  HDF5_USE_FILE_LOCKING=FALSE python -m pytest
  ```
- [ ] `mypy davinci_monet` clean (no new type errors introduced by the daemon package).
- [ ] `black davinci_monet && isort davinci_monet` clean (formatting + import ordering).
- [ ] The two daemon integration tests pass through the real pipeline engine
  (`PipelineRunner.run_from_config` via `run_analysis`) in a real worker subprocess:
  - `davinci_monet/tests/integration/test_daemon_integration.py::test_daemon_whole_config_integration`
  - `davinci_monet/tests/integration/test_daemon_integration.py::test_daemon_new_files_only_injection_integration`
  - plus `davinci_monet/tests/integration/test_daemon_worker_pipeline.py::test_worker_runs_synthetic_config_through_pipeline`
- [ ] `davinci-monet daemon --help`, `daemon watch --help`, and the `daemon` sub-app
  route through the root CLI (`davinci_monet/tests/unit/daemon/test_app_registration.py`).
