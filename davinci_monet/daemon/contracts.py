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
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from pydantic import ConfigDict, Field

# Reuse the project's pydantic base classes verbatim — DO NOT redefine them.
# Importing from the submodule directly avoids triggering the heavy geo/data
# stack that davinci_monet/config/__init__.py would otherwise pull in eagerly.
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

    def to_json(self) -> str:
        """Serialize to a compact JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> "JobSpec":
        """Deserialize from a JSON string."""
        return cls.model_validate_json(raw)


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


@runtime_checkable
class StateStore(Protocol):
    """SQLite-backed job history + watch runtime-status persistence.

    stdlib sqlite3 only. Single connection, check_same_thread=False, accessed
    only from the supervisor thread/loop. Conforming implementations must create
    or open the DB and apply the DDL (see ``SCHEMA_DDL`` below) before any
    other method is called. All timestamps written as ISO-8601
    (datetime.isoformat()); all dict/list fields json.dumps'd.
    """

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

    def add_live_rule(self, rule: Any) -> None:
        """Persist a runtime-added rule (source="live", rule_json=rule dump)."""
        ...

    def remove_watch(self, watch_name: str) -> None:
        """Delete the watch_status row (for live rules) / drop runtime overrides."""
        ...

    def disabled_names(self) -> set[str]:
        """Names with enabled=False. Fed to merge_rules() on reload."""
        ...

    def live_rules(self) -> dict[str, Any]:
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
        try:
            data = json.loads(line)
            if not isinstance(data, dict) or data.get("kind") not in PROGRESS_KINDS:
                return None
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return None


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

    def now(self) -> float: ...  # monotonic seconds

    def sleep(self, seconds: float) -> None: ...
