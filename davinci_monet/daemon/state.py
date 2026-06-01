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

from davinci_monet.daemon.contracts import (
    SCHEMA_DDL,
    JobRecord,
    JobStatus,
    OnFireMode,
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


def _require_dt(value: Any) -> datetime:
    """Like ``_parse_dt`` but raises if the result is None (NOT NULL columns)."""
    result = _parse_dt(value)
    if result is None:
        raise ValueError(f"Expected a non-null datetime, got {value!r}")
    return result


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
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
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
        last = cur.lastrowid
        assert last is not None, "INSERT did not return a rowid"
        return int(last)

    def get_job(self, job_id: int) -> Optional[JobRecord]:
        """Return the JobRecord for ``job_id`` or None if it does not exist."""
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def _duration_since_started(self, job_id: int, ended_iso: str) -> Optional[float]:
        """Compute ended-minus-started in seconds, or None if no start time."""
        row = self._conn.execute("SELECT started_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
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
            submitted_at=_require_dt(row["submitted_at"]),
            started_at=_parse_dt(row["started_at"]),
            ended_at=_parse_dt(row["ended_at"]),
            duration_s=row["duration_s"],
            exit_code=row["exit_code"],
            log_path=row["log_path"],
            result_summary=_loads(row["result_summary"], None),
            error=row["error"],
        )
