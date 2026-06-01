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

    @staticmethod
    def _row_to_watch_status(row: sqlite3.Row) -> WatchStatusRecord:
        return WatchStatusRecord(
            watch_name=row["watch_name"],
            enabled=bool(row["enabled"]),
            source=row["source"],
            rule_json=_loads(row["rule_json"], None),
            updated_at=_parse_dt(row["updated_at"]),
        )
