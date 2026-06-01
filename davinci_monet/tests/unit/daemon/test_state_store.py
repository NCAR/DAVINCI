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
        cur = store._conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
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
