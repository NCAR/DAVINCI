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
