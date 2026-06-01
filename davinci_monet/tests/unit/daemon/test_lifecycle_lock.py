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
