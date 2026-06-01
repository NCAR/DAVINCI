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
    ] or hooks.events[
        :3
    ] == ["fork", "setsid", "fork"]


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
