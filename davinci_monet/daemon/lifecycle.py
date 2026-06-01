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
        super().__init__(f"DAVINCI daemon already running (pid {pid}); refusing to start")
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
