"""Daemon process lifecycle: PID+lock, double-fork, signals, graceful drain.

Supervisor-side ONLY. Stdlib only; never imports the scientific stack.
"""

from __future__ import annotations

import fcntl
import os
import signal
import threading
import time
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


def install_signal_handlers(
    on_drain: OnDrainFn,
    *,
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
