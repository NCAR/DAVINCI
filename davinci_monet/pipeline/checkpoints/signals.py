"""Temporary signal handling for safely interruptible pipeline executions."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any


class PipelineInterrupted(KeyboardInterrupt):
    """Raised outside the signal handler's durable-I/O boundary."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"pipeline interrupted by {signal.Signals(signum).name}")


@contextmanager
def interruption_signals() -> Iterator[None]:
    """Convert SIGINT/SIGTERM to an exception and restore prior handlers."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    signums = (signal.Signals.SIGINT, signal.Signals.SIGTERM)
    previous: dict[signal.Signals, Any] = {}

    def handle(signum: int, _frame: FrameType | None) -> None:
        raise PipelineInterrupted(signum)

    try:
        for signum in signums:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handle)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
