"""Unit tests for the drain flag + signal handler wiring."""

from __future__ import annotations

import signal

from davinci_monet.daemon.lifecycle import DrainController, install_signal_handlers


def test_drain_flag_starts_clear_then_sets() -> None:
    ctrl = DrainController()
    assert ctrl.draining is False
    ctrl.request_drain()
    assert ctrl.draining is True


def test_request_drain_is_idempotent_and_records_reason() -> None:
    ctrl = DrainController()
    ctrl.request_drain(reason="SIGTERM")
    ctrl.request_drain(reason="second")
    assert ctrl.draining is True
    # First reason wins (records why drain started).
    assert ctrl.reason == "SIGTERM"


def test_wait_for_idle_returns_immediately_when_count_zero() -> None:
    ctrl = DrainController()
    # in_flight() reports zero workers -> drain completes instantly.
    completed = ctrl.wait_for_idle(in_flight=lambda: 0, timeout=0.5, poll=0.01)
    assert completed is True


def test_wait_for_idle_times_out_when_worker_never_finishes() -> None:
    ctrl = DrainController()
    completed = ctrl.wait_for_idle(in_flight=lambda: 1, timeout=0.05, poll=0.01)
    assert completed is False


def test_wait_for_idle_returns_true_when_worker_finishes() -> None:
    state = {"n": 2}

    def in_flight() -> int:
        # Drain one each poll until zero.
        if state["n"] > 0:
            state["n"] -= 1
        return state["n"]

    ctrl = DrainController()
    completed = ctrl.wait_for_idle(in_flight=in_flight, timeout=1.0, poll=0.001)
    assert completed is True


def test_install_signal_handlers_calls_on_drain_with_signal_name() -> None:
    # install_signal_handlers takes a bare Callable[[str], None]; the handler
    # must invoke it with the signal NAME (e.g. "SIGTERM").
    received: list[str] = []
    previous = install_signal_handlers(received.append)
    try:
        # Invoke the installed SIGTERM handler directly (no real signal).
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert received == ["SIGTERM"]
    finally:
        # Restore prior handlers so we don't leak into other tests.
        # previous[...] is typed object (the stored signal handler); the
        # restore is valid at runtime but mypy can't narrow object here.
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])  # type: ignore[arg-type]
        signal.signal(signal.SIGINT, previous[signal.SIGINT])  # type: ignore[arg-type]


def test_install_signal_handlers_accepts_drain_controller() -> None:
    # A DrainController.request_drain is itself a Callable[[str], None].
    ctrl = DrainController()
    previous = install_signal_handlers(ctrl.request_drain)
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert ctrl.draining is True
        assert ctrl.reason == "SIGINT"
    finally:
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])  # type: ignore[arg-type]
        signal.signal(signal.SIGINT, previous[signal.SIGINT])  # type: ignore[arg-type]
