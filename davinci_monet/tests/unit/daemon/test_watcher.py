"""Unit tests for davinci_monet.daemon.watcher.PollingWatcher.

Deterministic: an injectable monotonic Clock and an injectable filesystem scan
(no real sleeping, no real disk needed for the settle/sentinel logic).
"""

from __future__ import annotations

from datetime import datetime

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import TriggerEvent
from davinci_monet.daemon.watcher import FileStat, PollingWatcher


class FakeClock:
    """Injectable monotonic clock; advance() moves time forward."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:  # pragma: no cover - not used in tests
        self._t += seconds

    def advance(self, seconds: float) -> None:
        self._t += seconds


class FakeScan:
    """Scripted scan: returns the next queued snapshot for a glob pattern.

    snapshots is a list of {abs_path: FileStat}; each poll() consumes one.
    The last snapshot repeats once the script is exhausted (steady state).
    """

    def __init__(self, snapshots: list[dict[str, FileStat]]) -> None:
        self._snapshots = list(snapshots)
        self._i = 0

    def __call__(self, pattern: str) -> dict[str, FileStat]:
        snap = self._snapshots[min(self._i, len(self._snapshots) - 1)]
        self._i += 1
        return dict(snap)


def _rule(**kw: object) -> WatchRule:
    base = {"name": "w", "watch": "/data/*.nc", "run": "/cfg.yaml", "settle": 30.0}
    base.update(kw)
    return WatchRule(**base)  # type: ignore[arg-type]


def test_new_file_then_quiescence_fires_once() -> None:
    clock = FakeClock()
    f = "/data/a.nc"
    # tick0: file appears (size 100). tick1: same size (stable). steady state: same.
    scan = FakeScan(
        [
            {f: FileStat(size=100, mtime=1.0)},
            {f: FileStat(size=100, mtime=1.0)},
        ]
    )
    w = PollingWatcher(
        rules=[_rule(settle=30.0)],
        config=DaemonConfig(),
        clock=clock,
        scan=scan,
    )

    # tick0: file just appeared -> last_change set now; settle window not elapsed.
    assert w.poll() == []
    # advance only 10s (< 30s settle) -> still not fired.
    clock.advance(10.0)
    assert w.poll() == []
    # advance past the settle window with no size change -> fires exactly once.
    clock.advance(25.0)
    events = w.poll()
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, TriggerEvent)
    assert ev.watch_name == "w"
    assert ev.new_files == [f]
    assert ev.settle_mode == "quiescence"
    assert isinstance(ev.detected_at, datetime)
    # subsequent polls of the SAME settled set do not re-fire.
    clock.advance(100.0)
    assert w.poll() == []


def test_growing_file_does_not_fire_until_stable() -> None:
    clock = FakeClock()
    f = "/data/big.nc"
    scan = FakeScan(
        [
            {f: FileStat(size=100, mtime=1.0)},  # tick0 appears
            {f: FileStat(size=200, mtime=2.0)},  # tick1 grew
            {f: FileStat(size=300, mtime=3.0)},  # tick2 grew
            {f: FileStat(size=300, mtime=3.0)},  # tick3 stable
        ]
    )
    w = PollingWatcher(rules=[_rule(settle=30.0)], config=DaemonConfig(), clock=clock, scan=scan)

    assert w.poll() == []  # tick0 appears
    clock.advance(40.0)
    assert w.poll() == []  # tick1: grew -> resets settle timer, no fire despite 40s
    clock.advance(40.0)
    assert w.poll() == []  # tick2: grew again -> resets again
    clock.advance(40.0)
    events = w.poll()  # tick3: stable AND 40s elapsed since last change -> fires
    assert len(events) == 1
    assert events[0].new_files == [f]
    assert events[0].settle_mode == "quiescence"
