"""Unit tests for davinci_monet.daemon.watcher.PollingWatcher.

Deterministic: an injectable monotonic Clock and an injectable filesystem scan
(no real sleeping, no real disk needed for the settle/sentinel logic).
"""

from __future__ import annotations

import os
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


def test_sentinel_rule_fires_only_when_marker_appears(tmp_path) -> None:
    clock = FakeClock()
    marker = tmp_path / "DELIVERED"
    f = "/data/modis/a.hdf"
    # scan always returns one matching data file; firing is gated on the marker.
    scan = FakeScan([{f: FileStat(size=10, mtime=1.0)}])
    rule = _rule(
        name="modis",
        watch="/data/modis/*.hdf",
        sentinel=str(marker),
        settle=30.0,
    )
    assert rule.settle_mode == "sentinel"
    w = PollingWatcher(rules=[rule], config=DaemonConfig(), clock=clock, scan=scan)

    # Marker absent: never fires, no matter how much time passes.
    assert w.poll() == []
    clock.advance(10_000.0)
    assert w.poll() == []

    # Marker appears -> fires exactly once with settle_mode="sentinel".
    marker.write_text("ok")
    events = w.poll()
    assert len(events) == 1
    assert events[0].watch_name == "modis"
    assert events[0].settle_mode == "sentinel"
    assert events[0].new_files == [f]

    # Marker still present -> does NOT re-fire.
    assert w.poll() == []

    # Marker removed then re-delivered -> re-arms and fires again.
    os.remove(marker)
    assert w.poll() == []
    marker.write_text("ok2")
    events2 = w.poll()
    assert len(events2) == 1
    assert events2[0].settle_mode == "sentinel"


def test_max_settle_wait_force_fires_growing_file() -> None:
    clock = FakeClock()
    f = "/data/grow.nc"
    # File grows on EVERY poll, so it never quiesces on its own.
    snapshots = [{f: FileStat(size=100 * (i + 1), mtime=float(i))} for i in range(20)]
    scan = FakeScan(snapshots)
    # settle is large (never reached); max_settle_wait is the only thing that fires.
    cfg = DaemonConfig(max_settle_wait=100.0)
    w = PollingWatcher(rules=[_rule(settle=1000.0)], config=cfg, clock=clock, scan=scan)

    assert w.poll() == []  # first_seen recorded here.
    # Grow for 90s total (< 100s valve) -> still no fire despite constant growth.
    for _ in range(3):
        clock.advance(30.0)
        assert w.poll() == []
    # Cross the 100s valve while still growing -> force-fires once.
    clock.advance(30.0)
    events = w.poll()
    assert len(events) == 1
    assert events[0].watch_name == "w"
    assert events[0].settle_mode == "quiescence"


def test_max_settle_wait_none_never_force_fires() -> None:
    clock = FakeClock()
    f = "/data/grow.nc"
    snapshots = [{f: FileStat(size=100 * (i + 1), mtime=float(i))} for i in range(20)]
    scan = FakeScan(snapshots)
    cfg = DaemonConfig(max_settle_wait=None)  # valve disabled.
    w = PollingWatcher(rules=[_rule(settle=1000.0)], config=cfg, clock=clock, scan=scan)

    assert w.poll() == []
    for _ in range(10):
        clock.advance(10_000.0)
        assert w.poll() == []  # forever-growing + no valve -> never fires.


def test_second_batch_reanchors_max_settle_after_fire() -> None:
    # Regression: the max_settle_wait valve must measure from the START of the
    # current pending batch, NOT from the first file the rule ever saw. Otherwise
    # a new (still-growing) batch that arrives long after an earlier fire is
    # force-fired immediately, because first_seen_t is stale from batch 1.
    clock = FakeClock(start=1000.0)
    a = "/data/a.nc"
    b = "/data/b.nc"
    scan = FakeScan(
        [
            {a: FileStat(size=100, mtime=1.0)},  # poll0: a appears
            {a: FileStat(size=100, mtime=1.0)},  # poll1: a stable -> fires batch 1
            {
                a: FileStat(size=100, mtime=1.0),
                b: FileStat(size=100, mtime=2.0),
            },  # poll2: b appears
            {a: FileStat(size=100, mtime=1.0), b: FileStat(size=200, mtime=3.0)},  # poll3: b grows
            {a: FileStat(size=100, mtime=1.0), b: FileStat(size=300, mtime=4.0)},  # poll4: b grows
            {a: FileStat(size=100, mtime=1.0), b: FileStat(size=400, mtime=5.0)},  # poll5: b grows
            {a: FileStat(size=100, mtime=1.0), b: FileStat(size=500, mtime=6.0)},  # poll6: b grows
        ]
    )
    cfg = DaemonConfig(max_settle_wait=100.0)
    w = PollingWatcher(rules=[_rule(settle=30.0)], config=cfg, clock=clock, scan=scan)

    assert w.poll() == []  # poll0: a just appeared
    clock.advance(40.0)
    batch1 = w.poll()  # poll1: a stable for 40s -> fires
    assert len(batch1) == 1
    assert batch1[0].new_files == [a]

    # 200s later (>> the 100s valve measured from a's first-seen) a NEW file appears.
    clock.advance(200.0)
    assert w.poll() == []  # poll2: b's batch must NOT inherit a's start time
    # b keeps growing; it must not force-fire until 100s from ITS OWN start (t=1240).
    clock.advance(30.0)
    assert w.poll() == []  # poll3: t=1270, 30s into b's batch
    clock.advance(30.0)
    assert w.poll() == []  # poll4: t=1300, 60s
    clock.advance(30.0)
    assert w.poll() == []  # poll5: t=1330, 90s
    clock.advance(30.0)
    batch2 = w.poll()  # poll6: t=1360, 120s >= 100s valve -> force-fires
    assert len(batch2) == 1
    assert batch2[0].new_files == [b]
    assert batch2[0].settle_mode == "quiescence"
