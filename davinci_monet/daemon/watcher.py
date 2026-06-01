"""Polling file-watcher for the DAVINCI daemon.

Periodic stat+glob of each WatchRule's pattern. A rule fires a TriggerEvent
when its matching files quiesce (no new/modified files and stable size for the
rule's settle window) or, for sentinel rules, when the marker path appears.

Settle/quiescence math uses an injectable monotonic ``Clock``; the filesystem
scan is injectable too, so the firing logic is fully deterministic under test.
The supervisor calls ``poll()`` once per ``poll_interval``; ``poll()`` itself
never sleeps.
"""

from __future__ import annotations

import glob
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import Clock, TriggerEvent


@dataclass(frozen=True)
class FileStat:
    """Size + mtime snapshot of a single matched file."""

    size: int
    mtime: float


# A scan function maps a glob pattern -> {absolute_path: FileStat}.
ScanFn = Callable[[str], "dict[str, FileStat]"]


def default_scan(pattern: str) -> dict[str, FileStat]:
    """Production scan: glob the pattern and os.stat each match.

    Returns absolute paths only. Files that vanish between glob and stat are
    skipped silently (race-tolerant on busy scratch filesystems).
    """
    out: dict[str, FileStat] = {}
    for path in glob.glob(pattern):
        abspath = os.path.abspath(path)
        try:
            st = os.stat(abspath)
        except (FileNotFoundError, OSError):
            continue
        if os.path.isdir(abspath):
            continue
        out[abspath] = FileStat(size=st.st_size, mtime=st.st_mtime)
    return out


class RealClock:
    """Production monotonic clock (conforms to the contracts.Clock Protocol)."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass
class _RuleState:
    """Per-rule tracking carried across polls."""

    tracked: dict[str, FileStat] = field(default_factory=dict)
    last_change_t: Optional[float] = None  # monotonic time of last new/size change
    first_seen_t: Optional[float] = None  # monotonic time the rule first had a match
    settled: set[str] = field(default_factory=set)  # last batch already fired
    sentinel_fired: bool = False  # sentinel marker already consumed


class PollingWatcher:
    """Stat+glob watcher; one ``poll()`` per ``poll_interval`` tick.

    Parameters
    ----------
    rules
        The active WatchRules to monitor.
    config
        DaemonConfig (supplies ``max_settle_wait``).
    clock
        Monotonic clock (contracts.Clock); injected for deterministic tests.
    scan
        Filesystem scan function; injected for deterministic tests. Defaults to
        ``default_scan``.
    """

    def __init__(
        self,
        rules: list[WatchRule],
        config: DaemonConfig,
        clock: Clock,
        scan: ScanFn | None = None,
    ) -> None:
        self._config = config
        self._clock: Clock = clock
        self._scan: ScanFn = scan or default_scan
        self._state: dict[str, _RuleState] = {}
        self.set_rules(rules)

    # ---- rule set management ---------------------------------------------
    def set_rules(self, rules: list[WatchRule]) -> None:
        """Replace the active rule set, preserving state for surviving rules."""
        self._rules: dict[str, WatchRule] = {r.name: r for r in rules}
        for name in list(self._state):
            if name not in self._rules:
                del self._state[name]
        for name in self._rules:
            self._state.setdefault(name, _RuleState())

    # ---- the poll tick ----------------------------------------------------
    def poll(self) -> list[TriggerEvent]:
        """Scan every rule once and return the events that fired this tick."""
        events: list[TriggerEvent] = []
        for name, rule in self._rules.items():
            ev = self._poll_rule(rule, self._state[name])
            if ev is not None:
                events.append(ev)
        return events

    def _poll_rule(self, rule: WatchRule, st: _RuleState) -> Optional[TriggerEvent]:
        now = self._clock.now()
        if rule.settle_mode == "sentinel":
            return self._poll_sentinel(rule, st, now)
        return self._poll_quiescence(rule, st, now)

    # ---- quiescence detection --------------------------------------------
    def _poll_quiescence(
        self, rule: WatchRule, st: _RuleState, now: float
    ) -> Optional[TriggerEvent]:
        current = self._scan(rule.watch)
        if not current:
            # nothing matches: reset transient timing but keep settled history.
            st.tracked = {}
            st.last_change_t = None
            st.first_seen_t = None
            return None

        changed = self._detect_change(st.tracked, current)
        st.tracked = current
        if changed or st.last_change_t is None:
            st.last_change_t = now

        current_set = set(current)
        if current_set == st.settled:
            # identical to the last-fired batch: no pending batch is settling.
            st.first_seen_t = None
            return None  # do not re-fire.

        # A pending (not-yet-fired) batch exists. Anchor the max_settle_wait
        # valve to when THIS batch first diverged from the fired set, so a later
        # batch is never force-fired using an earlier batch's start time.
        if st.first_seen_t is None:
            st.first_seen_t = now

        quiesced = (now - st.last_change_t) >= rule.settle
        forced = self._max_settle_exceeded(st, now)
        if quiesced or forced:
            return self._fire(rule, st, current, "quiescence")
        return None

    @staticmethod
    def _detect_change(prev: dict[str, FileStat], cur: dict[str, FileStat]) -> bool:
        """True if any file appeared, vanished, or changed size vs prev poll."""
        if set(prev) != set(cur):
            return True
        for path, stat in cur.items():
            if prev[path].size != stat.size:
                return True
        return False

    def _max_settle_exceeded(self, st: _RuleState, now: float) -> bool:
        limit = self._config.max_settle_wait
        if limit is None or st.first_seen_t is None:
            return False
        return (now - st.first_seen_t) >= limit

    # ---- sentinel detection ----------------------------------------------
    def _poll_sentinel(self, rule: WatchRule, st: _RuleState, now: float) -> Optional[TriggerEvent]:
        assert rule.sentinel is not None
        marker = os.path.abspath(rule.sentinel)
        present = self._marker_present(marker)
        if not present:
            st.sentinel_fired = False  # re-arm once the marker is removed.
            return None
        if st.sentinel_fired:
            return None
        current = self._scan(rule.watch)
        st.tracked = current
        st.sentinel_fired = True
        return self._fire(rule, st, current, "sentinel")

    def _marker_present(self, marker: str) -> bool:
        """Sentinel presence via ``os.path.exists``; returns False on OSError."""
        try:
            return os.path.exists(marker)
        except OSError:
            return False

    # ---- firing ----------------------------------------------------------
    def _fire(
        self,
        rule: WatchRule,
        st: _RuleState,
        current: dict[str, FileStat],
        mode: str,
    ) -> TriggerEvent:
        new = sorted(p for p in current if p not in st.settled)
        if not new:
            new = sorted(current)
        st.settled = set(current)
        st.first_seen_t = None  # re-anchor the valve for the NEXT pending batch.
        return TriggerEvent(
            watch_name=rule.name,
            new_files=new,
            detected_at=datetime.now(),
            settle_mode=mode,  # type: ignore[arg-type]
        )
