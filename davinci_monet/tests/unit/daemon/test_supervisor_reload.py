"""Unit tests for the Supervisor `reload` control command.

`reload` re-reads watches.yaml from disk and reconciles the freshly-declared
rules against the runtime state (live rules + paused names) using
``config.merge_rules`` and the StateStore watch-status APIs, then pushes the new
rule set into the live PollingWatcher via ``set_rules``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from davinci_monet.daemon.config import DaemonConfig, WatchesFile, WatchRule, load_watches
from davinci_monet.daemon.state import StateStore


class FakeClock:
    def __init__(self) -> None:
        self._t = 100.0

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds


class FakeQueue:
    def submit(self, event) -> bool:  # pragma: no cover - unused in reload tests
        return False

    def pop(self):  # pragma: no cover - unused in reload tests
        return None

    def __len__(self) -> int:
        return 0


class RecordingWatcher:
    """Watcher stand-in that records the rule sets passed to set_rules."""

    def __init__(self) -> None:
        self.set_rules_calls: list[list[str]] = []

    def poll(self):  # pragma: no cover - unused in reload tests
        return []

    def set_rules(self, rules: list[WatchRule]) -> None:
        self.set_rules_calls.append([r.name for r in rules])


def _write_watches(path: Path, *, names: list[str], state_dir: Path) -> None:
    blocks = "\n".join(
        textwrap.dedent(
            f"""\
            {name}:
              watch: /tmp/{name}/*.nc
              run: /tmp/{name}.yaml
            """
        )
        for name in names
    )
    watches_body = textwrap.indent(blocks, "  ")
    path.write_text(
        textwrap.dedent(
            f"""\
            daemon:
              state_dir: {state_dir}
            watches:
            """
        )
        + watches_body
    )


def _build_sup(watches_path: Path, watcher: RecordingWatcher, state: StateStore):
    from davinci_monet.daemon.supervisor import Supervisor

    wf = load_watches(watches_path)
    return Supervisor(
        watches_file=wf,
        watcher=watcher,
        queue=FakeQueue(),
        dispatcher=lambda spec, cfg: None,
        state=state,
        notifier=None,
        clock=FakeClock(),
        watches_path=watches_path,
    )


def test_reload_adds_new_declared_rule(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    watches_path = tmp_path / "watches.yaml"
    _write_watches(watches_path, names=["cam"], state_dir=state_dir)

    state = StateStore(state_dir / "history.db")
    watcher = RecordingWatcher()
    sup = _build_sup(watches_path, watcher, state)

    assert set(sup.rules) == {"cam"}

    # Edit the file on disk: add a brand-new declared rule.
    _write_watches(watches_path, names=["cam", "modis"], state_dir=state_dir)

    resp = sup.handle_command("reload", {})

    assert resp.ok is True
    # Active rule dict updated to reflect the new declaration.
    assert set(sup.rules) == {"cam", "modis"}
    # The live watcher was handed the new rule set.
    assert watcher.set_rules_calls, "set_rules must be called on reload"
    assert set(watcher.set_rules_calls[-1]) == {"cam", "modis"}
    # Response summarizes the reconciliation.
    assert "modis" in resp.data["added"]
    assert resp.data["removed"] == []


def test_reload_removes_dropped_declared_rule(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    watches_path = tmp_path / "watches.yaml"
    _write_watches(watches_path, names=["cam", "modis"], state_dir=state_dir)

    state = StateStore(state_dir / "history.db")
    watcher = RecordingWatcher()
    sup = _build_sup(watches_path, watcher, state)
    assert set(sup.rules) == {"cam", "modis"}

    # Edit the file on disk: drop the modis rule.
    _write_watches(watches_path, names=["cam"], state_dir=state_dir)

    resp = sup.handle_command("reload", {})

    assert resp.ok is True
    assert set(sup.rules) == {"cam"}
    assert set(watcher.set_rules_calls[-1]) == {"cam"}
    assert "modis" in resp.data["removed"]
    assert resp.data["added"] == []


def test_reload_preserves_runtime_pause(tmp_path: Path) -> None:
    """A watch paused at runtime stays paused across a reload."""
    state_dir = tmp_path / "state"
    watches_path = tmp_path / "watches.yaml"
    _write_watches(watches_path, names=["cam"], state_dir=state_dir)

    state = StateStore(state_dir / "history.db")
    watcher = RecordingWatcher()
    sup = _build_sup(watches_path, watcher, state)

    # Pause cam at runtime (persists enabled=False to the state store).
    sup.handle_command("watch_pause", {"name": "cam"})
    assert sup.rules["cam"].enabled is False

    resp = sup.handle_command("reload", {})
    assert resp.ok is True
    # Even though watches.yaml declares cam as enabled, the runtime pause wins.
    assert sup.rules["cam"].enabled is False


def test_reload_without_path_returns_error(tmp_path: Path) -> None:
    """A Supervisor built without a watches_path cannot reload; returns an error."""
    from davinci_monet.daemon.supervisor import Supervisor

    wf = WatchesFile(
        daemon=DaemonConfig(state_dir=tmp_path / "state"),
        watches={"cam": WatchRule(name="cam", watch="/tmp/cam/*.nc", run="/tmp/cam.yaml")},
    )
    state = StateStore(tmp_path / "state" / "history.db")
    sup = Supervisor(
        watches_file=wf,
        watcher=RecordingWatcher(),
        queue=FakeQueue(),
        dispatcher=lambda spec, cfg: None,
        state=state,
        notifier=None,
        clock=FakeClock(),
        watches_path=None,
    )

    resp = sup.handle_command("reload", {})
    assert resp.ok is False
    assert resp.code in {"unsupported", "invalid_args"}
