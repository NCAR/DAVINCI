"""Unit tests for Supervisor.handle_command (control dispatch table)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pytest

from davinci_monet.daemon.config import DaemonConfig, WatchesFile, WatchRule
from davinci_monet.daemon.contracts import JobStatus


class FakeClock:
    def __init__(self) -> None:
        self._t = 100.0

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds


class FakeStateStore:
    def __init__(self) -> None:
        self._active: list[Any] = []
        self._jobs: dict[int, Any] = {}
        self._history: list[Any] = []
        self.enabled_calls: list[tuple[str, bool]] = []

    def active_jobs(self):
        return list(self._active)

    def list_jobs(self, watch_name=None, status=None, limit=50):
        return list(self._history)[:limit]

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def set_enabled(self, watch_name, enabled):
        self.enabled_calls.append((watch_name, enabled))


class FakeQueue:
    def __init__(self) -> None:
        self.submitted: list[Any] = []

    def submit(self, event) -> bool:
        coalesced = bool(self.submitted)
        self.submitted.append(event)
        return coalesced

    def pop(self):
        return None

    def __len__(self) -> int:
        return 0


class FakeWatcher:
    def poll(self):
        return []


def _sup(**overrides):
    from davinci_monet.daemon.supervisor import Supervisor

    wf = WatchesFile(
        daemon=DaemonConfig(max_concurrent=2),
        watches={
            "cam": WatchRule(name="cam", watch="/tmp/cam/*.nc", run="/tmp/cam.yaml"),
            "modis": WatchRule(
                name="modis",
                watch="/tmp/modis/*.hdf",
                run="/tmp/modis.yaml",
                on_fire="new_files_only",
                inject_into="modis",
                sentinel="/tmp/modis/DELIVERED",
            ),
        },
    )
    kwargs: dict[str, Any] = dict(
        watches_file=wf,
        watcher=FakeWatcher(),
        queue=FakeQueue(),
        dispatcher=lambda spec, cfg: None,
        state=FakeStateStore(),
        notifier=None,
        clock=FakeClock(),
    )
    kwargs.update(overrides)
    return Supervisor(**kwargs)


def test_ping_returns_version_and_uptime() -> None:
    from davinci_monet.daemon.contracts import PROTOCOL_VERSION

    sup = _sup()
    resp = sup.handle_command("ping", {})
    assert resp.ok is True
    assert resp.data["pong"] is True
    assert resp.data["version"] == PROTOCOL_VERSION
    assert "uptime_s" in resp.data


def test_watch_list_returns_summaries_with_settle_mode() -> None:
    sup = _sup()
    resp = sup.handle_command("watch_list", {})
    assert resp.ok is True
    watches = {w["name"]: w for w in resp.data["watches"]}
    assert watches["cam"]["settle_mode"] == "quiescence"
    assert watches["modis"]["settle_mode"] == "sentinel"
    assert watches["modis"]["on_fire"] == "new_files_only"
    assert watches["cam"]["enabled"] is True


def test_watch_pause_and_resume_persist_via_state() -> None:
    state = FakeStateStore()
    sup = _sup(state=state)
    paused = sup.handle_command("watch_pause", {"name": "cam"})
    assert paused.ok is True
    assert paused.data == {"name": "cam", "enabled": False}
    assert sup.rules["cam"].enabled is False
    assert ("cam", False) in state.enabled_calls

    resumed = sup.handle_command("watch_resume", {"name": "cam"})
    assert resumed.data == {"name": "cam", "enabled": True}
    assert sup.rules["cam"].enabled is True
    assert ("cam", True) in state.enabled_calls


def test_watch_pause_unknown_name_is_not_found() -> None:
    sup = _sup()
    resp = sup.handle_command("watch_pause", {"name": "nope"})
    assert resp.ok is False
    assert resp.code == "not_found"


def test_watch_trigger_enqueues_event() -> None:
    queue = FakeQueue()
    sup = _sup(queue=queue)
    resp = sup.handle_command("watch_trigger", {"name": "cam", "files": ["/tmp/cam/x.nc"]})
    assert resp.ok is True
    assert resp.data["coalesced"] is False
    assert len(queue.submitted) == 1
    assert queue.submitted[0].watch_name == "cam"
    assert queue.submitted[0].new_files == ["/tmp/cam/x.nc"]


def test_status_reports_draining_and_max_concurrent() -> None:
    sup = _sup()
    resp = sup.handle_command("status", {})
    assert resp.ok is True
    assert resp.data["max_concurrent"] == 2
    assert resp.data["draining"] is False
    assert isinstance(resp.data["watches"], list)
    assert "running" in resp.data and "queued" in resp.data

    sup.request_shutdown()
    resp2 = sup.handle_command("status", {})
    assert resp2.data["draining"] is True


def test_unknown_command_is_unsupported() -> None:
    sup = _sup()
    resp = sup.handle_command("frobnicate", {})
    assert resp.ok is False
    assert resp.code == "unsupported"


def test_shutdown_sets_draining_and_acks() -> None:
    sup = _sup()
    resp = sup.handle_command("shutdown", {"drain": True})
    assert resp.ok is True
    assert resp.data["shutting_down"] is True
    assert resp.data["draining"] is True
    assert sup.draining is True
