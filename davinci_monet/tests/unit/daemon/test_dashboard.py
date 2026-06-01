"""Unit tests for the `daemon top` dashboard renderers (no live loop)."""

from __future__ import annotations

import pytest
from rich.console import Console

from davinci_monet.daemon.contracts import StreamEvent
from davinci_monet.daemon.dashboard import (
    DashboardState,
    apply_stream_event,
    render_dashboard,
    render_queue_panel,
    render_recent_panel,
    render_running_panel,
    render_watches_panel,
)


@pytest.fixture
def sample_state() -> DashboardState:
    return DashboardState(
        version=1,
        pid=4242,
        uptime_s=3725.0,
        draining=False,
        max_concurrent=1,
        watches=[
            {
                "name": "cam_realtime",
                "enabled": True,
                "source": "file",
                "on_fire": "whole_config",
                "settle_mode": "quiescence",
                "watch": "/scratch/cam/incoming/*.nc",
                "run": "configs/asia-aq.yaml",
                "state": "running",
                "last_job_id": 7,
                "last_status": "running",
                "last_fired_at": "2026-05-31T12:00:00",
            },
            {
                "name": "modis_stream",
                "enabled": False,
                "source": "live",
                "on_fire": "new_files_only",
                "settle_mode": "sentinel",
                "watch": "/scratch/modis/*.hdf",
                "run": "configs/modis-aod.yaml",
                "state": "paused",
                "last_job_id": None,
                "last_status": None,
                "last_fired_at": None,
            },
        ],
        running=[
            {
                "id": 7,
                "watch_name": "cam_realtime",
                "config_path": "configs/asia-aq.yaml",
                "on_fire": "whole_config",
                "status": "running",
                "submitted_at": "2026-05-31T11:59:00",
                "started_at": "2026-05-31T12:00:00",
            }
        ],
        queued=[
            {
                "id": 8,
                "watch_name": "modis_stream",
                "config_path": "configs/modis-aod.yaml",
                "on_fire": "new_files_only",
                "status": "queued",
                "submitted_at": "2026-05-31T12:01:00",
            }
        ],
        recent=[
            {
                "id": 6,
                "watch_name": "cam_realtime",
                "config_path": "configs/asia-aq.yaml",
                "on_fire": "whole_config",
                "status": "completed",
                "submitted_at": "2026-05-31T11:00:00",
                "ended_at": "2026-05-31T11:30:00",
                "duration_s": 1800.0,
            },
            {
                "id": 5,
                "watch_name": "modis_stream",
                "config_path": "configs/modis-aod.yaml",
                "on_fire": "new_files_only",
                "status": "failed",
                "submitted_at": "2026-05-31T10:00:00",
                "ended_at": "2026-05-31T10:05:00",
                "duration_s": 300.0,
                "error": "config error: missing source",
            },
        ],
        progress={7: "Loading model: cam (1/2)"},
    )


def _render_to_text(renderable: object) -> str:
    console = Console(width=140, record=True)
    console.print(renderable)
    return console.export_text()


class TestDashboardStateFromStatus:
    def test_round_trip_from_status_dict(self) -> None:
        data = {
            "version": "2",
            "pid": "1234",
            "uptime_s": "99.5",
            "draining": True,
            "max_concurrent": "3",
            "running": [{"id": 1}],
            "queued": [],
            "watches": [{"name": "w1"}],
            "recent": [{"id": 0}],
        }
        state = DashboardState.from_status(data)
        assert state.version == 2
        assert state.pid == 1234
        assert state.uptime_s == 99.5
        assert state.draining is True
        assert state.max_concurrent == 3
        assert state.running == [{"id": 1}]
        assert state.queued == []
        assert state.watches == [{"name": "w1"}]
        assert state.recent == [{"id": 0}]
        # progress is not populated from status dict (comes from stream events)
        assert state.progress == {}


class TestWatchesPanel:
    def test_lists_both_watches(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_watches_panel(sample_state))
        assert "cam_realtime" in text
        assert "modis_stream" in text

    def test_shows_paused_and_enabled_state(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_watches_panel(sample_state))
        assert "paused" in text
        assert "running" in text

    def test_shows_settle_mode(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_watches_panel(sample_state))
        assert "sentinel" in text
        assert "quiescence" in text


class TestRunningPanel:
    def test_shows_running_job_and_progress(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_running_panel(sample_state))
        assert "cam_realtime" in text
        assert "7" in text  # job id
        assert "Loading model: cam (1/2)" in text  # progress message

    def test_empty_running_renders_placeholder(self) -> None:
        empty = DashboardState(version=1, pid=1, uptime_s=0.0, draining=False, max_concurrent=1)
        text = _render_to_text(render_running_panel(empty))
        assert "RUNNING" in text.upper()
        # No crash on empty; some idle marker present.
        assert "idle" in text.lower() or "—" in text or "-" in text


class TestQueuePanel:
    def test_shows_queued_job(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_queue_panel(sample_state))
        assert "modis_stream" in text
        assert "8" in text


class TestRecentPanel:
    def test_shows_completed_and_failed(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_recent_panel(sample_state))
        assert "completed" in text
        assert "failed" in text
        assert "cam_realtime" in text
        assert "modis_stream" in text


class TestRenderDashboard:
    def test_composite_contains_all_sections(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_dashboard(sample_state))
        # Header
        assert "DAVINCI" in text
        assert "4242" in text  # pid
        # Each panel's content
        assert "cam_realtime" in text  # watches + running + recent
        assert "modis_stream" in text  # queue + recent
        assert "Loading model: cam (1/2)" in text  # running progress
        assert "completed" in text  # recent

    def test_header_shows_uptime(self, sample_state: DashboardState) -> None:
        text = _render_to_text(render_dashboard(sample_state))
        # 3725s -> "1.0h" or "62.1m" formatting; just assert an uptime label.
        assert "uptime" in text.lower()

    def test_draining_flag_surfaced(self) -> None:
        draining = DashboardState(version=1, pid=9, uptime_s=1.0, draining=True, max_concurrent=1)
        text = _render_to_text(render_dashboard(draining))
        assert "draining" in text.lower()


class TestApplyStreamEvent:
    def test_progress_event_updates_progress_map(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(event="log_line", data={"job_id": 7, "message": "Pairing cam vs airnow"})
        apply_stream_event(sample_state, ev)
        assert sample_state.progress[7] == "Pairing cam vs airnow"

    def test_stage_progress_event(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(
            event="job_update", data={"job_id": 7, "stage": "statistics", "kind": "stage"}
        )
        apply_stream_event(sample_state, ev)
        assert "statistics" in sample_state.progress[7]

    def test_job_update_moves_running_to_recent(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(
            event="job_update",
            data={
                "id": 7,
                "watch_name": "cam_realtime",
                "config_path": "configs/asia-aq.yaml",
                "on_fire": "whole_config",
                "status": "completed",
                "submitted_at": "2026-05-31T11:59:00",
                "ended_at": "2026-05-31T12:30:00",
                "duration_s": 1860.0,
            },
        )
        apply_stream_event(sample_state, ev)
        assert all(j["id"] != 7 for j in sample_state.running)
        assert any(j["id"] == 7 and j["status"] == "completed" for j in sample_state.recent)

    def test_job_update_promotes_queued_to_running(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(
            event="job_update",
            data={
                "id": 8,
                "watch_name": "modis_stream",
                "config_path": "configs/modis-aod.yaml",
                "on_fire": "new_files_only",
                "status": "running",
                "submitted_at": "2026-05-31T12:01:00",
                "started_at": "2026-05-31T12:31:00",
            },
        )
        apply_stream_event(sample_state, ev)
        assert any(j["id"] == 8 for j in sample_state.running)
        assert all(j["id"] != 8 for j in sample_state.queued)

    def test_watch_update_replaces_summary(self, sample_state: DashboardState) -> None:
        ev = StreamEvent(
            event="watch_update",
            data={
                "name": "modis_stream",
                "enabled": True,
                "source": "live",
                "on_fire": "new_files_only",
                "settle_mode": "sentinel",
                "watch": "/scratch/modis/*.hdf",
                "run": "configs/modis-aod.yaml",
                "state": "armed",
                "last_job_id": None,
                "last_status": None,
                "last_fired_at": None,
            },
        )
        apply_stream_event(sample_state, ev)
        modis = next(w for w in sample_state.watches if w["name"] == "modis_stream")
        assert modis["enabled"] is True
        assert modis["state"] == "armed"

    def test_unknown_event_is_noop(self, sample_state: DashboardState) -> None:
        before = len(sample_state.running)
        apply_stream_event(sample_state, StreamEvent(event="mystery", data={}))
        assert len(sample_state.running) == before
