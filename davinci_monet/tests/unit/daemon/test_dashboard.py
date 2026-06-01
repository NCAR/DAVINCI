"""Unit tests for the `daemon top` dashboard renderers (no live loop)."""

from __future__ import annotations

import pytest
from rich.console import Console

from davinci_monet.daemon.dashboard import (
    DashboardState,
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
