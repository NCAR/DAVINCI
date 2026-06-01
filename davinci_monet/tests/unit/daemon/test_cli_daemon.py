"""Unit tests for the `daemon` CLI sub-app (mocked client, no real socket)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from davinci_monet.cli.commands import daemon as daemon_cli
from davinci_monet.daemon.contracts import ControlResponse

runner = CliRunner()


def test_daemon_subapp_help_lists_commands() -> None:
    result = runner.invoke(daemon_cli.app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    for sub in ("serve", "start", "stop", "status", "reload", "watch", "logs", "history"):
        assert sub in out


def test_status_invokes_client_status() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(
        ok=True,
        data={
            "version": 1,
            "pid": 4242,
            "uptime_s": 12.0,
            "draining": False,
            "max_concurrent": 1,
            "running": [],
            "queued": [],
            "watches": [
                {
                    "name": "cam",
                    "enabled": True,
                    "source": "file",
                    "on_fire": "whole_config",
                    "settle_mode": "quiescence",
                    "watch": "/tmp/cam/*.nc",
                    "run": "/tmp/cam.yaml",
                    "state": "armed",
                    "last_job_id": None,
                    "last_status": None,
                    "last_fired_at": None,
                }
            ],
            "recent": [],
        },
    )
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["status"])
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("status")
    assert "cam" in result.stdout
    assert "4242" in result.stdout


def test_status_when_daemon_not_running_reports_clearly() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = False
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["status"])
    assert result.exit_code != 0
    assert "not running" in result.stdout.lower()


def test_watch_pause_calls_client_with_name() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(ok=True, data={"name": "cam", "enabled": False})
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["watch", "pause", "cam"])
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("watch_pause", name="cam")
    assert "cam" in result.stdout


def test_watch_list_renders_rows() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(
        ok=True,
        data={
            "watches": [
                {
                    "name": "modis",
                    "enabled": False,
                    "source": "live",
                    "on_fire": "new_files_only",
                    "settle_mode": "sentinel",
                    "watch": "/tmp/modis/*.hdf",
                    "run": "/tmp/modis.yaml",
                    "state": "paused",
                    "last_job_id": 7,
                    "last_status": "completed",
                    "last_fired_at": None,
                }
            ]
        },
    )
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["watch", "list"])
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("watch_list")
    assert "modis" in result.stdout


def test_history_passes_filters() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(ok=True, data={"jobs": []})
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(
            daemon_cli.app, ["history", "--watch", "cam", "--failed", "--limit", "5"]
        )
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("history", watch="cam", failed=True, limit=5)


def test_reload_reports_changes() -> None:
    fake_client = MagicMock()
    fake_client.is_alive.return_value = True
    fake_client.call.return_value = ControlResponse(
        ok=True,
        data={"reloaded": True, "watches": [], "added": ["x"], "removed": [], "updated": ["cam"]},
    )
    with patch.object(daemon_cli, "_make_client", return_value=fake_client):
        result = runner.invoke(daemon_cli.app, ["reload"])
    assert result.exit_code == 0
    fake_client.call.assert_called_once_with("reload")
    assert "x" in result.stdout
