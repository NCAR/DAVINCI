"""The daemon sub-app must be registered on the main CLI app."""

from __future__ import annotations

from typer.testing import CliRunner

from davinci_monet.cli.app import app

runner = CliRunner()


def test_root_help_lists_daemon_subapp() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "daemon" in result.stdout.lower()


def test_daemon_help_routes_through_root_app() -> None:
    result = runner.invoke(app, ["daemon", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "serve" in out
    assert "status" in out
    assert "watch" in out


def test_daemon_watch_help_routes_through_root_app() -> None:
    result = runner.invoke(app, ["daemon", "watch", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "pause" in out
    assert "resume" in out
    assert "trigger" in out
