"""Unit tests for daemon desktop-notification dispatch (mocked subprocess)."""

from __future__ import annotations

import subprocess
from typing import Any

from davinci_monet.daemon.notify import DesktopNotifier, send_desktop_notification


class _FakeRunner:
    """Records calls; can be told to fail the first command (osascript)."""

    def __init__(self, fail_cmds: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_cmds = fail_cmds or set()

    def __call__(self, argv: list[str]) -> int:
        self.calls.append(list(argv))
        if argv and argv[0] in self.fail_cmds:
            raise FileNotFoundError(argv[0])
        return 0


def test_send_desktop_uses_osascript_by_default() -> None:
    runner = _FakeRunner()
    ok = send_desktop_notification("DAVINCI", "cam_realtime completed", runner=runner)
    assert ok is True
    assert len(runner.calls) == 1
    argv = runner.calls[0]
    assert argv[0] == "osascript"
    assert argv[1] == "-e"
    # AppleScript must carry both the title and the message text.
    assert "cam_realtime completed" in argv[2]
    assert "DAVINCI" in argv[2]


def test_send_desktop_falls_back_to_terminal_notifier() -> None:
    runner = _FakeRunner(fail_cmds={"osascript"})
    ok = send_desktop_notification("DAVINCI", "boom", runner=runner)
    assert ok is True
    assert [c[0] for c in runner.calls] == ["osascript", "terminal-notifier"]
    tn = runner.calls[1]
    assert "-message" in tn
    assert "boom" in tn


def test_send_desktop_returns_false_when_all_backends_missing() -> None:
    runner = _FakeRunner(fail_cmds={"osascript", "terminal-notifier"})
    ok = send_desktop_notification("DAVINCI", "x", runner=runner)
    assert ok is False
    assert [c[0] for c in runner.calls] == ["osascript", "terminal-notifier"]


def test_desktop_notifier_class_is_callable_wrapper() -> None:
    runner = _FakeRunner()
    notifier = DesktopNotifier(runner=runner)
    assert notifier("T", "M") is True
    assert runner.calls[0][0] == "osascript"


def test_quotes_in_message_do_not_break_applescript() -> None:
    runner = _FakeRunner()
    send_desktop_notification('Ti"tle', 'mes"sage', runner=runner)
    script = runner.calls[0][2]
    # Embedded double quotes must be escaped/stripped, never left raw-unbalanced.
    assert script.count('"') % 2 == 0
