"""Unit tests for the daemon control shell (parsing + dispatch)."""

from __future__ import annotations

import pytest

from davinci_monet.daemon.shell import ShellResult, parse_command


class TestParseCommand:
    def test_status(self) -> None:
        res = parse_command("status")
        assert isinstance(res, ShellResult)
        assert res.cmd == "status"
        assert res.args == {}
        assert res.streaming is False
        assert res.local is False

    def test_reload(self) -> None:
        res = parse_command("reload")
        assert res.cmd == "reload"
        assert res.args == {}

    def test_history_with_flags(self) -> None:
        res = parse_command("history --watch cam_realtime --failed --limit 10")
        assert res.cmd == "history"
        assert res.args == {"watch": "cam_realtime", "failed": True, "limit": 10}

    def test_history_defaults(self) -> None:
        res = parse_command("history")
        assert res.cmd == "history"
        assert res.args == {"watch": None, "failed": False, "limit": 50}

    def test_watch_list(self) -> None:
        res = parse_command("watch list")
        assert res.cmd == "watch_list"
        assert res.args == {}

    def test_watch_pause(self) -> None:
        res = parse_command("watch pause cam_realtime")
        assert res.cmd == "watch_pause"
        assert res.args == {"name": "cam_realtime"}

    def test_watch_resume(self) -> None:
        res = parse_command("watch resume cam_realtime")
        assert res.cmd == "watch_resume"
        assert res.args == {"name": "cam_realtime"}

    def test_watch_remove(self) -> None:
        res = parse_command("watch remove modis_stream")
        assert res.cmd == "watch_remove"
        assert res.args == {"name": "modis_stream"}

    def test_watch_trigger_no_files(self) -> None:
        res = parse_command("watch trigger cam_realtime")
        assert res.cmd == "watch_trigger"
        assert res.args == {"name": "cam_realtime", "files": []}

    def test_watch_trigger_with_files(self) -> None:
        res = parse_command("watch trigger cam_realtime /a/b.nc /a/c.nc")
        assert res.cmd == "watch_trigger"
        assert res.args == {
            "name": "cam_realtime",
            "files": ["/a/b.nc", "/a/c.nc"],
        }

    def test_watch_save(self) -> None:
        res = parse_command("watch save live_rule")
        assert res.cmd == "watch_save"
        assert res.args == {"name": "live_rule"}

    def test_job_get(self) -> None:
        res = parse_command("job 42")
        assert res.cmd == "job_get"
        assert res.args == {"job_id": 42}

    def test_logs_job_oneshot(self) -> None:
        res = parse_command("logs 7")
        assert res.cmd == "logs"
        assert res.args == {"target": "7", "kind": "job"}
        assert res.streaming is False

    def test_logs_watch(self) -> None:
        res = parse_command("logs cam_realtime --watch")
        assert res.cmd == "logs"
        assert res.args == {"target": "cam_realtime", "kind": "watch"}

    def test_logs_tail_streams(self) -> None:
        res = parse_command("logs 7 --tail")
        assert res.cmd == "logs_tail"
        assert res.args == {"target": "7", "kind": "job"}
        assert res.streaming is True

    def test_quit_is_local(self) -> None:
        for word in ("quit", "exit", "q"):
            res = parse_command(word)
            assert res.local is True
            assert res.cmd is None

    def test_help_is_local(self) -> None:
        res = parse_command("help")
        assert res.local is True
        assert res.cmd is None

    def test_empty_line_is_local_noop(self) -> None:
        res = parse_command("   ")
        assert res.local is True
        assert res.cmd is None

    def test_unknown_command_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown command"):
            parse_command("frobnicate")

    def test_watch_add_rejected_in_shell(self) -> None:
        with pytest.raises(ValueError, match="watch add is not available"):
            parse_command("watch add foo")

    def test_watch_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a watch name"):
            parse_command("watch pause")


from unittest.mock import MagicMock

from davinci_monet.daemon.contracts import ControlResponse, StreamEvent
from davinci_monet.daemon.shell import DaemonShell


class TestDaemonShellDispatch:
    def _shell(self) -> tuple[DaemonShell, MagicMock]:
        client = MagicMock()
        client.call.return_value = ControlResponse(ok=True, data={"pong": True})
        return DaemonShell(client), client

    def test_status_calls_client_call(self) -> None:
        shell, client = self._shell()
        resp = shell.execute("status")
        client.call.assert_called_once_with("status")
        client.stream.assert_not_called()
        assert resp is not None and resp.ok is True

    def test_watch_pause_dispatches_name(self) -> None:
        shell, client = self._shell()
        shell.execute("watch pause cam_realtime")
        client.call.assert_called_once_with("watch_pause", name="cam_realtime")

    def test_watch_trigger_dispatches_files(self) -> None:
        shell, client = self._shell()
        shell.execute("watch trigger cam /a.nc /b.nc")
        client.call.assert_called_once_with("watch_trigger", name="cam", files=["/a.nc", "/b.nc"])

    def test_history_flags_dispatched(self) -> None:
        shell, client = self._shell()
        shell.execute("history --watch cam --failed --limit 5")
        client.call.assert_called_once_with("history", watch="cam", failed=True, limit=5)

    def test_logs_oneshot_uses_call(self) -> None:
        shell, client = self._shell()
        shell.execute("logs 7")
        client.call.assert_called_once_with("logs", target="7", kind="job")
        client.stream.assert_not_called()

    def test_logs_tail_uses_stream(self) -> None:
        shell, client = self._shell()
        client.stream.return_value = iter(
            [StreamEvent(event="log_line", data={"message": "hello"})]
        )
        resp = shell.execute("logs 7 --tail")
        client.stream.assert_called_once_with("logs_tail", target="7", kind="job")
        client.call.assert_not_called()
        assert resp is None  # streaming returns None after draining

    def test_local_quit_does_not_touch_client(self) -> None:
        shell, client = self._shell()
        resp = shell.execute("quit")
        client.call.assert_not_called()
        client.stream.assert_not_called()
        assert resp is None

    def test_local_help_does_not_touch_client(self) -> None:
        shell, client = self._shell()
        shell.execute("help")
        client.call.assert_not_called()

    def test_is_quit(self) -> None:
        shell, _ = self._shell()
        assert shell.is_quit("quit") is True
        assert shell.is_quit("exit") is True
        assert shell.is_quit("status") is False


class TestShellNeverStopsDaemon:
    def test_quit_emits_no_shutdown(self) -> None:
        for word in ("quit", "exit", "q"):
            res = parse_command(word)
            assert res.cmd != "shutdown"
            assert res.cmd is None
            assert res.local is True

    def test_shutdown_verb_is_not_reachable(self) -> None:
        # The shell has no `shutdown`/`stop` verb; it must be an unknown command.
        with pytest.raises(ValueError, match="unknown command"):
            parse_command("shutdown")
        with pytest.raises(ValueError, match="unknown command"):
            parse_command("stop")

    def test_dispatch_quit_never_calls_shutdown(self) -> None:
        shell, client = TestDaemonShellDispatch()._shell()
        shell.execute("quit")
        # No call at all, and certainly not shutdown.
        client.call.assert_not_called()
        for call in client.call.call_args_list:
            assert call.args[0] != "shutdown"


class TestRunShell:
    def test_run_shell_dispatches_then_quits_cleanly(self) -> None:
        from davinci_monet.daemon.shell import run_shell

        client = MagicMock()
        client.call.return_value = ControlResponse(ok=True, data={"pong": True})
        # Scripted input: one real command, then quit ends the loop.
        scripted = iter(["status", "quit"])
        outputs: list[str] = []

        run_shell(
            client,
            input_fn=lambda _prompt: next(scripted),
            output_fn=outputs.append,
        )

        client.call.assert_called_once_with("status")
        # Quitting never sends shutdown and the loop returned cleanly.
        for call in client.call.call_args_list:
            assert call.args[0] != "shutdown"

    def test_run_shell_exits_on_eof(self) -> None:
        from davinci_monet.daemon.shell import run_shell

        client = MagicMock()

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        # EOF (Ctrl-D) ends the loop without touching the socket.
        run_shell(client, input_fn=_raise_eof, output_fn=lambda _s: None)
        client.call.assert_not_called()
