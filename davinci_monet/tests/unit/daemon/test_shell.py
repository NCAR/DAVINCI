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
