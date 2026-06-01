"""Unit tests for the daemon control client (group: control/client).

Drives a real ControlServer over a real AF_UNIX socket in a temp dir.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from davinci_monet.daemon.client import DaemonClient
from davinci_monet.daemon.contracts import ControlResponse, StreamEvent
from davinci_monet.daemon.control import ControlServer


@pytest.fixture()
def running_server(tmp_path: Path):
    sock_path = tmp_path / "control.sock"
    srv = ControlServer(sock_path)

    srv.register("ping", lambda a: ControlResponse(ok=True, data={"pong": True}))
    srv.register("echo", lambda a: ControlResponse(ok=True, data={"got": a}))
    srv.register("fail", lambda a: ControlResponse(ok=False, error="nope", code="not_found"))

    def sub(args: dict):
        yield ControlResponse(ok=True, data={"subscribed": args.get("topics", [])})
        for i in range(3):
            yield StreamEvent(event="tick", data={"i": i})

    srv.register_stream("subscribe", sub)
    srv.start()
    deadline = time.monotonic() + 5.0
    while not sock_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield sock_path
    finally:
        srv.stop()


class TestCall:
    def test_call_round_trip(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        resp = client.call("echo", a=1, b="two")
        assert isinstance(resp, ControlResponse)
        assert resp.ok is True
        assert resp.data == {"got": {"a": 1, "b": "two"}}

    def test_call_error_response_preserved(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        resp = client.call("fail")
        assert resp.ok is False
        assert resp.error == "nope"
        assert resp.code == "not_found"

    def test_is_alive_true_when_up(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        assert client.is_alive() is True

    def test_is_alive_false_when_no_socket(self, tmp_path: Path) -> None:
        client = DaemonClient(tmp_path / "nonexistent.sock")
        assert client.is_alive() is False

    def test_call_raises_on_dead_socket(self, tmp_path: Path) -> None:
        client = DaemonClient(tmp_path / "nope.sock")
        with pytest.raises((ConnectionError, FileNotFoundError, OSError)):
            client.call("ping")


class TestStream:
    def test_stream_yields_ack_then_events(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        items = list(client.stream("subscribe", topics=["jobs"]))
        ack = items[0]
        assert isinstance(ack, ControlResponse)
        assert ack.ok is True
        assert ack.data == {"subscribed": ["jobs"]}
        events = items[1:]
        assert all(isinstance(e, StreamEvent) for e in events)
        assert [e.event for e in events] == ["tick", "tick", "tick"]
        assert [e.data["i"] for e in events] == [0, 1, 2]

    def test_stream_ends_when_server_closes(self, running_server: Path) -> None:
        client = DaemonClient(running_server)
        count = 0
        for _ in client.stream("subscribe"):
            count += 1
        # ack + 3 events, then clean end
        assert count == 4
