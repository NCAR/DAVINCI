"""Unit tests for the daemon control socket server + client (group: control/client).

Round-trips real newline-delimited JSON over a real AF_UNIX socket in a temp
dir. No external datasets, no sci stack — stdlib socket only.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from davinci_monet.daemon.contracts import ControlResponse, StreamEvent
from davinci_monet.daemon.control import ControlServer


def _recv_line(sock: socket.socket, timeout: float = 5.0) -> str:
    """Read one newline-terminated line from a connected socket."""
    sock.settimeout(timeout)
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.decode("utf-8")


def _connect(socket_path: Path, timeout: float = 5.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(str(socket_path))
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)


@pytest.fixture()
def server(tmp_path: Path):
    sock_path = tmp_path / "control.sock"
    srv = ControlServer(sock_path)
    srv.start()
    # wait for the socket file to appear (server bound)
    deadline = time.monotonic() + 5.0
    while not sock_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield srv, sock_path
    finally:
        srv.stop()


class TestRequestResponse:
    def test_ping_round_trip(self, server) -> None:
        srv, sock_path = server

        def ping_handler(args: dict) -> ControlResponse:
            return ControlResponse(ok=True, data={"pong": True, "n": args.get("n", 0)})

        srv.register("ping", ping_handler)

        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "ping", "args": {"n": 7}}) + "\n").encode())
        line = _recv_line(conn)
        conn.close()

        payload = json.loads(line)
        assert payload["ok"] is True
        assert payload["data"] == {"pong": True, "n": 7}

    def test_missing_args_defaults_to_empty(self, server) -> None:
        srv, sock_path = server
        seen: dict = {}

        def h(args: dict) -> ControlResponse:
            seen.update({"args": args})
            return ControlResponse(ok=True, data=None)

        srv.register("noargs", h)
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "noargs"}) + "\n").encode())
        _recv_line(conn)
        conn.close()
        assert seen["args"] == {}

    def test_handler_exception_becomes_error_response(self, server) -> None:
        srv, sock_path = server

        def boom(args: dict) -> ControlResponse:
            raise RuntimeError("kaboom")

        srv.register("boom", boom)
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "boom", "args": {}}) + "\n").encode())
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert "kaboom" in payload["error"]
        assert payload["code"] == "handler_error"
