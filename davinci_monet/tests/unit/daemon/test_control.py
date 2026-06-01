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


class TestErrorEnvelope:
    def test_unknown_command(self, server) -> None:
        srv, sock_path = server
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "does_not_exist", "args": {}}) + "\n").encode())
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert payload["code"] == "unsupported"
        assert "does_not_exist" in payload["error"]

    def test_malformed_json(self, server) -> None:
        srv, sock_path = server
        conn = _connect(sock_path)
        conn.sendall(b"this is not json\n")
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert payload["code"] == "invalid_args"

    def test_missing_cmd(self, server) -> None:
        srv, sock_path = server
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"args": {"x": 1}}) + "\n").encode())
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert payload["code"] == "invalid_args"

    def test_non_object_args(self, server) -> None:
        srv, sock_path = server
        conn = _connect(sock_path)
        conn.sendall((json.dumps({"cmd": "x", "args": [1, 2, 3]}) + "\n").encode())
        payload = json.loads(_recv_line(conn))
        conn.close()
        assert payload["ok"] is False
        assert payload["code"] == "invalid_args"

    def test_dispatch_fallback_routes_unregistered_command(self, tmp_path: Path) -> None:
        # A ControlServer built with a `dispatch=` fallback routes any command not
        # in the per-command registry through dispatch(cmd, args) instead of
        # returning "unsupported". The supervisor passes handle_command here.
        seen: dict = {}

        def dispatch(cmd: str, args: dict) -> ControlResponse:
            seen.update({"cmd": cmd, "args": args})
            return ControlResponse(ok=True, data={"routed": cmd})

        sock_path = tmp_path / "fallback.sock"
        srv = ControlServer(sock_path, dispatch)
        srv.start()
        deadline = time.monotonic() + 5.0
        while not sock_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            conn = _connect(sock_path)
            conn.sendall((json.dumps({"cmd": "status", "args": {"n": 1}}) + "\n").encode())
            payload = json.loads(_recv_line(conn))
            conn.close()
        finally:
            srv.stop()
        assert payload["ok"] is True
        assert payload["data"] == {"routed": "status"}
        assert seen == {"cmd": "status", "args": {"n": 1}}
