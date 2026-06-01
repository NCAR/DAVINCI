"""Thin synchronous Unix-socket client for the DAVINCI daemon control server.

Used by ``daemon shell`` / ``daemon top`` / ``daemon status`` / ``daemon stop``.
Framing mirrors control.py (SHARED CONTRACTS §7): newline-delimited UTF-8 JSON,
ONE message per line. ``call`` is a request/response round-trip; ``stream`` yields
the ack ControlResponse then each pushed StreamEvent until the server closes.

Stdlib socket only; never imports the sci stack.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Iterator

from davinci_monet.daemon.contracts import ControlResponse, StreamEvent

_DEFAULT_TIMEOUT = 10.0


class DaemonClient:
    """Synchronous client over the daemon's AF_UNIX control socket."""

    def __init__(self, socket_path: str | Path, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout

    # ---- connection ------------------------------------------------------

    def connect(self) -> socket.socket:
        """Open a fresh connected AF_UNIX stream socket. Raises on failure."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(str(self.socket_path))
        return sock

    @staticmethod
    def _send_request(sock: socket.socket, cmd: str, args: dict[str, Any]) -> None:
        line = json.dumps({"cmd": cmd, "args": args}) + "\n"
        sock.sendall(line.encode("utf-8"))

    @staticmethod
    def _read_line(sock: socket.socket, buf: bytearray) -> str | None:
        """Read one '\\n'-terminated line, buffering any extra bytes in ``buf``."""
        while b"\n" not in buf:
            try:
                chunk = sock.recv(4096)
            except (socket.timeout, TimeoutError):
                raise
            if not chunk:
                if buf:
                    line = bytes(buf).decode("utf-8")
                    del buf[:]
                    return line
                return None
            buf.extend(chunk)
        idx = buf.index(b"\n")
        line = bytes(buf[:idx]).decode("utf-8")
        del buf[: idx + 1]
        return line

    # ---- request / response ---------------------------------------------

    def call(self, cmd: str, **args: Any) -> ControlResponse:
        """Send {cmd,args}; read one response line; return ControlResponse.

        Raises on transport error (no socket, refused, reset, timeout).
        """
        sock = self.connect()
        try:
            self._send_request(sock, cmd, args)
            buf = bytearray()
            line = self._read_line(sock, buf)
            if line is None:
                raise ConnectionError("daemon closed connection without a response")
            return ControlResponse.model_validate_json(line)
        finally:
            try:
                sock.close()
            except OSError:
                pass

    # ---- streaming -------------------------------------------------------

    def stream(self, cmd: str, **args: Any) -> Iterator[Any]:
        """Send a streaming cmd; yield the ack ControlResponse then StreamEvents.

        Iteration ends when the server closes the connection.
        """
        sock = self.connect()
        try:
            self._send_request(sock, cmd, args)
            buf = bytearray()
            ack_line = self._read_line(sock, buf)
            if ack_line is None:
                raise ConnectionError("daemon closed connection without an ack")
            ack = ControlResponse.model_validate_json(ack_line)
            yield ack
            if not ack.ok:
                return
            while True:
                line = self._read_line(sock, buf)
                if line is None:
                    return  # server closed the stream
                yield StreamEvent.model_validate_json(line)
        finally:
            try:
                sock.close()
            except OSError:
                pass

    # ---- liveness --------------------------------------------------------

    def is_alive(self) -> bool:
        """True if a ``ping`` round-trips (daemon up + socket healthy)."""
        try:
            resp = self.call("ping")
        except OSError:
            return False
        return bool(resp.ok)
