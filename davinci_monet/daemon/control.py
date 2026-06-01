"""Unix-domain-socket control server for the DAVINCI daemon.

Transport (SHARED CONTRACTS §7): AF_UNIX / SOCK_STREAM at ``socket_path``.
Framing is newline-delimited UTF-8 JSON, ONE message per line ('\\n').

Request  : {"cmd": <str>, "args": <object>}
Response : {"ok": true, "data": ...} | {"ok": false, "error": str, "code": str|null}

The supervisor registers per-command handlers. A request/response handler
conforms to ``ControlHandler``: ``(args: dict) -> ControlResponse``. A streaming
handler (``register_stream``) is a generator ``(args: dict) -> Iterator``; its
FIRST yielded value is the ack ``ControlResponse``, and every subsequent yielded
``StreamEvent`` is framed as ``{"event": str, "data": object}`` and pushed until
the generator ends or the client disconnects.

This module is part of the thin supervisor and imports only stdlib + the daemon
contracts. It MUST NOT import matplotlib / xarray / the pipeline.
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from davinci_monet.daemon.contracts import ControlHandler, ControlResponse, StreamEvent

# A streaming handler: generator whose first yield is the ack ControlResponse and
# whose remaining yields are StreamEvents pushed to the client.
StreamHandler = Callable[[dict[str, Any]], Iterator[Any]]

_BACKLOG = 64


class ControlServer:
    """Thread-per-connection AF_UNIX control server.

    Runs its accept loop in a background daemon thread so the supervisor's main
    loop is unblocked. Light traffic (one shell/top/status client at a time) makes
    thread-per-connection the simplest robust model; no selectors event loop is
    needed here.
    """

    def __init__(
        self,
        socket_path: "str | Path",
        dispatch: "Callable[[str, dict[str, Any]], ControlResponse] | None" = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self._handlers: dict[str, ControlHandler] = {}
        self._streamers: dict[str, StreamHandler] = {}
        # Optional fallback dispatcher: called as ``dispatch(cmd, args)`` for any
        # command not found in the per-command registry (and not a streamer).
        # The supervisor passes its ``handle_command`` here so the whole catalog
        # routes without registering each command individually.
        self._dispatch_fallback = dispatch
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._conns: set[socket.socket] = set()
        self._conns_lock = threading.Lock()

    # ---- registration ----------------------------------------------------

    def register(self, cmd: str, handler: ControlHandler) -> None:
        """Register a request/response handler for ``cmd``."""
        self._handlers[cmd] = handler

    def register_stream(self, cmd: str, handler: StreamHandler) -> None:
        """Register a streaming handler (generator: first yield = ack)."""
        self._streamers[cmd] = handler

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Bind the socket and launch the accept loop in a background thread."""
        self._bind()
        self._thread = threading.Thread(
            target=self._accept_loop, name="control-accept", daemon=True
        )
        self._thread.start()

    def _bind(self) -> None:
        # Remove a stale socket file before binding.
        try:
            if self.socket_path.exists() or self.socket_path.is_socket():
                self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(str(self.socket_path))
        sock.listen(_BACKLOG)
        sock.settimeout(0.5)  # so the accept loop can observe _stopping
        self._sock = sock

    def serve_forever(self) -> None:
        """Bind (if needed) and run the accept loop in the current thread."""
        if self._sock is None:
            self._bind()
        self._accept_loop()

    def stop(self) -> None:
        """Stop accepting, close live connections, remove the socket file."""
        self._stopping.set()
        with self._conns_lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                c.close()
            except OSError:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

    # ---- accept loop -----------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stopping.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed during stop()
            t = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True)
            t.start()

    def _serve_conn(self, conn: socket.socket) -> None:
        with self._conns_lock:
            self._conns.add(conn)
        try:
            conn.settimeout(None)
            line = self._read_line(conn)
            if line is None:
                return
            self._dispatch(conn, line)
        finally:
            with self._conns_lock:
                self._conns.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    # ---- framing ---------------------------------------------------------

    @staticmethod
    def _read_line(conn: socket.socket) -> Optional[str]:
        buf = b""
        while not buf.endswith(b"\n"):
            try:
                chunk = conn.recv(4096)
            except OSError:
                return None
            if not chunk:
                return buf.decode("utf-8") if buf else None
            buf += chunk
        return buf.decode("utf-8")

    @staticmethod
    def _send(conn: socket.socket, obj: str) -> bool:
        try:
            conn.sendall((obj + "\n").encode("utf-8"))
            return True
        except OSError:
            return False

    def _send_response(self, conn: socket.socket, resp: ControlResponse) -> bool:
        return self._send(conn, resp.model_dump_json())

    # ---- dispatch --------------------------------------------------------

    def _dispatch(self, conn: socket.socket, line: str) -> None:
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            self._send_response(
                conn,
                ControlResponse(ok=False, error="malformed JSON request", code="invalid_args"),
            )
            return
        if not isinstance(raw, dict) or "cmd" not in raw:
            self._send_response(
                conn,
                ControlResponse(ok=False, error="missing 'cmd'", code="invalid_args"),
            )
            return
        cmd: str = raw["cmd"]
        args = raw.get("args") or {}
        if not isinstance(args, dict):
            self._send_response(
                conn,
                ControlResponse(ok=False, error="'args' must be an object", code="invalid_args"),
            )
            return

        if cmd in self._streamers:
            self._dispatch_stream(conn, cmd, args)
            return
        if cmd not in self._handlers:
            if self._dispatch_fallback is not None:
                try:
                    resp = self._dispatch_fallback(cmd, args)
                except Exception as exc:  # noqa: BLE001 - handler isolation
                    self._send_response(
                        conn,
                        ControlResponse(ok=False, error=str(exc), code="handler_error"),
                    )
                    return
                self._send_response(conn, resp)
                return
            self._send_response(
                conn,
                ControlResponse(ok=False, error=f"unknown command: {cmd}", code="unsupported"),
            )
            return
        try:
            resp = self._handlers[cmd](args)
        except Exception as exc:  # noqa: BLE001 - handler isolation
            self._send_response(
                conn,
                ControlResponse(ok=False, error=str(exc), code="handler_error"),
            )
            return
        self._send_response(conn, resp)

    def _dispatch_stream(self, conn: socket.socket, cmd: str, args: dict) -> None:
        try:
            gen = self._streamers[cmd](args)
            iterator = iter(gen)
            ack = next(iterator)
        except StopIteration:
            self._send_response(
                conn,
                ControlResponse(ok=False, error="stream produced no ack", code="handler_error"),
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._send_response(
                conn,
                ControlResponse(ok=False, error=str(exc), code="handler_error"),
            )
            return
        if not isinstance(ack, ControlResponse):
            ack = ControlResponse(ok=True, data=ack)
        if not self._send_response(conn, ack) or not ack.ok:
            return
        for event in iterator:
            if self._stopping.is_set():
                break
            if not isinstance(event, StreamEvent):
                continue
            if not self._send(conn, event.model_dump_json()):
                break  # client disconnected
