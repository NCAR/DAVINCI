"""Interactive control shell for the DAVINCI daemon (`daemon shell`).

A thin REPL over ``DaemonClient``. Each input line is parsed by the pure,
side-effect-free ``parse_command`` into a ``ShellResult`` (control ``cmd`` +
``args`` matching the SHARED-CONTRACT command catalog), then dispatched over the
control socket. Quitting the shell never stops the daemon.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from davinci_monet.daemon.client import DaemonClient
from davinci_monet.daemon.contracts import ControlResponse

# Shell verbs that resolve to a single control cmd with no sub-verb.
SHELL_COMMANDS: tuple[str, ...] = (
    "status",
    "reload",
    "history",
    "watch",
    "logs",
    "job",
    "help",
    "quit",
    "exit",
)

_QUIT_WORDS = {"quit", "exit", "q"}
_HELP_WORDS = {"help", "?"}


@dataclass
class ShellResult:
    """Parsed shell line.

    ``local`` lines (quit/help/blank) never touch the socket; ``cmd`` is None.
    ``streaming`` selects ``DaemonClient.stream`` over ``DaemonClient.call``.
    """

    cmd: Optional[str] = None
    args: dict[str, Any] = field(default_factory=dict)
    streaming: bool = False
    local: bool = False


def _require_name(words: list[str], verb: str) -> str:
    if not words:
        raise ValueError(f"`watch {verb}` requires a watch name")
    return words[0]


def _parse_history(words: list[str]) -> ShellResult:
    watch: Optional[str] = None
    failed = False
    limit = 50
    i = 0
    while i < len(words):
        tok = words[i]
        if tok == "--watch":
            i += 1
            if i >= len(words):
                raise ValueError("--watch requires a value")
            watch = words[i]
        elif tok == "--failed":
            failed = True
        elif tok == "--limit":
            i += 1
            if i >= len(words):
                raise ValueError("--limit requires a value")
            limit = int(words[i])
        else:
            raise ValueError(f"unknown history option: {tok}")
        i += 1
    return ShellResult(
        cmd="history",
        args={"watch": watch, "failed": failed, "limit": limit},
    )


def _parse_watch(words: list[str]) -> ShellResult:
    if not words:
        raise ValueError("`watch` requires a sub-command (list/pause/resume/remove/trigger/save)")
    sub, rest = words[0], words[1:]
    if sub == "add":
        raise ValueError(
            "watch add is not available in the shell; " "use `davinci-monet daemon watch add`"
        )
    if sub == "list":
        return ShellResult(cmd="watch_list", args={})
    if sub == "pause":
        return ShellResult(cmd="watch_pause", args={"name": _require_name(rest, "pause")})
    if sub == "resume":
        return ShellResult(cmd="watch_resume", args={"name": _require_name(rest, "resume")})
    if sub == "remove":
        return ShellResult(cmd="watch_remove", args={"name": _require_name(rest, "remove")})
    if sub == "save":
        return ShellResult(cmd="watch_save", args={"name": _require_name(rest, "save")})
    if sub == "trigger":
        name = _require_name(rest, "trigger")
        return ShellResult(cmd="watch_trigger", args={"name": name, "files": rest[1:]})
    raise ValueError(f"unknown watch sub-command: {sub}")


def _parse_logs(words: list[str]) -> ShellResult:
    if not words:
        raise ValueError("`logs` requires a target (job id or watch name)")
    target: Optional[str] = None
    kind = "job"
    tail = False
    for tok in words:
        if tok == "--watch":
            kind = "watch"
        elif tok == "--tail":
            tail = True
        elif tok.startswith("--"):
            raise ValueError(f"unknown logs option: {tok}")
        elif target is None:
            target = tok
        else:
            raise ValueError(f"unexpected logs argument: {tok}")
    if target is None:
        raise ValueError("`logs` requires a target (job id or watch name)")
    return ShellResult(
        cmd="logs_tail" if tail else "logs",
        args={"target": target, "kind": kind},
        streaming=tail,
    )


def parse_command(line: str) -> ShellResult:
    """Parse one REPL line into a ``ShellResult``. Pure / no I/O."""
    words = shlex.split(line.strip())
    if not words:
        return ShellResult(local=True)
    verb, rest = words[0], words[1:]
    if verb in _QUIT_WORDS:
        return ShellResult(local=True)
    if verb in _HELP_WORDS:
        return ShellResult(local=True)
    if verb == "status":
        return ShellResult(cmd="status", args={})
    if verb == "reload":
        return ShellResult(cmd="reload", args={})
    if verb == "history":
        return _parse_history(rest)
    if verb == "watch":
        return _parse_watch(rest)
    if verb == "logs":
        return _parse_logs(rest)
    if verb == "job":
        if not rest:
            raise ValueError("`job` requires a job id")
        return ShellResult(cmd="job_get", args={"job_id": int(rest[0])})
    raise ValueError(f"unknown command: {verb}")


def format_response(cmd: str, resp: ControlResponse) -> str:
    """Render a one-shot ControlResponse as a short human string for the REPL."""
    if not resp.ok:
        code = f" [{resp.code}]" if resp.code else ""
        return f"error{code}: {resp.error}"
    return f"{cmd}: ok"


class DaemonShell:
    """Line-oriented REPL bound to a single ``DaemonClient``."""

    PROMPT = "davinci-daemon> "

    def __init__(self, client: DaemonClient) -> None:
        self._client = client

    def execute(self, line: str) -> Optional[ControlResponse]:
        """Parse + dispatch one line. Returns the response, or None for local lines.

        Streaming lines drain the client's event iterator to completion (the
        caller's terminal shows the pushed events). Local lines (quit/help/blank)
        return None without touching the socket.
        """
        result = parse_command(line)
        if result.local or result.cmd is None:
            return None
        if result.streaming:
            for _event in self._client.stream(result.cmd, **result.args):
                pass
            return None
        return self._client.call(result.cmd, **result.args)

    def is_quit(self, line: str) -> bool:
        """True if ``line`` is a quit/exit verb (REPL loop sentinel)."""
        words = shlex.split(line.strip())
        return bool(words) and words[0] in _QUIT_WORDS


def run_shell(
    client: DaemonClient,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the interactive daemon REPL until the user quits.

    Reads one line at a time via ``input_fn`` (defaulting to the builtin
    ``input``), dispatches it through a :class:`DaemonShell`, and prints any
    ``ControlResponse`` via ``output_fn``. Quitting (``quit``/``exit``/``q``) or
    EOF/Ctrl-D ends the loop and returns; it NEVER sends ``shutdown`` (the daemon
    keeps running). ``input_fn``/``output_fn`` are injected so the loop is
    unit-testable with scripted lines and no real terminal.
    """
    shell = DaemonShell(client)
    while True:
        try:
            line = input_fn(shell.PROMPT)
        except EOFError:
            output_fn("")
            return
        if shell.is_quit(line):
            return
        try:
            resp = shell.execute(line)
        except ValueError as exc:
            output_fn(f"error: {exc}")
            continue
        if resp is not None:
            stripped = line.strip()
            cmd_word = shlex.split(stripped)[0] if stripped else ""
            output_fn(format_response(cmd_word, resp))
