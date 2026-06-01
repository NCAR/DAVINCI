"""CLI command: the `daemon` sub-app — file-watching automation daemon.

`serve` runs the supervisor in the foreground. `start`/`stop`/`status`/`reload`/
`logs`/`history` and the nested `watch` verbs are thin clients that talk to a
running daemon over its Unix control socket via DaemonClient. This module never
imports the pipeline/sci-stack; the supervisor it launches forks workers that do.
"""

from __future__ import annotations

from typing import Any, Optional

import typer

# Define colour constants locally so importing this module never triggers
# davinci_monet.cli.app.register_commands(), which would pull in the pipeline
# (xarray / matplotlib / monet) at module load time.
ERROR_COLOR: str = typer.colors.BRIGHT_RED
INFO_COLOR: str = typer.colors.CYAN
SUCCESS_COLOR: str = typer.colors.GREEN

app = typer.Typer(
    name="daemon",
    help="File-watching automation daemon: auto-run analyses when data lands.",
)

watch_app = typer.Typer(
    name="watch",
    help="Manage individual watch rules (list/add/remove/pause/resume/trigger/save).",
)
app.add_typer(watch_app, name="watch")

_DEFAULT_WATCHES = "watches.yaml"


def _load_daemon_config(watches: str) -> Any:
    """Load + layer-1 expand watches.yaml -> WatchesFile (daemon policy + rules)."""
    from davinci_monet.daemon.config import load_watches

    return load_watches(watches)


def _make_client(watches: str) -> Any:
    """Build a DaemonClient bound to the socket from watches.yaml's state_dir."""
    from davinci_monet.daemon.client import DaemonClient

    cfg = _load_daemon_config(watches).daemon
    return DaemonClient(cfg.socket_path)


def _require_alive(client: Any) -> None:
    """Abort with a clear message if no daemon is answering the socket."""
    if not client.is_alive():
        typer.secho(
            "Daemon is not running (no response on the control socket). "
            "Start it with `davinci-monet daemon start` or `daemon serve`.",
            fg=ERROR_COLOR,
        )
        raise typer.Exit(1)


def _call(client: Any, cmd: str, **args: Any) -> Any:
    """Round-trip one command, aborting on a non-ok response."""
    resp = client.call(cmd, **args)
    if not resp.ok:
        typer.secho(f"Error: {resp.error}", fg=ERROR_COLOR)
        raise typer.Exit(1)
    return resp.data


@app.command()
def serve(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Run the daemon in the foreground (logs to stdout; tmux/systemd-friendly)."""
    from davinci_monet.daemon import lifecycle
    from davinci_monet.daemon.control import ControlServer
    from davinci_monet.daemon.supervisor import build_supervisor

    wf = _load_daemon_config(watches)
    supervisor = build_supervisor(wf, watches_path=watches)
    server = ControlServer(wf.daemon.socket_path, supervisor.handle_command)
    typer.secho(f"DAVINCI daemon serving ({len(wf.watches)} watches)...", fg=INFO_COLOR)
    lifecycle.install_signal_handlers(supervisor.request_shutdown)
    supervisor.serve(control_server=server)
    typer.secho("Daemon stopped.", fg=SUCCESS_COLOR)


@app.command()
def start(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Start the daemon in the background (PID + lock file; logs to daemon.log)."""
    from davinci_monet.daemon import lifecycle

    wf = _load_daemon_config(watches)

    def _serve() -> None:
        # Runs only in the backgrounded daemon child (after the double-fork).
        from davinci_monet.daemon.control import ControlServer
        from davinci_monet.daemon.supervisor import build_supervisor

        supervisor = build_supervisor(wf, watches_path=watches)
        server = ControlServer(wf.daemon.socket_path, supervisor.handle_command)
        lifecycle.install_signal_handlers(supervisor.request_shutdown)
        supervisor.serve(control_server=server)

    result = lifecycle.start_background(wf.daemon, _serve)
    if not result.started:
        typer.secho(result.message, fg=ERROR_COLOR)
        raise typer.Exit(1)
    typer.secho(result.message, fg=SUCCESS_COLOR)


@app.command()
def stop(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", help="Seconds to wait for graceful drain before giving up."
    ),
) -> None:
    """Stop a background daemon (graceful drain via SIGTERM)."""
    client = _make_client(watches)
    if client.is_alive():
        client.call("shutdown", drain=True, timeout=timeout)
        typer.secho("Shutdown requested (draining).", fg=INFO_COLOR)
        return
    from davinci_monet.daemon import lifecycle

    wf = _load_daemon_config(watches)
    result = lifecycle.stop_background(wf.daemon)
    if result.started:
        typer.secho(result.message, fg=SUCCESS_COLOR)
    else:
        typer.secho(result.message, fg=ERROR_COLOR)
        raise typer.Exit(1)


@app.command()
def status(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Print a one-shot summary of the running daemon."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "status")
    typer.secho(
        f"daemon up | pid {data['pid']} | uptime {data['uptime_s']:.0f}s | "
        f"max_concurrent {data['max_concurrent']} | "
        f"draining {data['draining']}",
        fg=INFO_COLOR,
    )
    typer.echo(f"running {len(data['running'])} | queued {len(data['queued'])}")
    typer.echo("watches:")
    for w in data["watches"]:
        flag = "paused" if not w["enabled"] else w["state"]
        typer.echo(f"  {w['name']:<20} {flag:<10} {w['settle_mode']:<10} -> {w['run']}")


@app.command()
def reload(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Re-read watches.yaml and reconcile declared vs. live rules."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "reload")
    typer.secho(
        f"reloaded | added {data.get('added', [])} | "
        f"removed {data.get('removed', [])} | updated {data.get('updated', [])}",
        fg=SUCCESS_COLOR,
    )


@app.command()
def shell(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Open the interactive daemon REPL."""
    from davinci_monet.daemon.shell import run_shell

    client = _make_client(watches)
    _require_alive(client)
    run_shell(client)


@app.command()
def top(
    watches: str = typer.Argument(_DEFAULT_WATCHES, help="Path to watches.yaml."),
) -> None:
    """Open the live dashboard (streams job/watch/stats updates)."""
    from davinci_monet.daemon.dashboard import run_dashboard

    client = _make_client(watches)
    _require_alive(client)
    run_dashboard(client)


@app.command()
def logs(
    target: str = typer.Argument(..., help="Job id or watch name whose log to show."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
    kind: str = typer.Option("job", "--kind", help="Resolve target as 'job' or 'watch'."),
    tail: bool = typer.Option(False, "--tail", help="Stream the log live until Ctrl-C."),
) -> None:
    """Show (or tail) the log for a job or watch."""
    client = _make_client(watches)
    _require_alive(client)
    if tail:
        for event in client.stream("logs_tail", target=target, kind=kind):
            d = event.data or {}
            line = d.get("line") or d.get("message", "")
            typer.echo(line)
        return
    data = _call(client, "logs", target=target, kind=kind)
    typer.echo(data.get("text", ""))


@app.command()
def history(
    watch: Optional[str] = typer.Option(None, "--watch", help="Filter by watch name."),
    failed: bool = typer.Option(False, "--failed", help="Only failed jobs."),
    limit: int = typer.Option(50, "--limit", help="Max rows (most recent first)."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Show recent job history from the daemon's state store."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "history", watch=watch, failed=failed, limit=limit)
    jobs = data.get("jobs", [])
    if not jobs:
        typer.echo("(no jobs)")
        return
    for job in jobs:
        typer.echo(
            f"#{job['id']:<5} {job['watch_name']:<18} {job['status']:<10} "
            f"{job.get('submitted_at', '')}"
        )


# ---- watch verbs ----------------------------------------------------------


@watch_app.command("list")
def watch_list(
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """List all watch rules and their runtime state."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_list")
    for w in data["watches"]:
        flag = "paused" if not w["enabled"] else w["state"]
        typer.echo(
            f"{w['name']:<20} {flag:<10} {w['source']:<6} {w['settle_mode']:<10} -> {w['run']}"
        )


@watch_app.command("pause")
def watch_pause(
    name: str = typer.Argument(..., help="Watch rule name to pause."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Pause a watch rule (stops firing until resumed)."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_pause", name=name)
    typer.secho(f"paused {data['name']}", fg=INFO_COLOR)


@watch_app.command("resume")
def watch_resume(
    name: str = typer.Argument(..., help="Watch rule name to resume."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Resume a paused watch rule."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_resume", name=name)
    typer.secho(f"resumed {data['name']}", fg=SUCCESS_COLOR)


@watch_app.command("trigger")
def watch_trigger(
    name: str = typer.Argument(..., help="Watch rule name to fire manually."),
    files: list[str] = typer.Option(
        [], "--file", "-f", help="Specific new file path(s) to inject (repeatable)."
    ),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Manually fire a watch rule now (optionally with explicit new files)."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_trigger", name=name, files=list(files))
    if data.get("coalesced"):
        typer.secho(f"{name}: coalesced into an already-pending run.", fg=INFO_COLOR)
    else:
        typer.secho(f"{name}: queued.", fg=SUCCESS_COLOR)


@watch_app.command("remove")
def watch_remove(
    name: str = typer.Argument(..., help="Watch rule name to remove."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Remove a (live-added) watch rule."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_remove", name=name)
    typer.secho(f"removed {data['removed']}", fg=INFO_COLOR)


@watch_app.command("add")
def watch_add(
    rule_yaml: str = typer.Argument(..., help="Path to a YAML file with one watch rule."),
    save: bool = typer.Option(False, "--save", help="Also write the rule back to watches.yaml."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Add a watch rule at runtime from a small YAML snippet."""
    import yaml

    from davinci_monet.config.parser import expand_env_vars

    client = _make_client(watches)
    _require_alive(client)
    with open(rule_yaml) as f:
        raw = expand_env_vars(yaml.safe_load(f) or {})
    data = _call(client, "watch_add", rule=raw, save=save)
    typer.secho(f"added {data['added']}", fg=SUCCESS_COLOR)


@watch_app.command("save")
def watch_save(
    name: str = typer.Argument(..., help="Live watch rule name to persist to file."),
    watches: str = typer.Option(_DEFAULT_WATCHES, "--watches", help="Path to watches.yaml."),
) -> None:
    """Persist a live-added watch rule back into watches.yaml."""
    client = _make_client(watches)
    _require_alive(client)
    data = _call(client, "watch_save", name=name)
    typer.secho(f"saved {data['saved']} -> {data['path']}", fg=SUCCESS_COLOR)
