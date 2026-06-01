"""`daemon top` live dashboard renderers.

Pure render functions build Rich panels from a ``DashboardState`` snapshot so
they are unit-testable without the live loop. NCAR brand colors are inlined below
to keep this module import-clean — importing it must not pull in matplotlib/xarray
(see the supervisor isolation invariant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# NSF NCAR brand hex values, inlined from davinci_monet.plots.style.NCAR_COLORS so
# this client module stays import-clean: importing it must NOT pull in
# matplotlib/xarray/cartopy (which plots.style does), or the supervisor isolation
# invariant breaks if the CLI imports the dashboard at module load.
_BLUE = "#0A5DDA"  # ncar_blue
_AQUA = "#00A2B4"  # aqua
_GREEN = "#2E8B57"  # green
_RED = "#D62839"  # red
_GRAY = "#58595B"  # gray
_ORANGE = "#FF8C00"  # orange

_STATUS_STYLE = {
    "completed": _GREEN,
    "failed": _RED,
    "running": _AQUA,
    "queued": _ORANGE,
    "skipped": _GRAY,
    "paused": _GRAY,
    "armed": _BLUE,
    "settling": _ORANGE,
}


@dataclass
class DashboardState:
    """Snapshot backing one dashboard frame (mirrors control ``StatusData``)."""

    version: int
    pid: int
    uptime_s: float
    draining: bool
    max_concurrent: int
    running: list[dict[str, Any]] = field(default_factory=list)
    queued: list[dict[str, Any]] = field(default_factory=list)
    watches: list[dict[str, Any]] = field(default_factory=list)
    recent: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_status(cls, data: dict[str, Any]) -> "DashboardState":
        """Build a snapshot from a ``status`` control response ``data`` dict."""
        return cls(
            version=int(data.get("version", 0)),
            pid=int(data.get("pid", 0)),
            uptime_s=float(data.get("uptime_s", 0.0)),
            draining=bool(data.get("draining", False)),
            max_concurrent=int(data.get("max_concurrent", 1)),
            running=list(data.get("running", [])),
            queued=list(data.get("queued", [])),
            watches=list(data.get("watches", [])),
            recent=list(data.get("recent", [])),
        )


def _styled_status(status: Optional[str]) -> Text:
    s = status or "-"
    return Text(s, style=_STATUS_STYLE.get(s, _GRAY))


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def render_watches_panel(state: DashboardState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("WATCH", style=f"bold {_BLUE}", no_wrap=True)
    table.add_column("STATE")
    table.add_column("SETTLE", style="dim")
    table.add_column("FIRE", style="dim")
    table.add_column("SRC", style="dim")
    table.add_column("PATTERN", style="dim", overflow="fold")
    for w in state.watches:
        state_str = "paused" if not w.get("enabled", True) else w.get("state", "armed")
        table.add_row(
            str(w.get("name", "?")),
            _styled_status(state_str),
            str(w.get("settle_mode", "-")),
            str(w.get("on_fire", "-")),
            str(w.get("source", "-")),
            str(w.get("watch", "-")),
        )
    if not state.watches:
        table.add_row("—", Text("no watches", style="dim"), "", "", "", "")
    return Panel(table, title="WATCHES", border_style=_BLUE, padding=(0, 1))


def render_running_panel(state: DashboardState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("JOB", style=f"bold {_AQUA}", no_wrap=True)
    table.add_column("WATCH", no_wrap=True)
    table.add_column("PROGRESS", overflow="fold")
    for job in state.running:
        job_id = job.get("id")
        msg = state.progress.get(int(job_id), "starting…") if job_id is not None else "starting…"
        table.add_row(str(job_id), str(job.get("watch_name", "?")), msg)
    if not state.running:
        table.add_row("—", Text("idle", style="dim"), "")
    title = "RUNNING (draining)" if state.draining else "RUNNING"
    return Panel(table, title=title, border_style=_AQUA, padding=(0, 1))


def render_queue_panel(state: DashboardState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("JOB", style=f"bold {_ORANGE}", no_wrap=True)
    table.add_column("WATCH", no_wrap=True)
    table.add_column("FIRE", style="dim")
    for job in state.queued:
        table.add_row(
            str(job.get("id")),
            str(job.get("watch_name", "?")),
            str(job.get("on_fire", "-")),
        )
    if not state.queued:
        table.add_row("—", Text("empty", style="dim"), "")
    return Panel(table, title="QUEUE", border_style=_ORANGE, padding=(0, 1))


def render_header(state: DashboardState) -> Panel:
    line = Text()
    line.append("DAVINCI daemon", style=f"bold {_AQUA}")
    line.append("   ")
    line.append(f"pid {state.pid}", style="dim")
    line.append("   ")
    line.append(f"uptime {_fmt_duration(state.uptime_s)}", style="dim")
    line.append("   ")
    line.append(f"concurrency {state.max_concurrent}", style="dim")
    line.append("   ")
    line.append(f"v{state.version}", style="dim")
    if state.draining:
        line.append("   ")
        line.append("draining", style=f"bold {_ORANGE}")
    return Panel(line, border_style=_AQUA, padding=(0, 2))


def render_recent_panel(state: DashboardState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    table.add_column("JOB", no_wrap=True)
    table.add_column("WATCH", no_wrap=True)
    table.add_column("STATUS")
    table.add_column("DUR", justify="right", style="dim")
    for job in state.recent:
        table.add_row(
            str(job.get("id")),
            str(job.get("watch_name", "?")),
            _styled_status(job.get("status")),
            _fmt_duration(job.get("duration_s")),
        )
    if not state.recent:
        table.add_row("—", Text("no history", style="dim"), "", "")
    return Panel(table, title="RECENT", border_style=_GREEN, padding=(0, 1))


def render_dashboard(state: DashboardState) -> Group:
    """Compose the full `daemon top` frame from a snapshot."""
    return Group(
        render_header(state),
        render_watches_panel(state),
        render_running_panel(state),
        render_queue_panel(state),
        render_recent_panel(state),
    )


def _job_id(data: dict[str, Any]) -> Optional[int]:
    raw = data.get("id", data.get("job_id"))
    return int(raw) if raw is not None else None


def _remove_by_id(jobs: list[dict[str, Any]], job_id: int) -> None:
    jobs[:] = [j for j in jobs if j.get("id") != job_id]


def _apply_job_update(state: DashboardState, data: dict[str, Any]) -> None:
    job_id = _job_id(data)
    if job_id is None:
        return
    status = data.get("status")
    # Drop any stale copy from every bucket, then re-file by status.
    for bucket in (state.running, state.queued, state.recent):
        _remove_by_id(bucket, job_id)
    record = dict(data)
    record.setdefault("id", job_id)
    if status == "running":
        state.running.insert(0, record)
    elif status == "queued":
        state.queued.append(record)
    else:  # completed | failed | skipped
        state.recent.insert(0, record)
        state.progress.pop(job_id, None)


def _apply_progress(state: DashboardState, data: dict[str, Any]) -> None:
    job_id = _job_id(data)
    if job_id is None:
        return
    if data.get("kind") == "stage" and data.get("stage"):
        stage = data["stage"]
        status = data.get("status")
        state.progress[job_id] = f"stage: {stage}" + (f" ({status})" if status else "")
    elif data.get("message") is not None:
        state.progress[job_id] = str(data["message"])


def _apply_watch_update(state: DashboardState, data: dict[str, Any]) -> None:
    name = data.get("name")
    if name is None:
        return
    for i, w in enumerate(state.watches):
        if w.get("name") == name:
            state.watches[i] = dict(data)
            return
    state.watches.append(dict(data))


def apply_stream_event(state: DashboardState, event: "Any") -> None:
    """Fold one pushed ``StreamEvent`` into ``state`` in place (pure reducer)."""
    data = dict(getattr(event, "data", {}) or {})
    kind = getattr(event, "event", None)
    if kind == "job_update":
        # A stage-only job_update carries no top-level status -> treat as progress.
        if data.get("status") is None and (data.get("stage") or data.get("message")):
            _apply_progress(state, data)
        else:
            _apply_job_update(state, data)
    elif kind in ("log_line", "progress"):
        _apply_progress(state, data)
    elif kind == "watch_update":
        _apply_watch_update(state, data)
    elif kind == "stats":
        if "running" in data:
            state.running = list(data["running"])
        if "queued" in data:
            state.queued = list(data["queued"])
    # Unknown events are intentionally ignored.


def run_dashboard(
    client: "Any",
    console: "Any" = None,
    refresh_per_second: float = 2.0,
    topics: Optional[list[str]] = None,
) -> None:  # pragma: no cover - blocking live loop, not unit-tested
    """Blocking `daemon top` loop: seed from ``status``, fold ``subscribe`` events."""
    from rich.console import Console
    from rich.live import Live

    console = console or Console()
    topics = topics or ["jobs", "watches", "stats"]

    seed = client.call("status")
    state = (
        DashboardState.from_status(seed.data)
        if getattr(seed, "ok", False) and seed.data
        else DashboardState(version=0, pid=0, uptime_s=0.0, draining=False, max_concurrent=1)
    )

    with Live(
        render_dashboard(state),
        console=console,
        refresh_per_second=refresh_per_second,
        screen=True,
    ) as live:
        for event in client.stream("subscribe", topics=topics):
            apply_stream_event(state, event)
            live.update(render_dashboard(state))
