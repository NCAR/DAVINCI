"""`daemon top` live dashboard renderers.

Pure render functions build Rich panels from a ``DashboardState`` snapshot so
they are unit-testable without the live loop. Colors come from the project's
single styling source of truth, ``davinci_monet.plots.style.NCAR_COLORS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from davinci_monet.plots.style import NCAR_COLORS

_BLUE = NCAR_COLORS["ncar_blue"]
_AQUA = NCAR_COLORS["aqua"]
_GREEN = NCAR_COLORS["green"]
_RED = NCAR_COLORS["red"]
_GRAY = NCAR_COLORS["gray"]
_ORANGE = NCAR_COLORS["orange"]

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
