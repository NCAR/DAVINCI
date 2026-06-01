"""Daemon outcome notifications: desktop (macOS) + iCloud copy + log.

Supervisor-side ONLY. Does not import matplotlib / xarray / the pipeline.
The actual subprocess runner and the file-copy primitive are injectable so
CI never shells out to a real ``osascript`` and never writes to real iCloud.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence

if TYPE_CHECKING:  # avoid import cycles / keep supervisor import-light
    from davinci_monet.daemon.config import DaemonConfig, WatchRule
    from davinci_monet.daemon.contracts import JobRecord

from davinci_monet.logging import get_logger

logger = get_logger(__name__)

# A runner takes an argv list and returns the process exit code; it raises
# FileNotFoundError if the backend binary is absent.
CommandRunner = Callable[[list[str]], int]


def _default_runner(argv: list[str]) -> int:
    """Run a command, returning its exit code; raises FileNotFoundError if absent."""
    completed = subprocess.run(  # noqa: S603 - argv is a fixed list, no shell
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode


def _applescript_escape(text: str) -> str:
    """Make a string safe to embed inside an AppleScript double-quoted literal."""
    # Backslash-escape embedded double quotes and backslashes.
    return text.replace("\\", "\\\\").replace('"', '\\"')


def send_desktop_notification(
    title: str,
    message: str,
    *,
    runner: CommandRunner | None = None,
) -> bool:
    """Post a macOS desktop notification.

    Tries ``osascript`` first; on failure (non-zero or FileNotFoundError) falls
    back to ``terminal-notifier``. Returns True iff a backend succeeded.
    """
    run = runner or _default_runner
    safe_title = _applescript_escape(title)
    safe_msg = _applescript_escape(message)
    script = f'display notification "{safe_msg}" with title "{safe_title}"'

    try:
        if run(["osascript", "-e", script]) == 0:
            return True
    except FileNotFoundError:
        logger.debug("osascript not found; trying terminal-notifier")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("osascript notification failed: %s", exc)

    try:
        rc = run(["terminal-notifier", "-title", title, "-message", message])
        if rc == 0:
            return True
    except FileNotFoundError:
        logger.debug("terminal-notifier not found")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("terminal-notifier notification failed: %s", exc)

    logger.info("No desktop notification backend available; skipped")
    return False


class DesktopNotifier:
    """Callable wrapper around send_desktop_notification with a bound runner."""

    def __init__(self, *, runner: CommandRunner | None = None) -> None:
        self._runner = runner

    def __call__(self, title: str, message: str) -> bool:
        return send_desktop_notification(title, message, runner=self._runner)


# A copy function takes (src, dst) absolute path strings.
CopyFn = Callable[[str, str], None]


def _default_copy(src: str, dst: str) -> None:
    shutil.copy2(src, dst)


def copy_to_icloud(
    *,
    icloud_dir: str | Path,
    plots: Sequence[str],
    summary_text: str,
    summary_name: str,
    copyfn: CopyFn | None = None,
) -> list[str]:
    """Copy generated plots into ``icloud_dir`` and write a Markdown summary.

    Creates ``icloud_dir`` if needed. Missing source plots are logged and
    skipped. Returns the list of destination paths actually written (plots +
    the summary file). The copy primitive is injectable for tests.
    """
    copy = copyfn or _default_copy
    dest_dir = Path(icloud_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for src in plots:
        src_path = Path(src)
        if not src_path.is_file():
            logger.warning("iCloud copy: source plot missing, skipped: %s", src)
            continue
        dst = dest_dir / src_path.name
        try:
            copy(str(src_path), str(dst))
            written.append(str(dst))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("iCloud copy failed for %s: %s", src, exc)

    summary_path = dest_dir / summary_name
    try:
        summary_path.write_text(summary_text)
        written.append(str(summary_path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("iCloud summary write failed: %s", exc)

    return written


class IcloudCopier:
    """Callable wrapper binding an icloud_dir + copy primitive."""

    def __init__(
        self,
        *,
        icloud_dir: str | Path,
        copyfn: CopyFn | None = None,
    ) -> None:
        self._dir = icloud_dir
        self._copyfn = copyfn

    def __call__(
        self,
        *,
        plots: Sequence[str],
        summary_text: str,
        summary_name: str,
    ) -> list[str]:
        return copy_to_icloud(
            icloud_dir=self._dir,
            plots=plots,
            summary_text=summary_text,
            summary_name=summary_name,
            copyfn=self._copyfn,
        )


class _DesktopProto(Protocol):
    def __call__(self, title: str, message: str) -> bool: ...


class _IcloudProto(Protocol):
    def __call__(
        self,
        *,
        plots: Sequence[str],
        summary_text: str,
        summary_name: str,
    ) -> list[str]: ...


def _resolve_channels(
    cfg: "DaemonConfig",
    rule: "Optional[WatchRule]",
) -> set[str]:
    """Resolve active notification channels.

    Per-rule ``notify:`` (if set) overrides the daemon defaults entirely;
    otherwise channels derive from the daemon NotificationConfig flags. "log"
    is always implicitly active. ``rule`` may be ``None`` (no per-rule override).
    """
    channels: set[str]
    if rule is not None and rule.notify is not None:
        channels = set(rule.notify)
        channels.add("log")
        return channels
    channels = {"log"}
    if cfg.notifications.desktop:
        channels.add("desktop")
    if cfg.notifications.icloud_copy:
        channels.add("icloud")
    return channels


def _build_summary_md(job: "JobRecord") -> str:
    """Render a short Markdown run summary for the iCloud copy."""
    summary = job.result_summary or {}
    lines = [
        f"# DAVINCI run: {job.watch_name} (job {job.id})",
        "",
        f"- status: {job.status.value}",
        f"- config: {job.config_path}",
        f"- submitted_at: {job.submitted_at.isoformat()}",
    ]
    if job.duration_s is not None:
        lines.append(f"- duration_s: {job.duration_s:.1f}")
    if job.files:
        lines.append(f"- files: {len(job.files)}")
    output_dir = summary.get("output_dir")
    if output_dir:
        lines.append(f"- output_dir: {output_dir}")
    if job.error:
        lines.append("")
        lines.append("## error")
        lines.append("```")
        lines.append(str(job.error))
        lines.append("```")
    return "\n".join(lines) + "\n"


def notify_outcome(
    job: "JobRecord",
    cfg: "DaemonConfig",
    rule: "Optional[WatchRule]" = None,
    *,
    desktop: Optional[_DesktopProto] = None,
    icloud: Optional[_IcloudProto] = None,
) -> None:
    """Route a finished job's outcome to log + (optionally) desktop + iCloud.

    Always logs. Posts a desktop notification when the "desktop" channel is
    active. On a COMPLETED job with the "icloud" channel active, copies the
    job's plots + a Markdown summary into ``cfg.notifications.icloud_dir``.

    ``desktop`` / ``icloud`` are injected callables (DesktopNotifier /
    IcloudCopier in production) so this is fully unit-testable with mocks.
    """
    # Late import keeps notify.py import-light at supervisor start.
    from davinci_monet.daemon.contracts import JobStatus

    status_word = job.status.value
    succeeded = job.status == JobStatus.COMPLETED

    # ---- always log -------------------------------------------------------
    log = logger.info if succeeded else logger.warning
    log(
        "Job %s for watch %r %s (config=%s)",
        job.id,
        job.watch_name,
        status_word,
        job.config_path,
    )

    channels = _resolve_channels(cfg, rule)

    # ---- desktop ----------------------------------------------------------
    if "desktop" in channels and desktop is not None:
        title = "DAVINCI"
        msg = f"{job.watch_name} {status_word}"
        if job.duration_s is not None:
            msg += f" in {job.duration_s:.0f}s"
        try:
            desktop(title, msg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("desktop notification raised: %s", exc)

    # ---- iCloud (success only) -------------------------------------------
    if succeeded and "icloud" in channels and icloud is not None:
        summary = job.result_summary or {}
        plots = list(summary.get("plots") or [])
        try:
            icloud(
                plots=plots,
                summary_text=_build_summary_md(job),
                summary_name=f"{job.watch_name}_job{job.id}.md",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("iCloud copy raised: %s", exc)


class Notifier:
    """Thin supervisor-facing facade over ``notify_outcome``.

    The supervisor's ``build_supervisor`` wiring constructs ONE Notifier bound to
    the daemon's NotificationConfig and (optionally) the desktop/iCloud callables,
    then calls :meth:`notify_result` once per finished job. ``hooks`` is a mapping
    of injectable side-effect callables (``{"desktop": ..., "icloud": ...}``);
    when omitted the production ``DesktopNotifier``/``IcloudCopier`` are used.
    """

    def __init__(
        self,
        daemon_cfg: "DaemonConfig",
        *,
        hooks: Optional[dict[str, object]] = None,  # noqa: ANN401
    ) -> None:
        self.daemon_cfg = daemon_cfg
        self.hooks = hooks or {}

    def notify_result(
        self,
        job: "JobRecord",
        rule: "Optional[WatchRule]" = None,
    ) -> None:
        """Route one finished job's outcome through ``notify_outcome``.

        Delegates verbatim to the module-level :func:`notify_outcome`, passing the
        bound NotificationConfig and the injected desktop/iCloud hooks (production
        ``DesktopNotifier``/``IcloudCopier`` when none were supplied).
        """
        _desktop_hook = self.hooks.get("desktop")
        _icloud_hook = self.hooks.get("icloud")
        desktop: _DesktopProto = (
            _desktop_hook  # type: ignore[assignment]
            if _desktop_hook is not None
            else DesktopNotifier()
        )
        icloud: _IcloudProto = (
            _icloud_hook  # type: ignore[assignment]
            if _icloud_hook is not None
            else IcloudCopier(icloud_dir=self.daemon_cfg.notifications.icloud_dir)
        )
        notify_outcome(
            job,
            self.daemon_cfg,
            rule,
            desktop=desktop,
            icloud=icloud,
        )
