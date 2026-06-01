"""Daemon outcome notifications: desktop (macOS) + iCloud copy + log.

Supervisor-side ONLY. Does not import matplotlib / xarray / the pipeline.
The actual subprocess runner and the file-copy primitive are injectable so
CI never shells out to a real ``osascript`` and never writes to real iCloud.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

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
