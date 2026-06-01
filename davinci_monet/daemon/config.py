"""Daemon configuration models and watches.yaml loader.

Owns the three pydantic config models (NotificationConfig, WatchRule,
DaemonConfig), the WatchesFile aggregate, the parse_duration helper, and the
load_watches / merge_rules functions. Runtime primitive enums/literals
(OnFireMode, NotifyChannel, SettleMode, WatchSource) are imported from
davinci_monet.daemon.contracts and never redefined here.

Pydantic style mirrors davinci_monet/config/schema.py (StrictModel /
FlexibleModel). Layer-1 ${VAR} expansion reuses the project config parser.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, field_validator

from davinci_monet.config.schema import FlexibleModel, StrictModel
from davinci_monet.daemon.contracts import (
    NotifyChannel,
    OnFireMode,
    SettleMode,
    WatchSource,
)

_DURATION_UNITS: dict[str, float] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def parse_duration(value: str | int | float | None) -> Optional[float]:
    """Parse a human duration like "30s", "5m", "2h", "1d" -> float seconds.

    Accepted suffixes (case-insensitive): s, m, h, d. A bare number (int/float
    or unsuffixed string) is interpreted as seconds. ``None`` -> ``None``.

    Raises
    ------
    ValueError
        On an unparseable / negative value.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise ValueError(f"Invalid duration: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"Duration must be non-negative: {value!r}")
        return seconds

    text = str(value).strip().lower()
    if not text:
        raise ValueError("Duration string is empty")

    unit = 1.0
    if text[-1] in _DURATION_UNITS:
        unit = _DURATION_UNITS[text[-1]]
        text = text[:-1].strip()
    elif text[-1].isalpha():
        raise ValueError(f"Unknown duration suffix in {value!r}")

    try:
        magnitude = float(text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse duration: {value!r}") from exc

    if magnitude < 0:
        raise ValueError(f"Duration must be non-negative: {value!r}")
    return magnitude * unit


def _expand_path_str(value: str) -> str:
    """Layer-1 path expansion: ${VAR}/$VAR then ~ (daemon environment)."""
    return os.path.expanduser(os.path.expandvars(value))


class NotificationConfig(FlexibleModel):
    """Daemon-level notification policy (the ``daemon.notifications`` block)."""

    desktop: bool = True
    icloud_copy: bool = True
    icloud_dir: Path = Field(default=Path("~/Library/Mobile Documents/com~apple~CloudDocs/Claude"))

    @field_validator("icloud_dir", mode="before")
    @classmethod
    def _expand_user(cls, v: Any) -> Any:
        """Expand ~ and ${VAR} at daemon load (layer-1 expansion)."""
        if v is None:
            return v
        return Path(_expand_path_str(str(v)))


class WatchRule(FlexibleModel):
    """A single declarative watch rule (one entry under ``watches:``).

    ``name`` is the rule's mapping key in watches.yaml; the loader injects it.
    Layer-1 ${VAR} expansion (watch/run/sentinel paths) has already been
    applied by the time a WatchRule is constructed. ``env`` is the per-rule
    overlay used for layer-2 (worker-side) expansion and is NOT expanded here.
    """

    name: str
    watch: str
    run: str
    on_fire: OnFireMode = "whole_config"
    inject_into: Optional[str] = None
    settle: float = Field(default=30.0)
    sentinel: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    notify: Optional[list[NotifyChannel]] = None
    enabled: bool = True

    @field_validator("settle", mode="before")
    @classmethod
    def _parse_settle(cls, v: Any) -> Any:
        """Accept "30s"/"5m"/number via parse_duration."""
        if v is None:
            return 30.0
        return parse_duration(v)

    @property
    def settle_mode(self) -> SettleMode:
        """'sentinel' if a sentinel path is set, else 'quiescence'."""
        return "sentinel" if self.sentinel else "quiescence"


class DaemonConfig(FlexibleModel):
    """Top-level daemon policy (the ``daemon:`` block of watches.yaml)."""

    state_dir: Path = Field(default=Path("~/.davinci/daemon"))
    poll_interval: float = Field(default=5.0)
    max_concurrent: int = 1
    hdf5_file_locking: bool = False
    max_settle_wait: Optional[float] = Field(default=1800.0)
    worker_timeout: Optional[float] = None
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    @field_validator("poll_interval", "max_settle_wait", "worker_timeout", mode="before")
    @classmethod
    def _parse_durations(cls, v: Any) -> Any:
        """Accept "5s"/"30m"/number via parse_duration; pass None through."""
        if v is None:
            return None
        return parse_duration(v)

    @field_validator("state_dir", mode="before")
    @classmethod
    def _expand_state_dir(cls, v: Any) -> Any:
        """Expand ~ and ${VAR} at daemon load (layer-1)."""
        if v is None:
            return v
        return Path(_expand_path_str(str(v)))

    @property
    def db_path(self) -> Path:
        return self.state_dir / "history.db"

    @property
    def socket_path(self) -> Path:
        return self.state_dir / "control.sock"

    @property
    def pid_path(self) -> Path:
        return self.state_dir / "daemon.pid"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "daemon.lock"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "daemon.log"
