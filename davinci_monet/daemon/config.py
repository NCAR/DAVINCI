"""Daemon configuration models and watches.yaml loader.

Owns the three pydantic config models (NotificationConfig, WatchRule,
DaemonConfig), the WatchesFile aggregate, the parse_duration helper, and the
load_watches / merge_rules functions. Runtime primitive enums/literals
(OnFireMode, NotifyChannel, SettleMode, WatchSource) are imported from
davinci_monet.daemon.contracts and never redefined here.

Pydantic style mirrors davinci_monet/config/schema.py (StrictModel /
FlexibleModel). Layer-1 ${VAR} expansion uses local helpers (os + yaml only)
so that the long-lived SUPERVISOR never transitively imports xarray/pandas.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, TextIO

import yaml
from pydantic import Field, field_validator

# Import ONLY from leaf submodules to avoid triggering the heavy geo/data stack:
#   - davinci_monet.config.schema  — pure pydantic, no xarray
#   - davinci_monet.daemon.contracts — pure pydantic + stdlib, no xarray
# Do NOT import from davinci_monet.config.parser (pulls in MonetConfig ->
# config.schema -> config.__init__ -> ... -> xarray) or from
# davinci_monet.core.exceptions (running core/__init__.py imports
# core.base/types/protocols which pull xarray).
from davinci_monet.config.schema import FlexibleModel, StrictModel
from davinci_monet.daemon.contracts import NotifyChannel, OnFireMode, SettleMode, WatchSource

# ---------------------------------------------------------------------------
# Lightweight YAML helpers (stdlib only — no project imports)
# ---------------------------------------------------------------------------


class ConfigurationError(Exception):
    """Raised by load_watches when a watches.yaml file is invalid.

    Defined here (not imported from davinci_monet.core.exceptions) so that
    importing this module does NOT trigger davinci_monet/core/__init__.py,
    which eagerly re-exports core.base / core.types / core.protocols and
    thereby drags xarray/pandas into the supervisor process.
    """


def _load_yaml(source: str | Path | TextIO) -> dict[str, Any]:
    """Load raw YAML from a file path or YAML string (stdlib + PyYAML only)."""
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                with open(path) as fh:
                    data = yaml.safe_load(fh)
            else:
                # Treat as a YAML string (useful in tests / pipes)
                data = yaml.safe_load(str(source))
        else:
            data = yaml.safe_load(source)

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"YAML root must be a mapping, got {type(data)}")
        return data
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML: {exc}") from exc
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {source}") from exc


def _expand_env_vars(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively expand ${VAR}/$VAR in string values (os.path.expandvars)."""

    def _expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {k: _expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_expand(item) for item in value]
        return value

    result: dict[str, Any] = _expand(data)
    return result


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


class WatchesFile(StrictModel):
    """The fully-parsed watches.yaml (daemon policy + the declared rules).

    ``watches`` is keyed by rule name; each value is a fully-constructed
    WatchRule whose ``name`` matches its key.
    """

    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    watches: dict[str, WatchRule] = Field(default_factory=dict)


def merge_rules(
    declared: dict[str, WatchRule],
    live: dict[str, WatchRule],
    disabled: set[str],
) -> dict[str, WatchRule]:
    """Reconcile file-declared rules with state-store live/runtime state.

    ``declared`` = rules from watches.yaml (source="file"). ``live`` =
    runtime-added rules from watch_status (source="live"). ``disabled`` = names
    paused at runtime. File-declared rules win on name collision. Each returned
    rule's ``enabled`` reflects the ``disabled`` set. Inputs are not mutated.
    """
    merged: dict[str, WatchRule] = {}

    # Live-added rules first; declared overrides any same-named live entry.
    for name, rule in live.items():
        merged[name] = rule
    for name, rule in declared.items():
        merged[name] = rule

    # Apply runtime pause/resume without mutating the source objects.
    result: dict[str, WatchRule] = {}
    for name, rule in merged.items():
        enabled = name not in disabled
        if rule.enabled != enabled:
            result[name] = rule.model_copy(update={"enabled": enabled})
        else:
            result[name] = rule
    return result


def load_watches(source: str | Path) -> WatchesFile:
    """Load + layer-1 env-expand + validate a watches.yaml into a WatchesFile.

    Uses local _load_yaml + _expand_env_vars for ${VAR} expansion against the
    DAEMON's os.environ, then constructs DaemonConfig and each WatchRule
    (injecting the mapping key as ``name``). on_fire == "new_files_only" with
    no ``inject_into`` is a validation error.
    """
    raw = _load_yaml(source)

    daemon_raw = raw.get("daemon") or {}
    if not isinstance(daemon_raw, dict):
        raise ConfigurationError("watches.yaml 'daemon' block must be a mapping")
    # Layer-1: expand the whole daemon policy block (paths, icloud_dir, ...).
    daemon_raw = _expand_env_vars(daemon_raw)

    watches_raw = raw.get("watches") or {}
    if not isinstance(watches_raw, dict):
        raise ConfigurationError("watches.yaml 'watches' block must be a mapping")

    try:
        daemon = DaemonConfig.model_validate(daemon_raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigurationError(f"Invalid daemon config: {exc}") from exc

    rules: dict[str, WatchRule] = {}
    for name, rule_raw in watches_raw.items():
        if rule_raw is None:
            rule_raw = {}
        if not isinstance(rule_raw, dict):
            raise ConfigurationError(f"watch '{name}' must be a mapping")

        # Layer-1: expand path-bearing string fields only; preserve env verbatim
        # (env is the layer-2 worker overlay and is expanded inside the worker).
        rule_data: dict[str, Any] = dict(rule_raw)
        env_overlay = rule_data.pop("env", None)
        rule_data = _expand_env_vars(rule_data)
        if env_overlay is not None:
            rule_data["env"] = env_overlay
        rule_data["name"] = str(name)

        try:
            rule = WatchRule.model_validate(rule_data)
        except Exception as exc:  # pydantic ValidationError (e.g. bad on_fire)
            raise ConfigurationError(f"Invalid watch '{name}': {exc}") from exc

        if rule.on_fire == "new_files_only" and not rule.inject_into:
            raise ConfigurationError(
                f"watch '{name}': on_fire 'new_files_only' requires 'inject_into'"
            )
        rules[str(name)] = rule

    return WatchesFile(daemon=daemon, watches=rules)
