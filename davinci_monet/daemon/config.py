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

from typing import Optional

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
