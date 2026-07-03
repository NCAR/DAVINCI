"""Pure parsing helpers for system-info display strings."""

from __future__ import annotations


def _cpu_name_from_cpuinfo(text: str) -> str | None:
    """Extract the CPU model name from ``/proc/cpuinfo`` text.

    Parameters
    ----------
    text
        Contents of ``/proc/cpuinfo`` (or an equivalent newline-delimited
        ``key\\t: value`` formatted string).

    Returns
    -------
    str | None
        The value of the first ``model name`` field, or ``None`` if the
        field is not present.
    """
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.split(":")[1].strip()
    return None
