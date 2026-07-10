"""Small helpers for carrying scientific scenario identity through analyses."""

from __future__ import annotations

from collections.abc import Iterable

import xarray as xr


def consistent_spec_hash(datasets: Iterable[xr.Dataset]) -> str | None:
    """Return the shared spec hash, rejecting mixed hashed inputs."""
    values: set[str] = set()
    for dataset in datasets:
        for raw_key, raw_value in dataset.attrs.items():
            key = str(raw_key)
            if key == "spec_hash" or key.endswith("_spec_hash"):
                value = str(raw_value).strip()
                if value:
                    values.add(value)
    if len(values) > 1:
        raise ValueError("analysis inputs contain inconsistent scientific spec hashes")
    return next(iter(values)) if values else None


__all__ = ["consistent_spec_hash"]
