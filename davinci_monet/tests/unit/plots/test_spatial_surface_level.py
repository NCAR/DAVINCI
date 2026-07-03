"""Tests for surface-level selection in spatial renderers.

Spatial overlays slice a single vertical level out of a 3-D dataset field. The
``surface_level_index`` helper mirrors the pairing-side ``_extract_surface``
auto-detection so overlays default to the **surface**, not the top of
atmosphere — the bug where a CESM-ordered field (pressure increasing with
index) was plotted at index 0 (stratosphere). See the CESM vertical-coordinate
warning in CLAUDE.md.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.plots.renderers.spatial.base import surface_level_index


def _field(z_values: list[float]) -> xr.DataArray:
    nz = len(z_values)
    return xr.DataArray(
        np.zeros((nz, 2, 2)),
        dims=["z", "lat", "lon"],
        coords={"z": ("z", z_values), "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


def test_ascending_pressure_selects_last_index() -> None:
    """CESM convention: pressure increases with index, surface is the last level."""
    assert surface_level_index(_field([10.0, 200.0, 1000.0]), "z") == -1


def test_ascending_altitude_selects_first_index() -> None:
    """Geometric height increases upward, so the surface is the lowest altitude."""
    field = _field([0.0, 1000.0, 5000.0])
    field["z"].attrs.update({"units": "m", "positive": "up"})
    assert surface_level_index(field, "z") == 0


def test_descending_pressure_selects_first_index() -> None:
    """Surface-first ordering: pressure decreases with index, surface is index 0."""
    assert surface_level_index(_field([1000.0, 200.0, 10.0]), "z") == 0


def test_single_level_selects_index_zero() -> None:
    assert surface_level_index(_field([500.0]), "z") == 0


def test_missing_level_coordinate_defaults_to_zero() -> None:
    """With no coordinate values to inspect, fall back to index 0."""
    field = xr.DataArray(np.zeros((3, 2, 2)), dims=["z", "lat", "lon"])
    assert surface_level_index(field, "z") == 0
