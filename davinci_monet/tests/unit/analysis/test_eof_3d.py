"""3-D EOF produces coupled (mode, lev, lat, lon) patterns and logs the mass-weight fallback."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis.eof import EOFAnalysis
from davinci_monet.config.schema import EOFSpec


def _planted_3d(nt=150, nlev=3, nlat=5, nlon=6, seed=2) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, np.pi, nlon)
    vstruct = np.array([1.0, 0.6, 0.2])[:nlev]
    p1 = vstruct[:, None, None] * np.cos(x)[None, None, :]
    p1 = np.broadcast_to(p1, (nlev, nlat, nlon))
    pc1 = rng.normal(size=nt)
    field = 3.0 * pc1[:, None, None, None] * p1[None] + 0.05 * rng.normal(
        size=(nt, nlev, nlat, nlon)
    )
    lat = np.linspace(-5, 5, nlat)
    lon = np.linspace(0, 30, nlon)
    return xr.Dataset(
        {"O3": (("time", "lev", "lat", "lon"), field, {"units": "ppb"})},
        coords={
            "time": np.arange(nt),
            "lev": np.arange(nlev),
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    )


def test_3d_eof_shapes_and_fallback_logs(caplog) -> None:
    ds = _planted_3d()
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=3)
    with caplog.at_level(logging.WARNING):
        out = EOFAnalysis().analyze(ds, spec)
    assert out["eofs"].dims == ("mode", "lev", "lat", "lon")
    assert out["explained_variance"].values[0] > 0.8  # one dominant coupled mode
    assert any("mass weighting unavailable" in r.message for r in caplog.records)


def test_level_select_reduces_to_2d() -> None:
    ds = _planted_3d()
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=2, level=-1)
    out = EOFAnalysis().analyze(ds, spec)
    assert out["eofs"].dims == ("mode", "lat", "lon")


def test_level_on_surface_field_logs_that_it_is_ignored(caplog) -> None:
    ds = _planted_3d().isel(lev=0, drop=True)
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=2, level=-1)

    with caplog.at_level(logging.WARNING, logger="davinci_monet.analysis.eof"):
        out = EOFAnalysis().analyze(ds, spec)

    assert out["eofs"].dims == ("mode", "lat", "lon")
    assert any(
        "level=-1 ignored for surface-only variable 'O3'" in record.getMessage()
        for record in caplog.records
    )


def test_unrecognized_extra_dimension_is_not_assumed_vertical() -> None:
    ds = _planted_3d().rename({"lev": "member"})
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=2)

    with pytest.raises(ValueError, match="unsupported non-spatial dimension 'member'"):
        EOFAnalysis().analyze(ds, spec)
