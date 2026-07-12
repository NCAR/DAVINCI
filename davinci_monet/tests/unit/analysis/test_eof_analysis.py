"""EOFAnalysis recovers two planted orthogonal patterns from a 2-D field."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.analysis.eof import EOFAnalysis
from davinci_monet.config.schema import EOFSpec


def _planted(nt=200, nlat=6, nlon=8, seed=0) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat = np.linspace(-5, 5, nlat)
    lon = np.linspace(0, 30, nlon)
    x = np.linspace(0, np.pi, nlon)
    p1 = np.cos(x)[None, :] * np.ones((nlat, 1))
    p2 = np.cos(2 * x)[None, :] * np.ones((nlat, 1))
    pc1 = rng.normal(size=nt)
    pc2 = rng.normal(size=nt)
    field = (
        3.0 * pc1[:, None, None] * p1[None]
        + 1.0 * pc2[:, None, None] * p2[None]
        + 0.05 * rng.normal(size=(nt, nlat, nlon))
    )
    ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={
            "time": np.arange(nt),
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    )
    return ds, p1.ravel(), pc1


def _corr(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return abs(float(np.corrcoef(a, b)[0, 1]))


def test_eof_recovers_patterns_and_pcs() -> None:
    ds, p1, pc1 = _planted()
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=3)
    out = EOFAnalysis().analyze(ds, spec)

    # Spatial patterns are stored under "eofs"; PCs under "pc".
    # xarray does not allow a data variable to share its name with one of its
    # dimensions, so the spatial patterns cannot be named "mode" when the mode
    # dimension is also called "mode".
    assert set(out.data_vars) >= {"eofs", "pc", "explained_variance", "explained_variance_error"}
    assert out["eofs"].dims == ("mode", "lat", "lon")
    assert out["pc"].dims == ("time", "mode")
    assert _corr(out["eofs"].sel(mode=1).values, p1) > 0.95
    assert _corr(out["pc"].sel(mode=1).values, pc1) > 0.9
    ev = out["explained_variance"].values
    assert ev[0] > ev[1] > ev[2]
    assert np.all((0.0 <= ev) & (ev <= 1.0))
    assert out["explained_variance"].attrs["display_as_percent"] is True
    assert "percent" not in out["explained_variance"].attrs
    assert float(out["pc"].sel(mode=1).std()) == pytest.approx(1.0, abs=0.05)


def test_sign_is_deterministic() -> None:
    ds, _, _ = _planted(seed=1)
    spec = EOFSpec(type="eof", source="cam", variable="O3", n_modes=2)
    a = EOFAnalysis().analyze(ds, spec)
    b = EOFAnalysis().analyze(ds, spec)
    assert np.allclose(a["eofs"].values, b["eofs"].values)
