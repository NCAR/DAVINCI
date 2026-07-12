"""WaveletAnalysis recovers an injected period and flags it significant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.analysis.wavelet import WaveletAnalysis
from davinci_monet.config.schema import WaveletSpec


def _injected(nt=256, period=16.0, seed=0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    t = np.arange(nt)
    y = np.sin(2 * np.pi * t / period) + 0.3 * rng.normal(size=nt)
    times = pd.date_range("2020-01-01", periods=nt, freq="D")
    field = np.broadcast_to(y[:, None, None], (nt, 2, 2)).copy()
    lat = np.array([-1.0, 1.0])
    lon = np.array([0.0, 1.0])
    return xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={
            "time": times,
            "lat": lat,
            "lon": lon,
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    )


def test_wavelet_recovers_injected_period() -> None:
    spec = WaveletSpec(type="wavelet", source="cam", variable="O3")
    out = WaveletAnalysis().analyze(_injected(period=16.0), spec)

    assert out["power"].dims == ("time", "period")
    assert set(out.data_vars) >= {
        "power",
        "power_significance",
        "coi",
        "global_power",
        "global_significance",
    }
    assert out["period"].attrs.get("units") == "days"

    period = out["period"].values
    gp = out["global_power"].values
    peak = period[int(np.argmax(gp))]
    assert 12.0 < peak < 22.0
    gsig = out["global_significance"].values
    i = int(np.argmax(gp))
    assert gp[i] > gsig[i]


def test_wavelet_reports_nonfinite_input_with_variable_count_and_time() -> None:
    data = _injected()
    data["O3"][10, :, :] = np.nan
    spec = WaveletSpec(type="wavelet", source="cam", variable="O3")

    with pytest.raises(
        ValueError,
        match=r"wavelet input 'O3' has 1 non-finite samples .*2020-01-11.*fill or subset",
    ):
        WaveletAnalysis().analyze(data, spec)
