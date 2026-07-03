"""Integration: wavelet runs through the pipeline, including on an EOF PC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner


def _grid_nc(path: Path) -> None:
    times = pd.date_range("2020-01-01", periods=256, freq="D")
    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(0, 30, 8)
    t = np.arange(len(times))
    x = np.linspace(0, np.pi, len(lon))
    rng = np.random.default_rng(0)
    pc1 = np.sin(2 * np.pi * t / 16.0) + 0.3 * rng.normal(size=len(times))
    p1 = np.cos(x)[None, :] * np.ones((len(lat), 1))
    field = 3.0 * pc1[:, None, None] * p1[None] + 0.1 * rng.normal(
        size=(len(times), len(lat), len(lon))
    )
    xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={
            "time": times,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    ).to_netcdf(path)


def _monthly_grid_nc(path: Path, n_months: int = 48) -> None:
    times = pd.date_range("2018-01-01", periods=n_months, freq="MS")
    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(0, 30, 8)
    t = np.arange(n_months)
    x = np.linspace(0, np.pi, len(lon))
    rng = np.random.default_rng(0)
    pc1 = np.sin(2 * np.pi * t / 12.0) + 0.3 * rng.normal(size=n_months)
    p1 = np.cos(x)[None, :] * np.ones((len(lat), 1))
    field = 3.0 * pc1[:, None, None] * p1[None] + 0.1 * rng.normal(
        size=(n_months, len(lat), len(lon))
    )
    xr.Dataset(
        {"O3": (("time", "lat", "lon"), field, {"units": "ppb"})},
        coords={
            "time": times,
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "latitude": ("lat", lat),
            "longitude": ("lon", lon),
        },
    ).to_netcdf(path)


@pytest.mark.integration
def test_areamean_wavelet(tmp_path: Path) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}
        },
        "analyses": {
            "cam_wav": {"type": "wavelet", "source": "cam", "variable": "O3", "reduce": "area_mean"}
        },
        "plots": {"scal": {"type": "wavelet_scalogram", "source": "cam_wav", "variable": "power"}},
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    assert "cam_wav" in ctx.sources
    pngs = [p for p in ctx.results["plotting"].data["plots_generated"] if p.endswith(".png")]
    assert any("scal" in p for p in pngs)


@pytest.mark.integration
def test_wavelet_of_eof_pc(tmp_path: Path) -> None:
    src = tmp_path / "grid.nc"
    _grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}
        },
        "analyses": {
            "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 3},
            "pc1_wav": {"type": "wavelet", "source": "cam_O3_eof", "variable": "pc", "mode": 1},
        },
        "plots": {"scal": {"type": "wavelet_scalogram", "source": "pc1_wav", "variable": "power"}},
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    assert "pc1_wav" in ctx.sources
    pngs = [p for p in ctx.results["plotting"].data["plots_generated"] if p.endswith(".png")]
    assert any("scal" in p for p in pngs)


@pytest.mark.integration
def test_wavelet_of_eof_pc_on_monthly_axis(tmp_path: Path) -> None:
    src = tmp_path / "monthly.nc"
    _monthly_grid_nc(src)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "cam": {"type": "generic", "files": str(src), "variables": {"O3": {"units": "ppb"}}}
        },
        "analyses": {
            "cam_O3_eof": {"type": "eof", "source": "cam", "variable": "O3", "n_modes": 3},
            "pc1_wav": {"type": "wavelet", "source": "cam_O3_eof", "variable": "pc", "mode": 1},
        },
        "plots": {"scal": {"type": "wavelet_scalogram", "source": "pc1_wav", "variable": "power"}},
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None
    assert "pc1_wav" in ctx.sources
    assert not ctx.metadata.get("analysis_errors")
    pngs = [p for p in ctx.results["plotting"].data["plots_generated"] if p.endswith(".png")]
    scals = [p for p in pngs if "scal" in p]
    assert scals
    assert Path(scals[0]).exists()
