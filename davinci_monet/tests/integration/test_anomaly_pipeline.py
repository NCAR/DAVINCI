"""Integration: an anomaly analysis runs through the pipeline and plots.

Drives ``PipelineRunner.run_from_config()`` -- the same path as
``davinci-monet run config.yaml`` -- so the derived pseudo-source, its
geometry, and the single-source plot path are all exercised together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.runner import PipelineRunner


def _point_nc(path: Path) -> None:
    """Monthly (time, site) series: seasonal cycle + per-site offset + warming."""
    times = pd.date_range("1991-01-01", periods=360, freq="MS")
    season = 20.0 * np.sin(2 * np.pi * (times.month.to_numpy() - 1) / 12.0)
    warming = np.linspace(0.0, 3.0, len(times))
    offsets = np.array([0.0, 40.0])
    values = (season + warming)[:, None] + offsets[None, :]
    lat = np.array([40.02, 19.54])
    lon = np.array([-105.27, -155.58])
    xr.Dataset(
        {"T2M": (("time", "site"), values, {"units": "K", "display_name": "2 m Temperature"})},
        coords={
            "time": times,
            "site": ("site", ["Boulder, CO", "Mauna Loa, HI"]),
            "latitude": ("site", lat),
            "longitude": ("site", lon),
        },
    ).to_netcdf(path)


def _config(src: Path, out: Path, **anomaly: object) -> dict:
    spec = {
        "type": "anomaly",
        "source": "power",
        "variable": "T2M",
        "baseline_start": "1991-01-01",
        "baseline_end": "2020-12-31",
        **anomaly,
    }
    return {
        "analysis": {"output_dir": str(out)},
        "sources": {
            "power": {
                "type": "pt_sfc",
                "filename": str(src),
                "variables": {"T2M": {"units": "K"}},
            }
        },
        "analyses": {"t2m_anomaly": spec},
        "plots": {
            "t2m_anomaly_record": {
                "type": "timeseries",
                "source": "t2m_anomaly",
                "variable": "T2M",
                "show_individual_sites": True,
                "site_label_var": "site",
                "title": "2 m Temperature Anomaly",
            }
        },
    }


@pytest.mark.integration
def test_anomaly_plots_through_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "points.nc"
    _point_nc(src)
    out = tmp_path / "out"

    result = PipelineRunner(show_progress=False).run_from_config(_config(src, out, smooth=12))

    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None

    # The derived source is registered and keeps the input's POINT geometry,
    # which is what lets the single-source timeseries path render it.
    derived = ctx.sources["t2m_anomaly"]
    assert derived.geometry is DataGeometry.POINT
    assert "T2M" in derived.data.data_vars
    assert derived.data["T2M"].dims == ("time", "site")

    plots = ctx.results["plotting"].data["plots_generated"]
    assert any("t2m_anomaly_record" in p and p.endswith(".png") for p in plots)
    assert (out / "t2m_anomaly_record.png").exists()


@pytest.mark.integration
def test_the_pipeline_anomaly_is_referenced_to_the_configured_baseline(tmp_path: Path) -> None:
    """The seasonal cycle is gone and the warming ramp survives, end to end."""
    src = tmp_path / "points.nc"
    _point_nc(src)

    result = PipelineRunner(show_progress=False).run_from_config(_config(src, tmp_path / "out"))

    assert result.success, getattr(result, "error", None)
    assert result.context is not None
    anomaly = result.context.sources["t2m_anomaly"].data["T2M"]

    # Seasonal cycle removed: what remains is the ramp, which is monotonic in
    # the annual mean rather than oscillating by +/-20 K.
    assert float(anomaly.max()) < 5.0
    early = float(anomaly.isel(time=slice(0, 12)).mean())
    late = float(anomaly.isel(time=slice(-12, None)).mean())
    assert late - early == pytest.approx(3.0, abs=0.2)


@pytest.mark.integration
def test_an_impossible_baseline_is_reported_not_silently_blank(tmp_path: Path) -> None:
    """A baseline outside the record must surface as an analysis error."""
    src = tmp_path / "points.nc"
    _point_nc(src)

    result = PipelineRunner(show_progress=False).run_from_config(
        _config(src, tmp_path / "out", baseline_start="1950-01-01", baseline_end="1955-12-31")
    )

    assert result.context is not None
    errors = result.context.metadata.get("analysis_errors") or []
    assert any("selects no times" in e for e in errors), errors
