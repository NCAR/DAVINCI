"""Pipeline integration tests for the NASA POWER source.

These run through ``PipelineRunner.run_from_config()`` -- the same code path
as ``davinci-monet run config.yaml`` -- not the reader API directly. The only
thing stubbed is the network seam: ``fetch_to_cache`` serves synthetic
POWER-shaped NetCDF, so the load -> pair -> stats -> plot chain is real.

Shapes mirror the live API measured 2026-07-15: point responses come back as
``(time, lat=1, lon=1)`` with no site dim, and units are POWER's native ones
(``C``, ``kW-hr/m^2/day``), so the catalog's normalization runs for real.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.datasets import power as power_reader
from davinci_monet.io.download import power as power_client
from davinci_monet.pipeline.runner import PipelineRunner

SITES = [
    {"name": "boulder", "latitude": 40.02, "longitude": -105.27},
    {"name": "table_mtn", "latitude": 40.125, "longitude": -105.24},
]
N_TIME = 24
TIMES = np.array([np.datetime64("2024-02-01") + np.timedelta64(i, "h") for i in range(N_TIME)])


def _power_point_response(site: dict, seed: int) -> xr.Dataset:
    """Synthetic POWER hourly point response: (time, lat=1, lon=1), native units."""
    rng = np.random.default_rng(seed)
    t2m_c = 5.0 + 8.0 * np.sin(np.arange(N_TIME) * np.pi / 12) + rng.normal(0, 0.3, N_TIME)
    return xr.Dataset(
        {
            "T2M": (
                ("time", "lat", "lon"),
                t2m_c.astype("float32").reshape(N_TIME, 1, 1),
                {"units": "C", "long_name": "Temperature at 2 Meters"},
            ),
        },
        coords={"time": TIMES, "lat": [site["latitude"]], "lon": [site["longitude"]]},
    )


def _model_grid(path: Path) -> Path:
    """A synthetic gridded model source carrying T2M in kelvin."""
    lats = np.arange(39.5, 41.0, 0.5)
    lons = np.arange(-106.0, -104.0, 0.5)
    rng = np.random.default_rng(7)
    base = 278.15 + 8.0 * np.sin(np.arange(N_TIME) * np.pi / 12)
    data = base[:, None, None] + rng.normal(0, 0.5, (N_TIME, lats.size, lons.size))
    ds = xr.Dataset(
        {"T2M": (("time", "lat", "lon"), data.astype("float32"), {"units": "K"})},
        coords={"time": TIMES, "lat": lats, "lon": lons},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def stub_power_fetch(monkeypatch: pytest.MonkeyPatch):
    """Serve synthetic POWER NetCDF from the client's fetch seam."""

    def _fake(request, cache_dir, **kwargs):  # type: ignore[no-untyped-def]
        site = next(s for s in SITES if s["name"] == request.site)
        path = power_client.cache_path(cache_dir, request)
        path.parent.mkdir(parents=True, exist_ok=True)
        _power_point_response(site, seed=hash(request.site) % 1000).to_netcdf(path)
        return path

    monkeypatch.setattr(power_reader, "fetch_to_cache", _fake)


def _config(tmp_path: Path, model_path: Path) -> dict:
    return {
        "analysis": {
            "start_time": "2024-02-01",
            "end_time": "2024-02-02",
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "model": {
                "type": "generic",
                "files": str(model_path),
                "radius_of_influence": 100000,
                "variables": {"T2M": {"units": "K"}},
            },
            "power": {
                "type": "power",
                "temporal": "hourly",
                "cache_dir": str(tmp_path / "cache"),
                "sites": SITES,
                "variables": {"T2M": {}},
            },
        },
        "pairs": {
            "model_vs_power_t2m": {
                "x": {"source": "power", "variable": "T2M"},
                "y": {"source": "model", "variable": "T2M"},
            },
        },
        "plots": {
            "t2m_scatter": {
                "type": "scatter",
                "pairs": ["model_vs_power_t2m"],
                "title": "T2M",
            },
            "t2m_timeseries": {
                "type": "timeseries",
                "pairs": ["model_vs_power_t2m"],
                "title": "T2M",
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
    }


def test_power_runs_through_the_pipeline_and_pairs_against_a_model(
    stub_power_fetch, tmp_path: Path
) -> None:
    """The whole chain: fetch -> cache -> read -> pair -> stats -> plot."""
    model_path = _model_grid(tmp_path / "model.nc")
    result = PipelineRunner(show_progress=False).run_from_config(_config(tmp_path, model_path))

    assert result.success, [f"{s.stage_name}: {s.error}" for s in result.failed_stages]

    stats_files = list((tmp_path / "output").rglob("*.csv"))
    assert stats_files, "pipeline wrote no statistics CSV"
    rows = list(csv.DictReader(stats_files[0].open()))
    assert rows, "statistics CSV is empty"

    plots = list((tmp_path / "output").rglob("*.png"))
    assert len(plots) >= 2, f"expected scatter + timeseries, got {[p.name for p in plots]}"


def test_power_units_are_normalized_before_statistics(stub_power_fetch, tmp_path: Path) -> None:
    """POWER sends C; the model is K. Without the catalog offset the bias is ~273."""
    model_path = _model_grid(tmp_path / "model.nc")
    result = PipelineRunner(show_progress=False).run_from_config(_config(tmp_path, model_path))
    assert result.success

    stats_files = list((tmp_path / "output").rglob("*.csv"))
    rows = list(csv.DictReader(stats_files[0].open()))
    bias = float(rows[0]["MB"])
    # Both sides are ~278 K by construction. A missing C->K conversion would
    # put MB near 273, so this asserts the normalization actually ran rather
    # than just that a number was produced.
    assert abs(bias) < 5.0, f"MB={bias}: units look unconverted (expected |MB| << 273)"


def test_power_sites_render_as_a_scatter_mark_not_a_mesh(stub_power_fetch, tmp_path: Path) -> None:
    """POWER's POINT geometry must drive the mark, verified programmatically.

    The failure this guards against is real: a POWER point response is
    ``(time, lat, lon)`` -- the same shape as a regional one -- so a renderer
    dispatching on coordinates alone could reasonably mistake two sites for a
    2x2 grid and draw a mesh.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection, QuadMesh

    from davinci_monet.core.protocols import DataGeometry
    from davinci_monet.plots.base import build_series
    from davinci_monet.plots.renderers.spatial.field import SpatialPlotter

    reader = power_reader.POWERReader()
    ds = reader.open(
        [],
        variables=["T2M"],
        temporal="hourly",
        sites=SITES,
        cache_dir=tmp_path / "cache",
        time_range=("2024-02-01", "2024-02-02"),
    )
    assert reader.geometry is DataGeometry.POINT
    assert ds.attrs["geometry"] == "point"

    result = SpatialPlotter().render(build_series(ds, "T2M"))
    fig = result[0][1] if isinstance(result, list) else result
    ax = fig.axes[0]
    marks = [type(c).__name__ for c in ax.collections]
    assert any(
        isinstance(c, PathCollection) for c in ax.collections
    ), f"POWER point data must render as scatter; got {marks}"
    assert not any(
        isinstance(c, QuadMesh) for c in ax.collections
    ), f"POWER point data must not render as a mesh; got {marks}"
    plt.close(fig)
