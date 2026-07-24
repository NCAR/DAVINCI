"""Integration: MERRA-2 reader through the full pipeline.

Exercises PipelineRunner.run_from_config() with a ``type: merra2`` GRID source
paired against a synthetic GRID geometry source, mirroring MERRA-2 AOD vs a gridded
AOD geometry. This is the pipeline path a user takes with ``davinci run``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.runner import PipelineRunner


def _grid(varname: str, seed: int) -> xr.Dataset:
    times = np.array(["2026-04-01", "2026-04-02", "2026-04-03"], dtype="datetime64[ns]")
    lat = np.linspace(-87.5, 87.5, 6)
    lon = np.linspace(-175.0, 175.0, 8)
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.05, 0.8, size=(3, 6, 8)).astype(np.float32)
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


@pytest.mark.integration
def test_merra2_reader_pipeline(tmp_path: Path) -> None:
    m_dir = tmp_path / "merra2"
    o_dir = tmp_path / "geometry"
    m_dir.mkdir()
    o_dir.mkdir()
    _grid("TOTEXTTAU", seed=1).to_netcdf(m_dir / "MERRA2_400.tavgM_2d_aer_Nx.202604.nc4")
    _grid("aod_550nm", seed=2).to_netcdf(o_dir / "geometry.nc")

    out_dir = tmp_path / "output"
    config = {
        "analysis": {
            "start_time": "2026-04-01",
            "end_time": "2026-04-03",
            "output_dir": str(out_dir),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "merra2": {
                "type": "merra2",
                "files": str(m_dir / "*.nc4"),
                "variables": {"TOTEXTTAU": {"units": "1"}},
            },
            "aod_grid": {
                "type": "generic",
                "files": str(o_dir / "*.nc"),
                "variables": {"aod_550nm": {"units": "1"}},
            },
        },
        "pairs": {
            "merra2_aod_grid": {
                "x": {"source": "aod_grid", "variable": "aod_550nm"},
                "y": {"source": "merra2", "variable": "TOTEXTTAU"},
            }
        },
        "plots": {
            "bias": {
                "type": "spatial_bias",
                "pairs": ["merra2_aod_grid"],
                "title": "AOD Bias",
            },
            "sc": {
                "type": "scatter",
                "pairs": ["merra2_aod_grid"],
                "title": "AOD Scatter",
            },
        },
        "stats": {"output_table": True, "metrics": ["N", "MB", "RMSE", "R"]},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))

    result = PipelineRunner(show_progress=False).run_from_config(str(cfg))

    failed = [
        f"{s.stage_name}: {s.error}" for s in result.stage_results if s.status.name == "FAILED"
    ]
    assert result.success, f"Pipeline failed: {failed}"
    assert sorted(out_dir.rglob("*.png")), "expected plots"
    assert list(out_dir.rglob("*.csv")), "expected a stats CSV"


_IO_AER = Path("/Volumes/Io/MERRA2_tavgM/aer_Nx")


@pytest.mark.skipif(
    not _IO_AER.is_dir(),
    reason="MERRA-2 monthly aerosol data not staged on /Volumes/Io",
)
@pytest.mark.real_data
def test_real_merra2_file_opens() -> None:
    """Smoke: open one staged monthly aerosol file via the MERRA-2 reader."""
    from davinci_monet.datasets.merra2 import MERRA2Reader

    files = sorted(
        f for f in _IO_AER.glob("MERRA2_*.tavgM_2d_aer_Nx.*.nc4") if not f.name.startswith("._")
    )
    if not files:
        pytest.skip("no monthly aerosol .nc4 files present")

    ds = MERRA2Reader().open([files[0]], variables=["TOTEXTTAU"])
    assert "TOTEXTTAU" in ds.data_vars
    assert {"time", "lat", "lon"} <= set(ds["TOTEXTTAU"].dims)
    assert ds.attrs["geometry"] == "grid"
    da = ds["TOTEXTTAU"]
    assert 0.0 <= float(da.min()) and float(da.max()) < 5.0  # physical AOD range
