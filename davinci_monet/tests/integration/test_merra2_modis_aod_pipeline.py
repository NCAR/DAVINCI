"""Integration test: MERRA2 vs MODIS AOD grid-to-grid pipeline.

Validates the full pipeline path via PipelineRunner.run_from_config() with
two synthetic GRID sources (both read by the ``generic`` reader) that mirror
the real MERRA2 vs MODIS Terra/Aqua AOD analysis.

This is also the renderer audit: it confirms that ``spatial_bias``,
``scatter``, and ``timeseries`` renderers handle GRID-geometry PAIRED data
(previously exercised mostly on point data).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.pipeline.runner import PipelineRunner


def _make_grid(varname: str, nt: int = 3, ny: int = 6, nx: int = 8, seed: int = 0) -> xr.Dataset:
    """Create a minimal synthetic GRID dataset.

    Parameters
    ----------
    varname
        Name of the single data variable.
    nt, ny, nx
        Temporal, latitude, and longitude sizes.
    seed
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Synthetic dataset with ``(time, lat, lon)`` dimensions.
    """
    rng = np.random.default_rng(seed)
    times = np.array(["2003-01-01", "2003-02-01", "2003-03-01"], dtype="datetime64[ns]")[:nt]
    lat = np.linspace(-87.5, 87.5, ny)
    lon = np.linspace(-175.0, 175.0, nx)
    data = rng.uniform(0.05, 0.8, size=(nt, ny, nx)).astype(np.float32)
    return xr.Dataset(
        {varname: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


@pytest.mark.integration
def test_merra2_modis_aod_pipeline(tmp_path: Path) -> None:
    """Grid-to-grid AOD pipeline runs end-to-end and produces plots + CSV.

    Uses two synthetic GRID sources (MERRA2-like TOTEXTTAU and MODIS-like
    aod_550nm) loaded by the ``generic`` reader.  Asserts that:
    - the pipeline succeeds (all stages pass),
    - at least 3 PNG plots were written (spatial_bias, scatter, timeseries),
    - at least one statistics CSV was written.
    """
    # ------------------------------------------------------------------ data --
    merra2_ds = _make_grid("TOTEXTTAU", seed=1)
    modis_ds = _make_grid("aod_550nm", seed=2)

    merra2_dir = tmp_path / "merra2"
    modis_dir = tmp_path / "modis"
    merra2_dir.mkdir()
    modis_dir.mkdir()
    merra2_ds.to_netcdf(merra2_dir / "merra2.nc")
    modis_ds.to_netcdf(modis_dir / "modis.nc")

    out_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"

    # ---------------------------------------------------------------- config --
    config: dict = {
        "analysis": {
            "start_time": "2003-01-01",
            "end_time": "2003-03-31",
            "output_dir": str(out_dir),
            "log_dir": str(log_dir),
        },
        "sources": {
            "merra2": {
                "type": "generic",
                "files": str(merra2_dir / "*.nc"),
                "variables": {"TOTEXTTAU": {"units": "1"}},
            },
            "modis_terra": {
                "type": "generic",
                "files": str(modis_dir / "*.nc"),
                "variables": {"aod_550nm": {"units": "1"}},
            },
        },
        "pairs": {
            "merra2_vs_terra": {
                "x": {"source": "modis_terra", "variable": "aod_550nm"},
                "y": {"source": "merra2", "variable": "TOTEXTTAU"},
            }
        },
        "plots": {
            "bias": {
                "type": "spatial_bias",
                "pairs": ["merra2_vs_terra"],
                "title": "AOD Spatial Bias",
            },
            "sc": {
                "type": "scatter",
                "pairs": ["merra2_vs_terra"],
                "title": "AOD Scatter",
            },
            "ts": {
                "type": "timeseries",
                "pairs": ["merra2_vs_terra"],
                "title": "AOD Time Series",
                # No aggregate_dim: the timeseries renderer auto-averages all
                # non-time dims (lat, lon) to produce a global-mean series.
            },
        },
        "stats": {
            "output_table": True,
            "metrics": ["N", "MB", "RMSE", "R"],
        },
    }

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    # --------------------------------------------------------------- execute --
    runner = PipelineRunner(show_progress=False)
    result = runner.run_from_config(str(cfg_path))

    # ---------------------------------------------------------------- assert --
    failed = [
        f"{s.stage_name}: {s.error}" for s in result.stage_results if s.status.name == "FAILED"
    ]
    assert result.success, f"Pipeline failed. Errors: {failed}"

    plots = sorted(out_dir.rglob("*.png"))
    assert len(plots) >= 3, (
        f"Expected >=3 PNG plots (spatial_bias, scatter, timeseries), "
        f"got {[p.name for p in plots]}"
    )
    for png in plots:
        assert (
            png.stat().st_size > 1024
        ), f"Plot {png.name} is suspiciously small ({png.stat().st_size} bytes)"

    csv_files = list(out_dir.rglob("*.csv"))
    assert csv_files, "Expected at least one statistics CSV in the output directory"


@pytest.mark.skipif(
    not (os.environ.get("MERRA2_DATA") and os.environ.get("MODIS_DATA")),
    reason="set MERRA2_DATA and MODIS_DATA to run the real-data smoke test",
)
@pytest.mark.real_data
def test_real_data_one_month(tmp_path: Path) -> None:
    """Read one real MOD08_M3 HDF4 file through the modis_viirs reader.

    Confirms that HDF4 decode, scale/fill application, and lat/lon coordinate
    attachment all work correctly on genuine MODIS Terra monthly L3 data.
    Set env vars to activate::

        export MERRA2_DATA=/Volumes/Io/MERRA2_tavgM
        export MODIS_DATA=/Volumes/Io
    """
    from davinci_monet.datasets.satellite.modis_viirs import MODISVIIRSReader

    modis_dir = Path(os.environ["MODIS_DATA"]) / "MOD08_M3"
    # Exclude macOS resource-fork files (._<name>.hdf) that appear on non-HFS volumes.
    files = sorted(f for f in modis_dir.glob("*.hdf") if not f.name.startswith("."))[:1]
    assert files, f"no MOD08_M3 files under {modis_dir}"
    ds = MODISVIIRSReader().open([str(files[0])], variables=["aod_550nm"], product="MOD08_M3")
    assert "aod_550nm" in ds and "time" in ds.coords
    assert {"lat", "lon"}.issubset(ds.coords)
    assert float(ds["aod_550nm"].max()) < 10.0  # physical AOD, scale applied
