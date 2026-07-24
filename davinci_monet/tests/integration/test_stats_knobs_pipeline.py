"""Integration: previously dead-pinned StatsConfig knobs are honored end-to-end.

Exercises the real ``PipelineRunner.run_from_config()`` path (the same one a user
hits with ``davinci run config.yaml``) to prove that ``stats.min_samples``
— a knob that StrictSchema used to reject and the stage used to read only as a
hardcoded default — now flows config -> validated MonetConfig -> typed
``stats_config()`` -> StatisticsCalculator and changes the result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.pipeline.runner import PipelineRunner


def _point_ds(lats: list[float], lons: list[float], vals: list[float], var: str) -> xr.Dataset:
    n = len(lats)
    return xr.Dataset(
        {var: (["site"], np.asarray(vals, float), {"units": "1"})},
        coords={
            "site": np.arange(n),
            "time": ("site", pd.to_datetime(["2024-02-01"] * n)),
            "latitude": ("site", np.asarray(lats, float)),
            "longitude": ("site", np.asarray(lons, float)),
        },
    )


def _config(tmp_path: Path, min_samples: int) -> dict:
    # Several points land in the same handful of 1-degree cells so both sources
    # overlap in multiple cells (enough valid pairs for the low-threshold run).
    lats = [10.2, 10.4, 10.6, 11.3, 11.6, 12.4]
    lons = [20.2, 20.4, 20.6, 21.3, 21.6, 22.4]
    x = _point_ds(lats, lons, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "aod")
    y = _point_ds(lats, lons, [1.2, 2.1, 3.3, 3.9, 5.2, 5.7], "AOD")
    xp, yp = tmp_path / "x.nc", tmp_path / "y.nc"
    x.to_netcdf(xp)
    y.to_netcdf(yp)
    return {
        "analysis": {"output_dir": str(tmp_path / f"out_{min_samples}")},
        "sources": {
            "obs": {"type": "generic", "files": str(xp), "variables": {"aod": {"units": "1"}}},
            "mod": {"type": "generic", "files": str(yp), "variables": {"AOD": {"units": "1"}}},
        },
        "pairs": {
            "obs_vs_mod": {
                "x": {"source": "obs", "variable": "aod"},
                "y": {"source": "mod", "variable": "AOD"},
                "method": "grid",
                "grid": {"horizontal_res": 1.0, "time_resolution": "1D", "min_sample_count": 1},
            }
        },
        "stats": {"metrics": ["MB", "RMSE"], "min_samples": min_samples},
    }


def _mb(result: object) -> float:
    stats = result.context.results["statistics"].data  # type: ignore[attr-defined]
    return float(stats["obs_vs_mod"]["aod"]["MB"])


@pytest.mark.integration
def test_stats_min_samples_is_honored_end_to_end(tmp_path: Path) -> None:
    # Low threshold: enough overlapping cells -> a finite metric is produced.
    low = PipelineRunner(show_progress=False).run_from_config(_config(tmp_path, min_samples=1))
    assert low.success, getattr(low, "error", None)
    assert not np.isnan(_mb(low))

    # High threshold: the same data has far fewer than 100000 valid pairs, so the
    # calculator masks every metric to NaN. Pre-fix this knob was unreachable
    # (StrictSchema rejected it) and the metric would have stayed finite.
    high = PipelineRunner(show_progress=False).run_from_config(
        _config(tmp_path, min_samples=100_000)
    )
    assert high.success, getattr(high, "error", None)
    assert np.isnan(_mb(high))
