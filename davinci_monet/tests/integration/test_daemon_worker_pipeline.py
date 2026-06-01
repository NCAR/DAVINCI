"""Integration: the daemon worker runs a synthetic config through the real pipeline.

Exercises PipelineRunner.run (the same engine run_analysis/run_from_config use) via
worker.run_job, with no monkeypatching of the pipeline. Synthetic NetCDF sources are
written to a temp dir; no external datasets are used.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from davinci_monet.daemon import worker
from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import ProgressEvent, TriggerEvent
from davinci_monet.daemon.dispatcher import build_job_spec


def _write_grid_source(path: Path) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    lat = np.array([40.0, 41.0])
    lon = np.array([-105.0, -104.0])
    values = np.arange(8, dtype=float).reshape(2, 2, 2)
    ds = xr.Dataset(
        {"O3": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"geometry": "grid"},
    )
    ds.to_netcdf(path)


def _write_point_source(path: Path) -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype="datetime64[m]")
    ds = xr.Dataset(
        {"o3": (("time", "site"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "time": times,
            "site": np.array([0, 1]),
            "latitude": ("site", np.array([40.0, 41.0])),
            "longitude": ("site", np.array([-105.0, -104.0])),
        },
        attrs={"geometry": "point"},
    )
    ds.to_netcdf(path)


def test_worker_runs_synthetic_config_through_pipeline(tmp_path, capsys):
    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    out_dir = tmp_path / "out"
    _write_grid_source(model_path)
    _write_point_source(obs_path)

    config = {
        "analysis": {"output_dir": str(out_dir)},
        "sources": {
            "cam": {
                "type": "generic",
                "role": "model",
                "files": str(model_path),
                "radius_of_influence": 200000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "airnow": {
                "type": "pt_sfc",
                "role": "obs",
                "filename": str(obs_path),
                "variables": {"o3": {"units": "ppb"}},
            },
        },
        "pairs": {
            "cam_airnow_o3": {
                "sources": ["cam", "airnow"],
                "reference": "airnow",
                "variables": {"cam": "O3", "airnow": "o3"},
            }
        },
        "stats": {"metrics": ["N", "MB"]},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))

    rule = WatchRule(
        name="cam_rt",
        watch=str(tmp_path / "*.nc"),
        run=str(config_path),
        on_fire="whole_config",
    )
    trigger = TriggerEvent(
        watch_name="cam_rt",
        new_files=[str(model_path)],
        detected_at=datetime(2026, 5, 31, 12, 0, 0),
        settle_mode="quiescence",
    )
    spec = build_job_spec(
        rule, trigger, DaemonConfig(hdf5_file_locking=False), job_id=1, base_env={}
    )

    code = worker.run_job(spec.to_json())

    assert code == 0, "worker should exit 0 on a successful pipeline run"

    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    events = [ProgressEvent.parse_line(l) for l in lines]
    events = [e for e in events if e is not None]

    kinds = [e.kind for e in events]
    assert kinds[0] == "started"
    assert kinds[-1] == "result"
    # The pipeline emitted at least one forwarded progress line
    assert any(e.kind == "progress" for e in events)

    result_evt = events[-1]
    assert result_evt.success is True
    assert result_evt.job_id == 1
    assert result_evt.output_dir == str(out_dir)
    assert out_dir.exists()
