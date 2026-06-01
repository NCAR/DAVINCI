"""Daemon-mode INTEGRATION tests (CLAUDE.md rule #1 compliant).

The analysis run flows through the REAL pipeline entry point
(``run_analysis`` -> ``PipelineRunner.run_from_config``) inside a REAL worker
subprocess spawned by the daemon dispatcher. Only the desktop/iCloud notify
side-effects are mocked; the pipeline, worker subprocess, and SQLite state
store are exercised for real.

See the module-level "Test path design" note in the plan for the entry points
called and the data flow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.daemon.config import load_watches
from davinci_monet.daemon.contracts import JobStatus
from davinci_monet.daemon.state import StateStore
from davinci_monet.daemon.supervisor import Supervisor, build_supervisor
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


class FakeClock:
    """Injectable monotonic clock (conforms to daemon contracts.Clock).

    ``now`` only advances when the test calls ``advance``; ``sleep`` advances
    the virtual clock instead of blocking, so settle/quiescence windows elapse
    deterministically with no real wall-clock waits.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += max(0.0, float(seconds))

    def advance(self, seconds: float) -> None:
        self._t += float(seconds)


def _write_synthetic_point_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Write a synthetic generic model + co-located pt_sfc obs to NetCDF.

    Mirrors the known-green TestPointPipeline recipe in test_integration.py:
    O3 model field with a small spatial gradient + bias, and obs sampled from
    the model so pairing/stats are well-defined. Returns (model_nc, obs_nc).
    """
    domain = Domain(lon_min=-105.0, lon_max=-95.0, lat_min=35.0, lat_max=45.0, n_lon=12, n_lat=12)
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-15 06:00", freq="1h")

    model_ds = create_model_dataset(variables=["O3"], domain=domain, time_config=time_cfg, seed=42)
    lat_vals = model_ds.lat.values
    lat_norm = (lat_vals - lat_vals.min()) / (lat_vals.max() - lat_vals.min())
    model_ds["O3"] = model_ds["O3"] + 20.0 * lat_norm[:, np.newaxis]

    scenario = PerfectMatchScenario(
        variables=["O3"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_obs=8,
        noise_level=0.0,
        seed=42,
    )
    obs_ds = scenario._generate_point_obs(model_ds)

    rng = np.random.default_rng(42)
    model_ds["O3"] = model_ds["O3"] + 5.0 + rng.normal(0, 3.0, size=model_ds["O3"].shape)

    model_nc = tmp_path / "model.nc"
    obs_nc = tmp_path / "obs.nc"
    model_ds.to_netcdf(model_nc)
    obs_ds.to_netcdf(obs_nc)
    return model_nc, obs_nc


def _minimal_point_config_dict(
    model_nc: Path, obs_nc: Path, output_dir: Path, log_dir: Path
) -> dict[str, Any]:
    """Minimal DAVINCI config that the real pipeline can run end-to-end."""
    return {
        "analysis": {
            "start_time": "2024-01-15 00:00:00",
            "end_time": "2024-01-15 06:00:00",
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
        },
        "model": {
            "synthetic": {
                "mod_type": "generic",
                "files": str(model_nc),
                "radius_of_influence": 50000,
                "mapping": {"surface": {"O3": "O3"}},
                "variables": {"O3": {"units": "ppb"}},
            }
        },
        "obs": {
            "surface": {
                "obs_type": "pt_sfc",
                "filename": str(obs_nc),
                "variables": {"O3": {"obs_min": 0, "obs_max": 200, "units": "ppb"}},
            }
        },
        "pairs": {
            "synthetic_surface": {
                "model": "synthetic",
                "obs": "surface",
                "variable": {"model_var": "O3", "obs_var": "O3"},
            }
        },
        "plots": {
            "scatter_o3": {
                "type": "scatter",
                "pairs": ["synthetic_surface"],
                "title": "O3: Model vs Observations",
            }
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
    }


def _drain_supervisor(
    supervisor: Supervisor, store: StateStore, *, timeout_s: float = 240.0
) -> None:
    """Serve the supervisor on a daemon thread until no QUEUED/RUNNING jobs remain.

    Runs the supervisor's ``serve`` loop on a background thread so that its
    ``clock.sleep(poll_interval)`` calls advance the injected FakeClock.  The
    settle/quiescence window only elapses when the clock advances, so the step
    path (``run_once`` / ``tick`` / ``process_pending``) cannot be used with a
    FakeClock — the clock never advances between bare step calls.  Using
    ``serve`` is therefore the correct approach: after each watcher poll the
    supervisor calls ``self.clock.sleep(poll_interval)`` which advances the
    FakeClock by ``poll_interval`` seconds, letting the settle window expire
    deterministically.  We then poll the real StateStore until the job leaves
    the active set (status transitions to COMPLETED or FAILED).
    """
    import threading

    serve = getattr(supervisor, "serve", None)
    assert serve is not None, "Supervisor exposes no serve() method"
    deadline = time.monotonic() + timeout_s
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        while time.monotonic() < deadline:
            jobs = store.list_jobs(limit=10)
            if jobs and not store.active_jobs():
                return
            time.sleep(0.1)
        raise AssertionError("supervisor did not drain within timeout")
    finally:
        # Signal the supervisor to stop; try common method names, then
        # fall back to request_shutdown which sets _draining=True.
        stop = (
            getattr(supervisor, "stop", None)
            or getattr(supervisor, "shutdown", None)
            or getattr(supervisor, "request_shutdown", None)
        )
        if stop is not None:
            stop()
        thread.join(timeout=10.0)


def _build_watches_yaml(
    tmp_path: Path,
    run_config_path: Path,
    incoming_glob: str,
    *,
    on_fire: str = "whole_config",
    inject_into: str | None = None,
    settle: str = "5s",
) -> Path:
    """Write a temp watches.yaml with one rule and a temp state_dir."""
    state_dir = tmp_path / "daemon_state"
    rule: dict[str, Any] = {
        "watch": incoming_glob,
        "run": str(run_config_path),
        "on_fire": on_fire,
        "settle": settle,
        "notify": ["desktop"],
    }
    if inject_into is not None:
        rule["inject_into"] = inject_into
    doc = {
        "daemon": {
            "state_dir": str(state_dir),
            "poll_interval": "1s",
            "max_concurrent": 1,
            "hdf5_file_locking": False,
            "notifications": {
                "desktop": True,
                "icloud_copy": True,
                "icloud_dir": str(tmp_path / "icloud"),
            },
        },
        "watches": {"realtime": rule},
    }
    watches_path = tmp_path / "watches.yaml"
    watches_path.write_text(yaml.safe_dump(doc))
    return watches_path


@pytest.fixture
def captured_notifications(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Mock ONLY the desktop/iCloud notify side-effects (never the pipeline)."""
    import davinci_monet.daemon.notify as notify_mod

    calls: list[dict[str, Any]] = []

    def _fake_desktop(*args: Any, **kwargs: Any) -> None:
        calls.append({"channel": "desktop", "args": args, "kwargs": kwargs})

    def _fake_icloud(*args: Any, **kwargs: Any) -> Any:
        calls.append({"channel": "icloud", "args": args, "kwargs": kwargs})
        return None

    # Patch whichever public hooks the notify module exposes; tolerate either
    # function-style or class-method-style implementations.
    for name in ("send_desktop_notification", "send_desktop", "desktop_notify", "notify_desktop"):
        if hasattr(notify_mod, name):
            monkeypatch.setattr(notify_mod, name, _fake_desktop, raising=False)
    for name in ("copy_to_icloud", "icloud_copy", "copy_outputs"):
        if hasattr(notify_mod, name):
            monkeypatch.setattr(notify_mod, name, _fake_icloud, raising=False)
    # Belt-and-suspenders: neutralize the raw side-effect primitives.
    monkeypatch.setattr(notify_mod, "subprocess", None, raising=False)
    return calls


def test_daemon_whole_config_integration(
    tmp_path: Path, captured_notifications: list[dict[str, Any]]
) -> None:
    # --- synthetic data + run config ---
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    model_nc, obs_nc = _write_synthetic_point_pair(data_dir)
    output_dir = tmp_path / "run_output"
    log_dir = tmp_path / "run_logs"
    run_cfg = _minimal_point_config_dict(model_nc, obs_nc, output_dir, log_dir)
    run_cfg_path = tmp_path / "run.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_cfg))

    # --- watched dir + watches.yaml ---
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    watches_path = _build_watches_yaml(
        tmp_path, run_cfg_path, str(incoming / "*.nc"), on_fire="whole_config"
    )

    watches_file = load_watches(watches_path)
    store = StateStore(watches_file.daemon.db_path)
    store.init_schema()

    clock = FakeClock()
    supervisor = build_supervisor(watches_file, state=store, clock=clock)

    # --- drop the trigger file, then let the settle window elapse ---
    trigger_nc = incoming / "new_data.nc"
    xr.open_dataset(model_nc).to_netcdf(trigger_nc)
    clock.advance(10.0)  # past the 5s settle window

    _drain_supervisor(supervisor, store)

    # --- assert: a job was recorded and COMPLETED ---
    jobs = store.list_jobs(watch_name="realtime", limit=10)
    assert jobs, "no job recorded for the fired watch"
    job = jobs[0]
    assert job.status == JobStatus.COMPLETED, f"job not completed: {job.status} / {job.error}"
    assert job.exit_code == 0
    assert job.result_summary is not None
    assert job.on_fire == "whole_config"

    # --- assert: the REAL pipeline produced outputs ---
    assert list(output_dir.rglob("statistics_summary.csv")), "no stats CSV from pipeline"
    pngs = list(output_dir.rglob("*.png"))
    assert pngs, "no plots from pipeline"
    assert all(p.stat().st_size > 1024 for p in pngs)
    assert list(log_dir.glob("pipeline_*.md")), "no per-run markdown log"

    # --- assert: notify hook fired (side-effects mocked) ---
    assert captured_notifications, "notify hook was not invoked on completion"

    store.close()


def _glob_model_config_dict(
    obs_nc: Path, model_glob: str, output_dir: Path, log_dir: Path
) -> dict[str, Any]:
    """Unified sources: config whose model source files: is a glob that is
    EMPTY until the daemon injects the newly-arrived file (on_fire=new_files_only).

    Uses the sources: schema (not the legacy model:/obs: schema) so that
    inject_new_files, which operates on config['sources'][inject_into], can
    override the target without requiring legacy-schema handling in the worker.
    """
    return {
        "analysis": {
            "start_time": "2024-01-15 00:00:00",
            "end_time": "2024-01-15 06:00:00",
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
        },
        "sources": {
            "synthetic": {
                "type": "generic",
                "role": "model",
                "files": model_glob,
                "radius_of_influence": 50000,
                "mapping": {"surface": {"O3": "O3"}},
                "variables": {"O3": {"units": "ppb"}},
            },
            "surface": {
                "type": "pt_sfc",
                "role": "obs",
                "filename": str(obs_nc),
                "variables": {"O3": {"obs_min": 0, "obs_max": 200, "units": "ppb"}},
            },
        },
        "pairs": {
            "synthetic_surface": {
                "sources": ["synthetic", "surface"],
                "variables": {"synthetic": "O3", "surface": "O3"},
            }
        },
        "plots": {
            "scatter_o3": {
                "type": "scatter",
                "pairs": ["synthetic_surface"],
                "title": "O3: Model vs Observations",
            }
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
    }


def test_daemon_new_files_only_injection_integration(
    tmp_path: Path, captured_notifications: list[dict[str, Any]]
) -> None:
    # --- synthetic data: obs lives outside the watched dir; model arrives later ---
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    model_nc, obs_nc = _write_synthetic_point_pair(data_dir)

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    output_dir = tmp_path / "run_output"
    log_dir = tmp_path / "run_logs"

    # Model source files: is a glob into the (initially empty) watched dir.
    model_glob = str(incoming / "*.nc")
    run_cfg = _glob_model_config_dict(obs_nc, model_glob, output_dir, log_dir)
    run_cfg_path = tmp_path / "run.yaml"
    run_cfg_path.write_text(yaml.safe_dump(run_cfg))

    watches_path = _build_watches_yaml(
        tmp_path,
        run_cfg_path,
        str(incoming / "*.nc"),
        on_fire="new_files_only",
        inject_into="synthetic",
    )

    watches_file = load_watches(watches_path)
    store = StateStore(watches_file.daemon.db_path)
    store.init_schema()

    clock = FakeClock()
    supervisor = build_supervisor(watches_file, state=store, clock=clock)

    # Drop the model file into the watched dir -> this is the injected new file.
    injected = incoming / "cam_model.nc"
    xr.open_dataset(model_nc).to_netcdf(injected)
    clock.advance(10.0)

    _drain_supervisor(supervisor, store)

    jobs = store.list_jobs(watch_name="realtime", limit=10)
    assert jobs, "no job recorded for the injection watch"
    job = jobs[0]
    assert job.on_fire == "new_files_only"
    resolved = {str(Path(f).resolve()) for f in job.files}
    assert str(injected.resolve()) in resolved, "injected file not on job"
    assert job.status == JobStatus.COMPLETED, f"injection run failed: {job.error}"
    assert job.exit_code == 0

    # The real pipeline ran with the injected file -> outputs exist.
    assert list(output_dir.rglob("statistics_summary.csv")), "no stats CSV from injected run"
    assert list(output_dir.rglob("*.png")), "no plots from injected run"
    assert list(log_dir.glob("pipeline_*.md")), "no per-run markdown log"

    assert captured_notifications, "notify hook not invoked for injection run"

    store.close()
