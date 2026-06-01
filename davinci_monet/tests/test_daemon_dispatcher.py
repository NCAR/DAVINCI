"""Tests for dispatcher.build_job_spec — new_files_only injection fields."""

from __future__ import annotations

from datetime import datetime

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import TriggerEvent
from davinci_monet.daemon.dispatcher import build_job_spec


def _trigger(name: str, files: list[str], mode: str = "quiescence") -> TriggerEvent:
    return TriggerEvent(
        watch_name=name,
        new_files=sorted(files),
        detected_at=datetime(2026, 5, 31, 12, 0, 0),
        settle_mode=mode,
    )


def test_build_job_spec_new_files_only_sets_injection_fields() -> None:
    rule = WatchRule(
        name="modis",
        watch="/data/modis/*.hdf",
        run="/configs/modis-aod.yaml",
        on_fire="new_files_only",
        inject_into="modis",
    )
    daemon_cfg = DaemonConfig(hdf5_file_locking=True, worker_timeout=900.0)
    trigger = _trigger("modis", ["/data/modis/b.hdf", "/data/modis/a.hdf"])

    spec = build_job_spec(rule, trigger, daemon_cfg, job_id=3, base_env={"DATA": "/data"})

    assert spec.on_fire == "new_files_only"
    assert spec.inject_into == "modis"
    # new_files are the injection override list, sorted from the trigger
    assert spec.new_files == ["/data/modis/a.hdf", "/data/modis/b.hdf"]
    assert spec.hdf5_file_locking is True
    assert spec.worker_timeout == 900.0


def test_build_job_spec_whole_config_ignores_inject_into() -> None:
    # inject_into is only meaningful for new_files_only; whole_config drops it
    rule = WatchRule(
        name="w",
        watch="/d/*.nc",
        run="/c.yaml",
        on_fire="whole_config",
        inject_into="should_be_ignored",
    )
    spec = build_job_spec(rule, _trigger("w", ["/d/x.nc"]), DaemonConfig(), job_id=1, base_env={})
    assert spec.inject_into is None
