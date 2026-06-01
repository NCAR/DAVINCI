"""Unit tests for the daemon dispatcher (job-spec building; no real run)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import datetime

from davinci_monet.daemon.config import DaemonConfig, WatchRule
from davinci_monet.daemon.contracts import JobSpec, ProgressEvent, SettleMode, TriggerEvent
from davinci_monet.daemon.dispatcher import build_job_spec


def _trigger(name: str, files: list[str], mode: SettleMode = "quiescence") -> TriggerEvent:
    return TriggerEvent(
        watch_name=name,
        new_files=sorted(files),
        detected_at=datetime(2026, 5, 31, 12, 0, 0),
        settle_mode=mode,
    )


def test_build_job_spec_whole_config_env_overlay_and_locking() -> None:
    rule = WatchRule(
        name="cam_rt",
        watch="/scratch/cam/incoming/*.nc",
        run="/configs/asia-aq.yaml",
        on_fire="whole_config",
        env={"DATA": "/scratch/cam", "EXTRA": "rule"},
    )
    daemon_cfg = DaemonConfig(hdf5_file_locking=False, worker_timeout=None)
    trigger = _trigger("cam_rt", ["/scratch/cam/incoming/a.nc"])

    spec = build_job_spec(
        rule,
        trigger,
        daemon_cfg,
        job_id=7,
        base_env={"PATH": "/usr/bin", "DATA": "/old"},
    )

    assert isinstance(spec, JobSpec)
    assert spec.job_id == 7
    assert spec.watch_name == "cam_rt"
    assert spec.config_path == "/configs/asia-aq.yaml"
    assert spec.on_fire == "whole_config"
    # whole_config does NOT inject files
    assert spec.inject_into is None
    assert spec.new_files == ["/scratch/cam/incoming/a.nc"]
    # per-job env = base_env overlaid with rule.env (rule wins)
    assert spec.env["PATH"] == "/usr/bin"
    assert spec.env["DATA"] == "/scratch/cam"
    assert spec.env["EXTRA"] == "rule"
    # HDF5 locking policy carried through
    assert spec.hdf5_file_locking is False
    assert spec.worker_timeout is None


def test_build_job_spec_round_trips_through_json() -> None:
    rule = WatchRule(name="w", watch="/d/*.nc", run="/c.yaml")
    daemon_cfg = DaemonConfig()
    spec = build_job_spec(rule, _trigger("w", ["/d/x.nc"]), daemon_cfg, job_id=1, base_env={})
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec


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


def test_spawn_worker_reads_progress_lines_and_exit_code(tmp_path, monkeypatch) -> None:
    from davinci_monet.daemon import dispatcher

    # A fake worker script that echoes a started + result JSON line and exits 0.
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        textwrap.dedent(
            """
            import json, sys
            raw = sys.stdin.read()
            spec = json.loads(raw)
            jid = spec["job_id"]
            print(json.dumps({"kind": "started", "job_id": jid, "ts": "2026-05-31T00:00:00", "pid": 1}))
            print("this is not json and must be ignored")
            print(json.dumps({"kind": "result", "job_id": jid, "success": True,
                              "total_duration_seconds": 0.1, "plots": [], "summary": {},
                              "ts": "2026-05-31T00:00:01"}))
            sys.stdout.flush()
            sys.exit(0)
            """
        )
    )

    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        # Replace the "-m module" invocation with running our fake worker file.
        new_cmd = [sys.executable, str(fake_worker)]
        return real_popen(new_cmd, **kwargs)

    monkeypatch.setattr(dispatcher.subprocess, "Popen", fake_popen)

    spec = build_job_spec(
        WatchRule(name="w", watch="/d/*.nc", run="/c.yaml"),
        _trigger("w", ["/d/x.nc"]),
        DaemonConfig(hdf5_file_locking=False),
        job_id=99,
        base_env={"PATH": os.environ.get("PATH", "")},
    )

    seen: list[ProgressEvent] = []
    result = dispatcher.spawn_worker(spec, on_event=seen.append)

    assert result.exit_code == 0
    assert result.success is True
    kinds = [e.kind for e in result.events]
    assert kinds == ["started", "result"]  # non-JSON line dropped
    assert [e.kind for e in seen] == ["started", "result"]
    assert result.result_event is not None
    assert result.result_event.success is True


def test_spawn_worker_attached_uses_events_file_and_inherits_stdout(tmp_path, monkeypatch) -> None:
    """attached=True: env carries DAEMON_EVENTS_PATH, stdout is inherited, and
    events are read back from the temp file (not from a stdout PIPE)."""
    from davinci_monet.daemon import dispatcher

    captured: dict = {}

    class _FakeProc:
        """Stand-in for a worker that writes its events to DAEMON_EVENTS_PATH."""

        def __init__(self, env, stdout, stderr):
            self.returncode = None
            self._env = env
            captured["env"] = env
            captured["stdout"] = stdout
            captured["stderr"] = stderr

            class _Stdin:
                def write(self, _s):
                    pass

                def flush(self):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

        def wait(self, timeout=None):
            # Simulate the worker writing started + result lines to the events
            # file, then exiting 0.
            events_path = self._env["DAEMON_EVENTS_PATH"]
            with open(events_path, "w", encoding="utf-8") as handle:
                handle.write('{"kind":"started","job_id":5,"ts":"2026-05-31T00:00:00","pid":1}\n')
                handle.write("not json — ignored by the parser\n")
                handle.write(
                    '{"kind":"result","job_id":5,"success":true,'
                    '"total_duration_seconds":0.2,"plots":["/out/x.png"],'
                    '"output_dir":"/out","summary":{},"ts":"2026-05-31T00:00:01"}\n'
                )
            self.returncode = 0
            return 0

    def fake_popen(cmd, **kwargs):
        return _FakeProc(kwargs["env"], kwargs.get("stdout"), kwargs.get("stderr"))

    monkeypatch.setattr(dispatcher.subprocess, "Popen", fake_popen)

    spec = build_job_spec(
        WatchRule(name="w", watch="/d/*.nc", run="/c.yaml"),
        _trigger("w", ["/d/x.nc"]),
        DaemonConfig(hdf5_file_locking=False),
        job_id=5,
        base_env={"PATH": os.environ.get("PATH", "")},
    )

    seen: list[ProgressEvent] = []
    result = dispatcher.spawn_worker(spec, on_event=seen.append, attached=True)

    # Worker env carried the events-file path that both sides agree on.
    assert "DAEMON_EVENTS_PATH" in captured["env"]
    # Attached mode inherits the serve terminal: stdout/stderr are NOT piped.
    assert captured["stdout"] is None
    assert captured["stderr"] is None

    # The temp events file is parsed exactly like headless stdout would be.
    assert result.exit_code == 0
    assert result.success is True
    assert [e.kind for e in result.events] == ["started", "result"]  # non-JSON dropped
    assert [e.kind for e in seen] == ["started", "result"]
    assert result.result_event is not None
    assert result.result_event.output_dir == "/out"
    assert result.result_event.plots == ["/out/x.png"]

    # The temp file is cleaned up after the run.
    assert not os.path.exists(captured["env"]["DAEMON_EVENTS_PATH"])
