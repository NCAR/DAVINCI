"""Tests for daemon dispatcher spawn_worker against a real subprocess."""

from __future__ import annotations

import subprocess
import sys
import textwrap
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


def test_spawn_worker_reads_progress_lines_and_exit_code(tmp_path, monkeypatch):
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
        base_env={"PATH": __import__("os").environ.get("PATH", "")},
    )

    seen = []
    result = dispatcher.spawn_worker(spec, on_event=seen.append)

    assert result.exit_code == 0
    assert result.success is True
    kinds = [e.kind for e in result.events]
    assert kinds == ["started", "result"]  # non-JSON line dropped
    assert [e.kind for e in seen] == ["started", "result"]
    assert result.result_event is not None
    assert result.result_event.success is True
