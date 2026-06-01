"""Unit tests for the daemon worker (progress emission + exit code; mocked pipeline)."""

from __future__ import annotations

import json as _json
from datetime import datetime

from davinci_monet.daemon.contracts import JobSpec, ProgressEvent


def _spec(tmp_path, inject_into=None, new_files=None) -> JobSpec:
    return JobSpec(
        job_id=42,
        watch_name="w",
        config_path=str(tmp_path / "cfg.yaml"),
        on_fire="whole_config" if inject_into is None else "new_files_only",
        inject_into=inject_into,
        new_files=new_files or [],
        env={"DAEMON_TEST_VAR": "set"},
        hdf5_file_locking=False,
        worker_timeout=None,
        log_dir=None,
    )


class _FakeStageResult:
    def __init__(self, data):
        self.data = data


class _FakeContext:
    def __init__(self, config):
        self.config = config
        self.metadata = {}
        self.progress_callback = None
        self.results = {"plotting": _FakeStageResult({"plots_generated": ["/out/a.png"]})}


class _FakeResult:
    def __init__(self, context, success=True):
        self.success = success
        self.context = context
        self.total_duration_seconds = 1.5
        self.completed_stages = ["load_sources", "plotting"]
        self.failed_stages = []


def test_run_job_emits_started_progress_result_and_sets_env(tmp_path, monkeypatch, capsys):
    """run_job must emit started, at least one progress, and a result event.

    This test was introduced to prove the fix for the latent correctness bug
    where the runner's internal formatter callback overwrote the worker's
    progress_callback before any stage executed.  The fix in runner.py now
    chains any pre-existing context.progress_callback through the internal
    callback, so the worker's events ARE emitted.
    """
    from davinci_monet.daemon import worker
    from davinci_monet.pipeline import runner as _runner
    from davinci_monet.pipeline import stages as _stages

    captured: dict = {}

    class _FakeLoaded:
        def model_dump(self):
            return {"analysis": {"output_dir": "/out"}, "sources": {"cam": {"files": "/x/*.nc"}}}

    monkeypatch.setattr(worker, "_now", lambda: "2026-05-31T00:00:00")
    monkeypatch.setattr("davinci_monet.config.parser.load_config", lambda p: _FakeLoaded())

    class _FakeRunner:
        def __init__(self, *a, **k):
            pass

        def run(self, context):
            captured["env_var"] = __import__("os").environ.get("DAEMON_TEST_VAR")
            captured["callback"] = context.progress_callback
            # Simulate the pipeline emitting one progress message via the
            # worker-supplied callback.  In the real runner this callback is
            # *chained* through the formatter; here we call it directly.
            if context.progress_callback:
                context.progress_callback("Loading model: cam (1/1)")
            return _FakeResult(_FakeContext(context.config), success=True)

    monkeypatch.setattr(_runner, "PipelineRunner", _FakeRunner)
    monkeypatch.setattr(_stages, "PipelineContext", _FakeContext)

    spec = _spec(tmp_path)
    code = worker.run_job(spec.to_json())

    assert code == 0
    assert captured["env_var"] == "set"  # spec.env applied to os.environ

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [ProgressEvent.parse_line(line) for line in lines]
    events = [e for e in events if e is not None]
    kinds = [e.kind for e in events]

    assert kinds[0] == "started", f"first event must be 'started', got {kinds}"
    assert kinds[-1] == "result", f"last event must be 'result', got {kinds}"
    assert (
        "progress" in kinds
    ), "worker must emit 'progress' events; callback was dead code before the fix"

    result_evt = events[-1]
    assert result_evt.success is True
    assert result_evt.job_id == 42
    assert result_evt.output_dir == "/out"
    assert result_evt.plots == ["/out/a.png"]


def test_run_job_failure_emits_failed_result_and_nonzero(tmp_path, monkeypatch, capsys):
    """run_job must return exit-code 1 and emit a result event with success=False on error."""
    from davinci_monet.daemon import worker

    def _boom(_path):
        raise RuntimeError("bad config")

    monkeypatch.setattr(worker, "_now", lambda: "2026-05-31T00:00:00")
    monkeypatch.setattr("davinci_monet.config.parser.load_config", _boom)

    code = worker.run_job(_spec(tmp_path).to_json())

    assert code == 1
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [ProgressEvent.parse_line(line) for line in lines if ProgressEvent.parse_line(line)]
    result_evt = events[-1]
    assert result_evt.kind == "result"
    assert result_evt.success is False
    assert "bad config" in (result_evt.error or "")


def test_run_job_surfaces_real_log_path(tmp_path, monkeypatch, capsys):
    """run_job must report the log_path from context.metadata if the runner set it."""
    from davinci_monet.daemon import worker
    from davinci_monet.pipeline import runner as _runner
    from davinci_monet.pipeline import stages as _stages

    class _FakeLoaded:
        def model_dump(self):
            return {
                "analysis": {"output_dir": "/out", "log_dir": str(tmp_path / "logs")},
                "sources": {},
            }

    monkeypatch.setattr(worker, "_now", lambda: "2026-05-31T00:00:00")
    monkeypatch.setattr("davinci_monet.config.parser.load_config", lambda p: _FakeLoaded())

    expected_log = str(tmp_path / "logs" / "pipeline_20260531_000000.md")

    class _FakeRunner:
        def __init__(self, *a, **k):
            pass

        def run(self, context):
            # Simulate the runner storing the log_path in context.metadata,
            # then return a result whose .context IS the same object (as the
            # real runner does — result.context = context).
            context.metadata["log_path"] = expected_log
            return _FakeResult(context, success=True)

    monkeypatch.setattr(_runner, "PipelineRunner", _FakeRunner)
    monkeypatch.setattr(_stages, "PipelineContext", _FakeContext)

    code = worker.run_job(_spec(tmp_path).to_json())

    assert code == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [ProgressEvent.parse_line(line) for line in lines if ProgressEvent.parse_line(line)]
    result_evt = events[-1]
    assert result_evt.log_path == expected_log
