import io
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


def test_run_job_emits_started_and_result_and_sets_env(tmp_path, monkeypatch, capsys):
    from davinci_monet import config as _cfg_pkg
    from davinci_monet.config import parser as _parser
    from davinci_monet.daemon import worker
    from davinci_monet.pipeline import runner as _runner
    from davinci_monet.pipeline import stages as _stages

    captured = {}

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
            if context.progress_callback:
                context.progress_callback("Loading model: cam (1/1)")
            return _FakeResult(_FakeContext(context.config), success=True)

    monkeypatch.setattr(_runner, "PipelineRunner", _FakeRunner)
    monkeypatch.setattr(_stages, "PipelineContext", _FakeContext)

    spec = _spec(tmp_path)
    code = worker.run_job(spec.to_json())

    assert code == 0
    assert captured["env_var"] == "set"  # spec.env applied to os.environ
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    events = [ProgressEvent.parse_line(l) for l in lines]
    events = [e for e in events if e is not None]
    kinds = [e.kind for e in events]
    assert kinds[0] == "started"
    assert kinds[-1] == "result"
    assert "progress" in kinds
    result_evt = events[-1]
    assert result_evt.success is True
    assert result_evt.job_id == 42
    assert result_evt.output_dir == "/out"
    assert result_evt.plots == ["/out/a.png"]


def test_run_job_failure_emits_failed_result_and_nonzero(tmp_path, monkeypatch, capsys):
    from davinci_monet.config import parser as _parser
    from davinci_monet.daemon import worker

    def _boom(_path):
        raise RuntimeError("bad config")

    monkeypatch.setattr(worker, "_now", lambda: "2026-05-31T00:00:00")
    monkeypatch.setattr("davinci_monet.config.parser.load_config", _boom)

    code = worker.run_job(_spec(tmp_path).to_json())

    assert code == 1
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    events = [ProgressEvent.parse_line(l) for l in lines if ProgressEvent.parse_line(l)]
    result_evt = events[-1]
    assert result_evt.kind == "result"
    assert result_evt.success is False
    assert "bad config" in (result_evt.error or "")
