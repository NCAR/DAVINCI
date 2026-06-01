"""Unit tests for the daemon worker (config injection + progress emission + exit code).

Injection tests exercise worker.inject_new_files() directly (pure logic, no I/O).
Progress/exit-code tests exercise worker.run_job() with a monkeypatched pipeline.
The chaining regression test exercises the real PipelineRunner.run() to confirm
that a pre-wired context.progress_callback survives the runner's internal
formatter installation at davinci_monet/pipeline/runner.py:1508-1602.
"""

from __future__ import annotations

from davinci_monet.daemon.contracts import JobSpec, ProgressEvent

# ---------------------------------------------------------------------------
# inject_new_files — pure logic, no I/O
# ---------------------------------------------------------------------------


def test_inject_new_files_overrides_target_source_files() -> None:
    from davinci_monet.daemon import worker

    config = {
        "analysis": {"output_dir": "/out"},
        "sources": {
            "modis": {"type": "modis", "files": "/data/modis/*.hdf"},
            "cam": {"type": "cesm_fv", "files": "/data/cam/*.nc"},
        },
    }
    new_files = ["/data/modis/new_b.hdf", "/data/modis/new_a.hdf"]

    out = worker.inject_new_files(config, inject_into="modis", new_files=new_files)

    # Target source files: replaced by the injected list (sorted), filename cleared
    assert out["sources"]["modis"]["files"] == [
        "/data/modis/new_a.hdf",
        "/data/modis/new_b.hdf",
    ]
    assert out["sources"]["modis"].get("filename") is None
    # Other sources untouched
    assert out["sources"]["cam"]["files"] == "/data/cam/*.nc"
    # Original config not mutated in place
    assert config["sources"]["modis"]["files"] == "/data/modis/*.hdf"


def test_inject_new_files_unknown_source_raises() -> None:
    from davinci_monet.daemon import worker

    config = {"sources": {"cam": {"type": "cesm_fv", "files": "/x/*.nc"}}}
    try:
        worker.inject_new_files(config, inject_into="missing", new_files=["/y/a.nc"])
    except KeyError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for unknown inject_into source")


def test_inject_new_files_noop_when_inject_into_none() -> None:
    from davinci_monet.daemon import worker

    config = {"sources": {"cam": {"files": "/x/*.nc"}}}
    out = worker.inject_new_files(config, inject_into=None, new_files=["/y/a.nc"])
    assert out["sources"]["cam"]["files"] == "/x/*.nc"


# ---------------------------------------------------------------------------
# Shared helpers for run_job tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# run_job — mocked pipeline
# ---------------------------------------------------------------------------


def test_run_job_emits_started_progress_result_and_sets_env(tmp_path, monkeypatch, capsys):
    """run_job must emit started, at least one progress, and a result event.

    This test verifies the wiring: spec.env is applied to os.environ, the
    progress callback installed on the context is forwarded through run_job's
    _emit() helper, and the terminal result event carries the expected metadata.
    Note: this test uses a _FakeRunner whose run() calls context.progress_callback
    directly.  For a test that exercises the real runner chaining path, see
    test_runner_chains_pre_existing_progress_callback below.
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
    assert "progress" in kinds, "worker must emit 'progress' events"

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


# ---------------------------------------------------------------------------
# Regression guard: real PipelineRunner must chain pre-existing callback
# ---------------------------------------------------------------------------


def test_runner_chains_pre_existing_progress_callback():
    """The real PipelineRunner.run() must forward messages to a callback set before run().

    This is the regression guard for the bug fixed at
    davinci_monet/pipeline/runner.py:1508-1602, where runner.run() now captures
    any pre-existing context.progress_callback BEFORE installing the internal
    formatter callback, then chains the two together.

    FAILURE MODE if the fix is reverted: runner.run() would overwrite
    context.progress_callback with the internal formatter callback, discarding
    the pre-wired outer callback.  The stage's log_progress() call would then
    never reach ``received``, and the final assert would fail.

    The test uses a real PipelineRunner with a single stub stage (no sci-stack
    imports needed) so it exercises the exact production code path.
    """
    from davinci_monet.pipeline.runner import PipelineRunner
    from davinci_monet.pipeline.stages import PipelineContext, StageResult, StageStatus

    received: list[str] = []

    # A minimal stage that emits one progress message then succeeds.
    class _EchoStage:
        name = "echo"

        def execute(self, context: PipelineContext) -> StageResult:
            context.log_progress("hello from echo stage")
            return StageResult(stage_name="echo", status=StageStatus.COMPLETED)

        def validate(self, context: PipelineContext) -> bool:
            return True

    runner = PipelineRunner(stages=[_EchoStage()], show_progress=False, show_plots=False)
    context = PipelineContext(config={"analysis": {}})
    # Pre-wire the callback BEFORE calling runner.run().  The runner must chain
    # through it even after installing its own internal formatter callback.
    context.progress_callback = received.append

    runner.run(context)

    assert received, (
        "context.progress_callback was never called: "
        "runner.run() likely overwrote the pre-wired callback instead of chaining it "
        "(runner.py:1508-1602 chaining fix may have been reverted)"
    )
    assert any(
        "hello from echo stage" in msg for msg in received
    ), f"expected 'hello from echo stage' in forwarded messages, got: {received}"
