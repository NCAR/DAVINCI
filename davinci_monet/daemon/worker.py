"""Daemon worker: the isolated child process that runs ONE pipeline.

Invoked as ``python -m davinci_monet.daemon.worker``. Reads a JobSpec JSON from
stdin, sets env, optionally injects new files into the resolved config, drives
PipelineRunner.run() directly (so a progress callback can be pre-wired to the
context before the runner's internal formatter is installed), streams
ProgressEvent-shaped JSON lines to stdout, and exits 0 iff
PipelineResult.success is True.  This is the ONLY daemon module that imports
the scientific pipeline.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Callable, Optional


def inject_new_files(
    config: dict[str, Any],
    *,
    inject_into: Optional[str],
    new_files: list[str],
) -> dict[str, Any]:
    """Return a copy of ``config`` with ``inject_into``'s files: overridden.

    For ``on_fire == "new_files_only"``: replace the named source's ``files:``
    with the sorted ``new_files`` list and clear any ``filename:`` so the glob is
    not also read. ``inject_into is None`` is a no-op (whole_config). An unknown
    source name raises KeyError. The input config is not mutated in place.
    """
    if inject_into is None:
        return config
    sources = config.get("sources") or {}
    if inject_into not in sources:
        raise KeyError(
            f"inject_into source '{inject_into}' not found in config sources " f"{sorted(sources)}"
        )
    out = copy.deepcopy(config)
    target = out["sources"][inject_into]
    target["files"] = sorted(new_files)
    target["filename"] = None
    return out


def _emit(event: dict[str, Any]) -> None:
    """Write one compact JSON progress line to the active event sink and flush.

    If ``DAEMON_EVENTS_PATH`` is set (attached mode), append the line to that
    file so stdout stays free for the pipeline's native animated progress
    display, which is inherited by the dispatcher's serve terminal. Otherwise
    (headless mode) write to stdout, which the dispatcher pipes and parses.
    """
    line = json.dumps(event, separators=(",", ":"))
    events_path = os.environ.get("DAEMON_EVENTS_PATH")
    if events_path:
        with open(events_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
        return
    sys.stdout.write(line)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _now() -> str:
    return datetime.now().isoformat()


def _make_progress_callback(job_id: int) -> Callable[[str], None]:
    """Build the pipeline progress_callback that forwards raw progress lines."""

    def callback(message: str) -> None:
        _emit(
            {
                "kind": "progress",
                "job_id": job_id,
                "message": message,
                "ts": _now(),
            }
        )

    return callback


def _collect_plots_and_output(result: Any) -> tuple[Optional[str], list[str]]:
    """Pull output_dir + generated plot paths out of a PipelineResult."""
    output_dir: Optional[str] = None
    plots: list[str] = []
    context = getattr(result, "context", None)
    if context is None:
        return output_dir, plots
    analysis = (context.config or {}).get("analysis", {})
    raw_output_dir = analysis.get("output_dir")
    output_dir = str(raw_output_dir) if raw_output_dir is not None else None
    for stage_name in ("plotting", "obs_plotting"):
        stage_result = context.results.get(stage_name)
        data = getattr(stage_result, "data", None)
        if isinstance(data, dict) and "plots_generated" in data:
            plots.extend(data["plots_generated"])
    return output_dir, plots


def run_job(spec_json: str) -> int:
    """Execute the job described by ``spec_json``; return the process exit code.

    Note: ``spec.worker_timeout`` is intentionally NOT enforced here.  Timeout
    enforcement (SIGKILL / SIGALRM) is the dispatcher's responsibility; the
    worker itself runs until the pipeline finishes or raises.
    """
    from davinci_monet.config import parser as _config_parser
    from davinci_monet.daemon.contracts import JobSpec

    # PipelineRunner is imported here (not run_analysis) so we can attach the
    # progress callback to the context before calling runner.run().
    # run_analysis would call run_from_config which creates a fresh context and
    # we would lose the callback.  PipelineRunner.run() preserves any
    # pre-existing context.progress_callback by chaining it through the
    # runner's internal formatter callback.
    from davinci_monet.pipeline.runner import (  # noqa: F401 — daemon isolation contract
        PipelineRunner,
    )
    from davinci_monet.pipeline.stages import PipelineContext

    spec = JobSpec.from_json(spec_json)
    job_id = spec.job_id

    for key, value in spec.env.items():
        os.environ[key] = value
    os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE" if spec.hdf5_file_locking else "FALSE"

    _emit(
        {
            "kind": "started",
            "job_id": job_id,
            "config_path": spec.config_path,
            "pid": os.getpid(),
            "ts": _now(),
        }
    )

    try:
        config = _config_parser.load_config(spec.config_path).model_dump()
        config = inject_new_files(config, inject_into=spec.inject_into, new_files=spec.new_files)
        if spec.log_dir is not None:
            config.setdefault("analysis", {})["log_dir"] = spec.log_dir

        # Attached mode (DAEMON_EVENTS_PATH set): stdout is inherited from the
        # serve terminal and progress JSON goes to the events file, so let the
        # pipeline animate its native progress display to that TTY. Headless
        # mode keeps stdout reserved for the JSON event stream the dispatcher
        # parses, so the pipeline must not write to it.
        attached = bool(os.environ.get("DAEMON_EVENTS_PATH"))
        runner = PipelineRunner(show_progress=attached, show_plots=False)
        context = PipelineContext(config=config)
        context.metadata["config_path"] = spec.config_path
        context.progress_callback = _make_progress_callback(job_id)
        result = runner.run(context)

        output_dir, plots = _collect_plots_and_output(result)
        # Retrieve the real log file path that the runner stored in
        # context.metadata["log_path"] (set only when analysis.log_dir is
        # configured).  This gives downstream consumers (notifications, etc.)
        # a direct link to the Markdown log without guessing the filename.
        log_path: Optional[str] = None
        if result.context is not None:
            log_path = result.context.metadata.get("log_path")
        _emit(
            {
                "kind": "result",
                "job_id": job_id,
                "success": bool(result.success),
                "total_duration_seconds": float(result.total_duration_seconds),
                "log_path": log_path,
                "output_dir": output_dir,
                "plots": plots,
                "summary": {
                    "completed_stages": list(result.completed_stages),
                    "failed_stages": [r.stage_name for r in result.failed_stages],
                },
                "error": None,
                "ts": _now(),
            }
        )
        return 0 if result.success else 1
    except Exception as exc:  # noqa: BLE001 - surfaced as a FAILED result line
        _emit(
            {
                "kind": "result",
                "job_id": job_id,
                "success": False,
                "total_duration_seconds": 0.0,
                "log_path": None,
                "output_dir": None,
                "plots": [],
                "summary": {},
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "ts": _now(),
            }
        )
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    """Entrypoint. Read the JobSpec JSON from a file arg or stdin; run it."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        with open(argv[0], "r", encoding="utf-8") as handle:
            spec_json = handle.read()
    else:
        spec_json = sys.stdin.read()
    return run_job(spec_json)


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
