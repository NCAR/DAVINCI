"""Subprocess proofs for scheduler signals and ungraceful process death."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from davinci_monet.pipeline.checkpoints.models import ExecutionStatus
from davinci_monet.pipeline.checkpoints.store import AttemptStore

_SCRIPT = r"""
import os
import sys
import time
from pathlib import Path

from davinci_monet.pipeline.runner import PipelineRunner
from davinci_monet.pipeline.stages import BaseStage, StageStatus

root = Path(sys.argv[1])
mode = sys.argv[2]
marker = root.parent / f"{root.name}.marker"

class First(BaseStage):
    def __init__(self):
        super().__init__("first")

    def execute(self, context):
        with marker.open("a", encoding="utf-8") as stream:
            stream.write("executed\n")
        return self._create_result(StageStatus.COMPLETED, data={"first": True})

class Second(BaseStage):
    def __init__(self):
        super().__init__("second")

    def execute(self, context):
        if mode == "signal":
            time.sleep(60)
        if mode == "crash":
            os._exit(9)
        return self._create_result(StageStatus.COMPLETED, data={"second": True})

config = {
    "run": {"id": "subprocess-resume-smoke", "kind": "smoke"},
    "execution": {
        "attempt_root": str(root),
        "checkpoints": {
            "mode": "required",
            "granularity": "stage",
            "loaded_sources": True,
            "retain": "all",
        },
    },
    "analysis": {
        "output_dir": str(root / "output"),
        "log_dir": str(root / "logs"),
    },
    "sources": {"fixture": {"type": "generic"}},
}
PipelineRunner(
    stages=[First(), Second()],
    show_progress=False,
).run_from_config(config, resume=(mode == "resume"))
"""


def _write_script(tmp_path: Path) -> Path:
    script = tmp_path / "subprocess_pipeline.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    return script


def _wait_for_receipt(root: Path, process: subprocess.Popen[str]) -> None:
    receipt = root / "checkpoints" / "first" / "stage" / "r001.json"
    # A cold subprocess hashes the complete production package before it can
    # publish its first receipt. Shared-filesystem load can make that startup
    # exceed 20 seconds even though the pipeline is progressing normally.
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if receipt.is_file():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"subprocess exited before receipt: {process.returncode}\n{stdout}\n{stderr}"
            )
        time.sleep(0.05)
    process.kill()
    process.wait()
    raise AssertionError("timed out waiting for finalized first-stage receipt")


def _launch(script: Path, root: Path, mode: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(script), str(root), mode],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_sigterm_records_interruption_and_resume_reuses_receipt(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    root = tmp_path / "a001"
    process = _launch(script, root, "signal")
    _wait_for_receipt(root, process)
    process.terminate()
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode != 0, (stdout, stderr)

    resumed = _launch(script, root, "resume")
    stdout, stderr = resumed.communicate(timeout=20)
    assert resumed.returncode == 0, (stdout, stderr)
    assert (tmp_path / "a001.marker").read_text(encoding="utf-8") == "executed\n"
    assert [record.status for record in AttemptStore(root).list_executions()] == [
        ExecutionStatus.INTERRUPTED,
        ExecutionStatus.COMPLETED,
    ]


def test_crashed_execution_is_abandoned_before_resume(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    root = tmp_path / "a002"
    process = _launch(script, root, "crash")
    _wait_for_receipt(root, process)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 9, (stdout, stderr)

    resumed = _launch(script, root, "resume")
    stdout, stderr = resumed.communicate(timeout=20)
    assert resumed.returncode == 0, (stdout, stderr)
    assert (tmp_path / "a002.marker").read_text(encoding="utf-8") == "executed\n"
    assert [record.status for record in AttemptStore(root).list_executions()] == [
        ExecutionStatus.ABANDONED,
        ExecutionStatus.COMPLETED,
    ]
