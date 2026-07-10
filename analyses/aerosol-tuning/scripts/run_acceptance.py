"""Run the synthetic-only FABLE acceptance gate with user-supplied seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from davinci_monet.tests.synthetic.aerosol_tuning import (
    evaluate_synthetic_recovery_gate,
    run_synthetic_acceptance,
)


def _file_identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _manifest_identity(context: Any) -> dict[str, str] | None:
    result = context.results.get("manifest")
    data = result.data if result is not None else None
    path = Path(str(data["manifest"])) if isinstance(data, Mapping) and "manifest" in data else None
    return _file_identity(path) if path is not None and path.is_file() else None


def _recovery_artifact(context: Any) -> dict[str, Any] | None:
    entries = context.metadata.get("analysis_artifacts", [])
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("analysis") == "recovery"
        and entry.get("role") == "recovery_report"
    ]
    if len(matches) != 1:
        return None
    return json.loads(json.dumps(dict(matches[0]), default=str))


def _json_report(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_report(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run_pipeline(config: Path) -> dict[str, Any]:
    from davinci_monet.pipeline.lifecycle import PipelineResourcePolicy
    from davinci_monet.pipeline.runner import PipelineRunner

    result = PipelineRunner(
        show_progress=True,
        close_datasets_after_run=False,
    ).run_from_config(str(config))
    try:
        payload: dict[str, Any] = {
            "status": "completed" if result.success else "failed",
            "success": result.success,
            "pipeline_success": result.success,
            "duration_seconds": result.total_duration_seconds,
            "completed_stages": result.completed_stages,
            "errors": json.loads(json.dumps(result.stage_errors, default=str)),
        }
        if result.context is not None:
            manifest = _manifest_identity(result.context)
            if manifest is not None:
                payload["manifest"] = manifest
        if result.success and result.context is not None and "recovery" in result.context.sources:
            report = result.context.sources["recovery"].data.compute()
            gate = evaluate_synthetic_recovery_gate(report)
            payload["recovery_gate"] = gate
            payload["recovery_report"] = _json_report(report.to_dict(data="list"))
            artifact = _recovery_artifact(result.context)
            if artifact is not None:
                payload["recovery_artifact"] = artifact
            payload["success"] = bool(gate["passed"])
            payload["status"] = "completed" if gate["passed"] else "failed_recovery_gate"
        return payload
    finally:
        if result.context is not None:
            PipelineResourcePolicy(close_datasets_after_run=False).cleanup_context_datasets(
                result.context
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="New destination for the locked acceptance run")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        required=True,
        help="User-supplied acceptance seed; provide exactly three distinct values",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate and hash all three OSSE bundles without running their pipelines",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and lock the supplied seeds without allocating OSSE arrays",
    )
    args = parser.parse_args(argv)
    try:
        record_path = run_synthetic_acceptance(
            args.root,
            args.seed,
            generate_only=bool(args.generate_only),
            dry_run=bool(args.dry_run),
            pipeline_executor=_run_pipeline,
        )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    print(record_path)
    return 0 if record["status"] in {"planned", "generated", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
