"""Deterministic run-directory inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from davinci_monet.inspection.presets import BUILTIN_INSPECTION_PRESETS


@dataclass(frozen=True)
class InspectionResult:
    """Result from inspecting a DAVINCI run directory."""

    passed: bool
    checks: list[dict[str, Any]]
    json_path: Path
    markdown_path: Path


def inspect_run_directory(
    run_dir: str | Path, *, presets: list[str] | tuple[str, ...]
) -> InspectionResult:
    """Inspect a run directory and write JSON and Markdown reports."""
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"run directory is not a directory: {root}")

    plots_dir = root / "plots"
    pdfs = (
        sorted(path for path in plots_dir.rglob("*.pdf") if path.is_file())
        if plots_dir.exists()
        else []
    )

    unknown_presets = sorted(set(presets) - BUILTIN_INSPECTION_PRESETS)
    checks: list[dict[str, Any]] = [
        {
            "name": "final_pdf_products_exist",
            "passed": bool(pdfs),
            "detail": f"{len(pdfs)} PDF plot(s) found",
            "pdfs": [str(path.relative_to(root)) for path in pdfs],
        },
        {
            "name": "known_preset_selected",
            "passed": not unknown_presets,
            "detail": ",".join(unknown_presets) if unknown_presets else ",".join(presets),
            "presets": list(presets),
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)

    inspection_dir = root / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    json_path = inspection_dir / "inspection.json"
    markdown_path = inspection_dir / "inspection.md"

    payload = {
        "passed": passed,
        "presets": list(presets),
        "checks": checks,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(payload))

    return InspectionResult(
        passed=passed,
        checks=checks,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    status = "passed" if payload["passed"] else "failed"
    lines = ["# DAVINCI Inspection", "", f"Status: {status}", "", "## Checks", ""]
    for check in payload["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {marker}: {check['name']} - {check['detail']}")
    lines.append("")
    return "\n".join(lines)
