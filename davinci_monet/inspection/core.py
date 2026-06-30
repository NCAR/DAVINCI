"""Deterministic run-directory inspection."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from davinci_monet.inspection.presets import BUILTIN_INSPECTION_PRESETS


@dataclass(frozen=True)
class InspectionResult:
    """Result from inspecting a DAVINCI run directory."""

    passed: bool
    checks: list[dict[str, Any]]
    json_path: Path
    markdown_path: Path
    preview_paths: list[Path]


def inspect_run_directory(
    run_dir: str | Path,
    *,
    presets: list[str] | tuple[str, ...],
    preview_format: Literal["png"] | None = None,
) -> InspectionResult:
    """Inspect a run directory and write deterministic inspection artifacts."""
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

    preview_paths: list[Path] = []
    if preview_format == "png":
        preview_result = _write_png_previews(root, plots_dir, pdfs)
        preview_paths = preview_result["paths"]
        checks.append(
            {
                "name": "inspection_previews_exist",
                "passed": preview_result["passed"],
                "detail": preview_result["detail"],
                "previews": [str(path.relative_to(root)) for path in preview_paths],
                "errors": preview_result["errors"],
            }
        )
    passed = all(bool(check["passed"]) for check in checks)

    inspection_dir = root / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    json_path = inspection_dir / "inspection.json"
    markdown_path = inspection_dir / "inspection.md"

    payload = {
        "passed": passed,
        "presets": list(presets),
        "checks": checks,
        "previews": [str(path.relative_to(root)) for path in preview_paths],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(payload))

    return InspectionResult(
        passed=passed,
        checks=checks,
        json_path=json_path,
        markdown_path=markdown_path,
        preview_paths=preview_paths,
    )


def _write_png_previews(
    root: Path, plots_dir: Path, pdfs: list[Path]
) -> dict[str, Any]:
    preview_root = root / "inspection" / "previews"
    if preview_root.exists():
        shutil.rmtree(preview_root)
    preview_paths: list[Path] = []
    errors: list[str] = []

    for pdf in pdfs:
        relative_plot = pdf.relative_to(plots_dir)
        preview_path = preview_root / relative_plot.with_suffix(".png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        output_prefix = preview_path.with_suffix("")
        command = [
            "pdftoppm",
            "-f",
            "1",
            "-singlefile",
            "-png",
            "-r",
            "144",
            str(pdf),
            str(output_prefix),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            errors.append("pdftoppm is not available on PATH")
            break
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            errors.append(f"{pdf.relative_to(root)}: {message}")
            continue
        except subprocess.TimeoutExpired:
            errors.append(f"{pdf.relative_to(root)}: preview conversion timed out")
            continue

        if preview_path.is_file():
            preview_paths.append(preview_path)
        else:
            errors.append(f"{pdf.relative_to(root)}: preview file was not created")

    passed = len(preview_paths) == len(pdfs) and not errors
    return {
        "passed": passed,
        "paths": preview_paths,
        "errors": errors,
        "detail": f"{len(preview_paths)} of {len(pdfs)} preview PNG(s) written",
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    status = "passed" if payload["passed"] else "failed"
    lines = ["# DAVINCI Inspection", "", f"Status: {status}", "", "## Checks", ""]
    for check in payload["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {marker}: {check['name']} - {check['detail']}")
    lines.append("")
    return "\n".join(lines)
