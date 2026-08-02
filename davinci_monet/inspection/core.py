"""Deterministic run-directory inspection."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from davinci_monet.inspection.presets import BUILTIN_INSPECTION_PRESETS
from davinci_monet.plots.contracts import AOD_CORRECTION_FIGURES, AOD_CORRECTION_PROTOCOL


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
    plot_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    plot_protocol_reports: Mapping[str, Any] | None = None,
) -> InspectionResult:
    """Inspect a run directory and write deterministic inspection artifacts."""
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"run directory is not a directory: {root}")

    plots_dir = root / "plots"
    pdfs = _collect_final_pdfs(root, plots_dir, plot_paths)

    unknown_presets = sorted(set(presets) - BUILTIN_INSPECTION_PRESETS)
    checks: list[dict[str, Any]] = [
        {
            "name": "final_pdf_products_exist",
            "passed": bool(pdfs),
            "detail": f"{len(pdfs)} PDF plot(s) found",
            "pdfs": [_format_relative(root, path) for path in pdfs],
        },
        {
            "name": "known_preset_selected",
            "passed": not unknown_presets,
            "detail": ",".join(unknown_presets) if unknown_presets else ",".join(presets),
            "presets": list(presets),
        },
    ]
    if "aod_correction" in presets:
        checks.append(_aod_correction_protocol_check(pdfs, plot_protocol_reports))

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
        "plot_protocol_reports": dict(plot_protocol_reports or {}),
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


def _aod_correction_protocol_check(
    pdfs: list[Path],
    plot_protocol_reports: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require the complete AOD suite and its renderer protocol evidence."""
    expected_figures = list(AOD_CORRECTION_FIGURES)
    pdf_names = [path.name for path in pdfs]
    missing_pdfs = [
        label
        for label in expected_figures
        if not any(name.endswith(f"_{label}.pdf") for name in pdf_names)
    ]
    reports = plot_protocol_reports if isinstance(plot_protocol_reports, Mapping) else {}
    matching_reports = [
        report
        for report in reports.values()
        if isinstance(report, Mapping)
        and report.get("protocol") == AOD_CORRECTION_PROTOCOL
        and report.get("passed") is True
        and report.get("figures") == expected_figures
    ]
    passed = not missing_pdfs and bool(matching_reports)
    detail_parts = [f"{len(expected_figures) - len(missing_pdfs)} of {len(expected_figures)} PDFs"]
    detail_parts.append(
        "protocol report passed" if matching_reports else "protocol report missing or invalid"
    )
    return {
        "name": "aod_correction_protocol",
        "passed": passed,
        "detail": "; ".join(detail_parts),
        "missing_figures": missing_pdfs,
        "protocol": AOD_CORRECTION_PROTOCOL,
    }


def _collect_final_pdfs(
    root: Path,
    plots_dir: Path,
    plot_paths: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    if plot_paths is not None:
        pdfs: list[Path] = []
        for raw_path in plot_paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            if path.suffix.lower() == ".pdf" and path.is_file():
                pdfs.append(path)
        return sorted(pdfs)

    return (
        sorted(path for path in plots_dir.rglob("*.pdf") if path.is_file())
        if plots_dir.exists()
        else []
    )


def _write_png_previews(root: Path, plots_dir: Path, pdfs: list[Path]) -> dict[str, Any]:
    preview_root = root / "inspection" / "previews"
    if preview_root.exists():
        shutil.rmtree(preview_root)
    preview_paths: list[Path] = []
    errors: list[str] = []

    for pdf in pdfs:
        relative_plot = _plot_relative_path(root, plots_dir, pdf)
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
            errors.append(f"{_format_relative(root, pdf)}: {message}")
            continue
        except subprocess.TimeoutExpired:
            errors.append(f"{_format_relative(root, pdf)}: preview conversion timed out")
            continue

        if preview_path.is_file():
            preview_paths.append(preview_path)
        else:
            errors.append(f"{_format_relative(root, pdf)}: preview file was not created")

    passed = len(preview_paths) == len(pdfs) and not errors
    return {
        "passed": passed,
        "paths": preview_paths,
        "errors": errors,
        "detail": f"{len(preview_paths)} of {len(pdfs)} preview PNG(s) written",
    }


def _plot_relative_path(root: Path, plots_dir: Path, pdf: Path) -> Path:
    try:
        return pdf.relative_to(plots_dir)
    except ValueError:
        try:
            return pdf.relative_to(root)
        except ValueError:
            return Path(pdf.name)


def _format_relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _render_markdown(payload: dict[str, Any]) -> str:
    status = "passed" if payload["passed"] else "failed"
    lines = ["# DAVINCI Inspection", "", f"Status: {status}", "", "## Checks", ""]
    for check in payload["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {marker}: {check['name']} - {check['detail']}")
    lines.append("")
    return "\n".join(lines)
