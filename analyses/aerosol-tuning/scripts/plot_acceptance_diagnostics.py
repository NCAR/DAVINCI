"""Generate read-only FABLE acceptance diagnostics through DAVINCI pipelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import xarray as xr
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from davinci_monet.pipeline.runner import run_analysis  # noqa: E402
from davinci_monet.tests.synthetic.fable_acceptance_diagnostics import (  # noqa: E402
    AcceptanceDiagnosticSource,
    acceptance_collection_config,
    build_acceptance_diagnostic_source,
    verify_wavelet_replay,
    write_acceptance_diagnostic_source,
)
from davinci_monet.tests.synthetic.fable_acceptance_plots import (  # noqa: E402
    registered_acceptance_plotters,
)
from davinci_monet.tests.synthetic.fable_v2_acceptance_diagnostics import (  # noqa: E402
    V2AcceptanceDiagnosticSource,
    build_v2_acceptance_diagnostic_source,
    is_v2_acceptance_record,
    v2_acceptance_collection_config,
    verify_v2_wavelet_replay,
    write_v2_acceptance_diagnostic_source,
)

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_ACCEPTANCE_ROOT = (
    REPOSITORY / "analyses/aerosol-tuning/synthetic/acceptance-1179-2358-11-attempt-2"
)
DEFAULT_ICLOUD_ROOT = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Claude"
DIAGNOSTIC_SCHEMA = "fable-acceptance-diagnostics-v1"
V2_DIAGNOSTIC_SCHEMA = "fable-v2-acceptance-diagnostics-v1"
OUTPUT_MARKER = ".fable-acceptance-diagnostics.json"
DiagnosticSource = AcceptanceDiagnosticSource | V2AcceptanceDiagnosticSource


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_diagnostic_source(acceptance_root: Path) -> tuple[DiagnosticSource, str]:
    if is_v2_acceptance_record(acceptance_root):
        return build_v2_acceptance_diagnostic_source(acceptance_root), V2_DIAGNOSTIC_SCHEMA
    return build_acceptance_diagnostic_source(acceptance_root), DIAGNOSTIC_SCHEMA


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _output_marker_document(
    acceptance_root: Path, schema_version: str = DIAGNOSTIC_SCHEMA
) -> dict[str, str]:
    return {
        "schema_version": schema_version,
        "acceptance_root": str(acceptance_root),
    }


def _prepare_output_root(
    acceptance_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    schema_version: str = DIAGNOSTIC_SCHEMA,
) -> None:
    if _paths_overlap(acceptance_root, output_root):
        raise ValueError("diagnostic output must not overlap immutable acceptance inputs")
    marker = output_root / OUTPUT_MARKER
    if output_root.exists():
        if not output_root.is_dir() or output_root.is_symlink():
            raise ValueError(f"diagnostic output is not a regular directory: {output_root}")
        if not overwrite:
            raise FileExistsError(f"diagnostic output exists: {output_root}; use --overwrite")
        try:
            actual_marker = json.loads(marker.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"refusing to overwrite unowned diagnostic output: {output_root}"
            ) from exc
        if actual_marker != _output_marker_document(acceptance_root, schema_version):
            raise ValueError(f"diagnostic output ownership does not match: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    marker.write_text(
        json.dumps(_output_marker_document(acceptance_root, schema_version), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _diagnostic_variables() -> dict[str, dict[str, str]]:
    names = (
        "truth_eof",
        "learned_eof_aligned",
        "eof_residual",
        "truth_correction_rms",
        "estimate_correction_rms",
        "residual_correction_rms",
        "best_representable_rms",
        "primary_valid_days",
        "raw_projected_pc",
        "raw_truth_pc",
        "wavelet_reconstruction_pc",
        "wavelet_truth_target_pc",
        "raw_eligible",
        "wavelet_valid_segment",
        "wavelet_coi_safe",
        "score_day",
        "mode_observable",
        "mode_similarity",
        "basis_scale_to_truth",
        "explained_variance",
        "coefficient_correlation",
        "coefficient_origin_slope",
        "coefficient_nrmse",
        "field_nrmse",
        "primary_valid_count",
        "primary_candidate_count",
        "subspace_angle_mean_degrees",
        "subspace_angle_max_degrees",
        "subspace_projector_error",
        "truth_snapshot",
        "estimate_snapshot",
        "residual_snapshot",
        "snapshot_valid_count",
        "snapshot_rmse",
        "snapshot_nrmse",
        "snapshot_active_domain",
    )
    return {name: {"units": "1"} for name in names}


def _wavelet_spec(source: str) -> dict[str, Any]:
    return {
        "type": "wavelet_filter",
        "source": source,
        "variable": "pc",
        "resolution_variable": "resolution",
        "min_resolution": 0.3,
        "keep_significant": False,
        "significance_level": 0.95,
        "band": {"min": 4.0, "max": 180.0, "units": "days"},
        "max_bridge_days": 7,
        "min_segment_days": 360,
        "omega0": 6.0,
        "dj": 0.25,
        "required": True,
    }


def _build_config(
    source: DiagnosticSource,
    diagnostic_source: Path,
    run_root: Path,
) -> dict[str, Any]:
    seeds = source.seeds
    sources: dict[str, Any] = {
        "acceptance_diagnostics": {
            "type": "generic",
            "files": str(diagnostic_source),
            "variables": _diagnostic_variables(),
        }
    }
    analyses: dict[str, Any] = {}
    plots: dict[str, Any] = {
        "spatial_recovery": {
            "type": "fable_spatial_recovery",
            "source": "acceptance_diagnostics",
            "variable": "truth_snapshot",
            "formats": ["png", "pdf"],
            "output_subdir": "summary",
        },
        "eof_comparison": {
            "type": "fable_eof_comparison",
            "source": "acceptance_diagnostics",
            "variable": "truth_eof",
            "formats": ["png", "pdf"],
            "output_subdir": "eofs",
        },
        "pc_reconstruction": {
            "type": "fable_pc_reconstruction",
            "source": "acceptance_diagnostics",
            "variable": "wavelet_reconstruction_pc",
            "formats": ["png", "pdf"],
            "output_subdir": "time_series",
        },
    }
    for seed in seeds:
        source_name = f"projection_{seed}"
        analysis_name = f"wavelet_replay_{seed}"
        sources[source_name] = _collection_config(
            source,
            seed,
            {"pc": {"units": "1"}, "resolution": {"units": "1"}},
        )
        analyses[analysis_name] = _wavelet_spec(source_name)
        for mode in (1, 2):
            plots[f"wavelet_seed_{seed}_mode_{mode}"] = {
                "type": "fable_wavelet_scalogram",
                "source": analysis_name,
                "variable": "power",
                "mode": mode,
                "title": f"FABLE Projected PC Wavelet Power | Seed {seed}, Mode {mode}",
                "formats": ["png", "pdf"],
                "output_subdir": "wavelet",
            }
    return {
        "analysis": {
            "workflow": "standard",
            "start_time": "2001-01-01 00:00:00",
            "end_time": "2008-12-31 23:59:59",
            "output_dir": str(run_root),
            "log_dir": str(run_root / "logs"),
            "style": {"theme": "ncar", "context": "publication"},
        },
        "sources": sources,
        "analyses": analyses,
        "plots": plots,
    }


def _generated_plots(result: Any, seeds: Sequence[int]) -> list[Path]:
    if result.context is None or "plotting" not in result.context.results:
        raise RuntimeError("diagnostic pipeline did not return a plotting result")
    errors = result.context.metadata.get("plot_errors") or []
    if errors:
        raise RuntimeError("diagnostic plotting was incomplete: " + "; ".join(map(str, errors)))
    values = result.context.results["plotting"].data.get("plots_generated", [])
    paths = [Path(value).resolve() for value in values]
    if not paths or not all(path.is_file() for path in paths):
        raise RuntimeError("diagnostic pipeline did not produce every requested plot")
    expected_pages = 2 + 4 * len(seeds)
    expected_products = 2 * expected_pages
    stems = {(path.parent, path.stem) for path in paths}
    png_count = sum(path.suffix.lower() == ".png" for path in paths)
    pdf_count = sum(path.suffix.lower() == ".pdf" for path in paths)
    if (
        len(paths) != expected_products
        or len(set(paths)) != expected_products
        or len(stems) != expected_pages
        or png_count != expected_pages
        or pdf_count != expected_pages
    ):
        raise RuntimeError(
            f"diagnostic pipeline produced {len(paths)} products; expected "
            f"{expected_pages} paired PNG/PDF pages"
        )
    return paths


def _collection_config(
    source: DiagnosticSource, seed: int, variables: dict[str, dict[str, str]]
) -> dict[str, Any]:
    if isinstance(source, V2AcceptanceDiagnosticSource):
        return v2_acceptance_collection_config(source, seed, "projection", variables)
    return acceptance_collection_config(source.acceptance_root, seed, "projection", variables)


def _verify_replays(result: Any, source: DiagnosticSource) -> dict[str, Any]:
    if result.context is None:
        raise RuntimeError("diagnostic pipeline has no result context")
    report: dict[str, Any] = {}
    for seed in source.seeds:
        name = f"wavelet_replay_{seed}"
        replay = result.context.sources.get(name)
        if replay is None:
            raise RuntimeError(f"diagnostic pipeline is missing {name}")
        if isinstance(source, V2AcceptanceDiagnosticSource):
            values = verify_v2_wavelet_replay(replay.data, source, seed)
        else:
            values = verify_wavelet_replay(replay.data, source.acceptance_root, seed)
        report[str(seed)] = values
    return report


def _gallery_html(title: str, images: Sequence[tuple[str, str]], disposition: str) -> str:
    figures = "\n".join(
        f'<figure><a href="{href}"><img src="{href}" alt="{label}"></a>'
        f"<figcaption>{label}</figcaption></figure>"
        for label, href in images
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ margin: 0; color: #011837; background: #fff; font-family: Arial, sans-serif; }}
header {{ padding: 24px 4vw 14px; border-bottom: 3px solid #0a5dda; }}
h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
p {{ margin: 0; color: #58595b; }}
main {{ padding: 20px 4vw 40px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; }}
figure {{ margin: 0; min-width: 0; border-bottom: 1px solid #cedff8; padding-bottom: 14px; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ margin-top: 8px; font-weight: 600; }}
@media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header><h1>{title}</h1><p>{disposition}</p></header>
<main>{figures}</main>
</body>
</html>
"""


def _write_local_gallery(output_root: Path, plots: Sequence[Path], disposition: str) -> Path:
    pngs = [path for path in plots if path.suffix.lower() == ".png"]
    images = [
        (path.stem.replace("_", " ").title(), os.path.relpath(path, output_root)) for path in pngs
    ]
    index = output_root / "index.html"
    index.write_text(
        _gallery_html("FABLE Synthetic Acceptance Diagnostics", images, disposition),
        encoding="utf-8",
    )
    return index


def _deliver(
    delivery_root: Path,
    acceptance_root: Path,
    plots: Sequence[Path],
) -> Path:
    destination = delivery_root / f"FABLE_{acceptance_root.name}_diagnostics"
    if _paths_overlap(acceptance_root, destination):
        raise ValueError("diagnostic delivery must not overlap immutable acceptance inputs")
    pdfs = [path for path in plots if path.suffix.lower() == ".pdf"]
    expected_names = {path.name for path in pdfs}
    if not pdfs or len(expected_names) != len(pdfs):
        raise ValueError("diagnostic PDFs must have unique delivery names")
    legacy_names = expected_names | {path.name for path in plots if path.suffix.lower() == ".png"}
    legacy_names.update(("diagnostic-record.json", "index.html"))
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise ValueError(f"diagnostic delivery is not a regular directory: {destination}")
        entries = list(destination.iterdir())
        unexpected = [
            path for path in entries if not path.is_file() or path.name not in legacy_names
        ]
        if unexpected:
            names = ", ".join(sorted(path.name for path in unexpected))
            raise ValueError(f"refusing to alter unrecognized diagnostic delivery files: {names}")
        for path in entries:
            if path.name not in expected_names:
                path.unlink()
    else:
        destination.mkdir(parents=True)
    for source in pdfs:
        target = destination / source.name
        shutil.copy2(source, target)
    return destination


def run(
    acceptance_root: Path,
    output_root: Path,
    delivery_root: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    source, schema_version = _load_diagnostic_source(acceptance_root)
    acceptance_root = source.acceptance_root
    _prepare_output_root(
        acceptance_root, output_root, overwrite=overwrite, schema_version=schema_version
    )

    destination = output_root / "data/diagnostics.nc"
    if isinstance(source, V2AcceptanceDiagnosticSource):
        data_path = write_v2_acceptance_diagnostic_source(source, destination)
    else:
        data_path = write_acceptance_diagnostic_source(source, destination)
    run_root = output_root / "run"
    config = _build_config(source, data_path, run_root)
    config_path = output_root / "diagnostics.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with registered_acceptance_plotters():
        result = run_analysis(str(config_path), show_progress=False)
    if not result.success:
        failures = "; ".join(
            f"{stage.stage_name}: {stage.error or 'unknown error'}"
            for stage in result.failed_stages
        )
        raise RuntimeError(f"diagnostic pipeline failed: {failures}")
    plots = _generated_plots(result, source.seeds)
    replay = _verify_replays(result, source)
    disposition = str(source.dataset.attrs["diagnostic_disposition"])
    record = {
        "schema_version": schema_version,
        "acceptance_root": str(acceptance_root),
        "acceptance_record_sha256": source.dataset.attrs["acceptance_record_sha256"],
        "diagnostic_source": {"path": str(data_path), "sha256": _sha256(data_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "snapshot_time": str(source.snapshot_time),
        "seeds": list(source.seeds),
        "wavelet_replay": replay,
        "plots": [{"path": str(path), "sha256": _sha256(path)} for path in sorted(plots)],
        "disposition": disposition,
    }
    record_path = output_root / "diagnostic-record.json"
    record_path.write_text(
        json.dumps(
            _json_value(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    local_index = _write_local_gallery(output_root, plots, disposition)
    delivery_path = _deliver(delivery_root, acceptance_root, plots)
    return local_index, delivery_path, record_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acceptance_root", nargs="?", type=Path, default=DEFAULT_ACCEPTANCE_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--icloud-root", type=Path, default=DEFAULT_ICLOUD_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    acceptance_root = args.acceptance_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else REPOSITORY / "analyses/aerosol-tuning/plots" / acceptance_root.name
    )
    local_index, delivery_path, record_path = run(
        acceptance_root,
        output_root,
        args.icloud_root.expanduser().resolve(),
        overwrite=bool(args.overwrite),
    )
    print(local_index)
    print(delivery_path)
    print(record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
