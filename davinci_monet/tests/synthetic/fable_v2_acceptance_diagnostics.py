"""Read-only plot sources for canonical FABLE v2 acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.analysis.artifacts import load_dataset_collection
from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec
from davinci_monet.tests.synthetic.fable_acceptance_diagnostics import (
    _close_seed_inputs,
    _finalize_seed,
    _prepare_seed,
    _SeedInputs,
)
from davinci_monet.tests.synthetic.fable_v2_acceptance import (
    V2_ACCEPTANCE_SCHEMA,
    V2AcceptanceRecord,
    load_v2_acceptance_record,
    validate_v2_acceptance_record,
)
from davinci_monet.tests.synthetic.fable_v2_evidence import (
    validate_v2_scenario_evidence,
)
from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_identity import (
    load_current_v2_preregistration,
)


@dataclass(frozen=True)
class V2AcceptanceDiagnosticSource:
    """Compact diagnostics plus validated roots for one v2 acceptance record."""

    dataset: xr.Dataset
    acceptance_root: Path
    record_path: Path
    seeds: tuple[int, ...]
    snapshot_time: np.datetime64
    run_roots: tuple[tuple[int, Path], ...]

    def run_root(self, seed: int) -> Path:
        """Return the fitting-manifest-derived root for one locked seed."""
        matches = [root for value, root in self.run_roots if value == seed]
        if len(matches) != 1:
            raise ValueError(f"seed {seed} is not in this v2 acceptance source")
        return matches[0]


def is_v2_acceptance_record(source: str | Path) -> bool:
    """Recognize a v2 envelope without weakening its canonical loader."""
    path = _record_path(source)
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(value, Mapping):
        return False
    record = value.get("record")
    return bool(
        "record_sha256" in value
        or (isinstance(record, Mapping) and record.get("schema_version") == V2_ACCEPTANCE_SCHEMA)
    )


def build_v2_acceptance_diagnostic_source(
    source: str | Path,
) -> V2AcceptanceDiagnosticSource:
    """Validate a canonical v2 record and prepare its read-only plot dataset."""
    record_path = _record_path(source)
    record = load_v2_acceptance_record(record_path)
    current, frozen = load_current_v2_preregistration(record.preregistration.file.path)
    record = validate_v2_acceptance_record(
        record,
        current,
        frozen,
        record.calibration_record,
        record.preflight_record,
    )
    root = _validated_generation_root(record_path, record)
    run_roots = _validated_run_roots(root, record)

    inputs: list[_SeedInputs] = []
    try:
        for result, (_, run_root) in zip(record.results, run_roots, strict=True):
            inputs.append(_load_seed_inputs(run_root, result.seed, result.evidence))
        prepared = [_prepare_seed(value) for value in inputs]
    finally:
        for value in inputs:
            _close_seed_inputs(value)

    common_score_day = prepared[0]["score_day"].astype(bool)
    for prepared_seed in prepared[1:]:
        common_score_day = common_score_day & prepared_seed["score_day"].astype(bool)
    common_times = common_score_day["time"].where(common_score_day, drop=True)
    if not common_times.size:
        raise ValueError("v2 acceptance seeds have no common primary-safe diagnostic day")
    snapshot_time = np.datetime64(common_times.values[common_times.size // 2], "ns")
    per_seed = [
        _finalize_seed(value, snapshot_time).expand_dims(seed=[int(value.attrs["root_seed"])])
        for value in prepared
    ]
    dataset = xr.concat(per_seed, dim="seed", join="exact", combine_attrs="drop_conflicts")
    disposition = _disposition(record.status)
    dataset.attrs.update(
        {
            "geometry": "grid",
            "source_label": "fable_v2_acceptance_diagnostics",
            "analysis_type": "fable_v2_acceptance_diagnostics",
            "acceptance_status": record.status,
            "acceptance_record_sha256": _sha256(record_path),
            "generation_lock_sha256": record.generation_lock.sha256,
            "selected_policy_id": record.selected_policy.policy_id,
            "snapshot_time": str(snapshot_time),
            "snapshot_selection": "temporal median of common primary-safe evaluation days",
            "primary_mask": (
                "development_test and coefficient_available and observable-mode valid_segment "
                "and coi>=band_max and spatial_support>0 and finite estimate/truth"
            ),
            "diagnostic_disposition": disposition,
        }
    )
    return V2AcceptanceDiagnosticSource(
        dataset,
        root,
        record_path,
        tuple(result.seed for result in record.results),
        snapshot_time,
        run_roots,
    )


def write_v2_acceptance_diagnostic_source(
    source: V2AcceptanceDiagnosticSource, destination: str | Path
) -> Path:
    """Write the compact, derived source outside the immutable v2 root."""
    path = Path(destination).expanduser().resolve()
    if path == source.acceptance_root or source.acceptance_root in path.parents:
        raise ValueError("v2 diagnostic output must not alter the acceptance root")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    encoding = {
        name: {"zlib": True, "complevel": 1, "shuffle": True}
        for name, value in source.dataset.data_vars.items()
        if value.ndim and value.dtype.kind in "fiu"
    }
    source.dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


def v2_acceptance_collection_config(
    source: V2AcceptanceDiagnosticSource,
    seed: int,
    artifact: str,
    variables: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Return an exact-file source config rooted in one validated v2 run."""
    details = {
        "projection": ("obs_pcs", "projection_fit"),
        "scaling": ("scaling", "scaling"),
    }
    try:
        analysis, role = details[artifact]
    except KeyError as exc:
        raise ValueError(f"unsupported v2 acceptance collection: {artifact}") from exc
    run_root = source.run_root(seed)
    manifest = run_root / "output/manifest.json"
    paths = _artifact_paths(manifest, run_root / "output", role, analysis)
    return {
        "type": "generic",
        "files": [str(path) for path in paths],
        "artifact_manifest": str(manifest),
        "artifact_role": role,
        "artifact_analysis": analysis,
        "combine": "nested",
        "concat_dim": "time",
        "data_vars": "minimal",
        "coords": "minimal",
        "compat": "override",
        "join": "exact",
        "variables": variables,
    }


def verify_v2_wavelet_replay(
    replayed: xr.Dataset,
    source: V2AcceptanceDiagnosticSource,
    seed: int,
) -> dict[str, list[float]]:
    """Require a v2 replay to match the validated persisted reconstruction."""
    run_root = source.run_root(seed)
    paths = _artifact_paths(
        run_root / "output/manifest.json", run_root / "output", "scaling", "scaling"
    )
    persisted = load_dataset_collection(paths)
    try:
        expected = persisted[["pc", "coi", "valid_segment"]].load()
    finally:
        persisted.close()
    for coord in ("time", "mode"):
        if not np.array_equal(replayed[coord].values, expected[coord].values):
            raise ValueError(f"seed {seed} replayed wavelet {coord} coordinate changed")
    for name in ("pc", "coi"):
        if not np.array_equal(replayed[name].values, expected[name].values, equal_nan=True):
            raise ValueError(f"seed {seed} replayed wavelet {name} changed")
    if not np.array_equal(replayed["valid_segment"].values, expected["valid_segment"].values):
        raise ValueError(f"seed {seed} replayed wavelet valid_segment changed")
    return {
        name: [float(value) for value in np.asarray(replayed[name].values).reshape(-1)]
        for name in (
            "retained_variance",
            "recon_error",
            "synth_fraction",
            "coi_valid_fraction",
        )
    }


def _validated_generation_root(record_path: Path, record: V2AcceptanceRecord) -> Path:
    record.generation_lock.verify()
    lock_path = Path(record.generation_lock.path).resolve()
    root = lock_path.parent
    if lock_path != root / "generation-lock.json" or record_path != root / "acceptance.json":
        raise ValueError("v2 acceptance record is outside its frozen generation root")
    return root


def _validated_run_roots(root: Path, record: V2AcceptanceRecord) -> tuple[tuple[int, Path], ...]:
    values: list[tuple[int, Path]] = []
    for result in record.results:
        evidence = result.evidence
        validate_v2_scenario_evidence(
            evidence,
            SyntheticTuningSpec.synthetic_osse(result.seed),
            record.selected_policy,
            score_kind="recovery",
            evaluation_splits=("development_test",),
            verify_files=True,
        )
        manifest = _identity(evidence, "fitting_manifest")
        run_root = Path(manifest.path).resolve().parents[1]
        expected = root / "runs" / record.selected_policy.policy_id / f"seed-{result.seed}"
        if (
            Path(manifest.path).resolve() != expected / "output/manifest.json"
            or run_root != expected
        ):
            raise ValueError(
                f"seed {result.seed} fitting manifest is outside its locked policy run root"
            )
        evaluation = _identity(evidence, "evaluation_manifest")
        if Path(evaluation.path).resolve() != run_root / "evaluation/manifest.json":
            raise ValueError(f"seed {result.seed} evaluation manifest has a different run root")
        bundle = root / "bundles" / f"seed-{result.seed}"
        if (run_root / "oracle").resolve() != (bundle / "oracle").resolve():
            raise ValueError(f"seed {result.seed} oracle link has a different bundle root")
        values.append((result.seed, run_root))
    return tuple(values)


def _load_seed_inputs(run_root: Path, seed: int, evidence: Mapping[str, Any]) -> _SeedInputs:
    fitting = Path(_identity(evidence, "fitting_manifest").path).resolve()
    evaluation = Path(_identity(evidence, "evaluation_manifest").path).resolve()
    basis_paths = _artifact_paths(fitting, run_root / "output", "basis_fit", "aod_basis")
    recovery_paths = _artifact_paths(
        evaluation, run_root / "evaluation", "recovery_report", "recovery"
    )
    if len(basis_paths) != 1 or len(recovery_paths) != 1:
        raise ValueError(f"seed {seed} requires single-file basis and recovery artifacts")
    return _SeedInputs(
        seed,
        xr.open_dataset(basis_paths[0]),
        load_dataset_collection(
            _artifact_paths(fitting, run_root / "output", "projection_fit", "obs_pcs")
        ),
        load_dataset_collection(
            _artifact_paths(fitting, run_root / "output", "scaling", "scaling")
        ),
        xr.open_dataset(run_root / "oracle/truth.nc"),
        xr.open_dataset(recovery_paths[0]),
    )


def _artifact_paths(
    manifest: Path, output_root: Path, role: str, analysis: str
) -> tuple[Path, ...]:
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid v2 pipeline manifest: {manifest}") from exc
    entries = document.get("analysis_artifacts") if isinstance(document, Mapping) else None
    matches = (
        [
            item
            for item in entries
            if isinstance(item, Mapping)
            and item.get("role") == role
            and item.get("analysis") == analysis
        ]
        if isinstance(entries, list)
        else []
    )
    if len(matches) != 1 or matches[0].get("kind") != "netcdf_collection":
        raise ValueError(f"v2 manifest must contain one {analysis}/{role} collection")
    entry = matches[0]
    artifact_root = Path(str(entry.get("artifact_dir", ""))).resolve()
    expected_root = (output_root / "artifacts" / analysis).resolve()
    checksums = entry.get("checksums")
    files = checksums.get("files") if isinstance(checksums, Mapping) else None
    if artifact_root != expected_root or not isinstance(files, Mapping) or not files:
        raise ValueError(f"v2 {analysis}/{role} collection has an invalid root or receipt")
    paths = tuple((artifact_root / str(name)).resolve() for name in sorted(files))
    if any(path.parent != artifact_root or not path.is_file() for path in paths):
        raise ValueError(f"v2 {analysis}/{role} collection contains an invalid path")
    return paths


def _identity(value: Mapping[str, Any], name: str) -> FrozenFileIdentity:
    raw = value.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"v2 evidence is missing {name}")
    return FrozenFileIdentity(str(raw.get("path")), str(raw.get("sha256")))


def _record_path(source: str | Path) -> Path:
    path = Path(source).expanduser().resolve()
    return path if path.is_file() else path / "acceptance.json"


def _disposition(status: str) -> str:
    if status == "passed_pending_user_review":
        return "diagnostic only; acceptance passed pending user review"
    if status == "failed":
        return "diagnostic only; acceptance failed"
    raise ValueError(f"unsupported v2 acceptance disposition: {status}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "V2AcceptanceDiagnosticSource",
    "build_v2_acceptance_diagnostic_source",
    "is_v2_acceptance_record",
    "v2_acceptance_collection_config",
    "verify_v2_wavelet_replay",
    "write_v2_acceptance_diagnostic_source",
]
