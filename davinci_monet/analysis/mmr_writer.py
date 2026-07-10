"""Atomic application of analysis-grid AOD ratios to native aerosol MMR files."""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from davinci_monet.analysis.base import (
    AnalysisExecutionError,
    AnalysisResult,
    AnalysisRuntime,
    DerivedAnalysis,
)
from davinci_monet.analysis.mmr_writer_hashes import (
    hash_application_config,
    hash_scaling_dataset,
    implementation_hash,
)
from davinci_monet.analysis.mmr_writer_io import (
    FileContract,
    atomic_scale_mmr_file,
    inspect_mmr_file,
    read_writer_hashes,
    sha256_file,
    validate_resumable_mmr_file,
)
from davinci_monet.analysis.mmr_writer_scaling import ValidatedScaling as _ValidatedScaling
from davinci_monet.analysis.mmr_writer_scaling import (
    interpolate_validated_native_ratio as _interpolate_validated_native_ratio,
)
from davinci_monet.analysis.mmr_writer_scaling import validate_scaling as _validated_scaling
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry

if TYPE_CHECKING:
    from davinci_monet.config.schema import MMRWriterSpec


DEFAULT_AEROSOL_SPECIES = (
    "DU001",
    "DU002",
    "DU003",
    "DU004",
    "DU005",
    "SS001",
    "SS002",
    "SS003",
    "SS004",
    "SS005",
    "SO4",
    "BCPHOBIC",
    "BCPHILIC",
    "OCPHOBIC",
    "OCPHILIC",
)
WRITER_VERSION = "fable-mmr-writer-v1"
CODE_SHA256 = implementation_hash()


@dataclass(frozen=True)
class _FilePlan:
    input_path: Path
    output_path: Path
    contract: FileContract
    input_sha256: str


@dataclass(frozen=True)
class _FileResult:
    input_path: Path
    input_sha256: str
    output_path: Path | None
    status: str
    output_sha256: str | None
    payload_sha256: str | None
    statistics: Mapping[str, int]


def interpolate_native_ratio(
    scaling: xr.Dataset,
    target_time: xr.DataArray | NDArray[np.datetime64] | Sequence[np.datetime64],
    target_lat: xr.DataArray | NDArray[np.floating] | Sequence[float],
    target_lon: xr.DataArray | NDArray[np.floating] | Sequence[float],
    *,
    outside_coverage: str = "identity",
    _validated: _ValidatedScaling | None = None,
) -> xr.Dataset:
    """Interpolate ``ln(r)`` periodically in space and linearly in time.

    Source time endpoints are never held outside coverage. Those samples are
    exact identity unless ``outside_coverage='error'`` requests a failure.
    """
    validated = _validated if _validated is not None else _validated_scaling(scaling)
    return _interpolate_validated_native_ratio(
        validated,
        target_time,
        target_lat,
        target_lon,
        outside_coverage=outside_coverage,
    )


def write_corrected_mmr_files(
    scaling: xr.Dataset,
    spec: MMRWriterSpec,
) -> AnalysisResult:
    """Write every direct-glob MMR input and return its artifact pseudo-source."""
    species = tuple(spec.species) if spec.species is not None else DEFAULT_AEROSOL_SPECIES
    input_paths = _expand_files(spec.files)
    plans = _preflight(input_paths, Path(spec.output_dir), species)
    validated = _validated_scaling(scaling)
    scaling_sha256 = hash_scaling_dataset(validated.ratio, validated.support, scaling.attrs)
    config_sha256 = hash_application_config(species, spec.time_interp, spec.outside_coverage)
    scenario_hash = str(scaling.attrs.get("spec_hash", scaling.attrs.get("scenario_hash", "")))
    for plan in plans:
        attributes = plan.contract.global_attributes
        input_scenario = str(attributes.get("spec_hash", attributes.get("scenario_hash", "")))
        if scenario_hash and input_scenario and input_scenario != scenario_hash:
            raise ValueError(
                f"synthetic scenario hash mismatch for {plan.input_path}: "
                f"{input_scenario} != {scenario_hash}"
            )
    results: list[_FileResult] = []

    for plan in plans:
        expected_hashes = {
            "input": plan.input_sha256,
            "config": config_sha256,
            "scaling": scaling_sha256,
            "code": CODE_SHA256,
        }
        provenance = {
            "input_sha256": plan.input_sha256,
            "config_sha256": config_sha256,
            "scaling_sha256": scaling_sha256,
            "code_sha256": CODE_SHA256,
            "scenario_hash": scenario_hash,
            "writer_version": WRITER_VERSION,
        }
        statistics: Mapping[str, int] = {}
        publish_attempted = False
        try:
            coordinates = _native_coordinates(plan.input_path)
            correction = interpolate_native_ratio(
                scaling,
                coordinates["time"],
                coordinates["lat"],
                coordinates["lon"],
                outside_coverage=spec.outside_coverage,
                _validated=validated,
            )
            inside = np.asarray(correction["inside_coverage"].values, dtype=np.bool_)
            statistics = _correction_statistics(correction, scaling)
            if spec.outside_coverage == "skip" and not inside.all():
                results.append(
                    _FileResult(
                        plan.input_path,
                        plan.input_sha256,
                        None,
                        "skipped_outside_coverage",
                        None,
                        None,
                        statistics,
                    )
                )
                continue

            if plan.output_path.exists():
                if _same_file(plan.input_path, plan.output_path):
                    raise ValueError(f"input and output paths alias: {plan.input_path}")
                if spec.resume:
                    try:
                        actual_hashes = read_writer_hashes(plan.output_path)
                        if actual_hashes != expected_hashes:
                            raise ValueError(
                                f"resume hashes do not match existing output {plan.output_path}"
                            )
                        payload_sha256 = validate_resumable_mmr_file(
                            plan.output_path, species, plan.contract, provenance
                        )
                    except (OSError, ValueError, RuntimeError):
                        if not spec.overwrite:
                            raise
                    else:
                        results.append(
                            _FileResult(
                                plan.input_path,
                                plan.input_sha256,
                                plan.output_path,
                                "resumed",
                                sha256_file(plan.output_path),
                                payload_sha256,
                                statistics,
                            )
                        )
                        continue
                elif not spec.overwrite:
                    raise FileExistsError(f"output already exists: {plan.output_path}")

            publish_attempted = True
            output_sha256, payload_sha256 = atomic_scale_mmr_file(
                plan.input_path,
                plan.output_path,
                species,
                np.asarray(correction["ratio"].values, dtype=np.float64),
                contract=plan.contract,
                provenance=provenance,
            )
            results.append(
                _FileResult(
                    plan.input_path,
                    plan.input_sha256,
                    plan.output_path,
                    "written",
                    output_sha256,
                    payload_sha256,
                    statistics,
                )
            )
        except Exception as exc:
            _record_finalized_after_error(
                results,
                plan,
                species,
                provenance,
                statistics,
                publish_attempted=publish_attempted,
            )
            if results:
                entries = tuple(
                    _manifest_entry(result, scaling_sha256, config_sha256, scenario_hash)
                    for result in results
                )
                raise AnalysisExecutionError(str(exc), manifest_entries=entries) from exc
            raise

    dataset = _artifact_dataset(results, scaling_sha256, config_sha256, scenario_hash)
    manifest_entries = tuple(
        _manifest_entry(result, scaling_sha256, config_sha256, scenario_hash) for result in results
    )
    return AnalysisResult(dataset=dataset, manifest_entries=manifest_entries)


def _record_finalized_after_error(
    results: list[_FileResult],
    plan: _FilePlan,
    species: Sequence[str],
    provenance: Mapping[str, str],
    statistics: Mapping[str, int],
    *,
    publish_attempted: bool,
) -> None:
    """Recover a receipt when replacement succeeded but final durability reporting failed."""
    if not publish_attempted or not statistics or not plan.output_path.is_file():
        return
    try:
        payload_sha256 = validate_resumable_mmr_file(
            plan.output_path, species, plan.contract, provenance
        )
        output_sha256 = sha256_file(plan.output_path)
    except (OSError, ValueError, RuntimeError):
        return
    results.append(
        _FileResult(
            plan.input_path,
            plan.input_sha256,
            plan.output_path,
            "finalized_before_failure",
            output_sha256,
            payload_sha256,
            statistics,
        )
    )


def _preflight(
    input_paths: Sequence[Path], output_dir: Path, species: Sequence[str]
) -> list[_FilePlan]:
    destinations = [output_dir / path.name for path in input_paths]
    if len({str(path.absolute()) for path in destinations}) != len(destinations):
        raise ValueError("input glob maps multiple files to the same output filename")
    plans: list[_FilePlan] = []
    for input_path, output_path in zip(input_paths, destinations):
        if _same_path(input_path, output_path):
            raise ValueError(f"input and output paths alias: {input_path}")
        contract = inspect_mmr_file(input_path, species)
        plans.append(
            _FilePlan(
                input_path=input_path,
                output_path=output_path,
                contract=contract,
                input_sha256=sha256_file(input_path),
            )
        )
    return plans


def _expand_files(pattern: str) -> list[Path]:
    expanded = os.path.expandvars(os.path.expanduser(pattern))
    paths = sorted(Path(value) for value in glob.glob(expanded, recursive=False))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"MMR file glob matched no files: {pattern}")
    return paths


def _native_coordinates(path: Path) -> dict[str, NDArray[Any]]:
    with xr.open_dataset(path, decode_times=True, mask_and_scale=False) as dataset:
        missing = [name for name in ("time", "lat", "lon") if name not in dataset.coords]
        if missing:
            raise ValueError(f"{path} is missing native coordinates: {', '.join(missing)}")
        coordinates = {
            name: np.asarray(dataset[name].values).copy() for name in ("time", "lat", "lon")
        }
    return coordinates


def _correction_statistics(correction: xr.Dataset, scaling: xr.Dataset) -> dict[str, int]:
    ratio = np.asarray(correction["ratio"].values, dtype=np.float64)
    support = np.asarray(correction["support"].values, dtype=np.float64)
    inside_time = np.asarray(correction["inside_coverage"].values, dtype=np.bool_)
    inside = np.broadcast_to(inside_time[:, None, None], ratio.shape)
    r_min = float(scaling.attrs.get("r_min", -np.inf))
    r_max = float(scaling.attrs.get("r_max", np.inf))
    return {
        "horizontal_time_cell_count": int(ratio.size),
        "inside_coverage_count": int(inside.sum()),
        "outside_coverage_identity_count": int((~inside).sum()),
        "spatial_support_identity_count": int((inside & (support <= 0.0)).sum()),
        "active_correction_count": int((inside & (support > 0.0)).sum()),
        "lower_clip_count": int((inside & np.isclose(ratio, r_min)).sum()),
        "upper_clip_count": int((inside & np.isclose(ratio, r_max)).sum()),
    }


def _artifact_dataset(
    results: Sequence[_FileResult], scaling_hash: str, config_hash: str, scenario_hash: str
) -> xr.Dataset:
    files = np.arange(len(results), dtype=np.int64)
    dataset = xr.Dataset(
        {
            "input_path": ("file", [str(result.input_path) for result in results]),
            "output_path": (
                "file",
                [
                    str(result.output_path) if result.output_path is not None else ""
                    for result in results
                ],
            ),
            "status": ("file", [result.status for result in results]),
            "sha256": ("file", [result.output_sha256 or "" for result in results]),
            "payload_sha256": ("file", [result.payload_sha256 or "" for result in results]),
            "outside_coverage_identity_count": (
                "file",
                [result.statistics["outside_coverage_identity_count"] for result in results],
            ),
            "spatial_support_identity_count": (
                "file",
                [result.statistics["spatial_support_identity_count"] for result in results],
            ),
        },
        coords={"file": files},
        attrs={
            "analysis_type": "mmr_writer",
            "scaling_sha256": scaling_hash,
            "config_sha256": config_hash,
            "code_sha256": CODE_SHA256,
            "scenario_hash": scenario_hash,
        },
    )
    return dataset


def _manifest_entry(
    result: _FileResult, scaling_hash: str, config_hash: str, scenario_hash: str
) -> Mapping[str, Any]:
    return {
        "role": "corrected_mmr",
        "kind": "mmr_file",
        "status": result.status,
        "input_path": str(result.input_path),
        "path": str(result.output_path) if result.output_path is not None else None,
        "checksums": {
            "input_sha256": result.input_sha256,
            "output_sha256": result.output_sha256,
            "payload_sha256": result.payload_sha256,
            "scaling_sha256": scaling_hash,
            "config_sha256": config_hash,
            "code_sha256": CODE_SHA256,
        },
        "scenario_hash": scenario_hash,
        "statistics": dict(result.statistics),
    }


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().absolute().resolve(
        strict=False
    ) == right.expanduser().absolute().resolve(strict=False)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return _same_path(left, right)


@analysis_registry.register("mmr_writer")
class MMRWriterAnalysis(DerivedAnalysis):
    """Pipeline adapter for atomic native-grid aerosol correction files."""

    name = "mmr_writer"
    long_name = "Aerosol MMR Writer"
    output_geometry = DataGeometry.ARTIFACT

    def analyze_inputs(
        self,
        inputs: Mapping[str, xr.Dataset],
        spec: MMRWriterSpec,
        runtime: AnalysisRuntime,
    ) -> AnalysisResult:
        del runtime
        if "scaling" not in inputs:
            raise ValueError("mmr_writer is missing named input 'scaling'")
        return write_corrected_mmr_files(inputs["scaling"], spec)


__all__ = [
    "DEFAULT_AEROSOL_SPECIES",
    "MMRWriterAnalysis",
    "interpolate_native_ratio",
    "write_corrected_mmr_files",
]
