"""Single-generation synthetic bundle sharing for isolated FABLE v2 policy runs."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import xarray as xr

from davinci_monet.tests.synthetic._aerosol_contracts import (
    SyntheticTuningBundle,
    SyntheticTuningSpec,
    spec_hash,
)
from davinci_monet.tests.synthetic._aerosol_io import (
    generate_aerosol_tuning_bundle,
    scientific_dataset_hash,
    write_aerosol_tuning_bundle,
)
from davinci_monet.tests.synthetic.fable_v2_freeze import FrozenFileIdentity
from davinci_monet.tests.synthetic.fable_v2_lifecycle import (
    V2GenerationLock,
    verify_v2_generation_lock,
)


def generate_v2_scenario_bundle(
    lock: V2GenerationLock, spec: SyntheticTuningSpec
) -> FrozenFileIdentity:
    """Generate and release one immutable role/seed bundle before policy runs."""
    _validate_lock_spec(lock, spec)
    destination = lock.root / "bundles" / f"seed-{spec.master_seed}"
    if destination.exists():
        raise FileExistsError(f"v2 role/seed bundle already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    bundle: SyntheticTuningBundle | None = None
    try:
        bundle = generate_aerosol_tuning_bundle(spec)
        manifest = write_aerosol_tuning_bundle(destination, bundle)
    finally:
        if bundle is not None:
            _release_bundle(bundle)
        gc.collect()
    validate_v2_scenario_bundle(destination, spec)
    return FrozenFileIdentity.capture(manifest)


def link_v2_scenario_bundle(
    lock: V2GenerationLock,
    destination: str | Path,
    spec: SyntheticTuningSpec,
    policy_id: str,
) -> Path:
    """Validate immutable inputs and link them into one isolated policy root."""
    _validate_lock_spec(lock, spec)
    root = Path(destination).expanduser().resolve()
    expected = lock.root / "runs" / policy_id / f"seed-{spec.master_seed}"
    if root != expected:
        raise ValueError(f"v2 policy run root must be the canonical locked path: {expected}")
    source = lock.root / "bundles" / f"seed-{spec.master_seed}"
    validate_v2_scenario_bundle(source, spec)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "oracle", "scenario.json"):
        link = root / name
        link.symlink_to(source / name, target_is_directory=name != "scenario.json")
        if not link.is_symlink() or link.resolve() != (source / name).resolve():
            raise ValueError(f"v2 policy input link target changed: {link}")
    return root / "scenario.json"


def _validate_lock_spec(lock: V2GenerationLock, spec: SyntheticTuningSpec) -> None:
    verify_v2_generation_lock(lock)
    if spec.master_seed not in lock.seeds:
        raise ValueError("v2 scenario seed is outside its verified generation lock")
    expected_scenario = (
        "synthetic_osse_null" if lock.role == "calibration_null" else "synthetic_osse"
    )
    if spec.scenario != expected_scenario:
        raise ValueError(f"v2 {lock.role} generation requires scenario {expected_scenario!r}")


def validate_v2_scenario_bundle(
    root: Path, spec: SyntheticTuningSpec, *, verify_scientific: bool = False
) -> Path:
    """Rehash a shared bundle and require the exact scenario/spec identity."""
    root = root.expanduser().resolve()
    manifest = root / "scenario.json"
    if not manifest.is_file():
        raise ValueError(f"shared v2 scenario manifest is missing: {manifest}")
    try:
        document = json.loads(manifest.read_text(encoding="ascii"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"shared v2 scenario manifest is invalid: {manifest}") from exc
    if (
        not isinstance(document, Mapping)
        or document.get("spec_hash") != spec_hash(spec)
        or document.get("root_seed") != spec.master_seed
        or document.get("scenario") != spec.scenario
    ):
        raise ValueError("shared v2 scenario does not match the requested role/seed spec")
    files = document.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("shared v2 scenario has no file identity table")
    scientific_receipts: list[tuple[str, str, str]] = []
    byte_digest = hashlib.sha256()
    for relative, identity in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(identity, Mapping):
            raise ValueError("shared v2 scenario file identity table is invalid")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("shared v2 scenario file identity escapes its root") from exc
        expected = identity.get("byte_sha256")
        actual_byte_sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        if not path.is_file() or not isinstance(expected, str) or actual_byte_sha256 != expected:
            raise ValueError(f"shared v2 scenario file identity changed: {relative}")
        byte_digest.update(relative.encode("utf-8"))
        byte_digest.update(actual_byte_sha256.encode("ascii"))
        scientific = identity.get("scientific_sha256")
        if not isinstance(scientific, str):
            raise ValueError(f"shared v2 scenario has no scientific receipt: {relative}")
        scientific_receipts.append((str(path), relative, scientific))
    if verify_scientific:
        _validate_scientific_receipts(tuple(scientific_receipts), byte_digest.hexdigest())
    return manifest


@lru_cache(maxsize=32)
def _validate_scientific_receipts(
    receipts: tuple[tuple[str, str, str], ...], byte_identity: str
) -> None:
    del byte_identity
    for raw_path, relative, expected in receipts:
        with xr.open_dataset(raw_path) as dataset:
            actual = scientific_dataset_hash(dataset)
        if actual != expected:
            raise ValueError(f"shared v2 scenario scientific identity changed: {relative}")


def _release_bundle(bundle: SyntheticTuningBundle) -> None:
    datasets = [bundle.model, *bundle.observations.values(), *bundle.mmr.values(), bundle.truth]
    for dataset in datasets:
        dataset.close()


__all__ = [
    "generate_v2_scenario_bundle",
    "link_v2_scenario_bundle",
    "validate_v2_scenario_bundle",
]
