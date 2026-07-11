"""Checksum validation bridge for finalized FABLE acceptance artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from davinci_monet.analysis.artifact_manifest import validate_finalized_artifact_manifest


def validate_recovery_artifact(value: Mapping[str, Any], manifest_path: str) -> None:
    """Validate the recovery entry against its completed run manifest and files."""
    checksums = value["checksums"]
    if value.get("kind") == "netcdf_collection":
        files = checksums.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("artifact collection has no file receipt")
        root = Path(str(value["artifact_dir"]))
        paths = [root / str(name) for name in sorted(files)]
    elif value.get("kind") == "product":
        paths = [Path(str(value["artifact_path"]))]
    else:
        raise ValueError("unsupported recovery artifact kind")
    manifest_entry = validate_finalized_artifact_manifest(
        manifest_path,
        paths,
        role=str(value["role"]),
        analysis=str(value["analysis"]),
    )
    if dict(value) != manifest_entry:
        raise ValueError("recorded recovery artifact does not match its pipeline manifest entry")


__all__ = ["validate_recovery_artifact"]
