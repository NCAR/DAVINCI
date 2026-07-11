"""Canonical preregistration and file identities for the FABLE v2 cycle."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from davinci_monet.tests.synthetic.fable_v2_protocol import (
    CYCLE_ID,
    PREREGISTRATION_SCHEMA,
    protocol_document,
    protocol_sha256,
)

_HASH_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class V2FileBinding:
    """Named file content bound into a preregistration."""

    name: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _trimmed(self.name, "file binding name")
        _trimmed(self.path, "file binding path")
        _sha256(self.sha256, "file binding sha256")

    @classmethod
    def capture(
        cls,
        name: str,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
    ) -> V2FileBinding:
        """Capture a named file with an optional stable repository-relative path."""
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"identity source is not a file: {source}")
        display = (
            source.relative_to(Path(relative_to).expanduser().resolve()) if relative_to else source
        )
        return cls(name, display.as_posix(), _file_sha256(source))

    def normalized(self) -> dict[str, str]:
        return {"name": self.name, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class V2TestEvidence:
    """Passing test evidence bound into a calibration preregistration."""

    suite: str
    command: str
    evidence_sha256: str
    passed: bool = True

    def __post_init__(self) -> None:
        _trimmed(self.suite, "test suite")
        _trimmed(self.command, "test command")
        _sha256(self.evidence_sha256, "test evidence sha256")
        if self.passed is not True:
            raise ValueError("preregistration requires passing test evidence")

    def normalized(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "evidence_sha256": self.evidence_sha256,
            "status": "passed",
            "suite": self.suite,
        }


@dataclass(frozen=True)
class V2Preregistration:
    """Current scientific identities required to open v2 calibration."""

    generator_schema_version: str
    generator_spec_sha256: str
    thresholds_sha256: str
    code_sha256: str
    environment_sha256: str
    configs: tuple[V2FileBinding, ...]
    tests: tuple[V2TestEvidence, ...]

    def __post_init__(self) -> None:
        _trimmed(self.generator_schema_version, "generator schema version")
        for name in (
            "generator_spec_sha256",
            "thresholds_sha256",
            "code_sha256",
            "environment_sha256",
        ):
            _sha256(getattr(self, name), name)
        configs = tuple(sorted(self.configs, key=lambda item: item.name))
        tests = tuple(sorted(self.tests, key=lambda item: item.suite))
        if not configs or len({item.name for item in configs}) != len(configs):
            raise ValueError("preregistration config bindings must be nonempty and uniquely named")
        if not tests or len({item.suite for item in tests}) != len(tests):
            raise ValueError("preregistration test evidence must be nonempty and uniquely named")
        object.__setattr__(self, "configs", configs)
        object.__setattr__(self, "tests", tests)

    def normalized(self) -> dict[str, Any]:
        return {
            "code_sha256": self.code_sha256,
            "configs": [item.normalized() for item in self.configs],
            "cycle_id": CYCLE_ID,
            "environment_sha256": self.environment_sha256,
            "generator": {
                "schema_version": self.generator_schema_version,
                "spec_sha256": self.generator_spec_sha256,
            },
            "protocol": protocol_document(),
            "protocol_sha256": protocol_sha256(),
            "schema_version": PREREGISTRATION_SCHEMA,
            "tests": [item.normalized() for item in self.tests],
            "thresholds_sha256": self.thresholds_sha256,
        }


@dataclass(frozen=True)
class FrozenFileIdentity:
    """Path and byte identity for a file that a later phase must reverify."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _trimmed(self.path, "frozen file path")
        _sha256(self.sha256, "frozen file sha256")

    @classmethod
    def capture(cls, path: str | Path) -> FrozenFileIdentity:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"frozen identity source is not a file: {source}")
        return cls(str(source), _file_sha256(source))

    def verify(self) -> None:
        source = Path(self.path)
        if not source.is_file() or not hmac.compare_digest(_file_sha256(source), self.sha256):
            raise ValueError(f"frozen file identity changed: {source}")

    def normalized(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class FrozenPreregistration:
    """Independent byte and scientific identities of a write-once preregistration."""

    file: FrozenFileIdentity
    preregistration_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.preregistration_sha256, "preregistration sha256")

    def normalized(self) -> dict[str, Any]:
        return {
            "file": self.file.normalized(),
            "preregistration_sha256": self.preregistration_sha256,
        }


def canonical_preregistration_json(preregistration: V2Preregistration) -> str:
    """Serialize only the scientific preregistration content canonically."""
    return _canonical_json(preregistration.normalized())


def preregistration_sha256(preregistration: V2Preregistration) -> str:
    payload = canonical_preregistration_json(preregistration).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_v2_preregistration(
    path: str | Path, preregistration: V2Preregistration
) -> FrozenPreregistration:
    """Atomically publish a canonical, read-only preregistration exactly once."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"v2 preregistration already exists: {destination}")
    scientific_hash = preregistration_sha256(preregistration)
    document = {
        "preregistration": preregistration.normalized(),
        "preregistration_sha256": scientific_hash,
    }
    payload = (_canonical_json(document) + "\n").encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"v2 preregistration already exists: {destination}") from exc
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return FrozenPreregistration(FrozenFileIdentity.capture(destination), scientific_hash)


def load_v2_preregistration(path: str | Path) -> V2Preregistration:
    """Load and verify canonical bytes, scientific hash, and the current protocol."""
    source = Path(path)
    raw = source.read_text(encoding="ascii")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid v2 preregistration JSON: {source}") from exc
    if not isinstance(document, Mapping) or set(document) != {
        "preregistration",
        "preregistration_sha256",
    }:
        raise ValueError("v2 preregistration envelope has unexpected fields")
    if raw != _canonical_json(document) + "\n":
        raise ValueError("v2 preregistration is not canonically encoded")
    value = document["preregistration"]
    stored_hash = document["preregistration_sha256"]
    if not isinstance(value, Mapping) or not isinstance(stored_hash, str):
        raise ValueError("v2 preregistration envelope has invalid value types")
    actual_hash = _sha256_json(value)
    if not hmac.compare_digest(stored_hash, actual_hash):
        raise ValueError("v2 preregistration SHA-256 mismatch")
    preregistration = _preregistration_from_normalized(value)
    if preregistration.normalized() != value:
        raise ValueError("v2 preregistration scientific identity mismatch")
    return preregistration


def verify_v2_preregistration(
    frozen: FrozenPreregistration, current: V2Preregistration
) -> V2Preregistration:
    """Require frozen bytes and every current scientific identity to remain unchanged."""
    frozen.file.verify()
    loaded = load_v2_preregistration(frozen.file.path)
    loaded_hash = preregistration_sha256(loaded)
    if not hmac.compare_digest(loaded_hash, frozen.preregistration_sha256):
        raise ValueError("frozen preregistration identity changed")
    if not hmac.compare_digest(loaded_hash, preregistration_sha256(current)):
        raise ValueError("current v2 scientific identity drifted after preregistration")
    return loaded


def _preregistration_from_normalized(value: Mapping[str, Any]) -> V2Preregistration:
    expected_keys = {
        "code_sha256",
        "configs",
        "cycle_id",
        "environment_sha256",
        "generator",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "tests",
        "thresholds_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("v2 preregistration has unexpected fields")
    if (
        value["cycle_id"] != CYCLE_ID
        or value["schema_version"] != PREREGISTRATION_SCHEMA
        or value["protocol"] != protocol_document()
        or value["protocol_sha256"] != protocol_sha256()
    ):
        raise ValueError("v2 preregistration protocol identity mismatch")
    try:
        generator = value["generator"]
        configs = value["configs"]
        tests = value["tests"]
        if not isinstance(generator, Mapping) or set(generator) != {
            "schema_version",
            "spec_sha256",
        }:
            raise TypeError
        if not isinstance(configs, list) or not isinstance(tests, list):
            raise TypeError
        config_values = tuple(
            V2FileBinding(item["name"], item["path"], item["sha256"])
            for item in configs
            if isinstance(item, Mapping) and set(item) == {"name", "path", "sha256"}
        )
        test_values = tuple(
            V2TestEvidence(
                item["suite"],
                item["command"],
                item["evidence_sha256"],
                item["status"] == "passed",
            )
            for item in tests
            if isinstance(item, Mapping)
            and set(item) == {"suite", "command", "evidence_sha256", "status"}
        )
        if len(config_values) != len(configs) or len(test_values) != len(tests):
            raise TypeError
        return V2Preregistration(
            generator_schema_version=generator["schema_version"],
            generator_spec_sha256=generator["spec_sha256"],
            thresholds_sha256=value["thresholds_sha256"],
            code_sha256=value["code_sha256"],
            environment_sha256=value["environment_sha256"],
            configs=config_values,
            tests=test_values,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v2 preregistration has invalid scientific identity data") from exc


def _trimmed(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HASH_CHARS for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FrozenFileIdentity",
    "FrozenPreregistration",
    "V2FileBinding",
    "V2Preregistration",
    "V2TestEvidence",
    "canonical_preregistration_json",
    "load_v2_preregistration",
    "preregistration_sha256",
    "verify_v2_preregistration",
    "write_v2_preregistration",
]
