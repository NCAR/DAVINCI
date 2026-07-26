"""Versioned durable models for checkpoint and resume state."""

from __future__ import annotations

import os
import re
from datetime import datetime
from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ATTEMPT_SCHEMA_VERSION: Final[Literal["davinci-attempt-v1"]] = "davinci-attempt-v1"
EXECUTION_SCHEMA_VERSION: Final[Literal["davinci-execution-v1"]] = "davinci-execution-v1"
OBJECT_SCHEMA_VERSION: Final[Literal["davinci-checkpoint-object-v1"]] = (
    "davinci-checkpoint-object-v1"
)
CHECKPOINT_SCHEMA_VERSION: Final[Literal["davinci-checkpoint-v1"]] = "davinci-checkpoint-v1"
RESUME_PLAN_SCHEMA_VERSION: Final[Literal["davinci-resume-plan-v1"]] = "davinci-resume-plan-v1"

_ATTEMPT_ID_PATTERN = re.compile(r"^a\d{3,}$")
_EXECUTION_ID_PATTERN = re.compile(r"^e\d{3,}$")
_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AttemptStatus(str, Enum):
    """Lifecycle status of one logical attempt."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionStatus(str, Enum):
    """Lifecycle status of one process or scheduler execution."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    ABANDONED = "abandoned"


class CheckpointStatus(str, Enum):
    """Durable publication state of a checkpoint receipt."""

    FINALIZED = "finalized"
    FAILED = "failed"


class ResumeDisposition(str, Enum):
    """How one stage or item is handled in an execution."""

    COMPUTED = "computed"
    RESTORED = "restored"
    RECOMPUTED = "recomputed"
    SKIPPED = "skipped"
    FAILED = "failed"


class _DurableModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


def _safe_component(value: str) -> str:
    if (
        not _SAFE_COMPONENT_PATTERN.fullmatch(value)
        or value in {".", ".."}
        or os.path.basename(value) != value
    ):
        raise ValueError("value must be one safe path component")
    return value


def _sha256(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase SHA-256 digest")
    return value


def _normalized_path(value: str) -> str:
    normalized = os.path.normpath(value)
    if not normalized or normalized == ".":
        raise ValueError("path must be nonempty")
    return normalized


class AttemptRecord(_DurableModel):
    """Immutable identity and current lifecycle state for an attempt."""

    schema_version: Literal["davinci-attempt-v1"]
    run_id: str = Field(min_length=1)
    run_kind: Literal["production", "preflight", "smoke", "example"]
    attempt_id: str
    attempt_root: str
    status: AttemptStatus
    config_path: str | None = None
    identities: dict[str, str] = Field(min_length=1)
    runtime_versions: dict[str, str] = Field(min_length=1)
    git_commit: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    host: str = Field(min_length=1)

    @field_validator("attempt_id")
    @classmethod
    def _validate_attempt_id(cls, value: str) -> str:
        if not _ATTEMPT_ID_PATTERN.fullmatch(value):
            raise ValueError("attempt_id must use aNNN notation")
        return value

    @field_validator("attempt_root", "config_path")
    @classmethod
    def _normalize_paths(cls, value: str | None) -> str | None:
        return None if value is None else _normalized_path(value)

    @field_validator("identities")
    @classmethod
    def _validate_identities(cls, value: dict[str, str]) -> dict[str, str]:
        required = {
            "config_sha256",
            "code_sha256",
            "environment_sha256",
            "source_inventory_sha256",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError("attempt identities missing: " + ", ".join(missing))
        return {str(name): _sha256(str(digest)) for name, digest in value.items()}

    @field_validator("runtime_versions")
    @classmethod
    def _validate_runtime_versions(cls, value: dict[str, str]) -> dict[str, str]:
        if "python" not in value:
            raise ValueError("runtime_versions must include python")
        return {str(name): str(version) for name, version in value.items()}

    @field_validator("git_commit")
    @classmethod
    def _validate_git_commit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("git_commit must be a Git object digest")
        return normalized

    @model_validator(mode="after")
    def _validate_completion_time(self) -> "AttemptRecord":
        if self.status is AttemptStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed attempts require completed_at")
        if self.status is AttemptStatus.IN_PROGRESS and self.completed_at is not None:
            raise ValueError("in-progress attempts must not set completed_at")
        return self


class ExecutionRecord(_DurableModel):
    """One local process or scheduler invocation within an attempt."""

    schema_version: Literal["davinci-execution-v1"]
    execution_id: str
    attempt_id: str
    status: ExecutionStatus
    started_at: datetime
    ended_at: datetime | None = None
    host: str = Field(min_length=1)
    pid: int = Field(ge=1)
    scheduler_job_id: str | None = None
    error: str | None = None

    @field_validator("execution_id")
    @classmethod
    def _validate_execution_id(cls, value: str) -> str:
        if not _EXECUTION_ID_PATTERN.fullmatch(value):
            raise ValueError("execution_id must use eNNN notation")
        return value

    @field_validator("attempt_id")
    @classmethod
    def _validate_attempt_id(cls, value: str) -> str:
        if not _ATTEMPT_ID_PATTERN.fullmatch(value):
            raise ValueError("attempt_id must use aNNN notation")
        return value

    @model_validator(mode="after")
    def _validate_end_time(self) -> "ExecutionRecord":
        terminal = self.status is not ExecutionStatus.RUNNING
        if terminal and self.ended_at is None:
            raise ValueError("terminal executions require ended_at")
        if not terminal and self.ended_at is not None:
            raise ValueError("running executions must not set ended_at")
        return self


class CheckpointDependency(_DurableModel):
    """One finalized upstream receipt consumed by an item."""

    stage: str
    item: str | None = None
    receipt_sha256: str

    @field_validator("stage")
    @classmethod
    def _validate_component(cls, value: str) -> str:
        return _safe_component(value)

    @field_validator("item")
    @classmethod
    def _validate_optional_component(cls, value: str | None) -> str | None:
        return None if value is None else _safe_component(value)

    @field_validator("receipt_sha256")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        return _sha256(value)


class CheckpointObject(_DurableModel):
    """One durable data or file object referenced by a receipt."""

    schema_version: Literal["davinci-checkpoint-object-v1"]
    object_id: str
    kind: Literal["dataset", "json", "files"]
    paths: tuple[str, ...] = Field(min_length=1)
    checksums: dict[str, str] = Field(min_length=1)
    size_bytes: int = Field(ge=1)

    @field_validator("object_id")
    @classmethod
    def _validate_object_id(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("paths")
    @classmethod
    def _normalize_object_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_normalized_path(path) for path in value)

    @field_validator("checksums")
    @classmethod
    def _validate_checksums(cls, value: dict[str, str]) -> dict[str, str]:
        return {str(name): _sha256(str(digest)) for name, digest in value.items()}


class CheckpointReceipt(_DurableModel):
    """Finalized stage or stage-item checkpoint receipt."""

    schema_version: Literal["davinci-checkpoint-v1"]
    generation: int = Field(default=1, ge=1)
    stage: str
    item: str | None = None
    status: CheckpointStatus
    identity_sha256: str
    dependencies: tuple[CheckpointDependency, ...] = ()
    objects: tuple[CheckpointObject, ...] = ()
    context_delta: dict[str, Any] = Field(default_factory=dict)
    disposition: ResumeDisposition
    execution_id: str
    finalized_at: datetime

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: str) -> str:
        return _safe_component(value)

    @field_validator("item")
    @classmethod
    def _validate_item(cls, value: str | None) -> str | None:
        return None if value is None else _safe_component(value)

    @field_validator("identity_sha256")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("execution_id")
    @classmethod
    def _validate_execution_id(cls, value: str) -> str:
        if not _EXECUTION_ID_PATTERN.fullmatch(value):
            raise ValueError("execution_id must use eNNN notation")
        return value

    @model_validator(mode="after")
    def _validate_reconstructable_state(self) -> "CheckpointReceipt":
        if (
            self.status is CheckpointStatus.FINALIZED
            and self.disposition is not ResumeDisposition.SKIPPED
            and not self.objects
            and not self.context_delta
        ):
            raise ValueError("finalized receipt must contain reconstructable state")
        return self


class ResumePlanItem(_DurableModel):
    """One deterministic item decision in a resume plan."""

    stage: str
    item: str | None = None
    disposition: ResumeDisposition
    reason: str = Field(min_length=1)
    identity_sha256: str
    invalidated_by: tuple[str, ...] = ()

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: str) -> str:
        return _safe_component(value)

    @field_validator("item")
    @classmethod
    def _validate_item(cls, value: str | None) -> str | None:
        return None if value is None else _safe_component(value)

    @field_validator("identity_sha256")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        return _sha256(value)


class ResumePlan(_DurableModel):
    """Read-only machine-readable plan for one resume invocation."""

    schema_version: Literal["davinci-resume-plan-v1"]
    attempt_id: str
    blocked: bool
    blocked_reasons: tuple[str, ...]
    items: tuple[ResumePlanItem, ...]

    @field_validator("attempt_id")
    @classmethod
    def _validate_attempt_id(cls, value: str) -> str:
        if not _ATTEMPT_ID_PATTERN.fullmatch(value):
            raise ValueError("attempt_id must use aNNN notation")
        return value

    @model_validator(mode="after")
    def _validate_blocked_reasons(self) -> "ResumePlan":
        if self.blocked != bool(self.blocked_reasons):
            raise ValueError("blocked and blocked_reasons must agree")
        return self
