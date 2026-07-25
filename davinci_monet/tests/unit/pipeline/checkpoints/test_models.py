"""Durable checkpoint record model contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from davinci_monet.pipeline.checkpoints.models import (
    ATTEMPT_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    EXECUTION_SCHEMA_VERSION,
    OBJECT_SCHEMA_VERSION,
    RESUME_PLAN_SCHEMA_VERSION,
    AttemptRecord,
    AttemptStatus,
    CheckpointDependency,
    CheckpointObject,
    CheckpointReceipt,
    CheckpointStatus,
    ExecutionRecord,
    ExecutionStatus,
    ResumeDisposition,
    ResumePlan,
    ResumePlanItem,
)

NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


def _object() -> CheckpointObject:
    return CheckpointObject(
        schema_version=OBJECT_SCHEMA_VERSION,
        object_id="a" * 64,
        kind="dataset",
        paths=("/run/objects/sha256/aa/chunk-00000.nc",),
        checksums={"chunk-00000.nc": "b" * 64},
        size_bytes=128,
    )


def test_attempt_record_json_round_trip() -> None:
    record = AttemptRecord(
        schema_version=ATTEMPT_SCHEMA_VERSION,
        run_id="aod-eof-wavelet-r01",
        run_kind="production",
        attempt_id="a001",
        attempt_root="/run/a001",
        status=AttemptStatus.IN_PROGRESS,
        config_path="/repo/control.yaml",
        identities={
            "config_sha256": "a" * 64,
            "code_sha256": "b" * 64,
            "environment_sha256": "d" * 64,
            "source_inventory_sha256": "c" * 64,
        },
        runtime_versions={"python": "3.13.14", "numpy": "2.3.0"},
        git_commit="e" * 40,
        created_at=NOW,
        host="derecho1",
    )

    restored = AttemptRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.attempt_root == "/run/a001"
    assert restored.runtime_versions["python"] == "3.13.14"


def test_execution_record_requires_terminal_end_time() -> None:
    with pytest.raises(ValidationError, match="ended_at"):
        ExecutionRecord(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id="e001",
            attempt_id="a001",
            status=ExecutionStatus.COMPLETED,
            started_at=NOW,
            host="derecho1",
            pid=123,
        )


@pytest.mark.parametrize("execution_id", ["1", "e1", "j001", "e00x"])
def test_execution_id_is_monotonic_execution_component(execution_id: str) -> None:
    with pytest.raises(ValidationError, match="execution_id"):
        ExecutionRecord(
            schema_version=EXECUTION_SCHEMA_VERSION,
            execution_id=execution_id,
            attempt_id="a001",
            status=ExecutionStatus.RUNNING,
            started_at=NOW,
            host="derecho1",
            pid=123,
        )


@pytest.mark.parametrize("label", ["../pair", "a/b", ".", "", " pair"])
def test_checkpoint_labels_are_safe_path_components(label: str) -> None:
    with pytest.raises(ValidationError, match="safe path component"):
        CheckpointDependency(
            stage=label,
            item="source",
            receipt_sha256="d" * 64,
        )


def test_checkpoint_object_requires_paths_and_checksums() -> None:
    with pytest.raises(ValidationError, match="paths"):
        CheckpointObject(
            schema_version=OBJECT_SCHEMA_VERSION,
            object_id="a" * 64,
            kind="dataset",
            paths=(),
            checksums={},
            size_bytes=0,
        )


def test_finalized_receipt_requires_reconstructable_state() -> None:
    with pytest.raises(ValidationError, match="reconstructable"):
        CheckpointReceipt(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            stage="analyses",
            item="basis",
            status=CheckpointStatus.FINALIZED,
            identity_sha256="a" * 64,
            dependencies=(),
            objects=(),
            context_delta={},
            disposition=ResumeDisposition.COMPUTED,
            execution_id="e001",
            finalized_at=NOW,
        )


def test_checkpoint_receipt_round_trip_preserves_lineage() -> None:
    receipt = CheckpointReceipt(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        stage="analyses",
        item="basis",
        status=CheckpointStatus.FINALIZED,
        identity_sha256="a" * 64,
        dependencies=(
            CheckpointDependency(
                stage="load_sources",
                item="model",
                receipt_sha256="c" * 64,
            ),
        ),
        objects=(_object(),),
        context_delta={"analysis_status": "completed"},
        disposition=ResumeDisposition.COMPUTED,
        execution_id="e001",
        finalized_at=NOW,
    )

    restored = CheckpointReceipt.model_validate_json(receipt.model_dump_json())

    assert restored == receipt
    assert restored.dependencies[0].stage == "load_sources"


def test_resume_plan_is_strict_and_machine_readable() -> None:
    plan = ResumePlan(
        schema_version=RESUME_PLAN_SCHEMA_VERSION,
        attempt_id="a001",
        blocked=False,
        blocked_reasons=(),
        items=(
            ResumePlanItem(
                stage="load_sources",
                item="model",
                disposition=ResumeDisposition.RESTORED,
                reason="valid_receipt",
                identity_sha256="a" * 64,
                invalidated_by=(),
            ),
        ),
    )

    payload = plan.model_dump(mode="json")

    assert payload["items"][0]["disposition"] == "restored"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ResumePlan.model_validate({**payload, "legacy_resume": True})
