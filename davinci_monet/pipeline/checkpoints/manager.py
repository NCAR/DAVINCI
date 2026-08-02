"""Attempt lifecycle, checkpoint lookup, and deterministic resume planning."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

from davinci_monet.config.schema import MonetConfig
from davinci_monet.core.identity import (
    canonical_sha256,
    canonicalize,
    code_tree_sha256,
    compose_checkpoint_identity,
    configuration_sha256,
    git_commit,
    inventory_sources,
    runtime_versions,
    source_inventory_sha256,
)
from davinci_monet.pipeline.checkpoints.codecs import CheckpointCodecs
from davinci_monet.pipeline.checkpoints.models import (
    ATTEMPT_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    EXECUTION_SCHEMA_VERSION,
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
from davinci_monet.pipeline.checkpoints.store import AttemptStore, AttemptStoreError

if TYPE_CHECKING:
    from davinci_monet.pipeline.stages.base import PipelineContext, SourceData, StageResult

REASON_VALID = "valid_receipt"
REASON_MISSING = "missing_receipt"
REASON_IDENTITY = "identity_changed"
REASON_CORRUPT = "object_invalid"
REASON_DEPENDENCY = "dependency_invalid"
REASON_RESTART = "operator_restart"
REASON_PRIOR_ATTEMPT = "pinned_prior_attempt"


@dataclass(frozen=True)
class CheckpointRequest:
    """Inputs required to plan or resolve one stage/item checkpoint."""

    stage: str
    item: str | None
    config: Any
    dependencies: tuple[tuple[str, str | None], ...] = ()
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckpointLookup:
    """One checkpoint lookup decision and its expected identity."""

    receipt: CheckpointReceipt | None
    disposition: ResumeDisposition
    reason: str
    identity_sha256: str
    dependencies: tuple[CheckpointDependency, ...]
    invalidated_by: tuple[str, ...] = ()


def _now() -> datetime:
    return datetime.now(UTC)


def _checkpoint_key(stage: str, item: str | None) -> str:
    return stage if item is None else f"{stage}:{item}"


def _extract_existing_source_paths(config: MonetConfig) -> tuple[str, ...]:
    """Find configured local source files without opening readers."""

    paths: set[str] = set()

    def visit(value: Any) -> None:
        if hasattr(value, "model_dump"):
            visit(value.model_dump(mode="python", exclude_none=True))
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)
        elif isinstance(value, (str, Path)):
            raw = str(value)
            if any(token in raw for token in ("*", "?", "[")):
                import glob

                for match in glob.glob(raw):
                    path = Path(match).expanduser()
                    if path.is_file():
                        paths.add(str(path.resolve()))
                return
            path = Path(raw).expanduser()
            if path.is_file():
                paths.add(str(path.resolve()))

    visit(config.sources)
    return tuple(sorted(paths))


def _json_safe(value: Any) -> Any:
    normalized = canonicalize(value)
    return json.loads(json.dumps(normalized, sort_keys=True, default=str))


def item_checkpoint_manager(context: Any) -> "CheckpointManager | None":
    """Return the manager only when this attempt enables item granularity."""
    manager = getattr(context, "checkpoint_manager", None)
    if manager is None or manager.config.execution is None:
        return None
    if manager.config.execution.checkpoints.granularity != "item":
        return None
    return manager


class CheckpointManager:
    """Coordinate one exact-identity DAVINCI attempt."""

    def __init__(
        self,
        *,
        config: MonetConfig,
        config_path: str | Path | None,
        store: AttemptStore,
        identities: Mapping[str, str],
        source_inventory: tuple[dict[str, Any], ...],
        resume: bool,
        read_only: bool = False,
        restart_from: str | None = None,
        blocked_reasons: Sequence[str] = (),
        restore_store: AttemptStore | None = None,
        restore_receipt: CheckpointReceipt | None = None,
        restore_codecs: CheckpointCodecs | None = None,
    ) -> None:
        self.config = config
        self.config_path = None if config_path is None else Path(config_path).resolve()
        self.store = store
        self.identities = dict(identities)
        self.source_inventory = source_inventory
        self.resume = resume
        self.read_only = read_only
        self.restart_from = self._parse_restart_from(restart_from)
        self.blocked_reasons = tuple(blocked_reasons)
        self.restore_store = restore_store
        self.restore_receipt = restore_receipt
        self.restore_codecs = restore_codecs
        self.code_sha256 = self.identities["code_sha256"]
        self.codecs = CheckpointCodecs(store.root)
        self.execution_id: str | None = None
        self._lock: AbstractContextManager[None] | None = None
        self._recorded_restorations: set[str] = set()
        self._pending_decisions: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._stage_order: tuple[str, ...] = ()
        self._restored_boundary_result: StageResult | None = None

    @classmethod
    def create(
        cls,
        config: MonetConfig,
        *,
        config_path: str | Path | None = None,
        resume: bool = False,
        read_only: bool = False,
        restart_from: str | None = None,
    ) -> "CheckpointManager | None":
        execution = config.execution
        if execution is None or execution.checkpoints.mode == "off":
            if resume or read_only or restart_from is not None:
                raise AttemptStoreError("resume requires enabled execution checkpoints")
            return None
        if restart_from is not None and not resume:
            raise AttemptStoreError("restart-from requires resume mode")
        if read_only and not resume:
            raise AttemptStoreError("resume planning requires resume mode")
        if config.run is None:
            raise AttemptStoreError("checkpointed execution requires run identity")

        attempt_root = execution.attempt_root.expanduser().resolve()
        restore_store: AttemptStore | None = None
        restore_receipt: CheckpointReceipt | None = None
        restore_codecs: CheckpointCodecs | None = None
        restore_config = execution.checkpoints.restore_from
        if restore_config is not None:
            source_root = restore_config.source_attempt_root.expanduser().resolve()
            if source_root == attempt_root:
                raise AttemptStoreError("checkpoint restore source must be a different attempt")
            restore_store = AttemptStore(source_root)
            source_attempt = restore_store.read_attempt()
            if source_attempt.status is AttemptStatus.IN_PROGRESS:
                raise AttemptStoreError("checkpoint restore source attempt must be terminal")
            restore_receipt = restore_store.read_receipt(
                restore_config.through_stage,
                None,
            )
            if restore_receipt is None:
                raise AttemptStoreError(
                    "checkpoint restore receipt does not exist: " f"{restore_config.through_stage}"
                )
            actual_receipt_sha256 = canonical_sha256(restore_receipt)
            if actual_receipt_sha256 != restore_config.receipt_sha256:
                raise AttemptStoreError("checkpoint restore receipt SHA-256 mismatch")
            restore_codecs = CheckpointCodecs(source_root)
            cls._validate_restore_receipt_chain(
                restore_store,
                restore_codecs,
                restore_receipt,
            )
            result_payload = restore_receipt.context_delta.get("result")
            if (
                not isinstance(result_payload, Mapping)
                or result_payload.get("status") != "COMPLETED"
            ):
                raise AttemptStoreError(
                    "checkpoint restore boundary must contain a completed stage result"
                )
        source_paths = _extract_existing_source_paths(config)
        inventory = inventory_sources(source_paths)
        package_root = Path(__file__).resolve().parents[2]
        versions = runtime_versions()
        identities = {
            "config_sha256": configuration_sha256(config),
            "code_sha256": code_tree_sha256(package_root, use_cache=False),
            "environment_sha256": canonical_sha256(versions),
            "source_inventory_sha256": source_inventory_sha256(inventory),
        }
        if resume:
            store = AttemptStore.open_for_resume(
                attempt_root,
                expected_identities=None if read_only else identities,
            )
            stored_identities = store.read_attempt().identities
            blocked_reasons = tuple(
                f"attempt_identity_mismatch:{name}"
                for name in sorted(set(stored_identities) | set(identities))
                if stored_identities.get(name) != identities.get(name)
            )
        else:
            blocked_reasons = ()
            record = AttemptRecord(
                schema_version=ATTEMPT_SCHEMA_VERSION,
                run_id=config.run.id,
                run_kind=config.run.kind,
                attempt_id=attempt_root.name,
                attempt_root=str(attempt_root),
                status=AttemptStatus.IN_PROGRESS,
                config_path=None if config_path is None else str(Path(config_path).resolve()),
                identities=identities,
                runtime_versions=versions,
                git_commit=git_commit(package_root),
                created_at=_now(),
                host=socket.gethostname(),
            )
            store = AttemptStore.initialize(attempt_root, record)
        return cls(
            config=config,
            config_path=config_path,
            store=store,
            identities=identities,
            source_inventory=inventory,
            resume=resume,
            read_only=read_only,
            restart_from=restart_from,
            blocked_reasons=blocked_reasons,
            restore_store=restore_store,
            restore_receipt=restore_receipt,
            restore_codecs=restore_codecs,
        )

    @classmethod
    def _validate_restore_receipt_chain(
        cls,
        store: AttemptStore,
        codecs: CheckpointCodecs,
        receipt: CheckpointReceipt,
        seen: set[tuple[str, str | None]] | None = None,
        validated_objects: set[str] | None = None,
    ) -> None:
        """Validate a pinned boundary and all receipts in its dependency chain."""
        if receipt.status is not CheckpointStatus.FINALIZED:
            raise AttemptStoreError("checkpoint restore receipt is not finalized")
        checked_objects = validated_objects if validated_objects is not None else set()
        for obj in receipt.objects:
            if obj.object_id in checked_objects:
                continue
            if not codecs.validate_object(obj):
                raise AttemptStoreError("checkpoint restore contains an invalid object")
            checked_objects.add(obj.object_id)
        visited = seen if seen is not None else set()
        key = (receipt.stage, receipt.item)
        if key in visited:
            return
        visited.add(key)
        for dependency in receipt.dependencies:
            upstream = store.read_receipt(dependency.stage, dependency.item)
            if upstream is None:
                raise AttemptStoreError(
                    "checkpoint restore dependency is missing: "
                    f"{_checkpoint_key(dependency.stage, dependency.item)}"
                )
            if cls.receipt_sha256(upstream) != dependency.receipt_sha256:
                raise AttemptStoreError(
                    "checkpoint restore dependency SHA-256 mismatch: "
                    f"{_checkpoint_key(dependency.stage, dependency.item)}"
                )
            cls._validate_restore_receipt_chain(
                store,
                codecs,
                upstream,
                visited,
                checked_objects,
            )

    @staticmethod
    def _parse_restart_from(value: str | None) -> tuple[str, str | None] | None:
        if value is None:
            return None
        stage, separator, item = value.partition(":")
        if not stage or (separator and not item) or ":" in item:
            raise AttemptStoreError("restart-from must use STAGE or STAGE:ITEM")
        return stage, item if separator else None

    @property
    def restore_through_stage(self) -> str | None:
        """Return the configured prior-attempt boundary stage, if any."""
        if self.config.execution is None:
            return None
        restore = self.config.execution.checkpoints.restore_from
        return None if restore is None else restore.through_stage

    def configure_stage_order(self, stage_names: Sequence[str]) -> None:
        """Validate and retain pipeline order for prior-attempt restoration."""
        self._stage_order = tuple(stage_names)
        through_stage = self.restore_through_stage
        if through_stage is not None and through_stage not in self._stage_order:
            raise AttemptStoreError(
                "checkpoint restore stage is not in this pipeline: " f"{through_stage}"
            )

    def restore_action(self, stage: str) -> str | None:
        """Classify a stage relative to a configured restored boundary."""
        through_stage = self.restore_through_stage
        if through_stage is None:
            return None
        if not self._stage_order:
            raise AttemptStoreError("checkpoint restore stage order is not configured")
        stage_index = self._stage_order.index(stage)
        boundary_index = self._stage_order.index(through_stage)
        if stage_index < boundary_index:
            return "skip"
        if stage_index == boundary_index:
            return "restore"
        return None

    def restore_boundary(
        self,
        request: CheckpointRequest,
        context: PipelineContext,
    ) -> StageResult:
        """Restore and adopt the pinned prior-attempt stage boundary."""
        if request.stage != self.restore_through_stage or request.item is not None:
            raise AttemptStoreError("checkpoint restore request is not the pinned boundary")
        if self._restored_boundary_result is not None:
            return self._restored_boundary_result

        if self.resume:
            lookup = self.lookup(request)
            if lookup.receipt is not None:
                result = self.restore_stage(lookup.receipt, context)
                result.metadata["checkpoint_reason"] = lookup.reason
                self._restored_boundary_result = result
                return result

        if (
            self.restore_store is None
            or self.restore_receipt is None
            or self.restore_codecs is None
        ):
            raise AttemptStoreError("checkpoint restore source is unavailable")

        result = self.restore_stage(self.restore_receipt, context)
        source_attempt = self.restore_store.read_attempt()
        lineage = {
            "source_attempt_root": str(self.restore_store.root),
            "source_run_id": source_attempt.run_id,
            "source_attempt_id": source_attempt.attempt_id,
            "source_attempt_status": source_attempt.status.value,
            "stage": self.restore_receipt.stage,
            "receipt_sha256": self.receipt_sha256(self.restore_receipt),
        }
        adopted_delta = {
            **dict(self.restore_receipt.context_delta),
            "upstream_checkpoint": lineage,
        }
        self.publish(
            request,
            objects=self.restore_receipt.objects,
            context_delta=adopted_delta,
            disposition=ResumeDisposition.RESTORED,
        )
        context.metadata["upstream_checkpoint"] = lineage
        result.duration_seconds = 0.0
        result.metadata.update(
            {
                "resume_disposition": ResumeDisposition.RESTORED.value,
                "checkpoint_reason": REASON_PRIOR_ATTEMPT,
                "upstream_checkpoint": lineage,
            }
        )
        self.store.append_event(
            {
                "event": "checkpoint_adopted",
                "execution_id": self.execution_id,
                "stage": request.stage,
                "item": None,
                "disposition": ResumeDisposition.RESTORED.value,
                "reason": REASON_PRIOR_ATTEMPT,
                "upstream_checkpoint": lineage,
                "at": _now().isoformat(),
            }
        )
        self._restored_boundary_result = result
        return result

    @property
    def attempt(self) -> AttemptRecord:
        return self.store.read_attempt()

    def begin_execution(self) -> str:
        if self.read_only:
            raise AttemptStoreError("read-only resume planning cannot start execution")
        self._lock = self.store.lock()
        self._lock.__enter__()
        try:
            for previous in self.store.list_executions():
                if previous.status is ExecutionStatus.RUNNING:
                    abandoned = previous.model_copy(
                        update={
                            "status": ExecutionStatus.ABANDONED,
                            "ended_at": _now(),
                            "error": "prior process ended without a terminal execution record",
                        }
                    )
                    self.store.publish_execution(abandoned)
            self.execution_id = self.store.next_execution_id()
            record = ExecutionRecord(
                schema_version=EXECUTION_SCHEMA_VERSION,
                execution_id=self.execution_id,
                attempt_id=self.attempt.attempt_id,
                status=ExecutionStatus.RUNNING,
                started_at=_now(),
                host=socket.gethostname(),
                pid=os.getpid(),
                scheduler_job_id=os.environ.get("PBS_JOBID"),
            )
            self.store.publish_execution(record)
            if self.attempt.status is not AttemptStatus.IN_PROGRESS:
                self.store.update_attempt(
                    self.attempt.model_copy(
                        update={"status": AttemptStatus.IN_PROGRESS, "completed_at": None}
                    )
                )
            self.store.append_event(
                {
                    "event": "execution_started",
                    "execution_id": self.execution_id,
                    "at": _now().isoformat(),
                }
            )
            return self.execution_id
        except BaseException:
            self._release_lock()
            raise

    def finish_execution(
        self,
        status: ExecutionStatus,
        *,
        error: str | None = None,
        finalize_attempt: bool = True,
        release_lock: bool = True,
    ) -> None:
        if self.execution_id is None:
            return
        try:
            started = next(
                record
                for record in self.store.list_executions()
                if record.execution_id == self.execution_id
            )
            terminal = started.model_copy(
                update={
                    "status": status,
                    "ended_at": _now(),
                    "error": error,
                }
            )
            self.store.publish_execution(terminal)
            if finalize_attempt:
                self.finalize_attempt(status)
            self.store.append_event(
                {
                    "event": "execution_finished",
                    "execution_id": self.execution_id,
                    "status": status.value,
                    "at": _now().isoformat(),
                    "error": error,
                }
            )
        finally:
            if release_lock:
                self._release_lock()

    def finalize_attempt(self, status: ExecutionStatus) -> None:
        """Close attempt lifecycle only after terminal output is durable."""
        attempt_status = (
            AttemptStatus.COMPLETED if status is ExecutionStatus.COMPLETED else AttemptStatus.FAILED
        )
        self.store.update_attempt(
            self.attempt.model_copy(
                update={
                    "status": attempt_status,
                    "completed_at": (_now() if attempt_status is AttemptStatus.COMPLETED else None),
                }
            )
        )

    def release_execution_lock(self) -> None:
        """Release the attempt writer lock after two-phase closeout."""
        self._release_lock()

    def _release_lock(self) -> None:
        if self._lock is not None:
            self._lock.__exit__(None, None, None)
            self._lock = None

    @staticmethod
    def receipt_sha256(receipt: CheckpointReceipt) -> str:
        return canonical_sha256(receipt)

    def _dependencies(
        self,
        keys: Iterable[tuple[str, str | None]],
    ) -> tuple[tuple[CheckpointDependency, ...], tuple[str, ...]]:
        dependencies: list[CheckpointDependency] = []
        missing: list[str] = []
        for stage, item in keys:
            receipt = self.store.read_receipt(stage, item)
            key = _checkpoint_key(stage, item)
            if (
                receipt is None
                or receipt.status is not CheckpointStatus.FINALIZED
                or any(not self.codecs.validate_object(obj) for obj in receipt.objects)
            ):
                missing.append(key)
                continue
            dependencies.append(
                CheckpointDependency(
                    stage=stage,
                    item=item,
                    receipt_sha256=self.receipt_sha256(receipt),
                )
            )
        return tuple(dependencies), tuple(missing)

    def identity(
        self,
        request: CheckpointRequest,
    ) -> tuple[str, tuple[CheckpointDependency, ...], tuple[str, ...]]:
        dependencies, missing = self._dependencies(request.dependencies)
        inventory = (
            inventory_sources(request.source_paths)
            if request.source_paths
            else self.source_inventory
        )
        parts = compose_checkpoint_identity(
            stage=request.stage,
            item=request.item,
            config=request.config,
            dependencies=dependencies,
            source_inventory=inventory,
            code_sha256=self.code_sha256,
        )
        return parts["identity_sha256"], dependencies, missing

    def _forced_restart(self, stage: str, item: str | None) -> bool:
        if self.restart_from is None:
            return False
        target_stage, target_item = self.restart_from
        if stage != target_stage:
            return False
        if target_item is None:
            return True
        return item is None or item == target_item

    def lookup(
        self, request: CheckpointRequest, *, mutate_restart: bool = True
    ) -> CheckpointLookup:
        if self.blocked_reasons:
            raise AttemptStoreError("resume is blocked: " + ", ".join(self.blocked_reasons))
        identity, dependencies, missing_dependencies = self.identity(request)
        forced = (
            self._forced_restart(request.stage, request.item)
            if mutate_restart
            else self._restart_matches(request.stage, request.item)
        )
        if forced:
            return self._remember_lookup(
                request,
                CheckpointLookup(
                    None,
                    ResumeDisposition.RECOMPUTED,
                    REASON_RESTART,
                    identity,
                    dependencies,
                ),
            )
        if missing_dependencies:
            return self._remember_lookup(
                request,
                CheckpointLookup(
                    None,
                    ResumeDisposition.RECOMPUTED,
                    REASON_DEPENDENCY,
                    identity,
                    dependencies,
                    missing_dependencies,
                ),
            )
        receipt = self.store.read_receipt(request.stage, request.item)
        if receipt is None:
            return self._remember_lookup(
                request,
                CheckpointLookup(
                    None,
                    ResumeDisposition.COMPUTED,
                    REASON_MISSING,
                    identity,
                    dependencies,
                ),
            )
        if receipt.identity_sha256 != identity:
            return self._remember_lookup(
                request,
                CheckpointLookup(
                    None,
                    ResumeDisposition.RECOMPUTED,
                    REASON_IDENTITY,
                    identity,
                    dependencies,
                    (_checkpoint_key(request.stage, request.item),),
                ),
            )
        if any(not self.codecs.validate_object(obj) for obj in receipt.objects):
            return self._remember_lookup(
                request,
                CheckpointLookup(
                    None,
                    ResumeDisposition.RECOMPUTED,
                    REASON_CORRUPT,
                    identity,
                    dependencies,
                    (_checkpoint_key(request.stage, request.item),),
                ),
            )
        lookup = CheckpointLookup(
            receipt,
            ResumeDisposition.RESTORED,
            REASON_VALID,
            identity,
            dependencies,
        )
        return self._remember_lookup(request, lookup)

    def _remember_lookup(
        self,
        request: CheckpointRequest,
        lookup: CheckpointLookup,
    ) -> CheckpointLookup:
        if self.read_only or self.execution_id is None:
            return lookup
        if lookup.receipt is not None:
            self._record_restoration(lookup.receipt)
        else:
            self._pending_decisions[_checkpoint_key(request.stage, request.item)] = (
                lookup.reason,
                lookup.invalidated_by,
            )
        return lookup

    def _record_restoration(self, receipt: CheckpointReceipt) -> None:
        """Record one execution-scoped restore decision without changing receipts."""
        if self.read_only or self.execution_id is None:
            return
        restored = [receipt]
        if receipt.item is None:
            for dependency in receipt.dependencies:
                if dependency.stage != receipt.stage or dependency.item is None:
                    continue
                item_receipt = self.store.read_receipt(
                    dependency.stage,
                    dependency.item,
                )
                if item_receipt is not None:
                    restored.append(item_receipt)
        for value in restored:
            key = _checkpoint_key(value.stage, value.item)
            if key in self._recorded_restorations:
                continue
            self.store.append_event(
                {
                    "event": "checkpoint_restored",
                    "execution_id": self.execution_id,
                    "stage": value.stage,
                    "item": value.item,
                    "generation": value.generation,
                    "disposition": ResumeDisposition.RESTORED.value,
                    "reason": REASON_VALID,
                    "invalidated_by": [],
                    "at": _now().isoformat(),
                }
            )
            self._recorded_restorations.add(key)

    def record_failed_stage(self, stage: str, error: str | None) -> None:
        """Record a failed stage decision for the current execution manifest."""
        if self.read_only or self.execution_id is None:
            return
        self.store.append_event(
            {
                "event": "checkpoint_decision",
                "execution_id": self.execution_id,
                "stage": stage,
                "item": None,
                "disposition": ResumeDisposition.FAILED.value,
                "reason": error or "stage_failed",
                "at": _now().isoformat(),
            }
        )

    def _restart_matches(self, stage: str, item: str | None) -> bool:
        return self._forced_restart(stage, item)

    def publish(
        self,
        request: CheckpointRequest,
        *,
        objects: Sequence[CheckpointObject] = (),
        context_delta: Mapping[str, Any] | None = None,
        disposition: ResumeDisposition | None = None,
    ) -> CheckpointReceipt:
        if self.read_only or self.execution_id is None:
            raise AttemptStoreError("checkpoint publication requires an active execution")
        identity, dependencies, _ = self.identity(request)
        existing = self.store.read_receipt(request.stage, request.item)
        resolved_disposition = disposition or (
            ResumeDisposition.RECOMPUTED if existing is not None else ResumeDisposition.COMPUTED
        )
        receipt = CheckpointReceipt(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            stage=request.stage,
            item=request.item,
            status=CheckpointStatus.FINALIZED,
            identity_sha256=identity,
            dependencies=dependencies,
            objects=tuple(objects),
            context_delta=dict(context_delta or {}),
            disposition=resolved_disposition,
            execution_id=self.execution_id,
            finalized_at=_now(),
        )
        path = self.store.publish_receipt(receipt)
        finalized = CheckpointReceipt.model_validate_json(path.read_text("utf-8"))
        reason, invalidated_by = self._pending_decisions.pop(
            _checkpoint_key(request.stage, request.item),
            ("checkpoint_published", ()),
        )
        self.store.append_event(
            {
                "event": "checkpoint_finalized",
                "execution_id": self.execution_id,
                "stage": request.stage,
                "item": request.item,
                "generation": finalized.generation,
                "disposition": finalized.disposition.value,
                "reason": reason,
                "invalidated_by": list(invalidated_by),
                "at": finalized.finalized_at.isoformat(),
            }
        )
        return finalized

    def capture_stage(
        self,
        request: CheckpointRequest,
        context: PipelineContext,
        result: StageResult,
        *,
        disposition: ResumeDisposition | None = None,
    ) -> CheckpointReceipt:
        from davinci_monet.pipeline.stages.base import SourceData

        objects: list[CheckpointObject] = []
        source_entries: list[dict[str, Any]] = []
        paired_entries: list[dict[str, Any]] = []
        for label, source in context.sources.items():
            dataset = source.data if hasattr(source, "data") else source
            if not isinstance(dataset, xr.Dataset):
                continue
            obj = (
                self.codecs.reference_finalized_dataset(dataset, source.config)
                if isinstance(source, SourceData)
                else None
            ) or self.codecs.write_dataset(dataset)
            objects.append(obj)
            metadata = (
                self.codecs.source_metadata(source)
                if isinstance(source, SourceData)
                else {"label": label, "raw_dataset": True}
            )
            source_entries.append(
                {
                    "label": label,
                    "object_id": obj.object_id,
                    "metadata": metadata,
                    "dataset_metadata": self.codecs.dataset_metadata(dataset),
                }
            )
        for label, paired in context.paired.items():
            dataset = paired.data if hasattr(paired, "data") else paired
            if not isinstance(dataset, xr.Dataset):
                continue
            obj = self.codecs.write_dataset(dataset)
            objects.append(obj)
            metadata = (
                self.codecs.paired_metadata(paired)
                if hasattr(paired, "x_source")
                else {"label": label, "raw_dataset": True}
            )
            paired_entries.append(
                {"label": label, "object_id": obj.object_id, "metadata": metadata}
            )
        result_files: set[Path] = set()

        def collect_files(value: Any) -> None:
            if isinstance(value, Mapping):
                for item in value.values():
                    collect_files(item)
            elif isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    collect_files(item)
            elif isinstance(value, (str, Path)):
                path = Path(value).expanduser()
                if path.is_file():
                    result_files.add(path.resolve())

        collect_files(result.data)
        if result_files:
            objects.append(self.codecs.capture_files(sorted(result_files, key=str)))
        result_payload = {
            "stage_name": result.stage_name,
            "status": result.status.name,
            "data": _json_safe(result.data),
            "metadata": _json_safe(result.metadata),
            "error": result.error,
            "error_type": result.error_type,
            "traceback_str": result.traceback_str,
            "duration_seconds": result.duration_seconds,
        }
        delta = {
            "sources": source_entries,
            "paired": paired_entries,
            "metadata": _json_safe(context.metadata),
            "result": result_payload,
        }
        return self.publish(
            request,
            objects=objects,
            context_delta=delta,
            disposition=disposition,
        )

    def capture_source(
        self,
        request: CheckpointRequest,
        source: SourceData,
        *,
        context_delta: Mapping[str, Any] | None = None,
    ) -> CheckpointReceipt:
        obj = self.codecs.reference_finalized_dataset(
            source.data, source.config
        ) or self.codecs.write_dataset(source.data)
        return self.publish(
            request,
            objects=(obj,),
            context_delta={
                "object_id": obj.object_id,
                "source": self.codecs.source_metadata(source),
                "dataset_metadata": self.codecs.dataset_metadata(source.data),
                **dict(context_delta or {}),
            },
        )

    def restore_source(self, receipt: CheckpointReceipt) -> SourceData:
        object_id = str(receipt.context_delta["object_id"])
        obj = next(obj for obj in receipt.objects if obj.object_id == object_id)
        dataset = self.codecs.read_dataset(
            obj,
            metadata=receipt.context_delta.get("dataset_metadata"),
        )
        return self.codecs.restore_source(dataset, receipt.context_delta["source"])

    def capture_paired(
        self,
        request: CheckpointRequest,
        paired: Any,
    ) -> CheckpointReceipt:
        obj = self.codecs.write_dataset(paired.data)
        return self.publish(
            request,
            objects=(obj,),
            context_delta={
                "object_id": obj.object_id,
                "paired": self.codecs.paired_metadata(paired),
            },
        )

    def restore_paired(self, receipt: CheckpointReceipt) -> Any:
        object_id = str(receipt.context_delta["object_id"])
        obj = next(obj for obj in receipt.objects if obj.object_id == object_id)
        dataset = self.codecs.read_dataset(obj)
        return self.codecs.restore_paired(dataset, receipt.context_delta["paired"])

    def capture_json(
        self,
        request: CheckpointRequest,
        value: Any,
        *,
        context_delta: Mapping[str, Any] | None = None,
    ) -> CheckpointReceipt:
        obj = self.codecs.write_json(value)
        return self.publish(
            request,
            objects=(obj,),
            context_delta={"object_id": obj.object_id, **dict(context_delta or {})},
        )

    def restore_json(self, receipt: CheckpointReceipt) -> Any:
        object_id = str(receipt.context_delta["object_id"])
        obj = next(obj for obj in receipt.objects if obj.object_id == object_id)
        return self.codecs.read_json(obj)

    def capture_files(
        self,
        request: CheckpointRequest,
        paths: Iterable[str | Path],
        *,
        context_delta: Mapping[str, Any] | None = None,
    ) -> CheckpointReceipt:
        obj = self.codecs.capture_files(paths)
        return self.publish(
            request,
            objects=(obj,),
            context_delta={"object_id": obj.object_id, **dict(context_delta or {})},
        )

    @staticmethod
    def restore_files(receipt: CheckpointReceipt) -> list[str]:
        object_id = str(receipt.context_delta["object_id"])
        obj = next(obj for obj in receipt.objects if obj.object_id == object_id)
        return list(obj.paths)

    def restore_stage(
        self,
        receipt: CheckpointReceipt,
        context: PipelineContext,
    ) -> StageResult:
        from davinci_monet.pipeline.stages.base import StageResult, StageStatus

        objects = {obj.object_id: obj for obj in receipt.objects}
        delta = receipt.context_delta
        sources: dict[str, Any] = {}
        for entry in delta.get("sources", []):
            obj = objects[str(entry["object_id"])]
            dataset = self.codecs.read_dataset(
                obj,
                metadata=entry.get("dataset_metadata"),
            )
            metadata = entry["metadata"]
            if metadata.get("raw_dataset"):
                sources[str(entry["label"])] = dataset
            else:
                sources[str(entry["label"])] = self.codecs.restore_source(dataset, metadata)
        paired: dict[str, Any] = {}
        for entry in delta.get("paired", []):
            obj = objects[str(entry["object_id"])]
            dataset = self.codecs.read_dataset(obj)
            metadata = entry["metadata"]
            if metadata.get("raw_dataset"):
                paired[str(entry["label"])] = dataset
            else:
                paired[str(entry["label"])] = self.codecs.restore_paired(dataset, metadata)
        context.sources = sources
        context.paired = paired
        metadata = delta.get("metadata")
        if isinstance(metadata, dict):
            context.metadata.update(metadata)
        payload = delta["result"]
        return StageResult(
            stage_name=str(payload["stage_name"]),
            status=StageStatus[str(payload["status"])],
            data=payload.get("data"),
            metadata={**dict(payload.get("metadata", {})), "resume_disposition": "restored"},
            error=payload.get("error"),
            error_type=payload.get("error_type"),
            traceback_str=payload.get("traceback_str"),
            duration_seconds=0.0,
        )

    def plan(self, requests: Sequence[CheckpointRequest]) -> ResumePlan:
        if self.blocked_reasons:
            return ResumePlan(
                schema_version=RESUME_PLAN_SCHEMA_VERSION,
                attempt_id=self.attempt.attempt_id,
                blocked=True,
                blocked_reasons=self.blocked_reasons,
                items=(),
            )
        items: list[ResumePlanItem] = []
        invalid: set[str] = set()
        for request in requests:
            key = _checkpoint_key(request.stage, request.item)
            dependency_keys = {_checkpoint_key(stage, item) for stage, item in request.dependencies}
            stage_invalid_dependencies = tuple(sorted(dependency_keys & invalid))
            if self._restart_matches(request.stage, request.item):
                identity, _, _ = self.identity(request)
                lookup = CheckpointLookup(
                    None,
                    ResumeDisposition.RECOMPUTED,
                    REASON_RESTART,
                    identity,
                    (),
                )
            elif stage_invalid_dependencies:
                identity, _, _ = self.identity(request)
                lookup = CheckpointLookup(
                    None,
                    ResumeDisposition.RECOMPUTED,
                    REASON_DEPENDENCY,
                    identity,
                    (),
                    stage_invalid_dependencies,
                )
            else:
                lookup = self.lookup(request, mutate_restart=False)
            if lookup.disposition is not ResumeDisposition.RESTORED:
                invalid.add(key)
            items.append(
                ResumePlanItem(
                    stage=request.stage,
                    item=request.item,
                    disposition=lookup.disposition,
                    reason=lookup.reason,
                    identity_sha256=lookup.identity_sha256,
                    invalidated_by=lookup.invalidated_by,
                )
            )
        return ResumePlan(
            schema_version=RESUME_PLAN_SCHEMA_VERSION,
            attempt_id=self.attempt.attempt_id,
            blocked=False,
            blocked_reasons=(),
            items=tuple(items),
        )

    def plan_attempt(self, stage_requests: Sequence[CheckpointRequest]) -> ResumePlan:
        """Inventory available item receipts and stage decisions without writes."""
        if self.blocked_reasons:
            return ResumePlan(
                schema_version=RESUME_PLAN_SCHEMA_VERSION,
                attempt_id=self.attempt.attempt_id,
                blocked=True,
                blocked_reasons=self.blocked_reasons,
                items=(),
            )

        receipts_by_stage: dict[str, list[CheckpointReceipt]] = {}
        for receipt in self.store.iter_receipts():
            if receipt.item is not None:
                receipts_by_stage.setdefault(receipt.stage, []).append(receipt)

        items: list[ResumePlanItem] = []
        invalid: set[str] = set()
        for request in stage_requests:
            restore_action = self.restore_action(request.stage)
            if restore_action == "skip":
                identity, _, _ = self.identity(request)
                items.append(
                    ResumePlanItem(
                        stage=request.stage,
                        item=request.item,
                        disposition=ResumeDisposition.RESTORED,
                        reason="pinned_prior_attempt_prefix",
                        identity_sha256=identity,
                    )
                )
                continue
            for receipt in sorted(
                receipts_by_stage.get(request.stage, []),
                key=lambda value: str(value.item),
            ):
                key = _checkpoint_key(receipt.stage, receipt.item)
                dependency_keys = {
                    _checkpoint_key(dependency.stage, dependency.item)
                    for dependency in receipt.dependencies
                }
                receipt_invalid_dependencies = set(dependency_keys & invalid)
                for dependency in receipt.dependencies:
                    current = self.store.read_receipt(dependency.stage, dependency.item)
                    if (
                        current is None
                        or self.receipt_sha256(current) != dependency.receipt_sha256
                        or any(not self.codecs.validate_object(obj) for obj in current.objects)
                    ):
                        receipt_invalid_dependencies.add(
                            _checkpoint_key(dependency.stage, dependency.item)
                        )

                if self._restart_matches(receipt.stage, receipt.item):
                    disposition = ResumeDisposition.RECOMPUTED
                    reason = REASON_RESTART
                    invalidated_by: tuple[str, ...] = ()
                elif receipt_invalid_dependencies:
                    disposition = ResumeDisposition.RECOMPUTED
                    reason = REASON_DEPENDENCY
                    invalidated_by = tuple(sorted(receipt_invalid_dependencies))
                elif receipt.status is not CheckpointStatus.FINALIZED or any(
                    not self.codecs.validate_object(obj) for obj in receipt.objects
                ):
                    disposition = ResumeDisposition.RECOMPUTED
                    reason = REASON_CORRUPT
                    invalidated_by = (key,)
                else:
                    disposition = ResumeDisposition.RESTORED
                    reason = REASON_VALID
                    invalidated_by = ()

                if disposition is not ResumeDisposition.RESTORED:
                    invalid.add(key)
                items.append(
                    ResumePlanItem(
                        stage=receipt.stage,
                        item=receipt.item,
                        disposition=disposition,
                        reason=reason,
                        identity_sha256=receipt.identity_sha256,
                        invalidated_by=invalidated_by,
                    )
                )

            key = _checkpoint_key(request.stage, request.item)
            dependency_keys = {_checkpoint_key(stage, item) for stage, item in request.dependencies}
            invalid_dependencies = tuple(sorted(dependency_keys & invalid))
            if self._restart_matches(request.stage, request.item):
                identity, _, _ = self.identity(request)
                lookup = CheckpointLookup(
                    None,
                    ResumeDisposition.RECOMPUTED,
                    REASON_RESTART,
                    identity,
                    (),
                )
            elif invalid_dependencies:
                identity, _, _ = self.identity(request)
                lookup = CheckpointLookup(
                    None,
                    ResumeDisposition.RECOMPUTED,
                    REASON_DEPENDENCY,
                    identity,
                    (),
                    invalid_dependencies,
                )
            elif (
                restore_action == "restore"
                and self.store.read_receipt(request.stage, request.item) is None
                and self.restore_receipt is not None
            ):
                identity, _, _ = self.identity(request)
                lookup = CheckpointLookup(
                    self.restore_receipt,
                    ResumeDisposition.RESTORED,
                    REASON_PRIOR_ATTEMPT,
                    identity,
                    (),
                )
            else:
                lookup = self.lookup(request, mutate_restart=False)
            if lookup.disposition is not ResumeDisposition.RESTORED:
                invalid.add(key)
            items.append(
                ResumePlanItem(
                    stage=request.stage,
                    item=request.item,
                    disposition=lookup.disposition,
                    reason=lookup.reason,
                    identity_sha256=lookup.identity_sha256,
                    invalidated_by=lookup.invalidated_by,
                )
            )

        return ResumePlan(
            schema_version=RESUME_PLAN_SCHEMA_VERSION,
            attempt_id=self.attempt.attempt_id,
            blocked=False,
            blocked_reasons=(),
            items=tuple(items),
        )

    def receipt_dependencies(self) -> tuple[tuple[str, str | None], ...]:
        """Return finalized stage receipts in publication order."""
        return tuple(
            (receipt.stage, receipt.item)
            for receipt in self.store.iter_receipts()
            if receipt.status is CheckpointStatus.FINALIZED
        )

    def stage_item_dependencies(self, stage: str) -> tuple[tuple[str, str | None], ...]:
        """Return the latest finalized item receipts owned by one stage."""
        return tuple(
            (receipt.stage, receipt.item)
            for receipt in self.store.iter_receipts()
            if receipt.stage == stage
            and receipt.item is not None
            and receipt.status is CheckpointStatus.FINALIZED
        )


__all__ = [
    "CheckpointLookup",
    "CheckpointManager",
    "CheckpointRequest",
    "item_checkpoint_manager",
    "REASON_CORRUPT",
    "REASON_DEPENDENCY",
    "REASON_IDENTITY",
    "REASON_MISSING",
    "REASON_PRIOR_ATTEMPT",
    "REASON_RESTART",
    "REASON_VALID",
]
