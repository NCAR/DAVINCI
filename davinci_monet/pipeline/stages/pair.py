"""Pairing stage.

Builds and executes source-pair jobs from explicit ``pairs:`` config.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import xarray as xr

from davinci_monet.core.base import PairedData, iter_paired_variable_xy
from davinci_monet.core.exceptions import PipelineError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourcePairJob,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.helpers import (
    _format_duration,
    _format_size,
)


class PairingStage(BaseStage):
    """Stage for pairing datasets by geometry."""

    def __init__(self) -> None:
        super().__init__("pairing")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that sources are loaded and pairable."""
        config = context.config_dict()
        pairs_config = config.get("pairs")
        if isinstance(pairs_config, dict) and any(
            isinstance(pair, dict) and bool(pair.get("x")) and bool(pair.get("y"))
            for pair in pairs_config.values()
        ):
            return True
        if pairs_config and len(context.sources) >= 2:
            return True
        return False

    def execute(self, context: PipelineContext) -> StageResult:
        """Pair source data through the pairing engine.

        Jobs come from the explicit ``pairs:`` block and run through
        :meth:`_execute_source_pair_jobs` / ``engine.pair_sources``.
        """
        import time

        start = time.time()

        # Get pairing configuration
        config = context.config_dict()
        pairing_config_dict = config.get("pairing", {})

        source_jobs, source_pair_errors = self._build_source_pair_jobs(context)
        if source_pair_errors:
            context.metadata.setdefault("pairing_errors", []).extend(source_pair_errors)
            return self._create_result(
                StageStatus.FAILED,
                data={"paired_keys": []},
                error="Invalid source pair configuration: " + "; ".join(source_pair_errors),
                duration=time.time() - start,
                count=0,
            )
        if not source_jobs:
            return self._create_result(
                StageStatus.COMPLETED,
                data={"paired_keys": []},
                duration=time.time() - start,
                count=0,
            )

        return self._execute_source_pair_jobs(
            context,
            source_jobs,
            pairing_config_dict,
            start,
        )

    def _build_source_pair_jobs(
        self, context: PipelineContext
    ) -> tuple[list[SourcePairJob], list[str]]:
        """Build pair jobs from the explicit ``pairs:`` block.

        Each pair is an ordered ``(x, y)``. ``x`` is the horizontal/reference
        axis; ``y`` is vertical. ``x.source`` is the reference (pairing) geometry
        that ``y.source`` is sampled onto; pairing *direction* still follows shape
        precedence (irregular outranks grid), with ``x`` as the same-shape
        tie-breaker.
        """
        from davinci_monet.pairing.direction import resolve_pair_direction

        jobs: list[SourcePairJob] = []
        errors: list[str] = []
        pair_index = 0
        config = context.config_dict()
        pairs_config = config.get("pairs")

        if not isinstance(pairs_config, dict):
            return [], errors

        for pair_name, raw_pair in pairs_config.items():
            if not isinstance(raw_pair, dict):
                errors.append(f"Pair '{pair_name}' must be a mapping")
                continue
            x_axis = raw_pair.get("x")
            y_axis = raw_pair.get("y")
            if not isinstance(x_axis, dict) or not isinstance(y_axis, dict):
                errors.append(f"Pair '{pair_name}' must declare nested 'x:' and 'y:' axes.")
                continue
            x_source = str(x_axis.get("source") or "")
            y_source = str(y_axis.get("source") or "")
            x_var = x_axis.get("variable")
            y_var = y_axis.get("variable")
            if not x_source or not y_source:
                errors.append(f"Pair '{pair_name}' axes must each name a 'source'.")
                continue
            if x_source not in context.sources or y_source not in context.sources:
                missing = [label for label in (x_source, y_source) if label not in context.sources]
                errors.append(
                    f"Pair '{pair_name}' names unknown source(s): " f"{', '.join(missing)}"
                )
                continue
            if not x_var or not y_var:
                missing = [
                    label for label, value in ((x_source, x_var), (y_source, y_var)) if not value
                ]
                errors.append(
                    f"Pair '{pair_name}' missing variable for axis source(s): "
                    f"{', '.join(missing)}"
                )
                continue
            x_obj = context.sources[x_source]
            y_obj = context.sources[y_source]
            # Validate that both source geometries are detectable. The actual
            # sampling direction is resolved per job so x/y can remain plot axes.
            resolve_pair_direction(
                self._source_geometry(x_obj),
                self._source_geometry(y_obj),
                explicit_geometry="a",
            )
            pair_index += 1
            jobs.append(
                SourcePairJob(
                    index=pair_index,
                    pair_key=str(pair_name),
                    x_source=x_source,
                    x_obj=x_obj,
                    y_source=y_source,
                    y_obj=y_obj,
                    x_var=str(x_var),
                    y_var=str(y_var),
                    radius_of_influence=self._pair_radius(raw_pair, y_obj),
                    strategy_options=self._strategy_options(
                        pairing_config_dict=config.get("pairing", {}),
                        pair_spec=raw_pair,
                    ),
                )
            )
        return jobs, errors

    @staticmethod
    def _lookup_source(context: PipelineContext, label: str) -> Any:
        """Resolve a source by label from the unified ``sources`` store."""
        return context.sources.get(label)

    @staticmethod
    def _source_dataset(obj: Any) -> xr.Dataset | None:
        data = obj.data if hasattr(obj, "data") else obj
        return data if isinstance(data, xr.Dataset) else None

    @staticmethod
    def _source_geometry(obj: Any) -> DataGeometry:
        geom = getattr(obj, "geometry", None)
        if isinstance(geom, DataGeometry):
            return geom
        data = PairingStage._source_dataset(obj)
        if data is not None and "geometry" in data.attrs:
            raw = data.attrs["geometry"]
            if isinstance(raw, DataGeometry):
                return raw
            if isinstance(raw, str):
                return DataGeometry[raw.upper()]
        from davinci_monet.pairing import PairingEngine

        if data is None:
            raise PipelineError("Cannot determine source geometry without a dataset")
        return PairingEngine()._detect_geometry(data)

    @staticmethod
    def _pair_radius(pair_spec: dict[str, Any], y_obj: Any) -> float:
        if pair_spec.get("radius_of_influence") is not None:
            return float(pair_spec["radius_of_influence"])
        cfg = getattr(y_obj, "config", None)
        if isinstance(cfg, dict) and cfg.get("radius_of_influence") is not None:
            return float(cfg["radius_of_influence"])
        return float(getattr(y_obj, "radius_of_influence", 12000.0))

    @staticmethod
    def _strategy_options(
        *,
        pairing_config_dict: dict[str, Any],
        pair_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Return strategy-specific options from global and pair config."""
        control_keys = {
            "x",
            "y",
            "radius_of_influence",
            "time_tolerance",
            "time_method",
            # vertical_method/horizontal_method are sourced from PairingConfig and
            # passed BY NAME by PairingEngine.pair_sources; they must never also
            # ride strategy_options or the **spread raises TypeError (got multiple
            # values for keyword argument).
            "vertical_method",
            "horizontal_method",
            "max_pair_workers",
            "dask_pair_workers",
        }
        options = {k: v for k, v in pairing_config_dict.items() if k not in control_keys}
        options.update({k: v for k, v in pair_spec.items() if k not in control_keys})
        # Flatten a nested ``grid:`` block (method: grid) into top-level kwargs.
        grid_block = options.pop("grid", None)
        if isinstance(grid_block, dict):
            options.update(grid_block)
        return {k: v for k, v in options.items() if v is not None}

    @staticmethod
    def _is_dask_backed(ds: xr.Dataset | None) -> bool:
        """Return ``True`` if any data variable of ``ds`` is dask-chunked.

        A chunked (lazy) dataset triggers the per-strategy
        ``dask.config.set(scheduler="threads").compute()`` — i.e. its own thread
        pool — during extraction. Eager (numpy-backed, unchunked) datasets hit a
        cheap no-op ``.compute()`` that spawns no dask threads. This predicate is
        what the executor uses to keep dask-backed pairs off the overlapping fast
        path (see :meth:`_execute_source_pair_jobs`).
        """
        if ds is None:
            return False
        return any(
            getattr(getattr(var, "data", None), "chunks", None) is not None
            for var in ds.data_vars.values()
        )

    @staticmethod
    def _use_config_x_as_sampling_geometry(
        x_geometry: DataGeometry,
        y_geometry: DataGeometry,
        method: str,
    ) -> bool:
        """Return whether config x should drive sampling geometry.

        ``method: grid`` bins both sources symmetrically, so it keeps config
        x/y order. ``method: auto`` follows geometry precedence: irregular
        geometries outrank GRID; same-precedence pairs use config x as the
        tie-breaker without emitting a user-facing warning.
        """
        if method == "grid":
            return True
        if (x_geometry is DataGeometry.GRID) != (y_geometry is DataGeometry.GRID):
            from davinci_monet.pairing.direction import resolve_pair_direction

            sampling_geometry, _sampled_geometry = resolve_pair_direction(x_geometry, y_geometry)
            return sampling_geometry is x_geometry
        return True

    def _sampling_call_args(
        self, job: SourcePairJob
    ) -> tuple[Any, Any, str, str, str, str, DataGeometry, DataGeometry]:
        """Return engine call args in sampling order for a source-pair job."""
        x_geometry = self._source_geometry(job.x_obj)
        y_geometry = self._source_geometry(job.y_obj)
        method = str(job.strategy_options.get("method", "auto"))
        use_config_x = self._use_config_x_as_sampling_geometry(x_geometry, y_geometry, method)
        if use_config_x:
            return (
                job.x_obj,
                job.y_obj,
                job.x_source,
                job.y_source,
                job.x_var,
                job.y_var,
                x_geometry,
                y_geometry,
            )
        return (
            job.y_obj,
            job.x_obj,
            job.y_source,
            job.x_source,
            job.y_var,
            job.x_var,
            y_geometry,
            x_geometry,
        )

    @staticmethod
    def _normalize_config_axes(
        paired_obj: PairedData,
        job: SourcePairJob,
        output_geometry: DataGeometry,
        sampling_source: str,
        sampled_source: str,
    ) -> PairedData:
        """Return paired data whose axis metadata follows config x/y."""
        data = paired_obj.data.copy()
        pair_canonical = job.x_var
        axis_metadata = {
            job.x_source: {
                "axis": "x",
                "source_label": job.x_source,
                "dataset_variable": job.x_var,
                "canonical_name": pair_canonical,
            },
            job.y_source: {
                "axis": "y",
                "source_label": job.y_source,
                "dataset_variable": job.y_var,
                "canonical_name": pair_canonical,
            },
        }
        for name in data.data_vars:
            source_label = data[name].attrs.get("source_label")
            if source_label in axis_metadata:
                data[name].attrs.update(axis_metadata[source_label])

        pairing_info = dict(paired_obj.pairing_info)
        pairing_info.update(
            {
                "x_source": job.x_source,
                "source_label": job.y_source,
                "geometry": output_geometry.name,
                "sampling_source": sampling_source,
                "sampled_source": sampled_source,
                "axis_variables": {"x": job.x_var, "y": job.y_var},
                "canonical_name": pair_canonical,
            }
        )
        return PairedData.from_sources(
            data=data,
            x_source=job.x_source,
            y_source=job.y_source,
            geometry=output_geometry,
            pairing_info=pairing_info,
        )

    @staticmethod
    def _pair_worker_counts(
        pairing_config_dict: dict[str, Any], n_eager: int, cpu: int
    ) -> tuple[int, int]:
        """Compute bounded ``(eager_workers, dask_workers)`` thread counts.

        Eager: ``max_pair_workers`` if set, else ``min(n_eager, max(1, cpu // 2))``;
        clamped to ``[1, 8]``. Dask: ``dask_pair_workers`` (default ``1`` →
        serial), clamped to ``>= 1``. Pure/static so the clamping is unit-testable
        without running the executor.
        """
        eager_workers = pairing_config_dict.get("max_pair_workers")
        if eager_workers is None:
            eager_workers = min(n_eager, max(1, cpu // 2))
        eager_workers = max(1, min(8, int(eager_workers)))
        dask_workers = max(1, int(pairing_config_dict.get("dask_pair_workers", 1)))
        return eager_workers, dask_workers

    def _run_pair_job(
        self,
        context: PipelineContext,
        job: SourcePairJob,
        pairing_config_dict: dict[str, Any],
        debug: bool,
    ) -> tuple[SourcePairJob, Any, str | None]:
        """Pair a single job. Thread-safe worker: returns, never mutates context.

        Builds its own :class:`PairingConfig` and a local :class:`PairingEngine`
        (no shared engine across threads), runs ``engine.pair_sources``, and
        returns ``(job, paired_obj, None)`` on success or ``(job, None, err)`` on
        failure. It does **not** touch
        ``context.paired``/``context.metadata`` and does **not** call
        ``context.log_progress`` — all shared-state mutation and progress logging
        stays on the main thread.
        """
        from davinci_monet.pairing import PairingConfig, PairingEngine

        (
            sampling_obj,
            sampled_obj,
            sampling_source,
            sampled_source,
            sampling_var,
            sampled_var,
            sampling_geometry,
            sampled_geometry,
        ) = self._sampling_call_args(job)
        sampling_ds = self._source_dataset(sampling_obj)
        sampled_ds = self._source_dataset(sampled_obj)
        if sampling_ds is None or sampled_ds is None:
            return job, None, f"{job.pair_key}: geometry or dataset data is None"

        try:
            pairing_cfg = PairingConfig(
                radius_of_influence=job.radius_of_influence,
                time_tolerance=pairing_config_dict.get("time_tolerance", "1h"),
                time_method=pairing_config_dict.get("time_method", "nearest"),
            )
            engine = PairingEngine()
            paired_obj = engine.pair_sources(
                x_data=sampling_ds,
                y_data=sampled_ds,
                x_vars=[sampling_var],
                y_vars=[sampled_var],
                output_geometry=sampling_geometry,
                y_geometry=sampled_geometry,
                config=pairing_cfg,
                x_source=sampling_source,
                y_source=sampled_source,
                **job.strategy_options,
            )
            return (
                job,
                self._normalize_config_axes(
                    paired_obj,
                    job,
                    sampling_geometry,
                    sampling_source,
                    sampled_source,
                ),
                None,
            )
        except Exception as e:
            return job, None, f"{job.pair_key}: {e}"

    def _execute_source_pair_jobs(
        self,
        context: PipelineContext,
        jobs: list[SourcePairJob],
        pairing_config_dict: dict[str, Any],
        start: float,
    ) -> StageResult:
        """Execute source-pair jobs through the pairing engine.

        Cross-pair concurrency is HDF5-safe and bounded. Jobs are partitioned by
        :meth:`_is_dask_backed`:

        * **eager jobs** (neither side dask-chunked) do a no-op ``.compute()`` and
          spawn no dask threads, so they overlap freely on a
          :class:`~concurrent.futures.ThreadPoolExecutor`.
        * **dask-backed jobs** (either side chunked) each open their own
          per-strategy dask thread pool during extraction. Overlapping them would
          nest thread pools → GIL/contention and the documented HDF5 segfaults, so
          they run **serially by default**.

        Thread-safety: worker threads (``_run_pair_job``) only read source data and
        mutate their own paired dataset attrs; they return results. Only the main
        thread writes ``context.paired``/``context.metadata`` and emits
        ``context.log_progress`` (as each result resolves), so the progress stream
        and shared state never race.

        Config knobs (under the ``pairing:`` block):

        * ``max_pair_workers`` (int, optional) — thread count for eager jobs.
          Default ``min(len(eager_jobs), max(1, cpu // 2))``, clamped to
          ``[1, 8]``.
        * ``dask_pair_workers`` (int, default ``1``) — thread count for
          dask-backed jobs. Default ``1`` keeps them serial (nested-pool safe);
          set higher only when each pair's own dask pool is constrained. Clamped
          to ``>= 1``.
        """
        import time

        config = context.config_dict()
        debug = config.get("analysis", {}).get("debug", False)
        paired_count = 0
        execution_errors: list[str] = []
        context.log_progress(f"    parallel_start: {len(jobs)}")

        # Partition by dask-backing: only fully-eager jobs are safe to overlap.
        eager_jobs: list[SourcePairJob] = []
        dask_jobs: list[SourcePairJob] = []
        for job in jobs:
            ref_ds = self._source_dataset(job.x_obj)
            comp_ds = self._source_dataset(job.y_obj)
            if self._is_dask_backed(ref_ds) or self._is_dask_backed(comp_ds):
                dask_jobs.append(job)
            else:
                eager_jobs.append(job)

        # Bounded worker counts.
        cpu = os.cpu_count() or 1
        eager_workers, dask_workers = self._pair_worker_counts(
            pairing_config_dict, len(eager_jobs), cpu
        )

        def _record(
            job: SourcePairJob, paired_obj: Any, error: str | None, pair_start: float
        ) -> None:
            """Main-thread sink for a resolved job: write state, log, count."""
            nonlocal paired_count
            if error is not None or paired_obj is None:
                execution_errors.append(
                    error or f"{job.pair_key}: geometry or dataset data is None"
                )
                context.log_progress(f"    parallel_completed: {job.pair_key} - FAILED")
                return
            paired_data = paired_obj.data
            context.paired[job.pair_key] = paired_obj
            paired_count += 1
            n_vars = len(iter_paired_variable_xy(paired_data))
            n_points = paired_data.sizes.get("time", paired_data.sizes.get("x", 0))
            timing_str = f" [{_format_duration(time.time() - pair_start)}]" if debug else ""
            context.log_progress(
                f"    parallel_completed: {job.pair_key} - "
                f"{n_vars} vars, {_format_size(n_points)} points{timing_str}"
            )

        # Eager jobs: overlap on a bounded thread pool.
        if eager_jobs:
            if eager_workers <= 1 or len(eager_jobs) <= 1:
                for job in eager_jobs:
                    pair_start = time.time()
                    context.log_progress(f"    parallel_started: {job.pair_key}")
                    j, paired_obj, error = self._run_pair_job(
                        context, job, pairing_config_dict, debug
                    )
                    _record(j, paired_obj, error, pair_start)
            else:
                with ThreadPoolExecutor(max_workers=eager_workers) as pool:
                    futures = {}
                    for job in eager_jobs:
                        context.log_progress(f"    parallel_started: {job.pair_key}")
                        futures[
                            pool.submit(
                                self._run_pair_job, context, job, pairing_config_dict, debug
                            )
                        ] = time.time()
                    for future in as_completed(futures):
                        pair_start = futures[future]
                        j, paired_obj, error = future.result()
                        _record(j, paired_obj, error, pair_start)

        # Dask-backed jobs: serial by default (nested dask pools are unsafe to
        # overlap); a bounded pool only if ``dask_pair_workers`` is raised.
        if dask_jobs:
            if dask_workers <= 1 or len(dask_jobs) <= 1:
                for job in dask_jobs:
                    pair_start = time.time()
                    context.log_progress(f"    parallel_started: {job.pair_key}")
                    j, paired_obj, error = self._run_pair_job(
                        context, job, pairing_config_dict, debug
                    )
                    _record(j, paired_obj, error, pair_start)
            else:
                with ThreadPoolExecutor(max_workers=dask_workers) as pool:
                    futures = {}
                    for job in dask_jobs:
                        context.log_progress(f"    parallel_started: {job.pair_key}")
                        futures[
                            pool.submit(
                                self._run_pair_job, context, job, pairing_config_dict, debug
                            )
                        ] = time.time()
                    for future in as_completed(futures):
                        pair_start = futures[future]
                        j, paired_obj, error = future.result()
                        _record(j, paired_obj, error, pair_start)

        context.log_progress("    parallel_end")
        if execution_errors:
            context.metadata.setdefault("pairing_errors", []).extend(execution_errors)
            if paired_count == 0:
                return self._create_result(
                    StageStatus.FAILED,
                    data={"paired_keys": list(context.paired.keys())},
                    error=f"all {len(jobs)} pairs failed",
                    duration=time.time() - start,
                    count=paired_count,
                    warnings=list(execution_errors),
                )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"paired_keys": list(context.paired.keys())},
                duration=time.time() - start,
                count=paired_count,
                warnings=list(execution_errors),
            )
        return self._create_result(
            StageStatus.COMPLETED,
            data={"paired_keys": list(context.paired.keys())},
            duration=time.time() - start,
            count=paired_count,
        )
