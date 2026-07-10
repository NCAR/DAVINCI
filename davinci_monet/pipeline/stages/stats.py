"""Statistics stage.

Computes comparison metrics on paired data or descriptive statistics on
unpaired single-source data.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from davinci_monet.config.schema import StatsConfig
from davinci_monet.core.base import iter_paired_variable_xy
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.helpers import iter_single_source_datasets


def _is_artifact_derived(source: Any, dataset: xr.Dataset) -> bool:
    """Identify persisted derived sources without inspecting their array payloads."""
    geometry = getattr(source, "geometry", None)
    if str(getattr(geometry, "name", "")).upper() == "ARTIFACT":
        return True
    config = getattr(source, "config", None)
    if isinstance(config, dict) and any(
        key in config for key in ("artifact_path", "artifact_dir", "artifact_glob")
    ):
        return True
    return str(dataset.attrs.get("artifact_finalized", "")).lower() == "true"


def _descriptive_sources(
    context: PipelineContext,
) -> tuple[list[tuple[str, Any, xr.Dataset]], list[str]]:
    included: list[tuple[str, Any, xr.Dataset]] = []
    skipped: list[str] = []
    for label, source, dataset in iter_single_source_datasets(context):
        if _is_artifact_derived(source, dataset):
            skipped.append(label)
        else:
            included.append((label, source, dataset))
    return included, skipped


def _effective_stats_config(context: PipelineContext) -> StatsConfig | None:
    """Preserve implicit legacy statistics only for standard workflows."""
    configured = context.stats_config()
    if configured is not None:
        return configured
    if context.analysis_config().workflow == "standard":
        return StatsConfig()
    return None


class StatisticsStage(BaseStage):
    """Stage for calculating statistics on paired or single-source data."""

    def __init__(self) -> None:
        super().__init__("statistics")

    def validate(self, context: PipelineContext) -> bool:
        """Run for configured stats or standard-workflow legacy defaults."""
        if _effective_stats_config(context) is None:
            return False
        if context.paired:
            return True
        sources, _skipped = _descriptive_sources(context)
        return bool(sources)

    def _execute_descriptive(
        self,
        context: PipelineContext,
        sources: list[tuple[str, Any, xr.Dataset]] | None = None,
    ) -> StageResult:
        """Descriptive stats for unpaired sources."""
        import time

        import numpy as np

        context.metadata["statistics_kind"] = "descriptive"
        start = time.time()
        all_stats: dict[str, dict[str, dict[str, float]]] = {}
        source_items = sources if sources is not None else _descriptive_sources(context)[0]
        for source_label, _source_obj, ds in source_items:
            source_stats: dict[str, dict[str, float]] = {}
            for var_name in ds.data_vars:
                var_key = str(var_name)
                if ds[var_key].dtype.kind not in "biuf":
                    continue
                values = ds[var_key].values.flatten()
                if values.dtype.kind == "b":
                    values = values.astype(np.float64)
                values = values[np.isfinite(values)]
                if len(values) < 1:
                    continue
                source_stats[var_key] = {
                    "N": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "p90": float(np.percentile(values, 90)),
                }
            all_stats[source_label] = source_stats
        return self._create_result(
            StageStatus.COMPLETED,
            data=all_stats,
            duration=time.time() - start,
        )

    def execute(self, context: PipelineContext) -> StageResult:
        """Comparison statistics on paired data, or descriptive statistics on
        unpaired sources."""
        import time

        stats_cfg = _effective_stats_config(context)
        if stats_cfg is None:
            return self._create_result(StageStatus.SKIPPED, data={})

        single_sources, skipped_sources = _descriptive_sources(context)
        if skipped_sources:
            context.metadata["statistics_skipped_artifact_sources"] = skipped_sources
        if not context.paired and single_sources:
            return self._execute_descriptive(context, single_sources)
        if not context.paired:
            return self._create_result(StageStatus.SKIPPED, data={})

        context.metadata["statistics_kind"] = "comparison"
        start = time.time()
        stats_results: dict[str, Any] = {}

        effective_cfg = stats_cfg
        requested_pairs = effective_cfg.data or []
        if isinstance(requested_pairs, str):
            requested_pairs = [requested_pairs]
        if requested_pairs:
            missing_pairs = [
                str(pair) for pair in requested_pairs if str(pair) not in context.paired
            ]
            if missing_pairs:
                message = "Unknown stats pair reference(s): " + ", ".join(missing_pairs)
                context.metadata.setdefault("stats_errors", []).append(message)
                return self._create_result(
                    StageStatus.FAILED,
                    data=stats_results,
                    error=message,
                    duration=time.time() - start,
                )
            pair_items = [(str(pair), context.paired[str(pair)]) for pair in requested_pairs]
        else:
            pair_items = list(context.paired.items())

        total_pairs = len(pair_items)
        pair_count = 0

        for pair_key, paired_obj in pair_items:
            try:
                pair_count += 1
                context.log_progress(f"    Stats: {pair_key} ({pair_count}/{total_pairs})")
                context.log_progress("step: Computing metrics...")

                # Handle PairedData objects
                paired_data = paired_obj.data if hasattr(paired_obj, "data") else paired_obj

                # Apply global stats-config domain filter if requested.
                from davinci_monet.util.domain import filter_paired_by_domain

                paired_data = filter_paired_by_domain(
                    paired_data,
                    effective_cfg.domain_type,
                    effective_cfg.domain_name,
                )

                # Calculate basic statistics
                pair_stats = self._calculate_stats(paired_data, effective_cfg)
                stats_results[pair_key] = pair_stats

                # Summary
                n_metrics = sum(len(v) for v in pair_stats.values())
                n_vars = len(pair_stats)
                context.log_progress(f"done: {n_vars} vars, {n_metrics} metrics")

            except Exception as e:
                context.metadata.setdefault("stats_errors", []).append(f"{pair_key}: {e}")
                context.log_progress(f"warning: stats failed for {pair_key}: {e}")

        errors = context.metadata.get("stats_errors") or []
        if errors:
            if not stats_results:
                return self._create_result(
                    StageStatus.FAILED,
                    data=stats_results,
                    error=f"all {len(errors)} stats pairs failed",
                    duration=time.time() - start,
                    warnings=list(errors),
                )
            return self._create_result(
                StageStatus.COMPLETED,
                data=stats_results,
                duration=time.time() - start,
                warnings=list(errors),
            )

        return self._create_result(
            StageStatus.COMPLETED,
            data=stats_results,
            duration=time.time() - start,
        )

    def _calculate_stats(
        self, paired_data: xr.Dataset, stats_cfg: StatsConfig | None
    ) -> dict[str, Any]:
        """Calculate statistics for a paired dataset."""
        import numpy as np

        from davinci_monet.stats import StatisticsCalculator, StatisticsConfig
        from davinci_monet.stats.metrics import STANDARD_METRICS

        stats: dict[str, Any] = {}

        effective_cfg = stats_cfg or StatsConfig()
        metrics = effective_cfg.metrics or effective_cfg.stat_list
        round_precision = effective_cfg.round_output
        include_counts = effective_cfg.include_counts
        remove_nan = effective_cfg.remove_nan
        min_samples = effective_cfg.min_samples

        calc_config = StatisticsConfig(
            metrics=list(metrics) if metrics else list(STANDARD_METRICS),
            round_precision=round_precision,
            include_counts=include_counts,
            remove_nan=remove_nan,
            min_samples=min_samples,
        )
        calculator = StatisticsCalculator(calc_config)

        # Pair x and y variables by canonical name.
        for x_var, y_var, base_name in iter_paired_variable_xy(paired_data):
            df = calculator.compute(
                paired_data,
                x_var=x_var,
                y_var=y_var,
                metrics=list(metrics) if metrics else None,
            )

            if df.empty:
                continue

            row = df.iloc[0].to_dict()
            # Normalize numpy scalars to Python types
            for key, value in list(row.items()):
                if isinstance(value, (np.floating, np.integer)):
                    row[key] = float(value)

            alias_map = {
                "n": "N",
                "mean_bias": "MB",
                "rmse": "RMSE",
                "correlation": "R",
                "y_mean": "MY",
                "x_mean": "MX",
            }
            for alias_key, metric_key in alias_map.items():
                if metric_key in row and alias_key not in row:
                    row[alias_key] = row[metric_key]

            stats[base_name] = row

        # Per-flight statistics (if flight coord exists and enabled)
        per_flight = effective_cfg.per_flight
        if per_flight and "flight" in paired_data.coords:
            stats["_per_flight"] = self._calculate_per_flight_stats(paired_data)

        return stats

    def _calculate_per_flight_stats(self, paired_data: xr.Dataset) -> list[dict[str, Any]]:
        """Calculate statistics for each flight.

        Parameters
        ----------
        paired_data : xr.Dataset
            Paired dataset with source-label variables and a flight coordinate.

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries with per-flight statistics.
        """
        import numpy as np

        flights = np.unique(paired_data["flight"].values)
        flight_stats: list[dict[str, Any]] = []

        var_pairs = iter_paired_variable_xy(paired_data)

        for flight in flights:
            mask = paired_data["flight"].values == flight
            flight_data = paired_data.isel(time=mask)

            for x_var, y_var, base_name in var_pairs:
                if x_var not in flight_data or y_var not in flight_data:
                    continue

                y_vals = flight_data[y_var].values.flatten()
                x_vals = flight_data[x_var].values.flatten()

                # Remove NaNs
                valid = ~(np.isnan(y_vals) | np.isnan(x_vals))
                if valid.sum() < 3:
                    continue

                m, o = y_vals[valid], x_vals[valid]
                diff = m - o

                row: dict[str, Any] = {
                    "flight": str(flight),
                    "variable": base_name,
                    "N": len(m),
                    "MX": float(np.mean(o)),
                    "MY": float(np.mean(m)),
                    "MB": float(np.mean(diff)),
                    "RMSE": float(np.sqrt(np.mean(diff**2))),
                    "R": float(np.corrcoef(o, m)[0, 1]) if len(m) > 1 else np.nan,
                }
                # NMB/NME
                if row["MX"] != 0:
                    row["NMB_%"] = (row["MB"] / row["MX"]) * 100
                    row["NME_%"] = (float(np.mean(np.abs(diff))) / row["MX"]) * 100
                else:
                    row["NMB_%"] = np.nan
                    row["NME_%"] = np.nan
                flight_stats.append(row)

        return flight_stats
