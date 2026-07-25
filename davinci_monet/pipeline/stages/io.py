"""Save-results stage.

Writes statistics (comparison, descriptive, and per-flight) to CSV files.
"""

from __future__ import annotations

from typing import Any

from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)


class SaveResultsStage(BaseStage):
    """Stage for saving analysis results."""

    def __init__(self) -> None:
        super().__init__("save_results")

    def execute(self, context: PipelineContext) -> StageResult:
        """Save analysis results to files."""
        import math
        import time
        from pathlib import Path

        import pandas as pd

        start = time.time()
        saved_files: list[str] = []
        saved_products: dict[str, str] = {}

        # Get output directory from analysis config
        output_dir = Path(context.analysis_config().output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save statistics from the statistics stage. Comparison stats go to
        # statistics_summary.csv for comparison stats; single-source descriptive
        # stats go to a separate statistics_descriptive.csv.
        stats_result = context.results.get("statistics")
        stats_kind = context.metadata.get("statistics_kind", "comparison")
        if stats_result and stats_result.data and stats_kind == "descriptive":
            context.log_progress("step: Writing descriptive statistics CSV...")
            desc_rows = []
            for source_label, var_stats in stats_result.data.items():
                for var_name, var_metrics in var_stats.items():
                    if var_name.startswith("_"):
                        continue
                    desc_rows.append({"Variable": var_name, "Source": source_label, **var_metrics})
            if desc_rows:
                desc_df = pd.DataFrame(desc_rows).set_index("Variable")
                desc_file = output_dir / "statistics_descriptive.csv"
                desc_df.to_csv(desc_file)
                saved_files.append(str(desc_file))
                saved_products["statistics_descriptive"] = str(desc_file)
                context.log_progress(f"done: {len(desc_rows)} rows saved")
        elif stats_result and stats_result.data:
            context.log_progress("step: Writing statistics CSV...")
            rows = []

            def _get_metric(stats: dict[str, Any], *keys: str, default: Any = float("nan")) -> Any:
                for key in keys:
                    if key in stats and stats[key] is not None:
                        return stats[key]
                return default

            for pair_key, pair_stats in stats_result.data.items():
                for var_name, var_stats in pair_stats.items():
                    # Skip internal keys like _per_flight
                    if var_name.startswith("_"):
                        continue
                    row = {"Variable": var_name, **var_stats}
                    row["N"] = _get_metric(var_stats, "N", "n", default=0)
                    mean_x = _get_metric(var_stats, "MX", "x_mean")
                    mean_y = _get_metric(var_stats, "MY", "y_mean")
                    row["Mean_X"] = mean_x
                    row["Mean_Y"] = mean_y
                    row["MB"] = _get_metric(var_stats, "MB", "mean_bias")
                    row["RMSE"] = _get_metric(var_stats, "RMSE", "rmse")
                    row["R"] = _get_metric(var_stats, "R", "correlation")
                    row["IOA"] = _get_metric(var_stats, "IOA", "ioa")

                    # Prefer computed NMB/NME if present; otherwise derive as fallback
                    nmb = _get_metric(var_stats, "NMB", default=float("nan"))
                    nme = _get_metric(var_stats, "NME", default=float("nan"))
                    x_mean = row["Mean_X"]

                    if isinstance(nmb, (int, float)) and not math.isnan(float(nmb)):
                        row["NMB_%"] = nmb
                    elif (
                        isinstance(x_mean, (int, float))
                        and x_mean not in (0, -0.0)
                        and not math.isnan(float(x_mean))
                    ):
                        row["NMB_%"] = (
                            (row["MB"] / x_mean) * 100
                            if isinstance(row["MB"], (int, float))
                            else float("nan")
                        )
                    else:
                        row["NMB_%"] = float("nan")

                    if isinstance(nme, (int, float)) and not math.isnan(float(nme)):
                        row["NME_%"] = nme
                    else:
                        # No correct fallback: NME requires per-point |y-x|,
                        # which can't be derived from RMSE or other summary scalars.
                        row["NME_%"] = float("nan")

                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                df = df.set_index("Variable")
                stats_file = output_dir / "statistics_summary.csv"
                df.to_csv(stats_file)
                saved_files.append(str(stats_file))
                saved_products["statistics_summary"] = str(stats_file)
                context.log_progress(f"done: {len(rows)} rows saved")

            # Save per-flight statistics if available
            all_flight_stats: list[dict] = []
            for pair_key, pair_stats in stats_result.data.items():
                if "_per_flight" in pair_stats:
                    all_flight_stats.extend(pair_stats["_per_flight"])

            if all_flight_stats:
                context.log_progress("step: Writing per-flight statistics CSV...")
                flight_df = pd.DataFrame(all_flight_stats)
                # Reorder columns
                cols = [
                    "variable",
                    "flight",
                    "N",
                    "MX",
                    "MY",
                    "MB",
                    "RMSE",
                    "R",
                    "NMB_%",
                    "NME_%",
                ]
                cols = [c for c in cols if c in flight_df.columns]
                flight_df = flight_df[cols]
                flight_df = flight_df.sort_values(["variable", "flight"])

                flight_stats_file = output_dir / "statistics_per_flight.csv"
                flight_df.to_csv(flight_stats_file, index=False)
                saved_files.append(str(flight_stats_file))
                saved_products["statistics_per_flight"] = str(flight_stats_file)
                context.log_progress(f"done: {len(flight_df)} rows")

        return self._create_result(
            StageStatus.COMPLETED,
            data={
                "saved_files": saved_files,
                "saved_products": saved_products,
            },
            duration=time.time() - start,
        )
