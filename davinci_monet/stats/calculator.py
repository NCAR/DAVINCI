"""Statistics calculator for paired source data.

This module provides the main interface for computing statistics
on paired x/y datasets, with support for grouping and multiple
metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from davinci_monet.core.exceptions import StatisticsError
from davinci_monet.stats.metrics import STANDARD_METRICS, get_metric, list_metrics

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StatisticsConfig:
    """Configuration for statistics calculation.

    Attributes
    ----------
    metrics : list[str]
        List of metric names to calculate.
    round_precision : int
        Number of decimal places for rounding results.
    include_counts : bool
        If True, include sample count in output.
    remove_nan : bool
        If True, remove NaN values before calculation.
    min_samples : int
        Minimum number of samples required for calculation.
    """

    metrics: list[str] = field(default_factory=lambda: list(STANDARD_METRICS))
    round_precision: int = 3
    include_counts: bool = True
    remove_nan: bool = True
    min_samples: int = 3

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> StatisticsConfig:
        """Create config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# Statistics Calculator
# =============================================================================


class StatisticsCalculator:
    """Calculator for computing statistics on paired data.

    Supports computing multiple metrics, grouping by dimensions,
    and outputting results as DataFrames.

    Parameters
    ----------
    config
        Statistics configuration.

    Examples
    --------
    >>> calc = StatisticsCalculator()
    >>> stats = calc.compute(paired_data, "x_o3", "y_o3")
    >>> print(stats)
           N     MG     MD      MB    RMSE      R
    0  1000  45.2  47.1    1.9    5.3  0.85

    >>> # Group by site
    >>> stats = calc.compute(
    ...     paired_data, "x_o3", "y_o3",
    ...     groupby="site"
    ... )
    """

    def __init__(self, config: StatisticsConfig | None = None) -> None:
        self.config = config or StatisticsConfig()

    def compute(
        self,
        paired_data: xr.Dataset,
        x_var: str | None = None,
        y_var: str | None = None,
        metrics: Sequence[str] | None = None,
        groupby: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics for paired data.

        Parameters
        ----------
        paired_data
            Paired dataset with x and y variables.
        metrics
            List of metric names. If None, uses config.metrics.
        groupby
            Optional dimension(s) to group by.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        **kwargs
            Additional options passed to individual metrics.

        Returns
        -------
        pd.DataFrame
            Statistics table with metrics as columns.
        """
        metrics = list(metrics) if metrics is not None else self.config.metrics
        if x_var is None or y_var is None:
            raise StatisticsError("Both x_var and y_var are required")

        # Get data arrays
        x_data = paired_data[x_var]
        y_data = paired_data[y_var]

        if groupby is not None:
            return self._compute_grouped(x_data, y_data, metrics, groupby, **kwargs)
        else:
            return self._compute_overall(x_data, y_data, metrics, **kwargs)

    def _compute_overall(
        self,
        x_data: xr.DataArray,
        y_data: xr.DataArray,
        metrics: list[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute overall statistics (no grouping).

        Parameters
        ----------
        x_data, y_data
            Data arrays.
        metrics
            List of metric names.
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with statistics.
        """
        x = x_data.values.flatten()
        y = y_data.values.flatten()

        results = self._compute_metrics(x, y, metrics, **kwargs)
        df = pd.DataFrame([results])

        if self.config.round_precision is not None:
            df = df.round(self.config.round_precision)

        return df

    def _compute_grouped(
        self,
        x_data: xr.DataArray,
        y_data: xr.DataArray,
        metrics: list[str],
        groupby: str | Sequence[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics grouped by dimension(s).

        Parameters
        ----------
        x_data, y_data
            Data arrays.
        metrics
            List of metric names.
        groupby
            Dimension(s) to group by.
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per group.
        """
        if isinstance(groupby, str):
            groupby = [groupby]

        # Handle special groupby patterns (e.g., "time.month")
        parsed_groupby = []
        for g in groupby:
            if "." in g:
                dim, accessor = g.split(".", 1)
                # Create grouping coordinate
                if dim == "time":
                    if accessor == "month":
                        group_coord = x_data.time.dt.month
                    elif accessor == "hour":
                        group_coord = x_data.time.dt.hour
                    elif accessor == "dayofweek":
                        group_coord = x_data.time.dt.dayofweek
                    elif accessor == "season":
                        group_coord = x_data.time.dt.season
                    else:
                        raise StatisticsError(f"Unknown time accessor: {accessor}")
                    parsed_groupby.append((g, group_coord))
                else:
                    raise StatisticsError(f"Unknown groupby pattern: {g}")
            else:
                parsed_groupby.append((g, g))

        # Simple case: single dimension groupby
        if len(parsed_groupby) == 1:
            name, coord = parsed_groupby[0]
            return self._compute_single_groupby(x_data, y_data, metrics, name, coord, **kwargs)

        # Multi-dimensional groupby
        return self._compute_multi_groupby(x_data, y_data, metrics, parsed_groupby, **kwargs)

    def _compute_single_groupby(
        self,
        x_data: xr.DataArray,
        y_data: xr.DataArray,
        metrics: list[str],
        name: str,
        coord: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics for single-dimension groupby.

        Parameters
        ----------
        x_data, y_data
            Data arrays.
        metrics
            List of metric names.
        name
            Groupby dimension name.
        coord
            Grouping coordinate.
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per group.
        """
        import xarray as xr

        # Group the data
        if isinstance(coord, str):
            x_grouped = x_data.groupby(coord, squeeze=False)
            y_grouped = y_data.groupby(coord, squeeze=False)
        else:
            x_grouped = x_data.groupby(coord, squeeze=False)
            y_grouped = y_data.groupby(coord, squeeze=False)

        results = []
        for (x_key, x_group), (_, y_group) in zip(x_grouped, y_grouped):
            x = x_group.values.flatten()
            y = y_group.values.flatten()

            row = {name: x_key}
            row.update(self._compute_metrics(x, y, metrics, **kwargs))
            results.append(row)

        df = pd.DataFrame(results)
        if self.config.round_precision is not None:
            # Only round numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(self.config.round_precision)

        return df

    def _compute_multi_groupby(
        self,
        x_data: xr.DataArray,
        y_data: xr.DataArray,
        metrics: list[str],
        parsed_groupby: list[tuple[str, Any]],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics for multi-dimension groupby.

        Parameters
        ----------
        x_data, y_data
            Data arrays.
        metrics
            List of metric names.
        parsed_groupby
            List of (name, coord) tuples.
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per group combination.
        """
        # For multi-dimensional groupby, we need a different approach
        # Stack all groupby dimensions and iterate

        # This is a simplified implementation - for now, convert to pandas
        # and use pandas groupby
        x_df = x_data.to_dataframe(name="x").reset_index()
        y_df = y_data.to_dataframe(name="y").reset_index()

        # Merge on common indices
        common_cols = list(set(x_df.columns) & set(y_df.columns) - {"x", "y"})
        df = pd.merge(x_df, y_df, on=common_cols)

        # Add groupby columns
        group_cols = []
        for name, coord in parsed_groupby:
            if "." in name:
                dim, accessor = name.split(".", 1)
                if dim == "time" and "time" in df.columns:
                    if accessor == "month":
                        df[name] = df["time"].dt.month
                    elif accessor == "hour":
                        df[name] = df["time"].dt.hour
                    elif accessor == "dayofweek":
                        df[name] = df["time"].dt.dayofweek
                    elif accessor == "season":
                        df[name] = df["time"].dt.month.map(
                            {
                                12: "DJF",
                                1: "DJF",
                                2: "DJF",
                                3: "MAM",
                                4: "MAM",
                                5: "MAM",
                                6: "JJA",
                                7: "JJA",
                                8: "JJA",
                                9: "SON",
                                10: "SON",
                                11: "SON",
                            }
                        )
            group_cols.append(name)

        # Group and compute
        results = []
        for group_keys, group_df in df.groupby(group_cols):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)

            x = group_df["x"].values
            y = group_df["y"].values

            row = dict(zip(group_cols, group_keys))
            row.update(self._compute_metrics(x, y, metrics, **kwargs))
            results.append(row)

        result_df = pd.DataFrame(results)
        if self.config.round_precision is not None:
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            result_df[numeric_cols] = result_df[numeric_cols].round(self.config.round_precision)

        return result_df

    def _compute_metrics(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metrics: list[str],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute all requested metrics.

        Parameters
        ----------
        x, y
            Arrays of x and y values.
        metrics
            List of metric names.
        **kwargs
            Additional options.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to values.
        """
        # Remove NaN if configured
        if self.config.remove_nan:
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

        results = {}

        # Check minimum samples
        if len(x) < self.config.min_samples:
            for metric_name in metrics:
                if metric_name == "N":
                    results[metric_name] = float(len(x))
                else:
                    results[metric_name] = np.nan
            return results

        # Compute each metric
        for metric_name in metrics:
            try:
                metric = get_metric(metric_name)
                results[metric_name] = metric.compute(x, y, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Metric '%s' raised an exception and will be set to NaN: %s",
                    metric_name,
                    exc,
                )
                results[metric_name] = np.nan

        return results

    def compute_summary(
        self,
        paired_data: xr.Dataset,
        x_var: str | None = None,
        y_var: str | None = None,
        metrics: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute summary statistics as a dictionary.

        Parameters
        ----------
        paired_data
            Paired dataset.
        metrics
            List of metric names.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        **kwargs
            Additional options.

        Returns
        -------
        dict[str, float]
            Dictionary of metric values.
        """
        df = self.compute(
            paired_data,
            x_var,
            y_var,
            metrics=metrics,
            **kwargs,
        )
        return df.iloc[0].to_dict()


# =============================================================================
# Convenience Functions
# =============================================================================


def calculate_statistics(
    paired_data: xr.Dataset,
    x_var: str | None = None,
    y_var: str | None = None,
    metrics: Sequence[str] | None = None,
    groupby: str | Sequence[str] | None = None,
    config: StatisticsConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Calculate statistics for paired data.

    This is the main entry point for statistics calculation.

    Parameters
    ----------
    paired_data
        Paired dataset with x and y variables.
    metrics
        List of metric names. If None, uses standard set.
    groupby
        Optional dimension(s) to group by.
    config
        Statistics configuration.
    x_var
        Name of the x variable.
    y_var
        Name of the y variable.
    **kwargs
        Additional options.

    Returns
    -------
    pd.DataFrame
        Statistics table.

    Examples
    --------
    >>> stats = calculate_statistics(
    ...     paired_data, "x_o3", "y_o3",
    ...     metrics=["MB", "RMSE", "R"],
    ... )

    >>> # Group by site and month
    >>> stats = calculate_statistics(
    ...     paired_data, "x_o3", "y_o3",
    ...     groupby=["site", "time.month"],
    ... )
    """
    if isinstance(config, dict):
        config = StatisticsConfig.from_dict(config)

    calc = StatisticsCalculator(config=config)
    return calc.compute(
        paired_data,
        x_var,
        y_var,
        metrics=metrics,
        groupby=groupby,
        **kwargs,
    )


def quick_stats(
    x: np.ndarray,
    y: np.ndarray,
    metrics: Sequence[str] | None = None,
    min_samples: int = 3,
) -> dict[str, float]:
    """Quick statistics calculation from arrays.

    Parameters
    ----------
    x
        Reference array.
    y
        Comparison array.
    metrics
        List of metric names. If None, uses standard set.
    min_samples
        Minimum valid-pair count required for metrics other than ``N``.

    Returns
    -------
    dict[str, float]
        Dictionary of metric values.

    Examples
    --------
    >>> stats = quick_stats(x_array, y_array)
    >>> print(f"RMSE: {stats['RMSE']:.2f}")
    """
    metrics = list(metrics) if metrics is not None else STANDARD_METRICS

    config = StatisticsConfig(metrics=metrics, min_samples=min_samples)
    return StatisticsCalculator(config)._compute_metrics(
        np.asarray(x).flatten(),
        np.asarray(y).flatten(),
        list(metrics),
    )
