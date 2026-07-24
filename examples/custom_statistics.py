#!/usr/bin/env python
"""Custom statistics example using DAVINCI stats module.

This script demonstrates using the statistics module directly for
custom analysis workflows.
"""

import numpy as np
import pandas as pd
import xarray as xr


def main():
    """Demonstrate statistics module usage."""
    print("DAVINCI Statistics Module Example")
    print("=" * 45)

    # =========================================================================
    # 1. Quick statistics from arrays
    # =========================================================================
    print("\n1. Quick Statistics from Arrays")
    print("-" * 40)

    from davinci_monet.stats import quick_stats

    np.random.seed(42)
    x = np.random.randn(1000) * 10 + 50
    y = x + np.random.randn(1000) * 5 + 3  # Positive bias, some scatter

    stats = quick_stats(x, y)

    print(f"   Sample size:  {stats['N']:.0f}")
    print(f"   Mean X:     {stats['MX']:.2f}")
    print(f"   Mean Y:   {stats['MY']:.2f}")
    print(f"   Mean Bias:    {stats['MB']:.2f}")
    print(f"   RMSE:         {stats['RMSE']:.2f}")
    print(f"   Correlation:  {stats['R']:.3f}")
    print(f"   NMB:          {stats['NMB']:.1f}%")

    # =========================================================================
    # 2. Individual metric calculation
    # =========================================================================
    print("\n2. Individual Metric Calculation")
    print("-" * 40)

    from davinci_monet.stats.metrics import compute_metric, get_metric, list_metrics

    # List available metrics
    print(f"   Available metrics: {len(list_metrics())}")
    print(f"   {list_metrics()[:10]}...")

    # Compute specific metrics
    rmse = compute_metric("RMSE", x, y)
    ioa = compute_metric("IOA", x, y)
    fb = compute_metric("FB", x, y)

    print(f"\n   RMSE = {rmse:.3f}")
    print(f"   IOA  = {ioa:.3f}")
    print(f"   FB   = {fb:.3f}")

    # Get metric with metadata
    metric = get_metric("NME")
    value = metric.compute(x, y)
    print(f"\n   {metric.long_name} ({metric.name}) = {value:.1f}%")

    # =========================================================================
    # 3. Statistics from paired dataset
    # =========================================================================
    print("\n3. Statistics from Paired Dataset")
    print("-" * 40)

    from davinci_monet.stats import calculate_statistics

    # Create synthetic paired dataset
    times = np.arange("2024-07-01", "2024-07-08", dtype="datetime64[h]")
    sites = [f"SITE_{i:02d}" for i in range(20)]

    x_data = np.random.randn(len(times), len(sites)) * 10 + 50
    y_data = x_data + np.random.randn(len(times), len(sites)) * 5 + 2

    paired = xr.Dataset(
        {
            "x_o3": (["time", "site"], x_data),
            "y_o3": (["time", "site"], y_data),
        },
        coords={
            "time": times,
            "site": sites,
        },
    )

    # Calculate with specific metrics
    stats_df = calculate_statistics(
        paired, "x_o3", "y_o3", metrics=["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"]
    )
    print("\n   Overall Statistics:")
    print(stats_df.to_string())

    # =========================================================================
    # 4. Grouped statistics
    # =========================================================================
    print("\n\n4. Grouped Statistics")
    print("-" * 40)

    # Add site metadata
    np.random.seed(123)
    paired["region"] = xr.DataArray(
        np.random.choice(["East", "West", "Central"], len(sites)), dims=["site"]
    )

    # Statistics by site
    print("\n   By Site (first 5):")
    stats_by_site = calculate_statistics(
        paired, "x_o3", "y_o3", metrics=["N", "MB", "RMSE", "R"], groupby="site"
    )
    print(stats_by_site.head().to_string())

    # Statistics by hour of day
    print("\n   By Hour of Day (sample):")
    stats_by_hour = calculate_statistics(
        paired, "x_o3", "y_o3", metrics=["N", "MB", "RMSE"], groupby="time.hour"
    )
    print(stats_by_hour.iloc[::6].to_string())  # Every 6 hours

    # =========================================================================
    # 5. Output formatting
    # =========================================================================
    print("\n\n5. Output Formatting")
    print("-" * 40)

    from davinci_monet.stats import (
        METRIC_FULL_NAMES,
        StatisticsFormatter,
        format_stats_summary,
        get_metric_fullname,
    )

    # Format summary string
    stats_dict = {"MB": 2.5, "RMSE": 8.3, "R": 0.89, "NMB": 4.5}
    summary = format_stats_summary(stats_dict)
    print("\n   Formatted Summary:")
    print("   " + summary.replace("\n", "\n   "))

    # Get full metric names
    print("\n   Metric Full Names:")
    for abbr in ["MB", "RMSE", "R", "NMB", "IOA"]:
        print(f"   {abbr:6s} -> {get_metric_fullname(abbr)}")

    # =========================================================================
    # 6. Export to files
    # =========================================================================
    print("\n\n6. Export to Files")
    print("-" * 40)

    from pathlib import Path

    from davinci_monet.stats import write_statistics_csv

    output_dir = Path("examples/output/custom_stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export overall stats to CSV
    write_statistics_csv(stats_df, output_dir / "overall_stats.csv")
    print(f"   Saved: {output_dir / 'overall_stats.csv'}")

    # Export grouped stats
    stats_by_site.to_csv(output_dir / "stats_by_site.csv")
    print(f"   Saved: {output_dir / 'stats_by_site.csv'}")

    # =========================================================================
    # 7. Using StatisticsCalculator class
    # =========================================================================
    print("\n\n7. Using StatisticsCalculator Class")
    print("-" * 40)

    from davinci_monet.stats import StatisticsCalculator, StatisticsConfig

    # Configure calculator
    config = StatisticsConfig(
        metrics=["N", "MX", "MY", "MB", "RMSE", "R", "R2", "NMB", "NME", "IOA", "d1"],
        round_precision=4,
    )

    calc = StatisticsCalculator(config)

    # Compute with custom config
    result = calc.compute(paired, "x_o3", "y_o3")
    print("\n   Calculator Result:")
    print(result.T.to_string())

    print("\n\nDone!")


if __name__ == "__main__":
    main()
