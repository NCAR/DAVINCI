"""Tests for the statistics module.

This module tests the statistical metrics, calculator, and output
functionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create simple x and y arrays."""
    np.random.seed(42)
    x = np.array([10, 20, 30, 40, 50], dtype=float)
    y = np.array([12, 18, 33, 38, 52], dtype=float)
    return x, y


@pytest.fixture
def random_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create random x and y arrays."""
    np.random.seed(42)
    x = np.random.normal(50, 10, 1000)
    y = x + np.random.normal(2, 5, 1000)  # Slight positive bias
    return x, y


@pytest.fixture
def paired_dataset() -> xr.Dataset:
    """Create a paired dataset for testing."""
    np.random.seed(42)
    n_times = 100
    n_sites = 5

    time = pd.date_range("2023-01-01", periods=n_times, freq="h")
    sites = [f"site_{i}" for i in range(n_sites)]

    x = np.random.normal(50, 10, (n_times, n_sites))
    y = x + np.random.normal(2, 5, (n_times, n_sites))

    ds = xr.Dataset(
        {
            "x_o3": (["time", "site"], x, {"units": "ppbv"}),
            "y_o3": (["time", "site"], y, {"units": "ppbv"}),
        },
        coords={
            "time": time,
            "site": sites,
        },
    )
    return ds


# =============================================================================
# Metric Tests
# =============================================================================


class TestBasicMetrics:
    """Tests for basic statistical measures."""

    def test_count(self, simple_arrays):
        """Test sample count."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("N")
        result = metric.compute(x, y)
        assert result == 5

    def test_mean_x(self, simple_arrays):
        """Test mean."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("MX")
        result = metric.compute(x, y)
        assert result == pytest.approx(30.0)

    def test_mean_y(self, simple_arrays):
        """Test mean."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("MY")
        result = metric.compute(x, y)
        assert result == pytest.approx(30.6)

    def test_std_x(self, simple_arrays):
        """Test x standard deviation."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("STDX")
        result = metric.compute(x, y)
        assert result == pytest.approx(np.std(x, ddof=1))


class TestBiasMetrics:
    """Tests for bias metrics."""

    def test_mean_bias(self, simple_arrays):
        """Test mean bias."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("MB")
        result = metric.compute(x, y)
        # MB = mean(y - x) = (2 + -2 + 3 + -2 + 2) / 5 = 0.6
        assert result == pytest.approx(0.6)

    def test_mean_bias_sign(self):
        """Test that positive bias means y > x."""
        from davinci_monet.stats import get_metric

        x = np.array([10, 20, 30])
        y = np.array([15, 25, 35])  # y is higher
        metric = get_metric("MB")
        result = metric.compute(x, y)
        assert result > 0  # Positive bias

    def test_normalized_mean_bias(self, simple_arrays):
        """Test normalized mean bias."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("NMB")
        result = metric.compute(x, y)
        # NMB = 100 * sum(y - x) / sum(x) = 100 * 3 / 150 = 2%
        assert result == pytest.approx(2.0)

    def test_fractional_bias(self):
        """Test fractional bias."""
        from davinci_monet.stats import get_metric

        x = np.array([10, 20, 30])
        y = np.array([10, 20, 30])  # Perfect match
        metric = get_metric("FB")
        result = metric.compute(x, y)
        assert result == pytest.approx(0.0)


class TestErrorMetrics:
    """Tests for error metrics."""

    def test_mean_error(self, simple_arrays):
        """Test mean error."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("ME")
        result = metric.compute(x, y)
        # ME = mean(|y - x|) = (2 + 2 + 3 + 2 + 2) / 5 = 2.2
        assert result == pytest.approx(2.2)

    def test_rmse(self, simple_arrays):
        """Test RMSE."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("RMSE")
        result = metric.compute(x, y)
        # RMSE = sqrt(mean((y - x)^2)) = sqrt((4+4+9+4+4)/5) = sqrt(5) ≈ 2.236
        expected = np.sqrt(np.mean((y - x) ** 2))
        assert result == pytest.approx(expected)

    def test_normalized_mean_error(self, simple_arrays):
        """Test normalized mean error."""
        from davinci_monet.stats import get_metric

        x, y = simple_arrays
        metric = get_metric("NME")
        result = metric.compute(x, y)
        # NME = 100 * sum(|y - x|) / sum(x) = 100 * 11 / 150 ≈ 7.33%
        assert result == pytest.approx(100 * 11 / 150)


class TestCorrelationMetrics:
    """Tests for correlation metrics."""

    def test_correlation_perfect(self):
        """Test perfect correlation."""
        from davinci_monet.stats import get_metric

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship
        metric = get_metric("R")
        result = metric.compute(x, y)
        assert result == pytest.approx(1.0)

    def test_correlation_negative(self):
        """Test negative correlation."""
        from davinci_monet.stats import get_metric

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])  # Perfect negative relationship
        metric = get_metric("R")
        result = metric.compute(x, y)
        assert result == pytest.approx(-1.0)

    def test_r2(self, random_arrays):
        """Test R^2."""
        from davinci_monet.stats import get_metric

        x, y = random_arrays
        r_metric = get_metric("R")
        r2_metric = get_metric("R2")
        r = r_metric.compute(x, y)
        r2 = r2_metric.compute(x, y)
        assert r2 == pytest.approx(r**2)

    def test_index_of_agreement(self):
        """Test index of agreement."""
        from davinci_monet.stats import get_metric

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])  # Perfect match
        metric = get_metric("IOA")
        result = metric.compute(x, y)
        assert result == pytest.approx(1.0)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        from davinci_monet.stats import get_metric

        x = np.array([])
        y = np.array([])
        metric = get_metric("MB")
        result = metric.compute(x, y)
        assert np.isnan(result)

    def test_nan_handling(self):
        """Test NaN handling."""
        from davinci_monet.stats import get_metric

        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([1, 2, 3, np.nan, 5])
        metric = get_metric("N")
        result = metric.compute(x, y)
        # Only 3 valid pairs
        assert result == 3

    def test_zero_variance(self):
        """Test handling of zero variance."""
        from davinci_monet.stats import get_metric

        x = np.array([5, 5, 5, 5, 5])
        y = np.array([5, 5, 5, 5, 5])
        metric = get_metric("R")
        result = metric.compute(x, y)
        assert np.isnan(result)

    def test_single_value(self):
        """Test single value arrays."""
        from davinci_monet.stats import get_metric

        x = np.array([5])
        y = np.array([6])
        metric = get_metric("MB")
        result = metric.compute(x, y)
        assert result == pytest.approx(1.0)


# =============================================================================
# Calculator Tests
# =============================================================================


class TestStatisticsCalculator:
    """Tests for StatisticsCalculator."""

    def test_basic_calculation(self, paired_dataset):
        """Test basic statistics calculation."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "x_o3", "y_o3")

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 1
        assert "MB" in stats.columns
        assert "RMSE" in stats.columns
        assert "R" in stats.columns

    def test_custom_metrics(self, paired_dataset):
        """Test custom metric selection."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(
            paired_dataset,
            "x_o3",
            "y_o3",
            metrics=["MB", "RMSE"],
        )

        assert set(stats.columns) == {"MB", "RMSE"}

    def test_groupby_site(self, paired_dataset):
        """Test grouping by site."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(
            paired_dataset,
            "x_o3",
            "y_o3",
            groupby="site",
        )

        assert len(stats) == 5  # 5 sites
        assert "site" in stats.columns

    def test_groupby_time_month(self, paired_dataset):
        """Test grouping by month."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(
            paired_dataset,
            "x_o3",
            "y_o3",
            groupby="time.month",
        )

        assert "time.month" in stats.columns
        # All data is in January
        assert stats["time.month"].iloc[0] == 1

    def test_compute_summary(self, paired_dataset):
        """Test summary dictionary output."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        summary = calc.compute_summary(
            paired_dataset,
            "x_o3",
            "y_o3",
            metrics=["MB", "RMSE", "R"],
        )

        assert isinstance(summary, dict)
        assert "MB" in summary
        assert "RMSE" in summary
        assert "R" in summary

    def test_xy_keywords(self, paired_dataset):
        """Test neutral x/y variable keywords."""
        from davinci_monet.stats import StatisticsCalculator

        calc = StatisticsCalculator()
        stats = calc.compute(
            paired_dataset,
            x_var="x_o3",
            y_var="y_o3",
            metrics=["N", "MB"],
        )

        assert int(stats["N"].iloc[0]) == 500
        assert "MB" in stats.columns

    def test_grouped_stats_raises_when_y_missing_x_key(self):
        """A group key present in x but absent from y must raise, not misalign.

        Regression test for the positional ``zip(x_grouped, y_grouped)`` bug:
        with asymmetric group keys, zipping by position silently pairs the
        wrong groups together instead of failing loudly.
        """
        from davinci_monet.core.exceptions import StatisticsError
        from davinci_monet.stats import StatisticsCalculator

        x_data = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims="site",
            coords={"site": ["site_a", "site_b", "site_c"]},
        )
        y_data = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims="site",
            coords={"site": ["site_a", "site_b", "site_d"]},
        )

        calc = StatisticsCalculator()

        with pytest.raises(StatisticsError, match="site_c"):
            calc._compute_single_groupby(x_data, y_data, ["N"], "site", "site")

    def test_grouped_stats_raises_when_x_missing_y_key(self):
        """A group key present in y but absent from x must also raise."""
        from davinci_monet.core.exceptions import StatisticsError
        from davinci_monet.stats import StatisticsCalculator

        x_data = xr.DataArray(
            [1.0, 2.0],
            dims="site",
            coords={"site": ["site_a", "site_b"]},
        )
        y_data = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims="site",
            coords={"site": ["site_a", "site_b", "site_d"]},
        )

        calc = StatisticsCalculator()

        with pytest.raises(StatisticsError, match="site_d"):
            calc._compute_single_groupby(x_data, y_data, ["N"], "site", "site")

    def test_min_samples_preserves_true_sample_count(self):
        """N reports valid-pair count even when other metrics are below the floor."""
        from davinci_monet.stats import StatisticsCalculator, StatisticsConfig

        ds = xr.Dataset(
            {
                "x_o3": ("time", np.array([1.0, 2.0])),
                "y_o3": ("time", np.array([2.0, 3.0])),
            },
            coords={"time": np.arange(2)},
        )
        calc = StatisticsCalculator(StatisticsConfig(min_samples=3))

        stats = calc.compute(ds, "x_o3", "y_o3", metrics=["N", "MB"])

        assert stats["N"].iloc[0] == 2
        assert np.isnan(stats["MB"].iloc[0])


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_statistics(self, paired_dataset):
        """Test calculate_statistics function."""
        from davinci_monet.stats import calculate_statistics

        stats = calculate_statistics(
            paired_dataset,
            "x_o3",
            "y_o3",
        )

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 1

    def test_calculate_statistics_geometry_dataset_keywords(self, paired_dataset):
        """Test convenience wrapper accepts neutral variable keywords."""
        from davinci_monet.stats import calculate_statistics

        stats = calculate_statistics(
            paired_dataset,
            x_var="x_o3",
            y_var="y_o3",
            metrics=["N"],
        )

        assert int(stats["N"].iloc[0]) == 500

    def test_quick_stats(self, random_arrays):
        """Test quick_stats function."""
        from davinci_monet.stats import quick_stats

        x, y = random_arrays
        stats = quick_stats(x, y)

        assert isinstance(stats, dict)
        assert "MB" in stats
        assert "RMSE" in stats
        assert "R" in stats
        # Check positive bias (we added positive offset)
        assert stats["MB"] > 0

    def test_quick_stats_respects_min_samples_floor(self):
        """quick_stats uses the same floor semantics as StatisticsCalculator."""
        from davinci_monet.stats import quick_stats

        stats = quick_stats(np.array([1.0, 2.0]), np.array([2.0, 3.0]), metrics=["N", "MB"])

        assert stats["N"] == 2
        assert np.isnan(stats["MB"])


# =============================================================================
# Output Tests
# =============================================================================


class TestStatisticsFormatter:
    """Tests for StatisticsFormatter."""

    def test_format_dataframe(self, paired_dataset):
        """Test DataFrame formatting."""
        from davinci_monet.stats import StatisticsCalculator, StatisticsFormatter

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "x_o3", "y_o3")

        formatter = StatisticsFormatter()
        formatted = formatter.format_dataframe(stats, transpose=True)

        assert "Stat_ID" in formatted.columns
        assert "Stat_FullName" in formatted.columns

    def test_to_csv(self, paired_dataset, tmp_path):
        """Test CSV output."""
        from davinci_monet.stats import StatisticsCalculator, StatisticsFormatter

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "x_o3", "y_o3")

        formatter = StatisticsFormatter()
        output_path = tmp_path / "stats.csv"
        result_path = formatter.to_csv(stats, output_path)

        assert result_path.exists()

        # Read back and verify
        df = pd.read_csv(result_path)
        assert "Stat_ID" in df.columns

    def test_to_json(self, paired_dataset, tmp_path):
        """Test JSON output."""
        from davinci_monet.stats import StatisticsCalculator, StatisticsFormatter

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "x_o3", "y_o3")

        formatter = StatisticsFormatter()
        output_path = tmp_path / "stats.json"
        result_path = formatter.to_json(stats, output_path)

        assert result_path.exists()

    def test_to_table_image(self, paired_dataset, tmp_path):
        """Test table image output."""
        import matplotlib

        matplotlib.use("Agg")

        from davinci_monet.stats import StatisticsCalculator, StatisticsFormatter

        calc = StatisticsCalculator()
        stats = calc.compute(paired_dataset, "x_o3", "y_o3")

        formatter = StatisticsFormatter()
        output_path = tmp_path / "stats.png"
        result_path = formatter.to_table_image(stats, output_path)

        assert result_path.exists()


class TestOutputFunctions:
    """Tests for output convenience functions."""

    def test_write_statistics_csv(self, paired_dataset, tmp_path):
        """Test write_statistics_csv function."""
        from davinci_monet.stats import calculate_statistics, write_statistics_csv

        stats = calculate_statistics(paired_dataset, "x_o3", "y_o3")
        output_path = tmp_path / "stats.csv"
        result_path = write_statistics_csv(stats, output_path)

        assert result_path.exists()

    def test_format_stats_summary(self, random_arrays):
        """Test format_stats_summary function."""
        from davinci_monet.stats import format_stats_summary, quick_stats

        x, y = random_arrays
        stats = quick_stats(x, y, metrics=["MB", "RMSE", "R"])
        summary = format_stats_summary(stats)

        assert "Mean Bias" in summary
        assert "Root Mean Square Error" in summary
        assert "Correlation Coefficient" in summary

    def test_format_stats_summary_handles_none_and_nan(self):
        """None/NaN metric values render as N/A instead of crashing.

        Regression test: the None-guard was inverted, so a None value entered
        the float-format branch and raised ``TypeError`` for a user-facing
        summary. Both None and NaN must degrade to "N/A".
        """
        import numpy as np

        from davinci_monet.stats import format_stats_summary

        summary = format_stats_summary({"MB": None, "RMSE": np.nan, "R": 0.5})

        assert "MB): N/A" in summary
        assert "RMSE): N/A" in summary
        assert "R): 0.5" in summary

    def test_get_metric_fullname(self):
        """Test get_metric_fullname function."""
        from davinci_monet.stats import get_metric_fullname

        assert get_metric_fullname("MB") == "Mean Bias"
        assert get_metric_fullname("RMSE") == "Root Mean Square Error"
        assert get_metric_fullname("MB", use_spaces=False) == "Mean_Bias"

    def test_create_comparison_table(self, paired_dataset):
        """Test create_comparison_table function."""
        from davinci_monet.stats import calculate_statistics, create_comparison_table

        # Create stats for two "datasets"
        stats1 = calculate_statistics(paired_dataset, "x_o3", "y_o3")
        stats2 = calculate_statistics(paired_dataset, "x_o3", "y_o3")

        comparison = create_comparison_table(
            {"Dataset A": stats1, "Dataset B": stats2},
            metrics=["MB", "RMSE", "R"],
        )

        assert "Dataset A" in comparison.columns
        assert "Dataset B" in comparison.columns
        assert "Stat_ID" in comparison.columns


# =============================================================================
# Registry Tests
# =============================================================================


class TestMetricRegistry:
    """Tests for metric registry."""

    def test_list_metrics(self):
        """Test listing available metrics."""
        from davinci_monet.stats import list_metrics

        metrics = list_metrics()
        assert len(metrics) >= 20
        assert "MB" in metrics
        assert "RMSE" in metrics
        assert "R" in metrics

    def test_standard_metrics(self):
        """Test standard metric set."""
        from davinci_monet.stats import STANDARD_METRICS

        assert "N" in STANDARD_METRICS
        assert "MB" in STANDARD_METRICS
        assert "RMSE" in STANDARD_METRICS
        assert "R" in STANDARD_METRICS

    def test_all_metrics(self):
        """Test all metrics set."""
        from davinci_monet.stats import ALL_METRICS

        assert len(ALL_METRICS) >= 20


# =============================================================================
# Workflow Tests
# =============================================================================


class TestStatsWorkflow:
    """End-to-end workflow tests for the statistics module (calls internal APIs directly)."""

    def test_full_workflow(self, paired_dataset, tmp_path):
        """Test complete statistics workflow."""
        from davinci_monet.stats import (
            calculate_statistics,
            format_stats_summary,
            write_statistics_csv,
        )

        # Calculate statistics
        stats = calculate_statistics(
            paired_dataset,
            "x_o3",
            "y_o3",
            metrics=["N", "MB", "RMSE", "R", "NMB", "NME"],
        )

        # Verify values
        assert stats["N"].iloc[0] == 500  # 100 times * 5 sites
        assert stats["MB"].iloc[0] != 0  # Should have some bias
        assert 0 < stats["R"].iloc[0] < 1  # Should have positive correlation

        # Write to CSV
        csv_path = write_statistics_csv(stats, tmp_path / "stats.csv")
        assert csv_path.exists()

        # Format summary
        summary = format_stats_summary(stats.iloc[0].to_dict())
        assert "Mean Bias" in summary

    def test_grouped_workflow(self, paired_dataset):
        """Test grouped statistics workflow."""
        from davinci_monet.stats import calculate_statistics

        # Group by site
        site_stats = calculate_statistics(
            paired_dataset,
            "x_o3",
            "y_o3",
            groupby="site",
            metrics=["N", "MB", "RMSE"],
        )

        assert len(site_stats) == 5
        assert all(site_stats["N"] == 100)  # 100 times per site
