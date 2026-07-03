"""Tests for dataset temporal averaging.

Tests for the module-level :func:`resample_dataset` with ``min_count`` and
``track_count``. These features enable averaging high-frequency datasets
(e.g. sub-hourly Pandora) to match model output resolution (e.g. hourly). The
current ``GeometryData.resample_data`` wrapper was removed; the unified source
loader resamples bare datasets through ``resample_dataset`` directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.datasets.base import resample_dataset


class TestTemporalAveraging:
    """Tests for dataset temporal averaging."""

    @pytest.fixture
    def high_freq_geometry(self) -> xr.Dataset:
        """Create sub-hourly dataset data (every 10 minutes)."""
        # 6 datasets per hour for 4 hours = 24 data points
        times = pd.date_range("2024-01-01", periods=24, freq="10min")
        return xr.Dataset(
            {
                "no2_column": (["time"], np.arange(24, dtype=float) + 10.0),
                "o3_column": (["time"], np.arange(24, dtype=float) + 50.0),
            },
            coords={
                "time": times,
                "latitude": 37.5,
                "longitude": -122.5,
            },
        )

    @pytest.fixture
    def high_freq_geometry_with_gaps(self) -> xr.Dataset:
        """Create sub-hourly data with some gaps (varying geometry per hour)."""
        # Hour 0: 6 geometry, Hour 1: 2 geometry, Hour 2: 1 geometry, Hour 3: 6 geometry
        times = pd.to_datetime(
            [
                # Hour 0: full coverage (6 geometry)
                "2024-01-01 00:00",
                "2024-01-01 00:10",
                "2024-01-01 00:20",
                "2024-01-01 00:30",
                "2024-01-01 00:40",
                "2024-01-01 00:50",
                # Hour 1: sparse (2 geometry)
                "2024-01-01 01:15",
                "2024-01-01 01:45",
                # Hour 2: very sparse (1 geometry)
                "2024-01-01 02:30",
                # Hour 3: full coverage (6 geometry)
                "2024-01-01 03:00",
                "2024-01-01 03:10",
                "2024-01-01 03:20",
                "2024-01-01 03:30",
                "2024-01-01 03:40",
                "2024-01-01 03:50",
            ]
        )
        n_points = len(times)
        return xr.Dataset(
            {
                "no2_column": (["time"], np.arange(n_points, dtype=float) + 10.0),
            },
            coords={
                "time": times,
                "latitude": 37.5,
                "longitude": -122.5,
            },
        )

    def test_resample_to_hourly(self, high_freq_geometry: xr.Dataset) -> None:
        """Test resampling high-frequency geometry to hourly."""
        result = resample_dataset(high_freq_geometry, "h")

        # 24 10-minute geometry -> 4 hourly averages
        assert len(result["time"]) == 4

        # Check that values are means of 6 geometry each
        # First hour: mean of 10, 11, 12, 13, 14, 15 = 12.5
        assert result["no2_column"].values[0] == pytest.approx(12.5)

    def test_resample_to_30min(self, high_freq_geometry: xr.Dataset) -> None:
        """Test resampling to 30-minute averages."""
        result = resample_dataset(high_freq_geometry, "30min")

        # 24 10-minute geometry -> 8 30-minute averages
        assert len(result["time"]) == 8

    def test_min_geometry_count_filtering(self, high_freq_geometry_with_gaps: xr.Dataset) -> None:
        """Test that averages with insufficient geometry are set to NaN."""
        result = resample_dataset(high_freq_geometry_with_gaps, "h", min_count=3)

        no2 = result["no2_column"].values

        # Hour 0: 6 geometry >= 3, should have value
        assert not np.isnan(no2[0])
        # Hour 1: 2 geometry < 3, should be NaN
        assert np.isnan(no2[1])
        # Hour 2: 1 geometry < 3, should be NaN
        assert np.isnan(no2[2])
        # Hour 3: 6 geometry >= 3, should have value
        assert not np.isnan(no2[3])

    def test_geometry_count_tracking(self, high_freq_geometry_with_gaps: xr.Dataset) -> None:
        """Test sample_count variable is added when requested."""
        result = resample_dataset(high_freq_geometry_with_gaps, "h", track_count=True)

        assert "sample_count" in result.data_vars

        counts = result["sample_count"].values
        assert counts[0] == 6  # Hour 0
        assert counts[1] == 2  # Hour 1
        assert counts[2] == 1  # Hour 2
        assert counts[3] == 6  # Hour 3

    def test_geometry_count_with_min_count(self, high_freq_geometry_with_gaps: xr.Dataset) -> None:
        """Test sample_count is added when using min_count together with track_count."""
        result = resample_dataset(high_freq_geometry_with_gaps, "h", min_count=3, track_count=True)

        # sample_count should still be present
        assert "sample_count" in result.data_vars

        # Values should be filtered by min_count
        no2 = result["no2_column"].values
        assert np.isnan(no2[1])  # 2 geometry < 3
        assert np.isnan(no2[2])  # 1 geometry < 3

    def test_resample_preserves_scalar_coords(self, high_freq_geometry: xr.Dataset) -> None:
        """Test that scalar lat/lon coordinates are preserved."""
        result = resample_dataset(high_freq_geometry, "h")

        assert "latitude" in result.coords
        assert "longitude" in result.coords
        assert float(result.coords["latitude"]) == pytest.approx(37.5)
        assert float(result.coords["longitude"]) == pytest.approx(-122.5)

    def test_resample_with_nans(self) -> None:
        """Test resampling data that contains NaN values."""
        times = pd.date_range("2024-01-01", periods=12, freq="10min")
        data = np.arange(12, dtype=float) + 10.0
        data[3] = np.nan  # Add a NaN in first hour
        data[4] = np.nan  # Add another NaN in first hour

        ds = xr.Dataset(
            {"no2_column": (["time"], data)},
            coords={"time": times, "latitude": 37.5, "longitude": -122.5},
        )
        result = resample_dataset(ds, "h", track_count=True)

        # First hour should have 4 valid geometry (6 - 2 NaN)
        # count() counts non-NaN, so this should work correctly
        counts = result["sample_count"].values
        assert counts[0] == 4  # 6 minus 2 NaN

    def test_resample_multiple_variables(self, high_freq_geometry: xr.Dataset) -> None:
        """Test resampling with multiple data variables."""
        result = resample_dataset(high_freq_geometry, "h", min_count=3, track_count=True)

        assert "no2_column" in result.data_vars
        assert "o3_column" in result.data_vars

        # With more than one data variable, the diagnostic count is emitted
        # per variable rather than as a single (potentially misleading)
        # shared "sample_count".
        assert "sample_count" not in result.data_vars
        assert "sample_count_no2_column" in result.data_vars
        assert "sample_count_o3_column" in result.data_vars

        # Both variables should be resampled
        assert len(result["no2_column"]) == 4
        assert len(result["o3_column"]) == 4

    def test_min_count_uses_each_variables_own_valid_count(self) -> None:
        """A sparse first variable must not mask valid bins for later variables."""
        times = pd.date_range("2024-01-01", periods=6, freq="10min")
        ds = xr.Dataset(
            {
                "no2_column": ("time", np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan])),
                "o3_column": ("time", np.arange(6, dtype=float) + 10.0),
            },
            coords={"time": times},
        )

        result = resample_dataset(ds, "h", min_count=3, track_count=True)

        assert np.isnan(result["no2_column"].isel(time=0))
        assert result["o3_column"].isel(time=0).item() == pytest.approx(12.5)

        # Per-variable sample counts describe each variable's own coverage,
        # not a max-across-variables value that overstates the sparser one.
        assert result["sample_count_no2_column"].isel(time=0).item() == 1
        assert result["sample_count_o3_column"].isel(time=0).item() == 6


class TestTemporalAveragingEdgeCases:
    """Edge case tests for temporal averaging."""

    def test_resample_no_time_dim(self) -> None:
        """Test resampling is skipped when no time dimension."""
        ds = xr.Dataset(
            {"no2_column": (["site"], np.arange(5, dtype=float))},
            coords={"site": np.arange(5)},
        )
        result = resample_dataset(ds, "h")

        # Should be unchanged
        assert "time" not in result.dims
        assert len(result["site"]) == 5

    def test_resample_excludes_coordinate_vars(self) -> None:
        """Test that lat/lon/alt aren't included in count calculation."""
        times = pd.date_range("2024-01-01", periods=6, freq="10min")
        ds = xr.Dataset(
            {
                "no2_column": (["time"], np.arange(6, dtype=float)),
                "latitude": (["time"], np.full(6, 37.5)),  # As variable not coord
                "longitude": (["time"], np.full(6, -122.5)),
            },
            coords={"time": times},
        )
        result = resample_dataset(ds, "h", track_count=True)

        # sample_count should be based on no2_column, not lat/lon
        assert "sample_count" in result.data_vars
        assert result["sample_count"].values[0] == 6

    def test_resample_daily(self) -> None:
        """Test daily resampling."""
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        ds = xr.Dataset(
            {"no2_column": (["time"], np.arange(48, dtype=float))},
            coords={"time": times, "latitude": 37.5, "longitude": -122.5},
        )
        result = resample_dataset(ds, "D", min_count=12, track_count=True)

        # 48 hourly -> 2 daily averages
        assert len(result["time"]) == 2
        # Each day has 24 geometry
        assert result["sample_count"].values[0] == 24
        assert result["sample_count"].values[1] == 24
