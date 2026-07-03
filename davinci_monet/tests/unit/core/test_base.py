"""Tests for core base data classes."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.base import (
    PairedData,
    validate_dataset_geometry,
)
from davinci_monet.core.exceptions import DataValidationError
from davinci_monet.core.protocols import DataGeometry


class TestPairedData:
    """Tests for PairedData class."""

    @pytest.fixture
    def paired_dataset(self) -> xr.Dataset:
        """Create a paired dataset for testing.

        Mirrors the ``axis``/``source_label``/``canonical_name`` attrs the
        pairing engine attaches to real paired variables (see
        ``pairing/engine.py:381-395``) so ``get_x``/``get_y``/``get_pair``
        can be checked against the real contract, not just the ``x_``/``y_``
        name-prefix fallback.
        """
        times = np.arange(10)
        sites = np.arange(5)
        ds = xr.Dataset(
            {
                "x_ozone": (["time", "site"], np.random.randn(10, 5) + 40),
                "y_ozone": (["time", "site"], np.random.randn(10, 5) + 42),
                "x_pm25": (["time", "site"], np.random.randn(10, 5) + 10),
                "y_pm25": (["time", "site"], np.random.randn(10, 5) + 12),
            },
            coords={
                "time": times,
                "site": sites,
                "lat": ("site", np.linspace(30, 40, 5)),
                "lon": ("site", np.linspace(-100, -90, 5)),
            },
        )
        ds["x_ozone"].attrs.update(
            {"axis": "x", "source_label": "airnow", "canonical_name": "ozone"}
        )
        ds["y_ozone"].attrs.update({"axis": "y", "source_label": "cmaq", "canonical_name": "ozone"})
        ds["x_pm25"].attrs.update({"axis": "x", "source_label": "airnow", "canonical_name": "pm25"})
        ds["y_pm25"].attrs.update({"axis": "y", "source_label": "cmaq", "canonical_name": "pm25"})
        return ds

    def test_pair_label(self, paired_dataset: xr.Dataset) -> None:
        """Test pair label generation."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        assert paired.pair_label == "airnow_cmaq"

    def test_from_sources_sets_geometry_and_dataset_labels(
        self, paired_dataset: xr.Dataset
    ) -> None:
        """construction is the preferred paired-data API."""
        paired = PairedData.from_sources(
            data=paired_dataset,
            x_source="airnow",
            y_source="cam",
            geometry=DataGeometry.POINT,
            pairing_info={"strategy": "PointStrategy"},
        )

        assert paired.x_source == "airnow"
        assert paired.y_source == "cam"
        assert paired.x_source == "airnow"
        assert paired.y_source == "cam"
        assert paired.pairing_info["strategy"] == "PointStrategy"

    def test_dataset_variables(self, paired_dataset: xr.Dataset) -> None:
        """Test listing dataset variables."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        assert "y_ozone" in paired.y_variables
        assert "y_pm25" in paired.y_variables

    def test_geometry_variables(self, paired_dataset: xr.Dataset) -> None:
        """Test listing geometry variables."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        assert "x_ozone" in paired.x_variables
        assert "x_pm25" in paired.x_variables

    def test_paired_variable_names(self, paired_dataset: xr.Dataset) -> None:
        """Test getting paired variable names."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        pairs = paired.paired_variable_names
        assert ("x_ozone", "y_ozone") in pairs
        assert ("x_pm25", "y_pm25") in pairs

    def test_get_x(self, paired_dataset: xr.Dataset) -> None:
        """Test getting x (geometry) variable."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        # With prefix
        x = paired.get_x("x_ozone")
        xr.testing.assert_equal(x, paired_dataset["x_ozone"])
        assert x.attrs["axis"] == "x"
        assert x.attrs["source_label"] == "airnow"
        # Without prefix
        x = paired.get_x("ozone")
        xr.testing.assert_equal(x, paired_dataset["x_ozone"])
        assert x.attrs["axis"] == "x"
        assert x.attrs["source_label"] == "airnow"

    def test_get_y(self, paired_dataset: xr.Dataset) -> None:
        """Test getting y (dataset) variable."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        # With prefix
        y = paired.get_y("y_ozone")
        xr.testing.assert_equal(y, paired_dataset["y_ozone"])
        assert y.attrs["axis"] == "y"
        assert y.attrs["source_label"] == "cmaq"
        # Without prefix
        y = paired.get_y("ozone")
        xr.testing.assert_equal(y, paired_dataset["y_ozone"])
        assert y.attrs["axis"] == "y"
        assert y.attrs["source_label"] == "cmaq"

    def test_get_pair(self, paired_dataset: xr.Dataset) -> None:
        """Test getting paired arrays."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        x, y = paired.get_pair("ozone")
        xr.testing.assert_equal(x, paired_dataset["x_ozone"])
        xr.testing.assert_equal(y, paired_dataset["y_ozone"])
        assert x.attrs["axis"] == "x"
        assert x.attrs["source_label"] == "airnow"
        assert y.attrs["axis"] == "y"
        assert y.attrs["source_label"] == "cmaq"

    def test_n_points(self, paired_dataset: xr.Dataset) -> None:
        """Test counting data points."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        assert paired.n_points == 50  # 10 times * 5 sites

    def test_to_dataframe(self, paired_dataset: xr.Dataset) -> None:
        """Test conversion to DataFrame."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        df = paired.to_dataframe()
        assert len(df) == 50
        assert "x_ozone" in df.columns
        assert "y_ozone" in df.columns

    def test_subset_time(self, paired_dataset: xr.Dataset) -> None:
        """Test subsetting paired data by time."""
        paired = PairedData(
            data=paired_dataset,
            y_source="cmaq",
            x_source="airnow",
            geometry=DataGeometry.POINT,
        )
        subset = paired.subset_time(start=2, end=5)  # type: ignore[arg-type]
        assert len(subset.data["time"]) == 4


class TestValidateDatasetGeometry:
    """Tests for validate_dataset_geometry function."""

    def test_point_geometry_valid(self) -> None:
        """Test valid POINT geometry."""
        ds = xr.Dataset(
            {"temp": (["time", "site"], np.random.randn(10, 5))},
            coords={"time": np.arange(10), "site": np.arange(5)},
            attrs={"geometry": "POINT"},
        )
        # Should not raise
        validate_dataset_geometry(ds, DataGeometry.POINT)

    def test_track_geometry_valid(self) -> None:
        """Test valid TRACK geometry."""
        ds = xr.Dataset(
            {"temp": (["time"], np.random.randn(100))},
            coords={
                "time": np.arange(100),
                "lat": ("time", np.random.randn(100)),
                "lon": ("time", np.random.randn(100)),
            },
            attrs={"geometry": "TRACK"},
        )
        validate_dataset_geometry(ds, DataGeometry.TRACK)

    def test_point_geometry_requires_site_or_x_dimension_without_attr(self) -> None:
        """A time-only dataset is not point geometry when no geometry attr is present."""
        ds = xr.Dataset(
            {"temp": (["time"], np.random.randn(10))},
            coords={"time": np.arange(10)},
        )
        with pytest.raises(DataValidationError, match="POINT geometry expects"):
            validate_dataset_geometry(ds, DataGeometry.POINT)

    def test_grid_geometry_valid(self) -> None:
        """Test valid GRID geometry."""
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.random.randn(10, 20, 30))},
            coords={
                "time": np.arange(10),
                "lat": np.linspace(-90, 90, 20),
                "lon": np.linspace(-180, 180, 30),
            },
        )
        validate_dataset_geometry(ds, DataGeometry.GRID)

    def test_geometry_mismatch_raises(self) -> None:
        """Test geometry mismatch raises error."""
        ds = xr.Dataset(
            {"temp": (["time"], np.random.randn(10))},
            attrs={"geometry": "TRACK"},
        )
        with pytest.raises(DataValidationError):
            validate_dataset_geometry(ds, DataGeometry.POINT)
