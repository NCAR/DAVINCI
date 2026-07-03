"""Tests for synthetic source-data generators."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.tests.synthetic.datasets import (
    DatasetConfig,
    create_3d_dataset,
    create_dataset_dataset,
    create_dataset_dataset_from_config,
    create_surface_dataset,
    create_variable_field,
)
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig


class TestCreateDatasetDataset:
    """Tests for create_dataset_dataset function."""

    def test_default_dataset(self) -> None:
        """Test creating dataset with defaults."""
        ds = create_dataset_dataset()

        assert isinstance(ds, xr.Dataset)
        assert "O3" in ds
        assert "PM25" in ds
        assert "time" in ds.dims
        assert "lat" in ds.dims
        assert "lon" in ds.dims

    def test_custom_variables(self) -> None:
        """Test creating dataset with custom variables."""
        ds = create_dataset_dataset(variables=["NO2", "CO", "SO2"])

        assert "NO2" in ds
        assert "CO" in ds
        assert "SO2" in ds
        assert "O3" not in ds

    def test_surface_only(self) -> None:
        """Test creating surface-only dataset (no levels)."""
        ds = create_dataset_dataset(n_levels=0)

        assert "level" not in ds.dims
        assert ds["O3"].dims == ("time", "lat", "lon")

    def test_3d_with_levels(self) -> None:
        """Test creating 3D dataset with levels."""
        ds = create_dataset_dataset(n_levels=20)

        assert "level" in ds.dims
        assert len(ds.level) == 20
        assert ds["O3"].dims == ("time", "level", "lat", "lon")

    def test_custom_domain(self) -> None:
        """Test creating dataset with custom domain."""
        domain = Domain(lon_min=-100, lon_max=-90, lat_min=30, lat_max=40, n_lon=21, n_lat=11)
        ds = create_dataset_dataset(domain=domain)

        assert len(ds.lon) == 21
        assert len(ds.lat) == 11
        assert float(ds.lon.min()) == -100.0
        assert float(ds.lon.max()) == -90.0

    def test_custom_time(self) -> None:
        """Test creating dataset with custom time config."""
        time_config = TimeConfig(start="2024-06-01", end="2024-06-02", freq="3h")
        ds = create_dataset_dataset(time_config=time_config)

        assert len(ds.time) == 9  # 24 hours / 3 hours + 1

    def test_reproducibility(self) -> None:
        """Test that same seed produces same data."""
        ds1 = create_dataset_dataset(seed=42)
        ds2 = create_dataset_dataset(seed=42)

        xr.testing.assert_equal(ds1, ds2)

    def test_different_seeds(self) -> None:
        """Test that different seeds produce different data."""
        ds1 = create_dataset_dataset(seed=42)
        ds2 = create_dataset_dataset(seed=43)

        # Data should be different
        assert not np.allclose(ds1["O3"].values, ds2["O3"].values)

    def test_global_attributes(self) -> None:
        """Test dataset has expected global attributes."""
        ds = create_dataset_dataset()

        assert "title" in ds.attrs
        assert "source" in ds.attrs
        assert "Conventions" in ds.attrs

    def test_variable_attributes(self) -> None:
        """Test variables have proper attributes."""
        ds = create_dataset_dataset(variables=["O3"])

        assert "units" in ds["O3"].attrs
        assert "long_name" in ds["O3"].attrs
        assert ds["O3"].attrs["units"] == "ppbv"


class TestCreateSurfaceDataset:
    """Tests for create_surface_dataset convenience function."""

    def test_creates_surface_data(self) -> None:
        """Test creates surface-only data."""
        ds = create_surface_dataset()

        assert "level" not in ds.dims
        assert "time" in ds.dims
        assert "lat" in ds.dims
        assert "lon" in ds.dims


class TestCreate3DDataset:
    """Tests for create_3d_dataset convenience function."""

    def test_creates_3d_data(self) -> None:
        """Test creates 3D data with levels."""
        ds = create_3d_dataset(n_levels=15)

        assert "level" in ds.dims
        assert len(ds.level) == 15


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = DatasetConfig()

        assert isinstance(config.domain, Domain)
        assert isinstance(config.time_config, TimeConfig)
        assert "O3" in config.variables

    def test_create_from_config(self) -> None:
        """Test creating dataset from config object."""
        config = DatasetConfig(
            variables=["NO2"],
            n_levels=10,
            seed=123,
        )
        ds = create_dataset_dataset_from_config(config)

        assert "NO2" in ds
        assert "level" in ds.dims
        assert len(ds.level) == 10


class TestCreateVariableField:
    """Tests for create_variable_field function."""

    def test_field_shape(self) -> None:
        """Test field has correct shape."""
        from davinci_monet.tests.synthetic.generators import get_variable_spec

        spec = get_variable_spec("O3")
        field = create_variable_field(spec, (10, 20, 30), ["time", "lat", "lon"], seed=42)

        assert field.shape == (10, 20, 30)
        assert field.dims == ("time", "lat", "lon")

    def test_field_attributes(self) -> None:
        """Test field has variable attributes."""
        from davinci_monet.tests.synthetic.generators import get_variable_spec

        spec = get_variable_spec("PM25")
        field = create_variable_field(spec, (5, 10), ["time", "site"], seed=42)

        assert field.attrs["units"] == "ug/m3"


class TestDataQuality:
    """Tests for data quality and realism."""

    def test_values_in_physical_range(self) -> None:
        """Test values are within physical bounds."""
        ds = create_dataset_dataset(variables=["O3", "PM25", "temperature"])

        # O3 should be positive and reasonable
        assert float(ds["O3"].min()) >= 0
        assert float(ds["O3"].max()) < 300

        # PM25 should be positive
        assert float(ds["PM25"].min()) >= 0

        # Temperature should be reasonable
        assert float(ds["temperature"].min()) > 150  # Above absolute zero
        assert float(ds["temperature"].max()) < 350

    def test_spatial_correlation(self) -> None:
        """Test that data has some spatial correlation."""
        ds = create_dataset_dataset(
            domain=Domain(n_lon=50, n_lat=50),
            time_config=TimeConfig(end="2024-01-01 01:00"),
        )

        # Get first time slice
        o3 = ds["O3"].isel(time=0).values

        # Compute autocorrelation by comparing to shifted version
        diff_horiz = np.abs(o3[:, 1:] - o3[:, :-1]).mean()
        random_diff = np.abs(o3 - np.roll(o3, 10, axis=1)).mean()

        # Adjacent values should be more similar than random shifts
        assert diff_horiz < random_diff
