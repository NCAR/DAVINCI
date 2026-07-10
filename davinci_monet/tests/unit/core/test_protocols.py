"""Tests for core protocol definitions.

These tests verify that:
1. Protocol classes are properly defined as runtime_checkable
2. The DataGeometry enum contains expected values
3. Classes implementing protocols are correctly recognized
"""

from __future__ import annotations

from typing import Any

from davinci_monet.core.protocols import (
    DataGeometry,
    PairingStrategy,
)


class TestDataGeometry:
    """Tests for DataGeometry enum."""

    def test_geometry_values_exist(self) -> None:
        """Verify all expected geometry types are defined."""
        assert hasattr(DataGeometry, "POINT")
        assert hasattr(DataGeometry, "TRACK")
        assert hasattr(DataGeometry, "PROFILE")
        assert hasattr(DataGeometry, "SWATH")
        assert hasattr(DataGeometry, "GRID")
        assert hasattr(DataGeometry, "SPECTRUM")
        assert hasattr(DataGeometry, "ARTIFACT")

    def test_geometry_count(self) -> None:
        """Verify exactly 7 geometry types."""
        assert len(DataGeometry) == 7

    def test_geometry_iteration(self) -> None:
        """Verify geometry can be iterated."""
        geometries = list(DataGeometry)
        assert len(geometries) == 7
        assert DataGeometry.POINT in geometries
        assert DataGeometry.GRID in geometries
        assert DataGeometry.ARTIFACT in geometries


class TestPairingStrategyProtocol:
    """Tests for PairingStrategy protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify PairingStrategy is runtime_checkable."""

        class MockPointStrategy:
            @property
            def geometry(self) -> DataGeometry:
                return DataGeometry.POINT

            def pair_sources(
                self,
                x_data: Any,
                y_data: Any,
                **kwargs: Any,
            ) -> Any:
                return None

        strategy = MockPointStrategy()
        assert isinstance(strategy, PairingStrategy)
