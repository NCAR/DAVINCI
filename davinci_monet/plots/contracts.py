"""Canonical plot API contracts.

This module is the single source of truth for plot type arity and public
category metadata. Config validation, registry category helpers, and pipeline
dispatch must consume this module instead of re-declaring plot type sets.
"""

from __future__ import annotations

from collections.abc import Iterator, Set
from enum import Enum
from typing import Any

from davinci_monet.core.exceptions import PlottingError
from davinci_monet.core.registry import plotter_registry


class PlotArity(str, Enum):
    """Supported renderer input shapes."""

    SINGLE_SOURCE = "single_source"
    PAIRWISE = "pairwise"
    MULTI_SOURCE = "multi_source"


def _ensure_default_plotters_registered() -> None:
    if "timeseries" not in plotter_registry:
        import davinci_monet.plots.renderers  # noqa: F401


def _coerce_plot_arity(value: Any) -> PlotArity:
    return value if isinstance(value, PlotArity) else PlotArity(str(value))


class _PlotTypeSet(Set[str]):
    """Live set view over plotter class contract metadata."""

    def __init__(
        self,
        *,
        arity: PlotArity | None = None,
        category: str | None = None,
    ) -> None:
        self._arity = arity
        self._category = category

    def __contains__(self, value: object) -> bool:
        return isinstance(value, str) and value in self._values()

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._values()))

    def __len__(self) -> int:
        return len(self._values())

    def __repr__(self) -> str:
        return repr(self._values())

    def _values(self) -> frozenset[str]:
        _ensure_default_plotters_registered()
        names: list[str] = []
        for name, plotter_cls in plotter_registry.items():
            if self._arity is not None:
                declared_arity = getattr(plotter_cls, "plot_arity", None)
                if declared_arity is None or _coerce_plot_arity(declared_arity) != self._arity:
                    continue
            if self._category is not None:
                declared_category = getattr(plotter_cls, "plot_category", None)
                if declared_category != self._category:
                    continue
            names.append(name)
        return frozenset(names)


SINGLE_SOURCE_PLOTS = _PlotTypeSet(arity=PlotArity.SINGLE_SOURCE)
PAIRWISE_PLOTS = _PlotTypeSet(arity=PlotArity.PAIRWISE)
MULTI_SOURCE_PLOTS = _PlotTypeSet(arity=PlotArity.MULTI_SOURCE)

TEMPORAL_PLOTS = _PlotTypeSet(category="temporal")
STATISTICAL_PLOTS = _PlotTypeSet(category="statistical")
SPATIAL_PLOTS = _PlotTypeSet(category="spatial")
SPECIALIZED_PLOTS = _PlotTypeSet(category="specialized")
ALL_PLOT_TYPES = _PlotTypeSet()


def _registered_plotter_attr(plot_type: str, attr: str) -> Any:
    _ensure_default_plotters_registered()
    plotter_cls = plotter_registry.get_or_none(plot_type)
    if plotter_cls is None:
        return None
    return getattr(plotter_cls, attr, None)


def plot_arity(plot_type: str) -> PlotArity:
    """Return the canonical arity for a registered plot type."""
    declared = _registered_plotter_attr(plot_type, "plot_arity")
    if declared is None:
        raise PlottingError(f"Plot type '{plot_type}' does not declare plot_arity")
    return _coerce_plot_arity(declared)


def plot_category(plot_type: str) -> str | None:
    """Return the public category for a registered plot type, if known."""
    declared = _registered_plotter_attr(plot_type, "plot_category")
    if declared is not None:
        return str(declared)
    return None


def validate_plot_shape(
    *,
    plot_name: str,
    plot_type: str,
    pairs: list[str],
    source: str | None,
    variable: str | None,
) -> list[str]:
    """Return config-shape validation errors for one plot spec."""
    arity = plot_arity(plot_type)
    has_pairs = bool(pairs)
    has_source = source is not None
    has_variable = variable is not None
    has_single = has_source or has_variable

    errors: list[str] = []
    if arity == PlotArity.SINGLE_SOURCE:
        if has_pairs:
            errors.append(
                f"plots.{plot_name}.pairs is invalid for single-source plot '{plot_type}'"
            )
        if not has_source:
            errors.append(
                f"plots.{plot_name}.source is required for single-source plot '{plot_type}'"
            )
        if not has_variable:
            errors.append(
                f"plots.{plot_name}.variable is required for single-source plot '{plot_type}'"
            )
    elif arity == PlotArity.PAIRWISE:
        if not has_pairs:
            errors.append(f"plots.{plot_name}.pairs is required for pairwise plot '{plot_type}'")
        if has_source:
            errors.append(f"plots.{plot_name}.source is invalid for pairwise plot '{plot_type}'")
        if has_variable:
            errors.append(f"plots.{plot_name}.variable is invalid for pairwise plot '{plot_type}'")
    elif arity == PlotArity.MULTI_SOURCE:
        if has_pairs and has_single:
            errors.append(
                f"plots.{plot_name} must use either pairs or source/variable for plot "
                f"'{plot_type}', not both"
            )
        if not has_pairs and not (has_source and has_variable):
            errors.append(
                f"plots.{plot_name} requires pairs or source+variable for plot '{plot_type}'"
            )
        if has_single and not (has_source and has_variable):
            errors.append(
                f"plots.{plot_name} source and variable must be provided together for plot "
                f"'{plot_type}'"
            )
    return errors


__all__ = [
    "PlotArity",
    "SINGLE_SOURCE_PLOTS",
    "PAIRWISE_PLOTS",
    "MULTI_SOURCE_PLOTS",
    "TEMPORAL_PLOTS",
    "STATISTICAL_PLOTS",
    "SPATIAL_PLOTS",
    "SPECIALIZED_PLOTS",
    "ALL_PLOT_TYPES",
    "plot_arity",
    "plot_category",
    "validate_plot_shape",
]
