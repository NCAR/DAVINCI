"""Plot registry and factory functions for DAVINCI.

This module provides the plotting registry and convenience functions
for creating plotters by name.

Example usage:
    # Get a plotter by name
    plotter = get_plotter("timeseries")
    fig = plotter.render(build_series(paired_data, "x_o3", "y_o3"))

    # List available plotters
    print(list_plotters())

    # Register a custom plotter
    @register_plotter("custom")
    class CustomPlotter(BasePlotter):
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from davinci_monet.core.registry import Registry, plotter_registry
from davinci_monet.plots.contracts import (
    ALL_PLOT_TYPES,
    MULTI_SOURCE_PLOTS,
    PAIRWISE_PLOTS,
    SINGLE_SOURCE_PLOTS,
    SPATIAL_PLOTS,
    SPECIALIZED_PLOTS,
    STATISTICAL_PLOTS,
    TEMPORAL_PLOTS,
    PlotArity,
)
from davinci_monet.plots.contracts import plot_category as _plot_category

if TYPE_CHECKING:
    from davinci_monet.plots.base import BasePlotter, PlotConfig

T = TypeVar("T")


def register_plotter(
    name: str,
    *,
    replace: bool = False,
    arity: PlotArity | str | None = None,
    category: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a plotter class.

    Parameters
    ----------
    name
        Unique name for the plotter (e.g., 'timeseries', 'scatter').
    replace
        If True, allow replacing an existing registration.
    arity
        Optional input-shape contract stored on the plotter class as
        ``plot_arity``.
    category
        Optional public plot category stored on the plotter class as
        ``plot_category``.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @register_plotter("my_plot")
    ... class MyPlotter(BasePlotter):
    ...     name = "my_plot"
    ...     def render(self, series, **kwargs):
    ...         ...
    """

    def decorator(plotter_cls: type[T]) -> type[T]:
        if arity is not None:
            setattr(plotter_cls, "plot_arity", arity)
        if category is not None:
            setattr(plotter_cls, "plot_category", category)
        return plotter_registry.register(name, replace=replace)(plotter_cls)

    return decorator


def get_plotter_class(name: str) -> type[BasePlotter]:
    """Get a plotter class by name.

    Parameters
    ----------
    name
        Plotter name.

    Returns
    -------
    type[BasePlotter]
        The plotter class.

    Raises
    ------
    ComponentNotFoundError
        If plotter is not registered.
    """
    return plotter_registry.get(name)


def get_plotter(
    name: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> BasePlotter:
    """Get a configured plotter instance by name.

    Parameters
    ----------
    name
        Plotter name (e.g., 'timeseries', 'scatter', 'taylor').
    config
        Plot configuration. Can be PlotConfig or dict.
    **kwargs
        Additional arguments passed to plotter constructor.

    Returns
    -------
    BasePlotter
        Configured plotter instance.

    Examples
    --------
    >>> plotter = get_plotter("timeseries", config={"vmin": 0, "vmax": 100})
    >>> fig = plotter.render(build_series(data, "x_o3", "y_o3"))
    """
    from davinci_monet.plots.base import PlotConfig

    plotter_cls = get_plotter_class(name)

    # Convert dict to PlotConfig if needed
    if isinstance(config, dict):
        config = PlotConfig.from_dict(config)

    return plotter_cls(config=config, **kwargs)


def list_plotters() -> list[str]:
    """List all registered plotter names.

    Returns
    -------
    list[str]
        Sorted list of plotter names.
    """
    return plotter_registry.list()


def has_plotter(name: str) -> bool:
    """Check if a plotter is registered.

    Parameters
    ----------
    name
        Plotter name to check.

    Returns
    -------
    bool
        True if plotter is registered.
    """
    return name in plotter_registry


# =============================================================================
# Plot Type Categories
# =============================================================================


def get_plot_category(name: str) -> str | None:
    """Get the category for a plot type.

    Parameters
    ----------
    name
        Plot type name.

    Returns
    -------
    str | None
        Category name, or None if not a standard type.
    """
    return _plot_category(name)


# Re-export the registry for direct access
__all__ = [
    "plotter_registry",
    "register_plotter",
    "get_plotter",
    "get_plotter_class",
    "list_plotters",
    "has_plotter",
    "get_plot_category",
    "SINGLE_SOURCE_PLOTS",
    "PAIRWISE_PLOTS",
    "MULTI_SOURCE_PLOTS",
    "TEMPORAL_PLOTS",
    "STATISTICAL_PLOTS",
    "SPATIAL_PLOTS",
    "SPECIALIZED_PLOTS",
    "ALL_PLOT_TYPES",
]
