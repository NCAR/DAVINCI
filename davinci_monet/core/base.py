"""Base data classes for DAVINCI.

This module provides the paired-data container and plot-series helpers that
wrap xarray Datasets with dataset-label metadata.

Key Classes:
    - PairedData: Container for paired source data
    - PlotSeries: One plottable series with dataset metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import xarray as xr

from davinci_monet.core.exceptions import DataValidationError, VariableNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeRange


def paired_variable_axis(dataset: xr.Dataset, var_name: str) -> str | None:
    """Return the pairing axis (``"x"``/``"y"``) of a paired variable."""
    if var_name in dataset.data_vars:
        axis = dataset[var_name].attrs.get("axis")
        if axis in ("x", "y"):
            return str(axis)
    lname = str(var_name).lower()
    if lname.startswith("x_"):
        return "x"
    if lname.startswith("y_"):
        return "y"
    return None


def paired_canonical_name(dataset: xr.Dataset, var_name: str) -> str:
    """Canonical name of a paired variable.

    Strips the source-label prefix (from the variable's ``source_label`` attr,
    e.g. ``cam_o3`` -> ``o3``) or the ``x_``/``y_`` prefix. Names with no
    recognized prefix are returned unchanged.
    """
    if var_name in dataset.data_vars:
        canonical = dataset[var_name].attrs.get("canonical_name")
        if canonical:
            return str(canonical)
        source_label = dataset[var_name].attrs.get("source_label")
        if source_label and var_name.startswith(f"{source_label}_"):
            return var_name[len(source_label) + 1 :]
    lname = str(var_name).lower()
    for prefix in ("x_", "y_"):
        if lname.startswith(prefix):
            return var_name[len(prefix) :]
    return var_name


def iter_paired_variable_xy(dataset: xr.Dataset) -> list[tuple[str, str, str]]:
    """Pair x variables with their y counterparts.

    Matched by canonical name (renderer rewire R-5).

    Returns ``(x_var, y_var, canonical)`` triples. One variable
    per axis is used, so dual-named data never double-counts.
    """
    xs: dict[str, str] = {}
    ys: dict[str, str] = {}
    for v in dataset.data_vars:
        name = str(v)
        axis = paired_variable_axis(dataset, name)
        if axis not in ("x", "y"):
            continue
        canonical = paired_canonical_name(dataset, name)
        (xs if axis == "x" else ys).setdefault(canonical, name)
    return [(xs[c], ys[c], c) for c in ys if c in xs]


@dataclass
class PlotSeries:
    """One plottable series from one dataset.

    The value object the unified renderer contract (``render(series)``) consumes.
    ``index`` is the series' 0-based position within its canonical group and is
    the styling hook for palette cycling.

    Attributes
    ----------
    dataset
        The dataset the variable lives in.
    var_name
        The actual variable name in ``dataset`` (e.g. ``cam_o3``/``airnow_o3``/``o3``).
    canonical
        The unprefixed canonical name (e.g. ``o3``).
    axis
        Pairing position (``"x"``/``"y"``) or ``None`` when unpaired.
    source_label
        The source's identity in a unified pair (e.g. ``airnow``/``cam``) or ``None``.
    index
        Position within the canonical group (0-based).
    """

    dataset: xr.Dataset
    var_name: str
    canonical: str
    axis: str | None
    source_label: str | None
    index: int


def iter_canonical_variable_series(dataset: xr.Dataset) -> dict[str, list[PlotSeries]]:
    """Group a dataset's source variables by canonical name into :class:`PlotSeries`.

    N-capable sibling of :func:`iter_paired_variable_xy`: where that returns a
    single ``(x, y)`` pair per canonical, this returns *every*
    source variable for each canonical as an ordered list (1 → single series,
    2 → x + y, N → multi-source overlay). Variables are included
    when they carry a ``source_label`` or a ``axis``. Series preserve
    ``data_vars`` order; ``index`` is the 0-based position within the canonical
    group.
    """
    groups: dict[str, list[PlotSeries]] = {}
    for v in dataset.data_vars:
        name = str(v)
        axis = paired_variable_axis(dataset, name)
        source_label = dataset[name].attrs.get("source_label")
        if source_label is None and axis is None:
            continue
        canonical = paired_canonical_name(dataset, name)
        group = groups.setdefault(canonical, [])
        group.append(
            PlotSeries(
                dataset=dataset,
                var_name=name,
                canonical=canonical,
                axis=axis,
                source_label=str(source_label) if source_label else None,
                index=len(group),
            )
        )
    return groups


@dataclass
class PairedData:
    """Container for paired source data.

    Canonically, paired data has an x source and a y source.
    Attributes
    ----------
    data : xr.Dataset
        The paired dataset with x and y variables.
    y_source : str
        Source label of the y-axis source.
    x_source : str
        Source label of the x-axis source.
    geometry : DataGeometry
        The geometry type of the x source.
    pairing_info : dict[str, Any]
        Metadata about the pairing process.
    """

    data: xr.Dataset
    y_source: str
    x_source: str
    geometry: DataGeometry
    pairing_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sources(
        cls,
        *,
        data: xr.Dataset,
        x_source: str,
        y_source: str,
        geometry: DataGeometry,
        pairing_info: dict[str, Any] | None = None,
    ) -> "PairedData":
        """Construct paired data from x and y source labels."""
        info = dict(pairing_info or {})
        info.setdefault("x_source", x_source)
        info.setdefault("source_label", y_source)
        return cls(
            data=data,
            y_source=y_source,
            x_source=x_source,
            geometry=geometry,
            pairing_info=info,
        )

    @property
    def pair_label(self) -> str:
        """Get combined pair label."""
        return f"{self.x_source}_{self.y_source}"

    @property
    def x_variables(self) -> list[str]:
        """List of x variables in the paired data."""
        return [
            str(v) for v in self.data.data_vars if paired_variable_axis(self.data, str(v)) == "x"
        ]

    @property
    def y_variables(self) -> list[str]:
        """List of y variables in the paired data."""
        return [
            str(v) for v in self.data.data_vars if paired_variable_axis(self.data, str(v)) == "y"
        ]

    @property
    def paired_variable_names(self) -> list[tuple[str, str]]:
        """List of (x_var, y_var) pairs, matched by canonical name."""
        return [(x_var, y_var) for x_var, y_var, _ in iter_paired_variable_xy(self.data)]

    def _resolve_axis_var(self, variable: str, axis: str) -> str | None:
        """Resolve a paired variable for ``axis`` (``"x"``/``"y"``)."""
        wanted = "x" if axis == "x" else "y"
        if variable in self.data.data_vars and paired_variable_axis(self.data, variable) == wanted:
            return variable
        prefix = "x_" if wanted == "x" else "y_"
        prefixed = variable if variable.startswith(prefix) else f"{prefix}{variable}"
        if prefixed in self.data.data_vars and paired_variable_axis(self.data, prefixed) == wanted:
            return prefixed
        target = paired_canonical_name(self.data, variable)
        for v in self.data.data_vars:
            name = str(v)
            if (
                paired_variable_axis(self.data, name) == wanted
                and paired_canonical_name(self.data, name) == target
            ):
                return name
        return None

    def get_x(self, variable: str) -> xr.DataArray:
        """Get an x (reference) variable."""
        name = self._resolve_axis_var(variable, "x")
        if name is None:
            raise VariableNotFoundError(
                f"X variable '{variable}' not found. " f"Available: {self.x_variables}"
            )
        result: xr.DataArray = self.data[name]
        return result

    def get_y(self, variable: str) -> xr.DataArray:
        """Get a y (comparison) variable."""
        name = self._resolve_axis_var(variable, "y")
        if name is None:
            raise VariableNotFoundError(
                f"Y variable '{variable}' not found. " f"Available: {self.y_variables}"
            )
        result: xr.DataArray = self.data[name]
        return result

    def get_pair(self, variable: str) -> tuple[xr.DataArray, xr.DataArray]:
        """Get both x and y arrays for a variable.

        Parameters
        ----------
        variable
            Base variable name (without prefix).

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            ``(x_array, y_array)``.
        """
        x = self.get_x(variable)
        y = self.get_y(variable)
        return (x, y)

    @property
    def time_range(self) -> TimeRange | None:
        """Get the time range of paired data."""
        if "time" not in self.data.dims:
            return None
        times = self.data["time"].values
        if len(times) == 0:
            return None
        return (times[0], times[-1])

    @property
    def n_points(self) -> int:
        """Get total number of paired data points."""
        # Count non-NaN pairs for first paired variable
        if not self.paired_variable_names:
            return 0
        x_var, y_var = self.paired_variable_names[0]
        x_data = self.data[x_var]
        y_data = self.data[y_var]
        valid = ~x_data.isnull() & ~y_data.isnull()
        return int(valid.sum().values)

    def to_dataframe(self) -> Any:
        """Convert paired data to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with time/location index and paired variables.
        """
        import pandas as pd

        df: pd.DataFrame = self.data.to_dataframe().reset_index()
        return df

    def subset_time(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> PairedData:
        """Subset paired data to a time range.

        Parameters
        ----------
        start
            Start time (inclusive).
        end
            End time (inclusive).

        Returns
        -------
        PairedData
            New PairedData with subsetted data.
        """
        if "time" not in self.data.dims:
            return self

        subset = self.data
        if start is not None:
            subset = subset.sel(time=slice(start, None))
        if end is not None:
            subset = subset.sel(time=slice(None, end))

        return PairedData(
            data=subset,
            y_source=self.y_source,
            x_source=self.x_source,
            geometry=self.geometry,
            pairing_info=self.pairing_info,
        )


def validate_dataset_geometry(
    data: xr.Dataset,
    expected_geometry: DataGeometry,
) -> None:
    """Validate that a dataset has the expected geometry.

    Parameters
    ----------
    data
        Dataset to validate.
    expected_geometry
        Expected geometry type.

    Raises
    ------
    DataValidationError
        If dataset doesn't match expected geometry.
    """
    # Check geometry attribute if present
    if "geometry" in data.attrs:
        actual = data.attrs["geometry"]
        if isinstance(actual, DataGeometry):
            if actual != expected_geometry:
                raise DataValidationError(
                    f"Expected geometry {expected_geometry.name}, got {actual.name}"
                )
            return
        if isinstance(actual, str):
            try:
                actual_geom = DataGeometry[actual.upper()]
                if actual_geom != expected_geometry:
                    raise DataValidationError(
                        f"Expected geometry {expected_geometry.name}, got {actual}"
                    )
                return
            except KeyError:
                pass

    # Validate based on expected dimensions
    dims = set(data.dims)

    if expected_geometry == DataGeometry.POINT:
        # Point: (time, site) or (time, x)
        if not ("time" in dims and ("site" in dims or "x" in dims)):
            raise DataValidationError(f"POINT geometry expects dims (time, site), got {dims}")

    elif expected_geometry == DataGeometry.TRACK:
        # Track: (time,) with lat/lon/alt coords
        if "time" not in dims:
            raise DataValidationError(f"TRACK geometry expects 'time' dimension, got {dims}")

    elif expected_geometry == DataGeometry.PROFILE:
        # Profile: (time, level) with lat/lon coords
        if not ("time" in dims and ("level" in dims or "z" in dims)):
            raise DataValidationError(f"PROFILE geometry expects dims (time, level), got {dims}")

    elif expected_geometry == DataGeometry.SWATH:
        # Swath: (time, scanline, pixel) or similar
        if "time" not in dims:
            raise DataValidationError(f"SWATH geometry expects 'time' dimension, got {dims}")

    elif expected_geometry == DataGeometry.GRID:
        # Grid: (time, lat, lon) or (time, y, x)
        has_spatial = ("lat" in dims or "latitude" in dims or "y" in dims) and (
            "lon" in dims or "longitude" in dims or "x" in dims
        )
        if not has_spatial:
            raise DataValidationError(f"GRID geometry expects spatial dims (lat, lon), got {dims}")
