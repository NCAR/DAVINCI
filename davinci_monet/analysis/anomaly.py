"""Departure of a source variable from its own climatology.

Anomalies exist to make a trend legible. In a raw four-decade record the
seasonal cycle is usually the largest signal by an order of magnitude --
surface shortwave at a polar site swings ~450 W m-2 between solstices, which
buries a decadal drift of a few W m-2 entirely. Removing a per-group
climatology leaves the departure, on an axis scaled to it.

The climatology is computed over a *baseline window* and applied to the whole
record, so the reference is a fixed period rather than one that drifts with
the trend under examination.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.analysis.base import DerivedAnalysis
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry

if TYPE_CHECKING:
    from davinci_monet.config.schema import AnomalySpec

logger = logging.getLogger(__name__)

_GROUP_COORD = {"month": "month", "dayofyear": "dayofyear"}


def _as_timestamp(value: object) -> pd.Timestamp | None:
    """Coerce a spec bound to a Timestamp, or None when unset."""
    if value is None:
        return None
    return pd.Timestamp(value)


def _input_geometry(data: xr.Dataset) -> DataGeometry:
    """Read the input's geometry so the anomaly can preserve it.

    An anomaly is shape-preserving, so its geometry is whatever came in. Every
    loaded source and every prior analysis output sets ``attrs['geometry']``
    (``load.py``, ``stages/analyses.py``), so a missing attr means the input
    never went through those paths and guessing would silently mis-route the
    pairing strategy later.
    """
    raw = data.attrs.get("geometry")
    if raw is None:
        raise ValueError(
            "anomaly input has no attrs['geometry']; cannot preserve the "
            "geometry of a source that never declared one"
        )
    name = str(raw).upper()
    if name not in DataGeometry.__members__:
        raise ValueError(f"anomaly input declares unknown geometry {raw!r}")
    return DataGeometry[name]


@analysis_registry.register("anomaly")
class AnomalyAnalysis(DerivedAnalysis):
    """Subtract a baseline-period climatology from a source variable."""

    name = "anomaly"
    long_name = "Anomaly from Climatology"
    #: Placeholder only -- ``analyze`` overwrites this per instance with the
    #: input's geometry, because the transform preserves shape.
    output_geometry = DataGeometry.POINT

    def analyze(self, data: xr.Dataset, spec: "AnomalySpec") -> xr.Dataset:
        if spec.variable not in data.data_vars:
            raise ValueError(
                f"anomaly variable '{spec.variable}' not in source "
                f"'{spec.source}' (has: {sorted(map(str, data.data_vars))})"
            )
        da = data[spec.variable]
        if "time" not in da.dims:
            raise ValueError(
                f"anomaly requires a time dimension; '{spec.variable}' has dims {da.dims}"
            )

        self.output_geometry = _input_geometry(data)

        baseline = self._baseline(da, spec)
        anomaly, climatology = self._departure(da, baseline, spec)
        anomaly = self._smooth(anomaly, spec)

        out = xr.Dataset(
            {
                spec.variable: self._label(anomaly, da, spec),
                f"{spec.variable}_climatology": self._label_climatology(climatology, da, spec),
            }
        )
        out.attrs.update(data.attrs)
        out.attrs["anomaly_variable"] = spec.variable
        self._warn_on_empty_groups(climatology, spec)
        return out

    # -- baseline ---------------------------------------------------------

    @staticmethod
    def _baseline(da: xr.DataArray, spec: "AnomalySpec") -> xr.DataArray:
        """Restrict to the baseline window, failing loudly on no overlap.

        An empty baseline yields an all-NaN climatology and therefore an
        all-NaN anomaly -- a blank plot with no error. That is the failure
        worth being noisy about, so it raises here instead.
        """
        start = _as_timestamp(spec.baseline_start)
        end = _as_timestamp(spec.baseline_end)
        if start is None and end is None:
            return da

        window = da.sel(time=slice(start, end))
        if window.sizes.get("time", 0) == 0:
            times = pd.to_datetime(da["time"].values)
            raise ValueError(
                f"anomaly baseline window "
                f"[{start.date() if start else '-inf'}, {end.date() if end else '+inf'}] "
                f"selects no times from '{spec.variable}', whose record runs "
                f"{times.min().date()} to {times.max().date()}"
            )
        return window

    # -- departure --------------------------------------------------------

    @staticmethod
    def _departure(
        da: xr.DataArray, baseline: xr.DataArray, spec: "AnomalySpec"
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return (anomaly over the full record, climatology over the baseline)."""
        if spec.climatology == "none":
            climatology = baseline.mean("time", skipna=True)
            return da - climatology, climatology

        group = _GROUP_COORD[spec.climatology]
        climatology = baseline.groupby(f"time.{group}").mean("time", skipna=True)
        anomaly = da.groupby(f"time.{group}") - climatology
        # groupby arithmetic leaves the grouping key behind as a scalar-per-step
        # coord; it is redundant with `time` and confuses downstream selection.
        return anomaly.drop_vars(group, errors="ignore"), climatology

    # -- smoothing --------------------------------------------------------

    @staticmethod
    def _smooth(anomaly: xr.DataArray, spec: "AnomalySpec") -> xr.DataArray:
        """Apply the centred rolling mean, if one was configured.

        Monthly anomalies over a multi-decade record are dominated by
        month-to-month weather, which hides the decadal signal the anomaly was
        computed to expose. A centred running mean over one full seasonal
        period removes it without shifting features in time, which a trailing
        mean would.
        """
        if not spec.smooth:
            return anomaly
        steps = int(anomaly.sizes.get("time", 0))
        if spec.smooth > steps:
            raise ValueError(
                f"anomaly smooth window ({spec.smooth}) exceeds the "
                f"{steps}-step time axis of '{spec.variable}'"
            )
        smoothed = anomaly.rolling(time=spec.smooth, center=True).mean()
        smoothed.attrs.update(anomaly.attrs)
        smoothed.attrs["smoothing"] = f"{spec.smooth}-step centred running mean"
        return smoothed

    # -- labelling --------------------------------------------------------

    @staticmethod
    def _quantity(da: xr.DataArray, spec: "AnomalySpec") -> str:
        return str(da.attrs.get("display_name") or da.attrs.get("long_name") or spec.variable)

    def _label(
        self, anomaly: xr.DataArray, source: xr.DataArray, spec: "AnomalySpec"
    ) -> xr.DataArray:
        """Carry units through and mark the quantity as an anomaly.

        Units are unchanged by subtraction, so they are copied verbatim; only
        the name changes, and it changes for both ``display_name`` (which the
        labeling module prefers) and ``long_name``.
        """
        anomaly = anomaly.copy()
        anomaly.attrs.update(source.attrs)
        quantity = f"{self._quantity(source, spec)} Anomaly"
        anomaly.attrs["display_name"] = quantity
        anomaly.attrs["long_name"] = quantity
        anomaly.attrs["climatology"] = spec.climatology
        start = _as_timestamp(spec.baseline_start)
        end = _as_timestamp(spec.baseline_end)
        anomaly.attrs["baseline"] = (
            f"{start.date() if start else 'record start'} to "
            f"{end.date() if end else 'record end'}"
        )
        return anomaly

    def _label_climatology(
        self, climatology: xr.DataArray, source: xr.DataArray, spec: "AnomalySpec"
    ) -> xr.DataArray:
        climatology = climatology.copy()
        climatology.attrs.update(source.attrs)
        quantity = f"{self._quantity(source, spec)} Climatology"
        climatology.attrs["display_name"] = quantity
        climatology.attrs["long_name"] = quantity
        return climatology

    # -- diagnostics ------------------------------------------------------

    @staticmethod
    def _warn_on_empty_groups(climatology: xr.DataArray, spec: "AnomalySpec") -> None:
        """Warn when some climatology groups had no baseline samples.

        A NaN climatology group silently NaNs out every step that maps to it --
        e.g. a solar record that starts in 1984 under a baseline beginning in
        1981 loses nothing, but one whose baseline misses a whole month loses
        that month across all forty years.
        """
        empty = int(np.isnan(np.asarray(climatology.values)).sum())
        if empty:
            logger.warning(
                "anomaly climatology for '%s' has %d empty %s group cell(s); "
                "the corresponding anomalies will be NaN",
                spec.variable,
                empty,
                spec.climatology,
            )
