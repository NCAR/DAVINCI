"""Generic dataset readers and dataset helpers.

This module provides geometry-keyed generic NetCDF readers and
:func:`resample_dataset`, a pure-function temporal resampler used by the source
loader. Dataset sources flow through registered reader classes that return
plain :class:`xarray.Dataset` objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import set_geometry_attr


def resample_dataset(
    data: "xr.Dataset",
    freq: str,
    min_count: int | None = None,
    track_count: bool = False,
) -> "xr.Dataset":
    """Resample a dataset along ``time``, masking sparse bins and optionally counting.

    Resamples bare datasets in the unified source loader: average to ``freq``,
    optionally drop bins with fewer than ``min_count`` datasets, and
    optionally emit an ``sample_count`` variable.
    """
    if "time" not in data.dims:
        return data
    resampler = data.resample(time=freq)
    result = resampler.mean()
    if track_count or min_count is not None:
        data_vars = [v for v in data.data_vars if v not in ("latitude", "longitude", "altitude")]
        if data_vars:
            count_ds = resampler.count()
            counts_by_var = {var: count_ds[var] for var in data_vars if var in count_ds}
            if track_count:
                count_arrays = list(counts_by_var.values())
                if count_arrays:
                    try:
                        result["sample_count"] = xr.concat(
                            count_arrays, dim="_sample_count_var"
                        ).max("_sample_count_var")
                    except ValueError:
                        result["sample_count"] = count_arrays[0]
            if min_count is not None:
                for var in data_vars:
                    if var in result and var in counts_by_var:
                        result[var] = result[var].where(counts_by_var[var] >= min_count)
    return result


class _GenericNetCDFReader:
    """Generic NetCDF source reader keyed only by geometry.

    Base for the generic readers (``pt_sfc``, ``aircraft``, ``profile``,
    ``gridded``). Opens plain NetCDF with no format-specific handling and
    reports a fixed :class:`DataGeometry`.
    """

    _name: str = "generic_dataset"
    _geometry: DataGeometry = DataGeometry.POINT

    @property
    def name(self) -> str:
        """Return reader name."""
        return self._name

    @property
    def geometry(self) -> DataGeometry:
        """Geometry of data this reader returns."""
        return self._geometry

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open one or more generic NetCDF files."""
        files = [str(Path(p).expanduser()) for p in file_paths]
        if not files:
            raise DataNotFoundError(f"No {self._name} dataset files provided")
        try:
            if len(files) == 1:
                ds = xr.open_dataset(files[0])
            else:
                ds = xr.open_mfdataset(files, combine="by_coords", parallel=True)
        except OSError as e:
            raise DataFormatError(f"Failed to open {self._name} files: {e}") from e

        if variables:
            keep_vars = [v for v in variables if v in ds.data_vars]
            ds = ds[keep_vars]

        if time_range and "time" in ds:
            start, end = time_range
            ds = ds.sel(time=slice(start, end))

        set_geometry_attr(ds, self.geometry)
        return ds


@source_registry.register("pt_sfc")
class PointSurfaceReader(_GenericNetCDFReader):
    """Generic point-surface (POINT) reader for unified ``sources:`` configs."""

    _name = "pt_sfc"
    _geometry = DataGeometry.POINT


@source_registry.register("aircraft")
class AircraftReader(_GenericNetCDFReader):
    """Generic aircraft/track (TRACK) NetCDF reader.

    Handles plain-NetCDF track files. For ICARTT ``.ict`` campaign files use
    the dedicated ``icartt`` reader instead.
    """

    _name = "aircraft"
    _geometry = DataGeometry.TRACK


@source_registry.register("profile")
class ProfileReader(_GenericNetCDFReader):
    """Generic vertical-profile (PROFILE) NetCDF reader.

    Handles plain-NetCDF profile files. For ozonesonde campaign files use the
    dedicated ``ozonesonde`` reader instead.
    """

    _name = "profile"
    _geometry = DataGeometry.PROFILE


@source_registry.register("gridded")
class GriddedDatasetReader(_GenericNetCDFReader):
    """Generic gridded (GRID) NetCDF dataset reader.

    Handles plain-NetCDF gridded geometry files. For gridded satellite L3
    products use the dedicated ``satellite_l3`` reader.
    """

    _name = "gridded"
    _geometry = DataGeometry.GRID
