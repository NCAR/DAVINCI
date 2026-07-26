"""Catalog-driven MODIS/VIIRS reader (L3 grid slice).

Initial vertical slice of the modis_viirs catalog reader design: Level-3
regular-grid atmosphere products (MOD08/MYD08 monthly M3 and daily D3). See
docs/superpowers/specs/2026-06-01-modis-viirs-catalog-readers-design.md.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.datasets.satellite.catalog import ProductEntry, get_catalog
from davinci_monet.io.reader_utils import apply_hdf4_scale, set_geometry_attr, validate_file_list

_DATE_TOKEN = re.compile(r"\.A(\d{7})\.")  # ".A2024032." -> 2024032
_HDF4_PACKING_ATTRS = frozenset(
    {
        "_FillValue",
        "add_offset",
        "missing_value",
        "scale_factor",
        "valid_range",
    }
)


@source_registry.register("modis_viirs")
class MODISVIIRSReader:
    """Catalog-driven reader for MODIS/VIIRS products (L3 grid path)."""

    def __init__(self) -> None:
        self.geometry: DataGeometry = DataGeometry.GRID

    @property
    def name(self) -> str:
        return "modis_viirs"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        product: str | None = None,
        level: str | None = None,
        time_range: tuple[Any, Any] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MODIS/VIIRS L3 grid files.

        Parameters
        ----------
        file_paths
            Paths to MODIS/VIIRS HDF4 or NetCDF files.
        variables
            Catalog display names of variables to load (e.g. ``"aod_550nm"``).
            Passing ``None`` or ``["*"]`` loads all cataloged variables.
        product
            Catalog product ID (e.g. ``"MOD08_M3"``). Required.
        level
            Optional level string for validation against catalog entry.
        time_range
            Accepted for interface compatibility but not applied here: time
            filtering is performed by the pipeline after ``open()``, so the
            reader does not apply it.
        progress_callback
            Optional callable invoked for each file as
            ``progress_callback(idx, total, filename)`` where ``idx`` is
            1-based.  Use this to display per-file loading progress.
        **kwargs
            Ignored; present for interface compatibility.

        Returns
        -------
        xr.Dataset
            Dataset with display-name variables, ``lat``/``lon`` coords,
            and a ``time`` dimension parsed from filenames.
        """
        if not product:
            raise ValueError("modis_viirs source requires a 'product' (e.g. MOD08_M3).")
        entry = get_catalog().resolve(product)  # raises UnknownProductError if absent
        if level and level.upper() != entry.level.upper():
            raise ValueError(
                f"Configured level '{level}' != catalog level '{entry.level}' for {product}."
            )
        if entry.geometry.upper() != "GRID":
            raise NotImplementedError(
                f"modis_viirs slice supports GRID products only; " f"{product} is {entry.geometry}."
            )

        files = validate_file_list(file_paths, source_label="MODIS/VIIRS")

        total = len(files)
        per_file: list[xr.Dataset | None] = []
        for idx, f in enumerate(files, start=1):
            if progress_callback is not None:
                progress_callback(idx, total, f.name)
            per_file.append(self._open_one(f, variables, entry))
        valid = [d for d in per_file if d is not None]
        if not valid:
            raise DataNotFoundError(
                f"No valid MODIS/VIIRS data found (0 of {len(files)} files opened successfully)"
            )

        ds = xr.concat(valid, dim="time").sortby("time")
        if entry.cadence == "daily":
            time = pd.DatetimeIndex(ds["time"].values)
            duplicate = time[time.duplicated()].unique()
            if len(duplicate):
                labels = ", ".join(stamp.strftime("%Y-%m-%d") for stamp in duplicate)
                raise ValueError(f"{entry.product_id}: duplicate daily timestamps: {labels}")
        self.geometry = DataGeometry.GRID
        set_geometry_attr(ds, DataGeometry.GRID)
        ds.attrs.update(
            product_id=entry.product_id,
            instrument=entry.instrument,
            platform=entry.platform,
            level=entry.level,
            daac=entry.daac,
            collection=entry.collection,
            cadence=entry.cadence,
        )
        return ds

    def _open_one(
        self, fpath: Path, variables: Sequence[str] | None, entry: ProductEntry
    ) -> xr.Dataset | None:
        """Open a single MODIS/VIIRS granule, loading only the requested SDS.

        Dispatch strategy:

        * ``.hdf`` files (real MODIS/VIIRS HDF4): use ``pyhdf`` to read only
          the requested SDS plus the grid-axis variables (``XDim``/``YDim``).
          This avoids constructing ~1,144 xarray variables for the many
          histogram SDS we don't need, saving ~140 ms per file vs a full
          ``xr.open_dataset`` call.
        * All other files (``.nc``, ``.nc4``, …): fall back to the original
          ``xr.open_dataset`` + subset approach, which is needed for the
          synthetic NetCDF fixtures used in unit tests (``pyhdf`` cannot open
          NetCDF files).
        """
        # Resolve which catalog variables to keep (display names -> SDS names).
        wanted: list[str] | None = list(variables) if variables else None
        if wanted is None or wanted == ["*"]:
            selected = entry.variables
        else:
            selected = []
            for name in wanted:
                v = entry.variable_by_display(name) or entry.variable_by_sds(name)
                if v is None:
                    warnings.warn(
                        f"{entry.product_id}: variable '{name}' not in catalog", UserWarning
                    )
                    continue
                selected.append(v)

        if fpath.suffix.lower() == ".hdf":
            return self._open_one_hdf4(fpath, selected, entry)
        return self._open_one_nc(fpath, selected, entry)

    def _open_one_hdf4(
        self,
        fpath: Path,
        selected: list[Any],
        entry: ProductEntry,
    ) -> xr.Dataset | None:
        """Read a native HDF4 MODIS granule via ``pyhdf``.

        Only the requested SDS and the grid-axis variables listed in
        ``entry.dim_aliases`` are read; the ~1,100+ histogram SDS are never
        touched.  Scale/fill/valid_range are applied manually (CF convention:
        ``physical = raw * scale_factor + add_offset``).
        """
        try:
            import pyhdf.SD as HDFSD

            hdf = HDFSD.SD(str(fpath), HDFSD.SDC.READ)
        except Exception as e:  # pragma: no cover - exercised via smoke test
            warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
            return None

        try:
            available_sds = set(hdf.datasets().keys())

            keep = {v.sds_name: v for v in selected if v.sds_name in available_sds}
            if not keep:
                warnings.warn(f"{fpath.name}: none of the requested SDS present", UserWarning)
                return None

            # Collect the grid-axis variable names we need as coords.
            # dim_aliases keys like "XDim", "YDim" (and colon-qualified variants)
            # are the 1-D coordinate SDS in real HDF4 files.  Keep only simple
            # names (no colon) that exist in the file as readable SDS.
            axis_names = [k for k in entry.dim_aliases if ":" not in k and k in available_sds]

            # Read each requested SDS.
            data_arrays: dict[str, tuple[list[str], np.ndarray, dict[str, Any]]] = {}
            for sds_name in keep:
                v = hdf.select(sds_name)
                raw = np.array(v[:])
                attrs: dict[str, Any] = dict(v.attributes())
                n_dims = v.info()[1]
                dims = [v.dim(i).info()[0] for i in range(n_dims)]
                v.endaccess()

                physical = self._apply_hdf4_scale(raw, attrs)
                decoded_attrs = {
                    str(name): value
                    for name, value in attrs.items()
                    if str(name) not in _HDF4_PACKING_ATTRS
                }
                data_arrays[sds_name] = (dims, physical, decoded_attrs)

            # Read axis variables (XDim, YDim) and record their dims.
            coord_data: dict[str, tuple[str, np.ndarray]] = {}
            for ax in axis_names:
                ax_v = hdf.select(ax)
                ax_arr = np.array(ax_v[:])
                ax_dim = list(ax_v.dimensions().keys())[0]
                ax_v.endaccess()
                coord_data[ax] = (ax_dim, ax_arr)

        finally:
            hdf.end()

        # Build a minimal xr.Dataset.
        data_vars: dict[str, Any] = {
            sds: (dims, arr, attrs) for sds, (dims, arr, attrs) in data_arrays.items()
        }
        coords: dict[str, Any] = {ax: (dim, arr) for ax, (dim, arr) in coord_data.items()}
        ds = xr.Dataset(data_vars, coords=coords)

        return self._finalize(ds, keep, entry, fpath)

    @staticmethod
    def _apply_hdf4_scale(raw: np.ndarray, attrs: dict[str, Any]) -> np.ndarray:
        """Delegate to the shared helper in ``io.reader_utils``."""
        return apply_hdf4_scale(raw, attrs)

    def _open_one_nc(
        self,
        fpath: Path,
        selected: list[Any],
        entry: ProductEntry,
    ) -> xr.Dataset | None:
        """Read a NetCDF (or any xarray-compatible) granule via ``xr.open_dataset``.

        This is the original implementation, kept for synthetic ``.nc`` unit-
        test fixtures and any future non-HDF4 product.  MOD08_M3/MYD08_M3
        HDF4 files contain histogram SDS with duplicate dimension names;
        xarray emits a ``UserWarning`` for those.  We suppress it here so it
        does not pollute test output or break suites with
        ``filterwarnings=error``.
        """
        _dup_dim_msg = "Duplicate dimension names present"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", _dup_dim_msg, UserWarning)
                raw = xr.open_dataset(str(fpath), engine="netcdf4", mask_and_scale=True)
        except Exception as e:  # pragma: no cover - exercised via smoke test
            warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
            return None

        keep = {v.sds_name: v for v in selected if v.sds_name in raw.data_vars}
        if not keep:
            warnings.warn(f"{fpath.name}: none of the requested SDS present", UserWarning)
            return None

        # Promote grid-axis variables (e.g. XDim, YDim) to coordinates before
        # selecting, so they survive the variable-subset and can be renamed to
        # lon/lat by _standardize_grid.  In real HDF4 files these are plain
        # data variables (not coords), so raw[list(keep)] would otherwise drop
        # them; the synthetic unit-test fixture creates them as coords, so this
        # is a no-op there.
        # set_coords() and variable selection both copy the full raw dataset,
        # triggering the duplicate-dim warning again — keep the filter active.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", _dup_dim_msg, UserWarning)
            axis_vars = [k for k in entry.dim_aliases if k in raw.data_vars and k not in raw.coords]
            if axis_vars:
                raw = raw.set_coords(axis_vars)
            ds = raw[list(keep)]

        return self._finalize(ds, keep, entry, fpath)

    def _finalize(
        self,
        ds: xr.Dataset,
        keep: dict[str, Any],
        entry: ProductEntry,
        fpath: Path,
    ) -> xr.Dataset:
        """Rename SDS→display name, attach metadata, standardize grid, add time."""
        self._validate_qa_contract(ds, keep, entry, fpath)

        # Rename SDS -> display name and attach variable metadata.
        ds = ds.rename({sds: v.display_name for sds, v in keep.items()})
        for v in keep.values():
            field = ds[v.display_name]
            attrs = dict(field.attrs)
            if v.valid_min is not None:
                field = field.where(field >= v.valid_min)
                attrs["valid_min"] = v.valid_min
            if v.valid_max is not None:
                field = field.where(field <= v.valid_max)
                attrs["valid_max"] = v.valid_max
            attrs["units"] = v.units
            if v.wavelength_nm is not None:
                attrs["wavelength_nm"] = v.wavelength_nm
            if v.long_name:
                attrs["long_name"] = v.long_name
            if v.qa_screening is not None:
                attrs["qa_screening"] = v.qa_screening
                attrs["qa_screened"] = "true"
                attrs["support_definition"] = (
                    "finite cells after source QA usefulness and valid-range screening"
                )
            field.attrs = attrs
            ds[v.display_name] = field

        # Standardize grid coords: dim_aliases maps file dim/coord names -> lon/lat.
        ds = self._standardize_grid(ds, entry)

        parsed_time = self._parse_time(fpath.name, entry)
        return self._standardize_time(ds, parsed_time, entry, fpath)

    @staticmethod
    def _validate_qa_contract(
        ds: xr.Dataset,
        keep: dict[str, Any],
        entry: ProductEntry,
        fpath: Path,
    ) -> None:
        """Require evidence that catalog-declared D3 QA screening was applied."""
        for sds_name, variable in keep.items():
            if variable.qa_screening != "level2_usefulness_flag":
                continue
            attrs = ds[sds_name].attrs
            raw_flag = attrs.get("Masked_With_QA_Usefulness_Flag")
            flag_is_true = str(raw_flag).strip().lower() in {"1", "true", "yes"}
            canonical_flag = attrs.get("qa_screening") == variable.qa_screening
            staged_product = ds.attrs.get("source_product") == entry.product_id
            if not (flag_is_true or canonical_flag or staged_product):
                raise ValueError(
                    f"{fpath.name}: cannot verify Level-2 QA usefulness screening "
                    f"for {sds_name}"
                )

    @staticmethod
    def _standardize_time(
        ds: xr.Dataset,
        parsed_time: np.datetime64,
        entry: ProductEntry,
        fpath: Path,
    ) -> xr.Dataset:
        """Attach parsed cadence time and validate a staged singleton time axis."""
        if "time" not in ds.dims:
            return ds.expand_dims(time=[parsed_time])
        if ds.sizes["time"] != 1:
            raise ValueError(
                f"{fpath.name}: expected one time sample per {entry.product_id} file; "
                f"found {ds.sizes['time']}"
            )
        if "time" in ds.coords:
            observed = pd.Timestamp(ds["time"].values[0])
            expected = pd.Timestamp(parsed_time)
            if entry.cadence == "monthly":
                matches = observed.to_period("M") == expected.to_period("M")
            else:
                matches = observed.normalize() == expected.normalize()
            if not matches:
                raise ValueError(
                    f"{fpath.name}: time {observed} does not match filename date {expected}"
                )
        return ds.assign_coords(time=[parsed_time])

    @staticmethod
    def _standardize_grid(ds: xr.Dataset, entry: ProductEntry) -> xr.Dataset:
        """Apply dim_aliases to rename dims and coords to lon/lat.

        xarray does not allow renaming a dim and a coord to the same target
        name in one call when a coord shares a different name than the dim.
        We therefore rename dims first, then coords separately.
        """
        aliases = entry.dim_aliases

        # --- 1. rename dims ---------------------------------------------------
        dim_renames = {k: v for k, v in aliases.items() if k in ds.dims}
        if dim_renames:
            ds = ds.rename_dims(dim_renames)

        # --- 2. rename coords -------------------------------------------------
        coord_renames = {k: v for k, v in aliases.items() if k in ds.coords}
        # Only rename if target name not already a coord (avoids collision).
        coord_renames = {k: v for k, v in coord_renames.items() if v not in ds.coords}
        if coord_renames:
            ds = ds.rename_vars(coord_renames)

        # Promote lon/lat to coordinates if they are bare 1-D variables.
        for axis in ("lon", "lat"):
            if axis in ds.variables and axis not in ds.coords:
                ds = ds.set_coords(axis)
        return ds

    @staticmethod
    def _parse_time(filename: str, entry: ProductEntry) -> np.datetime64:
        m = _DATE_TOKEN.search(filename)
        if not m:
            raise ValueError(f"Cannot parse date token from filename: {filename}")
        dt = datetime.strptime("A" + m.group(1), entry.time_parse)
        stamp = pd.Timestamp(dt)
        if entry.cadence == "monthly":
            stamp = stamp.to_period("M").to_timestamp()
        return np.datetime64(stamp, "ns")
