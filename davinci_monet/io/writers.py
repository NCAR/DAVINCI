"""File writers for various data formats.

This module provides functions for writing data to different file formats
including NetCDF and pickle.
"""

from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import xarray as xr

from davinci_monet.core.exceptions import DataFormatError


def write_dataset(
    ds: xr.Dataset,
    path: str | Path,
    engine: str | None = None,
    compress: bool = False,
    **kwargs: Any,
) -> None:
    """Write a dataset to file.

    Automatically detects format based on file extension.

    Parameters
    ----------
    ds
        Dataset to write.
    path
        Output file path.
    engine
        NetCDF engine to use.
    compress
        Whether to apply zlib compression to NetCDF variables.
    **kwargs
        Additional arguments passed to xarray.

    Raises
    ------
    DataFormatError
        If write fails.
    """
    path = Path(path)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    try:
        if suffix in (".pkl", ".pickle"):
            write_pickle(ds, path)
        elif suffix == ".zarr":
            _write_zarr(ds, path, **kwargs)
        else:
            # Default to NetCDF
            if engine is None:
                engine = "netcdf4"

            # Handle compression
            if compress:
                encoding = kwargs.pop("encoding", {})
                for var in ds.data_vars:
                    if var not in encoding:
                        encoding[var] = {}
                    encoding[var].setdefault("zlib", True)
                    encoding[var].setdefault("complevel", 4)
                kwargs["encoding"] = encoding

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                ds.to_netcdf(str(tmp_path), engine=engine, **kwargs)  # type: ignore[call-overload]
                os.replace(tmp_path, path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
    except Exception as e:
        raise DataFormatError(f"Failed to write {path}: {e}") from e


def _write_zarr(ds: xr.Dataset, path: Path, **kwargs: Any) -> None:
    """Write a dataset to a Zarr store, swapping it into place best-effort.

    Zarr stores are directories, so unlike the file writers this cannot use
    a single atomic ``os.replace``: an existing target directory must be
    removed before the temp directory can take its place, leaving a brief
    window where nothing valid exists at ``path``. The swap is therefore
    best-effort rather than atomic. The write itself still lands entirely
    in a temp directory first, so a crash during writing never truncates
    or corrupts an existing store.

    Parameters
    ----------
    ds
        Dataset to write.
    path
        Output Zarr store path.
    **kwargs
        Additional arguments passed to ``xr.Dataset.to_zarr``.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    try:
        ds.to_zarr(str(tmp_path), **kwargs)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp_path, path)


def write_pickle(
    data: Any,
    path: str | Path,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> None:
    """Write data to pickle file.

    Parameters
    ----------
    data
        Data to pickle.
    path
        Output file path.
    protocol
        Pickle protocol version.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)
        os.replace(tmp_path, path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise DataFormatError(f"Failed to write pickle {path}: {e}") from e
