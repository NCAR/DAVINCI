"""Pipeline process and dataset lifecycle policy."""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import Any

from davinci_monet.pipeline.stages.base import PipelineContext

logger = logging.getLogger(__name__)


@dataclass
class PipelineResourcePolicy:
    """Resource lifecycle rules applied around a pipeline run."""

    close_datasets_after_run: bool = True

    def prepare_before_run(self) -> None:
        """Apply process-level safety defaults and clear stale file state."""
        if "HDF5_USE_FILE_LOCKING" not in os.environ:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            logger.debug("HDF5_USE_FILE_LOCKING not set; defaulting to FALSE")
        self.cleanup_hdf5_state()

    def cleanup_after_run(self, context: PipelineContext) -> None:
        """Clean resources after run according to this policy."""
        if self.close_datasets_after_run:
            self.cleanup_context_datasets(context)

    def cleanup_hdf5_state(self) -> None:
        """Clear HDF5/NetCDF state to avoid transient file handle errors."""
        gc.collect()
        try:
            from xarray.backends.file_manager import FILE_CACHE

            FILE_CACHE.clear()
        except (ImportError, AttributeError):
            pass
        try:
            import netCDF4  # noqa: F401
        except ImportError:
            pass
        logger.debug("Cleared HDF5/NetCDF file state")

    def cleanup_context_datasets(self, context: PipelineContext) -> None:
        """Close all datasets in context to avoid transient file handle errors."""
        closed_ids: set[int] = set()
        for _label, source_data in list(context.sources.items()):
            self._close_source_data(source_data, closed_ids)
        for _label, paired_data in list(context.paired.items()):
            self._close_source_data(paired_data, closed_ids)

        gc.collect()
        try:
            from xarray.backends.file_manager import FILE_CACHE

            FILE_CACHE.clear()
        except (ImportError, AttributeError):
            pass
        logger.debug("Closed all context datasets")

    @staticmethod
    def _close_source_data(source_data: Any, closed_ids: set[int]) -> None:
        try:
            data = source_data.data if hasattr(source_data, "data") else source_data
            data_id = id(data)
            if data_id in closed_ids:
                return
            if hasattr(data, "close"):
                data.close()
                closed_ids.add(data_id)
        except Exception:
            pass
