"""Source-loading stage."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from davinci_monet.core.coordinates import normalize_post_load_coordinates
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.schema_utils import dump_schema, is_schema_object
from davinci_monet.io.source_registration import ensure_builtin_source_readers_registered
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourceData,
    StageResult,
    StageStatus,
)

SOURCE_LOADER_CONFIG_KEYS = frozenset(
    {
        "type",
        "files",
        "filename",
        "variables",
        "radius_of_influence",
        "display_name",
        "resample",
        "min_sample_count",
        "track_sample_count",
        "time_padding",
        "evaluation_only",
        "artifact_manifest",
        "artifact_role",
        "artifact_analysis",
    }
)


class LoadSourcesStage(BaseStage):
    """Standard data-source loading stage.

    Loads the ``sources:`` config directly through ``source_registry``. Loaded
    datasets are tagged with ``source_label`` and ``geometry`` in
    ``context.sources``.
    """

    def __init__(self) -> None:
        super().__init__("load_sources")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that at least one source is configured."""
        cfg = context.config_dict()
        has_config = "sources" in cfg
        # Already-populated sources (direct stage / unit-test use) are also a
        # valid starting point.
        return has_config or bool(context.sources)

    def execute(self, context: PipelineContext) -> StageResult:
        """Load and unify all data sources into ``context.sources``.

        ``sources:`` configs load directly through ``source_registry``. This
        stage also accepts pre-populated ``context.sources`` for direct stage
        and unit-test use.
        """
        import time

        start = time.time()

        config = context.config_dict()
        if config.get("sources"):
            try:
                source_configs = config.get("sources", {})
                total_sources = len(source_configs)
                for index, (label, raw_config) in enumerate(source_configs.items(), start=1):
                    context.log_progress(f"    Loading source: {label} ({index}/{total_sources})")
                    source = self._load_unified_source(label, raw_config, context)
                    self._register_source(context, label, source)
                return self._create_result(
                    StageStatus.COMPLETED,
                    data={"loaded_sources": list(context.sources.keys())},
                    duration=time.time() - start,
                    count=len(context.sources),
                )
            except Exception as e:
                return self._create_result(
                    StageStatus.FAILED,
                    error=f"Failed to load sources: {e}",
                    duration=time.time() - start,
                )

        # No source config at all: any pre-populated sources (tests/direct
        # stage use) just get their datasets tagged with source_label /
        # geometry so downstream stages can rely on those attrs.
        for label, obj in list(context.sources.items()):
            self._tag_source(label, obj)

        return self._create_result(
            StageStatus.COMPLETED,
            data={"loaded_sources": list(context.sources.keys())},
            duration=time.time() - start,
            count=len(context.sources),
        )

    @staticmethod
    def _as_dict(config: Any) -> dict[str, Any]:
        if is_schema_object(config):
            return dump_schema(config, exclude_none=True)
        if isinstance(config, dict):
            return dict(config)
        if hasattr(config, "__dict__"):
            return dict(config.__dict__)
        return {}

    @staticmethod
    def _normalize_var_configs(raw: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for name, cfg in raw.items():
            if is_schema_object(cfg):
                normalized[str(name)] = dump_schema(cfg, exclude_none=True)
            elif isinstance(cfg, dict):
                normalized[str(name)] = dict(cfg)
            elif hasattr(cfg, "__dict__"):
                normalized[str(name)] = dict(cfg.__dict__)
            else:
                normalized[str(name)] = {}
        return normalized

    @staticmethod
    def _file_list(files: Any) -> list[str]:
        from glob import glob
        from pathlib import Path

        if files is None:
            return []
        if isinstance(files, (list, tuple)):
            values = [str(Path(str(item)).expanduser()) for item in files]
        else:
            values = [str(Path(str(files)).expanduser())]

        expanded: list[str] = []
        for value in values:
            if "*" in value or "?" in value:
                expanded.extend(sorted(glob(value)))
            else:
                expanded.append(value)
        return expanded

    @staticmethod
    def _data_geometry(value: Any) -> DataGeometry:
        if isinstance(value, DataGeometry):
            return value
        if isinstance(value, str):
            return DataGeometry[value.upper()]
        raise ValueError(f"Unsupported geometry value: {value!r}")

    @staticmethod
    def _reader_kwargs(reader: Any, cfg: dict[str, Any]) -> dict[str, Any]:
        """Return source config kwargs accepted by ``reader.open``."""
        import inspect

        candidates = {
            key: value
            for key, value in cfg.items()
            if key not in SOURCE_LOADER_CONFIG_KEYS and value is not None
        }
        signature = inspect.signature(reader.open)
        has_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        if has_kwargs:
            return candidates

        accepted = {
            name
            for name, param in signature.parameters.items()
            if param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        accepted.difference_update({"files", "file_paths", "paths", "variables"})
        return {key: value for key, value in candidates.items() if key in accepted}

    def _load_unified_source(
        self,
        label: str,
        raw_config: Any,
        context: PipelineContext,
    ) -> SourceData:
        """Load one source directly through ``source_registry``."""
        from davinci_monet.core.registry import ComponentNotFoundError, source_registry

        ensure_builtin_source_readers_registered()

        cfg = self._as_dict(raw_config)
        source_type = str(cfg.get("type") or "generic")
        variables = self._normalize_var_configs(cfg.get("variables", {}))
        variable_names: list[str] | None = [
            str(vcfg.get("source_name") or name) for name, vcfg in variables.items()
        ]
        variable_names = variable_names or None
        file_paths = self._file_list(cfg.get("files") or cfg.get("filename"))
        artifact_entry = None
        if cfg.get("artifact_manifest") is not None:
            from davinci_monet.analysis.artifacts import validate_finalized_artifact_manifest

            artifact_entry = validate_finalized_artifact_manifest(
                str(cfg["artifact_manifest"]),
                file_paths,
                role=str(cfg["artifact_role"]),
                analysis=(
                    str(cfg["artifact_analysis"])
                    if cfg.get("artifact_analysis") is not None
                    else None
                ),
            )

        analysis = context.config_dict().get("analysis", {})
        time_range = None
        if analysis.get("start_time") and analysis.get("end_time"):
            time_range = (analysis["start_time"], analysis["end_time"])
        load_time_range = self._padded_time_range(time_range, cfg.get("time_padding"))

        try:
            reader_cls = source_registry.get(source_type)
        except ComponentNotFoundError as e:
            available = ", ".join(source_registry.list())
            raise ValueError(
                f"Unknown source type '{source_type}' for source '{label}'. "
                f"Available source types: {available}"
            ) from e

        reader = reader_cls()
        import inspect

        open_kwargs = self._reader_kwargs(reader, cfg)
        if "time_range" in inspect.signature(reader.open).parameters:
            open_kwargs["time_range"] = load_time_range
        if "progress_callback" in inspect.signature(reader.open).parameters:

            def _per_file_progress(i: int, total: int, name: str) -> None:
                # Right-align the counter to the width of `total` so it stays a
                # fixed width as files tick, and put the (often long) file name
                # on its own indented line so it is never truncated off the end
                # of the status line by the terminal width.
                width = len(str(total))
                context.log_progress(f"step: loading {label} [{i:>{width}}/{total}]\n      {name}")

            open_kwargs["progress_callback"] = _per_file_progress
        data = reader.open(file_paths, variables=variable_names, **open_kwargs)
        if artifact_entry is not None:
            import json

            data.attrs.update(
                artifact_finalized="true",
                artifact_manifest=str(cfg["artifact_manifest"]),
                artifact_role=str(cfg["artifact_role"]),
                artifact_identity=json.dumps(
                    artifact_entry["identity"], sort_keys=True, default=str
                ),
            )
        if load_time_range and "time" in data:
            start_time, end_time = load_time_range
            data = data.sel(time=self._time_slice_for_coord(data["time"], start_time, end_time))
        if variables:
            data = self._apply_variable_config(data, variables)
            # Subset to exactly the configured variables (coordinates preserved).
            # The reader's open(variables=) subset is unreliable for source_name
            # mappings: the names passed are the post-rename keys, which do not
            # match the raw source_name columns at open time, so select_variables
            # no-ops and keeps every column. Now that _apply_variable_config has
            # renamed source_name -> key (then key -> rename), the final names
            # match and coordinates are already promoted, so this drops only
            # unconfigured data variables, never lat/lon/alt/time.
            from davinci_monet.io.reader_utils import select_variables

            final_names = [vcfg.get("rename") or name for name, vcfg in variables.items()]
            data = select_variables(data, final_names)

        resample_freq = cfg.get("resample")
        if resample_freq:
            from davinci_monet.datasets.base import resample_dataset

            data = resample_dataset(
                data,
                str(resample_freq),
                min_count=cfg.get("min_sample_count"),
                track_count=bool(cfg.get("track_sample_count")),
            )

        geometry = self._data_geometry(getattr(reader, "geometry"))
        source = SourceData(
            data=data,
            label=label,
            source_type=source_type,
            geometry=geometry,
            variables=variables,
            config=cfg,
        )
        return source

    @staticmethod
    def _padded_time_range(
        time_range: tuple[Any, Any] | None,
        padding: Any,
    ) -> tuple[Any, Any] | None:
        """Expand a source load window without changing the analysis window."""
        if time_range is None or padding is None:
            return time_range

        import pandas as pd

        delta = pd.Timedelta(padding)
        if delta < pd.Timedelta(0):
            raise ValueError("source time_padding must be non-negative")
        start, end = time_range
        return (pd.Timestamp(start) - delta, pd.Timestamp(end) + delta)

    @staticmethod
    def _time_slice_for_coord(coord: xr.DataArray, start: Any, end: Any) -> slice:
        """Build a time slice compatible with cftime-backed coordinates."""
        try:
            import cftime
            import pandas as pd
        except ImportError:
            return slice(start, end)

        values = np.asarray(getattr(coord, "values", []), dtype=object)
        first = next((value for value in values.ravel() if value is not None), None)
        if not isinstance(first, cftime.datetime):
            return slice(start, end)

        def _coerce(value: Any) -> Any:
            if value is None or isinstance(value, cftime.datetime):
                return value
            stamp = pd.Timestamp(value)
            return type(first)(
                stamp.year,
                stamp.month,
                stamp.day,
                stamp.hour,
                stamp.minute,
                stamp.second,
                stamp.microsecond,
            )

        return slice(_coerce(start), _coerce(end))

    @staticmethod
    def _apply_variable_config(
        data: xr.Dataset,
        variables: dict[str, dict[str, Any]],
    ) -> xr.Dataset:
        """Apply common variable scaling, masking, renaming, and metadata."""
        ds = data
        for var_name, cfg in list(variables.items()):
            source = cfg.get("source_name")
            if source and source in ds and var_name not in ds:
                ds = ds.rename({source: var_name})
            if var_name not in ds:
                continue
            arr = ds[var_name]
            # Capture units before scaling: xarray arithmetic drops attrs, and
            # the schema default unit_scale=1.0 means the block below runs for
            # every variable. We restore the original units after a NO-OP scale
            # so reader/file-provided units (e.g. ICARTT ppbv, altitude m) are
            # not silently lost. A real conversion (scale != identity) leaves
            # units to the explicit config `units` field, since the old unit
            # would no longer be correct.
            orig_units = arr.attrs.get("units")
            scale_is_identity = True
            if "unit_scale" in cfg:
                scale = cfg["unit_scale"]
                method = cfg.get("unit_scale_method", "*")
                identity = 1.0 if method in {"*", "/"} else 0.0
                scale_is_identity = scale == identity
                if not scale_is_identity:
                    if arr.dtype.kind not in "biufc":
                        raise TypeError(
                            f"variable {var_name!r} cannot apply nonidentity "
                            f"unit scaling to dtype {arr.dtype}"
                        )
                    if method == "*":
                        arr = arr * scale
                    elif method == "/":
                        arr = arr / scale
                    elif method == "+":
                        arr = arr + scale
                    elif method == "-":
                        arr = arr - scale
            nan_value = cfg.get("nan_value")
            if nan_value is not None:
                arr = arr.where(arr != nan_value)
            # Valid-range clamp.
            min_val = cfg.get("valid_min")
            if min_val is not None:
                arr = arr.where(arr >= min_val)
            max_val = cfg.get("valid_max")
            if max_val is not None:
                arr = arr.where(arr <= max_val)
            if cfg.get("units"):
                arr.attrs["units"] = cfg["units"]
            elif orig_units and scale_is_identity and not arr.attrs.get("units"):
                arr.attrs["units"] = orig_units
            elif orig_units and not scale_is_identity and arr.attrs.get("units") == orig_units:
                del arr.attrs["units"]
            if cfg.get("display_name"):
                arr.attrs["display_name"] = cfg["display_name"]
            ds[var_name] = arr
            rename_to = cfg.get("rename")
            if rename_to and rename_to != var_name:
                ds = ds.rename({var_name: rename_to})

        return ds

    @staticmethod
    def _tag_source(label: str, obj: Any) -> None:
        """Tag a pre-populated source's dataset with source_label/geometry.

        Used for sources placed directly in ``context.sources`` (direct stage /
        unit-test use) that bypassed :meth:`_load_unified_source`.
        """
        data = getattr(obj, "data", None)
        if data is None:
            return
        data = normalize_post_load_coordinates(data)
        obj.data = data
        try:
            geometry = obj.geometry
            x_name = geometry.name.lower() if hasattr(geometry, "name") else str(geometry)
        except Exception:
            x_name = "unknown"
        data.attrs["source_label"] = label
        data.attrs["geometry"] = x_name

    @staticmethod
    def _register_source(context: PipelineContext, label: str, obj: SourceData) -> None:
        """Register a loaded source in the canonical ``context.sources`` view."""
        obj.data = normalize_post_load_coordinates(obj.data)
        context.sources[label] = obj
        obj.data.attrs["source_label"] = label
        obj.data.attrs["geometry"] = obj.geometry.name.lower()
