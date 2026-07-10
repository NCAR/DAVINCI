"""Plotting stage.

Generates comparison plots from paired data or single-source plots from
unpaired data through the unified renderer ``render(series)`` contract.
"""

from __future__ import annotations

from typing import Any

from davinci_monet.core.base import iter_paired_variable_xy, paired_canonical_name
from davinci_monet.core.exceptions import PlottingError
from davinci_monet.core.schema_utils import dump_schema, is_schema_object
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.helpers import (
    iter_single_source_datasets,
    tag_source_label,
)
from davinci_monet.pipeline.stages.plot_options import (
    build_comparison_plot_options,
    build_plot_subtitle,
    single_source_flight_plot_kwargs,
    single_source_plot_kwargs,
    timestamp_from_field,
)
from davinci_monet.plots.contracts import PlotArity, plot_arity


def _is_artifact_source(source: Any, dataset: Any) -> bool:
    """Return whether a pseudo-source is manifest-only rather than plottable data."""
    geometry = getattr(source, "geometry", None)
    if str(getattr(geometry, "name", "")).upper() == "ARTIFACT":
        return True
    return str(getattr(dataset, "attrs", {}).get("geometry", "")).lower() == "artifact"


def _has_finite_values(field: Any) -> bool:
    """Check a selected field lazily without materializing its full payload."""
    import numpy as np
    import xarray as xr

    if field.dtype.kind in "biufcMm":
        valid = xr.apply_ufunc(np.isfinite, field, dask="allowed")
    else:
        valid = field.notnull()
    any_valid = valid.any()
    if any_valid.chunks is not None:
        any_valid = any_valid.compute()
    return bool(any_valid.item())


class PlottingStage(BaseStage):
    """Stage for generating plots from paired or single-source data."""

    def __init__(self) -> None:
        super().__init__("plotting")

    @staticmethod
    def _config_dict(value: Any) -> dict[str, Any]:
        if is_schema_object(value):
            return dump_schema(value, exclude_none=True)
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _plot_formats(plot_spec: dict[str, Any]) -> tuple[str, ...]:
        raw_formats = plot_spec.get("formats", plot_spec.get("output_formats"))
        if raw_formats is None:
            return ("png", "pdf")
        if isinstance(raw_formats, str):
            raw_values = [raw_formats]
        else:
            try:
                raw_values = list(raw_formats)
            except TypeError as exc:
                raise PlottingError("plot formats must be a string or sequence") from exc

        formats: list[str] = []
        for raw_value in raw_values:
            fmt = str(raw_value).lower().lstrip(".")
            if fmt not in {"png", "pdf"}:
                raise PlottingError(f"unsupported plot format: {raw_value}")
            if fmt not in formats:
                formats.append(fmt)
        if not formats:
            raise PlottingError("plot formats must include at least one format")
        return tuple(formats)

    @staticmethod
    def _remove_unrequested_format_siblings(path_base: Any, formats: tuple[str, ...]) -> None:
        requested = set(formats)
        for fmt in {"png", "pdf"} - requested:
            stale_path = PlottingStage._format_path(path_base, fmt)
            if stale_path.exists():
                stale_path.unlink()

    @staticmethod
    def _format_path(path_base: Any, fmt: str) -> Any:
        return path_base.parent / f"{path_base.name}.{fmt}"

    @classmethod
    def _source_var_config(
        cls,
        config: dict[str, Any],
        source_label: str,
        variable_name: str,
    ) -> dict[str, Any]:
        """Return variable plot config for a source label."""
        sources_cfg = config.get("sources", {})
        if not isinstance(sources_cfg, dict):
            return {}
        source_cfg = cls._config_dict(sources_cfg.get(source_label, {}))
        variables = cls._config_dict(source_cfg.get("variables", {}))
        if variable_name in variables:
            return cls._config_dict(variables[variable_name])
        return {}

    @staticmethod
    def _resolve_paired_dataset_variable(
        paired_data: Any,
        *,
        source_label: str,
        y_variable: str,
        axis: str,
        fallback_name: str | None = None,
    ) -> str | None:
        """Resolve a variable from unified paired metadata."""
        candidates = list(paired_data.data_vars)
        if fallback_name in paired_data.data_vars:
            candidates.insert(0, str(fallback_name))

        seen: set[str] = set()
        requested = str(y_variable)
        requested_lower = requested.lower()
        for name in candidates:
            name = str(name)
            if name in seen:
                continue
            seen.add(name)
            attrs = paired_data[name].attrs
            if attrs.get("axis") != axis:
                continue
            if attrs.get("source_label") != source_label:
                continue
            actual_source_var = str(attrs.get("dataset_variable") or "")
            if actual_source_var and actual_source_var.lower() == requested_lower:
                return name
            if paired_canonical_name(paired_data, name).lower() == requested_lower:
                return name
            if name.lower() == requested_lower:
                return name
        return None

    def validate(self, context: PipelineContext) -> bool:
        """Run for paired comparisons or single-source plots."""
        return bool(context.paired) or bool(iter_single_source_datasets(context))

    def _render_single_source_plot(
        self,
        *,
        context: PipelineContext,
        plot_name: str,
        plot_type: str,
        plot_spec: dict[str, Any],
        analysis_config: dict[str, Any],
        output_dir: Any,
        source_map: dict[str, tuple[Any, Any]],
        plots_generated: list[str],
        file_index: int,
    ) -> int:
        """Render one validated single-source plot spec and save its figures.

        Single-source renderers keep their historical output names
        (``{plot_name}.png`` / ``.pdf`` plus flight and renderer labels) while
        advancing the shared preview ``file_index`` for mixed dispatch runs.
        """
        import logging
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        from davinci_monet.plots.base import build_series
        from davinci_monet.plots.registry import get_plotter

        _logger = logging.getLogger(__name__)
        errors: list[str] = []

        source_label = str(plot_spec.get("source") or "")
        variable = str(plot_spec.get("variable") or "")

        if not source_label:
            raise PlottingError(f"Plot '{plot_name}' is missing source")
        if not variable:
            raise PlottingError(f"Plot '{plot_name}' is missing variable")
        if source_label not in source_map:
            raise PlottingError(f"Source '{source_label}' not found for plot '{plot_name}'")

        source_obj, ds = source_map[source_label]

        if variable not in ds.data_vars:
            raise PlottingError(
                f"Variable '{variable}' not in source '{source_label}' for plot '{plot_name}'"
            )

        plotter = get_plotter(plot_type)
        if _is_artifact_source(source_obj, ds) and not bool(
            getattr(plotter, "supports_artifact", False)
        ):
            raise PlottingError(
                f"Source '{source_label}' has ARTIFACT geometry, which plot type "
                f"'{plot_type}' does not support"
            )
        plot_kwargs = single_source_plot_kwargs(
            plot_spec,
            analysis_config=analysis_config,
        )
        formats = self._plot_formats(plot_spec)
        plot_output_dir = output_dir
        output_subdir = plot_spec.get("output_subdir")
        if output_subdir:
            subdir = Path(str(output_subdir))
            plot_output_dir = subdir if subdir.is_absolute() else output_dir / subdir
        plot_output_dir.mkdir(parents=True, exist_ok=True)

        has_flights = "flight" in ds.coords
        flight_ids = sorted(set(np.unique(ds["flight"].values).tolist())) if has_flights else [None]

        for fid in flight_ids:
            if fid is not None:
                mask = ds["flight"].values == fid
                subset = ds.isel(time=mask)
                suffix = f"_{fid}"
                flight_base_kwargs = (
                    plot_kwargs
                    if plot_kwargs.get("title")
                    else {**plot_kwargs, "title": f"{variable} {plot_type}"}
                )
                flight_kwargs = single_source_flight_plot_kwargs(
                    flight_base_kwargs,
                    flight_id=fid,
                )
            else:
                subset = ds
                suffix = ""
                flight_kwargs = plot_kwargs

            try:
                # Select before probing so unrelated lazy mode chunks stay untouched.
                sel_mode = plot_spec.get("mode")
                if sel_mode is not None and variable in subset and "mode" in subset[variable].dims:
                    subset = subset.sel(mode=sel_mode)
                if not _has_finite_values(subset[variable]):
                    continue

                if "flight_tracks" in plot_spec:
                    flight_kwargs["x_datasets"] = {
                        label: source_ds for label, (_obj, source_ds) in source_map.items()
                    }

                # Tag the single source so build_series picks up its source label.
                tag_source_label(subset, source_label=source_label)
                render_kwargs = dict(flight_kwargs)
                title = render_kwargs.pop("title", None)
                if title:
                    plotter.config.title = str(title)
                plotter.config.subtitle = render_kwargs.pop("subtitle", None)
                result = plotter.render(build_series(subset, variable), **render_kwargs)
                figures: list[tuple[str | None, Any]]
                if isinstance(result, list):
                    figures = [(fig_label, fig) for fig_label, fig in result]
                else:
                    figures = [(None, result)]
                for fig_label, fig in figures:
                    fig_suffix = f"_{fig_label}" if fig_label else ""
                    path_base = plot_output_dir / f"{plot_name}{suffix}{fig_suffix}"
                    self._remove_unrequested_format_siblings(path_base, formats)
                    saved_paths = []
                    for fmt in formats:
                        out_path = self._format_path(path_base, fmt)
                        plotter.save(fig, out_path)
                        plots_generated.append(str(out_path))
                        saved_paths.append(out_path)
                    plt.close(fig)
                    file_index += 1
                    _logger.info(
                        "Saved source plot: %s",
                        ", ".join(str(path) for path in saved_paths),
                    )
            except Exception as e:
                label = f"'{plot_name}' (flight {fid})" if fid else f"'{plot_name}'"
                errors.append(f"Plot {label} failed: {e}")
                _logger.warning(f"Source plot {label} failed: {e}")

        if errors:
            raise PlottingError("; ".join(errors))

        return file_index

    def _resolve_pair_labels_and_vars(
        self,
        pair_name: str,
        pair_spec: dict[str, Any],
        context: PipelineContext,
    ) -> tuple[Any, str, str, dict[str, Any], str, str] | None:
        """Resolve the paired dataset, source labels, and variable names for a pair.

        Returns ``(paired_data, x_source, y_source, var_spec, geometry_var_name,
        dataset_var_name)`` or ``None`` when the pair cannot be resolved.
        """
        x_axis = pair_spec.get("x") if isinstance(pair_spec.get("x"), dict) else None
        y_axis = pair_spec.get("y") if isinstance(pair_spec.get("y"), dict) else None
        has_xy = x_axis is not None and y_axis is not None
        y_source = ""
        x_source = ""
        var_spec: dict[str, Any] = {}

        # Resolve the paired-data key. Unified ``x:``/``y:`` pairs and
        # implicitly auto-paired pairs are both keyed by the pair name
        # in ``context.paired``; for unified pairs we also recover the
        # x (reference) / y (sampled) labels from the spec.
        pair_key = pair_name
        if has_xy:
            assert x_axis is not None and y_axis is not None  # narrow for mypy
            x_source = str(x_axis.get("source") or "")
            y_source = str(y_axis.get("source") or "")
        if pair_key not in context.paired:
            return None

        paired_obj = context.paired[pair_key]
        paired_data = paired_obj.data if hasattr(paired_obj, "data") else paired_obj

        if has_xy:
            assert x_axis is not None and y_axis is not None  # narrow for mypy
            pair_vars = iter_paired_variable_xy(paired_data)
            if not pair_vars:
                return None
            fallback_geometry_name, fallback_dataset_name, fallback_var = pair_vars[0]
            if not x_source:
                x_source = str(
                    paired_data[fallback_geometry_name].attrs.get("source_label", "geometry")
                )
            if not y_source:
                y_source = str(
                    paired_data[fallback_dataset_name].attrs.get("source_label", "dataset")
                )
            x_var = str(
                x_axis.get("variable")
                or paired_data[fallback_geometry_name].attrs.get("dataset_variable")
                or fallback_var
            )
            y_var = str(
                y_axis.get("variable")
                or paired_data[fallback_dataset_name].attrs.get("dataset_variable")
                or fallback_var
            )
            var_spec = {
                "x_var": x_var,
                "y_var": y_var,
            }
            x_var_name = self._resolve_paired_dataset_variable(
                paired_data,
                source_label=x_source,
                y_variable=x_var,
                axis="x",
                fallback_name=fallback_geometry_name,
            )
            y_var_name = self._resolve_paired_dataset_variable(
                paired_data,
                source_label=y_source,
                y_variable=y_var,
                axis="y",
                fallback_name=fallback_dataset_name,
            )
            if x_var_name is None or y_var_name is None:
                return None
        else:
            # No pair spec: a plot can name a key already present in
            # context.paired. Recover labels and variables from paired attrs.
            pair_vars = iter_paired_variable_xy(paired_data)
            if not pair_vars:
                return None
            x_var_name, y_var_name, x_var = pair_vars[0]
            x_source = str(paired_data[x_var_name].attrs.get("source_label", "x"))
            y_source = str(paired_data[y_var_name].attrs.get("source_label", "y"))
            var_spec = {"x_var": x_var, "y_var": x_var}

        return (
            paired_data,
            x_source,
            y_source,
            var_spec,
            x_var_name,
            y_var_name,
        )

    @staticmethod
    def _apply_domain_filter(paired_data: Any, plot_spec: dict[str, Any]) -> Any:
        """Apply a per-plot domain filter to the paired data, if requested.

        ``domain_type``/``domain_name`` in the plot spec restrict ``paired_data``
        to a named lat/lon extent (e.g. 'conus', or 'epa_region' + R5). When the
        type is None / "all" or unrecognised, the helper returns the dataset
        unchanged.
        """
        from davinci_monet.util.domain import filter_paired_by_domain

        return filter_paired_by_domain(
            paired_data,
            plot_spec.get("domain_type"),
            plot_spec.get("domain_name"),
        )

    def _resolve_plot_options(
        self,
        *,
        context: PipelineContext,
        plot_type: str,
        plot_spec: dict[str, Any],
        analysis_config: dict[str, Any],
        title: str,
        paired_data: Any,
        var_spec: dict[str, Any],
        x_source: str,
        y_source: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Assemble the ``(plotter_config, plot_options)`` for a comparison plot.

        Resolves per-source variable limits, builds the renderer option dict
        (including ``spatial_overlay`` y-field/coord wiring), the title
        subtitle, and the axis-label overrides.
        """
        # Get plotter config from source variable settings. The y side
        # wins for comparison-specific plot limits; x settings are a
        # fallback.
        config = context.config_dict()
        y_var = var_spec.get("y_var", "")
        x_var = var_spec.get("x_var", "")
        var_config = self._source_var_config(config, y_source, y_var)
        if not var_config:
            var_config = self._source_var_config(config, x_source, x_var)
        vmin = var_config.get("vmin_plot")
        vmax = var_config.get("vmax_plot")
        vdiff = var_config.get("vdiff_plot")
        nlevels = var_config.get("nlevels_plot")
        style_preset = var_config.get("style_preset")
        levels = var_config.get("levels_plot")
        cmap = var_config.get("cmap_plot")
        extend = var_config.get("extend_plot")

        # Build plotter config
        plotter_config: dict[str, Any] = {"title": title}
        if plot_type == "spatial_bias":
            plotter_config["vmin"] = -vdiff if vdiff else None
            plotter_config["vmax"] = vdiff if vdiff else None
        elif plot_type == "timeseries":
            # Timeseries uses smart auto-scaling from data range
            # Don't pass vmin/vmax unless explicitly set in plot_spec
            if "vmin" in plot_spec:
                plotter_config["vmin"] = plot_spec["vmin"]
            if "vmax" in plot_spec:
                plotter_config["vmax"] = plot_spec["vmax"]
        else:
            plotter_config["vmin"] = vmin
            plotter_config["vmax"] = vmax

        plot_options = build_comparison_plot_options(
            plot_type,
            plot_spec,
            analysis_config,
            nlevels=nlevels,
        )
        if style_preset is not None:
            plot_options.setdefault("style_preset", style_preset)
        if levels is not None:
            plot_options.setdefault("levels", levels)
        if cmap is not None:
            plot_options.setdefault("cmap", cmap)
        if extend is not None:
            plot_options.setdefault("extend", extend)

        # spatial_overlay needs the raw gridded y field for the
        # contour layer; the paired dataset usually carries sampled
        # values at x locations only. The renderer option name is
        # `y_field`.
        if plot_type == "spatial_overlay":
            if "y_field" not in plot_options:
                source_obj = context.sources.get(y_source) or context.sources.get(x_source)
                if source_obj is not None:
                    source_ds = source_obj.data if hasattr(source_obj, "data") else source_obj
                    source_vars = getattr(source_ds, "data_vars", {})
                    field_var = y_var if y_var in source_vars else x_var
                    if source_ds is not None and field_var in source_vars:
                        plot_options["y_field"] = source_ds[field_var]
            # Dataset readers differ on coord naming
            # (`latitude`/`longitude` vs `lat`/`lon`). Pick whichever
            # the paired dataset actually carries.
            if "lat_var" not in plot_options:
                for cand in ("latitude", "lat"):
                    if cand in paired_data.coords or cand in paired_data:
                        plot_options["lat_var"] = cand
                        break
            if "lon_var" not in plot_options:
                for cand in ("longitude", "lon"):
                    if cand in paired_data.coords or cand in paired_data:
                        plot_options["lon_var"] = cand
                        break

        snapshot_str = ""
        if plot_type == "spatial_overlay" and "y_field" in plot_options:
            mf = plot_options["y_field"]
            time_idx = plot_options.get("time_index", 0)
            snapshot_str = timestamp_from_field(mf, time_idx)
        subtitle = build_plot_subtitle(
            analysis_config,
            snapshot_timestamp=snapshot_str,
        )
        if subtitle:
            plotter_config["subtitle"] = subtitle

        # Forward per-plot axis label overrides to the plotter config
        # so renderers like scatter can display source-named labels
        # (e.g. "MODIS Terra AOD") instead of "Dataset AOD (550 nm)".
        label_aliases = {
            "x_label": "x_label",
            "y_label": "y_label",
        }
        for input_key, plotter_key in label_aliases.items():
            if input_key in plot_spec:
                plotter_config[plotter_key] = plot_spec[input_key]

        return plotter_config, plot_options

    @staticmethod
    def _save_render_result(
        *,
        plotter: Any,
        paired_data: Any,
        x_var_name: str,
        y_var_name: str,
        plot_options: dict[str, Any],
        x_source_output_dir: Any,
        plot_name: str,
        file_index: int,
        plots_generated: list[str],
        context: PipelineContext,
        x_source: str,
        formats: tuple[str, ...],
    ) -> int:
        """Render and save one or more figures from the unified render contract."""
        import matplotlib.pyplot as plt

        from davinci_monet.plots.base import build_series

        result = plotter.render(build_series(paired_data, x_var_name, y_var_name), **plot_options)
        figures = result if isinstance(result, list) else [(None, result)]

        saved_count = 0
        for label, fig in figures:
            prefix = f"{label}_" if label else ""
            path_base = x_source_output_dir / f"{prefix}{file_index:02d}_{plot_name}"
            PlottingStage._remove_unrequested_format_siblings(path_base, formats)
            for fmt in formats:
                output_path = PlottingStage._format_path(path_base, fmt)
                if fmt == "png":
                    plotter.save(fig, output_path, dpi=300)
                else:
                    plotter.save(fig, output_path)
                plots_generated.append(str(output_path))

            plt.close(fig)
            file_index += 1
            saved_count += 1

        context.log_progress(f"done: saved {saved_count} plot(s) to {x_source}/")
        return file_index

    def _render_pair(
        self,
        *,
        context: PipelineContext,
        plot_name: str,
        pair_name: str,
        pairs_config: dict[str, Any],
        plot_type: str,
        plot_spec: dict[str, Any],
        analysis_config: dict[str, Any],
        title: str,
        output_dir: Any,
        file_index: int,
        plots_generated: list[str],
    ) -> int:
        """Render and save all figures for one (plot, pair) combination.

        Returns the advanced ``file_index``.
        """
        from davinci_monet.plots import get_plotter

        # Get pair configuration
        pair_spec = pairs_config.get(pair_name, {})
        if not isinstance(pair_spec, dict):
            pair_spec = {}

        resolved = self._resolve_pair_labels_and_vars(pair_name, pair_spec, context)
        if resolved is None:
            raise PlottingError(f"Pair '{pair_name}' not found or has no matching x/y variables")
        (
            paired_data,
            x_source,
            y_source,
            var_spec,
            x_var_name,
            y_var_name,
        ) = resolved

        paired_data = self._apply_domain_filter(paired_data, plot_spec)

        if x_var_name not in paired_data or y_var_name not in paired_data:
            raise PlottingError(
                f"Pair '{pair_name}' missing resolved variables "
                f"'{x_var_name}' and/or '{y_var_name}'"
            )

        plotter_config, plot_options = self._resolve_plot_options(
            context=context,
            plot_type=plot_type,
            plot_spec=plot_spec,
            analysis_config=analysis_config,
            title=title,
            paired_data=paired_data,
            var_spec=var_spec,
            x_source=x_source,
            y_source=y_source,
        )

        # Get plotter
        plotter = get_plotter(plot_type, config=plotter_config)

        # Create subdirectory by geometry source.
        x_source_output_dir = output_dir / x_source
        x_source_output_dir.mkdir(parents=True, exist_ok=True)

        return self._save_render_result(
            plotter=plotter,
            paired_data=paired_data,
            x_var_name=x_var_name,
            y_var_name=y_var_name,
            plot_options=plot_options,
            x_source_output_dir=x_source_output_dir,
            plot_name=plot_name,
            file_index=file_index,
            plots_generated=plots_generated,
            context=context,
            x_source=x_source,
            formats=self._plot_formats(plot_spec),
        )

    def execute(self, context: PipelineContext) -> StageResult:
        """Generate comparison plots from paired data, or plots from unpaired sources."""
        import time
        from pathlib import Path

        from davinci_monet.plots.labeling import title_text as _title_text

        start = time.time()
        plots_generated: list[str] = []

        config = context.config_dict()
        plot_config = config.get("plots", {})

        # Plotting is optional - if no config, skip
        if not plot_config:
            return self._create_result(
                StageStatus.SKIPPED,
                data={"message": "No plot configuration found"},
                duration=time.time() - start,
            )

        # Get output directory
        analysis_config = config.get("analysis", {})
        output_dir = Path(context.analysis_config().output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get pairs config for variable mapping
        pairs_config = config.get("pairs", {})
        total_plots = len(plot_config)
        plot_number = 0
        plot_count = 0
        file_index = 0  # Global counter for ordering files in preview
        source_map = {label: (obj, ds) for label, obj, ds in iter_single_source_datasets(context)}

        for plot_name, plot_spec in plot_config.items():
            try:
                plot_number += 1
                plot_type = plot_spec.get("type", "scatter")
                plot_pairs = plot_spec.get("pairs", [])
                title = _title_text(plot_spec.get("title", plot_name))
                arity = plot_arity(plot_type)
                file_index_before = file_index

                context.log_progress(f"    Plot: {plot_name} ({plot_number}/{total_plots})")
                context.log_progress(f"step: Rendering {plot_type}...")

                if arity == PlotArity.SINGLE_SOURCE:
                    file_index = self._render_single_source_plot(
                        context=context,
                        plot_name=plot_name,
                        plot_type=plot_type,
                        plot_spec=plot_spec,
                        analysis_config=analysis_config,
                        output_dir=output_dir,
                        source_map=source_map,
                        plots_generated=plots_generated,
                        file_index=file_index,
                    )
                elif arity == PlotArity.PAIRWISE:
                    if not plot_pairs:
                        raise PlottingError(f"Plot '{plot_name}' has no configured pairs")
                    for pair_name in plot_pairs:
                        file_index = self._render_pair(
                            context=context,
                            pair_name=pair_name,
                            pairs_config=pairs_config,
                            plot_type=plot_type,
                            plot_spec=plot_spec,
                            analysis_config=analysis_config,
                            title=title,
                            output_dir=output_dir,
                            file_index=file_index,
                            plots_generated=plots_generated,
                            plot_name=plot_name,
                        )
                elif arity == PlotArity.MULTI_SOURCE:
                    if plot_pairs:
                        for pair_name in plot_pairs:
                            file_index = self._render_pair(
                                context=context,
                                pair_name=pair_name,
                                pairs_config=pairs_config,
                                plot_type=plot_type,
                                plot_spec=plot_spec,
                                analysis_config=analysis_config,
                                title=title,
                                output_dir=output_dir,
                                file_index=file_index,
                                plots_generated=plots_generated,
                                plot_name=plot_name,
                            )
                    else:
                        file_index = self._render_single_source_plot(
                            context=context,
                            plot_name=plot_name,
                            plot_type=plot_type,
                            plot_spec=plot_spec,
                            analysis_config=analysis_config,
                            output_dir=output_dir,
                            source_map=source_map,
                            plots_generated=plots_generated,
                            file_index=file_index,
                        )

                plot_count += file_index - file_index_before

            except Exception as e:
                context.metadata.setdefault("plot_errors", []).append(f"{plot_name}: {e}")
                context.log_progress(f"warning: plot failed for {plot_name}: {e}")

        errors = context.metadata.get("plot_errors") or []
        if errors:
            if plot_count == 0:
                return self._create_result(
                    StageStatus.FAILED,
                    data={"plot_count": plot_count, "plots_generated": plots_generated},
                    error=f"all {len(errors)} plots failed",
                    duration=time.time() - start,
                    warnings=list(errors),
                )
            return self._create_result(
                StageStatus.COMPLETED,
                data={"plot_count": plot_count, "plots_generated": plots_generated},
                duration=time.time() - start,
                warnings=list(errors),
            )

        return self._create_result(
            StageStatus.COMPLETED if plot_count > 0 else StageStatus.SKIPPED,
            data={"plot_count": plot_count, "plots_generated": plots_generated},
            duration=time.time() - start,
        )
