"""Plotting-stage option assembly helpers."""

from __future__ import annotations

from typing import Any

from davinci_monet.plots.titles import is_date_label, strip_trailing_date_title

SINGLE_SOURCE_SCHEMA_KEYS = {
    "type",
    "geometry",
    "source",
    "variable",
    "mode",
    "fig_kwargs",
    "default_plot_kwargs",
    "text_kwargs",
    "domain_type",
    "domain_name",
    "pairs",
    "data_proc",
}


FORWARDED_COMPARISON_OPTION_KEYS = {
    "show_site_labels",
    "show_individual_sites",
    "show_uncertainty",
    "uncertainty_type",
    "resample",
    "aggregate_dim",
    "label_sites",
    "site_label_var",
    "city_labels",
    "show_density",
    "density_cmap",
    "alpha",
    "marker_size",
    "x_edgecolor",
    "x_linewidth",
    "plot_type",
    "cmap",
    "show_surface_map",
    "surface_map_resolution",
    "land_color",
    "ocean_color",
    "show_var",
    "elev",
    "azim",
    "show_coastlines",
    "show_borders",
    "show_projection",
    "alt_scale",
    "split_by_flight",
    "flight_coord",
    "min_points",
    "style_preset",
    "levels",
    "extend",
    "robust",
    "robust_pct",
}


def single_source_plot_kwargs(
    plot_spec: dict[str, Any],
    *,
    analysis_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return renderer kwargs for a single-source plot spec."""
    kwargs = {k: v for k, v in plot_spec.items() if k not in SINGLE_SOURCE_SCHEMA_KEYS}

    subtitle = build_plot_subtitle(analysis_config or {})
    if subtitle and not kwargs.get("subtitle"):
        kwargs["subtitle"] = subtitle

    title = kwargs.get("title")
    if isinstance(title, str):
        kwargs["title"] = strip_trailing_date_title(title)

    if plot_spec.get("display_level") is not None:
        kwargs["display_level"] = plot_spec["display_level"]

    # Forward a fixed map domain to spatial maps so every map shares one extent
    # (sparse-data maps otherwise auto-clip to their few sites). Per-plot
    # domain_type wins; otherwise the analysis-level `domain` applies. domain_type
    # is otherwise filtered out of render kwargs (SINGLE_SOURCE_SCHEMA_KEYS), so it
    # is re-injected here for the renderer's _resolve_extent.
    if str(plot_spec.get("type", "")).startswith("spatial"):
        per_plot = plot_spec.get("domain_type")
        if isinstance(per_plot, (list, tuple)):
            per_plot = per_plot[0] if per_plot else None
        # The schema defaults domain_type to "all" (no restriction); treat that
        # (and None/empty) as "unset" so the analysis-level `domain` can apply.
        if per_plot in (None, "", "all"):
            per_plot = None
        domain = per_plot or (analysis_config or {}).get("domain")
        if domain:
            kwargs["domain_type"] = domain
            dname = plot_spec.get("domain_name")
            if isinstance(dname, (list, tuple)):
                dname = dname[0] if dname else None
            if dname and dname != "all":
                kwargs["domain_name"] = dname

    return kwargs


def single_source_flight_plot_kwargs(
    plot_kwargs: dict[str, Any],
    *,
    flight_id: Any,
) -> dict[str, Any]:
    """Return per-flight kwargs without putting date labels in titles."""
    base_title = plot_kwargs.get("title")
    if is_date_label(flight_id):
        return {**plot_kwargs, "subtitle": str(flight_id)}
    if base_title:
        return {**plot_kwargs, "title": f"{base_title} \u2014 {flight_id}"}
    return {**plot_kwargs, "title": str(flight_id)}


def build_comparison_plot_options(
    plot_type: str,
    plot_spec: dict[str, Any],
    analysis_config: dict[str, Any],
    *,
    nlevels: int | None = None,
) -> dict[str, Any]:
    """Return renderer options for paired comparison plots."""
    options = {k: plot_spec[k] for k in FORWARDED_COMPARISON_OPTION_KEYS if k in plot_spec}

    if (
        plot_type.startswith("spatial") or plot_type == "track_map_3d"
    ) and "city_labels" not in options:
        city_labels = analysis_config.get("city_labels")
        if city_labels:
            options["city_labels"] = city_labels

    if plot_type.startswith("spatial") and nlevels is not None:
        options.setdefault("n_levels", nlevels)

    return options


def build_plot_subtitle(
    analysis_config: dict[str, Any],
    *,
    snapshot_timestamp: str | None = None,
) -> str:
    """Return the date/timestamp subtitle line for a plot.

    Delegates to ``labeling.subtitle_text`` so the date format is consistent
    across the pipeline and all renderers.  A ``snapshot_timestamp`` (e.g. a
    single UTC string for spatial-overlay time slices) bypasses the date range
    and is returned verbatim.
    """
    if snapshot_timestamp:
        return snapshot_timestamp

    from davinci_monet.plots.labeling import subtitle_text

    start_time = analysis_config.get("start_time", "")
    end_time = analysis_config.get("end_time", "")
    return subtitle_text(start_time or None, end_time or None)


def timestamp_from_field(field: Any, time_index: int = 0) -> str:
    """Return a display timestamp from a time-indexed xarray field."""
    if "time" not in field.dims or len(field["time"]) <= time_index:
        return ""
    ts = field["time"].values[time_index]
    try:
        import pandas as pd

        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)[:16] + " UTC"
