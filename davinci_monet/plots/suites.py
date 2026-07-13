from __future__ import annotations

from typing import Any

from davinci_monet.core.schema_utils import dump_schema, is_schema_object

_AOD_TITLES = {
    "observation_aod": "Observed AOD",
    "analyzed_aod": "Analyzed AOD",
    "first_guess_aod": "Pre-Correction AOD",
    "analysis_minus_observation_aod": "Analysis Minus Observation AOD",
    "analysis_increment_aod": "AOD Correction",
    "nudge_fraction": "Nudge Fraction",
    "observation_fraction": "Observation Fraction",
    "analysis_minus_free_running_aod": "Analyzed Minus Free-Running AOD",
}

_AOD_MAP_CONTEXT = {
    "show_coastlines": True,
    "show_countries": False,
    "show_states": False,
    "show_gridlines": True,
    "gridline_style": "-",
    "land_color": "none",
    "ocean_color": "none",
}

_SARB_TITLES = {
    "visible_column_aod": "Visible Aerosol Optical Depth",
    "visible_single_scatter_albedo": "Visible Single-Scatter Albedo",
    "visible_asymmetry": "Visible Asymmetry Parameter",
    "sw_spectral_profile": "SW Spectral Profile",
    "dust_vertical_extinction": "Dust Vertical Extinction",
    "lw_window_scattering": "Longwave Window Scattering Optical Depth",
    "lw_window_extinction": "Longwave Window Extinction Optical Depth",
}

_GLOBAL_OVERRIDE_KEYS = {
    "fig_kwargs",
    "default_plot_kwargs",
    "text_kwargs",
    "domain_type",
    "domain_name",
    "formats",
}

_GROUP_TITLE_PREFIXES = {
    "daily": "Daily",
    "day": "Daily",
    "five_day": "Mean",
    "five-day": "Mean",
    "all": "Mean",
}


def _suite_dict(suite: dict[str, Any] | Any) -> dict[str, Any]:
    if is_schema_object(suite):
        return dump_schema(suite, exclude_none=True)
    return dict(suite)


def _plot_overrides(
    overrides: Any,
    field: str,
    known_fields: set[str],
    *,
    allow_unknown_dict_globals: bool,
) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        return {}
    global_options = {
        k: v
        for k, v in overrides.items()
        if k not in known_fields
        and (allow_unknown_dict_globals or k in _GLOBAL_OVERRIDE_KEYS or not isinstance(v, dict))
    }
    field_options = overrides.get(field, {})
    if not isinstance(field_options, dict):
        field_options = {}
    return {**global_options, **field_options}


def _aod_plot_options(field: str) -> dict[str, Any]:
    if field.endswith("_aod") and "minus" not in field and "increment" not in field:
        return {"style_preset": "geosit_aod"}
    if "minus" in field or "increment" in field:
        return {"cmap": "RdBu_r", "robust": True, "symmetric": True}
    if "fraction" in field:
        return {"cmap": "gray_r", "vmin": 0.0, "vmax": 1.0}
    return {}


def _field_title(field: str, default: str, field_metadata: dict[str, dict[str, Any]]) -> str:
    metadata = field_metadata.get(field, {})
    for key in ("display_name", "long_name", "standard_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def _title_with_group_prefix(title: str, group: Any) -> str:
    if not group:
        return title
    key = str(group).strip().lower().replace(" ", "_")
    prefix = _GROUP_TITLE_PREFIXES.get(key)
    if not prefix or title == prefix or title.startswith(f"{prefix} "):
        return title
    return f"{prefix} {title}"


def _sarb_plot_options(field: str) -> dict[str, Any]:
    if field in {"visible_column_aod", "lw_window_extinction"}:
        return {"style_preset": "geosit_aod"}
    if "scatter_albedo" in field or "asymmetry" in field:
        return {"vmin": 0.0, "vmax": 1.0}
    return {}


def _sarb_plot_type(field: str) -> str:
    if "vertical" in field:
        return "vertical_profile"
    if "spectral" in field:
        return "timeseries"
    return "spatial"


def _expand_aod_suite(
    suite_name: str,
    suite: dict[str, Any],
    *,
    available_fields: list[str],
    field_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    source = suite["source"]
    fields = suite.get("fields") or {}
    overrides = suite.get("overrides") or {}
    available = {str(field) for field in available_fields}
    field_metadata = field_metadata or {}
    plots: dict[str, dict[str, Any]] = {}
    for field, default_title in _AOD_TITLES.items():
        variable = str(fields.get(field, field))
        if variable not in available:
            continue
        title = _title_with_group_prefix(
            _field_title(variable, default_title, field_metadata),
            suite.get("group"),
        )
        plot = {
            "type": "spatial",
            "source": source,
            "variable": variable,
            "title": title,
            "domain_type": ["all"],
            "formats": ["pdf"],
            **_AOD_MAP_CONTEXT,
            **_aod_plot_options(field),
            **_plot_overrides(
                overrides,
                field,
                set(_AOD_TITLES),
                allow_unknown_dict_globals=True,
            ),
        }
        if suite.get("output_subdir"):
            plot["output_subdir"] = suite["output_subdir"]
        plots[f"{suite_name}_{field}"] = plot
    return plots


def _expand_sarb_suite(
    suite_name: str,
    suite: dict[str, Any],
    *,
    available_fields: list[str],
) -> dict[str, dict[str, Any]]:
    source = suite["source"]
    fields = suite.get("fields") or {}
    overrides = suite.get("overrides") or {}
    available = {str(field) for field in available_fields}
    plots: dict[str, dict[str, Any]] = {}
    for field, title in _SARB_TITLES.items():
        variable = str(fields.get(field, field))
        if variable not in available:
            continue
        plot = {
            "type": _sarb_plot_type(field),
            "source": source,
            "variable": variable,
            "title": title,
            "domain_type": ["all"],
            "formats": ["pdf"],
            **_sarb_plot_options(field),
            **_plot_overrides(
                overrides,
                field,
                set(_SARB_TITLES),
                allow_unknown_dict_globals=False,
            ),
        }
        if suite.get("output_subdir"):
            plot["output_subdir"] = suite["output_subdir"]
        plots[f"{suite_name}_{field}"] = plot
    return plots


def expand_plot_suite(
    suite_name: str,
    suite: dict[str, Any] | Any,
    *,
    available_fields: list[str],
    field_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Expand a named plot suite into concrete DAVINCI plot configs."""

    suite = _suite_dict(suite)
    preset = suite["preset"]
    if preset == "gridded_aod_diagnostics":
        return _expand_aod_suite(
            suite_name,
            suite,
            available_fields=available_fields,
            field_metadata=field_metadata,
        )
    if preset == "sarb_band_aerosol_optics":
        return _expand_sarb_suite(suite_name, suite, available_fields=available_fields)
    raise ValueError(f"unknown plot suite preset: {preset}")
