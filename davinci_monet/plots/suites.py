from __future__ import annotations

from typing import Any

from davinci_monet.core.schema_utils import dump_schema, is_schema_object

_AOD_TITLES = {
    "observation_aod": "Observation AOD",
    "analyzed_aod": "CAM Analyzed AOD",
    "first_guess_aod": "CAM First-Guess AOD",
    "analysis_minus_observation_aod": "Analysis Minus Observation AOD",
    "analysis_increment_aod": "Analysis Increment AOD",
    "nudge_fraction": "Nudge Fraction",
    "observation_fraction": "Observation Fraction",
    "analysis_minus_free_running_aod": "Analyzed Minus Free-Running AOD",
}


def _suite_dict(suite: dict[str, Any] | Any) -> dict[str, Any]:
    if is_schema_object(suite):
        return dump_schema(suite, exclude_none=True)
    return dict(suite)


def _plot_overrides(overrides: Any, field: str) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        return {}
    global_options = {k: v for k, v in overrides.items() if k not in _AOD_TITLES}
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


def expand_plot_suite(
    suite_name: str,
    suite: dict[str, Any] | Any,
    *,
    available_fields: list[str],
) -> dict[str, dict[str, Any]]:
    """Expand a named plot suite into concrete DAVINCI plot configs."""

    suite = _suite_dict(suite)
    preset = suite["preset"]
    if preset != "gridded_aod_diagnostics":
        raise ValueError(f"unknown plot suite preset: {preset}")
    source = suite["source"]
    fields = suite.get("fields") or {}
    overrides = suite.get("overrides") or {}
    available = {str(field) for field in available_fields}
    plots: dict[str, dict[str, Any]] = {}
    for field, title in _AOD_TITLES.items():
        variable = str(fields.get(field, field))
        if variable not in available:
            continue
        plot = {
            "type": "spatial",
            "source": source,
            "variable": variable,
            "title": title,
            "domain_type": ["all"],
            "formats": ["pdf"],
            **_aod_plot_options(field),
            **_plot_overrides(overrides, field),
        }
        if suite.get("output_subdir"):
            plot["output_subdir"] = suite["output_subdir"]
        plots[f"{suite_name}_{field}"] = plot
    return plots
