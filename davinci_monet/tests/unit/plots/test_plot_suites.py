from __future__ import annotations

import pytest

from davinci_monet.plots.suites import expand_plot_suite


def test_gridded_aod_suite_expands_absolute_and_difference_plots() -> None:
    plots = expand_plot_suite(
        "daily_aod",
        {
            "preset": "gridded_aod_diagnostics",
            "source": "daily_aod",
            "output_subdir": "plots/daily",
        },
        available_fields=[
            "observation_aod",
            "analyzed_aod",
            "first_guess_aod",
            "analysis_minus_observation_aod",
            "analysis_increment_aod",
            "nudge_fraction",
        ],
    )
    assert "daily_aod_analyzed_aod" in plots
    assert plots["daily_aod_analyzed_aod"]["type"] == "spatial"
    assert plots["daily_aod_analyzed_aod"]["style_preset"] == "geosit_aod"
    assert plots["daily_aod_analysis_increment_aod"]["cmap"] == "RdBu_r"
    assert plots["daily_aod_analysis_increment_aod"]["robust"] is True
    assert plots["daily_aod_analysis_increment_aod"]["symmetric"] is True
    assert plots["daily_aod_nudge_fraction"]["vmin"] == 0.0
    assert plots["daily_aod_nudge_fraction"]["vmax"] == 1.0
    assert plots["daily_aod_analyzed_aod"]["formats"] == ["pdf"]
    assert plots["daily_aod_nudge_fraction"]["output_subdir"] == "plots/daily"
    assert plots["daily_aod_analyzed_aod"]["show_countries"] is False
    assert plots["daily_aod_analyzed_aod"]["show_states"] is False
    assert plots["daily_aod_analyzed_aod"]["gridline_style"] == "-"


def test_gridded_aod_suite_uses_field_metadata_for_titles() -> None:
    plots = expand_plot_suite(
        "daily_aod",
        {
            "preset": "gridded_aod_diagnostics",
            "source": "daily_aod",
        },
        available_fields=["observation_aod", "analyzed_aod", "first_guess_aod"],
        field_metadata={
            "observation_aod": {"display_name": "MODIS Observed AOD"},
            "analyzed_aod": {"display_name": "CAM7 Analyzed AOD"},
            "first_guess_aod": {"display_name": "CAM7 Pre-Correction AOD"},
        },
    )

    assert plots["daily_aod_observation_aod"]["title"] == "MODIS Observed AOD"
    assert plots["daily_aod_analyzed_aod"]["title"] == "CAM7 Analyzed AOD"
    assert plots["daily_aod_first_guess_aod"]["title"] == "CAM7 Pre-Correction AOD"


def test_gridded_aod_suite_uses_mean_prefix_for_five_day_group() -> None:
    plots = expand_plot_suite(
        "five_day_aod",
        {
            "preset": "gridded_aod_diagnostics",
            "source": "five_day_aod",
            "group": "five_day",
        },
        available_fields=["analyzed_aod"],
        field_metadata={"analyzed_aod": {"display_name": "CAM7 Analyzed AOD"}},
    )

    assert plots["five_day_aod_analyzed_aod"]["title"] == "Mean CAM7 Analyzed AOD"
    assert "Five Day" not in plots["five_day_aod_analyzed_aod"]["title"]


def test_gridded_aod_suite_generic_titles_do_not_name_models_or_products() -> None:
    plots = expand_plot_suite(
        "daily_aod",
        {
            "preset": "gridded_aod_diagnostics",
            "source": "daily_aod",
        },
        available_fields=["observation_aod", "analyzed_aod", "first_guess_aod"],
    )

    joined_titles = "\n".join(plot["title"] for plot in plots.values())
    for banned in ("CAM", "CAM7", "MODIS", "VIIRS", "GEOS", "WRF"):
        assert banned not in joined_titles


def test_gridded_aod_suite_maps_fields_and_applies_overrides() -> None:
    plots = expand_plot_suite(
        "daily",
        {
            "preset": "gridded_aod_diagnostics",
            "source": "daily_aod",
            "output_subdir": "plots/daily",
            "fields": {"analyzed_aod": "cam_analyzed_aod"},
            "overrides": {
                "fig_kwargs": {"figsize": [8.0, 4.0]},
                "analyzed_aod": {"title": "Custom Analyzed AOD"},
            },
        },
        available_fields=["cam_analyzed_aod"],
    )
    assert list(plots) == ["daily_analyzed_aod"]
    plot = plots["daily_analyzed_aod"]
    assert plot["variable"] == "cam_analyzed_aod"
    assert plot["title"] == "Custom Analyzed AOD"
    assert plot["fig_kwargs"] == {"figsize": [8.0, 4.0]}


def test_gridded_aod_suite_preserves_unknown_dict_global_overrides() -> None:
    plots = expand_plot_suite(
        "daily",
        {
            "preset": "gridded_aod_diagnostics",
            "source": "daily_aod",
            "overrides": {"custom_renderer_options": {"enabled": True}},
        },
        available_fields=["analyzed_aod"],
    )

    assert plots["daily_analyzed_aod"]["custom_renderer_options"] == {"enabled": True}


def test_unknown_plot_suite_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown plot suite preset"):
        expand_plot_suite(
            "daily",
            {"preset": "missing", "source": "daily_aod"},
            available_fields=["analyzed_aod"],
        )
