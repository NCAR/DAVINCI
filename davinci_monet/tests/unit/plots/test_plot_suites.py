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


def test_unknown_plot_suite_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown plot suite preset"):
        expand_plot_suite(
            "daily",
            {"preset": "missing", "source": "daily_aod"},
            available_fields=["analyzed_aod"],
        )
