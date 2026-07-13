from __future__ import annotations

import pytest

from davinci_monet.plots.suites import expand_plot_suite


def test_sarb_band_optics_suite_expands_expected_views() -> None:
    plots = expand_plot_suite(
        "optics",
        {
            "preset": "sarb_band_aerosol_optics",
            "source": "sarb_optics",
            "output_subdir": "plots/optics",
        },
        available_fields=[
            "visible_column_aod",
            "visible_single_scatter_albedo",
            "visible_asymmetry",
            "sw_spectral_profile",
            "dust_vertical_extinction",
            "lw_window_scattering",
            "lw_window_extinction",
        ],
    )

    assert "optics_visible_column_aod" in plots
    assert "optics_lw_window_scattering" in plots
    assert plots["optics_visible_column_aod"]["title"] == "Visible Aerosol Optical Depth"
    assert plots["optics_visible_asymmetry"]["title"] == "Visible Asymmetry Parameter"
    assert (
        plots["optics_lw_window_scattering"]["title"] == "Longwave Window Scattering Optical Depth"
    )
    assert (
        plots["optics_lw_window_extinction"]["title"] == "Longwave Window Extinction Optical Depth"
    )
    assert plots["optics_visible_column_aod"]["style_preset"] == "geosit_aod"
    assert plots["optics_lw_window_scattering"]["type"] == "spatial"
    assert plots["optics_sw_spectral_profile"]["type"] == "timeseries"
    assert plots["optics_dust_vertical_extinction"]["type"] == "vertical_profile"
    assert plots["optics_lw_window_extinction"]["formats"] == ["pdf"]
    assert plots["optics_lw_window_extinction"]["output_subdir"] == "plots/optics"


def test_sarb_band_optics_suite_ignores_unavailable_fields() -> None:
    plots = expand_plot_suite(
        "optics",
        {"preset": "sarb_band_aerosol_optics", "source": "sarb_optics"},
        available_fields=["visible_column_aod"],
    )

    assert list(plots) == ["optics_visible_column_aod"]


def test_sarb_band_optics_suite_allows_overrides() -> None:
    plots = expand_plot_suite(
        "optics",
        {
            "preset": "sarb_band_aerosol_optics",
            "source": "sarb_optics",
            "overrides": {
                "fig_kwargs": {"figsize": [8.0, 4.0]},
                "lw_window_scattering": {"title": "Custom LW Scattering"},
            },
        },
        available_fields=["lw_window_scattering"],
    )

    plot = plots["optics_lw_window_scattering"]
    assert plot["title"] == "Custom LW Scattering"
    assert plot["fig_kwargs"] == {"figsize": [8.0, 4.0]}


def test_sarb_band_optics_suite_expands_named_spectral_fields() -> None:
    plots = expand_plot_suite(
        "uv_optics",
        {
            "preset": "sarb_band_aerosol_optics",
            "source": "uv_optics",
            "fields": {
                "ultraviolet_column_aod": "column_aod",
                "ultraviolet_single_scatter_albedo": "single_scatter_albedo",
                "ultraviolet_asymmetry": "asymmetry",
            },
        },
        available_fields=["column_aod", "single_scatter_albedo", "asymmetry"],
        field_metadata={
            "column_aod": {"display_name": "Ultraviolet Aerosol Optical Depth (0.340 µm)"},
            "single_scatter_albedo": {
                "display_name": "Ultraviolet Single-Scatter Albedo (0.340 µm)"
            },
            "asymmetry": {"display_name": "Ultraviolet Asymmetry Parameter (0.340 µm)"},
        },
    )

    aod = plots["uv_optics_ultraviolet_column_aod"]
    ssa = plots["uv_optics_ultraviolet_single_scatter_albedo"]
    asymmetry = plots["uv_optics_ultraviolet_asymmetry"]
    assert aod["title"] == "Ultraviolet Aerosol Optical Depth (0.340 µm)"
    assert aod["style_preset"] == "geosit_aod"
    assert aod["global_mean_decimals"] == 3
    assert ssa["nice_range"] is True
    assert ssa["nice_range_bounds"] == [0.0, 1.0]
    assert asymmetry["nice_range"] is True
    assert asymmetry["robust_pct"] == [2.0, 98.0]


def test_unknown_sarb_field_override_is_not_global() -> None:
    plots = expand_plot_suite(
        "optics",
        {
            "preset": "sarb_band_aerosol_optics",
            "source": "sarb_optics",
            "overrides": {"not_a_field": {"title": "Ignored"}},
        },
        available_fields=["visible_column_aod"],
    )

    assert "not_a_field" not in plots["optics_visible_column_aod"]


def test_unknown_sarb_suite_preset_still_raises() -> None:
    with pytest.raises(ValueError, match="unknown plot suite preset"):
        expand_plot_suite(
            "optics",
            {"preset": "missing", "source": "sarb_optics"},
            available_fields=["visible_column_aod"],
        )
