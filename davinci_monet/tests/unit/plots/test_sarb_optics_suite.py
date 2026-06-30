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
