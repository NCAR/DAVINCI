"""Tests for plotting-stage option assembly helpers."""

from __future__ import annotations


def test_comparison_plot_options_are_assembled_outside_stage() -> None:
    from davinci_monet.pipeline.stages.plot_options import build_comparison_plot_options

    plot_spec = {
        "show_density": True,
        "marker_size": 12,
        "ignored": "not forwarded",
    }
    analysis_config = {"city_labels": [{"name": "Boulder", "lat": 40.0, "lon": -105.2}]}

    options = build_comparison_plot_options(
        "spatial_overlay",
        plot_spec,
        analysis_config,
        nlevels=21,
    )

    assert options["show_density"] is True
    assert options["marker_size"] == 12
    assert options["city_labels"] == analysis_config["city_labels"]
    assert options["n_levels"] == 21
    assert "ignored" not in options


def test_spatial_style_options_are_forwarded_from_single_source_plot_specs() -> None:
    from davinci_monet.pipeline.stages.plot_options import single_source_plot_kwargs

    kwargs = single_source_plot_kwargs(
        {
            "type": "spatial",
            "source": "modis",
            "variable": "aod",
            "style_preset": "geosit_aod",
            "levels": [0.0, 0.1, 0.5, 1.0],
            "cmap": "turbo",
            "extend": "max",
        }
    )

    assert kwargs["style_preset"] == "geosit_aod"
    assert kwargs["levels"] == [0.0, 0.1, 0.5, 1.0]
    assert kwargs["cmap"] == "turbo"
    assert kwargs["extend"] == "max"


def test_spatial_style_options_are_forwarded_from_comparison_plot_specs() -> None:
    from davinci_monet.pipeline.stages.plot_options import build_comparison_plot_options

    options = build_comparison_plot_options(
        "spatial",
        {
            "style_preset": "geosit_aod",
            "levels": [0.0, 0.1, 0.5, 1.0],
            "cmap": "turbo",
            "extend": "max",
        },
        {},
    )

    assert options == {
        "style_preset": "geosit_aod",
        "levels": [0.0, 0.1, 0.5, 1.0],
        "cmap": "turbo",
        "extend": "max",
    }


def test_spatial_robust_options_are_forwarded_from_comparison_plot_specs() -> None:
    from davinci_monet.pipeline.stages.plot_options import build_comparison_plot_options

    options = build_comparison_plot_options(
        "spatial",
        {
            "robust": True,
            "robust_pct": [5.0, 95.0],
        },
        {},
    )

    assert options == {"robust": True, "robust_pct": [5.0, 95.0]}


def test_plot_subtitle_uses_date_range_or_snapshot() -> None:
    from davinci_monet.pipeline.stages.plot_options import build_plot_subtitle

    assert (
        build_plot_subtitle(
            {"start_time": "2024-01-01", "end_time": "2024-01-03"},
            snapshot_timestamp=None,
        )
        == "2024-01-01 – 2024-01-03"
    )
    assert (
        build_plot_subtitle(
            {"start_time": "2024-01-01", "end_time": "2024-01-03"},
            snapshot_timestamp="2024-01-02 12:00 UTC",
        )
        == "2024-01-02 12:00 UTC"
    )


def test_single_source_options_move_trailing_date_to_subtitle() -> None:
    from davinci_monet.pipeline.stages.plot_options import single_source_plot_kwargs

    kwargs = single_source_plot_kwargs(
        {
            "type": "flight_track",
            "source": "dc8",
            "variable": "O3",
            "title": "DC-8 Flight Track: O3 (ppbv) \u2014 29 May 2012",
        },
        analysis_config={"start_time": "2012-05-29", "end_time": "2012-05-30"},
    )

    assert kwargs["title"] == "DC-8 Flight Track: O3 (ppbv)"
    assert kwargs["subtitle"] == "2012-05-29 – 2012-05-30"


def test_single_source_spatial_inherits_analysis_domain() -> None:
    """Analysis-level `domain` is forwarded to spatial maps as domain_type so
    every map shares the same fixed extent."""
    from davinci_monet.pipeline.stages.plot_options import single_source_plot_kwargs

    kw = single_source_plot_kwargs(
        {"type": "spatial", "source": "airnow", "variable": "o3"},
        analysis_config={"domain": "asia_aq"},
    )
    assert kw["domain_type"] == "asia_aq"


def test_single_source_default_domain_type_yields_analysis_domain() -> None:
    """The schema's default domain_type ('all' / ['all']) means 'no restriction'
    and must be treated as unset, so the analysis-level domain still applies."""
    from davinci_monet.pipeline.stages.plot_options import single_source_plot_kwargs

    kw = single_source_plot_kwargs(
        {
            "type": "spatial",
            "source": "airnow",
            "variable": "o3",
            "domain_type": ["all"],
            "domain_name": ["all"],
        },
        analysis_config={"domain": "asia_aq"},
    )
    assert kw["domain_type"] == "asia_aq"


def test_single_source_per_plot_domain_overrides_analysis() -> None:
    """A per-plot domain_type wins over the analysis-level domain."""
    from davinci_monet.pipeline.stages.plot_options import single_source_plot_kwargs

    kw = single_source_plot_kwargs(
        {"type": "spatial", "source": "airnow", "variable": "o3", "domain_type": "conus"},
        analysis_config={"domain": "asia_aq"},
    )
    assert kw["domain_type"] == "conus"


def test_non_spatial_single_source_ignores_analysis_domain() -> None:
    """Non-spatial plots (e.g. timeseries) do not get a map domain."""
    from davinci_monet.pipeline.stages.plot_options import single_source_plot_kwargs

    kw = single_source_plot_kwargs(
        {"type": "timeseries", "source": "airnow", "variable": "o3"},
        analysis_config={"domain": "asia_aq"},
    )
    assert "domain_type" not in kw


def test_single_source_flight_date_label_stays_out_of_title() -> None:
    from davinci_monet.pipeline.stages.plot_options import single_source_flight_plot_kwargs

    kwargs = single_source_flight_plot_kwargs(
        {
            "title": "DC-8 O3 Time Series",
            "subtitle": "2012-05-29 - 2012-05-30",
        },
        flight_id="2012-05-29",
    )

    assert kwargs["title"] == "DC-8 O3 Time Series"
    assert kwargs["subtitle"] == "2012-05-29"
