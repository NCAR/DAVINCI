"""Source-name label tests for NASA POWER.

Both cases are defects observed in rendered figures from the v1 POWER
campaign (2026-07-15), not hypotheticals.
"""

from __future__ import annotations

from davinci_monet.plots.labeling import source_display_name


def test_power_renders_as_the_acronym_not_title_case() -> None:
    """Observed: axes read "Power Surface Downwelling Shortwave".

    POWER is an acronym (Prediction Of Worldwide Energy Resources), so
    title-casing it to "Power" is simply wrong.
    """
    assert source_display_name("power") == "POWER"


def test_power_survives_as_an_acronym_inside_a_compound_key() -> None:
    """Observed: a bias colorbar read "MERRA-2 Rad - Power Grid".

    In a NASA POWER talk that parses as electrical infrastructure.
    """
    assert source_display_name("power_regional") == "POWER Regional"
    assert source_display_name("power_daily") == "POWER Daily"


def test_unrelated_sources_are_unaffected() -> None:
    """The acronym entry is additive: it must fire only on a `power` token."""
    assert source_display_name("merra2") == "MERRA-2"
    assert source_display_name("airnow") == "AirNow"
    assert source_display_name("cesm") == "CESM"
