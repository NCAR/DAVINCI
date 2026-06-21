"""Tests for davinci_monet.plots.labeling — TDD, task by task."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.plots import labeling as L

# ---------------------------------------------------------------------------
# Task 1: format_units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("mol/m2", "mol m$^{-2}$"),
        ("W m-2", "W m$^{-2}$"),
        ("mol/mol", "mol mol$^{-1}$"),
        ("kg/kg", "kg kg$^{-1}$"),
        ("m/s", "m s$^{-1}$"),
        ("ppb", "ppb"),
        ("K", "K"),
        ("1", ""),
        ("none", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_format_units(raw, expected):
    assert L.format_units(raw) == expected


def test_format_units_micrograms():
    assert L.format_units("ug/m3") == r"$\mu$g m$^{-3}$"


# ---------------------------------------------------------------------------
# Task 2: source_display_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        ("cesm_no2_column", r"CESM NO$_2$ Column"),
        ("airnow", "AirNow"),
        ("pandora", "Pandora"),
        ("merra2", "MERRA-2"),
        ("ceres", "CERES"),
        ("cam", "CAM"),
        ("", ""),
        (None, ""),
    ],
)
def test_source_display_name(key, expected):
    assert L.source_display_name(key) == expected


# ---------------------------------------------------------------------------
# Task 3: quantity_label
# ---------------------------------------------------------------------------


def _ds(var: str, **attrs: object) -> xr.Dataset:
    d = xr.Dataset({var: ("t", np.arange(3.0))})
    d[var].attrs.update(attrs)
    return d


def test_quantity_label_from_lookup():
    ds = _ds("no2_column")
    assert L.quantity_label(ds, "no2_column") == r"NO$_2$ Column"


def test_quantity_label_prefers_long_name():
    ds = _ds("X", long_name="Tropospheric NO2 Column")
    assert "Tropospheric" in L.quantity_label(ds, "X")


def test_quantity_label_subscripts_formula_from_long_name():
    # A chemical formula coming from a raw long_name must be subscripted and
    # title-cased so the colorbar/axis matches the title (NO2 -> NO$_2$, "total
    # column" -> "Total Column").
    ds = _ds("X", long_name="NO2 total column")
    assert L.quantity_label(ds, "X") == r"NO$_2$ Total Column"


# ---------------------------------------------------------------------------
# Task 4: axis_label
# ---------------------------------------------------------------------------


def test_axis_label_no_source():
    assert L.axis_label(r"NO$_2$ Column", "mol/m2") == r"NO$_2$ Column (mol m$^{-2}$)"


def test_axis_label_with_source():
    assert (
        L.axis_label(r"NO$_2$ Column", "mol/m2", source="pandora")
        == r"Pandora NO$_2$ Column (mol m$^{-2}$)"
    )


def test_axis_label_dedup():
    out = L.axis_label(r"NO$_2$ Column", "mol/m2", source="cesm_no2_column")
    assert out == r"CESM NO$_2$ Column (mol m$^{-2}$)"


def test_axis_label_dedup_partial_overlap():
    # source key embeds "NO2 Column" but the quantity is the longer
    # "Tropospheric NO2 Column" -> keep only the distinctive source token (CESM),
    # do NOT repeat "NO2 Column".
    out = L.axis_label(r"Tropospheric NO$_2$ Column", "mol/m2", source="cesm_no2_column")
    assert out == r"CESM Tropospheric NO$_2$ Column (mol m$^{-2}$)"


def test_axis_label_dedup_partial_quantity_keeps_word_order():
    # quantity "NO2" is a subset of the source's embedded "NO2 Column" -> use the
    # source name in good order, NOT a reordered "CESM Column NO2".
    out = L.axis_label(r"NO$_2$", "mol/m2", source="cesm_no2_column")
    assert out == r"CESM NO$_2$ Column (mol m$^{-2}$)"


def test_axis_label_no_units():
    assert L.axis_label("Altitude", "1") == "Altitude"


def test_axis_label_normalizes_raw_quantity():
    # Renderers (e.g. timeseries) pass a raw long_name straight to axis_label
    # (bypassing quantity_label), so axis_label itself must normalize: species
    # word -> formula and casing.
    assert L.axis_label("Ozone", "ppbv") == r"O$_3$ (ppbv)"
    assert L.axis_label("NO2 total column", "mol/m2") == r"NO$_2$ Total Column (mol m$^{-2}$)"


def test_axis_label_normalization_idempotent_on_lookup():
    # Already-formatted lookup quantities must be unchanged by the normalization.
    assert L.axis_label("AOD (500 nm)", "1") == "AOD (500 nm)"
    assert L.axis_label(r"Tropospheric NO$_2$ Column", "mol/m2") == (
        r"Tropospheric NO$_2$ Column (mol m$^{-2}$)"
    )


# ---------------------------------------------------------------------------
# Task 5: legend_label
# ---------------------------------------------------------------------------


def test_legend_label_plain():
    assert L.legend_label("cesm_no2_column") == r"CESM NO$_2$ Column"


def test_legend_label_uncertainty():
    assert L.legend_label("pandora", uncertainty="mean ± σ") == "Pandora (mean ± σ)"


# ---------------------------------------------------------------------------
# Task 6: bias_label
# ---------------------------------------------------------------------------


def test_bias_label_factors_shared_quantity():
    # 'Bias' lives in the title, so the colorbar is just direction + units.
    out = L.bias_label("merra2_olr", "ceres_olr", "W m-2")
    assert out == "MERRA-2 − CERES (W m$^{-2}$)"


def test_bias_label_no_shared_quantity():
    out = L.bias_label("cesm_no2_column", "pandora", "mol/m2")
    assert out == "CESM NO$_2$ Column − Pandora (mol m$^{-2}$)"


def test_bias_label_never_uses_xy():
    out = L.bias_label("cesm_no2_column", "pandora", None)
    assert " x" not in out.lower() and " y" not in out.lower()


def test_bias_label_no_redundant_bias_word():
    # The colorbar must NOT repeat "Bias" (the title carries it).
    out = L.bias_label("merra2_olr", "ceres_olr", "W m-2")
    assert "Bias" not in out


def test_bias_label_strips_quantity_when_given():
    # quantity is already in the title -> strip it from the source names so the
    # colorbar stays terse: "CESM − Pandora".
    out = L.bias_label("cesm_no2_column", "pandora", "mol/m2", quantity=r"NO$_2$ Column")
    assert out == "CESM − Pandora (mol m$^{-2}$)"


# ---------------------------------------------------------------------------
# Task 7: title_text + subtitle_text
# ---------------------------------------------------------------------------


def test_title_text_terse():
    assert L.title_text("NO2 Tropospheric Column") == r"NO$_2$ Tropospheric Column"


def test_title_text_operation():
    assert L.title_text("OLR", operation="Bias") == "OLR Bias"


def test_subtitle_range():
    assert L.subtitle_text("2024-02-01", "2024-02-29") == "2024-02-01 – 2024-02-29"


def test_subtitle_single():
    assert L.subtitle_text("2024-02-01", "2024-02-01") == "2024-02-01"


def test_subtitle_empty():
    assert L.subtitle_text(None, None) == ""


# ---------------------------------------------------------------------------
# sourced_title: single-source title with the SOURCE NAME FIRST (not in parens)
# ---------------------------------------------------------------------------


def test_sourced_title_source_first():
    # Source leads the title; no parenthesised "(AERONET)" suffix.
    assert L.sourced_title("aeronet", r"AOD (500 nm)") == r"AERONET AOD (500 nm)"
    assert L.sourced_title("airnow", r"O$_3$") == r"AirNow O$_3$"


def test_sourced_title_dedup():
    # Source key embeds the quantity -> only the distinctive token leads.
    assert L.sourced_title("cesm_no2_column", r"NO$_2$ Column") == r"CESM NO$_2$ Column"


def test_sourced_title_no_source():
    assert L.sourced_title(None, "O3") == "O3"
    assert L.sourced_title("", "O3") == "O3"


# ---------------------------------------------------------------------------
# Task 8: labels.format_units delegates to SI labeling
# ---------------------------------------------------------------------------


def test_labels_format_units_delegates_to_si():
    from davinci_monet.plots import labels

    assert labels.format_units("mol/m2") == "mol m$^{-2}$"


# ---------------------------------------------------------------------------
# Fix #5: source_display_name — skip empty tokens (double-underscore safety)
# ---------------------------------------------------------------------------


def test_source_display_name_double_underscore():
    """Double underscores (empty tokens) must not produce double spaces."""
    result = L.source_display_name("cesm__no2")
    assert "  " not in result, f"Double space found in: {result!r}"


def test_source_display_name_leading_underscore():
    """Leading underscore must not produce a leading space."""
    result = L.source_display_name("_cesm")
    assert not result.startswith(" "), f"Leading space found in: {result!r}"


def test_source_display_name_trailing_underscore():
    """Trailing underscore must not produce a trailing space."""
    result = L.source_display_name("cesm_")
    assert not result.endswith(" "), f"Trailing space found in: {result!r}"


# ---------------------------------------------------------------------------
# Fix #6: format_units ug replace — word-boundary safe
# ---------------------------------------------------------------------------


def test_format_units_ug_m3():
    """'ug/m3' must render as µg m^{-3}."""
    assert L.format_units("ug/m3") == r"$\mu$g m$^{-3}$"


def test_format_units_ug_space_m3():
    """'ug m-3' (space-separated) must render correctly."""
    assert L.format_units("ug m-3") == r"$\mu$g m$^{-3}$"


def test_format_units_bug_not_replaced():
    """'bug' must NOT have its 'ug' substring replaced."""
    result = L.format_units("bug")
    assert r"$\mu$g" not in result, f"False 'ug' substitution in 'bug': {result!r}"


def test_format_units_slug_not_replaced():
    """'slug' must NOT have its 'ug' substring replaced."""
    result = L.format_units("slug")
    assert r"$\mu$g" not in result, f"False 'ug' substitution in 'slug': {result!r}"


# ---------------------------------------------------------------------------
# quantity_label normalization (species-word → formula + smart title-case)
# ---------------------------------------------------------------------------


def test_quantity_label_ozone_word_to_formula():
    """long_name='Ozone' must become O$_3$, not the bare word."""
    ds = _ds("X", long_name="Ozone")
    assert L.quantity_label(ds, "X") == r"O$_3$"


def test_quantity_label_no2_total_column_title_case():
    """'NO2 total column' → 'NO$_2$ Total Column' (formula subscript + title-case)."""
    ds = _ds("X", long_name="NO2 total column")
    assert L.quantity_label(ds, "X") == r"NO$_2$ Total Column"


def test_quantity_label_carbon_monoxide_to_formula():
    """'carbon monoxide' (multi-word species name) → 'CO'."""
    ds = _ds("X", long_name="carbon monoxide")
    assert L.quantity_label(ds, "X") == "CO"


def test_quantity_label_ozone_mixing_ratio():
    """'ozone mixing ratio' → 'O$_3$ Mixing Ratio'."""
    ds = _ds("X", long_name="ozone mixing ratio")
    assert L.quantity_label(ds, "X") == r"O$_3$ Mixing Ratio"


# --- no-regress cases -------------------------------------------------------


def test_quantity_label_aod_500nm_preserved():
    """'AOD (500 nm)' must come through byte-identical: nm lowercase, 500 intact."""
    ds = _ds("X", long_name="AOD (500 nm)")
    assert L.quantity_label(ds, "X") == "AOD (500 nm)"


def test_quantity_label_toa_lw_all_mon_preserved():
    """All-upper-case tokens (TOA, LW) must stay all-caps; 'Mon' capitalised."""
    ds = _ds("X", long_name="TOA LW All Mon")
    assert L.quantity_label(ds, "X") == "TOA LW All Mon"


def test_quantity_label_tropospheric_no2_column():
    """'Tropospheric NO2 Column' → 'Tropospheric NO$_2$ Column'."""
    ds = _ds("X", long_name="Tropospheric NO2 Column")
    assert L.quantity_label(ds, "X") == r"Tropospheric NO$_2$ Column"


def test_quantity_label_lookup_no2_column_unchanged():
    """Lookup-table var no2_column (no long_name) must still give NO$_2$ Column."""
    ds = _ds("no2_column")
    assert L.quantity_label(ds, "no2_column") == r"NO$_2$ Column"


def test_quantity_label_lookup_o3_unchanged():
    """Lookup-table var o3 (no long_name) must still give O$_3$."""
    ds = _ds("o3")
    assert L.quantity_label(ds, "o3") == r"O$_3$"


def test_quantity_label_toa_olr_preserved():
    """'TOA OLR' — all-caps tokens must stay all-caps."""
    ds = _ds("X", long_name="TOA OLR")
    assert L.quantity_label(ds, "X") == "TOA OLR"
