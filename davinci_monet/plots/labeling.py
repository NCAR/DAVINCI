"""Central, publication-quality text composition for all DAVINCI plots.

Pure functions only — no matplotlib, no I/O. Renderers call these instead of
building label/title strings themselves.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from davinci_monet.plots.labels import (
    _QUANTITY_LOWERCASE_KEEP,
    SPECIES_WORD_TO_FORMULA,
    VARIABLE_DISPLAY_NAMES,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)

if TYPE_CHECKING:
    import xarray as xr

_DIMENSIONLESS = {"", "1", "none", "dimensionless", "unitless", "fraction"}


def format_units(units: str | None) -> str:
    """Rewrite a raw unit string to negative-exponent SI LaTeX (D3).

    Strings that already contain non-ASCII characters or LaTeX (``$``) are
    returned unchanged — they are assumed to be pre-formatted.
    """
    if units is None:
        return ""
    s = str(units).strip()
    if s.lower() in _DIMENSIONLESS:
        return ""
    # Pass through pre-formatted strings (Unicode or LaTeX already present).
    if "$" in s or not s.isascii():
        return s
    s = re.sub(r"(?<![A-Za-z])ug(?![A-Za-z])", r"$\\mu$g", s)

    # division: "/unitN" -> " unit^{-N}" (N defaults to 1)
    def _div(m: re.Match[str]) -> str:
        unit, power = m.group(1), m.group(2) or "1"
        return f" {unit}$^{{-{power}}}$"

    s = re.sub(r"/([A-Za-z]+)(\d+)?", _div, s)
    # space/dash negative exponents: "unit-N" -> "unit^{-N}"
    s = re.sub(r"([A-Za-z])-(\d+)", lambda m: f"{m.group(1)}$^{{-{m.group(2)}}}$", s)
    # remaining positive exponents: "unitN" -> "unit^{N}"
    s = re.sub(r"([A-Za-z])(\d+)", lambda m: f"{m.group(1)}$^{{{m.group(2)}}}$", s)
    return s.strip()


_SOURCE_ACRONYMS = {
    "cesm": "CESM",
    "cam": "CAM",
    "wrf": "WRF",
    "wrfchem": "WRF-Chem",
    "merra2": "MERRA-2",
    "geoschem": "GEOS-Chem",
    "geos": "GEOS",
    "airnow": "AirNow",
    "aeronet": "AERONET",
    "pandora": "Pandora",
    "ceres": "CERES",
    "modis": "MODIS",
    "tropomi": "TROPOMI",
    "tempo": "TEMPO",
    "ufs": "UFS",
    "cmaq": "CMAQ",
    "ebaf": "EBAF",
    "ssf": "SSF",
    # NASA POWER (Prediction Of Worldwide Energy Resources). Without this it
    # title-cases to "Power", and a key like `power_grid` renders "Power Grid"
    # -- which reads as electrical infrastructure rather than the NASA product.
    "power": "POWER",
}


def source_display_name(source_label: str | None) -> str:
    """Friendly source name from a raw config key (D5); never ALL-CAPS."""
    if not source_label:
        return ""
    out = []
    for tok in str(source_label).split("_"):
        if not tok:
            continue
        low = tok.lower()
        if low in _SOURCE_ACRONYMS:
            out.append(_SOURCE_ACRONYMS[low])
        elif low in VARIABLE_DISPLAY_NAMES:
            out.append(VARIABLE_DISPLAY_NAMES[low])
        else:
            out.append(tok.capitalize())
    return " ".join(out)


def _normalize_quantity_label(raw: str) -> str:
    """Normalize an auto-derived quantity label for consistent display.

    Applied only to labels that originate from raw data attributes
    (``long_name``, ``standard_name``) — NOT to user-supplied titles or
    lookup-table names.  Three passes, in order:

    1. Species word/phrase → LaTeX formula (case-insensitive, whole-word,
       longest-first, skips tokens already inside ``$…$``).
    2. Smart title-case: capitalise each word EXCEPT tokens that contain
       ``$`` (already LaTeX), tokens containing a digit, all-uppercase
       tokens, and a keep-set of unit abbreviations and particles.
    3. Formula subscript via ``format_plot_title`` (idempotent).
    """
    text = raw

    # ---- pass 1: species word → formula ------------------------------------
    # We must not touch substrings that are already inside $…$.  Strategy:
    # split on dollar-sign delimited segments and only process plain-text
    # segments.
    def _apply_species_map(s: str) -> str:
        """Replace species words in a plain-text segment."""
        for phrase, formula in SPECIES_WORD_TO_FORMULA:
            # Whole-word / whole-phrase match, case-insensitive.
            pattern = r"(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])"
            s = re.sub(pattern, formula, s, flags=re.IGNORECASE)
        return s

    # Split into (plain, latex, plain, latex, …) alternating segments.
    # Dollar-sign blocks: $...$ (non-nested, greedy).
    parts = re.split(r"(\$[^$]*\$)", text)
    text = "".join(
        part if (i % 2 == 1) else _apply_species_map(part) for i, part in enumerate(parts)
    )

    # ---- pass 2: smart title-case ------------------------------------------
    def _title_case_token(tok: str) -> str:
        # Tokens that are already LaTeX (contain $): leave untouched.
        if "$" in tok:
            return tok
        # Tokens containing a digit: leave untouched (NO2, 500, 2.5).
        if any(c.isdigit() for c in tok):
            return tok
        # All-uppercase tokens (TOA, LW, AOD, OLR, CO, UV, …): leave as-is.
        # A token is "all-uppercase" if stripping punctuation leaves only
        # uppercase ASCII letters (length ≥ 2, or a known 1-char acronym).
        alpha = re.sub(r"[^A-Za-z]", "", tok)
        if alpha and alpha == alpha.upper() and (len(alpha) >= 2 or alpha in {"K", "W"}):
            return tok
        # Keep-set (units + particles): strip surrounding punctuation before
        # comparing.
        stripped = tok.strip("()[].,;:!?").lower()
        if stripped in _QUANTITY_LOWERCASE_KEEP:
            return tok
        # Default: capitalise first letter, leave rest.
        return tok[0].upper() + tok[1:] if tok else tok

    # Tokenise preserving spaces and punctuation attached to words.
    # We split on spaces to handle multi-char tokens like "(500".
    text = " ".join(_title_case_token(t) for t in text.split(" "))

    # ---- pass 3: formula subscript (idempotent) ----------------------------
    return format_plot_title(text)


def quantity_label(dataset: "xr.Dataset", var_name: str) -> str:
    """Quantity name only (no source, no units), normalised and chem-formatted.

    When the label comes from the lookup table or a ``display_name`` attr it
    is returned as-is after a formula-subscript pass (idempotent for
    already-formatted strings).

    When the label comes from a raw ``long_name`` / ``standard_name`` attr it
    is additionally normalised:
      1. Species words/phrases are replaced with LaTeX formulas
         (e.g. "Ozone" → "O$_3$", "nitrogen dioxide" → "NO$_2$").
      2. Smart title-case is applied (preserving existing LaTeX tokens,
         digit-containing tokens, all-uppercase acronyms, and a keep-set of
         unit abbreviations and particles).
      3. Chemical formula subscripts are applied via ``format_plot_title``.

    ``title_text`` is deliberately NOT changed — user-supplied titles must
    pass through unchanged.
    """
    raw = get_variable_label(dataset, var_name, include_prefix=False)

    # Detect whether the label came from a raw attr or the lookup / display_name
    # path. Raw attrs produce labels that differ from what the lookup would give
    # for the same var. A simple heuristic: check the variable's own attrs.
    var_from_attr = False
    if var_name in dataset:
        attrs = dataset[var_name].attrs
        if attrs.get("display_name"):
            # display_name is already publication-quality: only do subscripts.
            return format_plot_title(raw)
        if attrs.get("long_name") or attrs.get("standard_name"):
            var_from_attr = True

    if var_from_attr:
        return _normalize_quantity_label(raw)
    # Lookup-table / format_variable_display_name path: subscripts only.
    return format_plot_title(raw)


def _distinctive_source_tokens(source: str, quantity: str) -> list[str]:
    """Source-name tokens that are NOT already part of the quantity.

    Lets a source key that redundantly embeds the quantity (e.g.
    ``cesm_no2_column``) contribute only its distinctive identifier (``CESM``)
    when placed next to that quantity, avoiding repetition.
    """
    q_tokens = set((quantity or "").split())
    return [t for t in source_display_name(source).split() if t not in q_tokens]


def distinctive_source_name(source_label: str, quantity: str) -> str:
    """Source display name for use NEXT TO a quantity (e.g. a single-source map
    title ``Quantity (Source)``), with quantity-overlapping tokens removed so a
    key like ``cesm_no2_column`` beside ``NO2 Column`` yields ``CESM`` (not
    ``CESM NO2 Column``). Falls back to the full display name if nothing is left.
    """
    toks = _distinctive_source_tokens(source_label, quantity)
    return " ".join(toks) or source_display_name(source_label)


def axis_label(quantity: str, units: str | None, source: str | None = None) -> str:
    """Compose an axis label (D2 context-aware, with de-dup).

    When the source name overlaps the quantity, only the source's distinctive
    tokens are kept, so e.g. ``cesm_no2_column`` + ``Tropospheric NO2 Column``
    renders ``CESM Tropospheric NO2 Column`` (not ``CESM NO2 Column
    Tropospheric NO2 Column``).

    The quantity is normalised here (species words → formulas, smart title-case,
    formula subscripts) so axes/colorbars are consistent regardless of whether
    the caller passed a lookup name or a raw ``long_name`` — normalisation is
    idempotent on already-formatted strings.
    """
    q = _normalize_quantity_label(quantity or "")
    if source:
        src = source_display_name(source)
        if q and q in src:
            # The source name already contains the quantity (in good order),
            # e.g. quantity "NO2" with source "CESM NO2 Column" -> use the source
            # name as-is rather than reordering tokens ("CESM Column NO2").
            label = src
        else:
            distinctive = _distinctive_source_tokens(source, q)
            if distinctive and q:
                label = f"{' '.join(distinctive)} {q}"
            elif distinctive:
                label = " ".join(distinctive)
            else:
                label = q
    else:
        label = q
    u = format_units(units)
    return f"{label} ({u})" if u else label


def legend_label(source_label: str, uncertainty: str | None = None) -> str:
    """Compose a legend entry: friendly name, optionally with uncertainty note."""
    name = source_display_name(source_label)
    return f"{name} ({uncertainty})" if uncertainty else name


def bias_label(
    y_source: str,
    x_source: str,
    units: str | None,
    quantity: str | None = None,
) -> str:
    """Bias colorbar label '<Ysrc> − <Xsrc> (units)' — viewer-facing, never x/y.

    The word "Bias" lives in the plot TITLE (``title_text(..., operation="Bias")``),
    so the colorbar carries only the direction and units to avoid repetition.
    When ``quantity`` is given (it already appears in the title), tokens each
    source shares with it are stripped so the label stays terse, e.g.
    ``cesm_no2_column`` vs ``pandora`` with quantity ``NO2 Column`` →
    ``CESM − Pandora``. Any remaining shared trailing tokens are factored.
    """
    if quantity:
        yw = _distinctive_source_tokens(y_source, quantity)
        xw = _distinctive_source_tokens(x_source, quantity)
    else:
        yw = source_display_name(y_source).split()
        xw = source_display_name(x_source).split()
    while yw and xw and yw[-1] == xw[-1]:
        yw.pop()
        xw.pop()
    y = " ".join(yw) or source_display_name(y_source)
    x = " ".join(xw) or source_display_name(x_source)
    core = f"{y} − {x}"
    u = format_units(units)
    return f"{core} ({u})" if u else core


def title_text(quantity: str, operation: str | None = None) -> str:
    """Terse plot title: chem-format the quantity, optionally append operation."""
    q = format_plot_title(quantity or "")
    return f"{q} {operation}".strip() if operation else q


def sourced_title(source_label: str | None, quantity: str) -> str:
    """Single-source plot title with the SOURCE NAME FIRST: ``<Source> <Quantity>``.

    The source's distinctive tokens (vs the quantity) lead the title, so a
    single-source map/timeseries reads ``AERONET AOD (500 nm)`` or ``AirNow O3``
    rather than ``AOD (500 nm) (AERONET)``. Mirrors :func:`axis_label`'s
    source-first, de-duplicated ordering. Falls back to the bare quantity when
    no source is given.
    """
    if not source_label:
        return quantity
    src = distinctive_source_name(source_label, quantity)
    return f"{src} {quantity}" if src else quantity


def subtitle_text(start: Any, end: Any) -> str:
    """Date-range subtitle: 'YYYY-MM-DD – YYYY-MM-DD' or single date or ''."""
    if not start:
        return ""
    s = str(start).split(" ")[0].split("T")[0]
    e = str(end).split(" ")[0].split("T")[0] if end else s
    return s if s == e else f"{s} – {e}"


__all__ = [
    "format_units",
    "source_display_name",
    "quantity_label",
    "axis_label",
    "distinctive_source_name",
    "legend_label",
    "bias_label",
    "title_text",
    "sourced_title",
    "subtitle_text",
]
