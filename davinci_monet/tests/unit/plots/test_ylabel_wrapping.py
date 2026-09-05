"""A y-axis label taller than its axes must wrap, not lose characters.

A rotated y label longer than the axes is silently truncated at BOTH ends by
the Agg backend, and ``bbox_inches="tight"`` does not rescue it -- the figure
saves cleanly with text missing, so nothing errors and nothing warns. Long
composed labels ("Surface Downwelling Shortwave Anomaly (W m^-2)") hit this at
presentation font sizes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from davinci_monet.plots.base import set_ylabel_wrapped  # noqa: E402

LONG = "Surface Downwelling Shortwave Anomaly (W m$^{-2}$)"
SHORT = "2 m Temperature (K)"


@pytest.fixture
def ax():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.canvas.draw()
    yield ax
    plt.close(fig)


def _rendered_height(ax) -> float:
    fig = ax.get_figure()
    fig.canvas.draw()
    return ax.yaxis.label.get_window_extent(renderer=fig.canvas.get_renderer()).height


def test_a_long_label_keeps_every_character(ax) -> None:
    set_ylabel_wrapped(ax, LONG, fontsize=18)

    assert ax.get_ylabel().replace("\n", " ") == LONG


def test_a_long_label_is_wrapped_onto_more_than_one_line(ax) -> None:
    set_ylabel_wrapped(ax, LONG, fontsize=18)

    assert "\n" in ax.get_ylabel()


def test_wrapping_brings_the_label_back_within_the_axes(ax) -> None:
    """The point of wrapping: the label must actually fit once wrapped."""
    set_ylabel_wrapped(ax, LONG, fontsize=18)

    assert _rendered_height(ax) <= ax.get_window_extent().height


def test_a_short_label_is_left_alone(ax) -> None:
    set_ylabel_wrapped(ax, SHORT, fontsize=18)

    assert ax.get_ylabel() == SHORT
    assert "\n" not in ax.get_ylabel()


def test_a_caller_supplied_wrap_is_respected(ax) -> None:
    deliberate = "Surface Downwelling\nShortwave"
    set_ylabel_wrapped(ax, deliberate, fontsize=18)

    assert ax.get_ylabel() == deliberate


def test_the_configured_font_size_is_not_shrunk_to_fit(ax) -> None:
    set_ylabel_wrapped(ax, LONG, fontsize=18)

    assert ax.yaxis.label.get_fontsize() == 18
