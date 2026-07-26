"""Tests for ProgressFormatter step-detail rendering in _create_stage_display.

These tests verify that fmt.step() sets a transient ``_current_step`` detail
that _create_stage_display() renders as the '› detail' line when no per-item
display is active, and that the detail is cleared between stages.
"""

from __future__ import annotations

import pytest

from davinci_monet.pipeline.runner import ProgressFormatter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fmt() -> ProgressFormatter:
    """Return a formatter with output disabled (no real Rich Live needed)."""
    return ProgressFormatter(show_output=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStepDetailRendersInStageDisplay:
    """_create_stage_display shows the step detail when set."""

    def test_step_detail_rendered_after_step_call(self) -> None:
        """After step(), the display plain text should contain the message."""
        fmt = _make_fmt()
        fmt.stage_start("load_sources")

        # Before any step() call: message must NOT appear
        display_before = fmt._create_stage_display()
        assert "loading modis_viirs" not in display_before.plain

        fmt.step("loading modis_viirs [3/47] MOD08_M3.A2003.hdf")

        display_after = fmt._create_stage_display()
        assert "loading modis_viirs [3/47]" in display_after.plain

    def test_step_detail_not_in_blank_stage(self) -> None:
        """A fresh stage with no step() call shows no step detail."""
        fmt = _make_fmt()
        fmt.stage_start("load_sources")

        display = fmt._create_stage_display()
        # The display should contain the stage name but no '›' detail
        assert "load_sources" in display.plain
        assert "loading" not in display.plain

    def test_step_detail_updates_to_latest_message(self) -> None:
        """Calling step() twice shows the second message, not the first."""
        fmt = _make_fmt()
        fmt.stage_start("load_sources")

        fmt.step("loading modis_viirs [1/47] MOD08_M3.A2003001.hdf")
        fmt.step("loading modis_viirs [2/47] MOD08_M3.A2003002.hdf")

        display = fmt._create_stage_display()
        assert "MOD08_M3.A2003002" in display.plain
        # First file should no longer be the active display
        assert "MOD08_M3.A2003001" not in display.plain


class TestStepDetailClearedBetweenStages:
    """Step detail is cleared when a new stage starts (or stage ends)."""

    def test_step_detail_cleared_on_stage_start(self) -> None:
        """New stage_start() clears the previous stage's step detail."""
        fmt = _make_fmt()
        fmt.stage_start("load_sources")
        fmt.step("loading modis_viirs [3/47] MOD08_M3.A2003003.hdf")

        # Confirm detail is visible
        assert "loading modis_viirs" in fmt._create_stage_display().plain

        # Start a new stage (no real Live, so just call stage_start directly)
        fmt._live = None  # Ensure stage_start doesn't try to start a Live
        fmt.stage_start("pairing")

        display_new = fmt._create_stage_display()
        # Previous detail must be gone
        assert "loading modis_viirs" not in display_new.plain
        assert "pairing" in display_new.plain

    def test_step_detail_cleared_on_stage_end(self) -> None:
        """stage_end() clears the step detail so it doesn't bleed forward."""
        fmt = _make_fmt()
        fmt.stage_start("load_sources")
        fmt.step("loading modis_viirs [5/47] MOD08.hdf")

        # Simulate stage end (Live is None because show_output=False)
        fmt.stage_end("load_sources", success=True, duration=12.3)

        # Now _current_step should be cleared
        assert fmt._current_step is None

    def test_stage_end_logs_checkpoint_disposition(self) -> None:
        fmt = _make_fmt()
        fmt.stage_start("load_sources")

        fmt.stage_end(
            "load_sources",
            success=True,
            duration=0.0,
            disposition="restored",
        )

        assert any("completed [restored]" in line for line in fmt._lines)


class TestCurrentItemTakesPrecedenceOverStepDetail:
    """When _current_item is set, item display takes priority over step detail."""

    def test_item_shown_when_both_item_and_step_set(self) -> None:
        """If item_start() was called, the item is shown (not just step detail)."""
        fmt = _make_fmt()
        fmt.stage_start("load_sources")

        # Set a step detail first
        fmt.step("loading modis_viirs [1/47] MOD08.hdf")

        # Then an item_start (e.g. sequential loading of a named source)
        fmt.item_start("geometry", "airnow", index=1, total=2)

        display = fmt._create_stage_display()
        # The item name should appear
        assert "airnow" in display.plain

    def test_parallel_mode_not_replaced_by_step_detail(self) -> None:
        """Parallel mode progress takes precedence over step detail."""
        fmt = _make_fmt()
        fmt.stage_start("pairing")
        fmt.start_parallel(total=3, loading_msg="loading pairs")

        fmt.step("some step detail")

        display = fmt._create_stage_display()
        # Parallel counter should be present; step detail must not overwrite it
        assert "[0/3]" in display.plain
