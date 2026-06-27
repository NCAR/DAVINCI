"""Tests for davinci_monet.plots.style module."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


class TestNCARColors:
    """Tests for NCAR color definitions."""

    def test_ncar_colors_are_valid_hex(self):
        """All NCAR colors should be valid hex codes."""
        from davinci_monet.plots.style import NCAR_COLORS

        for name, color in NCAR_COLORS.items():
            assert color.startswith("#"), f"{name} should start with #"
            assert len(color) == 7, f"{name} should be 7 characters"
            # Validate hex
            int(color[1:], 16)  # Should not raise

    def test_ncar_primary_colors_exist(self):
        """Primary color aliases should exist."""
        from davinci_monet.plots.style import NCAR_ACCENT, NCAR_PRIMARY, NCAR_SECONDARY

        assert NCAR_PRIMARY.startswith("#")
        assert NCAR_SECONDARY.startswith("#")
        assert NCAR_ACCENT.startswith("#")

    def test_geometry_dataset_colors_exist(self):
        """Dataset and dataset colors should be defined."""
        from davinci_monet.plots.style import X_COLOR, Y_COLOR

        assert X_COLOR.startswith("#")
        assert Y_COLOR.startswith("#")

    def test_ncar_palette_is_list(self):
        """NCAR palette should be a list of colors."""
        from davinci_monet.plots.style import NCAR_PALETTE

        assert isinstance(NCAR_PALETTE, list)
        assert len(NCAR_PALETTE) >= 6  # At least 6 colors for variety
        for color in NCAR_PALETTE:
            assert color.startswith("#")


class TestFontSizes:
    """Tests for font size configurations."""

    def test_font_sizes_dataclass(self):
        """FontSizes should be a proper dataclass."""
        from davinci_monet.plots.style import FontSizes

        sizes = FontSizes()
        assert hasattr(sizes, "axes_label")
        assert hasattr(sizes, "axes_title")
        assert hasattr(sizes, "tick_label")
        assert hasattr(sizes, "legend")
        assert hasattr(sizes, "figure_title")
        assert hasattr(sizes, "annotation")

    def test_font_size_presets_exist(self):
        """Font size presets should exist."""
        from davinci_monet.plots.style import (
            FONT_SIZES_DEFAULT,
            FONT_SIZES_PRESENTATION,
            FONT_SIZES_PUBLICATION,
        )

        assert FONT_SIZES_DEFAULT is not None
        assert FONT_SIZES_PRESENTATION is not None
        assert FONT_SIZES_PUBLICATION is not None

    def test_presentation_sizes_larger_than_publication(self):
        """Presentation sizes should be larger than publication sizes."""
        from davinci_monet.plots.style import FONT_SIZES_PRESENTATION, FONT_SIZES_PUBLICATION

        assert FONT_SIZES_PRESENTATION.axes_label > FONT_SIZES_PUBLICATION.axes_label
        assert FONT_SIZES_PRESENTATION.axes_title > FONT_SIZES_PUBLICATION.axes_title
        assert FONT_SIZES_PRESENTATION.legend > FONT_SIZES_PUBLICATION.legend


class TestApplyNCARStyle:
    """Tests for apply_ncar_style function."""

    def teardown_method(self):
        """Reset matplotlib after each test."""
        plt.rcdefaults()

    def test_apply_ncar_style_sets_font(self):
        """apply_ncar_style should set font family."""
        from davinci_monet.plots.style import apply_ncar_style

        apply_ncar_style(use_seaborn=False)

        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert "Poppins" in plt.rcParams["font.sans-serif"]

    def test_apply_ncar_style_sets_sizes(self):
        """apply_ncar_style should set font sizes."""
        from davinci_monet.plots.style import FONT_SIZES_DEFAULT, apply_ncar_style

        apply_ncar_style(use_seaborn=False)

        assert plt.rcParams["axes.labelsize"] == FONT_SIZES_DEFAULT.axes_label
        assert plt.rcParams["axes.titlesize"] == FONT_SIZES_DEFAULT.axes_title

    def test_apply_ncar_style_presentation_context(self):
        """apply_ncar_style with presentation context should use larger sizes."""
        from davinci_monet.plots.style import FONT_SIZES_PRESENTATION, apply_ncar_style

        apply_ncar_style(context="presentation", use_seaborn=False)

        assert plt.rcParams["axes.labelsize"] == FONT_SIZES_PRESENTATION.axes_label

    def test_apply_ncar_style_publication_context(self):
        """apply_ncar_style with publication context should use smaller sizes."""
        from davinci_monet.plots.style import FONT_SIZES_PUBLICATION, apply_ncar_style

        apply_ncar_style(context="publication", use_seaborn=False)

        assert plt.rcParams["axes.labelsize"] == FONT_SIZES_PUBLICATION.axes_label

    def test_apply_ncar_style_sets_color_cycle(self):
        """apply_ncar_style should set color cycle to NCAR palette."""
        from davinci_monet.plots.style import NCAR_PALETTE, apply_ncar_style

        apply_ncar_style(use_seaborn=False)

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        assert colors == NCAR_PALETTE


class TestResetStyle:
    """Tests for reset_style function."""

    def test_reset_style_restores_defaults(self):
        """reset_style should restore matplotlib defaults."""
        from davinci_monet.plots.style import apply_ncar_style, reset_style

        # Get original value
        original_labelsize = plt.rcParams["axes.labelsize"]

        # Apply style (changes rcParams)
        apply_ncar_style(use_seaborn=False)

        # Reset
        reset_style()

        # Should be back to default
        assert plt.rcParams["axes.labelsize"] == original_labelsize


class TestColorUtilities:
    """Tests for color utility functions."""

    def test_get_color_for_variable_x(self):
        """get_color_for_variable should return x color for x_ prefix."""
        from davinci_monet.plots.style import X_COLOR, get_color_for_variable

        assert get_color_for_variable("x_pm25") == X_COLOR
        assert get_color_for_variable("X_O3") == X_COLOR

    def test_get_color_for_variable_y(self):
        """get_color_for_variable should return y color for y_ prefix."""
        from davinci_monet.plots.style import Y_COLOR, get_color_for_variable

        assert get_color_for_variable("y_pm25") == Y_COLOR
        assert get_color_for_variable("Y_O3") == Y_COLOR

    def test_get_color_for_variable_bias(self):
        """get_color_for_variable should return red for bias_ prefix."""
        from davinci_monet.plots.style import NCAR_COLORS, get_color_for_variable

        assert get_color_for_variable("bias_pm25") == NCAR_COLORS["red"]

    def test_get_color_for_variable_default(self):
        """get_color_for_variable should return blue for unknown prefix."""
        from davinci_monet.plots.style import NCAR_COLORS, get_color_for_variable

        assert get_color_for_variable("temperature") == NCAR_COLORS["ncar_blue"]

    def test_get_palette_default(self):
        """get_palette without n_colors should return full palette."""
        from davinci_monet.plots.style import NCAR_PALETTE, get_palette

        palette = get_palette()
        assert palette == list(NCAR_PALETTE)

    def test_get_palette_n_colors(self):
        """get_palette with n_colors should return that many colors."""
        from davinci_monet.plots.style import get_palette

        palette = get_palette(3)
        assert len(palette) == 3

    def test_get_palette_cycles(self):
        """get_palette should cycle when n_colors > palette length."""
        from davinci_monet.plots.style import NCAR_PALETTE, get_palette

        n_colors = len(NCAR_PALETTE) + 2
        palette = get_palette(n_colors)
        assert len(palette) == n_colors
        # First colors should match
        assert palette[: len(NCAR_PALETTE)] == list(NCAR_PALETTE)
        # Should cycle
        assert palette[len(NCAR_PALETTE)] == NCAR_PALETTE[0]


class TestColormapUtilities:
    """Tests for colormap utility functions."""

    def test_geosit_aod_levels_cap_at_one_with_low_end_detail(self):
        """GEOSIT AOD levels use fine low-end bins and cap plotted range at 1."""
        from davinci_monet.plots.style import geosit_aod_levels

        levels = geosit_aod_levels(np.array([0.0, 0.012, 0.4, 2.3, np.nan]))

        np.testing.assert_allclose(levels[:7], [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05])
        assert levels[-1] == pytest.approx(1.0)
        np.testing.assert_allclose(np.diff(levels[6:]), 0.05, atol=1e-12)

    @pytest.mark.parametrize("values", [np.array([]), np.array([np.nan]), np.array([-1.0, 0.0])])
    def test_geosit_aod_levels_fall_back_for_empty_or_nonpositive_data(self, values):
        from davinci_monet.plots.style import geosit_aod_levels

        np.testing.assert_allclose(geosit_aod_levels(values), np.linspace(0.0, 1.0, 11))

    def test_get_geosit_aod_cmap(self):
        """GEOSIT AOD style uses the same color table as the reference plots."""
        from davinci_monet.plots.style import get_geosit_aod_cmap

        cmap_name = get_geosit_aod_cmap()
        assert cmap_name == "turbo"
        plt.get_cmap(cmap_name)

    def test_get_bias_cmap(self):
        """get_bias_cmap should return a diverging colormap name."""
        from davinci_monet.plots.style import get_bias_cmap

        cmap_name = get_bias_cmap()
        assert isinstance(cmap_name, str)
        # Verify it's a valid colormap
        plt.get_cmap(cmap_name)  # Should not raise

    def test_get_sequential_cmap(self):
        """get_sequential_cmap should return a sequential colormap name."""
        from davinci_monet.plots.style import get_sequential_cmap

        cmap_name = get_sequential_cmap()
        assert isinstance(cmap_name, str)
        plt.get_cmap(cmap_name)  # Should not raise

    def test_get_density_cmap(self):
        """get_density_cmap should return a colormap name."""
        from davinci_monet.plots.style import get_density_cmap

        cmap_name = get_density_cmap()
        assert isinstance(cmap_name, str)
        plt.get_cmap(cmap_name)  # Should not raise


class TestStyleWithPlotting:
    """Style + plotting interaction tests (calls style/plot APIs directly)."""

    def teardown_method(self):
        """Reset matplotlib after each test."""
        plt.rcdefaults()
        plt.close("all")

    def test_style_applied_to_plot(self):
        """Applied style should affect new plots."""
        from davinci_monet.plots.style import NCAR_PALETTE, apply_ncar_style

        apply_ncar_style(use_seaborn=False)

        fig, ax = plt.subplots()
        lines = []
        for i in range(3):
            (line,) = ax.plot([0, 1], [i, i + 1])
            lines.append(line)

        # Lines should use NCAR palette colors
        for i, line in enumerate(lines):
            assert line.get_color() == NCAR_PALETTE[i]

        plt.close(fig)

    def test_style_affects_text_sizes(self):
        """Applied style should affect text sizes."""
        from davinci_monet.plots.style import FONT_SIZES_DEFAULT, apply_ncar_style

        apply_ncar_style(use_seaborn=False)

        fig, ax = plt.subplots()
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")

        # Get the label font size (need to draw to get actual size)
        fig.canvas.draw()

        plt.close(fig)
